"""Node agent (PID 1 in the node image, stdlib only, Python 3.10+). Contract: node/README.md.

Runs every workload of the plan concurrently, balances them in seconds (a dedicated miner soaks the GPU cycles an
inference engine leaves idle), restarts crashed workloads with backoff, serves a local /status, propagates signals.
The controller only chooses the plan (PM_PLAN / PM_PARTS) and counts GPUs.
"""
from __future__ import annotations

import importlib
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drivers import ConfigError, Unsupported  # noqa: E402
from workername import make_worker  # noqa: E402

WORKER_ID_VARS = ("SALAD_MACHINE_ID", "SALAD_INSTANCE_ID", "PM_INSTANCE_ID")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} agent: {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentConfig:
    tick_s: float = 2.0
    status_port: int = 8085
    status_host: str = "127.0.0.1"
    busy_on_load: float = 0.30
    idle_after_s: float = 5.0
    throttle_floor: float = 0.0
    unknown_share: float = 0.5  # background duty while the foreground is alive but its load cannot be read
    no_work_after_s: float = 120.0
    restart_backoff_s: tuple[float, ...] = (5.0, 15.0, 60.0, 300.0)
    stable_s: float = 600.0  # a workload that ran this long resets its failure count
    bg_start_delay_s: float = 120.0  # max wait for the foreground to answer a load probe before the miner starts
    share_window_s: float = 600.0
    failed_log_s: float = 300.0  # a workload that stays FAILED repeats its reason on stdout this often
    exit_when_failed_s: float = 900.0  # every workload permanently failed: stay up (status says FAILED) this long, then exit 3

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> AgentConfig:
        def f(name: str, default: float) -> float:
            v = env.get(f"PM_AGENT_{name}", "").strip()
            try:
                return float(v) if v else default
            except ValueError as e:
                raise ConfigError(f"PM_AGENT_{name}={v!r} is not a number") from e

        d = cls()
        bo = env.get("PM_AGENT_RESTART_BACKOFF_S", "").replace("[", "").replace("]", "").strip()
        try:
            backoff = tuple(float(x) for x in bo.split(",") if x.strip()) if bo else d.restart_backoff_s
        except ValueError as e:
            raise ConfigError(f"PM_AGENT_RESTART_BACKOFF_S={bo!r} is not a list of numbers") from e
        return cls(tick_s=f("TICK_S", d.tick_s), status_port=int(f("STATUS_PORT", d.status_port)),
                   status_host=env.get("PM_AGENT_STATUS_HOST", d.status_host),
                   busy_on_load=f("BUSY_ON_LOAD", d.busy_on_load), idle_after_s=f("IDLE_AFTER_S", d.idle_after_s),
                   throttle_floor=f("THROTTLE_FLOOR", d.throttle_floor), unknown_share=f("UNKNOWN_SHARE", d.unknown_share),
                   no_work_after_s=f("NO_WORK_AFTER_S", d.no_work_after_s), restart_backoff_s=backoff or d.restart_backoff_s,
                   stable_s=f("STABLE_S", d.stable_s), bg_start_delay_s=f("BG_START_DELAY_S", d.bg_start_delay_s),
                   share_window_s=f("SHARE_WINDOW_S", d.share_window_s), failed_log_s=f("FAILED_LOG_S", d.failed_log_s),
                   exit_when_failed_s=f("EXIT_WHEN_FAILED_S", d.exit_when_failed_s))


def parse_parts(s: str) -> list[tuple[str, str]]:
    """'dolphin:foreground,pearl:background' -> [('dolphin', 'foreground'), ('pearl', 'background')]."""
    parts = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        net, _, mode = item.partition(":")
        if not net or not mode:
            raise ConfigError(f"PM_PARTS entry {item!r} is not network:mode")
        parts.append((net, mode))
    if not parts:
        raise ConfigError("PM_PARTS is empty (expected e.g. pearl:dedicated)")
    fg = [p for p in parts if p[1] == "foreground"]
    bg = [p for p in parts if p[1] == "background"]
    if len(fg) > 1 or len(bg) > 1 or (bg and not fg):
        raise ConfigError(f"PM_PARTS {s!r}: need at most one foreground, and a background only next to one")
    return parts


def resolve_worker(env: Mapping[str, str]) -> str:
    if env.get("WORKER", "").strip():
        return env["WORKER"].strip()
    for var in WORKER_ID_VARS:
        if env.get(var, "").strip():
            return make_worker(env.get("PM_WORKER_PREFIX", ""), env[var])
    raise ConfigError("no worker name: set WORKER, or PM_WORKER_PREFIX plus one of " + ", ".join(WORKER_ID_VARS))


# ---------------------------------------------------------------------------------------------------------------
# balancing (pure given a clock)
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Verdict:
    level: float  # background intensity 0..1
    state: str  # busy | idle | unknown | no_work | fg_down


class Balancer:
    """Foreground load -> background intensity. Busy (>= busy_on_load) drops the miner to throttle_floor at once; it
    returns to full only after idle_after_s below the threshold (no flapping); a foreground that reports exactly no
    load for no_work_after_s, or is down, hands the whole GPU to the background."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.busy = False
        self.idle_since: float | None = None
        self.zero_since: float | None = None

    def decide(self, load: float | None, fg_alive: bool, now: float) -> Verdict:
        c = self.cfg
        if not fg_alive:
            self.busy, self.idle_since, self.zero_since = False, None, None
            return Verdict(1.0, "fg_down")
        if load is None:
            self.zero_since = None
            return Verdict(max(c.throttle_floor, c.unknown_share), "unknown")
        if load >= c.busy_on_load:
            self.busy, self.idle_since, self.zero_since = True, None, None
            return Verdict(c.throttle_floor, "busy")
        if self.busy:
            if self.idle_since is None:
                self.idle_since = now
            if now - self.idle_since < c.idle_after_s:
                return Verdict(c.throttle_floor, "busy")
            self.busy, self.idle_since = False, None
        if load <= 0.01:
            if self.zero_since is None:
                self.zero_since = now
            if now - self.zero_since >= c.no_work_after_s:
                return Verdict(1.0, "no_work")
        else:
            self.zero_since = None
        return Verdict(1.0, "idle")


# ---------------------------------------------------------------------------------------------------------------
# supervision
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Slot:
    network: str
    mode: str
    driver: object
    started: bool = False
    started_at: float = 0.0
    next_start: float = 0.0
    fails: int = 0
    restarts: int = 0
    permanent: bool = False
    error: str = ""
    throttle: float = 1.0
    share: float | None = None  # moving average over share_window_s
    last_stats: dict = field(default_factory=dict)
    last_failure: str = ""  # why the workload last failed (kept after a restart, until it is verified running again)
    state: str = "pending"  # pending | installing | starting | running | failed (what the status shows)
    reason: str = ""
    state_since: float = 0.0
    logged: tuple[str, str] = ("", "")
    logged_at: float = 0.0


class Agent:
    def __init__(self, parts: Sequence[tuple[str, str]], drivers: Sequence[object], cfg: AgentConfig, plan: str = "",
                 worker: str = "", clock: Callable[[], float] = time.monotonic):
        self.cfg, self.plan, self.worker, self.clock = cfg, plan, worker, clock
        self.slots = [Slot(n, m, d) for (n, m), d in zip(parts, drivers, strict=True)]
        self.fg = next((s for s in self.slots if s.mode == "foreground"), None)
        self.bg = next((s for s in self.slots if s.mode == "background"), None)
        self.balancer = Balancer(cfg)
        self.verdict = Verdict(1.0, "idle")
        self.t0 = clock()
        self.stop_event = threading.Event()
        self._last_tick = self.t0
        self._all_failed_since: float | None = None
        self._lock = threading.Lock()

    # -- one balancing step ---------------------------------------------------------------------------------------
    def tick(self) -> None:
        now = self.clock()
        dt = max(0.0, now - self._last_tick)
        self._last_tick = now
        for s in self.slots:
            self._supervise(s, now)
        for s in self.slots:
            self._refresh_state(s, now)
        load = None
        if self.fg is not None:
            load = self._call(self.fg, "load")
            fg_alive = (self.fg.started and self.fg.driver.alive() and not self.fg.permanent
                        and self.fg.state != "failed")  # up but never verified counts as down: the miner keeps the GPU
            self.verdict = self.balancer.decide(load, fg_alive, now)
            if self.bg is not None and self.bg.started and not self.bg.permanent:
                self._apply_throttle(self.bg, self.verdict.level)
            self._ema(self.fg, 0.0 if load is None else min(1.0, load), dt)
            if self.bg is not None:
                self._ema(self.bg, self.bg.throttle if self.bg.started else 0.0, dt)
        for s in self.slots:
            s.last_stats = self._call(s, "stats") or {}

    # -- state shown in the status and on stdout --------------------------------------------------------------------
    def slot_state(self, s: Slot) -> tuple[str, str]:
        """(pending|installing|starting|running|failed, reason). 'failed' covers every way a workload is not running: no
        link/binary, download rejected, not executable, exited, up without evidence. 'No work right now' is NOT here: a
        running workload with load 0 is running (see work_state)."""
        if s.permanent:
            return "failed", f"{s.error} (not retrying)"
        if not s.started:
            return ("failed", s.last_failure or "not running") if s.fails else ("pending", "")
        health = getattr(s.driver, "health", None)
        if health is None:
            return ("running", "") if s.driver.alive() else ("failed", s.last_failure or "exited")
        try:
            h = health()
        except Exception as e:  # noqa: BLE001
            return "failed", f"health check crashed: {e!r}"
        return h.get("state", "running"), h.get("reason", "")

    def work_state(self, s: Slot) -> str | None:
        """For a running foreground: busy | idle (connected, no requests right now) | unknown (load unreadable)."""
        if s is not self.fg or s.state != "running":
            return None
        return {"busy": "busy", "idle": "idle", "no_work": "idle"}.get(self.verdict.state, "unknown")

    def _refresh_state(self, s: Slot, now: float) -> None:
        state, reason = self.slot_state(s)
        if state != s.state:
            s.state_since = now
        s.state, s.reason = state, reason
        key = (state, reason)
        if state == "failed":
            s.last_failure = reason if not s.permanent else s.last_failure
            if key != s.logged or now - s.logged_at >= self.cfg.failed_log_s:
                wait = max(0.0, s.next_start - now) if not s.permanent and not s.started else 0.0
                log(f"{s.network} FAILED: {reason}" + (f" (retrying in {wait:.0f}s)" if wait else ""))
                s.logged, s.logged_at = key, now
        elif key != s.logged:
            if state == "running":
                s.last_failure = ""
                log(f"{s.network} running" + (f" ({w})" if (w := self.work_state(s)) else ""))
            elif state != "pending":
                log(f"{s.network} {state}" + (f" (after: {s.last_failure})" if s.last_failure else ""))
            s.logged, s.logged_at = key, now

    def _ema(self, s: Slot, x: float, dt: float) -> None:
        a = min(1.0, dt / self.cfg.share_window_s) if self.cfg.share_window_s > 0 else 1.0
        s.share = x if s.share is None else s.share + a * (x - s.share)

    def _apply_throttle(self, s: Slot, level: float) -> None:
        if abs(level - s.throttle) > 1e-9:
            log(f"{s.network}: background intensity {s.throttle:.2f} -> {level:.2f} ({self.verdict.state})")
        s.throttle = level
        self._call(s, "throttle", level)

    def _call(self, s: Slot, method: str, *args):
        try:
            return getattr(s.driver, method)(*args)
        except Exception as e:  # a driver bug must never kill the agent or the other workload
            log(f"{s.network}.{method} failed: {e!r}")
            return None

    def _may_start(self, s: Slot, now: float) -> bool:
        """The background waits for the foreground to answer a load probe (VRAM goes to the model first), at most
        bg_start_delay_s, and starts at once if the foreground can never run."""
        if s is not self.bg or self.fg is None or self.fg.permanent or (self.fg.fails and not self.fg.started):
            return True
        if now - self.t0 >= self.cfg.bg_start_delay_s:
            return True
        return self.fg.started and self._call(self.fg, "load") is not None

    def _supervise(self, s: Slot, now: float) -> None:
        if s.permanent:
            return
        if s.started:
            if s.driver.alive():
                state, reason = self.slot_state(s)
                if state != "failed":
                    if s.fails and now - s.started_at >= self.cfg.stable_s:
                        s.fails = 0
                    return
                # Up but reported failed (never connected, download too slow): stop it and start it again after the
                # backoff, so the link is fetched afresh instead of the node sitting FAILED for the life of the container.
                self._call(s, "stop")
                s.error = s.last_failure = reason
                log(f"{s.network}: {reason}; stopping it to start again")
            else:
                why = getattr(s.driver, "exit_reason", None)
                s.error = s.last_failure = str(why()) if why else "exited"
            s.started = False
            s.fails += 1
            s.restarts += 1
            wait = self.cfg.restart_backoff_s[min(s.fails - 1, len(self.cfg.restart_backoff_s) - 1)]
            s.next_start = now + wait
            log(f"{s.network}: failed (failure {s.fails}), restarting in {wait:.0f}s: {s.error}")
        if now < s.next_start or not self._may_start(s, now):
            return
        try:
            s.driver.start()
        except Unsupported as e:
            s.permanent, s.error = True, str(e)
            log(f"{s.network}: unsupported, not retrying: {e}")
            return
        except Exception as e:
            s.fails += 1
            s.error = s.last_failure = str(e)
            wait = self.cfg.restart_backoff_s[min(s.fails - 1, len(self.cfg.restart_backoff_s) - 1)]
            s.next_start = now + wait
            log(f"{s.network}: start failed ({e}); retrying in {wait:.0f}s")
            return
        s.started, s.started_at, s.error = True, now, ""
        s.throttle = 1.0
        log(f"{s.network}: started ({s.mode})")

    # -- run / stop -----------------------------------------------------------------------------------------------
    def all_failed(self) -> bool:
        return all(s.permanent for s in self.slots)

    def run(self) -> int:
        order = sorted(self.slots, key=lambda s: s.mode != "foreground")  # foreground first
        self.slots = order
        while not self.stop_event.is_set():
            self.tick()
            if self.all_failed():
                # Exiting at once would make the provider restart the container in a hot loop (each restart burning
                # the short-lived link). Stay up and FAILED in the status; leave only after exit_when_failed_s.
                self._all_failed_since = self._all_failed_since if self._all_failed_since is not None else self.clock()
                if self.clock() - self._all_failed_since >= self.cfg.exit_when_failed_s:
                    log(f"every workload has been unsupported or failed for {self.cfg.exit_when_failed_s:.0f}s; exiting")
                    self.shutdown()
                    return 3
            else:
                self._all_failed_since = None
            self.stop_event.wait(self.cfg.tick_s)
        self.shutdown()
        return 0

    def shutdown(self) -> None:
        for s in reversed(self.slots):  # background (miner) first, the model last
            if not s.permanent:
                self._call(s, "stop")
                s.started = False

    # -- status ---------------------------------------------------------------------------------------------------
    def status(self) -> dict:
        now = self.clock()
        wl = []
        for s in self.slots:
            state, reason = self.slot_state(s)
            w = {"network": s.network, "mode": s.mode, "alive": bool(s.started and s.driver.alive()),
                 "state": state, "reason": reason, "state_for_s": int(max(0.0, now - s.state_since)),
                 "restarts": s.restarts,
                 "gpu_share": None if s.share is None else round(s.share, 3), "throttle": round(s.throttle, 3)}
            if state == "running" and (work := self.work_state(s)):
                w["work"] = work
            if state == "failed":
                w["permanent"] = s.permanent
                if not s.permanent and not s.started:
                    w["retry_in_s"] = max(0, int(s.next_start - now))
            elif s.last_failure:
                w["last_failure"] = s.last_failure
            if s.permanent:
                w["error"] = s.error
            health = getattr(s.driver, "health", None)
            if health is not None and state == "running":
                try:
                    w["evidence"] = health().get("evidence", [])
                except Exception:  # noqa: BLE001
                    pass
            for k in ("tokens_per_s", "hashrate_ths"):
                if k in s.last_stats:
                    w[k] = round(s.last_stats[k], 4)
            wl.append(w)
        primary = wl[0] if wl else {}
        return {"plan": self.plan, "worker": self.worker, "uptime_s": int(now - self.t0),
                "foreground_state": self.verdict.state if self.fg else None,
                "primary": {k: primary[k] for k in ("network", "state", "reason", "work") if k in primary},
                "ok": not any(w["state"] == "failed" for w in wl), "summary": self.summary(wl), "workloads": wl}

    @staticmethod
    def summary(wl: list[dict]) -> str:
        """One line: 'dolphin FAILED: <reason> · pearl running'. A failed workload never reads as idle."""
        parts = []
        for w in wl:
            if w["state"] == "failed":
                parts.append(f"{w['network']} FAILED: {w['reason']}")
            elif w["state"] == "running" and w.get("work"):
                parts.append(f"{w['network']} running ({w['work']})")
            else:
                parts.append(f"{w['network']} {w['state']}")
        return " · ".join(parts)


def serve_status(agent: Agent, host: str, port: int) -> ThreadingHTTPServer | None:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.split("?")[0] not in ("/", "/status"):
                self.send_error(404)
                return
            body = json.dumps(agent.status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    try:
        srv = ThreadingHTTPServer((host, port), H)
    except OSError as e:
        log(f"status server disabled: {e}")
        return None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ---------------------------------------------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------------------------------------------
class FailedDriver:
    """Stands in for a driver whose configuration is invalid, so the agent reports FAILED with the reason and the other
    workload (the miner) keeps the GPU, instead of the whole node exiting with a configuration error."""

    def __init__(self, reason: str):
        self.reason = reason

    def start(self) -> None:
        raise Unsupported(self.reason)

    def stop(self) -> None: ...
    def alive(self) -> bool:
        return False

    def load(self):
        return None

    def throttle(self, level: float) -> None: ...
    def stats(self) -> dict:
        return {}

    def describe(self) -> str:
        return f"(unsupported: {self.reason})"


def build(env: Mapping[str, str]) -> Agent:
    cfg = AgentConfig.from_env(env)
    parts = parse_parts(env.get("PM_PARTS") or env.get("PM_NETWORK_PARTS") or "pearl:dedicated")
    worker = resolve_worker(env)
    denv = {**env, "PM_WORKER": worker}
    drivers = []
    for net, mode in parts:
        try:
            mod = importlib.import_module(f"drivers.{net}")
        except ModuleNotFoundError as e:
            raise ConfigError(f"unknown network {net!r} in PM_PARTS") from e
        try:
            drivers.append(mod.Driver(denv, mode))
        except ConfigError as e:
            if mode == "foreground":  # stay up and say loudly why the primary is down (the miner, if any, keeps the GPU)
                drivers.append(FailedDriver(str(e)))
            else:
                raise
    return Agent(parts, drivers, cfg, plan=env.get("PM_PLAN", ""), worker=worker)


def main(env: Mapping[str, str] | None = None) -> int:
    env = dict(os.environ if env is None else env)
    try:
        agent = build(env)
    except ConfigError as e:
        print(f"pearl-node: configuration error: {e}", file=sys.stderr, flush=True)
        return 2
    if env.get("DRY_RUN") == "1":
        print(f"plan {agent.plan or '-'}  worker {agent.worker}")
        for s in agent.slots:
            print(f"[{s.network}:{s.mode}] {s.driver.describe()}")
        return 0
    log(f"plan {agent.plan} worker {agent.worker} parts "
        + ", ".join(f"{s.network}:{s.mode}" for s in agent.slots))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: agent.stop_event.set())
    serve_status(agent, agent.cfg.status_host, agent.cfg.status_port)
    return agent.run()


if __name__ == "__main__":
    sys.exit(main())
