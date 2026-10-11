"""Per-network drivers for the node agent. Each module exposes class Driver(env, mode) (contract in node/README.md).

Stdlib only; must stay Python 3.10 compatible (the image is Ubuntu 22.04).
"""
from __future__ import annotations

import collections
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping

# Modes the agent understands. foreground = inference with priority, background = miner soaking idle cycles,
# dedicated = owns the GPU, useful = same process as the foreground (Pearl via the vLLM plugin).
MODES = ("dedicated", "foreground", "background", "useful")


class DriverError(Exception):
    """Transient failure: the agent retries with backoff."""


class ConfigError(Exception):
    """Missing/invalid env. Raised from Driver.__init__; the agent fails fast with the message."""


class Unsupported(Exception):
    """This workload cannot run here (missing binary, mode not available). Permanent: no retries."""


def http_get(url: str, timeout: float = 2.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (local URLs only)
        return r.read().decode("utf-8", "replace")


def split_args(s: str) -> list[str]:
    return shlex.split(s) if s.strip() else []


class ProcessDriver:
    """Shared process handling: own process group, SIGSTOP/SIGCONT pause, graceful stop, duty-cycled throttle.

    Subclasses implement command()/popen_env() and may override stats()/load()."""
    name = "process"
    stop_grace_s = 15.0
    duty_period_s = 1.0
    capture_output = False  # True: stdout+stderr are piped, echoed as "<name>: line" and the last lines kept (log_tail)
    tail_lines = 40

    def __init__(self, env: Mapping[str, str], mode: str, clock: Callable[[], float] = time.monotonic):
        if mode not in MODES:
            raise ConfigError(f"{self.name}: unknown mode {mode!r}")
        self.env = dict(env)
        self.mode = mode
        self.clock = clock
        self.proc: subprocess.Popen | None = None
        self.level = 1.0
        self.paused = False  # SIGSTOPped right now
        self.parked = False  # killed on purpose while paused (pause_method=restart); alive() stays True
        self.paused_since: float | None = None
        self.pause_method = env.get("PM_PAUSE_METHOD") or env.get("PM_AGENT_PAUSE_METHOD") or "sigstop"
        self.restart_after_s = float(env.get("PM_RESTART_AFTER_S") or env.get("PM_AGENT_RESTART_AFTER_S") or "60")
        self._duty: threading.Thread | None = None
        self._duty_stop = threading.Event()
        self.exit_code: int | None = None
        self._tail: collections.deque[str] = collections.deque(maxlen=self.tail_lines)

    # -- to override --------------------------------------------------------------------------------------------
    def command(self) -> list[str]:
        raise NotImplementedError

    def popen_env(self) -> dict[str, str]:
        return dict(os.environ)

    def describe(self) -> str:
        """DRY_RUN text: the command line (never secrets passed through env)."""
        return shlex.join(self.command())

    def cwd(self) -> str | None:
        return None

    def on_line(self, line: str) -> None:
        """Called from the reader thread for every output line (capture_output only)."""

    def secrets(self) -> list[str]:
        """Values that must never appear in status text or logs we compose."""
        return []

    def redact(self, text: str, urls: bool = True) -> str:
        """Hide secrets (and, by default, every URL) in text we show. urls=False keeps ordinary URLs in worker output."""
        for v in sorted(self.secrets(), key=len, reverse=True):  # longest first: a whole link before its token
            if v and len(v) >= 4:
                text = text.replace(v, "***")
        if urls:
            text = re.sub(r"https?://\S+", "<url>", text)
        return text

    def log_tail(self, n: int = 3, width: int = 300) -> str:
        lines = [x.strip() for x in list(self._tail)[-12:] if x.strip()][-n:]
        return self.redact(" | ".join(lines))[-width:]

    def exit_reason(self) -> str:
        """Why the process is gone: exit code (or signal) plus the last output lines when they were captured."""
        rc = self.exit_code
        if rc is None and self.proc is not None:
            rc = self.proc.poll()
        what = "exited" if rc is None else (f"was killed by signal {-rc}" if rc < 0 else f"exited with code {rc}")
        tail = self.log_tail()
        return f"{self.name} {what}" + (f"; last output: {tail}" if tail else "")

    # -- lifecycle ----------------------------------------------------------------------------------------------
    def start(self) -> None:
        self.parked = False
        self.paused = False
        self.paused_since = None
        self._spawn(self.command())

    def _spawn(self, cmd: list[str]) -> None:
        kw: dict = {}
        if self.capture_output:
            kw = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True, "errors": "replace",
                  "bufsize": 1}
        try:
            proc = subprocess.Popen(cmd, env=self.popen_env(), cwd=self.cwd(), start_new_session=True, **kw)  # noqa: S603
        except FileNotFoundError as e:
            raise Unsupported(f"{self.name}: {cmd[0]} not found") from e
        except PermissionError as e:
            raise DriverError(f"{self.name}: {cmd[0]} is not executable (permission denied)") from e
        except OSError as e:
            raise DriverError(f"{self.name}: cannot start: {e}") from e
        self.proc = proc
        self.exit_code = None
        self.level = 1.0
        self._tail.clear()
        if self.capture_output:
            threading.Thread(target=self._read, args=(proc,), daemon=True).start()

    def _read(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip("\n")
                safe = self.redact(line, urls=False)  # a link or key the installer echoes must not reach docker logs
                self._tail.append(safe)
                print(f"{self.name}: {safe}", flush=True)  # keeps the worker output in `docker logs`
                try:
                    self.on_line(line)
                except Exception:  # noqa: BLE001 - a parser bug must not stop the reader
                    pass
        except (OSError, ValueError):
            pass

    def _signal(self, sig: int) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                pass

    def stop(self) -> None:
        self._duty_stop.set()
        p = self.proc
        if p is None:
            return
        if p.poll() is None:
            self._signal(signal.SIGCONT)
            self._signal(signal.SIGTERM)
            try:
                p.wait(self.stop_grace_s)
            except subprocess.TimeoutExpired:
                self._signal(signal.SIGKILL)
                p.wait(5)
        self.proc = None
        self.parked = self.paused = False

    def alive(self) -> bool:
        if self.parked:
            return True
        p = self.proc
        if p is None:
            return False
        rc = p.poll()
        if rc is None:
            return True
        self.exit_code = rc
        return False

    def load(self) -> float | None:
        return 0.0

    def stats(self) -> dict:
        return {}

    # -- throttle -----------------------------------------------------------------------------------------------
    def throttle(self, level: float) -> None:
        """0 pause .. 1 full. Fractions are a duty cycle of SIGSTOP/SIGCONT (GPU kernels in flight finish first)."""
        level = max(0.0, min(1.0, level))
        self.level = level
        if self.proc is None and not self.parked:
            return
        if level <= 0.01:
            self._stop_duty()
            self._pause()
            if (self.pause_method == "restart" and not self.parked and self.paused_since is not None
                    and self.clock() - self.paused_since >= self.restart_after_s):
                self.stop()
                self.parked = True
        else:
            if self.parked:
                self.start()
            if level >= 0.99:
                self._stop_duty()
                self._resume()
            else:
                self._start_duty()

    def _pause(self) -> None:
        if not self.paused:
            self._signal(signal.SIGSTOP)
            self.paused = True
            self.paused_since = self.clock()

    def _resume(self) -> None:
        if self.paused:
            self._signal(signal.SIGCONT)
            self.paused = False
            self.paused_since = None

    def _start_duty(self) -> None:
        if self._duty is not None and self._duty.is_alive():
            return
        self._duty_stop = threading.Event()
        self._duty = threading.Thread(target=self._duty_loop, args=(self._duty_stop,), daemon=True)
        self._duty.start()

    def _stop_duty(self) -> None:
        self._duty_stop.set()
        self._duty = None

    def _duty_loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            run = self.level * self.duty_period_s
            self._resume()
            if stop.wait(run):
                break
            self._pause()
            if stop.wait(self.duty_period_s - run):
                break
