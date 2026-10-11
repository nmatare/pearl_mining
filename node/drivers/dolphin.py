"""Dolphin driver (dphn.ai, serve LLM tokens). The worker is a closed per-key installer (research/dolphin_node.md), so
nothing is baked into the image. Launch sources, first match wins:
  DOLPHIN_WORKER_CMD   full command line (a path baked into a derived image or mounted volume)
  DOLPHIN_BOOTSTRAP    the dashboard's worker bootstrap link: its `curlCommand` (preferred), or its bootstrap/download URL
  DOLPHIN_WORKER_URL   the same, for a link that is a direct binary URL (`downloadUrl`) or a bootstrap script URL
  none of them         start() raises Unsupported: the agent reports "dolphin FAILED: no worker configured", never "idle"
A link value is classified (override with DOLPHIN_WORKER_KIND=binary|script|curl):
  curl command   run verbatim under `bash -o pipefail` in DOLPHIN_WORKER_DIR (`--fail` is added to curl when missing, so an
                 HTTP error is an error and not an HTML page piped into a shell)
  script URL     (path contains "bootstrap" or ends in .sh) downloaded to DOLPHIN_WORKER_DIR/bootstrap.sh and run with bash
  binary URL     downloaded to DOLPHIN_WORKER_PATH (default /opt/dolphin/dolphinpod-worker), chmod +x, run
The installer formats are UNVERIFIED until a real run (research/dolphin_node.md). If an installer exits 0 and leaves an
executable at DOLPHIN_WORKER_PATH, that executable is run next (DOLPHIN_WORKER_ARGS appends arguments).
Every start re-downloads (links are short-lived); if the re-download fails and an earlier download is on disk, that one is used.

Health (agent status): installing -> starting -> running | failed. "running" needs EVIDENCE: a worker log line matching
DOLPHIN_READY_REGEX (default "connected to lighthouse") or a reachable metrics endpoint. A process that is up but shows no
evidence after DOLPHIN_CONNECT_TIMEOUT_S (default 1800) is reported FAILED; the agent then stops it and starts it again with
backoff (re-fetching the link). A log line matching DOLPHIN_DISCONNECT_REGEX withdraws the evidence (FAILED if the worker
does not reconnect within DOLPHIN_RECONNECT_S, default 300), metrics evidence ages out after DOLPHIN_METRICS_TTL_S (120),
and a download that has not finished after DOLPHIN_FETCH_TOTAL_S (default 900) is FAILED ("download too slow").
Everything the worker prints is redacted (link, key, tokens) before it reaches stdout or a status reason.
No requests right now is NOT a failure: that is a connected worker with load 0 (the agent reports it as idle).

Env: DOLPHIN_WALLET (required), DOLPHIN_API_KEY (passed to the worker, never printed), DOLPHIN_MODE, DOLPHIN_METRICS_URL
(Prometheus text, default Sonar's port 2242, applicability to the worker UNCONFIRMED), DOLPHIN_SLOTS (concurrency slots).
Load = busy fraction from request counters; unknown (None) when the metrics endpoint is unreachable. Foreground only
('useful' is refused: the closed worker validates logprobs, the Pearl plugin cannot be injected).
"""
from __future__ import annotations

import os
import re
import stat
import threading
import urllib.error
import urllib.request
from urllib.parse import urlparse

from . import ConfigError, DriverError, ProcessDriver, Unsupported, http_get, split_args

DEFAULT_METRICS = "http://127.0.0.1:2242/metrics"
DEFAULT_READY = r"connected to lighthouse"
# a line that says the session is gone wins over a ready match in the same line ("NOT connected to lighthouse")
DEFAULT_LOST = (r"disconnect|not connected|connection (?:lost|closed|dropped)|lost connection|"
                r"failed to connect|unable to connect|reconnecting|can'?t connect|cannot connect")
KINDS = ("cmd", "binary", "script", "curl")
_LINE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf)\s*$")
_CURLISH = re.compile(r"^\s*(?:\$\s*)?(?:sudo\s+)?(?:curl|wget)\b")
_CURL_FAIL = re.compile(r"(?:^|\s)(?:--fail\b|-[A-Za-z]*f[A-Za-z]*\b)")
urlopen = urllib.request.urlopen  # patched in tests


def parse_prom(text: str) -> dict[str, float]:
    """Sum samples by metric name (labels dropped)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if m:
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
            except ValueError:
                pass
    return out


def _pick(m: dict[str, float], *needles: str) -> float | None:
    vals = [v for k, v in m.items() if any(n in k for n in needles)]
    return sum(vals) if vals else None


def classify_source(text: str, hint: str = "") -> tuple[str, str]:
    """Dashboard link value -> (kind, cleaned text). Raises ValueError when it is none of the known shapes."""
    t = text.strip().strip("'\"").strip()
    hint = hint.strip().lower()
    if hint:
        if hint not in ("binary", "script", "curl"):
            raise ValueError(f"DOLPHIN_WORKER_KIND={hint!r} is not binary, script or curl")
        return hint, t
    if _CURLISH.match(t):
        return "curl", t
    if re.match(r"^https?://\S+$", t):
        path = urlparse(t).path.lower()
        return ("script" if "bootstrap" in path or path.endswith(".sh") else "binary"), t
    raise ValueError("the Dolphin worker link is neither a URL nor a curl command")


def with_fail_flag(cmd: str) -> str:
    """curl without --fail pipes an HTML error page into the shell; make HTTP errors real errors."""
    m = re.match(r"^(\s*(?:\$\s*)?(?:sudo\s+)?curl\b)(.*)$", cmd, re.S)
    if not m or _CURL_FAIL.search(m.group(2).split("|", 1)[0]):
        return cmd
    return f"{m.group(1)} --fail{m.group(2)}"


def http_reason(code: int) -> str:
    if code in (401, 403):
        return f"download link rejected (HTTP {code})"
    if code == 404:
        return "download link not found (HTTP 404)"
    if code == 410:
        return "download link expired (HTTP 410)"
    if code == 429:
        return "download rate limited (HTTP 429)"
    if code >= 500:
        return f"download server error (HTTP {code})"
    return f"download failed (HTTP {code})"


def explain_exit(text: str) -> str | None:
    """A known failure pattern in the installer/worker output, or None."""
    low = text.lower()
    m = re.search(r"returned error:? (\d{3})", low) or re.search(r"\bhttp[/ ]?[\d.]* (\d{3})\b", low)
    if m and m.group(1)[0] in "45":
        return http_reason(int(m.group(1)))
    if "permission denied" in low:
        return "worker binary is not executable (permission denied)"
    if "could not resolve host" in low or "name or service not known" in low or "curl: (6)" in low:
        return "cannot reach the download server (DNS)"
    if "connection refused" in low or "curl: (7)" in low or "timed out" in low or "curl: (28)" in low:
        return "cannot reach the download server (network)"
    return None


class Driver(ProcessDriver):
    name = "dolphin"
    capture_output = True

    def __init__(self, env, mode, **kw):
        super().__init__(env, mode, **kw)
        if mode == "useful":
            raise ConfigError("dolphin: 'useful' mode is not supported (closed worker, logprob validation)")
        if mode == "background":
            raise ConfigError("dolphin: cannot run in the background of another workload")
        if not env.get("DOLPHIN_WALLET", "").strip():
            raise ConfigError("dolphin: DOLPHIN_WALLET is not set (EVM address of the Dolphin account)")
        self.cmd = split_args(env.get("DOLPHIN_WORKER_CMD", ""))
        self.dest = env.get("DOLPHIN_WORKER_PATH", "/opt/dolphin/dolphinpod-worker")
        self.workdir = env.get("DOLPHIN_WORKER_DIR", "") or os.path.dirname(self.dest) or "."
        self.worker_args = split_args(env.get("DOLPHIN_WORKER_ARGS", ""))
        self.metrics_url = env.get("DOLPHIN_METRICS_URL", DEFAULT_METRICS)
        self.slots = max(1.0, float(env.get("DOLPHIN_SLOTS", "8")))
        self.connect_timeout_s = float(env.get("DOLPHIN_CONNECT_TIMEOUT_S") or "1800")
        self.fetch_timeout_s = float(env.get("DOLPHIN_FETCH_TIMEOUT_S") or "120")
        self.fetch_total_s = float(env.get("DOLPHIN_FETCH_TOTAL_S") or "900")  # overall download deadline
        self.reconnect_grace_s = float(env.get("DOLPHIN_RECONNECT_S") or "300")  # a dropped session may come back
        self.metrics_ttl_s = float(env.get("DOLPHIN_METRICS_TTL_S") or "120")  # metrics evidence ages out
        try:
            self._ready = re.compile(env.get("DOLPHIN_READY_REGEX") or DEFAULT_READY, re.I)
        except re.error as e:
            raise ConfigError(f"dolphin: DOLPHIN_READY_REGEX is not a regular expression ({e})") from e
        try:
            self._lost = re.compile(env.get("DOLPHIN_DISCONNECT_REGEX") or DEFAULT_LOST, re.I)
        except re.error as e:
            raise ConfigError(f"dolphin: DOLPHIN_DISCONNECT_REGEX is not a regular expression ({e})") from e
        self.kind: str | None = None
        self.source = ""
        self.problem = ""  # what is wrong with the configuration (start() then raises Unsupported)
        link = (env.get("DOLPHIN_BOOTSTRAP", "") or env.get("DOLPHIN_WORKER_URL", "")).strip()
        if self.cmd:
            self.kind = "cmd"
        elif link:
            try:
                self.kind, self.source = classify_source(link, env.get("DOLPHIN_WORKER_KIND", ""))
            except ValueError as e:
                self.problem = f"{e}; paste the curlCommand or a URL from the Dolphin dashboard"
        self._last_tok: tuple[float, float] | None = None
        # health state
        self.phase = "idle"  # idle | downloading | install | run
        self.failure = ""  # install-thread failure (no process was started)
        self.connected = False
        self.lost_at: float | None = None  # when the worker last said its session dropped
        self.started_at: float | None = None
        self._metrics_ok_at: float | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self.note = ""

    # -- description / env ---------------------------------------------------------------------------------------
    def command(self) -> list[str]:
        if self.kind == "cmd":
            return self.cmd
        if self.kind == "binary":
            return [self.dest, *self.worker_args]
        if self.kind == "script":
            return ["bash", os.path.join(self.workdir, "bootstrap.sh")]
        if self.kind == "curl":
            return ["bash", "-o", "pipefail", "-c", with_fail_flag(self.source)]
        raise Unsupported(self._missing())

    def _missing(self) -> str:
        return self.problem or ("no Dolphin worker configured: set PM_SECRET_DOLPHIN_BOOTSTRAP (the curlCommand from the "
                                "Dolphin dashboard), PM_SECRET_DOLPHIN_WORKER_URL (its download or bootstrap URL) or "
                                "DOLPHIN_WORKER_CMD")

    def describe(self) -> str:
        if self.problem or self.kind is None:
            return f"(unsupported: {self._missing()})"
        if self.kind == "cmd":
            return super().describe()
        if self.kind == "curl":
            return "(runs the bootstrap curl command from DOLPHIN_BOOTSTRAP / DOLPHIN_WORKER_URL, hidden: it carries a key)"
        what = "bootstrap script" if self.kind == "script" else f"worker binary to {self.dest}"
        return f"(downloads the {what} from the Dolphin link in DOLPHIN_BOOTSTRAP / DOLPHIN_WORKER_URL, hidden, then runs it)"

    def popen_env(self) -> dict[str, str]:
        e = dict(os.environ)
        for k in ("DOLPHIN_WALLET", "DOLPHIN_API_KEY"):
            if self.env.get(k):
                e[k] = self.env[k]
        e.setdefault("DOLPHIN_WORKER_PATH", self.dest)
        return e

    def cwd(self) -> str | None:
        return self.workdir if self.kind in ("script", "curl") else None

    def secrets(self) -> list[str]:
        vals = [self.env.get("DOLPHIN_API_KEY", "")]
        for url in re.findall(r"https?://[^\s'\"]+", self.source):  # the whole link, then the tokens inside it
            vals.append(url)
            vals += [v for v in re.split(r"[&=]", url.partition("?")[2]) if len(v) >= 8]  # query values
        for tok in re.findall(r"[A-Za-z0-9_\-.~+/=]{16,}", self.source):  # header values, -d bodies, bearer tokens
            vals.append(tok)
        return [v for v in vals if v]

    # -- lifecycle -----------------------------------------------------------------------------------------------
    def start(self) -> None:
        if self.kind is None or self.problem:
            raise Unsupported(self._missing())
        with self._lock:
            self.parked = False
            self.paused = False
            self.paused_since = None
            self.failure = ""
            self.connected = False
            self.lost_at = None
            self._metrics_ok_at = None
            self._cancel = threading.Event()
            self.started_at = self.clock()
            if self.kind == "cmd":
                self.phase = "run"
                self._spawn(self.command())
            elif self.kind == "curl":
                self._ensure_dir()
                self.phase = "install"
                self._spawn(self.command())
            else:
                self.phase = "downloading"
                self._thread = threading.Thread(target=self._prepare, args=(self._cancel,), daemon=True)
                self._thread.start()

    def _ensure_dir(self) -> None:
        try:
            os.makedirs(self.workdir, exist_ok=True)
        except OSError as e:
            raise DriverError(f"dolphin: cannot create {self.workdir}: {e.strerror or e}") from e

    def _installing(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def _prepare(self, cancel: threading.Event) -> None:
        """Thread: fetch the binary/script (re-fetch every start, fall back to the earlier download), then spawn."""
        try:
            self._ensure_dir()
            path = self.dest if self.kind == "binary" else os.path.join(self.workdir, "bootstrap.sh")
            try:
                self._fetch(self.source, path, cancel)
            except DriverError as e:
                if cancel.is_set():
                    return
                if os.path.exists(path) and os.access(path, os.X_OK if self.kind == "binary" else os.R_OK):
                    self.note = f"re-download failed ({e}); reusing the earlier download"
                    print(f"dolphin: {self.note}", flush=True)
                else:
                    raise
            with self._lock:
                if cancel.is_set():
                    return
                self.phase = "run" if self.kind == "binary" else "install"
                self._spawn(self.command())
        except (DriverError, Unsupported) as e:
            if cancel.is_set():  # superseded by a newer start(): its state is not ours to overwrite
                return
            self.failure = self.redact(str(e).removeprefix("dolphin: "))
            print(f"dolphin: install failed: {self.failure}", flush=True)
        except Exception as e:  # noqa: BLE001 - never leave the slot in limbo
            if not cancel.is_set():
                self.failure = self.redact(f"install crashed: {type(e).__name__}: {e}")

    def _fetch(self, url: str, path: str, cancel: threading.Event) -> None:
        tmp = path + ".part"
        head = b""
        t_end = self.clock() + self.fetch_total_s
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pearl-node"})
            with urlopen(req, timeout=self.fetch_timeout_s) as r, open(tmp, "wb") as f:  # noqa: S310
                while chunk := r.read(1 << 20):
                    if cancel.is_set():
                        raise DriverError("cancelled")
                    if self.clock() > t_end:
                        raise DriverError(f"download too slow: not finished after {self.fetch_total_s / 60:.0f} min")
                    if len(head) < 256:
                        head += chunk[:256]
                    f.write(chunk)
        except urllib.error.HTTPError as e:
            raise DriverError(http_reason(e.code)) from e
        except urllib.error.URLError as e:
            raise DriverError(f"download failed: cannot reach the download server ({self.redact(str(e.reason))})") from e
        except OSError as e:
            raise DriverError(f"download failed: {self.redact(str(e))}") from e
        if not head.strip():
            raise DriverError("download failed: the link returned an empty file")
        if re.match(rb"\s*(<!doctype|<html)", head, re.I):
            raise DriverError("download failed: the link returned a web page, not the worker (expired or wrong link?)")
        try:
            os.replace(tmp, path)
            os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        except OSError as e:
            raise DriverError(f"download failed: cannot save the worker: {e.strerror or e}") from e

    def alive(self) -> bool:
        with self._lock:
            if self.parked or self._installing():
                return True
            p = self.proc
            if p is None:
                return False
            rc = p.poll()
            if rc is None:
                return True
            self.exit_code = rc
            # An installer that exits 0 and leaves the worker binary behind: run that binary next.
            if (self.phase == "install" and rc == 0 and not self._cancel.is_set() and os.path.isfile(self.dest)
                    and os.access(self.dest, os.X_OK)):
                try:
                    self.phase = "run"
                    self._spawn([self.dest, *self.worker_args])
                    return True
                except (DriverError, Unsupported) as e:
                    self.failure = self.redact(str(e).removeprefix("dolphin: "))
            return False

    def stop(self) -> None:
        self._cancel.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(5)
        super().stop()
        self.phase = "idle"

    # -- evidence / health ---------------------------------------------------------------------------------------
    def on_line(self, line: str) -> None:
        if self._lost.search(line):
            self.connected = False
            self.lost_at = self.clock()
        elif self._ready.search(line):
            self.connected = True
            self.lost_at = None

    def exit_reason(self) -> str:
        if self.failure:
            return self.failure
        rc = self.exit_code
        if self.phase == "install" and rc == 0:
            return ("bootstrap finished (exit 0) but left no worker at "
                    f"{self.dest} and nothing is running; set DOLPHIN_WORKER_CMD to the command that runs it")
        # Only an installer's output says anything about the download: a running worker that dies after printing
        # "HTTP 404" or "timed out" died of something else, so it gets the plain exit code and its last lines.
        known = explain_exit(" ".join(list(self._tail)[-12:])) if self.phase == "install" else None
        if known:
            return known + (f" (exit code {rc})" if rc is not None else "")
        return super().exit_reason().replace("dolphin ", "worker ", 1)

    def health(self) -> dict:
        """{'state': installing|starting|running|failed, 'reason': str, 'evidence': [str]}. Never says running without evidence."""
        if self._installing():
            waited = self.clock() - (self.started_at or self.clock())
            if waited > self.fetch_total_s + 30:  # the fetch loop checks its own deadline; this catches a hung read
                self._cancel.set()
                return {"state": "failed", "evidence": [],
                        "reason": f"download too slow: not finished after {waited / 60:.0f} min"}
            return {"state": "installing", "reason": "downloading the worker", "evidence": []}
        if self.failure:
            return {"state": "failed", "reason": self.failure, "evidence": []}
        if self.proc is None and not self.parked:
            return {"state": "starting", "reason": "", "evidence": []}
        if not self.alive():
            return {"state": "failed", "reason": self.exit_reason(), "evidence": []}
        ev = []
        if self.connected:
            ev.append("worker log: connected to lighthouse")
        if self._metrics_ok_at is not None and self.clock() - self._metrics_ok_at <= self.metrics_ttl_s:
            ev.append("metrics endpoint answers")
        if ev:
            return {"state": "running", "reason": "", "evidence": ev}
        if self.lost_at is not None:  # it was connected (or tried to) and said the session dropped
            gone = self.clock() - self.lost_at
            tail = self.log_tail()
            if gone < self.reconnect_grace_s:
                return {"state": "starting", "reason": "connection to lighthouse dropped, reconnecting", "evidence": []}
            return {"state": "failed", "evidence": [], "reason": (
                f"worker is not connected to lighthouse (it reported a dropped or failed connection {gone / 60:.0f} min ago)"
                + (f"; last output: {tail}" if tail else ""))}
        waited = self.clock() - (self.started_at or self.clock())
        if waited < self.connect_timeout_s:
            return {"state": "installing" if self.phase == "install" else "starting", "reason": "", "evidence": []}
        tail = self.log_tail()
        return {"state": "failed", "evidence": [], "reason": (
            f"worker is up but never reported 'connected to lighthouse' after {waited / 60:.0f} min"
            + (f"; last output: {tail}" if tail else ""))}

    def _metrics(self) -> dict[str, float] | None:
        try:
            m = parse_prom(http_get(self.metrics_url))
        except OSError:
            return None
        self._metrics_ok_at = self.clock()
        return m

    def load(self) -> float | None:
        m = self._metrics()
        if m is None:
            return None
        running = _pick(m, "num_requests_running") or 0.0
        waiting = _pick(m, "num_requests_waiting") or 0.0
        if waiting > 0:
            return 1.0
        return min(1.0, running / self.slots)

    def stats(self) -> dict:
        m = self._metrics()
        if m is None:
            return {}
        out: dict = {}
        tok = _pick(m, "generation_tokens_total")
        if tok is not None:
            now = self.clock()
            if self._last_tok and now > self._last_tok[0] and tok >= self._last_tok[1]:
                out["tokens_per_s"] = (tok - self._last_tok[1]) / (now - self._last_tok[0])
            self._last_tok = (now, tok)
        return out

    def throttle(self, level: float) -> None:
        """Foreground: never throttled."""
