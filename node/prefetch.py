"""Cold-start prefetch for the Dolphin worker (stdlib only, best effort, harmless when it fails).

The closed worker downloads its 5.5 GB runtime bundle and then the model (tens of GB) one after the other over single
connections. This module does both at node start, in parallel and over many ranged connections, into the places the worker
reads, so `dolphinpod-worker foreground` finds them and skips its own downloads:

  runtime   <worker dir>/text-v.tar.gz + text-v.tar.gz.sha256 (the side-loaded copy the worker verifies before it
            downloads; research/dolphin_worker_binary.md section 5). Skipped when /opt/dolphinpod/runtimes already holds
            this bundle's marker (a restarted warm instance) or the tarball is already here with the right sha256.
  model     a Hugging Face cache (models--<org>--<name>/{blobs,refs,snapshots}) under <XDG_CACHE_HOME>/huggingface/hub,
            every blob sha256-verified (LFS oid) or git-sha1-verified, plus links from the other places the worker might
            put HF_HOME (the exact value is unconfirmed: research section 5), and a watcher that logs, and when it can
            adopts, a cache directory the worker creates elsewhere.

Anything unexpected is logged and ignored: the worker then downloads for itself, exactly as it does without this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from http.client import HTTPException
from urllib.parse import quote, urlparse

DEFAULT_RUNTIME_URL = "https://updates.dphn.ai/runtimes/v2.4.3/text-v.tar.gz"
DEFAULT_AUTO_MODEL = "unsloth/Qwen3.8-27B-NVFP4"  # what the worker's model "auto" resolves to (research section 3.3)
RUNTIME_MARKER = "/opt/dolphinpod/runtimes/.dolphinpod-runtime.sha256"
CHUNK = 32 << 20
UA = "pearl-node-prefetch"


def log(msg: str) -> None:
    print(f"prefetch: {msg}", flush=True)


class PrefetchError(Exception):
    pass


class RangeUnsupported(PrefetchError):
    pass


class _StripAuth(urllib.request.HTTPRedirectHandler):
    """Never forward the Hugging Face token to the CDN host a download redirects to."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlparse(newurl).netloc != urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_opener = urllib.request.build_opener(_StripAuth)


def urlopen(req, timeout: float = 60):  # patched in tests
    return _opener.open(req, timeout=timeout)


def _get(url: str, headers: dict | None = None, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urlopen(req, timeout) as r:
        return r.read()


# ---------------------------------------------------------------------------------------------------------------
# ranged download
# ---------------------------------------------------------------------------------------------------------------
def probe_size(url: str, headers: dict | None = None) -> int:
    """Total size through a 1-byte ranged GET (works through redirects and on CDNs that answer HEAD badly)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0", **(headers or {})})
    with urlopen(req, 60) as r:
        cr = r.headers.get("Content-Range", "")
        m = re.search(r"/(\d+)\s*$", cr)
        if r.status == 206 and m:
            return int(m.group(1))
        if r.status == 200 and r.headers.get("Content-Length"):
            raise RangeUnsupported("server ignores Range")
    raise PrefetchError("cannot read the size")


def download_ranged(url: str, dest: str, size: int, *, headers: dict | None = None, workers: int = 8, chunk: int = CHUNK,
                    cancel: threading.Event | None = None, add: Callable[[int], None] = lambda n: None,
                    retries: int = 5, timeout: float = 60) -> None:
    """Fetch `size` bytes into dest over `workers` parallel Range requests (pwrite into a preallocated file). Falls back to
    one stream when the server does not honour Range. Raises PrefetchError."""
    cancel = cancel or threading.Event()
    hdr = {"User-Agent": UA, **(headers or {})}
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    fd = os.open(dest, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    done = [0]
    lock = threading.Lock()

    def bump(n: int) -> None:
        with lock:
            done[0] += n
        add(n)

    def one(off: int, ln: int) -> None:
        last: Exception | None = None
        for attempt in range(retries):
            if cancel.is_set():
                raise PrefetchError("cancelled")
            got = 0
            try:
                req = urllib.request.Request(url, headers={**hdr, "Range": f"bytes={off}-{off + ln - 1}"})
                with urlopen(req, timeout) as r:
                    if r.status != 206:
                        raise RangeUnsupported("server ignores Range")
                    while got < ln:
                        b = r.read(min(1 << 20, ln - got))
                        if not b:
                            raise OSError("short read")
                        os.pwrite(fd, b, off + got)
                        got += len(b)
                        bump(len(b))
                return
            except RangeUnsupported:
                raise
            except (OSError, HTTPException) as e:
                last = e
                bump(-got)  # this chunk starts over
                if cancel.wait(min(2.0 ** attempt, 10.0)):
                    raise PrefetchError("cancelled") from e
        raise PrefetchError(f"chunk at {off} failed after {retries} tries: {last}")

    try:
        os.ftruncate(fd, size)
        parts = [(o, min(chunk, size - o)) for o in range(0, size, chunk)]
        try:
            with ThreadPoolExecutor(max_workers=max(1, min(workers, len(parts) or 1))) as ex:
                futs = [ex.submit(one, o, n) for o, n in parts]
                try:
                    for f in as_completed(futs):
                        f.result()
                except BaseException:
                    for f in futs:
                        f.cancel()
                    raise
        except RangeUnsupported:
            bump(-done[0])
            os.ftruncate(fd, 0)
            os.ftruncate(fd, size)
            req = urllib.request.Request(url, headers=hdr)
            off = 0
            try:
                with urlopen(req, timeout) as r:
                    while b := r.read(1 << 20):
                        if cancel.is_set():
                            raise PrefetchError("cancelled")
                        os.pwrite(fd, b, off)
                        off += len(b)
                        bump(len(b))
            except (OSError, HTTPException) as e:
                raise PrefetchError(f"download failed: {e}") from e
            if off != size:
                raise PrefetchError(f"expected {size} bytes, got {off}")
    finally:
        os.close(fd)


def sha256_file(path: str, cancel: threading.Event | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(8 << 20):
            if cancel is not None and cancel.is_set():
                raise PrefetchError("cancelled")
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(path: str, size: int) -> str:
    h = hashlib.sha1(b"blob %d\0" % size, usedforsecurity=False)  # noqa: S324 - the Hugging Face file id, not security
    with open(path, "rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# config / progress
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class PrefetchConfig:
    worker_dir: str  # directory of the worker binary: the side-loaded runtime tarball goes here
    cache_home: str  # XDG_CACHE_HOME the worker runs with
    home: str  # HOME the worker runs with
    runtime_url: str = DEFAULT_RUNTIME_URL
    model: str = DEFAULT_AUTO_MODEL
    revision: str = "main"
    hf_endpoint: str = "https://huggingface.co"
    hf_token: str = ""
    runtime_workers: int = 8
    model_workers: int = 12  # per large file; up to model_files connections x this
    model_files: int = 3  # large files downloaded at once
    runtime_marker: str = RUNTIME_MARKER
    extra_homes: tuple[str, ...] = ()  # more places to link the HF cache from
    log_every_s: float = 15.0

    @property
    def hf_home(self) -> str:
        return os.path.join(self.cache_home, "huggingface")

    @property
    def hub_dir(self) -> str:
        return os.path.join(self.hf_home, "hub")

    def link_homes(self) -> list[str]:
        """Other places the worker might use as HF_HOME (unconfirmed), all pointing at the one real cache."""
        c = [os.path.join(self.cache_home, "dolphinpod", "huggingface"), os.path.join(self.cache_home, "dolphinpod", "hf"),
             os.path.join(self.cache_home, "hf"), os.path.join(self.home, ".cache", "huggingface"),
             os.path.join(self.home, "huggingface"), *self.extra_homes]
        return [p for p in dict.fromkeys(c) if p != self.hf_home]


@dataclass
class Progress:
    total: int = 0
    done: int = 0
    state: str = "pending"  # pending | skipped | running | done | failed
    note: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n

    def line(self, name: str) -> str:
        if self.state in ("done", "skipped", "failed"):
            return f"{name} {self.state}" + (f" ({self.note})" if self.note else "")
        gb = 1 << 30
        pct = f" {100 * self.done / self.total:.0f}%" if self.total else ""
        return f"{name} {self.done / gb:.1f}/{self.total / gb:.1f} GB{pct}"


# ---------------------------------------------------------------------------------------------------------------
# runtime bundle
# ---------------------------------------------------------------------------------------------------------------
def fetch_runtime(cfg: PrefetchConfig, prog: Progress, cancel: threading.Event) -> None:
    name = os.path.basename(urlparse(cfg.runtime_url).path) or "text-v.tar.gz"
    try:
        want = _get(cfg.runtime_url + ".sha256").decode("utf-8", "replace").split()[0].lower()
        if not re.fullmatch(r"[0-9a-f]{64}", want):
            raise PrefetchError("the runtime checksum file is not a sha256")
    except (OSError, HTTPException, IndexError, PrefetchError) as e:
        raise PrefetchError(f"cannot read the runtime checksum: {e}") from e
    try:
        with open(cfg.runtime_marker) as f:
            if f.read().strip().split()[:1] == [want]:
                prog.state, prog.note = "skipped", "runtime already installed"
                return
    except OSError:
        pass
    dest = os.path.join(cfg.worker_dir, name)
    sidecar = dest + ".sha256"
    if os.path.isfile(dest):
        try:
            if sha256_file(dest, cancel) == want:
                _write_sidecar(sidecar, want, name)
                prog.state, prog.note = "skipped", "tarball already here"
                return
        except OSError:
            pass
    size = probe_size(cfg.runtime_url)
    prog.total = size
    _check_space(cfg.worker_dir, size)
    part = dest + ".part"
    t0 = time.monotonic()
    download_ranged(cfg.runtime_url, part, size, workers=cfg.runtime_workers, cancel=cancel, add=prog.add)
    if sha256_file(part, cancel) != want:
        os.unlink(part)
        raise PrefetchError("runtime checksum mismatch")
    os.replace(part, dest)
    _write_sidecar(sidecar, want, name)
    prog.note = f"{size / (1 << 30):.1f} GB in {time.monotonic() - t0:.0f}s"


def _write_sidecar(path: str, digest: str, name: str) -> None:
    with open(path, "w") as f:
        f.write(f"{digest}  {name}\n")


def _check_space(path: str, need: int) -> None:
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < need * 1.05:
        raise PrefetchError(f"not enough disk in {path}: {free >> 30} GB free, {need >> 30} GB needed")


# ---------------------------------------------------------------------------------------------------------------
# model (Hugging Face cache layout)
# ---------------------------------------------------------------------------------------------------------------
def _hf_headers(cfg: PrefetchConfig) -> dict:
    return {"Authorization": f"Bearer {cfg.hf_token}"} if cfg.hf_token else {}


def list_model(cfg: PrefetchConfig) -> tuple[str, list[dict]]:
    """(commit sha, files[{path, size, blob, lfs}]) for cfg.model@cfg.revision."""
    base = f"{cfg.hf_endpoint.rstrip('/')}/api/models/{quote(cfg.model, safe='/')}"
    h = _hf_headers(cfg)
    rev = quote(cfg.revision, safe="")
    commit = json.loads(_get(f"{base}/revision/{rev}", h)).get("sha", "")
    tree = json.loads(_get(f"{base}/tree/{rev}?recursive=true", h))
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        raise PrefetchError("cannot resolve the model revision")
    files = []
    for e in tree:
        if e.get("type") != "file":
            continue
        lfs = e.get("lfs") or {}
        blob = lfs.get("oid") or e.get("oid")
        size = int(lfs.get("size") or e.get("size") or 0)
        if not blob or not isinstance(e.get("path"), str) or ".." in e["path"].split("/") or e["path"].startswith("/"):
            raise PrefetchError("unexpected file entry in the model listing")
        files.append({"path": e["path"], "size": size, "blob": blob, "lfs": bool(lfs.get("oid"))})
    if not files:
        raise PrefetchError("the model listing has no files")
    return commit, files


def repo_dir(hub_dir: str, model: str) -> str:
    return os.path.join(hub_dir, "models--" + model.replace("/", "--"))


def _fetch_blob(cfg: PrefetchConfig, f: dict, blobs: str, prog: Progress, cancel: threading.Event) -> None:
    final = os.path.join(blobs, f["blob"])
    if os.path.isfile(final) and os.path.getsize(final) == f["size"]:
        prog.add(f["size"])
        return
    url = f"{cfg.hf_endpoint.rstrip('/')}/{quote(cfg.model, safe='/')}/resolve/{quote(cfg.revision, safe='')}/{quote(f['path'])}"
    tmp = final + ".incomplete"
    if f["size"] == 0:
        open(tmp, "wb").close()
    else:
        download_ranged(url, tmp, f["size"], headers=_hf_headers(cfg), workers=cfg.model_workers, cancel=cancel, add=prog.add)
    got = sha256_file(tmp, cancel) if f["lfs"] else git_blob_sha1(tmp, f["size"])
    if got != f["blob"]:
        os.unlink(tmp)
        raise PrefetchError(f"checksum mismatch for {f['path']}")
    os.replace(tmp, final)


def fetch_model(cfg: PrefetchConfig, prog: Progress, cancel: threading.Event) -> None:
    commit, files = list_model(cfg)
    prog.total = sum(f["size"] for f in files)
    root = repo_dir(cfg.hub_dir, cfg.model)
    blobs = os.path.join(root, "blobs")
    snap = os.path.join(root, "snapshots", commit)
    marker = os.path.join(root, ".pm-prefetch-complete")
    if _read(marker) == commit:
        prog.add(prog.total)
        prog.state, prog.note = "skipped", "model already cached"
        link_homes(cfg)
        return
    os.makedirs(blobs, exist_ok=True)
    _check_space(blobs, max(prog.total - sum(os.path.getsize(os.path.join(blobs, x)) for x in os.listdir(blobs)
                                              if not x.endswith(".incomplete")), 0))
    t0 = time.monotonic()
    ordered = sorted(files, key=lambda f: -f["size"])
    with ThreadPoolExecutor(max_workers=max(1, cfg.model_files)) as ex:
        futs = [ex.submit(_fetch_blob, cfg, f, blobs, prog, cancel) for f in ordered]
        try:
            for fu in as_completed(futs):
                fu.result()
        except BaseException:
            for fu in futs:
                fu.cancel()
            raise
    for f in files:
        link = os.path.join(snap, f["path"])
        os.makedirs(os.path.dirname(link), exist_ok=True)
        target = os.path.relpath(os.path.join(blobs, f["blob"]), os.path.dirname(link))
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(target, link)
    os.makedirs(os.path.join(root, "refs"), exist_ok=True)
    with open(os.path.join(root, "refs", cfg.revision), "w") as fh:
        fh.write(commit)
    with open(marker, "w") as fh:
        fh.write(commit)
    link_homes(cfg)
    prog.note = f"{prog.total / (1 << 30):.1f} GB in {time.monotonic() - t0:.0f}s"


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def link_homes(cfg: PrefetchConfig) -> None:
    """models--X also directly under hf_home (if the worker treats HF_HOME as the hub dir), and the other candidate HF homes
    pointing at the one real cache. Never replaces something that already exists."""
    root = repo_dir(cfg.hub_dir, cfg.model)
    flat = os.path.join(cfg.hf_home, os.path.basename(root))
    _symlink(os.path.relpath(root, cfg.hf_home), flat)
    for alt in cfg.link_homes():
        _symlink(cfg.hf_home, alt)


def _symlink(target: str, link: str) -> None:
    try:
        if os.path.lexists(link):
            return
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(target, link)
    except OSError as e:
        log(f"cannot link {link}: {e.strerror or e}")


def adopt_foreign_cache(cfg: PrefetchConfig) -> str | None:
    """The worker created its own models--X directory somewhere we did not guess: log where, and when it is still (nearly)
    empty replace it with a link to ours, so the rest of its download is skipped. Returns the path it found."""
    name = os.path.basename(repo_dir(cfg.hub_dir, cfg.model))
    mine = os.path.realpath(repo_dir(cfg.hub_dir, cfg.model))
    for top in {cfg.cache_home, cfg.home}:
        base_depth = top.rstrip("/").count("/")
        for dp, dns, _ in os.walk(top):
            if dp.count("/") - base_depth > 5:
                dns[:] = []
                continue
            if name in dns:
                found = os.path.join(dp, name)
                if os.path.realpath(found) == mine:
                    dns.remove(name)
                    continue
                size = sum(os.path.getsize(os.path.join(a, b)) for a, _, bs in os.walk(found) for b in bs
                           if os.path.isfile(os.path.join(a, b)))
                if size < (64 << 20):
                    try:
                        shutil.rmtree(found)
                        os.symlink(mine, found)
                        log(f"the worker looks for the model in {found}; linked it to the prefetched cache")
                        return found
                    except OSError as e:
                        log(f"the worker looks for the model in {found}; cannot link it: {e.strerror or e}")
                else:
                    log(f"the worker is already downloading the model into {found}")
                return found
    return None


# ---------------------------------------------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------------------------------------------
class Prefetcher:
    """Runs runtime and model downloads in parallel threads. Never raises; wait() says whether both are in place."""

    def __init__(self, cfg: PrefetchConfig):
        self.cfg = cfg
        self.runtime = Progress()
        self.model = Progress()
        self.cancel_ev = threading.Event()
        self._threads: list[threading.Thread] = []
        self.started_at = 0.0

    def start(self) -> None:
        self.started_at = time.monotonic()
        for name, fn, prog in (("runtime", fetch_runtime, self.runtime), ("model", fetch_model, self.model)):
            t = threading.Thread(target=self._run, args=(name, fn, prog), daemon=True, name=f"prefetch-{name}")
            self._threads.append(t)
            t.start()
        t = threading.Thread(target=self._report, daemon=True, name="prefetch-report")
        self._threads.append(t)
        t.start()

    def _run(self, name: str, fn: Callable, prog: Progress) -> None:
        prog.state = "running"
        try:
            fn(self.cfg, prog, self.cancel_ev)
            if prog.state == "running":
                prog.state = "done"
        except Exception as e:  # noqa: BLE001 - best effort by design
            if prog.state != "skipped":
                prog.state, prog.note = "failed", str(e)[:200]
            log(f"{name} failed ({prog.note}); the worker will download it itself")

    def _report(self) -> None:
        while not self.cancel_ev.wait(self.cfg.log_every_s):
            if self.finished():
                break
            log(self.summary())
        log(self.summary())

    def finished(self) -> bool:
        return all(p.state in ("done", "skipped", "failed") for p in (self.runtime, self.model))

    def ok(self) -> bool:
        return all(p.state in ("done", "skipped") for p in (self.runtime, self.model))

    def summary(self) -> str:
        return f"{self.runtime.line('runtime')}, {self.model.line('model')}"

    def wait(self, timeout: float, until: Callable[[], bool] | None = None) -> bool:
        """Block until both parts finished (or `until()` says stop, or timeout). True when both are in place."""
        end = time.monotonic() + timeout
        while time.monotonic() < end and not self.finished():
            if until is not None and until():
                break
            time.sleep(0.2)
        return self.ok()

    def cancel(self) -> None:
        self.cancel_ev.set()
