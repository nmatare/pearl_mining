"""Pearl driver: dedicated SRBMiner (pearlhash) through a pool, or a reporting-only stub for the vLLM plugin mode.

Env: PEARL_WALLET (required for dedicated/background), PEARL_POOL, PEARL_ALGO, PEARL_MODE, PM_WORKER (set by the agent),
PEARL_MINER_BIN, PEARL_API_PORT, PEARL_EXTRA_ARGS, PEARL_GPU_VENDOR (nvidia|amd; default amd when PM_NODE_ARCH=amd,
else nvidia), PEARL_MINER (srbminer|krig; default srbminer; krig is the Kryptex ROCm miner shipped in the -amd image). Flags per research/miner.md (3.7.5): no overclocking, no --pearl-k*.
Throttling is SIGSTOP/SIGCONT (intensity flags are unconfirmed for pearlhash); see ProcessDriver.
"""
from __future__ import annotations

import json
import os

from . import ConfigError, ProcessDriver, http_get, split_args

DEFAULT_POOL = "prl.kryptex.network:7048"
KRIG_POOL = "prl.kryptex.network:8048"  # the TLS port from the Krig README
KRIG_BIN = "/opt/krig/krig-miner"


def parse_miner_stats(doc: object) -> dict:
    """Defensive SRBMiner JSON reader: the pearlhash layout is unverified (research/miner.md), so look for any
    `hashrate` object with a total and any accepted/rejected counters instead of a fixed path. H/s -> TH/s."""
    hashes: list[float] = []
    shares: dict[str, float] = {}

    def num(v: object) -> float | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        if isinstance(v, list) and v:
            return num(v[0])
        if isinstance(v, dict):
            for x in v.values():
                n = num(x)
                if n is not None:
                    return n
        return None

    def walk(o: object, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{path}/{k}".lower()
                kl = str(k).lower()
                if kl in ("accepted", "rejected") and (n := num(v)) is not None:
                    shares[kl] = shares.get(kl, 0.0) + n
                if "hashrate" in p and kl.startswith("total") and (n := num(v)) is not None:
                    hashes.append(n)
                walk(v, p)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(doc, "")
    out: dict = {}
    if hashes:
        out["hashrate_ths"] = max(hashes) / 1e12
    out.update({k: int(v) for k, v in shares.items()})
    return out


class Driver(ProcessDriver):
    name = "pearl"

    def __init__(self, env, mode, **kw):
        super().__init__(env, mode, **kw)
        if mode == "foreground":
            raise ConfigError("pearl: mode 'foreground' is not valid (use dedicated, background or useful)")
        self.wallet = env.get("PEARL_WALLET", "").strip()
        if mode != "useful" and not self.wallet:
            raise ConfigError("pearl: PEARL_WALLET is not set (the pool payout address, prl1p...); refusing to mine "
                              "without one")
        self.worker = env.get("PM_WORKER", "")
        self.pool = env.get("PEARL_POOL", DEFAULT_POOL)
        self.algo = env.get("PEARL_ALGO", "pearlhash")
        # absolute path: launched via the /usr/local/bin symlink SRBMiner aborts with "Unsupported runtime environment" (F02/L04)
        self.vendor = (env.get("PEARL_GPU_VENDOR") or ("amd" if env.get("PM_NODE_ARCH") == "amd" else "nvidia")).lower()
        if self.vendor not in ("nvidia", "amd"):
            raise ConfigError(f"pearl: PEARL_GPU_VENDOR must be nvidia or amd, got {self.vendor!r}")
        self.miner = (env.get("PEARL_MINER") or "srbminer").lower()
        if self.miner not in ("srbminer", "krig"):
            raise ConfigError(f"pearl: PEARL_MINER must be srbminer or krig, got {self.miner!r}")
        if self.miner == "krig":
            self.pool = env.get("PEARL_POOL", KRIG_POOL)
        self.bin = env.get("PEARL_MINER_BIN", KRIG_BIN if self.miner == "krig" else "/opt/srb/SRBMiner-MULTI")
        self.api_port = int(env.get("PEARL_API_PORT", "21550"))

    def cwd(self) -> str | None:
        return os.path.dirname(self.bin) or None  # run from the miner's own dir, as SRBMiner expects

    def command(self) -> list[str]:
        if self.miner == "krig":
            cmd = [self.bin, "--no-tui", "-u", f"{self.wallet}/{self.worker}" if self.worker else self.wallet,
                   "-o", self.pool, "--api-port", str(self.api_port)]
            if self.vendor == "amd":
                cmd.append("--no-cuda")
            return cmd + split_args(self.env.get("PEARL_EXTRA_ARGS", ""))
        # SRBMiner: switch off the other vendor's backend (the NVIDIA-only flag set would be wrong on AMD and vice versa)
        other = "--disable-gpu-nvidia" if self.vendor == "amd" else "--disable-gpu-amd"
        login = f"{self.wallet}.{self.worker}" if self.worker else self.wallet
        cmd = [self.bin, "--algorithm", self.algo, "--disable-cpu", other,
               "--pool", self.pool, "--wallet", login, "--password", "x",
               "--api-enable", "--api-port", str(self.api_port)]
        if self.worker:
            cmd += ["--api-rig-name", self.worker]
        cmd += ["--give-up-limit", "3", "--retry-time", "10", "--main-pool-reconnect", "120",
                "--max-no-share-sent", "900"]
        return cmd + split_args(self.env.get("PEARL_EXTRA_ARGS", ""))

    def describe(self) -> str:
        if self.mode == "useful":
            return "(no process: mining happens inside the inference engine via the vLLM Pearl plugin)"
        return super().describe()

    # 'useful': the inference process mines; nothing for us to run, pause or restart.
    def start(self) -> None:
        if self.mode != "useful":
            super().start()

    def stop(self) -> None:
        if self.mode != "useful":
            super().stop()

    def alive(self) -> bool:
        return True if self.mode == "useful" else super().alive()

    def throttle(self, level: float) -> None:
        if self.mode != "useful":
            super().throttle(level)

    def stats(self) -> dict:
        if self.mode == "useful" or self.proc is None:
            return {}
        if self.miner == "krig":
            return {}  # Krig's API layout is unmeasured; the pool side still sees the worker
        try:
            return parse_miner_stats(json.loads(http_get(f"http://127.0.0.1:{self.api_port}/")))
        except (OSError, ValueError):
            return {}

