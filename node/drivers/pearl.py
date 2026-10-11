"""Pearl driver: dedicated SRBMiner (pearlhash) through a pool, or a reporting-only stub for the vLLM plugin mode.

Env: PEARL_WALLET (required for dedicated/background), PEARL_POOL, PEARL_ALGO, PEARL_MODE, PM_WORKER (set by the agent),
PEARL_MINER_BIN, PEARL_API_PORT, PEARL_EXTRA_ARGS. Flags per research/miner.md (3.7.5): no overclocking, no --pearl-k*.
Throttling is SIGSTOP/SIGCONT (intensity flags are unconfirmed for pearlhash); see ProcessDriver.
"""
from __future__ import annotations

import json

from . import ConfigError, ProcessDriver, http_get, split_args

DEFAULT_POOL = "prl.kryptex.network:7048"


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
        self.bin = env.get("PEARL_MINER_BIN", "SRBMiner-MULTI")
        self.api_port = int(env.get("PEARL_API_PORT", "21550"))

    def command(self) -> list[str]:
        login = f"{self.wallet}.{self.worker}" if self.worker else self.wallet
        cmd = [self.bin, "--algorithm", self.algo, "--disable-cpu", "--disable-gpu-amd", "--gpu-sensors-disable",
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
        try:
            return parse_miner_stats(json.loads(http_get(f"http://127.0.0.1:{self.api_port}/")))
        except (OSError, ValueError):
            return {}

