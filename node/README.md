# Node image and agent

One image, tagged per arch (`ampere-ada`, `blackwell`) from the same build (research/miner.md section 6: SRBMiner ships its own kernels and links no CUDA toolkit, the host driver supplies `libcuda`). Base `nvidia/cuda:12.8.1-base-ubuntu22.04`; never overwrite `LD_LIBRARY_PATH`; no NVML on Salad. If the first real GPU run shows `libnvrtc`/`libcudart` load errors, switch the base to `-runtime-` (one line in `Dockerfile`).
Whatever network a market runs, the container is the same: `node/agent.py` is PID 1 and runs the plan's workloads together.

```
node/
  agent.py          PID 1, stdlib only: reads the plan, starts drivers, balances in real time, serves local status
  workername.py     verbatim copy of pearl_mining/workername.py (a test checks it)
  drivers/<net>.py  one Driver per network (pearl, dolphin; flops later)
  Dockerfile        SRBMiner 3.7.5 fetched with `ADD --checksum=sha256:...` (build fails on mismatch), python3-minimal, tini
  build.sh          local linux/amd64 build of both arch tags, never pushes
  VERSION           image version used in the tags
```

## Environment (set by the controller via Provider.set_desired -> Workload.env; see economics.compose_workload)
- `PM_PLAN` plan key (`solo:pearl`, `side:dolphin+pearl`, `useful:dolphin+pearl`); `PM_PARTS` = `network:mode,...` in plan order
  (first = foreground). Modes: `dedicated`, `foreground`, `background`, `useful`.
- `WORKER` (providers with per-instance env) or `PM_WORKER_PREFIX` + one of `SALAD_MACHINE_ID` / `SALAD_INSTANCE_ID` /
  `PM_INSTANCE_ID` -> `make_worker`. The worker name is passed to every driver as `PM_WORKER`.
- Per-network vars come from `Network.workload(...).env` (Pearl: `PEARL_POOL`, `PEARL_WALLET`, `PEARL_ALGO`, `PEARL_MODE`;
  Dolphin: `DOLPHIN_WALLET`, `DOLPHIN_API_KEY`, `DOLPHIN_MODE`).
- Balancing knobs (from `config.yaml node.agent`, as `PM_AGENT_<NAME>`): `TICK_S`, `STATUS_PORT`, `BUSY_ON_LOAD`, `IDLE_AFTER_S`,
  `THROTTLE_FLOOR`, `PAUSE_METHOD`, `RESTART_AFTER_S`, `VRAM_RESERVE_MB`, `RESTART_BACKOFF_S`, `NO_WORK_AFTER_S`.

## Agent contract
1. Parse `PM_PARTS`, build a Driver per part, start the foreground first (it needs its VRAM), reserve `VRAM_RESERVE_MB` for the
   background miner.
2. Every `TICK_S` seconds: read each driver's `load()` (0..1: vLLM `/metrics` running+waiting requests, GPU utilisation, or
   work-queue state), then `throttle()` the background driver: full intensity when the foreground has been below `BUSY_ON_LOAD` for
   `IDLE_AFTER_S`, `THROTTLE_FLOOR` (0 = paused) while it is busy. No restarts, no model reloads for seconds-scale changes; pauses
   longer than `RESTART_AFTER_S` may kill and respawn the miner instead of SIGSTOP (SIGSTOP can drop pool connections).
3. If the foreground reports no work for `NO_WORK_AFTER_S`, give the background the whole GPU (this is what makes a dead inference
   network degrade to Pearl-only with no controller action).
4. Restart a crashed workload with backoff `RESTART_BACKOFF_S`; count restarts in the status.
5. `useful:*` plans run ONE process (vLLM with the Pearl plugin); the Pearl driver is then a no-op that only reports hashrate.
6. Serve `GET /status` on `STATUS_PORT` (local; read by network plugins' telemetry where reachable): `{plan, worker, uptime_s, summary,
   ok, primary, workloads: [{network, mode, alive, state (pending|installing|starting|running|failed), reason, work?, restarts,
   gpu_share, tokens_per_s?, hashrate_ths?, throttle}]}`. A workload that is not running is `failed` with a reason, never "idle";
   `idle` only exists as `work` of a running foreground. Drivers may add `health() -> {state, reason, evidence}` and `exit_reason()`.

## Driver contract (`class Driver(env, mode)`)
`start()`, `stop()`, `alive() -> bool`, `load() -> float` (foreground: how busy; background: unused), `throttle(level: float)`
(0 pause .. 1 full; mechanism per driver: miner intensity/plimit flags, SIGSTOP/SIGCONT, CUDA MPS where available (likely not on
Salad's WSL2)), `stats() -> dict` (hashrate / tokens per second / last activity). Drivers never talk to the controller; networks
see the node only through their own telemetry (pool API, dashboards) or the local `/status`.

## Implementation notes
- Python 3.10 compatible (Ubuntu 22.04's python3), stdlib only. `agent.py` imports `drivers.<network>` and `workername` from its own directory.
- Entrypoint is `tini -- python3 /opt/node/agent.py`: tini reaps orphans and forwards SIGTERM; the agent stops the miner first, the model last, then exits 0.
- Exit codes: 0 clean stop, 2 configuration error (missing wallet or worker name: message on stderr), 3 every workload unsupported/failed permanently, but only after `EXIT_WHEN_FAILED_S` (900) of staying up with the status saying FAILED, so a provider never restart-loops a container that re-downloads a short-lived link each time (the controller's crash-loop check then takes the market to 0). A foreground whose configuration is invalid (e.g. no `DOLPHIN_WALLET`) is a FAILED workload, not an exit 2.
- `PM_PARTS` roles: `dedicated`/`useful` run alone; `foreground` + `background` is the side-by-side plan. Foreground starts first; the background waits until the foreground answers a load probe (max `PM_AGENT_BG_START_DELAY_S`, default 120) so the model gets its VRAM.
- Balancing (`Balancer`): load >= `BUSY_ON_LOAD` drops the miner to `THROTTLE_FLOOR` at once; it returns to full only after `IDLE_AFTER_S` below the threshold. Load unreadable while the foreground is alive -> duty `PM_AGENT_UNKNOWN_SHARE` (0.5). Foreground down/unsupported -> miner gets the whole GPU. Load exactly 0 for `NO_WORK_AFTER_S` -> state `no_work` (still full miner). Fractions are a SIGSTOP/SIGCONT duty cycle (1 s period). `PAUSE_METHOD=restart` kills the miner after `RESTART_AFTER_S` of pause and respawns it on resume.
- Restarts: backoff list `RESTART_BACKOFF_S`, capped at the last value (never a tight loop); the failure count resets after `STABLE_S` (600) healthy seconds. Missing binary / unsupported workload is permanent (no retries). A workload that is alive but reports `failed` (Dolphin up for 30 min without evidence, a download that is too slow) is stopped and started again with the same backoff, so the link is fetched afresh. Driver output is redacted (link, key, tokens) before it reaches stdout. Dolphin tuning: `DOLPHIN_FETCH_TOTAL_S` (900), `DOLPHIN_DISCONNECT_REGEX`, `DOLPHIN_RECONNECT_S` (300), `DOLPHIN_METRICS_TTL_S` (120).
- Not used: CUDA MPS (likely unavailable on Salad's WSL2), `--gpu-intensity` (unconfirmed for pearlhash).
- Local status: `GET http://127.0.0.1:8085/status` (`PM_AGENT_STATUS_HOST` to change). `gpu_share` is a 10 min moving average of the background intensity (miner) or foreground load.
- `DRY_RUN=1` prints the plan, worker and each driver's command (secrets travel by env and are never printed) and exits.

### Pearl driver
SRBMiner-MULTI `--algorithm pearlhash --disable-cpu --disable-gpu-amd --gpu-sensors-disable --pool $PEARL_POOL --wallet <PEARL_WALLET>.<worker> --api-enable --api-port 21550 ... --max-no-share-sent 900`, no overclock flags. `PEARL_EXTRA_ARGS` appends. `useful` mode runs nothing (the vLLM plugin lives in the inference process) and reports no hashrate. Stats come from the miner's local JSON API through a tolerant parser (layout unverified until the first GPU run).

### Dolphin driver
The worker is a closed per-key installer with no public image (research/dolphin_node.md), so it is not in the image. Sources, first
match wins: `DOLPHIN_WORKER_CMD` (a command line, e.g. a binary baked into a derived image); `DOLPHIN_BOOTSTRAP` / `DOLPHIN_WORKER_URL`
(the dashboard's worker bootstrap link). The link value may be any of the three things `POST /api/worker/bootstrap-links` returns:
- **`curlCommand`** (preferred, secret `PM_SECRET_DOLPHIN_BOOTSTRAP`): run verbatim with `bash -o pipefail -c` in `DOLPHIN_WORKER_DIR`
  (`--fail` is added to curl if missing, so an HTTP error is an error and not an HTML page piped into bash).
- **`bootstrapUrl`** (a URL whose path contains "bootstrap" or ends in `.sh`): downloaded to `bootstrap.sh` and run with bash.
- **`downloadUrl`** (any other URL): downloaded to `DOLPHIN_WORKER_PATH` (default `/opt/dolphin/dolphinpod-worker`), `chmod +x`, run.
`DOLPHIN_WORKER_KIND=binary|script|curl` overrides the guess. If an installer exits 0 and leaves an executable at `DOLPHIN_WORKER_PATH`,
that executable is run next (`DOLPHIN_WORKER_ARGS` appends arguments); an installer that exits 0 and leaves nothing is reported FAILED
with a hint to set `DOLPHIN_WORKER_CMD`. Installer formats are unverified until the first real run. Links may be short-lived: every
start re-downloads (a failed re-download falls back to the earlier file). Link text and the API key are never printed or put in the status.

Exact steps to get the link: log in at https://v2.dphn.ai, open the worker/node page, choose (or create) the API key, create a worker
bootstrap link, copy the `curlCommand` and set it as `PM_SECRET_DOLPHIN_BOOTSTRAP` (or copy `downloadUrl`/`bootstrapUrl` into
`PM_SECRET_DOLPHIN_WORKER_URL`); set `DOLPHIN_WALLET` to your account's wallet.

States (`GET /status` per workload, and `agent: dolphin ...` on stdout): `installing` (downloading / running the installer), `starting`
(process up, no evidence yet), `running` (evidence: a log line matching `DOLPHIN_READY_REGEX`, default `connected to lighthouse`, or the
metrics endpoint answers), `failed` with `reason`. A running worker has `work`: `busy`, `idle` (connected, no requests right now: healthy)
or `unknown` (load unreadable). FAILED reasons: no link configured, `download link rejected (HTTP 403)` (also 404, 410, 429, 5xx),
cannot reach the download server, `worker binary is not executable`, `exited with code N; last output: ...`, or up but never connected
after `DOLPHIN_CONNECT_TIMEOUT_S` (default 1800; the process is left running, it may still connect). A failed Dolphin is retried with
`RESTART_BACKOFF_S` (never a tight loop) and logged again every `PM_AGENT_FAILED_LOG_S` (300); the Pearl miner gets the whole GPU
meanwhile. With no link at all the driver is unsupported (permanent): status `dolphin FAILED: no Dolphin worker configured ...`.
Status also has `summary` (`dolphin FAILED: <reason> · pearl running`), `primary` and `ok`. A foreground whose configuration is invalid
(e.g. no `DOLPHIN_WALLET`) next to a background miner becomes a FAILED workload instead of a node exit, so the miner still runs.
Load comes from `DOLPHIN_METRICS_URL` (Prometheus text, default `http://127.0.0.1:2242/metrics`; whether the worker exposes it is
unconfirmed): `waiting > 0` -> 1.0, else running / `DOLPHIN_SLOTS`. No NVML fallback, because the side-by-side miner would drive GPU
utilisation itself. `useful` and `background` modes are refused.

## Building and pushing
```
node/build.sh            # local linux/amd64 build: ghcr.io/nmatare/pearl-node:<VERSION>-ampere-ada and -blackwell, nothing pushed
DRY_RUN=1 docker run --rm -e PM_PARTS=pearl:dedicated -e PEARL_WALLET=prl1p... -e WORKER=test ghcr.io/nmatare/pearl-node:0.1.2-ampere-ada
```
Pushing (not done by any script): `echo $GHCR_TOKEN | docker login ghcr.io -u <user> --password-stdin` with a classic PAT scoped `write:packages`, then
`docker push ghcr.io/nmatare/pearl-node:<VERSION>-ampere-ada` (and `-blackwell`).
- **Public package** (GitHub -> Packages -> pearl-node -> Package settings -> Change visibility): Salad pulls it with no credentials. The image holds only the pinned SRBMiner and our scripts; wallet, pool and keys arrive as env at runtime, never in the image. This is the simplest choice for the smoke test.
- **Private package**: create a PAT with `read:packages` only and give Salad `container.registry_authentication.basic {username, password}` on the container group. Never reuse the write token.
- Then set `node.images.*` in `config.yaml` to the pushed refs. Vast pulls the same refs (private needs registry login in the instance's `image_login`).
