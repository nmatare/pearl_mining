#!/bin/bash
# v3: which SRBMiner invocation triggers "Unsupported runtime environment"?
exec 2>&1
S=/opt/srb/SRBMiner-MULTI
W="${PEARL_WALLET}.dbgtest2"
BASE="--algorithm pearlhash --disable-cpu --disable-gpu-amd --pool prl.kryptex.network:7048 --wallet $W --password x"
try() { # label, prefix-cmd, args...
  local l="$1" pre="$2"; shift 2
  echo "--- $l"
  $pre $S "$@" > /tmp/v.txt 2>&1 &
  local p=$!
  sleep 15
  if kill -0 $p 2>/dev/null; then echo "ALIVE after 15 s"; else wait $p; echo "EXITED rc=$?"; fi
  head -c 1200 /tmp/v.txt
  kill $p 2>/dev/null; sleep 3; kill -9 $p 2>/dev/null; sleep 1
}
echo "== v2 srbminer variants"
try "A agent args, plain" "" $BASE --gpu-sensors-disable --api-enable --api-port 21550 --api-rig-name dbg --give-up-limit 3 --retry-time 10 --main-pool-reconnect 120 --max-no-share-sent 900
try "B agent args, setsid" "setsid" $BASE --gpu-sensors-disable --api-enable --api-port 21550 --api-rig-name dbg --give-up-limit 3 --retry-time 10 --main-pool-reconnect 120 --max-no-share-sent 900
try "C base + gpu-sensors-disable only" "" $BASE --gpu-sensors-disable
try "D base + api only" "" $BASE --api-enable --api-port 21550 --api-rig-name dbg
try "E base + give-up/retry only" "" $BASE --give-up-limit 3 --retry-time 10 --main-pool-reconnect 120 --max-no-share-sent 900
echo "== v2 done"
