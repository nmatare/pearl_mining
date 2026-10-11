#!/bin/bash
exec 2>&1
S=/opt/srb/SRBMiner-MULTI
W="${PEARL_WALLET}.dbgtest"
ARGS="--algorithm pearlhash --disable-cpu --disable-gpu-amd --pool prl.kryptex.network:7048 --wallet $W --password x"
run() { # label, command...
  local l="$1"; shift; echo "--- $l"; "$@" > /tmp/o.txt 2>&1; echo "rc=$?"; head -c 1500 /tmp/o.txt; echo
}
echo "== srbminer experiments"
grep -m1 -E "model name" /proc/cpuinfo; grep -m1 -o -w hypervisor /proc/cpuinfo; cat /sys/class/dmi/id/product_name 2>&1 | head -1
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
run "list-devices plain" $S --list-devices
run "list-devices no LD_LIBRARY_PATH" env -u LD_LIBRARY_PATH $S --list-devices
run "list-devices under timeout" timeout 20 $S --list-devices
(timeout 25 $S $ARGS > /tmp/o2.txt 2>&1; echo "rc=$?" >> /tmp/o2.txt) ; echo "--- mine 25 s plain"; head -c 2500 /tmp/o2.txt
echo "--- cuda libs present?"; ldconfig -p | grep -E "libcuda|libnvrtc|libcudart|libnvidia-ml" | head
echo "== install nvrtc/cudart and retry"
apt-get update -qq 2>&1 | tail -1
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-nvrtc-12-8 cuda-cudart-12-8 2>&1 | tail -2
ldconfig; ldconfig -p | grep -E "libnvrtc|libcudart" | head -4
(timeout 25 $S $ARGS > /tmp/o3.txt 2>&1; echo "rc=$?" >> /tmp/o3.txt) ; echo "--- mine 25 s after nvrtc"; head -c 2500 /tmp/o3.txt
echo "== sonar"
R=/opt/dolphinpod/runtimes/text-v
for i in $(seq 1 80); do [ -e $R/run-sonar ] && break; sleep 5; done
sleep 20
ls -la $R | head -30
file $R/run-sonar 2>&1 | head -2; head -c 1500 $R/run-sonar | cat -v | head -40
echo "--- run-sonar --help"
timeout 90 $R/run-sonar --help > /tmp/s1.txt 2>&1; echo "rc=$?"; head -c 3000 /tmp/s1.txt
echo "--- run-sonar serve --help"
timeout 90 $R/run-sonar serve --help > /tmp/s2.txt 2>&1; echo "rc=$?"; head -c 3000 /tmp/s2.txt
echo "== done v1"
