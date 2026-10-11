#!/bin/bash
# Diagnostics for the Pearl node image on a rented GPU (no secrets in here; run by the agent via PM_DEBUG_URL).
exec 2>&1
echo "== host"; date -u; uname -a; id; cat /proc/driver/nvidia/version 2>/dev/null | head -2
nvidia-smi 2>&1 | head -16
df -h / /tmp /dev/shm 2>&1 | cat; free -m; nproc; ulimit -a | head -12
echo "== srbminer"
ls -la /opt/srb | head; ldd /opt/srb/SRBMiner-MULTI 2>&1 | head -20
timeout 30 /opt/srb/SRBMiner-MULTI --list-devices 2>&1 | head -40
echo "== srbminer mining attempt (30 s)"
timeout 30 /opt/srb/SRBMiner-MULTI --algorithm pearlhash --disable-cpu --disable-gpu-amd --pool prl.kryptex.network:7048 --wallet "${PEARL_WALLET}.dbgtest" --password x 2>&1 | head -60
echo "== tools"; apt-get update -qq 2>&1 | tail -1; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq strace file procps 2>&1 | tail -1
echo "== srbminer under strace"
timeout 30 strace -f -o /tmp/srb.strace /opt/srb/SRBMiner-MULTI --algorithm pearlhash --disable-cpu --disable-gpu-amd --pool prl.kryptex.network:7048 --wallet "${PEARL_WALLET}.dbgtest" --password x >/tmp/srb.out 2>&1
head -30 /tmp/srb.out; echo "--- strace: failed syscalls and last lines"; grep -E " = -1 E" /tmp/srb.strace | grep -v -E "ENOENT.*(\.so|ld\.so|locale|gconv)" | tail -25; tail -15 /tmp/srb.strace
echo "== wait for the dolphin worker to download its runtime and fail/start"; sleep 200
echo "== dolphin tree"; ls -la /opt/dolphinpod /opt/dolphinpod/* 2>&1 | head -40; du -sh /opt/dolphinpod/* /var/lib/dolphin/* 2>&1 | head
find /opt/dolphinpod -maxdepth 3 -type f \( -name 'sonar*' -o -name 'run-sonar' -o -name '*.sha256' \) 2>/dev/null | head
echo "== recent logs"; for f in $(find / -xdev -name '*.log' -mmin -15 -size +0 2>/dev/null | grep -v -E '^/(proc|sys)' | head -8); do echo "--- $f"; tail -n 40 "$f"; done
ls -la /tmp | head -30; ls -la /var/lib/dolphin/cache/dolphinpod 2>&1 | head
S=$(find /opt/dolphinpod -maxdepth 4 -type f -name 'sonar' 2>/dev/null | head -1); echo "sonar=$S"
if [ -n "$S" ]; then file "$S"; ldd "$S" 2>&1 | head -30; timeout 60 "$S" --version 2>&1 | head -20; timeout 60 "$S" serve --help 2>&1 | head -60; fi
echo "== done"
