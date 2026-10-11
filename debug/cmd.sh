#!/bin/bash
# v4: why does the Dolphin worker sit after "selected GPUs"?
exec 2>&1
echo "== v4 $(date -u +%T)"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl procps iproute2 >/dev/null 2>&1; which curl ss pgrep strace
ps -eo pid,ppid,etime,stat,pcpu,rss,cmd --sort=start_time | cut -c1-200 | head -30
echo "--- sockets"; (ss -tnp 2>/dev/null || netstat -tnp 2>/dev/null) | head -20
W=$(pgrep -f dolphinpod-worker | head -1); echo "worker pid=$W"
[ -n "$W" ] && { ls -la /proc/$W/fd 2>/dev/null | head -20; cat /proc/$W/status | grep -E "State|Threads|VmRSS"; timeout 8 strace -f -p $W -e trace=network,openat,read,write -o /tmp/w.strace 2>&1 | head -3; tail -25 /tmp/w.strace 2>/dev/null | cut -c1-220; }
echo "--- dirs"; du -sh /opt/dolphinpod /var/lib/dolphin /tmp 2>/dev/null; ls -la /opt/dolphinpod/runtimes /var/lib/dolphin/cache 2>&1 | head
echo "--- network to updates.dphn.ai"
curl -sS -o /dev/null -w "head: %{http_code} %{time_total}s\n" -I https://updates.dphn.ai/runtimes/v2.4.3/text-v.tar.gz 2>&1 | tail -2
timeout 20 curl -sS -r 0-52428799 -o /dev/null -w "50MB range: %{http_code} %{speed_download} B/s %{time_total}s\n" https://updates.dphn.ai/runtimes/v2.4.3/text-v.tar.gz 2>&1 | tail -2
echo "== v4 done"
