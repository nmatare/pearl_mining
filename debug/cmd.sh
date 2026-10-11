#!/bin/bash
# v5: is the slow runtime download the disk, the host network, or updates.dphn.ai?
exec 2>&1
echo "== v5 $(date -u +%T)"
U=https://updates.dphn.ai/runtimes/v2.4.3/text-v.tar.gz
echo "--- disk write speed"; dd if=/dev/zero of=/opt/ddtest bs=1M count=1500 oflag=direct 2>&1 | tail -1; rm -f /opt/ddtest
echo "--- cloudflare speed test (100 MB)"; timeout 40 curl -sS -o /dev/null -w "%{http_code} %{speed_download} B/s %{time_total}s\n" "https://speed.cloudflare.com/__down?bytes=100000000" 2>&1 | tail -1
echo "--- updates.dphn.ai ranges (60 MB each, 25 s max)"
for off in 0 2000000000 4000000000; do
  timeout 25 curl -sS -r $off-$((off+59999999)) -o /dev/null -w "off=$off %{http_code} %{speed_download} B/s %{size_download} bytes %{time_total}s\n" $U 2>&1 | tail -1
done
curl -sI $U | grep -i -E "cf-cache-status|cf-ray|age|server|content-length" 
echo "--- worker progress"; du -sh /opt/dolphinpod; ls -la /opt/dolphinpod/runtimes/.text-v-*/text-v 2>/dev/null | head -5
echo "== v5 done"
