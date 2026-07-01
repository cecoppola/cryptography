#!/usr/bin/env bash
# crash_heartbeat.sh — stream a high-rate heartbeat to /dev/kmsg (→ netconsole)
# so the LAST line received before a silent freeze pinpoints the freeze moment
# (±interval) and the system state at that instant.
#
# Each beat carries: seq, monotonic ms, MemAvailable, 1-min load, and the
# contents of a progress file the workload updates (e.g. current mul index).
# Writes go through a process-substituted `sudo tee /dev/kmsg` (NOPASSWD); a
# TERM/INT/EXIT trap closes that fd so the root tee child exits cleanly — no
# leaked heartbeat/tee processes across runs.
#
# Usage:  scripts/crash_heartbeat.sh [progress_file] [interval_s]
set -u
PROG="${1:-/tmp/crash_progress}"
IVAL="${2:-0.25}"
TAG="CRASHHUNT_HB"

# fd 3 → root tee → /dev/kmsg; closing it (trap) sends EOF so tee exits.
exec 3> >(sudo -n tee /dev/kmsg >/dev/null 2>&1)
cleanup() { exec 3>&- 2>/dev/null; exit 0; }
trap cleanup TERM INT EXIT

seq=0
while :; do
    seq=$((seq + 1))
    mono=$(awk '{printf "%d", $1*1000}' /proc/uptime 2>/dev/null)
    mem=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null)
    load=$(awk '{print $1}' /proc/loadavg 2>/dev/null)
    p=$(tr -d '\n' < "$PROG" 2>/dev/null)
    printf '<6>%s seq=%d mono_ms=%s memavail_kb=%s load=%s prog=[%s]\n' \
           "$TAG" "$seq" "${mono:-0}" "${mem:-0}" "${load:-0}" "${p:-}" >&3
    sleep "$IVAL"
done
