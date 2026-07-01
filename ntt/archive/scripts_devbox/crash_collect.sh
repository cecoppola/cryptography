#!/usr/bin/env bash
# crash_collect.sh — called by the receiver (littleblue) over SSH immediately after
# a crash is detected. Dumps everything useful to stdout in one shot. The receiver
# saves the output to ~/crash-captures/<timestamp>/crash_collect.txt.
#
# Designed to run fast (< 8s total) and tolerate partial failures — each section
# is independent. Called with NOPASSWD sudo so no password prompt.
#
# Receiver invocation:
#   ssh machinus@192.168.50.226 'sudo bash ~/ntt/scripts/crash_collect.sh'
set -u

TS=$(date '+%Y-%m-%d %H:%M:%S')
TR=/sys/kernel/tracing; [ -d "$TR" ] || TR=/sys/kernel/debug/tracing
# Auto-detect the amdgpu debugfs DRI node (the GPU is card1 here, NOT dri/0).
DRI=/sys/kernel/debug/dri/0
for _d in /sys/kernel/debug/dri/*; do
    [ -e "$_d/amdgpu_fence_info" ] && { DRI="$_d"; break; }
done
# Auto-detect the amdgpu card sysfs node (for dpm/clock reads below).
CARD=/sys/class/drm/card0
for _c in /sys/class/drm/card[0-9]*; do
    grep -qi amdgpu "$_c/device/uevent" 2>/dev/null && { CARD="$_c"; break; }
done

section() { printf '\n\n=== %s [%s] ===\n' "$1" "$(date '+%H:%M:%S.%3N')"; }

section "CRASH_COLLECT START $TS"
echo "uptime: $(cat /proc/uptime)"
echo "loadavg: $(cat /proc/loadavg)"

section "DMESG TAIL (last 400 lines)"
dmesg --time-format=reltime 2>/dev/null | tail -400 || dmesg | tail -400

section "DRM FENCE INFO"
cat "$DRI/amdgpu_fence_info" 2>/dev/null | head -100 || echo "(unavailable)"

section "DRM SCHEDULER"
cat "$DRI/amdgpu_sched" 2>/dev/null | head -100 || echo "(unavailable)"

section "DRM GEM INFO"
cat "$DRI/amdgpu_gem_info" 2>/dev/null | head -60 || echo "(unavailable)"

section "DRM IB INFO"
cat "$DRI/amdgpu_ib_preempt" 2>/dev/null | head -40 || echo "(unavailable)"

section "FTRACE BUFFER (last 500 lines)"
if [ -f "$TR/trace" ]; then
  tail -500 "$TR/trace" 2>/dev/null || echo "(unavailable)"
else
  echo "(no ftrace mount at $TR)"
fi

section "GPU SYSFS (DPM level only — NO pp_dpm_sclk/mclk: those trigger TransferTableSmu2Dram)"
cat $CARD/device/power_dpm_force_performance_level 2>/dev/null || echo "(unavailable)"

section "PCIe AER COUNTERS"
_aer=/sys/bus/pci/devices/0000:0a:00.0/aer_dev_correctable
[ -f "$_aer" ] && cat "$_aer" 2>/dev/null || echo "(unavailable)"

section "INTERRUPTS (amdgpu / IH)"
grep -E 'amdgpu|IH|GPU|DRM' /proc/interrupts 2>/dev/null || true

section "PROCESS LIST (top CPU)"
ps aux --sort=-%cpu 2>/dev/null | head -30

section "MEMINFO"
cat /proc/meminfo

section "CRASH_COLLECT END"
echo "collector done at $(date '+%Y-%m-%d %H:%M:%S')"
