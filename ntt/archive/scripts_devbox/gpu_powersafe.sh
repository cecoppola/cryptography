#!/usr/bin/env bash
# gpu_powersafe.sh — reduce 6900XT electrical/thermal load below the CRASH 21
# failure threshold. CRASH 21 (2026-06-10) showed an instant full-load freeze
# with no kernel log, surviving a full software revert to the May-stable stack
# (332 prior successful runs) — i.e. a hardware/power-delivery/thermal-marginality
# regression, not a driver bug. Capping power and the clock ceiling cuts the peak
# draw and transient spikes the GPU crosses on the (warm) second run of a boot.
#
# Effects are RUNTIME ONLY and reset on reboot. Requires root (sysfs writes).
#
# Usage:
#   sudo bash scripts/gpu_powersafe.sh [cap_watts]   # apply (default 200W)
#   sudo bash scripts/gpu_powersafe.sh reset         # restore defaults
#
set -u
CARD=$(for c in /sys/class/drm/card*; do
         [ "$(basename "$(readlink "$c/device/driver" 2>/dev/null)" 2>/dev/null)" = amdgpu ] && { echo "$c"; break; }
       done)
[ -n "${CARD:-}" ] || { echo "no amdgpu card found"; exit 1; }
DEV="$CARD/device"
PCAP=$(ls "$DEV"/hwmon/hwmon*/power1_cap 2>/dev/null | head -1)
PMAX=$(ls "$DEV"/hwmon/hwmon*/power1_cap_max 2>/dev/null | head -1)
PERF="$DEV/power_dpm_force_performance_level"

PMIN=$(ls "$DEV"/hwmon/hwmon*/power1_cap_min 2>/dev/null | head -1)

if [ "${1:-}" = "reset" ]; then
  [ -n "$PCAP" ] && [ -n "$PMAX" ] && cat "$PMAX" > "$PCAP" && echo "power cap restored to max ($(($(cat "$PMAX")/1000000))W)"
  echo "auto" > "$PERF" 2>/dev/null && echo "perf level restored to auto"
  echo "current: perf=$(cat "$PERF") sclk=$(grep '\*' "$DEV/pp_dpm_sclk")"
  exit 0
fi

# This card's power1_cap floor is 231W (just under the 232W crash point) and the
# sclk DPM table is binary (500MHz idle / 2660MHz boost) with no OD interface —
# so the only lever that meaningfully cuts load is the performance-level PROFILE.
# profile_standard pins a lower fixed operating point (well under full boost),
# cutting voltage/clock and the transient slew implicated in CRASH 21.
PROFILE="${2:-profile_standard}"

# Power cap: clamp the requested watts into [min,max] so the write never EINVALs.
CAP_W="${1:-231}"
if [ -n "$PCAP" ] && [ -n "$PMIN" ] && [ -n "$PMAX" ]; then
  lo=$(( $(cat "$PMIN") / 1000000 )); hi=$(( $(cat "$PMAX") / 1000000 ))
  [ "$CAP_W" -lt "$lo" ] && CAP_W="$lo"; [ "$CAP_W" -gt "$hi" ] && CAP_W="$hi"
  if echo $(( CAP_W * 1000000 )) > "$PCAP" 2>/dev/null; then
    echo "power cap set: ${CAP_W}W (range ${lo}-${hi}W)"
  else
    echo "power cap write failed"; fi
fi

# Clock cap via performance profile.
if echo "$PROFILE" > "$PERF" 2>/dev/null; then
  echo "performance level set: $PROFILE"
else
  echo "perf level write failed (tried '$PROFILE')"; fi

sleep 1
echo "---"
echo "result: perf=$(cat "$PERF")  power_cap=$(($(cat "$PCAP")/1000000))W"
echo "        sclk=$(cat "$DEV/pp_dpm_sclk" | tr '\n' ' ')"
echo "Undo: sudo bash $0 reset   (or reboot)"
