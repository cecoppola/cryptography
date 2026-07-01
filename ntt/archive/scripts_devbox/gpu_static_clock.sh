#!/usr/bin/env bash
# gpu_static_clock.sh — pin the 6900XT to a static base clock (default 1300MHz
# sclk + max mclk) and disable all dynamic re-clocking during operation.
#
# WHY: every captured crash on this card lives in SMU message traffic, and the
# crashes cluster at dynamic transitions (idle->boost ramps at compute launch).
# Pinning via 'manual' performance level sends a one-time burst of SMU setup
# messages, after which the SMU receives no clock-management traffic at all:
# no ramps, no GfxOff (ppfeaturemask bit15), no metrics polling (all disabled).
# Residual unavoidable traffic: KFD sends SetWorkloadMask at HIP context
# open/close, and display events (modeset/DPMS) can message the SMU. If those
# alone still wedge the SMU, escalate to scripts/gpu_dpm_off.sh (dpm=0).
#
# Usage:
#   sudo bash scripts/gpu_static_clock.sh [target_sclk_mhz] [vddgfx_offset_mv]
#                                                            # default 1300, 0mV
#   sudo bash scripts/gpu_static_clock.sh 1500 50            # 1500MHz, +50mV
#   sudo bash scripts/gpu_static_clock.sh auto               # restore dynamic
#
# Pins are RUNTIME ONLY (reset by reboot). gpu_preflight.sh re-applies at boot.
set -u
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }

CARD=$(for c in /sys/class/drm/card*; do
         [ "$(basename "$(readlink "$c/device/driver" 2>/dev/null)" 2>/dev/null)" = amdgpu ] && { echo "$c/device"; break; }
       done)
[ -n "${CARD:-}" ] || { echo "no amdgpu card"; exit 1; }
PERF="$CARD/power_dpm_force_performance_level"

if [ "${1:-}" = "auto" ]; then
  echo auto > "$PERF" && echo "restored dynamic clocking (auto)"
  exit 0
fi

TARGET="${1:-1300}"   # 1300 = sustained-stable (1825 hard-locks large runs, 2026-06-16)
VO="${2:-0}"        # VDDGFX voltage offset in mV (RDNA2 'vo' command). 0 = stock
                    # curve voltage. Positive adds voltage (more stability margin
                    # at high clock, more power); negative undervolts. Used to
                    # explore stable V/f points without crashing (2026-06-15).
OD="$CARD/pp_od_clk_voltage"

# Enter manual mode first — the DPM table sysfs presentation can differ by mode.
echo manual > "$PERF" || { echo "cannot set manual perf level"; exit 1; }

if [ -w "$OD" ]; then
  # ── OverDrive path: the ONLY way to pin an EXACT MHz on RDNA2. ─────────────
  # pp_dpm_sclk's middle level on fine-grained-DPM cards is a floating
  # "current frequency" readout, NOT a base state — index-pinning it once
  # landed at 2565MHz instead of 1825 (2026-06-10). OD sets min=max=TARGET so
  # the SMU has a zero-width range: truly static. 'vo' applies the voltage
  # offset (same pp_od_clk_voltage interface = proven-safe SMU-write family).
  if [ "$VO" != "0" ]; then
    { echo "s 0 $TARGET"; echo "s 1 $TARGET"; echo "vo $VO"; echo "c"; } > "$OD" \
      || { echo "OD write failed — check dmesg; range limits may apply"; exit 1; }
  else
    { echo "s 0 $TARGET"; echo "s 1 $TARGET"; echo "c"; } > "$OD" \
      || { echo "OD write failed — check dmesg; range limits may apply"; exit 1; }
  fi
else
  echo "WARNING: pp_od_clk_voltage absent (ppfeaturemask bit 14 not set?)."
  echo "  Falling back to DPM-state pin — CANNOT guarantee target MHz."
  echo "  For exact pinning: sudo bash scripts/gpu_fix_gfxoff.sh (mask 0xffe77fff)"
  echo "  then COLD power cycle and re-run this script."
  awk -v t="$TARGET" -F'[: M]+' '
    /Mhz/ { d = ($2>t ? $2-t : t-$2); if (best=="" || d<bd) { bd=d; best=$1 } }
    END { if (best=="") exit 1; print best }' "$CARD/pp_dpm_sclk" > /tmp/.sidx \
    || { echo "no sclk states readable"; exit 1; }
  cat /tmp/.sidx > "$CARD/pp_dpm_sclk" || { echo "sclk pin write failed"; exit 1; }
fi

# Pin mclk to its highest state: NTT compute is memory-bound, and a statically
# LOW mclk would silently cost more performance than the sclk pin saves.
MIDX=$(awk -F: '/Mhz/ {i=$1} END {print i}' "$CARD/pp_dpm_mclk" 2>/dev/null | tr -d ' ')
[ -n "$MIDX" ] && echo "$MIDX" > "$CARD/pp_dpm_mclk" 2>/dev/null

sleep 1
_sclk_now=$(grep '\*' "$CARD/pp_dpm_sclk" | grep -oE '[0-9]+Mhz' | tr -d 'Mhz')
echo "--- result ---"
echo "perf:  $(cat "$PERF")"
echo "sclk:  $(grep '\*' "$CARD/pp_dpm_sclk" | tr -d '\n')"
echo "mclk:  $(grep '\*' "$CARD/pp_dpm_mclk" | tr -d '\n')"
[ -f "$OD" ] && { echo "od:"; sed 's/^/  /' "$OD" 2>/dev/null | head -8; }
# Verify the pin actually landed near TARGET — a silent wrong-frequency pin is
# exactly the failure mode that motivated the OD rewrite.
if [ -n "$_sclk_now" ] && [ "$(( _sclk_now > TARGET ? _sclk_now - TARGET : TARGET - _sclk_now ))" -gt 75 ]; then
  echo "FAIL: pinned sclk ${_sclk_now}MHz is >75MHz from target ${TARGET}MHz."
  exit 2
fi
echo "Static pin active. Revert: sudo bash $0 auto   (or reboot)"
