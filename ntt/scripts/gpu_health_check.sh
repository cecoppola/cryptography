#!/usr/bin/env bash
# gpu_health_check.sh — passive GPU driver sanity check.
#
# IMPORTANT: This script deliberately does NOT open the GPU device.
# rocminfo/rocm-smi open /dev/kfd and the render node; on a post-crash
# fragile amdgpu driver, even a "successful" query can leave HSA commands
# on the GPU ring that complete asynchronously, causing a kernel panic
# 30-60s after the tool exits. Confirmed on gfx1030 2026-06-06.
#
# After a BIOS-reset-level crash: power-cycle the machine and wait for a
# clean boot before running ANY GPU tool. This script only uses lsmod,
# dmesg, /dev/dri, and sysfs — no device opens.
#
# Returns 0 if GPU appears healthy, 1 if unhealthy, 2 if no GPU detected.
#
# Usage:
#   scripts/gpu_health_check.sh          # silent, returns exit code
#   scripts/gpu_health_check.sh --verbose

set -u
VERBOSE=0
[ "${1:-}" = "--verbose" ] && VERBOSE=1

log() { [ "$VERBOSE" -eq 1 ] && echo "[gpu_health] $*"; }
fail() { echo "[gpu_health] FAIL: $*" >&2; exit 1; }

# 0-SMU. Wedged-SMU detector (CRASH 21, 2026-06-10). A hard hang from the SMU
# interface mismatch does NOT clear on a warm reset — only a full cold power
# cycle re-inits the SMU. On a warm-rebooted-after-hang boot, reading an
# SMU-backed sysfs node returns EPERM immediately (or, for pp_features, hangs
# in uninterruptible sleep). Either state means the SMU is dead and ANY GPU
# work will hard-hang the machine. Probe gpu_busy_percent (EPERM is instant,
# never a D-state hang) under a hard timeout and refuse to run if it is not a
# plain integer. Requires a COLD power cycle to clear.
_dpm_par=$(cat /sys/module/amdgpu/parameters/dpm 2>/dev/null || echo -1)
if [ "$_dpm_par" = "0" ]; then
  log "amdgpu.dpm=0 — SMU disabled by design; skipping wedged-SMU probe (gpu_busy_percent legitimately unavailable)."
else
  _card_smu=""
  for _c in /sys/class/drm/card*; do
    [ "$(cat "${_c}/device/uevent" 2>/dev/null | grep -c amdgpu)" -gt 0 ] 2>/dev/null && { _card_smu="$_c"; break; }
  done
  [ -n "$_card_smu" ] || _card_smu=/sys/class/drm/card1
  _smu_probe=$(timeout -s KILL 3 cat "${_card_smu}/device/gpu_busy_percent" 2>&1)
  _smu_rc=$?
  if [ "$_smu_rc" -ne 0 ] || ! printf '%s' "$_smu_probe" | grep -qE '^[0-9]+$'; then
    echo "[gpu_health] SMU PROBE: gpu_busy_percent -> '${_smu_probe}' (rc=${_smu_rc})" >&2
    fail "SMU is WEDGED/unresponsive (EPERM or hang on SMU sysfs). This is a warm-reboot-after-hang state; a COLD POWER CYCLE is required before any GPU work. See [[crash21-hardware-marginal-regression]] / SMU-mismatch notes."
  fi
fi

# 0a. Crash marker check: gpu_run.sh writes a marker before LAUNCH and deletes
# it after clean EXIT.  If the marker survives to the next boot, the previous
# GPU compute crashed without cleanup.  Report it and the logs to diagnose.
_crash_marker_dir="/home/machinus/ntt/perf/crash_diag"
_crash_marker="${_crash_marker_dir}/CRASH_MARKER"
if [ -f "$_crash_marker" ]; then
  echo "[gpu_health] CRASH MARKER FOUND from previous session:" >&2
  cat "$_crash_marker" >&2
  echo "[gpu_health] Telemetry and kernel logs from that run:" >&2
  ls -lt "${_crash_marker_dir}"/run_*.csv "${_crash_marker_dir}"/run_*.log 2>/dev/null | head -4 >&2
  fail "previous GPU run crashed without clean exit. Inspect logs above, then: rm ${_crash_marker}"
fi

# 0b. Crash-boot detection: if the previous boot ended without a clean systemd
# shutdown, it was a hard crash (BIOS reset / GPU hang / kernel panic).
# A soft reboot after a GPU crash does NOT fully reset GPU hardware state —
# ring buffers and firmware state can persist, causing silent driver degradation
# that passes all sysfs checks but crashes on the next HIP device open.
# Confirmed pattern: 3 consecutive crash-recovery boots on gfx1030 2026-06-07,
# each one crashing despite clean sysfs/journalctl kernel messages.
# A full POWER CYCLE (power off, wait 30s, power on) is the only reliable reset.
PREV_LAST=$(journalctl -b -1 --no-pager 2>/dev/null | tail -30)
if [ -n "$PREV_LAST" ]; then
  if ! echo "$PREV_LAST" | grep -qiE \
      "systemd.*Shutdown|Stopped target.*Shutdown|Reached target.*Shutdown|systemd-shutdown|Rebooting\.|Halting\.|Powering off"; then
    # Previous boot crashed.  A soft reboot is unsafe (GPU hardware state persists).
    # A POWER CYCLE is safe: GPU hardware is fully reset.  We detect a post-power-cycle
    # boot by checking that the current boot has been stable for >300 s with no GPU
    # errors in the current journal.  This allows GPU work after a crash+power-cycle
    # without requiring a manual health check bypass (CRASH 14, 2026-06-08).
    _uptime_s=$(awk '{print int($1)}' /proc/uptime 2>/dev/null || echo 0)
    _cur_gpu_errors=$(journalctl -b 0 -k --no-pager 2>/dev/null | \
        grep -ciE "amdgpu.*GPU (reset|hang|fault|HANG)|gpu hang|ring.*timeout|fence.*timeout" || true)
    if [ "${_uptime_s:-0}" -ge 300 ] && [ "${_cur_gpu_errors:-1}" -eq 0 ]; then
      log "previous boot crashed but current boot stable ${_uptime_s}s with no GPU errors — power cycle confirmed safe."
    else
      fail "previous boot ended without a clean systemd shutdown — likely a GPU crash. Full POWER CYCLE required: hold power button until machine is completely off, wait 30s, then power on. Do NOT use 'sudo reboot' — it will not reset GPU hardware state."
    fi
  fi
fi

# 1. Check amdgpu module is loaded
if ! lsmod 2>/dev/null | grep -q "^amdgpu"; then
  log "amdgpu module not loaded — no GPU present or driver not started"
  exit 2
fi

# 2. Check for stale reset / fence timeout markers in kernel journal (current boot).
# Uses journalctl -b 0 -k instead of dmesg: on Ubuntu 24.04 dmesg requires
# elevated privileges and silently fails when invoked without sudo, producing
# empty output that always passes the grep — a vacuous check. journalctl reads
# the persistent journal without device opens and works as a normal user.
DMESG_TAIL=$(journalctl -b 0 -k --no-pager 2>/dev/null | tail -200 | grep -v "Command line:")
if echo "$DMESG_TAIL" | grep -qiE \
    "amdgpu.*GPU (reset|hang|fault|HANG)|gpu hang|ring.*timeout|fence.*timeout|job timeout|drm.*GPU reset"; then
  fail "amdgpu reset/hang markers found in kernel journal — power cycle required before GPU work"
fi

# 2b. Check for Chrome/Mesa GPU compositor stalls (current boot).
# A CompositorAnimationObserver stall > 60 s means the GPU rendering pipeline is
# blocked — symptom of driver degradation after a crash surviving a soft reboot.
# Running HIP compute on top of a stalled compositor has caused hard freezes on
# this machine (2026-06-07, boot -1: 410 s and 300 s stalls before crash).
MAX_STALL=$(journalctl -b 0 --no-pager -n 2000 2>/dev/null | \
  grep "CompositorAnimationObserver is active for too long" | \
  sed 's/.*(\([0-9]*\)\.[0-9]*s).*/\1/' | \
  awk 'BEGIN{m=0} $1+0>m{m=$1+0} END{print m}')
# On a crash-recovery boot, compositor stalls ≥ 60 s indicate driver degradation.
# On a stable clean boot (uptime > 300 s, no GPU errors), compositor stalls are just
# VS Code/Chrome idle periods and are not a safety concern (CRASH 14, 2026-06-08).
_uptime_s2=$(awk '{print int($1)}' /proc/uptime 2>/dev/null || echo 0)
# On a stable clean boot (uptime ≥ 300 s, no GPU errors in current journal),
# compositor stalls are just VS Code/Chrome idle — not driver degradation.
# Only apply this check on crash-recovery boots (low uptime or GPU errors present).
# _cur_gpu_errors is only set by the crash-recovery branch above; on a clean-shutdown
# boot it is unset and must be computed here, else the :-1 default falsely arms
# this check (false ABORT on benign Chrome UI stalls, 2026-06-09).
if [ -z "${_cur_gpu_errors:-}" ]; then
  _cur_gpu_errors=$(journalctl -b 0 -k --no-pager 2>/dev/null | \
      grep -ciE "amdgpu.*GPU (reset|hang|fault|HANG)|gpu hang|ring.*timeout|fence.*timeout" || true)
fi
_skip_stall_check=0
if [ "${_uptime_s2:-0}" -ge 300 ] && [ "${_cur_gpu_errors:-1}" -eq 0 ]; then
  _skip_stall_check=1
fi
if [ "${_skip_stall_check:-0}" -eq 0 ] && [ "${MAX_STALL:-0}" -gt 60 ] 2>/dev/null; then
  fail "Chrome GPU compositor stall ${MAX_STALL}s detected — GPU driver may be degraded; close Chrome and power-cycle if stalls persist"
fi

# 3. Check /sys/kernel/debug/dri is accessible (basic driver health)
if [ -d /sys/kernel/debug/dri ]; then
  log "/sys/kernel/debug/dri accessible"
fi

# 4. Check GPU render nodes exist
if ! ls /dev/dri/renderD* >/dev/null 2>&1; then
  fail "no /dev/dri/renderD* nodes — driver may be dead"
fi

# 5. Passive sysfs checks — NO device open (rocminfo/rocm-smi open /dev/kfd and
#    the render node; on a post-crash fragile driver this triggers a kernel panic
#    even if the query itself returns exit 0, because HSA discovery leaves async
#    commands on the GPU ring that complete after the tool exits).
# Find amdgpu DRM card (card index varies: card0 on bare-metal, card1 on some setups)
AMD_CARD=$(for c in /sys/class/drm/card[0-9]*/device/gpu_busy_percent; do
  [ -r "$c" ] && { echo "${c%/device/gpu_busy_percent}"; break; }; done)
if [ -n "$AMD_CARD" ] && [ -r "${AMD_CARD}/device/power_state" ]; then
  PWRST=$(cat "${AMD_CARD}/device/power_state" 2>/dev/null)
  log "power_state=$PWRST (${AMD_CARD##*/})"
fi
if [ -n "$AMD_CARD" ] && [ -r "${AMD_CARD}/device/vendor" ]; then
  log "vendor=$(cat "${AMD_CARD}/device/vendor" 2>/dev/null)"
fi
LT=$(cat /sys/module/amdgpu/parameters/lockup_timeout 2>/dev/null || echo "n/a")
GR=$(cat /sys/module/amdgpu/parameters/gpu_recovery   2>/dev/null || echo "n/a")
log "lockup_timeout=$LT  gpu_recovery=$GR"
if [ "$GR" = "0" ]; then
  log "WARNING: gpu_recovery=0 — any GPU hang will hard-freeze the machine (no auto-recovery). Fix: update grub and power-cycle."
fi

log "GPU appears healthy"
exit 0
