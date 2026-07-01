#!/usr/bin/env bash
# gpu_capture_run.sh — run a compute_e size with a DURABLE kernel-log capture so
# the cause of a hard-hang crash survives to the next boot.
#
# The crash is a total freeze that flushes nothing (no kdump/pstore/journal).
# tools/kmsg_sync_logger reads /dev/kmsg and fsync's every line to disk, so the
# last kernel message before the freeze (the amdgpu VM-fault / ring-timeout /
# SMU line that names the cause) is durably on the SSD. After the crash + COLD
# reboot, read it:   tail -40 perf/crash_diag/kmsg_durable.log
#
#   bash scripts/gpu_capture_run.sh <digits>      # prompts once for sudo
#
set -eu
D="${1:?usage: gpu_capture_run.sh <digits>}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$HERE/perf/crash_diag/kmsg_durable.log"
LOGGER="$HERE/tools/kmsg_sync_logger"

[ -x "$LOGGER" ] || { echo "build first: gcc -O2 -o $LOGGER tools/kmsg_sync_logger.c"; exit 1; }

# Root parts first (one password prompt, then cached for the rest).
sudo systemctl stop power-profiles-daemon
# Durable kmsg logger (mlockall + poll + O_SYNC) — captures the last kernel line
# (e.g. "flip_done timed out") before a freeze. NOTE: we deliberately do NOT arm
# the softlockup/hardlockup-panic monitor here — those panic on a transient stall
# and an RT logger tripped exactly that and rebooted the box (2026-06-12). The
# lightweight logger alone caught the flip_done cause last time; that's enough.
sudo setsid "$LOGGER" "$LOG" </dev/null >/dev/null 2>&1 &
sleep 1
if pgrep -f kmsg_sync_logger >/dev/null; then
  echo "[capture] durable kmsg logger running -> $LOG"
else
  echo "[capture] WARNING: logger not running (check sudo)";
fi
echo "[capture] power-profiles-daemon stopped"

rm -f "$HERE/perf/crash_diag/CRASH_MARKER"
cd "$HERE"
export COMPUTE_E_ALLOW_VSCODE=1 GPU_RUN_ALLOW_DESKTOP=1
# Reserve CUs for the display: limit compute to a subset so the compositor always
# has CUs to render its framebuffer -> the page flip completes (no flip_done hang).
# Inherited from the caller's env; echoed here so the run records what was used.
if [ -n "${HSA_CU_MASK:-}" ]; then
  export HSA_CU_MASK
  echo "[capture] HSA_CU_MASK=$HSA_CU_MASK (compute restricted; rest reserved for display)"
else
  echo "[capture] HSA_CU_MASK unset (gpu_run.sh will apply its default)"
fi
echo "[capture] launching d=$D (if it freezes: COLD reboot, then tail $LOG)"
exec bash scripts/gpu_run.sh 60 ./lib3/compute_e/compute_e_dev_l64 -d "$D"
