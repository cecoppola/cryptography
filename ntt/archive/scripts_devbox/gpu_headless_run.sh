#!/usr/bin/env bash
# gpu_headless_run.sh — DEPRECATED FOR THIS HARDWARE.
#
# WARNING: `systemctl stop gdm3` triggers an immediate amdgpu kernel panic on
# this 6900 XT setup (confirmed 3 times 2026-06-06, crashes in < 1 second).
# DO NOT USE THIS SCRIPT on this machine.
#
# Use instead (no sudo, no DM interaction):
#   GPU_RUN_ALLOW_LONG=1 scripts/gpu_run.sh <timeout> <binary> [args...] -f
#
# This script is retained only for MI300A/headless-server use where gdm3 is
# not present and the passthrough issue does not apply.
set -u
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [ "$(id -u)" -ne 0 ]; then
  echo "[headless] must run as root (stops/starts the display manager): sudo $0 $*" >&2
  exit 2
fi
if [ $# -lt 2 ]; then
  echo "usage: sudo $0 <timeout_secs> <binary> [args...]   (forwarded to gpu_run.sh)" >&2
  exit 2
fi

# Detect the active display manager unit (gdm/sddm/lightdm/...).
DM=""
for u in gdm gdm3 sddm lightdm; do
  systemctl is-active --quiet "$u" 2>/dev/null && DM="$u" && break
done

# DM restart with hard timeout: if systemctl start gdm3 hangs on a corrupted
# GPU driver, we must not freeze the machine (confirmed crash 2026-06-06).
restore() {
  if [ -n "$DM" ]; then
    echo "[headless] restarting $DM ..."
    if ! timeout 30 systemctl start "$DM"; then
      echo "[headless] WARNING: $DM restart timed out or failed. GPU driver may need power cycle." >&2
    fi
  fi
}
trap restore EXIT INT TERM

# Health check BEFORE stopping the DM.
# Stopping gdm3 on a pre-corrupted GPU driver can itself hard-hang the machine
# (confirmed crash 2026-06-06: DM stop on a corrupted gfx1030 froze the box).
echo "[headless] Pre-flight health check (before touching display manager) ..."
if ! "$SDIR/gpu_health_check.sh" --verbose; then
  hrc=$?
  if [ "$hrc" -eq 1 ]; then
    echo "[headless] ABORT: GPU driver unhealthy before DM stop. Power cycle required." >&2
    exit 1
  fi
  echo "[headless] No GPU detected (rc=$hrc) — proceeding anyway (may be MI300A path)."
fi

if [ -n "$DM" ]; then
  echo "[headless] GPU healthy. Stopping $DM (desktop will go dark; box stays up on SSH/TTY) ..."
  systemctl stop "$DM"
  sleep 3
else
  echo "[headless] no active display manager found — already headless."
fi

GPU_RUN_ALLOW_LONG=1 "$SDIR/gpu_run.sh" "$@"
rc=$?
echo "[headless] gpu_run.sh exit rc=$rc"
exit $rc
