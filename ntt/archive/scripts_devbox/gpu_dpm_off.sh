#!/usr/bin/env bash
# gpu_dpm_off.sh — disable amdgpu DPM/powerplay entirely (amdgpu.dpm=0).
#
# WHY: Every captured crash signature on this 6900XT (CRASHes 12-22) lives in
# the SMU message layer: 0xFFFFFFFF responses to GfxOff transitions and
# TransferTableSmu2Dram metrics exports, SMU wedges that survive warm reboot,
# and finally CRASH 22 — a hard freeze during *read-only SMU sysfs pre-checks*
# with GFXOFF already disabled. The KFD compute path itself has never faulted.
# The card's SMU has become intolerant of message traffic (stable in May,
# progressively worse through June, on identical software — kernel 6.8.0-117
# since May 16 throughout).
#
# amdgpu.dpm=0 skips SMU/powerplay init completely: no SMU messages can ever be
# sent, by anyone. The GPU runs at fixed VBIOS boot clocks.
#
# COSTS: no dynamic clocks (possibly low fixed clocks — measure, don't assume),
# no hwmon temp/power/freq sysfs, no GFXOFF/runtime-PM, fan policy from VBIOS.
# For a correctness-validation dev box, acceptable; performance numbers from
# this card are not project deliverables (MI300A is the target).
#
# Usage:
#   sudo bash scripts/gpu_dpm_off.sh        # apply
#   sudo bash scripts/gpu_dpm_off.sh undo   # remove dpm=0 again
# Then: FULL COLD POWER CYCLE (the SMU must come up from clean power-on).
#
set -eu

# ════════════════════════════════════════════════════════════════════════════
# REFUTED ON THIS HARDWARE (2026-06-10): amdgpu.dpm=0 makes the Sienna Cichlid
# driver FAIL PROBE ENTIRELY — "smu firmware loading failed → Fatal error
# during GPU init → probe failed with error -95". The card needs SMU firmware
# even with DPM off; dpm=0 produced an unbootable (black-screen) system and a
# recovery-mode session. Apply mode is disabled. 'undo' still works to clean
# a system that has dpm=0 stuck in its config.
# ════════════════════════════════════════════════════════════════════════════
if [ "${1:-}" != "undo" ]; then
  echo "REFUSING: amdgpu.dpm=0 breaks driver probe on Sienna Cichlid (error -95,"
  echo "confirmed 2026-06-10 — black screen / recovery mode). Not a viable mode."
  echo "Use scripts/gpu_static_clock.sh (OD pin) instead. 'undo' remains available."
  exit 1
fi

[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }

GRUB=/etc/default/grub
CONF=/etc/modprobe.d/amdgpu-tdr.conf

if [ "${1:-}" = "undo" ]; then
  sed -i 's/ amdgpu\.dpm=0//g' "$GRUB"
  sed -i 's/ dpm=0//g' "$CONF" 2>/dev/null || true
  update-grub && update-initramfs -u -k all
  echo "dpm=0 removed. COLD power cycle to take effect."
  exit 0
fi

# GRUB cmdline (covers early boot)
if ! grep -q "amdgpu.dpm=0" "$GRUB"; then
  sed -i 's/\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"/\1 amdgpu.dpm=0"/' "$GRUB"
  echo "added amdgpu.dpm=0 to $GRUB"
else
  echo "$GRUB already has amdgpu.dpm=0"
fi

# modprobe.d (covers module reload from rootfs)
if [ -f "$CONF" ] && ! grep -q "dpm=0" "$CONF"; then
  sed -i 's/^options amdgpu /options amdgpu dpm=0 /' "$CONF"
  echo "added dpm=0 to $CONF"
fi

update-grub
update-initramfs -u -k all
echo
echo "DONE. Now: FULL COLD POWER CYCLE (shutdown, power off 10s, power on)."
echo "Verify after boot:"
echo "  cat /sys/module/amdgpu/parameters/dpm        # -> 0"
echo "  journalctl -b 0 -k | grep -i 'SMU'           # -> NO 'SMU is initialized' line"
echo "Then run a single compute test via gpu_run.sh."
