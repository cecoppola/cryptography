#!/usr/bin/env bash
# fix_gpu_driver.sh — ensure amdgpu TDR and auto-recovery are correctly enabled.
#
# TWO-PART FIX:
#   PART 1 (immediate): PCIe Function-Level Reset (FLR) on 0a:00.0.
#     Attempts to reset 6900 XT hardware to factory state. NOTE: does not
#     work on an active display GPU (FLR not supported while display is active).
#     For a true immediate reset: full power cycle (hold power button, wait 30s).
#
#   PART 2 (persistent, survives reboot): write correct values to modprobe.d
#     AND grub — both must be correct, as modprobe.d overrides grub on module
#     reload. Correct values for this machine (gfx1030, display-sharing):
#       lockup_timeout=10000,600000,600000,600000  — GFX 10s / rest 600s TDR
#       gpu_recovery=1                              — auto-reset on hang
#
#   WHY TDR must be ENABLED (not disabled):
#     With TDR disabled (lockup_timeout=0, gpu_recovery=0), any GPU hang —
#     from multi-tenant contention, a buggy kernel, or a stray Mesa command —
#     becomes a permanent hard freeze requiring a BIOS reset. Confirmed pattern:
#     3 consecutive hard freezes on 2026-06-07 from running HIP compute with
#     Chrome GPU compositor active + lockup_timeout=0.
#     With TDR enabled, a hang fires the watchdog, the driver auto-resets the
#     GPU (display blinks once), the hung process gets an error, and the machine
#     stays up. This is the correct behaviour for a compute workstation.
#
#   AFTER THIS SCRIPT:
#     - Run: sudo update-grub && sudo update-initramfs -u
#     - Then full power cycle (not just reboot) for clean GPU hardware state.
#
# Usage: sudo bash scripts/fix_gpu_driver.sh

set -uo pipefail

GPU_PCI="0000:0a:00.0"
GPU_RESET="/sys/bus/pci/devices/${GPU_PCI}/reset"
MODPROBE_CONF="/etc/modprobe.d/amdgpu-tdr.conf"
GRUB_DEFAULT="/etc/default/grub"

LOCKUP="10000,600000,600000,600000"
RECOVERY="1"

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: must run as root. Use: sudo bash scripts/fix_gpu_driver.sh" >&2
  exit 1
fi

echo "=== amdgpu fix: Part 1 — PCIe Function-Level Reset ==="
if [ -w "$GPU_RESET" ]; then
  echo "  Writing 1 to $GPU_RESET ..."
  echo "  (display may flicker for ~1s — this is normal)"
  if echo 1 > "$GPU_RESET" 2>/dev/null; then
    sleep 2
    echo "  PCIe FLR complete."
    AMD_CARD=$(for c in /sys/class/drm/card[0-9]*/device/gpu_busy_percent; do
      [ -r "$c" ] && { echo "${c%/device/gpu_busy_percent}"; break; }; done)
    [ -n "$AMD_CARD" ] && [ -r "${AMD_CARD}/device/power_state" ] && \
      echo "  power_state=$(cat "${AMD_CARD}/device/power_state")"
  else
    echo "  NOTE: PCIe FLR not supported on this device (display-active GPU)."
    echo "  For immediate clean GPU state: full power cycle (hold button, wait 30s)."
  fi
else
  echo "  NOTE: $GPU_RESET not writable — skipping PCIe FLR."
  echo "  For immediate clean GPU state: full power cycle (hold button, wait 30s)."
fi

echo ""
echo "=== amdgpu fix: Part 2 — persistent TDR enable ==="

# Update modprobe.d — this applies on module reload (overrides grub on reload)
cat > "$MODPROBE_CONF" << EOF
# lockup_timeout=${LOCKUP}: GFX 10s / Compute+SDMA+Video 600s watchdog timeouts (enables TDR).
# gpu_recovery=${RECOVERY}: enable automatic GPU reset on hang (display blinks, process gets error, machine stays up).
options amdgpu lockup_timeout=${LOCKUP} gpu_recovery=${RECOVERY}
EOF
echo "  Written $MODPROBE_CONF"

# Update GRUB
if [ -f "$GRUB_DEFAULT" ]; then
  cp "$GRUB_DEFAULT" "${GRUB_DEFAULT}.bak.$(date +%Y%m%d_%H%M%S)"

  # Strip any existing amdgpu lockup/recovery params
  sed -i -E \
    's/ amdgpu\.lockup_timeout=[0-9,]+//g; s/ amdgpu\.gpu_recovery=-?[0-9]+//g' \
    "$GRUB_DEFAULT"

  # Append correct params
  sed -i -E \
    "s|(GRUB_CMDLINE_LINUX_DEFAULT=\"[^\"]*)\"$|\1 amdgpu.lockup_timeout=${LOCKUP} amdgpu.gpu_recovery=${RECOVERY}\"|" \
    "$GRUB_DEFAULT"

  echo "  Updated $GRUB_DEFAULT"
  echo "  $(grep GRUB_CMDLINE_LINUX_DEFAULT "$GRUB_DEFAULT")"
else
  echo "  WARNING: $GRUB_DEFAULT not found — GRUB not updated."
fi

echo ""
echo "=== Next steps (run these now) ==="
echo "  sudo update-grub"
echo "  sudo update-initramfs -u"
echo "  # Then full power cycle for clean GPU hardware state."
echo ""
echo "=== Verification (after power cycle) ==="
echo "  cat /sys/module/amdgpu/parameters/lockup_timeout   # should be ${LOCKUP}"
echo "  cat /sys/module/amdgpu/parameters/gpu_recovery     # should be ${RECOVERY}"
