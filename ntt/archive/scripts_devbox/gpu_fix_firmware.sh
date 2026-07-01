#!/usr/bin/env bash
# gpu_fix_firmware.sh — fix the Linux amdgpu SMU driver/firmware version mismatch
# that makes SMU messages fragile (driver if=0x40 vs loaded firmware if=0x41).
#
# Windows runs this same card flawlessly because its driver matches its firmware.
# On Linux the matching stock firmware (0x40, paired with the in-tree 6.8 driver)
# was removed and a newer mismatched firmware (0x41, version 58.89.0) was dropped
# in by the ROCm/amdgpu installer as an ORPHAN file. This restores Ubuntu's
# driver-matching firmware set.
#
# Fully reversible: the current /lib/firmware/amdgpu is backed up first.
#
#   sudo bash scripts/gpu_fix_firmware.sh          # apply (then COLD boot)
#   sudo bash scripts/gpu_fix_firmware.sh revert    # restore the ROCm firmware
#
set -eu
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }

BAK=/lib/firmware/amdgpu.rocm-bak

if [ "${1:-}" = "revert" ]; then
  [ -d "$BAK" ] || { echo "no backup at $BAK"; exit 1; }
  rm -rf /lib/firmware/amdgpu
  mv "$BAK" /lib/firmware/amdgpu
  update-initramfs -u -k all
  echo "reverted to backed-up (ROCm) firmware. COLD boot to apply."
  exit 0
fi

# 1. Back up the current GPU firmware (the orphan 0x41 set) — revertible.
if [ ! -d "$BAK" ]; then
  cp -a /lib/firmware/amdgpu "$BAK"
  echo "backed up current firmware -> $BAK"
else
  echo "backup already exists at $BAK (keeping it)"
fi

echo "current SMC firmware md5: $(md5sum /lib/firmware/amdgpu/sienna_cichlid_smc.bin.zst 2>/dev/null | cut -d' ' -f1)"

# 2. Restore Ubuntu's stock, kernel-matched firmware set (overwrites the orphan).
apt-get install -y --reinstall linux-firmware

echo "new SMC firmware md5:     $(md5sum /lib/firmware/amdgpu/sienna_cichlid_smc.bin.zst 2>/dev/null | cut -d' ' -f1)"
echo "  (different md5 => the firmware file changed, as intended)"

# 3. Refresh the early-boot firmware copy in the initramfs.
update-initramfs -u -k all

echo
echo "DONE. Now COLD power cycle, then verify the mismatch is gone:"
echo "    journalctl -b 0 -k | grep -i 'smu.*version'"
echo "  SUCCESS = the 'SMU driver if version not matched' line is ABSENT."
echo "  If ROCm/HIP compute misbehaves after this, revert:"
echo "    sudo bash scripts/gpu_fix_firmware.sh revert   (then COLD boot)"
