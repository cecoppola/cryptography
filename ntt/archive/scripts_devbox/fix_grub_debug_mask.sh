#!/bin/bash
# Adds amdgpu.debug_mask=0x306 to GRUB kernel cmdline so SMU/power-mgmt
# debug logging is captured from boot (modprobe.d cannot set this because
# amdgpu loads in the initramfs before /etc/modprobe.d is available).
set -e
GRUB=/etc/default/grub
if grep -q "amdgpu.debug_mask" "$GRUB"; then
  echo "[fix_grub] amdgpu.debug_mask already present — no change needed."
  grep "CMDLINE" "$GRUB"
  exit 0
fi
sed -i 's/amdgpu\.gpu_recovery=1"/amdgpu.gpu_recovery=1 amdgpu.debug_mask=0x306"/' "$GRUB"
grep "CMDLINE_LINUX_DEFAULT" "$GRUB"
update-grub
echo "[fix_grub] done. Effect on next reboot."
