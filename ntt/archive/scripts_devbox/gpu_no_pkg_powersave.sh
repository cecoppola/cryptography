#!/usr/bin/env bash
# gpu_no_pkg_powersave.sh — eliminate ALL power-state transitions on the GPU
# package (GPU function 0a:00.0 AND its HDMI/DP audio function 0a:00.1).
#
# Every idle crash on this card begins with "snd_hda_intel 0a:00.1: Unable to
# change power state from D3hot to D0". The audio function runtime-suspends to
# D3hot independently of amdgpu (runpm=0 only covers the GPU function), and the
# wake across the shared package then fails and wedges the SMU. GFXOFF-off,
# amdgpu.aspm=0, runpm=0 and pcie_aspm=off all left this one transition alive.
#
# This keeps both functions in D0 permanently: no D3hot, no wake, no wedge.
#
#   sudo bash scripts/gpu_no_pkg_powersave.sh          # apply now + persist
#   sudo bash scripts/gpu_no_pkg_powersave.sh off       # revert
#
set -eu
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }

GPU=/sys/bus/pci/devices/0000:0a:00.0
AUD=/sys/bus/pci/devices/0000:0a:00.1
HDACONF=/etc/modprobe.d/snd-hda-no-powersave.conf
UDEV=/etc/udev/rules.d/99-gpu-no-runtime-pm.rules

if [ "${1:-}" = "off" ]; then
  rm -f "$HDACONF" "$UDEV"
  echo 1 > /sys/module/snd_hda_intel/parameters/power_save 2>/dev/null || true
  echo auto > "$AUD/power/control" 2>/dev/null || true
  echo auto > "$GPU/power/control" 2>/dev/null || true
  udevadm control --reload 2>/dev/null || true
  echo "reverted (audio power-save + runtime PM back to defaults). reboot to fully apply modprobe change."
  exit 0
fi

# 1. Stop the HDMI/DP audio codec from runtime-suspending (now + persistent).
echo 0 > /sys/module/snd_hda_intel/parameters/power_save 2>/dev/null || true
echo 'options snd_hda_intel power_save=0' > "$HDACONF"
echo "snd_hda_intel power_save=0 (now + $HDACONF)"

# 2. Pin both PCI functions to D0 — no runtime suspend (now + persistent udev).
for d in "$GPU" "$AUD"; do
  echo on > "$d/power/control" 2>/dev/null || true
done
cat > "$UDEV" <<'EOF'
# Keep the 6900XT GPU + its HDMI/DP audio function in D0 (no runtime suspend):
# the audio D3hot->D0 wake failure wedges the SMU on this card.
ACTION=="add", SUBSYSTEM=="pci", KERNEL=="0000:0a:00.0", ATTR{power/control}="on"
ACTION=="add", SUBSYSTEM=="pci", KERNEL=="0000:0a:00.1", ATTR{power/control}="on"
EOF
udevadm control --reload 2>/dev/null || true
echo "both PCI functions control=on (now + $UDEV)"

echo
echo "state now:"
echo "  power_save: $(cat /sys/module/snd_hda_intel/parameters/power_save)"
echo "  gpu   0a:00.0: control=$(cat $GPU/power/control) status=$(cat $GPU/power/runtime_status)"
echo "  audio 0a:00.1: control=$(cat $AUD/power/control) status=$(cat $AUD/power/runtime_status)"
echo "(audio status should move to 'active' shortly; if still 'suspended', a reboot applies the modprobe option cleanly)"
