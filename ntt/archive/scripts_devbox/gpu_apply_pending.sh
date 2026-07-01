#!/usr/bin/env bash
# gpu_apply_pending.sh — commit the staged GPU-stability changes that are
# written to disk but not yet active in the running kernel.
#
# Activates (all on the next COLD boot):
#   - amdgpu.reset_method=1   (mode1 reset: survivable GPU recovery on a ring
#                              hang instead of a full machine wedge)
#   - amdgpu.aspm=0           (no PCIe link power-state transitions)
#   - amdgpu.lockup_timeout   (tight 10s on the gfx ring so an idle-display
#                              wedge auto-resets fast; 600s on compute/SDMA/video
#                              so compute_e is never falsely killed mid-run)
#   - snd_hda_intel blacklist (deletes the GPU audio function's D3hot->D0
#                              transition that begins every idle crash)
#
# Everything else (ppfeaturemask GFXOFF-off, static clock pin, audio D0 pin,
# runpm=0, pcie_aspm=off, the health-gated watchdog) is already LIVE.
#
#   sudo bash scripts/gpu_apply_pending.sh
#   <then a COLD power cycle — full shutdown + power off, NOT warm reboot>
#
set -eu
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }

GRUB=/etc/default/grub
BLACKLIST=/etc/modprobe.d/blacklist-gpu-audio.conf

echo "== before =="
grep GRUB_CMDLINE_LINUX_DEFAULT "$GRUB"

# 1. Keep gfx-ring lockup tight but compute generous (avoid false mid-run resets).
#    Normalises whatever single/multi value is currently there to 10s,600s,600s,600s.
sed -i -E 's/amdgpu\.lockup_timeout=[^ "]*/amdgpu.lockup_timeout=10000,600000,600000,600000/' "$GRUB"

# 2. Ensure the audio codec blacklist is present (idempotent).
grep -qxF 'blacklist snd_hda_intel' "$BLACKLIST" 2>/dev/null \
  || echo 'blacklist snd_hda_intel' > "$BLACKLIST"

echo
echo "== after =="
grep GRUB_CMDLINE_LINUX_DEFAULT "$GRUB"
echo "blacklist: $(cat "$BLACKLIST")"

# 3. Regenerate grub.cfg + initramfs so the staged tokens/blacklist take at boot.
echo
echo "== regenerating grub + initramfs (this is the slow part) =="
update-grub
update-initramfs -u

echo
echo "DONE. The changes are committed to disk."
echo "Now do a COLD power cycle: full shutdown, power OFF at the PSU/wall ~10s,"
echo "then power on. A warm reboot does NOT clear this card's SMU."
echo
echo "After it comes back up, verify with:"
echo "  grep -oE 'amdgpu.(reset_method|aspm|lockup_timeout)=[^ ]*' /proc/cmdline"
echo "  lsmod | grep -c snd_hda_intel   # expect 0 (blacklist active)"
echo "  systemctl is-active watchdog    # expect active"
