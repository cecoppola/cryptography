#!/usr/bin/env bash
# gpu_fix_gfxoff.sh — restore the GFXOFF-disable workaround for the 6900XT SMU
# interface mismatch (fw if 0x41 / driver 0x40). Disabling GFXOFF stops the
# SMU GfxOff transitions that return 0xFFFFFFFF and hard-hang the machine.
#
# This recreates /etc/modprobe.d/amdgpu-tdr.conf and the GRUB cmdline entry that
# were deleted during the 2026-06-09 "restore baseline" step — which is what let
# the crashes return (CRASH 21). PP_GFXOFF_MASK is bit 15, so the mask clears it:
#   0xffffffff & ~0x8000 = 0xffff7fff ; combined with the prior TDR/runpm bits
#   the value used and verified working is 0xffe73fff.
#
# Requires root. After running: COLD POWER CYCLE (not warm reboot — the SMU does
# not reset on warm reboot).
#
#   sudo bash scripts/gpu_fix_gfxoff.sh
#
set -eu
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }

# Bit 15 (PP_GFXOFF_MASK) CLEAR  — GfxOff transitions wedge the SMU (CRASH 13-21).
# Bit 14 (PP_OVERDRIVE_MASK) SET — enables pp_od_clk_voltage, the only interface
# that can pin an EXACT static MHz on RDNA2 (pp_dpm_sclk's middle level is a
# floating "current freq" readout on fine-grained-DPM cards, not a base state —
# index-pinning it landed at 2565MHz instead of 1825, 2026-06-10).
MASK="0xffe77fff"
CONF=/etc/modprobe.d/amdgpu-tdr.conf
GRUB=/etc/default/grub

cat > "$CONF" <<EOF
# amdgpu safety params. ppfeaturemask clears PP_GFXOFF_MASK (bit 15) to disable
# GFXOFF — the SMU GfxOff transition wedges on this fw/driver if-version
# mismatch (CRASH 13-21). lockup_timeout/gpu_recovery raise the TDR window;
# runpm=0 keeps the GPU from runtime-suspending between runs.
options amdgpu ppfeaturemask=${MASK} lockup_timeout=120000,600000,600000,600000 gpu_recovery=1 runpm=0
EOF
echo "wrote $CONF"

# Ensure the GRUB cmdline carries EXACTLY ONE coherent amdgpu param set.
# REPLACE, never append: appending stacked params from successive script runs
# (incl. a stale amdgpu.dpm=0, which makes the SC driver probe fail with -95 —
# "smu firmware loading failed") and broke boot on 2026-06-10. Strip every
# existing amdgpu.* token first, then add the canonical set.
_cur=$(grep -oP '(?<=^GRUB_CMDLINE_LINUX_DEFAULT=")[^"]*' "$GRUB")
_clean=$(printf '%s' "$_cur" | tr ' ' '\n' | grep -v '^amdgpu\.' | tr '\n' ' ' | sed 's/  */ /g; s/ $//')
_new="${_clean} amdgpu.ppfeaturemask=${MASK} amdgpu.gpu_recovery=1 amdgpu.lockup_timeout=120000,600000,600000,600000"
sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"${_new}\"|" "$GRUB"
echo "GRUB cmdline rewritten (single amdgpu param set): ${_new}"

update-grub
update-initramfs -u -k all
echo
echo "DONE. Now: full COLD POWER CYCLE (shutdown -h, wait 10s, power on)."
echo "Verify after boot: grep -i gfxoff /sys/class/drm/card1/device/pp_features  # -> disabled"
