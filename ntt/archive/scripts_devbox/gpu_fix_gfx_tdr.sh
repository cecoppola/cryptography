#!/usr/bin/env bash
# gpu_fix_gfx_tdr.sh — raise the GFX-ring TDR back to 120s so long compute runs
# don't trip the graphics watchdog.
#
# ROOT CAUSE of the d=1,100,000 hard freeze (2026-06-11): the GFX ring TDR
# (lockup_timeout[0]) was set to 10000ms (10s) in /etc/default/grub this session
# (for fast idle-wedge reset). With cwsr_enable=0 compute waves cannot be
# preempted, so a long run (d=1.1M ≈ 40s) starves the display compositor's gfx
# ring past 10s → GFX watchdog → GPU reset → total hard freeze. d=600k (22s) and
# d=800k (29s) stayed under the 10s window; 1.1M crossed it.
#
# /etc/modprobe.d/amdgpu-tdr.conf already carries the correct 120000ms, but the
# grub cmdline value overrides it. This makes grub match modprobe.d (both 120s).
# The compute/sdma/video rings stay at 600s. The health-gated watchdog daemon
# remains the fast-reset path for a genuine wedge, so the tight GFX TDR isn't
# needed. Bonus: S18's run clamp becomes 110s (no more GPU_RUN_MAX_TMO override).
#
#   sudo bash scripts/gpu_fix_gfx_tdr.sh        # apply, then COLD power cycle
#
set -eu
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }
GRUB=/etc/default/grub

echo "== before =="
grep -o 'amdgpu.lockup_timeout=[^ "]*' "$GRUB" | sort -u

# Raise the GFX value (first field) 10000 -> 120000 everywhere it appears.
sed -i -E 's/amdgpu\.lockup_timeout=10000,/amdgpu.lockup_timeout=120000,/g' "$GRUB"

echo "== after =="
grep -o 'amdgpu.lockup_timeout=[^ "]*' "$GRUB" | sort -u
echo "modprobe.d (should already match):"
grep -o 'lockup_timeout=[^ ]*' /etc/modprobe.d/amdgpu-tdr.conf | sort -u

update-grub
echo
echo "DONE. Now COLD power cycle (full shutdown, power OFF ~10s, power on)."
echo "  The cold cycle also clears the post-crash SMU state (warm reboot re-crashes)."
echo "After boot, verify:  cut -d, -f1 /sys/module/amdgpu/parameters/lockup_timeout"
echo "  expect 120000 (GFX TDR = 120s)."
