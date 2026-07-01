#!/usr/bin/env bash
# gpu_cu_mask.sh — compute an HSA_CU_MASK value that reserves a few Compute Units
# for the display compositor, so a page flip never starves behind non-preemptible
# compute (the flip_done-timeout hard hang on the display-sharing gfx1030).
#
# Prints the mask value "0:0-<lastCU>" to stdout, or nothing (reserve nothing) when:
#   - the reserve count is 0, or
#   - the GPU is NOT driving a display (e.g. headless MI300A) — no flip contention.
#
# IMPORTANT: use HSA_CU_MASK with the RANGE format, NOT ROC_GLOBAL_CU_MASK. Verified
# 2026-06-13 with tools/cu_probe: ROC_GLOBAL_CU_MASK (hex or range) was SILENTLY
# IGNORED for near-full masks (probe stayed at 40/40 WGPs); HSA_CU_MASK=0:0-N
# actually restricts (0:0-75 -> 38 WGPs, 0:0-71 -> 36). Consume as:
#   export HSA_CU_MASK=$(scripts/gpu_cu_mask.sh 4)
#
# Portable: the CU count is QUERIED from KFD topology (no hardcoded arch constant).
# RDNA2 pairs CUs into WGPs, so the reserve is rounded up to an even number.
set -u
RESERVE="${1:-4}"

# Reserve nothing if asked, or if no display is attached to a GPU (headless target).
[ "$RESERVE" -gt 0 ] 2>/dev/null || exit 0
_disp=0
for s in /sys/class/drm/card*/card*-*/status; do
  [ "$(cat "$s" 2>/dev/null)" = connected ] && { _disp=1; break; }
done
[ "$_disp" = 1 ] || exit 0   # no display sharing the GPU -> no reservation needed

# Query the GPU's CU count from KFD topology: CU = simd_count / simd_per_cu on the
# first node that has SIMDs (node 0 is the CPU, simd_count=0).
_cu=0
for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
  _sc=$(awk '/^simd_count /{print $2}'  "$p" 2>/dev/null)
  _sp=$(awk '/^simd_per_cu /{print $2}' "$p" 2>/dev/null)
  if [ "${_sc:-0}" -gt 0 ] && [ "${_sp:-0}" -gt 0 ]; then
    _cu=$(( _sc / _sp )); break
  fi
done
[ "$_cu" -gt 0 ] || exit 0   # couldn't determine CU count -> safe: no mask

# Round reserve up to even (WGP pairing on RDNA), keep at least 2 CUs for compute.
_res=$(( (RESERVE + 1) / 2 * 2 ))
[ $(( _cu - _res )) -ge 2 ] || _res=$(( _cu - 2 ))
_active=$(( _cu - _res ))

# Range format: enable CUs 0..(_active-1) for compute on GPU 0; the top _res CUs
# stay free for the display. (Verified to actually restrict — see header.)
echo "0:0-$(( _active - 1 ))"
