#!/usr/bin/env bash
# =============================================================================
# stockham_verify_all.sh — lib1 GPU Stockham vs CPU correctness sweep (U2)
#
# WHAT: For every GPU-eligible curated prime (q < 2^32, the lib1 GPU Stockham
#       reference hard-requires this) from the CURRENT 14-prime table in
#       lib1/ntt_moduli.h, over a watchdog-safe n-sweep, run the GPU Stockham
#       selftest (bin/ntt_gpu_stockham_6900xt <n> <q> <omega> 1). With
#       iters=1 the binary runs ONLY its selftest path (run_benchmark is
#       gated on iters>1), i.e. three checks per selftest:
#         - GPU round-trip (fwd→inv = identity)
#         - GPU vs CPU Stockham (element-wise)
#         - impulse [1,0,...,0] → all-ones
#       Each invocation is wrapped by the mandatory GPU safety wrapper
#       scripts/gpu_run.sh. This is the exact analogue of the CT-DIT driver
#       e2_cross_verify_all.sh, swept over the same GPU-eligible grid.
#
# WHY:  lib1/ntt_gpu_stockham.hip was previously only single-point selftested
#       (n=256). This driver sweeps the GPU-vs-CPU element-wise verification
#       across the current GPU-eligible curated-prime grid.
#
# NOTE ON THE BINARY'S FIXED SELFTESTS: main() ALWAYS runs two fixed
#       selftests first — ML-KEM (n=256 q=3329) and ML-DSA (n=256 q=8380417)
#       — and then the CLI selftest (n,q,omega) UNLESS the CLI point is one
#       of those two fixed points. So every invocation emits 2 or 3
#       per-selftest summary lines. The tally below counts those per-selftest
#       summary lines directly (see TALLY), which is correct and complete: a
#       sweep "all-pass" means every selftest line printed across the whole
#       run is "ALL PASS". Unlike the CT-DIT cross_verify binary (one
#       "Results: <P> passed <F> failed" line per run), the Stockham binary
#       prints one "ALL PASS n=.. q=.." / "FAIL n=.. q=.." line per selftest
#       — the awk below is adapted to that exact format.
#
# BUILD PREREQUISITE (run by the main agent, not here):
#       make gpu-stok-6900xt      # builds bin/ntt_gpu_stockham_6900xt from
#                                 # lib1/ntt_gpu_stockham.hip (gfx1030)
#
# GPU SAFETY: every binary invocation goes through scripts/gpu_run.sh, which
#       applies the full S1/S6/S7/S9/S10/S11/S15 safeguard stack including the
#       mandatory post-idle S6 cooldown (~5 s, auto-extended by S11 under
#       burst). The wrapper exits 99 if a guard trips or the GPU does not
#       return to a healthy idle state — this script then STOPS immediately
#       (S-regime STOP semantics) and reports the offending case.
#
# n-SWEEP: {256, 512, 1024, 4096}. All sizes <= 4096 (log2_n <= 12) to stay
#       inside the 6900XT gfx-ring watchdog / S7 envelope; larger sizes are
#       deliberately excluded (run the high-log_n curve on gfx942/MI300A).
#       A (q,n) pair is skipped automatically when n exceeds the prime's
#       max_log2_n (no n-th root of unity exists: n must divide q-1),
#       detected via perf/bench/e2_omega.py exactly as the CT-DIT driver.
#
# GPU-ELIGIBLE PRIMES (q < 2^32): 11 of 14.
#   257 3329 7681 12289 40961 65537 8380417 167772161 469762049
#   998244353 2013265921
# GPU-INELIGIBLE (q >= 2^32, CPU-only — NOT swept here, documented only):
#   1152921504606584833 (Solinas-60), 2287828610704211969 (Solinas-61),
#   18446744069414584321 (Goldilocks). lib1 GPU path requires q < 2^32.
#
# EXPECTED INVOCATIONS: 36  (8 (q,n) grid points skipped, 44 total)
#   q=257        : n=256                       (1)
#   q=3329       : n=256                       (1)
#   q=7681       : n=256,512                   (2)
#   q=12289      : n=256,512,1024,4096         (4)
#   q=40961      : n=256,512,1024,4096         (4)
#   q=65537      : n=256,512,1024,4096         (4)
#   q=8380417    : n=256,512,1024,4096         (4)
#   q=167772161  : n=256,512,1024,4096         (4)
#   q=469762049  : n=256,512,1024,4096         (4)
#   q=998244353  : n=256,512,1024,4096         (4)
#   q=2013265921 : n=256,512,1024,4096         (4)
#
# RUNTIME ESTIMATE: ~36 invocations x (per-run selftest time, sub-second at
#   these sizes with iters=1, + ~5–15 s S6/S11 cooldown between runs)
#   ≈ 4–10 minutes wall, dominated by the mandatory inter-run cooldowns.
#
# Usage (main agent only — do NOT run during authoring):
#       bash perf/bench/stockham_verify_all.sh
#   Each binary call is already gpu_run.sh-wrapped internally, so the agent
#   just executes this script once after `make gpu-stok-6900xt`.
# =============================================================================
set -u

ROOT="/home/machinus/ntt"
BIN="$ROOT/bin/ntt_gpu_stockham_6900xt"
WRAP="$ROOT/scripts/gpu_run.sh"
OMEGA_PY="$ROOT/perf/bench/e2_omega.py"
TMO=60                                  # per-run timeout (s) passed to wrapper

OUT_DIR="$ROOT/perf/results/sweep"
OUT="$OUT_DIR/stockham_verify_all_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$OUT_DIR"

# GPU-eligible curated primes (q < 2^32), in table order.
PRIMES="257 3329 7681 12289 40961 65537 8380417 167772161 469762049 998244353 2013265921"
NSWEEP="256 512 1024 4096"

if [ ! -x "$BIN" ]; then
  echo "ERROR: $BIN not found/executable. Build first: make gpu-stok-6900xt" | tee -a "$OUT"
  exit 1
fi
if [ ! -x "$WRAP" ]; then
  echo "ERROR: GPU safety wrapper missing: $WRAP" | tee -a "$OUT"
  exit 1
fi

echo "U2 Stockham verify-all sweep — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$OUT"
echo "bin=$BIN" | tee -a "$OUT"
echo "primes(q<2^32)=$PRIMES" | tee -a "$OUT"
echo "n-sweep=$NSWEEP" | tee -a "$OUT"
echo | tee -a "$OUT"

cases=0
skipped=0
for q in $PRIMES; do
  for n in $NSWEEP; do
    omega="$(python3 "$OMEGA_PY" "$q" "$n" 2>/dev/null)"
    if [ $? -ne 0 ] || [ -z "$omega" ]; then
      echo "--- SKIP n=$n q=$q (no primitive n-th root: n does not divide q-1) ---" | tee -a "$OUT"
      skipped=$((skipped + 1))
      continue
    fi

    echo "=== n=$n q=$q omega=$omega ===" | tee -a "$OUT"
    "$WRAP" "$TMO" "$BIN" "$n" "$q" "$omega" 1 >>"$OUT" 2>&1
    rc=$?
    echo "[case rc=$rc] n=$n q=$q omega=$omega" | tee -a "$OUT"
    cases=$((cases + 1))

    if [ "$rc" -eq 99 ]; then
      echo | tee -a "$OUT"
      echo "STOP: gpu_run.sh returned 99 (GPU-unhealthy / guard trip) on" | tee -a "$OUT"
      echo "      case n=$n q=$q omega=$omega — S-regime STOP. Aborting sweep." | tee -a "$OUT"
      echo "      Inspect rocm-smi / dmesg / perf/crash_diag before resuming." | tee -a "$OUT"
      exit 99
    fi
  done
done

# ── Tally: count per-selftest summary lines in the output ───────────────────
# The Stockham binary prints, per selftest, one summary line of the form
#   "ALL PASS  n=<n> q=<q>"   (success)
#   "FAIL  n=<n> q=<q>"       (any of the 3 sub-tests failed)
# (ANSI-wrapped). Strip escape codes, then count selftests by outcome. Each
# invocation contributes 2 fixed selftests (ML-KEM, ML-DSA) plus 1 CLI
# selftest unless the CLI point IS one of those fixed points, so the
# selftest count is >= cases; a clean sweep has zero FAIL lines.
SED_STRIP='s/\x1b\[[0-9;]*m//g'
PASS_TOTAL="$(sed "$SED_STRIP" "$OUT" \
  | awk '/^[[:space:]]*ALL PASS[[:space:]]+n=/ { c++ } END { print c+0 }')"
FAIL_TOTAL="$(sed "$SED_STRIP" "$OUT" \
  | awk '/^[[:space:]]*FAIL[[:space:]]+n=/ { c++ } END { print c+0 }')"

{
  echo
  echo "============================================================"
  echo " U2 STOCKHAM VERIFY SWEEP SUMMARY"
  echo "   cases run        : $cases"
  echo "   cases skipped    : $skipped (n > prime max_log2_n)"
  echo "   selftests passed : $PASS_TOTAL (ALL PASS summary lines)"
  echo "   selftests failed : $FAIL_TOTAL (FAIL summary lines)"
  echo "   output file      : $OUT"
  echo "============================================================"
} | tee -a "$OUT"

[ "${FAIL_TOTAL:-0}" -eq 0 ] && exit 0 || exit 1
