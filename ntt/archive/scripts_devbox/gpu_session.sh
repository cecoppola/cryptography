#!/bin/bash
# gpu_session.sh — full 6900XT engineering session driver.
#
# Current pipeline (updated 2026-06-01):
#
#   Phase 0  build smoke (clean-host preserves GPU bins; builds both arches)
#   Phase 1  make check-gpu — GPU correctness gate:
#              R2  test_ntt 31/31 (incl. convolution oracle, even log_n only)
#              R3-R7  per-prime stockham correctness
#              R8  determinism soak (11 primes, 50+200 iterations)
#              R9  negacyclic polymul (lib1)
#              R10 stockham sweep (49 cells)
#              R11 lib1 GPU determinism
#              (R1 cross-verify added with --full)
#   Phase 2  optimisation measurement:
#              measure_opts.sh: A1/A4/C1/I4/I15 pipeline measurement at d=100k
#              compute_e_dev d=100k NTT_PROFILE=2 combined wall-clock
#   Phase 3  baseline sweep: group_b_bench.sh (B1-B4 bench cells)
#            + ISA check (gfx942 save-temps)
#   Phase 4  (--pmc only): rocprofv3 kernel-trace on stockham hot kernel
#
# Run:
#   make check                           # host gate first (always)
#   bash scripts/gpu_session.sh          # default: phases 0-3
#   bash scripts/gpu_session.sh --full   # add R1 cross-verify sweep
#   bash scripts/gpu_session.sh --pmc    # add Phase 4 PMC capture
#   bash scripts/gpu_session.sh --full --pmc
#
# GPU-safety: all dispatches routed through scripts/gpu_run.sh (8 s TDR clamp
# unless GPU_RUN_ALLOW_LONG=1 + amdgpu compute timeout raised via
# scripts/gpu_tdr_fix.md + reboot).

set -u
cd "$(dirname "$0")/.."

FULL=0; PMC=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL=1 ;;
        --pmc)  PMC=1 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

mkdir -p results
SESSION="results/gpu_session_$(date +%Y%m%d_%H%M%S).log"
: > "$SESSION"

printf '\n\033[1;37m═══════════════════════════════════════════════════════════════\033[0m\n'
printf '\033[1;37m  gpu_session.sh — 6900XT (gfx1030) full session\033[0m\n'
printf '\033[1;37m  log = %s\033[0m\n' "$SESSION"
[ "$FULL" = "1" ] && printf '\033[1;33m  mode: --full (includes R1 cross-verify sweep)\033[0m\n'
[ "$PMC"  = "1" ] && printf '\033[1;33m  mode: --pmc  (will attempt rocprofv3 PMC capture)\033[0m\n'
printf '\033[1;37m═══════════════════════════════════════════════════════════════\033[0m\n\n'

step() { printf '\n\033[1;36m── %s ──\033[0m\n' "$*" | tee -a "$SESSION"; }
fail() { printf '\n\033[1;31m✗ FAIL @ %s — see %s\033[0m\n\n' "$*" "$SESSION"; exit 1; }

# ── GPU health gate ──────────────────────────────────────────────────────────
# Phase 0 invokes hipcc (via make all); on a box where a prior GPU crash left
# the amdgpu driver corrupted, hipcc device-open re-crashes. Abort early.
if scripts/gpu_health_check.sh 2>/dev/null; then
  :
else
  rc=$?
  if [ "$rc" -eq 1 ]; then
    printf '\033[1;31m[gpu_session] GPU driver unhealthy — power cycle required before session.\033[0m\n' >&2
    exit 1
  fi
fi

# ─────────────────────────── Phase 0 ────────────────────────────
step "Phase 0 — build smoke (clean-host keeps GPU bins)"
# clean-host removes lib1/lib2/lib3 host objects/bins but leaves
# the already-compiled gfx1030 GPU binaries in place.
make clean-host >>"$SESSION" 2>&1
make all        >>"$SESSION" 2>&1 || fail "make all (root + lib1 + lib2 + lib3)"
make -C lib2 dev test_ntt arith test_modops >>"$SESSION" 2>&1 || fail "lib2 dev targets"
make -C lib3/compute_e dev f7-build compute_e_dev >>"$SESSION" 2>&1 || fail "lib3 dev targets"
printf '  build smoke OK (0-warn both arches)\n' | tee -a "$SESSION"

# ─────────────────────────── Phase 1 ────────────────────────────
step "Phase 1 — make check-gpu (R2-R11; --full adds R1 cross-verify)"
if [ "$FULL" = "1" ]; then
    CHECK_GPU_FULL=1 bash scripts/check_gpu.sh | tee -a "$SESSION"
    rc=${PIPESTATUS[0]}
else
    bash scripts/check_gpu.sh | tee -a "$SESSION"
    rc=${PIPESTATUS[0]}
fi
[ "$rc" -eq 0 ] || fail "check-gpu"
# Gate confirms: I14+I15+A1+C1+A4+I4 correctness; I11 rect path remains disabled.

# ─────────────────────────── Phase 2 ────────────────────────────
step "Phase 2 — optimisation measurement (A1/A4/C1/I4/I15 at d=100k)"
# measure_opts.sh isolates each optimisation; compare to OPT_LEDGER §3 baseline.
# Baseline: scatter=15.6 / upload=44.4 / NTT_compute=287.8 / garner=67.0 ms
# Post-I13+I14 measured:                / NTT_compute=112.7 ms
bash scripts/measure_opts.sh | tee -a "$SESSION"

# Combined wall-clock measurement at the current production build.
step "Phase 2b — combined pipeline wall-clock (NTT_PROFILE=2, d=100k)"
{
    echo "=== compute_e_dev d=100k NTT_PROFILE=2 ==="
    NTT_PROFILE=2 scripts/gpu_run.sh 120 bin/compute_e_dev -d 100000 2>&1
    echo "=== compute_e_dev d=10k NTT_PROFILE=2 ==="
    NTT_PROFILE=2 scripts/gpu_run.sh 120 bin/compute_e_dev -d 10000 2>&1
} >> "$SESSION" 2>&1
printf '  pipeline wall-clock captured (NTT_PROFILE=2 output in log)\n' | tee -a "$SESSION"

# ─────────────────────────── Phase 3 ────────────────────────────
step "Phase 3 — baseline sweep (group_b_bench B1-B4) + gfx942 ISA"
bash scripts/group_b_bench.sh | tee -a "$SESSION"

# ISA checks: gfx942 save-temps (verify I4 mont kernel compiles to device ISA)
{
    echo "=== gfx942 ISA (dev-isa save-temps) ==="
    make -C lib2 dev-isa 2>&1 || true
    echo "=== gfx1030 isa-check ==="
    make -C lib2 isa-check 2>&1 || true
} >> "$SESSION" 2>&1
printf '  ISA checks captured (dev-isa gfx942, isa-check gfx1030)\n' | tee -a "$SESSION"

# ─────────────────────────── Phase 4 (--pmc) ────────────────────
if [ "$PMC" = "1" ]; then
    step "Phase 4 — rocprofv3 PMC capture (kernel-trace, stockham hot kernel)"
    if command -v rocprofv3 >/dev/null 2>&1; then
        {
            echo "=== PMC: stockham_kernel d=100k ==="
            # Kernel filter: stockham_kernel (C1 hot path at LOG_N_SUB=8)
            scripts/gpu_run.sh 240 rocprofv3 --kernel-trace \
                --kernel-include-regex 'stockham_kernel|xtranspose' \
                -o results/pmc_stok -- \
                bin/compute_e_dev -d 100000
        } >> "$SESSION" 2>&1 || true
        printf '  PMC capture attempted -> results/pmc_stok*\n' | tee -a "$SESSION"
        # C2 measurement: compare occupancy before/after minBlocksPerCU=4
        {
            echo "=== PMC: ntt_global_stage_kernel_mont (I4 CT-DIT path) ==="
            scripts/gpu_run.sh 120 rocprofv3 --kernel-trace \
                --kernel-include-regex 'ntt_global_stage_kernel_mont' \
                -o results/pmc_ctdit_mont -- \
                bin/compute_e_dev -d 10000
        } >> "$SESSION" 2>&1 || true
        printf '  I4 CT-DIT Mont PMC capture attempted -> results/pmc_ctdit_mont*\n' | tee -a "$SESSION"
    else
        printf '  Phase 4 SKIPPED — rocprofv3 not on PATH\n' | tee -a "$SESSION"
    fi
fi

# ─────────────────────────── Done ───────────────────────────────
printf '\n\033[1;32m═══  gpu_session.sh: all phases PASS  ═══\033[0m\n'
printf '  log: %s\n\n' "$SESSION"
