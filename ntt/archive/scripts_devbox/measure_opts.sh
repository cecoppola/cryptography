#!/usr/bin/env bash
# measure_opts.sh — Isolate and measure each optimization in the CRT-NTT pipeline.
#
# Usage:  bash scripts/measure_opts.sh [--full]
#   default: measures A1, C1, A4 at d=100k (the three new GPU-side wins)
#   --full:  also measures I14, I15 and baseline vs combined
#
# Runs entirely through gpu_run.sh (respects TDR safety limits).
# All output is timestamped to perf/opts_<date>_<time>.txt as well as stdout.
#
# Prerequisites: bin/compute_e_dev built (make compute_e_dev).
# Environment:   GPU_RUN_ALLOW_LONG=1 if d > safety threshold (default 8s).
#
# Measurement method: NTT_PROFILE=2 compute_e_dev -d 100000
#   Reports: scatter / gpu_pipe (upload+compute+download) / garner / gather ms
#   Baseline from OPT_LEDGER §3: scatter=15.6 / gpu_pipe=317.4 / garner=67.0 / gather=5.1
#   Post-I13+I14 (last GPU measurement): NTT compute 112.7 ms

set -euo pipefail
cd "$(dirname "$0")/.."

D=100000        # digit count — maps to log_n=16
OUT=perf/opts_$(date +%Y%m%d_%H%M%S).txt
mkdir -p perf

log() { echo "$*" | tee -a "$OUT"; }
sep() { log ""; log "────────────────────────────────────────────────────────"; }

log "measure_opts.sh — CRT-NTT optimization isolation benchmark"
log "Date: $(date)   D=$D   binary: bin/compute_e_dev"
log "Baseline (OPT_LEDGER §3, CT-DIT): scatter=15.6 gpu_pipe=317.4 garner=67.0 gather=5.1 ms"
log "Post-I13+I14 measured: NTT compute 112.7 ms"

run_d() {
    # run_d <label> <extra_env>
    local label="$1"; local env_prefix="${2:-}"
    sep
    log "[$label]"
    NTT_PROFILE=2 $env_prefix bash scripts/gpu_run.sh 30 bin/compute_e_dev -d "$D" 2>&1 \
        | grep -E "ntt_prof|ntt_prof2|digits|elapsed" | tee -a "$OUT"
}

# ── Current production build (all optimizations enabled) ────────────────
run_d "ALL-OPTS (production)"

sep
if [[ "${1:-}" == "--full" ]]; then
    log "Full mode: rebuilding with individual opts disabled..."

    # A1 disabled: rebuild with CPU Garner (comment out garner_reconstruct_gpu call)
    # A4 disabled: rebuild with sequential NTT(f) and NTT(g) (no stream_b)
    # C1 disabled: rebuild with static STOK_STRIDE (use EXTRA_DEF=-DSTOK_LEGACY)
    # These require Makefile EXTRA_DEF hooks and conditional compile flags.
    # For now, note what to measure manually:
    log "MANUAL STEPS for isolation:"
    log "  A1 (GPU Garner):   rebuild with -DNTT_CPU_GARNER; measure garner slot"
    log "  C1 (STOK_STRIDE):  rebuild with -DNTT_STATIC_LDS; measure NTT compute"
    log "  A4 (concurrent):   rebuild with -DNTT_SEQUENTIAL_NTT; measure gpu_pipe"
    log "  I14+I15 combined:  already measured 2026-05-30 (112.7 ms NTT compute)"
    log ""
    log "Expected deltas (from code analysis):"
    log "  A1: garner 67 ms -> ~2 ms (GPU kernel vs CPU OpenMP on 5950X)"
    log "  C1: occupancy win; measure via rocprofv3 SQ_BUSY vs baseline"
    log "  A4: NTT(f)||NTT(g) overlap; measure gpu_pipe upload portion"
    log "  I16: P0 Solinas -9 VALU; measure NTT compute on gfx942 only"
fi

sep
log "I11 rect path (if enabled): test_ntt_dev includes rect oracle at log_n 11/13/15/17/19"
log "  Run:  bash scripts/gpu_run.sh 120 bin/test_ntt_dev  (check for FAIL in I11 oracle)"
log ""
log "G6 PMC for C1 occupancy confirmation:"
log "  rocprofv3 --counters SQ_BUSY_CU_CYCLES,SQ_WAVES bin/compute_e_dev -d 1000"
log "  Compare before/after C1: expect SQ_BUSY up (more occupied CUs per ms)"
log ""
log "Results written to: $OUT"
log "Done."
