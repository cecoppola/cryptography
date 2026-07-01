#!/usr/bin/env bash
# run_full_sweep.sh — comprehensive correctness gate + benchmark sweep with
# crash-survivable progress logging. Every GPU launch goes through gpu_run.sh
# (S1-S18 safeguards + on-disk LAUNCH/EXIT ledger). This driver adds a single
# top-level progress log on PERSISTENT disk (results/, not /tmp) that is
# flushed+fsync'd after every step, so if the box crashes mid-run the log
# pinpoints exactly which phase/step was active.
#
# Phases:
#   1. make check-gpu      — correctness gate (R2-R11: NTT, polymul/negacyclic,
#                            determinism, stockham sweep, compute_e)
#   2. group_b_bench.sh    — benchmark sweep (CT-DIT / Stockham / polymul grids,
#                            compute_e d-scaling)
#
# GPU_RUN_ALLOW_LONG=1 lifts the 8s clamp (safe: TDR raised to 600s for
# compute, and the tree is rect-free / proven paths only).
set -u
cd "$(dirname "$0")/.."
export GPU_RUN_ALLOW_LONG=1

TS=$(date +%Y%m%d_%H%M%S)
LOG="results/full_sweep_${TS}.log"
mkdir -p results perf/results

step() {  # timestamped progress line, flushed + fsync'd to disk immediately
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
    sync
}

step "=== FULL SWEEP START (ts=$TS) ==="
step "GPU: busy_pct=$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null || echo '?')%"
step "TDR: $(cat /proc/cmdline | tr ' ' '\n' | grep -o 'lockup_timeout=[0-9,]*')"
step "tree: rect path DISABLED (odd log_n -> CT-DIT); even log_n -> Stockham"

# ── Phase 1: correctness gate ────────────────────────────────────────────────
step "--- PHASE 1: make check-gpu (correctness gate) START ---"
make check-gpu >>"$LOG" 2>&1
rc1=$?
step "--- PHASE 1: make check-gpu DONE rc=$rc1 ---"
if [ $rc1 -ne 0 ]; then
    step "!!! PHASE 1 FAILED (rc=$rc1) — stopping before benchmark. See $LOG + results/check_gpu_*.log"
    exit $rc1
fi

# ── Phase 2: benchmark sweep ─────────────────────────────────────────────────
step "--- PHASE 2: group_b_bench.sh (benchmark sweep) START ---"
bash scripts/group_b_bench.sh >>"$LOG" 2>&1
rc2=$?
step "--- PHASE 2: group_b_bench.sh DONE rc=$rc2 ---"

step "=== FULL SWEEP COMPLETE  (phase1 rc=$rc1, phase2 rc=$rc2) ==="
step "logs: $LOG  +  results/check_gpu_*  +  perf/results/GFX1030_NTT_BENCH_*"
exit $rc2
