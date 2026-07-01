#!/usr/bin/env bash
# Master runner for the remaining GPU tasks.
set -u
cd /home/machinus/ntt
STAMP() { date '+%H:%M:%S'; }
log() { echo "[$(STAMP)] $*"; }

log "E2 cross-verify all moduli"
bash bench/e2_cross_verify_all.sh >/tmp/e2.out 2>&1

log "D4 block-size sweep"
bash bench/d4_block_size.sh >/tmp/d4.out 2>&1

log "E1 n=4096 polymul + D2 (fused path)"
bash bench/e1_n4096_hybrid.sh >/tmp/e1.out 2>&1

log "D1 GPU vs CPU crossover"
bash bench/d1_gpu_vs_cpu.sh >/tmp/d1.out 2>&1

log "B1 random-seed stress (200 iter per config)"
bash bench/b1_stress.sh >/tmp/b1.out 2>&1

log "C2 hip-trace"
bash bench/c2_hip_trace.sh >/tmp/c2.out 2>&1

log "C1 PMC sweep"
bash bench/c1_pmc_sweep.sh >/tmp/c1.out 2>&1

log "D3 memcpy breakdown (post-process C2)"
bash bench/d3_memcpy_breakdown.sh >/tmp/d3.out 2>&1

log "ALL GPU TASKS COMPLETE"
