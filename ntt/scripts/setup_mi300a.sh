#!/bin/bash
# setup_mi300a.sh — source this before building or running on the MI300A cluster.
#
# Usage:
#   source scripts/setup_mi300a.sh   (or: . scripts/setup_mi300a.sh)
#
# Required for:
#   - Building lib/ (make all) and app/compute_e/ (make)
#   - Running any binary via srun
#
# All values confirmed from mi300a_environment_0509.txt.

# ── Modules ───────────────────────────────────────────────────────────────────
module load PrgEnv-amd/8.6.0
module load rocm/7.0.3
# Zen 4 ("Genoa") CPU target — NOT milan. Confirmed loaded on the live system
# (mi300a_environment_0509.txt §2). Matters: the Garner reconstruction is a
# 96-core OpenMP host loop; correct Zen4 codegen is on the critical path.
module load craype-x86-genoa
module load craype-accel-amd-gfx942
module load cray-shmem/12.0.0
module load cray-pmi/6.1.16

# ── ROCm paths ────────────────────────────────────────────────────────────────
export ROCM_PATH=/opt/rocm-7.0.3
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# ── Device visibility (all 4 APUs on the node) ────────────────────────────────
export ROCR_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3

# ── SDMA (required for hipMemcpyPeerAsync between APUs) ───────────────────────
export HSA_ENABLE_SDMA=1

# ── OpenMP (Garner reconstruction uses 96-core parallel loop) ─────────────────
export OMP_NUM_THREADS=96
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "MI300A environment ready."
echo "  ROCM_PATH=$ROCM_PATH"
echo "  ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""
echo "Build:  cd lib && make all"
echo "Test:   make check   (runs via srun -p mi -N 1)"
echo "Bench:  srun -p mi -N 1 -n 4 ./crt_ntt 24 100"
