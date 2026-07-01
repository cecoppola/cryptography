#!/usr/bin/env bash
# C1: PMC counter sweep per kernel. rocprof v1 with each metric set cycles.
set -u
cd /home/machinus/ntt
OUTDIR=results/rocprof
mkdir -p $OUTDIR

# Comprehensive metric list for gfx1030 — SQ_LDS_BANK_CONFLICT is CDNA-only.
cat > $OUTDIR/c1_pmc.txt <<'EOF'
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS SQ_INSTS_SMEM
pmc: SQ_WAIT_INST_LDS SQ_WAIT_INST_VMEM SQ_WAIT_ANY SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_LDS
pmc: GRBM_GUI_ACTIVE GRBM_COUNT
pmc: TA_TA_BUSY[0] TCC_HIT[0] TCC_MISS[0] TCC_EA_RDREQ[0] TCC_EA_WRREQ[0]
kernel: stockham_lds_fused_kernel stockham_lds_fused_otf_kernel stockham_lds_batched_kernel stockham_lds_fused_coarse_kernel stockham_lds_fused_reg2_kernel stockham_stage_kernel fused_polymul_kernel fused_polymul_batched_kernel
EOF

# Run rocprof against all relevant binaries; cycles through the pmc sets.
echo ">>> n=256 q=998244353"
rocprof -i $OUTDIR/c1_pmc.txt -o $OUTDIR/c1_stockham_n256.csv \
  ./bin/ntt_gpu_stockham_6900xt 256 998244353 $(python3 bench/a_omega.py 998244353 256) 300 \
  > $OUTDIR/c1_stockham_n256.log 2>&1

echo ">>> n=2048 q=998244353"
rocprof -i $OUTDIR/c1_pmc.txt -o $OUTDIR/c1_stockham_n2048.csv \
  ./bin/ntt_gpu_stockham_6900xt 2048 998244353 $(python3 bench/a_omega.py 998244353 2048) 200 \
  > $OUTDIR/c1_stockham_n2048.log 2>&1

echo ">>> polymul n=256 q=998244353"
rocprof -i $OUTDIR/c1_pmc.txt -o $OUTDIR/c1_polymul_n256.csv \
  ./bin/ntt_gpu_polymul_6900xt 256 998244353 $(python3 bench/a_omega.py 998244353 256) 200 \
  > $OUTDIR/c1_polymul_n256.log 2>&1

# --stats run (per-kernel timings) for the same binaries
rocprof --stats -o $OUTDIR/c1_stats_stockham.csv \
  ./bin/ntt_gpu_stockham_6900xt 256 998244353 $(python3 bench/a_omega.py 998244353 256) 500 \
  > $OUTDIR/c1_stats_stockham.log 2>&1
rocprof --stats -o $OUTDIR/c1_stats_polymul.csv \
  ./bin/ntt_gpu_polymul_6900xt 256 998244353 $(python3 bench/a_omega.py 998244353 256) 500 \
  > $OUTDIR/c1_stats_polymul.log 2>&1

echo "DONE C1"
