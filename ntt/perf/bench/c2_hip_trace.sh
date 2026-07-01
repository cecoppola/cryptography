#!/usr/bin/env bash
# C2: launch-timeline traces via rocprof --hip-trace --hsa-trace
set -u
cd /home/machinus/ntt
OUTDIR=results/trace
mkdir -p $OUTDIR

rocprof --hip-trace --hsa-trace -o $OUTDIR/c2_stockham_trace.csv \
  ./bin/ntt_gpu_stockham_6900xt 256 998244353 $(python3 bench/a_omega.py 998244353 256) 100 \
  > $OUTDIR/c2_stockham_trace.log 2>&1

rocprof --hip-trace --hsa-trace -o $OUTDIR/c2_polymul_trace.csv \
  ./bin/ntt_gpu_polymul_6900xt 256 998244353 $(python3 bench/a_omega.py 998244353 256) 100 \
  > $OUTDIR/c2_polymul_trace.log 2>&1

rocprof --hip-trace --hsa-trace -o $OUTDIR/c2_fourstep_trace.csv \
  ./bin/ntt_gpu_fourstep_6900xt 16384 998244353 $(python3 bench/a_omega.py 998244353 16384) 30 \
  > $OUTDIR/c2_fourstep_trace.log 2>&1

echo "DONE C2"
