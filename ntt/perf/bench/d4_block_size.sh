#!/usr/bin/env bash
# D4: block-size sweep for the multi-launch stockham_stage_kernel.
# The fused variants have fixed thread counts (n/2), so this measures the
# per-stage kernel path only.
set -u
cd /home/machinus/ntt
OUT=results/sweep/d4_block_size.csv
LOG=results/sweep/d4_block_size.log
echo "n,q,block_size,multi_launch_ntts" > "$OUT"
: > "$LOG"

BLOCKS="32 64 128 192 256"  # NTT_MAX_BLOCK=256
for n in 256 1024 2048; do
  for q in 998244353; do
    w=$(python3 bench/a_omega.py "$q" "$n") || continue
    for b in $BLOCKS; do
      raw=$(./bin/ntt_gpu_stockham_6900xt "$n" "$q" "$w" 1000 0 "$b" 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
      ml=$(echo "$raw" | awk '/multi-launch .global./ {gsub(/│/," "); for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/){print $i; exit}}')
      echo "$n,$q,$b,${ml:-NA}" >> "$OUT"
      echo "n=$n b=$b ml=$ml" >> "$LOG"
    done
  done
done
echo "DONE D4"
