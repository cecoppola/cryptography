#!/usr/bin/env bash
# A2: batch scaling curve — drives the batched fused kernel at many batch sizes.
set -u
cd /home/machinus/ntt
OUT=results/sweep/batch_scaling.csv
LOG=results/sweep/batch_scaling.log
echo "n,q,batch,ntts,ns_per_butterfly" > "$OUT"
: > "$LOG"

NS="256 1024 2048"
Q=998244353
BATCHES="1 2 4 8 16 32 40 48 64 80 128 256 512 1024 2048 4096"

for n in $NS; do
  w=$(python3 bench/a_omega.py "$Q" "$n") || continue
  # fewer iters per launch when batch is large (bat_iters = iters/batch handled in-binary)
  for b in $BATCHES; do
    iters=$(( 200 * b ))
    [ $iters -gt 20000 ] && iters=20000
    raw=$(./bin/ntt_gpu_stockham_6900xt "$n" "$Q" "$w" "$iters" "$b" 2>&1)
    echo "== n=$n b=$b iters=$iters ==" >> "$LOG"
    echo "$raw" >> "$LOG"
    # pick out "batched b=<b>" line
    row=$(echo "$raw" | sed 's/\x1b\[[0-9;]*m//g' | awk -v b="$b" '
      /^[[:space:]]*\│[[:space:]]*batched b=/ {
        gsub(/[│|]/," "); for(i=1;i<=NF;i++){if($i+0>1e3){print $i; break}}
      }')
    nspb=$(echo "$raw" | sed 's/\x1b\[[0-9;]*m//g' | awk -v b="$b" '
      /^[[:space:]]*\│[[:space:]]*batched b=/ {
        gsub(/[│|]/," "); cnt=0; for(i=1;i<=NF;i++){if($i ~ /^[0-9]+(\.[0-9]+)?$/){cnt++; if(cnt==2){print $i; break}}}
      }')
    echo "$n,$Q,$b,${row:-NA},${nspb:-NA}" >> "$OUT"
  done
done
echo "DONE A2"
