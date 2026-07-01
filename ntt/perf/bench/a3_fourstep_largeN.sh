#!/usr/bin/env bash
# A3: four-step GPU NTT large-N sweep.
set -u
cd /home/machinus/ntt
OUT=results/sweep/fourstep_largeN.csv
LOG=results/sweep/fourstep_largeN.log
echo "n,q,ntts,ms_per_ntt" > "$OUT"
: > "$LOG"

NS="16384 32768 65536 131072 262144 524288 1048576"
for q in 998244353 469762049 2013265921 8380417; do
  for n in $NS; do
    w=$(python3 bench/a_omega.py "$q" "$n" 2>/dev/null) || { echo "skip n=$n q=$q" >> "$LOG"; continue; }
    # fewer iters at very large n (internal in fourstep binary — just use default iters)
    iters=50
    [ $n -ge 262144 ] && iters=10
    [ $n -ge 1048576 ] && iters=3
    raw=$(./bin/ntt_gpu_fourstep_6900xt "$n" "$q" "$w" "$iters" 2>&1)
    echo "== n=$n q=$q iters=$iters ==" >> "$LOG"
    echo "$raw" >> "$LOG"
    parsed=$(echo "$raw" | python3 bench/parse.py fourstep)
    IFS=, read -r ntts ms <<< "$parsed"
    echo "$n,$q,${ntts:-NA},${ms:-NA}" >> "$OUT"
  done
done
echo "DONE A3"
