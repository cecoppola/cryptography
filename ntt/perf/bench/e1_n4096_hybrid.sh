#!/usr/bin/env bash
# E1: n=4096 polymul via separate (4-kernel) path — fused kernel is LDS-blocked.
# Selftest verifies round-trip correctness; benchmark captures throughput.
set -u
cd /home/machinus/ntt
OUT=results/sweep/e1_n4096.csv
LOG=results/sweep/e1_n4096.log
echo "n,q,path,polymul_per_s,us_per_call" > "$OUT"
: > "$LOG"

for q in 998244353 469762049 2013265921; do
  for n in 2048 4096; do
    w=$(python3 bench/a_omega.py "$q" "$n" 2>/dev/null) || continue
    iters=100
    [ $n -eq 4096 ] && iters=80
    raw=$(./bin/ntt_gpu_polymul_6900xt "$n" "$q" "$w" "$iters" 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
    echo "== n=$n q=$q iters=$iters ==" >> "$LOG"
    echo "$raw" >> "$LOG"
    # Rows: "separate (4 launches)", "fused (1 launch)", "batched b=... (D2)"
    sep=$(echo "$raw" | awk '/separate .4 launches./ {gsub(/│/," "); for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/){print $i; exit}}')
    fus=$(echo "$raw" | awk '/fused .1 launch./ {gsub(/│/," "); for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/){print $i; exit}}')
    bat=$(echo "$raw" | awk '/batched b=.*D2/ {gsub(/│/," "); for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/){print $i; exit}}')
    echo "$n,$q,separate,${sep:-NA},NA" >> "$OUT"
    [ -n "${fus:-}" ] && echo "$n,$q,fused,$fus,NA" >> "$OUT"
    [ -n "${bat:-}" ] && echo "$n,$q,batched_D2,$bat,NA" >> "$OUT"
  done
done
echo "DONE E1"
