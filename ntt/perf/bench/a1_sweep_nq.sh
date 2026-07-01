#!/usr/bin/env bash
# A1: n x q sweep for stockham GPU kernels (all variants in one run per cell).
set -u
cd /home/machinus/ntt
OUT=results/sweep/sweep_nq.csv
LOG=results/sweep/sweep_nq.log
echo "n,q,variant,ntts,ns_per_butterfly" > "$OUT"
: > "$LOG"

NS="32 64 128 256 512 1024 2048 4096"
QS="257 3329 7681 12289 40961 65537 8380417 167772161 469762049 998244353 2013265921"

# iters budget: want ~1s-2s per cell; rough iters = max(100, 100000/n)
for q in $QS; do
  for n in $NS; do
    w=$(python3 bench/a_omega.py "$q" "$n" 2>/dev/null || true)
    if [ -z "$w" ]; then
      echo "skip n=$n q=$q (no omega)" >> "$LOG"
      continue
    fi
    # Calibrate iters
    iters=$(( 200000 / n ))
    [ $iters -lt 200 ] && iters=200
    [ $iters -gt 5000 ] && iters=5000
    echo "==> n=$n q=$q omega=$w iters=$iters" >> "$LOG"
    raw=$(./bin/ntt_gpu_stockham_6900xt "$n" "$q" "$w" "$iters" 2>&1)
    echo "$raw" >> "$LOG"
    echo "$raw" | python3 bench/parse.py stockham | awk -F, -v n=$n -v q=$q '{print n","q","$1","$2","$3}' >> "$OUT"
  done
done
echo "DONE A1"
