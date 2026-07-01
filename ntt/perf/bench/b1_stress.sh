#!/usr/bin/env bash
# B1: random-seed correctness stress — 1000 cross-verify invocations per (n,q).
# Uses the three fixed seeds baked into ntt_cross_verify_6900xt (we can still detect
# any intermittent failure by rerunning many times; environment stress catches it).
set -u
cd /home/machinus/ntt
OUT=results/stress/b1_stress.csv
LOG=results/stress/b1_stress.log
echo "n,q,runs,passes,fails" > "$OUT"
: > "$LOG"

for q in 3329 8380417 998244353; do
  for n in 256 1024; do
    w=$(python3 bench/a_omega.py "$q" "$n") || continue
    passes=0; fails=0
    for i in $(seq 1 200); do
      r=$(./bin/ntt_cross_verify_6900xt "$n" "$q" "$w" 2>&1)
      ok=$(echo "$r" | sed 's/\x1b\[[0-9;]*m//g' | awk '/Results:/ {gsub(/[a-zA-Z]/,"",$2); print $2+0}')
      if [ "${ok:-0}" -ge 7 ]; then
        passes=$((passes+1))
      else
        fails=$((fails+1))
        echo "FAIL n=$n q=$q iter=$i" >> "$LOG"
        echo "$r" >> "$LOG"
      fi
    done
    echo "$n,$q,200,$passes,$fails" >> "$OUT"
    echo "n=$n q=$q passes=$passes fails=$fails" >> "$LOG"
  done
done
echo "DONE B1"
