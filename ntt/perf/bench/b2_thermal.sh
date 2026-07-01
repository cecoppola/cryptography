#!/usr/bin/env bash
# B2: 15-min thermal loop — sample NTT/s and GPU temp periodically.
set -u
cd /home/machinus/ntt
OUT=results/thermal/b2_thermal.csv
LOG=results/thermal/b2_thermal.log
echo "t_sec,ntts_fused,ntts_batched,gpu_temp_C,gpu_clk_MHz" > "$OUT"
: > "$LOG"

START=$(date +%s)
DURATION=900
N=256; Q=998244353; W=$(python3 bench/a_omega.py $Q $N)
ITERS=5000

while :; do
  now=$(date +%s); t=$((now - START))
  [ $t -ge $DURATION ] && break
  raw=$(./bin/ntt_gpu_stockham_6900xt $N $Q $W $ITERS 2>&1)
  fused=$(echo "$raw" | sed 's/\x1b\[[0-9;]*m//g' | awk '/LDS-fused .padded./ {gsub(/│/," "); for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/) {print $i; break}}')
  batch=$(echo "$raw" | sed 's/\x1b\[[0-9;]*m//g' | awk '/batched b=/ {gsub(/│/," "); cnt=0; for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/) {cnt++; if(cnt==2){print $i; break}}}')
  temp=$(rocm-smi -t 2>/dev/null | awk '/Temperature .Sensor edge/ {print $NF; exit}')
  clk=$(rocm-smi -g 2>/dev/null | awk '/sclk clock level/ {gsub(/[()Mhz]/,"",$NF); print $NF; exit}')
  echo "$t,${fused:-NA},${batch:-NA},${temp:-NA},${clk:-NA}" >> "$OUT"
  echo "t=$t fused=$fused batch=$batch temp=$temp clk=$clk" >> "$LOG"
done
echo "DONE B2"
