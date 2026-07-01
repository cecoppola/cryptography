#!/usr/bin/env bash
# D1: GPU vs CPU throughput at matched (n,q). Uses ntt_stockham (CPU) and
# ntt_gpu_stockham_6900xt (GPU fused path).
set -u
cd /home/machinus/ntt
OUT=results/sweep/d1_crossover.csv
LOG=results/sweep/d1_crossover.log
echo "n,q,cpu_ntts,gpu_fused_ntts,gpu_batch_ntts,ratio_gpufused_cpu,ratio_gpubatch_cpu" > "$OUT"
: > "$LOG"

QS="3329 8380417 998244353"
NS="32 64 128 256 512 1024 2048"

for q in $QS; do
  for n in $NS; do
    w=$(python3 bench/a_omega.py "$q" "$n" 2>/dev/null) || continue
    iters=$(( 200000 / n )); [ $iters -lt 500 ] && iters=500

    # CPU
    cpu_raw=$(./bin/ntt_stockham "$n" "$q" "$w" $iters 2>&1)
    cpu_ntts=$(echo "$cpu_raw" | sed 's/\x1b\[[0-9;]*m//g' | awk '/│ NTT\/s/ {gsub(/│/," "); for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/){print $i; exit}}')

    # GPU
    gpu_raw=$(./bin/ntt_gpu_stockham_6900xt "$n" "$q" "$w" $iters 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
    gpu_f=$(echo "$gpu_raw" | awk '/LDS-fused .padded./ {gsub(/│/," "); for(i=1;i<=NF;i++) if($i~/^[0-9]+$/){print $i; exit}}')
    gpu_b=$(echo "$gpu_raw" | awk '/batched b=/ {gsub(/│/," "); for(i=1;i<=NF;i++) if($i~/^[0-9]+$/ && $i+0 > 10000){print $i; exit}}')

    rf=$(awk -v a=${gpu_f:-0} -v b=${cpu_ntts:-1} 'BEGIN{if(b>0) printf "%.3f", a/b; else print "NA"}')
    rb=$(awk -v a=${gpu_b:-0} -v b=${cpu_ntts:-1} 'BEGIN{if(b>0) printf "%.3f", a/b; else print "NA"}')
    echo "$n,$q,${cpu_ntts:-NA},${gpu_f:-NA},${gpu_b:-NA},$rf,$rb" >> "$OUT"
    echo "n=$n q=$q cpu=$cpu_ntts gpu_f=$gpu_f gpu_b=$gpu_b" >> "$LOG"
  done
done
echo "DONE D1"
