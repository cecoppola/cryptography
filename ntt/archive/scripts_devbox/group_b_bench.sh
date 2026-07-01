#!/bin/bash
# group_b_bench.sh — comprehensive 6900XT perf baseline sweep.
#
# B1 CT-DIT bench (43 cells: 11 GPU-eligible primes × {256..4096})
# B2 Stockham bench (same grid; multi-variant: multi-launch / LDS-fused /
#                   reg2 / OTF / batched)
# B3 Polymul bench (same grid; separate / fused / batched)
# B4 compute_e_dev scaling (d ∈ {100, 1000, 10000} × LB ∈ {64, 112};
#                          d=100000 only at LB=64 to keep wall time tight)
#
# All under scripts/gpu_run.sh. Output: perf/results/GFX1030_NTT_BENCH_<TS>.md
# (raw per-run; consolidate notable findings into PERFORMANCE.md).
set -u
cd "$(dirname "$0")/.."

# GPU health gate — protects against hipcc hang and GPU binary hangs on a
# corrupted amdgpu driver (confirmed post-PMC-crash behaviour on 2026-06-06).
if scripts/gpu_health_check.sh 2>/dev/null; then
  :
else
  rc=$?
  [ "$rc" -eq 1 ] && { echo "[group_b_bench] GPU driver unhealthy — power cycle required." >&2; exit 1; }
fi

TS=$(date +%Y%m%d_%H%M%S)
MD=perf/results/GFX1030_NTT_BENCH_${TS}.md
LOG=results/groupB_${TS}.log
mkdir -p perf/results results
: > "$LOG"

# Build prerequisites: root binaries for B1-B3, lib3 dev binaries for B4.
# Without this the script silently runs every B4 cell as rc=127.
{
    make all >>"$LOG" 2>&1 \
        && make -C lib3/compute_e f7-build >>"$LOG" 2>&1
} || { echo "[group_b_bench] BUILD FAILED — see $LOG" >&2; exit 1; }

# Generate the 43-cell GPU-eligible grid (same as R1).
python3 << 'PYEOF' > /tmp/group_b_cells.txt
def mp(b,e,m):
    r=1; b%=m
    while e: r=r*b%m if e&1 else r; b=b*b%m; e>>=1
    return r
TABLE=[("Fermat-8",257,3,8),("ML-KEM",3329,3,8),("ML-KEM-v0",7681,17,9),
       ("FALCON",12289,11,12),("Proth-5-13",40961,3,13),("Fermat-16",65537,3,16),
       ("ML-DSA",8380417,10,13),("CRT-lo",167772161,3,25),("CRT-mid",469762049,3,26),
       ("CRT-hi",998244353,3,23),("FHE-RNS",2013265921,31,27)]
for nm,q,g,ml in TABLE:
    nmax=1<<ml
    for n in (256,512,1024,2048,4096):
        if n>nmax: continue
        print(f'{n} {q} {mp(g,(q-1)//n,q)} {nm}')
PYEOF
total_cells=$(wc -l < /tmp/group_b_cells.txt)

cat > "$MD" << EOF
# gfx1030 (6900 XT) — comprehensive NTT/polymul benchmarks

Captured ${TS} on AMD Radeon RX 6900 XT (gfx1030).
Source data: results/groupB_${TS}.log. Driver: scripts/group_b_bench.sh.
Prime set: the 14-prime curated table after 2026-05-18 g-fixes
(Solinas-60 g=3→10, q=7681 g=3→17). GPU-eligible subset (q<2^32, 11
primes) × n ∈ {256, 512, 1024, 2048, 4096} with the prime's
max_log_n respected.

All units = throughput per second unless noted. \`-\` = config not
admissible (n > 2^max_log_n) or wrapper rc != 0.

EOF

# wrapper with rc=99 retry
runw() {
    local secs="$1"; shift
    local out rc
    out=$(scripts/gpu_run.sh "$secs" "$@" 2>&1); rc=$?
    if [ "$rc" -eq 99 ]; then sleep 30; out=$(scripts/gpu_run.sh "$secs" "$@" 2>&1); rc=$?; fi
    echo "RC=$rc"; echo "$out"
}

# ─── B1 CT-DIT bench ────────────────────────────────────────────────────────
echo "" >>"$MD"; echo "## B1 — CT-DIT (lib1/ntt_gpu.hip), NTT/s" >>"$MD"; echo "" >>"$MD"
echo "| prime | n=256 | n=512 | n=1024 | n=2048 | n=4096 |" >>"$MD"
echo "|---|---:|---:|---:|---:|---:|" >>"$MD"
declare -A B1
declare -a primes_seen
while read -r n q w nm; do
    [[ " ${primes_seen[*]} " == *" $nm "* ]] || primes_seen+=("$nm")
    echo "===== B1 $nm n=$n q=$q ω=$w =====" >>"$LOG"
    R=$(runw 180 bin/ntt_gpu_6900xt "$n" "$q" "$w" 10000)
    rc=$(echo "$R" | head -1 | sed 's/RC=//')
    echo "$R" >>"$LOG"
    tp=$(echo "$R" | sed 's/\x1b\[[0-9;]*m//g' | grep -oE '│ +[0-9]+ /s' | head -1 | tr -dc 0-9)
    [ "$rc" = "0" ] && B1["${nm}_${n}"]="$tp" || B1["${nm}_${n}"]="-"
done < /tmp/group_b_cells.txt
for nm in "${primes_seen[@]}"; do
    row="| $nm"
    for n in 256 512 1024 2048 4096; do row+=" | ${B1[${nm}_${n}]:--}"; done
    echo "$row |" >>"$MD"
done

# ─── B2 Stockham bench ──────────────────────────────────────────────────────
echo "" >>"$MD"; echo "## B2 — Stockham (lib1/ntt_gpu_stockham.hip), LDS-fused NTT/s" >>"$MD"
echo "" >>"$MD"
echo "Headline = LDS-fused variant; the binary also reports multi-launch / reg2 / OTF / batched (see log)." >>"$MD"; echo "" >>"$MD"
echo "| prime | n=256 | n=512 | n=1024 | n=2048 | n=4096 |" >>"$MD"
echo "|---|---:|---:|---:|---:|---:|" >>"$MD"
declare -A B2
while read -r n q w nm; do
    echo "===== B2 $nm n=$n q=$q ω=$w =====" >>"$LOG"
    R=$(runw 180 bin/ntt_gpu_stockham_6900xt "$n" "$q" "$w" 10000)
    rc=$(echo "$R" | head -1 | sed 's/RC=//')
    echo "$R" >>"$LOG"
    # capture the LDS-fused (padded) row's throughput; first numeric column
    tp=$(echo "$R" | sed 's/\x1b\[[0-9;]*m//g' | grep "LDS-fused (padded)" | grep -oE '[0-9]+' | head -1)
    [ "$rc" = "0" ] && B2["${nm}_${n}"]="$tp" || B2["${nm}_${n}"]="-"
done < /tmp/group_b_cells.txt
for nm in "${primes_seen[@]}"; do
    row="| $nm"
    for n in 256 512 1024 2048 4096; do row+=" | ${B2[${nm}_${n}]:--}"; done
    echo "$row |" >>"$MD"
done

# ─── B3 Polymul bench ───────────────────────────────────────────────────────
echo "" >>"$MD"; echo "## B3 — Polymul (lib1/ntt_gpu_polymul.hip), fused polymul/s" >>"$MD"
echo "" >>"$MD"
echo "Headline = fused-launch polymul/s; binary also reports separate / batched (see log)." >>"$MD"; echo "" >>"$MD"
echo "| prime | n=256 | n=512 | n=1024 | n=2048 | n=4096 |" >>"$MD"
echo "|---|---:|---:|---:|---:|---:|" >>"$MD"
declare -A B3
while read -r n q w nm; do
    echo "===== B3 $nm n=$n q=$q ω=$w =====" >>"$LOG"
    R=$(runw 240 bin/ntt_gpu_polymul_6900xt "$n" "$q" "$w" 10000)
    rc=$(echo "$R" | head -1 | sed 's/RC=//')
    echo "$R" >>"$LOG"
    # Throughput is the column AFTER the label; strip everything up to and
    # including "launch)" first, else grep grabs the "1" from "(1 launch)".
    tp=$(echo "$R" | sed 's/\x1b\[[0-9;]*m//g' | grep "fused (1 launch)" | sed 's/.*launch)//' | grep -oE '[0-9]+' | head -1)
    [ "$rc" = "0" ] && B3["${nm}_${n}"]="$tp" || B3["${nm}_${n}"]="-"
done < /tmp/group_b_cells.txt
for nm in "${primes_seen[@]}"; do
    row="| $nm"
    for n in 256 512 1024 2048 4096; do row+=" | ${B3[${nm}_${n}]:--}"; done
    echo "$row |" >>"$MD"
done

# ─── B4 compute_e_dev scaling (refreshes PHASE_F_BENCH.md) ─────────────────
echo "" >>"$MD"; echo "## B4 — compute_e_dev scaling at LB=64 and LB=112" >>"$MD"; echo "" >>"$MD"
echo "| d | LB=64 total (s) | LB=112 total (s) | speedup (LB112/LB64) |" >>"$MD"
echo "|---:|---:|---:|---:|" >>"$MD"
extract_total() { echo "$1" | grep -oE 'total: +[0-9.]+ s' | head -1 | grep -oE '[0-9.]+'; }
for d in 100 1000 10000 100000; do
    echo "===== B4 d=$d LB=64 =====" >>"$LOG"
    secs=600; [ "$d" -ge 100000 ] && secs=1200
    R64=$(runw "$secs" lib3/compute_e/compute_e_dev_l64 -d "$d"); echo "$R64" >>"$LOG"
    R112=$(runw "$secs" lib3/compute_e/compute_e_dev_l112 -d "$d"); echo "$R112" >>"$LOG"
    t64=$(extract_total "$R64"); t112=$(extract_total "$R112")
    sp=$(python3 -c "
t64='$t64'; t112='$t112'
try:    print(f'{float(t64)/float(t112):.2f}x' if t112 else '-')
except: print('-')")
    echo "| $d | ${t64:--} | ${t112:--} | ${sp} |" >>"$MD"
done

cat >> "$MD" << EOF

---

## Methodology

Each cell was launched via \`scripts/gpu_run.sh\` (GPU-safety wrapper with
S1-S15 safeguards). rc=99 transient → 30s cooldown + retry once. Throughput
values are the binary's own reported steady-state numbers (best-of run).
B4 \`total\` includes binary_split + 10^D + Newton division + base
conversion (the same column reported by main.c).
EOF

echo
echo "Group B done. Markdown: $MD"
echo "Log:                  $LOG"
ls -la "$MD" "$LOG"
