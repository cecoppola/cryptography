#!/bin/bash
# coverage_full.sh — gcov line-coverage across ref + lib/arith + app/compute_e
# host code. Companion to coverage.sh (which measures only ref for speed).
#
# Strategy: for each lib, rebuild its host tests with --coverage instrumentation,
# run them, then for each subject TU compute the MAX line coverage across all
# binaries that exercised it. Assert each TU ≥ FLOOR%.
#
# Out: per-lib + combined table to results/coverage_full_<ts>.log.
# Exit 0 only if every measured TU meets the floor.
set -u
cd "$(dirname "$0")/.."

FLOOR="${COVERAGE_FLOOR:-80}"
mkdir -p results
LOG="results/coverage_full_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"

printf '\n\033[1;37m═══  coverage-all — ref + lib/arith + app/compute_e  ═══\033[0m\n\n'

run_ref() {
    printf '\n\033[1;36m── ref ──\033[0m\n' | tee -a "$LOG"
    local TARGETS=(test-curated test-ntt-rigor test-ntt-rigor-stok
                   test-polymul-integ test-negacyc-integ test-mlkem-kat)
    local SUBJECTS=(ntt_cpu ntt_stockham ntt_polymul ntt_polymul_negacyclic ntt_mlkem)
    make clean-host >/dev/null 2>&1
    for t in "${TARGETS[@]}"; do
        make "$t" HOSTCC=cc CFLAGS='--coverage -O0 -g' >>"$LOG" 2>&1 || return 1
    done
    for sub in "${SUBJECTS[@]}"; do
        local notes=( bin/*-"$sub".gcno )
        [ -e "${notes[0]}" ] || { echo "ref/$sub.c|MISS|0|0" >> /tmp/cov_rows.txt; continue; }
        local best_pct=0 best_tot=0
        for n in "${notes[@]}"; do
            local line=$(gcov -n "$n" 2>/dev/null | grep -A1 "File 'ref/$sub.c'" | grep 'Lines executed' | head -1)
            local pct=$(echo "$line" | sed -nE 's/.*Lines executed:([0-9.]+)% of ([0-9]+).*/\1/p')
            local tot=$(echo "$line" | sed -nE 's/.*Lines executed:([0-9.]+)% of ([0-9]+).*/\2/p')
            [ -z "$pct" ] && continue
            local cmp=$(awk -v a="$pct" -v b="$best_pct" 'BEGIN{print (a>b)?1:0}')
            [ "$cmp" = "1" ] && { best_pct=$pct; best_tot=$tot; }
        done
        echo "ref/$sub.c|OK|$best_pct|$best_tot" >> /tmp/cov_rows.txt
    done
    find . -name '*.gcda' -delete 2>/dev/null
    find . -name '*.gcno' -delete 2>/dev/null
    find . -name '*.gcov' -delete 2>/dev/null
}

run_lib() {
    printf '\n\033[1;36m── lib/arith ──\033[0m\n' | tee -a "$LOG"
    make -C lib clean >>"$LOG" 2>&1
    # Override GCC variable in lib/Makefile recipes so --coverage flows through
    for t in test-arith-dual test-arith-fuzz test-arith-gmp test-reduce; do
        make -C lib "$t" GCC="gcc -O0 -g --coverage" >>"$LOG" 2>&1 || true
    done
    for sub in bigint multiply newton base_convert; do
        # find any .gcno produced (could be at lib/arith/test_arith_*.gcno or similar)
        local notes=( lib/arith/*"$sub".gcno lib/*"$sub".gcno )
        local best_pct=0 best_tot=0
        local hits=0
        for n in "${notes[@]}"; do
            [ -e "$n" ] || continue
            hits=$((hits+1))
            local line=$(gcov -n "$n" 2>/dev/null | grep -A1 "File 'lib/arith/$sub.c'\|File 'arith/$sub.c'" | grep 'Lines executed' | head -1)
            local pct=$(echo "$line" | sed -nE 's/.*Lines executed:([0-9.]+)% of ([0-9]+).*/\1/p')
            local tot=$(echo "$line" | sed -nE 's/.*Lines executed:([0-9.]+)% of ([0-9]+).*/\2/p')
            [ -z "$pct" ] && continue
            local cmp=$(awk -v a="$pct" -v b="$best_pct" 'BEGIN{print (a>b)?1:0}')
            [ "$cmp" = "1" ] && { best_pct=$pct; best_tot=$tot; }
        done
        if [ "$best_tot" = "0" ]; then
            echo "lib/arith/$sub.c|MISS|0|0" >> /tmp/cov_rows.txt
        else
            echo "lib/arith/$sub.c|OK|$best_pct|$best_tot" >> /tmp/cov_rows.txt
        fi
    done
    find lib -name '*.gcda' -delete 2>/dev/null
    find lib -name '*.gcno' -delete 2>/dev/null
    find lib -name '*.gcov' -delete 2>/dev/null
    find . -maxdepth 1 -name '*.gcov' -delete 2>/dev/null
}

run_app() {
    printf '\n\033[1;36m── app/compute_e ──\033[0m\n' | tee -a "$LOG"
    make -C app/compute_e clean >>"$LOG" 2>&1
    # Override GCC so --coverage flows through BOTH compile and link
    # (C_CFLAGS reaches only compile; link line needs -lgcov via --coverage).
    for t in test-binsplit test-host; do
        make -C app/compute_e "$t" GCC="gcc --coverage" C_CFLAGS="-O0 -g -Wall -Wextra" >>"$LOG" 2>&1 || true
    done
    for sub in main binary_split mem_pool; do
        local notes=( app/compute_e/*"$sub".gcno app/compute_e/*"$sub"_host.gcno )
        local best_pct=0 best_tot=0
        for n in "${notes[@]}"; do
            [ -e "$n" ] || continue
            local line=$(gcov -n "$n" 2>/dev/null | grep -A1 "File 'app/compute_e/$sub.c'\|File '$sub.c'\|File '.*$sub\.c'" | grep 'Lines executed' | head -1)
            local pct=$(echo "$line" | sed -nE 's/.*Lines executed:([0-9.]+)% of ([0-9]+).*/\1/p')
            local tot=$(echo "$line" | sed -nE 's/.*Lines executed:([0-9.]+)% of ([0-9]+).*/\2/p')
            [ -z "$pct" ] && continue
            local cmp=$(awk -v a="$pct" -v b="$best_pct" 'BEGIN{print (a>b)?1:0}')
            [ "$cmp" = "1" ] && { best_pct=$pct; best_tot=$tot; }
        done
        if [ "$best_tot" = "0" ]; then
            echo "app/compute_e/$sub.c|MISS|0|0" >> /tmp/cov_rows.txt
        else
            echo "app/compute_e/$sub.c|OK|$best_pct|$best_tot" >> /tmp/cov_rows.txt
        fi
    done
    find app -name '*.gcda' -delete 2>/dev/null
    find app -name '*.gcno' -delete 2>/dev/null
    find app -name '*.gcov' -delete 2>/dev/null
    find . -maxdepth 1 -name '*.gcov' -delete 2>/dev/null
}

: > /tmp/cov_rows.txt
run_ref
run_lib
run_app

# Render combined table
printf '\n┌─────────────────────────────────────┬───────┬───────┬────────┬────────┐\n' | tee -a "$LOG"
printf '│ TU                                  │ exec  │ total │  %%     │ status │\n' | tee -a "$LOG"
printf '├─────────────────────────────────────┼───────┼───────┼────────┼────────┤\n' | tee -a "$LOG"

FAIL=0
while IFS='|' read -r sub kind pct tot; do
    if [ "$kind" = "MISS" ]; then
        printf '│ %-35s │  N/A  │  N/A  │  N/A   │ \033[1;31mMISS  \033[0m │\n' "$sub" | tee -a "$LOG"
        FAIL=1
        continue
    fi
    exec_lines=$(awk -v p="$pct" -v t="$tot" 'BEGIN{printf "%d", p*t/100+0.5}')
    pct_i=${pct%.*}
    if [ "${pct_i:-0}" -ge "$FLOOR" ]; then status="OK"; color="\033[1;32m"
    else status="LOW"; color="\033[1;31m"; FAIL=1; fi
    printf "│ %-35s │ %5d │ %5d │ %5s%% │ ${color}%-6s\033[0m │\n" \
        "$sub" "$exec_lines" "$tot" "$pct" "$status" | tee -a "$LOG"
done < /tmp/cov_rows.txt

printf '└─────────────────────────────────────┴───────┴───────┴────────┴────────┘\n' | tee -a "$LOG"
printf '  floor=%s%%   log=%s\n' "$FLOOR" "$LOG" | tee -a "$LOG"
rm -f /tmp/cov_rows.txt

exit "$FAIL"
