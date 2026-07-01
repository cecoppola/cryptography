#!/bin/bash
# coverage.sh — gcov line-coverage over the ref host test suite.
#
# Rebuilds the host suite with --coverage, runs each test, then for each
# subject TU computes the MAX line coverage achieved by ANY test binary
# that touched it. (Because the Makefile builds each test as a single
# compile+link, .gcno files are renamed <bin>-<src>.gcno; we glob over
# all such files per subject.) Asserts every measured TU >= FLOOR%
# (default 90 — measured actuals are 98-100%, so 90 is a real
# regression guard; override with COVERAGE_FLOOR=N).
#
# Out: fixed-width table + results/coverage_<ts>.log.
# Exit 0 only if every measured TU meets the floor.
set -u
cd "$(dirname "$0")/.."

FLOOR="${COVERAGE_FLOOR:-90}"
mkdir -p results
LOG="results/coverage_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"

# Targets we run; each leaves <bin>-<src>.gcno + <bin>-<src>.gcda under bin/.
TARGETS=(
    test-curated test-ntt-rigor test-ntt-rigor-stok
    test-polymul-integ test-negacyc-integ test-mlkem-kat
)
# Subject TUs we measure coverage for (basenames; we glob bin/*-<base>.gcno).
SUBJECTS=(
    ntt_cpu ntt_stockham ntt_polymul ntt_polymul_negacyclic ntt_mlkem
)

make clean-host >/dev/null 2>&1
for t in "${TARGETS[@]}"; do
    printf '\n=== building %s with --coverage ===\n' "$t" | tee -a "$LOG"
    if ! make "$t" HOSTCC=cc CFLAGS='--coverage -O0 -g' >>"$LOG" 2>&1; then
        printf '  BUILD/RUN FAILED — see %s\n' "$LOG"; exit 1
    fi
done

printf '\n┌─────────────────────────────────────┬───────┬───────┬────────┬────────┐\n' | tee -a "$LOG"
printf '│ TU                                  │ exec  │ total │  %%     │ status │\n' | tee -a "$LOG"
printf '├─────────────────────────────────────┼───────┼───────┼────────┼────────┤\n' | tee -a "$LOG"

FAIL=0
for sub in "${SUBJECTS[@]}"; do
    # Find every .gcno produced from this source TU (one per test binary).
    notes=( bin/*-"$sub".gcno )
    [ -e "${notes[0]}" ] || { printf '│ %-35s │  N/A  │  N/A  │  N/A   │ MISS   │\n' "ref/$sub.c" | tee -a "$LOG"; FAIL=1; continue; }

    best_pct=0; best_tot=0
    for n in "${notes[@]}"; do
        line=$(gcov -n "$n" 2>/dev/null | grep -A1 "File 'ref/$sub.c'" | grep 'Lines executed' | head -1)
        pct=$(echo "$line" | sed -nE 's/.*Lines executed:([0-9.]+)% of ([0-9]+).*/\1/p')
        tot=$(echo "$line" | sed -nE 's/.*Lines executed:([0-9.]+)% of ([0-9]+).*/\2/p')
        [ -z "$pct" ] && continue
        # Track the binary that achieved the highest pct for this TU.
        cmp=$(awk -v a="$pct" -v b="$best_pct" 'BEGIN{print (a>b)?1:0}')
        if [ "$cmp" = "1" ]; then best_pct="$pct"; best_tot="$tot"; fi
    done

    if [ "$best_tot" = "0" ]; then
        printf '│ %-35s │  N/A  │  N/A  │  N/A   │ MISS   │\n' "ref/$sub.c" | tee -a "$LOG"; FAIL=1; continue
    fi
    exec_lines=$(awk -v p="$best_pct" -v t="$best_tot" 'BEGIN{printf "%d", p*t/100+0.5}')
    pct_i=${best_pct%.*}
    if [ "${pct_i:-0}" -ge "$FLOOR" ]; then status="OK"; color="\033[1;32m"
    else status="LOW"; color="\033[1;31m"; FAIL=1; fi
    printf "│ %-35s │ %5d │ %5d │ %5s%% │ ${color}%-6s\033[0m │\n" \
        "ref/$sub.c" "$exec_lines" "$best_tot" "$best_pct" "$status" | tee -a "$LOG"
done

printf '└─────────────────────────────────────┴───────┴───────┴────────┴────────┘\n' | tee -a "$LOG"
printf '  floor=%s%%   log=%s\n' "$FLOOR" "$LOG" | tee -a "$LOG"

# Clean coverage debris so a normal build is unaffected.
find . -name '*.gcno' -delete 2>/dev/null
find . -name '*.gcda' -delete 2>/dev/null
find . -name '*.gcov' -delete 2>/dev/null

exit "$FAIL"
