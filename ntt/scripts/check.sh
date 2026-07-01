#!/bin/bash
# check.sh — unified host-side reliability gate for the whole project.
#
# Runs EVERY GPU-free test across all three libs, plus the sanitizer
# sweep, the fault-injection non-vacuity proof, and the coverage floor,
# then prints one summary table and returns a single exit code.
#
# This is the single command a developer (or CI) runs to answer "is the
# tree healthy?" — `make check`. Nothing here needs ROCm or a GPU; the
# GPU-only suites (cross-verify, determinism, polymul-sweep, modops)
# are deliberately excluded and remain gfx-gated.
#
# Exit 0 only if every suite passes.
set -u
cd "$(dirname "$0")/.."

# Each row: "<label>|<make-dir>|<make-target>|<extra-make-args>"
SUITES=(
    "ref curated primes      |.|test-curated|HOSTCC=cc"
    "ref NTT rigor (CT-DIT)  |.|test-ntt-rigor|HOSTCC=cc"
    "ref NTT rigor (Stockham)|.|test-ntt-rigor-stok|HOSTCC=cc"
    "ref polymul integ       |.|test-polymul-integ|HOSTCC=cc"
    "ref negacyclic integ    |.|test-negacyc-integ|HOSTCC=cc"
    "ref ML-KEM FIPS-203 KAT |.|test-mlkem-kat|HOSTCC=cc"
    "ref params boundary     |.|test-params-boundary|HOSTCC=cc"
    "ref mont correctness    |.|test-mont-correctness|HOSTCC=cc"
    "lib arith dual-width    |lib|test-arith-dual|"
    "lib arith fuzz          |lib|test-arith-fuzz|"
    "lib arith vs GMP oracle |lib|test-arith-gmp|"
    "lib reduce exhaustive   |lib|test-reduce|"
    "app binary_split        |app/compute_e|test-binsplit|"
    "app compute_e vs OEIS   |app/compute_e|test-host|"
    "meta non-vacuity proof   |.|verify-nonvacuous|HOSTCC=cc"
    "meta ASAN+UBSan sweep    |.|test-asan|"
    "meta coverage floor 90%  |.|coverage|"
)

mkdir -p results
LOG="results/check_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"

printf '\n\033[1;37m═══  make check — full host reliability gate  ═══\033[0m\n\n'

declare -a NAMES RESULTS
FAIL=0
for row in "${SUITES[@]}"; do
    IFS='|' read -r label dir target extra <<<"$row"
    label="${label%"${label##*[![:space:]]}"}"   # rtrim
    printf '  running %-26s ... ' "$label"
    {
        echo "===== $label ($dir: make $target $extra) ====="
        if [ -n "$extra" ]; then
            make -C "$dir" "$target" $extra
        else
            make -C "$dir" "$target"
        fi
    } >>"$LOG" 2>&1
    rc=$?
    NAMES+=("$label")
    if [ "$rc" -eq 0 ]; then
        RESULTS+=("PASS"); printf '\033[1;32mPASS\033[0m\n'
    else
        RESULTS+=("FAIL"); printf '\033[1;31mFAIL\033[0m (rc=%s)\n' "$rc"; FAIL=1
    fi
done

# ── Summary table ─────────────────────────────────────────────────────────────
printf '\n\033[1;37m┌──────────────────────────────┬────────┐\033[0m\n'
printf '\033[1;37m│\033[0m \033[1;36m%-28s\033[0m \033[1;37m│\033[0m \033[1;36m%-6s\033[0m \033[1;37m│\033[0m\n' "Suite" "Result"
printf '\033[1;37m├──────────────────────────────┼────────┤\033[0m\n'
for i in "${!NAMES[@]}"; do
    if [ "${RESULTS[$i]}" = "PASS" ]; then c='\033[1;32m'; else c='\033[1;31m'; fi
    printf '\033[1;37m│\033[0m %-28s \033[1;37m│\033[0m '"$c"'%-6s\033[0m \033[1;37m│\033[0m\n' \
        "${NAMES[$i]}" "${RESULTS[$i]}"
done
printf '\033[1;37m└──────────────────────────────┴────────┘\033[0m\n'

if [ "$FAIL" -eq 0 ]; then
    printf '\n  \033[1;32mALL %d host suites PASS\033[0m   log=%s\n\n' "${#NAMES[@]}" "$LOG"
else
    printf '\n  \033[1;31mFAILURES present — see %s\033[0m\n\n' "$LOG"
fi
exit "$FAIL"
