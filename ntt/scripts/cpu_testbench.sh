#!/bin/bash
# cpu_testbench.sh — build, selftest, and benchmark the CPU NTT reference set.
#
# Usage:   bash scripts/cpu_testbench.sh   (or: make cpu-all)
# Output:  results/cpu_testbench_YYYYMMDD_HHMMSS.log (plus live stdout)
# Exit:    0 only if (a) no FAIL/ERROR lines in log AND (b) every build/
#          run command exited 0. A failed `make` or a binary that crashes
#          before printing its verdict prints no FAIL text — rc tracking
#          (RC_FAIL) catches that silent-failure case.

set -u
cd "$(dirname "$0")/.."   # repo root (script lives in scripts/)
RC_FAIL=0
mkdir -p results
LOG="results/cpu_testbench_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"

say() { printf '\n\033[1;36m━━━━  %s  ━━━━\033[0m\n' "$*" | tee -a "$LOG"; }
run() {
    printf '\n\033[1;33m$ %s\033[0m\n' "$*" | tee -a "$LOG"
    eval "$*" 2>&1 | tee -a "$LOG"
    local rc="${PIPESTATUS[0]}"
    if [ "$rc" -ne 0 ]; then
        RC_FAIL=1
        printf '\033[1;31m[cpu_testbench] command exited rc=%s: %s\033[0m\n' \
               "$rc" "$*" | tee -a "$LOG"
    fi
    return "$rc"
}

say "Build all CPU targets"
# Force a clean rebuild so the warning-regression grep sees ALL TUs, not just
# whatever was stale. Capture build stderr into the log so the grep below
# catches any new -Wall -Wextra warning.
run "make clean-host >/dev/null"
run "make cpu stockham bench polymul negacyclic mlkem"
run "make test-curated test-ntt-rigor test-ntt-rigor-stok test-polymul-integ test-negacyc-integ test-mlkem-kat HOSTCC=cc"

say "ML-KEM (n=256, q=3329)"
run "bin/ntt_cpu 256 3329 17 100000"
run "bin/ntt_stockham 256 3329 17 100000"

say "ML-DSA (n=256, q=8380417)"
run "bin/ntt_cpu 256 8380417 1753 100000"
run "bin/ntt_stockham 256 8380417 1753 100000"

say "Polynomial multiplication"
run "bin/ntt_polymul 100000"
run "bin/ntt_polymul_negacyclic"

say "FIPS 203 ML-KEM 7-layer NTT"
run "bin/ntt_mlkem"

say "Full algorithm sweep (14 primes x all sizes)"
run "bin/ntt_bench"

say "Summary"
PASS=$(grep -cE 'PASS' "$LOG" 2>/dev/null || true)
FAIL=$(grep -cE '(FAIL|ERROR)' "$LOG" 2>/dev/null || true)
# Warning-regression: any -Wall/-Wextra warning emitted by the build step
# during this run is a regression. grep is case-sensitive on " warning:" to
# avoid false positives from result text containing the word "warning".
WARN=$(grep -cE ' warning:' "$LOG" 2>/dev/null || true)
printf '  PASS lines:       %s\n' "${PASS:-0}" | tee -a "$LOG"
printf '  FAIL/ERROR lines: %s\n' "${FAIL:-0}" | tee -a "$LOG"
printf '  Build warnings:   %s\n' "${WARN:-0}" | tee -a "$LOG"
printf '  cmd failures:     %s\n' "$RC_FAIL"   | tee -a "$LOG"
printf '  Log:              %s\n' "$LOG"       | tee -a "$LOG"
{ [ "${FAIL:-0}" -gt 0 ] || [ "${WARN:-0}" -gt 0 ] || [ "$RC_FAIL" -ne 0 ]; } && exit 1
exit 0
