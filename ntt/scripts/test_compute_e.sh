#!/bin/bash
# test_compute_e.sh — host-only differential test for app/compute_e.
#
# Runs the CPU-only build (compute_e_host, built by app/compute_e's
# `make test-host`) at several digit counts and checks the printed
# decimal expansion against the known digits of Euler's number e
# (OEIS A001113) — a fully INDEPENDENT oracle: e is a mathematical
# constant, unrelated to the project's binary-splitting algorithm.
#
# compute_e_host is built with HOST_SCHOOLBOOK (schoolbook multiply
# always), so the run is pure CPU and never touches the GPU at any
# digit count. The d<=200 cap here is only because the embedded e
# reference below is 200 digits long.
#
# Calibration: each comparison is exact, character-by-character; a
# single wrong digit fails the cell. Demonstrated non-vacuous — a
# deliberately corrupted reference string fails every cell.
set -u
cd "$(dirname "$0")/.."

BIN=app/compute_e/compute_e_host
[ -x "$BIN" ] || { echo "  $BIN not built — run: make -C app/compute_e test-host"; exit 1; }

# e fractional digits 1..200 (after the decimal point), OEIS A001113.
REF=71828182845904523536028747135266249775724709369995957496696762772407663035354759457138217852516642742746639193200305992181741359662904357290033429526059563073813232862794349076323382988075319525101901

FAIL=0
printf '\n\033[1;36m━━━━  compute_e host differential vs OEIS A001113  ━━━━\033[0m\n'

for d in 50 100 200; do
    out=$("$BIN" -d "$d" 2>&1)
    rc=$?
    if [ "$rc" -ne 0 ]; then
        printf '  [d=%-4s] \033[1;31mFAIL\033[0m — exit rc=%s\n' "$d" "$rc"
        echo "$out" | tail -2
        FAIL=1; continue
    fi
    # Extract the "e = 2.XXXX" line, keep the value, drop "2.".
    val=$(echo "$out" | sed -nE 's/^e = (2\.[0-9]+).*/\1/p')
    frac=${val#2.}
    if [ -z "$frac" ]; then
        printf '  [d=%-4s] \033[1;31mFAIL\033[0m — no "e = 2..." line in output\n' "$d"
        FAIL=1; continue
    fi
    # compute_e -d N prints N significant digits => N-1 fractional digits.
    want=${REF:0:${#frac}}
    if [ "$frac" = "$want" ]; then
        printf '  [d=%-4s] \033[1;32mPASS\033[0m — %d fractional digits match e\n' "$d" "${#frac}"
    else
        printf '  [d=%-4s] \033[1;31mFAIL\033[0m — digit mismatch\n' "$d"
        printf '           got  %s\n' "$frac"
        printf '           want %s\n' "$want"
        FAIL=1
    fi
done

if [ "$FAIL" -eq 0 ]; then
    printf '\n  \033[1;32mcompute_e host differential: all cells PASS\033[0m\n\n'
else
    printf '\n  \033[1;31mcompute_e host differential: FAILURES present\033[0m\n\n'
fi
exit "$FAIL"
