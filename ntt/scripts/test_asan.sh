#!/bin/bash
# test_asan.sh — rebuild every host C test suite under AddressSanitizer
# + UndefinedBehaviorSanitizer and run each.
#
# ASAN catches OOB heap/stack reads+writes, use-after-free, leaks.
# UBSan catches signed overflow, out-of-range shifts, misaligned access,
# and other UB that a plain -O2 build silently tolerates. -fno-sanitize-
# recover=all makes the FIRST violation abort with non-zero exit, so a
# regression cannot hide behind a still-printed "PASS".
#
# Coverage: ref host suite + lib arith (incl LB=64/112 dual-build,
# fuzz, GMP oracle, transfer-core scatter/gather/CLA, e2e CRT-NTT +
# negacyclic oracle) + lib reduce + app binsplit + app compute_e host.
# Pure host C — no ROCm/GPU. Exit 0 only if every suite builds + runs
# cleanly with zero sanitizer reports.
set -u
cd "$(dirname "$0")/.."

SAN='-fsanitize=undefined,address -fno-sanitize-recover=all -fno-omit-frame-pointer -g'

FAIL=0
run_suite() {  # tag | target | extra-make-args...
    local tag="$1" target="$2"; shift 2
    out=$(make "$@" "$target" 2>&1)
    rc=$?
    if [ "$rc" -ne 0 ] || echo "$out" | grep -qE 'runtime error|AddressSanitizer|UndefinedBehaviorSanitizer'; then
        printf '  [%-30s] \033[1;31mFAIL\033[0m\n' "$tag"
        echo "$out" | grep -E 'runtime error|AddressSanitizer|ERROR|SUMMARY' | head -5
        FAIL=1
    else
        printf '  [%-30s] \033[1;32mclean\033[0m\n' "$tag"
    fi
}

# ── ref host suite ──────────────────────────────────────────────────────────
make clean-host >/dev/null 2>&1
printf '\n\033[1;36m━━━━  ASAN + UBSan sweep — ref host suite  ━━━━\033[0m\n'
for t in test-curated test-ntt-rigor test-ntt-rigor-stok \
         test-polymul-integ test-negacyc-integ test-mlkem-kat; do
    run_suite "$t" "$t" HOSTCC=cc CFLAGS="$SAN"
done
make clean-host >/dev/null 2>&1

# ── lib host suites (arith + reduce) ────────────────────────────────────────
# Recipes use $(GCC); overriding GCC injects the sanitizer flags before the
# recipe's own -O2 -Wall -Wextra. Each target rebuilds its own binaries.
printf '\n\033[1;36m━━━━  ASAN + UBSan sweep — lib arith + reduce  ━━━━\033[0m\n'
make -C lib clean >/dev/null 2>&1
for t in test-arith-dual test-arith-fuzz test-arith-gmp test-reduce \
         test-transfer-core test-e2e-oracle; do
    run_suite "lib $t" "$t" -C lib GCC="gcc $SAN"
done
make -C lib clean >/dev/null 2>&1

# ── app compute_e host suites ───────────────────────────────────────────────
printf '\n\033[1;36m━━━━  ASAN + UBSan sweep — app compute_e  ━━━━\033[0m\n'
make -C app/compute_e clean >/dev/null 2>&1
for t in test-binsplit test-host; do
    run_suite "app $t" "$t" -C app/compute_e GCC="gcc $SAN" C_CFLAGS="$SAN -O1"
done
make -C app/compute_e clean >/dev/null 2>&1

if [ "$FAIL" -eq 0 ]; then
    printf '\n  \033[1;32mASAN+UBSan: all suites clean across ref + lib + app\033[0m\n\n'
else
    printf '\n  \033[1;31mASAN+UBSan: sanitizer report(s) present\033[0m\n\n'
fi
exit "$FAIL"
