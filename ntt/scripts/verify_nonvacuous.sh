#!/bin/bash
# verify_nonvacuous.sh — fault-injection proof that the host test
# suites actually detect a correctness regression. For each
# (label, source-file, exact-line, suite[, dir]) tuple, corrupt the file
# (literal replace via Python, no regex), rebuild, run, expect non-zero.
# Then restore + final baseline regression must pass.
#
# Coverage (F4 2026-06-24; +test-modops 2026-06-29): ref (curated/ntt-rigor/
# stok/polymul/negacyc) + lib (reduce, arith, transfer-core, arith-gmp, modops)
# + app (binsplit, compute_e host). All HOST-RUN (no kernel launch); test-modops
# is hipcc-compiled but runs its modular-op checks on host, so it fits here too.
set -u
cd "$(dirname "$0")/.."

FAIL=0

# Args: label, target, old_line, new_line, suite, [dir]
# dir empty  -> root host build  (make clean-host; make <suite> HOSTCC=cc)  [ref]
# dir set    -> sub-make build   (make -C <dir> clean; make -C <dir> <suite>) [lib/app]
inject() {
    local label="$1" target="$2" old="$3" new="$4" suite="$5" dir="${6:-}"
    local bak="/tmp/$(basename "$target").bak.$$"
    cp "$target" "$bak"
    python3 -c '
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if old not in s:
    sys.exit("PATTERN NOT FOUND: " + old)
open(p, "w").write(s.replace(old, new, 1))
' "$target" "$old" "$new" || { printf '  [%s] inject failed\n' "$label"; cp "$bak" "$target"; FAIL=1; return; }

    local rc=0
    if [ -n "$dir" ]; then
        make -C "$dir" clean >/dev/null 2>&1
        make -C "$dir" "$suite" >/tmp/inj_$$.log 2>&1 || rc=$?
        make -C "$dir" clean >/dev/null 2>&1
    else
        make clean-host >/dev/null 2>&1
        make "$suite" HOSTCC=cc >/tmp/inj_$$.log 2>&1 || rc=$?
    fi
    cp "$bak" "$target"
    rm -f "$bak"

    if [ "$rc" -eq 0 ]; then
        printf '  [%-22s] \033[1;31mVACUOUS\033[0m — suite PASSED with corruption!\n' "$label"
        FAIL=1
    else
        printf '  [%-22s] \033[1;32mNON-VACUOUS\033[0m — corruption -> rc=%s (expected)\n' "$label" "$rc"
    fi
    rm -f /tmp/inj_$$.log
}

printf '\n\033[1;36m━━━━  Fault-injection non-vacuity proof  ━━━━\033[0m\n'

OLD_CPU='tw[k] = (uint64_t)((__uint128_t)tw[k - 1] * p->omega % p->q);'
NEW_CPU='tw[k] = (uint64_t)((__uint128_t)tw[k - 1] * p->omega % (p->q - 1));'
OLD_PM='tw[k] = (uint64_t)((__uint128_t)tw[k-1] * p->omega % p->q);'
NEW_PM='tw[k] = (uint64_t)((__uint128_t)tw[k-1] * p->omega % (p->q - 1));'

inject "test-curated"        "ref/ntt_cpu.c"               "$OLD_CPU" "$NEW_CPU" test-curated
inject "test-ntt-rigor"      "ref/ntt_cpu.c"               "$OLD_CPU" "$NEW_CPU" test-ntt-rigor
inject "test-ntt-rigor-stok" "ref/ntt_stockham.c"          "$OLD_CPU" "$NEW_CPU" test-ntt-rigor-stok
inject "test-polymul-integ"  "ref/ntt_polymul.c"           "$OLD_PM"  "$NEW_PM"  test-polymul-integ
inject "test-negacyc-integ"  "ref/ntt_polymul_negacyclic.c" "$OLD_PM" "$NEW_PM"  test-negacyc-integ

# ── lib host suites (F4 2026-06-24: extend the gate past ref) ──────────────
# Each corrupts a distinct production path and expects the host suite to fail.
inject "lib test-reduce" "lib/primes.h" \
  '    __uint128_t R = (__uint128_t)m_hi * C1 + m_lo;  /* < 2*P1 */' \
  '    __uint128_t R = (__uint128_t)m_hi * C1 + m_lo + 1;  /* PERTURB */' \
  test-reduce lib
inject "lib test-arith" "lib/arith/bigint.c" \
  '        carry = (limb_t)(s >> LIMB_BITS);' \
  '        carry = 0; /* PERTURB */' \
  test-arith lib
inject "lib test-transfer-core" "lib/transfer_core.h" \
  '    return (__uint128_t)c.w[t];' \
  '    return (__uint128_t)c.w[t] + 1; /* PERTURB */' \
  test-transfer-core lib
inject "lib test-arith-gmp" "lib/arith/multiply.c" \
  '            c->limbs[i + j] = (limb_t)(acc & LIMB_MASK);' \
  '            c->limbs[i + j] = (limb_t)((acc + 1) & LIMB_MASK); /* PERTURB */' \
  test-arith-gmp lib
# test-modops (Phase B.2, 2026-06-29): hipcc-compiled but HOST-RUN (no kernel
# launch) — exercises addmod/submod/mulmod/montgomery_mul/shoup_mul vs an
# independent __uint128_t reference for all 4 primes. Corrupt addmod's reduction.
inject "lib test-modops" "lib/primes.h" \
  '    if (r >= PRIMES[PIDX]) r -= PRIMES[PIDX]; /* non-overflow reduction    */' \
  '    if (r >= PRIMES[PIDX]) r -= PRIMES[PIDX] + 1; /* PERTURB */' \
  test-modops lib

# ── app host suites ────────────────────────────────────────────────────────
inject "app test-binsplit" "app/compute_e/binary_split.c" \
  '    bigint_mul(&out->B, &L.B, &R.B);' \
  '    bigint_mul(&out->B, &L.B, &L.B); /* PERTURB */' \
  test-binsplit app/compute_e
inject "app test-host" "lib/arith/newton.c" \
  '        bigint_mul(&tmp, Q, x);' \
  '        bigint_mul(&tmp, Q, Q); /* PERTURB */' \
  test-host app/compute_e

make clean-host >/dev/null 2>&1
make -C lib clean >/dev/null 2>&1
make -C app/compute_e clean >/dev/null 2>&1
printf '\n\033[1;36m━━━━  Baseline regression (post-restore)  ━━━━\033[0m\n'
for t in test-curated test-ntt-rigor test-ntt-rigor-stok test-polymul-integ test-negacyc-integ; do
    if make "$t" HOSTCC=cc >/tmp/base_$$.log 2>&1; then
        printf '  [%-22s] \033[1;32mPASS\033[0m\n' "$t"
    else
        printf '  [%-22s] \033[1;31mFAIL (baseline broken!)\033[0m\n' "$t"
        FAIL=1
    fi
done
# lib/app host suites restored cleanly?  (label | dir | suite)
for ent in "lib test-reduce|lib|test-reduce" \
           "lib test-arith|lib|test-arith" \
           "lib transfer-core|lib|test-transfer-core" \
           "lib arith-gmp|lib|test-arith-gmp" \
           "lib modops|lib|test-modops" \
           "app binsplit|app/compute_e|test-binsplit" \
           "app host|app/compute_e|test-host"; do
    lbl="${ent%%|*}"; rest="${ent#*|}"; d="${rest%%|*}"; s="${rest#*|}"
    if make -C "$d" "$s" >/tmp/base_$$.log 2>&1; then
        printf '  [%-22s] \033[1;32mPASS\033[0m\n' "$lbl"
    else
        printf '  [%-22s] \033[1;31mFAIL (baseline broken!)\033[0m\n' "$lbl"
        FAIL=1
    fi
    make -C "$d" clean >/dev/null 2>&1
done
rm -f /tmp/base_$$.log

exit "$FAIL"
