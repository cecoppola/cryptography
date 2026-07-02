/*
 * test_arith_alias.c — In-place (aliased) safety test for BigInt arithmetic.
 *
 * bigint.c documents specific operand-aliasing guarantees:
 *   bigint_add  — "writes every limb before reading it (alias-safe)"  [c==a, c==b, c==a==b]
 *   bigint_sub  — same claim                                          [c==a, c==b, c==a==b]
 *   bigint_shr  — "reads higher-index limbs before writing lower ones (alias-safe when c==a)"
 * bigint_shl and bigint_mul_u64 zero c first and are NOT alias-safe — deliberately
 * not aliased here. This test verifies the CLAIMED-safe forms against a
 * non-aliased reference over random dual-width inputs (LIMB_BITS 64 and 112), and
 * under the ASAN+UBSan sweep also catches any overlapping-memcpy UB in the shr
 * bit_shift==0 (limb-aligned) aliased path.
 *
 * Built dual-width by lib/Makefile `test-arith-alias`; links bigint.c only.
 */

#include "bigint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static void ck(int cond, const char *what)
{
    if (!cond) { fprintf(stderr, "  FAIL: %s\n", what); failures++; }
}

/* xorshift128+ — deterministic. */
static uint64_t s0 = 0xCAFEF00DD15EA5E5ull, s1 = 0x0123456789ABCDEFull;
static uint64_t rnd(void)
{
    uint64_t x = s0, y = s1;
    s0 = y; x ^= x << 23; s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s1 + y;
}

/* Random trimmed BigInt with 0..max_limbs significant limbs. */
static BigInt rand_bigint(int max_limbs)
{
    int n = (int)(rnd() % (unsigned)(max_limbs + 1));   /* allow 0 (zero value) */
    BigInt a = bigint_alloc(max_limbs + 4);
    for (int i = 0; i < n; i++) {
        unsigned __int128 v = (unsigned __int128)rnd()
                            | ((unsigned __int128)rnd() << 64);
        a.limbs[i] = (limb_t)(v & LIMB_MASK);
    }
    a.n_limbs = n;
    while (a.n_limbs > 0 && a.limbs[a.n_limbs - 1] == 0) a.n_limbs--;
    return a;
}

static BigInt copy_cap(const BigInt *src, int cap)
{
    BigInt c = bigint_alloc(cap);
    bigint_copy(&c, src);
    return c;
}

int main(void)
{
    const int ITERS = 200000;
    const int MAXL   = 12;

    for (int it = 0; it < ITERS; it++) {
        BigInt A = rand_bigint(MAXL);
        BigInt B = rand_bigint(MAXL);
        int maxn = (A.n_limbs > B.n_limbs ? A.n_limbs : B.n_limbs);

        /* ── add: reference then each aliased form ───────────────────────── */
        BigInt ref = bigint_alloc(maxn + 2);
        bigint_add(&ref, &A, &B);

        BigInt caa = copy_cap(&A, maxn + 2);
        bigint_add(&caa, &caa, &B);                 /* c == a */
        ck(bigint_cmp(&ref, &caa) == 0, "add c==a");

        BigInt cab = copy_cap(&B, maxn + 2);
        bigint_add(&cab, &A, &cab);                 /* c == b */
        ck(bigint_cmp(&ref, &cab) == 0, "add c==b");

        BigInt selfref = bigint_alloc(A.n_limbs + 2);
        bigint_add(&selfref, &A, &A);
        BigInt caaa = copy_cap(&A, A.n_limbs + 2);
        bigint_add(&caaa, &caaa, &caaa);            /* c == a == b */
        ck(bigint_cmp(&selfref, &caaa) == 0, "add c==a==b");

        /* ── sub (ensure A >= B) ─────────────────────────────────────────── */
        const BigInt *hi = &A, *lo = &B;
        if (bigint_cmp(&A, &B) < 0) { hi = &B; lo = &A; }
        BigInt sref = bigint_alloc(hi->n_limbs + 2);
        bigint_sub(&sref, hi, lo);

        BigInt saa = copy_cap(hi, hi->n_limbs + 2);
        bigint_sub(&saa, &saa, lo);                 /* c == a */
        ck(bigint_cmp(&sref, &saa) == 0, "sub c==a");

        BigInt sab = copy_cap(lo, hi->n_limbs + 2);
        bigint_sub(&sab, hi, &sab);                 /* c == b */
        ck(bigint_cmp(&sref, &sab) == 0, "sub c==b");

        BigInt sxx = copy_cap(hi, hi->n_limbs + 2);
        bigint_sub(&sxx, &sxx, &sxx);               /* c == a == b -> 0 */
        ck(bigint_is_zero(&sxx), "sub c==a==b -> 0");

        /* ── shr aliased (c == a), across shift regimes ──────────────────── */
        int shifts[5] = { 0, 7, LIMB_BITS, LIMB_BITS + 5, 3 * LIMB_BITS };
        for (int s = 0; s < 5; s++) {
            int bits = shifts[s];
            BigInt rref = bigint_alloc(A.n_limbs + 1);
            bigint_shr(&rref, &A, bits);
            BigInt raa = copy_cap(&A, A.n_limbs + 1);
            bigint_shr(&raa, &raa, bits);           /* c == a */
            ck(bigint_cmp(&rref, &raa) == 0, "shr c==a");
            bigint_free(&rref); bigint_free(&raa);
        }

        /* ── add_u64 / sub_u64 (in-place by design) vs reference ─────────── */
        uint64_t v = rnd();
        BigInt vb = bigint_alloc(2); bigint_set_u64(&vb, v);
        BigInt auref = bigint_alloc(A.n_limbs + 3);
        bigint_add(&auref, &A, &vb);
        BigInt au = copy_cap(&A, A.n_limbs + 3);
        bigint_add_u64(&au, v);
        ck(bigint_cmp(&auref, &au) == 0, "add_u64 in-place");

        /* sub_u64 needs A >= v */
        if (bigint_cmp(&A, &vb) >= 0) {
            BigInt suref = bigint_alloc(A.n_limbs + 1);
            bigint_sub(&suref, &A, &vb);
            BigInt su = copy_cap(&A, A.n_limbs + 1);
            bigint_sub_u64(&su, v);
            ck(bigint_cmp(&suref, &su) == 0, "sub_u64 in-place");
            bigint_free(&suref); bigint_free(&su);
        }

        bigint_free(&A); bigint_free(&B); bigint_free(&ref);
        bigint_free(&caa); bigint_free(&cab);
        bigint_free(&selfref); bigint_free(&caaa);
        bigint_free(&sref); bigint_free(&saa); bigint_free(&sab); bigint_free(&sxx);
        bigint_free(&vb); bigint_free(&auref); bigint_free(&au);
    }

    printf("test_arith_alias (LIMB_BITS=%d): %s (%d failures)\n",
           LIMB_BITS, failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
