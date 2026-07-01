/*
 * binary_split.c — Binary splitting for e = sum_{k=0}^{N} 1/k!.
 *
 * Merge formula (derived from invariant: sum_{k=a}^{b-1} 1/k! = P/(b-1)!):
 *   Left  [a, m): P_L, B_L where B_L = (m-1)!/(a-1)!
 *   Right [m, b): P_R, B_R where B_R = (b-1)!/(m-1)!
 *   Combined: P = P_L*B_R + P_R,  B = B_L*B_R  (= (b-1)!/(a-1)!)
 *
 * At root (a=1, b=N+1):
 *   B_root = N!,  e - 1 = P_root / B_root.
 */

#include "binary_split.h"
#include "../../lib/arith/multiply.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void split_free(SplitResult *s)
{
    bigint_free(&s->P);
    bigint_free(&s->B);
}

/* ─── Recursion ─────────────────────────────────────────────────────────── */

void binary_split(SplitResult *out, long a, long b, MemPool *pool)
{
    /* Defensive contract (added 2026-05-19 adversarial audit):
     *   b <= a -> empty range identity.  sum over [a, a) is 0 and the
     *   partial factorial product over an empty range is 1.  Without this
     *   guard, m = a + (b - a)/2 = a, and binary_split(a, a) recurses
     *   infinitely (stack overflow).  Main.c never reaches this with the
     *   current term-count estimator but the binary_split() API is public
     *   and adversarial inputs (a = b, b < a) must not silently hang. */
    if (b <= a) {
        out->P = (pool) ? mem_pool_get(pool, 1) : bigint_alloc(1);
        out->B = (pool) ? mem_pool_get(pool, 1) : bigint_alloc(1);
        bigint_set_u64(&out->P, 0);
        bigint_set_u64(&out->B, 1);
        return;
    }
    if (b - a == 1) {
        /* Leaf: term 1/a!.  P = 1, B = a. */
        out->P = (pool) ? mem_pool_get(pool, 1) : bigint_alloc(1);
        out->B = (pool) ? mem_pool_get(pool, 2) : bigint_alloc(2);
        bigint_set_u64(&out->P, 1);
        bigint_set_u64(&out->B, (uint64_t)a);
        return;
    }

    long m = a + (b - a) / 2;   /* split point */

    SplitResult L, R;
    binary_split(&L, a, m, pool);
    binary_split(&R, m, b, pool);

    /* B = B_L * B_R */
    int bcap = L.B.n_limbs + R.B.n_limbs + 2;
    out->B = (pool) ? mem_pool_get(pool, bcap) : bigint_alloc(bcap);
    bigint_mul(&out->B, &L.B, &R.B);

    /* P = P_L * B_R + P_R */
    int pcap = L.P.n_limbs + R.B.n_limbs + 2;
    BigInt pLbR = (pool) ? mem_pool_get(pool, pcap) : bigint_alloc(pcap);
    bigint_mul(&pLbR, &L.P, &R.B);

    int pcap2 = (pLbR.n_limbs > R.P.n_limbs ? pLbR.n_limbs : R.P.n_limbs) + 2;
    out->P = (pool) ? mem_pool_get(pool, pcap2) : bigint_alloc(pcap2);
    bigint_add(&out->P, &pLbR, &R.P);

    /* Free temporaries and children. */
    if (pool) {
        mem_pool_put(pool, &pLbR);
        mem_pool_put(pool, &L.P);
        mem_pool_put(pool, &L.B);
        mem_pool_put(pool, &R.P);
        mem_pool_put(pool, &R.B);
    } else {
        bigint_free(&pLbR);
        bigint_free(&L.P);
        bigint_free(&L.B);
        bigint_free(&R.P);
        bigint_free(&R.B);
    }
}

/* ─── Term count estimation ─────────────────────────────────────────────── */

/*
 * We need sum_{k=N+1}^{inf} 1/k! < 10^(-D).
 * Bound: tail < 2/N! for large N.  Need N! > 2 * 10^D.
 * Stirling: ln(N!) ≈ N*ln(N) - N.  Solve N*ln(N) - N > D*ln(10) + ln(2).
 * Iterate fixed-point: N = (D*ln(10)) / (ln(N) - 1).
 */
long split_terms_needed(long n_digits)
{
    double target = (double)n_digits * log(10.0);
    double N = target / log(target);   /* initial estimate */
    for (int i = 0; i < 50; i++) {
        double logN = log(N);
        if (logN <= 1.0) { N = target; break; }
        N = target / (logN - 1.0);
    }
    /* Add 5% safety margin and round up. */
    long result = (long)(N * 1.05) + 64;
    return result;
}
