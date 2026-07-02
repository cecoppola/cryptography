/*
 * newton.c — Newton-Raphson reciprocal for large-integer division.
 *
 * newton_recip_at_scale(x, Q, scale) computes x ~= floor(2^scale / Q).
 * newton_reciprocal(x, Q) is the public API: delegates with scale=2*bits(Q).
 *
 * The Newton iteration converges from below, so the result r is a LOWER BOUND:
 * r <= floor(2^scale / Q), reaching equality when it fully converges (the
 * common large-Q case) and stopping at most a couple short otherwise (e.g.
 * Q=1 yields floor-1: the correction delta underflows before the last step).
 * This is intentional and safe — bigint_div_newton's bounded correction loop
 * absorbs the small slack. Callers needing an exact floor must not assume it.
 *
 * Seed derivation (bits(Q)=b, scale arbitrary):
 *   Q_top = top 64 bits of Q = floor(Q/2^(b-64)); ensures
 *   Q < (Q_top+1)*2^(b-64).  x0 = floor((2^128-1)/(Q_top+1)) ≈ 2^(b+64)/Q.
 *   Scale x0 by 2^(scale-(b+64)) to obtain x ≈ 2^scale/Q (initial 64-bit
 *   precision in relative terms; each Newton step doubles correct bits).
 *
 * bigint_div_newton computes floor(N/D) with at most 2 corrections:
 *   - when bits(N) ≤ 2*bits(D): uses the caller-supplied reciprocal directly.
 *   - when bits(N) > 2*bits(D): computes a fresh internal reciprocal at
 *     scale = bits(N)+bits(D)+1 so the error is bounded by 1, restoring O(1).
 *   CORRECTION_CAP of 4 catches a genuinely non-converging spin (corrupt
 *   upstream product); the expected maximum in correct operation is 2.
 */

#include "newton.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

/* Set x = 2^bits (single set bit). */
static void set_pow2(BigInt *x, int bits)
{
    int limb = bits / LIMB_BITS, bit = bits % LIMB_BITS;
    bigint_resize(x, limb + 1);
    bigint_zero(x);
    x->limbs[limb] = (limb_t)1 << bit;
    x->n_limbs = limb + 1;
}

/* Return bits [p, p+64) of Q as a uint64_t (limb-width-generic). Bit p
 * is in limb p/LIMB_BITS at offset p%LIMB_BITS; the 64-bit window spans
 * at most two limbs. At LIMB_BITS=64 this equals the former
 * `Q->limbs[ls]>>bs | Q->limbs[ls+1]<<(64-bs)`. */
static uint64_t bits_window_64(const BigInt *Q, int p)
{
    int L = p / LIMB_BITS, off = p % LIMB_BITS;
    unsigned __int128 lo = (L     < Q->n_limbs) ? (unsigned __int128)Q->limbs[L]     : 0;
    unsigned __int128 hi = (L + 1 < Q->n_limbs) ? (unsigned __int128)Q->limbs[L + 1] : 0;
    unsigned __int128 w = lo >> off;
    if (off > 0) w |= hi << (LIMB_BITS - off);
    return (uint64_t)w;
}

/* ─── Internal: compute floor(2^scale / Q) for arbitrary scale ───────────── */
/* Public API wrapper newton_reciprocal() calls this with scale=2*bits(Q).   */
static void newton_recip_at_scale(BigInt *x, const BigInt *Q, int scale)
{
    int b   = bigint_bits(Q);
    int cap = (scale / LIMB_BITS) + 8;

    /* Seed: x0 ≈ 2^128 / Q_top ≈ 2^(b+64) / Q.  After shifting by
     * (scale - (b+64)), x ≈ 2^scale / Q with ~64 bits of initial precision.
     * Each Newton step doubles the correct bit-count; convergence in
     * ceil(log2(scale/64)) iterations. */
    uint64_t Q_top = (b <= 64) ? (uint64_t)Q->limbs[0]
                               : bits_window_64(Q, b - 64);
    uint64_t denom = (Q_top < UINT64_MAX) ? Q_top + 1 : UINT64_MAX;
    unsigned __int128 x0 = (unsigned __int128)(-1) / denom;

    bigint_resize(x, cap);
    bigint_zero(x);
    x->limbs[0] = (limb_t)(x0 & LIMB_MASK);
    x->limbs[1] = (limb_t)(x0 >> LIMB_BITS);
    x->n_limbs  = (x->limbs[1] != 0) ? 2 : 1;

    /* x0 ≈ 2^x0_bits / Q where x0_bits depends on b:
     *   b > 64: Q_top ≈ Q>>( b-64), so x0 ≈ 2^128 / (Q>>(b-64)) = 2^(b+64)/Q.
     *   b ≤ 64: Q_top = full Q (≤64 bits),          so x0 ≈ 2^128 / Q.
     * Shift x by (scale - x0_bits) to get x ≈ 2^scale/Q. */
    int x0_bits = (b > 64) ? (b + 64) : 128;
    BigInt tmp = bigint_alloc(cap);
    int shift = scale - x0_bits;
    if (shift > 0) {
        bigint_shl(&tmp, x, shift);
        bigint_copy(x, &tmp);
    } else if (shift < 0) {
        bigint_shr(&tmp, x, -shift);
        bigint_copy(x, &tmp);
    }

    /* Newton iterations: x += floor(x*(2^scale - Q*x) / 2^scale) */
    BigInt two_k = bigint_alloc((scale / LIMB_BITS) + 2);
    BigInt e     = bigint_alloc(cap);
    BigInt delta = bigint_alloc(cap);
    set_pow2(&two_k, scale);

    for (int iter = 0; iter < 200; iter++) {
        bigint_mul(&tmp, Q, x);
        int cmp = bigint_cmp(&tmp, &two_k);
        if (cmp == 0) break;
        if (cmp > 0) { bigint_sub_u64(x, 1); break; }
        bigint_sub(&e, &two_k, &tmp);
        bigint_mul(&delta, x, &e);
        bigint_shr(&tmp, &delta, scale);
        if (bigint_is_zero(&tmp)) break;
        bigint_add(&delta, x, &tmp);
        bigint_copy(x, &delta);
    }

    bigint_free(&tmp);
    bigint_free(&two_k);
    bigint_free(&e);
    bigint_free(&delta);
}

void newton_reciprocal(BigInt *x, const BigInt *Q)
{
    if (bigint_is_zero(Q)) {
        fprintf(stderr, "newton_reciprocal: Q is zero\n");
        return;
    }
    newton_recip_at_scale(x, Q, 2 * bigint_bits(Q));
}

void bigint_div_newton(BigInt *q, const BigInt *N, const BigInt *D,
                       const BigInt *recip)
{
    int b      = bigint_bits(D);
    int bits_N = bigint_bits(N);
    int scale  = 2 * b;

    /* Choose the reciprocal to use.  The caller-supplied recip = floor(2^(2b)/D)
     * gives an O(1) correction only when bits(N) ≤ 2b.  When bits(N) > 2b, the
     * error in q = floor(N*recip/2^(2b)) can be ~N/2^(2b) which is large (e.g.
     * 123456789/7: bits_N=27, 2b=6, error≈275573).  Fix: compute a fresh
     * reciprocal at scale = bits_N+b+1 so error ≤ 1, restoring O(1) correction.
     * Base-convert (the hot caller) always satisfies bits(N) ≤ 2b structurally
     * and never pays this cost. */
    BigInt local_recip;
    const BigInt *use_recip;
    if (bits_N <= scale) {
        use_recip = recip;
    } else {
        scale = bits_N + b + 1;
        local_recip = bigint_alloc((scale / LIMB_BITS) + 8);
        newton_recip_at_scale(&local_recip, D, scale);
        use_recip = &local_recip;
    }

    /* q ≈ floor(N * use_recip / 2^scale).  With the correct scale the error
     * is at most 2, so the correction below runs at most 2 iterations. */
    int cap = N->n_limbs + use_recip->n_limbs + 4;
    BigInt tmp = bigint_alloc(cap);
    bigint_mul(&tmp, N, use_recip);
    bigint_shr(q, &tmp, scale);
    bigint_free(&tmp);

    if (bits_N > 2 * b) bigint_free(&local_recip);

    /* ── Correction ─────────────────────────────────────────────────────── */
    /* Adjust q by at most 2 (provably O(1) when the scale above is chosen
     * correctly).  The cap catches a genuinely non-converging spin from a
     * corrupt upstream product (bad NTT multiply); expected max is 2. */
    BigInt qD = bigint_alloc(q->n_limbs + D->n_limbs + 2);
    BigInt r  = bigint_alloc(N->n_limbs + 2);
    bigint_mul(&qD, q, D);

    #define CORRECTION_CAP 4
    long corr = 0;

    /* If q*D > N, q is too large; reduce. */
    while (bigint_cmp(&qD, N) > 0) {
        bigint_sub_u64(q, 1);
        bigint_sub(&qD, &qD, D);
        if (++corr > CORRECTION_CAP) {
            fprintf(stderr, "newton: division correction did not converge "
                    "(>%d down-steps) — likely a corrupt product/reciprocal. "
                    "Aborting cleanly instead of spinning.\n", CORRECTION_CAP);
            exit(1);
        }
    }

    /* r = N - q*D; while r >= D, increment q. */
    bigint_sub(&r, N, &qD);
    corr = 0;
    while (bigint_cmp(&r, D) >= 0) {
        bigint_add_u64(q, 1);
        bigint_sub(&r, &r, D);
        if (++corr > CORRECTION_CAP) {
            fprintf(stderr, "newton: division correction did not converge "
                    "(>%d up-steps) — likely a corrupt product/reciprocal. "
                    "Aborting cleanly instead of spinning.\n", CORRECTION_CAP);
            exit(1);
        }
    }

    bigint_free(&qD);
    bigint_free(&r);
}
#undef CORRECTION_CAP
