#pragma once
/*
 * transfer_core.h — Per-element core logic for GPU scatter / gather.
 *
 * Shared host/device building blocks for kernelizing the two host-side stages
 * of the CRT-NTT BigInt-multiply pipeline (arith/ntt_mul.hip steps 1 and 8, on
 * the CPU by default; gated GPU path via -DNTT_GPU_SCATTER_GATHER):
 *
 *   scatter — reduce each input limb mod PRIMES[pidx] and place it at its NTT
 *             coefficient slot (optionally in the I15 transposed layout).
 *   gather  — reconstruct the product BigInt from the Garner U256 coefficient
 *             array (optionally undoing the I15 transposition), as a base-
 *             2^LIMB_BITS positional sum with carry normalization.
 *
 * Design (pull formulation, no atomics):
 *   Each function computes the result for ONE output element from read-only
 *   inputs, so the GPU kernel is one thread per output element and the host
 *   test can call the identical function in a loop. This mirrors the existing
 *   reduce_pK / garner_one_dev pattern (host+device inline, host-tested).
 *
 *   gather is split into two phases so the parallel phase needs no atomics:
 *     phase 1 (parallel)  — gather_acc_limb(): one thread per OUTPUT limb pulls
 *                           every coefficient word/part that lands on that limb
 *                           into a 128-bit partial sum. Independent across limbs.
 *     phase 2 (sequential)— gather_carry_normalize(): a single carry sweep over
 *                           the partial sums. Identical to bigint_gather's carry
 *                           logic; parallelizing it (carry-lookahead) is a
 *                           hardware-side refinement, not needed for correctness.
 *
 * Equivalence: scatter_value / the gather phases reproduce bigint_scatter[_t]
 * and bigint_gather[_t] exactly (verified by arith/test_transfer_core.c at both
 * LIMB_BITS). reduce_prime() dispatches to the same reduce_pK that the NTT
 * butterfly uses (exhaustively checked by test_reduce.c).
 *
 * Included by transfer_kernels.hip (kernels) and arith/test_transfer_core.c
 * (host oracle). LIMB_BITS-dependent (compiled per width, like bigint.c).
 */

#include "arith/bigint.h"   /* BigInt, limb_t, LIMB_BITS, LIMB_MASK, U256, PRIMES, reduce_pK */

/* Number of base-2^LIMB_BITS limb positions a single U256 coefficient spans.
 * LB=64: the four 64-bit words ARE four consecutive limbs.
 * LB=112: a 256-bit value re-chunks into three 112-bit limbs (bigint.c F.6). */
#if LIMB_BITS == 64
#define COEFF_LIMB_SPAN 4
#elif LIMB_BITS == 112
#define COEFF_LIMB_SPAN 3
#else
#error "transfer_core.h: unsupported LIMB_BITS (expected 64 or 112)"
#endif

/* ─── Runtime prime dispatch ─────────────────────────────────────────────────
 * Selects reduce_pK at runtime from pidx. On the GPU pidx is a kernel argument
 * (uniform across the launch), so this is a scalar branch — no per-thread cost.
 * For x < 2^128 the result equals x % PRIMES[pidx] (reduce_pK is exact over the
 * full 128-bit domain; see test_reduce.c). */
__host__ __device__ __forceinline__
uint64_t reduce_prime(__uint128_t x, int pidx)
{
    switch (pidx) {
        case 0:  return reduce_p0(x);
        case 1:  return reduce_p1(x);
        case 2:  return reduce_p2(x);
        default: return reduce_p3(x);   /* pidx == 3 */
    }
}

/* ─── I15 transposition index ────────────────────────────────────────────────
 * For an N = M*M square layout, element x maps to (x%M)*M + (x/M); this is the
 * transpose_sq permutation and is its own inverse on [0, N). transposed==0
 * returns x unchanged (linear / CT-DIT / single-block paths).
 * Inputs:  x in [0, N); M = 2^(log_n/2); transposed in {0,1}.
 * Output:  storage index in [0, N). */
__host__ __device__ __forceinline__
int xpose_index(int x, int M, int transposed)
{
    return transposed ? (x % M) * M + (x / M) : x;
}

/* ─── Scatter: value of one NTT coefficient ──────────────────────────────────
 * Computes coefficient j of operand `limbs` modulo PRIMES[pidx], matching
 * bigint_scatter (linear) / bigint_scatter_t (transposed) for output index j.
 * Inputs:  limbs[0..n_limbs) little-endian; j the OUTPUT coefficient index;
 *          M, transposed select the layout; pidx in [0,4).
 * Output:  limbs[src] % PRIMES[pidx], or 0 when src >= n_limbs (zero-pad).
 * The caller derives src = xpose_index(j, M, transposed). */
__host__ __device__ __forceinline__
uint64_t scatter_value(const limb_t *limbs, int n_limbs, int src, int pidx)
{
    if (src >= n_limbs) return 0;
    return reduce_prime((__uint128_t)limbs[src], pidx);
}

/* ─── Gather phase 1: one U256 coefficient's contribution to limb offset t ────
 * Coefficient k contributes COEFF_LIMB_SPAN base-2^LIMB_BITS limbs at positions
 * k+0 .. k+(SPAN-1). This returns the part landing at offset t (0-based).
 * LB=64:  the t-th 64-bit word verbatim.
 * LB=112: the t-th 112-bit field of the 256-bit value (bigint.c F.6 layout).
 * Inputs:  c the coefficient; t in [0, COEFF_LIMB_SPAN).
 * Output:  the (<= 2^LIMB_BITS) contribution as a 128-bit value. */
__host__ __device__ __forceinline__
__uint128_t gather_part(U256 c, int t)
{
#if LIMB_BITS == 64
    return (__uint128_t)c.w[t];
#else /* LIMB_BITS == 112 */
    __uint128_t w0 = c.w[0], w1 = c.w[1], w2 = c.w[2], w3 = c.w[3];
    switch (t) {
        case 0:  return  w0 | ((w1 & (((__uint128_t)1 << 48) - 1)) << 64);
        case 1:  return (w1 >> 48) | (w2 << 16)
                       | ((w3 & (((__uint128_t)1 << 32) - 1)) << 80);
        default: return  w3 >> 32;   /* t == 2 */
    }
#endif
}

/* ─── Gather phase 1: un-normalized partial sum at one output limb ────────────
 * Pull formulation: output limb j receives the offset-t part of coefficient
 * k = j - t for every t in [0, COEFF_LIMB_SPAN) with k in range. Independent
 * across j, so the GPU kernel is one thread per limb with no atomics. The I15
 * transposition is folded in via xpose_index on the logical coefficient index.
 * Inputs:  coeffs[] the (possibly transposed) Garner U256 array; n_coeffs the
 *          number of logical coefficients; j the output limb index; M,
 *          transposed select the layout.
 * Output:  sum of all parts at position j, < 2^(LIMB_BITS+2) (fits 128 bits):
 *          LB=64 up to 4 words < 2^66; LB=112 up to 3 parts < 2^114.
 * Invariant: the positional sum over all j reproduces bigint_gather's value;
 *            a subsequent carry sweep yields the identical limb array. */
__host__ __device__ __forceinline__
__uint128_t gather_acc_limb(const U256 *coeffs, int n_coeffs,
                            int j, int M, int transposed)
{
    __uint128_t acc = 0;
    for (int t = 0; t < COEFF_LIMB_SPAN; t++) {
        int k = j - t;
        if (k < 0 || k >= n_coeffs) continue;
        acc += gather_part(coeffs[xpose_index(k, M, transposed)], t);
    }
    return acc;
}

/* ─── Gather phase 2: carry-normalize partial sums into limbs ─────────────────
 * Sequential single sweep turning the phase-1 partial sums (each split into
 * lo/hi 64-bit halves) into normalized base-2^LIMB_BITS limbs. Identical to the
 * carry logic in bigint_gather; kept sequential because the carry is tiny
 * (< 2^3) and the sweep is memory-trivial. Parallelizing via carry-lookahead is
 * a hardware-side optimization, deferred.
 * Inputs:  acc_lo[j] | acc_hi[j] = the 128-bit partial sum at limb j; n_out the
 *          output limb count (must be n_coeffs + 4 to absorb top carry).
 * Output:  limbs[0..n_out) normalized, each < 2^LIMB_BITS.
 * Returns: the final carry out of limb n_out-1 (0 on a correctly sized buffer). */
__host__ __device__ __forceinline__
uint64_t gather_carry_normalize(const uint64_t *acc_lo, const uint64_t *acc_hi,
                                limb_t *limbs, int n_out)
{
    __uint128_t carry = 0;
    for (int j = 0; j < n_out; j++) {
        __uint128_t s = (((__uint128_t)acc_hi[j] << 64) | acc_lo[j]) + carry;
        limbs[j] = (limb_t)(s & LIMB_MASK);
        carry = s >> LIMB_BITS;
    }
    return (uint64_t)carry;
}

/* ─── Parallel carry-lookahead (CLA) gather normalization ─────────────────────
 * gather_carry_normalize above is sequential: limb j cannot finish until limb
 * j-1's carry is known, an O(n_out) dependency chain. The carry-lookahead form
 * below makes the carry propagation parallelizable in O(log n_out) depth so the
 * GPU phase-2 kernel is not a single-thread scan. The decomposition is what
 * gets host-verified (arith/test_transfer_core.c, CLA section); the actual
 * GPU prefix scan (Hillis-Steele / Blelloch over cla_compose) is a standard
 * pattern whose monoid is provided here.
 *
 * Algorithm (B = 2^LIMB_BITS, base): each phase-1 partial sum acc[j] is in
 * [0, 4B) (cla_acc_invariant() asserts this). Split acc[j] = ghi[j]*B + r[j]
 * with r[j] in [0,B) and ghi[j] in {0,1,2,3} (the local high part, which belongs
 * to limb j+1). Folding the neighbour in, t[j] = r[j] + ghi[j-1] lies in
 * [0, B+4), so the leftover carry beyond this point is a THIN 0/1 carry — the
 * classic generate/propagate problem:
 *   gen[j]  = t[j] >= B            (limb j makes a thin carry regardless of cin)
 *   res[j]  = t[j] - gen[j]*B      (in [0,B); the digit before the thin carry)
 *   prop[j] = res[j] == B-1        (limb j passes an incoming thin carry through)
 * The carry into limb j is cin[j] = G of the prefix [0..j-1] under the monoid
 *   (Glo,Plo) then (Ghi,Phi)  ->  ( Ghi | (Phi & Glo),  Plo & Phi )
 * and the normalized limb is res[j] + cin[j] (mod B). gen excludes prop (gen=1
 * forces res<=3<B-1), so the monoid is the standard associative CLA operator. */

/* (generate, propagate) pair for the thin-carry prefix scan. */
typedef struct { unsigned char g, p; } ClaPair;

/* Local high part of a phase-1 partial sum: floor(acc / B), in {0,1,2,3}.
 * This is the carry limb j contributes to limb j+1 before thin propagation. */
__host__ __device__ __forceinline__
uint64_t cla_high(uint64_t acc_lo, uint64_t acc_hi)
{
#if LIMB_BITS == 64
    (void)acc_lo;  return acc_hi;                 /* acc >> 64  = hi (< 4)     */
#else /* 112 */
    (void)acc_lo;  return acc_hi >> 48;           /* acc >> 112 = hi>>48 (< 4) */
#endif
}

/* Base-B residue of a phase-1 partial sum: acc & (B-1), in [0, B). */
__host__ __device__ __forceinline__
limb_t cla_low(uint64_t acc_lo, uint64_t acc_hi)
{
#if LIMB_BITS == 64
    (void)acc_hi;  return (limb_t)acc_lo;
#else /* 112 */
    return (limb_t)acc_lo
         | (((limb_t)acc_hi & (((limb_t)1 << 48) - 1)) << 64);
#endif
}

/* Build limb j's (generate, propagate) cell and its pre-thin-carry digit.
 * Inputs:  acc_lo/acc_hi the phase-1 partial sum at limb j; g_prev = cla_high
 *          of limb j-1 (0 when j==0); res_out receives res[j] in [0,B).
 * Output:  ClaPair {gen, prop} for limb j.
 * Invariant (caller): acc[j] < 4B and g_prev < 4, so t = res+g_prev < B+4 and
 *          gen,prop are 0/1 and mutually exclusive. */
__host__ __device__ __forceinline__
ClaPair cla_cell(uint64_t acc_lo, uint64_t acc_hi, uint64_t g_prev, limb_t *res_out)
{
    const __uint128_t B = (__uint128_t)1 << LIMB_BITS;
    __uint128_t t = (__uint128_t)cla_low(acc_lo, acc_hi) + g_prev;  /* < B+4 */
    unsigned char gen = (t >= B) ? 1u : 0u;
    limb_t res = (limb_t)(gen ? (t - B) : t);                       /* [0,B) */
    *res_out = res;
    ClaPair c; c.g = gen; c.p = (res == (limb_t)LIMB_MASK) ? 1u : 0u;
    return c;
}

/* Associative CLA monoid: combine a lower-index block `lo` with the adjacent
 * higher-index block `hi`. Returns the combined block's (generate, propagate).
 * carry_out(combined, cin) = hi.g | (hi.p & (lo.g | (lo.p & cin))). */
__host__ __device__ __forceinline__
ClaPair cla_compose(ClaPair lo, ClaPair hi)
{
    ClaPair r;
    r.g = hi.g | (hi.p & lo.g);
    r.p = lo.p & hi.p;
    return r;
}

/* Final normalized limb from its pre-carry digit and the scanned thin carry-in.
 * res in [0,B), cin in {0,1}; res+cin in [0,B], and == B maps to 0 via the mask
 * (that case is exactly a propagated carry leaving limb j). */
__host__ __device__ __forceinline__
limb_t cla_finalize(limb_t res, uint64_t cin)
{
    return (limb_t)(((__uint128_t)res + cin) & LIMB_MASK);
}
