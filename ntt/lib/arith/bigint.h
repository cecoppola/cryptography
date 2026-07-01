#pragma once
/*
 * bigint.h — Arbitrary-precision unsigned integer for the CRT-NTT arith layer.
 *
 * Representation: little-endian array of limb_t limbs.
 *   value = sum_{i=0}^{n_limbs-1} limbs[i] * 2^(LIMB_BITS*i)
 *
 * Phase F dual-build (perf/results/PHASE_F_DESIGN.md): LIMB_BITS is a
 * compile-time switch — 64 (default; limb_t = uint64_t) or 112 (limb_t =
 * unsigned __int128, valid bits = LIMB_BITS, high bits masked by
 * LIMB_MASK). Built twice; the 64-bit build is the permanent differential
 * oracle for the 112-bit path. The limb-width dependence is NOT isolated
 * to the typedef — see the design doc's per-function migration map (the
 * old "typedef only" claim was audited false). Callers see only BigInt*.
 *
 * Memory: caller-managed via bigint_alloc / bigint_free.
 * All output arguments must be pre-allocated to sufficient capacity.
 * Use bigint_needed_limbs() to compute required capacity before allocation.
 */

#include <stdint.h>
#include <stddef.h>
#include "../crt_ntt.h"   /* U256 */
#include "../primes.h"     /* PRIMES[] */

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Limb width (Phase F dual-build; PHASE_F_DESIGN.md) ──────────────────── */
#ifndef LIMB_BITS
#define LIMB_BITS 64
#endif
#if LIMB_BITS == 64
typedef uint64_t limb_t;
#define LIMB_MASK (~(uint64_t)0)                       /* all 64 bits      */
#elif LIMB_BITS == 112
typedef unsigned __int128 limb_t;
#define LIMB_MASK ((((unsigned __int128)1) << 112) - 1)
#else
#error "LIMB_BITS must be 64 or 112"
#endif
#define LIMB_BYTES ((int)sizeof(limb_t))               /* storage cell size */

/* ─── Full limb*limb -> (hi:lo) via SUBLIMB split ────────────────────────────
 * limb = a0 + a1*2^S, S = LIMB_BITS/2. Every partial product is
 * < 2^LIMB_BITS <= 2^112, so all intermediates fit __uint128_t (a direct
 * limb*limb would be 2^224 at 112-bit). At LB=64 (S=32) this yields the
 * exact same value as (unsigned __int128)a*b — F.4 is a no-op there. */
#define SUBLIMB_BITS (LIMB_BITS / 2)
#define SUBLIMB_MASK ((((unsigned __int128)1) << SUBLIMB_BITS) - 1)
static inline void mul_limb(limb_t a, limb_t b, limb_t *lo, limb_t *hi)
{
    unsigned __int128 a0 = (unsigned __int128)a & SUBLIMB_MASK;
    unsigned __int128 a1 = (unsigned __int128)a >> SUBLIMB_BITS;
    unsigned __int128 b0 = (unsigned __int128)b & SUBLIMB_MASK;
    unsigned __int128 b1 = (unsigned __int128)b >> SUBLIMB_BITS;
    unsigned __int128 p0  = a0 * b0;
    unsigned __int128 p3  = a1 * b1;
    unsigned __int128 mid = a0 * b1 + a1 * b0;
    unsigned __int128 lo_acc = p0 + ((mid & SUBLIMB_MASK) << SUBLIMB_BITS);
    unsigned __int128 hi_acc = p3 + (mid >> SUBLIMB_BITS)
                                  + (lo_acc >> LIMB_BITS);
    *lo = (limb_t)(lo_acc & LIMB_MASK);
    *hi = (limb_t)(hi_acc & LIMB_MASK);
}

/* ─── Core type ──────────────────────────────────────────────────────────── */
typedef struct {
    limb_t   *limbs;   /* little-endian, LIMB_BITS valid bits per limb    */
    int       n_limbs; /* number of significant limbs (trimmed)          */
    int       cap;     /* allocated capacity in limbs                    */
    int       managed; /* 1 if limbs are hipMallocManaged; 0 if malloc   */
} BigInt;

/* ─── Allocation ─────────────────────────────────────────────────────────── */
BigInt  bigint_alloc(int cap);
/* Allocate limbs via hipMallocManaged (GPU-accessible, HBM3 on MI300A).
 * Falls back to calloc when linked without bigint_hip_alloc.o. */
BigInt  bigint_alloc_managed(int cap);
void    bigint_free(BigInt *a);
void    bigint_resize(BigInt *a, int new_cap); /* realloc; preserves value  */

/* ─── Assignment ─────────────────────────────────────────────────────────── */
void    bigint_zero(BigInt *a);
void    bigint_set_u64(BigInt *a, uint64_t v);
void    bigint_copy(BigInt *dst, const BigInt *src);

/* ─── Comparison ─────────────────────────────────────────────────────────── */
int     bigint_is_zero(const BigInt *a);
int     bigint_cmp(const BigInt *a, const BigInt *b);   /* -1 / 0 / +1   */

/* ─── Arithmetic ─────────────────────────────────────────────────────────── */
/* c = a + b.  c must have cap >= max(a->n_limbs, b->n_limbs) + 1.          */
void    bigint_add(BigInt *c, const BigInt *a, const BigInt *b);
/* c = a - b.  Requires a >= b (unsigned).                                   */
void    bigint_sub(BigInt *c, const BigInt *a, const BigInt *b);
/* a += v  (scalar, in-place).  a must have cap > a->n_limbs.               */
void    bigint_add_u64(BigInt *a, uint64_t v);
/* a -= v  (scalar, in-place).  Requires a >= v (unsigned).                  */
void    bigint_sub_u64(BigInt *a, uint64_t v);
/* c = a << bits.  c must have cap >= a->n_limbs + (bits+LIMB_BITS-1)/LIMB_BITS
 * (limb width is LIMB_BITS = 64 or 112, NOT hardcoded 64).                 */
void    bigint_shl(BigInt *c, const BigInt *a, int bits);
/* c = a >> bits.                                                             */
void    bigint_shr(BigInt *c, const BigInt *a, int bits);
/* c = a * v (scalar).  c must have cap >= a->n_limbs + 1.                  */
void    bigint_mul_u64(BigInt *c, const BigInt *a, uint64_t v);

/* ─── NTT interface ──────────────────────────────────────────────────────── */
/* Scatter: reduce each limb mod PRIMES[pidx]; write n_coeffs coefficients.
 * Limbs beyond a->n_limbs are written as 0 (zero-padding for NTT).
 * coeffs must hold n_coeffs uint64_t.                                       */
void    bigint_scatter(const BigInt *a, uint64_t *coeffs,
                       int n_coeffs, int pidx);

/* Transposed scatter (I15, Bailey 4-step only): writes coeffs[c*M + r] =
 * limbs[r*M + c] % p.  Eliminates the initial transpose_sq GPU kernel.
 * N = M*M must be a perfect square (even log_n > 10).  coeffs holds N entries. */
void    bigint_scatter_t(const BigInt *a, uint64_t *coeffs, int M, int pidx);

/* Gather: reconstruct a BigInt from an array of n_coeffs Garner U256
 * outputs.  Each U256 is the i-th polynomial coefficient; the BigInt
 * value is sum_{i} coeff[i] * 2^(LIMB_BITS*i) — scatter and gather are a
 * consistent base-2^LIMB_BITS pair (LIMB_BITS = 64 or 112, NOT fixed 64).
 * a must have cap >= n_coeffs + 4 to hold carry overflow.                   */
void    bigint_gather(BigInt *a, const U256 *coeffs, int n_coeffs);

/* Transposed gather (I15, Bailey 4-step only): coefficient k = r*M + c is
 * stored at coeffs[c*M + r].  Undoes the I15 scatter transposition so the
 * reconstructed BigInt is identical to the original.  M must match the M
 * used at scatter time (M = 2^(log_n/2)).                                    */
void    bigint_gather_t(BigInt *a, const U256 *coeffs, int n_coeffs, int M);

/* ─── Utility ────────────────────────────────────────────────────────────── */
/* Bits in value (0 for zero). */
int     bigint_bits(const BigInt *a);
/* Conservative limb count for n decimal digits. The /64 is a DELIBERATE
 * upper bound, not a LIMB_BITS bug: dividing the bit estimate by 64 yields
 * >= the true limb count for any LIMB_BITS >= 64, so it over-allocates at
 * LIMB_BITS=112 and never under-allocates. NOTE: currently unused (no
 * caller in the tree); kept as a safe helper. */
static inline int bigint_limbs_for_digits(int n_digits)
{
    /* log2(10) ≈ 3.32193 (·332194/100000); +4 limbs guard. */
    return (int)((uint64_t)n_digits * 332194 / 100000 / 64) + 4;
}
/* Debug helper: prints each limb as a 64-bit hex word (low 64 bits of the
 * limb at LIMB_BITS=112 — diagnostic only, not used in computation). */
void    bigint_print_hex(const BigInt *a, const char *label);

#ifdef __cplusplus
}
#endif
