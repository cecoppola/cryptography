#pragma once
/*
 * shoup.h — Shoup's modular multiplication algorithm
 *
 * When one operand (twiddle factor w) is fixed per butterfly group, its
 * Shoup reciprocal w_inv = floor(w * 2^64 / p) is precomputed once.
 * The butterfly multiply then costs 7-8 gfx942 instructions vs ~24 naive.
 *
 * Correctness: for any t < p,
 *   q = mulhi64(t, w_inv)    -- q ≈ w*t/p, error at most 1
 *   r = w*t - q*p            -- r = w*t mod p, or r+p
 *   if (r >= p) r -= p       -- at most one fixup: Shoup proves r in [0,2p)
 *
 * Instruction count on gfx942 (PIDX=0, Goldilocks):
 *   mulhi64(t, w_inv): __uint128_t >> 64 → ~4 instructions after DCE
 *   w*t low 64 bits:  → ~3 instructions
 *   q*p low 64 bits:  → ~3 instructions
 *   subtract + fixup: → 2 instructions
 *   Total: ~7 net instructions (compiler schedules, eliminates dead mul_hi)
 *
 * For PIDX=1 (p1, 61-bit): t_hi and w_inv_hi are both < 2^29, so three of
 * the four v_mul_hi_u32 calls in mulhi64 are provably zero. Compiler elides
 * them: ~8 instructions total.
 *
 * ShoupPair stores w and w_inv adjacently so a single 128-bit LDS load
 * fetches both at once during the NTT butterfly stage.
 */

#include <stdint.h>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif
/* primes.h provides __host__/__device__/__forceinline__ no-op fallbacks
 * for the pure-C/host path; mirror that here so shoup.h is host-includable
 * (e.g. for host-side Shoup verification) without ROCm, exactly as
 * primes.h already is. shoup_mul is C++-only (templated), guarded below. */
#include "primes.h"

typedef struct {
    uint64_t w;      /* twiddle factor in [0, p)          */
    uint64_t w_inv;  /* floor(w * 2^64 / p)               */
} ShoupPair;

/* Precompute a ShoupPair on the CPU. Called once per twiddle entry at init.  */
static inline ShoupPair make_shoup(uint64_t w, uint64_t p)
{
    /* w_inv = floor((w << 64) / p).                                          */
    /* Computed with __uint128_t: shift w left 64 bits, divide by p.          */
    /* Result fits in uint64_t since w < p implies w*2^64 < p*2^64.          */
    ShoupPair sp;
    sp.w     = w;
    sp.w_inv = (uint64_t)(
        (unsigned __int128)((unsigned __int128)w << 64)
        / (unsigned __int128)p
    );
    return sp;
}

/* ─── Shoup multiply: (w * t) mod p ─────────────────────────────────────── */
/* PIDX bakes the prime as a compile-time constant (ISA immediate on gfx942). */
/* __forceinline__ is critical: prevents function call overhead in the        */
/* butterfly loop where this is called once per element per stage.            */
/*                                                                             */
/* NOTE: For primes close to 2^64 (all four of our primes), r = w*t - q*p   */
/* can exceed 2^64, causing uint64_t wraparound. We use __uint128_t for the  */
/* subtraction and check the high word to detect overflow.                    */
/* If r128 >= 2^64: r = r128 mod 2^64, and r -= p gives r128 - p (correct), */
/* since 2^64 - p = p mod 2^64 for our primes (unsigned wraparound identity).*/
template<int PIDX>
__host__ __device__ __forceinline__
uint64_t shoup_mul(uint64_t t, uint64_t w, uint64_t w_inv)
{
#ifdef NTT_SOLINAS_BUTTERFLY
    /* A/B EXPERIMENT (2026-05-30): replace Shoup with a Solinas/Goldilocks
     * reduction of the full 128-bit product. For these four 2^64-2^k+1 primes
     * reduce_mod<PIDX> folds with shifts/adds (P0 multiply-free), trading
     * Shoup's two extra reduction multiplies for cheap ALU ops on the
     * VALU-bound butterfly. w_inv is unused in this mode. */
    (void)w_inv;
    return mulmod<PIDX>(t, w);
#elif defined(NTT_SOLINAS_P0) && (PIDX == 0)
    /* I16 (gfx942-only, 2026-06-01): Solinas reduction for P0 = 2^64-2^32+1
     * (Goldilocks). ISA analysis (§I12-ISA, OPT_LEDGER) shows -9 VALU vs Shoup
     * on gfx942 because the Goldilocks fold is pure shift-add (no extra mulhi).
     * P1/P2/P3 cost +9..+17 VALU with Solinas, so only P0 is specialized here.
     * On K4 each prime is an independent APU, so P0's APU runs this path while
     * P1-P3's APUs continue with Shoup.  Wall-clock impact pending G6 PMC. */
    (void)w_inv;
    return mulmod<0>(t, w);
#else
    const uint64_t p = PRIMES[PIDX];

    /* q = high 64 bits of (t * w_inv): ~4 instructions on gfx942.           */
    uint64_t q = (uint64_t)(((__uint128_t)t * w_inv) >> 64);

    /* Full 128-bit subtraction: r128 = w*t - q*p, guaranteed >= 0 by Shoup. */
    /* r128 < 2p by Shoup's theorem.  If r128 >= 2^64 (possible for p>2^63),*/
    /* the low 64 bits r have already "wrapped" by exactly 2^64; subtracting */
    /* p from r in uint64_t wraps again to give r128 - p, which is correct.  */
    __uint128_t r128 = (__uint128_t)w * t - (__uint128_t)q * p;
    uint64_t r = (uint64_t)r128;
    if ((uint64_t)(r128 >> 64) || r >= p) r -= p;
    return r;
#endif /* NTT_SOLINAS_BUTTERFLY / NTT_SOLINAS_P0 */
}
