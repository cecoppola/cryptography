#pragma once
/*
 * primes.h — Prime constants and modular arithmetic primitives
 *
 * Four NTT-friendly primes of the form 2^a - 2^b + 1.
 * Max NTT size: P0:2^32, P1:2^24, P2:2^34, P3:2^40 (minimum is 2^24).
 * Product p0*p1*p2*p3 ≈ 2^255 (> 2^248 needed for 112-bit inputs at n≤2^24).
 *
 * Reduction principle: 2^a ≡ 2^b - 1 (mod p).
 * For 128-bit product x = x_hi*2^64 + x_lo (all a=64: free split):
 *   x ≡ x_lo + x_hi*(2^b - 1)  (mod p)
 * Reduction method (all four primes):
 *   p0 b=32: 1 explicit round using 32-bit half-split (reduce_p0)
 *   p1 b=24: 2 Solinas rounds via __uint128_t; m_hi < 2^24 after round 1
 *   p2 b=34: 2 Solinas rounds + 4-bit tail round
 *   p3 b=40: 2 Solinas rounds + 16-bit tail round
 *
 * On gfx942: all VGPRs are 32-bit; uint64_t = two adjacent VGPRs.
 * __uint128_t arithmetic lowers to v_mul_lo_u32, v_mul_hi_u32,
 * v_add_u32, v_sub_u32 — all 1-cycle throughput on CDNA3.
 * __forceinline__ is mandatory to prevent call overhead in the
 * butterfly inner loop.
 */

#include <stdint.h>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
/* Allow pure-C compilation of arith/ host code without ROCm. */
#define __host__
#define __device__
#define __forceinline__ inline
#endif

/* ─── Prime constants (verified prime, verified NTT-friendly) ─────────────── */
/* P0: 2^64 - 2^32 + 1  (Goldilocks) */
#define P0  UINT64_C(0xffffffff00000001)
/* P1: 2^64 - 2^24 + 1 */
#define P1  UINT64_C(0xffffffffff000001)
/* P2: 2^64 - 2^34 + 1 */
#define P2  UINT64_C(0xfffffffc00000001)
/* P3: 2^64 - 2^40 + 1 */
#define P3  UINT64_C(0xffffff0000000001)

static const uint64_t PRIMES[4]    = { P0, P1, P2, P3 };
static const uint64_t PRIM_ROOT[4] = { 7, 43, 10, 19 };
static const int      PRIME_B[4]   = { 32, 24, 34, 40 };
static const int      PRIME_A[4]   = { 64, 64, 64, 64 };

/* Declared here; defined in garner.hip */
int verify_primes(void);

/* ─── Reduction: p0 = 2^64 - 2^32 + 1 (Goldilocks) ──────────────────────── */
/* a=64: x_hi and x_lo are the natural 64-bit halves — free split.            */
/* Carry after round 1: x_hi >> 32 (32 bits). One round suffices because      */
/* 32-bit carry * 2^32 fits exactly in 64 bits with no further overflow.      */
__host__ __device__ __forceinline__
uint64_t reduce_p0(__uint128_t x)
{
    const uint64_t EPS = UINT64_C(0xffffffff); /* 2^32 - 1 */
    uint64_t x_lo  = (uint64_t)x;
    uint64_t x_hi  = (uint64_t)(x >> 64);
    uint64_t x_hhi = x_hi >> 32;   /* upper 32 bits: the carry out of << 32 */
    uint64_t x_hlo = x_hi & EPS;   /* lower 32 bits: the value that shifts  */

    /* t0 = x_lo - x_hhi  (subtract carry, with borrow) */
    uint64_t t0 = x_lo - x_hhi;
    if (t0 > x_lo) t0 -= EPS;      /* borrow correction                      */

    /* t1 = x_hlo * (2^32 - 1) = (x_hlo << 32) - x_hlo */
    uint64_t t1 = (x_hlo << 32) - x_hlo;

    uint64_t r = t0 + t1;
    if (r < t0) r += EPS;          /* carry from addition: 2^64 ≡ EPS (mod P0) */
    /* r ≡ x (mod P0) and r < 2^64 = P0 + EPS, so r ∈ [0, P0+EPS):
     * one conditional subtract fully reduces it to [0, P0).            */
    if (r >= P0) r -= P0;
    return r;
}

/* ─── Reduction: p1 = 2^64 - 2^24 + 1 ───────────────────────────────────── */
/* a=64: free split. 2^64 ≡ C1 = 2^24-1 (mod P1).                           */
/* Round 1: m  = hi*C1 + lo   (< 2^88). m_hi = m>>64 < 2^24.                */
/* Round 2: R  = m_hi*C1 + m_lo. Bound: m_hi,m_lo ≤ 2^24-1, 2^64-1 →       */
/*   R ≤ (2^24-1)^2 + (2^64-1) = 2^64 + 2^48 - 2^25 < 2*P1.                 */
/*   So R is computed in __uint128_t and ONE conditional subtract of P1     */
/*   fully reduces it. The earlier uint64 round-2 form mishandled the case  */
/*   m_hi*2^24+m_lo ≥ 2^64 with true residue ≥ 2^64 (off by C1); doing the  */
/*   fold in 128-bit removes that edge case entirely.                       */
__host__ __device__ __forceinline__
uint64_t reduce_p1(__uint128_t x)
{
    const uint64_t C1 = (UINT64_C(1) << 24) - 1; /* 2^24 - 1 */
    uint64_t lo = (uint64_t)x;
    uint64_t hi = (uint64_t)(x >> 64);

    __uint128_t m = (__uint128_t)hi * C1 + lo;
    uint64_t m_lo = (uint64_t)m;
    uint64_t m_hi = (uint64_t)(m >> 64);  /* < 2^24 */

    __uint128_t R = (__uint128_t)m_hi * C1 + m_lo;  /* < 2*P1 */
    if (R >= P1) R -= P1;
    return (uint64_t)R;
}

/* ─── Reduction: p2 = 2^64 - 2^34 + 1 ───────────────────────────────────── */
/* a=64: free split. 2^64 ≡ C2 = 2^34-1 (mod P2).                           */
/* Round 1: m = hi*C2+lo (__uint128_t). m_hi < 2^35.                         */
/* Round 2: n = m_hi*C2+m_lo (__uint128_t). n_hi < 64.                       */
/* Round 3: add = n_hi*C2 < 2^40. r+add may overflow uint64_t when r>=P2.   */
/*   Overflow means result wrapped by 2^64 ≡ C2 (mod P2): add C2 back.      */
__host__ __device__ __forceinline__
uint64_t reduce_p2(__uint128_t x)
{
    const uint64_t C2 = (UINT64_C(1) << 34) - 1; /* 2^34 - 1 */
    uint64_t lo = (uint64_t)x;
    uint64_t hi = (uint64_t)(x >> 64);

    __uint128_t m  = (__uint128_t)hi * C2 + lo;
    uint64_t m_lo  = (uint64_t)m;
    uint64_t m_hi  = (uint64_t)(m >> 64);   /* < 2^35 */

    __uint128_t n  = (__uint128_t)m_hi * C2 + m_lo;
    uint64_t r     = (uint64_t)n;
    uint64_t r_hi  = (uint64_t)(n >> 64);   /* < 64  */

    uint64_t add   = r_hi * C2;             /* < 2^40 */
    uint64_t old_r = r;
    r += add;
    if (r < old_r) r += C2;                 /* carry: 2^64 ≡ C2 (mod P2) */
    if (r >= P2) r -= P2;
    if (r >= P2) r -= P2;
    return r;
}

/* ─── Reduction: p3 = 2^64 - 2^40 + 1 ───────────────────────────────────── */
/* a=64: free split. 2^64 ≡ C3 = 2^40-1 (mod P3).                           */
/* Round 1: m = hi*C3+lo (__uint128_t). m_hi < 2^41.                         */
/* Round 2: n = m_hi*C3+m_lo (__uint128_t). n_hi < 2^18.                     */
/* Round 3: n_hi*(2^40-1) < 2^58; r+add may wrap — carry adds C3 back.       */
__host__ __device__ __forceinline__
uint64_t reduce_p3(__uint128_t x)
{
    const uint64_t C3 = (UINT64_C(1) << 40) - 1; /* 2^40 - 1 */
    uint64_t lo = (uint64_t)x;
    uint64_t hi = (uint64_t)(x >> 64);

    __uint128_t m  = (__uint128_t)hi * C3 + lo;
    uint64_t m_lo  = (uint64_t)m;
    uint64_t m_hi  = (uint64_t)(m >> 64);   /* < 2^41 */

    __uint128_t n  = (__uint128_t)m_hi * C3 + m_lo;
    uint64_t r     = (uint64_t)n;
    uint64_t r_hi  = (uint64_t)(n >> 64);   /* < 2^18 */

    uint64_t add   = r_hi * C3;              /* < 2^58 */
    uint64_t old_r = r;
    r += add;
    if (r < old_r) r += C3;                 /* carry: 2^64 ≡ C3 (mod P3) */
    if (r >= P3) r -= P3;
    if (r >= P3) r -= P3;
    return r;
}

/* ─── Compile-time dispatch via C++ templates ────────────────────────────── */
/* PIDX selects the prime at compile time. The NTT kernel is instantiated    */
/* four times with PIDX=0..3. No runtime branching in the inner loop.        */
/* Guard: pure C (bigint.c/newton.c etc.) only needs PRIMES[] above;         */
/* the templates are visible only to C++ / hipcc translation units.          */
#ifdef __cplusplus
template<int PIDX>
__host__ __device__ __forceinline__
uint64_t reduce_mod(__uint128_t x);

template<> __host__ __device__ __forceinline__
uint64_t reduce_mod<0>(__uint128_t x) { return reduce_p0(x); }
template<> __host__ __device__ __forceinline__
uint64_t reduce_mod<1>(__uint128_t x) { return reduce_p1(x); }
template<> __host__ __device__ __forceinline__
uint64_t reduce_mod<2>(__uint128_t x) { return reduce_p2(x); }
template<> __host__ __device__ __forceinline__
uint64_t reduce_mod<3>(__uint128_t x) { return reduce_p3(x); }

/* ─── Additive primitives ────────────────────────────────────────────────── */
/* Both inputs must be in [0, p). Output is in [0, p).                       */
/* addmod: two checks needed for primes near 2^64 where a+b may overflow.    */
/*   Overflow (a+b >= 2^64): uint64_t r = a+b-2^64 (small positive).        */
/*   Detected by r < a. Fix: r -= p underflows to r + (2^64-p) = a+b-p. ✓  */
/*   Non-overflow, r >= p: r -= p as usual.                                  */
/* submod: a + p - b may also overflow, but double-wrapping gives the        */
/*   correct result via uint64_t arithmetic: no extra check required.        */

template<int PIDX>
__host__ __device__ __forceinline__
uint64_t addmod(uint64_t a, uint64_t b)
{
    uint64_t r = a + b;
    if (r < a) r -= PRIMES[PIDX];             /* overflow: r += 2^64-p     */
    if (r >= PRIMES[PIDX]) r -= PRIMES[PIDX]; /* non-overflow reduction    */
    return r;
}

template<int PIDX>
__host__ __device__ __forceinline__
uint64_t submod(uint64_t a, uint64_t b)
{
    return a >= b ? a - b : a + PRIMES[PIDX] - b;
}

/* Full modular multiply via reduction. Used for Hadamard products and        */
/* precomputation only — NOT the NTT inner loop (Shoup is used there).       */
template<int PIDX>
__host__ __device__ __forceinline__
uint64_t mulmod(uint64_t a, uint64_t b)
{
    return reduce_mod<PIDX>((__uint128_t)a * b);
}

/* ─── Montgomery multiplication (R = 2^64) ──────────────────────────────── */
/* P_INV[i] = -PRIMES[i]^{-1} mod 2^64. Verified: PRIMES[i] * P_INV[i] ≡ -1 (mod 2^64).
 * Trade-off vs Shoup: one extra 64x64 multiply per butterfly; but twiddle
 * storage halves (w only, not {w, w_inv}).  On bandwidth-bound NTTs
 * (N >= 2^20, HBM-limited), halving twiddle reads can offset the extra mul. */
static const uint64_t P_INV[4] = {
    UINT64_C(0xfffffffeffffffff),   /* -P0^{-1} mod 2^64 */
    UINT64_C(0xfffefffffeffffff),   /* -P1^{-1} mod 2^64 */
    UINT64_C(0xfffffffbffffffff),   /* -P2^{-1} mod 2^64 */
    UINT64_C(0xfffffeffffffffff),   /* -P3^{-1} mod 2^64 */
};

/* montgomery_mul<PIDX>: REDC(a*b) = a*b*R^{-1} mod p, R = 2^64.
 * Inputs a, b in [0, p). Output in [0, p).
 *
 * Carry-correct form: the textbook `(t + m*p) >> 64` is WRONG here because
 * for p near 2^64, t = a*b (≤(p-1)^2, ~128 bits) plus m*p (~128 bits) sums to
 * ~2^129 — it overflows __uint128_t, dropping the top carry, and the exact
 * (t+m*p)>>64 is a 65-bit value that does not fit uint64_t. We add with an
 * explicit 64-bit carry chain. By REDC construction (tlo+mplo) ≡ 0 (mod 2^64),
 * so only its carry-out propagates into the high words.
 *
 * Bound: thi = (a*b)>>64 < p and mphi = (m*p)>>64 < p (since m,a,b < 2^64,
 * p < 2^64), so u = thi+mphi+carry < 2p < 2^65 — one conditional subtract of
 * p fully reduces it. (Exhaustively verified in lib/test_modops.hip.)
 *
 * gfx942: the low-word add/carry lowers to v_add_co_u32/v_addc_co_u32 and the
 * 65-bit reduce to a __uint128_t compare+sub — same multiply count as before,
 * no extra 64x64 multiplies vs the (buggy) single-__uint128_t form. */
template<int PIDX>
__host__ __device__ __forceinline__
uint64_t montgomery_mul(uint64_t a, uint64_t b)
{
    const uint64_t p = PRIMES[PIDX];
    __uint128_t t   = (__uint128_t)a * b;
    uint64_t    tlo = (uint64_t)t;
    uint64_t    thi = (uint64_t)(t >> 64);
    uint64_t    m   = tlo * P_INV[PIDX];          /* mod 2^64 (implicit)      */
    __uint128_t mp  = (__uint128_t)m * p;
    uint64_t    mplo = (uint64_t)mp;
    uint64_t    mphi = (uint64_t)(mp >> 64);
    /* low 64 bits are 0 by construction; capture only the carry-out. */
    uint64_t carry = (uint64_t)(((__uint128_t)tlo + mplo) >> 64);   /* 0 or 1 */
    __uint128_t u  = (__uint128_t)thi + mphi + carry;               /* < 2p   */
    if (u >= p) u -= p;
    return (uint64_t)u;
}
#endif /* __cplusplus */

/* Scalar modular exponentiation. CPU only. Used in twiddle precomputation.   */
static inline uint64_t modpow(uint64_t base, uint64_t exp, uint64_t p)
{
    uint64_t r = 1;
    base %= p;
    while (exp > 0) {
        if (exp & 1) r = (uint64_t)((__uint128_t)r * base % p);
        base = (uint64_t)((__uint128_t)base * base % p);
        exp >>= 1;
    }
    return r;
}
