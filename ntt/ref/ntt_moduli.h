/*
 * ntt_moduli.h — NTT-friendly prime moduli: table, reduction functions,
 *                and arithmetic helpers for all 14 supported primes.
 *
 * Include this header in any binary that needs multi-modulus support.
 * The reduction functions replace the generic `% q` division in hot paths.
 *
 * REDUCTION STRATEGY BY PRIME CLASS
 * ──────────────────────────────────
 * MOD_FERMAT      (q = 2^m + 1):           exact: (t&mask) - (t>>m); 1 sub
 * MOD_DILITHIUM   (q = 2^23 - 2^13 + 1):  exact Solinas: B + (A<<13) - A
 * MOD_SOLINAS_60  (q = 2^60 - 2^18 + 1):  exact 2-pass Solinas; no 128-bit div
 * MOD_GOLDILOCKS  (q = 2^64 - 2^32 + 1):  exact 2-step; no 128-bit div
 * MOD_GENERIC     (any Proth or other):    __uint128_t product % q (hw divide)
 *
 * NOTE on K²RED: the Proth formula B - k·A (where A = t>>m, B = t&mask) gives
 * t - A·(2^m+k), which is NOT ≡ t (mod q) in software. K²RED is exact only on
 * FPGA where bounded-width arithmetic keeps A small enough. All Proth primes here
 * use reduce_generic (128-bit divide) until a correct Barrett table is added.
 *
 * OVERFLOW-SAFE BUTTERFLY HELPERS
 * ─────────────────────────────────
 * addmod(u, t, q) and submod(u, t, q) work correctly for all primes
 * including Goldilocks (q near 2^64) where u+t may overflow uint64_t.
 *
 * Build: cc -O2 -Wall -Wextra   (C99; no HIP required for CPU binaries)
 */

#ifndef NTT_MODULI_H
#define NTT_MODULI_H

#include <stdint.h>
#include <stddef.h>  /* NULL */

/* ── Goldilocks prime constant ───────────────────────────────────────────── */
#define GOLDILOCKS_Q  UINT64_C(18446744069414584321)  /* 2^64 - 2^32 + 1 */
#define GOLDILOCKS_MOD UINT64_C(0xFFFFFFFF)           /* 2^32 - 1 = 2^64 mod q */

/* ── Reduction function type ─────────────────────────────────────────────── */
/*
 * reduce_fn_t: pointer to a function that reduces a 128-bit product mod q.
 * product: result of a * b where a, b are NTT elements < q.
 * q:       the modulus (ignored by specialised functions; used by generic).
 * Returns: product mod q in [0, q).
 */
typedef uint64_t (*reduce_fn_t)(__uint128_t product, uint64_t q);

/* ── Modulus classes ─────────────────────────────────────────────────────── */
typedef enum {
    MOD_GENERIC    = 0,  /* hardware %: correct for any prime; use as fallback */
    MOD_FERMAT,          /* q = 2^m + 1: exact (t&mask)-(t>>m) reduction       */
    MOD_DILITHIUM,       /* q = 2^23 - 2^13 + 1: exact Solinas reduction       */
    MOD_SOLINAS_60,      /* q = 2^60 - 2^18 + 1: exact 2-pass Solinas          */
    MOD_GOLDILOCKS,      /* q = 2^64 - 2^32 + 1: exact 2-step reduction        */
} modulus_class_t;

/* ── Modulus table entry ─────────────────────────────────────────────────── */
typedef struct {
    uint64_t         q;            /* the prime modulus                        */
    uint64_t         g;            /* primitive root mod q                     */
    uint32_t         max_log2_n;   /* max NTT size: n = 2^max_log2_n          */
    modulus_class_t  cls;          /* reduction class                          */
    const char      *name;         /* short label                              */
    const char      *form;         /* algebraic form string                    */
    reduce_fn_t      reduce;       /* fast reduction function                  */
} ntt_modulus_info_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * OVERFLOW-SAFE BUTTERFLY ARITHMETIC
 * Works for all primes including Goldilocks (q near 2^64).
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * addmod: compute (u + t) mod q.
 * For small primes (q < 2^32): u+t < 2^33, no overflow possible; the
 *   `s < u` branch is dead and the compiler removes it.
 * For Goldilocks (q ≈ 2^64): u+t may overflow uint64_t; the overflow
 *   correction adds 2^64 mod q = 2^32-1.
 */
static inline uint64_t addmod(uint64_t u, uint64_t t, uint64_t q)
{
    uint64_t s = u + t;
    if (s < u) s += (UINT64_MAX - q + 1);  /* overflow: s_true = s + 2^64;
                                               (2^64 mod q) = UINT64_MAX-q+1 */
    if (s >= q) s -= q;
    return s;
}

/*
 * submod: compute (u - t) mod q.
 * In uint64_t: when t > u, `u - t + q` wraps correctly because the
 * double-wrap (add 2^64, then add q, then the combined overflow reduces)
 * yields the mathematical value u - t + q which is in [1, q-1].
 */
static inline uint64_t submod(uint64_t u, uint64_t t, uint64_t q)
{
    return (u >= t) ? u - t : u - t + q;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SPECIALIZED REDUCTION FUNCTIONS
 *
 * Each function reduces a 128-bit product mod the target prime.
 * The `q` parameter is accepted for interface uniformity but ignored by
 * specialised functions (q is encoded in the shift and mask constants).
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Generic: hardware divide. Correct for any prime; use as fallback. */
static inline uint64_t reduce_generic(__uint128_t t, uint64_t q)
{
    return (uint64_t)(t % q);
}

/*
 * Fermat primes: q = 2^m + 1.
 * t mod q = (t & mask) - (t >> m)   where mask = 2^m - 1.
 * Negative result: add q once.
 * q = 257   → m=8,  mask=0xFF
 * q = 65537 → m=16, mask=0xFFFF
 */
static inline uint64_t reduce_fermat8(__uint128_t t, uint64_t q_unused)
{
    (void)q_unused;
    /* q = 257 = 2^8 + 1; A = t>>8 can be up to q^2/256 ≈ 258 */
    uint64_t A = (uint64_t)(t >> 8);
    uint64_t B = (uint64_t)t & 0xFFULL;
    int64_t  r = (int64_t)B - (int64_t)(A % 257);  /* A % 257 ∈ {0, 1} */
    if (r < 0) r += 257;
    return (uint64_t)r;
}

static inline uint64_t reduce_fermat16(__uint128_t t, uint64_t q_unused)
{
    (void)q_unused;
    /* q = 65537 = 2^16 + 1; t < 65537^2 ≈ 2^32; split at bit 16 */
    uint64_t A = (uint64_t)(t >> 16);   /* A ≤ 65537; at most one subtract */
    uint64_t B = (uint64_t)t & 0xFFFFULL;
    if (A >= 65537) A -= 65537;
    int64_t r = (int64_t)B - (int64_t)A;
    if (r < 0) r += 65537;
    return (uint64_t)r;
}

/*
 * ML-DSA / Dilithium: q = 8380417 = 2^23 - 2^13 + 1 (Solinas form).
 * 2^23 ≡ 2^13 - 1 (mod q).  For t = A·2^23 + B:
 *   t mod q = B + A·(2^13 - 1) = B + (A<<13) - A.
 * A < q^2 / 2^23 ≈ q; the intermediate int64_t result may be slightly
 * negative; the final mod+add handles wrap-around.
 */
static inline uint64_t reduce_dilithium(__uint128_t t, uint64_t q_unused)
{
    (void)q_unused;
    const uint64_t Q = 8380417, MASK = (1ULL<<23)-1;
    uint64_t A = (uint64_t)(t >> 23);
    uint64_t B = (uint64_t)t & MASK;
    int64_t  r = (int64_t)B + (int64_t)((A<<13) - A);
    r = ((r % (int64_t)Q) + (int64_t)Q) % (int64_t)Q;
    return (uint64_t)r;
}

/*
 * 60-bit Solinas: q = 2^60 - 2^18 + 1 = 1152921504606584833.
 * 2^60 ≡ 2^18 - 1 (mod q).  For t = A·2^60 + B  (t < q^2 < 2^120):
 *   t mod q ≡ A·(2^18 - 1) + B.
 *
 * Pass 1: r128 = A·(2^18-1) + B   (A < q < 2^60; r128 < 2^79)
 * Pass 2: A2 = r128>>60, B2 = r128&mask; r = A2·(2^18-1) + B2
 *         A2 < 2^19, so A2·(2^18-1) < 2^37; r < 2^61 < 2q.
 * One conditional subtract yields the result in [0, q).
 */
static inline uint64_t reduce_solinas_60(__uint128_t t, uint64_t q_unused)
{
    (void)q_unused;
    const uint64_t Q    = UINT64_C(1152921504606584833);  /* 2^60 - 2^18 + 1 */
    const uint64_t MASK = (UINT64_C(1) << 60) - 1;
    const uint64_t MOD  = (UINT64_C(1) << 18) - 1;       /* 2^60 mod q = 2^18-1 */

    /* Pass 1: 120→79 bits */
    uint64_t A = (uint64_t)(t >> 60);
    uint64_t B = (uint64_t)t & MASK;
    __uint128_t r128 = (__uint128_t)A * MOD + B;

    /* Pass 2: 79→61 bits (fits uint64_t) */
    uint64_t A2 = (uint64_t)(r128 >> 60);
    uint64_t B2 = (uint64_t)r128 & MASK;
    uint64_t r  = A2 * MOD + B2;   /* A2 < 2^19, A2*MOD < 2^37; r < 2^61 < 2q */

    if (r >= Q) r -= Q;
    return r;
}

/*
 * Goldilocks: q = 2^64 - 2^32 + 1.
 * 2^64 ≡ 2^32 - 1 (mod q).  For 128-bit t = a·2^64 + b:
 *   t mod q = a·(2^32-1) + b.
 * Two-step reduction using the same identity avoids any 128-by-64 division.
 * Algorithm:
 *   Step 1: r = a·(2^32-1) + b  (compute in __uint128_t; r < 2^96)
 *   Step 2: c = r>>64 (< 2^32); d = (uint64_t)r
 *           s = c·(2^32-1) + d  (compute in uint64_t; may overflow once)
 *   Step 3: overflow correction and final conditional subtract.
 */
static inline uint64_t reduce_goldilocks(__uint128_t t, uint64_t q_unused)
{
    (void)q_unused;
    const uint64_t MOD = GOLDILOCKS_MOD;  /* 2^32 - 1 */
    const uint64_t Q   = GOLDILOCKS_Q;

    uint64_t a = (uint64_t)(t >> 64);
    uint64_t b = (uint64_t)t;

    /* Step 1: r = a*(2^32-1) + b; r < a*2^32 + b < q*2^32 + 2^64 < 2^96 */
    __uint128_t r128 = (__uint128_t)a * MOD + b;
    uint64_t c = (uint64_t)(r128 >> 64);   /* c < 2^32 (since r128 < 2^96) */
    uint64_t d = (uint64_t)r128;

    /* Step 2: s = c*(2^32-1) + d; c*MOD < 2^64; s may overflow once */
    uint64_t cm = c * MOD;                 /* c < 2^32, MOD = 2^32-1: cm < 2^64 */
    uint64_t s  = cm + d;
    if (s < cm) s += MOD;                  /* overflow: s_true = s + 2^64
                                              ≡ s + (2^32-1) = s + MOD (mod q) */
    if (s >= Q) s -= Q;
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MODULI TABLE — 14 NTT-friendly primes
 *
 * Entries are ordered by q.  Access via ntt_modulus_find() or iterate
 * NTT_MODULI[i].  Reduction class and function summarised below:
 *
 *  Class         Reduction path
 *  ─────────────────────────────────────────────────────────────────────────
 *  MOD_FERMAT    reduce_fermat8 / reduce_fermat16 — exact, 1 subtract
 *  MOD_DILITHIUM reduce_dilithium — exact Solinas, 2^23≡2^13−1 identity
 *  MOD_SOLINAS_60 reduce_solinas_60 — exact 2-pass, 2^60≡2^18−1 identity
 *  MOD_GOLDILOCKS reduce_goldilocks — exact 2-step, 2^64≡2^32−1 identity
 *  MOD_GENERIC   reduce_generic — __uint128_t % q; always correct
 *
 * CRT-NTT triple: 167772161 + 469762049 + 998244353 (all g=3).
 *   Product covers integers up to 167772161·469762049·998244353 ≈ 7.9×10^25.
 *   Use these three together for polynomial multiplication over ℤ.
 *
 * The two large Solinas primes (q=1152921504606584833, q=2287828610704211969)
 * are CPU-verified 2026-05-18 (ref/test_curated_primes.c, `make test-curated`):
 * reduction exhaustive-random vs __int128 %q (30M random + structured cases up
 * to q^2), addmod/submod/mulmod vs __int128, and NTT round-trip + cyclic +
 * negacyclic polymul vs O(n^2) schoolbook. q>=2^32 so the ref GPU path
 * (q<2^32) cannot exercise these; verification is inherently CPU-side.
 * Generator note: Solinas-60 uses g=10. The originally tabled g=3 was WRONG —
 * 3 is a quadratic residue mod 2^60-2^18+1, so 3^((q-1)/2)==1 and g=3 yields
 * no 2-power root of unity (test_curated_primes.c flagged this; g=10 is the
 * smallest primitive root and is verified). Solinas-61 g=3 is verified
 * (3 is a primitive root; supports n up to 2^54).
 * ML-KEM-v0 (q=7681) uses g=17. The originally tabled g=3 was WRONG —
 * 3 is a cubic residue mod 7681 (q-1 = 7680 = 2^9·3·5; 3^((q-1)/3)==1), so
 * g=3 is not a primitive root and yields an order-512 ω with ω^512≠1
 * (independent CPU verifier flagged this 2026-05-18). g=17 is the smallest
 * primitive root: ω=17^((q-1)/512)=7146, ω^512=1, ω^256=7680=q-1, verified.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define NTT_NUM_MODULI 14

static const ntt_modulus_info_t NTT_MODULI[NTT_NUM_MODULI] = {
    /*  q                    g   ml2n  class            name            form                 reduce            */
    { UINT64_C(         257),  3,   8, MOD_FERMAT,    "Fermat-8",    "2^8+1",               reduce_fermat8    },
    { UINT64_C(        3329),  3,   8, MOD_GENERIC,   "ML-KEM",      "13*2^8+1",            reduce_generic    },
    { UINT64_C(        7681), 17,   9, MOD_GENERIC,   "ML-KEM-v0",   "15*2^9+1",            reduce_generic    },
    { UINT64_C(       12289), 11,  12, MOD_GENERIC,   "FALCON",      "3*2^12+1",            reduce_generic    },
    { UINT64_C(       40961),  3,  13, MOD_GENERIC,   "Proth-5-13",  "5*2^13+1",            reduce_generic    },
    { UINT64_C(       65537),  3,  16, MOD_FERMAT,    "Fermat-16",   "2^16+1",              reduce_fermat16   },
    { UINT64_C(     8380417), 10,  13, MOD_DILITHIUM, "ML-DSA",      "2^23-2^13+1",         reduce_dilithium  },
    { UINT64_C(   167772161),  3,  25, MOD_GENERIC,   "CRT-lo",      "5*2^25+1",            reduce_generic    },
    { UINT64_C(   469762049),  3,  26, MOD_GENERIC,   "CRT-mid",     "7*2^26+1",            reduce_generic    },
    { UINT64_C(   998244353),  3,  23, MOD_GENERIC,   "CRT-hi",      "119*2^23+1",          reduce_generic    },
    { UINT64_C(  2013265921), 31,  27, MOD_GENERIC,   "FHE-RNS",     "15*2^27+1",           reduce_generic    },
    { UINT64_C(1152921504606584833),10, 18, MOD_SOLINAS_60, "Solinas-60", "2^60-2^18+1",    reduce_solinas_60 },
    { UINT64_C(2287828610704211969), 3, 54, MOD_GENERIC,   "Solinas-61", "2^61-2^54+1",     reduce_generic    },
    { GOLDILOCKS_Q,            7,  32, MOD_GOLDILOCKS,"Goldilocks",  "2^64-2^32+1",         reduce_goldilocks },
};

/* ═══════════════════════════════════════════════════════════════════════════
 * HELPER FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * ntt_modulus_find: look up a modulus by its q value.
 * Returns pointer to entry in NTT_MODULI[], or NULL if not found.
 */
static inline const ntt_modulus_info_t *ntt_modulus_find(uint64_t q)
{
    for (int i = 0; i < NTT_NUM_MODULI; i++)
        if (NTT_MODULI[i].q == q) return &NTT_MODULI[i];
    return NULL;
}

/*
 * ntt_modulus_omega: compute the primitive n-th root of unity for a modulus.
 * Uses omega = g^((q-1)/n) mod q.
 * Requires n to be a power of 2 with n <= 2^max_log2_n.
 * Returns 0 if n is invalid for this modulus.
 */
static inline uint64_t ntt_modulus_omega(const ntt_modulus_info_t *m, uint64_t n)
{
    if (n == 0 || (n & (n-1)) != 0) return 0;         /* not power of 2 */
    if (n > (UINT64_C(1) << m->max_log2_n)) return 0; /* exceeds max    */
    /* omega = g^((q-1)/n) mod q */
    /* (q-1)/n: for Goldilocks q-1 = 2^32*(2^32-1), division by n (power of 2)
     * is exact as long as n <= 2^32. */
    uint64_t exp = (m->q - 1) / n;
    /* mod_pow via __uint128_t */
    uint64_t base = m->g % m->q, result = 1;
    for (uint64_t e = exp; e > 0; e >>= 1) {
        if (e & 1) result = (uint64_t)((__uint128_t)result * base % m->q);
        base = (uint64_t)((__uint128_t)base * base % m->q);
    }
    return result;
}

#endif /* NTT_MODULI_H */
