/*
 * test_curated_primes.c — rigorous CPU verification of the two large curated
 *                         "unverified" NTT primes in ntt_moduli.h.
 *
 * Purpose:
 *   The 14-prime table in ntt_moduli.h flags the primitive roots / bespoke
 *   reduction of the two large Solinas primes as unverified, and they are
 *   q >= 2^32 so the ref GPU path (which hard-requires q < 2^32) can never
 *   exercise them. Verification is therefore inherently CPU-side. This test
 *   proves, for EACH of the two primes:
 *
 *     (a) REDUCTION CORRECTNESS — the prime's mapped reduce() function equals
 *         the exact reference (unsigned __int128) t % q over a large
 *         randomized + structured set of 128-bit inputs up to ~q^2, including
 *         all relevant boundary values.
 *     (b) MODULAR-ARITH SANITY — addmod/submod/mulmod (built on reduce) agree
 *         with an independent __int128 reference over randomized inputs; these
 *         q sit near 2^60 / 2^61 so u+t can approach but (for these q) not
 *         overflow uint64_t — checked explicitly.
 *     (c) NTT VALIDITY — derive omega = g^((q-1)/n) for the table's generator
 *         g, assert omega^n == 1 and omega^(n/2) != 1, run forward+inverse NTT
 *         via the ref CPU reference path (ntt_forward / ntt_inverse) and
 *         assert INTT(NTT(x)) == x; and a cyclic + negacyclic polynomial
 *         multiply checked against an independent O(n^2) __int128 schoolbook.
 *
 * The two primes under test (read from NTT_MODULI in ntt_moduli.h):
 *   Solinas-60: q = 1152921504606584833 = 2^60 - 2^18 + 1, g=3,
 *               class MOD_SOLINAS_60, reduce = reduce_solinas_60
 *   Solinas-61: q = 2287828610704211969 = 2^61 - 2^54 + 1, g=3,
 *               class MOD_GENERIC,     reduce = reduce_generic
 *
 * Deterministic xorshift128+ PRNG: any failure reproduces exactly.
 *
 * Build: cc -O2 -Wall -Wextra -o test_curated_primes test_curated_primes.c
 * Pure host C99; no HIP. Exit code 0 iff every check passes for both primes.
 */

#include "ntt.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ── ANSI formatting (matches ntt_cpu.c table style) ─────────────────────── */
#define ANSI_WHT "\033[1;37m"
#define ANSI_CYN "\033[1;36m"
#define ANSI_GRN "\033[1;32m"
#define ANSI_RED "\033[1;31m"
#define ANSI_RST "\033[0m"

/* ═══════════════════════════════════════════════════════════════════════════
 * DETERMINISTIC PRNG  (xorshift128+, Vigna 2014)
 * Fixed seed → identical stream every run → reproducible failures.
 * ═══════════════════════════════════════════════════════════════════════════ */
static uint64_t RNG_S0, RNG_S1;

static void rng_seed(uint64_t a, uint64_t b)
{
    RNG_S0 = a ? a : 0x9E3779B97F4A7C15ULL;
    RNG_S1 = b ? b : 0xBF58476D1CE4E5B9ULL;
}

static uint64_t rng_next(void)
{
    uint64_t x = RNG_S0, y = RNG_S1;
    RNG_S0 = y;
    x ^= x << 23;
    RNG_S1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return RNG_S1 + y;
}

/* Uniform value in [0, m) for m > 0. */
static uint64_t rng_below(uint64_t m)
{
    return rng_next() % m;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * (a) REDUCTION CORRECTNESS
 *
 * Compare m->reduce(t, q) against the exact reference (__uint128_t)t % q.
 * reduce_solinas_60's documented input invariant is t < q^2 (≈ 2^120),
 * which is exactly the bound seen in an NTT butterfly (tw*a, both < q),
 * so all random products are drawn in [0, q^2). Structured cases probe the
 * exact boundaries the 2-pass Solinas algebra hinges on.
 * ═══════════════════════════════════════════════════════════════════════════ */
static int check_reduction(const ntt_modulus_info_t *m, uint64_t n_random,
                           uint64_t *out_cases)
{
    const uint64_t q = m->q;
    const __uint128_t q2 = (__uint128_t)q * q;   /* upper bound on NTT products */
    uint64_t cases = 0;
    int fails = 0;

    /* ---- Structured / boundary inputs ---- */
    __uint128_t edge[32];
    size_t ne = 0;
    edge[ne++] = 0;
    edge[ne++] = 1;
    edge[ne++] = q - 1;
    edge[ne++] = q;
    edge[ne++] = q + 1;
    edge[ne++] = 2 * (__uint128_t)q - 1;
    edge[ne++] = 2 * (__uint128_t)q;
    edge[ne++] = (__uint128_t)(q - 1) * (q - 1);   /* max real NTT product */
    edge[ne++] = q2 - 1;
    edge[ne++] = q2 - q;
    edge[ne++] = ((__uint128_t)1 << 60);            /* exact split point (S60) */
    edge[ne++] = ((__uint128_t)1 << 60) - 1;
    edge[ne++] = ((__uint128_t)1 << 61);            /* exact split point (S61) */
    edge[ne++] = ((__uint128_t)1 << 79);            /* S60 pass-1 ceiling probe */
    edge[ne++] = ((__uint128_t)1 << 64) - 1;        /* 64-bit boundary */
    edge[ne++] = ((__uint128_t)1 << 64);
    edge[ne++] = ((__uint128_t)1 << 64) + 1;
    edge[ne++] = ((__uint128_t)q << 18);            /* A=q<<? structured */
    edge[ne++] = ((__uint128_t)q << 60) % q2;       /* keep in-range */
    edge[ne++] = q2 / 2;
    for (size_t i = 0; i < ne; i++) {
        __uint128_t t = edge[i] % q2;               /* respect t < q^2 invariant */
        uint64_t got = m->reduce(t, q);
        uint64_t exp = (uint64_t)(t % q);
        cases++;
        if (got != exp) {
            printf("  " ANSI_RED "FAIL" ANSI_RST " [%s reduce edge#%zu] "
                   "t=0x%016llx%016llx exp=%llu got=%llu\n",
                   m->name, i,
                   (unsigned long long)(uint64_t)(t >> 64),
                   (unsigned long long)(uint64_t)t,
                   (unsigned long long)exp, (unsigned long long)got);
            fails++;
        }
    }

    /* ---- Large randomized sweep: full-range products in [0, q^2) ---- */
    for (uint64_t i = 0; i < n_random; i++) {
        /* Build a 128-bit value < q^2. Mix of (a<q)*(b<q) products and
         * arbitrary 128-bit-then-masked values to cover the whole range. */
        __uint128_t t;
        if (i & 1) {
            uint64_t a = rng_below(q);
            uint64_t b = rng_below(q);
            t = (__uint128_t)a * b;                 /* genuine NTT product */
        } else {
            __uint128_t hi = rng_next();
            __uint128_t lo = rng_next();
            t = ((hi << 64) | lo) % q2;             /* uniform in [0, q^2) */
        }
        uint64_t got = m->reduce(t, q);
        uint64_t exp = (uint64_t)(t % q);
        cases++;
        if (got != exp) {
            printf("  " ANSI_RED "FAIL" ANSI_RST " [%s reduce rand#%llu] "
                   "t=0x%016llx%016llx exp=%llu got=%llu\n",
                   m->name, (unsigned long long)i,
                   (unsigned long long)(uint64_t)(t >> 64),
                   (unsigned long long)(uint64_t)t,
                   (unsigned long long)exp, (unsigned long long)got);
            fails++;
            if (fails > 8) break;                   /* enough evidence */
        }
    }

    *out_cases = cases;
    return fails ? -1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * (b) MODULAR-ARITH SANITY
 *
 * addmod / submod from ntt_moduli.h and a reduce-based mulmod, each compared
 * to an independent __int128 reference. Inputs are full-range residues
 * [0, q). For these q (< 2^61) u+t < 2^62, so addmod's overflow branch is
 * provably dead — we still exercise the path that would matter (u+t >= q).
 * ═══════════════════════════════════════════════════════════════════════════ */
static int check_arith(const ntt_modulus_info_t *m, uint64_t n_random,
                       uint64_t *out_cases)
{
    const uint64_t q = m->q;
    uint64_t cases = 0;
    int fails = 0;

    for (uint64_t i = 0; i < n_random; i++) {
        uint64_t u = rng_below(q);
        uint64_t t = rng_below(q);

        uint64_t add_got = addmod(u, t, q);
        uint64_t add_exp = (uint64_t)(((__uint128_t)u + t) % q);

        uint64_t sub_got = submod(u, t, q);
        uint64_t sub_exp = (uint64_t)((( (__int128)u - (__int128)t) % (__int128)q
                                        + (__int128)q) % (__int128)q);

        uint64_t mul_got = m->reduce((__uint128_t)u * t, q);
        uint64_t mul_exp = (uint64_t)(((__uint128_t)u * t) % q);

        cases += 3;
        if (add_got != add_exp || sub_got != sub_exp || mul_got != mul_exp) {
            printf("  " ANSI_RED "FAIL" ANSI_RST " [%s arith #%llu] "
                   "u=%llu t=%llu  add %llu/%llu sub %llu/%llu mul %llu/%llu\n",
                   m->name, (unsigned long long)i,
                   (unsigned long long)u, (unsigned long long)t,
                   (unsigned long long)add_got, (unsigned long long)add_exp,
                   (unsigned long long)sub_got, (unsigned long long)sub_exp,
                   (unsigned long long)mul_got, (unsigned long long)mul_exp);
            fails++;
            if (fails > 8) break;
        }
    }
    *out_cases = cases;
    return fails ? -1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * (c) NTT VALIDITY
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Local modular exponentiation (independent of the ref internal mod_pow). */
static uint64_t tp_mod_pow(uint64_t base, uint64_t exp, uint64_t q)
{
    uint64_t r = 1;
    base %= q;
    while (exp) {
        if (exp & 1) r = (uint64_t)((__uint128_t)r * base % q);
        base = (uint64_t)((__uint128_t)base * base % q);
        exp >>= 1;
    }
    return r;
}

/*
 * Cyclic convolution reference: c[k] = sum_{i+j == k mod n} a[i]*b[j] mod q.
 * Independent O(n^2) __int128 schoolbook — no NTT involved.
 */
static void schoolbook_cyclic(const uint64_t *a, const uint64_t *b,
                              uint64_t *c, uint64_t n, uint64_t q)
{
    for (uint64_t k = 0; k < n; k++) c[k] = 0;
    for (uint64_t i = 0; i < n; i++)
        for (uint64_t j = 0; j < n; j++) {
            uint64_t k = (i + j) % n;
            c[k] = (uint64_t)(((__uint128_t)c[k]
                              + (__uint128_t)a[i] * b[j]) % q);
        }
}

/*
 * Negacyclic convolution reference: product mod (x^n + 1).
 * c[k] = sum_{i+j==k} a[i]b[j] - sum_{i+j==k+n} a[i]b[j]  (mod q).
 */
static void schoolbook_negacyclic(const uint64_t *a, const uint64_t *b,
                                  uint64_t *c, uint64_t n, uint64_t q)
{
    for (uint64_t k = 0; k < n; k++) c[k] = 0;
    for (uint64_t i = 0; i < n; i++)
        for (uint64_t j = 0; j < n; j++) {
            uint64_t s = i + j;
            uint64_t k = s % n;
            __uint128_t prod = (__uint128_t)a[i] * b[j] % q;
            if ((s / n) & 1)               /* wrapped odd → subtract */
                c[k] = (uint64_t)(((__uint128_t)c[k] + q - prod) % q);
            else
                c[k] = (uint64_t)(((__uint128_t)c[k] + prod) % q);
        }
}

/*
 * check_ntt: omega derivation + round-trip + cyclic + negacyclic polymul.
 * Returns 0 if all sub-checks pass, -1 otherwise. n must be a power of 2
 * and <= 2^max_log2_n for this modulus.
 */
static int check_ntt(const ntt_modulus_info_t *m, uint64_t n)
{
    const uint64_t q = m->q;
    int fails = 0;

    /* ---- omega derivation and primitivity ---- */
    uint64_t omega = ntt_modulus_omega(m, n);
    if (omega == 0) {
        printf("  " ANSI_RED "FAIL" ANSI_RST
               " [%s ntt] ntt_modulus_omega returned 0 for n=%llu\n",
               m->name, (unsigned long long)n);
        return -1;
    }
    uint64_t w_n   = tp_mod_pow(omega, n, q);          /* must be 1 */
    uint64_t w_nh  = tp_mod_pow(omega, n / 2, q);      /* must be != 1 */
    if (w_n != 1 || w_nh == 1) {
        printf("  " ANSI_RED "FAIL" ANSI_RST
               " [%s ntt] non-primitive omega=%llu: w^n=%llu w^(n/2)=%llu\n",
               m->name, (unsigned long long)omega,
               (unsigned long long)w_n, (unsigned long long)w_nh);
        return -1;
    }

    ntt_params_t p;
    p.n = n; p.q = q; p.omega = omega;
    if (ntt_params_init(&p) != 0) {
        printf("  " ANSI_RED "FAIL" ANSI_RST
               " [%s ntt] ntt_params_init failed\n", m->name);
        return -1;
    }
    /* Confirm dispatch wired a reduce function behaviourally equivalent to
     * the prime's table entry. Pointer identity is NOT used: reduce_generic
     * is `static inline` in the header, so the copy ntt_params_init() takes
     * (in ntt_cpu.c's translation unit) and the copy this file sees can be
     * distinct addresses while being the same function. Check behaviour over
     * structured inputs spanning the reduction's range instead. */
    {
        const uint64_t probe[] = { 0, 1, q - 1,
                                   (uint64_t)(((__uint128_t)(q - 1) * (q - 1)) % q),
                                   q / 2 + 1 };
        for (size_t pi = 0; pi < sizeof probe / sizeof probe[0]; pi++) {
            __uint128_t t = (__uint128_t)probe[pi] * probe[pi];
            if (p.reduce(t, q) != m->reduce(t, q)
                || p.reduce(t, q) != (uint64_t)(t % q)) {
                printf("  " ANSI_RED "FAIL" ANSI_RST
                       " [%s ntt] dispatch reduce fn disagrees with table\n",
                       m->name);
                fails++;
                break;
            }
        }
    }

    uint64_t *tw  = ntt_alloc_twiddles(&p);
    uint64_t *twi = ntt_alloc_twiddles_inv(&p);
    uint64_t *x   = (uint64_t *)malloc(n * sizeof *x);
    uint64_t *y   = (uint64_t *)malloc(n * sizeof *y);
    uint64_t *xa  = (uint64_t *)malloc(n * sizeof *xa);
    uint64_t *xb  = (uint64_t *)malloc(n * sizeof *xb);
    uint64_t *ref = (uint64_t *)malloc(n * sizeof *ref);
    uint64_t *fa  = (uint64_t *)malloc(n * sizeof *fa);
    uint64_t *fb  = (uint64_t *)malloc(n * sizeof *fb);
    if (!tw || !twi || !x || !y || !xa || !xb || !ref || !fa || !fb) {
        printf("  " ANSI_RED "FAIL" ANSI_RST " [%s ntt] alloc failed\n", m->name);
        free(tw); free(twi); free(x); free(y);
        free(xa); free(xb); free(ref); free(fa); free(fb);
        return -1;
    }

    /* ---- Round-trip: INTT(NTT(x)) == x ---- */
    for (uint64_t i = 0; i < n; i++) x[i] = y[i] = rng_below(q);
    ntt_forward(x, tw, &p);
    ntt_inverse(x, twi, &p);
    for (uint64_t i = 0; i < n; i++)
        if (x[i] != y[i]) {
            printf("  " ANSI_RED "FAIL" ANSI_RST
                   " [%s ntt roundtrip] i=%llu exp=%llu got=%llu\n",
                   m->name, (unsigned long long)i,
                   (unsigned long long)y[i], (unsigned long long)x[i]);
            fails++;
            break;
        }

    /* ---- Cyclic polymul via NTT vs schoolbook ----
     * Pointwise product of forward NTTs, then inverse, equals the cyclic
     * convolution mod (x^n - 1). */
    for (uint64_t i = 0; i < n; i++) { xa[i] = rng_below(q); xb[i] = rng_below(q); }
    memcpy(fa, xa, n * sizeof *fa);
    memcpy(fb, xb, n * sizeof *fb);
    ntt_forward(fa, tw, &p);
    ntt_forward(fb, tw, &p);
    for (uint64_t i = 0; i < n; i++)
        fa[i] = p.reduce((__uint128_t)fa[i] * fb[i], q);
    ntt_inverse(fa, twi, &p);
    schoolbook_cyclic(xa, xb, ref, n, q);
    for (uint64_t i = 0; i < n; i++)
        if (fa[i] != ref[i]) {
            printf("  " ANSI_RED "FAIL" ANSI_RST
                   " [%s cyclic polymul] i=%llu exp=%llu got=%llu\n",
                   m->name, (unsigned long long)i,
                   (unsigned long long)ref[i], (unsigned long long)fa[i]);
            fails++;
            break;
        }

    /* ---- Negacyclic polymul via twisted NTT vs schoolbook ----
     * Need a primitive 2n-th root psi with psi^2 == omega. Twist input by
     * psi^i, do cyclic NTT product, untwist output by psi^-i. */
    uint64_t psi = ntt_modulus_omega(m, 2 * n);
    if (psi == 0 || tp_mod_pow(psi, 2, q) != omega) {
        /* Fall back: derive psi as g^((q-1)/(2n)) directly; require 2n | q-1. */
        if ((q - 1) % (2 * n) == 0)
            psi = tp_mod_pow(m->g, (q - 1) / (2 * n), q);
        else
            psi = 0;
    }
    if (psi == 0 || tp_mod_pow(psi, 2, q) != omega
                 || tp_mod_pow(psi, 2 * n, q) != 1
                 || tp_mod_pow(psi, n, q) == 1) {
        printf("  " ANSI_RED "FAIL" ANSI_RST
               " [%s negacyclic] no valid 2n-th root psi (psi=%llu)\n",
               m->name, (unsigned long long)psi);
        fails++;
    } else {
        uint64_t psi_inv = tp_mod_pow(psi, q - 2, q);
        for (uint64_t i = 0; i < n; i++) {
            uint64_t pw = tp_mod_pow(psi, i, q);
            fa[i] = p.reduce((__uint128_t)xa[i] * pw, q);
            fb[i] = p.reduce((__uint128_t)xb[i] * pw, q);
        }
        ntt_forward(fa, tw, &p);
        ntt_forward(fb, tw, &p);
        for (uint64_t i = 0; i < n; i++)
            fa[i] = p.reduce((__uint128_t)fa[i] * fb[i], q);
        ntt_inverse(fa, twi, &p);
        for (uint64_t i = 0; i < n; i++) {
            uint64_t ipw = tp_mod_pow(psi_inv, i, q);
            fa[i] = p.reduce((__uint128_t)fa[i] * ipw, q);
        }
        schoolbook_negacyclic(xa, xb, ref, n, q);
        for (uint64_t i = 0; i < n; i++)
            if (fa[i] != ref[i]) {
                printf("  " ANSI_RED "FAIL" ANSI_RST
                       " [%s negacyclic polymul] i=%llu exp=%llu got=%llu\n",
                       m->name, (unsigned long long)i,
                       (unsigned long long)ref[i], (unsigned long long)fa[i]);
                fails++;
                break;
            }
    }

    free(tw); free(twi); free(x); free(y);
    free(xa); free(xb); free(ref); free(fa); free(fb);
    return fails ? -1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DRIVER
 * ═══════════════════════════════════════════════════════════════════════════ */

/* The two curated-unverified primes, identified by q value. */
static const uint64_t TARGET_Q[2] = {
    UINT64_C(1152921504606584833),   /* Solinas-60: 2^60 - 2^18 + 1 */
    UINT64_C(2287828610704211969),   /* Solinas-61: 2^61 - 2^54 + 1 */
};

int main(int argc, char *argv[])
{
    uint64_t n_red  = (argc > 1) ? (uint64_t)strtoull(argv[1], NULL, 10)
                                 : UINT64_C(4000000);
    uint64_t n_ari  = (argc > 2) ? (uint64_t)strtoull(argv[2], NULL, 10)
                                 : UINT64_C(2000000);
    uint64_t ntt_n  = (argc > 3) ? (uint64_t)strtoull(argv[3], NULL, 10)
                                 : 1024;   /* O(n^2) schoolbook keeps n modest */

    printf("\n" ANSI_WHT
           "╔══════════════════════════════════════════════════════════╗\n"
           "║   NTT / MI300A — Curated Large-Prime CPU Verification    ║\n"
           "╚══════════════════════════════════════════════════════════╝\n"
           ANSI_RST "\n");
    printf("  reduction cases/prime ≈ %llu   arith cases/prime ≈ %llu   "
           "ntt n=%llu\n\n",
           (unsigned long long)(n_red + 32),
           (unsigned long long)(n_ari * 3),
           (unsigned long long)ntt_n);

    int any_fail = 0;
    struct { const char *name; uint64_t q; int red, ari, ntt;
             uint64_t rc, ac; } row[2];

    for (int k = 0; k < 2; k++) {
        const ntt_modulus_info_t *m = ntt_modulus_find(TARGET_Q[k]);
        if (!m) {
            printf("  " ANSI_RED "FAIL" ANSI_RST
                   " q=%llu not present in NTT_MODULI\n",
                   (unsigned long long)TARGET_Q[k]);
            any_fail = 1;
            row[k].name = "?"; row[k].q = TARGET_Q[k];
            row[k].red = row[k].ari = row[k].ntt = -1;
            row[k].rc = row[k].ac = 0;
            continue;
        }
        /* Deterministic, per-prime seed: reproducible, distinct streams. */
        rng_seed(0xD1CE5B9ULL ^ m->q, 0x9E3779B9ULL + m->q);

        uint64_t rc = 0, ac = 0;
        int r1 = check_reduction(m, n_red, &rc);
        int r2 = check_arith(m, n_ari, &ac);
        int r3 = check_ntt(m, ntt_n);

        row[k].name = m->name; row[k].q = m->q;
        row[k].red = r1; row[k].ari = r2; row[k].ntt = r3;
        row[k].rc = rc; row[k].ac = ac;
        if (r1 || r2 || r3) any_fail = 1;
    }

    /* ---- Fixed-width result table ---- */
    printf(ANSI_CYN
           "  ── Verification Results ────────────────────────────────────────\n"
           ANSI_RST);
    printf("  ┌──────────────┬──────────────────────┬──────┬───────┬──────────┐\n");
    printf("  │ " ANSI_CYN "%-12s" ANSI_RST " │ " ANSI_CYN "%-20s" ANSI_RST
           " │ " ANSI_CYN "%-4s" ANSI_RST " │ " ANSI_CYN "%-5s" ANSI_RST
           " │ " ANSI_CYN "%-8s" ANSI_RST " │\n",
           "Prime", "q", "Red", "Arith", "NTT");
    printf("  ├──────────────┼──────────────────────┼──────┼───────┼──────────┤\n");
    for (int k = 0; k < 2; k++) {
        const char *R = row[k].red == 0 ? ANSI_GRN "PASS" ANSI_RST
                                        : ANSI_RED "FAIL" ANSI_RST;
        const char *A = row[k].ari == 0 ? ANSI_GRN "PASS" ANSI_RST
                                        : ANSI_RED "FAIL" ANSI_RST;
        const char *T = row[k].ntt == 0 ? ANSI_GRN "PASS" ANSI_RST
                                        : ANSI_RED "FAIL" ANSI_RST;
        printf("  │ %-12s │ %20llu │ %s │ %s  │ %s │\n",
               row[k].name, (unsigned long long)row[k].q, R, A, T);
    }
    printf("  └──────────────┴──────────────────────┴──────┴───────┴──────────┘\n");

    int passed = !any_fail;
    printf("\n  %s  reduction+arith+ntt over both curated large primes "
           "(roundtrip + cyclic + negacyclic)\n\n",
           passed ? ANSI_GRN "ALL PASS" ANSI_RST
                  : ANSI_RED "FAILURES PRESENT" ANSI_RST);

    /* ---- One-line summary appended to a timestamped file (project rule 5) ---- */
    char fname[80];
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    strftime(fname, sizeof fname, "curated_primes_%Y%m%d_%H%M%S.txt", tm);
    FILE *f = fopen(fname, "w");
    if (f) {
        fprintf(f,
            "curated-prime-verify result=%s "
            "S60 q=%llu red=%s ari=%s ntt=%s | "
            "S61 q=%llu red=%s ari=%s ntt=%s | "
            "red_cases/prime=%llu ari_cases/prime=%llu ntt_n=%llu\n",
            passed ? "ALL_PASS" : "FAIL",
            (unsigned long long)row[0].q,
            row[0].red ? "FAIL" : "PASS",
            row[0].ari ? "FAIL" : "PASS",
            row[0].ntt ? "FAIL" : "PASS",
            (unsigned long long)row[1].q,
            row[1].red ? "FAIL" : "PASS",
            row[1].ari ? "FAIL" : "PASS",
            row[1].ntt ? "FAIL" : "PASS",
            (unsigned long long)row[0].rc,
            (unsigned long long)row[0].ac,
            (unsigned long long)ntt_n);
        fclose(f);
        printf("  Summary written to %s\n\n", fname);
    }

    return passed ? 0 : 1;
}
