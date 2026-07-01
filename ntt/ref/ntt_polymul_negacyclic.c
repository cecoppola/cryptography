/*
 * ntt_polymul_negacyclic.c — Twisted-NTT negacyclic polynomial multiply.
 *
 * Purpose:   CPU reference for negacyclic convolution mod (X^n + 1) mod q,
 *            as used in PQC schemes (ML-DSA q=8380417 n=256).
 *
 * Algorithm (twisted-NTT, Longa & Naehrig / CRYSTALS convention):
 *   Let psi be a primitive 2n-th root of unity (psi^{2n}=1, psi^n=-1 mod q).
 *   Let omega = psi^2 (a primitive n-th root, NTT base root).
 *
 *   Encode:    f_tw[i] = f[i] * psi^i  (pre-twist)
 *   NTT:       F = NTT(f_tw) using omega
 *   Pointwise: H[i] = F[i] * G[i] mod q
 *   INTT:      h_tw = INTT(H)
 *   Decode:    h[i] = h_tw[i] * psi^{-i}  (untwist)
 *
 *   Result h = f * g mod (X^n + 1) mod q.
 *
 * Lazy Stockham butterfly: intermediate values are not reduced every stage;
 * lazy-overflow guard in ntt_params_init rejects q > UINT64_MAX/(2n).
 *
 * Build (standalone):  cc -O2 -Wall -Wextra -o ntt_polymul_negacyclic \
 *                          ntt_polymul_negacyclic.c
 * Build (library TU):  add -DNTT_NEGACYC_NO_MAIN
 *
 * Usage: ./ntt_polymul_negacyclic [iterations]
 */

#include "ntt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ── ANSI ────────────────────────────────────────────────────────────────── */
#define ANSI_WHT "\033[1;37m"
#define ANSI_CYN "\033[1;36m"
#define ANSI_GRN "\033[1;32m"
#define ANSI_YLW "\033[1;33m"
#define ANSI_RED "\033[1;31m"
#define ANSI_RST "\033[0m"

/* ═══════════════════════════════════════════════════════════════════════════
 * MODULAR ARITHMETIC  (standalone — no link dep)
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint64_t h_mod_pow(uint64_t b, uint64_t e, uint64_t m)
{
    uint64_t r = 1; b %= m;
    for (; e; e >>= 1) {
        if (e & 1) r = (uint64_t)((__uint128_t)r * b % m);
        b = (uint64_t)((__uint128_t)b * b % m);
    }
    return r;
}

/* ntt_params_init: validate params and precompute derived values.
 *
 * Inputs:  p->n (transform size, power of 2 >= 2)
 *          p->q (modulus >= 2)
 *          p->omega (forward twiddle root — caller sets to psi^2 for negacyclic)
 * Outputs: p->omega_inv, p->n_inv, p->log2_n, p->reduce set.
 * Returns: 0 on success, -1 if params invalid or overflow-unsafe.
 *
 * Lazy-overflow guard: the unreduced Stockham butterfly lets intermediate
 * values grow additively by up to q per stage; require q <= UINT64_MAX/(2n)
 * so the worst-case accumulation cannot overflow uint64_t. Callers with
 * large q should use the lib CRT-NTT engine instead. */
int ntt_params_init(ntt_params_t *p)
{
    if (!p->q || p->q < 2) return -1;
    if (!p->n || (p->n & (p->n - 1))) return -1;
    if (p->q > (UINT64_MAX / (2 * p->n))) return -1;
    p->omega_inv = h_mod_pow(p->omega, p->q - 2, p->q);
    p->n_inv     = h_mod_pow(p->n,    p->q - 2, p->q);
    uint64_t t = p->n; p->log2_n = 0;
    while (t > 1) { t >>= 1; p->log2_n++; }
    const ntt_modulus_info_t *mi = ntt_modulus_find(p->q);
    p->reduce = mi ? mi->reduce : reduce_generic;
    return 0;
}

uint64_t *ntt_alloc_twiddles(const ntt_params_t *p)
{
    uint64_t *tw = (uint64_t *)malloc(p->n / 2 * sizeof(uint64_t));
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < p->n / 2; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k-1] * p->omega % p->q);
    return tw;
}

uint64_t *ntt_alloc_twiddles_inv(const ntt_params_t *p)
{
    uint64_t *tw = (uint64_t *)malloc(p->n / 2 * sizeof(uint64_t));
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < p->n / 2; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k-1] * p->omega_inv % p->q);
    return tw;
}

/* alloc_twist: psi^k mod q for k = 0 .. n-1.
 * psi must be a primitive 2n-th root of unity (psi^{2n}=1, psi^n = q-1). */
uint64_t *alloc_twist(uint64_t n, uint64_t q, uint64_t psi)
{
    uint64_t *tw = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < n; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k-1] * psi % q);
    return tw;
}

/* alloc_twist_inv: psi^{-k} mod q for k = 0 .. n-1. */
uint64_t *alloc_twist_inv(uint64_t n, uint64_t q, uint64_t psi)
{
    uint64_t psi_inv = h_mod_pow(psi, q - 2, q);
    uint64_t *tw = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < n; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k-1] * psi_inv % q);
    return tw;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * STOCKHAM NTT  (inline — no link dep on ntt_stockham.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

void ntt_forward(uint64_t *a, const uint64_t *tw, const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q, half = n >> 1;
    uint64_t *buf = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!buf) return;
    uint64_t *src = a, *dst = buf;
    for (uint32_t s = 0; s < p->log2_n; s++) {
        uint64_t pp = (uint64_t)1 << s, qs = n >> (s + 1), p2 = pp << 1;
        for (uint64_t j = 0; j < pp; j++) {
            uint64_t w = tw[j * qs];
            for (uint64_t k = 0; k < qs; k++) {
                uint64_t idx = j + k * pp;
                uint64_t u   = src[idx];
                uint64_t wv  = (uint64_t)((__uint128_t)w * (src[idx + half] % q) % q);
                dst[j + k * p2]      = u + wv;
                dst[j + k * p2 + pp] = u + q - wv;
            }
        }
        uint64_t *t = src; src = dst; dst = t;
    }
    for (uint64_t i = 0; i < n; i++) src[i] %= q;
    if (src != a) memcpy(a, src, n * sizeof(uint64_t));
    free(buf);
}

void ntt_inverse(uint64_t *a, const uint64_t *twi, const ntt_params_t *p)
{
    ntt_forward(a, twi, p);
    for (uint64_t i = 0; i < p->n; i++)
        a[i] = (uint64_t)((__uint128_t)a[i] * p->n_inv % p->q);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * POLYNOMIAL MULTIPLICATION
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * polymul_ntt_negacyclic: c = f * g mod (X^n + 1) mod q via twisted NTT.
 *
 * Inputs:  f, g       — input polynomials, coefficients in [0, q)
 *          tw, twi    — NTT twiddles for omega = psi^2 (forward and inverse)
 *          twist      — pre-twist table: twist[i] = psi^i mod q
 *          twist_inv  — untwist table:  twist_inv[i] = psi^{-i} mod q
 *          p          — NTT params (p->omega = psi^2; n, q set by caller)
 * Output:  c = f * g mod (X^n + 1) mod q.  c may alias f or g.
 */
void polymul_ntt_negacyclic(const uint64_t * restrict f,
                            const uint64_t * restrict g,
                                  uint64_t * restrict c,
                            const uint64_t *tw,  const uint64_t *twi,
                            const uint64_t *twist, const uint64_t *twist_inv,
                            const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;

    uint64_t *F = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *G = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!F || !G) { free(F); free(G); return; }

    /* Pre-twist: F[i] = f[i]*psi^i, G[i] = g[i]*psi^i */
    for (uint64_t i = 0; i < n; i++) {
        F[i] = (uint64_t)((__uint128_t)f[i] * twist[i] % q);
        G[i] = (uint64_t)((__uint128_t)g[i] * twist[i] % q);
    }

    ntt_forward(F, tw, p);
    ntt_forward(G, tw, p);

    /* Pointwise multiply in NTT domain */
    for (uint64_t i = 0; i < n; i++)
        c[i] = (uint64_t)((__uint128_t)F[i] * G[i] % q);

    ntt_inverse(c, twi, p);

    /* Untwist: c[i] = c[i] * psi^{-i} */
    for (uint64_t i = 0; i < n; i++)
        c[i] = (uint64_t)((__uint128_t)c[i] * twist_inv[i] % q);

    free(F); free(G);
}

/* polymul_ntt: stub — present so test_polymul_integ links when compiled with
 * -DINTEG_NEGACYCLIC (declaration visible at TU boundary, never called). */
void polymul_ntt(const uint64_t * restrict f,
                 const uint64_t * restrict g,
                       uint64_t * restrict c,
                 const uint64_t *tw, const uint64_t *twi,
                 const ntt_params_t *p)
{
    /* GCOV_EXCL_START — stub; never called at runtime, linker symbol only */
    (void)f; (void)g; (void)c; (void)tw; (void)twi; (void)p;
    fprintf(stderr, "polymul_ntt: not implemented in negacyclic TU\n");
    abort();
    /* GCOV_EXCL_STOP */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SCHOOLBOOK REFERENCE + SELFTEST  (compiled only when main() is present)
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef NTT_NEGACYC_NO_MAIN

/* c = f * g mod (X^n + 1) mod q, O(n^2). Independent reference for selftest.
 * When i+j >= n the monomial X^{i+j} = X^n * X^{i+j-n} wraps with sign -1. */
static void polymul_schoolbook_negacyclic(const uint64_t *f, const uint64_t *g,
                                          uint64_t *c, const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    for (uint64_t k = 0; k < n; k++) c[k] = 0;
    for (uint64_t i = 0; i < n; i++) {
        for (uint64_t j = 0; j < n; j++) {
            uint64_t prod = (uint64_t)((__uint128_t)f[i] * g[j] % q);
            uint64_t k    = (i + j) % n;
            if (i + j >= n)
                c[k] = (c[k] + q - prod) % q;   /* X^n = -1: subtract */
            else
                c[k] = (c[k] + prod) % q;
        }
    }
}

static int selftest(uint64_t n, uint64_t q, uint64_t psi, const char *label)
{
    printf(ANSI_CYN "  -- Selftest: %s (n=%lu q=%lu psi=%lu) ---\n"
           ANSI_RST, label, (unsigned long)n, (unsigned long)q, (unsigned long)psi);

    uint64_t omega = (uint64_t)((__uint128_t)psi * psi % q);
    ntt_params_t p = { n, q, omega, 0, 0, 0, NULL };
    if (ntt_params_init(&p) != 0) {
        printf("  " ANSI_YLW "SKIP" ANSI_RST
               " ntt_params_init rejected (lazy-overflow or bad params)\n\n");
        return 0;
    }

    uint64_t *tw   = ntt_alloc_twiddles(&p);
    uint64_t *twi  = ntt_alloc_twiddles_inv(&p);
    uint64_t *tws  = alloc_twist(n, q, psi);
    uint64_t *twsi = alloc_twist_inv(n, q, psi);
    uint64_t *f    = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *g    = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *c    = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *ref  = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tw || !twi || !tws || !twsi || !f || !g || !c || !ref) {
        free(tw); free(twi); free(tws); free(twsi);
        free(f); free(g); free(c); free(ref);
        return -1;
    }

    printf("  convolution type: negacyclic (mod X^n+1)\n");
    int ok = 1;

    /* T1: f * 1 = f */
    for (uint64_t i = 0; i < n; i++) { f[i] = (i * 31337 + 1) % q; g[i] = 0; }
    g[0] = 1;
    polymul_ntt_negacyclic(f, g, c, tw, twi, tws, twsi, &p);
    int t1 = 1;
    for (uint64_t i = 0; i < n; i++) if (c[i] != f[i]) { t1 = 0; break; }
    ok &= t1;
    printf("  %-36s %s\n", "f * 1 = f",
           t1 ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    /* T2: vs negacyclic schoolbook */
    for (uint64_t i = 0; i < n; i++) {
        f[i] = (i * 6364136223846793005ULL + 1) % q;
        g[i] = (i * 2862933555777941757ULL + 3) % q;
    }
    polymul_ntt_negacyclic(f, g, c, tw, twi, tws, twsi, &p);
    polymul_schoolbook_negacyclic(f, g, ref, &p);
    int t2 = (memcmp(c, ref, n * sizeof(uint64_t)) == 0);
    ok &= t2;
    printf("  %-36s %s\n", "NTT-negacyc vs schoolbook",
           t2 ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    /* T3: commutativity */
    uint64_t *c2 = (uint64_t *)malloc(n * sizeof(uint64_t));
    polymul_ntt_negacyclic(g, f, c2, tw, twi, tws, twsi, &p);
    int t3 = (c2 != NULL) && (memcmp(c, c2, n * sizeof(uint64_t)) == 0);
    ok &= t3;
    printf("  %-36s %s\n", "commutativity f*g = g*f",
           t3 ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);
    free(c2);

    /* T4: X^(n/2) * X^(n/2) = X^n = -1 (mod X^n+1) → c[0] = q-1 */
    for (uint64_t i = 0; i < n; i++) f[i] = g[i] = 0;
    f[n/2] = 1; g[n/2] = 1;
    polymul_ntt_negacyclic(f, g, c, tw, twi, tws, twsi, &p);
    int t4 = (c[0] == q - 1);
    for (uint64_t i = 1; i < n; i++) if (c[i] != 0) { t4 = 0; break; }
    ok &= t4;
    printf("  %-36s %s\n", "X^(n/2)^2 = -1 (negacyclic)",
           t4 ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    free(tw); free(twi); free(tws); free(twsi);
    free(f); free(g); free(c); free(ref);
    printf("  %s  n=%-4lu q=%lu\n\n",
           ok ? ANSI_GRN "ALL PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST,
           (unsigned long)n, (unsigned long)q);
    return ok ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK  (compiled only when main() is present)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void run_benchmark(uint64_t n, uint64_t q, uint64_t psi,
                          const char *label, uint64_t iters)
{
    uint64_t omega = (uint64_t)((__uint128_t)psi * psi % q);
    ntt_params_t p = { n, q, omega, 0, 0, 0, NULL };
    if (ntt_params_init(&p) != 0) return;

    uint64_t *tw   = ntt_alloc_twiddles(&p);
    uint64_t *twi  = ntt_alloc_twiddles_inv(&p);
    uint64_t *tws  = alloc_twist(n, q, psi);
    uint64_t *twsi = alloc_twist_inv(n, q, psi);
    uint64_t *f    = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *g    = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *c    = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tw || !twi || !tws || !twsi || !f || !g || !c) {
        free(tw); free(twi); free(tws); free(twsi);
        free(f); free(g); free(c); return;
    }
    for (uint64_t i = 0; i < n; i++) {
        f[i] = (i * 31337 + 1) % q;
        g[i] = (i * 99991 + 7) % q;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++)
        polymul_ntt_negacyclic(f, g, c, tw, twi, tws, twsi, &p);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double muls_s  = iters / elapsed;

    printf(ANSI_CYN "  -- Benchmark: polymul_ntt_negacyclic  [%s] --\n"
           ANSI_RST, label);
    printf("  +--------------------------+-------------------------+\n");
    printf("  | %-24s | %-23lu |\n", "n",           (unsigned long)n);
    printf("  | %-24s | %-23lu |\n", "q",           (unsigned long)q);
    printf("  | %-24s | %-23lu |\n", "psi",         (unsigned long)psi);
    printf("  | %-24s | %-23lu |\n", "iterations",  (unsigned long)iters);
    printf("  | %-24s | %-23.6f |\n", "elapsed (s)", elapsed);
    printf("  | %-24s | %-23.0f |\n", "polymul/s",   muls_s);
    printf("  | %-24s | %-23.2f |\n", "us/polymul",  1e6 / muls_s);
    printf("  +--------------------------+-------------------------+\n\n");

    free(tw); free(twi); free(tws); free(twsi);
    free(f); free(g); free(c);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    printf("\n" ANSI_WHT
           "+----------------------------------------------------------+\n"
           "|   NTT Negacyclic Polynomial Multiplication -- CPU ref   |\n"
           "+----------------------------------------------------------+\n"
           ANSI_RST "\n");

    uint64_t iters = (argc >= 2) ? (uint64_t)atoll(argv[1]) : 100000;

    /* ML-DSA: q=8380417, n=256, psi=1753.
     * 1753 is the primitive 512th root of unity (1753^256 = -1 mod 8380417).
     * omega = psi^2 = 3073009 is the primitive 256th root (NTT base). */
    int fail = 0;
    fail |= (selftest(256, 8380417, 1753, "ML-DSA") < 0);
    if (fail) return 1;

    run_benchmark(256, 8380417, 1753, "ML-DSA", iters);
    return 0;
}
#endif /* NTT_NEGACYC_NO_MAIN */
