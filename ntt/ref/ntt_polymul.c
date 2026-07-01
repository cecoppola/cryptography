/*
 * ntt_polymul.c — NTT-based polynomial multiplication for ML-KEM / ML-DSA
 *
 * Purpose:   NTT-based polynomial multiplication benchmark for MI300A Phase 4.
 *            Computes the cyclic convolution f * g mod (X^n - 1) mod q via:
 *            forward NTT both inputs, pointwise multiply mod q, inverse NTT.
 * Algorithm: Stockham NTT (no bit-reversal), pointwise multiply, Stockham INTT.
 *            CPU reference; GPU version to be written in HIP.
 *
 * Cyclic vs negacyclic: our NTT (omega a primitive n-th root of unity) computes
 *   the DFT over Z_q[X]/(X^n - 1), giving cyclic convolution.
 *   PQC (ML-KEM, ML-DSA) requires negacyclic convolution over Z_q[X]/(X^n + 1),
 *   which needs omega to be a primitive 2n-th root (twisted-NTT approach).
 *   For ML-DSA q=8380417 this is possible (2*256 | q-1). For ML-KEM q=3329 it
 *   requires the FIPS 203 specialized NTT (128-point with CRT decomposition).
 *   Both are Phase 4 extensions; the cyclic version here validates the core
 *   NTT+pointwise-multiply pipeline.
 *   Build:   cc -O2 -Wall -Wextra -o ntt_polymul ntt_polymul.c
 */

#include "ntt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

int ntt_params_init(ntt_params_t *p)
{
    if (!p->n || (p->n & (p->n-1))) return -1;
    /* q>=2: q=0 would SIGFPE in h_mod_pow (mod 0); q=1 is degenerate. */
    if (p->q < 2) return -1;
    /* Lazy-accumulation headroom guard. The Stockham butterfly below does
     * dst = u + wv and u + q - wv with NO per-stage reduction, so values
     * grow additively by up to q per stage. Require q <= UINT64_MAX/(2n)
     * so the unreduced accumulation provably cannot overflow uint64_t.
     * This admits the intended ML-KEM/ML-DSA small primes and loudly
     * rejects large-q (Goldilocks/Solinas ~2^60-2^64) for which this
     * lazy harness would be SILENTLY WRONG — use lib / the curated
     * full-reduction path for those instead. */
    if (p->q > (UINT64_MAX / (2 * p->n))) return -1;
    p->omega_inv = h_mod_pow(p->omega, p->q-2, p->q);
    p->n_inv     = h_mod_pow(p->n,    p->q-2, p->q);
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
    for (uint64_t k = 1; k < p->n/2; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k-1] * p->omega % p->q);
    return tw;
}

uint64_t *ntt_alloc_twiddles_inv(const ntt_params_t *p)
{
    uint64_t *tw = (uint64_t *)malloc(p->n / 2 * sizeof(uint64_t));
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < p->n/2; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k-1] * p->omega_inv % p->q);
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
        uint64_t pp = (uint64_t)1 << s, qs = n>>(s+1), p2 = pp<<1;
        for (uint64_t j = 0; j < pp; j++) {
            uint64_t w = tw[j * qs];
            for (uint64_t k = 0; k < qs; k++) {
                uint64_t idx = j + k*pp;
                uint64_t u   = src[idx];
                uint64_t wv  = (uint64_t)((__uint128_t)w * (src[idx+half]%q) % q);
                dst[j+k*p2]    = u + wv;
                dst[j+k*p2+pp] = u + q - wv;
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
 * polymul_ntt: compute c = f * g mod (X^n - 1) mod q using cyclic NTT.
 *
 * Steps:
 *   1. Forward NTT of f and g (in-place Stockham).
 *   2. Pointwise multiply: F[i] * G[i] mod q.
 *   3. Inverse NTT: c = INTT(F * G).
 *
 * This is a cyclic convolution (mod X^n - 1). For negacyclic (PQC use),
 * use a twisted-NTT parameterization with omega a 2n-th root of unity.
 *
 * c may alias f or g (a copy of f is made internally).
 */
void polymul_ntt(const uint64_t * restrict f,
                 const uint64_t * restrict g,
                       uint64_t * restrict c,
                 const uint64_t *tw, const uint64_t *twi,
                 const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;

    /* Copy inputs: forward NTT modifies in-place */
    uint64_t *F = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *G = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!F || !G) { free(F); free(G); return; }

    memcpy(F, f, n * sizeof(uint64_t));
    memcpy(G, g, n * sizeof(uint64_t));

    ntt_forward(F, tw, p);
    ntt_forward(G, tw, p);

    /* Pointwise multiply in NTT domain */
    for (uint64_t i = 0; i < n; i++)
        c[i] = (uint64_t)((__uint128_t)F[i] * G[i] % q);

    ntt_inverse(c, twi, p);

    free(F); free(G);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SELFTEST / BENCHMARK / MAIN
 *
 * Everything below is only reachable from this file's own main(). Compiled
 * out when NTT_POLYMUL_NO_MAIN is defined so an external driver (e.g.
 * test_polymul_integ.c) can link this TU for polymul_ntt + ntt_forward
 * without a duplicate main() or -Wunused-function under -Wall -Wextra.
 * Mirrors the NTT_CPU_NO_MAIN guard in ntt_cpu.c.
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef NTT_POLYMUL_NO_MAIN

/*
 * polymul_schoolbook: O(n^2) cyclic convolution mod (X^n - 1) mod q.
 * Reference for selftest. Our NTT always computes cyclic convolution regardless
 * of whether omega has order n (n-th root) or 2n (2n-th root) — the Stockham
 * and CT-DIT algorithms use twiddles at even powers, which are always n-th roots
 * of unity, so the NTT evaluates at the n-th roots and the product gives cyclic
 * convolution in both cases.
 *
 * For PQC negacyclic convolution (mod X^n + 1), a twisted-NTT with pre/post
 * multiplication by powers of a 2n-th root is required (Phase 4 extension).
 */
static void polymul_schoolbook(const uint64_t *f, const uint64_t *g,
                                uint64_t *c, const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    for (uint64_t k = 0; k < n; k++) c[k] = 0;
    for (uint64_t i = 0; i < n; i++)
        for (uint64_t j = 0; j < n; j++) {
            uint64_t prod = (uint64_t)((__uint128_t)f[i] * g[j] % q);
            c[(i + j) % n] = (c[(i + j) % n] + prod) % q;
        }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SELFTEST
 * ═══════════════════════════════════════════════════════════════════════════ */

static int selftest(const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    const char *lbl = (q==3329)?"ML-KEM":(q==8380417)?"ML-DSA":"custom";

    printf(ANSI_CYN "  ── Selftest: %s (n=%lu q=%lu ω=%lu) ──────────────────\n"
           ANSI_RST, lbl, n, q, p->omega);

    uint64_t *tw  = ntt_alloc_twiddles(p);
    uint64_t *twi = ntt_alloc_twiddles_inv(p);
    uint64_t *f   = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *g   = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *c   = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *ref = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tw || !twi || !f || !g || !c || !ref) {
        free(tw); free(twi); free(f); free(g); free(c); free(ref); return -1;
    }

    /* Our NTT always computes cyclic convolution (mod X^n - 1). */
    printf("  convolution type: cyclic (mod X^n-1)\n");
    int ok = 1;

    /* ── Test 1: f*1 = f  (multiply by the unit polynomial) ────────────── */
    for (uint64_t i = 0; i < n; i++) { f[i] = (i*31337+1)%q; g[i] = 0; }
    g[0] = 1;
    polymul_ntt(f, g, c, tw, twi, p);
    int t1 = 1;
    for (uint64_t i = 0; i < n; i++) if (c[i] != f[i]) { t1 = 0; break; }
    ok &= t1;
    printf("  %-36s %s\n", "f * 1 = f",
           t1 ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    /* ── Test 2: vs schoolbook reference ─────────────────────────────────── */
    for (uint64_t i = 0; i < n; i++) {
        f[i] = (i*6364136223846793005ULL + 1) % q;
        g[i] = (i*2862933555777941757ULL + 3) % q;
    }
    polymul_ntt(f, g, c, tw, twi, p);
    polymul_schoolbook(f, g, ref, p);
    int t2 = 1;
    for (uint64_t i = 0; i < n; i++) if (c[i] != ref[i]) { t2 = 0; break; }
    ok &= t2;
    printf("  %-36s %s\n", "NTT-polymul vs schoolbook",
           t2 ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    /* ── Test 3: commutativity  f*g = g*f ───────────────────────────────── */
    uint64_t *c2 = (uint64_t *)malloc(n * sizeof(uint64_t));
    polymul_ntt(g, f, c2, tw, twi, p);
    int t3 = (c2 != NULL);
    if (t3) for (uint64_t i = 0; i < n; i++) if (c[i] != c2[i]) { t3 = 0; break; }
    ok &= t3;
    printf("  %-36s %s\n", "commutativity f*g = g*f",
           t3 ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);
    free(c2);

    /* ── Test 4: X^(n/2) * X^(n/2) = X^n ≡ 1 (cyclic, mod X^n - 1) ─────── */
    for (uint64_t i = 0; i < n; i++) f[i] = g[i] = 0;
    f[n/2] = 1; g[n/2] = 1;
    polymul_ntt(f, g, c, tw, twi, p);
    int t4 = (c[0] == 1);
    for (uint64_t i = 1; i < n; i++) if (c[i] != 0) { t4 = 0; break; }
    ok &= t4;
    printf("  %-36s %s\n", "X^(n/2)^2 = X^n ≡ 1 (cyclic)",
           t4 ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    free(tw); free(twi); free(f); free(g); free(c); free(ref);
    printf("  %s  n=%-4lu q=%lu\n\n",
           ok ? ANSI_GRN "ALL PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST, n, q);
    return ok ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK
 * ═══════════════════════════════════════════════════════════════════════════ */

static void run_benchmark(const ntt_params_t *p, uint64_t iters)
{
    uint64_t n = p->n, q = p->q;
    uint64_t *tw  = ntt_alloc_twiddles(p);
    uint64_t *twi = ntt_alloc_twiddles_inv(p);
    uint64_t *f   = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *g   = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *c   = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tw || !twi || !f || !g || !c) {
        free(tw); free(twi); free(f); free(g); free(c); return;
    }
    for (uint64_t i = 0; i < n; i++) {
        f[i] = (i * 31337 + 1) % q;
        g[i] = (i * 99991 + 7) % q;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) polymul_ntt(f, g, c, tw, twi, p);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)*1e-9;
    double muls_s  = iters / elapsed;

    printf(ANSI_CYN "  ── Benchmark: polymul_ntt (CPU Stockham) ───────────────\n" ANSI_RST);
    printf("  ┌──────────────────────────┬─────────────────────────┐\n");
    printf("  │ %-24s │ %-23lu │\n", "n",           n);
    printf("  │ %-24s │ %-23lu │\n", "q",           q);
    printf("  │ %-24s │ %-23lu │\n", "iterations",  iters);
    printf("  │ %-24s │ %-23.6f │\n", "elapsed (s)", elapsed);
    printf("  │ %-24s │ %-23.0f │\n", "polymul/s",   muls_s);
    printf("  │ %-24s │ %-23.2f │\n", "µs/polymul",  1e6 / muls_s);
    printf("  └──────────────────────────┴─────────────────────────┘\n\n");

    free(tw); free(twi); free(f); free(g); free(c);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    printf("\n" ANSI_WHT
           "╔══════════════════════════════════════════════════════════╗\n"
           "║       NTT Polynomial Multiplication — CPU reference     ║\n"
           "╚══════════════════════════════════════════════════════════╝\n"
           ANSI_RST "\n");

    uint64_t iters = (argc >= 2) ? (uint64_t)atoll(argv[1]) : 100000;

    /* ML-KEM: omega=17 is a primitive 256th root of unity mod 3329 (17^256≡1). */
    ntt_params_t mlkem = { 256, 3329, 17, 0, 0, 0, NULL };
    /* ML-DSA: omega=1753 has order 512 (1753^256≡-1), so cyclic polymul requires
     * omega_cyclic = 1753^2 = 3073009 (order 256, 3073009^256≡1 mod 8380417).
     * The true ML-DSA NTT uses the negacyclic structure with omega=1753 and a
     * pre-twist — this is deferred to Phase 4 (twisted-NTT implementation). */
    ntt_params_t mldsa_cyc = { 256, 8380417, 3073009, 0, 0, 0, NULL };
    int fail = 0;
    if (ntt_params_init(&mlkem)     == 0) fail |= selftest(&mlkem);
    if (ntt_params_init(&mldsa_cyc) == 0) fail |= selftest(&mldsa_cyc);
    if (fail) return 1;

    if (ntt_params_init(&mlkem)     == 0) run_benchmark(&mlkem, iters);
    if (ntt_params_init(&mldsa_cyc) == 0) run_benchmark(&mldsa_cyc, iters);
    return 0;
}
#endif /* NTT_POLYMUL_NO_MAIN */
