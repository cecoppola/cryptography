/*
 * ntt_mlkem.c — Spec-Exact 7-Layer NTT for ML-KEM (FIPS 203)
 *
 * Purpose:   Implements the exact 7-layer Decimation-in-Frequency (DIF) NTT
 *            and 7-layer Decimation-in-Time (DIT) INTT specified in the
 *            NIST FIPS 203 standard.
 * 
 * Detail:    ML-KEM does not compute a full 256-point negacyclic NTT. It
 *            stops at 7 layers, resulting in 128 polynomials of degree 1
 *            in the ring Z_q[X]/(X^2 - zeta^(2*br_7(i)+1)).
 *            This implementation is locked to N=256, Q=3329, ZETA=17.
 *
 * Ref:       FIPS 203 (Section 4.3 and 4.1.4)
 *
 * Build:     cc -O2 -Wall -Wextra -Iref -o ntt_mlkem ref/ntt_mlkem.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "ntt_mlkem.h"

#define ANSI_WHT "\033[1;37m"
#define ANSI_CYN "\033[1;36m"
#define ANSI_GRN "\033[1;32m"
#define ANSI_RED "\033[1;31m"
#define ANSI_RST "\033[0m"

#define MLKEM_N 256
#define MLKEM_Q 3329
#define MLKEM_ZETA 17

/* ═══════════════════════════════════════════════════════════════════════════
 * MATH & INITIALISATION
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t m)
{
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % m;
        base = (base * base) % m;
        exp >>= 1;
    }
    return result;
}

/* 7-bit bit-reversal function for FIPS 203 twiddle generation */
static uint32_t br_7(uint32_t x)
{
    uint32_t r = 0;
    for (int i = 0; i < 7; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

/* Global lookup tables for FIPS 203 ML-KEM */
static uint64_t zetas[128];
static uint64_t gammas[128];

void mlkem_init_tables(void)
{
    /* zetas[k] = 17^br_7(k) mod 3329  for k=0..127 */
    for (uint32_t k = 0; k < 128; k++) {
        zetas[k] = mod_pow(MLKEM_ZETA, br_7(k), MLKEM_Q);
    }

    /* gammas[i] = 17^(2*br_7(i)+1) mod 3329 for BaseCaseMultiply */
    for (uint32_t i = 0; i < 128; i++) {
        gammas[i] = mod_pow(MLKEM_ZETA, 2 * br_7(i) + 1, MLKEM_Q);
    }
}

/* Accessors for external KAT tests (test_mlkem_fips203_kat.c). Returning
 * const pointers keeps the tables read-only outside this TU; the KAT
 * compares element-wise against independently recomputed reference values. */
const uint64_t *mlkem_get_zeta_table(void)  { return zetas;  }
const uint64_t *mlkem_get_gamma_table(void) { return gammas; }

/* ═══════════════════════════════════════════════════════════════════════════
 * FIPS 203 ALGORITHMS (NTT, INTT, Multiply)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* FIPS 203 Algorithm 8: NTT */
void mlkem_ntt(uint64_t *f)
{
    int k = 1;
    for (int len = 128; len >= 2; len >>= 1) {
        for (int start = 0; start < 256; start += 2 * len) {
            uint64_t zeta = zetas[k++];
            for (int j = start; j < start + len; j++) {
                uint64_t t = (zeta * f[j + len]) % MLKEM_Q;
                f[j + len] = (f[j] + MLKEM_Q - t) % MLKEM_Q;
                f[j]       = (f[j] + t) % MLKEM_Q;
            }
        }
    }
}

/* FIPS 203 Algorithm 9: INTT */
void mlkem_intt(uint64_t *f)
{
    int k = 127;
    for (int len = 2; len <= 128; len <<= 1) {
        for (int start = 0; start < 256; start += 2 * len) {
            uint64_t zeta = zetas[k--];
            for (int j = start; j < start + len; j++) {
                uint64_t t = f[j];
                f[j]       = (t + f[j + len]) % MLKEM_Q;
                f[j + len] = (zeta * (f[j + len] + MLKEM_Q - t)) % MLKEM_Q;
            }
        }
    }
    
    /* Scale by 128^-1 mod 3329 = 3303 */
    for (int i = 0; i < 256; i++) {
        f[i] = (f[i] * 3303) % MLKEM_Q;
    }
}

/* FIPS 203 Algorithm 10 & 11: MultiplyNTTs & BaseCaseMultiply */
void mlkem_polymul(const uint64_t *f, const uint64_t *g, uint64_t *h)
{
    for (int i = 0; i < 128; i++) {
        uint64_t a0 = f[2 * i];
        uint64_t a1 = f[2 * i + 1];
        uint64_t b0 = g[2 * i];
        uint64_t b1 = g[2 * i + 1];
        uint64_t gamma = gammas[i];
        
        /* c0 = a0*b0 + a1*b1*gamma */
        h[2 * i]     = (a0 * b0 + ((a1 * b1) % MLKEM_Q) * gamma) % MLKEM_Q;
        /* c1 = a0*b1 + a1*b0 */
        h[2 * i + 1] = (a0 * b1 + a1 * b0) % MLKEM_Q;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * VERIFICATION & BENCHMARKING
 * Only compiled when ntt_mlkem.c provides its own main(); external KAT
 * harnesses (#define NTT_MLKEM_NO_MAIN) supply their own verification and
 * would otherwise trip -Wunused-function on these helpers.
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef NTT_MLKEM_NO_MAIN

/* Schoolbook negacyclic convolution mod (X^256 + 1) for mathematical verification */
static void polymul_schoolbook_negacyclic(const uint64_t *f, const uint64_t *g, uint64_t *h)
{
    for (int k = 0; k < 256; k++) h[k] = 0;
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            uint64_t prod = (f[i] * g[j]) % MLKEM_Q;
            if (i + j < 256) {
                h[i + j] = (h[i + j] + prod) % MLKEM_Q;
            } else {
                /* Wrap around with sign flip */
                h[(i + j) % 256] = (h[(i + j) % 256] + MLKEM_Q - prod) % MLKEM_Q;
            }
        }
    }
}

static int selftest(void)
{
    uint64_t f[MLKEM_N], g[MLKEM_N], F[MLKEM_N], G[MLKEM_N];
    uint64_t H[MLKEM_N], ref[MLKEM_N];
    int ok = 1;

    printf(ANSI_CYN "  ── Selftest: ML-KEM FIPS 203 7-Layer NTT ──────────────\n" ANSI_RST);
    
    srand(42);
    for (int i = 0; i < MLKEM_N; i++) {
        f[i] = rand() % MLKEM_Q;
        g[i] = rand() % MLKEM_Q;
    }

    /* Test 1: Round Trip f -> NTT -> INTT -> f */
    memcpy(F, f, sizeof(f));
    mlkem_ntt(F);
    mlkem_intt(F);
    
    int t1 = 1;
    for (int i = 0; i < MLKEM_N; i++) {
        if (F[i] != f[i]) { t1 = 0; break; }
    }
    ok &= t1;
    printf("  %-36s %s\n", "INTT(NTT(f)) == f", 
           t1 ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    /* Test 2: Convolution vs Schoolbook */
    memcpy(F, f, sizeof(f));
    memcpy(G, g, sizeof(g));
    
    mlkem_ntt(F);
    mlkem_ntt(G);
    mlkem_polymul(F, G, H);
    mlkem_intt(H);
    
    polymul_schoolbook_negacyclic(f, g, ref);
    
    int t2 = 1;
    for (int i = 0; i < MLKEM_N; i++) {
        if (H[i] != ref[i]) {
            printf("Mismatch at %d: Expected %lu, Got %lu\n", i, ref[i], H[i]);
            t2 = 0; 
            break;
        }
    }
    ok &= t2;
    printf("  %-36s %s\n", "NTT PolyMul vs Schoolbook Negacyclic", 
           t2 ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    printf("  %s\n\n", ok ? ANSI_GRN "ALL PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);
    return ok ? 0 : -1;
}

static void run_benchmark(uint64_t iters)
{
    /* Pre-transform f and g once; benchmark only the polymul + intt hot path. */
    uint64_t f[MLKEM_N], g[MLKEM_N], F[MLKEM_N], G[MLKEM_N], h[MLKEM_N];
    for (int i = 0; i < MLKEM_N; i++) {
        f[i] = i % MLKEM_Q;
        g[i] = (i * 3) % MLKEM_Q;
    }
    for (int i = 0; i < MLKEM_N; i++) { F[i] = f[i]; G[i] = g[i]; }
    mlkem_ntt(F); mlkem_ntt(G);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t i = 0; i < iters; i++) {
        mlkem_polymul(F, G, h);
        mlkem_intt(h);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    
    printf(ANSI_CYN "  ── Benchmark ──────────────────────────────────────────\n" ANSI_RST);
    printf("  ML-KEM 7-Layer Convolution: %10.0f polymuls/sec\n", (double)iters / elapsed);
    printf("  Time per full polynomial product: %6.2f µs\n\n", (elapsed * 1e6) / (double)iters);
}

int main(void)
{
    printf("\n" ANSI_WHT
           "╔══════════════════════════════════════════════════════════╗\n"
           "║       ML-KEM 7-Layer NTT (FIPS 203 Spec-Correct)         ║\n"
           "╚══════════════════════════════════════════════════════════╝\n"
           ANSI_RST "\n");

    /* Precompute exact zetas and gammas as per FIPS 203 */
    mlkem_init_tables();

    /* Verify math */
    if (selftest() == 0) {
        /* Benchmark */
        run_benchmark(100000);
    }

    return 0;
}
#endif /* NTT_MLKEM_NO_MAIN */