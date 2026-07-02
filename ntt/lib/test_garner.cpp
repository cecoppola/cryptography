/*
 * test_garner.cpp — Host correctness test for Garner CRT reconstruction.
 *
 * Garner (garner_precompute / garner_one / garner_reconstruct) is the final
 * integer-assembly step of every CRT-NTT multiply: it rebuilds each ~248-bit
 * product coefficient from its four 64-bit residues. It runs on the GPU in
 * production, but garner.hip is host-compilable (the HIP kernels are behind
 * #ifdef __HIPCC__), so this test exercises the exact per-coefficient math on
 * the CPU and is part of the GPU-free `make check` gate.
 *
 * Three checks (all via the public garner_reconstruct):
 *   A. Known single-word vectors: residues r_i = x for a small/large 64-bit x
 *      (x < every p_i) must reconstruct to the exact integer U256{x,0,0,0}.
 *      This anchors the result to a true value, not merely a consistent one.
 *   B. Edge residues: all-zero -> 0; all p_i-1; single-lane; each checked by
 *      the CRT self-consistency property X mod p_i == r_i.
 *   C. Randomized fuzz: many random residue tuples, self-consistency.
 *
 * Compiled by lib/Makefile `test-garner` with g++ -fopenmp (garner_reconstruct
 * uses OpenMP; the math is identical to the device garner_one_dev kernel).
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "crt_ntt.h"
#include "garner.hip"

/* X mod p over the four little-endian 64-bit words (Horner from the top). */
static uint64_t u256_mod(const U256 *X, uint64_t p)
{
    unsigned __int128 r = 0;
    for (int i = 3; i >= 0; i--) {
        unsigned __int128 hi = (r % p);              /* r < p */
        unsigned __int128 t  = (hi << 64) % p;       /* hi < p < 2^64 */
        t += (unsigned __int128)X->w[i] % p;
        r  = t % p;
    }
    return (uint64_t)r;
}

static int u256_eq_u64(const U256 *X, uint64_t v)
{
    return X->w[0] == v && X->w[1] == 0 && X->w[2] == 0 && X->w[3] == 0;
}

/* xorshift128+ — deterministic, fixed seed for reproducibility. */
static uint64_t s0 = 0x9E3779B97F4A7C15ull, s1 = 0xD1B54A32D192ED03ull;
static uint64_t rnd(void)
{
    uint64_t x = s0, y = s1;
    s0 = y; x ^= x << 23; s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s1 + y;
}

static int check_selfconsistent(const uint64_t *r0, const uint64_t *r1,
                                const uint64_t *r2, const uint64_t *r3,
                                const U256 *X, int n)
{
    int errs = 0, shown = 0;
    for (int i = 0; i < n; i++) {
        uint64_t c0 = u256_mod(&X[i], PRIMES[0]), c1 = u256_mod(&X[i], PRIMES[1]);
        uint64_t c2 = u256_mod(&X[i], PRIMES[2]), c3 = u256_mod(&X[i], PRIMES[3]);
        if (c0 != r0[i] || c1 != r1[i] || c2 != r2[i] || c3 != r3[i]) {
            if (shown++ < 5)
                fprintf(stderr, "  self-consistency FAIL @%d: r=(%llu,%llu,%llu,%llu) "
                        "X mod p=(%llu,%llu,%llu,%llu)\n", i,
                        (unsigned long long)r0[i], (unsigned long long)r1[i],
                        (unsigned long long)r2[i], (unsigned long long)r3[i],
                        (unsigned long long)c0, (unsigned long long)c1,
                        (unsigned long long)c2, (unsigned long long)c3);
            errs++;
        }
    }
    return errs;
}

int main(int argc, char **argv)
{
    int n_rand = argc > 1 ? atoi(argv[1]) : 2000000;
    GarnerConsts g = garner_precompute();
    long errs = 0;

    /* ── A. Known single-word vectors ────────────────────────────────────── */
    {
        const uint64_t xs[] = {
            0, 1, 2, 7, 10, 0xFFFFFFFFull, 0x100000000ull,
            0xFFFFFFFFFEull, 0x7FFFFFFFFFFFFFFFull, 0xFEDCBA9876543210ull
        };
        const int nx = (int)(sizeof(xs) / sizeof(xs[0]));
        uint64_t r0[16], r1[16], r2[16], r3[16];
        U256 X[16];
        for (int i = 0; i < nx; i++) {
            /* xs[i] < every p_i (all p_i > 2^63.9), so r_i = x for all lanes. */
            r0[i] = xs[i]; r1[i] = xs[i]; r2[i] = xs[i]; r3[i] = xs[i];
        }
        garner_reconstruct(r0, r1, r2, r3, X, nx, &g);
        int a_err = 0;
        for (int i = 0; i < nx; i++)
            if (!u256_eq_u64(&X[i], xs[i])) {
                fprintf(stderr, "  known-vector FAIL x=%llu -> "
                        "X=(%llu,%llu,%llu,%llu)\n", (unsigned long long)xs[i],
                        (unsigned long long)X[i].w[0], (unsigned long long)X[i].w[1],
                        (unsigned long long)X[i].w[2], (unsigned long long)X[i].w[3]);
                a_err++;
            }
        printf("  A known single-word vectors (%2d)      : %s\n",
               nx, a_err ? "FAIL" : "ok");
        errs += a_err;
    }

    /* ── B. Edge residue tuples (self-consistency) ───────────────────────── */
    {
        uint64_t r0[8], r1[8], r2[8], r3[8];
        U256 X[8];
        int m = 0;
        /* all zero -> X == 0 */                     r0[m]=0;               r1[m]=0;               r2[m]=0;               r3[m]=0;               m++;
        /* all p_i-1 (max residues) */               r0[m]=PRIMES[0]-1;     r1[m]=PRIMES[1]-1;     r2[m]=PRIMES[2]-1;     r3[m]=PRIMES[3]-1;     m++;
        /* single lanes set */                       r0[m]=PRIMES[0]-1;     r1[m]=0;               r2[m]=0;               r3[m]=0;               m++;
                                                     r0[m]=0;               r1[m]=PRIMES[1]-1;     r2[m]=0;               r3[m]=0;               m++;
                                                     r0[m]=0;               r1[m]=0;               r2[m]=PRIMES[2]-1;     r3[m]=0;               m++;
                                                     r0[m]=0;               r1[m]=0;               r2[m]=0;               r3[m]=PRIMES[3]-1;     m++;
        /* mid values */                             r0[m]=PRIMES[0]/2;     r1[m]=PRIMES[1]/2;     r2[m]=PRIMES[2]/2;     r3[m]=PRIMES[3]/2;     m++;
        /* mixed */                                  r0[m]=1;               r1[m]=PRIMES[1]-1;     r2[m]=2;               r3[m]=PRIMES[3]-3;     m++;
        garner_reconstruct(r0, r1, r2, r3, X, m, &g);
        int b_err = check_selfconsistent(r0, r1, r2, r3, X, m);
        if (!u256_eq_u64(&X[0], 0)) { fprintf(stderr, "  edge FAIL: all-zero residues did not reconstruct to 0\n"); b_err++; }
        printf("  B edge residue tuples (%d)              : %s\n",
               m, b_err ? "FAIL" : "ok");
        errs += b_err;
    }

    /* ── C. Randomized self-consistency fuzz ─────────────────────────────── */
    {
        int n = n_rand;
        uint64_t *r0 = (uint64_t*)malloc((size_t)n*8), *r1 = (uint64_t*)malloc((size_t)n*8);
        uint64_t *r2 = (uint64_t*)malloc((size_t)n*8), *r3 = (uint64_t*)malloc((size_t)n*8);
        U256 *X = (U256*)malloc((size_t)n*sizeof(U256));
        if (!r0||!r1||!r2||!r3||!X) { fprintf(stderr, "test_garner: OOM\n"); return 2; }
        for (int i = 0; i < n; i++) {
            r0[i] = rnd() % PRIMES[0]; r1[i] = rnd() % PRIMES[1];
            r2[i] = rnd() % PRIMES[2]; r3[i] = rnd() % PRIMES[3];
        }
        garner_reconstruct(r0, r1, r2, r3, X, n, &g);
        int c_err = check_selfconsistent(r0, r1, r2, r3, X, n);
        printf("  C random self-consistency (%d) : %s\n", n, c_err ? "FAIL" : "ok");
        errs += c_err;
        free(r0); free(r1); free(r2); free(r3); free(X);
    }

    printf("test_garner: %s (%ld errors)\n", errs ? "FAIL" : "PASS", errs);
    return errs ? 1 : 0;
}
