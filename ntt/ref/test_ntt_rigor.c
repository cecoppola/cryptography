/*
 * test_ntt_rigor.c — adversarial CPU NTT correctness test over the FULL
 * 14-prime curated table (ntt_moduli.h), every reduction class.
 *
 * Motivation: this project's real defects lived in *vacuous* tests
 * (round-trip passes for ANY invertible omega; a "reference" sharing the
 * code under test; a non-primitive tabled generator g=3). This test is
 * built to be non-vacuous and to be a permanent REGRESSION for the exact
 * bug classes already hit:
 *
 *   PRIM  primitive-root guard — omega = ntt_modulus_omega(m,n) must be a
 *         genuine primitive n-th root: omega!=0,1 ; omega^n==1 ;
 *         omega^(n/2)==q-1 (==-1). This FAILS if a table generator g is
 *         non-primitive (the Solinas-60 g=3 / q=7681 g=3 class).
 *   DFT   forward NTT vs an INDEPENDENT O(n^2) DFT, X[k]=sum_j x[j]
 *         omega^{jk} mod q, computed with unsigned __int128 and NOT the
 *         project butterfly/reduction. Catches wrong twiddles/order/
 *         reduction that a round-trip cannot.
 *   RT    INTT(NTT(x))==x for structured+random inputs, PLUS a vacuity
 *         guard: for a non-constant x, NTT(x) must differ from x and not
 *         be constant (kills the omega=1 / identity-transform degeneracy
 *         that a bare round-trip would wave through).
 *   CONV  cyclic convolution via NTT vs an INDEPENDENT O(n^2) schoolbook
 *         sum a[i]b[(k-i) mod n] mod q. Catches Hadamard/scale errors.
 *
 * All references use plain (unsigned __int128) %q arithmetic — disjoint
 * from the Solinas/Montgomery code under test. CPU only, no GPU, no GMP.
 * Deterministic splitmix64. Exit 0 iff every (modulus,n,test) PASSes.
 *
 * Build: cc -O2 -Wall -Wextra -DNTT_CPU_NO_MAIN \
 *           ref/test_ntt_rigor.c ref/ntt_cpu.c -o bin/test_ntt_rigor
 */
#include "ntt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── deterministic PRNG ──────────────────────────────────────────────────── */
static uint64_t SM(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ── independent modular arithmetic (NOT the code under test) ────────────── */
static uint64_t rmul(uint64_t a, uint64_t b, uint64_t q)
{
    return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) % q);
}
static uint64_t rpow(uint64_t b, uint64_t e, uint64_t q)
{
    uint64_t r = 1 % q;
    b %= q;
    while (e) { if (e & 1) r = rmul(r, b, q); b = rmul(b, b, q); e >>= 1; }
    return r;
}

/* independent O(n^2) forward DFT: X[k] = sum_j x[j] * omega^{jk} mod q */
static void ref_dft(const uint64_t *x, uint64_t *X, uint64_t n,
                     uint64_t omega, uint64_t q)
{
    for (uint64_t k = 0; k < n; k++) {
        unsigned __int128 acc = 0;        /* 128-bit: q may be ~2^64 */
        uint64_t wk = rpow(omega, k, q), cur = 1;
        for (uint64_t j = 0; j < n; j++) {
            acc += rmul(x[j], cur, q);    /* < n * 2^64 << 2^128     */
            cur  = rmul(cur, wk, q);
        }
        X[k] = (uint64_t)(acc % q);
    }
}

/* independent O(n^2) cyclic convolution mod q */
static void ref_conv(const uint64_t *a, const uint64_t *b, uint64_t *c,
                      uint64_t n, uint64_t q)
{
    for (uint64_t k = 0; k < n; k++) {
        unsigned __int128 acc = 0;        /* 128-bit: q may be ~2^64 */
        for (uint64_t i = 0; i < n; i++) {
            uint64_t bi = (k >= i) ? (k - i) : (k + n - i);
            acc += rmul(a[i], b[bi], q);
        }
        c[k] = (uint64_t)(acc % q);
    }
}

static int all_equal(const uint64_t *v, uint64_t n)
{
    for (uint64_t i = 1; i < n; i++) if (v[i] != v[0]) return 0;
    return 1;
}
static int vec_eq(const uint64_t *a, const uint64_t *b, uint64_t n)
{
    return memcmp(a, b, (size_t)n * sizeof(uint64_t)) == 0;
}

/* one (modulus, n) cell: returns 0 on full PASS, else a bitmask of fails */
static int run_cell(const ntt_modulus_info_t *m, uint64_t n, uint64_t *seed,
                    int *prim_ok, int *dft_ok, int *rt_ok, int *conv_ok)
{
    uint64_t q = m->q;
    *prim_ok = *dft_ok = *rt_ok = *conv_ok = 0;

    /* PRIM: genuine primitive n-th root (the g=3-class regression) */
    uint64_t omega = ntt_modulus_omega(m, n);
    if (omega == 0 || omega == 1) return 1;             /* PRIM fail */
    if (rpow(omega, n, q) != 1) return 1;               /* omega^n != 1 */
    if (rpow(omega, n / 2, q) != q - 1) return 1;       /* omega^(n/2) != -1 */
    *prim_ok = 1;

    ntt_params_t p;
    memset(&p, 0, sizeof p);
    p.n = n; p.q = q; p.omega = omega;
    if (ntt_params_init(&p) != 0) return 2;
    uint64_t *tw  = ntt_alloc_twiddles(&p);
    uint64_t *twi = ntt_alloc_twiddles_inv(&p);
    if (!tw || !twi) { free(tw); free(twi); return 2; }

    uint64_t *x   = malloc((size_t)n * sizeof(uint64_t));
    uint64_t *y   = malloc((size_t)n * sizeof(uint64_t));
    uint64_t *Xr  = malloc((size_t)n * sizeof(uint64_t));
    uint64_t *a   = malloc((size_t)n * sizeof(uint64_t));
    uint64_t *b   = malloc((size_t)n * sizeof(uint64_t));
    uint64_t *fa  = malloc((size_t)n * sizeof(uint64_t));
    uint64_t *fb  = malloc((size_t)n * sizeof(uint64_t));
    uint64_t *cr  = malloc((size_t)n * sizeof(uint64_t));
    if (!x||!y||!Xr||!a||!b||!fa||!fb||!cr) {
        free(tw);free(twi);free(x);free(y);free(Xr);free(a);free(b);
        free(fa);free(fb);free(cr); return 2;
    }

    /* DFT: forward NTT vs independent O(n^2) DFT (non-constant input) */
    for (uint64_t i = 0; i < n; i++) x[i] = SM(seed) % q;
    if (all_equal(x, n)) x[0] = (x[0] + 1) % q;          /* ensure non-const */
    memcpy(y, x, (size_t)n * sizeof(uint64_t));
    ref_dft(x, Xr, n, omega, q);
    ntt_forward(y, tw, &p);
    int dft = vec_eq(y, Xr, n)
              && !vec_eq(y, x, n)        /* vacuity: transform must mix */
              && !all_equal(y, n);       /* vacuity: spectrum not constant */
    *dft_ok = dft;

    /* RT: INTT(NTT(.)) == . for structured + random inputs */
    int rt = 1;
    for (int t = 0; t < 5 && rt; t++) {
        for (uint64_t i = 0; i < n; i++) {
            switch (t) {
              case 0: x[i] = (i == 0);                       break; /* delta */
              case 1: x[i] = 1;                              break; /* ones  */
              case 2: x[i] = (i & 1) ? (q - 1) : 1;          break; /* alt   */
              case 3: x[i] = SM(seed) % q;                   break; /* rand  */
              default:x[i] = (uint64_t)((i * 2654435761u) % q);     /* ramp  */
            }
        }
        memcpy(y, x, (size_t)n * sizeof(uint64_t));
        ntt_forward(y, tw, &p);
        if (t >= 2 && (vec_eq(y, x, n) || all_equal(y, n)))
            rt = 0;                       /* vacuity: non-const must mix */
        ntt_inverse(y, twi, &p);
        if (!vec_eq(y, x, n)) rt = 0;
    }
    *rt_ok = rt;

    /* CONV: cyclic convolution via NTT vs independent schoolbook */
    for (uint64_t i = 0; i < n; i++) { a[i] = SM(seed) % q; b[i] = SM(seed) % q; }
    memcpy(fa, a, (size_t)n * sizeof(uint64_t));
    memcpy(fb, b, (size_t)n * sizeof(uint64_t));
    ntt_forward(fa, tw, &p);
    ntt_forward(fb, tw, &p);
    for (uint64_t i = 0; i < n; i++) fa[i] = p.reduce((unsigned __int128)fa[i] * fb[i], q);
    ntt_inverse(fa, twi, &p);
    ref_conv(a, b, cr, n, q);
    *conv_ok = vec_eq(fa, cr, n);

    free(tw);free(twi);free(x);free(y);free(Xr);free(a);free(b);
    free(fa);free(fb);free(cr);
    return (*prim_ok && *dft_ok && *rt_ok && *conv_ok) ? 0 : 1;
}

int main(int argc, char **argv)
{
    /* n sweep: 64 (log2=6 even), 128 (log2=7 odd — covers Stockham memcpy path),
     * 256 (log2=8 even), 1024 (log2=10 even) */
    uint64_t ns[] = { 64, 128, 256, 1024 };
    int nns = (int)(sizeof ns / sizeof ns[0]);
    if (argc > 1) { ns[0] = (uint64_t)strtoull(argv[1], 0, 10); nns = 1; }
    uint64_t seed = 0xA5A5C0DE12345678ULL;

    printf("\n  NTT / MI300A  —  RIGOROUS CPU NTT TEST (14-prime table)\n");
    printf("  %-12s %-10s %-5s  %-4s %-4s %-4s %-4s\n",
           "modulus", "q", "n", "PRIM", "DFT", "RT", "CONV");
    printf("  ---------------------------------------------------------\n");

    int total = 0, passed = 0, cells = 0, skipped = 0;
    for (int mi = 0; mi < NTT_NUM_MODULI; mi++) {
        const ntt_modulus_info_t *m = &NTT_MODULI[mi];
        for (int k = 0; k < nns; k++) {
            uint64_t n = ns[k];
            if (n < 4 || (n & (n - 1))) continue;
            if (ntt_modulus_omega(m, n) == 0) { skipped++; continue; }
            int pr, df, rt, cv;
            int rc = run_cell(m, n, &seed, &pr, &df, &rt, &cv);
            cells++;
            int cell_pass = (rc == 0);
            total += 4;
            passed += pr + df + rt + cv;
            printf("  %-12s %-10llu %-5llu  %-4s %-4s %-4s %-4s%s\n",
                   m->name, (unsigned long long)m->q, (unsigned long long)n,
                   pr?"OK":"FAIL", df?"OK":"FAIL", rt?"OK":"FAIL",
                   cv?"OK":"FAIL", cell_pass?"":"   <== CELL FAIL");
        }
    }
    printf("  ---------------------------------------------------------\n");
    printf("  cells=%d  skipped=%d  subtests %d/%d passed\n",
           cells, skipped, passed, total);
    int ok = (cells > 0 && passed == total);
    printf("  %s\n\n", ok ? "ALL PASS" : "FAILURES PRESENT");

    char ts[32]; time_t tt = time(0);
    strftime(ts, sizeof ts, "%Y%m%d_%H%M%S", localtime(&tt));
    char fn[64]; snprintf(fn, sizeof fn, "ntt_rigor_%s.txt", ts);
    FILE *f = fopen(fn, "w");
    if (f) {
        fprintf(f, "test_ntt_rigor %s : cells=%d skipped=%d %d/%d %s\n",
                ts, cells, skipped, passed, total, ok ? "ALL PASS" : "FAIL");
        fclose(f);
    }
    return ok ? 0 : 1;
}
