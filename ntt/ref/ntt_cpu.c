/*
 * ntt_cpu.c — Cooley-Tukey DIT NTT, CPU reference implementation
 *
 * Purpose:   Correctness-first, readable CPU baseline for the NTT/MI300A
 *            project. All GPU kernels are verified against this output.
 * Algorithm: Cooley-Tukey decimation-in-time (DIT), radix-2.
 * Reduction: lazy — multiplications reduce immediately via __uint128_t;
 *            additions accumulate to ≤ (log2(n)+1)·q, then one final pass
 *            reduces all values to [0, q).
 * Ref:       Longa & Naehrig, "Speeding up the Number Theoretic Transform
 *            for Faster Ideal Lattice-Based Cryptography," CANS 2016, Alg. 1.
 * Compiler:  gcc or clang, C99+; no HIP headers required.
 *   Build:   cc -O2 -Wall -Wextra -o ntt_cpu ntt_cpu.c
 */

#include "ntt.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ── ANSI formatting ─────────────────────────────────────────────────────── */
#define ANSI_WHT "\033[1;37m"
#define ANSI_CYN "\033[1;36m"
#define ANSI_GRN "\033[1;32m"
#define ANSI_YLW "\033[1;33m"
#define ANSI_RED "\033[1;31m"
#define ANSI_RST "\033[0m"

/* ── Hardware info struct ────────────────────────────────────────────────── */

/*
 * hw_info_t: system properties queried at startup via sysconf().
 * Printed as a formatted table before any computation.
 */
typedef struct {
    long     n_cpus;      /* logical CPUs online                             */
    long     page_sz;     /* system page size in bytes                       */
    long     phys_pages;  /* total physical memory pages                     */
    uint64_t mem_bytes;   /* total RAM in bytes (page_sz × phys_pages)       */
} hw_info_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * MODULAR ARITHMETIC
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * mod_pow: compute base^exp mod m by square-and-multiply.
 * Purpose:  twiddle factor generation and modular inverse via Fermat's thm.
 * Inputs:   base, exp, m — all fit in uint64_t; m > 0.
 * Output:   base^exp mod m in [0, m).
 * Ref:      Knuth, TAOCP Vol 2, §4.6.3.
 */
static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t m)
{
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1)
            result = (uint64_t)((__uint128_t)result * base % m);
        base = (uint64_t)((__uint128_t)base * base % m);
        exp >>= 1;
    }
    return result;
}

/*
 * mod_inv: modular inverse of a mod m via Fermat's little theorem.
 * Purpose:  compute omega_inv and n_inv for INTT setup.
 * Inputs:   a in (0, m), m prime.
 * Output:   a^{m-2} mod m = a^{-1} mod m.
 * Invariant: m must be prime; behaviour undefined if gcd(a,m) ≠ 1.
 */
static uint64_t mod_inv(uint64_t a, uint64_t m)
{
    return mod_pow(a, m - 2, m);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TWIDDLE FACTOR PRECOMPUTATION
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * ntt_alloc_twiddles: build the table tw[k] = root^k mod q for k in [0, n/2).
 * Purpose:  precompute all twiddle factors once; accessed as tw[j·stride]
 *           in the NTT inner loop (stride = n / current_stage_length).
 * Inputs:   p->omega is the primitive n-th root of unity mod p->q.
 * Output:   heap-allocated array of n/2 uint64_t values in [0, q).
 *           Returns NULL on allocation failure; caller must free().
 */
uint64_t *ntt_alloc_twiddles(const ntt_params_t *p)
{
    uint64_t half = p->n >> 1;
    uint64_t *tw  = malloc(half * sizeof *tw);
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < half; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k - 1] * p->omega % p->q);
    return tw;
}

/*
 * ntt_alloc_twiddles_inv: same as ntt_alloc_twiddles but using omega_inv.
 * Purpose:  twiddle table for ntt_inverse.
 * Output:   heap-allocated array of n/2 values; caller must free().
 */
uint64_t *ntt_alloc_twiddles_inv(const ntt_params_t *p)
{
    uint64_t half = p->n >> 1;
    uint64_t *tw  = malloc(half * sizeof *tw);
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < half; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k - 1] * p->omega_inv % p->q);
    return tw;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BIT-REVERSAL PERMUTATION
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * bit_reverse_perm: permute a[0..n-1] into bit-reversed index order in place.
 * Purpose:  required input reordering for Cooley-Tukey DIT.
 * Inputs:   a[] of length n (power of 2); log2_n = log2(n).
 * Output:   a[i] ↔ a[reverse_bits(i, log2_n)] for all i.
 * Algorithm: scan all indices; swap i with its bit-reversal if i < rev(i)
 *            to avoid double-swapping. O(n · log2(n)) bit operations.
 */
static void bit_reverse_perm(uint64_t *a, uint64_t n, uint32_t log2_n)
{
    for (uint64_t i = 0; i < n; i++) {
        uint64_t rev = 0, x = i;
        for (uint32_t b = 0; b < log2_n; b++) {
            rev = (rev << 1) | (x & 1);
            x >>= 1;
        }
        if (i < rev) {
            uint64_t tmp = a[i];
            a[i] = a[rev];
            a[rev] = tmp;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CORE NTT
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * ct_dit_loop: CT-DIT butterfly stages (shared by ntt_forward and ntt_inverse).
 * Applies log2(n) butterfly stages after bit-reversal. Uses the caller-supplied
 * reduce function for the multiply-reduce step; addmod/submod keep values in
 * [0, q) after every butterfly, making the code safe for Goldilocks (q ≈ 2^64)
 * where lazy accumulation (u + q in uint64_t) would overflow.
 * Ref: Longa & Naehrig CANS 2016, Alg. 1.
 */
static void ct_dit_loop(uint64_t *a, const uint64_t * restrict tw,
                        uint64_t n, uint64_t q, uint32_t log2_n,
                        reduce_fn_t reduce)
{
    bit_reverse_perm(a, n, log2_n);
    for (uint64_t len = 2; len <= n; len <<= 1) {
        uint64_t half = len >> 1, stride = n / len;
        for (uint64_t k = 0; k < n; k += len)
            for (uint64_t j = 0; j < half; j++) {
                uint64_t u = a[k + j];
                uint64_t t = reduce((__uint128_t)tw[j*stride] * a[k+j+half], q);
                a[k + j]        = addmod(u, t, q);
                a[k + j + half] = submod(u, t, q);
            }
    }
}

/*
 * ntt_forward: in-place CT-DIT NTT over Z_q.
 * Input:  a[0..n-1] in natural order, values in [0, q).
 * Output: NTT(a), values in [0, q).
 * twiddles: ntt_alloc_twiddles(p) — tw[k] = omega^k.
 * Uses p->reduce for the butterfly multiply-reduce; addmod/submod keep all
 * intermediate values in [0, q), so no final reduction pass is needed.
 */
void ntt_forward(uint64_t *a, const uint64_t * restrict twiddles,
                 const ntt_params_t *p)
{
    ct_dit_loop(a, twiddles, p->n, p->q, p->log2_n, p->reduce);
}

/*
 * ntt_inverse: in-place INTT over Z_q.
 * Input:  a[0..n-1] in NTT domain, values in [0, q).
 * Output: INTT(a) = n^{-1} * NTT^{-1}(a), values in [0, q).
 * twiddles_inv: ntt_alloc_twiddles_inv(p) — tw[k] = omega_inv^k.
 */
void ntt_inverse(uint64_t *a, const uint64_t * restrict twiddles_inv,
                 const ntt_params_t *p)
{
    ct_dit_loop(a, twiddles_inv, p->n, p->q, p->log2_n, p->reduce);
    /* Scale by n^{-1}: values from ct_dit_loop are already in [0, q). */
    for (uint64_t i = 0; i < p->n; i++)
        a[i] = p->reduce((__uint128_t)a[i] * p->n_inv, p->q);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PARAMETER INITIALISATION
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * ntt_params_init: fill derived fields of p given n, q, omega.
 * Purpose:  compute omega_inv, n_inv, log2_n from the three primary inputs.
 * Inputs:   p->n, p->q, p->omega must be set by the caller.
 * Output:   p->omega_inv, p->n_inv, p->log2_n filled in.
 * Returns:  0 on success; -1 if n is not a power of 2.
 */
int ntt_params_init(ntt_params_t *p)
{
    if (p->q < 2) return -1;                                 /* q=0/1 degenerate */
    if (p->n == 0 || (p->n & (p->n - 1)) != 0) return -1;  /* not power of 2 */
    p->omega_inv = mod_inv(p->omega, p->q);
    p->n_inv     = mod_inv(p->n,     p->q);
    uint64_t tmp = p->n;
    p->log2_n    = 0;
    while (tmp > 1) { tmp >>= 1; p->log2_n++; }
    /* Wire up fast reduction: look up q in NTT_MODULI; fall back to generic. */
    const ntt_modulus_info_t *mi = ntt_modulus_find(p->q);
    p->reduce = mi ? mi->reduce : reduce_generic;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DISPLAY HELPERS / SELFTEST / BENCHMARK / MAIN
 * compiled only when ntt_cpu.c is the top-level translation unit
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef NTT_CPU_NO_MAIN

static void print_hw_summary(const hw_info_t *hw)
{
    printf("\n" ANSI_CYN "  ── System Information ──────────────────────────────────\n" ANSI_RST);
    printf("  ┌──────────────────────────┬─────────────────────────┐\n");
    printf("  │ " ANSI_CYN "%-24s" ANSI_RST " │ " ANSI_CYN "%-23s" ANSI_RST " │\n",
           "Property", "Value");
    printf("  ├──────────────────────────┼─────────────────────────┤\n");
    printf("  │ %-24s │ %-23ld │\n", "Logical CPUs",    hw->n_cpus);
    printf("  │ %-24s │ %-23ld │\n", "Page size (B)",   hw->page_sz);
    printf("  │ %-24s │ %-.2f %-18s │\n", "Total RAM",
           hw->mem_bytes / (1024.0 * 1024.0 * 1024.0), "GB");
    printf("  └──────────────────────────┴─────────────────────────┘\n\n");
}

static void print_params(const ntt_params_t *p)
{
    printf(ANSI_CYN "  ── NTT Parameters ──────────────────────────────────────\n" ANSI_RST);
    printf("  ┌──────────────────────────┬─────────────────────────┐\n");
    printf("  │ " ANSI_CYN "%-24s" ANSI_RST " │ " ANSI_CYN "%-23s" ANSI_RST " │\n",
           "Parameter", "Value");
    printf("  ├──────────────────────────┼─────────────────────────┤\n");
    printf("  │ %-24s │ %-23lu │\n", "n",         p->n);
    printf("  │ %-24s │ %-23lu │\n", "q",         p->q);
    printf("  │ %-24s │ %-23lu │\n", "omega",     p->omega);
    printf("  │ %-24s │ %-23lu │\n", "omega_inv", p->omega_inv);
    printf("  │ %-24s │ %-23lu │\n", "n_inv",     p->n_inv);
    printf("  │ %-24s │ %-23u │\n", "log2(n)",    p->log2_n);
    printf("  └──────────────────────────┴─────────────────────────┘\n\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CORRECTNESS TEST
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * selftest: round-trip NTT → INTT and verify recovery of the original array.
 * Purpose:  smoke test for ntt_forward and ntt_inverse.
 * Inputs:   p — transform parameters; n=256, q=3329, omega=17 for ML-KEM.
 * Output:   prints PASS/FAIL; returns 0 on success, nonzero on failure.
 * Invariant: INTT(NTT(a)) = a element-wise mod q for any a in [0, q)^n.
 */
static int selftest(const ntt_params_t *p)
{
    uint64_t n = p->n;
    uint64_t *orig = malloc(n * sizeof *orig);
    uint64_t *a    = malloc(n * sizeof *a);
    uint64_t *tw   = ntt_alloc_twiddles(p);
    uint64_t *twi  = ntt_alloc_twiddles_inv(p);
    int ok = 0;

    if (!orig || !a || !tw || !twi) {
        fprintf(stderr, "selftest: allocation failed\n");
        ok = -1;
        goto done;
    }

    /* Fill with pseudo-random values in [0, q) */
    for (uint64_t i = 0; i < n; i++)
        orig[i] = a[i] = (uint64_t)(i * 1234567891ULL + 42) % p->q;

    ntt_forward(a, tw, p);
    ntt_inverse(a, twi, p);

    for (uint64_t i = 0; i < n; i++) {
        if (a[i] != orig[i]) {
            printf("  " ANSI_RED "FAIL" ANSI_RST
                   " at index %lu: expected %lu got %lu\n", i, orig[i], a[i]);
            ok = -1;
            goto done;
        }
    }

done:
    free(orig); free(a); free(tw); free(twi);
    return ok;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * run_benchmark: time ntt_forward over many iterations; report NTT/s.
 * Purpose:  establish a CPU baseline throughput measurement.
 * Inputs:   p — transform parameters; iters — number of forward NTTs to run.
 * Output:   prints results table to stdout; writes same to a timestamped file.
 */
static void run_benchmark(const ntt_params_t *p, uint64_t iters)
{
    uint64_t n  = p->n;
    uint64_t *a = malloc(n * sizeof *a);
    uint64_t *tw = ntt_alloc_twiddles(p);
    if (!a || !tw) { fprintf(stderr, "bench: alloc failed\n"); free(a); free(tw); return; }

    for (uint64_t i = 0; i < n; i++)
        a[i] = i % p->q;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++)
        ntt_forward(a, tw, p);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double ntts_per_sec = (double)iters / elapsed;
    double ns_per_ntt   = elapsed * 1e9 / (double)iters;
    double butterflies  = (double)n / 2.0 * (double)p->log2_n;
    double ns_per_bf    = ns_per_ntt / butterflies;

    printf(ANSI_CYN "  ── Benchmark: ntt_forward (CPU) ────────────────────────\n" ANSI_RST);
    printf("  ┌────────────────────┬──────────┬────────────┬────────────────┐\n");
    printf("  │ " ANSI_CYN "%-18s" ANSI_RST " │ " ANSI_CYN "%-8s" ANSI_RST
           " │ " ANSI_CYN "%-10s" ANSI_RST " │ " ANSI_CYN "%-14s" ANSI_RST " │\n",
           "Metric", "n", "Iters", "Value");
    printf("  ├────────────────────┼──────────┼────────────┼────────────────┤\n");
    printf("  │ %-18s │ %-8lu │ %-10lu │ %11.0f /s │\n",
           "NTT throughput", n, iters, ntts_per_sec);
    printf("  │ %-18s │ %-8lu │ %-10lu │ %11.1f ns  │\n",
           "ns / NTT", n, iters, ns_per_ntt);
    printf("  │ %-18s │ %-8lu │ %-10lu │ %11.2f ns  │\n",
           "ns / butterfly", n, iters, ns_per_bf);
    printf("  └────────────────────┴──────────┴────────────┴────────────────┘\n\n");

    /* Write same results to a timestamped file for later comparison */
    char fname[64];
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    strftime(fname, sizeof fname, "bench_%Y%m%d_%H%M%S.txt", tm);
    FILE *f = fopen(fname, "w");
    if (f) {
        fprintf(f, "kernel=ntt_forward_cpu n=%lu q=%lu omega=%lu iters=%lu\n",
                n, p->q, p->omega, iters);
        fprintf(f, "elapsed_s=%.6f ntts_per_s=%.0f ns_per_ntt=%.1f ns_per_butterfly=%.2f\n",
                elapsed, ntts_per_sec, ns_per_ntt, ns_per_bf);
        fclose(f);
        printf("  Results written to %s\n\n", fname);
    }

    free(a); free(tw);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * main: parse args, print system summary, run selftest, run benchmark.
 * Usage:  ntt_cpu [n] [q] [omega] [iters]
 *   n:     transform size (default 256, ML-KEM)
 *   q:     prime modulus  (default 3329, ML-KEM)
 *   omega: primitive root (default 17, ML-KEM n=256)
 *   iters: benchmark iterations (default 100000)
 * All NTT-friendly defaults correspond to FIPS 203 (ML-KEM) parameters.
 */
int main(int argc, char *argv[])
{
    ntt_params_t p;
    p.n     = (argc > 1) ? (uint64_t)atol(argv[1]) : 256;
    p.q     = (argc > 2) ? (uint64_t)atol(argv[2]) : 3329;
    p.omega = (argc > 3) ? (uint64_t)atol(argv[3]) : 17;
    uint64_t iters = (argc > 4) ? (uint64_t)atol(argv[4]) : 100000;

    printf("\n" ANSI_WHT
           "╔══════════════════════════════════════════════════════════╗\n"
           "║          NTT / MI300A  —  CPU Reference Kernel          ║\n"
           "╚══════════════════════════════════════════════════════════╝\n"
           ANSI_RST "\n");

    /* Hardware summary */
    hw_info_t hw;
    hw.n_cpus     = sysconf(_SC_NPROCESSORS_ONLN);
    hw.page_sz    = sysconf(_SC_PAGE_SIZE);
    hw.phys_pages = sysconf(_SC_PHYS_PAGES);
    hw.mem_bytes  = (uint64_t)hw.phys_pages * (uint64_t)hw.page_sz;
    print_hw_summary(&hw);

    /* Parameter initialisation */
    if (ntt_params_init(&p) != 0) {
        fprintf(stderr, "error: n=%lu is not a power of 2\n", p.n);
        return 1;
    }
    print_params(&p);

    /* Correctness test */
    printf(ANSI_CYN "  ── Selftest ─────────────────────────────────────────────\n" ANSI_RST);
    int result = selftest(&p);
    if (result == 0)
        printf("  " ANSI_GRN "PASS" ANSI_RST
               "  INTT(NTT(a)) == a  for n=%lu, q=%lu, omega=%lu\n\n",
               p.n, p.q, p.omega);
    else
        printf("  " ANSI_RED "FAIL" ANSI_RST "  see above\n\n");

    if (result != 0) return 1;

    /* Benchmark */
    run_benchmark(&p, iters);

    return 0;
}
#endif /* NTT_CPU_NO_MAIN */
