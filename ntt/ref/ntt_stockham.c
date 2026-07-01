/*
 * ntt_stockham.c — Stockham auto-sort NTT, CPU implementation
 *
 * Purpose:   GPU-preferred NTT variant that eliminates the bit-reversal
 *            permutation pass required by Cooley-Tukey DIT. Double-buffer
 *            ping-pong structure maps cleanly to GPU shared-memory staging.
 * Algorithm: Stockham DIT (decimation-in-time), radix-2.
 *            Double-buffer ping-pong: each stage reads src[j+k*p] (first half)
 *            and src[j+k*p+n/2] (second half), writes dst[j+k*(2p)] and
 *            dst[j+k*(2p)+p]. Natural-order output — no bit-reversal needed.
 * Reduction: every butterfly fully reduces — twiddle multiply via the
 *            per-modulus reduce fn, add/sub via addmod/submod — values
 *            stay in [0, q) at every stage (no lazy accumulation, no
 *            final pass; required as some moduli are near 2^64).
 * Ref:       Gentleman & Sande, "Fast Fourier Transforms for Fun and Profit,"
 *            AFIPS 1966 FJCC. Stockham, "High-Speed Convolution and
 *            Correlation," AFIPS 1966 SJCC.
 * Compiler:  gcc or clang, C99+; no HIP headers required.
 *   Build:   cc -O2 -Wall -Wextra -o ntt_stockham ntt_stockham.c
 */

#include "ntt.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ── ANSI formatting ─────────────────────────────────────────────────────── */
#define ANSI_WHT "\033[1;37m"
#define ANSI_CYN "\033[1;36m"
#define ANSI_GRN "\033[1;32m"
#define ANSI_YLW "\033[1;33m"
#define ANSI_RED "\033[1;31m"
#define ANSI_RST "\033[0m"

/* ═══════════════════════════════════════════════════════════════════════════
 * MODULAR ARITHMETIC  (standalone copies — no link dependency on ntt_cpu.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t m)
{
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = (uint64_t)((__uint128_t)result * base % m);
        base = (uint64_t)((__uint128_t)base * base % m);
        exp >>= 1;
    }
    return result;
}

static uint64_t mod_inv(uint64_t a, uint64_t m)
{
    return mod_pow(a, m - 2, m);   /* Fermat: m must be prime */
}

/* ntt_params_init: compute derived fields from p->n, p->q, p->omega. */
int ntt_params_init(ntt_params_t *p)
{
    if (p->n == 0 || (p->n & (p->n - 1)) != 0) return -1;
    /* q>=2: q=0 would SIGFPE in mod_inv->mod_pow (base %= 0); q=1 is degenerate.
     * Same footgun class as G4b GPU-host params_init + ref/ntt_polymul guards. */
    if (p->q < 2) return -1;
    p->omega_inv = mod_inv(p->omega, p->q);
    p->n_inv     = mod_inv(p->n,     p->q);
    uint64_t tmp = p->n;
    p->log2_n    = 0;
    while (tmp > 1) { tmp >>= 1; p->log2_n++; }
    const ntt_modulus_info_t *mi = ntt_modulus_find(p->q);
    p->reduce = mi ? mi->reduce : reduce_generic;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TWIDDLE TABLES
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Stockham accesses tw[j * t] where t = n/2^(s+1), j in [0, n/2^(s+1)).
 * Maximum index: s = log2(n)-1, t=1, j up to n/2-1 → tw[n/2-1].
 * Identical layout to the CT-DIT twiddle table: tw[k] = omega^k, k in [0, n/2).
 */
uint64_t *ntt_alloc_twiddles(const ntt_params_t *p)
{
    uint64_t *tw = (uint64_t *)malloc(p->n / 2 * sizeof(uint64_t));
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < p->n / 2; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k - 1] * p->omega % p->q);
    return tw;
}

uint64_t *ntt_alloc_twiddles_inv(const ntt_params_t *p)
{
    uint64_t *tw = (uint64_t *)malloc(p->n / 2 * sizeof(uint64_t));
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < p->n / 2; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k - 1] * p->omega_inv % p->q);
    return tw;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * STOCKHAM CORE
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * stockham_core: in-place Stockham DIT NTT using double-buffer ping-pong.
 *
 * Each stage s reads from src[0..n/2) (u) and src[n/2..n) (v), writes to dst,
 * then swaps src/dst. This auto-sort index structure produces natural-order
 * output from natural-order input — no bit-reversal needed.
 *
 * At stage s:
 *   p  = 2^s             stride in source (= number of groups at this stage)
 *   qs = n / 2^(s+1)     elements per group (inner loop bound)
 *
 *   For j in [0, p), k in [0, qs):
 *     u  = src[j + k*p]               (always in first half:  index < n/2)
 *     v  = src[j + k*p + n/2]         (always in second half: index ≥ n/2)
 *     w  = tw[j*qs] = omega^(j*qs)    twiddle for this (j, s) group
 *   Output:
 *     dst[j + k*(2p)]     = u + w*v   (lazy — wv reduced, u+wv accumulates)
 *     dst[j + k*(2p) + p] = u - w*v + q
 *
 * Invariant: after every stage, src[0..n/2) holds the "positive" butterfly
 * outputs and src[n/2..n) holds the "negative" outputs. This ensures the next
 * stage always reads u from the first half and v from the second half.
 *
 * Final reduction pass: reduce all values from [0, (L+1)*q) to [0, q).
 *
 * Parameters:
 *   a, b   — two buffers of size n; a is the input
 *   tw     — twiddle table: tw[k] = omega^k for k in [0, n/2)
 *   n, q   — transform size and modulus
 *   log2_n — log2(n)
 *
 * Returns pointer to the buffer holding the result (either a or b).
 */
static uint64_t *stockham_core(uint64_t * restrict a, uint64_t * restrict b,
                                const uint64_t * restrict tw,
                                uint64_t n, uint64_t q, uint32_t log2_n,
                                reduce_fn_t reduce)
{
    uint64_t *src  = a;
    uint64_t *dst  = b;
    uint64_t half  = n >> 1;

    for (uint32_t s = 0; s < log2_n; s++) {
        uint64_t p  = (uint64_t)1 << s;    /* stride in source = 2^s      */
        uint64_t qs = n >> (s + 1);        /* elements per group           */
        uint64_t p2 = p << 1;              /* 2*p: stride in destination   */

        for (uint64_t j = 0; j < p; j++) {
            uint64_t w = tw[j * qs];       /* omega^(j * qs)                */
            for (uint64_t k = 0; k < qs; k++) {
                uint64_t idx = j + k * p;
                uint64_t u   = src[idx];
                /* v is in the second half of src; apply fast reduce then    *
                 * addmod/submod to stay in [0, q) — safe for Goldilocks.    */
                uint64_t wv  = reduce((__uint128_t)w * src[idx + half], q);
                dst[j + k * p2]      = addmod(u, wv, q);
                dst[j + k * p2 + p]  = submod(u, wv, q);
            }
        }

        /* Swap buffers for the next stage. */
        uint64_t *tmp = src; src = dst; dst = tmp;
    }

    /* addmod/submod in the butterfly keeps all values in [0, q) throughout;
     * no final reduction pass is needed. */
    return src;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC API  (matches ntt.h — drop-in replacement for ntt_forward/inverse)
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * ntt_forward: forward Stockham NTT, in-place.
 * Allocates a scratch buffer internally; result overwrites 'a'.
 * Same signature as ntt_cpu.c ntt_forward.
 */
void ntt_forward(uint64_t *a, const uint64_t *twiddles, const ntt_params_t *p)
{
    uint64_t *scratch = (uint64_t *)malloc(p->n * sizeof(uint64_t));
    if (!scratch) { fprintf(stderr, "ntt_forward: malloc failed\n"); return; }

    uint64_t *result = stockham_core(a, scratch, twiddles, p->n, p->q, p->log2_n, p->reduce);

    if (result != a) {
        /* Result ended up in scratch; copy back to a. */
        memcpy(a, result, p->n * sizeof(uint64_t));
    }
    free(scratch);
}

/*
 * ntt_inverse: inverse Stockham NTT (INTT), in-place.
 * Runs forward Stockham with omega_inv twiddles, then scales by n_inv.
 */
void ntt_inverse(uint64_t *a, const uint64_t *twiddles_inv, const ntt_params_t *p)
{
    ntt_forward(a, twiddles_inv, p);

    for (uint64_t i = 0; i < p->n; i++)
        a[i] = p->reduce((__uint128_t)a[i] * p->n_inv, p->q);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * REFERENCE CT-DIT  (static — used only for selftest cross-check)
 *
 * selftest/run_benchmark and the static reference helpers below are only
 * reachable from this file's own main(). They are compiled out together
 * with main() when NTT_STOCKHAM_NO_MAIN is defined so external drivers that
 * link this file as the NTT implementation (e.g. test_ntt_rigor.c via the
 * test-ntt-rigor-stok target) do not trip -Wunused-function under
 * -Wall -Wextra. Mirrors the NTT_CPU_NO_MAIN guard in ntt_cpu.c.
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef NTT_STOCKHAM_NO_MAIN

/*
 * ref_bit_reverse_perm: bit-reverse permutation, same as ntt_cpu.c.
 * Used only in the selftest to generate the CT-DIT reference output.
 */
static void ref_bit_reverse_perm(uint64_t *a, uint64_t n, uint32_t log2_n)
{
    for (uint64_t i = 0; i < n; i++) {
        uint64_t rev = 0, x = i;
        for (uint32_t b = 0; b < log2_n; b++) { rev = (rev << 1) | (x & 1); x >>= 1; }
        if (i < rev) { uint64_t tmp = a[i]; a[i] = a[rev]; a[rev] = tmp; }
    }
}

/*
 * ref_ct_forward: CT-DIT NTT (reference, static copy).
 * Writes output into the provided array, which is overwritten in place.
 */
static void ref_ct_forward(uint64_t *a, const uint64_t *tw, const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    ref_bit_reverse_perm(a, n, p->log2_n);
    for (uint64_t len = 1; len < n; len <<= 1) {
        uint64_t step = n / (len << 1);
        for (uint64_t i = 0; i < n; i += len << 1) {
            for (uint64_t j = 0; j < len; j++) {
                uint64_t u = a[i + j];
                uint64_t v = p->reduce((__uint128_t)tw[j * step] * a[i + j + len], q);
                a[i + j]       = addmod(u, v, q);
                a[i + j + len] = submod(u, v, q);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SELFTEST
 * ═══════════════════════════════════════════════════════════════════════════ */

static int selftest(const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    const char *label = (q == 3329) ? "ML-KEM" : (q == 8380417) ? "ML-DSA" : "custom";

    printf(ANSI_CYN "  ── Selftest: %s (n=%lu q=%lu ω=%lu) ──────────────────\n"
           ANSI_RST, label, n, q, p->omega);

    uint64_t *tw  = ntt_alloc_twiddles(p);
    uint64_t *twi = ntt_alloc_twiddles_inv(p);
    uint64_t *a   = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *ref = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tw || !twi || !a || !ref) { free(tw); free(twi); free(a); free(ref); return -1; }

    /* Fill with a pseudo-random pattern. */
    for (uint64_t i = 0; i < n; i++) a[i] = ref[i] = (i * 6364136223846793005ULL + 1) % q;

    /* ── Test 1: Stockham round-trip (forward → inverse = identity) ──── */
    ntt_forward(a, tw, p);
    ntt_inverse(a, twi, p);
    int rt_ok = 1;
    for (uint64_t i = 0; i < n; i++) if (a[i] != ref[i]) { rt_ok = 0; break; }
    printf("  %-36s %s\n", "round-trip (fwd→inv = identity)",
           rt_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    /* ── Test 2: Stockham vs CT-DIT output element-wise ─────────────── */
    /* Re-fill, then run both transforms. */
    for (uint64_t i = 0; i < n; i++) a[i] = ref[i] = (i * 6364136223846793005ULL + 1) % q;
    ntt_forward(a,   tw, p);       /* Stockham output */
    ref_ct_forward(ref, tw, p);    /* CT-DIT output   */

    int cmp_ok = 1;
    for (uint64_t i = 0; i < n; i++) if (a[i] != ref[i]) { cmp_ok = 0; break; }
    printf("  %-36s %s\n", "vs CT-DIT reference (element-wise)",
           cmp_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    /* ── Test 3: unit impulse → all-ones ──────────────────────────────── */
    /* NTT of [1, 0, 0, ...] = [1, 1, ..., 1] because omega^(k*0)=1. */
    a[0] = 1; for (uint64_t i = 1; i < n; i++) a[i] = 0;
    ntt_forward(a, tw, p);
    int imp_ok = 1;
    for (uint64_t i = 0; i < n; i++) if (a[i] != 1) { imp_ok = 0; break; }
    printf("  %-36s %s\n", "impulse [1,0,...,0] → all-ones",
           imp_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    free(tw); free(twi); free(a); free(ref);

    int ok = rt_ok && cmp_ok && imp_ok;
    printf("  %s  n=%-4lu q=%lu\n\n",
           ok ? ANSI_GRN "ALL PASS" ANSI_RST : ANSI_RED "FAIL"  ANSI_RST, n, q);
    return ok ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK
 * ═══════════════════════════════════════════════════════════════════════════ */

static void run_benchmark(const ntt_params_t *p, uint64_t iters)
{
    uint64_t n = p->n, q = p->q;
    uint64_t *tw = ntt_alloc_twiddles(p);
    uint64_t *a  = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tw || !a) { free(tw); free(a); return; }
    for (uint64_t i = 0; i < n; i++) a[i] = (i * 31337 + 1) % q;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) ntt_forward(a, tw, p);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double ntts_per_s = iters / elapsed;
    double ns_per_butterfly = elapsed * 1e9 / ((double)iters * n * p->log2_n / 2.0);

    printf(ANSI_CYN "  ── Benchmark: ntt_forward (Stockham) ───────────────────\n" ANSI_RST);
    printf("  ┌──────────────────────────┬─────────────────────────┐\n");
    printf("  │ " ANSI_CYN "%-24s" ANSI_RST " │ " ANSI_CYN "%-23s" ANSI_RST " │\n", "Parameter", "Value");
    printf("  ├──────────────────────────┼─────────────────────────┤\n");
    printf("  │ %-24s │ %-23lu │\n", "n",          n);
    printf("  │ %-24s │ %-23lu │\n", "q",          q);
    printf("  │ %-24s │ %-23lu │\n", "iterations", iters);
    printf("  │ %-24s │ %-23.6f │\n", "elapsed (s)",  elapsed);
    printf("  │ %-24s │ %-23.0f │\n", "NTT/s",        ntts_per_s);
    printf("  │ %-24s │ %-23.2f │\n", "ns/butterfly", ns_per_butterfly);
    printf("  └──────────────────────────┴─────────────────────────┘\n\n");

    free(tw); free(a);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    printf(ANSI_WHT
           "╔══════════════════════════════════════════════════════════╗\n"
           "║         NTT / Stockham auto-sort — CPU reference        ║\n"
           "╚══════════════════════════════════════════════════════════╝\n"
           ANSI_RST "\n");

    ntt_params_t p;
    p.n     = (argc > 1) ? (uint64_t)atoll(argv[1]) : 256;
    p.q     = (argc > 2) ? (uint64_t)atoll(argv[2]) : 3329;
    p.omega = (argc > 3) ? (uint64_t)atoll(argv[3]) : 17;
    uint64_t iters = (argc > 4) ? (uint64_t)atoll(argv[4]) : 100000;

    if (ntt_params_init(&p) != 0) {
        fprintf(stderr, "ntt_params_init failed — n must be a power of 2\n");
        return 1;
    }

    printf(ANSI_CYN "  ── NTT Parameters ──────────────────────────────────────\n" ANSI_RST);
    printf("  n=%-6lu q=%-10lu omega=%-10lu log2_n=%u\n\n", p.n, p.q, p.omega, p.log2_n);

    /* Selftests for both standard parameter sets. */
    int fail = 0;
    ntt_params_t mlkem = { 256, 3329, 17, 0, 0, 0, NULL };
    ntt_params_t mldsa = { 256, 8380417, 3073009, 0, 0, 0, NULL };
    if (ntt_params_init(&mlkem) == 0) fail |= selftest(&mlkem);
    if (ntt_params_init(&mldsa) == 0) fail |= selftest(&mldsa);

    /* Also run selftest for the user-supplied parameters if different. */
    if (p.n != 256 || (p.q != 3329 && p.q != 8380417)) {
        fail |= selftest(&p);
    }

    if (iters > 1) run_benchmark(&p, iters);

    return (fail != 0) ? 1 : 0;
}
#endif /* NTT_STOCKHAM_NO_MAIN */
