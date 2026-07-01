/*
 * ntt_bench.c — Side-by-side CPU algorithm benchmark across transform sizes
 *
 * Purpose:   Compare CT-DIT (lazy), Stockham (lazy), and Montgomery NTT
 *            across all 14 moduli at min(max_n, 1024)
 *            parameter sets. Output a formatted table for analysis.
 * Algorithm: All three algorithms implement the same ntt_forward() signature;
 *            this file compiles all three inline as static functions with
 *            distinct prefixes (ct_, stk_, mnt_) so there are no link conflicts.
 * Build:     cc -O2 -Wall -Wextra -o ntt_bench ntt_bench.c
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
#define ANSI_RST "\033[0m"

/* ═══════════════════════════════════════════════════════════════════════════
 * SHARED HELPERS
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
static uint64_t h_mod_inv(uint64_t a, uint64_t m) { return h_mod_pow(a, m-2, m); }

/* shared params_init (no link conflict — ntt.h is included, but we provide impl) */
int ntt_params_init(ntt_params_t *p)
{
    if (!p->n || (p->n & (p->n-1))) return -1;
    /* q>=2: q=0 SIGFPEs in h_mod_inv->mod_pow (base %= 0); q=1 degenerate.
     * Same SIGFPE footgun class generalized across all ref ntt_params_init. */
    if (p->q < 2) return -1;
    p->omega_inv = h_mod_inv(p->omega, p->q);
    p->n_inv     = h_mod_inv(p->n,     p->q);
    uint64_t t = p->n; p->log2_n = 0;
    while (t > 1) { t >>= 1; p->log2_n++; }
    const ntt_modulus_info_t *mi = ntt_modulus_find(p->q);
    p->reduce = mi ? mi->reduce : reduce_generic;
    return 0;
}

/* twiddle alloc shared */
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

/* ═══════════════════════════════════════════════════════════════════════════
 * ALGORITHMS + TIMING  (compiled only when ntt_bench.c is top-level TU)
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef NTT_BENCH_NO_MAIN

static void ct_bit_rev(uint64_t *a, uint64_t n, uint32_t log2_n)
{
    for (uint64_t i = 0; i < n; i++) {
        uint64_t rev = 0, x = i;
        for (uint32_t b = 0; b < log2_n; b++) { rev = (rev<<1)|(x&1); x>>=1; }
        if (i < rev) { uint64_t t = a[i]; a[i] = a[rev]; a[rev] = t; }
    }
}

static void ct_ntt(uint64_t *a, const uint64_t *tw, const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    ct_bit_rev(a, n, p->log2_n);
    for (uint64_t len = 1; len < n; len <<= 1) {
        uint64_t step = n / (len << 1);
        for (uint64_t i = 0; i < n; i += len << 1)
            for (uint64_t j = 0; j < len; j++) {
                uint64_t u = a[i+j];
                uint64_t v = p->reduce((__uint128_t)tw[j*step] * a[i+j+len], q);
                a[i+j]     = addmod(u, v, q);
                a[i+j+len] = submod(u, v, q);
            }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ALGORITHM 2: Stockham auto-sort (lazy reduction)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void stk_ntt(uint64_t *a, const uint64_t *tw, const ntt_params_t *p)
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
                uint64_t wv  = p->reduce((__uint128_t)w * src[idx+half], q);
                dst[j+k*p2]    = addmod(u, wv, q);
                dst[j+k*p2+pp] = submod(u, wv, q);
            }
        }
        uint64_t *t = src; src = dst; dst = t;
    }
    if (src != a) memcpy(a, src, n * sizeof(uint64_t));
    free(buf);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ALGORITHM 3: Montgomery multiplication
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct { uint64_t q, r2; uint32_t q_prime; } mont_ctx_t;

static mont_ctx_t mont_init(uint64_t q)
{
    mont_ctx_t c;
    c.q = q;
    /* q_prime: q * q_prime ≡ -1 (mod 2^32) via Hensel lifting */
    uint32_t x = 1;
    for (int i = 0; i < 5; i++) x *= 2 - (uint32_t)q * x;
    c.q_prime = (uint32_t)(-(int32_t)x);       /* q_prime = -q^{-1} mod 2^32 */
    /* R^2 mod q, R = 2^32 */
    c.r2 = (uint64_t)(((__uint128_t)1 << 64) % q);
    return c;
}

static uint64_t mont_mul(uint64_t a, uint64_t b, const mont_ctx_t *c)
{
    uint64_t t    = a * b;
    uint32_t mp   = (uint32_t)t * c->q_prime;
    uint64_t corr = (uint64_t)mp * c->q;
    uint64_t lo   = t + corr;
    uint64_t carry = (lo < t) ? 1ULL : 0ULL;
    uint64_t u    = (lo >> 32) | (carry << 32);
    return u >= c->q ? u - c->q : u;
}

static uint64_t mont_enter(uint64_t a, const mont_ctx_t *c)
{ return mont_mul(a, c->r2, c); }
static uint64_t mont_exit(uint64_t a, const mont_ctx_t *c)
{ return mont_mul(a, 1ULL, c); }

static void mnt_ntt(uint64_t *a, const uint64_t *tw_std, const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    mont_ctx_t mc = mont_init(q);

    /* Build Montgomery twiddle table */
    uint64_t *tw_m = (uint64_t *)malloc((n/2) * sizeof(uint64_t));
    if (!tw_m) return;
    uint64_t omega_m = mont_enter(p->omega, &mc);
    tw_m[0] = mont_enter(1ULL, &mc);
    for (uint64_t k = 1; k < n/2; k++)
        tw_m[k] = mont_mul(tw_m[k-1], omega_m, &mc);

    /* Convert input to Montgomery domain */
    for (uint64_t i = 0; i < n; i++) a[i] = mont_enter(a[i], &mc);

    /* CT-DIT with Montgomery butterflies */
    ct_bit_rev(a, n, p->log2_n);
    for (uint64_t len = 1; len < n; len <<= 1) {
        uint64_t step = n / (len << 1);
        for (uint64_t i = 0; i < n; i += len << 1)
            for (uint64_t j = 0; j < len; j++) {
                uint64_t u = a[i+j];
                uint64_t v = mont_mul(tw_m[j*step], a[i+j+len], &mc);
                uint64_t s = u + v; if (s >= q) s -= q;
                uint64_t d = u - v + q; if (d >= q) d -= q;
                a[i+j]     = s;
                a[i+j+len] = d;
            }
    }

    /* Convert back from Montgomery domain */
    for (uint64_t i = 0; i < n; i++) a[i] = mont_exit(a[i], &mc);
    free(tw_m);
    (void)tw_std;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TIMING HELPER
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef void (*ntt_fn)(uint64_t *, const uint64_t *, const ntt_params_t *);

static double time_ntt(ntt_fn fn, const ntt_params_t *p, uint64_t iters)
{
    uint64_t *tw = ntt_alloc_twiddles(p);
    uint64_t *a  = (uint64_t *)malloc(p->n * sizeof(uint64_t));
    if (!tw || !a) { free(tw); free(a); return -1.0; }
    for (uint64_t i = 0; i < p->n; i++) a[i] = (i*31337+1) % p->q;

    /* Warm-up */
    fn(a, tw, p);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) fn(a, tw, p);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    free(tw); free(a);
    return (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── Montgomery not valid for q ≥ 2^32 (R=2^32 must exceed q for REDC). ─── */
#define MONT_MAX_Q  UINT64_C(0xFFFFFFFF)   /* 2^32 - 1 */

int main(void)
{
    printf("\n" ANSI_WHT
           "╔══════════════════════════════════════════════════════════╗\n"
           "║      NTT CPU Algorithm Benchmark — 14 moduli sweep       ║\n"
           "╚══════════════════════════════════════════════════════════╝\n"
           ANSI_RST "\n");

    /* One benchmark row per modulus at the maximum supported transform size. */
    printf("  ┌─────────────┬────────────────────┬────────┬──────────────┬──────────────┬──────────────┐\n");
    printf("  │ " ANSI_CYN "%-11s" ANSI_RST
           " │ " ANSI_CYN "%-18s" ANSI_RST
           " │ " ANSI_CYN "%-6s" ANSI_RST
           " │ " ANSI_CYN "%-12s" ANSI_RST
           " │ " ANSI_CYN "%-12s" ANSI_RST
           " │ " ANSI_CYN "%-12s" ANSI_RST " │\n",
           "Prime", "Form", "max n", "CT-DIT NTT/s", "Stk NTT/s", "Mont NTT/s");
    printf("  ├─────────────┼────────────────────┼────────┼──────────────┼──────────────┼──────────────┤\n");

    for (int mi = 0; mi < NTT_NUM_MODULI; mi++) {
        const ntt_modulus_info_t *m = &NTT_MODULI[mi];
        uint64_t max_n = UINT64_C(1) << m->max_log2_n;
        /* Cap benchmark n to avoid excessive runtime for huge sizes. */
        uint64_t bench_n = (max_n > 1024) ? 1024 : max_n;

        /* Compute primitive bench_n-th root of unity for this prime. */
        uint64_t omega = ntt_modulus_omega(m, bench_n);
        if (omega == 0) {
            printf("  │ %-11s │ %-18s │ %-6lu │ %-12s │ %-12s │ %-12s │\n",
                   m->name, m->form, bench_n, "skip", "skip", "skip");
            continue;
        }

        ntt_params_t p;
        p.n = bench_n; p.q = m->q; p.omega = omega;
        if (ntt_params_init(&p) != 0) continue;

        /* Scale iterations: fewer for large n to keep runtime reasonable. */
        //uint64_t iters = 10000 / (bench_n / 64);
        uint64_t iters = 100000 / (bench_n / 64);
        if (iters < 200) iters = 200;

        double t_ct  = time_ntt(ct_ntt,  &p, iters);
        double t_stk = time_ntt(stk_ntt, &p, iters);
        double r_ct  = (t_ct  > 0) ? iters / t_ct  : 0;
        double r_stk = (t_stk > 0) ? iters / t_stk : 0;

        /* Montgomery only valid for q < 2^32 (R = 2^32 must exceed q). */
        int mont_ok = (m->q <= MONT_MAX_Q);
        double r_mnt = 0;
        if (mont_ok) {
            double t_mnt = time_ntt(mnt_ntt, &p, iters);
            r_mnt = (t_mnt > 0) ? iters / t_mnt : 0;
        }

        /* Green-highlight the fastest algorithm in each row. */
        double best = r_ct;
        if (r_stk > best) best = r_stk;
        if (mont_ok && r_mnt > best) best = r_mnt;

        const char *hl_ct  = (r_ct  == best) ? ANSI_GRN : "";
        const char *hl_stk = (r_stk == best) ? ANSI_GRN : "";
        const char *hl_mnt = (mont_ok && r_mnt == best) ? ANSI_GRN : "";

        char mont_str[20];
        if (mont_ok) snprintf(mont_str, sizeof mont_str, "%.0f", r_mnt);
        else         snprintf(mont_str, sizeof mont_str, "N/A");

        printf("  │ %-11s │ %-18s │ %6lu │ %s%-12.0f" ANSI_RST
               " │ %s%-12.0f" ANSI_RST " │ %s%-12s" ANSI_RST " │\n",
               m->name, m->form, bench_n,
               hl_ct,  r_ct,
               hl_stk, r_stk,
               hl_mnt, mont_str);
    }

    printf("  └─────────────┴────────────────────┴────────┴──────────────┴──────────────┴──────────────┘\n");
    printf("\n  " ANSI_YLW "Note:" ANSI_RST
           " Benchmarked at min(max_n, 1024). Mont N/A for q ≥ 2^32 (R=2^32 < q).\n"
           "  All algorithms use correct primitive roots via ntt_modulus_omega().\n\n");
    return 0;
}
#endif /* NTT_BENCH_NO_MAIN */
