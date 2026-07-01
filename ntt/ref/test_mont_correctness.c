/*
 * test_mont_correctness.c — Verify Montgomery NTT matches CT-DIT NTT output
 *
 * Purpose:  Differential correctness test: mnt_ntt (Montgomery domain
 *           multiply) vs ntt_forward (lazy CT-DIT) over all 14 curated
 *           primes at sizes n=64, 128, and 256.
 *           Note: Montgomery uses R=2^32; primes q >= 2^32 are skipped
 *           (Solinas-60, Solinas-61, Goldilocks) — 64-bit Montgomery would
 *           be a different implementation not present in ntt_bench.c.
 * Method:   Same random input, same twiddle table, compare spectra.
 *           mnt_ntt builds its own Montgomery twiddles internally; the
 *           standard twiddle pointer is passed for API uniformity.
 * Build:    cc -O2 -Wall -Wextra -DNTT_CPU_NO_MAIN -o test_mont_correctness
 *               test_mont_correctness.c ntt_cpu.c
 */

#include "ntt.h"
#include "ntt_moduli.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── ANSI ────────────────────────────────────────────────────────────────── */
#define ANSI_GRN "\033[1;32m"
#define ANSI_RED "\033[1;31m"
#define ANSI_CYN "\033[1;36m"
#define ANSI_WHT "\033[1;37m"
#define ANSI_RST "\033[0m"

/* ── Montgomery helpers (independent copy from ntt_bench.c) ─────────────── */
typedef struct { uint64_t q, r2; uint32_t q_prime; } mont_ctx_t;

static mont_ctx_t mont_init(uint64_t q)
{
    mont_ctx_t c; c.q = q;
    uint32_t x = 1;
    for (int i = 0; i < 5; i++) x *= 2 - (uint32_t)q * x;
    c.q_prime = (uint32_t)(-(int32_t)x);
    c.r2 = (uint64_t)(((__uint128_t)1 << 64) % q);
    return c;
}
static uint64_t mont_mul(uint64_t a, uint64_t b, const mont_ctx_t *c)
{
    uint64_t t = a * b;
    uint32_t mp = (uint32_t)t * c->q_prime;
    uint64_t corr = (uint64_t)mp * c->q;
    uint64_t lo = t + corr;
    uint64_t carry = (lo < t) ? 1ULL : 0ULL;
    uint64_t u = (lo >> 32) | (carry << 32);
    return u >= c->q ? u - c->q : u;
}
static uint64_t mont_enter(uint64_t a, const mont_ctx_t *c) { return mont_mul(a, c->r2, c); }
static uint64_t mont_exit(uint64_t a, const mont_ctx_t *c)  { return mont_mul(a, 1ULL, c); }

static void ct_bit_rev(uint64_t *a, uint64_t n, uint32_t log2_n)
{
    for (uint64_t i = 0; i < n; i++) {
        uint64_t rev = 0, x = i;
        for (uint32_t b = 0; b < log2_n; b++) { rev = (rev<<1)|(x&1); x>>=1; }
        if (i < rev) { uint64_t t = a[i]; a[i] = a[rev]; a[rev] = t; }
    }
}

static void mnt_ntt_local(uint64_t *a, const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    mont_ctx_t mc = mont_init(q);
    uint64_t *tw_m = malloc((n/2) * sizeof(uint64_t));
    if (!tw_m) return;
    uint64_t omega_m = mont_enter(p->omega, &mc);
    tw_m[0] = mont_enter(1ULL, &mc);
    for (uint64_t k = 1; k < n/2; k++)
        tw_m[k] = mont_mul(tw_m[k-1], omega_m, &mc);
    for (uint64_t i = 0; i < n; i++) a[i] = mont_enter(a[i], &mc);
    ct_bit_rev(a, n, p->log2_n);
    for (uint64_t len = 1; len < n; len <<= 1) {
        uint64_t step = n / (len << 1);
        for (uint64_t i = 0; i < n; i += len << 1)
            for (uint64_t j = 0; j < len; j++) {
                uint64_t u = a[i+j];
                uint64_t v = mont_mul(tw_m[j*step], a[i+j+len], &mc);
                uint64_t s = u + v; if (s >= q) s -= q;
                uint64_t d = u - v + q; if (d >= q) d -= q;
                a[i+j] = s; a[i+j+len] = d;
            }
    }
    for (uint64_t i = 0; i < n; i++) a[i] = mont_exit(a[i], &mc);
    free(tw_m);
}

int main(void)
{
    uint64_t ns[] = { 64, 128, 256 };
    int nns = (int)(sizeof ns / sizeof ns[0]);
    int total = 0, passed = 0;

    printf("\n  " ANSI_WHT "Montgomery vs CT-DIT NTT — differential correctness" ANSI_RST "\n");
    printf("  %-14s %-5s %-4s\n", "modulus", "n", "MATCH");
    printf("  ──────────────────────────────\n");

    for (int mi = 0; mi < NTT_NUM_MODULI; mi++) {
        const ntt_modulus_info_t *m = &NTT_MODULI[mi];
        for (int k = 0; k < nns; k++) {
            uint64_t n = ns[k];
            /* Montgomery uses R=2^32; only valid for q < 2^32 */
            if (m->q >= (1ULL << 32)) continue;
            uint64_t omega = ntt_modulus_omega(m, n);
            if (omega == 0) continue;

            ntt_params_t p;
            memset(&p, 0, sizeof p);
            p.n = n; p.q = m->q; p.omega = omega;
            if (ntt_params_init(&p) != 0) continue;
            uint64_t *tw = ntt_alloc_twiddles(&p);
            if (!tw) continue;

            uint64_t *x_ct  = malloc(n * sizeof(uint64_t));
            uint64_t *x_mnt = malloc(n * sizeof(uint64_t));
            if (!x_ct || !x_mnt) { free(tw); free(x_ct); free(x_mnt); continue; }

            uint64_t seed = 0xDEADBEEFCAFE1234ULL ^ m->q ^ n;
            for (uint64_t i = 0; i < n; i++) {
                seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                x_ct[i] = x_mnt[i] = seed % m->q;
            }

            ntt_forward(x_ct, tw, &p);
            mnt_ntt_local(x_mnt, &p);

            int match = (memcmp(x_ct, x_mnt, n * sizeof(uint64_t)) == 0);
            /* vacuity: ensure input was not trivially zero */
            int nonzero = 0;
            for (uint64_t i = 0; i < n; i++) if (x_ct[i]) { nonzero = 1; break; }

            total++;
            if (match && nonzero) passed++;
            printf("  %-14s %-5llu %s\n", m->name, (unsigned long long)n,
                   (match && nonzero) ? ANSI_GRN "OK" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

            free(tw); free(x_ct); free(x_mnt);
        }
    }

    printf("  ──────────────────────────────\n");
    printf("  %d/%d PASS\n", passed, total);
    printf("  %s\n\n", (passed == total && total > 0) ? ANSI_GRN "ALL PASS" ANSI_RST
                                                      : ANSI_RED "FAILURES PRESENT" ANSI_RST);
    return (passed == total && total > 0) ? 0 : 1;
}
