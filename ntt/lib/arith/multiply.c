/*
 * multiply.c — Hybrid BigInt multiply: schoolbook or NTT dispatch.
 *
 * Schoolbook forms each limb*limb partial via mul_limb (bigint.h),
 * which SUBLIMB-splits so every intermediate fits __uint128_t at both
 * LIMB_BITS=64 (32-bit halves) and 112 (56-bit halves). The two-limb
 * product (r1:r0) is accumulated with a __uint128_t carry. At LB=64
 * this is numerically identical to the former direct-u128 path
 * (perf/results/PHASE_F_DESIGN.md unit F.4).
 *
 * NTT path: declared extern; implemented in ntt_mul.hip to access HIP APIs.
 */

#include "multiply.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

/* ntt_bigint_mul prototype now lives in multiply.h (shared by all callers). */

/* ntt_bigint_mul uses a single shared device buffer set (g_nms.d_in_a/b,
 * g_nms.stream[]).  Concurrent callers from different CPU threads would
 * corrupt each other's scatter inputs.  Serialize with a mutex so that
 * OpenMP-parallel callers in dc_convert stay correct.  The GPU is the
 * bottleneck anyway; this serialization does not reduce GPU throughput. */
static pthread_mutex_t ntt_gpu_lock = PTHREAD_MUTEX_INITIALIZER;

/* ─── Schoolbook multiply ────────────────────────────────────────────────── */
/*
 * Per (i,j): (r1:r0) = a[i]*b[j] via mul_limb; acc = c[i+j] + r0 +
 * (carry mod 2^LB); store acc mod 2^LB; carry = r1 + acc>>LB +
 * carry>>LB. carry is __uint128_t (holds the full high part). O(na*nb).
 */
static void mul_schoolbook(BigInt *c, const BigInt *a, const BigInt *b)
{
    int na = a->n_limbs, nb = b->n_limbs;
    int nc = na + nb;
    bigint_resize(c, nc);
    bigint_zero(c);
    c->n_limbs = nc;

    for (int i = 0; i < na; i++) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < nb; j++) {
            limb_t r0, r1;
            mul_limb(a->limbs[i], b->limbs[j], &r0, &r1);
            unsigned __int128 acc = (unsigned __int128)c->limbs[i + j]
                                  + r0 + (carry & LIMB_MASK);
            c->limbs[i + j] = (limb_t)(acc & LIMB_MASK);
            carry = (unsigned __int128)r1 + (acc >> LIMB_BITS)
                                          + (carry >> LIMB_BITS);
        }
        /* Propagate carry past the b-span. */
        int k = i + nb;
        while (carry) {
            unsigned __int128 s = (unsigned __int128)c->limbs[k]
                                + (carry & LIMB_MASK);
            c->limbs[k++] = (limb_t)(s & LIMB_MASK);
            carry = (s >> LIMB_BITS) + (carry >> LIMB_BITS);
        }
    }

    /* Trim leading zeros. */
    while (c->n_limbs > 0 && c->limbs[c->n_limbs - 1] == 0)
        c->n_limbs--;
}

/* ─── Runtime threshold ──────────────────────────────────────────────────── */

static int g_threshold = BIGINT_MUL_THRESHOLD;

int  bigint_mul_get_threshold(void)  { return g_threshold; }
void bigint_mul_set_threshold(int t) { g_threshold = t; }

/* ─── Public dispatcher ──────────────────────────────────────────────────── */

void bigint_mul(BigInt *c, const BigInt *a, const BigInt *b)
{
    int na = a->n_limbs, nb = b->n_limbs;
    int threshold = g_threshold;

    if (na <= threshold && nb <= threshold) {
        mul_schoolbook(c, a, b);
    } else {
        pthread_mutex_lock(&ntt_gpu_lock);
        ntt_bigint_mul(c, a, b);
        pthread_mutex_unlock(&ntt_gpu_lock);
    }
}

/* ─── Init / teardown (forwarded to ntt_mul.hip) ────────────────────────── */

extern void ntt_mul_init_impl(int max_log_n);
extern void ntt_mul_teardown_impl(void);

void ntt_mul_init(int max_log_n)     { ntt_mul_init_impl(max_log_n); }
void ntt_mul_teardown(void)          { ntt_mul_teardown_impl(); }
