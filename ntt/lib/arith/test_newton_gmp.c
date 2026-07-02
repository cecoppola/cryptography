/*
 * test_newton_gmp.c — GMP cross-oracle for Newton reciprocal + division.
 *
 * newton.c is only exercised on its happy path by compute_e. This fuzzes it
 * directly against GMP (an independent bignum), covering both division regimes:
 *   - bits_N <= 2*bits(D): uses the caller-supplied reciprocal (base_convert's
 *     structural case, O(1) correction);
 *   - bits_N >  2*bits(D): bigint_div_newton builds a fresh internal reciprocal
 *     at scale bits_N+bits(D)+1. The test asserts BOTH regimes are hit.
 * Also cross-checks newton_reciprocal(Q) == floor(2^(2*bits(Q))/Q), and pins
 * edge cases (D=1, N<D -> 0, N==D -> 1, exact multiples, powers of two).
 *
 * Schoolbook multiply is forced (bigint_mul_set_threshold high) so the NTT
 * path is never taken — the test is pure host, no GPU. Dual-width (LB 64/112).
 * Built by lib/Makefile `test-newton-gmp`; links bigint/newton/multiply + GMP.
 */

#include "bigint.h"
#include "newton.h"
#include "multiply.h"
#include <gmp.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define GRN "\033[1;32m"
#define RED "\033[1;31m"
#define CYN "\033[1;36m"
#define RST "\033[0m"

/* NTT stub: schoolbook is forced below, so this must never be reached. */
void ntt_bigint_mul(BigInt *c, const BigInt *a, const BigInt *b) {
    (void)c; (void)a; (void)b;
    fprintf(stderr, "FAIL: ntt_bigint_mul called (threshold not forced?)\n");
    exit(1);
}
void ntt_mul_init_impl(int mln) { (void)mln; }
void ntt_mul_teardown_impl(void) {}

static int fails = 0;
#define CHECK(cond, label) do { \
    if (!(cond)) { fails++; fprintf(stderr, "  " RED "FAIL" RST " %s\n", label); } \
} while (0)

/* newton_reciprocal converges from below: its result r satisfies
 * r <= floor(2^scale/D), reaching equality when Newton fully converges (the
 * common large-D case) and stopping at most RECIP_TOL short otherwise (e.g.
 * D=1 gives floor-1). bigint_div_newton's correction loop absorbs that slack;
 * what MUST hold is r <= floor and floor-r small. */
#define RECIP_TOL 2
static long recip_err_max = 0;

/* mpz_t -> BigInt via limb-copy through mpz_export (LIMB_BITS-width chunks). */
static void mpz_to_bigint(BigInt *out, const mpz_t z)
{
    bigint_zero(out);
    if (mpz_sgn(z) == 0) return;
    size_t nbytes = (mpz_sizeinbase(z, 2) + 7) / 8;
    unsigned char *bytes = (unsigned char *)calloc(nbytes + 16, 1);
    size_t got = 0;
    mpz_export(bytes, &got, -1, 1, 1, 0, z);
    const int LB_BYTES_INT = (LIMB_BITS + 7) / 8;
    int n_limbs = (int)((got + LB_BYTES_INT - 1) / LB_BYTES_INT);
    if (n_limbs > out->cap) { fprintf(stderr, "mpz_to_bigint: cap overflow\n"); exit(2); }
    for (int i = 0; i < n_limbs; i++) {
        limb_t v = 0;
        for (int b = 0; b < LB_BYTES_INT; b++) {
            size_t bi = (size_t)i * LB_BYTES_INT + b;
            if (bi < got) v |= ((limb_t)bytes[bi]) << (8 * b);
        }
#if LIMB_BITS == 112
        v &= (((limb_t)1 << LIMB_BITS) - 1);
#endif
        out->limbs[i] = v;
    }
    out->n_limbs = n_limbs;
    while (out->n_limbs > 1 && out->limbs[out->n_limbs - 1] == 0) out->n_limbs--;
    free(bytes);
}

static void bigint_to_mpz(mpz_t out, const BigInt *a)
{
    const int LB_BYTES_INT = (LIMB_BITS + 7) / 8;
    size_t n = (size_t)a->n_limbs * LB_BYTES_INT;
    if (n == 0) { mpz_set_ui(out, 0); return; }
    unsigned char *bytes = (unsigned char *)calloc(n + 16, 1);
    for (int i = 0; i < a->n_limbs; i++) {
        limb_t v = a->limbs[i];
        for (int b = 0; b < LB_BYTES_INT; b++)
            bytes[(size_t)i * LB_BYTES_INT + b] = (unsigned char)(v >> (8 * b));
    }
    mpz_import(out, n, -1, 1, 1, 0, bytes);
    free(bytes);
}

/* Smallest bits(D) at which the reciprocal is required to be a tight lower
 * bound. Below this the reciprocal is a degenerate under-approximation the
 * division correction handles (division is still checked exactly); such tiny
 * D never occur in production (D is 10 or N!-sized). */
#define RECIP_MIN_BITS 8

/* One division trial: q_lib = floor(N/D) vs GMP; returns 1 if bits_N > 2*bits(D). */
static int div_trial(const mpz_t N, const mpz_t D,
                     BigInt *n, BigInt *d, BigInt *recip, BigInt *q,
                     int *div_ok, int *recip_ok, int *recip_checked)
{
    mpz_to_bigint(n, N);
    mpz_to_bigint(d, D);

    /* reciprocal ~= floor(2^(2*bits(D)) / D); tight-lower-bound cross-check
     * vs GMP, only meaningful (and asserted) for non-trivially-small D. */
    newton_reciprocal(recip, d);
    unsigned long bd = mpz_sizeinbase(D, 2);
    if (bd >= RECIP_MIN_BITS) {
        (*recip_checked)++;
        mpz_t P, rref, rgot, diff; mpz_inits(P, rref, rgot, diff, NULL);
        mpz_setbit(P, 2 * bd);                 /* P = 2^(2*bits(D)) */
        mpz_fdiv_q(rref, P, D);
        bigint_to_mpz(rgot, recip);
        mpz_sub(diff, rref, rgot);              /* floor - recip, must be in [0,TOL] */
        if (mpz_sgn(diff) >= 0 && mpz_cmp_ui(diff, RECIP_TOL) <= 0) {
            (*recip_ok)++;
            long e = mpz_get_si(diff);
            if (e > recip_err_max) recip_err_max = e;
        } else {
            char *sd = mpz_get_str(NULL,10,D), *se = mpz_get_str(NULL,10,rref),
                 *sg = mpz_get_str(NULL,10,rgot);
            fprintf(stderr, "  recip OUT-OF-TOL D=%s floor=%s got=%s\n", sd, se, sg);
            free(sd); free(se); free(sg);
        }
        mpz_clears(P, rref, rgot, diff, NULL);
    }

    bigint_div_newton(q, n, d, recip);

    mpz_t qref, qgot; mpz_inits(qref, qgot, NULL);
    mpz_fdiv_q(qref, N, D);
    bigint_to_mpz(qgot, q);
    if (mpz_cmp(qref, qgot) == 0) (*div_ok)++;
    else {
        char *sN = mpz_get_str(NULL,10,N), *sD = mpz_get_str(NULL,10,D);
        char *se = mpz_get_str(NULL,10,qref), *sg = mpz_get_str(NULL,10,qgot);
        fprintf(stderr, "  div MISMATCH N=%s D=%s exp=%s got=%s\n", sN, sD, se, sg);
        free(sN); free(sD); free(se); free(sg);
    }
    mpz_clears(qref, qgot, NULL);

    return (long)mpz_sizeinbase(N, 2) > 2L * (long)mpz_sizeinbase(D, 2);
}

int main(void)
{
    printf("\n" CYN "═══ newton reciprocal + division vs GMP (LB=%d) ═══" RST "\n", LIMB_BITS);

    /* Force schoolbook: no NTT dispatch, fully host-side and deterministic. */
    bigint_mul_set_threshold(INT_MAX);

    const int TRIALS = 400;
    const int MAXDB  = 800;   /* max bits(D); N up to ~3*MAXDB bits */
    int cap_small = MAXDB / LIMB_BITS + 16;
    int cap_big   = 4 * MAXDB / LIMB_BITS + 16;

    BigInt d     = bigint_alloc(cap_small);
    BigInt recip = bigint_alloc(2 * cap_small + 16);
    BigInt n     = bigint_alloc(cap_big);
    BigInt q     = bigint_alloc(cap_big + 8);

    gmp_randstate_t st; gmp_randinit_mt(st); gmp_randseed_ui(st, 0x5A17ED);
    mpz_t N, D; mpz_inits(N, D, NULL);

    int div_ok = 0, recip_ok = 0, recip_checked = 0, n_big = 0, n_small = 0;

    for (int t = 0; t < TRIALS; t++) {
        unsigned long bd = 1 + (unsigned long)(gmp_urandomb_ui(st, 12) % MAXDB);   /* 1..MAXDB */
        mpz_urandomb(D, st, bd);
        if (mpz_sgn(D) == 0) mpz_set_ui(D, 1);
        /* N bits: sometimes < bd (q=0), often up to ~3*bd to hit the >2b branch. */
        unsigned long span = (t & 1) ? (3 * bd + 64) : (bd + 32);
        unsigned long bn = 1 + (unsigned long)(gmp_urandomb_ui(st, 13) % span);
        mpz_urandomb(N, st, bn);
        if (div_trial(N, D, &n, &d, &recip, &q, &div_ok, &recip_ok, &recip_checked))
            n_big++; else n_small++;
    }

    printf("  division        vs mpz_fdiv_q:      %4d/%d\n", div_ok, TRIALS);
    printf("  newton_reciprocal (<=floor, err<=%d): %4d/%d (bits(D)>=%d; max under-approx %ld)\n",
           RECIP_TOL, recip_ok, recip_checked, RECIP_MIN_BITS, recip_err_max);
    printf("  branch coverage: bits_N<=2b:%d  bits_N>2b:%d\n", n_small, n_big);

    CHECK(div_ok == TRIALS,             "division oracle");
    CHECK(recip_ok == recip_checked,    "reciprocal tight-lower-bound oracle");
    CHECK(recip_checked > 0,            "non-vacuity: reciprocal checked");
    CHECK(n_small > 0,                  "non-vacuity: bits_N<=2b branch exercised");
    CHECK(n_big  > 0,                   "non-vacuity: bits_N>2b (fresh-reciprocal) branch exercised");

    /* ── Explicit edge cases ─────────────────────────────────────────────── */
    struct { const char *N, *D; } edges[] = {
        {"0", "1"}, {"1", "1"}, {"7", "1"}, {"6", "7"},          /* q=0 / q=N / N<D */
        {"100", "10"}, {"999999999999999999999", "1000000000"},  /* exact-ish */
        {"340282366920938463463374607431768211455", "18446744073709551616"}, /* 2^128-1 / 2^64 */
        {"123456789012345678901234567890", "987654321"},
    };
    int e_ok = 0, e_n = (int)(sizeof(edges)/sizeof(edges[0]));
    for (int i = 0; i < e_n; i++) {
        mpz_set_str(N, edges[i].N, 10);
        mpz_set_str(D, edges[i].D, 10);
        int ro = 0, dvo = 0, rc = 0;   /* edges assert DIVISION exactness (recip
                                        * tolerance is regime-scoped above) */
        div_trial(N, D, &n, &d, &recip, &q, &dvo, &ro, &rc);
        e_ok += dvo;
    }
    printf("  edge-case division:                 %4d/%d\n", e_ok, e_n);
    CHECK(e_ok == e_n, "edge-case division (incl. D=1, N<D, exact multiples)");

    bigint_free(&d); bigint_free(&recip); bigint_free(&n); bigint_free(&q);
    mpz_clears(N, D, NULL); gmp_randclear(st);

    printf("\n  %s%d failures%s\n\n", fails ? RED : GRN, fails, RST);
    return fails ? 1 : 0;
}
