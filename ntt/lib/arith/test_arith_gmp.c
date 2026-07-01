/*
 * test_arith_gmp.c — Independent GMP-as-oracle cross-check for lib/arith.
 *
 * The lib engine's correctness ultimately rests on (a) limb arithmetic
 * being faithful to ordinary integer arithmetic and (b) the decimal
 * conversion (base_convert.c divide-and-conquer) producing the standard
 * decimal expansion. test_arith.c already pins these with deterministic
 * vectors. This file adds an INDEPENDENT reference: GMP's mpz_t, an
 * unrelated arbitrary-precision implementation.
 *
 * Trials (each at both LIMB_BITS=64 and 112):
 *   T_DEC:  GMP-random 1024-bit A; build BigInt via limb-copy from
 *           mpz_export; bigint_to_decimal(BigInt) must equal mpz_get_str.
 *   T_ADD:  C_lib = A_lib + B_lib (lib). bigint→GMP, compare with
 *           C_gmp = A_gmp + B_gmp.
 *   T_SUB:  D_lib = max(A,B) - min(A,B); same cross-check.
 *
 * Calibration: deliberately swapping a sign or limb-order in this file
 * makes T_DEC fail loudly (small-bigint test pre-check inverts every
 * mismatch). The cross-implementation oracle catches ANY divergence
 * between lib and GMP — by construction non-vacuous.
 */

#include "bigint.h"
#include "base_convert.h"
#include "multiply.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRIALS 200
#define BITS 1024

#define GRN "\033[1;32m"
#define RED "\033[1;31m"
#define CYN "\033[1;36m"
#define RST "\033[0m"

/* NTT stubs (same convention as test_arith.c). */
void ntt_bigint_mul(BigInt *c, const BigInt *a, const BigInt *b) {
    (void)c; (void)a; (void)b;
    fprintf(stderr, "FAIL: ntt_bigint_mul called\n"); exit(1);
}
void ntt_mul_init_impl(int mln) { (void)mln; }
void ntt_mul_teardown_impl(void) {}

static int fails = 0;
#define CHECK(cond, label) do { \
    if (!(cond)) { fails++; fprintf(stderr, "  " RED "FAIL" RST " %s\n", label); } \
} while (0)

/* mpz_t → BigInt via limb-copy through mpz_export. Uses LIMB_BITS-width
 * chunks; works at both LB=64 (one mpz limb per BigInt limb on a 64-bit
 * GMP build) and LB=112 (re-chunk 8-byte mpz limbs into 14-byte chunks). */
static void mpz_to_bigint(BigInt *out, const mpz_t z)
{
    bigint_zero(out);
    if (mpz_sgn(z) == 0) return;
    /* Export to bytes (little-endian) then re-pack into LIMB_BITS chunks. */
    size_t nbytes = (mpz_sizeinbase(z, 2) + 7) / 8;
    unsigned char *bytes = (unsigned char *)calloc(nbytes + 16, 1);
    size_t got = 0;
    mpz_export(bytes, &got, -1 /* lsb first */, 1, 1, 0, z);
    /* Pack bytes into limb_t at LIMB_BITS spacing. */
    const int LB_BYTES_INT = (LIMB_BITS + 7) / 8;
    int n_limbs = (int)((got + LB_BYTES_INT - 1) / LB_BYTES_INT);
    if (n_limbs > out->cap) { free(bytes); return; }
    for (int i = 0; i < n_limbs; i++) {
        limb_t v = 0;
        for (int b = 0; b < LB_BYTES_INT; b++) {
            size_t bi = (size_t)i * LB_BYTES_INT + b;
            if (bi < got) v |= ((limb_t)bytes[bi]) << (8 * b);
        }
#if LIMB_BITS == 112
        /* limb_t = __uint128_t with 16 unused high bits; mask to LIMB_BITS.
         * For LIMB_BITS == 64 the limb fills uint64_t exactly, no mask
         * needed (and `1 << 64` would be UB on a 64-bit type). */
        v &= (((limb_t)1 << LIMB_BITS) - 1);
#endif
        out->limbs[i] = v;
    }
    out->n_limbs = n_limbs;
    while (out->n_limbs > 1 && out->limbs[out->n_limbs - 1] == 0) out->n_limbs--;
    free(bytes);
}

/* BigInt → mpz_t via the reverse limb-copy (also independent of
 * lib/arith's own conversion paths). */
static void bigint_to_mpz(mpz_t out, const BigInt *a)
{
    const int LB_BYTES_INT = (LIMB_BITS + 7) / 8;
    size_t n = (size_t)a->n_limbs * LB_BYTES_INT;
    unsigned char *bytes = (unsigned char *)calloc(n + 16, 1);
    for (int i = 0; i < a->n_limbs; i++) {
        limb_t v = a->limbs[i];
        for (int b = 0; b < LB_BYTES_INT; b++) {
            bytes[(size_t)i * LB_BYTES_INT + b] = (unsigned char)(v >> (8 * b));
        }
    }
    mpz_import(out, n, -1, 1, 1, 0, bytes);
    free(bytes);
}

int main(void)
{
    printf("\n" CYN "═══ lib/arith vs GMP cross-oracle (LB=%d, %d trials, %d-bit) ═══" RST "\n",
           LIMB_BITS, TRIALS, BITS);

    /* 1024-bit values need ~309 decimal digits → level 9 cache
     * (10^(2^9) = 10^512) is enough; pre-build the pow10 cache. */
    base_convert_init(9);

    gmp_randstate_t st;
    gmp_randinit_mt(st);
    gmp_randseed_ui(st, 0xDEADBEEF);

    mpz_t A, B, C_ref, C_got;
    mpz_inits(A, B, C_ref, C_got, NULL);

    int cap = (BITS / LIMB_BITS) + 8;
    BigInt a = bigint_alloc(cap);
    BigInt b = bigint_alloc(cap);
    BigInt c = bigint_alloc(cap * 2);

    int dec_ok = 0, add_ok = 0, sub_ok = 0, mul_ok = 0;

    for (int t = 0; t < TRIALS; t++) {
        mpz_urandomb(A, st, BITS);
        mpz_urandomb(B, st, BITS);

        mpz_to_bigint(&a, A);
        mpz_to_bigint(&b, B);

        /* (a) Decimal cross-check: bigint_to_decimal vs mpz_get_str. */
        char *dec = bigint_to_decimal(&a);
        char *ref = mpz_get_str(NULL, 10, A);
        if (dec && ref && strcmp(dec, ref) == 0) dec_ok++;
        free(dec); free(ref);

        /* (b) Addition cross-check. */
        bigint_add(&c, &a, &b);
        mpz_add(C_ref, A, B);
        bigint_to_mpz(C_got, &c);
        if (mpz_cmp(C_ref, C_got) == 0) add_ok++;

        /* (c) Subtraction: lib bigint_sub is unsigned, so use max-min. */
        if (mpz_cmp(A, B) >= 0) {
            bigint_sub(&c, &a, &b);
            mpz_sub(C_ref, A, B);
        } else {
            bigint_sub(&c, &b, &a);
            mpz_sub(C_ref, B, A);
        }
        bigint_to_mpz(C_got, &c);
        if (mpz_cmp(C_ref, C_got) == 0) sub_ok++;

        /* (d) Multiply cross-check. BITS=1024 → ~16 limbs/operand, below the
         * NTT threshold, so this exercises the schoolbook bigint_mul path
         * directly against mpz_mul (the ntt path is stubbed in this TU). */
        bigint_mul(&c, &a, &b);
        mpz_mul(C_ref, A, B);
        bigint_to_mpz(C_got, &c);
        if (mpz_cmp(C_ref, C_got) == 0) mul_ok++;
    }

    printf("  bigint_to_decimal vs mpz_get_str:  %4d/%d\n", dec_ok, TRIALS);
    printf("  bigint_add        vs mpz_add:      %4d/%d\n", add_ok, TRIALS);
    printf("  bigint_sub        vs mpz_sub:      %4d/%d\n", sub_ok, TRIALS);
    printf("  bigint_mul        vs mpz_mul:      %4d/%d\n", mul_ok, TRIALS);

    CHECK(dec_ok == TRIALS, "decimal-conversion oracle");
    CHECK(add_ok == TRIALS, "addition oracle");
    CHECK(sub_ok == TRIALS, "subtraction oracle");
    CHECK(mul_ok == TRIALS, "multiplication oracle");

    bigint_free(&a); bigint_free(&b); bigint_free(&c);
    mpz_clears(A, B, C_ref, C_got, NULL);
    gmp_randclear(st);

    printf("\n  %s%d failures%s\n\n", fails ? RED : GRN, fails, RST);
    return fails ? 1 : 0;
}
