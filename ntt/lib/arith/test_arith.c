/*
 * test_arith.c — Unit tests for lib/arith/ (CPU schoolbook path only).
 *
 * Provides stubs for ntt_bigint_mul / ntt_mul_init_impl / ntt_mul_teardown_impl
 * so the file links without HIP. All test cases stay below BIGINT_MUL_THRESHOLD
 * (64 limbs), so the schoolbook path is always used and the NTT stub is never
 * called.
 *
 * Build (from lib/arith/):
 *   gcc -O2 -Wall -Wextra -I.. test_arith.c \
 *       bigint.c multiply.c newton.c base_convert.c -o test_arith
 */

#include "bigint.h"
#include "multiply.h"
#include "newton.h"
#include "base_convert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ─── NTT stubs (schoolbook path only; NTT never called below threshold) ──── */

void ntt_bigint_mul(BigInt *c, const BigInt *a, const BigInt *b)
{
    (void)c; (void)a; (void)b;
    fprintf(stderr, "FAIL: ntt_bigint_mul called — operands exceeded threshold\n");
    exit(1);
}
void ntt_mul_init_impl(int max_log_n) { (void)max_log_n; }
void ntt_mul_teardown_impl(void) {}

/* ─── Test harness ────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do {                               \
    if (cond) { printf("  [PASS] %s\n", msg); g_pass++; }  \
    else       { printf("  [FAIL] %s\n", msg); g_fail++; } \
} while (0)

static uint64_t to_u64(const BigInt *a)
{
    return (a->n_limbs > 0) ? (uint64_t)a->limbs[0] : 0;
}

/* Representation-agnostic value check: does `a` equal the decimal
 * string `expect`? Uses bigint_to_decimal — a different code path from
 * add/shl/mul (no circularity) and itself verified against known
 * constants in test_base_convert. Requires base_convert_init() active
 * (main() initialises it for the whole run). This replaces the former
 * 64-bit limb-LAYOUT asserts (n_limbs/limbs[]) so the suite is a valid
 * oracle at both LIMB_BITS=64 and 112. */
static int eq_dec(const BigInt *a, const char *expect)
{
    char *s = bigint_to_decimal(a);
    int ok = (s != NULL) && (strcmp(s, expect) == 0);
    free(s);
    return ok;
}

/* ─── basic alloc / set / compare ────────────────────────────────────────── */

static void test_basic(void)
{
    printf("\n--- basic: alloc / zero / set / cmp / bits ---\n");
    BigInt a = bigint_alloc(4), b = bigint_alloc(4), z = bigint_alloc(2);

    bigint_zero(&z);
    CHECK(bigint_is_zero(&z),    "zero → is_zero");

    bigint_set_u64(&a, 1);
    bigint_set_u64(&b, 2);
    CHECK(!bigint_is_zero(&a),          "set_u64(1) not zero");
    CHECK(bigint_cmp(&a, &b) == -1,     "cmp(1, 2) = -1");
    CHECK(bigint_cmp(&b, &a) ==  1,     "cmp(2, 1) = 1");
    CHECK(bigint_cmp(&a, &a) ==  0,     "cmp(1, 1) = 0");
    CHECK(bigint_bits(&z) == 0,         "bits(0) = 0");
    CHECK(bigint_bits(&a) == 1,         "bits(1) = 1");

    bigint_set_u64(&a, UINT64_C(0xffffffffffffffff));
    CHECK(bigint_bits(&a) == 64,        "bits(UINT64_MAX) = 64");

    bigint_free(&a); bigint_free(&b); bigint_free(&z);
}

/* ─── add / sub ───────────────────────────────────────────────────────────── */

static void test_add_sub(void)
{
    printf("\n--- add / sub / add_u64 / sub_u64 ---\n");
    BigInt a = bigint_alloc(4), b = bigint_alloc(4), c = bigint_alloc(6);

    bigint_set_u64(&a, 1);
    bigint_set_u64(&b, 2);
    bigint_add(&c, &a, &b);
    CHECK(to_u64(&c) == 3, "1 + 2 = 3");

    /* UINT64_MAX + 1 overflows into limb 1 */
    bigint_set_u64(&a, UINT64_C(0xffffffffffffffff));
    bigint_set_u64(&b, 1);
    bigint_add(&c, &a, &b);
    CHECK(eq_dec(&c, "18446744073709551616"), "UINT64_MAX + 1 = 2^64");

    /* 10 - 3 = 7 */
    bigint_set_u64(&a, 10);
    bigint_set_u64(&b, 3);
    bigint_sub(&c, &a, &b);
    CHECK(to_u64(&c) == 7, "10 - 3 = 7");

    /* 2^64 - 1 = UINT64_MAX via sub_u64 */
    bigint_set_u64(&a, 0);
    bigint_add_u64(&a, 1);
    bigint_set_u64(&b, 1);
    bigint_add(&c, &a, &a);      /* c = 2 */
    bigint_shl(&a, &b, 64);      /* a = 2^64 */
    bigint_sub_u64(&a, 1);
    CHECK(a.n_limbs == 1 && a.limbs[0] == UINT64_C(0xffffffffffffffff),
          "2^64 - 1 = UINT64_MAX");

    /* add_u64 carry propagation: UINT64_MAX + 42 */
    bigint_set_u64(&a, UINT64_C(0xffffffffffffffff));
    bigint_add_u64(&a, 42);
    CHECK(eq_dec(&a, "18446744073709551657"), "UINT64_MAX + 42 = 2^64 + 41");

    bigint_free(&a); bigint_free(&b); bigint_free(&c);
}

/* ─── shifts ──────────────────────────────────────────────────────────────── */

static void test_shifts(void)
{
    printf("\n--- shl / shr ---\n");
    BigInt one = bigint_alloc(2), a = bigint_alloc(8), b = bigint_alloc(8);
    bigint_set_u64(&one, 1);

    bigint_shl(&a, &one, 0);
    CHECK(to_u64(&a) == 1,  "1 << 0 = 1");

    bigint_shl(&a, &one, 63);
    CHECK(a.n_limbs == 1 && a.limbs[0] == (UINT64_C(1) << 63),
          "1 << 63 = 2^63");

    bigint_shl(&a, &one, 64);
    CHECK(eq_dec(&a, "18446744073709551616"), "1 << 64 = 2^64");

    bigint_shl(&a, &one, 128);
    CHECK(eq_dec(&a, "340282366920938463463374607431768211456"),
          "1 << 128 = 2^128");
    CHECK(bigint_bits(&a) == 129, "bits(2^128) = 129");

    bigint_shr(&b, &a, 64);
    CHECK(eq_dec(&b, "18446744073709551616"), "2^128 >> 64 = 2^64");

    bigint_shr(&b, &a, 128);
    CHECK(b.n_limbs == 1 && b.limbs[0] == 1,
          "2^128 >> 128 = 1");

    bigint_shr(&b, &a, 129);
    CHECK(bigint_is_zero(&b), "2^128 >> 129 = 0");

    /* Bit-level shift: 3 << 1 = 6 */
    bigint_set_u64(&a, 3);
    bigint_shl(&b, &a, 1);
    CHECK(to_u64(&b) == 6, "3 << 1 = 6");

    /* 7 >> 2 = 1 */
    bigint_set_u64(&a, 7);
    bigint_shr(&b, &a, 2);
    CHECK(to_u64(&b) == 1, "7 >> 2 = 1");

    bigint_free(&one); bigint_free(&a); bigint_free(&b);
}

/* ─── mul_u64 ─────────────────────────────────────────────────────────────── */

static void test_mul_u64(void)
{
    printf("\n--- mul_u64 ---\n");
    BigInt a = bigint_alloc(4), c = bigint_alloc(6);

    bigint_set_u64(&a, 6);
    bigint_mul_u64(&c, &a, 7);
    CHECK(to_u64(&c) == 42, "6 * 7 = 42");

    bigint_set_u64(&a, 0);
    bigint_mul_u64(&c, &a, 999);
    CHECK(bigint_is_zero(&c), "0 * 999 = 0");

    /* UINT64_MAX * 2 = 2^65 - 2 */
    bigint_set_u64(&a, UINT64_C(0xffffffffffffffff));
    bigint_mul_u64(&c, &a, 2);
    CHECK(eq_dec(&c, "36893488147419103230"), "UINT64_MAX * 2 = 2^65 - 2");

    bigint_free(&a); bigint_free(&c);
}

/* ─── schoolbook multiply ─────────────────────────────────────────────────── */

static void test_mul(void)
{
    printf("\n--- schoolbook multiply (operands < %d limbs) ---\n",
           BIGINT_MUL_THRESHOLD);
    BigInt a = bigint_alloc(12), b = bigint_alloc(12), c = bigint_alloc(20);
    BigInt one = bigint_alloc(2);
    bigint_set_u64(&one, 1);

    /* 3 * 5 = 15 */
    bigint_set_u64(&a, 3);
    bigint_set_u64(&b, 5);
    bigint_mul(&c, &a, &b);
    CHECK(to_u64(&c) == 15, "3 * 5 = 15");

    /* 999999999^2 = 999999998000000001 */
    bigint_set_u64(&a, 999999999ULL);
    bigint_mul(&c, &a, &a);
    CHECK(to_u64(&c) == 999999998000000001ULL, "999999999^2 = 999999998000000001");

    /* 2^64 * 2^64 = 2^128 */
    bigint_shl(&a, &one, 64);
    bigint_mul(&c, &a, &a);
    CHECK(eq_dec(&c, "340282366920938463463374607431768211456"),
          "2^64 * 2^64 = 2^128");

    /* 2^256 * 2^256 = 2^512. Compared against an independently built
     * 2^512 via shl (decimal-validated above at 64/128) — avoids
     * transcribing a 155-digit constant; representation-agnostic. */
    bigint_shl(&a, &one, 256);
    bigint_mul(&c, &a, &a);
    {
        BigInt e = bigint_alloc(20);
        bigint_shl(&e, &one, 512);
        CHECK(bigint_cmp(&c, &e) == 0, "2^256 * 2^256 = 2^512");
        bigint_free(&e);
    }

    /* 0 * anything = 0 */
    bigint_zero(&a);
    bigint_set_u64(&b, 12345);
    bigint_mul(&c, &a, &b);
    CHECK(bigint_is_zero(&c), "0 * 12345 = 0");

    bigint_free(&a); bigint_free(&b); bigint_free(&c); bigint_free(&one);
}

/* ─── newton_reciprocal + bigint_div_newton ───────────────────────────────── */

static void test_newton_div(void)
{
    printf("\n--- newton reciprocal + division ---\n");
    BigInt N = bigint_alloc(12), D = bigint_alloc(8);
    BigInt recip = bigint_alloc(24), q = bigint_alloc(12);

    /* Helper: verify q*D + r = N, 0 <= r < D. Returns 1 on success. */
#define DIV_OK(q_, D_, N_) ({ \
    BigInt qd_ = bigint_alloc((q_)->n_limbs + (D_)->n_limbs + 2); \
    BigInt r_  = bigint_alloc((D_)->n_limbs + 2); \
    bigint_mul(&qd_, (q_), (D_)); \
    int ok_ = (bigint_cmp(&qd_, (N_)) <= 0); \
    bigint_sub(&r_, (N_), &qd_); \
    ok_ = ok_ && (bigint_cmp(&r_, (D_)) < 0); \
    bigint_add(&qd_, &qd_, &r_); \
    ok_ = ok_ && (bigint_cmp(&qd_, (N_)) == 0); \
    bigint_free(&qd_); bigint_free(&r_); \
    ok_; })

    /* 100 / 10 = 10 */
    bigint_set_u64(&N, 100);
    bigint_set_u64(&D, 10);
    newton_reciprocal(&recip, &D);
    bigint_div_newton(&q, &N, &D, &recip);
    CHECK(to_u64(&q) == 10 && DIV_OK(&q, &D, &N), "100 / 10 = 10");

    /* 123456789 / 7 = 17636684 r 1 */
    bigint_set_u64(&N, 123456789ULL);
    bigint_set_u64(&D, 7);
    newton_reciprocal(&recip, &D);
    bigint_div_newton(&q, &N, &D, &recip);
    CHECK(to_u64(&q) == 17636684ULL && DIV_OK(&q, &D, &N),
          "123456789 / 7 = 17636684");

    /* 10^18 / 999999937 */
    bigint_set_u64(&N, 1000000000000000000ULL);
    bigint_set_u64(&D, 999999937ULL);
    newton_reciprocal(&recip, &D);
    bigint_div_newton(&q, &N, &D, &recip);
    CHECK(DIV_OK(&q, &D, &N), "10^18 / 999999937: q*d + r = n, 0 <= r < d");

    /* 2^64 / (2^32+1): 2-limb N, 1-limb D, bits(N)=65 <= 2*bits(D)=66 ✓
     * Exact: q = 2^32-1 = 4294967295, r = 1 */
    {
        BigInt one = bigint_alloc(2), big = bigint_alloc(4);
        bigint_set_u64(&one, 1);
        bigint_shl(&big, &one, 64);                            /* big = 2^64 */
        bigint_set_u64(&D, ((uint64_t)1 << 32) + 1);          /* D = 2^32+1 */
        newton_reciprocal(&recip, &D);
        bigint_div_newton(&q, &big, &D, &recip);
        CHECK(DIV_OK(&q, &D, &big) && to_u64(&q) == 4294967295ULL,
              "2^64 / (2^32+1) = 4294967295 r 1");
        bigint_free(&one); bigint_free(&big);
    }

    /* 2^128 / (2^64 - 59) — large divisor, tests multi-limb reciprocal.
     * Exact result: q = 2^64+59, r = 3481.
     * Precondition: bits(N) <= 2*bits(D) so the Newton correction is O(1). */
    {
        BigInt one = bigint_alloc(2), big = bigint_alloc(6), d2 = bigint_alloc(4);
        bigint_set_u64(&one, 1);
        bigint_shl(&big, &one, 128);
        bigint_shl(&d2, &one, 64);
        bigint_sub_u64(&d2, 59);     /* d2 = 2^64 - 59 */
        newton_reciprocal(&recip, &d2);
        bigint_div_newton(&q, &big, &d2, &recip);
        CHECK(DIV_OK(&q, &d2, &big), "2^128 / (2^64-59): invariant holds");
        bigint_free(&one); bigint_free(&big); bigint_free(&d2);
    }

#undef DIV_OK

    bigint_free(&N); bigint_free(&D); bigint_free(&recip); bigint_free(&q);
}

/* ─── base conversion ─────────────────────────────────────────────────────── */

static void test_base_convert(void)
{
    printf("\n--- base conversion ---\n");
    /* Cache initialised globally in main() (level 8). */

    BigInt a = bigint_alloc(6);
    char *s;

    bigint_zero(&a);
    s = bigint_to_decimal(&a);
    CHECK(s && strcmp(s, "0") == 0, "0 → \"0\"");
    free(s);

    bigint_set_u64(&a, 1);
    s = bigint_to_decimal(&a);
    CHECK(s && strcmp(s, "1") == 0, "1 → \"1\"");
    free(s);

    bigint_set_u64(&a, 42);
    s = bigint_to_decimal(&a);
    CHECK(s && strcmp(s, "42") == 0, "42 → \"42\"");
    free(s);

    bigint_set_u64(&a, 1000000000ULL);
    s = bigint_to_decimal(&a);
    CHECK(s && strcmp(s, "1000000000") == 0, "10^9 → \"1000000000\"");
    free(s);

    bigint_set_u64(&a, UINT64_C(0xffffffffffffffff));
    s = bigint_to_decimal(&a);
    CHECK(s && strcmp(s, "18446744073709551615") == 0,
          "UINT64_MAX → \"18446744073709551615\"");
    free(s);

    /* 2^128 = 340282366920938463463374607431768211456 */
    {
        BigInt one = bigint_alloc(2), big = bigint_alloc(4);
        bigint_set_u64(&one, 1);
        bigint_shl(&big, &one, 128);
        s = bigint_to_decimal(&big);
        CHECK(s && strcmp(s, "340282366920938463463374607431768211456") == 0,
              "2^128 → \"340282366920938463463374607431768211456\"");
        free(s);
        bigint_free(&one); bigint_free(&big);
    }

    /* 2^192 = 6277101735386680763835789423207666416102355444464034512896 */
    {
        BigInt one = bigint_alloc(2), big = bigint_alloc(6);
        bigint_set_u64(&one, 1);
        bigint_shl(&big, &one, 192);
        s = bigint_to_decimal(&big);
        CHECK(s && strcmp(s,
              "6277101735386680763835789423207666416102355444464034512896") == 0,
              "2^192 → 58-digit decimal");
        free(s);
        bigint_free(&one); bigint_free(&big);
    }
    /* (base_convert_teardown handled globally in main) */

    /* pow10_exact: 10^7 = 10000000 */
    {
        BigInt p = bigint_alloc(4);
        base_convert_pow10_exact(&p, 7);
        s = bigint_to_decimal(&p);
        CHECK(s && strcmp(s, "10000000") == 0, "10^7 → \"10000000\"");
        free(s);
        bigint_free(&p);
    }

    /* pow10_exact: 10^20 — check decimal and equality with pow10_exact(10)*pow10_exact(10) */
    {
        BigInt p = bigint_alloc(4), p2 = bigint_alloc(4), prod = bigint_alloc(8);
        base_convert_pow10_exact(&p, 20);
        s = bigint_to_decimal(&p);
        /* 10^20 = 100000000000000000000 (21 chars) */
        CHECK(s && strlen(s) == 21 && s[0] == '1' && s[1] == '0',
              "10^20 has 21 digits starting with '10'");
        /* Verify: 10^10 * 10^10 = 10^20 (uses schoolbook, not Newton division) */
        base_convert_pow10_exact(&p2, 10);
        bigint_mul(&prod, &p2, &p2);
        CHECK(bigint_cmp(&prod, &p) == 0, "10^10 * 10^10 = 10^20");
        free(s);
        bigint_free(&p); bigint_free(&p2); bigint_free(&prod);
    }

    bigint_free(&a);   /* base_convert_teardown handled globally in main */
}

/* ─── scatter (CPU-only residue check) ───────────────────────────────────── */

static void test_scatter(void)
{
    printf("\n--- scatter (residue correctness) ---\n");
    BigInt a = bigint_alloc(4);
    uint64_t coeffs[8] = {0};

    /* a = 12345; scatter mod each prime → residue = 12345 % p = 12345 (< all primes) */
    bigint_set_u64(&a, 12345ULL);
    for (int pidx = 0; pidx < 4; pidx++) {
        bigint_scatter(&a, coeffs, 8, pidx);
        CHECK(coeffs[0] == 12345ULL && coeffs[1] == 0,
              pidx == 0 ? "scatter p0: 12345 % P0 = 12345" :
              pidx == 1 ? "scatter p1: 12345 % P1 = 12345" :
              pidx == 2 ? "scatter p2: 12345 % P2 = 12345" :
                          "scatter p3: 12345 % P3 = 12345");
    }

    /* a = UINT64_MAX; scatter mod P0 = 2^64-2^32+1.
     * UINT64_MAX = P0 + 2^32 - 2, so UINT64_MAX % P0 = 2^32 - 2 = 4294967294. */
    bigint_set_u64(&a, UINT64_C(0xffffffffffffffff));
    bigint_scatter(&a, coeffs, 4, 0);  /* pidx=0: P0=2^64-2^32+1 */
    CHECK(coeffs[0] == UINT64_C(4294967294), "scatter p0: UINT64_MAX % P0 = 2^32-2");

    bigint_free(&a);
}

/* ─── gather (NTT-boundary limb_pack; CPU-direct, dual-build critical) ────── */
/*
 * bigint_gather is normally only reached via the GPU NTT path (stubbed
 * in this file), so its LIMB_BITS=112 limb_pack was previously
 * unexercised on CPU. These call it directly. Value expectations are
 * width-INDEPENDENT (the integer is identical; only the internal limb
 * re-chunking differs), so pass/fail is the same at LB=64 and LB=112.
 */
static void set_u256(U256 *c, uint64_t w0, uint64_t w1,
                     uint64_t w2, uint64_t w3)
{
    c->w[0] = w0; c->w[1] = w1; c->w[2] = w2; c->w[3] = w3;
}

static void test_gather(void)
{
    printf("\n--- gather (NTT-boundary limb_pack) ---\n");
    BigInt g = bigint_alloc(40);
    U256 c[8];

    /* 1 coeff = 42 → 42 */
    memset(c, 0, sizeof c);
    set_u256(&c[0], 42, 0, 0, 0);
    bigint_gather(&g, c, 1);
    CHECK(eq_dec(&g, "42"), "gather [42] = 42");

    /* 1 coeff = 2^128 (w2=1) — known constant (verified in base conv) */
    memset(c, 0, sizeof c);
    set_u256(&c[0], 0, 0, 1, 0);
    bigint_gather(&g, c, 1);
    CHECK(eq_dec(&g, "340282366920938463463374607431768211456"),
          "gather [2^128] = 2^128");

    /* 1 coeff = 2^192 (w3=1) — full 256-bit re-chunk; verified constant */
    memset(c, 0, sizeof c);
    set_u256(&c[0], 0, 0, 0, 1);
    bigint_gather(&g, c, 1);
    CHECK(eq_dec(&g,
          "6277101735386680763835789423207666416102355444464034512896"),
          "gather [2^192] = 2^192");

    /* 1 coeff = 1 + 2^64 + 2^128 + 2^192 (all words set): densest
     * limb_pack. Expected built independently via shl/add (different
     * code path from gather), compared by value. */
    {
        BigInt e = bigint_alloc(8), t = bigint_alloc(8), one = bigint_alloc(2);
        bigint_set_u64(&one, 1);
        bigint_set_u64(&e, 1);
        bigint_shl(&t, &one, 64);  bigint_add(&e, &e, &t);
        bigint_shl(&t, &one, 128); bigint_add(&e, &e, &t);
        bigint_shl(&t, &one, 192); bigint_add(&e, &e, &t);
        memset(c, 0, sizeof c);
        set_u256(&c[0], 1, 1, 1, 1);
        bigint_gather(&g, c, 1);
        CHECK(bigint_cmp(&g, &e) == 0,
              "gather [1+2^64+2^128+2^192]");
        bigint_free(&e); bigint_free(&t); bigint_free(&one);
    }

    /* Round-trip: coeff_i = A.limbs[i] (its low 128 bits) → gather → A.
     * Width-agnostic by construction (coeff value == limb value, placed
     * at limb i). Exercises multi-limb pack + inter-coeff carry. */
    {
        BigInt A = bigint_alloc(8), one = bigint_alloc(2);
        bigint_set_u64(&one, 1);
        bigint_shl(&A, &one, 300);       /* 2^300: multi-limb both widths */
        bigint_add_u64(&A, 123456789);
        memset(c, 0, sizeof c);
        for (int i = 0; i < A.n_limbs && i < 8; i++) {
            c[i].w[0] = (uint64_t)A.limbs[i];
            c[i].w[1] = (uint64_t)((unsigned __int128)A.limbs[i] >> 64);
        }
        bigint_gather(&g, c, A.n_limbs);
        CHECK(bigint_cmp(&g, &A) == 0,
              "gather round-trip: A = 2^300 + 123456789");
        bigint_free(&A); bigint_free(&one);
    }

    bigint_free(&g);
}

/* ─── scatter_t / gather_t roundtrip (I15 CPU gate) ─────────────────────── */
static void test_scatter_t_gather_t(void)
{
    printf("\n--- scatter_t / gather_t (I15 transposed-scatter roundtrip) ---\n");

    /* M=4 (log_n=4, N=16): small exhaustive check. */
    {
        int M = 4, N = 16;
        uint64_t coeffs[16] = {0};
        U256 crt[16];
        memset(crt, 0, sizeof crt);

        BigInt a = bigint_alloc(N);
        bigint_set_u64(&a, 0);
        /* Fill limbs 0..N-1 with distinct values. */
        for (int k = 0; k < N; k++)
            a.limbs[k] = (uint64_t)(k + 1);
        a.n_limbs = N;

        /* scatter_t pidx=0; since all values < P0, residues equal limbs. */
        bigint_scatter_t(&a, coeffs, M, 0);

        /* Verify transposition: coeffs[c*M + r] == a.limbs[r*M + c]. */
        int ok = 1;
        for (int r = 0; r < M && ok; r++)
            for (int c = 0; c < M && ok; c++)
                if (coeffs[c * M + r] != a.limbs[r * M + c]) ok = 0;
        CHECK(ok, "scatter_t: coeffs[c*M+r] == limbs[r*M+c]");

        /* Build U256 coefficients matching the scatter output: coeff k holds
         * the value that was stored at coeffs[k] (the transposed index). */
        for (int k = 0; k < N; k++) {
            crt[k].w[0] = coeffs[k];
            crt[k].w[1] = crt[k].w[2] = crt[k].w[3] = 0;
        }

        BigInt b = bigint_alloc(N + 4);
        bigint_gather_t(&b, crt, N, M);

        /* gather_t must recover the original BigInt. */
        CHECK(bigint_cmp(&a, &b) == 0,
              "scatter_t -> gather_t round-trip: M=4 N=16");

        bigint_free(&a); bigint_free(&b);
    }

    /* M=8 (log_n=6, N=64): exercise a larger square. */
    {
        int M = 8, N = 64;
        uint64_t *coeffs = (uint64_t *)calloc((size_t)N, sizeof(uint64_t));
        U256 *crt = (U256 *)calloc((size_t)N, sizeof(U256));
        BigInt a = bigint_alloc(N);
        bigint_set_u64(&a, 0);
        /* Use limbs = 1..N (all < P0, so residues == limbs). */
        for (int k = 0; k < N; k++) a.limbs[k] = (uint64_t)(k + 1);
        a.n_limbs = N;

        bigint_scatter_t(&a, coeffs, M, 0);
        for (int k = 0; k < N; k++) {
            crt[k].w[0] = coeffs[k];
            crt[k].w[1] = crt[k].w[2] = crt[k].w[3] = 0;
        }
        BigInt b = bigint_alloc(N + 4);
        bigint_gather_t(&b, crt, N, M);
        CHECK(bigint_cmp(&a, &b) == 0,
              "scatter_t -> gather_t round-trip: M=8 N=64");

        bigint_free(&a); bigint_free(&b);
        free(coeffs); free(crt);
    }
}

/* ─── Main ───────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== lib/arith/ unit tests (CPU schoolbook path) ===\n");

    /* Level 8 = 2^8 = 256 decimal digits — covers every value tested
     * here (max is 2^512 = 155 digits). Active for the whole run so
     * eq_dec() works in every test, not just base conversion. */
    base_convert_init(8);

    test_basic();
    test_add_sub();
    test_shifts();
    test_mul_u64();
    test_mul();
    test_newton_div();
    test_base_convert();
    test_scatter();
    test_gather();
    test_scatter_t_gather_t();

    base_convert_teardown();

    printf("\n");
    if (g_fail == 0)
        printf("=== All %d tests passed ===\n", g_pass);
    else
        printf("=== %d/%d passed, %d FAILED ===\n",
               g_pass, g_pass + g_fail, g_fail);
    return g_fail ? 1 : 0;
}
