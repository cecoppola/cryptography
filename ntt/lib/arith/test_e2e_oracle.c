/*
 * test_e2e_oracle.c — Host-only end-to-end CRT-NTT BigInt-multiply oracle.
 *
 * Validates the ENTIRE multiply pipeline's logic on CPU, independently of the
 * GPU kernels, against GMP as ground truth. For random BigInts A, B:
 *
 *   1. bigint_scatter()  — the REAL project scatter: digit_k = limb_k mod P_i
 *   2. cyclic_conv()     — an INDEPENDENT pure-C radix-2 NTT convolution mod P_i
 *                          (plain %p arithmetic, self-discovered N-th roots),
 *                          one per prime — mirrors the GPU per-APU NTT lane
 *   3. crt_combine()     — Garner/CRT reconstruction of the 4 residues per
 *                          coefficient into a U256 (constants derived fresh via
 *                          GMP, cross-checking the project's PRIMES[])
 *   4. bigint_gather()   — the REAL project gather: positional base-2^LIMB_BITS
 *                          carry reconstruction of the product BigInt
 *   5. compare to mpz(A) * mpz(B)                       [GMP ground truth]
 *
 * If the composed result equals GMP's product over random + adversarial inputs
 * at both LIMB_BITS, then the design — primes, NTT roots, CRT headroom (Q≈2^255
 * covers n·(2^LIMB_BITS-1)^2 for the tested n), scatter/gather base, and the
 * cyclic=linear sizing — is sound end-to-end before any GPU bring-up. The NTT
 * here is deliberately a separate, simple implementation from the GPU kernels so
 * this is a true cross-check of the method, not a tautology.
 *
 * A second mode validates the negacyclic path (main.hip pipeline_negacyclic):
 * c = a*b mod (X^n+1) via psi twist -> length-n cyclic NTT -> psi^{-k} untwist
 * (psi a primitive 2n-th root, psi^n = -1), per prime, CRT-combined, compared
 * to a GMP schoolbook negacyclic convolution. Exercises the psi roots and the
 * twist/untwist convention independent of the GPU twist/untwist kernels.
 *
 * A third mode (trial_core_routed) is the integration check: it runs the
 * multiply through the EXACT GPU-kernel cores (transfer_core.h scatter_value +
 * gather_acc_limb + gather_carry_normalize) in pipeline position vs GMP,
 * confirming the cores compose correctly in the data path the gated GPU
 * pipeline (transfer_kernels.hip -DNTT_GPU_SCATTER_GATHER) will run.
 *
 * Built dual-width by lib/Makefile (test-e2e-oracle), pure C (gcc) + libgmp,
 * no GPU. Self-checks (root order, CRT headroom) assert inside the run.
 */

#include "../transfer_core.h"   /* GPU scatter/gather cores (integration test) */
#include "bigint.h"
#include "primes.h"
#include "crt_ntt.h"
#include <gmp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GRN "\033[1;32m"
#define RED "\033[1;31m"
#define CYN "\033[1;36m"
#define RST "\033[0m"

static int fails = 0, checks = 0;
#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { fails++; printf("  " RED "FAIL" RST " "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

/* Deterministic xorshift PRNG. */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t rng64(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static limb_t rand_limb(void)
{
    limb_t v = (limb_t)rng64();
#if LIMB_BITS == 112
    v |= ((limb_t)rng64()) << 64;
    v &= LIMB_MASK;
#endif
    return v;
}

/* ─── Modular arithmetic mod a 64-bit prime (independent of primes.h reduce) ── */
static uint64_t modadd(uint64_t a, uint64_t b, uint64_t p)
{
    __uint128_t s = (__uint128_t)a + b;
    if (s >= p) s -= p;
    return (uint64_t)s;
}
static uint64_t modsub(uint64_t a, uint64_t b, uint64_t p)
{
    return (a >= b) ? (a - b) : (uint64_t)((__uint128_t)a + p - b);
}
static uint64_t modmul(uint64_t a, uint64_t b, uint64_t p)
{
    return (uint64_t)(((__uint128_t)a * b) % p);
}
static uint64_t o_modpow(uint64_t b, uint64_t e, uint64_t p)
{
    uint64_t r = 1 % p; b %= p;
    while (e) { if (e & 1) r = modmul(r, b, p); b = modmul(b, b, p); e >>= 1; }
    return r;
}
static uint64_t modinv(uint64_t a, uint64_t p) { return o_modpow(a, p - 2, p); } /* Fermat */

/* Smallest primitive N-th root of unity mod p (N a power of two, N | p-1).
 * Searches h=2,3,...: w=h^((p-1)/N); accept iff order(w)==N. Asserts success. */
static uint64_t find_root(int N, uint64_t p)
{
    uint64_t e = (p - 1) / (uint64_t)N;
    for (uint64_t h = 2; h < 1000; h++) {
        uint64_t w = o_modpow(h, e, p);
        if (o_modpow(w, (uint64_t)N, p) == 1 && o_modpow(w, (uint64_t)N / 2, p) != 1)
            return w;
    }
    fprintf(stderr, "find_root: no primitive %d-th root mod %llu\n",
            N, (unsigned long long)p);
    exit(2);
}

/* In-place iterative radix-2 DIT NTT. w = primitive N-th root of unity. */
static void ntt(uint64_t *a, int N, uint64_t w, uint64_t p)
{
    for (int i = 1, j = 0; i < N; i++) {            /* bit-reversal permute */
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; }
    }
    for (int len = 2; len <= N; len <<= 1) {
        uint64_t wlen = o_modpow(w, (uint64_t)(N / len), p); /* primitive len-th root */
        for (int i = 0; i < N; i += len) {
            uint64_t wk = 1;
            for (int k = 0; k < len / 2; k++) {
                uint64_t u = a[i + k];
                uint64_t v = modmul(a[i + k + len / 2], wk, p);
                a[i + k]           = modadd(u, v, p);
                a[i + k + len / 2] = modsub(u, v, p);
                wk = modmul(wk, wlen, p);
            }
        }
    }
}

/* out = cyclic convolution of fa, fb (length N) mod p, via NTT. */
static void cyclic_conv(const uint64_t *fa, const uint64_t *fb, uint64_t *out,
                        int N, uint64_t p)
{
    uint64_t w = find_root(N, p);
    /* self-check: w has order exactly N */
    if (o_modpow(w, (uint64_t)N, p) != 1) { fprintf(stderr, "root order != N\n"); exit(2); }
    uint64_t winv = modinv(w, p);
    uint64_t ninv = modinv((uint64_t)N % p, p);
    uint64_t *A = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
    uint64_t *B = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
    memcpy(A, fa, (size_t)N * sizeof(uint64_t));
    memcpy(B, fb, (size_t)N * sizeof(uint64_t));
    ntt(A, N, w, p);
    ntt(B, N, w, p);
    for (int k = 0; k < N; k++) out[k] = modmul(A[k], B[k], p);
    ntt(out, N, winv, p);
    for (int k = 0; k < N; k++) out[k] = modmul(out[k], ninv, p);
    free(A); free(B);
}

/* ─── CRT reconstruction constants (GMP; derived from PRIMES[]) ─────────────── */
static mpz_t g_Q;            /* P0*P1*P2*P3 */
static mpz_t g_Ncoef[4];     /* M_i * inv(M_i mod P_i),  M_i = Q / P_i  */

static void crt_setup(void)
{
    mpz_t P[4], Mi, t;
    mpz_init(g_Q); mpz_init(Mi); mpz_init(t);
    mpz_set_ui(g_Q, 1);
    for (int i = 0; i < 4; i++) {
        mpz_init_set_ui(P[i], (unsigned long)PRIMES[i]);
        mpz_mul(g_Q, g_Q, P[i]);
    }
    for (int i = 0; i < 4; i++) {
        mpz_init(g_Ncoef[i]);
        mpz_divexact(Mi, g_Q, P[i]);          /* M_i = Q / P_i             */
        mpz_mod(t, Mi, P[i]);                  /* M_i mod P_i               */
        mpz_invert(t, t, P[i]);                /* y_i = inv(M_i) mod P_i    */
        mpz_mul(g_Ncoef[i], Mi, t);            /* N_i = M_i * y_i           */
    }
    for (int i = 0; i < 4; i++) mpz_clear(P[i]);
    mpz_clear(Mi); mpz_clear(t);
}

/* Reconstruct one coefficient: X = (sum r_i * N_i) mod Q, exported to U256. */
/* Export a non-negative mpz in [0, 2^256) to a U256 (little-endian 64-bit words). */
static void mpz_to_u256(U256 *out, const mpz_t x)
{
    memset(out->w, 0, sizeof(out->w));
    size_t count = 0;
    mpz_export(out->w, &count, -1, sizeof(uint64_t), 0, 0, x);
    if (count > 4) { fprintf(stderr, "mpz_to_u256: value >= 2^256\n"); exit(2); }
}

static void crt_combine(const uint64_t r[4], U256 *out)
{
    mpz_t X, t;
    mpz_init_set_ui(X, 0); mpz_init(t);
    for (int i = 0; i < 4; i++) {
        mpz_mul_ui(t, g_Ncoef[i], (unsigned long)r[i]);
        mpz_add(X, X, t);
    }
    mpz_mod(X, X, g_Q);
    mpz_to_u256(out, X);
    mpz_clear(X); mpz_clear(t);
}

/* ─── BigInt <-> mpz (base 2^LIMB_BITS) ─────────────────────────────────────── */
static void limb_to_mpz(mpz_t t, limb_t v)
{
    uint64_t lo = (uint64_t)v;
#if LIMB_BITS == 112
    uint64_t hi = (uint64_t)(v >> 64);
    mpz_set_ui(t, hi); mpz_mul_2exp(t, t, 64); mpz_add_ui(t, t, lo);
#else
    mpz_set_ui(t, lo);
#endif
}
static void bigint_to_mpz(mpz_t out, const BigInt *a)
{
    mpz_t t; mpz_init(t);
    mpz_set_ui(out, 0);
    for (int i = a->n_limbs - 1; i >= 0; i--) {
        mpz_mul_2exp(out, out, LIMB_BITS);
        limb_to_mpz(t, a->limbs[i]);
        mpz_add(out, out, t);
    }
    mpz_clear(t);
}

static void fill_random_bigint(BigInt *a, int n_limbs, int all_max)
{
    bigint_zero(a);
    for (int i = 0; i < n_limbs; i++)
        a->limbs[i] = all_max ? (limb_t)LIMB_MASK : rand_limb();
    a->n_limbs = n_limbs;
}

static int next_pow2(int x) { int n = 1; while (n < x) n <<= 1; return n; }

/* ─── One end-to-end trial: result = A*B via CRT-NTT, compared to GMP ───────── */
static void trial(int la, int lb, int all_max)
{
    BigInt A = bigint_alloc(la + 4), B = bigint_alloc(lb + 4);
    fill_random_bigint(&A, la, all_max);
    fill_random_bigint(&B, lb, all_max);

    int n_coeffs = la + lb;                 /* product spans < la+lb coefficients */
    int N = next_pow2(n_coeffs);            /* cyclic length: no wrap when N>=n_c  */

    uint64_t *fa = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
    uint64_t *fb = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
    uint64_t *rc[4];
    for (int i = 0; i < 4; i++) rc[i] = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));

    for (int i = 0; i < 4; i++) {
        bigint_scatter(&A, fa, N, i);       /* REAL scatter (linear, zero-padded) */
        bigint_scatter(&B, fb, N, i);
        cyclic_conv(fa, fb, rc[i], N, PRIMES[i]);  /* independent NTT convolution  */
    }

    U256 *coeffs = (U256 *)malloc((size_t)n_coeffs * sizeof(U256));
    for (int k = 0; k < n_coeffs; k++) {
        uint64_t r[4] = { rc[0][k], rc[1][k], rc[2][k], rc[3][k] };
        crt_combine(r, &coeffs[k]);
    }

    BigInt C = bigint_alloc(n_coeffs + 8);
    bigint_gather(&C, coeffs, n_coeffs);    /* REAL gather (carry reconstruction) */

    /* Ground truth: mpz(A) * mpz(B). */
    mpz_t mA, mB, mProd, mC;
    mpz_inits(mA, mB, mProd, mC, NULL);
    bigint_to_mpz(mA, &A); bigint_to_mpz(mB, &B);
    mpz_mul(mProd, mA, mB);
    bigint_to_mpz(mC, &C);
    CHECK(mpz_cmp(mC, mProd) == 0, "e2e la=%d lb=%d N=%d max=%d", la, lb, N, all_max);

    mpz_clears(mA, mB, mProd, mC, NULL);
    free(fa); free(fb); for (int i = 0; i < 4; i++) free(rc[i]);
    free(coeffs); bigint_free(&A); bigint_free(&B); bigint_free(&C);
}

/* ─── Negacyclic path: c = a*b mod (X^n + 1) over the composite modulus ──────
 * Mirrors main.hip's pipeline_negacyclic: pre-twist by psi^k, length-n cyclic
 * NTT, post-untwist by psi^{-k}, with psi = a primitive 2n-th root of unity
 * (psi^n = -1, the only property the separate twist needs). Validates the
 * negacyclic mode end-to-end (psi roots, twist/untwist, CRT) vs a GMP
 * schoolbook negacyclic convolution — independent of the GPU kernels. */

/* Primitive 2n-th root of unity mod p: psi^n == -1 (== p-1). 2n | p-1 holds for
 * the tested n since each prime has 2-adic valuation b >= 24. */
static uint64_t find_psi(int n, uint64_t p)
{
    uint64_t e = (p - 1) / ((uint64_t)2 * (uint64_t)n);
    for (uint64_t h = 2; h < 2000; h++) {
        uint64_t psi = o_modpow(h, e, p);
        if (o_modpow(psi, (uint64_t)n, p) == p - 1) return psi;
    }
    fprintf(stderr, "find_psi: none for n=%d p=%llu\n", n, (unsigned long long)p);
    exit(2);
}

/* out = a (*) b  mod (X^n + 1)  mod p,  via twist -> cyclic NTT -> untwist. */
static void negacyclic_conv(const uint64_t *a, const uint64_t *b, uint64_t *out,
                            int n, uint64_t p)
{
    uint64_t psi = find_psi(n, p);
    uint64_t psi_inv = modinv(psi, p);
    uint64_t *A = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    uint64_t *B = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    uint64_t pk = 1;
    for (int k = 0; k < n; k++) {                    /* pre-twist by psi^k */
        A[k] = modmul(a[k] % p, pk, p);
        B[k] = modmul(b[k] % p, pk, p);
        pk = modmul(pk, psi, p);
    }
    cyclic_conv(A, B, out, n, p);
    uint64_t ik = 1;
    for (int k = 0; k < n; k++) {                    /* post-untwist by psi^{-k} */
        out[k] = modmul(out[k], ik, p);
        ik = modmul(ik, psi_inv, p);
    }
    free(A); free(B);
}

/* One negacyclic trial: c = a*b mod (X^n+1) via per-prime negacyclic NTT + CRT,
 * compared coefficient-by-coefficient to a GMP schoolbook negacyclic product. */
static void trial_negacyclic(int n, int all_max)
{
    uint64_t *a = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    uint64_t *b = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    for (int k = 0; k < n; k++) {
        a[k] = all_max ? UINT64_MAX : rng64();
        b[k] = all_max ? UINT64_MAX : rng64();
    }

    uint64_t *rc[4];
    for (int i = 0; i < 4; i++) {
        rc[i] = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
        negacyclic_conv(a, b, rc[i], n, PRIMES[i]);
    }
    U256 *coeffs = (U256 *)malloc((size_t)n * sizeof(U256));
    for (int k = 0; k < n; k++) {
        uint64_t r[4] = { rc[0][k], rc[1][k], rc[2][k], rc[3][k] };
        crt_combine(r, &coeffs[k]);
    }

    /* Ground truth: c_k = sum_{i+j=k} a_i b_j - sum_{i+j=k+n} a_i b_j, mod Q. */
    mpz_t ck, t, exp;
    mpz_inits(ck, t, exp, NULL);
    int ok = 1, bad = -1;
    for (int k = 0; k < n && ok; k++) {
        mpz_set_ui(ck, 0);
        for (int i = 0; i < n; i++) {
            int jp = k - i;                          /* i+j = k    (positive)  */
            if (jp >= 0 && jp < n) {
                mpz_set_ui(t, a[i]); mpz_mul_ui(t, t, b[jp]); mpz_add(ck, ck, t);
            }
            int jn = k + n - i;                      /* i+j = k+n  (negative)  */
            if (jn >= 0 && jn < n) {
                mpz_set_ui(t, a[i]); mpz_mul_ui(t, t, b[jn]); mpz_sub(ck, ck, t);
            }
        }
        mpz_mod(ck, ck, g_Q);                        /* [0, Q) */
        U256 want; mpz_to_u256(&want, ck);
        if (memcmp(&want, &coeffs[k], sizeof(U256)) != 0) { ok = 0; bad = k; }
    }
    CHECK(ok, "negacyclic n=%d max=%d: coeff %d mismatch", n, all_max, bad);

    mpz_clears(ck, t, exp, NULL);
    free(a); free(b); for (int i = 0; i < 4; i++) free(rc[i]); free(coeffs);
}

/* ─── Integration: the GPU scatter/gather cores in pipeline position ──────────
 * Routes a full multiply through the EXACT per-element logic the GPU kernels
 * wrap (transfer_core.h) — scatter_value for the residues, then gather_acc_limb
 * (phase-1 pull/untranspose core) + gather_carry_normalize (phase-2 core, the
 * pair launch_gather_gpu wraps) for reconstruction — composed through scatter ->
 * NTT -> CRT -> gather and checked vs GMP. Unlike the unit tests (cores vs
 * bigint_scatter/gather in isolation), this confirms the cores compose
 * correctly in the integrated data path the gated GPU pipeline will run. The
 * CLA parallel phase-2 is a verified drop-in for gather_carry_normalize
 * (test_transfer_core, incl. adversarial carry chains random operands can't
 * reach). Linear layout (transposed=0); the I15 transposed layout's core
 * equivalence is covered by test_transfer_core. */
static void trial_core_routed(int la, int lb)
{
    BigInt A = bigint_alloc(la + 4), B = bigint_alloc(lb + 4);
    fill_random_bigint(&A, la, 0);
    fill_random_bigint(&B, lb, 0);
    int n_coeffs = la + lb, N = next_pow2(n_coeffs);

    uint64_t *fa = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
    uint64_t *fb = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
    uint64_t *rc[4];
    for (int i = 0; i < 4; i++) {
        rc[i] = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
        for (int j = 0; j < N; j++) {        /* scatter via the GPU scatter core */
            fa[j] = scatter_value(A.limbs, A.n_limbs, j, i);
            fb[j] = scatter_value(B.limbs, B.n_limbs, j, i);
        }
        cyclic_conv(fa, fb, rc[i], N, PRIMES[i]);
    }
    U256 *coeffs = (U256 *)malloc((size_t)n_coeffs * sizeof(U256));
    for (int k = 0; k < n_coeffs; k++) {
        uint64_t r[4] = { rc[0][k], rc[1][k], rc[2][k], rc[3][k] };
        crt_combine(r, &coeffs[k]);
    }

    /* gather via the phase-1 pull core + phase-2 carry-normalize core. */
    int n_out = n_coeffs + 4;
    uint64_t *acc_lo = (uint64_t *)malloc((size_t)n_out * sizeof(uint64_t));
    uint64_t *acc_hi = (uint64_t *)malloc((size_t)n_out * sizeof(uint64_t));
    for (int j = 0; j < n_out; j++) {
        __uint128_t acc = gather_acc_limb(coeffs, n_coeffs, j, 0, 0);
        acc_lo[j] = (uint64_t)acc;
        acc_hi[j] = (uint64_t)(acc >> 64);
    }
    BigInt C = bigint_alloc(n_out + 1);
    bigint_zero(&C);
    uint64_t carry = gather_carry_normalize(acc_lo, acc_hi, C.limbs, n_out);
    C.n_limbs = n_out;
    CHECK(carry == 0, "core-routed la=%d lb=%d: nonzero top carry", la, lb);

    mpz_t mA, mB, mProd, mC;
    mpz_inits(mA, mB, mProd, mC, NULL);
    bigint_to_mpz(mA, &A); bigint_to_mpz(mB, &B);
    mpz_mul(mProd, mA, mB);
    bigint_to_mpz(mC, &C);
    CHECK(mpz_cmp(mC, mProd) == 0, "core-routed la=%d lb=%d N=%d", la, lb, N);

    mpz_clears(mA, mB, mProd, mC, NULL);
    free(fa); free(fb); for (int i = 0; i < 4; i++) free(rc[i]); free(coeffs);
    free(acc_lo); free(acc_hi);
    bigint_free(&A); bigint_free(&B); bigint_free(&C);
}

int main(void)
{
    printf(CYN "=== end-to-end CRT-NTT multiply oracle vs GMP (LIMB_BITS=%d) ==="
           RST "\n", LIMB_BITS);
    crt_setup();

    /* CRT headroom self-check: Q must exceed the largest coefficient that the
     * tested sizes can produce, else cyclic-mod-Q != true product. */
    {
        mpz_t maxc, base; mpz_inits(maxc, base, NULL);
        mpz_set_ui(base, 1); mpz_mul_2exp(base, base, LIMB_BITS); /* 2^LIMB_BITS */
        mpz_sub_ui(base, base, 1);                                /* 2^LIMB_BITS-1 */
        mpz_mul(maxc, base, base);                                /* (.)^2 */
        mpz_mul_ui(maxc, maxc, 1024);                             /* * max N tested */
        CHECK(mpz_cmp(maxc, g_Q) < 0, "CRT headroom: max coeff < Q");
        mpz_clears(maxc, base, NULL);
    }

    printf("\n" CYN "--- fixed small sizes ---" RST "\n");
    int sz[] = { 1, 2, 3, 5, 8, 13, 32 };
    for (size_t i = 0; i < sizeof(sz)/sizeof(sz[0]); i++)
        for (size_t j = 0; j < sizeof(sz)/sizeof(sz[0]); j++)
            trial(sz[i], sz[j], 0);

    printf(CYN "--- adversarial all-max limbs (carry + max coeff stress) ---" RST "\n");
    int amax[] = { 1, 2, 7, 16, 64, 100 };
    for (size_t i = 0; i < sizeof(amax)/sizeof(amax[0]); i++)
        trial(amax[i], amax[i], 1);

    printf(CYN "--- randomized soak (300 trials) ---" RST "\n");
    for (int t = 0; t < 300; t++) {
        int la = 1 + (int)(rng64() % 200);
        int lb = 1 + (int)(rng64() % 200);
        trial(la, lb, 0);
    }

    /* Negacyclic mode: c = a*b mod (X^n+1), per-prime negacyclic NTT + CRT. */
    printf(CYN "--- negacyclic mod (X^n+1): fixed sizes ---" RST "\n");
    int ns[] = { 2, 4, 8, 16, 32, 64, 128, 256 };
    for (size_t i = 0; i < sizeof(ns)/sizeof(ns[0]); i++) trial_negacyclic(ns[i], 0);
    printf(CYN "--- negacyclic: adversarial all-max coefficients ---" RST "\n");
    for (size_t i = 0; i < sizeof(ns)/sizeof(ns[0]); i++) trial_negacyclic(ns[i], 1);
    printf(CYN "--- negacyclic: randomized soak (200 trials) ---" RST "\n");
    for (int t = 0; t < 200; t++) {
        int pw = 1 + (int)(rng64() % 8);             /* n in {2,4,...,256} */
        trial_negacyclic(1 << pw, 0);
    }

    /* Integration: GPU scatter/gather cores composed in pipeline position. */
    printf(CYN "--- core-routed integration (scatter_value + gather cores) ---" RST "\n");
    int cr[] = { 1, 2, 5, 16, 64, 200, 600 };   /* spans n_out below + above 1024 */
    for (size_t i = 0; i < sizeof(cr)/sizeof(cr[0]); i++)
        for (size_t j = 0; j < sizeof(cr)/sizeof(cr[0]); j++)
            trial_core_routed(cr[i], cr[j]);
    for (int t = 0; t < 100; t++)
        trial_core_routed(1 + (int)(rng64() % 300), 1 + (int)(rng64() % 300));

    for (int i = 0; i < 4; i++) mpz_clear(g_Ncoef[i]);
    mpz_clear(g_Q);

    printf("\n");
    if (fails == 0) printf(GRN "=== All %d checks passed ===" RST "\n", checks);
    else            printf(RED "=== %d/%d checks FAILED ===" RST "\n", fails, checks);
    return fails ? 1 : 0;
}
