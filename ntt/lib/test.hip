/*
 * test.hip — Correctness and timing tests for the CRT-NTT pipeline
 *
 * All HIP API return values wrapped in (void)() per environment rules.
 * Kernel launches (<<<...>>>) are NOT wrapped — they are void.
 *
 * Test order:
 *   1. verify_primes()          — constant values and NTT-friendliness
 *   2. test_lds_idx             — padding arithmetic spot checks
 *   3. test_reduction<PIDX>     — reduce_mod on GPU vs CPU reference
 *   4. test_shoup_mul<PIDX>     — shoup_mul on GPU vs CPU reference
 *   5. test_ntt_roundtrip<PIDX> — forward + inverse NTT = identity
 *   6. test_garner_small        — N=8, all residues=28, expect output=28
 *   7. test_garner_known        — known 128-bit value reconstructed exactly
 *   8. test_pipeline_oracle     — I15 GATE: ntt_bigint_mul vs bigint_mul on
 *                                 random BigInts; exercises the FULL pipeline
 *                                 (scatter → 4×NTT → 4×Hadamard → 4×INTT →
 *                                 GPU Garner → bigint_gather) and catches any
 *                                 layout mismatch introduced by the proposed
 *                                 transposed-scatter/gather fold (I15). Any
 *                                 layout change that silently breaks Garner or
 *                                 gather will cause a FAIL here.
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "primes.h"
#include "shoup.h"
#include "crt_ntt.h"
#include "arith/bigint.h"
#include "arith/multiply.h"

template<int PIDX>
void launch_ntt(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_intt(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_hadamard(uint64_t*, const uint64_t*, int, hipStream_t);
template<int PIDX>
void launch_stockham_ntt(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_intt(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_ntt_rect(uint64_t*, uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_intt_rect(uint64_t*, uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_ntt_ct_dit_mont(uint64_t*, const uint64_t*, int, hipStream_t);
template<int PIDX>
void launch_intt_ct_dit_mont(uint64_t*, const uint64_t*, uint64_t, int, hipStream_t);
GarnerConsts garner_precompute(void);
void garner_reconstruct(const uint64_t*, const uint64_t*, const uint64_t*,
                        const uint64_t*, U256*, int, const GarnerConsts*);
int verify_primes(void);

/* ─── LCG RNG ────────────────────────────────────────────────────────────── */
static uint64_t rng_state = UINT64_C(0xdeadbeefcafe1234);
static uint64_t rand64(void) {
    rng_state = rng_state * UINT64_C(6364136223846793005)
              + UINT64_C(1442695040888963407);
    return rng_state;
}

/* Set in main() via hipGetDeviceCount; all hipSetDevice calls use % g_n_devs
 * so the test suite runs on a single device (e.g. 6900XT dev machine). */
static int g_n_devs = 1;

static void pass(const char* name) { printf("[PASS] %s\n", name); }
static void fail(const char* name, const char* msg) {
    printf("[FAIL] %s: %s\n", name, msg);
}

/* ─── Test 1: LDS index padding ──────────────────────────────────────────── */
static int lds_idx_host(int i) { return (i >> 5) * 33 + (i & 31); }

static void test_lds_idx(void)
{
    int ok = (lds_idx_host(0)  == 0
           && lds_idx_host(31) == 31
           && lds_idx_host(32) == 33
           && lds_idx_host(63) == 64
           && lds_idx_host(64) == 66);
    /* Injectivity check: no two distinct logical indices map to same padded index */
    int i, j;
    for (i = 0; i < 1024 && ok; i++)
        for (j = 0; j < i && ok; j++)
            if (lds_idx_host(i) == lds_idx_host(j)) ok = 0;

    if (ok) pass("lds_idx padding"); else fail("lds_idx padding", "mapping error");
}

/* ─── Test 1b: STOK_PAD injectivity ─────────────────────────────────────── */
/*
 * STOK_PAD(i) = i + (i >> 5): pads one slot every 32 elements to break LDS
 * bank conflicts in stockham_kernel.  STOK_STRIDE = 1057 is the GLOBAL upper
 * bound used here (the maximum of all per-LOG_N_SUB strides after C1).
 * After C1, each stockham_kernel<PIDX,LOG_N_SUB> uses its own constexpr
 * STOK_STRIDE_C = (2*half-1) + ((2*half-1)>>5) + 1, which is always ≤ 1057.
 * This test remains valid as a conservative bound: if STOK_PAD is injective on
 * [0,1024) within STOK_STRIDE=1057, it is also injective within any smaller
 * STOK_STRIDE_C.  Two invariants verified:
 *   (a) STOK_PAD is injective on [0, N_MAX) where N_MAX = 1024.
 *   (b) max(STOK_PAD(i)) < STOK_STRIDE, so the two ping-pong buffers
 *       (at offsets 0 and STOK_STRIDE) do not overlap in LDS.
 */
static void test_stok_pad(void)
{
    const int N_MAX = 1024;
    int seen[STOK_STRIDE] = {0};
    int ok = 1;

    for (int i = 0; i < N_MAX && ok; i++) {
        int idx = STOK_PAD(i);
        if (idx < 0 || idx >= STOK_STRIDE) {
            ok = 0;
        } else if (seen[idx]) {
            ok = 0;   /* collision: two distinct i map to same padded slot */
        } else {
            seen[idx] = 1;
        }
    }

    if (ok) pass("stok_pad injective [0,1024) within STOK_STRIDE");
    else    fail("stok_pad", "collision or out-of-bounds padded index");
}

/* ─── Test 2: GPU reduction ──────────────────────────────────────────────── */
template<int PIDX>
__global__ void kern_reduce(const uint64_t* a, const uint64_t* b,
                             uint64_t* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = reduce_mod<PIDX>((__uint128_t)a[i] * b[i]);
}

template<int PIDX>
static void test_reduction(int n)
{
    char name[32];
    snprintf(name, sizeof(name), "reduction p%d", PIDX);

    uint64_t p = PRIMES[PIDX];
    uint64_t *h_a = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
    uint64_t *h_b = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
    uint64_t *h_out = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));

    int i;
    for (i = 0; i < n; i++) { h_a[i] = rand64() % p; h_b[i] = rand64() % p; }

    uint64_t *d_a, *d_b, *d_out;
    (void)(hipSetDevice(PIDX % g_n_devs));
    (void)(hipMalloc(&d_a,   (size_t)n * sizeof(uint64_t)));
    (void)(hipMalloc(&d_b,   (size_t)n * sizeof(uint64_t)));
    (void)(hipMalloc(&d_out, (size_t)n * sizeof(uint64_t)));
    (void)(hipMemcpy(d_a, h_a, (size_t)n * sizeof(uint64_t), hipMemcpyHostToDevice));
    (void)(hipMemcpy(d_b, h_b, (size_t)n * sizeof(uint64_t), hipMemcpyHostToDevice));

    kern_reduce<PIDX><<<(n + 255) / 256, 256>>>(d_a, d_b, d_out, n);
    (void)(hipDeviceSynchronize());
    (void)(hipMemcpy(h_out, d_out, (size_t)n * sizeof(uint64_t), hipMemcpyDeviceToHost));

    int errs = 0;
    for (i = 0; i < n; i++) {
        uint64_t want = (uint64_t)((__uint128_t)h_a[i] * h_b[i] % p);
        if (h_out[i] != want && ++errs <= 3)
            fprintf(stderr, "  i=%d got=%llu want=%llu\n", i,
                    (unsigned long long)h_out[i], (unsigned long long)want);
    }
    if (!errs) pass(name); else fail(name, "mismatch");

    (void)(hipFree(d_a)); (void)(hipFree(d_b)); (void)(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);
}

/* ─── Test 3: GPU Shoup multiply ─────────────────────────────────────────── */
template<int PIDX>
__global__ void kern_shoup(const uint64_t* t_arr, const uint64_t* w_arr,
                            const uint64_t* wi_arr, uint64_t* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = shoup_mul<PIDX>(t_arr[i], w_arr[i], wi_arr[i]);
}

template<int PIDX>
static void test_shoup_mul(int n)
{
    char name[32];
    snprintf(name, sizeof(name), "shoup_mul p%d", PIDX);

    uint64_t p = PRIMES[PIDX];
    uint64_t *h_t  = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
    uint64_t *h_w  = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
    uint64_t *h_wi = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));
    uint64_t *h_out = (uint64_t*)malloc((size_t)n * sizeof(uint64_t));

    int i;
    for (i = 0; i < n; i++) {
        uint64_t wv  = rand64() % p;
        ShoupPair sp = make_shoup(wv, p);
        h_t[i]  = rand64() % p;
        h_w[i]  = sp.w;
        h_wi[i] = sp.w_inv;
    }

    uint64_t *d_t, *d_w, *d_wi, *d_out;
    (void)(hipSetDevice(PIDX % g_n_devs));
    (void)(hipMalloc(&d_t,   (size_t)n * sizeof(uint64_t)));
    (void)(hipMalloc(&d_w,   (size_t)n * sizeof(uint64_t)));
    (void)(hipMalloc(&d_wi,  (size_t)n * sizeof(uint64_t)));
    (void)(hipMalloc(&d_out, (size_t)n * sizeof(uint64_t)));
    (void)(hipMemcpy(d_t,  h_t,  (size_t)n * sizeof(uint64_t), hipMemcpyHostToDevice));
    (void)(hipMemcpy(d_w,  h_w,  (size_t)n * sizeof(uint64_t), hipMemcpyHostToDevice));
    (void)(hipMemcpy(d_wi, h_wi, (size_t)n * sizeof(uint64_t), hipMemcpyHostToDevice));

    kern_shoup<PIDX><<<(n + 255) / 256, 256>>>(d_t, d_w, d_wi, d_out, n);
    (void)(hipDeviceSynchronize());
    (void)(hipMemcpy(h_out, d_out, (size_t)n * sizeof(uint64_t), hipMemcpyDeviceToHost));

    int errs = 0;
    for (i = 0; i < n; i++) {
        uint64_t want = (uint64_t)((__uint128_t)h_w[i] * h_t[i] % p);
        if (h_out[i] != want && ++errs <= 3)
            fprintf(stderr, "  i=%d\n", i);
    }
    if (!errs) pass(name); else fail(name, "mismatch");

    (void)(hipFree(d_t)); (void)(hipFree(d_w));
    (void)(hipFree(d_wi)); (void)(hipFree(d_out));
    free(h_t); free(h_w); free(h_wi); free(h_out);
}

/* ─── Stockham flat twiddle table ────────────────────────────────────────── */
/*
 * Purpose: build the N/2-entry flat Stockham twiddle table for a log_n-point NTT.
 * Inputs:  dev=device index, p=prime, g_root=primitive root, log_n=transform size.
 * Output:  device pointer to N/2 ShoupPairs (caller must hipFree).
 *
 * For log_n <= 10: tw[k] = shoup(omega^k, p)  k in [0, N/2).
 * For log_n >  10 (4-step, N1=N2=2^(log_n/2)):
 *   tw[0 .. N1/2-1]   = flat N1-point table, omega_N1 = g^{(p-1)/N1}
 *   tw[N1/2 .. N/2-1] = flat N2-point table, omega_N2 = g^{(p-1)/N2}
 */
static ShoupPair* make_stok_flat(int dev, uint64_t p, uint64_t g_root, int log_n)
{
    int N    = 1 << log_n;
    int n_tw = N / 2;
    ShoupPair* h = (ShoupPair*)malloc((size_t)n_tw * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "make_stok_flat: malloc failed\n"); exit(1); }

    if (log_n <= 10) {
        uint64_t omega = modpow(g_root, (p - 1) / (uint64_t)N, p);
        uint64_t cur = 1;
        for (int k = 0; k < n_tw; k++) {
            h[k] = make_shoup(cur, p);
            cur = (uint64_t)((__uint128_t)cur * omega % p);
        }
    } else {
        int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
        int N1 = 1 << log_n1, N2 = 1 << log_n2;
        uint64_t omega1 = modpow(g_root, (p - 1) / (uint64_t)N1, p);
        uint64_t omega2 = modpow(g_root, (p - 1) / (uint64_t)N2, p);
        uint64_t cur = 1;
        for (int k = 0; k < N1 / 2; k++) {
            h[k] = make_shoup(cur, p);
            cur = (uint64_t)((__uint128_t)cur * omega1 % p);
        }
        cur = 1;
        for (int k = 0; k < N2 / 2; k++) {
            h[N1 / 2 + k] = make_shoup(cur, p);
            cur = (uint64_t)((__uint128_t)cur * omega2 % p);
        }
    }

    ShoupPair* d;
    (void)(hipSetDevice(dev % g_n_devs));
    (void)(hipMalloc(&d, (size_t)n_tw * sizeof(ShoupPair)));
    (void)(hipMemcpy(d, h, (size_t)n_tw * sizeof(ShoupPair), hipMemcpyHostToDevice));
    free(h);
    return d;
}

/* ─── Cross-twiddle table for 4-step Stockham (log_n > 10) ──────────────── */
/*
 * Purpose: build the N-entry cross-twiddle table for the 4-step Stockham NTT.
 * Entry at flat index r*N2+c = shoup(scale * omega^{r*c}, p)
 *   where omega = g_root^{(p-1)/N} is the N-th primitive root.
 *   scale = 1         for the forward NTT cross-twiddle.
 *   scale = N^{-1}    for the inverse (INTT) cross-twiddle (N^{-1} pre-baked).
 * Inputs:  dev, p, g_root (use g^{-1} for INTT), log_n, scale.
 * Output:  device pointer to N ShoupPairs (caller must hipFree).
 */
static ShoupPair* make_stok_cross(int dev, uint64_t p, uint64_t g_root,
                                   int log_n, uint64_t scale)
{
    int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
    int N1 = 1 << log_n1, N2 = 1 << log_n2;
    int N  = N1 * N2;
    uint64_t omega = modpow(g_root, (p - 1) / (uint64_t)N, p);

    ShoupPair* h = (ShoupPair*)malloc((size_t)N * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "make_stok_cross: malloc failed\n"); exit(1); }

    for (int r = 0; r < N1; r++) {
        uint64_t omega_r = modpow(omega, (uint64_t)r, p); /* omega^r */
        uint64_t cur = scale; /* cur = scale * omega^{r*c} at c=0 */
        for (int c = 0; c < N2; c++) {
            h[r * N2 + c] = make_shoup(cur, p);
            cur = (uint64_t)((__uint128_t)cur * omega_r % p);
        }
    }

    ShoupPair* d;
    (void)(hipSetDevice(dev % g_n_devs));
    (void)(hipMalloc(&d, (size_t)N * sizeof(ShoupPair)));
    (void)(hipMemcpy(d, h, (size_t)N * sizeof(ShoupPair), hipMemcpyHostToDevice));
    free(h);
    return d;
}

/* Montgomery-form CT-DIT twiddle table for launch_ntt_ct_dit_mont.
 * Stage-bucketed layout (stage s at offset 2^s-1, mirroring make_tw_local in
 * arith/ntt_mul.hip), entries stored as w * R mod p  (Montgomery representation,
 * R = 2^64, R mod p = 2^b - 1). Inputs: dev, p, g_root (use g^{-1} for the
 * inverse table), log_n, prime_b. The CT-DIT path is Montgomery-only since I4;
 * the earlier Shoup-form builder (make_ctdit_flat) was retired with it.
 * Returns device pointer to N-1 uint64_t entries (caller must hipFree). */
static uint64_t* make_mont_flat(int dev, uint64_t p, uint64_t g_root,
                                int log_n, int prime_b)
{
    int n_tw = (1 << log_n) - 1;
    uint64_t* h = (uint64_t*)malloc((size_t)n_tw * sizeof(uint64_t));
    if (!h) { fprintf(stderr, "make_mont_flat: malloc failed\n"); exit(1); }
    uint64_t R_mod_p = (UINT64_C(1) << prime_b) - 1;
    uint64_t omega   = modpow(g_root, (p - 1) / (uint64_t)(1 << log_n), p);
    int offset = 0;
    for (int s = 0; s < log_n; s++) {
        int cnt = 1 << s;
        uint64_t root = modpow(omega, (uint64_t)((1 << log_n) >> (s + 1)), p);
        uint64_t cur  = 1;
        for (int j = 0; j < cnt; j++) {
            h[offset + j] = (uint64_t)((__uint128_t)cur * R_mod_p % p);
            cur = (uint64_t)((__uint128_t)cur * root % p);
        }
        offset += cnt;
    }
    uint64_t* d;
    (void)hipSetDevice(dev % g_n_devs);
    (void)hipMalloc(&d, (size_t)n_tw * sizeof(uint64_t));
    (void)hipMemcpy(d, h, (size_t)n_tw * sizeof(uint64_t), hipMemcpyHostToDevice);
    free(h);
    return d;
}

/* ─── Test 4: NTT round-trip (Stockham) ─────────────────────────────────── */
/*
 * Uses launch_stockham_ntt + launch_stockham_intt: natural-order in/out, no
 * bit-reversal, clean round-trip semantics.  For log_n > 10 a proper
 * cross-twiddle table is built (avoids the NULL-deref that broke the old
 * CT-DIT path).  N^{-1} is pre-baked into the inverse cross-twiddle so
 * the INTT scaling is fused at no extra HBM pass.
 */
template<int PIDX>
static void test_ntt_roundtrip(int log_n)
{
    char name[48];
    snprintf(name, sizeof(name), "NTT round-trip (Stockham) p%d N=2^%d", PIDX, log_n);

    int N = 1 << log_n;
    uint64_t p = PRIMES[PIDX], g = PRIM_ROOT[PIDX];
    uint64_t N_inv = modpow((uint64_t)N, p - 2, p);
    uint64_t g_inv = modpow(g, p - 2, p);
    int dev = PIDX % g_n_devs;

    ShoupPair* d_tw_fwd = make_stok_flat(dev, p, g,     log_n);
    ShoupPair* d_tw_inv = make_stok_flat(dev, p, g_inv, log_n);

    /* Cross-twiddle only needed for 4-step (log_n > 10).                     */
    /* INTT cross-twiddle has N^{-1} pre-baked so cross_twiddle_kernel fuses  */
    /* the scaling at no added HBM pass.                                      */
    ShoupPair* d_tw_cross     = NULL;
    ShoupPair* d_tw_cross_inv = NULL;
    if (log_n > 10) {
        d_tw_cross     = make_stok_cross(dev, p, g,     log_n, 1);
        d_tw_cross_inv = make_stok_cross(dev, p, g_inv, log_n, N_inv);
    }

    uint64_t *d_data;
    uint64_t *h_ref = (uint64_t*)malloc((size_t)N * sizeof(uint64_t));
    uint64_t *h_out = (uint64_t*)malloc((size_t)N * sizeof(uint64_t));
    if (!h_ref || !h_out) { fprintf(stderr, "test_ntt_roundtrip: malloc failed\n"); exit(1); }

    (void)(hipSetDevice(dev));
    (void)(hipMalloc(&d_data, (size_t)N * sizeof(uint64_t)));

    int i;
    for (i = 0; i < N; i++) h_ref[i] = rand64() % p;
    (void)(hipMemcpy(d_data, h_ref, (size_t)N * sizeof(uint64_t), hipMemcpyHostToDevice));

    hipStream_t s;
    (void)(hipStreamCreate(&s));

    /* Odd log_n>10 uses the rectangular 4-step (needs a size-N scratch). */
    int odd = (log_n > 10) && (log_n & 1);
    uint64_t *d_scr = NULL;
    if (odd) (void)(hipMalloc(&d_scr, (size_t)N * sizeof(uint64_t)));
    if (odd) launch_stockham_ntt_rect<PIDX>(d_data, d_scr, d_tw_fwd, d_tw_cross, log_n, s);
    else     launch_stockham_ntt<PIDX>(d_data, d_tw_fwd, d_tw_cross, log_n, s);
    (void)(hipStreamSynchronize(s));
    if (odd) launch_stockham_intt_rect<PIDX>(d_data, d_scr, N_inv, d_tw_inv, d_tw_cross_inv, log_n, s);
    else     launch_stockham_intt<PIDX>(d_data, N_inv, d_tw_inv, d_tw_cross_inv, log_n, s);
    (void)(hipStreamSynchronize(s));
    if (d_scr) (void)(hipFree(d_scr));

    (void)(hipMemcpy(h_out, d_data, (size_t)N * sizeof(uint64_t), hipMemcpyDeviceToHost));

    int errs = 0;
    for (i = 0; i < N; i++) {
        if (h_out[i] != h_ref[i] && ++errs <= 3)
            fprintf(stderr, "  i=%d got=%llu want=%llu\n", i,
                    (unsigned long long)h_out[i], (unsigned long long)h_ref[i]);
    }
    if (!errs) pass(name); else fail(name, "round-trip mismatch");

    (void)(hipFree(d_data));
    (void)(hipFree(d_tw_fwd)); (void)(hipFree(d_tw_inv));
    if (d_tw_cross)     (void)(hipFree(d_tw_cross));
    if (d_tw_cross_inv) (void)(hipFree(d_tw_cross_inv));
    (void)(hipStreamDestroy(s));
    free(h_ref); free(h_out);
}

/* ─── Test 4b: NTT CONVOLUTION (fwd·fwd·Hadamard·inv vs naive) ───────────── */
/*
 * A roundtrip only proves inv∘fwd = identity; the production multiply
 * additionally requires that pointwise multiply in the NTT domain equals
 * cyclic convolution. This test exercises EXACTLY the production sequence
 * (launch_stockham_ntt ×2, launch_hadamard, launch_stockham_intt) and
 * compares against a naive O(N²) cyclic convolution mod p. Use only at
 * small 4-step sizes (log_n 12/14) where the naive reference is cheap.
 */
/* ─── I11 oracle: rect NTT polynomial product vs CT-DIT reference ───────── */
/*
 * Verifies the rectangular Bailey 4-step (I11) for ODD log_n > 10 by
 * computing the polynomial product of two random arrays via BOTH the rect
 * path and the CT-DIT path (proven correct), then comparing element-by-element.
 * CT-DIT is the confirmed production path for odd log_n — any divergence is a
 * DFT-correctness failure in the rect kernel.  O(N) comparison: runs fast at
 * all sizes (N=2^17 and 2^19 are the two untested sizes from the I11 gap).
 */
template<int PIDX>
static void test_ntt_rect_vs_ctdit(int log_n)
{
    char name[64];
    snprintf(name, sizeof(name),
             "rect vs CT-DIT polymul p%d N=2^%d", PIDX, log_n);

    int N = 1 << log_n;
    uint64_t p = PRIMES[PIDX], g = PRIM_ROOT[PIDX];
    uint64_t N_inv = modpow((uint64_t)N, p - 2, p);
    uint64_t g_inv = modpow(g, p - 2, p);
    int dev = PIDX % g_n_devs;

    /* Twiddle tables for the rect (Stockham) path. */
    ShoupPair* d_stok_fwd  = make_stok_flat(dev, p, g,     log_n);
    ShoupPair* d_stok_inv  = make_stok_flat(dev, p, g_inv, log_n);
    ShoupPair* d_cross_fwd = make_stok_cross(dev, p, g,     log_n, 1);
    /* launch_stockham_intt_rect ignores the n_inv parameter — N^{-1} must be
     * pre-baked into the cross-twiddle table (matches test_ntt_convolution). */
    ShoupPair* d_cross_inv = make_stok_cross(dev, p, g_inv, log_n, N_inv);

    /* Twiddle tables for the CT-DIT-Mont reference (I4): handles any log_n;
     * replaces the old launch_ntt/launch_intt which only work for even log_n
     * (square 4-step layout). */
    uint64_t* d_ct_fwd = make_mont_flat(dev, p, g,     log_n, PRIME_B[PIDX]);
    uint64_t* d_ct_inv = make_mont_flat(dev, p, g_inv, log_n, PRIME_B[PIDX]);

    uint64_t *ha = (uint64_t*)malloc((size_t)N*8);
    uint64_t *hb = (uint64_t*)malloc((size_t)N*8);
    uint64_t *ho_rect = (uint64_t*)malloc((size_t)N*8);
    uint64_t *ho_ct   = (uint64_t*)malloc((size_t)N*8);
    if (!ha||!hb||!ho_rect||!ho_ct){
        fprintf(stderr,"rect_vs_ctdit: malloc failed\n"); exit(1); }
    for (int i = 0; i < N; i++) { ha[i] = rand64() % p; hb[i] = rand64() % p; }

    uint64_t *da_rect, *db_rect, *da_ct, *db_ct, *d_scr;
    (void)hipSetDevice(dev);
    (void)hipMalloc(&da_rect, (size_t)N*8); (void)hipMalloc(&db_rect, (size_t)N*8);
    (void)hipMalloc(&da_ct,   (size_t)N*8); (void)hipMalloc(&db_ct,   (size_t)N*8);
    (void)hipMalloc(&d_scr,   (size_t)N*8);
    (void)hipMemcpy(da_rect, ha, (size_t)N*8, hipMemcpyHostToDevice);
    (void)hipMemcpy(db_rect, hb, (size_t)N*8, hipMemcpyHostToDevice);
    (void)hipMemcpy(da_ct,   ha, (size_t)N*8, hipMemcpyHostToDevice);
    (void)hipMemcpy(db_ct,   hb, (size_t)N*8, hipMemcpyHostToDevice);
    hipStream_t s; (void)hipStreamCreate(&s);

    /* Rect path: fwd_rect(a) * fwd_rect(b), inv_rect. */
    launch_stockham_ntt_rect<PIDX>(da_rect, d_scr, d_stok_fwd, d_cross_fwd, log_n, s);
    (void)hipStreamSynchronize(s);
    launch_stockham_ntt_rect<PIDX>(db_rect, d_scr, d_stok_fwd, d_cross_fwd, log_n, s);
    (void)hipStreamSynchronize(s);
    launch_hadamard<PIDX>(da_rect, db_rect, N, s);
    launch_stockham_intt_rect<PIDX>(da_rect, d_scr, N_inv, d_stok_inv, d_cross_inv, log_n, s);
    (void)hipStreamSynchronize(s);
    (void)hipMemcpy(ho_rect, da_rect, (size_t)N*8, hipMemcpyDeviceToHost);

    /* CT-DIT-Mont reference (I4): handles all log_n including odd > 10. */
    launch_ntt_ct_dit_mont<PIDX>(da_ct, d_ct_fwd, log_n, s);
    (void)hipStreamSynchronize(s);
    launch_ntt_ct_dit_mont<PIDX>(db_ct, d_ct_fwd, log_n, s);
    (void)hipStreamSynchronize(s);
    launch_hadamard<PIDX>(da_ct, db_ct, N, s);
    launch_intt_ct_dit_mont<PIDX>(da_ct, d_ct_inv, N_inv, log_n, s);
    (void)hipStreamSynchronize(s);
    (void)hipMemcpy(ho_ct, da_ct, (size_t)N*8, hipMemcpyDeviceToHost);

    /* Compare: rect and CT-DIT must give the identical polynomial product. */
    int errs = 0;
    for (int i = 0; i < N; i++)
        if (ho_rect[i] != ho_ct[i] && ++errs <= 3)
            fprintf(stderr, "  k=%d rect=%llu ctdit=%llu\n", i,
                    (unsigned long long)ho_rect[i],
                    (unsigned long long)ho_ct[i]);
    if (!errs) pass(name);
    else       fail(name, "rect != CT-DIT poly product");

    (void)hipFree(da_rect); (void)hipFree(db_rect);
    (void)hipFree(da_ct);   (void)hipFree(db_ct);
    (void)hipFree(d_scr);
    (void)hipFree(d_stok_fwd); (void)hipFree(d_stok_inv);
    (void)hipFree(d_cross_fwd); (void)hipFree(d_cross_inv);
    (void)hipFree(d_ct_fwd); (void)hipFree(d_ct_inv);
    (void)hipStreamDestroy(s);
    free(ha); free(hb); free(ho_rect); free(ho_ct);
}

template<int PIDX>
static void test_ntt_convolution(int log_n)
{
    char name[56];
    snprintf(name, sizeof(name), "NTT convolution (Stockham) p%d N=2^%d", PIDX, log_n);

    int N = 1 << log_n;
    uint64_t p = PRIMES[PIDX], g = PRIM_ROOT[PIDX];
    uint64_t N_inv = modpow((uint64_t)N, p - 2, p);
    uint64_t g_inv = modpow(g, p - 2, p);
    int dev = PIDX % g_n_devs;

    ShoupPair* d_tw_fwd = make_stok_flat(dev, p, g,     log_n);
    ShoupPair* d_tw_inv = make_stok_flat(dev, p, g_inv, log_n);
    ShoupPair* d_tw_cross     = (log_n > 10) ? make_stok_cross(dev, p, g,     log_n, 1)     : NULL;
    ShoupPair* d_tw_cross_inv = (log_n > 10) ? make_stok_cross(dev, p, g_inv, log_n, N_inv) : NULL;

    uint64_t *ha = (uint64_t*)malloc((size_t)N*sizeof(uint64_t));
    uint64_t *hb = (uint64_t*)malloc((size_t)N*sizeof(uint64_t));
    uint64_t *ho = (uint64_t*)malloc((size_t)N*sizeof(uint64_t));
    uint64_t *ref= (uint64_t*)malloc((size_t)N*sizeof(uint64_t));
    if (!ha||!hb||!ho||!ref){ fprintf(stderr,"conv malloc failed\n"); exit(1);}
    for (int i=0;i<N;i++){ ha[i]=rand64()%p; hb[i]=rand64()%p; }

    /* naive cyclic convolution mod p */
    for (int k=0;k<N;k++){
        __uint128_t acc=0;
        for (int i=0;i<N;i++){
            int j=(k-i)&(N-1);
            acc += (__uint128_t)ha[i]*hb[j] % p;
            if (acc >= ((__uint128_t)1<<120)) acc %= p;
        }
        ref[k]=(uint64_t)(acc % p);
    }

    /* I15: launch_stockham_large (even 4-step, log_n>10 even) skips the initial
     * T, expecting pre-transposed input from bigint_scatter_t. Pre-transpose
     * on the CPU here; post-un-transpose after INTT (which also skips final T). */
    int even4 = (log_n > 10) && !(log_n & 1);
    int M4    = even4 ? (1 << (log_n / 2)) : 0;

    uint64_t *da,*db;
    (void)hipSetDevice(dev);
    (void)hipMalloc(&da,(size_t)N*sizeof(uint64_t));
    (void)hipMalloc(&db,(size_t)N*sizeof(uint64_t));
    if (even4) {
        uint64_t *ha_t = (uint64_t*)malloc((size_t)N * sizeof(uint64_t));
        uint64_t *hb_t = (uint64_t*)malloc((size_t)N * sizeof(uint64_t));
        for (int r = 0; r < M4; r++)
            for (int c = 0; c < M4; c++) {
                ha_t[c * M4 + r] = ha[r * M4 + c];
                hb_t[c * M4 + r] = hb[r * M4 + c];
            }
        (void)hipMemcpy(da,ha_t,(size_t)N*sizeof(uint64_t),hipMemcpyHostToDevice);
        (void)hipMemcpy(db,hb_t,(size_t)N*sizeof(uint64_t),hipMemcpyHostToDevice);
        free(ha_t); free(hb_t);
    } else {
        (void)hipMemcpy(da,ha,(size_t)N*sizeof(uint64_t),hipMemcpyHostToDevice);
        (void)hipMemcpy(db,hb,(size_t)N*sizeof(uint64_t),hipMemcpyHostToDevice);
    }
    hipStream_t s; (void)hipStreamCreate(&s);

    int odd = (log_n > 10) && (log_n & 1);
    uint64_t *d_scr = NULL;
    if (odd) (void)hipMalloc(&d_scr, (size_t)N*sizeof(uint64_t));
    if (odd) { launch_stockham_ntt_rect<PIDX>(da, d_scr, d_tw_fwd, d_tw_cross, log_n, s);
               (void)hipStreamSynchronize(s);
               launch_stockham_ntt_rect<PIDX>(db, d_scr, d_tw_fwd, d_tw_cross, log_n, s); }
    else     { launch_stockham_ntt<PIDX>(da, d_tw_fwd, d_tw_cross, log_n, s);
               launch_stockham_ntt<PIDX>(db, d_tw_fwd, d_tw_cross, log_n, s); }
    (void)hipStreamSynchronize(s);
    launch_hadamard<PIDX>(da, db, N, s);
    if (odd) launch_stockham_intt_rect<PIDX>(da, d_scr, N_inv, d_tw_inv, d_tw_cross_inv, log_n, s);
    else     launch_stockham_intt<PIDX>(da, N_inv, d_tw_inv, d_tw_cross_inv, log_n, s);
    (void)hipStreamSynchronize(s);
    if (d_scr) (void)hipFree(d_scr);
    (void)hipMemcpy(ho,da,(size_t)N*sizeof(uint64_t),hipMemcpyDeviceToHost);
    if (even4) {
        uint64_t *ho_t = (uint64_t*)malloc((size_t)N * sizeof(uint64_t));
        for (int r = 0; r < M4; r++)
            for (int c = 0; c < M4; c++)
                ho_t[r * M4 + c] = ho[c * M4 + r];
        for (int k = 0; k < N; k++) ho[k] = ho_t[k];
        free(ho_t);
    }

    int errs=0;
    for (int i=0;i<N;i++)
        if (ho[i]!=ref[i] && ++errs<=3)
            fprintf(stderr,"  k=%d got=%llu want=%llu\n",i,
                    (unsigned long long)ho[i],(unsigned long long)ref[i]);
    /* 4-step Stockham fixed 2026-05-29 (Bailey transpose + reverse-order
     * inverse, ntt_kernel.hip). Convolution is now a TRUE DFT at all sizes,
     * so this is a hard assertion — any mismatch is a real regression. */
    if (!errs) pass(name); else fail(name,"convolution mismatch");

    (void)hipFree(da);(void)hipFree(db);
    (void)hipFree(d_tw_fwd);(void)hipFree(d_tw_inv);
    if(d_tw_cross)(void)hipFree(d_tw_cross);
    if(d_tw_cross_inv)(void)hipFree(d_tw_cross_inv);
    (void)hipStreamDestroy(s);
    free(ha);free(hb);free(ho);free(ref);
}

/* ─── Test 5: Garner small (N=8, residues=28) ───────────────────────────── */
static void test_garner_small(void)
{
    const int N = 8;
    GarnerConsts g = garner_precompute();

    uint64_t *d_r[4];
    uint64_t vals[8];
    int d, i;
    for (i = 0; i < N; i++) vals[i] = 28;

    for (d = 0; d < 4; d++) {
        (void)(hipSetDevice(d % g_n_devs));
        (void)(hipMalloc(&d_r[d], (size_t)N * sizeof(uint64_t)));
        (void)(hipMemcpy(d_r[d], vals, (size_t)N * sizeof(uint64_t), hipMemcpyHostToDevice));
        (void)(hipDeviceSynchronize());
    }

    U256* h_out = (U256*)malloc((size_t)N * sizeof(U256));
    if (!h_out) { fprintf(stderr, "test_garner_small: malloc failed\n"); exit(1); }
#ifdef GFX1030_LOCAL
    /* 6900 XT: device pointers not CPU-accessible (PCIe, not UMA). Copy to host. */
    uint64_t *h_r[4];
    for (d = 0; d < 4; d++) {
        h_r[d] = (uint64_t*)malloc((size_t)N * sizeof(uint64_t));
        (void)(hipMemcpy(h_r[d], d_r[d], (size_t)N * sizeof(uint64_t), hipMemcpyDeviceToHost));
    }
    garner_reconstruct(h_r[0], h_r[1], h_r[2], h_r[3], h_out, N, &g);
    for (d = 0; d < 4; d++) free(h_r[d]);
#else
    garner_reconstruct(d_r[0], d_r[1], d_r[2], d_r[3], h_out, N, &g);
#endif

    int errs = 0;
    for (i = 0; i < N; i++) {
        if (h_out[i].w[0] != 28 || h_out[i].w[1] != 0 ||
            h_out[i].w[2] != 0  || h_out[i].w[3] != 0) {
            if (++errs <= 3)
                fprintf(stderr, "  i=%d w0=%llu w1=%llu\n", i,
                        (unsigned long long)h_out[i].w[0],
                        (unsigned long long)h_out[i].w[1]);
        }
    }
    if (!errs) pass("garner_small (N=8, residues=28)");
    else       fail("garner_small", "mismatch");

    for (d = 0; d < 4; d++) (void)(hipFree(d_r[d]));
    free(h_out);
}

/* ─── Test 6: Garner known 128-bit value ─────────────────────────────────── */
static void test_garner_known(void)
{
    __uint128_t x128 = ((__uint128_t)UINT64_C(0x123456789abcdef0) << 64)
                     | UINT64_C(0xfedcba9876543210);

    uint64_t r[4];
    r[0] = (uint64_t)(x128 % P0);
    r[1] = (uint64_t)(x128 % P1);
    r[2] = (uint64_t)(x128 % P2);
    r[3] = (uint64_t)(x128 % P3);

    uint64_t *d_r[4];
    int d;
    for (d = 0; d < 4; d++) {
        (void)(hipSetDevice(d % g_n_devs));
        (void)(hipMalloc(&d_r[d], sizeof(uint64_t)));
        (void)(hipMemcpy(d_r[d], &r[d], sizeof(uint64_t), hipMemcpyHostToDevice));
        (void)(hipDeviceSynchronize());
    }

    GarnerConsts g = garner_precompute();
    U256 out;
#ifdef GFX1030_LOCAL
    /* 6900 XT: device pointers not CPU-accessible (PCIe, not UMA). Copy to host. */
    uint64_t h_r[4];
    for (d = 0; d < 4; d++)
        (void)(hipMemcpy(&h_r[d], d_r[d], sizeof(uint64_t), hipMemcpyDeviceToHost));
    garner_reconstruct(&h_r[0], &h_r[1], &h_r[2], &h_r[3], &out, 1, &g);
#else
    garner_reconstruct(d_r[0], d_r[1], d_r[2], d_r[3], &out, 1, &g);
#endif

    uint64_t x_lo = (uint64_t)x128;
    uint64_t x_hi = (uint64_t)(x128 >> 64);
    if (out.w[0] == x_lo && out.w[1] == x_hi && out.w[2] == 0 && out.w[3] == 0)
        pass("garner_known (128-bit value)");
    else
        fail("garner_known", "output does not match input");

    for (d = 0; d < 4; d++) (void)(hipFree(d_r[d]));
}

/* ─── Test 7: Garner CPU-only (no GPU required) ──────────────────────────── */
/* garner_reconstruct() dereferences its residue pointers via OMP on the CPU. */
/* On MI300A (UMA) device pointers are CPU-accessible; here we pass plain     */
/* malloc'd host arrays — the function is unaware of the difference.          */
/* Tests Garner math correctness independently of device count or UMA.        */
static void test_garner_cpu_only(void)
{
    __uint128_t x = ((__uint128_t)UINT64_C(0xdeadbeef12345678) << 64)
                  | UINT64_C(0xabcdef0123456789);
    uint64_t r0 = (uint64_t)(x % P0);
    uint64_t r1 = (uint64_t)(x % P1);
    uint64_t r2 = (uint64_t)(x % P2);
    uint64_t r3 = (uint64_t)(x % P3);

    GarnerConsts g = garner_precompute();
    U256 out;
    garner_reconstruct(&r0, &r1, &r2, &r3, &out, 1, &g);

    uint64_t x_lo = (uint64_t)x;
    uint64_t x_hi = (uint64_t)(x >> 64);
    if (out.w[0] == x_lo && out.w[1] == x_hi && out.w[2] == 0 && out.w[3] == 0)
        pass("garner_cpu_only (128-bit value, host pointers)");
    else
        fail("garner_cpu_only", "output does not match input");
}

/* ─── I15 Pipeline oracle ────────────────────────────────────────────────── */
/* Purpose: verify ntt_bigint_mul(c, a, b) == bigint_mul(c, a, b) for random
 * BigInt operands at several even log_n sizes (the Stockham production path).
 * This is the gate for I15 (transposed scatter/gather fold): any layout change
 * in scatter_kernel or bigint_gather that silently corrupts Garner or the CRT
 * product will cause a FAIL here, whereas test_ntt_convolution (NTT kernels
 * only) would not catch it.
 *
 * The oracle uses random limb values (LIMB_BITS-wide each) to produce inputs
 * whose product NTT log_n is sufficient.  ntt_bigint_mul dispatches at
 * max_log_n from ntt_mul_init; the test uses a modest limb count so the
 * expected log_n is even (Stockham path). */

extern "C" {
    void ntt_mul_init_impl(int max_log_n);
    void ntt_mul_teardown_impl(void);
    void ntt_bigint_mul(BigInt *c, const BigInt *a, const BigInt *b);
    int  bigint_bits(const BigInt *a);
}

static void test_pipeline_oracle(void)
{
    /* Tests span schoolbook fallback, NTT threshold, even log_n, and the I11
     * rect Stockham path (odd log_n > 10) through ntt_bigint_mul.  The 513-
     * and 700-limb cases produce n_coeffs=1026/1400 → log_n=11 (N=2048),
     * exercising the production I11 dispatch that was previously CT-DIT-Mont. */
    static const struct { int limbs; const char *label; } cases[] = {
        {  4, "4-limb (schoolbook fallback)"          },
        { 32, "32-limb (near NTT threshold)"          },
        { 64, "64-limb (NTT path, even log_n)"        },
        {513, "513-limb (I11 rect NTT, odd log_n=11)" },  /* na+nb=1026 → log_n=11 */
        {700, "700-limb (I11 rect NTT, odd log_n=11)" },  /* na+nb=1400 → log_n=11 */
        {  0, NULL }
    };

    /* ntt_mul_init: choose max_log_n large enough for 64*2=128 limbs product.
     * 128 limbs * LIMB_BITS bits/limb = 8192 bits → log_n ≥ ceil(log2(8192/LIMB_BITS)+1).
     * Use log_n=14 (N=16384 ≥ 128+1 coefficients at any LIMB_BITS ≥ 64). */
    ntt_mul_init_impl(14);

    for (int ci = 0; cases[ci].label; ci++) {
        int lim = cases[ci].limbs;
        char name[80];
        snprintf(name, sizeof(name),
                 "pipeline oracle: ntt_bigint_mul vs bigint_mul (%s)", cases[ci].label);

        /* Build random operands. */
        BigInt a = bigint_alloc(lim + 2);
        BigInt b = bigint_alloc(lim + 2);
        a.n_limbs = lim;
        b.n_limbs = lim;
        for (int i = 0; i < lim; i++) {
            a.limbs[i] = (limb_t)(rand64() & LIMB_MASK);
            b.limbs[i] = (limb_t)(rand64() & LIMB_MASK);
        }
        /* Ensure no leading zeros (trim). */
        while (a.n_limbs > 1 && a.limbs[a.n_limbs-1] == 0) a.n_limbs--;
        while (b.n_limbs > 1 && b.limbs[b.n_limbs-1] == 0) b.n_limbs--;

        /* Reference: schoolbook multiply. */
        BigInt ref = bigint_alloc(a.n_limbs + b.n_limbs + 2);
        bigint_mul(&ref, &a, &b);

        /* CRT-NTT multiply. */
        BigInt ntt = bigint_alloc(a.n_limbs + b.n_limbs + 4);
        ntt_bigint_mul(&ntt, &a, &b);

        if (bigint_cmp(&ref, &ntt) == 0) {
            pass(name);
        } else {
            fail(name, "ntt_bigint_mul != bigint_mul");
            fprintf(stderr,
                    "  bits_a=%d bits_b=%d bits_ref=%d bits_ntt=%d\n",
                    bigint_bits(&a), bigint_bits(&b),
                    bigint_bits(&ref), bigint_bits(&ntt));
        }

        bigint_free(&a); bigint_free(&b);
        bigint_free(&ref); bigint_free(&ntt);
    }

    ntt_mul_teardown_impl();
}

/* ─── I11 production stress test ────────────────────────────────────────────
 * Runs ntt_bigint_mul at log_n=19 (N=2^19) for N_ITER iterations.
 * All dispatches use the rect Stockham (I11) path since log_n=19 is odd and >10.
 * Correctness verified at every iteration via bigint_mul reference.
 * Purpose: detect cumulative GPU state corruption that single-shot tests miss. */
static void test_pipeline_stress_log19(int n_iter)
{
    printf("  [I11 stress] log_n=19  %d iterations ...\n", n_iter); fflush(stdout);

    ntt_mul_init_impl(19);

    /* Use 64-limb inputs: small enough for bigint_mul reference to be fast,
     * large enough to exercise a real multiply.  n_coeffs = 128 << 19=524288, so
     * the GPU buffers are fully exercised at N=2^19 with lots of zero padding. */
    static const int LIM = 64;
    int all_pass = 1;
    for (int iter = 0; iter < n_iter; iter++) {
        BigInt a = bigint_alloc(LIM + 2);
        BigInt b = bigint_alloc(LIM + 2);
        a.n_limbs = LIM;
        b.n_limbs = LIM;
        for (int i = 0; i < LIM; i++) {
            a.limbs[i] = (limb_t)(rand64() & LIMB_MASK);
            b.limbs[i] = (limb_t)(rand64() & LIMB_MASK);
        }
        while (a.n_limbs > 1 && a.limbs[a.n_limbs-1] == 0) a.n_limbs--;
        while (b.n_limbs > 1 && b.limbs[b.n_limbs-1] == 0) b.n_limbs--;

        BigInt ref = bigint_alloc(a.n_limbs + b.n_limbs + 2);
        bigint_mul(&ref, &a, &b);

        BigInt ntt = bigint_alloc(a.n_limbs + b.n_limbs + 4);
        ntt_bigint_mul(&ntt, &a, &b);

        if (bigint_cmp(&ref, &ntt) != 0) {
            char name[64];
            snprintf(name, sizeof(name),
                     "I11 stress log_n=19 iter %d", iter);
            fail(name, "ntt != ref");
            all_pass = 0;
        }

        bigint_free(&a); bigint_free(&b);
        bigint_free(&ref); bigint_free(&ntt);

        if ((iter + 1) % 10 == 0) {
            printf("  [I11 stress] %d/%d ok\n", iter + 1, n_iter);
            fflush(stdout);
        }
    }

    ntt_mul_teardown_impl();

    char name[80];
    snprintf(name, sizeof(name), "I11 stress log_n=19 x%d", n_iter);
    if (all_pass) pass(name);
}

/* ─── main ───────────────────────────────────────────────────────────────── */
int main(void)
{
    printf("=== CRT-NTT correctness test suite ===\n\n");

    (void)(hipGetDeviceCount(&g_n_devs));
    if (g_n_devs < 1) g_n_devs = 1;
    printf("Devices available: %d (tests use device PIDX %% %d)\n\n",
           g_n_devs, g_n_devs);

    printf("--- prime verification ---\n");
    if (!verify_primes()) { printf("FATAL: primes wrong, aborting.\n"); return 1; }
    printf("\n");

    printf("--- LDS index padding ---\n");
    test_lds_idx();
    test_stok_pad();
    printf("\n");

    printf("--- GPU reduction (100k tests each prime) ---\n");
    test_reduction<0>(100000);
    test_reduction<1>(100000);
    test_reduction<2>(100000);
    test_reduction<3>(100000);
    printf("\n");

    printf("--- Shoup multiply (100k tests each prime) ---\n");
    test_shoup_mul<0>(100000);
    test_shoup_mul<1>(100000);
    test_shoup_mul<2>(100000);
    test_shoup_mul<3>(100000);
    printf("\n");

    printf("--- NTT round-trip N=2^10 ---\n");
    test_ntt_roundtrip<0>(10);
    test_ntt_roundtrip<1>(10);
    test_ntt_roundtrip<2>(10);
    test_ntt_roundtrip<3>(10);
    printf("\n");

    /* 4-step localization: the Stockham 4-step (log_n>10) was only ever
     * asserted at log_n=20 (N1=N2=1024). compute_e's hot sizes are log_n
     * 11..19; the production swap hung at log_n=16. Cover every even
     * 4-step size to localize. Odd log_n>10 aborts by design (no square
     * decomposition), so only even sizes are testable here. */
    printf("--- NTT round-trip 4-step EVEN sizes (square, prime 0) ---\n");
    test_ntt_roundtrip<0>(12);
    test_ntt_roundtrip<0>(14);
    test_ntt_roundtrip<0>(16);
    test_ntt_roundtrip<0>(18);
    printf("\n");

    /* ODD-log_n rectangular rect path (I11) is DISABLED — it crashed the box
     * at production scale (compute_e d=1e6/log_n=19) and could not be
     * certified by code review. test_ntt does NOT exercise it, so check-gpu /
     * benchmarking never touch the uncertified path. Re-enable these only
     * after a bounded headless reproduction + a CONVOLUTION assert at every
     * odd size 11..19 (roundtrip alone is insufficient — it is identity for
     * any invertible op). See OPT_LEDGER.md I11. */

    /* Convolution = the production scenario (pointwise multiply in NTT
     * domain). Small 4-step sizes only (naive reference is O(N^2)). This is
     * what a roundtrip does NOT cover. */
    printf("--- NTT convolution (naive oracle; all primes 4-step) ---\n");
    test_ntt_convolution<0>(10);   /* single-block */
    test_ntt_convolution<0>(12);   /* 4-step even, prime 0 */
    test_ntt_convolution<0>(14);   /* 4-step even, prime 0 */
    test_ntt_convolution<1>(12);   /* prime 1 (prime-independence) */
    test_ntt_convolution<2>(12);   /* prime 2 */
    test_ntt_convolution<3>(12);   /* prime 3 */
    /* Odd-log_n rectangular Bailey 4-step (I11) convolution oracle.
     * log_n 11, 13: naive O(N^2) oracle — fast (N<=8192, <1s).
     * log_n 15:     naive O(N^2) oracle — slow (~5s); certifies the kernel
     *               at a size the original crash review missed.
     * log_n 17, 19: CT-DIT comparison — O(N) fast; proves rect gives the
     *               same polynomial product as the proven CT-DIT path.
     * These all run the rect KERNEL (launch_stockham_ntt_rect); they do NOT
     * exercise the production ntt_bigint_mul odd-log_n dispatch (which still
     * rounds up to even). Pass here => kernel arithmetic correct; production
     * re-enable still requires a bounded headless production-scale test. */
    printf("--- I11 rect NTT oracle (odd log_n; re-certification) ---\n");
    test_ntt_convolution<0>(11);   /* naive oracle, N=2048 */
    test_ntt_convolution<0>(13);   /* naive oracle, N=8192 */
    /* log_n 15/17/19: CT-DIT-Mont reference (I4) vs rect — O(N) fast.
     * Certifies that the rect kernel produces the correct DFT at every odd
     * size.  Rect is STILL DISABLED in ntt_bigint_mul production dispatch
     * (OPT_LEDGER I11 / ROADMAP R7); passing here proves kernel arithmetic
     * only.  Re-enable production dispatch after bounded headless test. */
    test_ntt_rect_vs_ctdit<0>(15); /* N=32768  */
    test_ntt_rect_vs_ctdit<0>(17); /* N=131072 */
    test_ntt_rect_vs_ctdit<0>(19); /* N=524288 — PIDX=0 uses Solinas path */
    /* PIDX=1,2,3 at log_n=19: exercise Shoup path at LOG_N_SUB=10 (never tested
     * before; production uses all 4 primes but check-gpu only covered PIDX=0). */
    test_ntt_rect_vs_ctdit<1>(19);
    test_ntt_rect_vs_ctdit<2>(19);
    test_ntt_rect_vs_ctdit<3>(19);
    printf("\n");

    printf("--- NTT round-trip N=2^20 ---\n");
    test_ntt_roundtrip<0>(20);
    test_ntt_roundtrip<1>(20);
    test_ntt_roundtrip<2>(20);
    test_ntt_roundtrip<3>(20);
    printf("\n");

    printf("--- Garner CRT reconstruction (GPU arrays) ---\n");
    test_garner_small();
    test_garner_known();
    printf("\n");

    printf("--- Garner CRT reconstruction (CPU host arrays, no GPU required) ---\n");
    test_garner_cpu_only();
    printf("\n");

    /* ── I15 PIPELINE ORACLE ────────────────────────────────────────────────
     * Test the FULL CRT-NTT pipeline (ntt_bigint_mul) against the schoolbook
     * reference (bigint_mul) on random BigInts. This catches:
     *   (a) any current regression in the pipeline — should always be green.
     *   (b) any layout mismatch from a future I15 transposed-scatter/gather
     *       fold — because test_ntt_convolution ONLY exercises the per-prime
     *       NTT kernels, NOT the scatter/Garner/gather boundary.
     * Runs EVEN log_n only (the Stockham production path); CT-DIT (odd log_n)
     * is exercised by compute_e digit-verification. */
    printf("--- Pipeline oracle: ntt_bigint_mul vs bigint_mul (even + odd I11 log_n) ---\n");
    test_pipeline_oracle();
    printf("\n");

    /* I11 production stress: 1000 iterations at log_n=19 via the full pipeline.
     * compute_e at d=1M does ~3348 dispatches; 1000 covers ~30% of that load.
     * Detects cumulative GPU instability that single-shot oracle tests miss. */
    printf("--- I11 production stress: log_n=19 x1000 ---\n");
    test_pipeline_stress_log19(1000);
    printf("\n");

    printf("=== All tests complete ===\n");
    return 0;
}
