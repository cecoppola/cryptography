/*
 * ntt_gpu_determinism.hip — GPU determinism / repeatability self-test
 *
 * Purpose:   Catch latent nondeterminism and data races in the ref reference
 *            GPU NTT kernels that pass/fail correctness tests miss. A kernel
 *            can be functionally correct on average yet produce
 *            run-to-run-variable output if it has an unsynchronised race, a
 *            read-modify-write hazard, or relies on undefined launch ordering.
 *            Correctness tests compare against a tolerance or a single CPU
 *            reference and so are blind to "right on average, jittery in the
 *            low bits". This test asserts BYTE-exact reproducibility instead.
 *
 * Tests:
 *   R1  Forward repeatability: GPU forward NTT of one fixed seeded input,
 *       run REPEATS times, must be byte-identical (memcmp) to run 0 every
 *       time. Any difference is a nondeterministic kernel — a real bug.
 *   R2  Round-trip repeatability + identity: GPU INTT(NTT(x)) run REPEATS
 *       times must (a) be byte-identical across runs and (b) equal the
 *       original x exactly. Exercises forward, inverse and the n^{-1} scale
 *       under repetition.
 *   R3  Cross-algorithm agreement: CT-DIT and Stockham GPU kernels compute
 *       the SAME mathematical transform, so for identical (x,n,q,omega)
 *       their forward-NTT outputs must be byte-identical. A mismatch is a
 *       genuine bug in one of the two kernels (it is not a tolerance issue —
 *       both use exact modular integer arithmetic).
 *
 * Why byte-exact is the right bar: every kernel here uses exact integer
 * modular arithmetic (no floating point, no atomics, no shared accumulators).
 * A correct, race-free implementation is therefore a pure function of its
 * inputs and MUST be bit-reproducible across launches and across the two
 * algorithms. Any observed variance localises a real defect.
 *
 * Both ref GPU sources (ntt_gpu.hip, ntt_gpu_stockham.hip) export the same
 * public symbols (ntt_forward/ntt_inverse/main) and so cannot be linked
 * together. Following the ntt_cross_verify.hip pattern, the relevant kernels
 * and host drivers are inlined here under distinct prefixes (ctdit_ / stok_)
 * so a single translation unit can drive both.
 *
 * Safety cap: n is capped at NTT_N_CAP (4096). The 6900XT dev box has a ~2 s
 * gfx-ring watchdog (the S7 hard-lock regime); a determinism run launches the
 * whole NTT REPEATS times, so a large n multiplies kernel time by REPEATS and
 * risks tripping the watchdog / locking the box. 4096 keeps every individual
 * launch tiny while still covering all production sizes (ML-KEM n=256,
 * ML-DSA n=256, n=512). Do NOT raise this cap without re-checking watchdog
 * headroom on the dev hardware.
 *
 * Build:  hipcc -O2 -Wall -Wextra --offload-arch=gfx1030 \
 *               -o ntt_gpu_determinism ntt_gpu_determinism.hip
 * Run:    ./ntt_gpu_determinism [n] [q] [omega] [repeats]
 *         defaults: n=256 q=3329 omega=17 repeats=50  (ML-KEM / FIPS 203)
 * Ref:    ntt_gpu.hip (CT-DIT), ntt_gpu_stockham.hip (Stockham).
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ── ANSI ────────────────────────────────────────────────────────────────── */
#define WHT "\033[1;37m"
#define CYN "\033[1;36m"
#define GRN "\033[1;32m"
#define YLW "\033[1;33m"
#define RED "\033[1;31m"
#define RST "\033[0m"

/* ── Error-checking macro ────────────────────────────────────────────────────
 * Aborts the host process on any failed HIP call so a bad alloc/copy can
 * never reach a kernel launch (an unchecked error there would corrupt the
 * very determinism signal this test measures). */
#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t _e = (call);                                             \
        if (_e != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s:%d — %s\n",                      \
                    __FILE__, __LINE__, hipGetErrorString(_e));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* Watchdog/S7-safety cap on transform length (see file header). */
#define NTT_N_CAP 4096

/* ── Shared parameter struct (matches ntt.h layout) ──────────────────────── */
typedef struct {
    uint64_t n;
    uint64_t q;
    uint64_t omega;
    uint64_t omega_inv;
    uint64_t n_inv;
    uint32_t log2_n;
} ntt_params_t;

/* Launch block size derived from runtime warpSize (filled in main). */
static int g_block_size = 128;

/* ── Kernel launch bounds (match the reference kernels) ───────────────────── */
#define NTT_MAX_BLOCK  256
#define NTT_MIN_BLOCKS    1

/* ═══════════════════════════════════════════════════════════════════════════
 * HOST MODULAR ARITHMETIC (twiddle precompute, root validation)
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint64_t host_mod_pow(uint64_t base, uint64_t exp, uint64_t m)
{
    uint64_t r = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) r = (uint64_t)((__uint128_t)r * base % m);
        base = (uint64_t)((__uint128_t)base * base % m);
        exp >>= 1;
    }
    return r;
}

static uint64_t host_mod_inv(uint64_t a, uint64_t m)
{
    return host_mod_pow(a, m - 2, m);   /* Fermat: m prime */
}

/*
 * root_is_primitive: check omega is a primitive n-th root of unity mod q.
 * Conditions: omega^n == 1 (mod q) and omega^(n/2) != 1 (mod q). The second
 * condition rules out lower-order roots that would still satisfy the first.
 * Inputs: n (power of 2), q (prime), omega in [1,q). Returns 1 if valid.
 */
static int root_is_primitive(uint64_t n, uint64_t q, uint64_t omega)
{
    if (omega == 0) return 0;
    if (host_mod_pow(omega, n, q) != 1) return 0;
    if (host_mod_pow(omega, n >> 1, q) == 1) return 0;
    return 1;
}

/*
 * alloc_twiddles: tw[k] = root^k mod q for k in [0, n/2). Caller frees.
 * Identical recurrence to ntt_gpu.hip / ntt_gpu_stockham.hip.
 */
static uint64_t *alloc_twiddles(const ntt_params_t *p, uint64_t root)
{
    uint64_t half = p->n >> 1;
    uint64_t *tw  = (uint64_t *)malloc(half * sizeof *tw);
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < half; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k - 1] * root % p->q);
    return tw;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CT-DIT GPU KERNELS  (inlined from ntt_gpu.hip, prefix ctdit_)
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * ctdit_bitrev_kernel: in-place bit-reversal permutation.
 * One thread per element i in [0,n); swaps a[i] with a[reverse(i)] once.
 */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void ctdit_bitrev_kernel(uint64_t * __restrict__ a, uint64_t n, uint32_t log2_n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t rev = 0, x = i;
    for (uint32_t b = 0; b < log2_n; b++) { rev = (rev << 1) | (x & 1); x >>= 1; }
    if (i < rev) { uint64_t t = a[i]; a[i] = a[rev]; a[rev] = t; }
}

/*
 * ctdit_stage_kernel: one radix-2 CT-DIT butterfly stage, per-stage reduction.
 * One thread per butterfly pair (n/2 threads). Safe for q < 2^32.
 */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void ctdit_stage_kernel(uint64_t * __restrict__ a,
                         const uint64_t * __restrict__ tw,
                         uint64_t n, uint64_t q, uint64_t half, uint64_t stride)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n >> 1) return;
    uint64_t len = half << 1;
    uint64_t k = (tid / half) * len, j = tid % half;
    uint64_t u = a[k + j], v = a[k + j + half];
    uint64_t t = (tw[j * stride] * (v % q)) % q;
    uint64_t s = u + t, d = u + q - t;
    a[k + j]        = s >= q ? s - q : s;
    a[k + j + half] = d >= q ? d - q : d;
}

/*
 * ctdit_scale_kernel: a[i] <- a[i] * sc mod q (INTT 1/n scaling).
 * One thread per element.
 */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void ctdit_scale_kernel(uint64_t * __restrict__ a, uint64_t n,
                         uint64_t sc, uint64_t q)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = (a[i] * sc) % q;
}

/*
 * ctdit_ntt: drive the CT-DIT GPU NTT on a host array.
 * Inputs:  a_host (in/out), tw_host (read-only twiddles), p.
 *          scale: 0/1 = forward (no scale), else multiply by scale (INTT).
 * Output:  a_host holds the transform result. Every HIP call is checked.
 */
static void ctdit_ntt(uint64_t *a_host, const uint64_t *tw_host,
                      const ntt_params_t *p, uint64_t scale)
{
    uint64_t n = p->n, q = p->q, hn = n >> 1;
    int bsz = g_block_size;
    uint64_t *da = NULL, *dtw = NULL;

    HIP_CHECK(hipMalloc((void **)&da,  n  * sizeof *da));
    HIP_CHECK(hipMalloc((void **)&dtw, hn * sizeof *dtw));
    HIP_CHECK(hipMemcpy(da,  a_host,  n  * sizeof *da,  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dtw, tw_host, hn * sizeof *dtw, hipMemcpyHostToDevice));

    int gn = (int)((n  + (uint64_t)bsz - 1) / (uint64_t)bsz);
    int gh = (int)((hn + (uint64_t)bsz - 1) / (uint64_t)bsz);

    hipLaunchKernelGGL(ctdit_bitrev_kernel, gn, bsz, 0, 0, da, n, p->log2_n);
    HIP_CHECK(hipGetLastError());
    for (uint64_t len = 2; len <= n; len <<= 1) {
        uint64_t half = len >> 1, stride = n / len;
        hipLaunchKernelGGL(ctdit_stage_kernel, gh, bsz, 0, 0,
                           da, dtw, n, q, half, stride);
        HIP_CHECK(hipGetLastError());
    }
    if (scale != 0 && scale != 1) {
        hipLaunchKernelGGL(ctdit_scale_kernel, gn, bsz, 0, 0, da, n, scale, q);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(a_host, da, n * sizeof *da, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(da));
    HIP_CHECK(hipFree(dtw));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * STOCKHAM GPU KERNELS  (inlined from ntt_gpu_stockham.hip, prefix stok_)
 * Multi-launch per-stage path: one kernel per stage, ping-pong buffers,
 * then a reduce pass and (for INTT) a scale pass. This is the
 * non-LDS-fused public Stockham transform and is the simplest deterministic
 * formulation; its mathematical output equals the CT-DIT output.
 * ═══════════════════════════════════════════════════════════════════════════ */

__device__ __forceinline__
uint64_t stok_dev_mulmod(uint64_t a, uint64_t b, uint64_t q)
{
    return (a * b) % q;   /* a,b < q < 2^32 → product < 2^64 */
}

/*
 * stok_stage_kernel: one Stockham DIT butterfly stage (lazy add, no reduce).
 * One thread per butterfly (n/2 threads). src→dst, no in-place hazard.
 */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void stok_stage_kernel(const uint64_t * __restrict__ src,
                              uint64_t * __restrict__ dst,
                        const uint64_t * __restrict__ tw,
                        uint64_t n, uint64_t q, uint32_t s)
{
    uint64_t tid  = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = n >> 1;
    if (tid >= half) return;
    uint64_t p  = (uint64_t)1 << s;
    uint64_t qs = n >> (s + 1);
    uint64_t p2 = p << 1;
    uint64_t j  = tid & (p - 1);
    uint64_t k  = tid >> s;
    uint64_t idx_u = j + k * p;
    uint64_t idx_v = idx_u + half;
    uint64_t u  = src[idx_u];
    uint64_t v  = src[idx_v];
    uint64_t w  = tw[j * qs];
    uint64_t wv = stok_dev_mulmod(w, v % q, q);
    dst[j + k * p2]     = u + wv;
    dst[j + k * p2 + p] = u + q - wv;
}

/*
 * stok_reduce_kernel: final a[i] %= q after the lazy-add stages.
 * One thread per element.
 */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void stok_reduce_kernel(uint64_t * __restrict__ a, uint64_t n, uint64_t q)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] %= q;
}

/*
 * stok_scale_kernel: a[i] <- a[i] * n_inv mod q (INTT scaling).
 * One thread per element; a[i] already reduced to [0,q).
 */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void stok_scale_kernel(uint64_t * __restrict__ a, uint64_t n,
                        uint64_t q, uint64_t n_inv)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = stok_dev_mulmod(a[i], n_inv, q);
}

/*
 * stok_ntt: drive the multi-launch Stockham GPU NTT on a host array.
 * Inputs:  a_host (in/out), tw_host (read-only twiddles), p.
 *          is_inverse: 0 = forward, 1 = inverse (adds the scale pass).
 * Output:  a_host holds the natural-order transform result.
 */
static void stok_ntt(uint64_t *a_host, const uint64_t *tw_host,
                      const ntt_params_t *p, int is_inverse)
{
    uint64_t n = p->n, half = n >> 1;
    int bsz = g_block_size;
    int gsz_half = (int)((half + (uint64_t)bsz - 1) / (uint64_t)bsz);
    int gsz_full = (int)((n    + (uint64_t)bsz - 1) / (uint64_t)bsz);

    uint64_t *d_a = NULL, *d_b = NULL, *d_tw = NULL;
    HIP_CHECK(hipMalloc((void **)&d_a,  n    * sizeof *d_a));
    HIP_CHECK(hipMalloc((void **)&d_b,  n    * sizeof *d_b));
    HIP_CHECK(hipMalloc((void **)&d_tw, half * sizeof *d_tw));
    HIP_CHECK(hipMemcpy(d_a,  a_host,  n    * sizeof *d_a,  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tw, tw_host, half * sizeof *d_tw, hipMemcpyHostToDevice));

    uint64_t *d_src = d_a, *d_dst = d_b;
    for (uint32_t s = 0; s < p->log2_n; s++) {
        hipLaunchKernelGGL(stok_stage_kernel, gsz_half, bsz, 0, 0,
                           d_src, d_dst, d_tw, n, p->q, s);
        HIP_CHECK(hipGetLastError());
        uint64_t *tmp = d_src; d_src = d_dst; d_dst = tmp;
    }
    hipLaunchKernelGGL(stok_reduce_kernel, gsz_full, bsz, 0, 0,
                       d_src, n, p->q);
    HIP_CHECK(hipGetLastError());
    if (is_inverse) {
        hipLaunchKernelGGL(stok_scale_kernel, gsz_full, bsz, 0, 0,
                           d_src, n, p->q, p->n_inv);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(a_host, d_src, n * sizeof *d_a, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_tw));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PARAMETER INIT
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * params_init: validate and derive parameters.
 * Returns  0 ok; -1 n not pow2; -2 q out of [2, 2^32); -3 n over safety cap;
 *         -4 omega not a primitive n-th root mod q.
 */
static int params_init(ntt_params_t *p)
{
    if (!p->n || (p->n & (p->n - 1))) return -1;
    if (p->q < 2 || p->q > 0xFFFFFFFFULL) return -2;  /* q<2: host_mod_pow %0 → SIGFPE */
    if (p->n > NTT_N_CAP)              return -3;
    if (!root_is_primitive(p->n, p->q, p->omega)) return -4;
    p->omega_inv = host_mod_inv(p->omega, p->q);
    p->n_inv     = host_mod_inv(p->n,     p->q);
    uint64_t t = p->n; p->log2_n = 0;
    while (t > 1) { t >>= 1; p->log2_n++; }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST FRAMEWORK
 * ═══════════════════════════════════════════════════════════════════════════ */

static int total_pass = 0, total_fail = 0;

static void print_result(const char *name, int pass, const char *detail)
{
    const char *tag = pass ? GRN "PASS" RST : RED "FAIL" RST;
    printf("  │ %-36s │ %s │ %-26s │\n", name, tag, detail);
}

/* Deterministic seeded fill: pure function of (i, seed, q). */
static void fill_seeded(uint64_t *x, uint64_t n, uint64_t q, uint64_t seed)
{
    for (uint64_t i = 0; i < n; i++)
        x[i] = (i * 6364136223846793005ULL + seed) % q;
}

/*
 * R1 — forward repeatability.
 * Run the CT-DIT forward NTT of one fixed input REPEATS times; every run
 * must be byte-identical (memcmp) to run 0. A correct race-free integer
 * kernel is a pure function of its input, so any byte difference proves
 * nondeterminism (race / RMW hazard / launch-order dependence).
 */
static void test_r1_forward_repeat(const ntt_params_t *p,
                                    const uint64_t *tw, uint64_t repeats)
{
    uint64_t n = p->n;
    uint64_t *x0  = (uint64_t *)malloc(n * sizeof *x0);
    uint64_t *ref = (uint64_t *)malloc(n * sizeof *ref);
    uint64_t *cur = (uint64_t *)malloc(n * sizeof *cur);
    if (!x0 || !ref || !cur) { free(x0); free(ref); free(cur); return; }

    fill_seeded(x0, n, p->q, 0xA5A5ULL);
    memcpy(ref, x0, n * sizeof *ref);
    ctdit_ntt(ref, tw, p, 0);

    uint64_t bad_run = 0;
    int ok = 1;
    for (uint64_t r = 0; r < repeats && ok; r++) {
        memcpy(cur, x0, n * sizeof *cur);
        ctdit_ntt(cur, tw, p, 0);
        if (memcmp(cur, ref, n * sizeof *cur) != 0) { ok = 0; bad_run = r; }
    }

    char det[64];
    if (ok) snprintf(det, sizeof det, "%lu runs byte-identical", repeats);
    else    snprintf(det, sizeof det, "diverged at run %lu", bad_run);
    print_result("R1 fwd repeatability (CT-DIT)", ok, det);
    ok ? total_pass++ : total_fail++;
    free(x0); free(ref); free(cur);
}

/*
 * R2 — round-trip repeatability + identity.
 * INTT(NTT(x)) run REPEATS times must be byte-identical across runs AND
 * equal the original x. Repetition surfaces nondeterminism in the forward,
 * inverse and 1/n-scale paths together; the identity check additionally
 * confirms the repeatable result is the correct one (a stuck-but-stable
 * kernel would pass R1's repeatability alone).
 */
static void test_r2_roundtrip_repeat(const ntt_params_t *p,
                                      const uint64_t *tw, const uint64_t *twi,
                                      uint64_t repeats)
{
    uint64_t n = p->n;
    uint64_t *orig = (uint64_t *)malloc(n * sizeof *orig);
    uint64_t *ref  = (uint64_t *)malloc(n * sizeof *ref);
    uint64_t *cur  = (uint64_t *)malloc(n * sizeof *cur);
    if (!orig || !ref || !cur) { free(orig); free(ref); free(cur); return; }

    fill_seeded(orig, n, p->q, 0x1234ULL);

    memcpy(ref, orig, n * sizeof *ref);
    ctdit_ntt(ref, tw,  p, 0);
    ctdit_ntt(ref, twi, p, p->n_inv);

    int ident_ref = (memcmp(ref, orig, n * sizeof *ref) == 0);
    int stable = 1;
    uint64_t bad_run = 0;
    for (uint64_t r = 0; r < repeats && stable; r++) {
        memcpy(cur, orig, n * sizeof *cur);
        ctdit_ntt(cur, tw,  p, 0);
        ctdit_ntt(cur, twi, p, p->n_inv);
        if (memcmp(cur, ref, n * sizeof *cur) != 0) { stable = 0; bad_run = r; }
    }

    int ok = stable && ident_ref;
    char det[64];
    if (!stable)          snprintf(det, sizeof det, "unstable at run %lu", bad_run);
    else if (!ident_ref)  snprintf(det, sizeof det, "stable but != x");
    else                  snprintf(det, sizeof det, "%lu runs == x, identical", repeats);
    print_result("R2 round-trip repeat+identity", ok, det);
    ok ? total_pass++ : total_fail++;
    free(orig); free(ref); free(cur);
}

/*
 * R3 — cross-algorithm agreement.
 * CT-DIT and Stockham implement the same forward NTT, so for identical
 * (x,n,q,omega) their outputs must be byte-identical (both use exact
 * modular integer arithmetic — there is no rounding/tolerance excuse for
 * any mismatch). Repeated to also catch a kernel that disagrees only
 * intermittently.
 */
static void test_r3_cross_algo(const ntt_params_t *p,
                                const uint64_t *tw, uint64_t repeats)
{
    uint64_t n = p->n;
    uint64_t *x0 = (uint64_t *)malloc(n * sizeof *x0);
    uint64_t *ca = (uint64_t *)malloc(n * sizeof *ca);
    uint64_t *sa = (uint64_t *)malloc(n * sizeof *sa);
    if (!x0 || !ca || !sa) { free(x0); free(ca); free(sa); return; }

    fill_seeded(x0, n, p->q, 0xCAFEULL);

    int ok = 1;
    uint64_t bad_run = 0, first_bad = 0;
    for (uint64_t r = 0; r < repeats && ok; r++) {
        memcpy(ca, x0, n * sizeof *ca);
        memcpy(sa, x0, n * sizeof *sa);
        ctdit_ntt(ca, tw, p, 0);
        stok_ntt (sa, tw, p, 0);
        if (memcmp(ca, sa, n * sizeof *ca) != 0) {
            ok = 0; bad_run = r;
            for (uint64_t i = 0; i < n; i++)
                if (ca[i] != sa[i]) { first_bad = i; break; }
        }
    }

    char det[64];
    if (ok) snprintf(det, sizeof det, "%lu runs identical", repeats);
    else    snprintf(det, sizeof det, "run %lu mismatch @%lu", bad_run, first_bad);
    print_result("R3 CT-DIT vs Stockham (fwd)", ok, det);
    ok ? total_pass++ : total_fail++;
    free(x0); free(ca); free(sa);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * main: query GPU, derive block size from warpSize, run R1/R2/R3, print a
 * fixed-width table, append a one-line summary to det_<timestamp>.txt.
 * Usage:   ntt_gpu_determinism [n] [q] [omega] [repeats]
 * Defaults: n=256 q=3329 omega=17 repeats=50  (ML-KEM / FIPS 203).
 *
 * Curated default + alternative (n,q,omega) triples (all validated at
 * runtime by root_is_primitive — a bad CLI omega is rejected):
 *   n=256 q=3329    omega=17       ML-KEM / FIPS 203 prime (13*256+1).
 *   n=256 q=8380417 omega=5801164  ML-DSA prime; primitive 256th root
 *                                  (= g^((q-1)/256), g=5).
 *   n=512 q=12289   omega=3400     NewHope/Kyber-family prime (3*2^12+1);
 *                                  primitive 512th root (= g^((q-1)/512),
 *                                  g=11).
 */
int main(int argc, char *argv[])
{
    ntt_params_t p;
    p.n     = (argc > 1) ? (uint64_t)atol(argv[1]) : 256;
    p.q     = (argc > 2) ? (uint64_t)atol(argv[2]) : 3329;
    p.omega = (argc > 3) ? (uint64_t)atol(argv[3]) : 17;
    uint64_t repeats = (argc > 4) ? (uint64_t)atol(argv[4]) : 50;
    if (repeats == 0) repeats = 1;

    printf("\n" WHT
           "╔══════════════════════════════════════════════════════════╗\n"
           "║       NTT / MI300A  —  GPU Determinism Self-Test        ║\n"
           "╚══════════════════════════════════════════════════════════╝\n"
           RST "\n");

    int rc = params_init(&p);
    if (rc != 0) {
        const char *why =
            rc == -1 ? "n is not a power of 2" :
            rc == -2 ? "q must satisfy 2 <= q < 2^32 (GPU kernels)" :
            rc == -3 ? "n exceeds watchdog-safe cap (4096)" :
                       "omega is not a primitive n-th root mod q";
        fprintf(stderr, "error: %s  (n=%lu q=%lu omega=%lu)\n",
                why, p.n, p.q, p.omega);
        return 1;
    }

    /* Query GPU; choose block size as a multiple of warpSize, capped at
     * NTT_MAX_BLOCK so every kernel's __launch_bounds__ contract holds.
     * RDNA2 (6900XT) warp=32 → 128; CDNA3 (MI300A) warp=64 → 256. */
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    g_block_size = 4 * prop.warpSize;
    if (g_block_size > NTT_MAX_BLOCK) g_block_size = NTT_MAX_BLOCK;

    printf(CYN "  Device: " RST "%s  warp=%d  CUs=%d  block=%d\n",
           prop.name, prop.warpSize, prop.multiProcessorCount, g_block_size);
    printf(CYN "  Params: " RST "n=%lu  q=%lu  omega=%lu  repeats=%lu\n\n",
           p.n, p.q, p.omega, repeats);

    uint64_t *tw  = alloc_twiddles(&p, p.omega);
    uint64_t *twi = alloc_twiddles(&p, p.omega_inv);
    if (!tw || !twi) { fprintf(stderr, "alloc failed\n"); free(tw); free(twi); return 1; }

    printf("  ┌──────────────────────────────────────┬──────┬────────────────────────────┐\n");
    printf("  │ " CYN "%-36s" RST " │ " CYN "%-4s" RST " │ " CYN "%-26s" RST " │\n",
           "Test", "Res.", "Detail");
    printf("  ├──────────────────────────────────────┼──────┼────────────────────────────┤\n");

    test_r1_forward_repeat(&p, tw, repeats);
    test_r2_roundtrip_repeat(&p, tw, twi, repeats);
    test_r3_cross_algo(&p, tw, repeats);

    printf("  └──────────────────────────────────────┴──────┴────────────────────────────┘\n\n");
    printf("  Results: " GRN "%d passed" RST "  " RED "%d failed" RST "\n\n",
           total_pass, total_fail);

    /* Append a one-line summary to a timestamped file (project rule 5). */
    char fname[64];
    time_t now = time(NULL);
    struct tm *tmv = localtime(&now);
    strftime(fname, sizeof fname, "det_%Y%m%d_%H%M%S.txt", tmv);
    FILE *f = fopen(fname, "a");
    if (f) {
        fprintf(f,
            "det device=%s n=%lu q=%lu omega=%lu repeats=%lu "
            "pass=%d fail=%d result=%s\n",
            prop.name, p.n, p.q, p.omega, repeats,
            total_pass, total_fail, total_fail ? "FAIL" : "PASS");
        fclose(f);
        printf("  Summary appended to %s\n\n", fname);
    }

    free(tw); free(twi);
    return total_fail ? 1 : 0;
}
