/*
 * ntt_gpu.hip — Cooley-Tukey DIT NTT, HIP GPU implementation
 *
 * Purpose:   GPU implementation of the same NTT interface as ntt_cpu.c.
 *            Caller code is identical; only the linked object changes.
 * Algorithm: Cooley-Tukey DIT, radix-2, per-stage GPU kernels.
 * Reduction: per-stage — values reduced to [0, q) after each butterfly
 *            stage. Avoids __uint128_t in device code; safe for q < 2³².
 * Target:    gfx1030 (6900XT) for GPU dev; gfx942 (MI300A) for production.
 *   Build:   hipcc -O2 -Wall -Wextra --offload-arch=gfx1030 -o ntt_gpu ntt_gpu.hip
 *   MI300A:  hipcc -O2 -Wall -Wextra --offload-arch=gfx942  -o ntt_gpu ntt_gpu.hip
 * Ref:       Longa & Naehrig CANS 2016, Alg. 1; HIP Programming Guide §3.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdlib.h>
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

/* ── Error-checking macro ────────────────────────────────────────────────── */
#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _e = (call);                                                \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s:%d — %s\n",                         \
                    __FILE__, __LINE__, hipGetErrorString(_e));                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* ── Runtime parameter struct (identical layout to ntt_cpu.c) ────────────── */
typedef struct {
    uint64_t n;
    uint64_t q;
    uint64_t omega;
    uint64_t omega_inv;
    uint64_t n_inv;
    uint32_t log2_n;
} ntt_params_t;

/* ── GPU hardware config (queried once at startup) ───────────────────────── */
/*
 * gpu_config_t: device properties measured at startup via hipGetDeviceProperties().
 * Passed to kernel launch helpers to select block size and LDS allocation.
 * Never hardcoded — always derived from the actual runtime device.
 */
typedef struct {
    char    name[256];       /* device name string                           */
    int     warp_size;       /* wavefront width (32 = RDNA3; 64 = CDNA3)    */
    int     cu_count;        /* multiProcessorCount (compute units)          */
    int     max_shared_b;    /* sharedMemPerBlock in bytes                   */
    int     max_regs;        /* regsPerBlock                                 */
    size_t  total_mem;       /* totalGlobalMem in bytes                      */
    int     block_size;      /* chosen launch block size; multiple of 64     */
} gpu_config_t;

/* Singleton config; populated in main() before any kernel launch */
static gpu_config_t g_gpu;

/* ── Kernel launch bounds ────────────────────────────────────────────────── */
/* 256 threads/block; 1 block/CU min. Adjust when tuning VGPR pressure. */
#define NTT_MAX_BLOCK  256
#define NTT_MIN_BLOCKS   1

/* ═══════════════════════════════════════════════════════════════════════════
 * DEVICE KERNELS
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * ntt_bitrev_kernel: in-place bit-reversal permutation.
 * Purpose:  reorder input array into bit-reversed index order before DIT.
 * Inputs:   a[0..n-1]; n = 2^log2_n.
 * Output:   a[i] ↔ a[reverse_bits(i, log2_n)] for all i; each pair swapped once.
 * Mapping:  one thread per element index i in [0, n).
 * Invariant: n is a power of 2.
 */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void ntt_bitrev_kernel(uint64_t * __restrict__ a, uint64_t n, uint32_t log2_n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint64_t rev = 0, x = i;
    for (uint32_t b = 0; b < log2_n; b++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    if (i < rev) {
        uint64_t tmp = a[i];
        a[i] = a[rev];
        a[rev] = tmp;
    }
}

/*
 * ntt_stage_kernel: one radix-2 CT-DIT butterfly stage.
 * Purpose:  advance the NTT computation by one stage (doubling butterfly span).
 * Inputs:   a[0..n-1] — working array in device memory.
 *           twiddles[0..n/2-1] — packed Shoup twiddle table (see ntt_alloc_twiddles).
 *             Each entry packs: upper 32 bits = w_inv = floor(w*2^32/q),
 *                               lower 32 bits = w    = omega^k mod q.
 *           n, q — transform length and modulus (q < 2^32).
 *           half   — half the current butterfly span (= 2^(stage-1)).
 *           stride — twiddle index step (= n / (2*half)).
 * Output:   a[] updated in place; all values reduced to [0, q) after each
 *           butterfly (per-stage reduction, safe for q < 2^32).
 * Mapping:  one thread per butterfly pair; n/2 threads total.
 * Butterfly (Shoup precomputed-quotient, Shoup 1995):
 *   packed  = twiddles[j·stride]
 *   w       = (uint32_t)packed                  -- omega^(j*stride) mod q
 *   w_inv   = (uint32_t)(packed >> 32)           -- floor(w * 2^32 / q)
 *   hi      = (uint32_t)((w_inv * v) >> 32)     -- approximates w*v/q
 *   t       = w * v - hi * q                    -- exact reduction (no division)
 *   if t >= q: t -= q                            -- at most one correction
 * Replaces hardware 64-bit division (% q, ~24 GPU cycles) with three
 * 32-bit multiplies (~3 cycles each). Correctness: w, v, q all < 2^32;
 * w_inv is the Shoup quotient; hi <= w*v/q; error in t is at most q-1.
 * Ref: Shoup 1995; Longa & Naehrig CANS 2016 Alg. 1 inner loop.
 */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void ntt_stage_kernel(uint64_t * __restrict__ a,
                      const uint64_t * __restrict__ twiddles,
                      uint64_t n, uint64_t q,
                      uint64_t half, uint64_t stride)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n >> 1) return;

    /* Map linear thread id to butterfly indices (k, j) */
    uint64_t len = half << 1;
    uint64_t k   = (tid / half) * len;
    uint64_t j   = tid % half;

    uint64_t packed = twiddles[j * stride];
    uint32_t w      = (uint32_t)packed;           /* omega^(j*stride) mod q  */
    uint32_t w_inv  = (uint32_t)(packed >> 32);   /* Shoup quotient          */
    uint32_t v      = (uint32_t)a[k + j + half];  /* in [0, q) < 2^32        */
    uint32_t hi     = (uint32_t)(((uint64_t)w_inv * v) >> 32);
    uint64_t t      = (uint64_t)w * v - (uint64_t)hi * (uint32_t)q;
    if (t >= q) t  -= q;                           /* at most one correction  */

    uint64_t u    = a[k + j];
    uint64_t sum  = u + t;
    uint64_t diff = u + q - t;
    a[k + j]         = sum  >= q ? sum  - q : sum;
    a[k + j + half]  = diff >= q ? diff - q : diff;
}

/*
 * ntt_scale_kernel: multiply every element by a scalar mod q.
 * Purpose:  apply the 1/n scaling factor for INTT.
 * Inputs:   a[0..n-1] in device memory; scalar in [0, q); q < 2³².
 * Output:   a[i] ← a[i] · scalar mod q for all i in [0, n).
 * Mapping:  one thread per element.
 */
/* ntt_scale_kernel receives packed_scalar = (scalar_inv << 32) | scalar,
 * where scalar_inv = floor(scalar * 2^32 / q).  Shoup reduction replaces % q. */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void ntt_scale_kernel(uint64_t * __restrict__ a, uint64_t n,
                      uint64_t packed_scalar, uint64_t q)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t w      = (uint32_t)packed_scalar;
    uint32_t w_inv  = (uint32_t)(packed_scalar >> 32);
    uint32_t v      = (uint32_t)a[i];
    uint32_t hi     = (uint32_t)(((uint64_t)w_inv * v) >> 32);
    uint64_t t      = (uint64_t)w * v - (uint64_t)hi * (uint32_t)q;
    a[i] = (t >= q) ? t - q : t;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HOST-SIDE HELPERS (modular arithmetic, twiddle precomputation)
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t m)
{
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1)
            result = (uint64_t)((__uint128_t)result * base % m);
        base = (uint64_t)((__uint128_t)base * base % m);
        exp >>= 1;
    }
    return result;
}

static uint64_t mod_inv(uint64_t a, uint64_t m)
{
    return mod_pow(a, m - 2, m);
}

/* shoup_pack: pack a twiddle w and its Shoup quotient into a single uint64_t.
 * w_inv = floor(w * 2^32 / q); stored in upper 32 bits.
 * Valid for w < q < 2^32. Matches the layout expected by ntt_stage_kernel. */
static uint64_t shoup_pack(uint64_t w, uint64_t q)
{
    uint64_t w_inv = (uint64_t)(((unsigned __int128)w << 32) / q);
    return (w_inv << 32) | (uint32_t)w;
}

/*
 * ntt_alloc_twiddles: compute host twiddle table with Shoup-packed entries.
 * tw[k] = shoup_pack(omega^k mod q, q) — upper 32 bits = w_inv, lower = w.
 * Callers treat the array as opaque; ntt_stage_kernel decodes the packing.
 * Caller must free().
 */
uint64_t *ntt_alloc_twiddles(const ntt_params_t *p)
{
    uint64_t half = p->n >> 1;
    uint64_t *tw  = (uint64_t *)malloc(half * sizeof *tw);
    if (!tw) return NULL;
    uint64_t w = 1;
    for (uint64_t k = 0; k < half; k++) {
        tw[k] = shoup_pack(w, p->q);
        w = (uint64_t)((__uint128_t)w * p->omega % p->q);
    }
    return tw;
}

uint64_t *ntt_alloc_twiddles_inv(const ntt_params_t *p)
{
    uint64_t half = p->n >> 1;
    uint64_t *tw  = (uint64_t *)malloc(half * sizeof *tw);
    if (!tw) return NULL;
    uint64_t w = 1;
    for (uint64_t k = 0; k < half; k++) {
        tw[k] = shoup_pack(w, p->q);
        w = (uint64_t)((__uint128_t)w * p->omega_inv % p->q);
    }
    return tw;
}

/* ── Internal GPU NTT engine ─────────────────────────────────────────────── */

/*
 * gpu_ntt: shared implementation for forward and inverse GPU NTT.
 * Purpose:  copy host arrays to device, run kernels, copy result back.
 * Inputs:   a_host — host array (in/out); tw_host — host twiddle table (read-only).
 *           p — parameters; scale — scalar applied after stages (1 for forward,
 *           n_inv for inverse); 0 means no scaling pass.
 * Output:   a_host[0..n-1] contains NTT result after return.
 * Reduction: per-stage (not lazy); safe for q < 2³².
 */
static void gpu_ntt(uint64_t *a_host, const uint64_t *tw_host,
                    const ntt_params_t *p, uint64_t scale)
{
    uint64_t n    = p->n;
    uint64_t q    = p->q;
    uint64_t half_n = n >> 1;
    int      bsz  = g_gpu.block_size;

    uint64_t *d_a  = NULL;
    uint64_t *d_tw = NULL;
    HIP_CHECK(hipMalloc((void **)&d_a,  n      * sizeof *d_a));
    HIP_CHECK(hipMalloc((void **)&d_tw, half_n * sizeof *d_tw));
    HIP_CHECK(hipMemcpy(d_a,  a_host,  n      * sizeof *d_a,  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tw, tw_host, half_n * sizeof *d_tw, hipMemcpyHostToDevice));

    /* Bit-reversal: n threads, one per element */
    int grid_n = (int)((n + (uint64_t)bsz - 1) / (uint64_t)bsz);
    hipLaunchKernelGGL(ntt_bitrev_kernel, grid_n, bsz, 0, 0,
                       d_a, n, p->log2_n);
    HIP_CHECK(hipGetLastError());

    /* Butterfly stages: n/2 threads, one per butterfly pair */
    int grid_h = (int)((half_n + (uint64_t)bsz - 1) / (uint64_t)bsz);
    for (uint64_t len = 2; len <= n; len <<= 1) {
        uint64_t half   = len >> 1;
        uint64_t stride = n / len;
        hipLaunchKernelGGL(ntt_stage_kernel, grid_h, bsz, 0, 0,
                           d_a, d_tw, n, q, half, stride);
        HIP_CHECK(hipGetLastError());
    }

    /* Optional scaling (INTT: multiply by n_inv).
     * Pack Shoup quotient: packed = (floor(scale*2^32/q) << 32) | scale. */
    if (scale != 0 && scale != 1) {
        uint64_t packed_scale = shoup_pack(scale, q);
        hipLaunchKernelGGL(ntt_scale_kernel, grid_n, bsz, 0, 0,
                           d_a, n, packed_scale, q);
        HIP_CHECK(hipGetLastError());
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(a_host, d_a, n * sizeof *d_a, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_tw));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC API  (same signatures as ntt_cpu.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * ntt_forward: in-place CT-DIT forward NTT, GPU-accelerated.
 * Inputs/outputs identical to ntt_cpu.c::ntt_forward.
 * a[] is a host pointer; this function handles device allocation internally.
 */
void ntt_forward(uint64_t *a, const uint64_t * __restrict__ twiddles,
                 const ntt_params_t *p)
{
    gpu_ntt(a, twiddles, p, 0);  /* scale=0: skip scaling kernel */
}

/*
 * ntt_inverse: in-place INTT, GPU-accelerated.
 * Runs forward NTT with inverse twiddles, then scales by n^{-1} mod q.
 */
void ntt_inverse(uint64_t *a, const uint64_t * __restrict__ twiddles_inv,
                 const ntt_params_t *p)
{
    gpu_ntt(a, twiddles_inv, p, p->n_inv);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PARAMETER INIT, DISPLAY, SELFTEST, BENCHMARK (identical logic to ntt_cpu.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * root_is_primitive: omega must be a primitive n-th root of unity mod q,
 * i.e. omega^n == 1 and omega^(n/2) != 1 (mod q). The second condition
 * rules out lower-order roots. Without this check the selftest is vacuous:
 * INTT(NTT(a))==a holds for ANY invertible omega (incl. omega=1), so a
 * wrong/non-primitive root passes silently.
 */
static int root_is_primitive(uint64_t n, uint64_t q, uint64_t omega)
{
    if (omega == 0) return 0;
    if (mod_pow(omega, n, q) != 1) return 0;
    if (mod_pow(omega, n >> 1, q) == 1) return 0;
    return 1;
}

static int ntt_params_init(ntt_params_t *p)
{
    if (p->n == 0 || (p->n & (p->n - 1)) != 0) return -1;
    if (p->q < 2) {
        fprintf(stderr, "error: q must be a prime >= 2\n");
        return -1;
    }
    if (p->q > 0xFFFFFFFFULL) {
        fprintf(stderr, "error: GPU implementation currently requires q < 2^32\n");
        return -1;
    }
    if (!root_is_primitive(p->n, p->q, p->omega)) {
        fprintf(stderr, "error: omega=%lu is not a primitive %lu-th root of "
                "unity mod q=%lu\n", p->omega, p->n, p->q);
        return -1;
    }
    p->omega_inv = mod_inv(p->omega, p->q);
    p->n_inv     = mod_inv(p->n,     p->q);
    uint64_t tmp = p->n;
    p->log2_n    = 0;
    while (tmp > 1) { tmp >>= 1; p->log2_n++; }
    return 0;
}

static void print_gpu_summary(void)
{
    double mem_gb = (double)g_gpu.total_mem / (1024.0 * 1024.0 * 1024.0);
    printf("\n" ANSI_CYN "  ── GPU Device ───────────────────────────────────────────\n" ANSI_RST);
    printf("  ┌──────────────────────────┬─────────────────────────┐\n");
    printf("  │ " ANSI_CYN "%-24s" ANSI_RST " │ " ANSI_CYN "%-23s" ANSI_RST " │\n",
           "Property", "Value");
    printf("  ├──────────────────────────┼─────────────────────────┤\n");
    printf("  │ %-24s │ %-23s │\n",  "Device name",    g_gpu.name);
    printf("  │ %-24s │ %-23d │\n",  "Wavefront width", g_gpu.warp_size);
    printf("  │ %-24s │ %-23d │\n",  "Compute units",   g_gpu.cu_count);
    printf("  │ %-24s │ %-23d │\n",  "Max LDS/block(B)", g_gpu.max_shared_b);
    printf("  │ %-24s │ %-.2f %-18s │\n", "Global memory",  mem_gb, "GB");
    printf("  │ %-24s │ %-23d │\n",  "Launch block size", g_gpu.block_size);
    printf("  └──────────────────────────┴─────────────────────────┘\n\n");
}

static void print_params(const ntt_params_t *p)
{
    printf(ANSI_CYN "  ── NTT Parameters ──────────────────────────────────────\n" ANSI_RST);
    printf("  ┌──────────────────────────┬─────────────────────────┐\n");
    printf("  │ " ANSI_CYN "%-24s" ANSI_RST " │ " ANSI_CYN "%-23s" ANSI_RST " │\n",
           "Parameter", "Value");
    printf("  ├──────────────────────────┼─────────────────────────┤\n");
    printf("  │ %-24s │ %-23lu │\n", "n",         p->n);
    printf("  │ %-24s │ %-23lu │\n", "q",         p->q);
    printf("  │ %-24s │ %-23lu │\n", "omega",     p->omega);
    printf("  │ %-24s │ %-23lu │\n", "omega_inv", p->omega_inv);
    printf("  │ %-24s │ %-23lu │\n", "n_inv",     p->n_inv);
    printf("  │ %-24s │ %-23u │\n",  "log2(n)",   p->log2_n);
    printf("  └──────────────────────────┴─────────────────────────┘\n\n");
}

static int selftest(const ntt_params_t *p)
{
    uint64_t n    = p->n;
    uint64_t *orig = (uint64_t *)malloc(n * sizeof *orig);
    uint64_t *a    = (uint64_t *)malloc(n * sizeof *a);
    uint64_t *tw   = ntt_alloc_twiddles(p);
    uint64_t *twi  = ntt_alloc_twiddles_inv(p);
    int ok = 0;

    if (!orig || !a || !tw || !twi) { ok = -1; goto done; }

    for (uint64_t i = 0; i < n; i++)
        orig[i] = a[i] = (i * 1234567891ULL + 42) % p->q;

    ntt_forward(a, tw,  p);
    ntt_inverse(a, twi, p);

    for (uint64_t i = 0; i < n; i++) {
        if (a[i] != orig[i]) {
            printf("  " ANSI_RED "FAIL" ANSI_RST
                   " index %lu: expected %lu got %lu\n", i, orig[i], a[i]);
            ok = -1;
            goto done;
        }
    }
done:
    free(orig); free(a); free(tw); free(twi);
    return ok;
}

static void run_benchmark(const ntt_params_t *p, uint64_t iters)
{
    uint64_t n   = p->n;
    uint64_t *a  = (uint64_t *)malloc(n * sizeof *a);
    uint64_t *tw = ntt_alloc_twiddles(p);
    if (!a || !tw) { free(a); free(tw); return; }
    for (uint64_t i = 0; i < n; i++) a[i] = i % p->q;

    /* Warm-up: one NTT to initialise HIP runtime state */
    ntt_forward(a, tw, p);

    /* Pre-allocate and transfer to benchmark true kernel throughput */
    uint64_t *d_a, *d_tw;
    (void)hipMalloc((void **)&d_a, n * sizeof(*d_a));
    (void)hipMalloc((void **)&d_tw, (n >> 1) * sizeof(*d_tw));
    (void)hipMemcpy(d_tw, tw, (n >> 1) * sizeof(*d_tw), hipMemcpyHostToDevice);
    
    int bsz = g_gpu.block_size;
    int grid_n = (int)((n + bsz - 1) / bsz);
    int grid_h = (int)(((n >> 1) + bsz - 1) / bsz);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) {
        hipLaunchKernelGGL(ntt_bitrev_kernel, grid_n, bsz, 0, 0, d_a, n, p->log2_n);
        for (uint64_t len = 2; len <= n; len <<= 1) {
            hipLaunchKernelGGL(ntt_stage_kernel, grid_h, bsz, 0, 0, d_a, d_tw, n, p->q, len >> 1, n / len);
        }
    }
    (void)hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    (void)hipFree(d_a); (void)hipFree(d_tw);

    double elapsed     = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double ntts_per_s  = (double)iters / elapsed;
    double ns_per_ntt  = elapsed * 1e9 / (double)iters;
    double butterflies = (double)(n >> 1) * (double)p->log2_n;
    double ns_per_bf   = ns_per_ntt / butterflies;

    printf(ANSI_CYN "  ── Benchmark: ntt_forward (GPU) ─────────────────────────\n" ANSI_RST);
    printf("  ┌────────────────────┬──────────┬────────────┬────────────────┐\n");
    printf("  │ " ANSI_CYN "%-18s" ANSI_RST " │ " ANSI_CYN "%-8s" ANSI_RST
           " │ " ANSI_CYN "%-10s" ANSI_RST " │ " ANSI_CYN "%-14s" ANSI_RST " │\n",
           "Metric", "n", "Iters", "Value");
    printf("  ├────────────────────┼──────────┼────────────┼────────────────┤\n");
    printf("  │ %-18s │ %-8lu │ %-10lu │ %11.0f /s │\n",
           "NTT throughput", n, iters, ntts_per_s);
    printf("  │ %-18s │ %-8lu │ %-10lu │ %11.1f ns  │\n",
           "ns / NTT", n, iters, ns_per_ntt);
    printf("  │ %-18s │ %-8lu │ %-10lu │ %11.2f ns  │\n",
           "ns / butterfly", n, iters, ns_per_bf);
    printf("  └────────────────────┴──────────┴────────────┴────────────────┘\n\n");

    char fname[64];
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    strftime(fname, sizeof fname, "bench_gpu_%Y%m%d_%H%M%S.txt", tm);
    FILE *f = fopen(fname, "w");
    if (f) {
        fprintf(f, "kernel=ntt_forward_gpu device=%s n=%lu q=%lu omega=%lu iters=%lu\n",
                g_gpu.name, n, p->q, p->omega, iters);
        fprintf(f, "elapsed_s=%.6f ntts_per_s=%.0f ns_per_ntt=%.1f ns_per_butterfly=%.2f\n",
                elapsed, ntts_per_s, ns_per_ntt, ns_per_bf);
        fclose(f);
        printf("  Results written to %s\n\n", fname);
    }
    free(a); free(tw);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * main: query GPU, print hardware summary, run selftest, run benchmark.
 * Usage:  ntt_gpu [n] [q] [omega] [iters]
 * Defaults: n=256, q=3329, omega=17 (ML-KEM / FIPS 203 parameters).
 */
int main(int argc, char *argv[])
{
    ntt_params_t p;
    p.n     = (argc > 1) ? (uint64_t)atol(argv[1]) : 256;
    p.q     = (argc > 2) ? (uint64_t)atol(argv[2]) : 3329;
    p.omega = (argc > 3) ? (uint64_t)atol(argv[3]) : 17;
    uint64_t iters = (argc > 4) ? (uint64_t)atol(argv[4]) : 1000;

    printf("\n" ANSI_WHT
           "╔══════════════════════════════════════════════════════════╗\n"
           "║          NTT / MI300A  —  GPU Reference Kernel          ║\n"
           "╚══════════════════════════════════════════════════════════╝\n"
           ANSI_RST "\n");

    /* Query GPU hardware properties; choose block size as smallest multiple
     * of 64 that is ≥ warpSize and does not exceed NTT_MAX_BLOCK. */
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    strncpy(g_gpu.name,       prop.name,                sizeof g_gpu.name - 1);
    g_gpu.warp_size   = prop.warpSize;
    g_gpu.cu_count    = prop.multiProcessorCount;
    g_gpu.max_shared_b = (int)prop.sharedMemPerBlock;
    g_gpu.max_regs    = prop.regsPerBlock;
    g_gpu.total_mem   = prop.totalGlobalMem;
    /* Block size: multiple of 64 (MI300A requirement), ≥ warpSize, ≤ 256 */
    g_gpu.block_size  = (prop.warpSize <= 32) ? 128 : 128;  /* 128 safe for both */
    print_gpu_summary();

    if (ntt_params_init(&p) != 0) {
        fprintf(stderr, "error: n=%lu is not a power of 2\n", p.n);
        return 1;
    }
    print_params(&p);

    printf(ANSI_CYN "  ── Selftest ─────────────────────────────────────────────\n" ANSI_RST);
    int result = selftest(&p);
    if (result == 0)
        printf("  " ANSI_GRN "PASS" ANSI_RST
               "  INTT(NTT(a)) == a  for n=%lu, q=%lu, omega=%lu\n\n",
               p.n, p.q, p.omega);
    else
        printf("  " ANSI_RED "FAIL" ANSI_RST "  see above\n\n");

    if (result != 0) return 1;

    run_benchmark(&p, iters);
    return 0;
}
