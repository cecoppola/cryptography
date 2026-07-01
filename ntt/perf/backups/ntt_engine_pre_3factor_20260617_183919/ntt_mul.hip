/*
 * ntt_mul.hip — NTT-based BigInt multiply for the CRT-NTT arith layer.
 *
 * Uses the 4-prime CRT pipeline (one GPU per prime) to compute the
 * polynomial product of two BigInts represented as coefficient arrays.
 *
 * Each BigInt with k 64-bit limbs is treated as a degree-(k-1) polynomial
 * evaluated at x=2^64.  The convolution product has 2k-1 coefficients,
 * each fitting in U256 (≤ k * (2^64)^2 ≤ 2^24 * 2^128 = 2^152 < 2^255).
 *
 * Pipeline per call:
 *   1. bigint_scatter → 4 per-prime host arrays (CPU, parallel)
 *   2. hipMemcpyAsync → 4 device arrays (one per GPU)
 *   3. launch_ntt<PIDX> → 4 forward NTTs (one per GPU)
 *   4. launch_hadamard<PIDX> → 4 pointwise multiplies
 *   5. launch_intt_fused<PIDX> → 4 inverse NTTs (fused n^-1 scale)
 *   6. hipMemcpyAsync → download 4 residue arrays (one per GPU)
 *   7. garner_reconstruct → U256 coefficient array (CPU, OpenMP)
 *   8. bigint_gather → output BigInt
 *
 * State is global and single-threaded.  Call ntt_mul_init_impl() once.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

#include "../primes.h"
#include "../shoup.h"
#include "../crt_ntt.h"
#include "bigint.h"

/* ─── Forward declarations from other translation units ──────────────────── */
template<int PIDX>
void launch_ntt(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_intt(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_intt_fused(uint64_t*, uint64_t, const ShoupPair*,
                       const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_ntt(uint64_t*, const ShoupPair*, const ShoupPair*,
                         int, hipStream_t);
template<int PIDX>
void launch_stockham_intt(uint64_t*, uint64_t, const ShoupPair*,
                          const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_intt_hadamard(uint64_t*, const uint64_t*, uint64_t,
                          const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_ntt_rect(uint64_t*, uint64_t*, const ShoupPair*,
                              const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_intt_rect(uint64_t*, uint64_t*, uint64_t, const ShoupPair*,
                               const ShoupPair*, int, hipStream_t);
void launch_transpose_sq(uint64_t* data, int M, hipStream_t stream);
void garner_reconstruct_gpu(const uint64_t*, const uint64_t*, const uint64_t*,
                            const uint64_t*, U256*, int, const GarnerConsts*,
                            hipStream_t);
template<int PIDX>
void launch_hadamard(uint64_t*, const uint64_t*, int, hipStream_t);

GarnerConsts garner_precompute(void);
void garner_reconstruct(const uint64_t*, const uint64_t*, const uint64_t*,
                        const uint64_t*, U256*, int, const GarnerConsts*);

/* ─── HIP error gate ─────────────────────────────────────────────────────── */
/* GPU-SAFETY (2026-05-16 incident): the engine previously discarded every
 * HIP return via (void).  A failed hipMalloc/hipMemcpy then left NULL/invalid
 * device pointers, the kernel did an illegal access, and the amdgpu ring
 * faulted → GPU reset.  HIP_CHECK aborts on the HOST the instant a HIP call
 * fails, so a transient failure can never cascade into a GPU crash.  Used
 * only in cold setup/teardown paths — zero hot-loop cost (CDNA3 & RDNA2). */
#define HIP_CHECK(expr) do {                                                  \
    hipError_t _e = (expr);                                                   \
    if (_e != hipSuccess) {                                                   \
        fprintf(stderr, "ntt_mul: HIP error %d (%s) at %s:%d: %s\n",          \
                (int)_e, hipGetErrorString(_e), __FILE__, __LINE__, #expr);   \
        fflush(stderr);                                                       \
        exit(1);                                                              \
    }                                                                         \
} while (0)

/* ─── GPU scatter kernel ─────────────────────────────────────────────────── */
/* For one prime, reduce each limb of the input BigInt to its residue mod p
 * and write to the per-prime device coefficient array. Positions beyond
 * n_limbs are zero-padded so the NTT input length matches N exactly.
 * Embarrassingly parallel — replaces the CPU OpenMP scatter that ran on
 * only 4 threads (one per prime). */
template<int PIDX>
__global__ __launch_bounds__(256, 8)
static void scatter_kernel(const limb_t* __restrict__ d_in,
                           int n_limbs,
                           uint64_t* __restrict__ d_out,
                           int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i < n_limbs) {
        /* limb_t is uint64_t (LB=64) or unsigned __int128 (LB=112); both
         * are 128-bit-promotable for the modular reduction. */
        d_out[i] = (uint64_t)((__uint128_t)d_in[i] % PRIMES[PIDX]);
    } else {
        d_out[i] = 0;
    }
}

template<int PIDX>
static void launch_scatter(const limb_t* d_in, int n_limbs,
                           uint64_t* d_out, int N, hipStream_t stream)
{
    const int block = 256;
    int grid = (N + block - 1) / block;
    if (grid < 1) grid = 1;
    scatter_kernel<PIDX><<<grid, block, 0, stream>>>(d_in, n_limbs, d_out, N);
}

/* ─── Internal twiddle helpers (mirrors main.hip) ────────────────────────── */

/* Flat (Stockham) twiddle helpers — mirror lib2/main.hip make_tw_stockham. */
static void fill_flat_table_local(ShoupPair* h, uint64_t p, uint64_t g_root, int log_m)
{
    int M = 1 << log_m;
    uint64_t omega = modpow(g_root, (p - 1) / (uint64_t)M, p);
    uint64_t cur = 1;
    for (int k = 0; k < M / 2; k++) {
        h[k] = make_shoup(cur, p);
        cur = (uint64_t)((__uint128_t)cur * omega % p);
    }
}

static ShoupPair* make_tw_stockham_local(int dev, uint64_t p, uint64_t g_root, int log_n)
{
    int alloc_size, log_n1, log_n2;
    if (log_n <= 10) {
        alloc_size = (1 << log_n) / 2;
    } else {
        log_n1 = log_n / 2; log_n2 = log_n - log_n1;
        alloc_size = (1 << (log_n1 - 1)) + (1 << (log_n2 - 1));
    }
    ShoupPair* h = (ShoupPair*)malloc((size_t)alloc_size * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "ntt_mul: make_tw_stockham malloc failed\n"); exit(1); }
    if (log_n <= 10) {
        fill_flat_table_local(h, p, g_root, log_n);
    } else {
        log_n1 = log_n / 2; log_n2 = log_n - log_n1;
        fill_flat_table_local(h,                    p, g_root, log_n1);
        fill_flat_table_local(h + (1 << (log_n1-1)), p, g_root, log_n2);
    }
    ShoupPair* d;
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMalloc(&d, (size_t)alloc_size * sizeof(ShoupPair)));
    HIP_CHECK(hipMemcpy(d, h, (size_t)alloc_size * sizeof(ShoupPair), hipMemcpyHostToDevice));
    free(h);
    return d;
}

static ShoupPair* make_tw_stockham_inv_local(int dev, uint64_t p, uint64_t g_root, int log_n)
{
    uint64_t g_root_inv = modpow(g_root, p - 2, p);
    return make_tw_stockham_local(dev, p, g_root_inv, log_n);
}

static ShoupPair* make_tw_local(int dev, uint64_t p, uint64_t g_root, int log_n)
{
    /* N-1 twiddle entries total: stage s contributes 2^s entries at offset 2^s-1.
     * Sum_{s=0}^{log_n-1} 2^s = N-1.  Must allocate N-1, not N/2. */
    int n_tw = (1 << log_n) - 1;
    ShoupPair* h = (ShoupPair*)malloc((size_t)n_tw * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "ntt_mul: make_tw malloc failed\n"); exit(1); }
    uint64_t omega  = modpow(g_root, (p - 1) / (uint64_t)(1 << log_n), p);
    int offset = 0;
    for (int s = 0; s < log_n; s++) {
        int cnt = 1 << s;
        uint64_t root = modpow(omega, (uint64_t)((1 << log_n) >> (s + 1)), p);
        uint64_t cur  = 1;
        for (int j = 0; j < cnt; j++) {
            h[offset + j] = make_shoup(cur, p);
            cur = (uint64_t)((__uint128_t)cur * root % p);
        }
        offset += cnt;
    }
    ShoupPair* d;
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMalloc(&d, (size_t)n_tw * sizeof(ShoupPair)));
    HIP_CHECK(hipMemcpy(d, h, (size_t)n_tw * sizeof(ShoupPair), hipMemcpyHostToDevice));
    free(h);
    return d;
}

static ShoupPair* make_tw_cross_local(int dev, uint64_t p, uint64_t g_root, int log_n)
{
    int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
    int N1 = 1 << log_n1, N2 = 1 << log_n2;
    ShoupPair* h = (ShoupPair*)malloc((size_t)(N1 * N2) * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "ntt_mul: make_tw_cross malloc failed\n"); exit(1); }
    uint64_t omega = modpow(g_root, (p - 1) / (uint64_t)(N1 * N2), p);
    for (int r = 0; r < N1; r++) {
        uint64_t rb = modpow(omega, (uint64_t)r, p);
        uint64_t cur = 1;
        for (int c = 0; c < N2; c++) {
            h[r * N2 + c] = make_shoup(cur, p);
            cur = (uint64_t)((__uint128_t)cur * rb % p);
        }
    }
    ShoupPair* d;
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMalloc(&d, (size_t)(N1 * N2) * sizeof(ShoupPair)));
    HIP_CHECK(hipMemcpy(d, h, (size_t)(N1 * N2) * sizeof(ShoupPair), hipMemcpyHostToDevice));
    free(h);
    return d;
}

static ShoupPair* make_tw_cross_inv_scaled_local(int dev, uint64_t p,
                                                  uint64_t g_root, int log_n,
                                                  uint64_t n_inv)
{
    int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
    int N1 = 1 << log_n1, N2 = 1 << log_n2;
    ShoupPair* h = (ShoupPair*)malloc((size_t)(N1 * N2) * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "ntt_mul: make_tw_cross_inv malloc failed\n"); exit(1); }
    /* omega_inv = g_root^((p-1) - (p-1)/N) mod p */
    uint64_t N = (uint64_t)N1 * N2;
    uint64_t omega_inv = modpow(g_root, (p - 1) - (p - 1) / N, p);
    for (int r = 0; r < N1; r++) {
        uint64_t rb = modpow(omega_inv, (uint64_t)r, p);
        uint64_t cur = 1;
        for (int c = 0; c < N2; c++) {
            uint64_t val = (uint64_t)((__uint128_t)cur * n_inv % p);
            h[r * N2 + c] = make_shoup(val, p);
            cur = (uint64_t)((__uint128_t)cur * rb % p);
        }
    }
    ShoupPair* d;
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMalloc(&d, (size_t)(N1 * N2) * sizeof(ShoupPair)));
    HIP_CHECK(hipMemcpy(d, h, (size_t)(N1 * N2) * sizeof(ShoupPair), hipMemcpyHostToDevice));
    free(h);
    return d;
}

/* ─── Global NTT multiply state ──────────────────────────────────────────── */

typedef struct {
    int          log_n;
    int          n_devs;    /* actual GPU count (1..4); primes round-robin across devs */
    uint64_t*    d_a[4];    /* device coefficient arrays for operand A */
    uint64_t*    d_b[4];    /* device coefficient arrays for operand B */
    ShoupPair*   d_tw[4];
    ShoupPair*   d_tw_inv[4];
    ShoupPair*   d_tw_cross[4];
    ShoupPair*   d_tw_cross_inv_scaled[4];
    ShoupPair*   d_tw_stok[4];      /* forward Stockham twiddle table */
    ShoupPair*   d_tw_stok_inv[4];  /* inverse Stockham twiddle table */
    uint64_t*    d_scr[4];          /* size-N scratch for odd-log_n rect transpose (I11) */
    uint64_t     n_inv[4];   /* N^{-1} mod p_i; computed once in init */
    hipStream_t  stream[4];
    GarnerConsts g;
    U256*        h_out;     /* host Garner output: N U256s */
    U256*        d_out;     /* device Garner output (on device 0); copied to h_out */
    limb_t*      d_in_a;    /* device-0 buffer for a->limbs (size N × LIMB_BYTES) */
    limb_t*      d_in_b;    /* device-0 buffer for b->limbs */
    uint64_t*    h_a[4];   /* host scatter buffers */
    uint64_t*    h_b[4];
} NttMulState;

static NttMulState g_nms;
static int         g_nms_ready = 0;
static unsigned long g_gpu_dispatch_count = 0;

/* ─── Opt-in per-phase profiler (NTT_PROFILE=1) ──────────────────────────── */
/* g_prof is resolved once in ntt_mul_init_impl() via prof_resolve() (see
 * below) so the teardown gate (g_prof>0 && g_prof_calls>0) and the per-call
 * accumulators always observe a definite value — not a -1 "unresolved" left
 * by a never-taken lazy path. Zero overhead when off: g_prof==0 short-circuits
 * every prof_ms()/accumulate site, so an OFF run is bit-identical. Reuses the
 * existing post-download stream-sync as the GPU-phase barrier — adds NO extra
 * syncs; an ON run's only perturbation is the clock_gettime calls (~20 ns). */
#include <time.h>
static int    g_prof = -1;            /* -1 unresolved, 0 off, 1 on          */
static double g_prof_ms[4];           /* scatter / gpu-pipe / garner / gather */
static long   g_prof_calls = 0;
static double g_prof_ms2[3];          /* deep (NTT_PROFILE=2): upload/compute/download
                                       * — adds transient syncs, profile-only */
static inline double prof_ms(void){ struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec*1e3 + (double)t.tv_nsec*1e-6; }

/* Resolve g_prof from NTT_PROFILE exactly once, idempotently. Called from
 * ntt_mul_init_impl() (single-threaded, before any multiply) so the teardown
 * gate and per-call accumulator always observe a resolved value, and as a
 * defensive fallback at the top of ntt_bigint_mul() in case a future caller
 * multiplies without init. Reads the env only on the -1→{0,1,2} transition;
 * once resolved this is a single predictable branch. Zero cost when off:
 * g_prof==0 makes every prof_ms()/accumulate site below a no-op. */
static inline void prof_resolve(void){
    if (g_prof < 0) { const char *pe = getenv("NTT_PROFILE");
                      g_prof = pe ? atoi(pe) : 0; } /* 0 off, 1 phase, 2 deep */
}

extern "C" void ntt_mul_init_impl(int max_log_n)
{
    prof_resolve();   /* resolve NTT_PROFILE once, single-threaded, pre-multiply */
    if (g_nms_ready) return;
    /* GPU-SAFETY: bound log_n.
     *   Lower bound 3: log_n<3 is degenerate (NTT semantics).
     *   Upper bound 22: >22 would request cross-twiddle tables of 2^log_n
     *   ShoupPair (16 B each) per prime — e.g. log_n=24 → 256 MiB ×4 ×4
     *   tables ≈ 4 GiB, courting OOM on the 6900XT (16 GiB) and, with any
     *   failed alloc, a GPU fault. CRT-NTT call sites need ≤ ~20 in
     *   practice. launch_ntt routes log_n!=10 through the global path,
     *   so small N is safe (see lib2/ntt_kernel.hip). Reject loudly. */
    /* Default cap 22 (validated 2026-06-16: log_n=22 GMP-verified; even log_n>=24
     * exceeds the LDS-Stockham sub-transform limit, odd log_n=23 untested). The
     * cap is env-overridable (NTT_MAX_LOGN) so the GMP oracle can probe 23/24/25
     * without changing the production default. */
    int _cap = 22;
    { const char *_e = getenv("NTT_MAX_LOGN");
      if (_e) { int _v = atoi(_e); if (_v >= 3 && _v <= 26) _cap = _v; } }
    if (max_log_n < 3 || max_log_n > _cap) {
        fprintf(stderr, "ntt_mul_init: log_n=%d out of safe range [3,%d] (set NTT_MAX_LOGN to raise)\n",
                max_log_n, _cap);
        exit(1);
    }
    int log_n = max_log_n;
    int N     = 1 << log_n;
    g_nms.log_n = log_n;

    g_nms.g   = garner_precompute();
    HIP_CHECK(hipGetDeviceCount(&g_nms.n_devs));
    if (g_nms.n_devs < 1) g_nms.n_devs = 1;
    if (g_nms.n_devs > 4) g_nms.n_devs = 4;
    /* h_out: GPU Garner output.
     *   GFX1030_LOCAL: standard host malloc + separate device d_out, D2H copy.
     *   MI300A:        hipHostMalloc gives a host pointer in unified HBM3,
     *                  GPU writes there directly; d_out unused. */
    HIP_CHECK(hipSetDevice(0));
#ifdef GFX1030_LOCAL
    g_nms.h_out = (U256*)malloc((size_t)N * sizeof(U256));
    if (!g_nms.h_out) { fprintf(stderr, "ntt_mul_init: OOM\n"); exit(1); }
    HIP_CHECK(hipMalloc(&g_nms.d_out, (size_t)N * sizeof(U256)));
#else
    HIP_CHECK(hipHostMalloc(&g_nms.h_out, (size_t)N * sizeof(U256),
                             hipHostMallocNonCoherent));
    g_nms.d_out = g_nms.h_out;
#endif
    HIP_CHECK(hipMalloc(&g_nms.d_in_a, (size_t)N * sizeof(limb_t)));
    HIP_CHECK(hipMalloc(&g_nms.d_in_b, (size_t)N * sizeof(limb_t)));

    /* Display-coexistence: on a single GPU that also drives the desktop, run the
     * CRT-prime compute streams at the LOWEST hardware-scheduler priority so the
     * graphics/display queue preempts our NTT kernels and page-flips are never
     * starved (the log_n=21 hard-hang class). Requires cwsr_enable=1 (compute
     * waves preemptible), which is set on this box. Gated by NTT_STREAM_LOWPRIO
     * (default 1; set 0 to restore default-priority streams). */
    int s_lo_prio = 0;
    {
        const char *e = getenv("NTT_STREAM_LOWPRIO");
        int want_lowprio = e ? atoi(e) : 1;
        if (want_lowprio) {
            int prio_least = 0, prio_greatest = 0;
            if (hipDeviceGetStreamPriorityRange(&prio_least, &prio_greatest) == hipSuccess)
                s_lo_prio = prio_least;   /* numerically highest = lowest priority */
        }
    }
    for (int i = 0; i < 4; i++) {
        HIP_CHECK(hipSetDevice(i % g_nms.n_devs));
        HIP_CHECK(hipStreamCreateWithPriority(&g_nms.stream[i], hipStreamDefault, s_lo_prio));
        HIP_CHECK(hipMalloc(&g_nms.d_a[i], (size_t)N * sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&g_nms.d_b[i], (size_t)N * sizeof(uint64_t)));
        /* I11: scratch for the odd-log_n rectangular 4-step transpose. Only
         * touched when g_nms.log_n is odd>10, but allocated unconditionally
         * (size N) so the dispatch never branches on allocation. */
        HIP_CHECK(hipMalloc(&g_nms.d_scr[i], (size_t)N * sizeof(uint64_t)));
        g_nms.h_a[i] = (uint64_t*)malloc((size_t)N * sizeof(uint64_t));
        g_nms.h_b[i] = (uint64_t*)malloc((size_t)N * sizeof(uint64_t));
        if (!g_nms.h_a[i] || !g_nms.h_b[i]) {
            fprintf(stderr, "ntt_mul_init: OOM h\n"); exit(1);
        }
        int dev        = i % g_nms.n_devs;
        uint64_t p     = PRIMES[i];
        uint64_t root  = PRIM_ROOT[i];
        uint64_t n_inv = modpow((uint64_t)N, p - 2, p);
        g_nms.n_inv[i] = n_inv;   /* cache so ntt_bigint_mul doesn't recompute */
        g_nms.d_tw[i]       = make_tw_local(dev, p, root, log_n);
        /* Inverse twiddle: use root^{-1} = root^{p-2}. */
        uint64_t root_inv    = modpow(root, p - 2, p);
        g_nms.d_tw_inv[i]   = make_tw_local(dev, p, root_inv, log_n);
        g_nms.d_tw_cross[i] = make_tw_cross_local(dev, p, root, log_n);
        g_nms.d_tw_cross_inv_scaled[i] =
            make_tw_cross_inv_scaled_local(dev, p, root, log_n, n_inv);
        /* Stockham tables: flat layout per stage; smaller LDS path used by
         * launch_stockham_ntt/intt avoids the per-stage HBM round-trip the
         * old CT-DIT global path incurred for log_n>10. */
        g_nms.d_tw_stok[i]     = make_tw_stockham_local(dev, p, root, log_n);
        g_nms.d_tw_stok_inv[i] = make_tw_stockham_inv_local(dev, p, root, log_n);
    }
    g_nms_ready = 1;
}

extern "C" void ntt_mul_teardown_impl(void)
{
    if (!g_nms_ready) return;
    if (g_prof > 0 && g_prof_calls > 0) {
        double tot = g_prof_ms[0]+g_prof_ms[1]+g_prof_ms[2]+g_prof_ms[3];
        if (tot <= 0) tot = 1;
        fprintf(stderr,
          "[ntt_prof] calls=%ld  scatter=%.1fms(%.0f%%)  gpu_pipe=%.1fms(%.0f%%)"
          "  garner=%.1fms(%.0f%%)  gather=%.1fms(%.0f%%)  total=%.1fms\n",
          g_prof_calls,
          g_prof_ms[0],100*g_prof_ms[0]/tot, g_prof_ms[1],100*g_prof_ms[1]/tot,
          g_prof_ms[2],100*g_prof_ms[2]/tot, g_prof_ms[3],100*g_prof_ms[3]/tot,
          tot);
        if (g_prof > 1)
            fprintf(stderr, "[ntt_prof2] gpu_pipe split: upload=%.1fms  "
              "compute(NTT+Had+INTT)=%.1fms  download=%.1fms  "
              "(transfers=%.1fms = %.0f%% of pipe — the zero-copy ceiling)\n",
              g_prof_ms2[0], g_prof_ms2[1], g_prof_ms2[2],
              g_prof_ms2[0]+g_prof_ms2[2],
              100.0*(g_prof_ms2[0]+g_prof_ms2[2])/
                    (g_prof_ms2[0]+g_prof_ms2[1]+g_prof_ms2[2] > 0 ?
                     g_prof_ms2[0]+g_prof_ms2[1]+g_prof_ms2[2] : 1));
        fflush(stderr);
    }
    /* Flush all in-flight GPU work before releasing resources.  hipStreamDestroy
     * drains its own queue, but a device-wide sync here makes KFD workqueue
     * cleanup faster and more deterministic, closing the PASID-reuse race window
     * for the next HIP process opened on the same boot (CRASH 15, 2026-06-08). */
    (void)hipDeviceSynchronize();
    int N = 1 << g_nms.log_n;
    (void)N;
    for (int i = 0; i < 4; i++) {
        (void)hipSetDevice(i % g_nms.n_devs);
        (void)hipFree(g_nms.d_a[i]);  (void)hipFree(g_nms.d_b[i]);
        (void)hipFree(g_nms.d_tw[i]); (void)hipFree(g_nms.d_tw_inv[i]);
        (void)hipFree(g_nms.d_tw_cross[i]);
        (void)hipFree(g_nms.d_tw_cross_inv_scaled[i]);
        (void)hipFree(g_nms.d_tw_stok[i]);
        (void)hipFree(g_nms.d_tw_stok_inv[i]);
        (void)hipFree(g_nms.d_scr[i]);
        (void)hipStreamDestroy(g_nms.stream[i]);
        free(g_nms.h_a[i]); free(g_nms.h_b[i]);
    }
    (void)hipSetDevice(0);
    (void)hipFree(g_nms.d_in_a);
    (void)hipFree(g_nms.d_in_b);
#ifdef GFX1030_LOCAL
    (void)hipFree(g_nms.d_out);
    free(g_nms.h_out);
#else
    /* h_out and d_out alias the same hipHostMalloc'd unified buffer. */
    (void)hipHostFree(g_nms.h_out);
#endif
    g_nms_ready = 0;
    g_gpu_dispatch_count = 0;
}

/* ─── GPU dispatch telemetry ─────────────────────────────────────────────── */
extern "C" unsigned long ntt_get_gpu_dispatch_count(void) { return g_gpu_dispatch_count; }

/* ─── Forward / inverse NTT dispatch (Stockham vs CT-DIT) ─────────────────── */
/* I1 (2026-05-29, re-applied after the 4-step DFT fix): the LDS-fused
 * Stockham path keeps butterfly stages in LDS — ~1 HBM round-trip for
 * log_n<=10 and ~5 (transpose/F1/transpose/cross/F2) for the even-log_n
 * Bailey 4-step — versus CT-DIT's log_n separate global-stage kernels.
 * The lib2 4-step Stockham was fixed (ntt_kernel.hip: Bailey transpose +
 * reverse-order inverse) and is now convolution-correct (test_ntt_convolution
 * PASSES all primes at 4-step sizes). Stockham's square decomposition needs
 * even log_n>10; log_n<=10 runs single-block (any parity); odd log_n>10 has
 * no square split, so it falls back to CT-DIT. All tables built in init. */
/* Dispatch routing:
 *   log_n <= 10:        single-block Stockham (any parity).
 *   log_n > 10 even:    square Bailey 4-step Stockham (N1=N2=2^(log_n/2)).
 *   log_n > 10 odd:     rectangular Bailey 4-step Stockham (I11, N2=2*N1).
 *
 * I11 re-enabled (2026-06-07).  Exhaustive static analysis of launch bounds,
 * twiddle layout, LDS indices, and grid dimensions for log_n=19 found no
 * kernel defect.  Previous crashes at d=1M were attributed to cumulative GPU
 * driver state corruption from the preceding crash chain (crashes 1-6), not
 * the rect kernels.  The kernels pass test_ntt_rect_vs_ctdit at log_n=19 and
 * ntt_bigint_mul at log_n=11 (R11, d=10000).  Re-enabled incrementally: run
 * check-gpu, then d=200k, then d=1M to confirm. */
/* I11 (rect Stockham for odd log_n > 10) is DISABLED in production.
 * It GPU-faulted at d=1e6 (log_n=19) and at d=200k (log_n=17) in
 * production scale (real HBM pressure, 4 primes × 4 streams, hundreds of
 * dispatches) despite passing bounded unit tests at all odd log_n 11-19.
 * Odd log_n > 10 falls back to CT-DIT (launch_ntt / launch_intt), which
 * completed d=1e6 correctly in 10.25 s.  I11 remains in ntt_kernel.hip for
 * future offline root-cause investigation. */
/* log_n=22 FIXED 2026-06-16 (verified vs GMP, all 4.19M product limbs match):
 * the Stockham sub-transform is one-butterfly-per-thread and needs 2^(LOG_N_SUB-1)
 * threads; at LOG_N_SUB=11 (log_n=22) that is 1024, but it was capped at 512
 * (BLOCK_SIZE / __launch_bounds__) → half the butterflies never ran → corrupt
 * product. Fix in ntt_kernel.hip: stok_block clamp 512→1024, __launch_bounds__
 * →1024 for LOG_N_SUB>=11, StokMinBlks→1 block for that LDS footprint.
 * STILL UNSUPPORTED: even log_n>=24 (LOG_N_SUB>=12 needs >1024 threads → would
 * require a butterfly LOOP, not just more threads) and odd log_n=23 (CT-DIT 2^12
 * sub, untested). ntt_mul_init caps log_n<=22, so 22 is the validated ceiling. */
static inline int use_stockham(int log_n) { return log_n <= 10 || (log_n & 1) == 0; }

template<int PIDX>
static inline void fwd_ntt_dispatch(uint64_t* d, int log_n, hipStream_t s)
{
    if (use_stockham(log_n))
        launch_stockham_ntt<PIDX>(d, g_nms.d_tw_stok[PIDX], g_nms.d_tw_cross[PIDX], log_n, s);
    else
        /* odd log_n > 10: CT-DIT fallback (I11 disabled — see comment above) */
        launch_ntt<PIDX>(d, g_nms.d_tw[PIDX], g_nms.d_tw_cross[PIDX], log_n, s);
}

template<int PIDX>
static inline void inv_ntt_dispatch(uint64_t* d, int log_n, hipStream_t s)
{
    if (use_stockham(log_n))
        launch_stockham_intt<PIDX>(d, g_nms.n_inv[PIDX], g_nms.d_tw_stok_inv[PIDX],
                                   g_nms.d_tw_cross_inv_scaled[PIDX], log_n, s);
    else
        /* odd log_n > 10: CT-DIT fallback (I11 disabled) */
        launch_intt<PIDX>(d, g_nms.n_inv[PIDX], g_nms.d_tw_inv[PIDX],
                          g_nms.d_tw_cross[PIDX], log_n, s);
}

/* Hadamard-fused inverse for the even/Stockham path (I14): the pointwise
 * A*=B is folded into the inverse's first stockham pass, eliminating the
 * standalone hadamard_kernel launch + its HBM round trip. Even log_n only. */
template<int PIDX>
static inline void inv_had_dispatch(uint64_t* d, const uint64_t* b,
                                    int log_n, hipStream_t s)
{
    launch_stockham_intt_hadamard<PIDX>(d, b, g_nms.n_inv[PIDX],
        g_nms.d_tw_stok_inv[PIDX], g_nms.d_tw_cross_inv_scaled[PIDX], log_n, s);
}

/* ─── NTT BigInt multiply ────────────────────────────────────────────────── */

extern "C" void ntt_bigint_mul(BigInt *c, const BigInt *a, const BigInt *b)
{
    if (!g_nms_ready) {
        fprintf(stderr, "ntt_bigint_mul: call ntt_mul_init first\n"); exit(1);
    }
    g_gpu_dispatch_count++;
    int log_n = g_nms.log_n;
    int N     = 1 << log_n;

    prof_resolve();   /* normally a no-op: resolved in ntt_mul_init_impl() */
    double _pt = g_prof ? prof_ms() : 0.0;
    double _pt2 = (g_prof >= 2) ? prof_ms() : 0.0;

    /* Step 1+2: upload source limbs to device 0 (one H2D each), then run
     * GPU scatter per prime.  When an operand is managed (hipMallocManaged),
     * its limbs already reside in GPU-accessible memory (HBM3 on MI300A,
     * VRAM on gfx1030) so the H2D staging copy is skipped and the scatter
     * kernel reads the managed pointer directly.  On MI300A this eliminates
     * the CPU→HBM3 copy entirely; on gfx1030 the GPU reads VRAM directly
     * (faster than the staging path, though CPU ops on managed data pay PCIe). */
    int n_a_limbs = a->n_limbs, n_b_limbs = b->n_limbs;
    if (n_a_limbs > N) n_a_limbs = N;
    if (n_b_limbs > N) n_b_limbs = N;
    HIP_CHECK(hipSetDevice(0));
    const limb_t *src_a = a->managed ? a->limbs : g_nms.d_in_a;
    const limb_t *src_b = b->managed ? b->limbs : g_nms.d_in_b;
    if (!a->managed)
        HIP_CHECK(hipMemcpyAsync(g_nms.d_in_a, a->limbs,
                                 (size_t)n_a_limbs * sizeof(limb_t),
                                 hipMemcpyHostToDevice, g_nms.stream[0]));
    if (!b->managed)
        HIP_CHECK(hipMemcpyAsync(g_nms.d_in_b, b->limbs,
                                 (size_t)n_b_limbs * sizeof(limb_t),
                                 hipMemcpyHostToDevice, g_nms.stream[0]));
    /* Sync only when a staging copy was issued — if both inputs are managed
     * the scatter kernels can be launched immediately with no prior sync. */
    if (!a->managed || !b->managed)
        HIP_CHECK(hipStreamSynchronize(g_nms.stream[0]));
    if (g_prof) { g_prof_ms[0] += prof_ms() - _pt; _pt = prof_ms(); }

    HIP_CHECK(hipSetDevice(0 % g_nms.n_devs));
    launch_scatter<0>(src_a, n_a_limbs, g_nms.d_a[0], N, g_nms.stream[0]);
    launch_scatter<0>(src_b, n_b_limbs, g_nms.d_b[0], N, g_nms.stream[0]);
    HIP_CHECK(hipSetDevice(1 % g_nms.n_devs));
    launch_scatter<1>(src_a, n_a_limbs, g_nms.d_a[1], N, g_nms.stream[1]);
    launch_scatter<1>(src_b, n_b_limbs, g_nms.d_b[1], N, g_nms.stream[1]);
    HIP_CHECK(hipSetDevice(2 % g_nms.n_devs));
    launch_scatter<2>(src_a, n_a_limbs, g_nms.d_a[2], N, g_nms.stream[2]);
    launch_scatter<2>(src_b, n_b_limbs, g_nms.d_b[2], N, g_nms.stream[2]);
    HIP_CHECK(hipSetDevice(3 % g_nms.n_devs));
    launch_scatter<3>(src_a, n_a_limbs, g_nms.d_a[3], N, g_nms.stream[3]);
    launch_scatter<3>(src_b, n_b_limbs, g_nms.d_b[3], N, g_nms.stream[3]);

    /* I15: launch_stockham_large skips the initial in-place transpose (T is
     * pre-applied by scatter on the transfer_gpu path).  Here scatter is GPU-
     * side and writes natural order, so we must transpose explicitly before
     * handing data to launch_stockham_ntt for even log_n > 10. */
    if ((log_n > 10) && !(log_n & 1)) {
        int M4 = 1 << (log_n / 2);
        HIP_CHECK(hipSetDevice(0 % g_nms.n_devs));
        launch_transpose_sq(g_nms.d_a[0], M4, g_nms.stream[0]);
        launch_transpose_sq(g_nms.d_b[0], M4, g_nms.stream[0]);
        HIP_CHECK(hipSetDevice(1 % g_nms.n_devs));
        launch_transpose_sq(g_nms.d_a[1], M4, g_nms.stream[1]);
        launch_transpose_sq(g_nms.d_b[1], M4, g_nms.stream[1]);
        HIP_CHECK(hipSetDevice(2 % g_nms.n_devs));
        launch_transpose_sq(g_nms.d_a[2], M4, g_nms.stream[2]);
        launch_transpose_sq(g_nms.d_b[2], M4, g_nms.stream[2]);
        HIP_CHECK(hipSetDevice(3 % g_nms.n_devs));
        launch_transpose_sq(g_nms.d_a[3], M4, g_nms.stream[3]);
        launch_transpose_sq(g_nms.d_b[3], M4, g_nms.stream[3]);
    }

    if (g_prof >= 2) { for (int i=0;i<4;i++){ (void)hipSetDevice(i%g_nms.n_devs);
        (void)hipStreamSynchronize(g_nms.stream[i]); }
        g_prof_ms2[0] += prof_ms() - _pt2; _pt2 = prof_ms(); }

    /* Step 3: forward NTT — CT-DIT path (launch_ntt).  Root cause of the
     * log_n=16 Stockham regression identified and fixed in ntt_kernel.hip
     * (launch_stockham_large used (N1*N2)/1024 blocks for both row and col
     * passes; correct counts are N2 and N1 respectively — only equal at
     * log_n=20 where N1=N2=1024, explaining why test_ntt at log_n=20
     * passed while compute_e at log_n=16 failed).  Stockham can be
     * re-enabled here once verified on gfx1030 (G9 task). */
    fwd_ntt_dispatch<0>(g_nms.d_a[0], log_n, g_nms.stream[0]);
    fwd_ntt_dispatch<1>(g_nms.d_a[1], log_n, g_nms.stream[1]);
    fwd_ntt_dispatch<2>(g_nms.d_a[2], log_n, g_nms.stream[2]);
    fwd_ntt_dispatch<3>(g_nms.d_a[3], log_n, g_nms.stream[3]);
    fwd_ntt_dispatch<0>(g_nms.d_b[0], log_n, g_nms.stream[0]);
    fwd_ntt_dispatch<1>(g_nms.d_b[1], log_n, g_nms.stream[1]);
    fwd_ntt_dispatch<2>(g_nms.d_b[2], log_n, g_nms.stream[2]);
    fwd_ntt_dispatch<3>(g_nms.d_b[3], log_n, g_nms.stream[3]);

    /* Cooperative display yield — mid-dispatch (between forward and inverse NTT).
     * Sync all four streams so the GFX ring is momentarily idle, then sleep.
     * This bounds the maximum ring-busy window to max(fwd_NTT_time, inv_NTT_time)
     * instead of the full mul duration.  Critical on gfx1030 with cwsr_enable=0:
     * the top-level binary-split NTT mul at d>=1M saturated the ring for >5s,
     * causing DRM flip_done timeout and hard-hang (CRASH class, 2026-06-13).
     * The cwsr_enable=1 + GFXOFF-disabled system fix is the primary guard; this
     * is the code-layer defense-in-depth — it works even if cwsr is 0.
     * Controlled by NTT_DISPLAY_YIELD_US (0 = skip sync entirely, no overhead). */
    {
        static int s_mid_us = -1;
        if (__builtin_expect(s_mid_us < 0, 0)) {
            const char *e = getenv("NTT_DISPLAY_YIELD_US");
            s_mid_us = (e && atoi(e) > 0) ? atoi(e) : 0;
        }
        if (s_mid_us > 0) {
            for (int _i = 0; _i < 4; _i++) {
                (void)hipSetDevice(_i % g_nms.n_devs);
                (void)hipStreamSynchronize(g_nms.stream[_i]);
            }
            usleep((unsigned)s_mid_us);
        }
    }

    /* Steps 4+5: Hadamard A*=B then inverse NTT. The even/Stockham path
     * (I14) fuses the Hadamard into the inverse's first stockham pass; the
     * CT-DIT odd path keeps the standalone hadamard_kernel. */
    if (use_stockham(log_n)) {
        inv_had_dispatch<0>(g_nms.d_a[0], g_nms.d_b[0], log_n, g_nms.stream[0]);
        inv_had_dispatch<1>(g_nms.d_a[1], g_nms.d_b[1], log_n, g_nms.stream[1]);
        inv_had_dispatch<2>(g_nms.d_a[2], g_nms.d_b[2], log_n, g_nms.stream[2]);
        inv_had_dispatch<3>(g_nms.d_a[3], g_nms.d_b[3], log_n, g_nms.stream[3]);
        /* I15: INTT output is in transposed order for even log_n > 10; un-transpose
         * so Garner reads natural-order coefficients k=0..N-1. */
        if (log_n > 10) {
            int M4 = 1 << (log_n / 2);
            HIP_CHECK(hipSetDevice(0 % g_nms.n_devs));
            launch_transpose_sq(g_nms.d_a[0], M4, g_nms.stream[0]);
            HIP_CHECK(hipSetDevice(1 % g_nms.n_devs));
            launch_transpose_sq(g_nms.d_a[1], M4, g_nms.stream[1]);
            HIP_CHECK(hipSetDevice(2 % g_nms.n_devs));
            launch_transpose_sq(g_nms.d_a[2], M4, g_nms.stream[2]);
            HIP_CHECK(hipSetDevice(3 % g_nms.n_devs));
            launch_transpose_sq(g_nms.d_a[3], M4, g_nms.stream[3]);
        }
    } else {
        launch_hadamard<0>(g_nms.d_a[0], g_nms.d_b[0], N, g_nms.stream[0]);
        launch_hadamard<1>(g_nms.d_a[1], g_nms.d_b[1], N, g_nms.stream[1]);
        launch_hadamard<2>(g_nms.d_a[2], g_nms.d_b[2], N, g_nms.stream[2]);
        launch_hadamard<3>(g_nms.d_a[3], g_nms.d_b[3], N, g_nms.stream[3]);
        inv_ntt_dispatch<0>(g_nms.d_a[0], log_n, g_nms.stream[0]);
        inv_ntt_dispatch<1>(g_nms.d_a[1], log_n, g_nms.stream[1]);
        inv_ntt_dispatch<2>(g_nms.d_a[2], log_n, g_nms.stream[2]);
        inv_ntt_dispatch<3>(g_nms.d_a[3], log_n, g_nms.stream[3]);
    }

    if (g_prof >= 2) { for (int i=0;i<4;i++){ (void)hipSetDevice(i%g_nms.n_devs);
        (void)hipStreamSynchronize(g_nms.stream[i]); }
        g_prof_ms2[1] += prof_ms() - _pt2; _pt2 = prof_ms(); }

    /* Step 6: synchronize all four NTT/INTT streams. No D2H residue copy
     * — Garner runs on the GPU and reads residues device-local (or via
     * peer access on MI300A 4-APU). */
    for (int i = 0; i < 4; i++) {
        HIP_CHECK(hipSetDevice(i % g_nms.n_devs));
        HIP_CHECK(hipStreamSynchronize(g_nms.stream[i]));
    }
    HIP_CHECK(hipGetLastError());
    if (g_prof) { g_prof_ms[1] += prof_ms() - _pt; _pt = prof_ms(); }
    if (g_prof >= 2) { g_prof_ms2[2] += prof_ms() - _pt2; }

    /* Step 7: GPU Garner CRT → U256 array.
     *   GFX1030_LOCAL: writes to device-side d_out, then D2H to h_out.
     *   MI300A: d_out alias h_out (unified HBM); GPU write lands in
     *   host-visible memory and the D2H disappears. */
    HIP_CHECK(hipSetDevice(0));
    garner_reconstruct_gpu(g_nms.d_a[0], g_nms.d_a[1], g_nms.d_a[2], g_nms.d_a[3],
                           g_nms.d_out, N, &g_nms.g, g_nms.stream[0]);
#ifdef GFX1030_LOCAL
    HIP_CHECK(hipMemcpyAsync(g_nms.h_out, g_nms.d_out,
                             (size_t)N * sizeof(U256),
                             hipMemcpyDeviceToHost, g_nms.stream[0]));
#endif
    HIP_CHECK(hipStreamSynchronize(g_nms.stream[0]));
    if (g_prof) { g_prof_ms[2] += prof_ms() - _pt; _pt = prof_ms(); }

    /* Cooperative display yield — end-of-mul (post-Garner+D2H, ring idle).
     * Paired with the mid-dispatch yield above; together they bound the max
     * ring-busy window to max(fwd_NTT_time, inv_NTT_time) per mul call.
     * The ring is idle here: all streams are synced and Garner+D2H complete.
     * See mid-dispatch block above for the full crash-class explanation.
     *
     * Env: NTT_DISPLAY_YIELD_US    — sleep duration in µs (0 = disabled)
     *      NTT_DISPLAY_YIELD_EVERY — yield every N end-of-mul calls (default 1) */
    {
        static int           s_yield_us = -1;
        static int           s_yield_ev =  1;
        static unsigned long s_yield_n  =  0;
        if (__builtin_expect(s_yield_us < 0, 0)) {
            const char *e  = getenv("NTT_DISPLAY_YIELD_US");
            const char *ev = getenv("NTT_DISPLAY_YIELD_EVERY");
            s_yield_us = e  ? atoi(e)  : 0;
            s_yield_ev = (ev && atoi(ev) > 0) ? atoi(ev) : 1;
        }
        if (s_yield_us > 0 && (++s_yield_n % (unsigned)s_yield_ev == 0))
            usleep((unsigned)s_yield_us);
    }

    int n_coeffs = (a->n_limbs + b->n_limbs); /* product has at most this many significant coeffs */
    if (n_coeffs > N) n_coeffs = N;
    bigint_resize(c, n_coeffs + 4);
    bigint_gather(c, g_nms.h_out, n_coeffs);
    if (g_prof) { g_prof_ms[3] += prof_ms() - _pt; g_prof_calls++; }
}

/* ─── Emergency GPU reset (signal handler path) ─────────────────────────── */
/* Called from main.c's SIGINT/SIGTERM/SIGPIPE handler.  Drains in-flight GPU
 * work so rings are idle when the process exits via _exit().
 *
 * DO NOT call hipDeviceReset() here.  On gfx1030 (display GPU), hipDeviceReset
 * followed by KFD driver teardown on process exit conflicts and hard-crashes the
 * machine (CRASH 14, 2026-06-08).  hipDeviceSynchronize() alone is sufficient:
 * it waits for all submitted kernels to complete, leaving the ring empty.  The
 * KFD then cleans up the compute context on exit; gpu_recovery=1 handles any
 * residual ring state automatically.  Best-effort: errors silently ignored. */
extern "C" void ntt_gpu_emergency_reset(void)
{
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) n = 1;
    for (int i = 0; i < n && i < 4; i++) {
        (void)hipSetDevice(i);
        (void)hipDeviceSynchronize();
        /* hipDeviceReset() deliberately omitted — see comment above. */
    }
}
