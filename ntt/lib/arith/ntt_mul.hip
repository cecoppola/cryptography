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
 * Pipeline per call (ntt_bigint_mul):
 *   1. upload operand limbs to device 0 (H2D; skipped for managed inputs)
 *   2. GPU scatter per prime → 4 device coefficient arrays (scatter_dispatch)
 *   3. forward NTT per prime (fwd_ntt_dispatch: LDS-Stockham 4-step for even
 *      log_n>10, recursive 3-factor for log_n=24, single LDS block for
 *      log_n<=10, CT-DIT for odd log_n>10)
 *   4+5. Hadamard A*=B fused into the inverse NTT's first pass (even/Stockham
 *      and 3-factor paths); the CT-DIT odd path does a standalone Hadamard +
 *      inverse NTT (fused n^-1 scale)
 *   6. synchronize the 4 streams — NO residue D2H copy (Garner runs on the GPU)
 *   7. GPU Garner CRT → U256 coefficient array (device d_out; aliases the host
 *      buffer on MI300A unified HBM, D2H copy on gfx1030)
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
/* 3-factor (recursive 4-step) NTT for log_n=24 (outer 12+12, inner 6+6). */
template<int PIDX>
void launch_stockham_3factor_ntt(uint64_t*, const ShoupPair*, const ShoupPair*,
                                 const ShoupPair*, hipStream_t);
template<int PIDX>
void launch_stockham_3factor_intt(uint64_t*, const ShoupPair*, const ShoupPair*,
                                  const ShoupPair*, hipStream_t);
template<int PIDX>
void launch_stockham_3factor_intt_hadamard(uint64_t*, const uint64_t*,
                                           const ShoupPair*, const ShoupPair*,
                                           const ShoupPair*, hipStream_t);
template<int PIDX>
void launch_stockham_3factor_F1_fwd(uint64_t*, const ShoupPair*, const ShoupPair*, hipStream_t);
template<int PIDX>
void launch_stockham_3factor_F1_inv(uint64_t*, const ShoupPair*, const ShoupPair*, hipStream_t);
void launch_transpose_sq(uint64_t* data, int M, hipStream_t stream);
void garner_reconstruct_gpu(const uint64_t*, const uint64_t*, const uint64_t*,
                            const uint64_t*, U256*, int, const GarnerConsts*,
                            hipStream_t);
/* I15: transposed-read Garner (un-transpose folded into the read; d_out natural). */
void garner_reconstruct_gpu_t(const uint64_t*, const uint64_t*, const uint64_t*,
                              const uint64_t*, U256*, int, int, const GarnerConsts*,
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

/* I15 transposed scatter: writes limb i to the TRANSPOSED position of the M×M
 * matrix (i = (r,c) row-major, r=i/M, c=i%M → position c*M+r), producing exactly
 * what natural scatter + an in-place transpose_sq would, so the even-log_n 4-step
 * (launch_stockham_ntt, which expects pre-transposed input) can run with the
 * standalone input transpose_sq eliminated. Same cost as natural scatter, just a
 * different store index. (Opt-in: only used when M>0, set under -DNTT_I15.) */
template<int PIDX>
__global__ __launch_bounds__(256, 8)
static void scatter_kernel_t(const limb_t* __restrict__ d_in,
                             int n_limbs, uint64_t* __restrict__ d_out, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    size_t dst = (size_t)(i % M) * (size_t)M + (size_t)(i / M);   /* transpose(i) */
    d_out[dst] = (i < n_limbs) ? (uint64_t)((__uint128_t)d_in[i] % PRIMES[PIDX]) : 0;
}

template<int PIDX>
static void launch_scatter_t(const limb_t* d_in, int n_limbs,
                             uint64_t* d_out, int N, int M, hipStream_t stream)
{
    const int block = 256;
    int grid = (N + block - 1) / block;
    if (grid < 1) grid = 1;
    scatter_kernel_t<PIDX><<<grid, block, 0, stream>>>(d_in, n_limbs, d_out, N, M);
}

/* Dispatch: transposed scatter when M_t>0 (I15 even-log_n path), else natural.
 * M_t is 0 unless built with -DNTT_I15, so the default path is byte-identical. */
template<int PIDX>
static inline void scatter_dispatch(const limb_t* d_in, int n_limbs,
                                    uint64_t* d_out, int N, int M_t, hipStream_t stream)
{
#ifdef NTT_I15
    if (M_t > 0) { launch_scatter_t<PIDX>(d_in, n_limbs, d_out, N, M_t, stream); return; }
#else
    (void)M_t;
#endif
    launch_scatter<PIDX>(d_in, n_limbs, d_out, N, stream);
}

/* ─── Internal twiddle helpers (mirrors main.hip) ────────────────────────── */

/* Flat (Stockham) twiddle helpers — mirror lib/main.hip make_tw_stockham. */
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
    /* 3-factor (recursive 4-step) tables — only allocated when log_n==24.
     * d_tw_stok_leaf: log_n=12 Stockham table (64 entries, inner 6+6 leaf NTTs).
     * d_tw_cross_inner: inner 64x64 cross-twiddle (4096 entries, omega_{4096}^{rc}).
     * d_tw_cross_inner_inv: inner 64x64 INVERSE cross-twiddle (omega_{4096}^{-rc},
     *   unscaled — the inverse 4-step needs the conjugate inner diagonal; N^{-1}
     *   scaling lives only in the outer d_tw_cross_inv_scaled). */
    ShoupPair*   d_tw_stok_leaf[4];
    ShoupPair*   d_tw_stok_leaf_inv[4];
    ShoupPair*   d_tw_cross_inner[4];
    ShoupPair*   d_tw_cross_inner_inv[4];
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
    /* GPU-SAFETY: bound log_n so an unbounded request can't OOM/GPU-fault.
     *   Lower bound 3: log_n<3 is degenerate (NTT semantics).
     *   Default cap 22 (conservative production default). The twiddle tables
     *   grow with N: the cross-twiddle is 2^log_n ShoupPair (16 B) per prime,
     *   so log_n=24 → 256 MiB per cross table ×2 (fwd+inv) ×4 primes ≈ 2 GiB,
     *   plus the rest of the working set (~6.9 GiB total, measured) — fits the
     *   16 GiB 6900XT but leaves little margin, hence the conservative default.
     * Higher log_n is supported but OPT-IN via NTT_MAX_LOGN (clamped to ≤26):
     *   log_n=23 routes through CT-DIT; log_n=24 through the recursive 3-factor
     *   4-step (fwd_ntt_dispatch). Both are GMP-verified; log_n=24 was also
     *   validated in a full 32M-digit compute_e run (2026-06-30). The 2-factor
     *   LDS-Stockham path itself still caps at even log_n<=22 (LOG_N_SUB<=11);
     *   24 needs the 3-factor tables, allocated only when log_n==24 (below). */
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

    /* Multi-device peer access (4-APU MI300A). Two hot cross-device reads make
     * this mandatory once n_devs>1: (1) the scatter kernels for primes 1..3 run
     * on devices 1..3 but read the operand-limb staging buffers d_in_a/d_in_b,
     * which live on device 0; (2) the GPU Garner on device 0 reads the per-prime
     * residue arrays d_a[1..3] on devices 1..3. Both dereference a peer device's
     * HBM, so every device must peer-access every other or the kernel page-faults
     * (DEPLOYMENT_DIAGNOSTIC P6). Mirrors lib/main.hip's crt_ntt init.
     * On the single-GPU dev box (n_devs==1) the loop body never executes — this
     * block is a no-op there and that path stays byte-identical.
     * hipErrorPeerAccessAlreadyEnabled is tolerated (a prior context or a second
     * init may have enabled it) and the sticky error is cleared, so it can never
     * trip the HIP_CHECK(hipGetLastError()) in ntt_bigint_mul. */
    for (int pi = 0; pi < g_nms.n_devs; pi++) {
        HIP_CHECK(hipSetDevice(pi));
        for (int pj = 0; pj < g_nms.n_devs; pj++) {
            if (pi == pj) continue;
            int can = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&can, pi, pj));
            if (!can) {
                fprintf(stderr, "ntt_mul_init: WARNING: device %d cannot peer-access "
                        "device %d — cross-device scatter/Garner will fault.\n", pi, pj);
                continue;
            }
            hipError_t _pe = hipDeviceEnablePeerAccess(pj, 0);
            if (_pe != hipSuccess && _pe != hipErrorPeerAccessAlreadyEnabled) {
                fprintf(stderr, "ntt_mul: HIP error %d (%s) enabling peer access "
                        "%d->%d at %s:%d\n", (int)_pe, hipGetErrorString(_pe),
                        pi, pj, __FILE__, __LINE__);
                fflush(stderr);
                exit(1);
            }
            (void)hipGetLastError();   /* clear a benign AlreadyEnabled sticky error */
        }
    }

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
        /* 3-factor tables: inner leaf (log_n=12 Stockham, 64 entries) and
         * inner 64x64 cross-twiddle (4096 entries); NULL for other sizes. */
        if (log_n == 24) {
            g_nms.d_tw_stok_leaf[i]       = make_tw_stockham_local(dev, p, root, 12);
            g_nms.d_tw_stok_leaf_inv[i]   = make_tw_stockham_inv_local(dev, p, root, 12);
            g_nms.d_tw_cross_inner[i]     = make_tw_cross_local(dev, p, root, 12);
            /* INVERSE inner cross-twiddle = omega^{-rc}: build with root_inv so the
             * inverse 4-step's inner transpose applies the conjugate diagonal
             * (forward table here corrupts every row r>=1; row 0 has omega^0=1
             * and was the only part that survived — the roundtrip-fail signature). */
            g_nms.d_tw_cross_inner_inv[i] = make_tw_cross_local(dev, p, root_inv, 12);
        } else {
            g_nms.d_tw_stok_leaf[i]       = NULL;
            g_nms.d_tw_stok_leaf_inv[i]   = NULL;
            g_nms.d_tw_cross_inner[i]     = NULL;
            g_nms.d_tw_cross_inner_inv[i] = NULL;
        }
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
        (void)hipFree(g_nms.d_tw_stok_leaf[i]);
        (void)hipFree(g_nms.d_tw_stok_leaf_inv[i]);
        (void)hipFree(g_nms.d_tw_cross_inner[i]);
        (void)hipFree(g_nms.d_tw_cross_inner_inv[i]);
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
 * The lib 4-step Stockham was fixed (ntt_kernel.hip: Bailey transpose +
 * reverse-order inverse) and is now convolution-correct (test_ntt_convolution
 * PASSES all primes at 4-step sizes). Stockham's square decomposition needs
 * even log_n>10; log_n<=10 runs single-block (any parity); odd log_n>10 has
 * no square split, so it falls back to CT-DIT. All tables built in init. */
/* Dispatch routing:
 *   log_n <= 10:        single-block Stockham (any parity).
 *   log_n > 10 even:    square Bailey 4-step Stockham (N1=N2=2^(log_n/2)).
 *   log_n > 10 odd:     CT-DIT (the I11 rect-Stockham path is disabled — see
 *                       the I11 note below). */
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
 * Beyond the 2-factor ceiling (even log_n=22): even log_n=24 is handled by the
 * recursive 3-factor 4-step (fwd_ntt_dispatch routes log_n==24), and odd
 * log_n=23 by CT-DIT — both GMP-verified, opt-in via NTT_MAX_LOGN. The default
 * ntt_mul_init cap stays 22; 24 is the validated ceiling (full 32M-digit run,
 * 2026-06-30). */
static inline int use_stockham(int log_n) { return log_n <= 10 || (log_n & 1) == 0; }

template<int PIDX>
static inline void fwd_ntt_dispatch(uint64_t* d, int log_n, hipStream_t s)
{
    if (log_n == 24) {   /* 3-factor recursive 4-step (outer 12+12, inner 6+6) */
        launch_stockham_3factor_ntt<PIDX>(d, g_nms.d_tw_stok_leaf[PIDX],
            g_nms.d_tw_cross_inner[PIDX], g_nms.d_tw_cross[PIDX], s);
        return;
    }
    if (use_stockham(log_n))
        launch_stockham_ntt<PIDX>(d, g_nms.d_tw_stok[PIDX], g_nms.d_tw_cross[PIDX], log_n, s);
    else
        /* odd log_n > 10: CT-DIT fallback (I11 disabled — see comment above) */
        launch_ntt<PIDX>(d, g_nms.d_tw[PIDX], g_nms.d_tw_cross[PIDX], log_n, s);
}

template<int PIDX>
static inline void inv_ntt_dispatch(uint64_t* d, int log_n, hipStream_t s)
{
    if (log_n == 24) {   /* 3-factor recursive 4-step inverse (plain, no Hadamard) */
        launch_stockham_3factor_intt<PIDX>(d, g_nms.d_tw_stok_leaf_inv[PIDX],
            g_nms.d_tw_cross_inner_inv[PIDX], g_nms.d_tw_cross_inv_scaled[PIDX], s);
        return;
    }
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
    if (log_n == 24) {   /* 3-factor Hadamard-fused inverse (inner uses inverse cross-tw) */
        launch_stockham_3factor_intt_hadamard<PIDX>(d, b, g_nms.d_tw_stok_leaf_inv[PIDX],
            g_nms.d_tw_cross_inner_inv[PIDX], g_nms.d_tw_cross_inv_scaled[PIDX], s);
        return;
    }
    launch_stockham_intt_hadamard<PIDX>(d, b, g_nms.n_inv[PIDX],
        g_nms.d_tw_stok_inv[PIDX], g_nms.d_tw_cross_inv_scaled[PIDX], log_n, s);
}

/* ─── 3-factor round-trip test ───────────────────────────────────────────── */
/* Tests NTT(x) then INTT on the result and checks equality with x.
 * Requires ntt_mul_init(24) to have been called. Returns 0 on pass, error count. */
extern "C" int ntt_mul_test_3factor_roundtrip(void)
{
    if (!g_nms_ready || g_nms.log_n != 24) {
        fprintf(stderr, "3factor_roundtrip: requires ntt_mul_init(24)\n");
        return -1;
    }
    int log_n = 24;
    size_t N = (size_t)1 << log_n;
    size_t bytes = N * sizeof(uint64_t);

    /* Allocate host + device buffers for prime 0 only (sufficient for roundtrip). */
    uint64_t* h_orig = (uint64_t*)malloc(bytes);
    uint64_t* h_out  = (uint64_t*)malloc(bytes);
    uint64_t* d_buf;
    if (!h_orig || !h_out) { fprintf(stderr, "3factor_roundtrip: OOM\n"); return -1; }
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&d_buf, bytes));

    /* Fill with pseudorandom values mod p (prime 0). */
    uint64_t p0 = PRIMES[0];
    uint64_t v = 0x123456789ABCDEFull;
    for (size_t i = 0; i < N; i++) {
        v = v * 6364136223846793005ull + 1442695040888963407ull;
        h_orig[i] = v % p0;
    }
    HIP_CHECK(hipMemcpy(d_buf, h_orig, bytes, hipMemcpyHostToDevice));

    hipStream_t s = g_nms.stream[0];

    /* Inspect forward NTT output on delta input to sanity-check the forward. */
    { uint64_t* h_delta = (uint64_t*)calloc(N, sizeof(uint64_t));
      uint64_t* d_delta;
      if (h_delta) {
          h_delta[0] = 1;
          HIP_CHECK(hipMalloc(&d_delta, bytes));
          HIP_CHECK(hipMemcpy(d_delta, h_delta, bytes, hipMemcpyHostToDevice));
          launch_stockham_3factor_ntt<0>(d_delta,
              g_nms.d_tw_stok_leaf[0], g_nms.d_tw_cross_inner[0],
              g_nms.d_tw_cross[0], s);
          HIP_CHECK(hipStreamSynchronize(s));
          HIP_CHECK(hipMemcpy(h_delta, d_delta, bytes, hipMemcpyDeviceToHost));
          /* NTT of delta = all-ones (regardless of output order). Count non-ones. */
          size_t bad_delta = 0;
          for (size_t i = 0; i < N; i++) if (h_delta[i] != 1) bad_delta++;
          fprintf(stderr, "  forward NTT delta check: %zu/%zu elements != 1\n",
                  bad_delta, N);
          if (bad_delta < 8) {
              for (size_t i = 0; i < N; i++)
                  if (h_delta[i] != 1)
                      fprintf(stderr, "    delta[%zu] = %llu\n", i, (unsigned long long)h_delta[i]);
          } else {
              fprintf(stderr, "    delta[0]=%llu [1]=%llu [64]=%llu [4096]=%llu\n",
                      (unsigned long long)h_delta[0], (unsigned long long)h_delta[1],
                      (unsigned long long)h_delta[64], (unsigned long long)h_delta[4096]);
          }
          (void)hipFree(d_delta);
          free(h_delta);
      }
    }

    /* Forward 3-factor NTT. */
    launch_stockham_3factor_ntt<0>(d_buf,
        g_nms.d_tw_stok_leaf[0], g_nms.d_tw_cross_inner[0],
        g_nms.d_tw_cross[0], s);
    HIP_CHECK(hipStreamSynchronize(s));
    { hipError_t e = hipGetLastError();
      if (e != hipSuccess)
          fprintf(stderr, "3factor_roundtrip: forward NTT GPU error: %s\n",
                  hipGetErrorString(e)); }
    /* Inverse 3-factor INTT. (inner uses the INVERSE inner cross-twiddle.) */
    launch_stockham_3factor_intt<0>(d_buf,
        g_nms.d_tw_stok_leaf_inv[0], g_nms.d_tw_cross_inner_inv[0],
        g_nms.d_tw_cross_inv_scaled[0], s);
    HIP_CHECK(hipStreamSynchronize(s));
    { hipError_t e = hipGetLastError();
      if (e != hipSuccess)
          fprintf(stderr, "3factor_roundtrip: inverse INTT GPU error: %s\n",
                  hipGetErrorString(e)); }

    HIP_CHECK(hipStreamSynchronize(s));
    HIP_CHECK(hipMemcpy(h_out, d_buf, bytes, hipMemcpyDeviceToHost));

    /* Scan for correct-element pattern to diagnose structural bugs. */
    size_t first_bad = N, first_good_after_bad = N;
    for (size_t i = 0; i < N; i++) {
        if (h_out[i] != h_orig[i] && first_bad == N) first_bad = i;
        if (h_out[i] == h_orig[i] && first_bad != N && first_good_after_bad == N)
            first_good_after_bad = i;
    }
    fprintf(stderr, "  first_bad=%zu  first_good_after_first_bad=%zu\n",
            first_bad, first_good_after_bad);
    /* Print first 4 correct indices and first 4 incorrect indices. */
    { int ng=0, nb=0;
      for (size_t i = 0; i < N && (ng < 4 || nb < 4); i++) {
          if (h_out[i] == h_orig[i] && ng < 4) {
              fprintf(stderr, "  GOOD[%d] @ %zu\n", ng, i); ng++;
          } else if (h_out[i] != h_orig[i] && nb < 4) {
              fprintf(stderr, "  BAD[%d]  @ %zu\n", nb, i); nb++;
          }
      }
    }

    int errs = 0;
    for (size_t i = 0; i < N; i++) {
        if (h_out[i] != h_orig[i]) errs++;
    }
    (void)hipFree(d_buf);
    free(h_orig); free(h_out);
    return errs;
}

/* Dense single-prime convolution: n random full-range coefficients vs a CPU
 * O(n^2) cyclic-convolution reference. Catches density/value-dependent bugs a
 * sparse input cannot, while staying out of scatter/Garner. n=8200 spans 3
 * outer-blocks, so it also exercises the outer 4-step (the multi-block path). */
template<int PIDX>
static int conv_dense(void)
{
    size_t N = (size_t)1 << 24, bytes = N * sizeof(uint64_t);
    uint64_t p = PRIMES[PIDX];
    int n = 8200;   /* spans 3 outer-blocks (>2*4096): exercises cross-outer-block */
    uint64_t *h_a = (uint64_t*)calloc(N, 8), *h_b = (uint64_t*)calloc(N, 8);
    uint64_t *h_c = (uint64_t*)malloc(bytes), *exp = (uint64_t*)calloc(N, 8);
    if (!h_a || !h_b || !h_c || !exp) { fprintf(stderr, "conv_dense: OOM\n"); return -1; }
    uint64_t st = 0x9e3779b97f4a7c15ULL ^ ((uint64_t)PIDX * 1234567ULL);
    for (int i = 0; i < n; i++) {
        st = st*6364136223846793005ULL + 1; h_a[i] = st % p;
        st = st*6364136223846793005ULL + 1; h_b[i] = st % p;
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int k = i + j;
            exp[k] = (uint64_t)(((__uint128_t)exp[k] + (__uint128_t)h_a[i]*h_b[j]) % p);
        }
    uint64_t *d_a, *d_b;
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&d_a, bytes)); HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));
    hipStream_t s = g_nms.stream[0];
    launch_stockham_3factor_ntt<PIDX>(d_a, g_nms.d_tw_stok_leaf[PIDX],
        g_nms.d_tw_cross_inner[PIDX], g_nms.d_tw_cross[PIDX], s);
    launch_stockham_3factor_ntt<PIDX>(d_b, g_nms.d_tw_stok_leaf[PIDX],
        g_nms.d_tw_cross_inner[PIDX], g_nms.d_tw_cross[PIDX], s);
    launch_hadamard<PIDX>(d_a, d_b, (int)N, s);
    launch_stockham_3factor_intt<PIDX>(d_a, g_nms.d_tw_stok_leaf_inv[PIDX],
        g_nms.d_tw_cross_inner_inv[PIDX], g_nms.d_tw_cross_inv_scaled[PIDX], s);
    HIP_CHECK(hipStreamSynchronize(s));
    HIP_CHECK(hipMemcpy(h_c, d_a, bytes, hipMemcpyDeviceToHost));
    int errs = 0, shown = 0;
    for (size_t k = 0; k < N; k++)
        if (h_c[k] != exp[k]) {
            if (shown < 4) { fprintf(stderr, "  P%d dense MISMATCH @%zu: gpu=%llu exp=%llu\n",
                PIDX, k, (unsigned long long)h_c[k], (unsigned long long)exp[k]); shown++; }
            errs++;
        }
    fprintf(stderr, "  P%d conv_dense(n=%d): %d mismatches\n", PIDX, n, errs);
    (void)hipFree(d_a); (void)hipFree(d_b);
    free(h_a); free(h_b); free(h_c); free(exp);
    return errs;
}

/* Position-sweep: a = A.d0 + B.dP, b = C.d0 + D.dP (full-range), so
 * a (*) b = AC.d0 + (AD+BC).dP + BD.d2P. Tests whether a NONZERO input at
 * position P (leaf-block P/64, outer-block P/4096) is transformed correctly.
 * Pinpoints the index threshold where the forward goes wrong. */
template<int PIDX>
static int conv_pos(int P)
{
    size_t N = (size_t)1 << 24, bytes = N * sizeof(uint64_t);
    uint64_t p = PRIMES[PIDX];
    uint64_t A = 0xdeadbeefcafef00dULL % p, B = 0x123456789abcdef0ULL % p;
    uint64_t C = 0xfedcba9876543210ULL % p, D = 0x0f1e2d3c4b5a6978ULL % p;
    uint64_t *h_a = (uint64_t*)calloc(N, 8), *h_b = (uint64_t*)calloc(N, 8);
    uint64_t *h_c = (uint64_t*)malloc(bytes), *exp = (uint64_t*)calloc(N, 8);
    if (!h_a || !h_b || !h_c || !exp) { fprintf(stderr, "conv_pos: OOM\n"); return -1; }
    h_a[0] = A; h_a[P] = B; h_b[0] = C; h_b[P] = D;
    int ai[2] = {0,P}; uint64_t av[2] = {A,B};
    int bi[2] = {0,P}; uint64_t bv[2] = {C,D};
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            size_t k = ((size_t)ai[i] + (size_t)bi[j]) % N;
            exp[k] = (uint64_t)(((__uint128_t)exp[k] + (__uint128_t)av[i]*bv[j]) % p);
        }
    uint64_t *d_a, *d_b;
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&d_a, bytes)); HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));
    hipStream_t s = g_nms.stream[0];
    launch_stockham_3factor_ntt<PIDX>(d_a, g_nms.d_tw_stok_leaf[PIDX],
        g_nms.d_tw_cross_inner[PIDX], g_nms.d_tw_cross[PIDX], s);
    launch_stockham_3factor_ntt<PIDX>(d_b, g_nms.d_tw_stok_leaf[PIDX],
        g_nms.d_tw_cross_inner[PIDX], g_nms.d_tw_cross[PIDX], s);
    launch_hadamard<PIDX>(d_a, d_b, (int)N, s);
    launch_stockham_3factor_intt<PIDX>(d_a, g_nms.d_tw_stok_leaf_inv[PIDX],
        g_nms.d_tw_cross_inner_inv[PIDX], g_nms.d_tw_cross_inv_scaled[PIDX], s);
    HIP_CHECK(hipStreamSynchronize(s));
    HIP_CHECK(hipMemcpy(h_c, d_a, bytes, hipMemcpyDeviceToHost));
    int errs = 0;
    for (size_t k = 0; k < N && errs == 0; k++) if (h_c[k] != exp[k]) errs = 1;
    (void)hipFree(d_a); (void)hipFree(d_b);
    free(h_a); free(h_b); free(h_c); free(exp);
    return errs;
}

/* Isolates the INNER 4-step (F1) alone: convolution within a single 4096-block
 * (input in [0,2048) of block 0) vs CPU 4096-cyclic reference. F1_inv∘F1_fwd =
 * 4096*identity, so scale by 1/4096. PASS => inner 4-step correct, bug is in the
 * outer T.X or F2. FAIL => bug is inside the inner 4-step. */
template<int PIDX>
static int conv_F1(void)
{
    size_t N = (size_t)1 << 24, bytes = N * sizeof(uint64_t);
    uint64_t p = PRIMES[PIDX];
    int n = 2048;
    uint64_t n_inv_4096 = modpow(4096, p - 2, p);
    uint64_t *h_a = (uint64_t*)calloc(N, 8), *h_b = (uint64_t*)calloc(N, 8);
    uint64_t *h_c = (uint64_t*)malloc(bytes), *exp = (uint64_t*)calloc(4096, 8);
    if (!h_a || !h_b || !h_c || !exp) { fprintf(stderr, "conv_F1: OOM\n"); return -1; }
    uint64_t st = 0xabcdef0123456789ULL ^ ((uint64_t)PIDX * 7777ULL);
    for (int i = 0; i < n; i++) {
        st = st*6364136223846793005ULL + 1; h_a[i] = st % p;
        st = st*6364136223846793005ULL + 1; h_b[i] = st % p;
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {           /* i+j < 4096: no wrap */
            int k = i + j;
            exp[k] = (uint64_t)(((__uint128_t)exp[k] + (__uint128_t)h_a[i]*h_b[j]) % p);
        }
    uint64_t *d_a, *d_b;
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMalloc(&d_a, bytes)); HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));
    hipStream_t s = g_nms.stream[0];
    launch_stockham_3factor_F1_fwd<PIDX>(d_a, g_nms.d_tw_stok_leaf[PIDX], g_nms.d_tw_cross_inner[PIDX], s);
    launch_stockham_3factor_F1_fwd<PIDX>(d_b, g_nms.d_tw_stok_leaf[PIDX], g_nms.d_tw_cross_inner[PIDX], s);
    launch_hadamard<PIDX>(d_a, d_b, (int)N, s);
    launch_stockham_3factor_F1_inv<PIDX>(d_a, g_nms.d_tw_stok_leaf_inv[PIDX], g_nms.d_tw_cross_inner_inv[PIDX], s);
    HIP_CHECK(hipStreamSynchronize(s));
    HIP_CHECK(hipMemcpy(h_c, d_a, bytes, hipMemcpyDeviceToHost));
    int errs = 0, shown = 0;
    for (int k = 0; k < 4096; k++) {
        uint64_t v = (uint64_t)((__uint128_t)h_c[k] * n_inv_4096 % p);
        if (v != exp[k]) {
            if (shown < 5) { fprintf(stderr, "  F1 conv MISMATCH @%d: gpu=%llu exp=%llu\n",
                k, (unsigned long long)v, (unsigned long long)exp[k]); shown++; }
            errs++;
        }
    }
    fprintf(stderr, "  P%d conv_F1 (inner 4-step only, 4096-block): %d mismatches\n", PIDX, errs);
    (void)hipFree(d_a); (void)hipFree(d_b);
    free(h_a); free(h_b); free(h_c); free(exp);
    return errs;
}

/* Checks F1_fwd produces NATURAL-ORDER DFT output (not just a consistent
 * convolution engine): F1_fwd(delta_1)[k] must equal omega_4096^k for k in
 * [0,4096). The outer 4-step REQUIRES F1's output to be natural order. */
template<int PIDX>
static int test_F1_order(void)
{
    size_t N = (size_t)1 << 24, bytes = N * sizeof(uint64_t);
    uint64_t p = PRIMES[PIDX], root = PRIM_ROOT[PIDX];
    uint64_t w4096 = modpow(root, (p - 1) / 4096, p);
    uint64_t *h = (uint64_t*)calloc(N, 8); if (!h) return -1;
    h[1] = 1;
    uint64_t *d; HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipMalloc(&d, bytes));
    HIP_CHECK(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice));
    launch_stockham_3factor_F1_fwd<PIDX>(d, g_nms.d_tw_stok_leaf[PIDX],
        g_nms.d_tw_cross_inner[PIDX], g_nms.stream[PIDX]);
    HIP_CHECK(hipStreamSynchronize(g_nms.stream[PIDX]));
    HIP_CHECK(hipMemcpy(h, d, bytes, hipMemcpyDeviceToHost));
    int errs = 0, shown = 0; uint64_t cur = 1;  /* omega^k */
    for (int k = 0; k < 4096; k++) {
        if (h[k] != cur) {
            if (shown < 5) { fprintf(stderr, "  F1_order MISMATCH @%d: got=%llu want omega^%d=%llu\n",
                k, (unsigned long long)h[k], k, (unsigned long long)cur); shown++; }
            errs++;
        }
        cur = (uint64_t)((__uint128_t)cur * w4096 % p);
    }
    fprintf(stderr, "  F1_order (delta_1 -> omega_4096^k natural): %d mismatches\n", errs);
    (void)hipFree(d); free(h);
    return errs;
}

extern "C" int ntt_mul_test_3factor_convolution(void)
{
    if (!g_nms_ready || g_nms.log_n != 24) {
        fprintf(stderr, "3factor_conv: requires ntt_mul_init(24)\n"); return -1;
    }
    /* 3-factor regression suite (each returns a mismatch count; sum==0 on pass):
     *   test_F1_order  F1 output is in natural DFT order (catches a missing T_final)
     *   conv_F1        inner 4-step is a correct 4096-pt cyclic convolution (one block)
     *   conv_pos sweep single delta at increasing OUTER-block offsets (outer handling)
     *   conv_dense     dense convolution spanning multiple outer-blocks (full path). */
    int e = test_F1_order<0>() + conv_F1<0>();
    int Ps[] = {1, 4097, 8193, 12289, 100001, 1000001};
    for (size_t i = 0; i < sizeof(Ps)/sizeof(Ps[0]); i++) {
        int m = conv_pos<0>(Ps[i]);
        fprintf(stderr, "  conv_pos P=%-8d (outer-block %d): %s\n",
                Ps[i], Ps[i]/4096, m ? "WRONG" : "ok");
        e += m;
    }
    e += conv_dense<0>();
    return e;
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

    /* Linear-convolution size guard. This engine computes a size-N CYCLIC
     * convolution (the NTT length is fixed at init to N=2^log_n). The polynomial
     * product of a (n_a coeffs) and b (n_b coeffs) has n_a+n_b-1 coefficients and
     * equals the cyclic result WITHOUT wraparound only when n_a+n_b-1 <= N. If a
     * caller ever presents operands violating this, the top coefficients fold back
     * onto the low ones (mod x^N - 1) and the product is silently WRONG — the
     * n_coeffs clamp below only truncates the output, it cannot detect the fold.
     * compute_e sizes max_log_n with margin so this never fires in-tree, but
     * ntt_bigint_mul is a public API: abort loudly rather than return garbage. */
    long need = (long)a->n_limbs + (long)b->n_limbs - 1;
    if (need > N) {
        fprintf(stderr,
            "ntt_bigint_mul: operands exceed the fixed NTT size — product needs "
            "%ld coeffs but N=%d (log_n=%d). Re-init ntt_mul with a larger "
            "max_log_n; the size-N cyclic convolution would otherwise wrap "
            "silently and corrupt the result.\n", need, N, log_n);
        fflush(stderr);
        exit(1);
    }

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

    /* I15 (opt-in via -DNTT_I15): on the even-log_n 4-step path, scatter writes
     * the operand TRANSPOSED so the standalone input transpose_sq is skipped.
     * M15 is 0 on every other path — and always 0 without -DNTT_I15 — so the
     * default build is byte-identical to natural scatter + explicit transpose. */
    int M15 = 0;
#ifdef NTT_I15
    if ((log_n > 10) && !(log_n & 1) && log_n != 24) M15 = 1 << (log_n / 2);
#endif
    HIP_CHECK(hipSetDevice(0 % g_nms.n_devs));
    scatter_dispatch<0>(src_a, n_a_limbs, g_nms.d_a[0], N, M15, g_nms.stream[0]);
    scatter_dispatch<0>(src_b, n_b_limbs, g_nms.d_b[0], N, M15, g_nms.stream[0]);
    HIP_CHECK(hipSetDevice(1 % g_nms.n_devs));
    scatter_dispatch<1>(src_a, n_a_limbs, g_nms.d_a[1], N, M15, g_nms.stream[1]);
    scatter_dispatch<1>(src_b, n_b_limbs, g_nms.d_b[1], N, M15, g_nms.stream[1]);
    HIP_CHECK(hipSetDevice(2 % g_nms.n_devs));
    scatter_dispatch<2>(src_a, n_a_limbs, g_nms.d_a[2], N, M15, g_nms.stream[2]);
    scatter_dispatch<2>(src_b, n_b_limbs, g_nms.d_b[2], N, M15, g_nms.stream[2]);
    HIP_CHECK(hipSetDevice(3 % g_nms.n_devs));
    scatter_dispatch<3>(src_a, n_a_limbs, g_nms.d_a[3], N, M15, g_nms.stream[3]);
    scatter_dispatch<3>(src_b, n_b_limbs, g_nms.d_b[3], N, M15, g_nms.stream[3]);

    /* launch_stockham_ntt (even log_n>10) expects PRE-TRANSPOSED input. Default:
     * natural scatter above, so transpose explicitly here. With I15 (M15>0) the
     * scatter already wrote transposed, so this standalone pass is eliminated.
     * log_n==24 (3-factor) is natural→natural: input pre-transpose AND the output
     * un-transpose (below) are both skipped. */
    if ((log_n > 10) && !(log_n & 1) && log_n != 24 && M15 == 0) {
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

    /* Step 3: forward NTT per prime via fwd_ntt_dispatch — LDS-fused Stockham
     * 4-step for even log_n>10, the recursive 3-factor 4-step for log_n=24, a
     * single LDS Stockham block for log_n<=10, and CT-DIT for odd log_n>10
     * (the I11 rect-Stockham path being disabled — see the dispatch comment). */
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
     * (I14) fuses the Hadamard into the inverse's first pass; log_n==24 uses the
     * 3-factor fused-Hadamard inverse (inv_had_dispatch); the CT-DIT odd path
     * keeps the standalone hadamard_kernel. */
    if (use_stockham(log_n)) {
        inv_had_dispatch<0>(g_nms.d_a[0], g_nms.d_b[0], log_n, g_nms.stream[0]);
        inv_had_dispatch<1>(g_nms.d_a[1], g_nms.d_b[1], log_n, g_nms.stream[1]);
        inv_had_dispatch<2>(g_nms.d_a[2], g_nms.d_b[2], log_n, g_nms.stream[2]);
        inv_had_dispatch<3>(g_nms.d_a[3], g_nms.d_b[3], log_n, g_nms.stream[3]);
        /* INTT output is in transposed order for even log_n > 10. Default: un-
         * transpose here so Garner reads natural coefficients. With I15 (M15>0)
         * the transposed-read Garner (garner_reconstruct_gpu_t below) folds the
         * un-transpose into its read, so this standalone pass is eliminated.
         * log_n==24 (3-factor) already returns natural order — skip (see input). */
        if (log_n > 10 && log_n != 24 && M15 == 0) {
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
    if (M15 > 0)   /* I15: read residues transposed; write d_out natural */
        garner_reconstruct_gpu_t(g_nms.d_a[0], g_nms.d_a[1], g_nms.d_a[2], g_nms.d_a[3],
                                 g_nms.d_out, N, M15, &g_nms.g, g_nms.stream[0]);
    else
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
