/*
 * transfer_gpu.hip — NTT-based BigInt multiply for the CRT-NTT arith layer.
 *
 * Uses the 4-prime CRT pipeline (one GPU per prime) to compute the
 * polynomial product of two BigInts represented as coefficient arrays.
 *
 * Each BigInt with k 64-bit limbs is treated as a degree-(k-1) polynomial
 * evaluated at x=2^64.  The convolution product has 2k-1 coefficients,
 * each fitting in U256 (≤ k * (2^64)^2 ≤ 2^24 * 2^128 = 2^152 < 2^255).
 *
 * Pipeline per call (log_n > 10 — Bailey 4-step Stockham path):
 *   1. bigint_scatter_t → 4 per-prime host arrays, written transposed (I15)
 *   2. hipMemcpyAsync   → d_a on stream[i], d_b on stream_b[i] concurrently (A4)
 *   3. NTT(f) on stream[i] || NTT(g) on stream_b[i] in parallel (A4);
 *      hipStreamWaitEvent gates step 4+5 on both NTT completions
 *   4+5. launch_stockham_intt_hadamard<PIDX> → Hadamard fused into first
 *        INTT butterfly load (I14) + inverse NTT; final transpose_sq
 *        eliminated — output remains in transposed order on device (I15)
 *   6. hipStreamSynchronize × 4 (no D2H — GPU Garner reads d_a[i] directly)
 *   7. garner_reconstruct_gpu → N-parallel GPU Garner CRT on device 0,
 *      peer-reads d_a[1..3] via Infinity Fabric, writes to pinned h_out (A1)
 *   8. inline reorder (I15) + bigint_gather → untranspose h_gather_tmp, reconstruct
 *
 * For log_n <= 10 (single-block Stockham): bigint_scatter / bigint_gather
 * (no transposition); concurrent upload (A4); launch_stockham_ntt single-block;
 * launch_stockham_intt_hadamard; garner_reconstruct_gpu (A1); bigint_gather.
 *
 * State is global and single-threaded.  Call ntt_mul_init_impl() once.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#include "primes.h"
#include "shoup.h"
#include "crt_ntt.h"
#include "arith/bigint.h"

/* ─── Forward declarations from other translation units ──────────────────── */
template<int PIDX>
void launch_ntt(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_ntt(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_intt_fused(uint64_t*, uint64_t, const ShoupPair*,
                       const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_intt(uint64_t*, uint64_t, const ShoupPair*,
                          const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_hadamard(uint64_t*, const uint64_t*, int, hipStream_t);
template<int PIDX>
void launch_stockham_intt_hadamard(uint64_t*, const uint64_t*, uint64_t,
                                   const ShoupPair*, const ShoupPair*,
                                   int, hipStream_t);
template<int PIDX>
void launch_ntt_ct_dit_mont(uint64_t*, const uint64_t*, int, hipStream_t);
template<int PIDX>
void launch_intt_ct_dit_mont(uint64_t*, const uint64_t*, uint64_t, int, hipStream_t);
template<int PIDX>
void launch_stockham_ntt_rect(uint64_t*, uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_intt_rect(uint64_t*, uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);

GarnerConsts garner_precompute(void);
void garner_reconstruct(const uint64_t*, const uint64_t*, const uint64_t*,
                        const uint64_t*, U256*, int, const GarnerConsts*);
void garner_reconstruct_gpu(const uint64_t*, const uint64_t*,
                            const uint64_t*, const uint64_t*,
                            U256*, int, const GarnerConsts*, hipStream_t);

/* ─── GPU scatter/gather/CLA pipeline path (MI300A unified memory) ────────────
 * Opt-in via -DNTT_GPU_SCATTER_GATHER; inactive on the PCIe dev build
 * (GFX1030_LOCAL), where host scatter buffers are not device-local. DEFAULT OFF
 * — when off, the preprocessor removes every block below and the CPU
 * scatter/gather path is byte-for-byte unchanged. Enabling additionally
 * requires: (a) linking transfer_kernels.o, and (b) operand/result BigInt limbs
 * residing in unified (managed) memory so the kernels can read/write them. The
 * per-element cores are host-verified (test_transfer_core, and the e2e
 * core-routed integration trial); GPU runtime is pending MI300A (ROADMAP
 * G10/G11). Moves scatter (steps 1+2) + gather (step 8) off the CPU. */
#if defined(NTT_GPU_SCATTER_GATHER) && !defined(GFX1030_LOCAL)
#define NTT_GPU_SG 1
#else
#define NTT_GPU_SG 0
#endif

#if NTT_GPU_SG
/* Launchers from transfer_kernels.hip (host-verified cores in transfer_core.h). */
void launch_scatter_gpu(const limb_t*, int, uint64_t*, int, int, int, int, hipStream_t);
void launch_gather_cla_multiblock(const U256*, int, int, int, uint64_t*, uint64_t*,
                                  limb_t*, unsigned char*, unsigned char*,
                                  unsigned char*, unsigned char*, limb_t*, int, hipStream_t);
#endif

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

/* ─── Internal twiddle helpers (mirrors main.hip) ────────────────────────── */

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

/* I4: Stage-bucketed Montgomery twiddle table.  Format matches make_tw_local
 * but stores w*R mod p (8 B/entry) instead of ShoupPair (16 B/entry).
 * Used by launch_ntt_ct_dit_mont / launch_intt_ct_dit_mont for odd log_n > 10
 * where the Stockham 4-step is unavailable (requires even log_n / N1==N2). */
static uint64_t* make_tw_mont_local(int dev, uint64_t p, uint64_t g_root,
                                    int log_n, int prime_b)
{
    int n_tw = (1 << log_n) - 1;
    uint64_t* h = (uint64_t*)malloc((size_t)n_tw * sizeof(uint64_t));
    if (!h) { fprintf(stderr, "ntt_mul: make_tw_mont malloc failed\n"); exit(1); }
    /* R = 2^64; R mod p = 2^prime_b - 1 for these NTT-friendly primes. */
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
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMalloc(&d, (size_t)n_tw * sizeof(uint64_t)));
    HIP_CHECK(hipMemcpy(d, h, (size_t)n_tw * sizeof(uint64_t), hipMemcpyHostToDevice));
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
            /* n_inv is baked into every entry: cur starts at n_inv (not 1) so
             * every element carries the scale factor. This distributes the
             * N^{-1} scaling across the cross-twiddle pass at no extra cost. */
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

/* Stockham-format twiddle table: N1/2 + N2/2 ShoupPairs (log_n > 10) or
 * N/2 entries (log_n <= 10).  Passed to launch_stockham_ntt / _intt_hadamard.
 * Different from d_tw (CT-DIT stage-bucketed, N-1 entries) — wrong format
 * there caused Failure B (convolution wrong for even 4-step path). */
static ShoupPair* make_tw_stok_local(int dev, uint64_t p, uint64_t g_root,
                                      int log_n)
{
    int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
    int alloc_size = (log_n <= 10) ? (1 << log_n) / 2
                                   : (1 << (log_n1 - 1)) + (1 << (log_n2 - 1));
    ShoupPair* h = (ShoupPair*)malloc((size_t)alloc_size * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "ntt_mul: make_tw_stok malloc failed\n"); exit(1); }
    if (log_n <= 10) {
        uint64_t omega = modpow(g_root, (p - 1) / (uint64_t)(1 << log_n), p);
        uint64_t cur = 1;
        for (int k = 0; k < alloc_size; k++) {
            h[k] = make_shoup(cur, p);
            cur = (uint64_t)((__uint128_t)cur * omega % p);
        }
    } else {
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
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMalloc(&d, (size_t)alloc_size * sizeof(ShoupPair)));
    HIP_CHECK(hipMemcpy(d, h, (size_t)alloc_size * sizeof(ShoupPair),
                        hipMemcpyHostToDevice));
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
    /* C1/I14/I15: Stockham-format twiddles for the even-log_n 4-step path.
     * launch_stockham_ntt / launch_stockham_intt_hadamard take N1/2+N2/2
     * flat entries — NOT the stage-bucketed CT-DIT format in d_tw/d_tw_inv. */
    ShoupPair*   d_tw_stok[4];
    ShoupPair*   d_tw_stok_inv[4];
    /* I4: Montgomery 8-byte twiddles for CT-DIT odd log_n > 10 path (kept for
     * fallback / MI300A comparison; I11 rect path is the production odd path). */
    uint64_t*    d_tw_mont[4];
    uint64_t*    d_tw_inv_mont[4];
    /* I11: scratch buffers for out-of-place rect transpose (size N uint64_t per prime).
     * d_scr[i] for stream[i] (NTT(f)/INTT), d_scr_b[i] for stream_b[i] (NTT(g)).
     * Two separate arrays prevent the A4 concurrent streams from aliasing the same buffer. */
    uint64_t*    d_scr[4];
    uint64_t*    d_scr_b[4];
    hipStream_t  stream[4];    /* NTT(f) + Hadamard + INTT + Garner */
    hipStream_t  stream_b[4]; /* A4: NTT(g) concurrent with NTT(f) */
    hipEvent_t   ev_ntt_b[4]; /* A4: signals NTT(g) done per prime  */
    GarnerConsts g;
    U256*        h_out;         /* host Garner output: N U256s */
    uint64_t*    h_a[4];        /* host scatter buffers */
    uint64_t*    h_b[4];
    /* A2: n_inv precomputed at init for every log_n in [1, log_n].
     * Eliminates 4 modpow calls per ntt_bigint_mul call. Index: [prime][log_n]. */
    uint64_t     n_inv_table[4][23];
    /* A3: pre-allocated reorder buffer for the I15 inline untranspose.
     * The production path inlines the reorder loop (h_out[c*M+r] → tmp[k])
     * then calls bigint_gather; bigint_gather_t (which mallocs) is test-only. */
    U256*        h_gather_tmp;
#if NTT_GPU_SG
    /* Device scratch for the GPU gather + CLA carry-lookahead (device 0). */
    uint64_t*      d_gacc_lo;     /* gather phase-1 accumulator low half  */
    uint64_t*      d_gacc_hi;     /* gather phase-1 accumulator high half */
    limb_t*        d_gres;        /* CLA pre-carry digits                 */
    unsigned char* d_ggen;        /* CLA generate flags                   */
    unsigned char* d_gprop;       /* CLA propagate flags                  */
    unsigned char* d_gagg_g;      /* per-block aggregate generate         */
    unsigned char* d_gagg_p;      /* per-block aggregate propagate        */
    limb_t*        d_glimbs;      /* gathered product limbs (unified)     */
#endif
} NttMulState;

static NttMulState g_nms;
static int           g_nms_ready = 0;
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
    /* GPU-SAFETY: bound log_n. log_n<3 is degenerate; >22 would request
     * cross-twiddle tables of 2^log_n ShoupPair (16 B each) per prime —
     * e.g. log_n=24 → 256 MiB ×4 ×4 tables ≈ 4 GiB, courting OOM on the
     * 6900XT (16 GiB) and, with any failed alloc, a GPU fault.  The CRT-NTT
     * call sites need ≤ ~20 in practice.  Reject loudly, never silently. */
    if (max_log_n < 3 || max_log_n > 22) {
        fprintf(stderr, "ntt_mul_init: log_n=%d out of safe range [3,22]\n",
                max_log_n);
        exit(1);
    }
    int log_n = max_log_n;
    int N     = 1 << log_n;
    g_nms.log_n = log_n;

    g_nms.g   = garner_precompute();
    HIP_CHECK(hipGetDeviceCount(&g_nms.n_devs));
    if (g_nms.n_devs < 1) g_nms.n_devs = 1;
    if (g_nms.n_devs > 4) g_nms.n_devs = 4;
    /* R3: NUMA-aware pinned allocation.  hipHostMalloc with hipSetDevice
     * active places memory in the HBM domain of that APU on MI300A.  Pinned
     * memory also enables DMA-direct H2D/D2H, avoiding a system-memory
     * bounce buffer.  hipHostMallocNonCoherent matches the main.hip benchmark
     * convention for MI300A unified-HBM buffers. */
    HIP_CHECK(hipSetDevice(0));   /* h_out lives on APU 0 (Garner kernel domain) */
    HIP_CHECK(hipHostMalloc(&g_nms.h_out, (size_t)N * sizeof(U256),
                            hipHostMallocNonCoherent));
    if (!g_nms.h_out) { fprintf(stderr, "ntt_mul_init: OOM\n"); exit(1); }

#if NTT_GPU_SG
    /* GPU gather + CLA scratch on device 0 (gather reads peer residues there).
     * Sized for n_out = n_coeffs+4 <= N+4; rounded to N+8. d_glimbs is unified so
     * the host can read the product back. */
    {
        int M_ent = N + 8;
        int Bmax  = (M_ent + 1023) / 1024;   /* 1024 = CLA_SCAN_BLOCK (transfer_kernels.hip) */
        HIP_CHECK(hipMalloc(&g_nms.d_gacc_lo, (size_t)M_ent * sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&g_nms.d_gacc_hi, (size_t)M_ent * sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&g_nms.d_gres,    (size_t)M_ent * sizeof(limb_t)));
        HIP_CHECK(hipMalloc(&g_nms.d_ggen,    (size_t)M_ent));
        HIP_CHECK(hipMalloc(&g_nms.d_gprop,   (size_t)M_ent));
        HIP_CHECK(hipMalloc(&g_nms.d_gagg_g,  (size_t)Bmax));
        HIP_CHECK(hipMalloc(&g_nms.d_gagg_p,  (size_t)Bmax));
        HIP_CHECK(hipHostMalloc(&g_nms.d_glimbs, (size_t)M_ent * sizeof(limb_t),
                                hipHostMallocNonCoherent));
    }
#endif

    for (int i = 0; i < 4; i++) {
        int dev = i % g_nms.n_devs;
        HIP_CHECK(hipSetDevice(dev));
        HIP_CHECK(hipStreamCreate(&g_nms.stream[i]));
        HIP_CHECK(hipStreamCreate(&g_nms.stream_b[i]));   /* A4: NTT(g) stream */
        HIP_CHECK(hipEventCreateWithFlags(&g_nms.ev_ntt_b[i], hipEventDisableTiming));
        HIP_CHECK(hipMalloc(&g_nms.d_a[i], (size_t)N * sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&g_nms.d_b[i], (size_t)N * sizeof(uint64_t)));
        /* h_a[i] / h_b[i]: scatter buffers pinned to the same APU domain as
         * d_a[i]/d_b[i] so H2D copies land in NUMA-local HBM. */
        HIP_CHECK(hipHostMalloc(&g_nms.h_a[i], (size_t)N * sizeof(uint64_t),
                                hipHostMallocNonCoherent));
        HIP_CHECK(hipHostMalloc(&g_nms.h_b[i], (size_t)N * sizeof(uint64_t),
                                hipHostMallocNonCoherent));
        if (!g_nms.h_a[i] || !g_nms.h_b[i]) {
            fprintf(stderr, "ntt_mul_init: OOM h\n"); exit(1);
        }
        uint64_t p     = PRIMES[i];
        uint64_t root  = PRIM_ROOT[i];
        uint64_t n_inv = modpow((uint64_t)N, p - 2, p);
        g_nms.d_tw[i]       = make_tw_local(dev, p, root, log_n);
        /* Inverse twiddle: use root^{-1} = root^{p-2}. */
        uint64_t root_inv    = modpow(root, p - 2, p);
        g_nms.d_tw_inv[i]   = make_tw_local(dev, p, root_inv, log_n);
        g_nms.d_tw_cross[i] = make_tw_cross_local(dev, p, root, log_n);
        g_nms.d_tw_cross_inv_scaled[i] =
            make_tw_cross_inv_scaled_local(dev, p, root, log_n, n_inv);
        /* I4: Montgomery twiddle tables for CT-DIT odd-log_n path. */
        g_nms.d_tw_mont[i]     = make_tw_mont_local(dev, p, root,     log_n, PRIME_B[i]);
        g_nms.d_tw_inv_mont[i] = make_tw_mont_local(dev, p, root_inv, log_n, PRIME_B[i]);
        /* C1/I15: Stockham flat twiddles for the even-log_n 4-step path.
         * Also used by I11 rect path for odd log_n. */
        g_nms.d_tw_stok[i]     = make_tw_stok_local(dev, p, root,     log_n);
        g_nms.d_tw_stok_inv[i] = make_tw_stok_local(dev, p, root_inv, log_n);
        /* I11: scratch buffers — d_scr for stream[i], d_scr_b for stream_b[i]. */
        HIP_CHECK(hipMalloc(&g_nms.d_scr[i],   (size_t)N * sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&g_nms.d_scr_b[i], (size_t)N * sizeof(uint64_t)));
    }
    /* A1: enable peer access from device 0 to all other devices so the GPU
     * Garner kernel can read residue arrays on devices 1..n_devs-1 via
     * Infinity Fabric (MI300A) or PCIe (multi-GPU dev setups). */
    if (g_nms.n_devs > 1) {
        HIP_CHECK(hipSetDevice(0));
        for (int j = 1; j < g_nms.n_devs; j++) {
            hipError_t pe = hipDeviceEnablePeerAccess(j, 0);
            if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
                fprintf(stderr, "ntt_mul_init: peer access 0->%d unavailable "
                        "(GPU Garner will fall back to CPU)\n", j);
        }
    }

    /* A2: precompute n_inv for every log_n in [1, log_n] × 4 primes.
     * Avoids 4 modpow calls in the hot ntt_bigint_mul path; log_n ≤ 22
     * (enforced above), so n_inv_table[4][23] always covers the range. */
    for (int i = 0; i < 4; i++) {
        uint64_t p = PRIMES[i];
        for (int ln = 1; ln <= log_n; ln++)
            g_nms.n_inv_table[i][ln] = modpow((uint64_t)1 << ln, p - 2, p);
    }
    /* A3: pre-allocate the gather_t reorder buffer (N U256s = up to 2 MB).
     * Eliminates one malloc/free pair per ntt_bigint_mul call. */
    g_nms.h_gather_tmp = (U256*)malloc((size_t)N * sizeof(U256));
    if (!g_nms.h_gather_tmp) { fprintf(stderr, "ntt_mul_init: OOM gather_tmp\n"); exit(1); }
    g_nms_ready = 1;
}

extern "C" void ntt_mul_teardown_impl(void)
{
    if (!g_nms_ready) return;
    /* Flush all in-flight GPU work before releasing any resource.  hipStreamDestroy
     * drains its own queue, but a device-wide sync here ensures nothing is still
     * in flight on any ring when we start freeing VRAM and destroying streams.
     * This makes the KFD workqueue cleanup faster and more deterministic, reducing
     * the PASID-reuse race window for the next HIP process (CRASH 15, 2026-06-08). */
    (void)hipDeviceSynchronize();
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
    int N = 1 << g_nms.log_n;
    (void)N;
    for (int i = 0; i < 4; i++) {
        (void)hipSetDevice(i % g_nms.n_devs);
        (void)hipFree(g_nms.d_a[i]);  (void)hipFree(g_nms.d_b[i]);
        (void)hipFree(g_nms.d_tw[i]); (void)hipFree(g_nms.d_tw_inv[i]);
        (void)hipFree(g_nms.d_tw_cross[i]);
        (void)hipFree(g_nms.d_tw_cross_inv_scaled[i]);
        (void)hipFree(g_nms.d_tw_mont[i]); (void)hipFree(g_nms.d_tw_inv_mont[i]);
        (void)hipFree(g_nms.d_tw_stok[i]); (void)hipFree(g_nms.d_tw_stok_inv[i]);
        (void)hipFree(g_nms.d_scr[i]); (void)hipFree(g_nms.d_scr_b[i]);
        (void)hipStreamDestroy(g_nms.stream[i]);
        (void)hipStreamDestroy(g_nms.stream_b[i]);
        (void)hipEventDestroy(g_nms.ev_ntt_b[i]);
        (void)hipHostFree(g_nms.h_a[i]); (void)hipHostFree(g_nms.h_b[i]);
    }
    (void)hipHostFree(g_nms.h_out);
    free(g_nms.h_gather_tmp);
#if NTT_GPU_SG
    (void)hipSetDevice(0);
    (void)hipFree(g_nms.d_gacc_lo); (void)hipFree(g_nms.d_gacc_hi);
    (void)hipFree(g_nms.d_gres);
    (void)hipFree(g_nms.d_ggen);    (void)hipFree(g_nms.d_gprop);
    (void)hipFree(g_nms.d_gagg_g);  (void)hipFree(g_nms.d_gagg_p);
    (void)hipHostFree(g_nms.d_glimbs);
#endif
    g_nms_ready = 0;
    g_gpu_dispatch_count = 0;
}

/* ─── GPU dispatch telemetry ─────────────────────────────────────────────── */
/* Incremented each time ntt_bigint_mul dispatches to the GPU.  Reset by
 * ntt_mul_teardown. Checked by check_gpu.sh R7 to confirm GPU is actually used. */
extern "C" unsigned long ntt_get_gpu_dispatch_count(void) { return g_gpu_dispatch_count; }

/* ─── NTT BigInt multiply ────────────────────────────────────────────────── */

extern "C" void ntt_bigint_mul(BigInt *c, const BigInt *a, const BigInt *b)
{
    if (!g_nms_ready) {
        fprintf(stderr, "ntt_bigint_mul: call ntt_mul_init first\n"); exit(1);
    }
    g_gpu_dispatch_count++;
    /* R6: per-call log_n — use the smallest NTT that fits the product.
     * Stage-bucketed twiddle tables are prefix-valid: a sub-size call
     * reads only the first (N-1) entries of the table allocated for
     * g_nms.log_n, so no re-allocation is needed.  This reduces scatter,
     * H2D upload, NTT/INTT, and D2H work for small operands. */
    int log_n = 1;
    int n_needed = a->n_limbs + b->n_limbs;
    while ((1 << log_n) < n_needed) log_n++;
    if (log_n > g_nms.log_n) log_n = g_nms.log_n;
    /* I4: odd log_n > 10 uses CT-DIT with 8-byte Montgomery twiddles (avoids
     * the old 2× size overhead from rounding up to even).  Even log_n > 10
     * uses the Stockham 4-step (requires N1==N2, i.e. even log_n).
     * log_n <= 10 is always single-block Stockham (handles any parity). */
    /* True for odd log_n > 10. NOTE: the name is historical — this case is now
     * dispatched to the I11 rect Bailey 4-step (launch_stockham_ntt_rect), NOT
     * CT-DIT (see the forward-NTT dispatch below). lib3/ntt_mul.hip keeps I11
     * DISABLED here and uses CT-DIT instead — the two engines diverge (F6). */
    int use_ct_dit = (log_n > 10 && (log_n & 1));
    int N = 1 << log_n;

    prof_resolve();   /* normally a no-op: resolved in ntt_mul_init_impl() */
    double _pt = g_prof ? prof_ms() : 0.0;
    double _pt2 = (g_prof >= 2) ? prof_ms() : 0.0;

    /* Step 1: scatter operands to per-prime host arrays.
     * I15: even 4-step path (log_n > 10, use_ct_dit==0) writes transposed so
     * the GPU NTT skips the initial transpose_sq.  CT-DIT and single-block
     * paths scatter in natural order (no transposition needed). */
    int M = 1 << (log_n / 2);  /* sqrt(N); valid only for even log_n > 10 */
#if NTT_GPU_SG
    /* Steps 1+2 fused on-device: scatter directly into the residue arrays, one
     * kernel per prime per operand, no host buffer and no H2D. (a/b limbs must
     * be in unified memory — see NTT_GPU_SG note.) */
    int trn = (log_n > 10 && !use_ct_dit);   /* I15 transposed layout selector */
    for (int i = 0; i < 4; i++) {
        HIP_CHECK(hipSetDevice(i % g_nms.n_devs));
        launch_scatter_gpu(a->limbs, a->n_limbs, g_nms.d_a[i], N, M, trn, i, g_nms.stream[i]);
        launch_scatter_gpu(b->limbs, b->n_limbs, g_nms.d_b[i], N, M, trn, i, g_nms.stream_b[i]);
    }
    if (g_prof) { g_prof_ms[0] += prof_ms() - _pt; _pt = prof_ms(); }
#else
#pragma omp parallel for schedule(static) num_threads(4)
    for (int i = 0; i < 4; i++) {
        if (log_n > 10 && !use_ct_dit) {
            bigint_scatter_t(a, g_nms.h_a[i], M, i);
            bigint_scatter_t(b, g_nms.h_b[i], M, i);
        } else {
            bigint_scatter(a, g_nms.h_a[i], N, i);
            bigint_scatter(b, g_nms.h_b[i], N, i);
        }
    }
    if (g_prof) { g_prof_ms[0] += prof_ms() - _pt; _pt = prof_ms(); }

    /* Step 2: upload to device.
     * A4: d_a uploaded on stream[i], d_b uploaded on stream_b[i] concurrently. */
    for (int i = 0; i < 4; i++) {
        HIP_CHECK(hipSetDevice(i % g_nms.n_devs));
        HIP_CHECK(hipMemcpyAsync(g_nms.d_a[i], g_nms.h_a[i],
                             (size_t)N * sizeof(uint64_t),
                             hipMemcpyHostToDevice, g_nms.stream[i]));
        HIP_CHECK(hipMemcpyAsync(g_nms.d_b[i], g_nms.h_b[i],
                             (size_t)N * sizeof(uint64_t),
                             hipMemcpyHostToDevice, g_nms.stream_b[i]));
    }
#endif

    if (g_prof >= 2) { for (int i=0;i<4;i++){ (void)hipSetDevice(i%g_nms.n_devs);
        (void)hipStreamSynchronize(g_nms.stream[i]);
        (void)hipStreamSynchronize(g_nms.stream_b[i]); }
        g_prof_ms2[0] += prof_ms() - _pt2; _pt2 = prof_ms(); }

    /* Step 3: NTT(f) on stream[i] and NTT(g) on stream_b[i] concurrently (A4).
     * Three sub-cases:
     *   log_n <= 10:  single-block CT-DIT Shoup (launch_ntt) with d_tw.
     *   log_n > 10, odd:  I11 rect Bailey 4-step Stockham (launch_stockham_ntt_rect).
     *                     Uses d_tw_stok + d_tw_cross; d_scr is the out-of-place
     *                     transpose scratch (size N per prime).
     *   log_n > 10, even: square Bailey 4-step Stockham (launch_stockham_ntt). */
    if (log_n <= 10) {
        launch_ntt<0>(g_nms.d_a[0], g_nms.d_tw[0], g_nms.d_tw_cross[0], log_n, g_nms.stream[0]);
        launch_ntt<1>(g_nms.d_a[1], g_nms.d_tw[1], g_nms.d_tw_cross[1], log_n, g_nms.stream[1]);
        launch_ntt<2>(g_nms.d_a[2], g_nms.d_tw[2], g_nms.d_tw_cross[2], log_n, g_nms.stream[2]);
        launch_ntt<3>(g_nms.d_a[3], g_nms.d_tw[3], g_nms.d_tw_cross[3], log_n, g_nms.stream[3]);
        launch_ntt<0>(g_nms.d_b[0], g_nms.d_tw[0], g_nms.d_tw_cross[0], log_n, g_nms.stream_b[0]);
        launch_ntt<1>(g_nms.d_b[1], g_nms.d_tw[1], g_nms.d_tw_cross[1], log_n, g_nms.stream_b[1]);
        launch_ntt<2>(g_nms.d_b[2], g_nms.d_tw[2], g_nms.d_tw_cross[2], log_n, g_nms.stream_b[2]);
        launch_ntt<3>(g_nms.d_b[3], g_nms.d_tw[3], g_nms.d_tw_cross[3], log_n, g_nms.stream_b[3]);
    } else if (use_ct_dit) {
        /* I11: rect Bailey 4-step for odd log_n > 10.
         * scr buffer is N elements per prime; two out-of-place transposes cancel,
         * so output is in natural order (same gather path as CT-DIT). */
        launch_stockham_ntt_rect<0>(g_nms.d_a[0], g_nms.d_scr[0], g_nms.d_tw_stok[0], g_nms.d_tw_cross[0], log_n, g_nms.stream[0]);
        launch_stockham_ntt_rect<1>(g_nms.d_a[1], g_nms.d_scr[1], g_nms.d_tw_stok[1], g_nms.d_tw_cross[1], log_n, g_nms.stream[1]);
        launch_stockham_ntt_rect<2>(g_nms.d_a[2], g_nms.d_scr[2], g_nms.d_tw_stok[2], g_nms.d_tw_cross[2], log_n, g_nms.stream[2]);
        launch_stockham_ntt_rect<3>(g_nms.d_a[3], g_nms.d_scr[3], g_nms.d_tw_stok[3], g_nms.d_tw_cross[3], log_n, g_nms.stream[3]);
        launch_stockham_ntt_rect<0>(g_nms.d_b[0], g_nms.d_scr_b[0], g_nms.d_tw_stok[0], g_nms.d_tw_cross[0], log_n, g_nms.stream_b[0]);
        launch_stockham_ntt_rect<1>(g_nms.d_b[1], g_nms.d_scr_b[1], g_nms.d_tw_stok[1], g_nms.d_tw_cross[1], log_n, g_nms.stream_b[1]);
        launch_stockham_ntt_rect<2>(g_nms.d_b[2], g_nms.d_scr_b[2], g_nms.d_tw_stok[2], g_nms.d_tw_cross[2], log_n, g_nms.stream_b[2]);
        launch_stockham_ntt_rect<3>(g_nms.d_b[3], g_nms.d_scr_b[3], g_nms.d_tw_stok[3], g_nms.d_tw_cross[3], log_n, g_nms.stream_b[3]);
    } else {
        launch_stockham_ntt<0>(g_nms.d_a[0], g_nms.d_tw_stok[0], g_nms.d_tw_cross[0], log_n, g_nms.stream[0]);
        launch_stockham_ntt<1>(g_nms.d_a[1], g_nms.d_tw_stok[1], g_nms.d_tw_cross[1], log_n, g_nms.stream[1]);
        launch_stockham_ntt<2>(g_nms.d_a[2], g_nms.d_tw_stok[2], g_nms.d_tw_cross[2], log_n, g_nms.stream[2]);
        launch_stockham_ntt<3>(g_nms.d_a[3], g_nms.d_tw_stok[3], g_nms.d_tw_cross[3], log_n, g_nms.stream[3]);
        launch_stockham_ntt<0>(g_nms.d_b[0], g_nms.d_tw_stok[0], g_nms.d_tw_cross[0], log_n, g_nms.stream_b[0]);
        launch_stockham_ntt<1>(g_nms.d_b[1], g_nms.d_tw_stok[1], g_nms.d_tw_cross[1], log_n, g_nms.stream_b[1]);
        launch_stockham_ntt<2>(g_nms.d_b[2], g_nms.d_tw_stok[2], g_nms.d_tw_cross[2], log_n, g_nms.stream_b[2]);
        launch_stockham_ntt<3>(g_nms.d_b[3], g_nms.d_tw_stok[3], g_nms.d_tw_cross[3], log_n, g_nms.stream_b[3]);
    }
    /* Signal NTT(g) completion; stream[i] waits before Hadamard+INTT. */
    for (int i = 0; i < 4; i++) {
        HIP_CHECK(hipSetDevice(i % g_nms.n_devs));
        HIP_CHECK(hipEventRecord(g_nms.ev_ntt_b[i], g_nms.stream_b[i]));
        HIP_CHECK(hipStreamWaitEvent(g_nms.stream[i], g_nms.ev_ntt_b[i], 0));
    }

    /* Steps 4+5: Hadamard + INTT.
     * A2: n_inv precomputed at init for all log_n ≤ g_nms.log_n; zero cost here.
     * log_n <= 10: CT-DIT Shoup (standalone Hadamard + launch_intt_fused).
     * log_n > 10 odd: CT-DIT Montgomery (standalone Hadamard + launch_intt_ct_dit_mont).
     * log_n > 10 even: Hadamard fused into first INTT butterfly (I14). */
    uint64_t n_inv[4];
    for (int i = 0; i < 4; i++)
        n_inv[i] = g_nms.n_inv_table[i][log_n];
    if (log_n <= 10) {
        /* Single-block: CT-DIT Shoup with prefix-valid d_tw_inv. */
        launch_hadamard<0>(g_nms.d_a[0], g_nms.d_b[0], N, g_nms.stream[0]);
        launch_hadamard<1>(g_nms.d_a[1], g_nms.d_b[1], N, g_nms.stream[1]);
        launch_hadamard<2>(g_nms.d_a[2], g_nms.d_b[2], N, g_nms.stream[2]);
        launch_hadamard<3>(g_nms.d_a[3], g_nms.d_b[3], N, g_nms.stream[3]);
        launch_intt_fused<0>(g_nms.d_a[0], n_inv[0], g_nms.d_tw_inv[0],
                             g_nms.d_tw_cross_inv_scaled[0], log_n, g_nms.stream[0]);
        launch_intt_fused<1>(g_nms.d_a[1], n_inv[1], g_nms.d_tw_inv[1],
                             g_nms.d_tw_cross_inv_scaled[1], log_n, g_nms.stream[1]);
        launch_intt_fused<2>(g_nms.d_a[2], n_inv[2], g_nms.d_tw_inv[2],
                             g_nms.d_tw_cross_inv_scaled[2], log_n, g_nms.stream[2]);
        launch_intt_fused<3>(g_nms.d_a[3], n_inv[3], g_nms.d_tw_inv[3],
                             g_nms.d_tw_cross_inv_scaled[3], log_n, g_nms.stream[3]);
    } else if (use_ct_dit) {
        /* I11: rect Bailey 4-step INTT for odd log_n > 10.
         * Standalone Hadamard (not fused, unlike even path) followed by rect INTT.
         * n_inv is pre-baked into tw_cross_inv_scaled so the kernel ignores it. */
        launch_hadamard<0>(g_nms.d_a[0], g_nms.d_b[0], N, g_nms.stream[0]);
        launch_hadamard<1>(g_nms.d_a[1], g_nms.d_b[1], N, g_nms.stream[1]);
        launch_hadamard<2>(g_nms.d_a[2], g_nms.d_b[2], N, g_nms.stream[2]);
        launch_hadamard<3>(g_nms.d_a[3], g_nms.d_b[3], N, g_nms.stream[3]);
        launch_stockham_intt_rect<0>(g_nms.d_a[0], g_nms.d_scr[0], n_inv[0], g_nms.d_tw_stok_inv[0], g_nms.d_tw_cross_inv_scaled[0], log_n, g_nms.stream[0]);
        launch_stockham_intt_rect<1>(g_nms.d_a[1], g_nms.d_scr[1], n_inv[1], g_nms.d_tw_stok_inv[1], g_nms.d_tw_cross_inv_scaled[1], log_n, g_nms.stream[1]);
        launch_stockham_intt_rect<2>(g_nms.d_a[2], g_nms.d_scr[2], n_inv[2], g_nms.d_tw_stok_inv[2], g_nms.d_tw_cross_inv_scaled[2], log_n, g_nms.stream[2]);
        launch_stockham_intt_rect<3>(g_nms.d_a[3], g_nms.d_scr[3], n_inv[3], g_nms.d_tw_stok_inv[3], g_nms.d_tw_cross_inv_scaled[3], log_n, g_nms.stream[3]);
    } else {
        /* Even log_n > 10: Hadamard fused into first Stockham INTT pass (I14). */
        launch_stockham_intt_hadamard<0>(g_nms.d_a[0], g_nms.d_b[0], n_inv[0],
                                        g_nms.d_tw_stok_inv[0], g_nms.d_tw_cross_inv_scaled[0],
                                        log_n, g_nms.stream[0]);
        launch_stockham_intt_hadamard<1>(g_nms.d_a[1], g_nms.d_b[1], n_inv[1],
                                        g_nms.d_tw_stok_inv[1], g_nms.d_tw_cross_inv_scaled[1],
                                        log_n, g_nms.stream[1]);
        launch_stockham_intt_hadamard<2>(g_nms.d_a[2], g_nms.d_b[2], n_inv[2],
                                        g_nms.d_tw_stok_inv[2], g_nms.d_tw_cross_inv_scaled[2],
                                        log_n, g_nms.stream[2]);
        launch_stockham_intt_hadamard<3>(g_nms.d_a[3], g_nms.d_b[3], n_inv[3],
                                        g_nms.d_tw_stok_inv[3], g_nms.d_tw_cross_inv_scaled[3],
                                        log_n, g_nms.stream[3]);
    }

    if (g_prof >= 2) { for (int i=0;i<4;i++){ (void)hipSetDevice(i%g_nms.n_devs);
        (void)hipStreamSynchronize(g_nms.stream[i]); }
        g_prof_ms2[1] += prof_ms() - _pt2; _pt2 = prof_ms(); }

    /* Step 6: synchronize all INTT streams so residues in d_a[0..3] are stable.
     * A1: residues are read by GPU Garner directly — no D2H of per-prime arrays. */
    for (int i = 0; i < 4; i++) {
        HIP_CHECK(hipSetDevice(i % g_nms.n_devs));
        HIP_CHECK(hipStreamSynchronize(g_nms.stream[i]));
    }
    /* One error gate for the whole NTT/Hadamard/INTT kernel chain. */
    HIP_CHECK(hipGetLastError());
    if (g_prof) { g_prof_ms[1] += prof_ms() - _pt; _pt = prof_ms(); }
    if (g_prof >= 2) { g_prof_ms2[2] += prof_ms() - _pt2; }

    /* Step 7: GPU Garner CRT reconstruction (A1).
     * garner_reconstruct_gpu runs on device 0, reads d_a[1..3] via peer access
     * (Infinity Fabric on MI300A; PCIe on multi-GPU dev). Writes directly to
     * h_out (hipHostMalloc NonCoherent — device-writable pinned/unified memory).
     * Replaces the CPU OpenMP Garner (~67 ms at d=100k on 5950X dev host). */
    HIP_CHECK(hipSetDevice(0));
    garner_reconstruct_gpu(g_nms.d_a[0], g_nms.d_a[1], g_nms.d_a[2], g_nms.d_a[3],
                           g_nms.h_out, N, &g_nms.g, g_nms.stream[0]);
    HIP_CHECK(hipStreamSynchronize(g_nms.stream[0]));
    HIP_CHECK(hipGetLastError());
    if (g_prof) { g_prof_ms[2] += prof_ms() - _pt; _pt = prof_ms(); }
    int n_coeffs = (a->n_limbs + b->n_limbs); /* product has at most this many significant coeffs */
    if (n_coeffs > N) n_coeffs = N;
    bigint_resize(c, n_coeffs + 4);
#if NTT_GPU_SG
    /* Step 8 on-device: gather from the Garner U256 output with in-kernel
     * untranspose (trn) + carry-lookahead phase 2. Product limbs land in unified
     * d_glimbs; copy into c (value-correct; high zero limbs left untrimmed). */
    {
        int n_out = n_coeffs + 4;
        HIP_CHECK(hipSetDevice(0));
        launch_gather_cla_multiblock(g_nms.h_out, n_coeffs, M, trn,
                                     g_nms.d_gacc_lo, g_nms.d_gacc_hi, g_nms.d_gres,
                                     g_nms.d_ggen, g_nms.d_gprop, g_nms.d_gagg_g,
                                     g_nms.d_gagg_p, g_nms.d_glimbs, n_out,
                                     g_nms.stream[0]);
        HIP_CHECK(hipStreamSynchronize(g_nms.stream[0]));
        for (int j = 0; j < n_out; j++) c->limbs[j] = g_nms.d_glimbs[j];
        c->n_limbs = n_out;
    }
#else
    /* I15: even 4-step path: Garner output is in transposed order (final
     * transpose_sq removed from kernels); untranspose via h_gather_tmp.
     * CT-DIT and single-block paths: natural order, gather directly. */
    if (log_n > 10 && !use_ct_dit) {
        for (int k = 0; k < n_coeffs; k++) {
            int rr = k / M, cc = k % M;
            g_nms.h_gather_tmp[k] = g_nms.h_out[cc * M + rr];
        }
        bigint_gather(c, g_nms.h_gather_tmp, n_coeffs);
    } else {
        bigint_gather(c, g_nms.h_out, n_coeffs);
    }
#endif
    if (g_prof) { g_prof_ms[3] += prof_ms() - _pt; g_prof_calls++; }
}
