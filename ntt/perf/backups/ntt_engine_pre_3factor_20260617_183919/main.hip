/*
 * main.hip — CRT-NTT polynomial multiplication pipeline and benchmark
 *
 * Compiled with: hipcc -O3 --offload-arch=gfx942
 *
 * REQUIRED runtime environment (set before srun):
 *   export ROCM_PATH=/opt/rocm-7.0.3
 *   export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
 *   export HSA_ENABLE_SDMA=1           # activates SDMA for hipMemcpyPeerAsync
 *   export ROCR_VISIBLE_DEVICES=0,1,2,3
 *   export HIP_VISIBLE_DEVICES=0,1,2,3
 *
 * REQUIRED: all execution via srun (direct ./binary fails, cgroup restriction):
 *   Single-node:  srun -p mi300a -N 1 -n 1 ./crt_ntt 20 100
 *   Multi-node:   srun -N 2 -n 2 --ntasks-per-node=1 ./crt_ntt 20 100
 *
 * Pipeline (single node, N-point polynomial multiply f*g, four primes):
 *   1. Upload:    hipMemcpyAsync host -> APU 0 HBM (f and g)
 *   2. Broadcast: hipMemcpyPeerAsync APU 0 -> APUs 1,2,3 (f then g)
 *                 (66.63 GB/s SDMA per link; adequate for N <= 2^22)
 *                 (for N > 2^23 use kernel-driven P2P for 104 GB/s)
 *   3. NTT(f)||NTT(g): launch on stream (f) and stream_g (g) concurrently
 *                      hipEventRecord(ev_g) + hipStreamWaitEvent gates Hadamard
 *   5. Hadamard:  4 concurrent launch_hadamard<0..3>
 *   6. INTT:      4 concurrent launch_intt<0..3> (inverse twiddles + scale)
 *   7. Sync:      hipStreamSynchronize x4
 *   8. Garner:    96-core OpenMP CRT reconstruction
 *                 (OMP_PROC_BIND=spread places 24 threads per APU)
 *
 * Measured performance baselines (MI300A_env0429.md, bench.pdf 2026-05-04):
 *   HBM bandwidth:  4.0 TB/s (measured; 5.3 TB/s is theoretical peak)
 *   LDS bandwidth:  5.0 TB/s (measured)
 *   P2P per-link:   66.63 GB/s SDMA unidirectional, 104 GB/s kernel-driven
 *   P2P all-to-all: 650 GB/s sustained (60.2% of 978 GB/s theoretical peak)
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "primes.h"
#include "shoup.h"
#include "crt_ntt.h"

/* GPU-SAFETY: HIP_CHECK aborts on the host the instant a HIP call fails,
 * so a transient failure (e.g. allocator exhaustion on one of 4 APUs)
 * cannot cascade into a kernel reading a NULL device pointer / GPU reset.
 * Mirrors lib2/arith/ntt_mul.hip. Cold setup/teardown only; zero hot cost. */
#define HIP_CHECK(expr) do {                                                  \
    hipError_t _e = (expr);                                                   \
    if (_e != hipSuccess) {                                                   \
        fprintf(stderr, "lib2: HIP error %d (%s) at %s:%d: %s\n",            \
                (int)_e, hipGetErrorString(_e), __FILE__, __LINE__, #expr);   \
        fflush(stderr);                                                       \
        exit(1);                                                              \
    }                                                                         \
} while (0)

/* Forward declarations — implementations in other translation units */
template<int PIDX>
void launch_ntt(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_intt(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_hadamard(uint64_t*, const uint64_t*, int, hipStream_t);
template<int PIDX>
void launch_intt_fused(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_twist(uint64_t*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_untwist(uint64_t*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_mont_ntt(uint64_t*, const uint64_t*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_ntt(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_stockham_intt(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
void init_ntt_device(int dev);
GarnerConsts garner_precompute(void);
void garner_reconstruct(const uint64_t*, const uint64_t*, const uint64_t*,
                        const uint64_t*, U256*, int, const GarnerConsts*);
void garner_reconstruct_gpu(const uint64_t*, const uint64_t*, const uint64_t*,
                            const uint64_t*, U256*, int, const GarnerConsts*,
                            hipStream_t);
int  verify_primes(void);
/* transfer_gpu.hip */
void broadcast_input(const uint64_t*, uint64_t*[4], hipStream_t[4][4], size_t);
/* transfer_shmem.c (only used in multi-node builds) */
void shmem_distribute_input_collective(void*, size_t, int);

/* ─── Per-APU context ────────────────────────────────────────────────────── */
typedef struct {
    int         dev_id;
    uint64_t*   d_coeffs;       /* polynomial f residues: N uint64_t          */
    uint64_t*   d_g_coeffs;     /* polynomial g residues: N uint64_t          */
    ShoupPair*  d_tw;           /* forward twiddles: N/2 ShoupPairs           */
    ShoupPair*  d_tw_inv;       /* inverse twiddles: N/2 ShoupPairs           */
    ShoupPair*  d_tw_cross;     /* forward cross-twiddles (4-step, N entries) */
    uint64_t    n_inv;          /* N^{-1} mod p[dev_id], for INTT scaling     */
    hipStream_t stream;         /* NTT compute stream for f                   */
    hipStream_t stream_g;       /* second stream: NTT(g) concurrent with f    */
    hipEvent_t  ev_g;           /* signals NTT(g) completion on stream_g      */
    ShoupPair*  d_psi;          /* psi^k table: negacyclic pre-twist          */
    ShoupPair*  d_psi_inv;           /* psi^{-k} table: negacyclic post-untwist    */
    ShoupPair*  d_tw_cross_inv_scaled; /* inverse cross-twiddles with n_inv baked in */
    uint64_t*   d_tw_mont;            /* forward twiddles in Montgomery form: 8B/entry */
    ShoupPair*  d_tw_stok;            /* forward flat Stockham twiddle table           */
    ShoupPair*  d_tw_stok_inv;        /* inverse flat Stockham twiddle table           */
    U256*       d_out;                /* device U256 output for GPU Garner             */
} APUCtx;

static APUCtx       ctx[4];
static GarnerConsts gconsts;

/* One dedicated stream per directed link (src, dst).
 * CONFIRMED: sharing streams between concurrent peer copies causes HSA
 * command queue overflow on this system. */
static hipStream_t link_streams[4][4];

/* Actual APU count: set in init() via hipGetDeviceCount, capped at 4.
 * 4 NTT primes are always assigned; when n_devs < 4, prime i runs on
 * device ctx[i].dev_id = i % g_n_devs. */
static int g_n_devs = 4;

/* ─── Twiddle table builders ─────────────────────────────────────────────── */
static ShoupPair* make_tw(int dev, uint64_t p, uint64_t g_root, int log_n)
{
    int N    = 1 << log_n;
    /* Stage-bucketed table: stage s holds 2^s entries, total
     * sum_{s=0}^{log_n-1} 2^s = N-1. Must allocate N-1 (NOT N/2): the fill
     * loop below writes h[0 .. N-2] and the kernel reads tw[tw_base+pos]
     * with tw_base accumulating 2^s up to N-2. (Matches the correct
     * ntt_mul.hip:make_tw_local; the old N/2 overran the host buffer and
     * truncated the device table for every log_n >= 2.) */
    int n_tw = N - 1;
    ShoupPair* h = (ShoupPair*)malloc((size_t)n_tw * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "make_tw: malloc failed\n"); exit(1); }

    uint64_t omega = modpow(g_root, (p - 1) / (uint64_t)N, p);
    int offset = 0, s;
    for (s = 0; s < log_n; s++) {
        int cnt = 1 << s;
        uint64_t root = modpow(omega, (uint64_t)(N >> (s + 1)), p);
        uint64_t cur  = 1;
        int j;
        for (j = 0; j < cnt; j++) {
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

/* Inverse twiddle table: same layout as make_tw but using omega^{-1}.       */
static ShoupPair* make_tw_inv(int dev, uint64_t p, uint64_t g_root, int log_n)
{
    /* g_root_inv generates the same cyclic group in reverse order.           */
    /* g_root_inv^{(p-1)/N} = omega^{-1}, the inverse N-th root of unity.   */
    uint64_t g_root_inv = modpow(g_root, p - 2, p);
    return make_tw(dev, p, g_root_inv, log_n);
}

static ShoupPair* make_tw_cross(int dev, uint64_t p, uint64_t g_root, int log_n)
{
    if (log_n <= 10) return NULL;

    int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
    int N1 = 1 << log_n1, N2 = 1 << log_n2;
    int total = N1 * N2;

    uint64_t omega = modpow(g_root, (p - 1) / (uint64_t)total, p);
    ShoupPair* h = (ShoupPair*)malloc((size_t)total * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "make_tw_cross: malloc failed\n"); exit(1); }

    int r, c;
    for (r = 0; r < N1; r++) {
        uint64_t row_base = modpow(omega, (uint64_t)r, p);
        uint64_t cur = 1;
        for (c = 0; c < N2; c++) {
            h[r * N2 + c] = make_shoup(cur, p);
            cur = (uint64_t)((__uint128_t)cur * row_base % p);
        }
    }
    ShoupPair* d;
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMalloc(&d, (size_t)total * sizeof(ShoupPair)));
    HIP_CHECK(hipMemcpy(d, h, (size_t)total * sizeof(ShoupPair), hipMemcpyHostToDevice));
    free(h);
    return d;
}


/* Inverse cross-twiddle table with N^{-1} pre-multiplied into every entry.  */
/* Used by launch_intt_fused for log_n > 10: N^{-1} is distributed across all*/
/* N elements via the existing cross_twiddle_kernel at no extra runtime cost. */
static ShoupPair* make_tw_cross_inv_scaled(int dev, uint64_t p, uint64_t g_root,
                                            int log_n, uint64_t n_inv)
{
    if (log_n <= 10) return NULL;

    int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
    int N1 = 1 << log_n1, N2 = 1 << log_n2;
    int total = N1 * N2;

    uint64_t g_root_inv = modpow(g_root, p - 2, p);
    uint64_t omega_inv  = modpow(g_root_inv, (p - 1) / (uint64_t)total, p);

    ShoupPair* h = (ShoupPair*)malloc((size_t)total * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "make_tw_cross_inv_scaled: malloc failed\n"); exit(1); }

    int r, c;
    for (r = 0; r < N1; r++) {
        uint64_t row_base = modpow(omega_inv, (uint64_t)r, p);
        /* Start each row at n_inv (not 1), baking n_inv * omega_inv^(r*c). */
        uint64_t cur = n_inv;
        for (c = 0; c < N2; c++) {
            h[r * N2 + c] = make_shoup(cur, p);
            cur = (uint64_t)((__uint128_t)cur * row_base % p);
        }
    }
    ShoupPair* d;
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMalloc(&d, (size_t)total * sizeof(ShoupPair)));
    HIP_CHECK(hipMemcpy(d, h, (size_t)total * sizeof(ShoupPair), hipMemcpyHostToDevice));
    free(h);
    return d;
}

/* Negacyclic twist table: psi^k for k = 0..N-1, psi = g^{(p-1)/(2N)}.     */
/* If inv=1, builds psi^{-k} (untwist). Both tables are N entries.          */
static ShoupPair* make_psi_table(int dev, uint64_t p, uint64_t g_root,
                                  int log_n, int inv)
{
    int N = 1 << log_n;
    uint64_t psi = modpow(g_root, (p - 1) / ((uint64_t)2 * (uint64_t)N), p);
    if (inv) psi = modpow(psi, p - 2, p);

    ShoupPair* h = (ShoupPair*)malloc((size_t)N * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "make_psi_table: malloc failed\n"); exit(1); }

    uint64_t cur = 1;
    int k;
    for (k = 0; k < N; k++) {
        h[k] = make_shoup(cur, p);
        cur  = (uint64_t)((__uint128_t)cur * psi % p);
    }
    ShoupPair* d;
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMalloc(&d, (size_t)N * sizeof(ShoupPair)));
    HIP_CHECK(hipMemcpy(d, h, (size_t)N * sizeof(ShoupPair), hipMemcpyHostToDevice));
    free(h);
    return d;
}

/* Montgomery twiddle table: same twiddle sequence as make_tw() but each      */
/* entry stored as w * R mod p (uint64_t, 8 B) rather than a ShoupPair (16B).*/
/* R = 2^64; since p = 2^64 - 2^b + 1, R mod p = 2^b - 1.                  */
/* montgomery_mul(data, w*R mod p) = data * w mod p (data stays in normal    */
/* domain; no domain conversion needed when twiddles carry the R factor).    */
static uint64_t* make_tw_mont(int dev, uint64_t p, uint64_t g_root,
                               int log_n, int prime_b)
{
    uint64_t R_mod_p = (UINT64_C(1) << prime_b) - 1;  /* 2^64 mod p */
    int N    = 1 << log_n;
    /* Stage-bucketed: total entries = sum 2^s = N-1 (see make_tw). The old
     * N/2 allocation overran the host buffer and truncated the device
     * Montgomery twiddle table for every log_n >= 2. */
    int n_tw = N - 1;
    uint64_t* h = (uint64_t*)malloc((size_t)n_tw * sizeof(uint64_t));
    if (!h) { fprintf(stderr, "make_tw_mont: malloc failed\n"); exit(1); }

    uint64_t omega = modpow(g_root, (p - 1) / (uint64_t)N, p);
    int offset = 0, s;
    for (s = 0; s < log_n; s++) {
        int cnt = 1 << s;
        uint64_t root = modpow(omega, (uint64_t)(N >> (s + 1)), p);
        uint64_t cur  = 1;
        int j;
        for (j = 0; j < cnt; j++) {
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

/* ─── Flat Stockham twiddle helpers ──────────────────────────────────────── */
/* fill_flat_table: h[k] = make_shoup(root_N^k, p) for k=0..M/2-1.           */
/* root_N is the M-th root of unity derived from g_root.                      */
static void fill_flat_table(ShoupPair* h, uint64_t p, uint64_t g_root, int log_m)
{
    int M = 1 << log_m;
    uint64_t omega = modpow(g_root, (p - 1) / (uint64_t)M, p);
    uint64_t cur = 1;
    int k;
    for (k = 0; k < M / 2; k++) {
        h[k] = make_shoup(cur, p);
        cur = (uint64_t)((__uint128_t)cur * omega % p);
    }
}

/* make_tw_stockham: allocate and upload forward flat Stockham twiddle table.  */
/* log_n <= 10: N/2 ShoupPairs, flat table for ω_N.                           */
/* log_n > 10:  N1/2 + N2/2 pairs; [0..N1/2-1] → ω_N1, rest → ω_N2.        */
static ShoupPair* make_tw_stockham(int dev, uint64_t p, uint64_t g_root, int log_n)
{
    int alloc_size, log_n1, log_n2;
    if (log_n <= 10) {
        alloc_size = (1 << log_n) / 2;
    } else {
        log_n1 = log_n / 2; log_n2 = log_n - log_n1;
        alloc_size = (1 << (log_n1 - 1)) + (1 << (log_n2 - 1));
    }
    ShoupPair* h = (ShoupPair*)malloc((size_t)alloc_size * sizeof(ShoupPair));
    if (!h) { fprintf(stderr, "make_tw_stockham: malloc failed\n"); exit(1); }

    if (log_n <= 10) {
        fill_flat_table(h, p, g_root, log_n);
    } else {
        log_n1 = log_n / 2; log_n2 = log_n - log_n1;
        fill_flat_table(h,                    p, g_root, log_n1);
        fill_flat_table(h + (1 << (log_n1-1)), p, g_root, log_n2);
    }

    ShoupPair* d;
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMalloc(&d, (size_t)alloc_size * sizeof(ShoupPair)));
    HIP_CHECK(hipMemcpy(d, h, (size_t)alloc_size * sizeof(ShoupPair),
                    hipMemcpyHostToDevice));
    free(h);
    return d;
}

/* make_tw_stockham_inv: inverse flat Stockham twiddle table.                  */
/* Uses g_root^{-1} mod p so that root_inv_M = ω_M^{-1}.                     */
static ShoupPair* make_tw_stockham_inv(int dev, uint64_t p, uint64_t g_root, int log_n)
{
    uint64_t g_root_inv = modpow(g_root, p - 2, p);
    return make_tw_stockham(dev, p, g_root_inv, log_n);
}

/* ─── Initialization ─────────────────────────────────────────────────────── */
void init(int log_n)
{
    int N = 1 << log_n;
    int i, j;

    if (!verify_primes()) {
        fprintf(stderr, "FATAL: prime verification failed\n");
        exit(1);
    }
    gconsts = garner_precompute();

    HIP_CHECK(hipGetDeviceCount(&g_n_devs));
    if (g_n_devs < 1) g_n_devs = 1;
    if (g_n_devs > 4) g_n_devs = 4;
    printf("init: %d physical APU(s) detected; 4 primes mapped across them\n",
           g_n_devs);

    /* Enable peer access between all distinct physical device pairs. */
    for (i = 0; i < g_n_devs; i++) {
        HIP_CHECK(hipSetDevice(i));
        for (j = 0; j < g_n_devs; j++) {
            if (i == j) continue;
            int can = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&can, i, j));
            if (can) {
                HIP_CHECK(hipDeviceEnablePeerAccess(j, 0));
            } else {
                fprintf(stderr, "WARNING: APU %d cannot peer-access APU %d\n", i, j);
            }
        }
    }

    for (i = 0; i < 4; i++) {
        ctx[i].dev_id = i % g_n_devs;
        HIP_CHECK(hipSetDevice(ctx[i].dev_id));
        HIP_CHECK(hipStreamCreate(&ctx[i].stream));
        HIP_CHECK(hipStreamCreate(&ctx[i].stream_g));
        HIP_CHECK(hipEventCreate(&ctx[i].ev_g));
        HIP_CHECK(hipMalloc(&ctx[i].d_coeffs,   (size_t)N * sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&ctx[i].d_g_coeffs, (size_t)N * sizeof(uint64_t)));
        ctx[i].d_tw           = make_tw(ctx[i].dev_id,           PRIMES[i], PRIM_ROOT[i], log_n);
        ctx[i].d_tw_inv       = make_tw_inv(ctx[i].dev_id,       PRIMES[i], PRIM_ROOT[i], log_n);
        ctx[i].d_tw_cross     = make_tw_cross(ctx[i].dev_id,     PRIMES[i], PRIM_ROOT[i], log_n);
        ctx[i].d_psi          = make_psi_table(ctx[i].dev_id,    PRIMES[i], PRIM_ROOT[i], log_n, 0);
        ctx[i].d_psi_inv      = make_psi_table(ctx[i].dev_id,    PRIMES[i], PRIM_ROOT[i], log_n, 1);
        ctx[i].n_inv          = modpow((uint64_t)N, PRIMES[i] - 2, PRIMES[i]);
        ctx[i].d_tw_cross_inv_scaled = make_tw_cross_inv_scaled(
            ctx[i].dev_id, PRIMES[i], PRIM_ROOT[i], log_n, ctx[i].n_inv);
        ctx[i].d_tw_mont     = make_tw_mont(ctx[i].dev_id, PRIMES[i], PRIM_ROOT[i],
                                            log_n, PRIME_B[i]);
        ctx[i].d_tw_stok     = make_tw_stockham(ctx[i].dev_id,     PRIMES[i], PRIM_ROOT[i], log_n);
        ctx[i].d_tw_stok_inv = make_tw_stockham_inv(ctx[i].dev_id, PRIMES[i], PRIM_ROOT[i], log_n);
        /* d_out: GPU Garner writes the reconstructed U256 array here. Only
         * device 0 needs it (the kernel aggregates four residues into one
         * U256 stream on the lead APU). For other devices keep NULL. */
        ctx[i].d_out = NULL;
        if (i == 0) {
            HIP_CHECK(hipMalloc(&ctx[i].d_out, (size_t)N * sizeof(U256)));
        }
        init_ntt_device(ctx[i].dev_id);

        /* One dedicated stream per directed link from device i.
         * Skip same physical device: no peer copy needed. */
        for (j = 0; j < 4; j++) {
            if (ctx[i].dev_id == ctx[j].dev_id) continue;
            HIP_CHECK(hipSetDevice(ctx[i].dev_id));
            HIP_CHECK(hipStreamCreate(&link_streams[i][j]));
        }
    }
}

/* ─── Pipeline: compute h = f * g mod (p0*p1*p2*p3) ─────────────────────── */
/*                                                                            */
/* h_f, h_g: host arrays of N uint64_t coefficients in [0, min(pi)-1).      */
/*   (Caller is responsible for range-checking inputs against each prime.)   */
/* h_output: host array of N U256 results. Must be pre-allocated by caller.  */
void pipeline(const uint64_t* h_f, const uint64_t* h_g,
              U256* h_output, int log_n)
{
    int N = 1 << log_n;
    size_t bytes = (size_t)N * sizeof(uint64_t);
    int d;

    /* 1. Upload f and g to APU 0 */
    HIP_CHECK(hipSetDevice(ctx[0].dev_id));
    HIP_CHECK(hipMemcpyAsync(ctx[0].d_coeffs,   h_f, bytes,
                          hipMemcpyHostToDevice, ctx[0].stream));
    HIP_CHECK(hipMemcpyAsync(ctx[0].d_g_coeffs, h_g, bytes,
                          hipMemcpyHostToDevice, ctx[0].stream));
    HIP_CHECK(hipStreamSynchronize(ctx[0].stream));

    /* 2. Broadcast f then g: APU 0 -> APUs 1,2,3 (66.63 GB/s SDMA per link) */
    uint64_t* f_dsts[4] = {
        ctx[0].d_coeffs, ctx[1].d_coeffs,
        ctx[2].d_coeffs, ctx[3].d_coeffs
    };
    broadcast_input(ctx[0].d_coeffs, f_dsts, link_streams, bytes);

    uint64_t* g_dsts[4] = {
        ctx[0].d_g_coeffs, ctx[1].d_g_coeffs,
        ctx[2].d_g_coeffs, ctx[3].d_g_coeffs
    };
    broadcast_input(ctx[0].d_g_coeffs, g_dsts, link_streams, bytes);

    /* 3. NTT(f) on stream; NTT(g) on stream_g.
     * Stockham 4-step only works for square decomposition (even log_n);
     * fall back to CT-DIT for odd log_n. */
    const bool use_stockham = !(log_n & 1);
    if (use_stockham) {
        HIP_CHECK(hipSetDevice(ctx[0].dev_id));
        launch_stockham_ntt<0>(ctx[0].d_coeffs,   ctx[0].d_tw_stok, ctx[0].d_tw_cross, log_n, ctx[0].stream);
        launch_stockham_ntt<0>(ctx[0].d_g_coeffs, ctx[0].d_tw_stok, ctx[0].d_tw_cross, log_n, ctx[0].stream_g);
        HIP_CHECK(hipSetDevice(ctx[1].dev_id));
        launch_stockham_ntt<1>(ctx[1].d_coeffs,   ctx[1].d_tw_stok, ctx[1].d_tw_cross, log_n, ctx[1].stream);
        launch_stockham_ntt<1>(ctx[1].d_g_coeffs, ctx[1].d_tw_stok, ctx[1].d_tw_cross, log_n, ctx[1].stream_g);
        HIP_CHECK(hipSetDevice(ctx[2].dev_id));
        launch_stockham_ntt<2>(ctx[2].d_coeffs,   ctx[2].d_tw_stok, ctx[2].d_tw_cross, log_n, ctx[2].stream);
        launch_stockham_ntt<2>(ctx[2].d_g_coeffs, ctx[2].d_tw_stok, ctx[2].d_tw_cross, log_n, ctx[2].stream_g);
        HIP_CHECK(hipSetDevice(ctx[3].dev_id));
        launch_stockham_ntt<3>(ctx[3].d_coeffs,   ctx[3].d_tw_stok, ctx[3].d_tw_cross, log_n, ctx[3].stream);
        launch_stockham_ntt<3>(ctx[3].d_g_coeffs, ctx[3].d_tw_stok, ctx[3].d_tw_cross, log_n, ctx[3].stream_g);
    } else {
        HIP_CHECK(hipSetDevice(ctx[0].dev_id));
        launch_ntt<0>(ctx[0].d_coeffs,   ctx[0].d_tw, ctx[0].d_tw_cross, log_n, ctx[0].stream);
        launch_ntt<0>(ctx[0].d_g_coeffs, ctx[0].d_tw, ctx[0].d_tw_cross, log_n, ctx[0].stream_g);
        HIP_CHECK(hipSetDevice(ctx[1].dev_id));
        launch_ntt<1>(ctx[1].d_coeffs,   ctx[1].d_tw, ctx[1].d_tw_cross, log_n, ctx[1].stream);
        launch_ntt<1>(ctx[1].d_g_coeffs, ctx[1].d_tw, ctx[1].d_tw_cross, log_n, ctx[1].stream_g);
        HIP_CHECK(hipSetDevice(ctx[2].dev_id));
        launch_ntt<2>(ctx[2].d_coeffs,   ctx[2].d_tw, ctx[2].d_tw_cross, log_n, ctx[2].stream);
        launch_ntt<2>(ctx[2].d_g_coeffs, ctx[2].d_tw, ctx[2].d_tw_cross, log_n, ctx[2].stream_g);
        HIP_CHECK(hipSetDevice(ctx[3].dev_id));
        launch_ntt<3>(ctx[3].d_coeffs,   ctx[3].d_tw, ctx[3].d_tw_cross, log_n, ctx[3].stream);
        launch_ntt<3>(ctx[3].d_g_coeffs, ctx[3].d_tw, ctx[3].d_tw_cross, log_n, ctx[3].stream_g);
    }

    /* 4. Make stream wait for NTT(g) completion — GPU-side, no CPU stall. */
    for (d = 0; d < 4; d++) {
        HIP_CHECK(hipSetDevice(ctx[d].dev_id));
        HIP_CHECK(hipEventRecord(ctx[d].ev_g, ctx[d].stream_g));
        HIP_CHECK(hipStreamWaitEvent(ctx[d].stream, ctx[d].ev_g, 0));
    }

    /* 5. Hadamard: d_coeffs[i] *= d_g_coeffs[i] mod p[i] */
    HIP_CHECK(hipSetDevice(ctx[0].dev_id)); launch_hadamard<0>(ctx[0].d_coeffs, ctx[0].d_g_coeffs, N, ctx[0].stream);
    HIP_CHECK(hipSetDevice(ctx[1].dev_id)); launch_hadamard<1>(ctx[1].d_coeffs, ctx[1].d_g_coeffs, N, ctx[1].stream);
    HIP_CHECK(hipSetDevice(ctx[2].dev_id)); launch_hadamard<2>(ctx[2].d_coeffs, ctx[2].d_g_coeffs, N, ctx[2].stream);
    HIP_CHECK(hipSetDevice(ctx[3].dev_id)); launch_hadamard<3>(ctx[3].d_coeffs, ctx[3].d_g_coeffs, N, ctx[3].stream);

    /* 6. INTT: same parity dispatch as forward NTT. */
    if (use_stockham) {
        HIP_CHECK(hipSetDevice(ctx[0].dev_id)); launch_stockham_intt<0>(ctx[0].d_coeffs, ctx[0].n_inv, ctx[0].d_tw_stok_inv, ctx[0].d_tw_cross_inv_scaled, log_n, ctx[0].stream);
        HIP_CHECK(hipSetDevice(ctx[1].dev_id)); launch_stockham_intt<1>(ctx[1].d_coeffs, ctx[1].n_inv, ctx[1].d_tw_stok_inv, ctx[1].d_tw_cross_inv_scaled, log_n, ctx[1].stream);
        HIP_CHECK(hipSetDevice(ctx[2].dev_id)); launch_stockham_intt<2>(ctx[2].d_coeffs, ctx[2].n_inv, ctx[2].d_tw_stok_inv, ctx[2].d_tw_cross_inv_scaled, log_n, ctx[2].stream);
        HIP_CHECK(hipSetDevice(ctx[3].dev_id)); launch_stockham_intt<3>(ctx[3].d_coeffs, ctx[3].n_inv, ctx[3].d_tw_stok_inv, ctx[3].d_tw_cross_inv_scaled, log_n, ctx[3].stream);
    } else {
        HIP_CHECK(hipSetDevice(ctx[0].dev_id)); launch_intt_fused<0>(ctx[0].d_coeffs, ctx[0].n_inv, ctx[0].d_tw_inv, ctx[0].d_tw_cross_inv_scaled, log_n, ctx[0].stream);
        HIP_CHECK(hipSetDevice(ctx[1].dev_id)); launch_intt_fused<1>(ctx[1].d_coeffs, ctx[1].n_inv, ctx[1].d_tw_inv, ctx[1].d_tw_cross_inv_scaled, log_n, ctx[1].stream);
        HIP_CHECK(hipSetDevice(ctx[2].dev_id)); launch_intt_fused<2>(ctx[2].d_coeffs, ctx[2].n_inv, ctx[2].d_tw_inv, ctx[2].d_tw_cross_inv_scaled, log_n, ctx[2].stream);
        HIP_CHECK(hipSetDevice(ctx[3].dev_id)); launch_intt_fused<3>(ctx[3].d_coeffs, ctx[3].n_inv, ctx[3].d_tw_inv, ctx[3].d_tw_cross_inv_scaled, log_n, ctx[3].stream);
    }

    /* 7. Synchronize all four streams */
    for (d = 0; d < 4; d++) {
        HIP_CHECK(hipSetDevice(ctx[d].dev_id));
        HIP_CHECK(hipStreamSynchronize(ctx[d].stream));
    }

    /* 8. Garner CRT reconstruction — GPU kernel on device 0.
     *   GFX1030_LOCAL: write to device-side ctx[0].d_out, then D2H to
     *     caller's h_output (PCIe-attached discrete GPU, no UMA).
     *   MI300A:        h_output is a unified-HBM buffer the caller
     *     allocated via hipHostMalloc; GPU writes land in host-visible
     *     memory with no separate D2H copy. The caller's contract
     *     (see benchmark()) takes care of the allocation. */
    HIP_CHECK(hipSetDevice(ctx[0].dev_id));
#ifdef GFX1030_LOCAL
    garner_reconstruct_gpu(
        ctx[0].d_coeffs, ctx[1].d_coeffs,
        ctx[2].d_coeffs, ctx[3].d_coeffs,
        ctx[0].d_out, N, &gconsts, ctx[0].stream);
    HIP_CHECK(hipMemcpyAsync(h_output, ctx[0].d_out,
                             (size_t)N * sizeof(U256),
                             hipMemcpyDeviceToHost, ctx[0].stream));
#else
    garner_reconstruct_gpu(
        ctx[0].d_coeffs, ctx[1].d_coeffs,
        ctx[2].d_coeffs, ctx[3].d_coeffs,
        h_output, N, &gconsts, ctx[0].stream);
#endif
    HIP_CHECK(hipStreamSynchronize(ctx[0].stream));
}

/* ─── Negacyclic pipeline: h = f * g mod (X^N + 1) over composite modulus ── */
/*                                                                            */
/* Same as pipeline() but wraps NTT with twist/untwist using the 2N-th root  */
/* psi = g^{(p-1)/(2N)}: pre-multiplies f[k] and g[k] by psi^k before NTT,  */
/* then post-multiplies the INTT output by psi^{-k}.                         */
/* Gives mod-(X^N+1) convolution rather than mod-(X^N-1) cyclic convolution. */
void pipeline_negacyclic(const uint64_t* h_f, const uint64_t* h_g,
                          U256* h_output, int log_n)
{
    int N = 1 << log_n;
    size_t bytes = (size_t)N * sizeof(uint64_t);
    int d;

    /* 1. Upload f and g to APU 0 */
    HIP_CHECK(hipSetDevice(ctx[0].dev_id));
    HIP_CHECK(hipMemcpyAsync(ctx[0].d_coeffs,   h_f, bytes,
                          hipMemcpyHostToDevice, ctx[0].stream));
    HIP_CHECK(hipMemcpyAsync(ctx[0].d_g_coeffs, h_g, bytes,
                          hipMemcpyHostToDevice, ctx[0].stream));
    HIP_CHECK(hipStreamSynchronize(ctx[0].stream));

    /* 2. Broadcast f then g to APUs 1,2,3 */
    uint64_t* f_dsts[4] = {
        ctx[0].d_coeffs, ctx[1].d_coeffs,
        ctx[2].d_coeffs, ctx[3].d_coeffs
    };
    broadcast_input(ctx[0].d_coeffs, f_dsts, link_streams, bytes);
    uint64_t* g_dsts[4] = {
        ctx[0].d_g_coeffs, ctx[1].d_g_coeffs,
        ctx[2].d_g_coeffs, ctx[3].d_g_coeffs
    };
    broadcast_input(ctx[0].d_g_coeffs, g_dsts, link_streams, bytes);

    /* 3. Twist + NTT (parity dispatch — see pipeline() above). */
    const bool use_stockham = !(log_n & 1);
    HIP_CHECK(hipSetDevice(ctx[0].dev_id));
    launch_twist<0>(ctx[0].d_coeffs,   ctx[0].d_psi, N, ctx[0].stream);
    launch_twist<0>(ctx[0].d_g_coeffs, ctx[0].d_psi, N, ctx[0].stream_g);
    if (use_stockham) {
        launch_stockham_ntt<0>(ctx[0].d_coeffs,   ctx[0].d_tw_stok, ctx[0].d_tw_cross, log_n, ctx[0].stream);
        launch_stockham_ntt<0>(ctx[0].d_g_coeffs, ctx[0].d_tw_stok, ctx[0].d_tw_cross, log_n, ctx[0].stream_g);
    } else {
        launch_ntt<0>(ctx[0].d_coeffs,   ctx[0].d_tw, ctx[0].d_tw_cross, log_n, ctx[0].stream);
        launch_ntt<0>(ctx[0].d_g_coeffs, ctx[0].d_tw, ctx[0].d_tw_cross, log_n, ctx[0].stream_g);
    }
    HIP_CHECK(hipSetDevice(ctx[1].dev_id));
    launch_twist<1>(ctx[1].d_coeffs,   ctx[1].d_psi, N, ctx[1].stream);
    launch_twist<1>(ctx[1].d_g_coeffs, ctx[1].d_psi, N, ctx[1].stream_g);
    if (use_stockham) {
        launch_stockham_ntt<1>(ctx[1].d_coeffs,   ctx[1].d_tw_stok, ctx[1].d_tw_cross, log_n, ctx[1].stream);
        launch_stockham_ntt<1>(ctx[1].d_g_coeffs, ctx[1].d_tw_stok, ctx[1].d_tw_cross, log_n, ctx[1].stream_g);
    } else {
        launch_ntt<1>(ctx[1].d_coeffs,   ctx[1].d_tw, ctx[1].d_tw_cross, log_n, ctx[1].stream);
        launch_ntt<1>(ctx[1].d_g_coeffs, ctx[1].d_tw, ctx[1].d_tw_cross, log_n, ctx[1].stream_g);
    }
    HIP_CHECK(hipSetDevice(ctx[2].dev_id));
    launch_twist<2>(ctx[2].d_coeffs,   ctx[2].d_psi, N, ctx[2].stream);
    launch_twist<2>(ctx[2].d_g_coeffs, ctx[2].d_psi, N, ctx[2].stream_g);
    if (use_stockham) {
        launch_stockham_ntt<2>(ctx[2].d_coeffs,   ctx[2].d_tw_stok, ctx[2].d_tw_cross, log_n, ctx[2].stream);
        launch_stockham_ntt<2>(ctx[2].d_g_coeffs, ctx[2].d_tw_stok, ctx[2].d_tw_cross, log_n, ctx[2].stream_g);
    } else {
        launch_ntt<2>(ctx[2].d_coeffs,   ctx[2].d_tw, ctx[2].d_tw_cross, log_n, ctx[2].stream);
        launch_ntt<2>(ctx[2].d_g_coeffs, ctx[2].d_tw, ctx[2].d_tw_cross, log_n, ctx[2].stream_g);
    }
    HIP_CHECK(hipSetDevice(ctx[3].dev_id));
    launch_twist<3>(ctx[3].d_coeffs,   ctx[3].d_psi, N, ctx[3].stream);
    launch_twist<3>(ctx[3].d_g_coeffs, ctx[3].d_psi, N, ctx[3].stream_g);
    if (use_stockham) {
        launch_stockham_ntt<3>(ctx[3].d_coeffs,   ctx[3].d_tw_stok, ctx[3].d_tw_cross, log_n, ctx[3].stream);
        launch_stockham_ntt<3>(ctx[3].d_g_coeffs, ctx[3].d_tw_stok, ctx[3].d_tw_cross, log_n, ctx[3].stream_g);
    } else {
        launch_ntt<3>(ctx[3].d_coeffs,   ctx[3].d_tw, ctx[3].d_tw_cross, log_n, ctx[3].stream);
        launch_ntt<3>(ctx[3].d_g_coeffs, ctx[3].d_tw, ctx[3].d_tw_cross, log_n, ctx[3].stream_g);
    }

    /* 4. Make stream wait for NTT(g) completion — GPU-side, no CPU stall. */
    for (d = 0; d < 4; d++) {
        HIP_CHECK(hipSetDevice(ctx[d].dev_id));
        HIP_CHECK(hipEventRecord(ctx[d].ev_g, ctx[d].stream_g));
        HIP_CHECK(hipStreamWaitEvent(ctx[d].stream, ctx[d].ev_g, 0));
    }

    /* 5. Hadamard: d_coeffs[i] *= d_g_coeffs[i] mod p[i] */
    HIP_CHECK(hipSetDevice(ctx[0].dev_id)); launch_hadamard<0>(ctx[0].d_coeffs, ctx[0].d_g_coeffs, N, ctx[0].stream);
    HIP_CHECK(hipSetDevice(ctx[1].dev_id)); launch_hadamard<1>(ctx[1].d_coeffs, ctx[1].d_g_coeffs, N, ctx[1].stream);
    HIP_CHECK(hipSetDevice(ctx[2].dev_id)); launch_hadamard<2>(ctx[2].d_coeffs, ctx[2].d_g_coeffs, N, ctx[2].stream);
    HIP_CHECK(hipSetDevice(ctx[3].dev_id)); launch_hadamard<3>(ctx[3].d_coeffs, ctx[3].d_g_coeffs, N, ctx[3].stream);

    /* 6. INTT: parity dispatch as forward. */
    if (use_stockham) {
        HIP_CHECK(hipSetDevice(ctx[0].dev_id)); launch_stockham_intt<0>(ctx[0].d_coeffs, ctx[0].n_inv, ctx[0].d_tw_stok_inv, ctx[0].d_tw_cross_inv_scaled, log_n, ctx[0].stream);
        HIP_CHECK(hipSetDevice(ctx[1].dev_id)); launch_stockham_intt<1>(ctx[1].d_coeffs, ctx[1].n_inv, ctx[1].d_tw_stok_inv, ctx[1].d_tw_cross_inv_scaled, log_n, ctx[1].stream);
        HIP_CHECK(hipSetDevice(ctx[2].dev_id)); launch_stockham_intt<2>(ctx[2].d_coeffs, ctx[2].n_inv, ctx[2].d_tw_stok_inv, ctx[2].d_tw_cross_inv_scaled, log_n, ctx[2].stream);
        HIP_CHECK(hipSetDevice(ctx[3].dev_id)); launch_stockham_intt<3>(ctx[3].d_coeffs, ctx[3].n_inv, ctx[3].d_tw_stok_inv, ctx[3].d_tw_cross_inv_scaled, log_n, ctx[3].stream);
    } else {
        HIP_CHECK(hipSetDevice(ctx[0].dev_id)); launch_intt_fused<0>(ctx[0].d_coeffs, ctx[0].n_inv, ctx[0].d_tw_inv, ctx[0].d_tw_cross_inv_scaled, log_n, ctx[0].stream);
        HIP_CHECK(hipSetDevice(ctx[1].dev_id)); launch_intt_fused<1>(ctx[1].d_coeffs, ctx[1].n_inv, ctx[1].d_tw_inv, ctx[1].d_tw_cross_inv_scaled, log_n, ctx[1].stream);
        HIP_CHECK(hipSetDevice(ctx[2].dev_id)); launch_intt_fused<2>(ctx[2].d_coeffs, ctx[2].n_inv, ctx[2].d_tw_inv, ctx[2].d_tw_cross_inv_scaled, log_n, ctx[2].stream);
        HIP_CHECK(hipSetDevice(ctx[3].dev_id)); launch_intt_fused<3>(ctx[3].d_coeffs, ctx[3].n_inv, ctx[3].d_tw_inv, ctx[3].d_tw_cross_inv_scaled, log_n, ctx[3].stream);
    }

    /* 7. Untwist: post-multiply by psi^{-k} to complete negacyclic transform */
    HIP_CHECK(hipSetDevice(ctx[0].dev_id)); launch_untwist<0>(ctx[0].d_coeffs, ctx[0].d_psi_inv, N, ctx[0].stream);
    HIP_CHECK(hipSetDevice(ctx[1].dev_id)); launch_untwist<1>(ctx[1].d_coeffs, ctx[1].d_psi_inv, N, ctx[1].stream);
    HIP_CHECK(hipSetDevice(ctx[2].dev_id)); launch_untwist<2>(ctx[2].d_coeffs, ctx[2].d_psi_inv, N, ctx[2].stream);
    HIP_CHECK(hipSetDevice(ctx[3].dev_id)); launch_untwist<3>(ctx[3].d_coeffs, ctx[3].d_psi_inv, N, ctx[3].stream);

    /* 8. Synchronize all four streams */
    for (d = 0; d < 4; d++) {
        HIP_CHECK(hipSetDevice(ctx[d].dev_id));
        HIP_CHECK(hipStreamSynchronize(ctx[d].stream));
    }

    /* 9. Garner CRT reconstruction — same MI300A vs gfx1030 gate as
     * pipeline() above. */
    HIP_CHECK(hipSetDevice(ctx[0].dev_id));
#ifdef GFX1030_LOCAL
    garner_reconstruct_gpu(
        ctx[0].d_coeffs, ctx[1].d_coeffs,
        ctx[2].d_coeffs, ctx[3].d_coeffs,
        ctx[0].d_out, N, &gconsts, ctx[0].stream);
    HIP_CHECK(hipMemcpyAsync(h_output, ctx[0].d_out,
                             (size_t)N * sizeof(U256),
                             hipMemcpyDeviceToHost, ctx[0].stream));
#else
    garner_reconstruct_gpu(
        ctx[0].d_coeffs, ctx[1].d_coeffs,
        ctx[2].d_coeffs, ctx[3].d_coeffs,
        h_output, N, &gconsts, ctx[0].stream);
#endif
    HIP_CHECK(hipStreamSynchronize(ctx[0].stream));
}

/* ─── Benchmark ──────────────────────────────────────────────────────────── */
void benchmark(int log_n, int n_trials)
{
    int N = 1 << log_n;
    uint64_t* h_f;
    uint64_t* h_g;
    U256*     h_out;
#ifdef GFX1030_LOCAL
    /* Discrete-VRAM dev path: standard host buffers + DMA H2D/D2H. */
    h_f   = (uint64_t*)malloc((size_t)N * sizeof(uint64_t));
    h_g   = (uint64_t*)malloc((size_t)N * sizeof(uint64_t));
    h_out = (U256*)    malloc((size_t)N * sizeof(U256));
#else
    /* MI300A unified HBM: h_out is hipHostMalloc'd so the GPU Garner
     * writes directly into host-visible memory — eliminates the
     * d_out→h_out D2H copy on the production target. */
    HIP_CHECK(hipHostMalloc(&h_f,   (size_t)N * sizeof(uint64_t), hipHostMallocNonCoherent));
    HIP_CHECK(hipHostMalloc(&h_g,   (size_t)N * sizeof(uint64_t), hipHostMallocNonCoherent));
    HIP_CHECK(hipHostMalloc(&h_out, (size_t)N * sizeof(U256),     hipHostMallocNonCoherent));
#endif
    if (!h_f || !h_g || !h_out) {
        fprintf(stderr, "benchmark: malloc failed\n");
        exit(1);
    }

    int i, d;
    for (i = 0; i < N; i++) {
        h_f[i] = (uint64_t)(i % 1000003);
        h_g[i] = (uint64_t)((i * 7 + 1) % 1000003);
    }

    /* Warm-up: 5 iterations to settle caches and HIP JIT */
    for (i = 0; i < 5; i++) pipeline(h_f, h_g, h_out, log_n);
    for (d = 0; d < 4; d++) {
        HIP_CHECK(hipSetDevice(ctx[d].dev_id));
        HIP_CHECK(hipDeviceSynchronize());
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (i = 0; i < n_trials; i++) pipeline(h_f, h_g, h_out, log_n);
    for (d = 0; d < 4; d++) {
        HIP_CHECK(hipSetDevice(ctx[d].dev_id));
        HIP_CHECK(hipDeviceSynchronize());
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = ((t1.tv_sec  - t0.tv_sec)  * 1e3
               + (t1.tv_nsec - t0.tv_nsec) * 1e-6) / n_trials;
    /* 4 primes × 2 uploads × 3 broadcasts × N × 8 bytes → data moved */
    printf("N=2^%d  %d trials  %.2f ms/polymul  %.2f Gcoeff/s\n",
           log_n, n_trials, ms, (double)N / (ms * 1e-3) / 1e9);

#ifdef GFX1030_LOCAL
    free(h_f);   free(h_g);   free(h_out);
#else
    (void)hipHostFree(h_f);  (void)hipHostFree(h_g);  (void)hipHostFree(h_out);
#endif
}

/* ─── Montgomery vs Shoup NTT benchmark ──────────────────────────────────── */
/* Compares Shoup (ShoupPair, 16B/twiddle) vs Montgomery (uint64_t, 8B/twiddle)
 * on device 0 only. At large N the twiddle table no longer fits in L2/L1 and
 * HBM reads dominate; Montgomery's 2x density advantage reduces twiddle reads
 * by 50% (~64 μs at N=2^24 on 4 TB/s HBM3) at the cost of +3 ALU instr/butterfly.*/
static void benchmark_montgomery(int log_n, int n_trials)
{
    int N = 1 << log_n;
    int i;
    uint64_t* h_f = (uint64_t*)malloc((size_t)N * sizeof(uint64_t));
    if (!h_f) { fprintf(stderr, "benchmark_montgomery: malloc failed\n"); exit(1); }
    for (i = 0; i < N; i++) h_f[i] = (uint64_t)(i % 1000003);

    HIP_CHECK(hipSetDevice(ctx[0].dev_id));
    HIP_CHECK(hipMemcpy(ctx[0].d_coeffs, h_f, (size_t)N * sizeof(uint64_t),
                     hipMemcpyHostToDevice));

    /* Warm-up */
    for (i = 0; i < 5; i++) {
        launch_ntt<0>(ctx[0].d_coeffs, ctx[0].d_tw, ctx[0].d_tw_cross, log_n, ctx[0].stream);
        HIP_CHECK(hipStreamSynchronize(ctx[0].stream));
        launch_mont_ntt<0>(ctx[0].d_coeffs, ctx[0].d_tw_mont, ctx[0].d_tw_cross, log_n, ctx[0].stream);
        HIP_CHECK(hipStreamSynchronize(ctx[0].stream));
    }

    struct timespec t0, t1;

    /* Shoup baseline */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (i = 0; i < n_trials; i++) {
        launch_ntt<0>(ctx[0].d_coeffs, ctx[0].d_tw, ctx[0].d_tw_cross, log_n, ctx[0].stream);
        HIP_CHECK(hipStreamSynchronize(ctx[0].stream));
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms_shoup = ((t1.tv_sec  - t0.tv_sec)  * 1e3
                     + (t1.tv_nsec - t0.tv_nsec) * 1e-6) / n_trials;

    /* Montgomery */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (i = 0; i < n_trials; i++) {
        launch_mont_ntt<0>(ctx[0].d_coeffs, ctx[0].d_tw_mont, ctx[0].d_tw_cross, log_n, ctx[0].stream);
        HIP_CHECK(hipStreamSynchronize(ctx[0].stream));
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms_mont = ((t1.tv_sec  - t0.tv_sec)  * 1e3
                    + (t1.tv_nsec - t0.tv_nsec) * 1e-6) / n_trials;

    size_t tw_shoup_mb = (size_t)(N / 2) * sizeof(ShoupPair) / (1024 * 1024);
    size_t tw_mont_mb  = (size_t)(N / 2) * sizeof(uint64_t)  / (1024 * 1024);
    printf("Shoup vs Montgomery NTT  N=2^%d  %d trials  device 0:\n",
           log_n, n_trials);
    printf("  %-12s  %7.3f ms  twiddle %zu MB (16 B/entry)\n",
           "Shoup", ms_shoup, tw_shoup_mb);
    printf("  %-12s  %7.3f ms  twiddle %zu MB  (8 B/entry)\n",
           "Montgomery", ms_mont, tw_mont_mb);
    printf("  Speedup: %.3fx\n", ms_shoup / ms_mont);

    free(h_f);
}

/* ─── Stockham NTT benchmark: Shoup-CT-DIT vs Stockham-LDS ─────────────── */
/* Times launch_ntt<0> (CT-DIT) and launch_stockham_ntt<0> on device 0 and   */
/* prints a side-by-side comparison table.                                    */
static void benchmark_stockham(int log_n, int n_trials)
{
    int N = 1 << log_n;
    size_t sz = (size_t)N * sizeof(uint64_t);
    uint64_t* h_f = (uint64_t*)malloc(sz);
    if (!h_f) { fprintf(stderr, "benchmark_stockham: malloc failed\n"); exit(1); }
    int i;
    for (i = 0; i < N; i++) h_f[i] = (uint64_t)i % PRIMES[0];
    HIP_CHECK(hipSetDevice(ctx[0].dev_id));
    HIP_CHECK(hipMemcpy(ctx[0].d_coeffs, h_f, sz, hipMemcpyHostToDevice));

    struct timespec t0, t1;

    /* Warm-up */
    for (i = 0; i < 5; i++) {
        launch_ntt<0>(ctx[0].d_coeffs, ctx[0].d_tw, ctx[0].d_tw_cross,
                      log_n, ctx[0].stream);
        launch_stockham_ntt<0>(ctx[0].d_coeffs, ctx[0].d_tw_stok,
                               ctx[0].d_tw_cross, log_n, ctx[0].stream);
        HIP_CHECK(hipStreamSynchronize(ctx[0].stream));
    }

    /* CT-DIT (Shoup) */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (i = 0; i < n_trials; i++) {
        launch_ntt<0>(ctx[0].d_coeffs, ctx[0].d_tw, ctx[0].d_tw_cross,
                      log_n, ctx[0].stream);
        HIP_CHECK(hipStreamSynchronize(ctx[0].stream));
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms_ctdit = ((t1.tv_sec  - t0.tv_sec)  * 1e3
                     + (t1.tv_nsec - t0.tv_nsec) * 1e-6) / n_trials;

    /* Stockham LDS */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (i = 0; i < n_trials; i++) {
        launch_stockham_ntt<0>(ctx[0].d_coeffs, ctx[0].d_tw_stok,
                               ctx[0].d_tw_cross, log_n, ctx[0].stream);
        HIP_CHECK(hipStreamSynchronize(ctx[0].stream));
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms_stok = ((t1.tv_sec  - t0.tv_sec)  * 1e3
                    + (t1.tv_nsec - t0.tv_nsec) * 1e-6) / n_trials;

    printf("CT-DIT vs Stockham-LDS NTT  N=2^%d  %d trials  device 0:\n",
           log_n, n_trials);
    printf("  %-14s  %7.3f ms\n", "CT-DIT (Shoup)",  ms_ctdit);
    printf("  %-14s  %7.3f ms\n", "Stockham (LDS)", ms_stok);
    printf("  Speedup: %.3fx\n", ms_ctdit / ms_stok);

    free(h_f);
}

/* ─── main ───────────────────────────────────────────────────────────────── */
/* log_n upper bound: P1 = 2^64 - 2^24 + 1 supports at most 2^24 NTT length.
 * Lower bound 3: log_n<3 has no sensible NTT semantics. launch_ntt routes
 * log_n != 10 through the bit-reverse + global-stage path (works for any
 * N); only log_n == 10 uses the 1024-element LDS kernel. For Stockham
 * 4-step (log_n>10): cross_twiddle_kernel indexes assuming square
 * decomposition (N1=N2), so odd log_n in (11..23) is rejected by
 * launch_stockham_ntt — see ntt_kernel.hip. n_trials must be positive
 * (callers divide by it). Validation runs before init() so a CLI typo
 * can never reach a kernel launch with a small/zero N. */
#define LIB2_LOG_N_MIN 3
#define LIB2_LOG_N_MAX 24

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [log_n] [n_trials] [--stockham] [--montgomery]\n"
        "  log_n     transform exponent, %d..%d (default 20; N = 2^log_n)\n"
        "  n_trials  positive integer (default 100)\n"
        "Examples:  %s 20 100        %s 22 50 --stockham\n",
        prog, LIB2_LOG_N_MIN, LIB2_LOG_N_MAX, prog, prog);
}

int main(int argc, char** argv)
{
    int log_n      = 20;
    int n_trials   = 100;
    int do_mont    = 0;
    int do_stok    = 0;
    int log_n_set  = 0;   /* sentinel: distinguishes default from explicit 20 */
    int n_trials_set = 0;
    int i;

    for (i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--montgomery") == 0) { do_mont = 1; }
        else if (strcmp(argv[i], "--stockham")   == 0) { do_stok = 1; }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]); return 0;
        }
        else if (!log_n_set    && argv[i][0] != '-') { log_n    = atoi(argv[i]); log_n_set    = 1; }
        else if (!n_trials_set && argv[i][0] != '-') { n_trials = atoi(argv[i]); n_trials_set = 1; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 2; }
    }

    if (log_n < LIB2_LOG_N_MIN || log_n > LIB2_LOG_N_MAX) {
        fprintf(stderr, "error: log_n=%d out of range [%d, %d]\n",
                log_n, LIB2_LOG_N_MIN, LIB2_LOG_N_MAX);
        usage(argv[0]); return 2;
    }
    if (n_trials <= 0) {
        fprintf(stderr, "error: n_trials=%d must be positive\n", n_trials);
        usage(argv[0]); return 2;
    }

    printf("CRT-NTT polymul: N=2^%d, %d trials%s%s\n",
           log_n, n_trials,
           do_mont ? " [+montgomery]" : "",
           do_stok ? " [+stockham]"   : "");
    init(log_n);

    if (do_mont) {
        benchmark_montgomery(log_n, n_trials);
        printf("\n");
    }
    if (do_stok) {
        benchmark_stockham(log_n, n_trials);
        printf("\n");
    }
    benchmark(log_n, n_trials);
    return 0;
}
