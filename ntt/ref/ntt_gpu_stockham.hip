/*
 * ntt_gpu_stockham.hip — Stockham auto-sort NTT, GPU kernel for MI300A / gfx942
 *
 * Purpose:   GPU implementation of the Stockham auto-sort NTT. Eliminates
 *            bit-reversal via double-buffer ping-pong between two device
 *            arrays. Produces natural-order output from natural-order input.
 * Algorithm: Stockham DIT, radix-2. Per-stage kernel launch; each thread
 *            handles one butterfly. After log2(n) stages no bit-reversal pass
 *            is needed — output is ready in-order in the result buffer.
 * Reduction: per-stage, using 64-bit multiply (q < 2^32 guarantees no
 *            overflow: w*(v%q) < q^2 < 2^64). No __uint128_t on device.
 * Ref:       Gentleman & Sande, AFIPS 1966. Stockham, AFIPS 1966.
 * Target:    AMD MI300A (gfx942, CDNA3, wavefront=64). Dev: 6900XT (gfx1030).
 *   Build:   hipcc -O2 -Wall --offload-arch=gfx942 -o ntt_gpu_stockham_mi300a \
 *                  ntt_gpu_stockham.hip
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

/* ── HIP error check ─────────────────────────────────────────────────────── */
#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t _e = (call);                                                \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s:%d — %s\n",                         \
                    __FILE__, __LINE__, hipGetErrorString(_e));                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

/* ── NTT parameter struct (matches ntt.h layout) ────────────────────────── */
typedef struct {
    uint64_t n;
    uint64_t q;
    uint64_t omega;
    uint64_t omega_inv;
    uint64_t n_inv;
    uint32_t log2_n;
} ntt_params_t;

/* ── GPU hardware config ─────────────────────────────────────────────────── */
typedef struct {
    char   name[256];
    int    warp_size;
    int    cu_count;
    int    max_shared_b;
    size_t total_mem;
    int    block_size;
} gpu_config_t;
static gpu_config_t g_gpu;

/* ── Kernel launch bounds ────────────────────────────────────────────────── */
/* 256 threads/block, ≥2 blocks/CU: broad occupancy across CDNA3 and RDNA3. */
#define NTT_MAX_BLOCK  256
#define NTT_MIN_BLOCKS   2

/* ═══════════════════════════════════════════════════════════════════════════
 * DEVICE ARITHMETIC
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * dev_shoup_mul: Shoup precomputed-quotient multiply mod q.
 * packed = (w_inv << 32) | w  — both fields 32-bit, q < 2^32.
 * w_inv = floor(w * 2^32 / q), precomputed by alloc_twiddles.
 * b must be < q (enforced by caller via conditional subtract or initial reduction).
 * Replaces hardware 64-bit division (~24 cycles) with 3 × 32-bit multiplies.
 * Ref: Shoup 1995.
 */
__device__ __forceinline__
uint64_t dev_shoup_mul(uint64_t packed, uint64_t b, uint64_t q)
{
    uint32_t w     = (uint32_t)packed;
    uint32_t w_inv = (uint32_t)(packed >> 32);
    uint32_t b32   = (uint32_t)b;
    uint32_t hi    = (uint32_t)(((uint64_t)w_inv * b32) >> 32);
    uint64_t t     = (uint64_t)w * b32 - (uint64_t)hi * (uint32_t)q;
    return (t >= q) ? t - q : t;
}

/* dev_reduce: conditional subtract — reduce x from [0, 2q) to [0, q). */
__device__ __forceinline__
uint64_t dev_reduce(uint64_t x, uint64_t q) { return (x >= q) ? x - q : x; }

/*
 * dev_mod_pow: (base ^ exp) mod q, base < q < 2^32. Used by the
 * on-the-fly (G6) kernel to synthesize twiddles without a precomputed
 * table. log2(exp) mulmods — expensive per butterfly but drops all
 * twiddle HBM traffic.
 */
__device__ __forceinline__
uint64_t dev_mod_pow(uint64_t base, uint64_t exp, uint64_t q)
{
    uint64_t r = 1;
    base %= q;
    while (exp) {
        if (exp & 1) r = (r * base) % q;
        base = (base * base) % q;
        exp >>= 1;
    }
    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNELS
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * stockham_stage_kernel: one Stockham DIT butterfly stage.
 *
 * Each thread tid in [0, n/2) handles one butterfly.
 *
 * Stage geometry (p = 2^s, qs = n / 2^(s+1)):
 *   j   = tid % p              group index   (j in [0, p))
 *   k   = tid / p              butterfly index within group (k in [0, qs))
 *   src indices: j+k*p (first half) and j+k*p + n/2 (second half)
 *   dst indices: j+k*(2p) and j+k*(2p)+p
 *   twiddle:     tw[j * qs]
 *
 * src[0..n/2) always holds the "top" halves and src[n/2..n) the "bottom"
 * halves due to the write pattern of the previous stage. No branch needed.
 */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void stockham_stage_kernel(const uint64_t * __restrict__ src,
                                 uint64_t * __restrict__ dst,
                           const uint64_t * __restrict__ tw,
                           uint64_t n, uint64_t q, uint32_t s)
{
    uint64_t tid  = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = n >> 1;
    if (tid >= half) return;

    uint64_t p  = (uint64_t)1 << s;    /* group stride    */
    uint64_t qs = n >> (s + 1);        /* butterflies/grp  */
    uint64_t p2 = p << 1;

    uint64_t j  = tid & (p - 1);       /* tid % p (p is power of 2) */
    uint64_t k  = tid >> s;            /* tid / p                   */

    uint64_t idx_u = j + k * p;        /* always in [0, n/2)  */
    uint64_t idx_v = idx_u + half;     /* always in [n/2, n)  */

    uint64_t u  = src[idx_u];
    uint64_t v  = src[idx_v];
    uint64_t w  = tw[j * qs];

    /* Both u and v may be in [0, 2q) from the previous lazy stage output.
     * Reduce both to [0, q) so butterfly output stays in [0, 2q). */
    u  = dev_reduce(u, q);
    uint64_t wv = dev_shoup_mul(w, dev_reduce(v, q), q);

    dst[j + k * p2]      = u + wv;       /* lazy — will be reduced next stage */
    dst[j + k * p2 + p]  = u + q - wv;  /* keep ≥ 0                          */
}

/*
 * stockham_lds_fused_kernel: single-launch Stockham NTT with all stages
 * running entirely in LDS.
 *
 * Preconditions: grid = 1, blockDim.x = n/2, n ≤ 2048 (so half ≤ 1024,
 * the HIP per-block thread limit). Dynamic LDS is sized to hold two
 * padded buffers of n elements each (see LDS_PAD / LDS_STRIDE below).
 *
 * Padding: LDS_PAD(i) = i + (i >> 5) injects one extra uint64_t every
 * 32 elements. This breaks the simple modular alignment that makes
 * stride-matching strides collide on the same banks in later stages
 * where the group stride p approaches half. Without the pad, stride
 * patterns at stages log2(n) − 5 and later degrade to >=8-way bank
 * conflicts on the 32-bank RDNA2/CDNA3 LDS; with the pad, all stages
 * sit at the uint64_t theoretical minimum of 2-way.
 *
 * Outputs values reduced mod q; inverse NTT scale is applied here too
 * (when n_inv != 0) to avoid a second kernel launch.
 */
#define LDS_PAD(i)    ((i) + ((i) >> 5))
#define LDS_STRIDE(n) ((n) + ((n) >> 5) + 1)

__global__ __launch_bounds__(1024, 2)
void stockham_lds_fused_kernel(const uint64_t * __restrict__ in_g,
                                     uint64_t * __restrict__ out_g,
                               const uint64_t * __restrict__ tw,
                               uint64_t n, uint64_t q, uint32_t log2_n,
                               uint64_t n_inv)
{
    extern __shared__ uint64_t lds_raw[];
    uint64_t stride = LDS_STRIDE(n);
    uint64_t *bufs[2];
    bufs[0] = lds_raw;
    bufs[1] = lds_raw + stride;

    uint64_t tid  = threadIdx.x;
    uint64_t half = n >> 1;

    /* Load input into buf 0 via padded addressing. Each thread loads 2. */
    if (tid < half) {
        bufs[0][LDS_PAD(tid)]         = in_g[tid];
        bufs[0][LDS_PAD(tid + half)]  = in_g[tid + half];
    }
    __syncthreads();

    int src_idx = 0;
    for (uint32_t s = 0; s < log2_n; s++) {
        uint64_t p  = (uint64_t)1 << s;
        uint64_t qs = n >> (s + 1);
        uint64_t p2 = p << 1;

        uint64_t j = tid & (p - 1);
        uint64_t k = tid >> s;

        uint64_t idx_u = j + k * p;
        uint64_t idx_v = idx_u + half;

        if (tid < half) {
            uint64_t u  = bufs[src_idx][LDS_PAD(idx_u)];
            uint64_t v  = bufs[src_idx][LDS_PAD(idx_v)];
            uint64_t w  = tw[j * qs];
            /* Both u and v may be in [0, 2q) from the previous lazy stage.
             * Reduce both to [0, q) so butterfly output stays in [0, 2q). */
            u  = dev_reduce(u, q);
            uint64_t wv = dev_shoup_mul(w, dev_reduce(v, q), q);
            bufs[1 - src_idx][LDS_PAD(j + k * p2)]     = u + wv;
            bufs[1 - src_idx][LDS_PAD(j + k * p2 + p)] = u + q - wv;
        }
        __syncthreads();
        src_idx = 1 - src_idx;
    }

    /* Final reduction + optional 1/N scale, then store to global. */
    if (tid < half) {
        uint64_t a0 = dev_reduce(bufs[src_idx][LDS_PAD(tid)],        q);
        uint64_t a1 = dev_reduce(bufs[src_idx][LDS_PAD(tid + half)], q);
        if (n_inv) {
            /* Shoup quotient for n_inv: floor(n_inv * 2^32 / q).
             * Computed once per kernel (one 64-bit div, amortized over n/2 threads). */
            uint64_t ni_inv = ((uint64_t)n_inv << 32) / q;
            uint64_t ni_pk  = (ni_inv << 32) | (uint32_t)n_inv;
            a0 = dev_shoup_mul(ni_pk, a0, q);
            a1 = dev_shoup_mul(ni_pk, a1, q);
        }
        out_g[tid]        = a0;
        out_g[tid + half] = a1;
    }
}

/*
 * stockham_lds_batched_kernel — grid = batch size, each block runs one
 * independent NTT of size n using the G3 fine-fused LDS layout. The
 * 39-of-40 CU idle problem at small n disappears: at batch ≥ CU_count
 * the GPU is saturated. Same LDS footprint and thread shape as G3.
 */
__global__ __launch_bounds__(1024, 2)
void stockham_lds_batched_kernel(const uint64_t * __restrict__ in_g,
                                       uint64_t * __restrict__ out_g,
                                 const uint64_t * __restrict__ tw,
                                 uint64_t n, uint64_t q, uint32_t log2_n,
                                 uint64_t n_inv)
{
    extern __shared__ uint64_t lds_raw[];
    uint64_t stride = LDS_STRIDE(n);
    uint64_t *bufs[2];
    bufs[0] = lds_raw;
    bufs[1] = lds_raw + stride;

    uint64_t tid  = threadIdx.x;
    uint64_t half = n >> 1;
    uint64_t off  = (uint64_t)blockIdx.x * n;   /* per-batch input offset */

    if (tid < half) {
        bufs[0][LDS_PAD(tid)]        = in_g[off + tid];
        bufs[0][LDS_PAD(tid + half)] = in_g[off + tid + half];
    }
    __syncthreads();

    int src_idx = 0;
    for (uint32_t s = 0; s < log2_n; s++) {
        uint64_t p  = (uint64_t)1 << s;
        uint64_t qs = n >> (s + 1);
        uint64_t p2 = p << 1;
        uint64_t j  = tid & (p - 1);
        uint64_t k  = tid >> s;
        uint64_t idx_u = j + k * p;
        uint64_t idx_v = idx_u + half;
        if (tid < half) {
            uint64_t u  = bufs[src_idx][LDS_PAD(idx_u)];
            uint64_t v  = bufs[src_idx][LDS_PAD(idx_v)];
            uint64_t w  = tw[j * qs];
            /* Both u and v may be in [0, 2q) from the previous lazy stage.
             * Reduce both to [0, q) so butterfly output stays in [0, 2q). */
            u  = dev_reduce(u, q);
            uint64_t wv = dev_shoup_mul(w, dev_reduce(v, q), q);
            bufs[1 - src_idx][LDS_PAD(j + k * p2)]     = u + wv;
            bufs[1 - src_idx][LDS_PAD(j + k * p2 + p)] = u + q - wv;
        }
        __syncthreads();
        src_idx = 1 - src_idx;
    }

    if (tid < half) {
        uint64_t a0 = dev_reduce(bufs[src_idx][LDS_PAD(tid)],        q);
        uint64_t a1 = dev_reduce(bufs[src_idx][LDS_PAD(tid + half)], q);
        if (n_inv) {
            /* Shoup quotient for n_inv: floor(n_inv * 2^32 / q).
             * Computed once per kernel (one 64-bit div, amortized over n/2 threads). */
            uint64_t ni_inv = ((uint64_t)n_inv << 32) / q;
            uint64_t ni_pk  = (ni_inv << 32) | (uint32_t)n_inv;
            a0 = dev_shoup_mul(ni_pk, a0, q);
            a1 = dev_shoup_mul(ni_pk, a1, q);
        }
        out_g[off + tid]        = a0;
        out_g[off + tid + half] = a1;
    }
}

/*
 * stockham_lds_fused_otf_kernel — G6 on-the-fly twiddle variant of the
 * G3 fused kernel. Takes the active Nth root (omega for forward, omega_inv
 * for inverse) instead of a precomputed twiddle pointer. Each butterfly
 * synthesizes its twiddle via dev_mod_pow(root, j * qs, q).
 *
 * Cost: log2(j*qs) ≤ log2(n/2) mulmods per butterfly (versus 1 LDS load
 * with the tabled variant). Benefit: zero twiddle HBM/LDS bandwidth.
 * Expected to be slower on 6900XT but documents the memory-free path.
 */
__global__ __launch_bounds__(1024, 2)
void stockham_lds_fused_otf_kernel(const uint64_t * __restrict__ in_g,
                                         uint64_t * __restrict__ out_g,
                                   uint64_t root,
                                   uint64_t n, uint64_t q, uint32_t log2_n,
                                   uint64_t n_inv)
{
    extern __shared__ uint64_t lds_raw[];
    uint64_t stride = LDS_STRIDE(n);
    uint64_t *bufs[2];
    bufs[0] = lds_raw;
    bufs[1] = lds_raw + stride;

    uint64_t tid  = threadIdx.x;
    uint64_t half = n >> 1;

    if (tid < half) {
        bufs[0][LDS_PAD(tid)]        = in_g[tid];
        bufs[0][LDS_PAD(tid + half)] = in_g[tid + half];
    }
    __syncthreads();

    int src_idx = 0;
    for (uint32_t s = 0; s < log2_n; s++) {
        uint64_t p  = (uint64_t)1 << s;
        uint64_t qs = n >> (s + 1);
        uint64_t p2 = p << 1;

        uint64_t j = tid & (p - 1);
        uint64_t k = tid >> s;
        uint64_t idx_u = j + k * p;
        uint64_t idx_v = idx_u + half;

        if (tid < half) {
            uint64_t u  = bufs[src_idx][LDS_PAD(idx_u)];
            uint64_t v  = bufs[src_idx][LDS_PAD(idx_v)];
            /* Synthesize the twiddle w = root^(j*qs), then PACK it into Shoup
             * form (w_inv<<32 | w) exactly as alloc_twiddles packs the tabled
             * twiddles — dev_shoup_mul needs the precomputed quotient
             * w_inv = floor(w*2^32/q) in the high 32 bits. The OTF path
             * previously passed the RAW w (w_inv=0), so dev_shoup_mul did no
             * reduction and returned an unreduced product -> the "OTF vs tabled
             * fused FAIL". q < 2^32 on the GPU-eligible bench set, so w<<32
             * fits in 64 bits. */
            uint64_t w_raw = dev_mod_pow(root, j * qs, q);
            uint64_t w_inv = (w_raw << 32) / q;
            uint64_t w     = (w_inv << 32) | (uint32_t)w_raw;
            /* Both u and v may be in [0, 2q) from the previous lazy stage.
             * Reduce both to [0, q) so butterfly output stays in [0, 2q). */
            u  = dev_reduce(u, q);
            uint64_t wv = dev_shoup_mul(w, dev_reduce(v, q), q);
            bufs[1 - src_idx][LDS_PAD(j + k * p2)]     = u + wv;
            bufs[1 - src_idx][LDS_PAD(j + k * p2 + p)] = u + q - wv;
        }
        __syncthreads();
        src_idx = 1 - src_idx;
    }

    if (tid < half) {
        uint64_t a0 = dev_reduce(bufs[src_idx][LDS_PAD(tid)],        q);
        uint64_t a1 = dev_reduce(bufs[src_idx][LDS_PAD(tid + half)], q);
        if (n_inv) {
            /* Shoup quotient for n_inv: floor(n_inv * 2^32 / q).
             * Computed once per kernel (one 64-bit div, amortized over n/2 threads). */
            uint64_t ni_inv = ((uint64_t)n_inv << 32) / q;
            uint64_t ni_pk  = (ni_inv << 32) | (uint32_t)n_inv;
            a0 = dev_shoup_mul(ni_pk, a0, q);
            a1 = dev_shoup_mul(ni_pk, a1, q);
        }
        out_g[tid]        = a0;
        out_g[tid + half] = a1;
    }
}

/*
 * stockham_lds_fused_reg2_kernel — G5 register-blocked variant of the G3
 * fused kernel. Each thread owns TWO butterflies per stage (indices tid
 * and tid + n/4). All six reads (u1, v1, w1, u2, v2, w2) are issued
 * together, the two butterfly computes are written independently, and
 * both result pairs are stored before the next __syncthreads(). This
 * exposes the two butterfly dep-chains to the compiler in parallel so it
 * can issue mulmods and addmods back-to-back, hiding mulmod latency.
 *
 * Shape constraint: threads = n/4 ≤ 1024 → n ≤ 4096. Combined with the
 * padded-LDS cap of n ≤ 2048, this kernel is dispatched for 4 ≤ n ≤ 2048.
 */
__global__ __launch_bounds__(1024, 2)
void stockham_lds_fused_reg2_kernel(const uint64_t * __restrict__ in_g,
                                          uint64_t * __restrict__ out_g,
                                    const uint64_t * __restrict__ tw,
                                    uint64_t n, uint64_t q, uint32_t log2_n,
                                    uint64_t n_inv)
{
    extern __shared__ uint64_t lds_raw[];
    uint64_t stride = LDS_STRIDE(n);
    uint64_t *bufs[2];
    bufs[0] = lds_raw;
    bufs[1] = lds_raw + stride;

    uint64_t tid   = threadIdx.x;
    uint64_t half  = n >> 1;
    uint64_t quart = n >> 2;

    /* Load 4 elements per thread. */
    if (tid < quart) {
        bufs[0][LDS_PAD(tid)]                 = in_g[tid];
        bufs[0][LDS_PAD(tid + quart)]         = in_g[tid + quart];
        bufs[0][LDS_PAD(tid + half)]          = in_g[tid + half];
        bufs[0][LDS_PAD(tid + half + quart)]  = in_g[tid + half + quart];
    }
    __syncthreads();

    int src_idx = 0;
    for (uint32_t s = 0; s < log2_n; s++) {
        uint64_t p  = (uint64_t)1 << s;
        uint64_t qs = n >> (s + 1);
        uint64_t p2 = p << 1;

        if (tid < quart) {
            /* Butterfly 1 at bid = tid. */
            uint64_t b1 = tid;
            uint64_t j1 = b1 & (p - 1);
            uint64_t k1 = b1 >> s;
            /* Butterfly 2 at bid = tid + quart. */
            uint64_t b2 = tid + quart;
            uint64_t j2 = b2 & (p - 1);
            uint64_t k2 = b2 >> s;

            uint64_t u1 = bufs[src_idx][LDS_PAD(j1 + k1 * p)];
            uint64_t v1 = bufs[src_idx][LDS_PAD(j1 + k1 * p + half)];
            uint64_t w1 = tw[j1 * qs];
            uint64_t u2 = bufs[src_idx][LDS_PAD(j2 + k2 * p)];
            uint64_t v2 = bufs[src_idx][LDS_PAD(j2 + k2 * p + half)];
            uint64_t w2 = tw[j2 * qs];

            u1 = dev_reduce(u1, q);
            u2 = dev_reduce(u2, q);
            uint64_t wv1 = dev_shoup_mul(w1, dev_reduce(v1, q), q);
            uint64_t wv2 = dev_shoup_mul(w2, dev_reduce(v2, q), q);

            bufs[1 - src_idx][LDS_PAD(j1 + k1 * p2)]     = u1 + wv1;
            bufs[1 - src_idx][LDS_PAD(j1 + k1 * p2 + p)] = u1 + q - wv1;
            bufs[1 - src_idx][LDS_PAD(j2 + k2 * p2)]     = u2 + wv2;
            bufs[1 - src_idx][LDS_PAD(j2 + k2 * p2 + p)] = u2 + q - wv2;
        }
        __syncthreads();
        src_idx = 1 - src_idx;
    }

    if (tid < quart) {
        uint64_t a0 = dev_reduce(bufs[src_idx][LDS_PAD(tid)],                q);
        uint64_t a1 = dev_reduce(bufs[src_idx][LDS_PAD(tid + quart)],        q);
        uint64_t a2 = dev_reduce(bufs[src_idx][LDS_PAD(tid + half)],         q);
        uint64_t a3 = dev_reduce(bufs[src_idx][LDS_PAD(tid + half + quart)], q);
        if (n_inv) {
            uint64_t ni_inv = ((uint64_t)n_inv << 32) / q;
            uint64_t ni_pk  = (ni_inv << 32) | (uint32_t)n_inv;
            a0 = dev_shoup_mul(ni_pk, a0, q);
            a1 = dev_shoup_mul(ni_pk, a1, q);
            a2 = dev_shoup_mul(ni_pk, a2, q);
            a3 = dev_shoup_mul(ni_pk, a3, q);
        }
        out_g[tid]                = a0;
        out_g[tid + quart]        = a1;
        out_g[tid + half]         = a2;
        out_g[tid + half + quart] = a3;
    }
}

/*
 * stockham_lds_fused_coarse_kernel — G4 extension of the G3 fused kernel.
 * Each thread covers multiple butterflies via a strided loop so we can
 * fuse all stages for n up to 4096 on a single block (one butterfly per
 * thread would require >1024 threads, over the HIP per-block limit).
 *
 * LDS padding is dropped here because two padded buffers for n=4096 just
 * exceed the 64 KB per-block cap; stride-1 Stockham access is already at
 * the uint64_t 2-way theoretical minimum so the empirical penalty is
 * small — the fusion win dominates.
 */
__global__ __launch_bounds__(1024, 2)
void stockham_lds_fused_coarse_kernel(const uint64_t * __restrict__ in_g,
                                            uint64_t * __restrict__ out_g,
                                      const uint64_t * __restrict__ tw,
                                      uint64_t n, uint64_t q, uint32_t log2_n,
                                      uint64_t n_inv)
{
    extern __shared__ uint64_t lds_raw[];
    uint64_t *bufs[2];
    bufs[0] = lds_raw;
    bufs[1] = lds_raw + n;  /* unpadded: total 2*n uint64_t = 2*n*8 bytes */

    uint64_t tid  = threadIdx.x;
    uint64_t nth  = blockDim.x;
    uint64_t half = n >> 1;

    /* Load: each thread covers n/nth contiguous elements. */
    for (uint64_t i = tid; i < n; i += nth) {
        bufs[0][i] = in_g[i];
    }
    __syncthreads();

    int src_idx = 0;
    for (uint32_t s = 0; s < log2_n; s++) {
        uint64_t p  = (uint64_t)1 << s;
        uint64_t qs = n >> (s + 1);
        uint64_t p2 = p << 1;

        for (uint64_t bid = tid; bid < half; bid += nth) {
            uint64_t j = bid & (p - 1);
            uint64_t k = bid >> s;
            uint64_t idx_u = j + k * p;
            uint64_t idx_v = idx_u + half;
            uint64_t u  = bufs[src_idx][idx_u];
            uint64_t v  = bufs[src_idx][idx_v];
            uint64_t w  = tw[j * qs];
            /* Both u and v may be in [0, 2q) from the previous lazy stage.
             * Reduce both to [0, q) so butterfly output stays in [0, 2q). */
            u  = dev_reduce(u, q);
            uint64_t wv = dev_shoup_mul(w, dev_reduce(v, q), q);
            bufs[1 - src_idx][j + k * p2]     = u + wv;
            bufs[1 - src_idx][j + k * p2 + p] = u + q - wv;
        }
        __syncthreads();
        src_idx = 1 - src_idx;
    }

    if (n_inv) {
        uint64_t ni_inv = ((uint64_t)n_inv << 32) / q;
        uint64_t ni_pk  = (ni_inv << 32) | (uint32_t)n_inv;
        for (uint64_t i = tid; i < n; i += nth)
            out_g[i] = dev_shoup_mul(ni_pk, dev_reduce(bufs[src_idx][i], q), q);
    } else {
        for (uint64_t i = tid; i < n; i += nth)
            out_g[i] = dev_reduce(bufs[src_idx][i], q);
    }
}

/*
 * stockham_reduce_kernel: final reduction pass — bring all values to [0, q).
 * Each thread reduces one element. Run after all log2(n) butterfly stages.
 */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void stockham_reduce_kernel(uint64_t * __restrict__ a, uint64_t n, uint64_t q)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) a[tid] = dev_reduce(a[tid], q);
}

/*
 * stockham_scale_kernel: multiply every element by n_inv mod q.
 * Applied after the inverse NTT butterfly stages.
 * packed_n_inv = (floor(n_inv*2^32/q) << 32) | n_inv — Shoup packing.
 */
__global__ __launch_bounds__(NTT_MAX_BLOCK, NTT_MIN_BLOCKS)
void stockham_scale_kernel(uint64_t * __restrict__ a, uint64_t n,
                           uint64_t q, uint64_t packed_n_inv)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) a[tid] = dev_shoup_mul(packed_n_inv, dev_reduce(a[tid], q), q);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MODULAR ARITHMETIC  (host-side: twiddle precomputation)
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint64_t h_mod_pow(uint64_t base, uint64_t exp, uint64_t m)
{
    uint64_t r = 1; base %= m;
    for (; exp; exp >>= 1) {
        if (exp & 1) r = (uint64_t)((__uint128_t)r * base % m);
        base = (uint64_t)((__uint128_t)base * base % m);
    }
    return r;
}
static uint64_t h_mod_inv(uint64_t a, uint64_t m) { return h_mod_pow(a, m-2, m); }

/*
 * root_is_primitive: omega must be a primitive n-th root of unity mod q
 * (omega^n == 1 and omega^(n/2) != 1). Without this the selftest is
 * vacuous: round-trip and "GPU vs CPU Stockham" both hold for ANY
 * invertible omega (incl. omega=1) because the CPU ref uses the same
 * twiddle table and algorithm, and impulse->all-ones holds for any tw.
 */
static int root_is_primitive(uint64_t n, uint64_t q, uint64_t omega)
{
    if (omega == 0) return 0;
    if (h_mod_pow(omega, n, q) != 1) return 0;
    if (h_mod_pow(omega, n >> 1, q) == 1) return 0;
    return 1;
}

static int ntt_params_init(ntt_params_t *p)
{
    if (!p->n || (p->n & (p->n-1))) return -1;
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
    p->omega_inv = h_mod_inv(p->omega, p->q);
    p->n_inv     = h_mod_inv(p->n,     p->q);
    uint64_t t = p->n; p->log2_n = 0;
    while (t > 1) { t >>= 1; p->log2_n++; }
    return 0;
}

/* Pack (w_inv, w) into a single uint64_t for Shoup reduction on the GPU.
 * Upper 32 bits = w_inv = floor(w * 2^32 / q); lower 32 bits = w.
 * Callers (kernels) decode via dev_shoup_mul. */
static uint64_t shoup_pack_stok(uint64_t w, uint64_t q)
{
    uint64_t w_inv = (uint64_t)(((unsigned __int128)w << 32) / q);
    return (w_inv << 32) | (uint32_t)w;
}

static uint64_t *alloc_twiddles(const ntt_params_t *p, uint64_t root)
{
    uint64_t *tw = (uint64_t *)malloc(p->n / 2 * sizeof(uint64_t));
    if (!tw) return NULL;
    uint64_t w = 1;
    for (uint64_t k = 0; k < p->n / 2; k++) {
        tw[k] = shoup_pack_stok(w, p->q);
        w = (uint64_t)((__uint128_t)w * root % p->q);
    }
    return tw;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GPU NTT LAUNCH WRAPPER
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * gpu_stockham: run Stockham NTT on device.
 * Allocates two device buffers; ping-pongs between them for log2(n) stages.
 * The result always ends up back in d_a via a final device memcpy if needed.
 * After all stages, applies reduce; for INTT also applies scale.
 *
 * is_inverse: 0 = forward NTT, 1 = inverse (adds scale pass).
 */
/*
 * LDS-fused path eligibility (fine, G3): one block owns the whole NTT,
 * one butterfly per thread, padded LDS.
 *   - n/2 ≤ 1024 (HIP per-block thread limit) → n ≤ 2048
 *   - 2 padded buffers fit in per-block LDS
 */
static int fused_lds_eligible(uint64_t n, int max_shared_b)
{
    if (n < 2 || n > 2048) return 0;
    size_t stride = (size_t)(n + (n >> 5) + 1);
    size_t bytes  = 2 * stride * sizeof(uint64_t);
    return (int)bytes <= max_shared_b;
}

/*
 * Coarse fused-LDS eligibility (G4): one block, strided loop so each
 * thread handles multiple butterflies. Unpadded LDS (2*n*sizeof uint64_t)
 * so the block fits n up to 4096 on 64 KB LDS hardware.
 */
static int fused_lds_coarse_eligible(uint64_t n, int max_shared_b)
{
    if (n < 4 || (n & (n - 1)) != 0) return 0;
    size_t bytes = 2 * (size_t)n * sizeof(uint64_t);
    return (int)bytes <= max_shared_b;
}

/*
 * Register-blocked fused eligibility (G5): like G3 (padded) but threads
 * = n/4 instead of n/2, with 2 butterflies per thread.
 */
static int fused_lds_reg2_eligible(uint64_t n, int max_shared_b)
{
    if (n < 4 || n > 2048) return 0;
    size_t stride = (size_t)(n + (n >> 5) + 1);
    size_t bytes  = 2 * stride * sizeof(uint64_t);
    return (int)bytes <= max_shared_b;
}

/* On-the-fly twiddle fused eligibility (G6): same shape as G3 padded. */
static int fused_lds_otf_eligible(uint64_t n, int max_shared_b)
{
    return fused_lds_eligible(n, max_shared_b);
}

/* Flag used by benchmark to skip the multi-launch path when a fused
 * launch already covered the NTT. Set on every gpu_stockham call. */
static int g_used_fused = 0;

static void gpu_stockham(uint64_t *a_host, const uint64_t *tw_host,
                         const ntt_params_t *p, int is_inverse)
{
    uint64_t n    = p->n;
    uint64_t half = n >> 1;
    int      bsz  = g_gpu.block_size;
    int      gsz_half = (int)((half + bsz - 1) / bsz);
    int      gsz_full = (int)((n    + bsz - 1) / bsz);

    uint64_t *d_a, *d_b, *d_tw;
    HIP_CHECK(hipMalloc((void **)&d_a,  n    * sizeof *d_a));
    HIP_CHECK(hipMalloc((void **)&d_b,  n    * sizeof *d_b));
    HIP_CHECK(hipMalloc((void **)&d_tw, half * sizeof *d_tw));

    HIP_CHECK(hipMemcpy(d_a,  a_host,  n    * sizeof *d_a,  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tw, tw_host, half * sizeof *d_tw, hipMemcpyHostToDevice));

    uint64_t *d_result = d_a;

    /*
     * Dispatch note: G5's stockham_lds_fused_reg2_kernel is kept for
     * evaluation on MI300A (warp=64, different occupancy dynamics) but
     * not selected by default — on 6900XT it loses to the G3 fine path
     * at n=256..2048 because halving the thread count leaves too few
     * wavefronts on one CU to hide latency. See run_benchmark output
     * and G5 row in STATUS.md.
     */
    if (fused_lds_eligible(n, g_gpu.max_shared_b)) {
        g_used_fused = 1;
        size_t stride  = (size_t)(n + (n >> 5) + 1);
        size_t lds_sz  = 2 * stride * sizeof(uint64_t);
        int    threads = (int)half;
        uint64_t n_inv_arg = is_inverse ? p->n_inv : 0ULL;
        hipLaunchKernelGGL(stockham_lds_fused_kernel, dim3(1), dim3(threads),
                           lds_sz, 0, d_a, d_b, d_tw, n, p->q, p->log2_n,
                           n_inv_arg);
        HIP_CHECK(hipGetLastError());
        d_result = d_b;
    } else if (fused_lds_coarse_eligible(n, g_gpu.max_shared_b)) {
        g_used_fused = 2;
        size_t lds_sz = 2 * (size_t)n * sizeof(uint64_t);
        int    threads = (half < 1024) ? (int)half : 1024;
        uint64_t n_inv_arg = is_inverse ? p->n_inv : 0ULL;
        hipLaunchKernelGGL(stockham_lds_fused_coarse_kernel, dim3(1),
                           dim3(threads), lds_sz, 0,
                           d_a, d_b, d_tw, n, p->q, p->log2_n, n_inv_arg);
        HIP_CHECK(hipGetLastError());
        d_result = d_b;
    } else {
        g_used_fused = 0;
        /* Multi-launch fallback: one kernel per stage, ping-pong between buffers. */
        uint64_t *d_src = d_a, *d_dst = d_b;
        for (uint32_t s = 0; s < p->log2_n; s++) {
            hipLaunchKernelGGL(stockham_stage_kernel, gsz_half, bsz, 0, 0,
                               d_src, d_dst, d_tw, n, p->q, s);
            HIP_CHECK(hipGetLastError());
            uint64_t *tmp = d_src; d_src = d_dst; d_dst = tmp;
        }
        hipLaunchKernelGGL(stockham_reduce_kernel, gsz_full, bsz, 0, 0,
                           d_src, n, p->q);
        HIP_CHECK(hipGetLastError());
        if (is_inverse) {
            uint64_t ni_inv = (uint64_t)(((unsigned __int128)p->n_inv << 32) / p->q);
            uint64_t ni_pk  = (ni_inv << 32) | (uint32_t)p->n_inv;
            hipLaunchKernelGGL(stockham_scale_kernel, gsz_full, bsz, 0, 0,
                               d_src, n, p->q, ni_pk);
            HIP_CHECK(hipGetLastError());
        }
        d_result = d_src;
    }

    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(a_host, d_result, n * sizeof *d_a, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_tw));
}

/* ── Public API (matches ntt.h signatures) ──────────────────────────────── */
void ntt_forward(uint64_t *a, const uint64_t *twiddles, const ntt_params_t *p)
{
    gpu_stockham(a, twiddles, p, 0);
}

void ntt_inverse(uint64_t *a, const uint64_t *twiddles_inv, const ntt_params_t *p)
{
    gpu_stockham(a, twiddles_inv, p, 1);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DISPLAY HELPERS
 * ═══════════════════════════════════════════════════════════════════════════ */

static void print_gpu_summary(void)
{
    double mem_gb = g_gpu.total_mem / (1024.0 * 1024.0 * 1024.0);
    printf(ANSI_CYN "  ── GPU Hardware ─────────────────────────────────────────\n" ANSI_RST);
    printf("  ┌──────────────────────────┬─────────────────────────┐\n");
    printf("  │ %-24s │ %-23s │\n", "Device", g_gpu.name);
    printf("  │ %-24s │ %-23d │\n", "Wavefront size",   g_gpu.warp_size);
    printf("  │ %-24s │ %-23d │\n", "Compute units",    g_gpu.cu_count);
    printf("  │ %-24s │ %-23d │\n", "LDS/block (bytes)",g_gpu.max_shared_b);
    printf("  │ %-24s │ %.2f %-18s │\n", "Total memory", mem_gb, "GB");
    printf("  │ %-24s │ %-23d │\n", "Launch block size",g_gpu.block_size);
    printf("  └──────────────────────────┴─────────────────────────┘\n\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CPU REFERENCE  (static — for selftest cross-check only)
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * ref_stockham_core: CPU Stockham (same algorithm as ntt_stockham.c).
 * Used to generate expected output for the GPU selftest without linking
 * against ntt_stockham.c.
 */
static void ref_stockham_forward(uint64_t *a, const uint64_t *tw,
                                  const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q, half = n >> 1;
    uint64_t *buf = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!buf) { fprintf(stderr, "ref_stockham: malloc failed\n"); return; }

    uint64_t *src = a, *dst = buf;
    for (uint32_t s = 0; s < p->log2_n; s++) {
        uint64_t pp = (uint64_t)1 << s;
        uint64_t qs = n >> (s + 1);
        uint64_t p2 = pp << 1;
        for (uint64_t j = 0; j < pp; j++) {
            uint64_t w = (uint32_t)tw[j * qs];  /* unpack Shoup-packed twiddle: lower 32 bits = w */
            for (uint64_t k = 0; k < qs; k++) {
                uint64_t idx = j + k * pp;
                uint64_t u   = src[idx];
                uint64_t wv  = (uint64_t)((__uint128_t)w * (src[idx + half] % q) % q);
                dst[j + k * p2]      = u + wv;
                dst[j + k * p2 + pp] = u + q - wv;
            }
        }
        uint64_t *tmp = src; src = dst; dst = tmp;
    }
    for (uint64_t i = 0; i < n; i++) src[i] %= q;
    if (src != a) memcpy(a, src, n * sizeof(uint64_t));
    free(buf);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SELFTEST
 * ═══════════════════════════════════════════════════════════════════════════ */

static int selftest(const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    const char *lbl = (q==3329)?"ML-KEM":(q==8380417)?"ML-DSA":"custom";

    printf(ANSI_CYN "  ── Selftest: %s (n=%lu q=%lu ω=%lu) ──────────────────\n"
           ANSI_RST, lbl, n, q, p->omega);

    uint64_t *tw  = alloc_twiddles(p, p->omega);
    uint64_t *twi = alloc_twiddles(p, p->omega_inv);
    uint64_t *a   = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *ref = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tw || !twi || !a || !ref) {
        free(tw); free(twi); free(a); free(ref); return -1;
    }

    /* Fill with pseudo-random pattern */
    for (uint64_t i = 0; i < n; i++) a[i] = ref[i] = (i * 6364136223846793005ULL + 1) % q;

    /* Test 1: GPU round-trip */
    ntt_forward(a, tw,  p);
    ntt_inverse(a, twi, p);
    int rt_ok = 1;
    for (uint64_t i = 0; i < n; i++) if (a[i] != ref[i]) { rt_ok = 0; break; }
    printf("  %-36s %s\n", "GPU round-trip (fwd→inv = identity)",
           rt_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    /* Test 2: GPU forward vs CPU Stockham reference element-wise */
    for (uint64_t i = 0; i < n; i++) a[i] = ref[i] = (i * 6364136223846793005ULL + 1) % q;
    ntt_forward(a, tw, p);
    ref_stockham_forward(ref, tw, p);
    int cmp_ok = 1;
    for (uint64_t i = 0; i < n; i++) if (a[i] != ref[i]) { cmp_ok = 0; break; }
    printf("  %-36s %s\n", "GPU vs CPU Stockham (element-wise)",
           cmp_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    /* Test 3: unit impulse → all-ones */
    a[0] = 1; for (uint64_t i = 1; i < n; i++) a[i] = 0;
    ntt_forward(a, tw, p);
    int imp_ok = 1;
    for (uint64_t i = 0; i < n; i++) if (a[i] != 1) { imp_ok = 0; break; }
    printf("  %-36s %s\n", "impulse [1,0,...,0] → all-ones",
           imp_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    free(tw); free(twi); free(a); free(ref);

    int ok = rt_ok && cmp_ok && imp_ok;
    printf("  %s  n=%-4lu q=%lu\n\n",
           ok ? ANSI_GRN "ALL PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST, n, q);
    return ok ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK
 * ═══════════════════════════════════════════════════════════════════════════ */

static double bench_multi_launch(const ntt_params_t *p, const uint64_t *tw,
                                 uint64_t iters)
{
    uint64_t n = p->n;
    uint64_t *d_a, *d_b, *d_tw;
    (void)hipMalloc((void **)&d_a, n * sizeof(*d_a));
    (void)hipMalloc((void **)&d_b, n * sizeof(*d_b));
    (void)hipMalloc((void **)&d_tw, (n >> 1) * sizeof(*d_tw));
    (void)hipMemcpy(d_tw, tw, (n >> 1) * sizeof(*d_tw), hipMemcpyHostToDevice);

    int bsz      = g_gpu.block_size;
    int gsz_half = (int)(((n >> 1) + bsz - 1) / bsz);
    int gsz_full = (int)((n + bsz - 1) / bsz);

    /* Warm up */
    uint64_t *ws = d_a, *wd = d_b;
    for (uint32_t s = 0; s < p->log2_n; s++) {
        hipLaunchKernelGGL(stockham_stage_kernel, gsz_half, bsz, 0, 0,
                           ws, wd, d_tw, n, p->q, s);
        uint64_t *t = ws; ws = wd; wd = t;
    }
    hipLaunchKernelGGL(stockham_reduce_kernel, gsz_full, bsz, 0, 0, ws, n, p->q);
    (void)hipDeviceSynchronize();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) {
        uint64_t *d_src = d_a, *d_dst = d_b;
        for (uint32_t s = 0; s < p->log2_n; s++) {
            hipLaunchKernelGGL(stockham_stage_kernel, gsz_half, bsz, 0, 0,
                               d_src, d_dst, d_tw, n, p->q, s);
            uint64_t *tmp = d_src; d_src = d_dst; d_dst = tmp;
        }
        hipLaunchKernelGGL(stockham_reduce_kernel, gsz_full, bsz, 0, 0,
                           d_src, n, p->q);
    }
    (void)hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    (void)hipFree(d_a); (void)hipFree(d_b); (void)hipFree(d_tw);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static double bench_lds_fused(const ntt_params_t *p, const uint64_t *tw,
                              uint64_t iters)
{
    uint64_t n = p->n;
    int use_fine   = fused_lds_eligible(n, g_gpu.max_shared_b);
    int use_coarse = !use_fine && fused_lds_coarse_eligible(n, g_gpu.max_shared_b);
    if (!use_fine && !use_coarse) return -1.0;

    uint64_t *d_a, *d_b, *d_tw;
    (void)hipMalloc((void **)&d_a,  n * sizeof(*d_a));
    (void)hipMalloc((void **)&d_b,  n * sizeof(*d_b));
    (void)hipMalloc((void **)&d_tw, (n >> 1) * sizeof(*d_tw));
    (void)hipMemcpy(d_tw, tw, (n >> 1) * sizeof(*d_tw), hipMemcpyHostToDevice);

    size_t lds_sz;
    int    threads;
    if (use_fine) {
        size_t stride = (size_t)(n + (n >> 5) + 1);
        lds_sz  = 2 * stride * sizeof(uint64_t);
        threads = (int)(n >> 1);
    } else {
        lds_sz  = 2 * (size_t)n * sizeof(uint64_t);
        threads = ((n >> 1) < 1024) ? (int)(n >> 1) : 1024;
    }

    /* Warm up */
    if (use_fine) {
        hipLaunchKernelGGL(stockham_lds_fused_kernel, dim3(1), dim3(threads),
                           lds_sz, 0, d_a, d_b, d_tw, n, p->q, p->log2_n, 0ULL);
    } else {
        hipLaunchKernelGGL(stockham_lds_fused_coarse_kernel, dim3(1), dim3(threads),
                           lds_sz, 0, d_a, d_b, d_tw, n, p->q, p->log2_n, 0ULL);
    }
    (void)hipDeviceSynchronize();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) {
        if (use_fine) {
            hipLaunchKernelGGL(stockham_lds_fused_kernel, dim3(1), dim3(threads),
                               lds_sz, 0, d_a, d_b, d_tw, n, p->q, p->log2_n, 0ULL);
        } else {
            hipLaunchKernelGGL(stockham_lds_fused_coarse_kernel, dim3(1), dim3(threads),
                               lds_sz, 0, d_a, d_b, d_tw, n, p->q, p->log2_n, 0ULL);
        }
    }
    (void)hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    (void)hipFree(d_a); (void)hipFree(d_b); (void)hipFree(d_tw);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/*
 * bench_lds_otf: G6 on-the-fly twiddle kernel bench. Uses the same
 * padded-LDS block shape as the fine fused kernel, so eligibility tracks
 * fused_lds_eligible. Takes p->omega directly; no d_tw allocation.
 */
static double bench_lds_otf(const ntt_params_t *p, uint64_t iters)
{
    uint64_t n = p->n;
    if (!fused_lds_otf_eligible(n, g_gpu.max_shared_b)) return -1.0;

    uint64_t *d_a, *d_b;
    (void)hipMalloc((void **)&d_a, n * sizeof(*d_a));
    (void)hipMalloc((void **)&d_b, n * sizeof(*d_b));

    size_t stride  = (size_t)(n + (n >> 5) + 1);
    size_t lds_sz  = 2 * stride * sizeof(uint64_t);
    int    threads = (int)(n >> 1);

    hipLaunchKernelGGL(stockham_lds_fused_otf_kernel, dim3(1), dim3(threads),
                       lds_sz, 0, d_a, d_b, p->omega, n, p->q, p->log2_n, 0ULL);
    (void)hipDeviceSynchronize();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) {
        hipLaunchKernelGGL(stockham_lds_fused_otf_kernel, dim3(1), dim3(threads),
                           lds_sz, 0, d_a, d_b, p->omega, n, p->q, p->log2_n,
                           0ULL);
    }
    (void)hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    (void)hipFree(d_a); (void)hipFree(d_b);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/*
 * verify_otf_matches_tabled: run OTF and tabled fused kernels on the same
 * input and memcmp the device outputs. Returns 0 on match, -1 on mismatch
 * or skip (n ineligible for the fine kernel).
 */
static int verify_otf_matches_tabled(const ntt_params_t *p, const uint64_t *tw)
{
    uint64_t n = p->n;
    if (!fused_lds_eligible(n, g_gpu.max_shared_b)) return -1;

    uint64_t *h_in  = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *h_ot  = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *h_tb  = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!h_in || !h_ot || !h_tb) { free(h_in); free(h_ot); free(h_tb); return -1; }
    for (uint64_t i = 0; i < n; i++) h_in[i] = (i * 2654435761ULL + 1) % p->q;

    uint64_t *d_in, *d_out, *d_tw;
    (void)hipMalloc((void **)&d_in,  n * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_out, n * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_tw,  (n >> 1) * sizeof(uint64_t));
    (void)hipMemcpy(d_tw, tw, (n >> 1) * sizeof(uint64_t), hipMemcpyHostToDevice);

    size_t stride = (size_t)(n + (n >> 5) + 1);
    size_t lds_sz = 2 * stride * sizeof(uint64_t);
    int    threads = (int)(n >> 1);

    /* Tabled */
    (void)hipMemcpy(d_in, h_in, n * sizeof(uint64_t), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(stockham_lds_fused_kernel, dim3(1), dim3(threads),
                       lds_sz, 0, d_in, d_out, d_tw, n, p->q, p->log2_n, 0ULL);
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(h_tb, d_out, n * sizeof(uint64_t), hipMemcpyDeviceToHost);

    /* OTF */
    (void)hipMemcpy(d_in, h_in, n * sizeof(uint64_t), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(stockham_lds_fused_otf_kernel, dim3(1), dim3(threads),
                       lds_sz, 0, d_in, d_out, p->omega, n, p->q, p->log2_n,
                       0ULL);
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(h_ot, d_out, n * sizeof(uint64_t), hipMemcpyDeviceToHost);

    int ok = (memcmp(h_ot, h_tb, n * sizeof(uint64_t)) == 0);

    (void)hipFree(d_in); (void)hipFree(d_out); (void)hipFree(d_tw);
    free(h_in); free(h_ot); free(h_tb);
    return ok ? 0 : -1;
}

/*
 * bench_lds_batched: time `batch` independent NTTs per launch via
 * stockham_lds_batched_kernel. Returns seconds for iters*batch total
 * NTTs. Also verifies against the single-NTT fused kernel on batch 0.
 */
static double bench_lds_batched(const ntt_params_t *p, const uint64_t *tw,
                                uint64_t iters, uint64_t batch, int *ok_out)
{
    uint64_t n = p->n;
    if (!fused_lds_eligible(n, g_gpu.max_shared_b)) return -1.0;

    uint64_t *d_a, *d_b, *d_tw;
    (void)hipMalloc((void **)&d_a,  batch * n * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_b,  batch * n * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_tw, (n >> 1) * sizeof(uint64_t));
    (void)hipMemcpy(d_tw, tw, (n >> 1) * sizeof(uint64_t), hipMemcpyHostToDevice);

    /* Fill input: all batches get the same pattern so batch 0 can be
     * cross-checked against the single-NTT fused kernel. */
    uint64_t *h_in = (uint64_t *)malloc(batch * n * sizeof(uint64_t));
    for (uint64_t b = 0; b < batch; b++)
        for (uint64_t i = 0; i < n; i++)
            h_in[b * n + i] = (i * 2654435761ULL + 1) % p->q;
    (void)hipMemcpy(d_a, h_in, batch * n * sizeof(uint64_t),
                    hipMemcpyHostToDevice);

    size_t stride  = (size_t)(n + (n >> 5) + 1);
    size_t lds_sz  = 2 * stride * sizeof(uint64_t);
    int    threads = (int)(n >> 1);

    hipLaunchKernelGGL(stockham_lds_batched_kernel, dim3((uint32_t)batch),
                       dim3(threads), lds_sz, 0, d_a, d_b, d_tw,
                       n, p->q, p->log2_n, 0ULL);
    (void)hipDeviceSynchronize();

    /* Correctness: batch 0 output vs single-kernel reference. */
    if (ok_out) {
        uint64_t *h_ref = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *h_bat = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *d_ref_in, *d_ref_out;
        (void)hipMalloc((void **)&d_ref_in,  n * sizeof(uint64_t));
        (void)hipMalloc((void **)&d_ref_out, n * sizeof(uint64_t));
        (void)hipMemcpy(d_ref_in, h_in, n * sizeof(uint64_t),
                        hipMemcpyHostToDevice);
        hipLaunchKernelGGL(stockham_lds_fused_kernel, dim3(1), dim3(threads),
                           lds_sz, 0, d_ref_in, d_ref_out, d_tw, n, p->q,
                           p->log2_n, 0ULL);
        (void)hipDeviceSynchronize();
        (void)hipMemcpy(h_ref, d_ref_out, n * sizeof(uint64_t),
                        hipMemcpyDeviceToHost);
        (void)hipMemcpy(h_bat, d_b, n * sizeof(uint64_t),
                        hipMemcpyDeviceToHost);
        *ok_out = (memcmp(h_ref, h_bat, n * sizeof(uint64_t)) == 0);
        free(h_ref); free(h_bat);
        (void)hipFree(d_ref_in); (void)hipFree(d_ref_out);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) {
        hipLaunchKernelGGL(stockham_lds_batched_kernel, dim3((uint32_t)batch),
                           dim3(threads), lds_sz, 0, d_a, d_b, d_tw,
                           n, p->q, p->log2_n, 0ULL);
    }
    (void)hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    free(h_in);
    (void)hipFree(d_a); (void)hipFree(d_b); (void)hipFree(d_tw);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static double bench_lds_reg2(const ntt_params_t *p, const uint64_t *tw,
                             uint64_t iters)
{
    uint64_t n = p->n;
    if (!fused_lds_reg2_eligible(n, g_gpu.max_shared_b)) return -1.0;

    uint64_t *d_a, *d_b, *d_tw;
    (void)hipMalloc((void **)&d_a,  n * sizeof(*d_a));
    (void)hipMalloc((void **)&d_b,  n * sizeof(*d_b));
    (void)hipMalloc((void **)&d_tw, (n >> 1) * sizeof(*d_tw));
    (void)hipMemcpy(d_tw, tw, (n >> 1) * sizeof(*d_tw), hipMemcpyHostToDevice);

    size_t stride  = (size_t)(n + (n >> 5) + 1);
    size_t lds_sz  = 2 * stride * sizeof(uint64_t);
    int    threads = (int)(n >> 2);
    if (threads < 1) threads = 1;

    hipLaunchKernelGGL(stockham_lds_fused_reg2_kernel, dim3(1), dim3(threads),
                       lds_sz, 0, d_a, d_b, d_tw, n, p->q, p->log2_n, 0ULL);
    (void)hipDeviceSynchronize();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) {
        hipLaunchKernelGGL(stockham_lds_fused_reg2_kernel, dim3(1), dim3(threads),
                           lds_sz, 0, d_a, d_b, d_tw, n, p->q, p->log2_n, 0ULL);
    }
    (void)hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    (void)hipFree(d_a); (void)hipFree(d_b); (void)hipFree(d_tw);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static uint64_t g_batch_override = 0;  /* A2 sweep: 0 = use CU count */

static void run_benchmark(const ntt_params_t *p, uint64_t iters)
{
    uint64_t n = p->n, q = p->q;
    uint64_t *tw = alloc_twiddles(p, p->omega);
    uint64_t *a  = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!tw || !a) { free(tw); free(a); return; }
    for (uint64_t i = 0; i < n; i++) a[i] = (i * 31337 + 1) % q;

    /* G6 correctness check before timing. */
    int otf_ok = (verify_otf_matches_tabled(p, tw) == 0);
    printf("  %-26s %s\n", "OTF vs tabled fused",
           otf_ok ? ANSI_GRN "PASS" ANSI_RST :
                    fused_lds_eligible(n, g_gpu.max_shared_b)
                    ? ANSI_RED "FAIL" ANSI_RST : ANSI_YLW "skip" ANSI_RST);

    double t_multi = bench_multi_launch(p, tw, iters);
    double t_fused = bench_lds_fused   (p, tw, iters);
    double t_reg2  = bench_lds_reg2    (p, tw, iters);
    double t_otf   = bench_lds_otf     (p, iters);

    double s_multi = iters / t_multi;
    double ns_multi = t_multi * 1e9 / ((double)iters * n * p->log2_n / 2.0);

    printf(ANSI_CYN "  ── Benchmark: GPU Stockham (n=%lu q=%lu) ──────────────\n"
           ANSI_RST, n, q);
    printf("  ┌──────────────────────────┬──────────────┬──────────────┐\n");
    printf("  │ %-24s │ %-12s │ %-12s │\n",
           "variant", "NTT/s", "ns/butterfly");
    printf("  ├──────────────────────────┼──────────────┼──────────────┤\n");
    printf("  │ %-24s │ %12.0f │ %12.2f │\n",
           "multi-launch (global)", s_multi, ns_multi);
    if (t_fused > 0.0) {
        double s_fused  = iters / t_fused;
        double ns_fused = t_fused * 1e9 / ((double)iters * n * p->log2_n / 2.0);
        const char *variant = fused_lds_eligible(n, g_gpu.max_shared_b)
                              ? "LDS-fused (padded)" : "LDS-fused (coarse)";
        printf("  │ %-24s │ %12.0f │ %12.2f │\n", variant, s_fused, ns_fused);
        printf("  │ %-24s │ %12.2fx │ %12s │\n",
               "speedup (fused/multi)", s_fused / s_multi, "");
    } else {
        printf("  │ %-24s │ %-12s │ %-12s │\n",
               "LDS-fused", "— n too big", "");
    }
    if (t_reg2 > 0.0) {
        double s_reg2  = iters / t_reg2;
        double ns_reg2 = t_reg2 * 1e9 / ((double)iters * n * p->log2_n / 2.0);
        printf("  │ %-24s │ %12.0f │ %12.2f │\n",
               "LDS-fused reg2 (G5)", s_reg2, ns_reg2);
        if (t_fused > 0.0) {
            double ratio = (iters / t_reg2) / (iters / t_fused);
            printf("  │ %-24s │ %12.2fx │ %12s │\n",
                   "speedup (reg2/fused)", ratio, "");
        }
    }
    if (t_otf > 0.0) {
        double s_otf  = iters / t_otf;
        double ns_otf = t_otf * 1e9 / ((double)iters * n * p->log2_n / 2.0);
        printf("  │ %-24s │ %12.0f │ %12.2f │\n",
               "LDS-fused OTF (G6)", s_otf, ns_otf);
        if (t_fused > 0.0) {
            double ratio = (iters / t_otf) / (iters / t_fused);
            printf("  │ %-24s │ %12.2fx │ %12s │\n",
                   "speedup (OTF/fused)", ratio, "");
        }
    }
    /* Batched — measure throughput at batch = 40 (6900XT CU count). */
    if (fused_lds_eligible(n, g_gpu.max_shared_b)) {
        uint64_t batch = g_batch_override ? g_batch_override : (uint64_t)g_gpu.cu_count;
        if (!g_batch_override && batch < 8) batch = 8;
        uint64_t bat_iters = (iters + batch - 1) / batch;
        if (bat_iters < 1) bat_iters = 1;
        int bat_ok = 0;
        double t_bat = bench_lds_batched(p, tw, bat_iters, batch, &bat_ok);
        if (t_bat > 0.0) {
            double total_ntts = (double)bat_iters * (double)batch;
            double s_bat  = total_ntts / t_bat;
            double ns_bat = t_bat * 1e9 / (total_ntts * n * p->log2_n / 2.0);
            char label[64];
            snprintf(label, sizeof label, "batched b=%lu (correctness %s)",
                     batch, bat_ok ? "PASS" : "FAIL");
            printf("  │ %-24s │ %12.0f │ %12.2f │\n", label, s_bat, ns_bat);
            if (t_fused > 0.0) {
                double ratio = s_bat / (iters / t_fused);
                printf("  │ %-24s │ %12.2fx │ %12s │\n",
                       "speedup (batched/fused)", ratio, "");
            }
        }
    }
    printf("  └──────────────────────────┴──────────────┴──────────────┘\n\n");

    free(tw); free(a);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    ntt_params_t p;
    p.n     = (argc > 1) ? (uint64_t)atol(argv[1]) : 256;
    p.q     = (argc > 2) ? (uint64_t)atol(argv[2]) : 3329;
    p.omega = (argc > 3) ? (uint64_t)atol(argv[3]) : 17;
    uint64_t iters = (argc > 4) ? (uint64_t)atol(argv[4]) : 1000;
    /* argv[5]: optional batch override (A2 batch scaling sweep). */
    uint64_t batch_override = (argc > 5) ? (uint64_t)atol(argv[5]) : 0;
    /* argv[6]: optional block-size override for multi-launch stage kernel (D4 sweep). */
    int block_override = (argc > 6) ? atoi(argv[6]) : 0;

    printf("\n" ANSI_WHT
           "╔══════════════════════════════════════════════════════════╗\n"
           "║       NTT / MI300A  —  GPU Stockham auto-sort           ║\n"
           "╚══════════════════════════════════════════════════════════╝\n"
           ANSI_RST "\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    strncpy(g_gpu.name, prop.name, sizeof g_gpu.name - 1);
    g_gpu.warp_size   = prop.warpSize;
    g_gpu.cu_count    = prop.multiProcessorCount;
    g_gpu.max_shared_b = (int)prop.sharedMemPerBlock;
    g_gpu.total_mem   = prop.totalGlobalMem;
    g_gpu.block_size  = 128;   /* safe for CDNA3 (wavefront=64) and RDNA3 (wavefront=32) */
    /* __launch_bounds__ ceiling for stockham_stage_kernel is NTT_MAX_BLOCK (256). */
    if (block_override >= 32 && block_override <= NTT_MAX_BLOCK) g_gpu.block_size = block_override;
    print_gpu_summary();

    if (ntt_params_init(&p) != 0) {
        fprintf(stderr, "error: n=%lu is not a power of 2\n", p.n);
        return 1;
    }

    ntt_params_t mlkem = { 256, 3329, 17, 0, 0, 0 };
    ntt_params_t mldsa = { 256, 8380417, 1753, 0, 0, 0 };
    int fail = 0;
    if (ntt_params_init(&mlkem) == 0) fail |= selftest(&mlkem);
    if (ntt_params_init(&mldsa) == 0) fail |= selftest(&mldsa);
    if (p.n != 256 || (p.q != 3329 && p.q != 8380417)) fail |= selftest(&p);

    if (fail) return 1;

    g_batch_override = batch_override;
    if (iters > 1) run_benchmark(&p, iters);
    return 0;
}
