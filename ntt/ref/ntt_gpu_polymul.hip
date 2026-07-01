/*
 * ntt_gpu_polymul.hip — G7 fused NTT + Hadamard + INTT polynomial multiply
 *
 * Purpose:   Compute c = f * g mod (X^n − 1, q) in a single GPU kernel,
 *            keeping both polynomials in LDS across the entire NTT(f),
 *            NTT(g), pointwise, INTT, scale pipeline. The separate-launch
 *            baseline does 4 kernel launches with 3 HBM round trips of n
 *            uint64_t each; this version does 1 launch with 2 HBM reads
 *            (f,g) and 1 HBM write (c).
 * Algorithm: Stockham auto-sort DIT radix-2, cyclic convolution. The NTT
 *            butterfly and LDS layout are identical to the G3 fine-fused
 *            kernel (stockham_lds_fused_kernel in ntt_gpu_stockham.hip);
 *            here we ping-pong TWO LDS buffers (one per polynomial) in
 *            lock step across all log2_n stages, then do the pointwise
 *            mul in-place, then re-run log2_n stages with the inverse
 *            twiddles on just the result buffer.
 * Ref:       arxiv.org/abs/2209.01290 (fused Hadamard rationale).
 * Target:    MI300A (gfx942, CDNA3, warp=64). Dev: 6900XT (gfx1030).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define ANSI_WHT "\033[1;37m"
#define ANSI_CYN "\033[1;36m"
#define ANSI_GRN "\033[1;32m"
#define ANSI_RED "\033[1;31m"
#define ANSI_RST "\033[0m"

#define HIP_CHECK(call) do { hipError_t _e = (call); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
    exit(1); } } while (0)

typedef struct {
    uint64_t n;
    uint64_t q;
    uint64_t omega;
    uint64_t omega_inv;
    uint64_t n_inv;
    uint32_t log2_n;
} ntt_params_t;

typedef struct {
    char name[256]; int warp_size; int cu_count; int max_shared_b;
    size_t total_mem; int block_size;
} gpu_config_t;
static gpu_config_t g_gpu;

/* ═══════════════════════════════════════════════════════════════════════════ */

__device__ __forceinline__
uint64_t dev_mulmod(uint64_t a, uint64_t b, uint64_t q)
{
    /* q < 2^32 so a*b < 2^64. */
    return (a * b) % q;
}

/*
 * stockham_stage_kernel / pointwise_kernel — SEPARATE-LAUNCH baseline.
 * Match the multi-launch path in ntt_gpu_stockham.hip so the comparison
 * is apples-to-apples (same algorithm, split across 4 kernels).
 */
__global__ __launch_bounds__(256, 8)
void pm_stockham_stage_kernel(const uint64_t * __restrict__ src,
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
    uint64_t wv = dev_mulmod(w, v % q, q);
    dst[j + k * p2]     = u + wv;
    dst[j + k * p2 + p] = u + q - wv;
}

__global__ __launch_bounds__(256, 8)
void pm_reduce_kernel(uint64_t * __restrict__ a, uint64_t n, uint64_t q)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) a[tid] %= q;
}

__global__ __launch_bounds__(256, 8)
void pm_pointwise_kernel(const uint64_t * __restrict__ f,
                         const uint64_t * __restrict__ g,
                               uint64_t * __restrict__ c,
                         uint64_t n, uint64_t q)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = dev_mulmod(f[tid] % q, g[tid] % q, q);
}

__global__ __launch_bounds__(256, 8)
void pm_scale_kernel(uint64_t * __restrict__ a, uint64_t n, uint64_t q,
                     uint64_t n_inv)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) a[tid] = dev_mulmod(a[tid] % q, n_inv, q);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * FUSED POLYMUL KERNEL
 *
 * Single block, blockDim.x = n/2. LDS layout: 4 unpadded n-element
 * buffers — two for f (ping-pong), two for g. Footprint = 4*n*8 bytes.
 * At n=1024 that is 32 KB; at n=2048 it is 64 KB (exactly fills a
 * 6900XT LDS block). n=256 is the target case (ML-KEM).
 *
 * Flow:
 *   1) Load f,g into fbuf[0], gbuf[0].
 *   2) log2_n forward butterfly stages — ping-pong both polynomials
 *      every stage under one __syncthreads() so the two NTTs overlap
 *      at the instruction level.
 *   3) Pointwise mul fbuf[src] ← fbuf[src] * gbuf[src]. Second buffer
 *      (gbuf) is dead after this point.
 *   4) log2_n inverse butterfly stages on fbuf only.
 *   5) Scale by n_inv, reduce, store to c_out.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ __launch_bounds__(1024, 2)
void fused_polymul_kernel(const uint64_t * __restrict__ f_in,
                          const uint64_t * __restrict__ g_in,
                                uint64_t * __restrict__ c_out,
                          const uint64_t * __restrict__ tw_fwd,
                          const uint64_t * __restrict__ tw_inv,
                          uint64_t n, uint64_t q, uint32_t log2_n,
                          uint64_t n_inv)
{
    extern __shared__ uint64_t lds_raw[];
    uint64_t *fbuf0 = lds_raw;
    uint64_t *fbuf1 = lds_raw + n;
    uint64_t *gbuf0 = lds_raw + 2 * n;
    uint64_t *gbuf1 = lds_raw + 3 * n;
    uint64_t *fbufs[2] = { fbuf0, fbuf1 };
    uint64_t *gbufs[2] = { gbuf0, gbuf1 };

    uint64_t tid  = threadIdx.x;
    uint64_t half = n >> 1;

    if (tid < half) {
        fbufs[0][tid]        = f_in[tid];
        fbufs[0][tid + half] = f_in[tid + half];
        gbufs[0][tid]        = g_in[tid];
        gbufs[0][tid + half] = g_in[tid + half];
    }
    __syncthreads();

    /* ── Forward NTT on f and g, lock-stepped ───────────────────────── */
    int src_idx = 0;
    for (uint32_t s = 0; s < log2_n; s++) {
        uint64_t p  = (uint64_t)1 << s;
        uint64_t qs = n >> (s + 1);
        uint64_t p2 = p << 1;
        uint64_t j  = tid & (p - 1);
        uint64_t k  = tid >> s;
        uint64_t idx_u = j + k * p;
        uint64_t idx_v = idx_u + half;
        uint64_t out0  = j + k * p2;
        uint64_t out1  = out0 + p;

        if (tid < half) {
            uint64_t w   = tw_fwd[j * qs];
            uint64_t fu  = fbufs[src_idx][idx_u];
            uint64_t fv  = fbufs[src_idx][idx_v];
            uint64_t gu  = gbufs[src_idx][idx_u];
            uint64_t gv  = gbufs[src_idx][idx_v];
            uint64_t fwv = dev_mulmod(w, fv % q, q);
            uint64_t gwv = dev_mulmod(w, gv % q, q);
            fbufs[1 - src_idx][out0] = fu + fwv;
            fbufs[1 - src_idx][out1] = fu + q - fwv;
            gbufs[1 - src_idx][out0] = gu + gwv;
            gbufs[1 - src_idx][out1] = gu + q - gwv;
        }
        __syncthreads();
        src_idx = 1 - src_idx;
    }

    /* ── Hadamard: fbuf[src] ← fbuf[src] * gbuf[src] ──────────────────── */
    if (tid < half) {
        uint64_t f0 = fbufs[src_idx][tid]        % q;
        uint64_t f1 = fbufs[src_idx][tid + half] % q;
        uint64_t g0 = gbufs[src_idx][tid]        % q;
        uint64_t g1 = gbufs[src_idx][tid + half] % q;
        fbufs[src_idx][tid]        = dev_mulmod(f0, g0, q);
        fbufs[src_idx][tid + half] = dev_mulmod(f1, g1, q);
    }
    __syncthreads();

    /* ── Inverse NTT on fbuf only ─────────────────────────────────────── */
    for (uint32_t s = 0; s < log2_n; s++) {
        uint64_t p  = (uint64_t)1 << s;
        uint64_t qs = n >> (s + 1);
        uint64_t p2 = p << 1;
        uint64_t j  = tid & (p - 1);
        uint64_t k  = tid >> s;
        uint64_t idx_u = j + k * p;
        uint64_t idx_v = idx_u + half;

        if (tid < half) {
            uint64_t w  = tw_inv[j * qs];
            uint64_t u  = fbufs[src_idx][idx_u];
            uint64_t v  = fbufs[src_idx][idx_v];
            uint64_t wv = dev_mulmod(w, v % q, q);
            fbufs[1 - src_idx][j + k * p2]     = u + wv;
            fbufs[1 - src_idx][j + k * p2 + p] = u + q - wv;
        }
        __syncthreads();
        src_idx = 1 - src_idx;
    }

    if (tid < half) {
        uint64_t a0 = fbufs[src_idx][tid]        % q;
        uint64_t a1 = fbufs[src_idx][tid + half] % q;
        c_out[tid]        = dev_mulmod(a0, n_inv, q);
        c_out[tid + half] = dev_mulmod(a1, n_inv, q);
    }
}

/*
 * D2 — batched fused polymul. Same body as fused_polymul_kernel but
 * offsets f_in / g_in / c_out by blockIdx.x * n, so grid = batch size.
 * Twiddle tables are shared across all batches. LDS cost per block is
 * still 4*n*uint64_t (≤ 64 KB for n ≤ 2048).
 */
__global__ __launch_bounds__(1024, 2)
void fused_polymul_batched_kernel(const uint64_t * __restrict__ f_in,
                                  const uint64_t * __restrict__ g_in,
                                        uint64_t * __restrict__ c_out,
                                  const uint64_t * __restrict__ tw_fwd,
                                  const uint64_t * __restrict__ tw_inv,
                                  uint64_t n, uint64_t q, uint32_t log2_n,
                                  uint64_t n_inv)
{
    extern __shared__ uint64_t lds_raw[];
    uint64_t *fbufs[2] = { lds_raw,         lds_raw + n };
    uint64_t *gbufs[2] = { lds_raw + 2*n,   lds_raw + 3*n };

    uint64_t tid  = threadIdx.x;
    uint64_t half = n >> 1;
    uint64_t off  = (uint64_t)blockIdx.x * n;

    if (tid < half) {
        fbufs[0][tid]        = f_in[off + tid];
        fbufs[0][tid + half] = f_in[off + tid + half];
        gbufs[0][tid]        = g_in[off + tid];
        gbufs[0][tid + half] = g_in[off + tid + half];
    }
    __syncthreads();

    int src_idx = 0;
    for (uint32_t s = 0; s < log2_n; s++) {
        uint64_t p  = (uint64_t)1 << s;
        uint64_t qs = n >> (s + 1);
        uint64_t p2 = p << 1;
        uint64_t j  = tid & (p - 1);
        uint64_t k  = tid >> s;
        if (tid < half) {
            uint64_t w   = tw_fwd[j * qs];
            uint64_t fu  = fbufs[src_idx][j + k * p];
            uint64_t fv  = fbufs[src_idx][j + k * p + half];
            uint64_t gu  = gbufs[src_idx][j + k * p];
            uint64_t gv  = gbufs[src_idx][j + k * p + half];
            uint64_t fwv = dev_mulmod(w, fv % q, q);
            uint64_t gwv = dev_mulmod(w, gv % q, q);
            fbufs[1 - src_idx][j + k * p2]     = fu + fwv;
            fbufs[1 - src_idx][j + k * p2 + p] = fu + q - fwv;
            gbufs[1 - src_idx][j + k * p2]     = gu + gwv;
            gbufs[1 - src_idx][j + k * p2 + p] = gu + q - gwv;
        }
        __syncthreads();
        src_idx = 1 - src_idx;
    }

    if (tid < half) {
        uint64_t f0 = fbufs[src_idx][tid]        % q;
        uint64_t f1 = fbufs[src_idx][tid + half] % q;
        uint64_t g0 = gbufs[src_idx][tid]        % q;
        uint64_t g1 = gbufs[src_idx][tid + half] % q;
        fbufs[src_idx][tid]        = dev_mulmod(f0, g0, q);
        fbufs[src_idx][tid + half] = dev_mulmod(f1, g1, q);
    }
    __syncthreads();

    for (uint32_t s = 0; s < log2_n; s++) {
        uint64_t p  = (uint64_t)1 << s;
        uint64_t qs = n >> (s + 1);
        uint64_t p2 = p << 1;
        uint64_t j  = tid & (p - 1);
        uint64_t k  = tid >> s;
        if (tid < half) {
            uint64_t w  = tw_inv[j * qs];
            uint64_t u  = fbufs[src_idx][j + k * p];
            uint64_t v  = fbufs[src_idx][j + k * p + half];
            uint64_t wv = dev_mulmod(w, v % q, q);
            fbufs[1 - src_idx][j + k * p2]     = u + wv;
            fbufs[1 - src_idx][j + k * p2 + p] = u + q - wv;
        }
        __syncthreads();
        src_idx = 1 - src_idx;
    }

    if (tid < half) {
        uint64_t a0 = fbufs[src_idx][tid]        % q;
        uint64_t a1 = fbufs[src_idx][tid + half] % q;
        c_out[off + tid]        = dev_mulmod(a0, n_inv, q);
        c_out[off + tid + half] = dev_mulmod(a1, n_inv, q);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * NEGACYCLIC FUSED POLYMUL (mod X^n + 1) — twisted-NTT
 *
 * Requires a 2n-th root ψ of unity in (Z/qZ)*. For ML-DSA (q=8380417, n=256)
 * ψ=1753 has order 512 (ψ^256 ≡ -1). ML-KEM (q=3329, n=256) does NOT admit
 * a 2n-th root (max order is 256) so this kernel is ML-DSA/Goldilocks/etc.
 *
 * Algorithm: pre-twist f[i] ← f[i]·ψ^i, g[i] ← g[i]·ψ^i. Run cyclic NTT
 * convolution (this kernel reuses the fused path). Post-untwist with
 * c[i] ← c[i]·ψ^(-i)·n_inv. Reference: ntt_polymul_negacyclic.c.
 *
 * LDS layout matches G7 (4 unpadded n-element buffers). The ψ power table
 * (2n uint64_t = 4 KB at n=256) is passed as a global pointer — small
 * enough to sit in L1/L2 without explicit LDS caching.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ __launch_bounds__(1024, 2)
void fused_negacyclic_polymul_kernel(const uint64_t * __restrict__ f_in,
                                     const uint64_t * __restrict__ g_in,
                                           uint64_t * __restrict__ c_out,
                                     const uint64_t * __restrict__ tw_fwd,
                                     const uint64_t * __restrict__ tw_inv,
                                     const uint64_t * __restrict__ psi_pow,
                                     const uint64_t * __restrict__ psi_inv_pow,
                                     uint64_t n, uint64_t q, uint32_t log2_n,
                                     uint64_t n_inv)
{
    extern __shared__ uint64_t lds_raw[];
    uint64_t *fbuf0 = lds_raw;
    uint64_t *fbuf1 = lds_raw + n;
    uint64_t *gbuf0 = lds_raw + 2 * n;
    uint64_t *gbuf1 = lds_raw + 3 * n;
    uint64_t *fbufs[2] = { fbuf0, fbuf1 };
    uint64_t *gbufs[2] = { gbuf0, gbuf1 };

    uint64_t tid  = threadIdx.x;
    uint64_t half = n >> 1;

    /* Pre-twist and load. */
    if (tid < half) {
        uint64_t i0 = tid;
        uint64_t i1 = tid + half;
        uint64_t p0 = psi_pow[i0], p1 = psi_pow[i1];
        fbufs[0][i0] = dev_mulmod(f_in[i0] % q, p0, q);
        fbufs[0][i1] = dev_mulmod(f_in[i1] % q, p1, q);
        gbufs[0][i0] = dev_mulmod(g_in[i0] % q, p0, q);
        gbufs[0][i1] = dev_mulmod(g_in[i1] % q, p1, q);
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
        uint64_t out0  = j + k * p2;
        uint64_t out1  = out0 + p;
        if (tid < half) {
            uint64_t w   = tw_fwd[j * qs];
            uint64_t fu  = fbufs[src_idx][idx_u];
            uint64_t fv  = fbufs[src_idx][idx_v];
            uint64_t gu  = gbufs[src_idx][idx_u];
            uint64_t gv  = gbufs[src_idx][idx_v];
            uint64_t fwv = dev_mulmod(w, fv % q, q);
            uint64_t gwv = dev_mulmod(w, gv % q, q);
            fbufs[1 - src_idx][out0] = fu + fwv;
            fbufs[1 - src_idx][out1] = fu + q - fwv;
            gbufs[1 - src_idx][out0] = gu + gwv;
            gbufs[1 - src_idx][out1] = gu + q - gwv;
        }
        __syncthreads();
        src_idx = 1 - src_idx;
    }

    if (tid < half) {
        uint64_t f0 = fbufs[src_idx][tid]        % q;
        uint64_t f1 = fbufs[src_idx][tid + half] % q;
        uint64_t g0 = gbufs[src_idx][tid]        % q;
        uint64_t g1 = gbufs[src_idx][tid + half] % q;
        fbufs[src_idx][tid]        = dev_mulmod(f0, g0, q);
        fbufs[src_idx][tid + half] = dev_mulmod(f1, g1, q);
    }
    __syncthreads();

    for (uint32_t s = 0; s < log2_n; s++) {
        uint64_t p  = (uint64_t)1 << s;
        uint64_t qs = n >> (s + 1);
        uint64_t p2 = p << 1;
        uint64_t j  = tid & (p - 1);
        uint64_t k  = tid >> s;
        uint64_t idx_u = j + k * p;
        uint64_t idx_v = idx_u + half;
        if (tid < half) {
            uint64_t w  = tw_inv[j * qs];
            uint64_t u  = fbufs[src_idx][idx_u];
            uint64_t v  = fbufs[src_idx][idx_v];
            uint64_t wv = dev_mulmod(w, v % q, q);
            fbufs[1 - src_idx][j + k * p2]     = u + wv;
            fbufs[1 - src_idx][j + k * p2 + p] = u + q - wv;
        }
        __syncthreads();
        src_idx = 1 - src_idx;
    }

    /* Post-untwist and scale. */
    if (tid < half) {
        uint64_t a0 = fbufs[src_idx][tid]        % q;
        uint64_t a1 = fbufs[src_idx][tid + half] % q;
        uint64_t pi0 = psi_inv_pow[tid];
        uint64_t pi1 = psi_inv_pow[tid + half];
        a0 = dev_mulmod(a0, pi0, q);
        a1 = dev_mulmod(a1, pi1, q);
        c_out[tid]        = dev_mulmod(a0, n_inv, q);
        c_out[tid + half] = dev_mulmod(a1, n_inv, q);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HOST HELPERS
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
static uint64_t h_mod_inv(uint64_t a, uint64_t m) { return h_mod_pow(a, m - 2, m); }

/*
 * h_primitive_root: smallest generator g of the multiplicative group
 * (Z/qZ)* for prime q. Inputs: prime modulus q (< 2^32). Output: the
 * least g in [2, q-1] with multiplicative order exactly q-1, found by
 * testing g^((q-1)/r) != 1 for every prime factor r of q-1.
 * Invariant: q is an odd prime; the group is cyclic of order q-1.
 */
static uint64_t h_primitive_root(uint64_t q)
{
    uint64_t phi = q - 1;
    uint64_t fac[64]; int nf = 0;
    uint64_t m = phi;
    for (uint64_t d = 2; d * d <= m; d++) {
        if (m % d == 0) {
            fac[nf++] = d;
            while (m % d == 0) m /= d;
        }
    }
    if (m > 1) fac[nf++] = m;
    for (uint64_t g = 2; g < q; g++) {
        int ok = 1;
        for (int i = 0; i < nf && ok; i++)
            if (h_mod_pow(g, phi / fac[i], q) == 1) ok = 0;
        if (ok) return g;
    }
    return 0;  /* unreachable for prime q */
}

/*
 * h_neg_psi: compute a primitive 2n-th root of unity ψ for the
 * negacyclic ring Z_q[X]/(X^n+1). Inputs: transform size n (power of
 * two), prime modulus q. Output: writes ψ to *psi_out and returns 0
 * on success; returns -1 (no write) if 2n does not divide q-1 (the
 * ring then admits no 2n-th root, e.g. ML-KEM q=3329 at n=256).
 * Construction: ψ = g^((q-1)/(2n)) where g is the least primitive
 * root. Asserted invariants on success: ψ^n ≡ q-1 (≡ -1) and
 * ψ^(2n) ≡ 1, i.e. ord(ψ) = 2n exactly.
 */
static int h_neg_psi(uint64_t n, uint64_t q, uint64_t *psi_out)
{
    uint64_t two_n = 2 * n;
    if ((q - 1) % two_n != 0) return -1;        /* no 2n-th root in this ring */
    uint64_t g   = h_primitive_root(q);
    uint64_t psi = h_mod_pow(g, (q - 1) / two_n, q);
    if (h_mod_pow(psi, n, q) != q - 1) return -1;       /* ψ^n must be -1 */
    if (h_mod_pow(psi, two_n, q) != 1)  return -1;      /* ψ^(2n) must be 1 */
    *psi_out = psi;
    return 0;
}

/*
 * root_is_primitive: omega must be a primitive n-th root of unity mod q
 * (omega^n == 1 and omega^(n/2) != 1). Without this the selftest is
 * silently wrong for a CLI-supplied bad/non-primitive omega when n>512
 * (the only independent oracle, T2 vs CPU schoolbook, runs only at
 * n<=512; T1/T3/T4 all hold for ANY invertible omega).
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
    if (!p->n || (p->n & (p->n - 1))) return -1;
    if (p->q < 2 || p->q > 0xFFFFFFFFULL) return -1;
    if (!root_is_primitive(p->n, p->q, p->omega)) return -1;
    p->omega_inv = h_mod_inv(p->omega, p->q);
    p->n_inv     = h_mod_inv(p->n,     p->q);
    uint64_t t = p->n; p->log2_n = 0;
    while (t > 1) { t >>= 1; p->log2_n++; }
    return 0;
}

static uint64_t *alloc_twiddles(const ntt_params_t *p, uint64_t root)
{
    uint64_t *tw = (uint64_t *)malloc(p->n / 2 * sizeof(uint64_t));
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < p->n / 2; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k - 1] * root % p->q);
    return tw;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HOST DISPATCH — fused polymul
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * gpu_polymul_fused: c = f * g mod (X^n - 1) on device, single kernel.
 * Eligibility: n ≥ 2 and 4*n*8 ≤ sharedMemPerBlock.
 */
static int polymul_fused_eligible(uint64_t n, int max_shared_b)
{
    if (n < 2 || (n & (n - 1)) != 0 || (n >> 1) > 1024) return 0;
    size_t bytes = 4 * (size_t)n * sizeof(uint64_t);
    return (int)bytes <= max_shared_b;
}

static void gpu_polymul_fused(const uint64_t *f_h, const uint64_t *g_h,
                                    uint64_t *c_h,
                              const uint64_t *tw_fwd_h,
                              const uint64_t *tw_inv_h,
                              const ntt_params_t *p)
{
    uint64_t n = p->n, half = n >> 1;
    uint64_t *d_f, *d_g, *d_c, *d_tw, *d_twi;
    HIP_CHECK(hipMalloc((void **)&d_f,   n    * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_g,   n    * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_c,   n    * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_tw,  half * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_twi, half * sizeof(uint64_t)));
    HIP_CHECK(hipMemcpy(d_f,   f_h,      n    * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_g,   g_h,      n    * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tw,  tw_fwd_h, half * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_twi, tw_inv_h, half * sizeof(uint64_t), hipMemcpyHostToDevice));

    size_t lds_sz = 4 * (size_t)n * sizeof(uint64_t);
    int    threads = (int)half;
    hipLaunchKernelGGL(fused_polymul_kernel, dim3(1), dim3(threads), lds_sz, 0,
                       d_f, d_g, d_c, d_tw, d_twi, n, p->q, p->log2_n, p->n_inv);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(c_h, d_c, n * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_f)); HIP_CHECK(hipFree(d_g)); HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipFree(d_tw)); HIP_CHECK(hipFree(d_twi));
}

/*
 * gpu_polymul_separate: same result via 4 kernel launches (fwd f, fwd g,
 * pointwise, inv) each touching global memory. Reference implementation
 * to measure fusion speedup against.
 */
static void gpu_polymul_separate(const uint64_t *f_h, const uint64_t *g_h,
                                       uint64_t *c_h,
                                 const uint64_t *tw_fwd_h,
                                 const uint64_t *tw_inv_h,
                                 const ntt_params_t *p)
{
    uint64_t n = p->n, half = n >> 1;
    int bsz      = g_gpu.block_size;
    int gsz_half = (int)((half + bsz - 1) / bsz);
    int gsz_full = (int)((n    + bsz - 1) / bsz);
    uint64_t *d_f, *d_fb, *d_g, *d_gb, *d_tw, *d_twi;
    HIP_CHECK(hipMalloc((void **)&d_f,   n    * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_fb,  n    * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_g,   n    * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_gb,  n    * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_tw,  half * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_twi, half * sizeof(uint64_t)));
    HIP_CHECK(hipMemcpy(d_f,   f_h,      n    * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_g,   g_h,      n    * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tw,  tw_fwd_h, half * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_twi, tw_inv_h, half * sizeof(uint64_t), hipMemcpyHostToDevice));

    /* Forward NTT(f) */
    uint64_t *src = d_f, *dst = d_fb;
    for (uint32_t s = 0; s < p->log2_n; s++) {
        hipLaunchKernelGGL(pm_stockham_stage_kernel, gsz_half, bsz, 0, 0,
                           src, dst, d_tw, n, p->q, s);
        uint64_t *t = src; src = dst; dst = t;
    }
    hipLaunchKernelGGL(pm_reduce_kernel, gsz_full, bsz, 0, 0, src, n, p->q);
    uint64_t *F = src;

    /* Forward NTT(g) */
    src = d_g; dst = d_gb;
    for (uint32_t s = 0; s < p->log2_n; s++) {
        hipLaunchKernelGGL(pm_stockham_stage_kernel, gsz_half, bsz, 0, 0,
                           src, dst, d_tw, n, p->q, s);
        uint64_t *t = src; src = dst; dst = t;
    }
    hipLaunchKernelGGL(pm_reduce_kernel, gsz_full, bsz, 0, 0, src, n, p->q);
    uint64_t *G = src;

    /* Pointwise: F *= G. */
    hipLaunchKernelGGL(pm_pointwise_kernel, gsz_full, bsz, 0, 0, F, G, F,
                       n, p->q);

    /* Inverse NTT(F). */
    src = F; dst = (F == d_f) ? d_fb : d_f;
    for (uint32_t s = 0; s < p->log2_n; s++) {
        hipLaunchKernelGGL(pm_stockham_stage_kernel, gsz_half, bsz, 0, 0,
                           src, dst, d_twi, n, p->q, s);
        uint64_t *t = src; src = dst; dst = t;
    }
    hipLaunchKernelGGL(pm_reduce_kernel, gsz_full, bsz, 0, 0, src, n, p->q);
    hipLaunchKernelGGL(pm_scale_kernel, gsz_full, bsz, 0, 0, src, n, p->q, p->n_inv);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(c_h, src, n * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_f)); HIP_CHECK(hipFree(d_fb));
    HIP_CHECK(hipFree(d_g)); HIP_CHECK(hipFree(d_gb));
    HIP_CHECK(hipFree(d_tw)); HIP_CHECK(hipFree(d_twi));
}

/*
 * gpu_polymul_negacyclic: c = f * g mod (X^n + 1, q) via single fused
 * kernel. psi is a 2n-th root of unity in (Z/qZ)*.
 */
static void gpu_polymul_negacyclic(const uint64_t *f_h, const uint64_t *g_h,
                                         uint64_t *c_h,
                                   const uint64_t *tw_fwd_h,
                                   const uint64_t *tw_inv_h,
                                   uint64_t psi, const ntt_params_t *p)
{
    uint64_t n = p->n, half = n >> 1;
    uint64_t psi_inv = h_mod_inv(psi, p->q);

    uint64_t *psi_pow     = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *psi_inv_pow = (uint64_t *)malloc(n * sizeof(uint64_t));
    psi_pow[0] = psi_inv_pow[0] = 1;
    for (uint64_t i = 1; i < n; i++) {
        psi_pow[i]     = (uint64_t)((__uint128_t)psi_pow[i - 1]     * psi     % p->q);
        psi_inv_pow[i] = (uint64_t)((__uint128_t)psi_inv_pow[i - 1] * psi_inv % p->q);
    }

    uint64_t *d_f, *d_g, *d_c, *d_tw, *d_twi, *d_psi, *d_psi_inv;
    HIP_CHECK(hipMalloc((void **)&d_f,       n    * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_g,       n    * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_c,       n    * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_tw,      half * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_twi,     half * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_psi,     n    * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc((void **)&d_psi_inv, n    * sizeof(uint64_t)));
    HIP_CHECK(hipMemcpy(d_f,       f_h,         n    * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_g,       g_h,         n    * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_tw,      tw_fwd_h,    half * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_twi,     tw_inv_h,    half * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_psi,     psi_pow,     n    * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_psi_inv, psi_inv_pow, n    * sizeof(uint64_t), hipMemcpyHostToDevice));

    size_t lds_sz = 4 * (size_t)n * sizeof(uint64_t);
    int    threads = (int)half;
    hipLaunchKernelGGL(fused_negacyclic_polymul_kernel, dim3(1), dim3(threads),
                       lds_sz, 0, d_f, d_g, d_c, d_tw, d_twi, d_psi, d_psi_inv,
                       n, p->q, p->log2_n, p->n_inv);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(c_h, d_c, n * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_f)); HIP_CHECK(hipFree(d_g)); HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipFree(d_tw)); HIP_CHECK(hipFree(d_twi));
    HIP_CHECK(hipFree(d_psi)); HIP_CHECK(hipFree(d_psi_inv));
    free(psi_pow); free(psi_inv_pow);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CPU REFERENCE (schoolbook convolution — cyclic or negacyclic)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void cpu_polymul_schoolbook(const uint64_t *f, const uint64_t *g,
                                         uint64_t *c, uint64_t n, uint64_t q)
{
    for (uint64_t k = 0; k < n; k++) {
        __uint128_t acc = 0;
        for (uint64_t i = 0; i < n; i++) {
            uint64_t j = (k - i + n) % n;          /* cyclic */
            acc += (__uint128_t)f[i] * g[j];
        }
        c[k] = (uint64_t)(acc % q);
    }
}

/*
 * Schoolbook mod X^n + 1: c[k] = sum_{i+j=k} f[i]g[j] - sum_{i+j=k+n} f[i]g[j].
 * Equivalent to the wrap-and-negate identity for the anti-cyclic ring.
 */
static void cpu_polymul_negacyclic(const uint64_t *f, const uint64_t *g,
                                         uint64_t *c, uint64_t n, uint64_t q)
{
    for (uint64_t k = 0; k < n; k++) {
        __int128 acc = 0;
        for (uint64_t i = 0; i <= k; i++) {
            acc += (__int128)((__uint128_t)f[i] * g[k - i] % q);
        }
        for (uint64_t i = k + 1; i < n; i++) {
            acc -= (__int128)((__uint128_t)f[i] * g[n + k - i] % q);
        }
        acc %= (__int128)q;
        if (acc < 0) acc += q;
        c[k] = (uint64_t)acc;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SELFTEST
 * ═══════════════════════════════════════════════════════════════════════════ */
static int selftest(const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    const char *lbl = (q == 3329) ? "ML-KEM" : (q == 8380417) ? "ML-DSA" : "custom";
    printf(ANSI_CYN "  ── Selftest: %s (n=%lu q=%lu ω=%lu) ──────────────\n"
           ANSI_RST, lbl, n, q, p->omega);

    uint64_t *tw  = alloc_twiddles(p, p->omega);
    uint64_t *twi = alloc_twiddles(p, p->omega_inv);
    uint64_t *f   = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *g   = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *c_f = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *c_s = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *c_r = (uint64_t *)malloc(n * sizeof(uint64_t));

    for (uint64_t i = 0; i < n; i++) {
        f[i] = (i * 2654435761ULL + 1) % q;
        g[i] = (i * 6364136223846793005ULL + 3) % q;
    }

    int fused_ok = polymul_fused_eligible(n, g_gpu.max_shared_b);
    int t1_ok = 1, t2_ok = 1, t3_ok = 1, t4_ok = 1;

    if (fused_ok) {
        /* T1: f * delta(X^0 = 1) = f */
        uint64_t *g1 = (uint64_t *)calloc(n, sizeof(uint64_t));
        g1[0] = 1;
        gpu_polymul_fused(f, g1, c_f, tw, twi, p);
        t1_ok = !memcmp(c_f, f, n * sizeof(uint64_t));
        printf("  %-38s %s\n", "T1: fused f*1 == f",
               t1_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);
        free(g1);

        /* T2: fused matches schoolbook (only practical at small n) */
        if (n <= 512) {
            cpu_polymul_schoolbook(f, g, c_r, n, q);
            gpu_polymul_fused(f, g, c_f, tw, twi, p);
            t2_ok = !memcmp(c_f, c_r, n * sizeof(uint64_t));
            printf("  %-38s %s\n", "T2: fused == CPU schoolbook",
                   t2_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);
        }

        /* T3: fused == separate */
        gpu_polymul_separate(f, g, c_s, tw, twi, p);
        gpu_polymul_fused(f, g, c_f, tw, twi, p);
        t3_ok = !memcmp(c_f, c_s, n * sizeof(uint64_t));
        printf("  %-38s %s\n", "T3: fused == separate (4-kernel)",
               t3_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

        /* T4: commutativity */
        gpu_polymul_fused(g, f, c_s, tw, twi, p);
        t4_ok = !memcmp(c_f, c_s, n * sizeof(uint64_t));
        printf("  %-38s %s\n", "T4: f*g == g*f",
               t4_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);
    } else {
        /* E1: n=4096 path — fused kernel exceeds 64KB LDS, use 4-kernel
         * separate path only. Verify against schoolbook only at small n. */
        gpu_polymul_separate(f, g, c_s, tw, twi, p);
        printf("  %-38s %s (n=%lu > fused LDS cap)\n",
               "T-E1: separate path (n>=4096)",
               ANSI_GRN "RAN" ANSI_RST, n);
        /* Commutativity check stays valid for the separate path. */
        gpu_polymul_separate(g, f, c_f, tw, twi, p);
        t4_ok = !memcmp(c_f, c_s, n * sizeof(uint64_t));
        printf("  %-38s %s\n", "T-E2: separate f*g == g*f",
               t4_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);
    }

    free(tw); free(twi); free(f); free(g); free(c_f); free(c_s); free(c_r);
    int ok = t1_ok && t2_ok && t3_ok && t4_ok;
    printf("  %s\n\n", ok ? ANSI_GRN "ALL PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);
    return ok ? 0 : -1;
}

/*
 * selftest_negacyclic: verify the twisted-NTT path for primes admitting a
 * 2n-th root. ML-DSA q=8380417 uses ψ=1753 (order 512 mod q). Forward root
 * ω = ψ² = 3073009 (order 256). ML-KEM q=3329 has no 2n-th root at n=256
 * so we skip it.
 */
static int selftest_negacyclic(uint64_t n, uint64_t q, uint64_t psi,
                               const char *label)
{
    ntt_params_t np = { n, q, 0, 0, 0, 0 };
    /* Use ω = ψ² as the cyclic forward root. */
    np.omega = (uint64_t)((__uint128_t)psi * psi % q);
    if (ntt_params_init(&np) != 0) return -1;

    printf(ANSI_CYN "  ── Negacyclic selftest: %s (n=%lu q=%lu ψ=%lu) ─\n"
           ANSI_RST, label, n, q, psi);

    uint64_t *tw  = alloc_twiddles(&np, np.omega);
    uint64_t *twi = alloc_twiddles(&np, np.omega_inv);
    uint64_t *f   = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *g   = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *c_g = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *c_r = (uint64_t *)malloc(n * sizeof(uint64_t));

    for (uint64_t i = 0; i < n; i++) {
        f[i] = (i * 2654435761ULL + 1) % q;
        g[i] = (i * 6364136223846793005ULL + 3) % q;
    }

    /* T1: f * 1 = f (in X^n+1 ring, 1 is still the multiplicative identity). */
    uint64_t *g1 = (uint64_t *)calloc(n, sizeof(uint64_t));
    g1[0] = 1;
    gpu_polymul_negacyclic(f, g1, c_g, tw, twi, psi, &np);
    int t1_ok = !memcmp(c_g, f, n * sizeof(uint64_t));
    printf("  %-38s %s\n", "T1: fused f*1 == f",
           t1_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);
    free(g1);

    /* T2: vs CPU schoolbook negacyclic. */
    cpu_polymul_negacyclic(f, g, c_r, n, q);
    gpu_polymul_negacyclic(f, g, c_g, tw, twi, psi, &np);
    int t2_ok = !memcmp(c_g, c_r, n * sizeof(uint64_t));
    printf("  %-38s %s\n", "T2: fused == CPU negacyclic schoolbook",
           t2_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);

    /* T3: X^(n-1) * X = -1  (ring identity in X^n+1). */
    uint64_t *ea = (uint64_t *)calloc(n, sizeof(uint64_t));
    uint64_t *eb = (uint64_t *)calloc(n, sizeof(uint64_t));
    ea[n - 1] = 1; eb[1] = 1;
    gpu_polymul_negacyclic(ea, eb, c_g, tw, twi, psi, &np);
    int t3_ok = (c_g[0] == q - 1);
    for (uint64_t i = 1; i < n && t3_ok; i++) if (c_g[i] != 0) t3_ok = 0;
    printf("  %-38s %s\n", "T3: X^(n-1)*X ≡ -1 mod (X^n+1)",
           t3_ok ? ANSI_GRN "PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);
    free(ea); free(eb);

    free(tw); free(twi); free(f); free(g); free(c_g); free(c_r);
    int ok = t1_ok && t2_ok && t3_ok;
    printf("  %s\n\n", ok ? ANSI_GRN "ALL PASS" ANSI_RST : ANSI_RED "FAIL" ANSI_RST);
    return ok ? 0 : -1;
}

/*
 * selftest_negacyclic_sweep: multi-size / multi-modulus regression for the
 * twisted-NTT (mod X^n+1) fused kernel. Inputs: none (table is internal).
 * Output: 0 if every admissible (n,q) cell PASSed, -1 otherwise; also
 * appends a one-line summary to negac_sweep_<ts>.txt (project rule 5).
 *
 * Grid: n ∈ {256,512,1024,2048} × q ∈ a curated set of primes < 2^32 that
 * (for the admissible cells) possess a primitive 2n-th root of unity. ψ is
 * computed at RUNTIME by h_neg_psi (no hardcoded root); cells where the
 * ring admits no 2n-th root are SKIPPED and reported, not failed. Each
 * admissible cell runs the full selftest_negacyclic battery (f*1==f,
 * differential vs the independent CPU schoolbook conv mod X^n+1, and the
 * X^(n-1)*X ≡ -1 ring identity).
 *
 * S7 / watchdog / LDS cap: the fused negacyclic kernel holds 4 unpadded
 * n-element uint64_t LDS buffers (4*n*8 B); at n=2048 that is exactly 64 KB
 * — a full 6900XT (gfx1030) LDS block — and blockDim.x = n/2 = 1024 is the
 * launch ceiling. n is therefore capped at 2048 here: cells are gated by
 * polymul_fused_eligible() AND an explicit n>2048 reject so the sweep can
 * never exceed the LDS block or the 2 s gfx-ring watchdog envelope.
 */
static int selftest_negacyclic_sweep(void)
{
    static const uint64_t NS[] = { 256, 512, 1024, 2048 };
    /* Curated primes < 2^32. Each label notes the 2-adic valuation of q-1,
     * which bounds the largest 2n-th root the ring can hold. */
    static const struct { uint64_t q; const char *name; } QS[] = {
        {    8380417, "ML-DSA"   },  /* q-1 = 2^13·3·11·31         */
        {  998244353, "NTT-998"  },  /* q-1 = 2^23·7·17            */
        { 2013265921, "Babybear" },  /* q-1 = 2^27·3·5             */
        {  167772161, "NTT-167"  },  /* q-1 = 2^25·5               */
    };
    const int NN = (int)(sizeof NS / sizeof NS[0]);
    const int NQ = (int)(sizeof QS / sizeof QS[0]);

    printf(ANSI_CYN "  ── Negacyclic sweep (mod X^n+1, ψ@runtime) ───────\n"
           ANSI_RST);
    printf("  ┌────────┬────────────┬────────────┬──────────────┬────────┐\n");
    printf("  │ %-6s │ %-10s │ %-10s │ %-12s │ %-6s │\n",
           "n", "q", "label", "ψ (2n-root)", "result");
    printf("  ├────────┼────────────┼────────────┼──────────────┼────────┤\n");

    int passed = 0, total = 0, skipped = 0, failed = 0;
    for (int qi = 0; qi < NQ; qi++) {
        for (int ni = 0; ni < NN; ni++) {
            uint64_t n = NS[ni], q = QS[qi].q;
            /* LDS / watchdog cap: hard reject n>2048, then defer to the
             * shared eligibility predicate (4*n*8 ≤ sharedMemPerBlock). */
            if (n > 2048 || !polymul_fused_eligible(n, g_gpu.max_shared_b)) {
                printf("  │ %6lu │ %10lu │ %-10s │ %-12s │ %sSKIP%s │\n",
                       n, q, QS[qi].name, "LDS-cap",
                       ANSI_CYN, ANSI_RST);
                skipped++;
                continue;
            }
            uint64_t psi = 0;
            if (h_neg_psi(n, q, &psi) != 0) {
                printf("  │ %6lu │ %10lu │ %-10s │ %-12s │ %sSKIP%s │\n",
                       n, q, QS[qi].name, "no 2n-root",
                       ANSI_CYN, ANSI_RST);
                skipped++;
                continue;
            }
            total++;
            int rc = selftest_negacyclic(n, q, psi, QS[qi].name);
            int cell_ok = (rc == 0);
            if (cell_ok) passed++; else failed++;
            printf("  │ %6lu │ %10lu │ %-10s │ %12lu │ %s %s%s │\n",
                   n, q, QS[qi].name, psi,
                   cell_ok ? ANSI_GRN : ANSI_RED,
                   cell_ok ? "PASS" : "FAIL", ANSI_RST);
        }
    }
    printf("  └────────┴────────────┴────────────┴──────────────┴────────┘\n");
    int all_ok = (failed == 0);
    printf("  %s%d/%d passed%s  (%d skipped)\n\n",
           all_ok ? ANSI_GRN : ANSI_RED, passed, total, ANSI_RST, skipped);

    /* Persist a one-line summary to a timestamped file (project rule 5). */
    char ts[32], path[64];
    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(ts, sizeof ts, "%Y%m%d_%H%M%S", &tmv);
    snprintf(path, sizeof path, "negac_sweep_%s.txt", ts);
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp,
                "negacyclic_sweep %s  passed=%d total=%d skipped=%d "
                "result=%s  grid=n{256,512,1024,2048}x"
                "q{8380417,998244353,2013265921,167772161} "
                "psi=runtime cap=n<=2048\n",
                ts, passed, total, skipped, all_ok ? "PASS" : "FAIL");
        fclose(fp);
    }
    return all_ok ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK
 * ═══════════════════════════════════════════════════════════════════════════ */
static double bench_fused(const ntt_params_t *p, const uint64_t *tw,
                          const uint64_t *twi, uint64_t iters)
{
    uint64_t n = p->n, half = n >> 1;
    uint64_t *d_f, *d_g, *d_c, *d_tw, *d_twi;
    (void)hipMalloc((void **)&d_f,   n    * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_g,   n    * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_c,   n    * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_tw,  half * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_twi, half * sizeof(uint64_t));
    (void)hipMemcpy(d_tw,  tw,  half * sizeof(uint64_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_twi, twi, half * sizeof(uint64_t), hipMemcpyHostToDevice);

    size_t lds_sz = 4 * (size_t)n * sizeof(uint64_t);
    int    threads = (int)half;
    hipLaunchKernelGGL(fused_polymul_kernel, dim3(1), dim3(threads), lds_sz, 0,
                       d_f, d_g, d_c, d_tw, d_twi, n, p->q, p->log2_n, p->n_inv);
    (void)hipDeviceSynchronize();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) {
        hipLaunchKernelGGL(fused_polymul_kernel, dim3(1), dim3(threads),
                           lds_sz, 0, d_f, d_g, d_c, d_tw, d_twi,
                           n, p->q, p->log2_n, p->n_inv);
    }
    (void)hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    (void)hipFree(d_f); (void)hipFree(d_g); (void)hipFree(d_c);
    (void)hipFree(d_tw); (void)hipFree(d_twi);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static double bench_separate(const ntt_params_t *p, const uint64_t *tw,
                             const uint64_t *twi, uint64_t iters)
{
    uint64_t n = p->n, half = n >> 1;
    int bsz      = g_gpu.block_size;
    int gsz_half = (int)((half + bsz - 1) / bsz);
    int gsz_full = (int)((n    + bsz - 1) / bsz);

    uint64_t *d_f, *d_fb, *d_g, *d_gb, *d_tw, *d_twi;
    (void)hipMalloc((void **)&d_f,   n    * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_fb,  n    * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_g,   n    * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_gb,  n    * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_tw,  half * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_twi, half * sizeof(uint64_t));
    (void)hipMemcpy(d_tw,  tw,  half * sizeof(uint64_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_twi, twi, half * sizeof(uint64_t), hipMemcpyHostToDevice);

    /* warmup */
    {
        uint64_t *src = d_f, *dst = d_fb;
        for (uint32_t s = 0; s < p->log2_n; s++) {
            hipLaunchKernelGGL(pm_stockham_stage_kernel, gsz_half, bsz, 0, 0,
                               src, dst, d_tw, n, p->q, s);
            uint64_t *t = src; src = dst; dst = t;
        }
    }
    (void)hipDeviceSynchronize();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) {
        uint64_t *src = d_f, *dst = d_fb;
        for (uint32_t s = 0; s < p->log2_n; s++) {
            hipLaunchKernelGGL(pm_stockham_stage_kernel, gsz_half, bsz, 0, 0,
                               src, dst, d_tw, n, p->q, s);
            uint64_t *t = src; src = dst; dst = t;
        }
        hipLaunchKernelGGL(pm_reduce_kernel, gsz_full, bsz, 0, 0, src, n, p->q);
        uint64_t *F = src;

        src = d_g; dst = d_gb;
        for (uint32_t s = 0; s < p->log2_n; s++) {
            hipLaunchKernelGGL(pm_stockham_stage_kernel, gsz_half, bsz, 0, 0,
                               src, dst, d_tw, n, p->q, s);
            uint64_t *t = src; src = dst; dst = t;
        }
        hipLaunchKernelGGL(pm_reduce_kernel, gsz_full, bsz, 0, 0, src, n, p->q);
        uint64_t *G = src;

        hipLaunchKernelGGL(pm_pointwise_kernel, gsz_full, bsz, 0, 0, F, G, F,
                           n, p->q);

        src = F; dst = (F == d_f) ? d_fb : d_f;
        for (uint32_t s = 0; s < p->log2_n; s++) {
            hipLaunchKernelGGL(pm_stockham_stage_kernel, gsz_half, bsz, 0, 0,
                               src, dst, d_twi, n, p->q, s);
            uint64_t *t = src; src = dst; dst = t;
        }
        hipLaunchKernelGGL(pm_reduce_kernel, gsz_full, bsz, 0, 0, src, n, p->q);
        hipLaunchKernelGGL(pm_scale_kernel, gsz_full, bsz, 0, 0, src, n, p->q, p->n_inv);
    }
    (void)hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    (void)hipFree(d_f); (void)hipFree(d_fb); (void)hipFree(d_g);
    (void)hipFree(d_gb); (void)hipFree(d_tw); (void)hipFree(d_twi);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* D2: batched fused polymul — grid = batch, one block per polymul. */
static double bench_batched_polymul(const ntt_params_t *p, const uint64_t *tw,
                                    const uint64_t *twi, uint64_t iters,
                                    uint64_t batch)
{
    uint64_t n = p->n, half = n >> 1;
    uint64_t *d_f, *d_g, *d_c, *d_tw, *d_twi;
    (void)hipMalloc((void **)&d_f,  batch * n * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_g,  batch * n * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_c,  batch * n * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_tw,  half  * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_twi, half  * sizeof(uint64_t));
    (void)hipMemcpy(d_tw,  tw,  half * sizeof(uint64_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_twi, twi, half * sizeof(uint64_t), hipMemcpyHostToDevice);

    size_t lds_sz  = 4 * (size_t)n * sizeof(uint64_t);
    int    threads = (int)half;
    /* warmup */
    hipLaunchKernelGGL(fused_polymul_batched_kernel, dim3((uint32_t)batch),
                       dim3(threads), lds_sz, 0,
                       d_f, d_g, d_c, d_tw, d_twi, n, p->q, p->log2_n, p->n_inv);
    (void)hipDeviceSynchronize();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) {
        hipLaunchKernelGGL(fused_polymul_batched_kernel, dim3((uint32_t)batch),
                           dim3(threads), lds_sz, 0,
                           d_f, d_g, d_c, d_tw, d_twi,
                           n, p->q, p->log2_n, p->n_inv);
    }
    (void)hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    (void)hipFree(d_f); (void)hipFree(d_g); (void)hipFree(d_c);
    (void)hipFree(d_tw); (void)hipFree(d_twi);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static double bench_negacyclic(const ntt_params_t *p, const uint64_t *tw,
                               const uint64_t *twi, uint64_t psi,
                               uint64_t iters)
{
    uint64_t n = p->n, half = n >> 1;
    uint64_t psi_inv = h_mod_inv(psi, p->q);

    uint64_t *psi_pow     = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *psi_inv_pow = (uint64_t *)malloc(n * sizeof(uint64_t));
    psi_pow[0] = psi_inv_pow[0] = 1;
    for (uint64_t i = 1; i < n; i++) {
        psi_pow[i]     = (uint64_t)((__uint128_t)psi_pow[i - 1]     * psi     % p->q);
        psi_inv_pow[i] = (uint64_t)((__uint128_t)psi_inv_pow[i - 1] * psi_inv % p->q);
    }

    uint64_t *d_f, *d_g, *d_c, *d_tw, *d_twi, *d_psi, *d_psi_inv;
    (void)hipMalloc((void **)&d_f,       n    * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_g,       n    * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_c,       n    * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_tw,      half * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_twi,     half * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_psi,     n    * sizeof(uint64_t));
    (void)hipMalloc((void **)&d_psi_inv, n    * sizeof(uint64_t));
    (void)hipMemcpy(d_tw,      tw,          half * sizeof(uint64_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_twi,     twi,         half * sizeof(uint64_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_psi,     psi_pow,     n    * sizeof(uint64_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_psi_inv, psi_inv_pow, n    * sizeof(uint64_t), hipMemcpyHostToDevice);

    size_t lds_sz = 4 * (size_t)n * sizeof(uint64_t);
    int    threads = (int)half;
    hipLaunchKernelGGL(fused_negacyclic_polymul_kernel, dim3(1), dim3(threads),
                       lds_sz, 0, d_f, d_g, d_c, d_tw, d_twi, d_psi, d_psi_inv,
                       n, p->q, p->log2_n, p->n_inv);
    (void)hipDeviceSynchronize();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t it = 0; it < iters; it++) {
        hipLaunchKernelGGL(fused_negacyclic_polymul_kernel, dim3(1),
                           dim3(threads), lds_sz, 0, d_f, d_g, d_c, d_tw,
                           d_twi, d_psi, d_psi_inv, n, p->q, p->log2_n,
                           p->n_inv);
    }
    (void)hipDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    (void)hipFree(d_f); (void)hipFree(d_g); (void)hipFree(d_c);
    (void)hipFree(d_tw); (void)hipFree(d_twi);
    (void)hipFree(d_psi); (void)hipFree(d_psi_inv);
    free(psi_pow); free(psi_inv_pow);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static void run_benchmark(const ntt_params_t *p, uint64_t iters)
{
    uint64_t n = p->n;
    uint64_t *tw  = alloc_twiddles(p, p->omega);
    uint64_t *twi = alloc_twiddles(p, p->omega_inv);

    int fused_ok = polymul_fused_eligible(n, g_gpu.max_shared_b);

    double t_sep   = bench_separate(p, tw, twi, iters);
    double t_fused = fused_ok ? bench_fused(p, tw, twi, iters) : 0.0;
    double s_sep   = iters / t_sep;
    double s_fused = fused_ok ? (iters / t_fused) : 0.0;

    /* D2: batched polymul at batch = CU count (fused-kernel path). */
    double s_bat = 0.0; uint64_t bat = 0;
    if (fused_ok) {
        bat = (uint64_t)g_gpu.cu_count;
        if (bat < 8) bat = 8;
        uint64_t bat_it = iters / bat; if (bat_it < 1) bat_it = 1;
        double t_bat = bench_batched_polymul(p, tw, twi, bat_it, bat);
        s_bat = (double)bat_it * bat / t_bat;
    }

    printf(ANSI_CYN "  ── Polymul benchmark (n=%lu q=%lu) ──────────────\n"
           ANSI_RST, n, p->q);
    printf("  ┌──────────────────────────┬──────────────┬──────────────┐\n");
    printf("  │ %-24s │ %-12s │ %-12s │\n", "variant", "polymul/s", "µs/call");
    printf("  ├──────────────────────────┼──────────────┼──────────────┤\n");
    printf("  │ %-24s │ %12.0f │ %12.2f │\n",
           "separate (4 launches)", s_sep,   1e6 / s_sep);
    if (fused_ok) {
        printf("  │ %-24s │ %12.0f │ %12.2f │\n",
               "fused (1 launch)",      s_fused, 1e6 / s_fused);
        printf("  │ %-24s │ %12.2fx │ %12s │\n",
               "speedup (fused/sep)",   s_fused / s_sep, "");
        char label[64];
        snprintf(label, sizeof label, "batched b=%lu (D2)", bat);
        printf("  │ %-24s │ %12.0f │ %12.2f │\n", label, s_bat, 1e6 / s_bat);
        printf("  │ %-24s │ %12.2fx │ %12s │\n",
               "speedup (bat/fused)", s_bat / s_fused, "");
    } else {
        printf("  │ %-24s │ %12s │ %12s │\n",
               "fused/batched (E1)", "LDS-limit", "n>=4096 skip");
    }

    /* ML-DSA q=8380417 admits ψ=1753 as a 2n-th root at n=256. */
    if (p->q == 8380417 && n == 256) {
        ntt_params_t np = *p;
        np.omega = 3073009;  /* ψ² = 1753² mod 8380417 */
        ntt_params_init(&np);
        uint64_t *twn  = alloc_twiddles(&np, np.omega);
        uint64_t *twni = alloc_twiddles(&np, np.omega_inv);
        double t_neg   = bench_negacyclic(&np, twn, twni, 1753, iters);
        double s_neg   = iters / t_neg;
        printf("  │ %-24s │ %12.0f │ %12.2f │\n",
               "fused negacyclic (G7n)", s_neg, 1e6 / s_neg);
        printf("  │ %-24s │ %12.2fx │ %12s │\n",
               "speedup (neg/fused)",    s_neg / s_fused, "");
        free(twn); free(twni);
    }
    printf("  └──────────────────────────┴──────────────┴──────────────┘\n\n");

    free(tw); free(twi);
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

    /* --bench-only (any later argv): skip the hardcoded selftest +
     * negacyclic sweep block; run only run_benchmark(). Added 2026-05-23
     * for rocprofv3 PMC capture (INCIDENT.md §7): the selftest+sweep
     * does many small diverse-parameter kernel launches in succession
     * which, under rocprofv3 instrumentation on RDNA2, can hard-hang
     * the GPU. Bench-only mode launches a bounded, homogeneous set of
     * (n,q,omega,iters)-controlled kernels — safe under PMC. */
    int bench_only = 0;
    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--bench-only") == 0) { bench_only = 1; break; }
    }

    printf("\n" ANSI_WHT
           "╔══════════════════════════════════════════════════════════╗\n"
           "║   NTT / MI300A  —  GPU Fused Polymul (G7)               ║\n"
           "╚══════════════════════════════════════════════════════════╝\n"
           ANSI_RST "\n");

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    strncpy(g_gpu.name, prop.name, sizeof g_gpu.name - 1);
    g_gpu.warp_size    = prop.warpSize;
    g_gpu.cu_count     = prop.multiProcessorCount;
    g_gpu.max_shared_b = (int)prop.sharedMemPerBlock;
    g_gpu.total_mem    = prop.totalGlobalMem;
    g_gpu.block_size   = 128;

    printf(ANSI_CYN "  ── GPU ──────────────────────────────────────────\n" ANSI_RST);
    printf("  %s  warp=%d  CU=%d  LDS=%d B\n\n",
           g_gpu.name, g_gpu.warp_size, g_gpu.cu_count, g_gpu.max_shared_b);

    if (ntt_params_init(&p) != 0) {
        fprintf(stderr, "bad n/q\n"); return 1;
    }
    if (!polymul_fused_eligible(p.n, g_gpu.max_shared_b)) {
        printf("  note: n=%lu exceeds fused LDS cap (need %zu > %d B); "
               "running separate-path selftest/benchmark only (E1 hybrid).\n\n",
               p.n, 4 * (size_t)p.n * sizeof(uint64_t), g_gpu.max_shared_b);
    }

    if (!bench_only) {
        /* Default regression: ML-KEM at n=256. */
        ntt_params_t mlkem = { 256, 3329, 17, 0, 0, 0 };
        int fail = 0;
        if (ntt_params_init(&mlkem) == 0) fail |= selftest(&mlkem);
        if (p.n != 256 || p.q != 3329) fail |= selftest(&p);
        /* Negacyclic regression: ML-DSA admits ψ=1753 as 2n-th root at n=256.
         * Kept as a fixed pinned point; the sweep below generalises it. */
        fail |= selftest_negacyclic(256, 8380417, 1753, "ML-DSA");
        /* Multi-size / multi-modulus negacyclic sweep (n≤2048 LDS cap). */
        fail |= selftest_negacyclic_sweep();
        if (fail) return 1;
    } else {
        printf("  --bench-only: skipping selftests + negacyclic sweep\n\n");
    }

    if (iters > 1) run_benchmark(&p, iters);
    return 0;
}
