/*
 * ntt_kernel_cswitch.hip — pure-C runtime-switch variant of ntt_kernel.hip
 *
 * G9 PREP (BACKLOG §3): this is the "C++ in kernels" deviation A/B sibling
 * of lib2/ntt_kernel.hip. That file is pervasively `template<int PIDX>`:
 * the prime / Solinas reduction is baked per-prime at COMPILE time, so four
 * kernel instantiations are emitted with the modulus as an ISA immediate and
 * no runtime branch in the butterfly.
 *
 * This file removes every C++ template. Each per-prime arithmetic primitive
 * (shoup_mul / addmod / submod / mulmod / montgomery_mul) is provided as a
 * single `__forceinline__ __device__` function that takes a runtime
 * `int pidx` and dispatches with `switch (pidx)` to a `_pN`-suffixed body
 * that hardcodes prime N's constant. The constant is therefore STILL a
 * literal inside each switch arm — the question G9 must answer from the
 * emitted ISA is whether the compiler keeps it as an immediate (and keeps
 * the four arms branch-free / specialised) once dispatch is a runtime switch
 * instead of a template instantiation.
 *
 * The public launch_* signatures are call-compatible with the templated
 * file: wherever ntt_kernel.hip had `<PIDX>` this file takes a leading
 * runtime `int pidx` argument. C-style per .claude/rules/c-style.md:
 * .hip = C++ frontend, so `__restrict__` (not `restrict`); no STL, no
 * classes, no new/delete, no templates.
 *
 * Structure / LDS / twiddle layout are byte-identical to ntt_kernel.hip;
 * see that file's header for the algorithm description. This translation
 * unit only needs to COMPILE and emit ISA for the G9 diff — it is not
 * linked into a runnable binary here.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>   /* abort() for launcher pre-condition checks */
#include "primes.h"
#include "shoup.h"
#include "crt_ntt.h"

/* ─── LDS constants (identical to ntt_kernel.hip) ────────────────────────── */
#define BLOCK_SIZE  512             /* 8 wave64 (gfx942) / 16 wave32 (gfx1030) */
#define LDS_STRIDE  33              /* 32 elements + 1 padding (32 LDS banks)  */
#define LDS_TOTAL   (32 * LDS_STRIDE) /* = 1056 uint64_t = 8448 bytes/block  */

#if defined(GFX1030_LOCAL)
#  define NTT_LAUNCH_BOUNDS __launch_bounds__(512, 2)  /* RDNA2 dev path     */
#else
#  define NTT_LAUNCH_BOUNDS __launch_bounds__(512, 2)  /* CDNA3 native; tune */
#endif

__device__ __forceinline__
int lds_idx(int i) { return (i >> 5) * LDS_STRIDE + (i & 31); }

/* ─── Per-prime arithmetic, runtime switch (replaces template<int PIDX>) ──── */
/* The bodies below are the exact arithmetic of shoup.h / primes.h with the
 * prime constant hardcoded as a literal (P0..P3) instead of PRIMES[PIDX].
 * Each public helper dispatches `switch (pidx)` to the matching `_pN` body.
 * `__forceinline__` keeps the inlined arithmetic in the butterfly so the
 * only structural difference vs the templated build is the dispatch form. */

/* shoup_mul: (w * t) mod pN, w_inv = floor(w*2^64/pN). See shoup.h. */
#define SHOUP_BODY(P)                                                         \
    uint64_t q = (uint64_t)(((__uint128_t)t * w_inv) >> 64);                  \
    __uint128_t r128 = (__uint128_t)w * t - (__uint128_t)q * (P);             \
    uint64_t r = (uint64_t)r128;                                              \
    if ((uint64_t)(r128 >> 64) || r >= (P)) r -= (P);                         \
    return r;

__device__ __forceinline__
uint64_t cs_shoup_mul_p0(uint64_t t, uint64_t w, uint64_t w_inv) { SHOUP_BODY(P0) }
__device__ __forceinline__
uint64_t cs_shoup_mul_p1(uint64_t t, uint64_t w, uint64_t w_inv) { SHOUP_BODY(P1) }
__device__ __forceinline__
uint64_t cs_shoup_mul_p2(uint64_t t, uint64_t w, uint64_t w_inv) { SHOUP_BODY(P2) }
__device__ __forceinline__
uint64_t cs_shoup_mul_p3(uint64_t t, uint64_t w, uint64_t w_inv) { SHOUP_BODY(P3) }

__device__ __forceinline__
uint64_t cs_shoup_mul(int pidx, uint64_t t, uint64_t w, uint64_t w_inv)
{
    switch (pidx) {
        case 0:  return cs_shoup_mul_p0(t, w, w_inv);
        case 1:  return cs_shoup_mul_p1(t, w, w_inv);
        case 2:  return cs_shoup_mul_p2(t, w, w_inv);
        default: return cs_shoup_mul_p3(t, w, w_inv);
    }
}

/* addmod: a,b in [0,pN) → (a+b) mod pN. See primes.h addmod<PIDX>. */
#define ADDMOD_BODY(P)                                                        \
    uint64_t r = a + b;                                                       \
    if (r < a) r -= (P);                                                      \
    if (r >= (P)) r -= (P);                                                   \
    return r;

__device__ __forceinline__
uint64_t cs_addmod_p0(uint64_t a, uint64_t b) { ADDMOD_BODY(P0) }
__device__ __forceinline__
uint64_t cs_addmod_p1(uint64_t a, uint64_t b) { ADDMOD_BODY(P1) }
__device__ __forceinline__
uint64_t cs_addmod_p2(uint64_t a, uint64_t b) { ADDMOD_BODY(P2) }
__device__ __forceinline__
uint64_t cs_addmod_p3(uint64_t a, uint64_t b) { ADDMOD_BODY(P3) }

__device__ __forceinline__
uint64_t cs_addmod(int pidx, uint64_t a, uint64_t b)
{
    switch (pidx) {
        case 0:  return cs_addmod_p0(a, b);
        case 1:  return cs_addmod_p1(a, b);
        case 2:  return cs_addmod_p2(a, b);
        default: return cs_addmod_p3(a, b);
    }
}

/* submod: a,b in [0,pN) → (a-b) mod pN. See primes.h submod<PIDX>. */
#define SUBMOD_BODY(P)  return a >= b ? a - b : a + (P) - b;

__device__ __forceinline__
uint64_t cs_submod_p0(uint64_t a, uint64_t b) { SUBMOD_BODY(P0) }
__device__ __forceinline__
uint64_t cs_submod_p1(uint64_t a, uint64_t b) { SUBMOD_BODY(P1) }
__device__ __forceinline__
uint64_t cs_submod_p2(uint64_t a, uint64_t b) { SUBMOD_BODY(P2) }
__device__ __forceinline__
uint64_t cs_submod_p3(uint64_t a, uint64_t b) { SUBMOD_BODY(P3) }

__device__ __forceinline__
uint64_t cs_submod(int pidx, uint64_t a, uint64_t b)
{
    switch (pidx) {
        case 0:  return cs_submod_p0(a, b);
        case 1:  return cs_submod_p1(a, b);
        case 2:  return cs_submod_p2(a, b);
        default: return cs_submod_p3(a, b);
    }
}

/* reduce_pN are pure-C already (primes.h, no template). Reuse directly. */
__device__ __forceinline__
uint64_t cs_reduce(int pidx, __uint128_t x)
{
    switch (pidx) {
        case 0:  return reduce_p0(x);
        case 1:  return reduce_p1(x);
        case 2:  return reduce_p2(x);
        default: return reduce_p3(x);
    }
}

/* mulmod: full modular multiply via reduction. See primes.h mulmod<PIDX>. */
__device__ __forceinline__
uint64_t cs_mulmod(int pidx, uint64_t a, uint64_t b)
{
    return cs_reduce(pidx, (__uint128_t)a * b);
}

/* montgomery_mul: REDC(a*b)=a*b*R^{-1} mod pN, R=2^64. See primes.h. */
#define MONT_BODY(P, PINV)                                                    \
    const uint64_t p = (P);                                                   \
    __uint128_t t   = (__uint128_t)a * b;                                     \
    uint64_t    tlo = (uint64_t)t;                                            \
    uint64_t    thi = (uint64_t)(t >> 64);                                    \
    uint64_t    m   = tlo * (PINV);                                           \
    __uint128_t mp  = (__uint128_t)m * p;                                     \
    uint64_t    mplo = (uint64_t)mp;                                          \
    uint64_t    mphi = (uint64_t)(mp >> 64);                                  \
    uint64_t carry = (uint64_t)(((__uint128_t)tlo + mplo) >> 64);             \
    __uint128_t u  = (__uint128_t)thi + mphi + carry;                         \
    if (u >= p) u -= p;                                                       \
    return (uint64_t)u;

__device__ __forceinline__
uint64_t cs_mont_mul_p0(uint64_t a, uint64_t b)
{ MONT_BODY(P0, UINT64_C(0xfffffffeffffffff)) }
__device__ __forceinline__
uint64_t cs_mont_mul_p1(uint64_t a, uint64_t b)
{ MONT_BODY(P1, UINT64_C(0xfffefffffeffffff)) }
__device__ __forceinline__
uint64_t cs_mont_mul_p2(uint64_t a, uint64_t b)
{ MONT_BODY(P2, UINT64_C(0xfffffffbffffffff)) }
__device__ __forceinline__
uint64_t cs_mont_mul_p3(uint64_t a, uint64_t b)
{ MONT_BODY(P3, UINT64_C(0xfffffeffffffffff)) }

__device__ __forceinline__
uint64_t cs_montgomery_mul(int pidx, uint64_t a, uint64_t b)
{
    switch (pidx) {
        case 0:  return cs_mont_mul_p0(a, b);
        case 1:  return cs_mont_mul_p1(a, b);
        case 2:  return cs_mont_mul_p2(a, b);
        default: return cs_mont_mul_p3(a, b);
    }
}

/* ─── Cooley-Tukey DIT butterfly ────────────────────────────────────────── */
/* In-place: (lds[lu], lds[lv]) → (u + w*v, u - w*v) mod p (runtime pidx).  */
__device__ __forceinline__
void ct_butterfly(int pidx, uint64_t* lds, int lu, int lv,
                  uint64_t w, uint64_t w_inv)
{
    uint64_t u = lds[lu];
    uint64_t t = cs_shoup_mul(pidx, lds[lv], w, w_inv);
    lds[lu] = cs_addmod(pidx, u, t);
    lds[lv] = cs_submod(pidx, u, t);
}

/* ─── Single-block NTT kernel ────────────────────────────────────────────── */
/* Same grid/LDS/twiddle layout as ntt_kernel.hip; pidx is a runtime arg.    */
__global__ NTT_LAUNCH_BOUNDS void ntt_kernel_cs(
    int pidx,
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ tw,
    int log_n,
    int tw_offset)
{
    __shared__ uint64_t lds[LDS_TOTAL];

    const int tid  = threadIdx.x;
    const int base = blockIdx.x * 1024;

    int i0 = tid * 2,     i1 = tid * 2 + 1;
    int j0 = __brev(i0) >> (32 - log_n);
    int j1 = __brev(i1) >> (32 - log_n);
    int l0 = lds_idx(i0), l1 = lds_idx(i1);
    lds[l0] = data[base + j0];
    lds[l1] = data[base + j1];
    __syncthreads();

    int tw_base = tw_offset;
    for (int s = 0; s < log_n; s++) {
        int half  = 1 << s;
        int group = tid >> s;
        int pos   = tid & (half - 1);

        int idx_u = (group << (s + 1)) + pos;
        int idx_v = idx_u + half;

        uint64_t w     = tw[tw_base + pos].w;
        uint64_t w_inv = tw[tw_base + pos].w_inv;

        ct_butterfly(pidx, lds, lds_idx(idx_u), lds_idx(idx_v), w, w_inv);
        __syncthreads();

        tw_base += (1 << s);
    }

    data[base + i0] = lds[l0];
    data[base + i1] = lds[l1];
}

/* ─── Cross-twiddle kernel (4-step NTT step 2) ───────────────────────────── */
/* See ntt_kernel.hip C7 finding; logic identical, pidx is runtime.          */
__global__ void cross_twiddle_kernel_cs(
    int pidx,
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ tw_cross,
    int N2)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (c >= N2) return;
    int idx = r * N2 + c;
    data[idx] = cs_shoup_mul(pidx, data[idx],
                             tw_cross[idx].w, tw_cross[idx].w_inv);
}

/* ─── Bit-reversal permutation kernel (prime-independent, no pidx) ───────── */
__global__ void bit_reverse_kernel_cs(
    uint64_t* __restrict__ data,
    int N,
    int log_n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = __brev((unsigned int)i) >> (32 - log_n);
    if (i < j) {
        uint64_t tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

static void launch_bit_reverse_cs(uint64_t* data, int log_n, hipStream_t stream)
{
    int N     = 1 << log_n;
    int nblks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bit_reverse_kernel_cs<<<nblks, BLOCK_SIZE, 0, stream>>>(data, N, log_n);
}

/* ─── Global-memory per-stage butterfly kernel ───────────────────────────── */
/* tw layout: stage-bucketed (same as ntt_kernel.hip), tw_base=(1<<s)-1.     */
__global__ void ntt_global_stage_kernel_cs(
    int pidx,
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ tw,
    int half,      /* 1 << s */
    int n_half,    /* N/2; threads beyond this must NOT touch data */
    int tw_base)   /* sum_{t<s} 2^t = (1<<s)-1 */
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_half) return;             /* small-N: extra threads idle */
    int pos = i & (half - 1);
    int u   = (i - pos) * 2 + pos;       /* u_idx = group*2*half + pos */
    int v   = u + half;

    uint64_t w     = tw[tw_base + pos].w;
    uint64_t w_inv = tw[tw_base + pos].w_inv;
    uint64_t a     = data[u];
    uint64_t t     = cs_shoup_mul(pidx, data[v], w, w_inv);
    data[u] = cs_addmod(pidx, a, t);
    data[v] = cs_submod(pidx, a, t);
}

/* ─── Global multi-stage NTT launcher ───────────────────────────────────── */
static void launch_ntt_global_cs(
    int              pidx,
    uint64_t*        data,
    const ShoupPair* tw,
    int              log_n,
    hipStream_t      stream)
{
    int N      = 1 << log_n;
    int n_half = N / 2;
    /* Adaptive block: see lib2/ntt_kernel.hip launch_ntt_global. */
    int block = BLOCK_SIZE;
    if (n_half < BLOCK_SIZE) {
        block = 64;
        while (block < n_half) block <<= 1;
        if (block < 64) block = 64;
    }
    int nblks  = (n_half + block - 1) / block;
    if (nblks < 1) nblks = 1;
    int tw_base = 0;
    for (int s = 0; s < log_n; s++) {
        int half = 1 << s;
        ntt_global_stage_kernel_cs<<<nblks, block, 0, stream>>>(
            pidx, data, tw, half, n_half, tw_base);
        tw_base += half;   /* stage s contributes 2^s twiddle entries */
    }
}

/* NOTE: ntt_kernel.hip carries an UNUSED `launch_ntt_large<PIDX>` template
 * (incorrect for odd log_n, kept for reference). A template that is never
 * instantiated emits no code and no -Wunused-function warning. The
 * equivalent here would be a concrete `static` function, which DOES warn
 * under -Wall. Since it is dead in both builds and irrelevant to the G9
 * ISA diff (no kernel/launcher references it), it is intentionally omitted
 * here rather than suppressed — keeps this TU 0-warning. */

/* ─── Public NTT launcher ────────────────────────────────────────────────── */
void launch_ntt_cs(
    int pidx,
    uint64_t* data,
    const ShoupPair* tw,
    const ShoupPair* tw_cross,
    int log_n,
    hipStream_t stream)
{
    /* Only log_n == 10 uses the LDS kernel (hardcoded for one 1024-elem
     * block). All other sizes go through the bit-reverse + global path. */
    (void)tw_cross;
    if (log_n == 10) {
        int N = 1 << log_n;
        int n_blocks = N / 1024;
        ntt_kernel_cs<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
            pidx, data, tw, log_n, 0);
    } else {
        launch_bit_reverse_cs(data, log_n, stream);
        launch_ntt_global_cs(pidx, data, tw, log_n, stream);
    }
}

/* ─── Scale kernel (INTT final step) ─────────────────────────────────────── */
__global__ void scale_kernel_cs(
    int pidx,
    uint64_t* __restrict__ data,
    uint64_t               n_inv,
    int                    N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] = cs_mulmod(pidx, data[idx], n_inv);
}

/* ─── Hadamard (pointwise multiply) kernel ───────────────────────────────── */
__global__ void hadamard_kernel_cs(
    int pidx,
    uint64_t* __restrict__       f,
    const uint64_t* __restrict__ g,
    int                          N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    f[idx] = cs_mulmod(pidx, f[idx], g[idx]);
}

/* ─── Public INTT launcher ───────────────────────────────────────────────── */
void launch_intt_cs(
    int              pidx,
    uint64_t*        data,
    uint64_t         n_inv,
    const ShoupPair* tw_inv,
    const ShoupPair* tw_cross_inv,
    int              log_n,
    hipStream_t      stream)
{
    launch_ntt_cs(pidx, data, tw_inv, tw_cross_inv, log_n, stream);
    int N       = 1 << log_n;
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scale_kernel_cs<<<nblocks, BLOCK_SIZE, 0, stream>>>(pidx, data, n_inv, N);
}

/* ─── Public Hadamard launcher ───────────────────────────────────────────── */
void launch_hadamard_cs(
    int             pidx,
    uint64_t*       f,
    const uint64_t* g,
    int             N,
    hipStream_t     stream)
{
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hadamard_kernel_cs<<<nblocks, BLOCK_SIZE, 0, stream>>>(pidx, f, g, N);
}

/* ─── Scaled single-block NTT kernel (INTT final pass) ──────────────────── */
__global__ NTT_LAUNCH_BOUNDS void ntt_kernel_scaled_cs(
    int pidx,
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ tw,
    int log_n,
    int tw_offset,
    uint64_t n_inv_w,     /* Shoup w     for N^{-1} */
    uint64_t n_inv_winv)  /* Shoup w_inv for N^{-1} */
{
    __shared__ uint64_t lds[LDS_TOTAL];

    const int tid  = threadIdx.x;
    const int base = blockIdx.x * 1024;

    int i0 = tid * 2,     i1 = tid * 2 + 1;
    int j0 = __brev(i0) >> (32 - log_n);
    int j1 = __brev(i1) >> (32 - log_n);
    int l0 = lds_idx(i0), l1 = lds_idx(i1);
    lds[l0] = data[base + j0];
    lds[l1] = data[base + j1];
    __syncthreads();

    int tw_base = tw_offset;
    for (int s = 0; s < log_n; s++) {
        int half  = 1 << s;
        int group = tid >> s;
        int pos   = tid & (half - 1);

        int idx_u = (group << (s + 1)) + pos;
        int idx_v = idx_u + half;

        uint64_t w     = tw[tw_base + pos].w;
        uint64_t w_inv = tw[tw_base + pos].w_inv;

        ct_butterfly(pidx, lds, lds_idx(idx_u), lds_idx(idx_v), w, w_inv);
        __syncthreads();

        tw_base += (1 << s);
    }

    data[base + i0] = cs_shoup_mul(pidx, lds[l0], n_inv_w, n_inv_winv);
    data[base + i1] = cs_shoup_mul(pidx, lds[l1], n_inv_w, n_inv_winv);
}

/* ─── Fused INTT launcher: no separate scale_kernel pass ────────────────── */
void launch_intt_fused_cs(
    int              pidx,
    uint64_t*        data,
    uint64_t         n_inv,
    const ShoupPair* tw_inv,
    const ShoupPair* tw_cross_inv_scaled,
    int              log_n,
    hipStream_t      stream)
{
    (void)tw_cross_inv_scaled;
    if (log_n == 10) {
        ShoupPair nsp  = make_shoup(n_inv, PRIMES[pidx]);
        int N          = 1 << log_n;
        int n_blocks   = N / 1024;
        ntt_kernel_scaled_cs<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
            pidx, data, tw_inv, log_n, 0, nsp.w, nsp.w_inv);
    } else {
        launch_bit_reverse_cs(data, log_n, stream);
        launch_ntt_global_cs(pidx, data, tw_inv, log_n, stream);
        int N     = 1 << log_n;
        int nblks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scale_kernel_cs<<<nblks, BLOCK_SIZE, 0, stream>>>(pidx, data, n_inv, N);
    }
}

/* ─── Twist / untwist kernels (negacyclic NTT pre/post multiply) ─────────── */
__global__ void twist_kernel_cs(
    int pidx,
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ psi_table,
    int                           N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] = cs_shoup_mul(pidx, data[idx],
                             psi_table[idx].w, psi_table[idx].w_inv);
}

__global__ void untwist_kernel_cs(
    int pidx,
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ psi_inv_table,
    int                           N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] = cs_shoup_mul(pidx, data[idx],
                             psi_inv_table[idx].w, psi_inv_table[idx].w_inv);
}

void launch_twist_cs(
    int              pidx,
    uint64_t*        data,
    const ShoupPair* psi_table,
    int              N,
    hipStream_t      stream)
{
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    twist_kernel_cs<<<nblocks, BLOCK_SIZE, 0, stream>>>(pidx, data, psi_table, N);
}

void launch_untwist_cs(
    int              pidx,
    uint64_t*        data,
    const ShoupPair* psi_inv_table,
    int              N,
    hipStream_t      stream)
{
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    untwist_kernel_cs<<<nblocks, BLOCK_SIZE, 0, stream>>>(
        pidx, data, psi_inv_table, N);
}

/* ─── Device query: report wavefront geometry (prime-independent) ────────── */
void init_ntt_device_cs(int dev)
{
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, dev);
    if (prop.warpSize != 64) {
        fprintf(stderr,
            "[ntt_kernel_cs] device %d: warpSize=%d (dev/RDNA2 path). "
            "BLOCK_SIZE=%d = %d wavefronts — correct, MI300A target is "
            "wave64 (8 wavefronts).\n",
            dev, prop.warpSize, BLOCK_SIZE, BLOCK_SIZE / prop.warpSize);
    }
}

/* ─── Stockham LDS-fused NTT kernel ─────────────────────────────────────── */
/* Same algorithm as ntt_kernel.hip stockham_kernel; pidx is runtime.        */
template<int LOG_N_SUB>
__global__ NTT_LAUNCH_BOUNDS void stockham_kernel_cs(
    int pidx,
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ tw)
{
    constexpr int half_c = 1 << (LOG_N_SUB - 1);
    constexpr int STOK_STRIDE_C = (2*half_c-1) + ((2*half_c-1) >> 5) + 1;
    __shared__ uint64_t lds[2 * STOK_STRIDE_C];

    constexpr int half = half_c;
    const int tid  = threadIdx.x;
    const int base = blockIdx.x * (half * 2);

    int src_buf = 0;

    if (tid < half) {
        lds[STOK_PAD(tid)]        = data[base + tid];
        lds[STOK_PAD(tid + half)] = data[base + tid + half];
    }
    __syncthreads();

    for (int s = 0; s < LOG_N_SUB; s++) {
        int p   = 1 << s;
        int qs  = half >> s;
        int p2  = p << 1;
        int j   = tid & (p - 1);
        int k   = tid >> s;
        int dst_buf = 1 - src_buf;

        if (tid < half) {
            uint64_t u  = lds[src_buf * STOK_STRIDE_C + STOK_PAD(j + k * p)];
            uint64_t v  = lds[src_buf * STOK_STRIDE_C + STOK_PAD(j + k * p + half)];
            uint64_t w     = tw[j * qs].w;
            uint64_t w_inv = tw[j * qs].w_inv;
            uint64_t wv    = cs_shoup_mul(pidx, v, w, w_inv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2)]     = cs_addmod(pidx, u, wv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2 + p)] = cs_submod(pidx, u, wv);
        }
        __syncthreads();
        src_buf = 1 - src_buf;
    }

    if (tid < half) {
        data[base + tid]        = lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid)];
        data[base + tid + half] = lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid + half)];
    }
}

template<int LOG_N_SUB>
__global__ NTT_LAUNCH_BOUNDS void stockham_kernel_scaled_cs(
    int pidx,
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ tw,
    uint64_t n_inv_w,
    uint64_t n_inv_winv)
{
    constexpr int half_c = 1 << (LOG_N_SUB - 1);
    constexpr int STOK_STRIDE_C = (2*half_c-1) + ((2*half_c-1) >> 5) + 1;
    __shared__ uint64_t lds[2 * STOK_STRIDE_C];

    constexpr int half = half_c;
    const int tid  = threadIdx.x;
    const int base = blockIdx.x * (half * 2);
    int src_buf    = 0;

    if (tid < half) {
        lds[STOK_PAD(tid)]        = data[base + tid];
        lds[STOK_PAD(tid + half)] = data[base + tid + half];
    }
    __syncthreads();

    for (int s = 0; s < LOG_N_SUB; s++) {
        int p   = 1 << s;
        int qs  = half >> s;
        int p2  = p << 1;
        int j   = tid & (p - 1);
        int k   = tid >> s;
        int dst_buf = 1 - src_buf;

        if (tid < half) {
            uint64_t u     = lds[src_buf * STOK_STRIDE_C + STOK_PAD(j + k * p)];
            uint64_t v     = lds[src_buf * STOK_STRIDE_C + STOK_PAD(j + k * p + half)];
            uint64_t wv    = cs_shoup_mul(pidx, v, tw[j * qs].w, tw[j * qs].w_inv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2)]     = cs_addmod(pidx, u, wv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2 + p)] = cs_submod(pidx, u, wv);
        }
        __syncthreads();
        src_buf = 1 - src_buf;
    }

    if (tid < half) {
        data[base + tid]        = cs_shoup_mul(pidx,
            lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid)],        n_inv_w, n_inv_winv);
        data[base + tid + half] = cs_shoup_mul(pidx,
            lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid + half)], n_inv_w, n_inv_winv);
    }
}

/* Threads-per-block for a stockham sub-NTT: fit the block to the work (one
 * butterfly per thread = `half` = 2^(log_n_sub-1) threads), clamped to
 * [64, 1024]. Local copy of the helper in ntt_kernel.hip (this cswitch sibling
 * deliberately duplicates kernel helpers for the G9 templates-vs-switch A/B). */
static inline int stok_block(int log_n_sub)
{
    int h = 1 << (log_n_sub - 1);
    if (h < 64)   h = 64;
    if (h > 1024) h = 1024;
    return h;
}

/* C1: dispatch helpers for cswitch kernels — select LOG_N_SUB specialization. */
static void launch_stok_cs(int pidx, uint64_t* data, const ShoupPair* tw,
                           int n_blocks, int log_n_sub, hipStream_t stream)
{
    int blk = stok_block(log_n_sub);
    switch (log_n_sub) {
        case  1: stockham_kernel_cs< 1><<<n_blocks,blk,0,stream>>>(pidx,data,tw); break;
        case  2: stockham_kernel_cs< 2><<<n_blocks,blk,0,stream>>>(pidx,data,tw); break;
        case  3: stockham_kernel_cs< 3><<<n_blocks,blk,0,stream>>>(pidx,data,tw); break;
        case  4: stockham_kernel_cs< 4><<<n_blocks,blk,0,stream>>>(pidx,data,tw); break;
        case  5: stockham_kernel_cs< 5><<<n_blocks,blk,0,stream>>>(pidx,data,tw); break;
        case  6: stockham_kernel_cs< 6><<<n_blocks,blk,0,stream>>>(pidx,data,tw); break;
        case  7: stockham_kernel_cs< 7><<<n_blocks,blk,0,stream>>>(pidx,data,tw); break;
        case  8: stockham_kernel_cs< 8><<<n_blocks,blk,0,stream>>>(pidx,data,tw); break;
        case  9: stockham_kernel_cs< 9><<<n_blocks,blk,0,stream>>>(pidx,data,tw); break;
        case 10: stockham_kernel_cs<10><<<n_blocks,blk,0,stream>>>(pidx,data,tw); break;
        case 11: stockham_kernel_cs<11><<<n_blocks,blk,0,stream>>>(pidx,data,tw); break;
        default: fprintf(stderr,"launch_stok_cs: bad log_n_sub=%d\n",log_n_sub); abort();
    }
}

static void launch_stok_scaled_cs(int pidx, uint64_t* data, const ShoupPair* tw,
                                  int n_blocks, int log_n_sub,
                                  uint64_t n_inv_w, uint64_t n_inv_winv,
                                  hipStream_t stream)
{
    int blk = stok_block(log_n_sub);
    switch (log_n_sub) {
        case  1: stockham_kernel_scaled_cs< 1><<<n_blocks,blk,0,stream>>>(pidx,data,tw,n_inv_w,n_inv_winv); break;
        case  2: stockham_kernel_scaled_cs< 2><<<n_blocks,blk,0,stream>>>(pidx,data,tw,n_inv_w,n_inv_winv); break;
        case  3: stockham_kernel_scaled_cs< 3><<<n_blocks,blk,0,stream>>>(pidx,data,tw,n_inv_w,n_inv_winv); break;
        case  4: stockham_kernel_scaled_cs< 4><<<n_blocks,blk,0,stream>>>(pidx,data,tw,n_inv_w,n_inv_winv); break;
        case  5: stockham_kernel_scaled_cs< 5><<<n_blocks,blk,0,stream>>>(pidx,data,tw,n_inv_w,n_inv_winv); break;
        case  6: stockham_kernel_scaled_cs< 6><<<n_blocks,blk,0,stream>>>(pidx,data,tw,n_inv_w,n_inv_winv); break;
        case  7: stockham_kernel_scaled_cs< 7><<<n_blocks,blk,0,stream>>>(pidx,data,tw,n_inv_w,n_inv_winv); break;
        case  8: stockham_kernel_scaled_cs< 8><<<n_blocks,blk,0,stream>>>(pidx,data,tw,n_inv_w,n_inv_winv); break;
        case  9: stockham_kernel_scaled_cs< 9><<<n_blocks,blk,0,stream>>>(pidx,data,tw,n_inv_w,n_inv_winv); break;
        case 10: stockham_kernel_scaled_cs<10><<<n_blocks,blk,0,stream>>>(pidx,data,tw,n_inv_w,n_inv_winv); break;
        default: fprintf(stderr,"launch_stok_scaled_cs: bad log_n_sub=%d\n",log_n_sub); abort();
    }
}

/* ─── 4-step Stockham launcher ───────────────────────────────────────────── */
static void launch_stockham_large_cs(
    int              pidx,
    uint64_t*        data,
    const ShoupPair* tw_stok,
    const ShoupPair* tw_cross,
    int              log_n,
    hipStream_t      stream)
{
    int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
    int N1 = 1 << log_n1, N2 = 1 << log_n2;

    /* Row pass: N2 independent sub-NTTs of size N1 (C1 dispatch). */
    launch_stok_cs(pidx, data, tw_stok, N2, log_n1, stream);
    dim3 grid2((N2 + 255) / 256, N1);
    cross_twiddle_kernel_cs<<<grid2, 256, 0, stream>>>(pidx, data, tw_cross, N2);
    /* Col pass: N1 independent sub-NTTs of size N2 (C1 dispatch). */
    launch_stok_cs(pidx, data, tw_stok + N1 / 2, N1, log_n2, stream);
}

/* ─── Public Stockham NTT launcher ──────────────────────────────────────── */
void launch_stockham_ntt_cs(
    int              pidx,
    uint64_t*        data,
    const ShoupPair* tw_stok,
    const ShoupPair* tw_cross,
    int              log_n,
    hipStream_t      stream)
{
    if (log_n > 10 && (log_n & 1)) {
        fprintf(stderr, "launch_stockham_ntt_cs: log_n=%d odd; 4-step "
                        "requires even log_n\n", log_n);
        abort();
    }
    int N = 1 << log_n;
    if (log_n <= 10) {
        int n_blocks = (N < 1024) ? 1 : N / 1024;
        launch_stok_cs(pidx, data, tw_stok, n_blocks, log_n, stream);
    } else {
        launch_stockham_large_cs(pidx, data, tw_stok, tw_cross, log_n, stream);
    }
}

/* ─── Public Stockham INTT launcher ─────────────────────────────────────── */
void launch_stockham_intt_cs(
    int              pidx,
    uint64_t*        data,
    uint64_t         n_inv,
    const ShoupPair* tw_stok_inv,
    const ShoupPair* tw_cross_inv_scaled,
    int              log_n,
    hipStream_t      stream)
{
    if (log_n > 10 && (log_n & 1)) {
        fprintf(stderr, "launch_stockham_intt_cs: log_n=%d odd; 4-step "
                        "requires even log_n\n", log_n);
        abort();
    }
    if (log_n <= 10) {
        ShoupPair nsp  = make_shoup(n_inv, PRIMES[pidx]);
        int N          = 1 << log_n;
        int n_blocks   = (N < 1024) ? 1 : N / 1024;
        launch_stok_scaled_cs(pidx, data, tw_stok_inv, n_blocks, log_n,
                              nsp.w, nsp.w_inv, stream);
    } else {
        int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
        int N1 = 1 << log_n1, N2 = 1 << log_n2;

        /* Row pass: N2 independent sub-INTTs of size N1 (C1 dispatch). */
        launch_stok_cs(pidx, data, tw_stok_inv, N2, log_n1, stream);
        dim3 grid2((N2 + 255) / 256, N1);
        cross_twiddle_kernel_cs<<<grid2, 256, 0, stream>>>(
            pidx, data, tw_cross_inv_scaled, N2);
        /* Col pass: N1 independent sub-INTTs of size N2 (C1 dispatch). */
        launch_stok_cs(pidx, data, tw_stok_inv + N1 / 2, N1, log_n2, stream);
    }
}

/* ─── Montgomery NTT kernel ──────────────────────────────────────────────── */
__global__ NTT_LAUNCH_BOUNDS void mont_ntt_kernel_cs(
    int pidx,
    uint64_t* __restrict__       data,
    const uint64_t* __restrict__ tw_mont,  /* w * R mod p per entry, 8 B each */
    int log_n,
    int tw_offset)
{
    __shared__ uint64_t lds[LDS_TOTAL];

    const int tid  = threadIdx.x;
    const int base = blockIdx.x * 1024;

    /* Bit-reversed load — CT-DIT stage-bucketed twiddle layout requires
     * bit-reversed input. Mirrors ntt_kernel_cs above and matches the
     * corresponding fix in lib2/ntt_kernel.hip mont_ntt_kernel. */
    int i0 = tid * 2,     i1 = tid * 2 + 1;
    int j0 = __brev(i0) >> (32 - log_n);
    int j1 = __brev(i1) >> (32 - log_n);
    int l0 = lds_idx(i0), l1 = lds_idx(i1);
    lds[l0] = data[base + j0];
    lds[l1] = data[base + j1];
    __syncthreads();

    int tw_base = tw_offset;
    for (int s = 0; s < log_n; s++) {
        int half  = 1 << s;
        int group = tid >> s;
        int pos   = tid & (half - 1);
        int idx_u = (group << (s + 1)) + pos;
        int idx_v = idx_u + half;

        uint64_t w = tw_mont[tw_base + pos];
        uint64_t u = lds[lds_idx(idx_u)];
        uint64_t t = cs_montgomery_mul(pidx, lds[lds_idx(idx_v)], w);
        lds[lds_idx(idx_u)] = cs_addmod(pidx, u, t);
        lds[lds_idx(idx_v)] = cs_submod(pidx, u, t);
        __syncthreads();

        tw_base += (1 << s);
    }

    data[base + i0] = lds[l0];
    data[base + i1] = lds[l1];
}

static void launch_mont_ntt_large_cs(
    int              pidx,
    uint64_t*        data,
    const uint64_t*  tw_mont,
    const ShoupPair* tw_cross,
    int log_n,
    hipStream_t stream)
{
    int log_n1 = log_n / 2;
    int log_n2 = log_n - log_n1;
    int N1     = 1 << log_n1;
    int N2     = 1 << log_n2;
    int total_blocks = (N1 * N2) / 1024;

    mont_ntt_kernel_cs<<<total_blocks, BLOCK_SIZE, 0, stream>>>(
        pidx, data, tw_mont, log_n1, 0);
    dim3 grid2((N2 + 255) / 256, N1);
    cross_twiddle_kernel_cs<<<grid2, 256, 0, stream>>>(
        pidx, data, tw_cross, N2);
    mont_ntt_kernel_cs<<<total_blocks, BLOCK_SIZE, 0, stream>>>(
        pidx, data, tw_mont + (N1 / 2), log_n2, 0);
}

void launch_mont_ntt_cs(
    int              pidx,
    uint64_t*        data,
    const uint64_t*  tw_mont,
    const ShoupPair* tw_cross,
    int log_n,
    hipStream_t stream)
{
    int N = 1 << log_n;
    if (log_n <= 10) {
        int n_blocks = (N < 1024) ? 1 : N / 1024;
        mont_ntt_kernel_cs<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
            pidx, data, tw_mont, log_n, 0);
    } else {
        launch_mont_ntt_large_cs(pidx, data, tw_mont, tw_cross, log_n, stream);
    }
}
