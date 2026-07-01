/*
 * ntt_kernel.hip — GPU NTT butterfly kernel
 *
 * A single template handles all four primes; PIDX is a compile-time
 * parameter that selects the reduction function and bakes the prime
 * constant as an ISA immediate. Four kernel instantiations are emitted;
 * no runtime branching in the butterfly inner loop.
 *
 * LDS layout:
 *   Each block processes 1024 elements. With 512 threads (8 wave64
 *   wavefronts), each thread handles 2 elements.
 *   LDS has 32 banks x 4 bytes. A uint64_t spans 2 banks. At butterfly
 *   stride 32, all threads map to the same 2 banks: 32-way conflict.
 *   Fix: insert 1 uint64_t padding every 32 elements (LDS_STRIDE = 33).
 *   lds_idx(i) = (i>>5)*33 + (i&31). Check: lds_idx(32)=33, lds_idx(64)=66.
 *
 * Twiddle layout:
 *   Bit-reversed Cooley-Tukey order. Stage s contributes 2^s entries.
 *   Offsets: stage 0 at tw[0], stage 1 at tw[1], stage 2 at tw[3], etc.
 *   Precomputed on CPU, DMA'd to device HBM once at init.
 *
 * Transform paths in this file (all share the butterfly + reduction above):
 *   - CT-DIT single-block + classic 4-step (ntt_kernel, launch_ntt): the
 *     reference path. The 4-step does N = N1*N2 as row-NTT → cross-twiddle →
 *     column-NTT with no explicit transpose (column passes read the strided
 *     row-NTT output, which the algorithm accounts for).
 *   - LDS-fused Stockham 4-step (stockham_kernel, launch_stockham_large): the
 *     production even-log_n path — sub-NTTs stay in LDS, bracketed by explicit
 *     square transposes (transpose_sq) plus a fused transpose+cross-twiddle (I13).
 *   - Recursive 3-factor 4-step (launch_stockham_3factor_ntt): log_n=24 only —
 *     outer 12+12, inner 6+6 leaves, with batched transposes (2026-06-17).
 *   Montgomery-twiddle CT-DIT (I4), Hadamard, twist/untwist, the rectangular
 *   odd-log_n path (I11, disabled in production), and the inverse variants of
 *   each round out the file.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>   /* abort() for launcher pre-condition checks */
#include "primes.h"
#include "shoup.h"
#include "crt_ntt.h"

/* ─── LDS constants ──────────────────────────────────────────────────────── */
/* Wave-size note: every LDS kernel below synchronises ONLY with block-level
 * __syncthreads() and uses NO warp intrinsics (__shfl/__ballot/readlane/...),
 * so each kernel is correct for ANY wavefront width. BLOCK_SIZE=512 is, by
 * construction, MI300A-native: 8 full wave64 wavefronts on gfx942 with zero
 * partial waves; on gfx1030 it is 16 wave32 warps (also exact). No separate
 * 32-thread kernel is required for correctness — only the occupancy hint
 * differs per arch (NTT_LAUNCH_BOUNDS).  LDS bank count is 32 on BOTH RDNA2
 * and CDNA3, so LDS_STRIDE=33 (bank-conflict padding) is wave-independent —
 * the old "runtime block dispatch needs LDS restructuring" claim was wrong. */
#define BLOCK_SIZE  512             /* 8 wave64 (gfx942) / 16 wave32 (gfx1030) */
#define LDS_STRIDE  33              /* 32 elements + 1 padding (32 LDS banks)  */
#define LDS_TOTAL   (32 * LDS_STRIDE) /* = 1056 uint64_t = 8448 bytes/block  */

/* Arch-keyed occupancy hint. gfx1030 is the explicit special case; the
 * default IS the MI300A/CDNA3 (wave64) target. The minBlocksPerCU value is
 * kept at the gfx1030-verified 2 for both until measured gfx942 occupancy
 * (STATUS task G6/G8) justifies raising it — per the project rule against
 * hardcoded, unmeasured architecture constants. Single knob, not a magic
 * literal scattered across 5 kernels. */
#if defined(GFX1030_LOCAL)
#  define NTT_LAUNCH_BOUNDS __launch_bounds__(512, 2)  /* RDNA2 dev path     */
#else
#  define NTT_LAUNCH_BOUNDS __launch_bounds__(512, 2)  /* CDNA3 native; minBlocks=2
     kept until G6 PMC confirms occupancy gap — C1 reduced LDS at LOG_N_SUB=8 from
     16912 B to 4208 B, so minBlocks=4 is now LDS-feasible for that hot path. */
#endif

/* Logical-to-padded LDS index. Eliminates bank conflicts at all strides.    */
__device__ __forceinline__
int lds_idx(int i) { return (i >> 5) * LDS_STRIDE + (i & 31); }

/* ─── Cooley-Tukey DIT butterfly ────────────────────────────────────────── */
/* In-place: (lds[lu], lds[lv]) → (u + w*v, u - w*v) mod p                 */
template<int PIDX>
__device__ __forceinline__
void ct_butterfly(uint64_t* lds, int lu, int lv, uint64_t w, uint64_t w_inv)
{
    uint64_t u = lds[lu];
    uint64_t t = shoup_mul<PIDX>(lds[lv], w, w_inv);
    lds[lu] = addmod<PIDX>(u, t);
    lds[lv] = submod<PIDX>(u, t);
}

/* ─── Single-block NTT kernel ────────────────────────────────────────────── */
/*                                                                            */
/* Grid:  dim3(max(1, N/1024)) blocks                                         */
/* Block: dim3(BLOCK_SIZE) = dim3(512) threads                               */
/* Each block processes exactly 1024 contiguous elements.                    */
/*                                                                            */
/* tw: twiddle table in bit-reversed CT order (stage 0 first, 2^s entries    */
/* per stage). Pointer into device HBM; GPU reads at full 5.3 TB/s.         */
/* tw_offset: starting position in tw[] for this block's chunk of the NTT.  */
/*   For single-pass NTT (log_n <= 10), tw_offset = 0.                       */
/*   For 4-step NTT row/col passes, tw_offset selects the correct sub-table. */

template<int PIDX>
__global__ NTT_LAUNCH_BOUNDS void ntt_kernel(
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ tw,
    int log_n,
    int tw_offset)
{
    __shared__ uint64_t lds[LDS_TOTAL];

    const int tid  = threadIdx.x;
    const int base = blockIdx.x * 1024;

    /* Load 2 elements per thread into padded LDS.
     * Load from bit-reversed global positions so DIT butterfly computes the
     * standard DFT (natural-order output).  Bit-reversal: brev(i, log_n). */
    int i0 = tid * 2,     i1 = tid * 2 + 1;
    int j0 = __brev(i0) >> (32 - log_n);
    int j1 = __brev(i1) >> (32 - log_n);
    int l0 = lds_idx(i0), l1 = lds_idx(i1);
    lds[l0] = data[base + j0];
    lds[l1] = data[base + j1];
    __syncthreads();

    /* log_n butterfly stages (Cooley-Tukey DIT) */
    int tw_base = tw_offset;
    for (int s = 0; s < log_n; s++) {
        int half  = 1 << s;
        int group = tid >> s;
        int pos   = tid & (half - 1);

        int idx_u = (group << (s + 1)) + pos;
        int idx_v = idx_u + half;

        /* All elements at the same pos within their group share one twiddle.  */
        /* pos ∈ [0, 2^s) indexes into the 2^s entries allocated for stage s. */
        uint64_t w     = tw[tw_base + pos].w;
        uint64_t w_inv = tw[tw_base + pos].w_inv;

        ct_butterfly<PIDX>(lds, lds_idx(idx_u), lds_idx(idx_v), w, w_inv);
        __syncthreads();

        tw_base += (1 << s); /* stage s contributes 2^s twiddle entries       */
    }

    /* Store back to global memory */
    data[base + i0] = lds[l0];
    data[base + i1] = lds[l1];
}

/* ─── Cross-twiddle kernel (4-step NTT step 2) ───────────────────────────── */
/* Square-decomposition only: index r*N2+c is correct iff N1=N2 (even      */
/* log_n). Stockham's natural-in/natural-out hides the transpose only when */
/* the decomposition is square; for odd log_n the algorithm additionally   */
/* needs a transpose between the two stockham passes which is not          */
/* implemented here. launch_stockham_ntt rejects odd log_n; callers must   */
/* fall back to launch_ntt (CT-DIT) for that case. */
template<int PIDX>
__global__ void cross_twiddle_kernel(
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ tw_cross,
    int N2)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (c >= N2) return;
    int idx = r * N2 + c;
    data[idx] = shoup_mul<PIDX>(data[idx], tw_cross[idx].w, tw_cross[idx].w_inv);
}

/* ─── Bit-reversal permutation kernel ───────────────────────────────────── */
/* Swaps element i with element bitrev(i, log_n) for all i < bitrev(i).     */
/* Required before NTT or INTT so that DIT butterfly computes standard DFT. */
__global__ void bit_reverse_kernel(
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

static void launch_bit_reverse(uint64_t* data, int log_n, hipStream_t stream)
{
    int N     = 1 << log_n;
    int nblks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bit_reverse_kernel<<<nblks, BLOCK_SIZE, 0, stream>>>(data, N, log_n);
}

/* ─── Global-memory per-stage butterfly kernel ───────────────────────────── */
/* Processes one CT-DIT stage across all N elements using global memory.
 * Each thread handles exactly one butterfly pair (u, v) where v = u + half.
 * Correct for any log_n without shared memory or transposition.
 * Used for log_n > 10 (N > 1024) where the LDS-based kernel covers only
 * N ≤ 1024 elements per block.
 * tw layout: stage-bucketed (same as make_tw_local), tw_base = (1<<s)-1.   */
template<int PIDX>
__global__ void ntt_global_stage_kernel(
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
    uint64_t t     = shoup_mul<PIDX>(data[v], w, w_inv);
    data[u] = addmod<PIDX>(a, t);
    data[v] = submod<PIDX>(a, t);
}

/* ─── Global multi-stage NTT launcher ───────────────────────────────────── */
/* Runs log_n stage kernels sequentially in the given stream.
 * Correct for all log_n; used when N > 1024 (LDS kernel can't fit full N). */
template<int PIDX>
static void launch_ntt_global(
    uint64_t*        data,
    const ShoupPair* tw,
    int              log_n,
    hipStream_t      stream)
{
    int N      = 1 << log_n;
    int n_half = N / 2;
    /* Adaptive block: at small N, launching BLOCK_SIZE=512 leaves most of
     * a wavefront idle. Pick the next pow2 ≥ n_half capped at BLOCK_SIZE,
     * with a floor of 64 to remain a multiple of the MI300A wavefront. */
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
        ntt_global_stage_kernel<PIDX><<<nblks, block, 0, stream>>>(
            data, tw, half, n_half, tw_base);
        tw_base += half;   /* stage s contributes 2^s twiddle entries */
    }
}

/* ─── Public launch function ─────────────────────────────────────────────── */
template<int PIDX>
void launch_ntt(
    uint64_t* data,
    const ShoupPair* tw,
    const ShoupPair* tw_cross,
    int log_n,
    hipStream_t stream)
{
    /* Only log_n == 10 uses the LDS kernel — it's hardcoded for one
     * 1024-element block. Smaller N would overrun the device allocation
     * (the kernel always launches 512 threads writing data[base..base+1023]);
     * larger N doesn't fit a single LDS block. Both extremes route through
     * the global multi-stage path (correct for any log_n ≥ 1). */
    (void)tw_cross;   /* not used in either kept path */
    if (log_n == 10) {
        int N = 1 << log_n;
        int n_blocks = N / 1024;
        ntt_kernel<PIDX><<<n_blocks, BLOCK_SIZE, 0, stream>>>(
            data, tw, log_n, 0);
    } else {
        launch_bit_reverse(data, log_n, stream);
        launch_ntt_global<PIDX>(data, tw, log_n, stream);
    }
}

/* Forward declaration — scale_kernel body appears after the CT-DIT block.  */
template<int PIDX> __global__ void scale_kernel(uint64_t* __restrict__, uint64_t, int);

/* ─── Global-stage CT-DIT kernel with Montgomery twiddles (I4) ──────────── */
/* Montgomery variant of ntt_global_stage_kernel: reads 8 B/entry (uint64_t)  */
/* instead of 16 B/entry (ShoupPair). montgomery_mul replaces shoup_mul.      */
/* HBM bandwidth: N-1 entries × 8 B vs N-1 entries × 16 B — half the traffic. */
/* Cost: ~10 gfx942 VALU vs Shoup's ~7; net positive when memory-bound        */
/* (CT-DIT at log_n > 10 issues N global passes each reading N/2 entries).    */
template<int PIDX>
__global__ void ntt_global_stage_kernel_mont(
    uint64_t* __restrict__       data,
    const uint64_t* __restrict__ tw_mont,  /* stage-bucketed Montgomery w*R, 8 B */
    int half,
    int n_half,
    int tw_base)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_half) return;
    int pos = i & (half - 1);
    int u   = (i - pos) * 2 + pos;
    int v   = u + half;

    uint64_t w = tw_mont[tw_base + pos];
    uint64_t a = data[u];
    uint64_t t = montgomery_mul<PIDX>(data[v], w);
    data[u] = addmod<PIDX>(a, t);
    data[v] = submod<PIDX>(a, t);
}

/* Chains log_n Montgomery global-stage kernels sequentially. */
template<int PIDX>
static void launch_ntt_global_mont(
    uint64_t*       data,
    const uint64_t* tw_mont,
    int             log_n,
    hipStream_t     stream)
{
    int N      = 1 << log_n;
    int n_half = N / 2;
    int block  = BLOCK_SIZE;
    if (n_half < BLOCK_SIZE) {
        block = 64;
        while (block < n_half) block <<= 1;
        if (block < 64) block = 64;
    }
    int nblks   = (n_half + block - 1) / block;
    if (nblks < 1) nblks = 1;
    int tw_base = 0;
    for (int s = 0; s < log_n; s++) {
        int half = 1 << s;
        ntt_global_stage_kernel_mont<PIDX><<<nblks, block, 0, stream>>>(
            data, tw_mont, half, n_half, tw_base);
        tw_base += half;
    }
}

/* CT-DIT forward NTT with 8-byte Montgomery twiddles, any log_n.             */
/* For odd log_n > 10: avoids the 2× size overhead of rounding up to even.    */
template<int PIDX>
void launch_ntt_ct_dit_mont(
    uint64_t*       data,
    const uint64_t* tw_mont,
    int             log_n,
    hipStream_t     stream)
{
    launch_bit_reverse(data, log_n, stream);
    launch_ntt_global_mont<PIDX>(data, tw_mont, log_n, stream);
}

/* CT-DIT inverse NTT with Montgomery twiddles + N^{-1} scaling.              */
/* Call with inverse Montgomery twiddle table (built from root_inv).           */
template<int PIDX>
void launch_intt_ct_dit_mont(
    uint64_t*       data,
    const uint64_t* tw_inv_mont,
    uint64_t        n_inv,
    int             log_n,
    hipStream_t     stream)
{
    launch_bit_reverse(data, log_n, stream);
    launch_ntt_global_mont<PIDX>(data, tw_inv_mont, log_n, stream);
    int N       = 1 << log_n;
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scale_kernel<PIDX><<<nblocks, BLOCK_SIZE, 0, stream>>>(data, n_inv, N);
}

/* ─── Scale kernel (INTT final step) ─────────────────────────────────────── */
/* Multiplies every element by the scalar n_inv = N^{-1} mod p.              */
/* Called once per INTT, after the inverse-twiddle NTT pass.                 */
template<int PIDX>
__global__ void scale_kernel(
    uint64_t* __restrict__ data,
    uint64_t               n_inv,
    int                    N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] = mulmod<PIDX>(data[idx], n_inv);
}

/* ─── Hadamard (pointwise multiply) kernel ───────────────────────────────── */
/* f[i] = f[i] * g[i] mod p for all i in [0, N).                            */
/* Called between NTT(f) and INTT in the polymul pipeline.                   */
template<int PIDX>
__global__ void hadamard_kernel(
    uint64_t* __restrict__       f,
    const uint64_t* __restrict__ g,
    int                          N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    f[idx] = mulmod<PIDX>(f[idx], g[idx]);
}

/* ─── Public INTT launcher ───────────────────────────────────────────────── */
/* INTT(Y) = N^{-1} * NTT_{omega^{-1}}(Y).                                  */
/* tw_inv and tw_cross_inv are the inverse-twiddle tables (built with         */
/* omega_inv = omega^{p-2} mod p). n_inv = N^{-1} mod p.                    */
template<int PIDX>
void launch_intt(
    uint64_t*        data,
    uint64_t         n_inv,
    const ShoupPair* tw_inv,
    const ShoupPair* tw_cross_inv,
    int              log_n,
    hipStream_t      stream)
{
    launch_ntt<PIDX>(data, tw_inv, tw_cross_inv, log_n, stream);
    int N       = 1 << log_n;
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scale_kernel<PIDX><<<nblocks, BLOCK_SIZE, 0, stream>>>(data, n_inv, N);
}

/* ─── Public Hadamard launcher ───────────────────────────────────────────── */
template<int PIDX>
void launch_hadamard(
    uint64_t*       f,
    const uint64_t* g,
    int             N,
    hipStream_t     stream)
{
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hadamard_kernel<PIDX><<<nblocks, BLOCK_SIZE, 0, stream>>>(f, g, N);
}

/* ─── Scaled single-block NTT kernel (INTT final pass) ──────────────────── */
/* Same butterfly structure as ntt_kernel, but fuses N^{-1} Shoup scaling   */
/* into the write-back to global memory. Replaces a separate scale_kernel    */
/* pass, saving one HBM read-write cycle                                     */
/* (~256 MB = 64 μs at 4 TB/s HBM3 for N=2^24).                             */
template<int PIDX>
__global__ NTT_LAUNCH_BOUNDS void ntt_kernel_scaled(
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

    /* Same bit-reversed load as ntt_kernel: DIT on bit-reversed input gives
     * standard IDFT output (natural order) when using omega_inv twiddles. */
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

        ct_butterfly<PIDX>(lds, lds_idx(idx_u), lds_idx(idx_v), w, w_inv);
        __syncthreads();

        tw_base += (1 << s);
    }

    /* Write-back: scale by N^{-1} fused into the store (no extra HBM pass). */
    data[base + i0] = shoup_mul<PIDX>(lds[l0], n_inv_w, n_inv_winv);
    data[base + i1] = shoup_mul<PIDX>(lds[l1], n_inv_w, n_inv_winv);
}

/* ─── Fused INTT launcher: no separate scale_kernel pass ────────────────── */
/* For log_n <= 10: N^{-1} is fused into ntt_kernel_scaled write-back.       */
/* For log_n > 10:  caller provides tw_cross_inv with N^{-1} pre-multiplied  */
/*                  into every entry; cross_twiddle_kernel distributes the   */
/*                  scale at no extra cost (it already touches all N elems).  */
template<int PIDX>
void launch_intt_fused(
    uint64_t*        data,
    uint64_t         n_inv,
    const ShoupPair* tw_inv,
    const ShoupPair* tw_cross_inv_scaled,
    int              log_n,
    hipStream_t      stream)
{
    /* Same dispatch logic as launch_ntt: only log_n == 10 uses the fused-
     * scale LDS kernel; everything else uses the global path + an explicit
     * scale_kernel pass (correct for any log_n ≥ 1). */
    (void)tw_cross_inv_scaled;
    if (log_n == 10) {
        ShoupPair nsp  = make_shoup(n_inv, PRIMES[PIDX]);
        int N          = 1 << log_n;
        int n_blocks   = N / 1024;
        ntt_kernel_scaled<PIDX><<<n_blocks, BLOCK_SIZE, 0, stream>>>(
            data, tw_inv, log_n, 0, nsp.w, nsp.w_inv);
    } else {
        launch_bit_reverse(data, log_n, stream);
        launch_ntt_global<PIDX>(data, tw_inv, log_n, stream);
        int N     = 1 << log_n;
        int nblks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scale_kernel<PIDX><<<nblks, BLOCK_SIZE, 0, stream>>>(data, n_inv, N);
    }
}

/* ─── Twist kernel (negacyclic NTT pre-multiply) ────────────────────────── */
/* Multiplies data[k] by psi^k mod p for k in [0, N).                        */
/* psi_table[k] holds the Shoup pair for the k-th power of psi = g^{(p-1)/2N},
 * the primitive 2N-th root of unity. Build table once at init via CPU.      */
template<int PIDX>
__global__ void twist_kernel(
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ psi_table,
    int                           N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] = shoup_mul<PIDX>(data[idx], psi_table[idx].w, psi_table[idx].w_inv);
}

/* ─── Untwist kernel (negacyclic NTT post-multiply) ─────────────────────── */
/* Multiplies data[k] by psi^{-k} mod p after INTT to complete the negacyclic
 * polynomial multiplication (removes the twist applied before NTT).         */
template<int PIDX>
__global__ void untwist_kernel(
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ psi_inv_table,
    int                           N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] = shoup_mul<PIDX>(data[idx], psi_inv_table[idx].w, psi_inv_table[idx].w_inv);
}

/* ─── Public twist launcher ──────────────────────────────────────────────── */
template<int PIDX>
void launch_twist(
    uint64_t*        data,
    const ShoupPair* psi_table,
    int              N,
    hipStream_t      stream)
{
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    twist_kernel<PIDX><<<nblocks, BLOCK_SIZE, 0, stream>>>(data, psi_table, N);
}

/* ─── Public untwist launcher ────────────────────────────────────────────── */
template<int PIDX>
void launch_untwist(
    uint64_t*        data,
    const ShoupPair* psi_inv_table,
    int              N,
    hipStream_t      stream)
{
    int nblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    untwist_kernel<PIDX><<<nblocks, BLOCK_SIZE, 0, stream>>>(data, psi_inv_table, N);
}

/* ─── Device query: report wavefront geometry ───────────────────────────── */
/* Call once per device at startup. The kernels are wave-size-agnostic (block
 * sync only, no warp intrinsics): BLOCK_SIZE=512 = 8 full wave64 wavefronts
 * on gfx942 (MI300A-native) and 16 wave32 warps on gfx1030 (both exact, both
 * correct). This is informational, not a defect warning — it prints only
 * when NOT the wave64 target so the dev-on-gfx1030 case is visible without
 * implying incorrectness. */
void init_ntt_device(int dev)
{
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, dev);
    if (prop.warpSize != 64) {
        fprintf(stderr,
            "[ntt_kernel] device %d: warpSize=%d (dev/RDNA2 path). "
            "BLOCK_SIZE=%d = %d wavefronts — correct, MI300A target is "
            "wave64 (8 wavefronts).\n",
            dev, prop.warpSize, BLOCK_SIZE, BLOCK_SIZE / prop.warpSize);
    }
}

/* ─── Stockham LDS-fused NTT kernel ─────────────────────────────────────── */
/*                                                                            */
/* Alternative to CT-DIT (ntt_kernel): Stockham auto-sort algorithm.          */
/* Advantages over CT-DIT: (1) no bit-reversal pass; (2) natural-order        */
/* output from natural-order input; (3) all inter-stage data stays in LDS,   */
/* eliminating global memory round-trips between butterfly stages.            */
/*                                                                            */
/* Twiddle layout (FLAT, unlike CT-DIT's stage-bucketed layout):              */
/*   tw[k] = shoup(ω^k) for k = 0 .. N_sub/2 - 1.                           */
/*   Thread tid at stage s accesses tw[j * qs] where j = tid & (p-1)         */
/*   and qs = N_sub / 2^(s+1). Max index = (p-1)*qs < N_sub/2. ✓            */
/*   Table size: same N/2 ShoupPairs as CT-DIT, different contents.          */
/*                                                                            */
/* LDS layout (C1): STOK_STRIDE_C is compile-time per LOG_N_SUB.             */
/*   4-step hot path (log_n=16, LOG_N_SUB=8): STOK_STRIDE_C=263, LDS=4208 B  */
/*   vs the old static 16912 B (−75%). Enables 15 blocks/CU at LOG_N_SUB=8.  */
/*   Also fully unrolls the butterfly loop (LOG_N_SUB is compile-time).       */
/*                                                                            */
/* Grid / dispatch: launch_stok<PIDX,LOG_N_SUB>() in launch_stockham_large    */
/*   selects the correct specialization based on log_n/2.                     */

/* STOK_PAD and STOK_STRIDE defined in crt_ntt.h */

/* C2: compile-time occupancy hint for stockham_kernel specializations.
 * LDS per block = 2 * STOK_STRIDE_C * 8 bytes.  The 65 kB LDS/CU can fit
 * 4 blocks when that footprint ≤ 65536/4 = 16384 B (i.e. STOK_STRIDE_C ≤ 1024).
 * At LOG_N_SUB=8 (hot 4-step): STOK_STRIDE_C=263, LDS=4208 B → 4 fits.
 * At LOG_N_SUB=10 (single-block): STOK_STRIDE_C=1055, LDS=16880 B → 2 fits. */
template<int LOG_N_SUB>
struct StokMinBlks {
    static constexpr int half_c = 1 << (LOG_N_SUB - 1);
    static constexpr int stride = (2*half_c - 1) + ((2*half_c - 1) >> 5) + 1;
    /* LOG_N_SUB=11 (log_n=22): LDS≈33.8KB → only 1 block/CU fits in 64KB
     * (2 would need 67KB). Earlier 2-way (4/2) gave 2 here → LDS over-subscribe
     * once 1024 threads are used. 3-way 4/2/1 keeps small sizes unchanged. */
    static constexpr int value  = (2 * stride * 8 <= 65536 / 4) ? 4
                                : (2 * stride * 8 <= 65536 / 2) ? 2 : 1;
};

template<int PIDX, int LOG_N_SUB>
__global__ __launch_bounds__((LOG_N_SUB >= 11 ? 1024 : 512), StokMinBlks<LOG_N_SUB>::value) void stockham_kernel(
    uint64_t* __restrict__        data,
    const ShoupPair* __restrict__ tw)      /* flat Stockham twiddle table     */
{
    /* C1: compile-time STOK_STRIDE_C eliminates the static 1057-slot overalloc. */
    constexpr int half_c = 1 << (LOG_N_SUB - 1);
    constexpr int STOK_STRIDE_C = (2*half_c-1) + ((2*half_c-1) >> 5) + 1;
    __shared__ uint64_t lds[2 * STOK_STRIDE_C];

    constexpr int half = half_c;
    const int tid  = threadIdx.x;
    const int base = blockIdx.x * (half * 2);

    int src_buf = 0;

    /* Load from global memory into LDS buffer 0. */
    if (tid < half) {
        lds[STOK_PAD(tid)]        = data[base + tid];
        lds[STOK_PAD(tid + half)] = data[base + tid + half];
    }
    __syncthreads();

    for (int s = 0; s < LOG_N_SUB; s++) {
        int p   = 1 << s;
        int qs  = half >> s;      /* N_sub / 2^(s+1) */
        int p2  = p << 1;
        int j   = tid & (p - 1); /* position within group [0, p) */
        int k   = tid >> s;       /* group index  [0, qs) */
        int dst_buf = 1 - src_buf;

        if (tid < half) {
            uint64_t u  = lds[src_buf * STOK_STRIDE_C + STOK_PAD(j + k * p)];
            uint64_t v  = lds[src_buf * STOK_STRIDE_C + STOK_PAD(j + k * p + half)];
            /* tw[j*qs]: twiddle ω^(j*qs) for this butterfly. */
            uint64_t w     = tw[j * qs].w;
            uint64_t w_inv = tw[j * qs].w_inv;
            uint64_t wv    = shoup_mul<PIDX>(v, w, w_inv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2)]     = addmod<PIDX>(u, wv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2 + p)] = submod<PIDX>(u, wv);
        }
        __syncthreads();
        src_buf = 1 - src_buf;
    }

    /* Write natural-order output back to global memory. */
    if (tid < half) {
        data[base + tid]        = lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid)];
        data[base + tid + half] = lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid + half)];
    }
}

/* Scaled variant: fuses N^{-1} Shoup multiply into the final write-back.   */
/* Used by launch_stockham_intt for log_n <= 10 to avoid a separate scale   */
/* kernel launch (saves one HBM read-write cycle).                           */
template<int PIDX, int LOG_N_SUB>
__global__ __launch_bounds__((LOG_N_SUB >= 11 ? 1024 : 512), StokMinBlks<LOG_N_SUB>::value) void stockham_kernel_scaled(
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
            uint64_t wv    = shoup_mul<PIDX>(v, tw[j * qs].w, tw[j * qs].w_inv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2)]     = addmod<PIDX>(u, wv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2 + p)] = submod<PIDX>(u, wv);
        }
        __syncthreads();
        src_buf = 1 - src_buf;
    }

    if (tid < half) {
        data[base + tid]        = shoup_mul<PIDX>(
            lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid)],        n_inv_w, n_inv_winv);
        data[base + tid + half] = shoup_mul<PIDX>(
            lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid + half)], n_inv_w, n_inv_winv);
    }
}

/* ─── Hadamard-fused Stockham (I14) ─────────────────────────────────────── */
/* The pointwise frequency-domain product A *= B sits between the forward and
 * inverse NTTs.  Both operands are in the same transposed-freq order (from
 * identical forward transforms), so the multiply is applied on the LOAD of
 * the inverse's FIRST stockham pass — folding the standalone hadamard_kernel
 * (its launch + a full read-B/read-A/write-A HBM round trip) into work the
 * inverse already does.  mulmod (general Solinas reduce) is used since B is
 * not a fixed twiddle.  These mirror stockham_kernel / _scaled exactly except
 * for the initial `data[i] *= b[i]` on load. Used only by the even-log_n
 * (Stockham) inverse path via launch_stockham_intt_hadamard. */
template<int PIDX, int LOG_N_SUB>
__global__ __launch_bounds__((LOG_N_SUB >= 11 ? 1024 : 512), StokMinBlks<LOG_N_SUB>::value) void stockham_kernel_had(
    uint64_t* __restrict__        data,
    const uint64_t* __restrict__  b,      /* hadamard operand B (same layout) */
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
        lds[STOK_PAD(tid)]        = mulmod<PIDX>(data[base + tid],        b[base + tid]);
        lds[STOK_PAD(tid + half)] = mulmod<PIDX>(data[base + tid + half], b[base + tid + half]);
    }
    __syncthreads();

    for (int s = 0; s < LOG_N_SUB; s++) {
        int p = 1 << s, qs = half >> s, p2 = p << 1;
        int j = tid & (p - 1), k = tid >> s;
        int dst_buf = 1 - src_buf;
        if (tid < half) {
            uint64_t u  = lds[src_buf * STOK_STRIDE_C + STOK_PAD(j + k * p)];
            uint64_t v  = lds[src_buf * STOK_STRIDE_C + STOK_PAD(j + k * p + half)];
            uint64_t wv = shoup_mul<PIDX>(v, tw[j * qs].w, tw[j * qs].w_inv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2)]     = addmod<PIDX>(u, wv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2 + p)] = submod<PIDX>(u, wv);
        }
        __syncthreads();
        src_buf = 1 - src_buf;
    }

    if (tid < half) {
        data[base + tid]        = lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid)];
        data[base + tid + half] = lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid + half)];
    }
}

/* Hadamard-fused + N^{-1}-scaled: the single-pass inverse (log_n<=10) folds
 * BOTH the hadamard (on load) and the N^{-1} scale (on store). */
template<int PIDX, int LOG_N_SUB>
__global__ __launch_bounds__((LOG_N_SUB >= 11 ? 1024 : 512), StokMinBlks<LOG_N_SUB>::value) void stockham_kernel_scaled_had(
    uint64_t* __restrict__        data,
    const uint64_t* __restrict__  b,
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
    int src_buf = 0;

    if (tid < half) {
        lds[STOK_PAD(tid)]        = mulmod<PIDX>(data[base + tid],        b[base + tid]);
        lds[STOK_PAD(tid + half)] = mulmod<PIDX>(data[base + tid + half], b[base + tid + half]);
    }
    __syncthreads();

    for (int s = 0; s < LOG_N_SUB; s++) {
        int p = 1 << s, qs = half >> s, p2 = p << 1;
        int j = tid & (p - 1), k = tid >> s;
        int dst_buf = 1 - src_buf;
        if (tid < half) {
            uint64_t u  = lds[src_buf * STOK_STRIDE_C + STOK_PAD(j + k * p)];
            uint64_t v  = lds[src_buf * STOK_STRIDE_C + STOK_PAD(j + k * p + half)];
            uint64_t wv = shoup_mul<PIDX>(v, tw[j * qs].w, tw[j * qs].w_inv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2)]     = addmod<PIDX>(u, wv);
            lds[dst_buf * STOK_STRIDE_C + STOK_PAD(j + k * p2 + p)] = submod<PIDX>(u, wv);
        }
        __syncthreads();
        src_buf = 1 - src_buf;
    }

    if (tid < half) {
        data[base + tid]        = shoup_mul<PIDX>(
            lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid)],        n_inv_w, n_inv_winv);
        data[base + tid + half] = shoup_mul<PIDX>(
            lds[src_buf * STOK_STRIDE_C + STOK_PAD(tid + half)], n_inv_w, n_inv_winv);
    }
}

/* ─── Square in-place transpose (Bailey 4-step) ─────────────────────────── */
/* Transpose the M×M row-major matrix in `data` in place (M = 2^(log_n/2),
 * always a multiple of TT_DIM here since log_n>10 ⇒ M≥64). Tiled via LDS so
 * BOTH global reads and writes are coalesced (the naive scalar-swap version
 * had one strided access per element). Each block owns the tile-PAIR
 * (bi,bj)/(bj,bi) with bj≥bi: load both tiles, write each other's transpose;
 * the diagonal tile (bi==bj) is transposed in place. LDS rows padded by 1 to
 * avoid 32-way bank conflicts. Block = TT_DIM² = 256 threads (mult. of 64). */
#define TT_DIM 16
__global__ __launch_bounds__(TT_DIM*TT_DIM, 4)
void transpose_sq_kernel(uint64_t* __restrict__ data, int M)
{
    __shared__ uint64_t sa[TT_DIM][TT_DIM + 1];
    __shared__ uint64_t sb[TT_DIM][TT_DIM + 1];
    int bi = blockIdx.y, bj = blockIdx.x;
    if (bj < bi) return;                       /* each tile-pair handled once */
    int tx = threadIdx.x, ty = threadIdx.y;
    size_t pa = (size_t)(bi*TT_DIM + ty) * M + (bj*TT_DIM + tx);   /* tile (bi,bj) */
    size_t pb = (size_t)(bj*TT_DIM + ty) * M + (bi*TT_DIM + tx);   /* tile (bj,bi) */
    sa[ty][tx] = data[pa];
    if (bi != bj) sb[ty][tx] = data[pb];
    __syncthreads();
    data[pb] = sa[tx][ty];                      /* (bj,bi) ← (bi,bj)^T */
    if (bi != bj) data[pa] = sb[tx][ty];        /* (bi,bj) ← (bj,bi)^T */
}

void launch_transpose_sq(uint64_t* data, int M, hipStream_t stream)
{
    dim3 tb(TT_DIM, TT_DIM);
    dim3 tg((unsigned)(M / TT_DIM), (unsigned)(M / TT_DIM));
    transpose_sq_kernel<<<tg, tb, 0, stream>>>(data, M);
}

/* ─── Fused transpose + cross-twiddle (I13) ─────────────────────────────── */
/* The Bailey 4-step applies a transpose T immediately adjacent to the
 * cross-twiddle X (forward: T·X; inverse: X·T). transpose_sq_kernel already
 * streams every element through LDS, so the cross-twiddle Shoup multiply is
 * folded into that same pass at no extra HBM traffic — eliminating the
 * standalone cross_twiddle_kernel launch (~11% of compute at log16).
 *
 * Forward (T then X): out[p] = in[transpose(p)] · tw_cross[p]. The multiply
 * lands on the STORE at the destination position p (∈ {pa,pb}); tw_cross[p]
 * is read coalesced (tx is the fastest index).
 *
 * Inverse (X then T): out[p] = (in·tw_inv)[transpose(p)]
 *                            = in[transpose(p)] · tw_inv[transpose(p)].
 * The multiply lands on the LOAD at the source position (pa/pb), so the
 * pre-transpose twiddle index is read coalesced; the transpose then carries
 * the scaled value to its destination. tw_inv = tw_cross_inv_scaled (N^{-1}
 * pre-baked), matching the standalone X^{-1} pass it replaces.
 *
 * Both are in-place: each block owns the tile-pair (bi,bj)/(bj,bi) (bj≥bi),
 * exactly as transpose_sq_kernel, so no scratch buffer is needed. */
template<int PIDX>
__global__ __launch_bounds__(TT_DIM*TT_DIM, 4)
void xtranspose_fwd_kernel(uint64_t* __restrict__ data,
                           const ShoupPair* __restrict__ tw_cross, int M)
{
    __shared__ uint64_t sa[TT_DIM][TT_DIM + 1];
    __shared__ uint64_t sb[TT_DIM][TT_DIM + 1];
    int bi = blockIdx.y, bj = blockIdx.x;
    if (bj < bi) return;
    int tx = threadIdx.x, ty = threadIdx.y;
    size_t pa = (size_t)(bi*TT_DIM + ty) * M + (bj*TT_DIM + tx);   /* tile (bi,bj) */
    size_t pb = (size_t)(bj*TT_DIM + ty) * M + (bi*TT_DIM + tx);   /* tile (bj,bi) */
    sa[ty][tx] = data[pa];
    if (bi != bj) sb[ty][tx] = data[pb];
    __syncthreads();
    data[pb] = shoup_mul<PIDX>(sa[tx][ty], tw_cross[pb].w, tw_cross[pb].w_inv);
    if (bi != bj)
        data[pa] = shoup_mul<PIDX>(sb[tx][ty], tw_cross[pa].w, tw_cross[pa].w_inv);
}

template<int PIDX>
__global__ __launch_bounds__(TT_DIM*TT_DIM, 4)
void xtranspose_inv_kernel(uint64_t* __restrict__ data,
                           const ShoupPair* __restrict__ tw_inv, int M)
{
    __shared__ uint64_t sa[TT_DIM][TT_DIM + 1];
    __shared__ uint64_t sb[TT_DIM][TT_DIM + 1];
    int bi = blockIdx.y, bj = blockIdx.x;
    if (bj < bi) return;
    int tx = threadIdx.x, ty = threadIdx.y;
    size_t pa = (size_t)(bi*TT_DIM + ty) * M + (bj*TT_DIM + tx);   /* tile (bi,bj) */
    size_t pb = (size_t)(bj*TT_DIM + ty) * M + (bi*TT_DIM + tx);   /* tile (bj,bi) */
    sa[ty][tx] = shoup_mul<PIDX>(data[pa], tw_inv[pa].w, tw_inv[pa].w_inv);
    if (bi != bj)
        sb[ty][tx] = shoup_mul<PIDX>(data[pb], tw_inv[pb].w, tw_inv[pb].w_inv);
    __syncthreads();
    data[pb] = sa[tx][ty];
    if (bi != bj) data[pa] = sb[tx][ty];
}

/* ─── Batched in-place transpose + cross-twiddle (recursive 4-step, 2026-06-17) ─ */
/* Identical to xtranspose_{fwd,inv}_kernel but applied to gridDim.z independent
 * square a×a sub-blocks, each a contiguous a*a-element region of `data`
 * (block base = blockIdx.z * a*a). tw_cross is the a*a cross-twiddle for the
 * inner transform size A=a*a, shared by every sub-block. This lets the size-M
 * sub-transforms of the outer 4-step (when M>2^11) be 4-stepped themselves —
 * i.e. recursive decomposition keeping every leaf NTT <= 2^11 (fits LDS).
 * Used by the 3-factor launchers (launch_stockham_3factor_*) for log_n=24,
 * in production since 2026-06-29 (32M-digit run verified 2026-06-30). */
template<int PIDX>
__global__ __launch_bounds__(TT_DIM*TT_DIM, 4)
void xtranspose_fwd_batched_kernel(uint64_t* __restrict__ data,
                                   const ShoupPair* __restrict__ tw_cross, int a)
{
    __shared__ uint64_t sa[TT_DIM][TT_DIM + 1];
    __shared__ uint64_t sb[TT_DIM][TT_DIM + 1];
    size_t blk = (size_t)blockIdx.z * ((size_t)a * (size_t)a);
    int bi = blockIdx.y, bj = blockIdx.x;
    if (bj < bi) return;
    int tx = threadIdx.x, ty = threadIdx.y;
    size_t pa = (size_t)(bi*TT_DIM + ty) * a + (bj*TT_DIM + tx);
    size_t pb = (size_t)(bj*TT_DIM + ty) * a + (bi*TT_DIM + tx);
    sa[ty][tx] = data[blk + pa];
    if (bi != bj) sb[ty][tx] = data[blk + pb];
    __syncthreads();
    data[blk + pb] = shoup_mul<PIDX>(sa[tx][ty], tw_cross[pb].w, tw_cross[pb].w_inv);
    if (bi != bj)
        data[blk + pa] = shoup_mul<PIDX>(sb[tx][ty], tw_cross[pa].w, tw_cross[pa].w_inv);
}

template<int PIDX>
__global__ __launch_bounds__(TT_DIM*TT_DIM, 4)
void xtranspose_inv_batched_kernel(uint64_t* __restrict__ data,
                                   const ShoupPair* __restrict__ tw_inv, int a)
{
    __shared__ uint64_t sa[TT_DIM][TT_DIM + 1];
    __shared__ uint64_t sb[TT_DIM][TT_DIM + 1];
    size_t blk = (size_t)blockIdx.z * ((size_t)a * (size_t)a);
    int bi = blockIdx.y, bj = blockIdx.x;
    if (bj < bi) return;
    int tx = threadIdx.x, ty = threadIdx.y;
    size_t pa = (size_t)(bi*TT_DIM + ty) * a + (bj*TT_DIM + tx);
    size_t pb = (size_t)(bj*TT_DIM + ty) * a + (bi*TT_DIM + tx);
    sa[ty][tx] = shoup_mul<PIDX>(data[blk + pa], tw_inv[pa].w, tw_inv[pa].w_inv);
    if (bi != bj)
        sb[ty][tx] = shoup_mul<PIDX>(data[blk + pb], tw_inv[pb].w, tw_inv[pb].w_inv);
    __syncthreads();
    data[blk + pb] = sa[tx][ty];
    if (bi != bj) data[blk + pa] = sb[tx][ty];
}

/* Batched PLAIN transpose (no twiddle): gridDim.z independent a×a sub-blocks.
 * Used as the initial/final inner transpose so the inner 4-step is natural-in
 * /natural-out (the F1a→T.X→F1b body alone is transposed-order). */
__global__ __launch_bounds__(TT_DIM*TT_DIM, 4)
void transpose_sq_batched_kernel(uint64_t* __restrict__ data, int a)
{
    __shared__ uint64_t sa[TT_DIM][TT_DIM + 1];
    __shared__ uint64_t sb[TT_DIM][TT_DIM + 1];
    size_t blk = (size_t)blockIdx.z * ((size_t)a * (size_t)a);
    int bi = blockIdx.y, bj = blockIdx.x;
    if (bj < bi) return;
    int tx = threadIdx.x, ty = threadIdx.y;
    size_t pa = (size_t)(bi*TT_DIM + ty) * a + (bj*TT_DIM + tx);
    size_t pb = (size_t)(bj*TT_DIM + ty) * a + (bi*TT_DIM + tx);
    sa[ty][tx] = data[blk + pa];
    if (bi != bj) sb[ty][tx] = data[blk + pb];
    __syncthreads();
    data[blk + pb] = sa[tx][ty];
    if (bi != bj) data[blk + pa] = sb[tx][ty];
}

/* Hadamard-fused batched transpose (I14 analogue for the 3-factor inverse):
 * folds the pointwise A*=B into the inverse's first inner T_init, which already
 * reads+writes the data — eliminating the standalone hadamard_kernel pass and its
 * extra HBM read-A/read-B/write-A. Valid because the data entering the inverse and
 * b[] are BOTH in natural (forward-output) order, so the multiply is index-aligned
 * on the LOAD (before the transpose moves the element). mulmod (general Solinas
 * reduce) since b is not a fixed twiddle. */
template<int PIDX>
__global__ __launch_bounds__(TT_DIM*TT_DIM, 4)
void transpose_sq_batched_had_kernel(uint64_t* __restrict__ data,
                                     const uint64_t* __restrict__ b, int a)
{
    __shared__ uint64_t sa[TT_DIM][TT_DIM + 1];
    __shared__ uint64_t sb[TT_DIM][TT_DIM + 1];
    size_t blk = (size_t)blockIdx.z * ((size_t)a * (size_t)a);
    int bi = blockIdx.y, bj = blockIdx.x;
    if (bj < bi) return;
    int tx = threadIdx.x, ty = threadIdx.y;
    size_t pa = (size_t)(bi*TT_DIM + ty) * a + (bj*TT_DIM + tx);
    size_t pb = (size_t)(bj*TT_DIM + ty) * a + (bi*TT_DIM + tx);
    sa[ty][tx] = mulmod<PIDX>(data[blk + pa], b[blk + pa]);
    if (bi != bj) sb[ty][tx] = mulmod<PIDX>(data[blk + pb], b[blk + pb]);
    __syncthreads();
    data[blk + pb] = sa[tx][ty];
    if (bi != bj) data[blk + pa] = sb[tx][ty];
}

/* ─── Out-of-place rectangular transpose (odd-log_n 4-step, I11) ─────────── */
/* out = in^T:  in is R×C row-major (in[r*C+c]); out is C×R (out[c*R+r]).
 * Tiled+padded for coalesced read and write. R,C are powers of two and
 * multiples of TT_DIM here (odd log_n>10 ⇒ N1≥32, N2≥64). Used by the
 * rectangular Bailey 4-step where N1≠N2 (in-place swap is not possible). */
__global__ __launch_bounds__(TT_DIM*TT_DIM, 4)
void transpose_rect_kernel(uint64_t* __restrict__ out,
                           const uint64_t* __restrict__ in, int R, int C)
{
    __shared__ uint64_t tile[TT_DIM][TT_DIM + 1];
    int tx = threadIdx.x, ty = threadIdx.y;
    int r = blockIdx.y * TT_DIM + ty;   /* row in `in`  [0,R) */
    int c = blockIdx.x * TT_DIM + tx;   /* col in `in`  [0,C) */
    if (r < R && c < C) tile[ty][tx] = in[(size_t)r * C + c];
    __syncthreads();
    int orow = blockIdx.x * TT_DIM + ty;  /* row in `out` [0,C) */
    int ocol = blockIdx.y * TT_DIM + tx;  /* col in `out` [0,R) */
    if (orow < C && ocol < R) out[(size_t)orow * R + ocol] = tile[tx][ty];
}

static void launch_transpose_rect(uint64_t* out, const uint64_t* in,
                                  int R, int C, hipStream_t stream)
{
    dim3 tb(TT_DIM, TT_DIM);
    dim3 tg((unsigned)((C + TT_DIM - 1) / TT_DIM),
            (unsigned)((R + TT_DIM - 1) / TT_DIM));
    transpose_rect_kernel<<<tg, tb, 0, stream>>>(out, in, R, C);
}

/* I8: thread count for a stockham_kernel sub-NTT of size 2^log_n_sub. Only
 * `half = 2^(log_n_sub-1)` threads do work (the kernel guards tid<half), so
 * launching the fixed BLOCK_SIZE=512 left up to 6 of 8 wave64 idle at
 * log_n_sub<10 (e.g. log_n=16 4-step ⇒ sub=8 ⇒ half=128 ⇒ 75% idle). Size
 * the block to the work, clamped to [64, BLOCK_SIZE] (64 = one wave64).
 * C1: LDS is now compile-time per LOG_N_SUB, not a fixed STOK_STRIDE. */
static inline int stok_block(int log_n_sub)
{
    int h = 1 << (log_n_sub - 1);
    if (h < 64)   h = 64;
    /* stockham_kernel is one-butterfly-per-thread (needs `half` threads). At
     * LOG_N_SUB=11 (log_n=22) half=1024; clamping to BLOCK_SIZE=512 left half the
     * butterflies uncomputed → corrupt product (the log_n=22 bug). gfx1030 allows
     * 1024 threads/block, so clamp at 1024. LOG_N_SUB<=10 (h<=512) is unaffected. */
    if (h > 1024) h = 1024;
    return h;
}

/* C1: dispatch helper — selects stockham_kernel<PIDX,LOG_N_SUB> specialization.
 * Used by launch_stockham_large for the forward 4-step pass.
 * log_n_sub ranges 6..11 for even log_n in [12..22]. */
template<int PIDX>
static void launch_stok(uint64_t* data, const ShoupPair* tw,
                        int n_blocks, int log_n_sub, hipStream_t stream)
{
    int blk = stok_block(log_n_sub);
    switch (log_n_sub) {
        case  1: stockham_kernel<PIDX, 1><<<n_blocks, blk, 0, stream>>>(data, tw); break;
        case  2: stockham_kernel<PIDX, 2><<<n_blocks, blk, 0, stream>>>(data, tw); break;
        case  3: stockham_kernel<PIDX, 3><<<n_blocks, blk, 0, stream>>>(data, tw); break;
        case  4: stockham_kernel<PIDX, 4><<<n_blocks, blk, 0, stream>>>(data, tw); break;
        case  5: stockham_kernel<PIDX, 5><<<n_blocks, blk, 0, stream>>>(data, tw); break;
        case  6: stockham_kernel<PIDX, 6><<<n_blocks, blk, 0, stream>>>(data, tw); break;
        case  7: stockham_kernel<PIDX, 7><<<n_blocks, blk, 0, stream>>>(data, tw); break;
        case  8: stockham_kernel<PIDX, 8><<<n_blocks, blk, 0, stream>>>(data, tw); break;
        case  9: stockham_kernel<PIDX, 9><<<n_blocks, blk, 0, stream>>>(data, tw); break;
        case 10: stockham_kernel<PIDX,10><<<n_blocks, blk, 0, stream>>>(data, tw); break;
        case 11: stockham_kernel<PIDX,11><<<n_blocks, blk, 0, stream>>>(data, tw); break;
        default:
            fprintf(stderr, "launch_stok: unsupported log_n_sub=%d\n", log_n_sub);
            abort();
    }
}

/* C1: dispatch helper for the Hadamard-fused INTT first pass. */
template<int PIDX>
static void launch_stok_had(uint64_t* data, const uint64_t* b,
                            const ShoupPair* tw,
                            int n_blocks, int log_n_sub, hipStream_t stream)
{
    int blk = stok_block(log_n_sub);
    switch (log_n_sub) {
        case  6: stockham_kernel_had<PIDX, 6><<<n_blocks, blk, 0, stream>>>(data, b, tw); break;
        case  7: stockham_kernel_had<PIDX, 7><<<n_blocks, blk, 0, stream>>>(data, b, tw); break;
        case  8: stockham_kernel_had<PIDX, 8><<<n_blocks, blk, 0, stream>>>(data, b, tw); break;
        case  9: stockham_kernel_had<PIDX, 9><<<n_blocks, blk, 0, stream>>>(data, b, tw); break;
        case 10: stockham_kernel_had<PIDX,10><<<n_blocks, blk, 0, stream>>>(data, b, tw); break;
        case 11: stockham_kernel_had<PIDX,11><<<n_blocks, blk, 0, stream>>>(data, b, tw); break;
        default:
            fprintf(stderr, "launch_stok_had: unsupported log_n_sub=%d\n", log_n_sub);
            abort();
    }
}

/* C1: dispatch helpers for scaled variants (log_n <= 10 single-block INTT). */
template<int PIDX>
static void launch_stok_scaled(uint64_t* data, const ShoupPair* tw,
                               int n_blocks, int log_n_sub,
                               uint64_t n_inv_w, uint64_t n_inv_winv,
                               hipStream_t stream)
{
    int blk = stok_block(log_n_sub);
    switch (log_n_sub) {
        case  1: stockham_kernel_scaled<PIDX, 1><<<n_blocks,blk,0,stream>>>(data,tw,n_inv_w,n_inv_winv); break;
        case  2: stockham_kernel_scaled<PIDX, 2><<<n_blocks,blk,0,stream>>>(data,tw,n_inv_w,n_inv_winv); break;
        case  3: stockham_kernel_scaled<PIDX, 3><<<n_blocks,blk,0,stream>>>(data,tw,n_inv_w,n_inv_winv); break;
        case  4: stockham_kernel_scaled<PIDX, 4><<<n_blocks,blk,0,stream>>>(data,tw,n_inv_w,n_inv_winv); break;
        case  5: stockham_kernel_scaled<PIDX, 5><<<n_blocks,blk,0,stream>>>(data,tw,n_inv_w,n_inv_winv); break;
        case  6: stockham_kernel_scaled<PIDX, 6><<<n_blocks,blk,0,stream>>>(data,tw,n_inv_w,n_inv_winv); break;
        case  7: stockham_kernel_scaled<PIDX, 7><<<n_blocks,blk,0,stream>>>(data,tw,n_inv_w,n_inv_winv); break;
        case  8: stockham_kernel_scaled<PIDX, 8><<<n_blocks,blk,0,stream>>>(data,tw,n_inv_w,n_inv_winv); break;
        case  9: stockham_kernel_scaled<PIDX, 9><<<n_blocks,blk,0,stream>>>(data,tw,n_inv_w,n_inv_winv); break;
        case 10: stockham_kernel_scaled<PIDX,10><<<n_blocks,blk,0,stream>>>(data,tw,n_inv_w,n_inv_winv); break;
        default: fprintf(stderr,"launch_stok_scaled: bad log_n_sub=%d\n",log_n_sub); abort();
    }
}

template<int PIDX>
static void launch_stok_scaled_had(uint64_t* data, const uint64_t* b,
                                   const ShoupPair* tw,
                                   int n_blocks, int log_n_sub,
                                   uint64_t n_inv_w, uint64_t n_inv_winv,
                                   hipStream_t stream)
{
    int blk = stok_block(log_n_sub);
    switch (log_n_sub) {
        case  1: stockham_kernel_scaled_had<PIDX, 1><<<n_blocks,blk,0,stream>>>(data,b,tw,n_inv_w,n_inv_winv); break;
        case  2: stockham_kernel_scaled_had<PIDX, 2><<<n_blocks,blk,0,stream>>>(data,b,tw,n_inv_w,n_inv_winv); break;
        case  3: stockham_kernel_scaled_had<PIDX, 3><<<n_blocks,blk,0,stream>>>(data,b,tw,n_inv_w,n_inv_winv); break;
        case  4: stockham_kernel_scaled_had<PIDX, 4><<<n_blocks,blk,0,stream>>>(data,b,tw,n_inv_w,n_inv_winv); break;
        case  5: stockham_kernel_scaled_had<PIDX, 5><<<n_blocks,blk,0,stream>>>(data,b,tw,n_inv_w,n_inv_winv); break;
        case  6: stockham_kernel_scaled_had<PIDX, 6><<<n_blocks,blk,0,stream>>>(data,b,tw,n_inv_w,n_inv_winv); break;
        case  7: stockham_kernel_scaled_had<PIDX, 7><<<n_blocks,blk,0,stream>>>(data,b,tw,n_inv_w,n_inv_winv); break;
        case  8: stockham_kernel_scaled_had<PIDX, 8><<<n_blocks,blk,0,stream>>>(data,b,tw,n_inv_w,n_inv_winv); break;
        case  9: stockham_kernel_scaled_had<PIDX, 9><<<n_blocks,blk,0,stream>>>(data,b,tw,n_inv_w,n_inv_winv); break;
        case 10: stockham_kernel_scaled_had<PIDX,10><<<n_blocks,blk,0,stream>>>(data,b,tw,n_inv_w,n_inv_winv); break;
        default: fprintf(stderr,"launch_stok_scaled_had: bad log_n_sub=%d\n",log_n_sub); abort();
    }
}

/* ─── 4-step Stockham launcher (Bailey, transpose-based) ────────────────── */
/* Correct DFT_N for N=N1·N2 (square: N1==N2==M, even log_n). The earlier
 * transpose-FREE version DFT'd the LOW index in BOTH passes, leaving the
 * high index untransformed — invertible (roundtrip OK) but NOT a DFT, so
 * pointwise multiply != convolution (the bug behind the I1 hang, exposed by
 * test_ntt_convolution). Bailey's algorithm:
 *   T  transpose            (original columns become contiguous rows)
 *   F1 contiguous N1-point DFT on each row  (= the column DFTs)
 *   T  transpose back
 *   X  cross-twiddle  data[r·N2+c] *= ω_N^{r·c}
 *   F2 contiguous N2-point DFT on each row  (the row DFTs)
 * Output is in transposed frequency order — a consistent permutation that
 * launch_stockham_intt inverts (roundtrip = identity). tw_stok layout
 * unchanged: [0..N1/2) row table, [N1/2..) col table. */
template<int PIDX>
static void launch_stockham_large(
    uint64_t*        data,
    const ShoupPair* tw_stok,
    const ShoupPair* tw_cross,
    int              log_n,
    hipStream_t      stream)
{
    int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
    int N1 = 1 << log_n1, N2 = 1 << log_n2;   /* N1==N2==M for even log_n */
    int M  = N1;
    dim3 tb(TT_DIM, TT_DIM);
    dim3 tg((unsigned)(M / TT_DIM), (unsigned)(M / TT_DIM));
    /* I15: initial T eliminated — scatter writes data already transposed.
     * C1: launch_stok selects the LOG_N_SUB specialization with exact LDS. */
    launch_stok<PIDX>(data, tw_stok, N2, log_n1, stream);                   /* F1  */
    xtranspose_fwd_kernel<PIDX><<<tg, tb, 0, stream>>>(data, tw_cross, M);   /* T·X (fused) */
    launch_stok<PIDX>(data, tw_stok + N1 / 2, N1, log_n2, stream);          /* F2  */
}

/* ─── 3-factor (recursive 4-step) NTT for log_n=24 ─────────────────────── */
/* Outer split: 24 = 12+12 (N_outer=4096 columns of 4096-point sub-NTTs).
 * Inner split: 12 = 6+6  (N_inner=64  per 4096-point sub-NTT).
 * All leaf NTTs are 64-point, kept in LDS (LOG_N_SUB=6, 32 threads < 1024).
 *
 * tw_leaf:         Stockham table for log_n=12 built as 6+6 (64 entries).
 *                  [0..31]  = F1 leaf twiddles (omega_6)
 *                  [32..63] = F2 leaf twiddles (same omega_6, square split)
 * tw_cross_inner:  inner 64x64 cross-twiddle (4096 entries, omega_{4096}^{rc}).
 * tw_cross_outer:  outer 4096x4096 cross-twiddle (2^24 entries, omega_{2^24}^{rc}).
 *
 * Factor accounting (each Stockham sub-NTT outputs N_sub * DFT, unnormalized):
 *   Forward: two inner Stockham levels x 64 x 64 = 4096 per outer half
 *            -> per-element NTT output = N * (correct DFT coefficient)
 *   Inverse: matching 4096^2 factor cancelled by N^{-1} in tw_cross_outer_inv_scaled */
template<int PIDX>
void launch_stockham_3factor_ntt(
    uint64_t*        data,
    const ShoupPair* tw_leaf,          /* log_n=12 Stockham table (64 entries) */
    const ShoupPair* tw_cross_inner,   /* inner 64x64 cross-twiddle (4096 entries) */
    const ShoupPair* tw_cross_outer,   /* outer 4096x4096 cross-twiddle (2^24 entries) */
    hipStream_t      stream)
{
    enum { LOG_OUTER = 12, LOG_LEAF = 6 };
    int N_outer = 1 << LOG_OUTER;   /* 4096: outer N1=N2 */
    int N_inner = 1 << LOG_LEAF;    /* 64:   inner N1i=N2i */
    dim3 tb(TT_DIM, TT_DIM);
    /* outer 4096x4096 transpose: (4096/16) x (4096/16) blocks */
    dim3 tg_out((unsigned)(N_outer/TT_DIM), (unsigned)(N_outer/TT_DIM));
    /* inner batched 64x64 transpose: (64/16)x(64/16) grid, N_outer z-slices */
    dim3 tg_in((unsigned)(N_inner/TT_DIM), (unsigned)(N_inner/TT_DIM),
               (unsigned)N_outer);

    /* Outer T_init: the four-step's FIRST sub-DFT must run over the OUTER (strided
     * by N_outer) index, but launch_stok works on contiguous blocks — so transpose
     * the full N_outer×N_outer matrix first to make the outer index contiguous.
     * Without this, F1 DFTs the wrong (inner) index first; single-outer-block and
     * sparse inputs mask it (trivial outer twiddle), dense multi-block exposes it. */
    launch_transpose_sq(data, N_outer, stream);

    /* F1: N_outer independent 4096-point sub-NTTs via inner 4-step (6+6).
     * Each inner 4-step is made natural-in/natural-out by a batched transpose
     * BEFORE (cancel input perm) and AFTER (cancel output perm) the
     * leaf→T.X→leaf body — verified by conv_F1 + test_F1_order. Without T_final
     * F1's output is in transposed DFT order, which mis-aligns the outer T.X and
     * corrupts multi-outer-block (full-size) convolutions. */
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);     /* F1 T_init */
    launch_stok<PIDX>(data, tw_leaf, N_outer * N_inner, LOG_LEAF, stream);
    xtranspose_fwd_batched_kernel<PIDX><<<tg_in, tb, 0, stream>>>(
        data, tw_cross_inner, N_inner);
    launch_stok<PIDX>(data, tw_leaf + N_inner/2, N_outer * N_inner, LOG_LEAF, stream);
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);     /* F1 T_final */

    /* Outer T.X: single 4096x4096 transpose + outer cross-twiddle */
    xtranspose_fwd_kernel<PIDX><<<tg_out, tb, 0, stream>>>(
        data, tw_cross_outer, N_outer);

    /* F2: N_outer independent 4096-point sub-NTTs via inner 4-step (same). */
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);     /* F2 T_init */
    launch_stok<PIDX>(data, tw_leaf, N_outer * N_inner, LOG_LEAF, stream);
    xtranspose_fwd_batched_kernel<PIDX><<<tg_in, tb, 0, stream>>>(
        data, tw_cross_inner, N_inner);
    launch_stok<PIDX>(data, tw_leaf + N_inner/2, N_outer * N_inner, LOG_LEAF, stream);
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);     /* F2 T_final */
}

/* DEBUG-ONLY: the INNER 4-step alone (F1), exposed to isolate it from the outer
 * stage + F2. Computes N_outer independent unnormalized 4096-point DFTs (one per
 * contiguous 4096-element block), as 64x64 (6+6). Forward and inverse mirror. */
template<int PIDX>
void launch_stockham_3factor_F1_fwd(uint64_t* data, const ShoupPair* tw_leaf,
                                    const ShoupPair* tw_cross_inner, hipStream_t stream)
{
    enum { LOG_OUTER = 12, LOG_LEAF = 6 };
    int N_outer = 1 << LOG_OUTER, N_inner = 1 << LOG_LEAF;
    dim3 tb(TT_DIM, TT_DIM);
    dim3 tg_in((unsigned)(N_inner/TT_DIM), (unsigned)(N_inner/TT_DIM), (unsigned)N_outer);
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);  /* T_init (cancel input perm) */
    launch_stok<PIDX>(data, tw_leaf, N_outer * N_inner, LOG_LEAF, stream);
    xtranspose_fwd_batched_kernel<PIDX><<<tg_in, tb, 0, stream>>>(data, tw_cross_inner, N_inner);
    launch_stok<PIDX>(data, tw_leaf + N_inner/2, N_outer * N_inner, LOG_LEAF, stream);
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);  /* T_final (cancel output perm -> natural order) */
}
template<int PIDX>
void launch_stockham_3factor_F1_inv(uint64_t* data, const ShoupPair* tw_leaf_inv,
                                    const ShoupPair* tw_cross_inner_inv, hipStream_t stream)
{
    enum { LOG_OUTER = 12, LOG_LEAF = 6 };
    int N_outer = 1 << LOG_OUTER, N_inner = 1 << LOG_LEAF;
    dim3 tb(TT_DIM, TT_DIM);
    dim3 tg_in((unsigned)(N_inner/TT_DIM), (unsigned)(N_inner/TT_DIM), (unsigned)N_outer);
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);  /* T_init (mirror of fwd T_final) */
    launch_stok<PIDX>(data, tw_leaf_inv + N_inner/2, N_outer * N_inner, LOG_LEAF, stream);
    xtranspose_inv_batched_kernel<PIDX><<<tg_in, tb, 0, stream>>>(data, tw_cross_inner_inv, N_inner);
    launch_stok<PIDX>(data, tw_leaf_inv, N_outer * N_inner, LOG_LEAF, stream);
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);  /* T_final (mirror of fwd T_init) */
}

/* Inverse 3-factor NTT (plain, no Hadamard fusion).
 * tw_cross_outer_inv_scaled: tw_cross_outer with N^{-1} = (2^24)^{-1} pre-multiplied.
 *   (= d_tw_cross_inv_scaled for log_n=24; inner cross-twiddle is unscaled.) */
template<int PIDX>
void launch_stockham_3factor_intt(
    uint64_t*        data,
    const ShoupPair* tw_leaf_inv,               /* log_n=12 inverse Stockham (64 entries) */
    const ShoupPair* tw_cross_inner_inv,        /* inner 64x64 INVERSE cross-twiddle (omega^{-rc}, unscaled) */
    const ShoupPair* tw_cross_outer_inv_scaled, /* outer with N^{-1} */
    hipStream_t      stream)
{
    enum { LOG_OUTER = 12, LOG_LEAF = 6 };
    int N_outer = 1 << LOG_OUTER;
    int N_inner = 1 << LOG_LEAF;
    dim3 tb(TT_DIM, TT_DIM);
    dim3 tg_out((unsigned)(N_outer/TT_DIM), (unsigned)(N_outer/TT_DIM));
    dim3 tg_in((unsigned)(N_inner/TT_DIM), (unsigned)(N_inner/TT_DIM),
               (unsigned)N_outer);

    /* F2^{-1}: inverse of the 4096-point F2 sub-NTTs. Each inner 4-step is
     * bracketed by batched transposes (mirror of the forward's T_init/T_final). */
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);  /* inner T_init (mirror) */
    launch_stok<PIDX>(data, tw_leaf_inv + N_inner/2, N_outer * N_inner, LOG_LEAF, stream);
    xtranspose_inv_batched_kernel<PIDX><<<tg_in, tb, 0, stream>>>(
        data, tw_cross_inner_inv, N_inner);
    launch_stok<PIDX>(data, tw_leaf_inv, N_outer * N_inner, LOG_LEAF, stream);
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);  /* inner T_final (mirror) */

    /* Outer T^{-1}.X^{-1}: single 4096x4096 inverse transpose + scaled cross-tw. */
    xtranspose_inv_kernel<PIDX><<<tg_out, tb, 0, stream>>>(
        data, tw_cross_outer_inv_scaled, N_outer);

    /* F1^{-1}: inverse of the 4096-point F1 sub-NTTs. */
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);  /* inner T_init (mirror) */
    launch_stok<PIDX>(data, tw_leaf_inv + N_inner/2, N_outer * N_inner, LOG_LEAF, stream);
    xtranspose_inv_batched_kernel<PIDX><<<tg_in, tb, 0, stream>>>(
        data, tw_cross_inner_inv, N_inner);
    launch_stok<PIDX>(data, tw_leaf_inv, N_outer * N_inner, LOG_LEAF, stream);
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);  /* inner T_final (mirror) */
    launch_transpose_sq(data, N_outer, stream);   /* outer T_final (mirror of fwd outer T_init) */
}

/* Hadamard-fused inverse 3-factor NTT: pointwise data[i] *= b[i] is folded
 * into the first Stockham sub-NTT pass (F2b^{-1}), eliminating the separate
 * hadamard_kernel launch and its HBM round-trip (I14 analogue for 3-factor). */
template<int PIDX>
void launch_stockham_3factor_intt_hadamard(
    uint64_t*        data,
    const uint64_t*  b,
    const ShoupPair* tw_leaf_inv,
    const ShoupPair* tw_cross_inner_inv,        /* inner 64x64 INVERSE cross-twiddle (omega^{-rc}, unscaled) */
    const ShoupPair* tw_cross_outer_inv_scaled,
    hipStream_t      stream)
{
    enum { LOG_OUTER = 12, LOG_LEAF = 6 };
    int N_outer = 1 << LOG_OUTER;
    int N_inner = 1 << LOG_LEAF;
    dim3 tb(TT_DIM, TT_DIM);
    dim3 tg_out((unsigned)(N_outer/TT_DIM), (unsigned)(N_outer/TT_DIM));
    dim3 tg_in((unsigned)(N_inner/TT_DIM), (unsigned)(N_inner/TT_DIM),
               (unsigned)N_outer);

    /* Hadamard FUSED into the inverse's first kernel: the F2^{-1} inner T_init is a
     * batched transpose that reads the data in NATURAL (forward-output) order — the
     * same order as b — so the pointwise A*=B is folded onto its LOAD, eliminating
     * the standalone hadamard_kernel pass (I14 analogue). */
    transpose_sq_batched_had_kernel<PIDX><<<tg_in, tb, 0, stream>>>(data, b, N_inner);  /* hadamard + F2^{-1} inner T_init */
    launch_stok<PIDX>(data, tw_leaf_inv + N_inner/2, N_outer * N_inner, LOG_LEAF, stream);
    xtranspose_inv_batched_kernel<PIDX><<<tg_in, tb, 0, stream>>>(
        data, tw_cross_inner_inv, N_inner);
    launch_stok<PIDX>(data, tw_leaf_inv, N_outer * N_inner, LOG_LEAF, stream);
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);  /* inner T_final (mirror) */

    xtranspose_inv_kernel<PIDX><<<tg_out, tb, 0, stream>>>(
        data, tw_cross_outer_inv_scaled, N_outer);

    /* F1^{-1} (T_init/T_final bracketed inner 4-step). */
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);  /* inner T_init (mirror) */
    launch_stok<PIDX>(data, tw_leaf_inv + N_inner/2, N_outer * N_inner, LOG_LEAF, stream);
    xtranspose_inv_batched_kernel<PIDX><<<tg_in, tb, 0, stream>>>(
        data, tw_cross_inner_inv, N_inner);
    launch_stok<PIDX>(data, tw_leaf_inv, N_outer * N_inner, LOG_LEAF, stream);
    transpose_sq_batched_kernel<<<tg_in, tb, 0, stream>>>(data, N_inner);  /* inner T_final (mirror) */
    launch_transpose_sq(data, N_outer, stream);   /* outer T_final (mirror of fwd outer T_init) */
}

/* ─── Public Stockham NTT launcher ──────────────────────────────────────── */
template<int PIDX>
void launch_stockham_ntt(
    uint64_t*        data,
    const ShoupPair* tw_stok,
    const ShoupPair* tw_cross,
    int              log_n,
    hipStream_t      stream)
{
    /* 4-step path (log_n>10) requires square decomposition; cross_twiddle
     * indexing + the missing transpose between the two stockham passes
     * only line up when N1==N2 (even log_n). Callers must dispatch by
     * parity (Stockham for even, CT-DIT launch_ntt for odd). */
    if (log_n > 10 && (log_n & 1)) {
        fprintf(stderr,
            "launch_stockham_ntt: log_n=%d odd; 4-step path requires "
            "even log_n (cross_twiddle_kernel layout constraint)\n", log_n);
        abort();
    }
    int N = 1 << log_n;
    if (log_n <= 10) {
        int n_blocks = (N < 1024) ? 1 : N / 1024;
        /* C1: use LOG_N_SUB dispatch; single-block path, log_n IS the sub-size. */
        launch_stok<PIDX>(data, tw_stok, n_blocks, log_n, stream);
    } else {
        launch_stockham_large<PIDX>(data, tw_stok, tw_cross, log_n, stream);
    }
}

/* ─── Public Stockham INTT launcher ─────────────────────────────────────── */
/* For log_n <= 10: fuses N^{-1} into write-back via stockham_kernel_scaled. */
/* For log_n > 10:  N^{-1} is pre-baked in tw_cross_inv_scaled, same as     */
/*                  launch_intt_fused — passes through cross_twiddle_kernel. */
template<int PIDX>
void launch_stockham_intt(
    uint64_t*        data,
    uint64_t         n_inv,
    const ShoupPair* tw_stok_inv,
    const ShoupPair* tw_cross_inv_scaled,
    int              log_n,
    hipStream_t      stream)
{
    if (log_n > 10 && (log_n & 1)) {
        fprintf(stderr,
            "launch_stockham_intt: log_n=%d odd; 4-step path requires "
            "even log_n (cross_twiddle_kernel layout constraint)\n", log_n);
        abort();
    }
    if (log_n <= 10) {
        ShoupPair nsp  = make_shoup(n_inv, PRIMES[PIDX]);
        int N          = 1 << log_n;
        int n_blocks   = (N < 1024) ? 1 : N / 1024;
        /* C1: LOG_N_SUB dispatch for exact LDS allocation. */
        launch_stok_scaled<PIDX>(data, tw_stok_inv, n_blocks, log_n,
                                 nsp.w, nsp.w_inv, stream);
    } else {
        /* Exact inverse of the Bailey forward (launch_stockham_large):
         * forward = T·F1·T·X·F2, so inverse = F2^{-1}·X^{-1}·T·F1^{-1}·T
         * applied in REVERSE order with inverse tables. N^{-1} is pre-baked
         * into tw_cross_inv_scaled (X^{-1}). Block counts mirror the forward:
         * F2 used N1 blocks (log_n2), F1 used N2 blocks (log_n1). */
        int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
        int N1 = 1 << log_n1, N2 = 1 << log_n2;   /* N1==N2==M */
        int M  = N1;
        dim3 tb(TT_DIM, TT_DIM);
        dim3 tg((unsigned)(M / TT_DIM), (unsigned)(M / TT_DIM));
        /* C1: use LOG_N_SUB dispatch helpers. */
        launch_stok<PIDX>(data, tw_stok_inv + N1/2, N1, log_n2, stream); /* F2^{-1} */
        xtranspose_inv_kernel<PIDX><<<tg, tb, 0, stream>>>(
            data, tw_cross_inv_scaled, M);                  /* X^{-1}·T (fused) */
        launch_stok<PIDX>(data, tw_stok_inv, N2, log_n1, stream);        /* F1^{-1} */
        /* I15: final T eliminated — gather reads data already transposed. */
    }
}

/* ─── Hadamard-fused inverse Stockham (I14) ─────────────────────────────── */
/* Identical to launch_stockham_intt except the inverse's FIRST stockham pass
 * multiplies data *= b (the frequency-domain Hadamard product) on load,
 * folding away the standalone hadamard_kernel.  Even log_n only (the Stockham
 * path); the caller (ntt_mul.hip) keeps the standalone hadamard for the CT-DIT
 * odd path.  data and b must be in the same transposed-freq order. */
template<int PIDX>
void launch_stockham_intt_hadamard(
    uint64_t*        data,
    const uint64_t*  b,
    uint64_t         n_inv,
    const ShoupPair* tw_stok_inv,
    const ShoupPair* tw_cross_inv_scaled,
    int              log_n,
    hipStream_t      stream)
{
    if (log_n <= 10) {
        ShoupPair nsp  = make_shoup(n_inv, PRIMES[PIDX]);
        int N          = 1 << log_n;
        int n_blocks   = (N < 1024) ? 1 : N / 1024;
        /* C1: LOG_N_SUB dispatch for exact LDS allocation. */
        launch_stok_scaled_had<PIDX>(data, b, tw_stok_inv, n_blocks, log_n,
                                    nsp.w, nsp.w_inv, stream);
    } else {
        int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
        int N1 = 1 << log_n1, N2 = 1 << log_n2;
        int M  = N1;
        dim3 tb(TT_DIM, TT_DIM);
        dim3 tg((unsigned)(M / TT_DIM), (unsigned)(M / TT_DIM));
        /* C1: launch_stok_had / launch_stok select LOG_N_SUB specialization. */
        launch_stok_had<PIDX>(data, b, tw_stok_inv + N1/2,
                              N1, log_n2, stream);           /* (A∘B)·F2^{-1} */
        xtranspose_inv_kernel<PIDX><<<tg, tb, 0, stream>>>(
            data, tw_cross_inv_scaled, M);                  /* X^{-1}·T (fused) */
        launch_stok<PIDX>(data, tw_stok_inv, N2, log_n1, stream); /* F1^{-1}  */
        /* I15: final T eliminated — gather reads data already transposed. */
    }
}

/* ─── Rectangular Bailey 4-step for ODD log_n>10 (I11) ───────────────────── */
/* For odd log_n the split is N1=2^(log_n/2), N2=2*N1 (N1≠N2), so the square
 * in-place transpose can't be used — we transpose out-of-place through a
 * caller-provided scratch `scr` (size N). Same Bailey structure as the square
 * path (T·F1·T·X·F2), and the SAME twiddle tables (make_tw_stockham builds
 * N1/2+N2/2 entries; make_tw_cross builds N1*N2) — only the transpose differs.
 * Brings odd-log_n sizes (e.g. d=1e6 ⇒ log_n=19) from the memory-bound CT-DIT
 * path up to the LDS-fused Stockham path. */
template<int PIDX>
void launch_stockham_ntt_rect(
    uint64_t* data, uint64_t* scr,
    const ShoupPair* tw_stok, const ShoupPair* tw_cross,
    int log_n, hipStream_t stream)
{
    int log_n1 = log_n / 2, log_n2 = log_n - log_n1;   /* log_n2 = log_n1+1 */
    int N1 = 1 << log_n1, N2 = 1 << log_n2;            /* N2 = 2*N1         */
    dim3 grid2((N2 + 255) / 256, N1);
    launch_transpose_rect(scr, data, N1, N2, stream);                 /* T1: data(N1xN2)->scr(N2xN1) */
    launch_stok<PIDX>(scr, tw_stok, N2, log_n1, stream);             /* F1 (size N1) */
    launch_transpose_rect(data, scr, N2, N1, stream);                 /* T2: scr->data(N1xN2)         */
    cross_twiddle_kernel<PIDX><<<grid2, 256, 0, stream>>>(data, tw_cross, N2);          /* X            */
    launch_stok<PIDX>(data, tw_stok + N1 / 2, N1, log_n2, stream);   /* F2 (size N2) */
}

template<int PIDX>
void launch_stockham_intt_rect(
    uint64_t* data, uint64_t* scr, uint64_t n_inv,
    const ShoupPair* tw_stok_inv, const ShoupPair* tw_cross_inv_scaled,
    int log_n, hipStream_t stream)
{
    (void)n_inv;   /* N^{-1} pre-baked into tw_cross_inv_scaled */
    int log_n1 = log_n / 2, log_n2 = log_n - log_n1;
    int N1 = 1 << log_n1, N2 = 1 << log_n2;
    dim3 grid2((N2 + 255) / 256, N1);
    /* Exact reverse of the forward: F2^{-1}·X^{-1}·T2^{-1}·F1^{-1}·T1^{-1}. */
    launch_stok<PIDX>(data, tw_stok_inv + N1/2, N1, log_n2, stream); /* F2^{-1}  */
    cross_twiddle_kernel<PIDX><<<grid2, 256, 0, stream>>>(
        data, tw_cross_inv_scaled, N2);                               /* X^{-1}   */
    launch_transpose_rect(scr, data, N1, N2, stream);                 /* T2^{-1}: data(N1xN2)->scr(N2xN1) */
    launch_stok<PIDX>(scr, tw_stok_inv, N2, log_n1, stream);         /* F1^{-1}  */
    launch_transpose_rect(data, scr, N2, N1, stream);                 /* T1^{-1}: scr->data(N1xN2)        */
}

/* ─── Montgomery NTT kernel ──────────────────────────────────────────────── */
/* Identical butterfly to ntt_kernel but twiddle table stores one uint64_t   */
/* per entry (w * R mod p) instead of a ShoupPair (16 bytes). This halves    */
/* twiddle HBM bandwidth: 8 B/entry vs 16 B/entry (Shoup).                  */
/* Correctness: montgomery_mul(data, w*R mod p) = data * w mod p             */
/* because mont_mul(a, b) = a*b*R^{-1} mod p and b = w*R gives a*w mod p.  */
/* Cost: ~10 gfx942 instructions vs Shoup's ~7. Favoured when HBM bandwidth  */
/* dominates (N >= 2^20 on MI300A with 4 TB/s HBM3).                        */
template<int PIDX>
__global__ NTT_LAUNCH_BOUNDS void mont_ntt_kernel(
    uint64_t* __restrict__       data,
    const uint64_t* __restrict__ tw_mont,  /* w * R mod p per entry, 8 B each */
    int log_n,
    int tw_offset)
{
    __shared__ uint64_t lds[LDS_TOTAL];

    const int tid  = threadIdx.x;
    const int base = blockIdx.x * 1024;

    /* Load bit-reversed (CT-DIT stage-bucketed twiddle layout requires
     * bit-reversed input — mirrors ntt_kernel above). Previously loaded
     * natural order, which silently computed DFT(bitrev(x)) instead of
     * DFT(x). Caller benchmark_montgomery was the only consumer. */
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
        uint64_t t = montgomery_mul<PIDX>(lds[lds_idx(idx_v)], w);
        lds[lds_idx(idx_u)] = addmod<PIDX>(u, t);
        lds[lds_idx(idx_v)] = submod<PIDX>(u, t);
        __syncthreads();

        tw_base += (1 << s);
    }

    data[base + i0] = lds[l0];
    data[base + i1] = lds[l1];
}

/* 4-step launcher variant using mont_ntt_kernel for row/col passes.          */
/* Cross-twiddle kernel is unchanged (ShoupPair); only butterfly tables change.*/
template<int PIDX>
static void launch_mont_ntt_large(
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

    mont_ntt_kernel<PIDX><<<total_blocks, BLOCK_SIZE, 0, stream>>>(
        data, tw_mont, log_n1, 0);
    dim3 grid2((N2 + 255) / 256, N1);
    cross_twiddle_kernel<PIDX><<<grid2, 256, 0, stream>>>(
        data, tw_cross, N2);
    mont_ntt_kernel<PIDX><<<total_blocks, BLOCK_SIZE, 0, stream>>>(
        data, tw_mont + (N1 / 2), log_n2, 0);
}

template<int PIDX>
void launch_mont_ntt(
    uint64_t*        data,
    const uint64_t*  tw_mont,
    const ShoupPair* tw_cross,
    int log_n,
    hipStream_t stream)
{
    int N = 1 << log_n;
    if (log_n <= 10) {
        int n_blocks = (N < 1024) ? 1 : N / 1024;
        mont_ntt_kernel<PIDX><<<n_blocks, BLOCK_SIZE, 0, stream>>>(
            data, tw_mont, log_n, 0);
    } else {
        launch_mont_ntt_large<PIDX>(data, tw_mont, tw_cross, log_n, stream);
    }
}

/* ─── Explicit instantiations ────────────────────────────────────────────── */
template __global__ void ntt_global_stage_kernel<0>(uint64_t*, const ShoupPair*, int, int, int);
template __global__ void ntt_global_stage_kernel<1>(uint64_t*, const ShoupPair*, int, int, int);
template __global__ void ntt_global_stage_kernel<2>(uint64_t*, const ShoupPair*, int, int, int);
template __global__ void ntt_global_stage_kernel<3>(uint64_t*, const ShoupPair*, int, int, int);

template __global__ void ntt_global_stage_kernel_mont<0>(uint64_t*, const uint64_t*, int, int, int);
template __global__ void ntt_global_stage_kernel_mont<1>(uint64_t*, const uint64_t*, int, int, int);
template __global__ void ntt_global_stage_kernel_mont<2>(uint64_t*, const uint64_t*, int, int, int);
template __global__ void ntt_global_stage_kernel_mont<3>(uint64_t*, const uint64_t*, int, int, int);

template void launch_ntt_ct_dit_mont<0>(uint64_t*, const uint64_t*, int, hipStream_t);
template void launch_ntt_ct_dit_mont<1>(uint64_t*, const uint64_t*, int, hipStream_t);
template void launch_ntt_ct_dit_mont<2>(uint64_t*, const uint64_t*, int, hipStream_t);
template void launch_ntt_ct_dit_mont<3>(uint64_t*, const uint64_t*, int, hipStream_t);

template void launch_intt_ct_dit_mont<0>(uint64_t*, const uint64_t*, uint64_t, int, hipStream_t);
template void launch_intt_ct_dit_mont<1>(uint64_t*, const uint64_t*, uint64_t, int, hipStream_t);
template void launch_intt_ct_dit_mont<2>(uint64_t*, const uint64_t*, uint64_t, int, hipStream_t);
template void launch_intt_ct_dit_mont<3>(uint64_t*, const uint64_t*, uint64_t, int, hipStream_t);

template __global__ void ntt_kernel<0>(uint64_t*, const ShoupPair*, int, int);
template __global__ void ntt_kernel<1>(uint64_t*, const ShoupPair*, int, int);
template __global__ void ntt_kernel<2>(uint64_t*, const ShoupPair*, int, int);
template __global__ void ntt_kernel<3>(uint64_t*, const ShoupPair*, int, int);

template __global__ void cross_twiddle_kernel<0>(uint64_t*, const ShoupPair*, int);
template __global__ void cross_twiddle_kernel<1>(uint64_t*, const ShoupPair*, int);
template __global__ void cross_twiddle_kernel<2>(uint64_t*, const ShoupPair*, int);
template __global__ void cross_twiddle_kernel<3>(uint64_t*, const ShoupPair*, int);

template __global__ void scale_kernel<0>(uint64_t*, uint64_t, int);
template __global__ void scale_kernel<1>(uint64_t*, uint64_t, int);
template __global__ void scale_kernel<2>(uint64_t*, uint64_t, int);
template __global__ void scale_kernel<3>(uint64_t*, uint64_t, int);

template __global__ void hadamard_kernel<0>(uint64_t*, const uint64_t*, int);
template __global__ void hadamard_kernel<1>(uint64_t*, const uint64_t*, int);
template __global__ void hadamard_kernel<2>(uint64_t*, const uint64_t*, int);
template __global__ void hadamard_kernel<3>(uint64_t*, const uint64_t*, int);

template void launch_ntt<0>(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_ntt<1>(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_ntt<2>(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_ntt<3>(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);

template void launch_intt<0>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_intt<1>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_intt<2>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_intt<3>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);

template void launch_hadamard<0>(uint64_t*, const uint64_t*, int, hipStream_t);
template void launch_hadamard<1>(uint64_t*, const uint64_t*, int, hipStream_t);
template void launch_hadamard<2>(uint64_t*, const uint64_t*, int, hipStream_t);
template void launch_hadamard<3>(uint64_t*, const uint64_t*, int, hipStream_t);

template __global__ void twist_kernel<0>(uint64_t*, const ShoupPair*, int);
template __global__ void twist_kernel<1>(uint64_t*, const ShoupPair*, int);
template __global__ void twist_kernel<2>(uint64_t*, const ShoupPair*, int);
template __global__ void twist_kernel<3>(uint64_t*, const ShoupPair*, int);

template __global__ void untwist_kernel<0>(uint64_t*, const ShoupPair*, int);
template __global__ void untwist_kernel<1>(uint64_t*, const ShoupPair*, int);
template __global__ void untwist_kernel<2>(uint64_t*, const ShoupPair*, int);
template __global__ void untwist_kernel<3>(uint64_t*, const ShoupPair*, int);

template __global__ void ntt_kernel_scaled<0>(uint64_t*, const ShoupPair*, int, int, uint64_t, uint64_t);
template __global__ void ntt_kernel_scaled<1>(uint64_t*, const ShoupPair*, int, int, uint64_t, uint64_t);
template __global__ void ntt_kernel_scaled<2>(uint64_t*, const ShoupPair*, int, int, uint64_t, uint64_t);
template __global__ void ntt_kernel_scaled<3>(uint64_t*, const ShoupPair*, int, int, uint64_t, uint64_t);

template void launch_intt_fused<0>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_intt_fused<1>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_intt_fused<2>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_intt_fused<3>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);

template void launch_twist<0>(uint64_t*, const ShoupPair*, int, hipStream_t);
template void launch_twist<1>(uint64_t*, const ShoupPair*, int, hipStream_t);
template void launch_twist<2>(uint64_t*, const ShoupPair*, int, hipStream_t);
template void launch_twist<3>(uint64_t*, const ShoupPair*, int, hipStream_t);

template void launch_untwist<0>(uint64_t*, const ShoupPair*, int, hipStream_t);
template void launch_untwist<1>(uint64_t*, const ShoupPair*, int, hipStream_t);
template void launch_untwist<2>(uint64_t*, const ShoupPair*, int, hipStream_t);
template void launch_untwist<3>(uint64_t*, const ShoupPair*, int, hipStream_t);

template __global__ void mont_ntt_kernel<0>(uint64_t*, const uint64_t*, int, int);
template __global__ void mont_ntt_kernel<1>(uint64_t*, const uint64_t*, int, int);
template __global__ void mont_ntt_kernel<2>(uint64_t*, const uint64_t*, int, int);
template __global__ void mont_ntt_kernel<3>(uint64_t*, const uint64_t*, int, int);

template void launch_mont_ntt<0>(uint64_t*, const uint64_t*, const ShoupPair*, int, hipStream_t);
template void launch_mont_ntt<1>(uint64_t*, const uint64_t*, const ShoupPair*, int, hipStream_t);
template void launch_mont_ntt<2>(uint64_t*, const uint64_t*, const ShoupPair*, int, hipStream_t);
template void launch_mont_ntt<3>(uint64_t*, const uint64_t*, const ShoupPair*, int, hipStream_t);

/* C1: stockham_kernel<PIDX,LOG_N_SUB> and stockham_kernel_scaled<PIDX,LOG_N_SUB>
 * are instantiated via the launch_stok / launch_stok_scaled dispatch helpers
 * (switch 1-11 / 1-10 for each PIDX); no standalone explicit instantiations needed. */
template void launch_stockham_ntt<0>(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_ntt<1>(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_ntt<2>(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_ntt<3>(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);

template void launch_stockham_intt<0>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_intt<1>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_intt<2>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_intt<3>(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);

template void launch_stockham_intt_hadamard<0>(uint64_t*, const uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_intt_hadamard<1>(uint64_t*, const uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_intt_hadamard<2>(uint64_t*, const uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_intt_hadamard<3>(uint64_t*, const uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);

template void launch_stockham_ntt_rect<0>(uint64_t*, uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_ntt_rect<1>(uint64_t*, uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_ntt_rect<2>(uint64_t*, uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_ntt_rect<3>(uint64_t*, uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);

template void launch_stockham_intt_rect<0>(uint64_t*, uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_intt_rect<1>(uint64_t*, uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_intt_rect<2>(uint64_t*, uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template void launch_stockham_intt_rect<3>(uint64_t*, uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);

template void launch_stockham_3factor_ntt<0>(uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);
template void launch_stockham_3factor_ntt<1>(uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);
template void launch_stockham_3factor_ntt<2>(uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);
template void launch_stockham_3factor_ntt<3>(uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);

template void launch_stockham_3factor_intt<0>(uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);
template void launch_stockham_3factor_intt<1>(uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);
template void launch_stockham_3factor_intt<2>(uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);
template void launch_stockham_3factor_intt<3>(uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);

template void launch_stockham_3factor_intt_hadamard<0>(uint64_t*, const uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);
template void launch_stockham_3factor_intt_hadamard<1>(uint64_t*, const uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);
template void launch_stockham_3factor_intt_hadamard<2>(uint64_t*, const uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);
template void launch_stockham_3factor_intt_hadamard<3>(uint64_t*, const uint64_t*, const ShoupPair*, const ShoupPair*, const ShoupPair*, hipStream_t);

template void launch_stockham_3factor_F1_fwd<0>(uint64_t*, const ShoupPair*, const ShoupPair*, hipStream_t);
template void launch_stockham_3factor_F1_inv<0>(uint64_t*, const ShoupPair*, const ShoupPair*, hipStream_t);
