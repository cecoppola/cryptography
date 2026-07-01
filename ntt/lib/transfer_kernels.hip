/*
 * transfer_kernels.hip — GPU scatter / gather kernels for the CRT-NTT pipeline.
 *
 * A fully on-device scatter + gather for the gated path. The default engine
 * (arith/ntt_mul.hip) already GPU-scatters after an H2D limb upload, but still
 * GATHERS on the host (bigint_gather, step 8); these kernels move the gather
 * on-device too and offer a transposed-layout-aware scatter that reads unified
 * limbs directly (no H2D). On MI300A (unified HBM3) every buffer is GPU-
 * addressable, so the operand limbs / Garner U256 output are read and the
 * residue / product arrays written entirely on-device — eliminating the host-
 * side gather loop and the H2D scatter copy (see ROADMAP "GPU scatter/gather",
 * PERFORMANCE §"4-APU posture").
 *
 * Per-element logic lives in transfer_core.h (host+device inline) and is the
 * SAME code the host oracle arith/test_transfer_core.c verifies against the
 * bigint_scatter[_t] / bigint_gather[_t] reference. The kernels here are thin
 * one-thread-per-output-element wrappers plus host launch helpers.
 *
 * Integration note: wired into arith/ntt_mul.hip's hot path behind
 * -DNTT_GPU_SCATTER_GATHER (default OFF; auto-off on GFX1030_LOCAL), where it
 * replaces the H2D-upload + GPU scatter (steps 1+2) and the host gather (step 8).
 * Enabling is gated on unified memory and on MI300A performance measurement
 * (P2P-bound scatter vs APU0-local gather); on the PCIe dev box these would
 * regress vs the host gather because host buffers are not device-local.
 * Correctness is hardware-independent
 * and is what gets verified here (and in test_e2e_oracle's core-routed trial).
 *
 * C-only kernel bodies (no C++ containers/templates); pidx is a runtime kernel
 * argument (uniform across a launch — scalar branch, no per-thread cost).
 */

#include <hip/hip_runtime.h>
#include "transfer_core.h"

#ifdef __HIPCC__

/* ─── Scatter kernel: one thread per output coefficient ──────────────────────
 * d_coeffs[j] = limbs[xpose_index(j)] % PRIMES[pidx], zero-padded past n_limbs.
 * transposed==1 pre-applies the I15 transpose_sq (Bailey 4-step path). */
__global__ __launch_bounds__(256)
static void scatter_kernel(const limb_t* __restrict__ limbs, int n_limbs,
                           uint64_t* __restrict__ d_coeffs, int n_coeffs,
                           int M, int transposed, int pidx)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_coeffs) return;
    int src = xpose_index(j, M, transposed);
    d_coeffs[j] = scatter_value(limbs, n_limbs, src, pidx);
}

/* ─── Gather phase-1 kernel: one thread per output limb ──────────────────────
 * Pulls every U256 part landing on limb j into a 128-bit partial sum, split
 * into d_acc_lo/d_acc_hi for the phase-2 carry sweep. No atomics (pull form).
 * Undoes the I15 transposition inline when transposed==1. */
__global__ __launch_bounds__(256)
static void gather_accum_kernel(const U256* __restrict__ d_coeffs, int n_coeffs,
                                int M, int transposed,
                                uint64_t* __restrict__ d_acc_lo,
                                uint64_t* __restrict__ d_acc_hi, int n_out)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    __uint128_t acc = gather_acc_limb(d_coeffs, n_coeffs, j, M, transposed);
    d_acc_lo[j] = (uint64_t)acc;
    d_acc_hi[j] = (uint64_t)(acc >> 64);
}

/* ─── Gather phase-2 kernel: single-thread carry normalize ───────────────────
 * Sequential carry sweep (carry is < 2^3; sweep is memory-trivial). A parallel
 * carry-lookahead is a hardware-side refinement; a single thread is correct and
 * the established floor (~0.004 ms at log_n=20, PERFORMANCE §3). */
__global__
static void gather_carry_kernel(const uint64_t* __restrict__ d_acc_lo,
                                const uint64_t* __restrict__ d_acc_hi,
                                limb_t* __restrict__ d_limbs, int n_out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    (void)gather_carry_normalize(d_acc_lo, d_acc_hi, d_limbs, n_out);
}

/* ─── Launch helpers ─────────────────────────────────────────────────────────
 * Drop-in launchers for arith/ntt_mul.hip. d_* are device/unified pointers on
 * the prime's owning APU. Sizes: d_coeffs has n_coeffs entries; the gather
 * scratch (d_acc_lo/hi) and output d_limbs have n_out = n_coeffs + 4 entries. */

/* scatter: residue array d_coeffs <- operand limbs, prime pidx, given layout. */
void launch_scatter_gpu(const limb_t* limbs, int n_limbs,
                        uint64_t* d_coeffs, int n_coeffs,
                        int M, int transposed, int pidx, hipStream_t stream)
{
    const int block = 256;
    int grid = (n_coeffs + block - 1) / block;
    if (grid < 1) grid = 1;
    scatter_kernel<<<grid, block, 0, stream>>>(limbs, n_limbs, d_coeffs,
                                               n_coeffs, M, transposed, pidx);
}

/* gather: product limbs d_limbs <- Garner U256 array d_coeffs, given layout.
 * Two launches on the same stream: parallel accumulate then carry normalize. */
void launch_gather_gpu(const U256* d_coeffs, int n_coeffs, int M, int transposed,
                       uint64_t* d_acc_lo, uint64_t* d_acc_hi,
                       limb_t* d_limbs, int n_out, hipStream_t stream)
{
    const int block = 256;
    int grid = (n_out + block - 1) / block;
    if (grid < 1) grid = 1;
    gather_accum_kernel<<<grid, block, 0, stream>>>(d_coeffs, n_coeffs, M,
                                                    transposed, d_acc_lo,
                                                    d_acc_hi, n_out);
    gather_carry_kernel<<<1, 1, 0, stream>>>(d_acc_lo, d_acc_hi, d_limbs, n_out);
}

/* ─── Parallel carry-lookahead (CLA) phase-2 building blocks ─────────────────
 * Replace the single-thread carry sweep with an O(log n) generate/propagate
 * prefix. The decomposition and the cla_compose monoid are host-verified
 * (arith/test_transfer_core.c CLA section, incl. adversarial long-carry chains).
 * Provided here: the embarrassingly-parallel decompose/finalize kernels and a
 * single-block LDS prefix scan (correct for n_out <= CLA_SCAN_BLOCK). The
 * multi-block prefix-scan orchestration (segmented scan over cla_compose across
 * blocks) is the remaining GPU bring-up step; launch_gather_cla() uses the
 * verified sequential normalizer for larger n_out until that lands. */
#define CLA_SCAN_BLOCK 1024

/* Decompose each partial sum into its pre-carry digit res[j] and (gen,prop)
 * flags. Reads acc[j] and acc[j-1] (for the local-high fold). Parallel. */
__global__ __launch_bounds__(256)
static void cla_decompose_kernel(const uint64_t* __restrict__ acc_lo,
                                 const uint64_t* __restrict__ acc_hi, int n_out,
                                 limb_t* __restrict__ res,
                                 unsigned char* __restrict__ gen,
                                 unsigned char* __restrict__ prop)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    uint64_t g_prev = (j == 0) ? 0 : cla_high(acc_lo[j-1], acc_hi[j-1]);
    limb_t r;
    ClaPair c = cla_cell(acc_lo[j], acc_hi[j], g_prev, &r);
    res[j] = r; gen[j] = c.g; prop[j] = c.p;
}

/* Single-block Hillis-Steele inclusive scan over (gen,prop) in LDS, writing the
 * exclusive carry-in cin[j] for each limb. Precondition: n_out <= CLA_SCAN_BLOCK
 * and launched with blockDim.x == CLA_SCAN_BLOCK. Identity pad = {g=0,p=1}. */
__global__ __launch_bounds__(CLA_SCAN_BLOCK)
static void cla_scan_singleblock_kernel(const unsigned char* __restrict__ gen,
                                        const unsigned char* __restrict__ prop,
                                        int n_out, unsigned char* __restrict__ cin)
{
    __shared__ ClaPair buf[2][CLA_SCAN_BLOCK];
    int t = threadIdx.x;
    int cur = 0;
    ClaPair me;
    if (t < n_out) { me.g = gen[t]; me.p = prop[t]; }
    else           { me.g = 0;      me.p = 1;       }   /* identity */
    buf[cur][t] = me;
    __syncthreads();
    for (int d = 1; d < n_out; d <<= 1) {
        ClaPair v = buf[cur][t];
        if (t >= d) v = cla_compose(buf[cur][t-d], buf[cur][t]);
        buf[1-cur][t] = v;
        cur = 1 - cur;
        __syncthreads();
    }
    if (t < n_out) cin[t] = (t == 0) ? 0 : buf[cur][t-1].g;
}

/* Finalize each limb from its pre-carry digit and scanned carry-in. Parallel. */
__global__ __launch_bounds__(256)
static void cla_finalize_kernel(const limb_t* __restrict__ res,
                                const unsigned char* __restrict__ cin, int n_out,
                                limb_t* __restrict__ limbs)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    limbs[j] = cla_finalize(res[j], cin[j]);
}

/* ─── Multi-block CLA scan (mirrors the host-verified blocked_scan) ──────────
 * For n_out beyond one block the scan runs in levels: each block inclusive-scans
 * its own (gen,prop) cells in LDS and emits its aggregate; the aggregate array
 * is scanned the same way; each block then composes the exclusive prefix of all
 * earlier blocks (from the LOWER side — the monoid is non-commutative) into its
 * local results. The recursive structure and the apply order are host-verified
 * in arith/test_transfer_core.c (test_blocked_scan, deep recursion). */

/* One scan level: in-place LDS inclusive scan of gen/prop within each block;
 * thread 0 writes the block aggregate to agg_g/agg_p[blockIdx] (NULL to skip,
 * e.g. the top level). blockDim.x must be CLA_SCAN_BLOCK. */
__global__ __launch_bounds__(CLA_SCAN_BLOCK)
static void cla_block_scan_kernel(unsigned char* __restrict__ gen,
                                  unsigned char* __restrict__ prop, int n,
                                  unsigned char* __restrict__ agg_g,
                                  unsigned char* __restrict__ agg_p)
{
    __shared__ ClaPair buf[2][CLA_SCAN_BLOCK];
    int base = blockIdx.x * blockDim.x;
    int t = threadIdx.x;
    int gj = base + t;
    int cnt = n - base; if (cnt > (int)blockDim.x) cnt = blockDim.x;  /* valid in block */
    int cur = 0;
    ClaPair me;
    if (t < cnt) { me.g = gen[gj]; me.p = prop[gj]; }
    else         { me.g = 0;       me.p = 1;        }   /* identity */
    buf[cur][t] = me;
    __syncthreads();
    for (int d = 1; d < cnt; d <<= 1) {
        ClaPair v = buf[cur][t];
        if (t >= d) v = cla_compose(buf[cur][t-d], buf[cur][t]);
        buf[1-cur][t] = v;
        cur = 1 - cur;
        __syncthreads();
    }
    if (t < cnt) { gen[gj] = buf[cur][t].g; prop[gj] = buf[cur][t].p; }
    if (t == 0 && agg_g) {                              /* aggregate = last element */
        agg_g[blockIdx.x] = buf[cur][cnt-1].g;
        agg_p[blockIdx.x] = buf[cur][cnt-1].p;
    }
}

/* Compose the exclusive block prefix (inclusive-scanned aggregate of blocks
 * [0,b)) into every element of block b>0, turning block-local inclusive scans
 * into the global inclusive prefix. blockDim.x must match the scan block size. */
__global__ __launch_bounds__(CLA_SCAN_BLOCK)
static void cla_apply_block_prefix_kernel(unsigned char* __restrict__ gen,
                                          unsigned char* __restrict__ prop, int n,
                                          const unsigned char* __restrict__ inc_agg_g,
                                          const unsigned char* __restrict__ inc_agg_p)
{
    int b = blockIdx.x;
    if (b == 0) return;                                 /* first block unchanged */
    int j = b * blockDim.x + threadIdx.x;
    if (j >= n) return;
    ClaPair bp;    bp.g    = inc_agg_g[b-1]; bp.p    = inc_agg_p[b-1];  /* excl. prefix */
    ClaPair local; local.g = gen[j];         local.p = prop[j];
    ClaPair r = cla_compose(bp, local);
    gen[j] = r.g; prop[j] = r.p;
}

/* Finalize from the GLOBAL inclusive prefix in pref_g: cin[j] = pref_g[j-1]. */
__global__ __launch_bounds__(256)
static void cla_global_finalize_kernel(const unsigned char* __restrict__ pref_g,
                                       const limb_t* __restrict__ res, int n,
                                       limb_t* __restrict__ limbs)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned char cin = (j == 0) ? 0 : pref_g[j-1];
    limbs[j] = cla_finalize(res[j], cin);
}

/* gather with parallel carry-lookahead phase 2 (single-block scan path).
 * d_res / d_gen / d_prop / d_cin are scratch of n_out elements. For
 * n_out > CLA_SCAN_BLOCK, falls back to the verified sequential normalizer
 * (use launch_gather_cla_multiblock for the multi-block path). */
void launch_gather_cla(const U256* d_coeffs, int n_coeffs, int M, int transposed,
                       uint64_t* d_acc_lo, uint64_t* d_acc_hi,
                       limb_t* d_res, unsigned char* d_gen, unsigned char* d_prop,
                       unsigned char* d_cin, limb_t* d_limbs, int n_out,
                       hipStream_t stream)
{
    const int block = 256;
    int grid = (n_out + block - 1) / block;
    if (grid < 1) grid = 1;
    gather_accum_kernel<<<grid, block, 0, stream>>>(d_coeffs, n_coeffs, M,
                                                    transposed, d_acc_lo,
                                                    d_acc_hi, n_out);
    if (n_out <= CLA_SCAN_BLOCK) {
        cla_decompose_kernel<<<grid, block, 0, stream>>>(d_acc_lo, d_acc_hi,
                                                         n_out, d_res, d_gen, d_prop);
        cla_scan_singleblock_kernel<<<1, CLA_SCAN_BLOCK, 0, stream>>>(d_gen, d_prop,
                                                                      n_out, d_cin);
        cla_finalize_kernel<<<grid, block, 0, stream>>>(d_res, d_cin, n_out, d_limbs);
    } else {
        gather_carry_kernel<<<1, 1, 0, stream>>>(d_acc_lo, d_acc_hi, d_limbs, n_out);
    }
}

/* gather with parallel carry-lookahead phase 2 (general multi-block path,
 * mirrors the host-verified blocked_scan). Two scan levels cover n_out up to
 * CLA_SCAN_BLOCK^2 (~1.05M, i.e. log_n<=20); beyond that a third level is
 * needed and this falls back to the verified sequential normalizer. Scratch:
 * d_res/d_gen/d_prop hold n_out elements; d_agg_g/d_agg_p hold B = ceil(n_out/
 * CLA_SCAN_BLOCK) aggregates. */
void launch_gather_cla_multiblock(const U256* d_coeffs, int n_coeffs, int M,
                                  int transposed, uint64_t* d_acc_lo, uint64_t* d_acc_hi,
                                  limb_t* d_res, unsigned char* d_gen, unsigned char* d_prop,
                                  unsigned char* d_agg_g, unsigned char* d_agg_p,
                                  limb_t* d_limbs, int n_out, hipStream_t stream)
{
    const int block = 256;
    int grid = (n_out + block - 1) / block;
    if (grid < 1) grid = 1;
    gather_accum_kernel<<<grid, block, 0, stream>>>(d_coeffs, n_coeffs, M, transposed,
                                                    d_acc_lo, d_acc_hi, n_out);

    int B = (n_out + CLA_SCAN_BLOCK - 1) / CLA_SCAN_BLOCK;
    if (B > CLA_SCAN_BLOCK) {                 /* > 2 levels (log_n>20): not yet wired */
        gather_carry_kernel<<<1, 1, 0, stream>>>(d_acc_lo, d_acc_hi, d_limbs, n_out);
        return;
    }
    cla_decompose_kernel<<<grid, block, 0, stream>>>(d_acc_lo, d_acc_hi, n_out,
                                                     d_res, d_gen, d_prop);
    if (B <= 1) {                            /* single block: one inclusive scan */
        cla_block_scan_kernel<<<1, CLA_SCAN_BLOCK, 0, stream>>>(d_gen, d_prop, n_out, 0, 0);
    } else {                                 /* two levels: scan, scan aggregates, apply */
        cla_block_scan_kernel<<<B, CLA_SCAN_BLOCK, 0, stream>>>(d_gen, d_prop, n_out,
                                                                d_agg_g, d_agg_p);
        cla_block_scan_kernel<<<1, CLA_SCAN_BLOCK, 0, stream>>>(d_agg_g, d_agg_p, B, 0, 0);
        cla_apply_block_prefix_kernel<<<B, CLA_SCAN_BLOCK, 0, stream>>>(d_gen, d_prop, n_out,
                                                                        d_agg_g, d_agg_p);
    }
    cla_global_finalize_kernel<<<grid, block, 0, stream>>>(d_gen, d_res, n_out, d_limbs);
}

#endif /* __HIPCC__ */
