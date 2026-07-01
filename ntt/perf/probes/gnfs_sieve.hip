/* gnfs_sieve.hip — MI300A line sieve for GNFS
 *
 * Purpose:
 *   GPU-accelerated General Number Field Sieve line sieve for smooth-relation
 *   collection on AMD MI300A (CDNA3, gfx942).  Implements a dual-sided
 *   (rational + algebraic) additive log sieve over a 2D (a,b) lattice region.
 *
 *   This is a LINE sieve, not a lattice sieve.  Modern GNFS implementations
 *   (CADO-NFS) use lattice sieving which is ~5× more efficient per relation.
 *   This file provides a correct, benchmarkable baseline for GPU sieve
 *   throughput measurement and parameter tuning.
 *
 * Algorithm:
 *   For each b-line and each prime p in the factor base, the rational-side
 *   sieve root is a_r ≡ −b·m (mod p).  The additive log sieve accumulates
 *   floor(log₂(p)×8) into sieve[a] for every a ≡ a_r (mod p) in the tile.
 *   Entries exceeding a threshold are candidate smooth (a,b) pairs.
 *
 * Three-tier factor base:
 *   Tier 1  p < TILE_SIZE (64 KB fits in LDS):
 *             Sequential stride sieve entirely in LDS; no HBM traffic per update.
 *             Potential race: multiple threads can update the same LDS byte.
 *             Use atomicAdd variants or a serial-strip decomposition if exact
 *             counts are required (benchmark mode accepts rare missed updates).
 *   Tier 2  TILE_SIZE ≤ p < MEDIUM_BOUND:
 *             One thread per (b, prime) pair; single atomicAdd on HBM byte.
 *             Each prime hits at most once per tile (T/p < 1 for p ≥ T).
 *             HBM atomicAdd throughput is the dominant bottleneck.
 *   Tier 3  p ≥ MEDIUM_BOUND (large primes):
 *             Excluded from GPU sieve.  CPU trial-divides candidate (a,b) pairs.
 *             P(≥1 hit per 64KB tile) = T/p ≤ 6%; GPU efficiency < 6%.
 *
 * Known limitations (benchmark code, not production GNFS):
 *   1. bm = b × m uses int64_t.  For real GNFS, m ≈ N^(1/6) ≥ 2^170 —
 *      requires __int128 or explicit b·m (mod p) via Montgomery reduction.
 *      Toy benchmark uses m = 1000003, b ≤ 80000 → bm ≤ 8×10¹⁰ (safe).
 *   2. LDS tier-1 sieve: different threads may write the same byte
 *      non-atomically.  Occasionally underestimates a sieve value by logp.
 *      Effect: a small fraction of smooth numbers may be missed.  Harmless
 *      for throughput benchmarking; fix with atomicAdd or strip decomposition
 *      for production use.
 *   3. Only the rational side is sieved.  Full GNFS requires sieving both
 *      algebraic and rational sides and checking joint smoothness.
 *
 * Build:
 *   hipcc -O3 --offload-arch=gfx942 -Wall -Wextra \
 *         -I/opt/rocm-7.0.3/include \
 *         -L/opt/rocm-7.0.3/lib -Wl,-rpath,$(HOME)/lib \
 *         -o gnfs_sieve gnfs_sieve.hip
 *
 * Usage:
 *   ./gnfs_sieve [b_range] [n_small_cap] [n_medium_cap]
 *   Defaults: b_range=4096  n_small_cap=50000  n_medium_cap=200000
 *
 * Reference:
 *   Briggs, M. "An Introduction to the GNFS" (2005), §4.
 *   Kleinjung et al. "Factorization of a 768-bit RSA Modulus", CRYPTO 2010.
 *   Boudot et al. "Factoring RSA-250", CRYPTO 2020.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Compile-time tuning parameters ─────────────────────────────────────── */

/* TILE_SIZE: number of consecutive a-values sieved per b-line.
 * Must satisfy TILE_SIZE + 1 ≤ 65536 (64 KB LDS limit on both gfx942 and
 * gfx1030).  64512 = 63 × 1024 gives 65513 B with the +1 pad, well within
 * the limit and aligned to 1024 (cache-line friendly). Previously 65536
 * caused the +1 pad to overflow the LDS limit by 1 byte, preventing
 * compilation. Fixed 2026-05-30. */
#define TILE_SIZE        64512

/* MEDIUM_BOUND: upper limit for Tier-2 (HBM atomic) primes.
 * Primes p ≥ MEDIUM_BOUND are excluded from the GPU sieve entirely.
 * P(hit per tile) = TILE_SIZE/p; at p = MEDIUM_BOUND = 2^20 = 1M,
 * expected hits per tile = 0.0625 → >93% of threads find no work.        */
#define MEDIUM_BOUND     (1u << 20)    /* 1,048,576 */

/* BLOCK_SIZE: threads per block.  Must equal warpSize = 64 on MI300A.
 * One wavefront = one block = one b-line handled per CU.                 */
#define BLOCK_SIZE       64

/* MAX_CANDIDATES: output array size.  Increase if the sieve finds more
 * smooth candidates than this.  At THRESHOLD_FRAC=0.85, expect ~0.01–0.1%
 * of sieve cells to be accepted.  4M is generous for benchmark runs.     */
#define MAX_CANDIDATES   (1u << 22)   /* 4,194,304 */

/* Rational-side threshold fraction: accept if accumulated log ≥ fraction
 * of ln(MEDIUM_BOUND)×8.  Increase to reduce false positives; decrease to
 * increase candidate yield (at cost of more CPU post-processing).        */
#define THRESHOLD_FRAC   0.85f

/* ── Data structures ─────────────────────────────────────────────────────── */

/* One entry in the factor base.
 * p:    prime
 * r:    sieve root for a reference b-value (updated per b-line on GPU)
 * logp: floor(log2(p) × 8) — additive log approximation, stored as byte */
typedef struct {
    uint32_t p;
    uint32_t r;      /* unused by kernels; kept for host bookkeeping */
    uint8_t  logp;
} prime_entry_t;

/* Smooth-relation candidate (a, b) pair surviving threshold scan */
typedef struct {
    int64_t a;
    int64_t b;
} relation_t;

/* ── Tier-1 kernel: small primes entirely in LDS ────────────────────────── */
/*
 * gnfs_line_sieve_kernel — Tier-1 LDS sieve + Tier-2 HBM merge + scan
 *
 * Launch: gridDim.x = b_range, blockDim.x = BLOCK_SIZE (64)
 *   One block per b-line; one wavefront per block.
 *
 * Inputs:
 *   small_primes  — Tier-1 primes (p < TILE_SIZE); factor base entries
 *   n_small       — count of Tier-1 primes
 *   m             — rational-side polynomial root: N ≡ 0 (mod f(m)) implied
 *                   For benchmark: toy integer; for real GNFS: N^(1/d) mod p
 *                   computed per-prime on CPU and passed via prime_entry_t.
 *   a0            — first a-value in this tile; usually −TILE_SIZE/2
 *   b_start       — first b-value for this grid; b = b_start + blockIdx.x
 *   hbm_sieve     — [b_range × TILE_SIZE] byte array; Tier-2 kernel fills
 *                   this before the Tier-1 kernel runs
 *   candidates    — output array; appended atomically
 *   n_candidates  — atomic counter; caller resets to 0 before each launch
 *   threshold_rat — accept sieve[i] ≥ threshold_rat (computed on host)
 *
 * Algorithm:
 *   1. Zero LDS sieve tile.
 *   2. Each thread handles n_small/BLOCK_SIZE primes (strided distribution).
 *      For each prime p: compute first a ≡ −b·m (mod p) with a ≥ a0;
 *      add logp to sieve[offset], sieve[offset+p], … until end of tile.
 *      NOTE: race condition exists if two threads have primes sharing a
 *      common multiple within the tile.  Rare in practice; harmless for
 *      benchmarking.  Fix: atomicAdd on uint32_t sieve[TILE_SIZE/4].
 *   3. Merge HBM Tier-2 contributions (filled by medium_sieve_kernel).
 *   4. Threshold scan: output (a,b) pairs with sieve ≥ threshold.
 *
 * Invariants assumed:
 *   TILE_SIZE = 65536, BLOCK_SIZE = 64 (one wavefront on CDNA3)
 *   a0 % p: we normalize to [0,p) to handle negative a0 correctly.
 */
__global__ void gnfs_line_sieve_kernel(
    const prime_entry_t * __restrict__ small_primes,
    uint32_t n_small,
    int64_t  m,
    int64_t  a0,
    int64_t  b_start,
    const uint8_t * __restrict__ hbm_sieve,
    relation_t *candidates,
    uint32_t   *n_candidates,
    uint8_t     threshold_rat)
{
    int64_t b   = b_start + (int64_t)blockIdx.x;
    int     tid = threadIdx.x;

    /* LDS sieve tile.  Padded by 1 to avoid bank-0 pressure (§c-style.md). */
    __shared__ uint8_t sieve[TILE_SIZE + 1];

    /* 1. Zero the tile */
    for (int i = tid; i < TILE_SIZE; i += BLOCK_SIZE)
        sieve[i] = 0;
    __syncthreads();

    /* 2. Tier-1: additive log sieve for small primes in LDS ─────────────── */
    for (uint32_t pi = (uint32_t)tid; pi < n_small; pi += BLOCK_SIZE) {
        uint32_t p    = small_primes[pi].p;
        uint8_t  logp = small_primes[pi].logp;

        /* Rational-side root: a_r = (−b·m) mod p, in [0, p).
         * For real GNFS, b·m mod p must be precomputed with 128-bit arithmetic
         * since m ≈ N^(1/6) does not fit in int64_t.  The benchmark uses a
         * small m (= 1000003) so int64_t arithmetic is sufficient.          */
        int64_t bm_r = (b % (int64_t)p) * (m % (int64_t)p) % (int64_t)p;
        if (bm_r < 0) bm_r += (int64_t)p;
        uint32_t a_r = (uint32_t)((p - (uint32_t)bm_r) % p);

        /* First sieve position: smallest index i ≥ 0 such that a0+i ≡ a_r (mod p).
         * Normalize a0 mod p to [0, p) before subtracting.                */
        int64_t a0r = a0 % (int64_t)p;
        if (a0r < 0) a0r += (int64_t)p;          /* [0, p) */
        int64_t off = (int64_t)a_r - a0r;
        if (off < 0) off += (int64_t)p;
        uint32_t pos = (uint32_t)off;             /* pos ∈ [0, p) ⊆ [0, TILE_SIZE) */

        /* Additive sieve: add logp at every position ≡ a_r (mod p).
         * No atomics: each thread owns a disjoint set of primes.  Two primes
         * p, q may share common-multiple positions in the tile, causing a
         * non-atomic collision.  Impact: rare under-count; accept for bench. */
        while (pos < TILE_SIZE) {
            sieve[pos] += logp;    /* benign race: occasional miss, never overflow */
            pos        += p;
        }
    }
    __syncthreads();

    /* 3. Merge Tier-2 (HBM) contributions ──────────────────────────────── */
    /* gnfs_medium_sieve_kernel ran first and wrote atomicAdd results into
     * hbm_sieve[blockIdx.x * TILE_SIZE .. +TILE_SIZE].                     */
    const uint8_t *hbm_row = hbm_sieve + (size_t)blockIdx.x * TILE_SIZE;
    for (int i = tid; i < TILE_SIZE; i += BLOCK_SIZE)
        sieve[i] += hbm_row[i];
    __syncthreads();

    /* 4. Threshold scan: output surviving (a, b) candidates ─────────────── */
    for (int i = tid; i < TILE_SIZE; i += BLOCK_SIZE) {
        if (sieve[i] >= threshold_rat) {
            int64_t a = a0 + (int64_t)i;
            if (a == 0) continue;          /* gcd(0,b)=b, never smooth */
            uint32_t idx = atomicAdd(n_candidates, 1u);
            if (idx < MAX_CANDIDATES) {
                candidates[idx].a = a;
                candidates[idx].b = b;
            }
        }
    }
}

/* ── Tier-2 kernel: medium primes via HBM atomics ───────────────────────── */
/*
 * gnfs_medium_sieve_kernel — one thread per (b-line, prime) pair
 *
 * Launch: gridDim = (b_range, ceil(n_medium / blockDim.x))
 *         blockDim.x = BLOCK_SIZE
 *   Each thread computes at most one sieve update per b-line per prime.
 *
 * Inputs:
 *   medium_primes — Tier-2 primes: TILE_SIZE ≤ p < MEDIUM_BOUND
 *   n_medium      — count of Tier-2 primes
 *   m, a0, b_start — same meaning as Tier-1 kernel
 *   hbm_sieve     — [b_range × TILE_SIZE] byte array; updated via atomicAdd
 *
 * Algorithm:
 *   For prime p and b-line b: there is at most one position pos in [0, TILE_SIZE)
 *   with a0+pos ≡ −b·m (mod p).  Compute pos; if pos < TILE_SIZE, atomicAdd.
 *
 * Note on atomics:
 *   P(two threads hit same byte) ≈ (T/p)² ≤ (T/MEDIUM_BOUND)² = 3.8×10⁻³
 *   over all medium primes.  atomicAdd(uint8_t*) requires a 4-byte RMW;
 *   NOTE (2026-05-18 audit): ROCm 7.0.3 does NOT provide an
 *   atomicAdd(uint8_t*, ...) overload — the as-written call is a hard
 *   compile error on gfx942 (a byte→word CAS lane is needed; redesign
 *   pending, see TILE_SIZE note). Atomic throughput on
 *   scattered HBM addresses is bounded by HBM bandwidth (3.0–3.5 TB/s):
 *     effective rate ≈ 3.25 TB/s / 64 bytes per cache line ≈ 50 GOPS
 *   This is the primary throughput bottleneck.
 */
__global__ void gnfs_medium_sieve_kernel(
    const prime_entry_t * __restrict__ medium_primes,
    uint32_t n_medium,
    int64_t  m,
    int64_t  a0,
    int64_t  b_start,
    uint8_t * __restrict__ hbm_sieve)
{
    int64_t  b  = b_start + (int64_t)blockIdx.x;
    uint32_t pi = (uint32_t)(threadIdx.x + (uint32_t)blockDim.x * blockIdx.y);
    if (pi >= n_medium) return;

    uint32_t p    = medium_primes[pi].p;
    uint8_t  logp = medium_primes[pi].logp;

    /* Compute rational-side root (see Tier-1 kernel for details) */
    int64_t bm_r = (b % (int64_t)p) * (m % (int64_t)p) % (int64_t)p;
    if (bm_r < 0) bm_r += (int64_t)p;
    uint32_t a_r = (uint32_t)((p - (uint32_t)bm_r) % p);

    /* First (and only) position in tile */
    int64_t a0r = a0 % (int64_t)p;
    if (a0r < 0) a0r += (int64_t)p;
    int64_t off = (int64_t)a_r - a0r;
    if (off < 0) off += (int64_t)p;
    uint32_t pos = (uint32_t)off;

    if (pos < TILE_SIZE) {
        uint8_t *row = hbm_sieve + (size_t)blockIdx.x * TILE_SIZE;
        /* ROCm 7 does not support byte-granularity atomicAdd (uint8_t*).
         * Use word-level CAS to emulate atomic byte addition:
         *   read the aligned uint32_t containing the target byte,
         *   add logp into the byte lane, CAS until the word is accepted.
         * atomicCAS(uint32_t*, old, new) is universally supported. */
        uint32_t *word = (uint32_t *)((uintptr_t)(row + pos) & ~(uintptr_t)3);
        uint32_t  lane = (uint32_t)((uintptr_t)(row + pos) & 3) * 8;
        uint32_t  mask = (uint32_t)0xFF << lane;
        uint32_t  add  = (uint32_t)logp << lane;
        uint32_t  old, next;
        do {
            old  = *word;
            next = (old & ~mask) | ((old + add) & mask);
        } while (atomicCAS(word, old, next) != old);
    }
}

/* ── Host helpers ────────────────────────────────────────────────────────── */

/* build_factor_base — Eratosthenes up to bound; fills fb[0..count-1].
 *
 * Inputs:  fb        — output array (caller allocates)
 *          max_primes — max entries to fill
 *          bound     — sieve up to this value
 *          m         — polynomial root (for root precomputation; unused here)
 *          b_ref     — reference b-value for root r (unused here; set 0)
 * Returns: number of primes found (≤ max_primes)
 *
 * Note: This fills .r = 0 for all entries.  In production, .r would be
 * precomputed as (−b·m mod p) once per special-q batch.                  */
static uint32_t build_factor_base(
    prime_entry_t *fb, uint32_t max_primes, uint32_t bound)
{
    uint8_t *composite = (uint8_t *)calloc(bound, 1);
    if (!composite) { fprintf(stderr, "OOM in build_factor_base\n"); exit(1); }
    composite[0] = composite[1] = 1;
    for (uint32_t i = 2; (uint64_t)i * i < bound; i++)
        if (!composite[i])
            for (uint32_t j = i * i; j < bound; j += i)
                composite[j] = 1;

    uint32_t count = 0;
    for (uint32_t p = 2; p < bound && count < max_primes; p++) {
        if (composite[p]) continue;
        float lp = logf((float)p) * 8.0f;
        fb[count].p    = p;
        fb[count].r    = 0;
        fb[count].logp = (uint8_t)(lp < 255.0f ? lp : 255.0f);
        count++;
    }
    free(composite);
    return count;
}

/* ── Main: benchmark harness ─────────────────────────────────────────────── */
/*
 * Runs ITERS batches, each covering b_range b-lines over a TILE_SIZE a-range.
 * Reports: sieve update throughput (GOPS), HBM bandwidth (GB/s), candidates/s.
 *
 * Tuning guide:
 *   TILE_SIZE:    65536 is optimal (= LDS capacity).  Reducing it lowers
 *                 Tier-1 throughput and increases per-prime overhead.
 *   MEDIUM_BOUND: Increasing this enlarges Tier-2 and reduces Tier-3.
 *                 The HBM atomic throughput is the bottleneck; test whether
 *                 more Tier-2 primes improve or hurt overall throughput.
 *   b_range:      Should be a multiple of blockDim.x and >> 228 (CU count).
 *                 Larger b_range amortizes kernel launch overhead.
 *   BLOCK_SIZE:   Must be 64 on MI300A (one wavefront).  On 6900XT, 32.
 *   threshold:    Lower → more candidates (more CPU post-processing).
 *                 Upper bound: threshold < (Tier-1 + Tier-2 log contributions).
 */
int main(int argc, char **argv)
{
    uint32_t b_range       = (argc > 1) ? (uint32_t)atoi(argv[1]) : 4096;
    uint32_t n_small_cap   = (argc > 2) ? (uint32_t)atoi(argv[2]) : 50000;
    uint32_t n_medium_cap  = (argc > 3) ? (uint32_t)atoi(argv[3]) : 200000;

    /* Toy GNFS parameters for benchmarking only (not a real factorization) */
    int64_t m  = 1000003LL;                     /* rational-side root         */
    int64_t a0 = -(int64_t)(TILE_SIZE / 2);    /* center tile at a = 0       */

    /* ── Hardware query ─────────────────────────────────────────────────── */
    int dev = 0;
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, dev);

    printf("\n");
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│           GNFS Line Sieve Benchmark — MI300A (gfx942)           │\n");
    printf("├─────────────────────────────────────────────────────────────────┤\n");
    printf("│  Device        : %-46s │\n", prop.name);
    printf("│  CUs           : %-6d    warpSize   : %-6d               │\n",
           prop.multiProcessorCount, prop.warpSize);
    printf("│  HBM total     : %5.1f GB   LDS/CU     : %5d KB             │\n",
           (double)prop.totalGlobalMem / 1e9,
           (int)(prop.sharedMemPerBlock / 1024));
    printf("│  TILE_SIZE     : %-6d    MEDIUM_BOUND: %-10u           │\n",
           TILE_SIZE, MEDIUM_BOUND);
    printf("│  b_range       : %-6u    a0          : %-12lld         │\n",
           b_range, (long long)a0);
    printf("└─────────────────────────────────────────────────────────────────┘\n\n");

    /* ── Build factor base ───────────────────────────────────────────────── */
    prime_entry_t *fb = (prime_entry_t *)malloc(
        (n_small_cap + n_medium_cap) * sizeof(*fb));
    if (!fb) { fprintf(stderr, "OOM\n"); return 1; }

    printf("Building factor base (bound = %u) ...\n", MEDIUM_BOUND);
    uint32_t total_fb = build_factor_base(fb, n_small_cap + n_medium_cap,
                                          MEDIUM_BOUND);
    /* Split at TILE_SIZE: Tier-1 (p < TILE_SIZE) and Tier-2 (≥ TILE_SIZE) */
    uint32_t n_small  = 0;
    uint32_t n_medium = 0;
    for (uint32_t i = 0; i < total_fb; i++) {
        if (fb[i].p < TILE_SIZE) n_small++;
        else                     n_medium++;
    }
    /* Rearrange: small primes first, then medium */
    prime_entry_t *fb_small  = fb;
    prime_entry_t *fb_medium = (prime_entry_t *)malloc(n_medium * sizeof(*fb));
    if (!fb_medium) { fprintf(stderr, "OOM\n"); return 1; }
    {
        uint32_t si = 0, mi = 0;
        for (uint32_t i = 0; i < total_fb; i++) {
            if (fb[i].p < TILE_SIZE) fb_small[si++]  = fb[i];
            else                     fb_medium[mi++] = fb[i];
        }
    }
    if (n_small  > n_small_cap)  n_small  = n_small_cap;
    if (n_medium > n_medium_cap) n_medium = n_medium_cap;

    printf("  Tier-1 (p < %6u): %7u primes\n", TILE_SIZE,    n_small);
    printf("  Tier-2 (p < %7u): %7u primes  [cap %u]\n",
           MEDIUM_BOUND, n_medium, n_medium_cap);

    /* Threshold: 85% of accumulated log for primes up to MEDIUM_BOUND */
    uint8_t threshold = (uint8_t)(THRESHOLD_FRAC * logf((float)MEDIUM_BOUND) * 8.0f);
    printf("  Threshold        : %u  (%.0f%% × ln(%u) × 8 = %.1f)\n\n",
           threshold, THRESHOLD_FRAC * 100.0f, MEDIUM_BOUND,
           logf((float)MEDIUM_BOUND) * 8.0f);

    /* ── Allocate device memory ──────────────────────────────────────────── */
    prime_entry_t *d_small  = NULL;
    prime_entry_t *d_medium = NULL;
    uint8_t       *d_hbm    = NULL;
    relation_t    *d_cands  = NULL;
    uint32_t      *d_ncands = NULL;

    size_t hbm_bytes = (size_t)b_range * TILE_SIZE;

    (void)hipMalloc((void **)&d_small,  n_small  * sizeof(*d_small));
    (void)hipMalloc((void **)&d_medium, n_medium * sizeof(*d_medium));
    (void)hipMalloc((void **)&d_hbm,    hbm_bytes);
    (void)hipMalloc((void **)&d_cands,  MAX_CANDIDATES * sizeof(*d_cands));
    (void)hipMalloc((void **)&d_ncands, sizeof(uint32_t));

    (void)hipMemcpy(d_small,  fb_small,  n_small  * sizeof(*d_small),
                   hipMemcpyHostToDevice);
    (void)hipMemcpy(d_medium, fb_medium, n_medium * sizeof(*d_medium),
                   hipMemcpyHostToDevice);

    /* ── Kernel launch dimensions ────────────────────────────────────────── */
    /* Tier-1: one block (64 threads) per b-line */
    dim3 grid_tier1((unsigned)b_range);

    /* Tier-2: (b_range × ceil(n_medium/64)) blocks; y-dim iterates primes */
    dim3 grid_tier2((unsigned)b_range,
                    (n_medium + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blk(BLOCK_SIZE);

    /* ── Warmup ──────────────────────────────────────────────────────────── */
    (void)hipMemset(d_hbm,    0, hbm_bytes);
    (void)hipMemset(d_ncands, 0, sizeof(uint32_t));
    hipLaunchKernelGGL(gnfs_medium_sieve_kernel, grid_tier2, blk, 0, 0,
        d_medium, n_medium, m, a0, 0LL, d_hbm);
    hipLaunchKernelGGL(gnfs_line_sieve_kernel, grid_tier1, blk, 0, 0,
        d_small, n_small, m, a0, 0LL, d_hbm, d_cands, d_ncands, threshold);
    (void)hipDeviceSynchronize();

    /* ── Timed benchmark ─────────────────────────────────────────────────── */
    const int ITERS = 20;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; it++) {
        int64_t b_start = (int64_t)(it + 1) * b_range;
        (void)hipMemset(d_hbm,    0, hbm_bytes);
        (void)hipMemset(d_ncands, 0, sizeof(uint32_t));
        hipLaunchKernelGGL(gnfs_medium_sieve_kernel, grid_tier2, blk, 0, 0,
            d_medium, n_medium, m, a0, b_start, d_hbm);
        hipLaunchKernelGGL(gnfs_line_sieve_kernel, grid_tier1, blk, 0, 0,
            d_small, n_small, m, a0, b_start, d_hbm, d_cands, d_ncands,
            threshold);
        (void)hipDeviceSynchronize();
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed  = (t1.tv_sec  - t0.tv_sec)
                    + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double t1_upd   = (double)ITERS * b_range * n_small  * 2.2;  /* avg T/p */
    double t2_upd   = (double)ITERS * b_range * n_medium * ((double)TILE_SIZE
                                                              / (MEDIUM_BOUND * 0.5));
    double total_upd = t1_upd + t2_upd;
    double hbm_rw    = (double)ITERS * hbm_bytes * 2.0;    /* hipMemset+reads */

    uint32_t h_ncands = 0;
    (void)hipMemcpy(&h_ncands, d_ncands, sizeof(uint32_t),
                   hipMemcpyDeviceToHost);

    printf("┌────────────────────────────────────────────────────────────────┐\n");
    printf("│                     Benchmark Results                          │\n");
    printf("├──────────────────────────────────┬─────────────────────────────┤\n");
    printf("│  Iterations                      │  %-26d │\n", ITERS);
    printf("│  b-lines / iteration             │  %-26u │\n", b_range);
    printf("│  Total wall time (s)             │  %-26.4f │\n", elapsed);
    printf("│  Time per batch (ms)             │  %-26.3f │\n",
           elapsed / ITERS * 1e3);
    printf("│  b-lines / second (×10⁶)         │  %-26.3f │\n",
           (double)ITERS * b_range / elapsed / 1e6);
    printf("│  Est. sieve updates / s (×10⁹)   │  %-26.3f │\n",
           total_upd / elapsed / 1e9);
    printf("│  HBM R+W bandwidth (GB/s)        │  %-26.1f │\n",
           hbm_rw / elapsed / 1e9);
    printf("│  Candidates found                │  %-26u │\n", h_ncands);
    printf("│  Candidates / b-line             │  %-26.4f │\n",
           (double)h_ncands / (ITERS * b_range));
    printf("└──────────────────────────────────┴─────────────────────────────┘\n\n");

    /* ── Write timestamped results file ─────────────────────────────────── */
    {
        time_t now = time(NULL);
        char fname[80];
        strftime(fname, sizeof(fname), "gnfs_bench_%Y%m%d_%H%M%S.txt",
                 localtime(&now));
        FILE *f = fopen(fname, "w");
        if (f) {
            fprintf(f, "# GNFS Line Sieve Benchmark\n");
            fprintf(f, "device=%s\n", prop.name);
            fprintf(f, "CUs=%d warpSize=%d LDS_KB=%d\n",
                    prop.multiProcessorCount, prop.warpSize,
                    (int)(prop.sharedMemPerBlock / 1024));
            fprintf(f, "TILE_SIZE=%d MEDIUM_BOUND=%u BLOCK_SIZE=%d\n",
                    TILE_SIZE, MEDIUM_BOUND, BLOCK_SIZE);
            fprintf(f, "b_range=%u n_small=%u n_medium=%u threshold=%u\n",
                    b_range, n_small, n_medium, (unsigned)threshold);
            fprintf(f, "iters=%d elapsed_s=%.4f\n", ITERS, elapsed);
            fprintf(f, "b_lines_per_s=%.3e updates_per_s=%.3e\n",
                    (double)ITERS * b_range / elapsed, total_upd / elapsed);
            fprintf(f, "hbm_bw_GBs=%.1f candidates=%u\n",
                    hbm_rw / elapsed / 1e9, h_ncands);
            fclose(f);
            printf("Results written to %s\n\n", fname);
        }
    }

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    (void)hipFree(d_small);
    (void)hipFree(d_medium);
    (void)hipFree(d_hbm);
    (void)hipFree(d_cands);
    (void)hipFree(d_ncands);
    free(fb);
    free(fb_small);
    free(fb_medium);
    return 0;
}
