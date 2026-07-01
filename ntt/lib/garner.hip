/*
 * garner.hip — Garner CRT reconstruction and supporting functions
 *
 * After all four NTTs complete on the GPU, this reconstructs each ~248-bit
 * coefficient from four 64-bit residues using Garner's algorithm.
 *
 * TWO PATHS (same per-coefficient math):
 *   - PRODUCTION: garner_reconstruct_gpu / _gpu_t — a GPU kernel, one thread
 *     per coefficient (Garner is ~110 ALU ops/coeff, embarrassingly parallel).
 *     This is what the engine (ntt_mul.hip) calls.
 *   - CPU OpenMP: garner_reconstruct — runs on all 96 Zen 4 cores (4 APUs x 24)
 *     via OpenMP; kept for the host-only CRT verification test (lib/test.hip)
 *     and as the non-GPU reconstruction path. NUMA notes below apply to it.
 *
 * NUMA binding for the CPU path (OMP_PROC_BIND=spread OMP_PLACES=cores):
 *   Threads  0-23  run on APU 0's cores → read d_r0 locally at HBM speed
 *   Threads 24-47  run on APU 1's cores → read d_r1 locally
 *   Threads 48-71  run on APU 2's cores → read d_r2 locally
 *   Threads 72-95  run on APU 3's cores → read d_r3 locally
 *   Remote reads (3 of 4 per thread) traverse Infinity Fabric at ~104 GB/s.
 *
 * d_r0..d_r3 are hipMalloc'd on their respective devices. After
 * hipDeviceEnablePeerAccess and hipStreamSynchronize, the CPU reads
 * these pointers directly — Infinity Fabric serves the loads transparently.
 *
 * Garner's algorithm for 4 primes:
 *   a0 = r0
 *   a1 = (r1 - a0) * inv(p0, p1)                              mod p1
 *   a2 = ((r2 - a0 - a1*p0) * inv(p0,p2) ...) * inv(p1,p2)   mod p2
 *   a3 = similar triangular reduction                          mod p3
 *   x  = a0 + a1*p0 + a2*p0*p1 + a3*p0*p1*p2   (as 256-bit integer)
 * All coefficient indices are independent: embarrassingly parallel.
 */

/* garner.hip uses NO HIP runtime API (pure CPU + OpenMP); the header is
 * only needed when hipcc compiles this TU as part of a device build.
 * Guarding it (mirrors primes.h / shoup.h) makes garner.hip host-
 * compilable for the committed CPU CRT verification test. */
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "primes.h"
#include "crt_ntt.h"

/* GPU-SAFETY: HIP_CHECK aborts on the host the instant a HIP call fails,
 * so a transient failure (e.g. allocator exhaustion on one of 4 APUs)
 * cannot cascade into a kernel reading a NULL device pointer / GPU reset.
 * Mirrors lib/arith/ntt_mul.hip. Cold setup/teardown only; zero hot cost. */
#define HIP_CHECK(expr) do {                                                  \
    hipError_t _e = (expr);                                                   \
    if (_e != hipSuccess) {                                                   \
        fprintf(stderr, "lib: HIP error %d (%s) at %s:%d: %s\n",            \
                (int)_e, hipGetErrorString(_e), __FILE__, __LINE__, #expr);   \
        fflush(stderr);                                                       \
        exit(1);                                                              \
    }                                                                         \
} while (0)

/* ─── 256-bit helpers ────────────────────────────────────────────────────── */

/* acc += scalar * constant (all limbs, with carry propagation) */
static inline void u256_addmul(U256* acc, uint64_t scalar, const U256* constant)
{
    __uint128_t carry = 0;
    int i;
    for (i = 0; i < 4; i++) {
        carry += (__uint128_t)scalar * constant->w[i] + acc->w[i];
        acc->w[i] = (uint64_t)carry;
        carry >>= 64;
    }
    /* carry must be 0 if result fits in 256 bits (it does for our modulus) */
}

/* Compute a*b as U256 (128-bit product stored in low two limbs) */
static inline U256 u256_from_mul64(uint64_t a, uint64_t b)
{
    __uint128_t p = (__uint128_t)a * b;
    U256 r;
    r.w[0] = (uint64_t)p;
    r.w[1] = (uint64_t)(p >> 64);
    r.w[2] = 0;
    r.w[3] = 0;
    return r;
}

/* Multiply U256 by 64-bit scalar, return new U256 */
static inline U256 u256_mul_scalar(const U256* a, uint64_t s)
{
    U256 r = {{0, 0, 0, 0}};
    __uint128_t carry = 0;
    int i;
    for (i = 0; i < 4; i++) {
        carry += (__uint128_t)a->w[i] * s;
        r.w[i] = (uint64_t)carry;
        carry >>= 64;
    }
    return r;
}

/* ─── Precompute all Garner constants (call once at startup) ─────────────── */
GarnerConsts garner_precompute(void)
{
    GarnerConsts g;

    /* Modular inverses via Fermat's little theorem.                          */
    /* Always reduce the base mod the target modulus first, since e.g.        */
    /* P0 > P1 and modpow(P0, P1-2, P1) would compute P0 mod P1 internally, */
    /* but being explicit avoids relying on that implementation detail.       */
    g.inv_p0_p1  = modpow(P0 % P1, P1 - 2, P1);
    g.inv_p0_p2  = modpow(P0 % P2, P2 - 2, P2);
    g.inv_p0_p3  = modpow(P0 % P3, P3 - 2, P3);
    g.inv_p1_p2  = modpow(P1 % P2, P2 - 2, P2);
    g.inv_p1_p3  = modpow(P1 % P3, P3 - 2, P3);
    g.inv_p2_p3  = modpow(P2 % P3, P3 - 2, P3);

    /* Pre-reduced products: avoid % inside the parallel Garner loop         */
    g.p0_mod_p2   = P0 % P2;
    g.p0_mod_p3   = P0 % P3;
    g.p1_mod_p3   = P1 % P3;
    g.p0p1_mod_p3 = (uint64_t)((__uint128_t)(P0 % P3) * (P1 % P3) % P3);

    /* Multi-precision products for 256-bit assembly                         */
    g.P0_256       = (U256){{P0, 0, 0, 0}};
    g.P0P1_256     = u256_from_mul64(P0, P1);
    g.P0P1P2_256   = u256_mul_scalar(&g.P0P1_256, P2);

    return g;
}

/* ─── Single-coefficient Garner reconstruction ───────────────────────────── */
static inline U256 garner_one(
    uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3,
    const GarnerConsts* g)
{
    /* Step 1: compute mixed-radix digits a0..a3 */

    uint64_t a0 = r0;

    /* a1 = (r1 - a0 mod p1) * inv_p0_p1 mod p1 */
    uint64_t t1 = r1 - (a0 % P1);
    if (t1 > r1) t1 += P1;          /* underflow: result was negative mod p1 */
    uint64_t a1 = mulmod<1>(t1, g->inv_p0_p1);

    /* a2 = (r2 - a0 mod p2 - a1*p0 mod p2) * inv_p1_p2 mod p2 */
    uint64_t base2 = addmod<2>(a0 % P2, mulmod<2>(a1, g->p0_mod_p2));
    uint64_t t2 = r2 - base2;
    if (t2 > r2) t2 += P2;
    if (t2 >= P2) t2 -= P2;
    /* a2 = t2 * (P0*P1)^{-1} = t2 * inv_p0_p2 * inv_p1_p2 (mod P2) */
    uint64_t a2 = mulmod<2>(mulmod<2>(t2, g->inv_p0_p2), g->inv_p1_p2);

    /* a3 = (r3 - a0 - a1*p0 - a2*p0*p1) * inv_p2_p3 mod p3 */
    uint64_t base3 = a0 % P3;
    base3 = addmod<3>(base3, mulmod<3>(a1, g->p0_mod_p3));
    base3 = addmod<3>(base3, mulmod<3>(a2, g->p0p1_mod_p3));
    uint64_t t3 = r3 - base3;
    if (t3 > r3) t3 += P3;
    if (t3 >= P3) t3 -= P3;
    /* a3 = t3 * (P0*P1*P2)^{-1} = t3 * inv_p0_p3 * inv_p1_p3 * inv_p2_p3 (mod P3) */
    uint64_t a3 = mulmod<3>(mulmod<3>(mulmod<3>(t3, g->inv_p0_p3), g->inv_p1_p3), g->inv_p2_p3);

    /* Step 2: 256-bit assembly                                               */
    /* x = a0 + a1*P0 + a2*P0*P1 + a3*P0*P1*P2                              */
    U256 x = {{a0, 0, 0, 0}};
    u256_addmul(&x, a1, &g->P0_256);
    u256_addmul(&x, a2, &g->P0P1_256);
    u256_addmul(&x, a3, &g->P0P1P2_256);
    return x;
}

#ifdef __HIPCC__
/* ─── GPU Garner kernel ──────────────────────────────────────────────────── */
/* Same per-coefficient math as garner_one, runs on the GPU. Each thread
 * reconstructs one U256 coefficient. The kernel reads four residues —
 * on a single-device build (6900XT) all four buffers are on the same
 * device; on MI300A 4-APU peer access (already enabled at init) lets the
 * kernel pull residues from the three peer APUs via Infinity Fabric.
 *
 * Replaces the CPU OpenMP Garner for the production path. Compute-wise
 * Garner is ~110 ALU ops/coefficient — embarrassingly parallel and a near-
 * perfect fit for the GPU. The CPU implementation above remains for the
 * host-only verification test (lib/test.hip CPU-pointer Garner case).  */
__device__ static inline U256 garner_one_dev(
    uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3,
    const GarnerConsts* g)
{
    uint64_t a0 = r0;
    uint64_t t1 = r1 - (a0 % P1);
    if (t1 > r1) t1 += P1;
    uint64_t a1 = mulmod<1>(t1, g->inv_p0_p1);

    uint64_t base2 = addmod<2>(a0 % P2, mulmod<2>(a1, g->p0_mod_p2));
    uint64_t t2 = r2 - base2;
    if (t2 > r2) t2 += P2;
    if (t2 >= P2) t2 -= P2;
    uint64_t a2 = mulmod<2>(mulmod<2>(t2, g->inv_p0_p2), g->inv_p1_p2);

    uint64_t base3 = a0 % P3;
    base3 = addmod<3>(base3, mulmod<3>(a1, g->p0_mod_p3));
    base3 = addmod<3>(base3, mulmod<3>(a2, g->p0p1_mod_p3));
    uint64_t t3 = r3 - base3;
    if (t3 > r3) t3 += P3;
    if (t3 >= P3) t3 -= P3;
    uint64_t a3 = mulmod<3>(mulmod<3>(mulmod<3>(t3, g->inv_p0_p3),
                                       g->inv_p1_p3), g->inv_p2_p3);

    /* 256-bit assembly inlined: x = a0 + a1*P0 + a2*P0*P1 + a3*P0*P1*P2 */
    U256 x = {{a0, 0, 0, 0}};
    __uint128_t carry = 0;
    for (int i = 0; i < 4; i++) {
        carry += (__uint128_t)a1 * g->P0_256.w[i] + x.w[i];
        x.w[i] = (uint64_t)carry; carry >>= 64;
    }
    carry = 0;
    for (int i = 0; i < 4; i++) {
        carry += (__uint128_t)a2 * g->P0P1_256.w[i] + x.w[i];
        x.w[i] = (uint64_t)carry; carry >>= 64;
    }
    carry = 0;
    for (int i = 0; i < 4; i++) {
        carry += (__uint128_t)a3 * g->P0P1P2_256.w[i] + x.w[i];
        x.w[i] = (uint64_t)carry; carry >>= 64;
    }
    return x;
}

__global__ __launch_bounds__(256, 8)
static void garner_kernel(
    const uint64_t* __restrict__ d_r0,
    const uint64_t* __restrict__ d_r1,
    const uint64_t* __restrict__ d_r2,
    const uint64_t* __restrict__ d_r3,
    U256* __restrict__           d_out,
    int                          N,
    GarnerConsts                 g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    d_out[i] = garner_one_dev(d_r0[i], d_r1[i], d_r2[i], d_r3[i], &g);
}

/* GPU Garner launcher.  d_out must be a device buffer of N U256 entries on
 * the same device as d_r0 (single-device 6900XT: all four buffers live
 * there; MI300A: d_r1..d_r3 reached via peer access from device 0). */
void garner_reconstruct_gpu(
    const uint64_t* d_r0, const uint64_t* d_r1,
    const uint64_t* d_r2, const uint64_t* d_r3,
    U256* d_out, int N, const GarnerConsts* g,
    hipStream_t stream)
{
    const int block = 256;
    int grid = (N + block - 1) / block;
    if (grid < 1) grid = 1;
    garner_kernel<<<grid, block, 0, stream>>>(d_r0, d_r1, d_r2, d_r3,
                                              d_out, N, *g);
}

/* I15 transposed-read Garner: the even-log_n INTT output is left TRANSPOSED (the
 * standalone output transpose_sq is eliminated), so coefficient k lives at the
 * M×M transposed position transpose(k) = (k%M)*M + (k/M). Reading the residues
 * from there and writing d_out[k] natural folds the un-transpose into the Garner
 * pass (which already streams every coefficient) — no separate transpose kernel,
 * and bigint_gather still reads d_out in natural order. */
__global__ __launch_bounds__(256, 8)
static void garner_kernel_t(
    const uint64_t* __restrict__ d_r0, const uint64_t* __restrict__ d_r1,
    const uint64_t* __restrict__ d_r2, const uint64_t* __restrict__ d_r3,
    U256* __restrict__ d_out, int N, int M, GarnerConsts g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    size_t t = (size_t)(i % M) * (size_t)M + (size_t)(i / M);   /* transpose(i) */
    d_out[i] = garner_one_dev(d_r0[t], d_r1[t], d_r2[t], d_r3[t], &g);
}

void garner_reconstruct_gpu_t(
    const uint64_t* d_r0, const uint64_t* d_r1,
    const uint64_t* d_r2, const uint64_t* d_r3,
    U256* d_out, int N, int M, const GarnerConsts* g, hipStream_t stream)
{
    const int block = 256;
    int grid = (N + block - 1) / block;
    if (grid < 1) grid = 1;
    garner_kernel_t<<<grid, block, 0, stream>>>(d_r0, d_r1, d_r2, d_r3,
                                                d_out, N, M, *g);
}
#endif /* __HIPCC__ */

/* ─── Parallel Garner reconstruction over N coefficients ─────────────────── */
/*                                                                            */
/* Precondition: all four NTT streams have been synchronized by the caller.  */
/* d_r0..d_r3 are hipMalloc'd on devices 0..3. After peer access is enabled, */
/* the CPU reads them via Infinity Fabric at ~104 GB/s per link.             */
/*                                                                            */
/* h_output is a host allocation (malloc or hipHostMalloc). Writing to host  */
/* memory from the CPU avoids routing output through IF unnecessarily.       */

void garner_reconstruct(
    const uint64_t* d_r0,
    const uint64_t* d_r1,
    const uint64_t* d_r2,
    const uint64_t* d_r3,
    U256* h_output,
    int N,
    const GarnerConsts* g)
{
    /* schedule(static): gives each thread a contiguous chunk of N/96         */
    /* coefficients, maximizing spatial locality for sequential HBM reads.   */
    /* Thread count = omp_get_max_threads() → honours OMP_NUM_THREADS (env
     * file sets 96 on the 4-APU MI300A; on the 16-core 5950X dev box this
     * avoids 96-thread oversubscription and lets the Garner CPU-scaling
     * sweep work). OMP_PROC_BIND=spread/OMP_PLACES=cores set at runtime.   */
    #pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int i = 0; i < N; i++) {
        /* Prefetch 32 iterations ahead from all four HBM pools.
         * MI300A Infinity Fabric cross-APU latency is ~150-200 cycles; the
         * earlier +16 distance was tuned for tighter PCIe access and left
         * cross-NUMA reads partially stalled. +32 covers the longer pipe
         * while staying within typical prefetch-buffer capacity. */
        if (i + 32 < N) {
            __builtin_prefetch(&d_r0[i + 32], 0, 1);
            __builtin_prefetch(&d_r1[i + 32], 0, 1);
            __builtin_prefetch(&d_r2[i + 32], 0, 1);
            __builtin_prefetch(&d_r3[i + 32], 0, 1);
        }
        h_output[i] = garner_one(d_r0[i], d_r1[i], d_r2[i], d_r3[i], g);
    }
}

/* ─── Prime verification ─────────────────────────────────────────────────── */
/* Call at startup. Returns 1 if all checks pass, 0 otherwise.               */
int verify_primes(void)
{
    /* Check constants match the computed values */
    const uint64_t expected[4] = {
        UINT64_C(0xffffffff00000001),   /* 2^64 - 2^32 + 1 */
        UINT64_C(0xffffffffff000001),   /* 2^64 - 2^24 + 1 */
        UINT64_C(0xfffffffc00000001),   /* 2^64 - 2^34 + 1 */
        UINT64_C(0xffffff0000000001)    /* 2^64 - 2^40 + 1 */
    };
    int ok = 1;
    int i;
    for (i = 0; i < 4; i++) {
        if (PRIMES[i] != expected[i]) {
            printf("verify_primes: PRIMES[%d] = %llx, expected %llx\n",
                   i, (unsigned long long)PRIMES[i],
                   (unsigned long long)expected[i]);
            ok = 0;
        }
        /* NTT-friendliness: 2^24 must divide p-1 (minimum across all four primes) */
        if ((PRIMES[i] - 1) % (UINT64_C(1) << 24) != 0) {
            printf("verify_primes: PRIMES[%d]: 2^24 does not divide p-1\n", i);
            ok = 0;
        }
    }
    if (ok) printf("[OK] verify_primes: all four primes valid\n");
    return ok;
}
