// mfma_probe.hip — REFERENCE TEMPLATE only (not built by default).
// Canonical copy: ~/mi300a-dev/mfma_probe.hip. Mirrored here 2026-04-29
// so readers of the NTT tree can see the canonical per-arch-gated MFMA
// pattern in context.
//
// **MFMA is NOT used by NTT itself** — see
// ~/ntt/results/framework/A2_MFMA_EVALUATION.md for why
// (modular integer arithmetic does not survive FP32 / FP64 / INT32 MFMA
// without re-reduction; mantissa precision is insufficient for q ≥ ~14
// bits). Keep this file as a pattern reference for adjacent kernels
// (e.g. lattice INT8 bucketing in BKZ) where MFMA *does* apply.
//
// Original header follows.
// ──────────────────────────────────────────────────────────────────────
// mfma_probe.hip — compile-only template for MFMA matrix-unit kernels.
//
// MFMA (Matrix Fused Multiply-Add) is a CDNA-only instruction family
// available on gfx908+ (MI100/MI200/MI300A). It is NOT present on
// RDNA gfx1030 (6900 XT). This file demonstrates a portable shape:
//
//   - Under -DMFMA_TARGET (set automatically by Makefile when
//     HIP_ARCH=gfx942), the kernel uses
//     __builtin_amdgcn_mfma_f32_16x16x4f32 to issue a real 16x16x4
//     FP32 matrix multiply on the matrix unit.
//   - Otherwise, the same kernel falls back to a scalar VALU
//     implementation that produces the same result (slowly).
//
// This is meant as a TEMPLATE. The real test is that gfx942 builds
// pull in the MFMA intrinsic without any module/sdk changes.
//
// Build matrix:
//   make mfma_probe                       # gfx1030 default (scalar fallback)
//   HIP_ARCH=gfx942 make -B mfma_probe    # gfx942 (real MFMA)
//
// Both must compile cleanly. Only the gfx942 binary actually exercises
// the matrix unit; the gfx1030 binary is a smoke test of the fallback
// path and should be runnable on the 6900 XT.

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>

#if defined(MFMA_TARGET)
// ============================================================================
// gfx942 path: real MFMA matrix multiply
// ============================================================================
typedef float v4f __attribute__((ext_vector_type(4)));

__global__ void k_mfma_16x16x4(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float       *__restrict__ C) {
    // 1 wave (64 threads on CDNA3) computes a 16x16 FP32 output tile.
    // Intrinsic signature (ROCm 7): mfma_f32_16x16x4f32(float a, float b,
    //   v4f c, int cbsz, int abid, int blgp) -> v4f. Each lane supplies one
    // scalar from A and B; c is the per-lane 4-element accumulator slice.
    int lane = threadIdx.x;
    float a = A[lane % 64];
    float b = B[lane % 64];
    v4f   c = {0.0f, 0.0f, 0.0f, 0.0f};
    c = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
    // Each lane writes its 4-element slice contiguously.
    if (lane < 64) {
        for (int i = 0; i < 4; i++) C[lane*4 + i] = c[i];
    }
}

#else
// ============================================================================
// gfx1030 path: scalar VALU fallback (same math, no matrix unit)
// ============================================================================
__global__ void k_mfma_16x16x4(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float       *__restrict__ C) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= 16) return;
    for (int col = 0; col < 16; col++) {
        float acc = 0.0f;
        for (int k = 0; k < 4; k++) {
            acc += A[row*4 + k] * B[k*16 + col];
        }
        C[row*16 + col] = acc;
    }
}
#endif

int main() {
    const int M = 16, N = 16, K = 4;
    float *A, *B, *C;
    (void)hipMalloc(&A, M*K*sizeof(float));
    (void)hipMalloc(&B, K*N*sizeof(float));
    (void)hipMalloc(&C, M*N*sizeof(float));

    // Initialize A,B to identity-ish so the result is deterministic
    float hA[M*K], hB[K*N];
    for (int i=0;i<M*K;i++) hA[i] = (i%5)*0.25f;
    for (int i=0;i<K*N;i++) hB[i] = ((i+1)%7)*0.125f;
    (void)hipMemcpy(A, hA, sizeof(hA), hipMemcpyHostToDevice);
    (void)hipMemcpy(B, hB, sizeof(hB), hipMemcpyHostToDevice);

#if defined(MFMA_TARGET)
    printf("mfma_probe: gfx942 path (MFMA 16x16x4 FP32)\n");
    k_mfma_16x16x4<<<1, 64>>>(A, B, C);
#else
    printf("mfma_probe: gfx1030 fallback (scalar VALU)\n");
    k_mfma_16x16x4<<<1, 16>>>(A, B, C);
#endif
    (void)hipDeviceSynchronize();

    float hC[M*N];
    (void)hipMemcpy(hC, C, sizeof(hC), hipMemcpyDeviceToHost);
    // Sanity: print first 4 elements
    printf("C[0..3] = %.3f %.3f %.3f %.3f\n", hC[0], hC[1], hC[2], hC[3]);
    (void)hipFree(A); (void)hipFree(B); (void)hipFree(C);
    return 0;
}
