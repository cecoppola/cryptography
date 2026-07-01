/*
 * test_ntt_kernel.hip — Direct NTT kernel correctness test.
 *
 * Bypasses ntt_bigint_mul to test the GPU NTT pipeline in isolation.
 *
 * The forward (launch_ntt) and inverse (launch_intt_fused) kernels are, from
 * the caller's view, natural-order-in / natural-order-out standard (I)DFTs:
 * ntt_kernel reads its input through an internal bit-reversal and writes the
 * result in natural order, so the public transform is the plain
 *   X[k] = sum_n x[n] omega^{n k}  (and the matching N^{-1}-scaled inverse),
 * with omega = root^{(p-1)/N}. No bit-reversal is applied to the output —
 * single-frequency probes must therefore use the un-permuted spectrum.
 *
 * Known-answer tests (all mod PRIMES[0] = P0 = 2^64-2^32+1):
 *   KA1: NTT([1, 0, ..., 0]) → all outputs should be 1.
 *        (DFT of [1, 0, ..., 0] evaluated at omega^k is 1 for all k)
 *   KA2: INTT(NTT([1, 0, ..., 0])) / N → [1, 0, ..., 0].
 *        Verifies the round-trip.
 *   KA3: NTT([1, 0, ..., 0]) pointwise-mul NTT([0, 1, 0, ..., 0])
 *        → INTT → should be [0, 1, 0, ..., 0] (convolution: shift by 1).
 *        Verifies Hadamard and INTT.
 *   KA4: NTT(e_1) → geometric sequence omega^k (un-permuted single
 *        frequency; expected vector host-recomputed from omega).
 *   KA5: INTT([omega^k]) → e_1 = [0,1,0,...,0]. Direct inverse-DFT KAT,
 *        independent of NTT (does not call launch_ntt).
 *   KA6: Hadamard([1,1,...], [1,-1,1,-1,...]) → [1,-1,1,-1,...].
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../lib/primes.h"
#include "../../lib/shoup.h"
#include "../../lib/crt_ntt.h"

/* ─── Forward declarations (from ntt_kernel.hip compilation unit) ─────────── */
template<int PIDX>
void launch_ntt(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_intt_fused(uint64_t*, uint64_t, const ShoupPair*, const ShoupPair*, int, hipStream_t);
template<int PIDX>
void launch_hadamard(uint64_t*, const uint64_t*, int, hipStream_t);

/* ─── Twiddle generation for prime 0 ─────────────────────────────────────── */
static ShoupPair* make_tw(uint64_t p, uint64_t g_root, int log_n)
{
    int n_tw = (1 << log_n) - 1;
    ShoupPair* h = (ShoupPair*)malloc((size_t)n_tw * sizeof(ShoupPair));
    uint64_t omega = modpow(g_root, (p - 1) / (uint64_t)(1 << log_n), p);
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
    (void)hipMalloc(&d, (size_t)n_tw * sizeof(ShoupPair));
    (void)hipMemcpy(d, h, (size_t)n_tw * sizeof(ShoupPair), hipMemcpyHostToDevice);
    free(h);
    return d;
}

static void check_hip(hipError_t e, const char* ctx)
{
    if (e != hipSuccess) {
        fprintf(stderr, "HIP error at %s: %s\n", ctx, hipGetErrorString(e));
        exit(1);
    }
}

int main(void)
{
    int log_n = 10;
    int N     = 1 << log_n;
    uint64_t p    = PRIMES[0];
    uint64_t root = PRIM_ROOT[0];

    /* Query device */
    int n_devs = 0;
    check_hip(hipGetDeviceCount(&n_devs), "hipGetDeviceCount");
    printf("Device count: %d\n", n_devs);
    check_hip(hipSetDevice(0), "hipSetDevice");

    /* Precompute twiddle tables for prime 0 */
    printf("Building twiddle tables (log_n=%d) ...\n", log_n);
    uint64_t root_inv = modpow(root, p - 2, p);
    ShoupPair* d_tw         = make_tw(p, root, log_n);
    ShoupPair* d_tw_inv     = make_tw(p, root_inv, log_n);
    /* Cross-twiddle: not used in single-block path, can be NULL for log_n=10 */
    ShoupPair* d_tw_cross = NULL;
    uint64_t n_inv = modpow((uint64_t)N, p - 2, p);
    /* cross_inv_scaled: also not used for log_n<=10 */
    ShoupPair* d_tw_cross_inv = NULL;

    hipStream_t stream;
    check_hip(hipStreamCreate(&stream), "hipStreamCreate");

    uint64_t* d_a;
    uint64_t* d_b;
    check_hip(hipMalloc(&d_a, (size_t)N * sizeof(uint64_t)), "hipMalloc d_a");
    check_hip(hipMalloc(&d_b, (size_t)N * sizeof(uint64_t)), "hipMalloc d_b");

    uint64_t* h_in  = (uint64_t*)calloc(N, sizeof(uint64_t));
    uint64_t* h_out = (uint64_t*)calloc(N, sizeof(uint64_t));

    int passed = 0, total = 0;

    /* ── KA1: NTT([1, 0, ..., 0]) should give all-ones ─────────────────────── */
    {
        total++;
        printf("\nKA1: NTT([1,0,...,0]) -> all 1s\n");
        memset(h_in, 0, N * sizeof(uint64_t));
        h_in[0] = 1;

        check_hip(hipMemcpy(d_a, h_in, N * sizeof(uint64_t), hipMemcpyHostToDevice), "upload KA1");
        launch_ntt<0>(d_a, d_tw, d_tw_cross, log_n, stream);
        check_hip(hipStreamSynchronize(stream), "sync KA1");
        check_hip(hipGetLastError(), "kernel error KA1");
        check_hip(hipMemcpy(h_out, d_a, N * sizeof(uint64_t), hipMemcpyDeviceToHost), "download KA1");

        int ok = 1;
        int first_fail = -1;
        for (int i = 0; i < N; i++) {
            if (h_out[i] != 1) {
                if (first_fail < 0) first_fail = i;
                ok = 0;
            }
        }
        if (ok) {
            printf("  PASS: all %d outputs are 1\n", N);
            passed++;
        } else {
            printf("  FAIL: first mismatch at index %d: got %016llx expected 1\n",
                   first_fail, (unsigned long long)h_out[first_fail]);
            /* Print first 10 outputs for debugging */
            printf("  First 10 outputs: ");
            for (int i = 0; i < 10 && i < N; i++)
                printf("%016llx ", (unsigned long long)h_out[i]);
            printf("\n");
        }
    }

    /* ── KA2: Round-trip: INTT(NTT([1,0,...,0])) = [1, 0, ..., 0] ─────────── */
    {
        total++;
        printf("\nKA2: INTT(NTT([1,0,...,0])) -> [1,0,...,0]\n");
        memset(h_in, 0, N * sizeof(uint64_t));
        h_in[0] = 1;

        check_hip(hipMemcpy(d_a, h_in, N * sizeof(uint64_t), hipMemcpyHostToDevice), "upload KA2");
        launch_ntt<0>(d_a, d_tw, d_tw_cross, log_n, stream);
        launch_intt_fused<0>(d_a, n_inv, d_tw_inv, d_tw_cross_inv, log_n, stream);
        check_hip(hipStreamSynchronize(stream), "sync KA2");
        check_hip(hipGetLastError(), "kernel error KA2");
        check_hip(hipMemcpy(h_out, d_a, N * sizeof(uint64_t), hipMemcpyDeviceToHost), "download KA2");

        int ok = (h_out[0] == 1);
        for (int i = 1; i < N; i++) if (h_out[i] != 0) { ok = 0; break; }
        if (ok) {
            printf("  PASS: recovered [1, 0, ..., 0]\n");
            passed++;
        } else {
            printf("  FAIL:\n");
            printf("  h_out[0] = %016llx (expected 1)\n", (unsigned long long)h_out[0]);
            int first_nonzero = -1;
            for (int i = 1; i < N; i++) {
                if (h_out[i]) { first_nonzero = i; break; }
            }
            if (first_nonzero >= 0)
                printf("  first non-zero at h_out[%d] = %016llx\n",
                       first_nonzero, (unsigned long long)h_out[first_nonzero]);
        }
    }

    /* ── KA3: Convolution [1,0,...] * [0,1,0,...] = [0,1,0,...] ─────────────── */
    {
        total++;
        printf("\nKA3: conv([1,0,...], [0,1,0,...]) = [0,1,0,...]\n");
        memset(h_in, 0, N * sizeof(uint64_t));
        h_in[0] = 1;
        check_hip(hipMemcpy(d_a, h_in, N * sizeof(uint64_t), hipMemcpyHostToDevice), "upload d_a KA3");

        memset(h_in, 0, N * sizeof(uint64_t));
        h_in[1] = 1;
        check_hip(hipMemcpy(d_b, h_in, N * sizeof(uint64_t), hipMemcpyHostToDevice), "upload d_b KA3");

        launch_ntt<0>(d_a, d_tw, d_tw_cross, log_n, stream);
        launch_ntt<0>(d_b, d_tw, d_tw_cross, log_n, stream);
        launch_hadamard<0>(d_a, d_b, N, stream);
        launch_intt_fused<0>(d_a, n_inv, d_tw_inv, d_tw_cross_inv, log_n, stream);
        check_hip(hipStreamSynchronize(stream), "sync KA3");
        check_hip(hipGetLastError(), "kernel error KA3");
        check_hip(hipMemcpy(h_out, d_a, N * sizeof(uint64_t), hipMemcpyDeviceToHost), "download KA3");

        int ok = (h_out[0] == 0 && h_out[1] == 1);
        for (int i = 2; i < N; i++) if (h_out[i] != 0) { ok = 0; break; }
        if (ok) {
            printf("  PASS: result is [0, 1, 0, ..., 0]\n");
            passed++;
        } else {
            printf("  FAIL:\n");
            printf("  h_out[0]=%016llx h_out[1]=%016llx (expected 0, 1)\n",
                   (unsigned long long)h_out[0], (unsigned long long)h_out[1]);
            for (int i = 0; i < 8; i++)
                printf("  h_out[%d]=%016llx\n", i, (unsigned long long)h_out[i]);
        }
    }

    /* ── KA4: NTT(e_1) = the geometric sequence omega^k ──────────────────────── */
    /* launch_ntt<0>(log_n<=10) calls ntt_kernel, which reads its input in       */
    /* bit-reversed order (data[base+brev(i)] -> lds[i]) and writes its output   */
    /* in NATURAL order (lds[i] -> data[base+i]). The internal bit-reversal of   */
    /* the load is exactly what a CT-DIT pass needs to produce the *standard*    */
    /* forward DFT from natural-order input: from the caller's view the          */
    /* transform is natural-in / natural-out and computes                        */
    /*     X[k] = sum_{n} x[n] * omega^{n k}   (mod p),                          */
    /* with omega = root^{(p-1)/N} (the same omega make_tw bakes into d_tw).     */
    /* There is NO bitrev applied to the output, so the spectrum of a unit       */
    /* vector is a plain (un-permuted) single frequency. For e_1 (x[n]=delta_{n,1}) */
    /*     X[k] = omega^{1*k} = omega^k.                                          */
    /* The previous probe wrongly assumed bitrev-OUT ordering (expecting         */
    /* omega^{bitrev(1) k} = (-1)^k); that mis-modelled the kernel's I/O order   */
    /* and is the actual bug behind the old KA4 failure. This is a genuine KAT:  */
    /* the expected vector is the host-recomputed geometric sequence omega^k,    */
    /* not a round-trip identity. */
    {
        total++;
        printf("\nKA4: NTT(e_1) = geometric sequence omega^k (omega=root^((p-1)/N))\n");
        memset(h_in, 0, N * sizeof(uint64_t));
        h_in[1] = 1;  /* e_1: unit vector with 1 at position 1 */
        check_hip(hipMemcpy(d_a, h_in, N * sizeof(uint64_t), hipMemcpyHostToDevice), "upload KA4");
        launch_ntt<0>(d_a, d_tw, d_tw_cross, log_n, stream);
        check_hip(hipStreamSynchronize(stream), "sync KA4");
        check_hip(hipGetLastError(), "kernel error KA4");
        check_hip(hipMemcpy(h_out, d_a, N * sizeof(uint64_t), hipMemcpyDeviceToHost), "download KA4");

        /* Expected[k] = omega^k, omega = root^{(p-1)/N} mod p — recomputed on
         * the host from the SAME definition make_tw uses, independent of the
         * GPU result. (Sanity-checked offline: omega = 0x9d8f2ad78bfed972,
         * omega^N = 1, omega^{N/2} = p-1.) */
        uint64_t omega = modpow(root, (p - 1) / (uint64_t)N, p);
        int ok = 1;
        int first_fail = -1;
        uint64_t expected = 1;  /* omega^0 */
        for (int j = 0; j < N; j++) {
            if (h_out[j] != expected) {
                if (ok) { first_fail = j; }
                ok = 0;
            }
            expected = (uint64_t)((__uint128_t)expected * omega % p);
        }
        if (ok) {
            printf("  PASS: NTT(e_1)[k] = omega^k for all k\n");
            passed++;
        } else {
            uint64_t exp_ff = modpow(omega, (uint64_t)first_fail, p);
            printf("  FAIL at index %d: got %016llx expected %016llx\n",
                   first_fail, (unsigned long long)h_out[first_fail],
                   (unsigned long long)exp_ff);
            printf("  First 8 outputs: ");
            for (int i = 0; i < 8; i++)
                printf("%016llx ", (unsigned long long)h_out[i]);
            printf("\n");
        }
    }

    /* ── KA5: INTT([omega^k]) directly → e_1 = [0,1,0,...,0] ─────────────────── */
    /* Direct INTT KAT, independent of NTT/Hadamard (does NOT call launch_ntt).  */
    /* launch_intt_fused<0>(log_n<=10) runs ntt_kernel_scaled with omega^{-1}    */
    /* twiddles + N^{-1} scaling: same bit-reversed-IN / natural-OUT order as    */
    /* the forward kernel, so from the caller's view it computes the standard    */
    /* inverse DFT (natural-in / natural-out):                                   */
    /*     x[n] = N^{-1} sum_{k} y[k] * omega^{-n k}   (mod p).                  */
    /* Feeding y[k] = omega^k (the host-computed geometric sequence — the exact  */
    /* KA4 spectrum of e_1) must recover x = e_1 = [0,1,0,...,0], since this is  */
    /* the analytic inverse of X[k]=omega^k. The expected output is a precise    */
    /* derived array (delta at index 1), not a round-trip identity (KA2 already  */
    /* covers round-trips); the input is independently host-derived, so this is  */
    /* a genuine known-answer test of launch_intt_fused in isolation.            */
    /* The OLD probe fed [1,-1,1,-1,...] expecting e_1 — that holds only under   */
    /* a bitrev-out spectrum the kernel does not produce (same ordering bug as   */
    /* the old KA4); it is now corrected to the kernel's true natural ordering.  */
    {
        total++;
        printf("\nKA5: INTT([omega^k]) -> [0,1,0,...,0]\n");
        uint64_t omega = modpow(root, (p - 1) / (uint64_t)N, p);
        /* Upload y[k] = omega^k (host-derived, == KA4's expected spectrum). */
        uint64_t cur = 1;  /* omega^0 */
        for (int i = 0; i < N; i++) {
            h_in[i] = cur;
            cur = (uint64_t)((__uint128_t)cur * omega % p);
        }
        check_hip(hipMemcpy(d_a, h_in, N * sizeof(uint64_t), hipMemcpyHostToDevice), "upload KA5");
        launch_intt_fused<0>(d_a, n_inv, d_tw_inv, d_tw_cross_inv, log_n, stream);
        check_hip(hipStreamSynchronize(stream), "sync KA5");
        check_hip(hipGetLastError(), "kernel error KA5");
        check_hip(hipMemcpy(h_out, d_a, N * sizeof(uint64_t), hipMemcpyDeviceToHost), "download KA5");
        int ok = (h_out[0] == 0 && h_out[1] == 1);
        for (int i = 2; i < N; i++) if (h_out[i]) { ok = 0; break; }
        if (ok) {
            printf("  PASS: recovered [0, 1, 0, ..., 0]\n");
            passed++;
        } else {
            printf("  FAIL: h_out[0]=%016llx h_out[1]=%016llx\n",
                   (unsigned long long)h_out[0], (unsigned long long)h_out[1]);
            for (int i = 0; i < 8; i++)
                printf("  h_out[%d]=%016llx\n", i, (unsigned long long)h_out[i]);
        }
    }

    /* ── KA6: Check Hadamard output before INTT ──────────────────────────────── */
    /* Upload NTT(e_0) and NTT(e_1) directly, apply Hadamard, check result.     */
    {
        total++;
        printf("\nKA6: Hadamard([1,1,...],[1,-1,1,-1,...]) -> [1,-1,1,-1,...]\n");
        /* f = [1,1,...,1] */
        for (int i = 0; i < N; i++) h_in[i] = 1;
        check_hip(hipMemcpy(d_a, h_in, N * sizeof(uint64_t), hipMemcpyHostToDevice), "upload d_a KA6");
        /* g = [1,p-1,1,p-1,...] */
        for (int i = 0; i < N; i++) h_in[i] = (i % 2 == 0) ? 1ULL : (p - 1);
        check_hip(hipMemcpy(d_b, h_in, N * sizeof(uint64_t), hipMemcpyHostToDevice), "upload d_b KA6");
        launch_hadamard<0>(d_a, d_b, N, stream);
        check_hip(hipStreamSynchronize(stream), "sync KA6");
        check_hip(hipGetLastError(), "kernel error KA6");
        check_hip(hipMemcpy(h_out, d_a, N * sizeof(uint64_t), hipMemcpyDeviceToHost), "download KA6");
        int ok = 1;
        for (int i = 0; i < N; i++) {
            uint64_t expected = (i % 2 == 0) ? 1ULL : (p - 1);
            if (h_out[i] != expected) { ok = 0; break; }
        }
        if (ok) {
            printf("  PASS: Hadamard result is [1,-1,1,-1,...]\n");
            passed++;
        } else {
            printf("  FAIL:\n");
            for (int i = 0; i < 8; i++) {
                uint64_t expected = (i % 2 == 0) ? 1ULL : (p - 1);
                printf("  h_out[%d]=%016llx expected=%016llx %s\n",
                       i, (unsigned long long)h_out[i], (unsigned long long)expected,
                       h_out[i] == expected ? "OK" : "FAIL");
            }
        }
    }

    printf("\nResult: %d/%d passed\n", passed, total);

    /* Cleanup (return values intentionally ignored on teardown) */
    (void)hipFree(d_a); (void)hipFree(d_b);
    (void)hipFree(d_tw); (void)hipFree(d_tw_inv);
    (void)hipStreamDestroy(stream);
    free(h_in); free(h_out);
    return (passed == total) ? 0 : 1;
}
