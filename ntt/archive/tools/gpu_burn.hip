/* gpu_burn.hip — minimal sustained-compute stress test (platform control).
 *
 * Purpose: isolate "can ROCm/gfx1030 sustain ANY compute past ~60s" from our
 * NTT code. A trivial compute-bound FMA-loop kernel, SINGLE default stream, one
 * buffer, no CRT/multi-prime/multi-stream/cwsr-heavy dispatch — the simplest
 * possible sustained GPU load. If this hard-locks the box at minute scale, the
 * cause is the platform (driver/firmware/config), not our NTT.
 *
 * Args: argv[1]=target seconds (default 90), argv[2]=inner FMA iters (default 30000).
 * Writes progress to $CRASH_PROGRESS for the crash-hunt heartbeat.
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void burn_kernel(float* __restrict__ x, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = x[i];
    float b = 1.0001f;
    for (int k = 0; k < iters; k++) {     /* compute-bound: keeps ALUs busy */
        a = a * b + 0.5f;
        b = b * 1.00001f - a * 1e-7f;
    }
    x[i] = a;
}

#define HC(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    return 2; } } while (0)

int main(int argc, char** argv)
{
    double secs  = (argc > 1) ? atof(argv[1]) : 90.0;
    int    kiter = (argc > 2) ? atoi(argv[2]) : 30000;
    int    N     = 1 << 22;                 /* 4,194,304 threads */
    int    block = 256;
    int    grid  = N / block;

    const char* prog = getenv("CRASH_PROGRESS");
    if (!prog) prog = "/tmp/crash_progress";

    printf("gpu_burn: target=%.0fs inner_iters=%d threads=%d (single stream, compute-bound)\n",
           secs, kiter, N);

    float* d = NULL;
    HC(hipMalloc(&d, (size_t)N * sizeof(float)));
    HC(hipMemset(d, 0, (size_t)N * sizeof(float)));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long it = 0;
    double el = 0.0;
    while (el < secs) {
        it++;
        FILE* pf = fopen(prog, "w");
        if (pf) { fprintf(pf, "burn iter %ld t=%.0fs", it, el); fclose(pf); }
        hipLaunchKernelGGL(burn_kernel, dim3(grid), dim3(block), 0, 0, d, kiter);
        HC(hipDeviceSynchronize());
        clock_gettime(CLOCK_MONOTONIC, &t1);
        el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("  burn iter=%ld elapsed=%.1fs\n", it, el);
        fflush(stdout);
    }
    HC(hipDeviceSynchronize());
    hipFree(d);
    printf("gpu_burn DONE: %ld kernels, %.1fs sustained, no crash\n", it, el);
    return 0;
}
