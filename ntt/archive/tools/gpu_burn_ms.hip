/* gpu_burn_ms.hip — multi-stream / high-launch-rate sustained-compute stress.
 *
 * Isolates whether our NTT crash is caused by the DISPATCH PATTERN rather than
 * the platform: the simple single-stream gpu_burn ran 90s clean, so this adds
 * the two biggest structural differences vs our NTT —
 *   (1) N concurrent streams (our NTT uses 4, one per CRT prime), and
 *   (2) a ~1000x higher kernel-launch rate (short kernels, synced per round,
 *       like our NTT's ~370 launches/mul at ~10k launches/s).
 * If THIS hard-locks the box at minute scale (while single-stream gpu_burn did
 * not), the trigger is our multi-stream / high-rate dispatch, fixable in our code.
 *
 * Args: argv[1]=seconds(90) argv[2]=inner_iters(200, short=high rate) argv[3]=nstreams(4)
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void burn_kernel(float* __restrict__ x, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = x[i], b = 1.0001f;
    for (int k = 0; k < iters; k++) { a = a * b + 0.5f; b = b * 1.00001f - a * 1e-7f; }
    x[i] = a;
}

#define HC(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    return 2; } } while (0)

int main(int argc, char** argv)
{
    double secs   = (argc > 1) ? atof(argv[1]) : 90.0;
    int    kiter  = (argc > 2) ? atoi(argv[2]) : 200;     /* short -> high launch rate */
    int    nstr   = (argc > 3) ? atoi(argv[3]) : 4;       /* mirror 4 CRT-prime streams */
    int    N      = 1 << 20;                              /* per-stream buffer */
    int    block  = 256, grid = N / block;

    const char* prog = getenv("CRASH_PROGRESS");
    if (!prog) prog = "/tmp/crash_progress";

    printf("gpu_burn_ms: target=%.0fs inner_iters=%d nstreams=%d (multi-stream, high launch rate)\n",
           secs, kiter, nstr);

    hipStream_t* s = (hipStream_t*)malloc(nstr * sizeof(hipStream_t));
    float**      d = (float**)malloc(nstr * sizeof(float*));
    for (int j = 0; j < nstr; j++) {
        HC(hipStreamCreate(&s[j]));
        HC(hipMalloc(&d[j], (size_t)N * sizeof(float)));
        HC(hipMemset(d[j], 0, (size_t)N * sizeof(float)));
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long rounds = 0, launches = 0;
    double el = 0.0;
    while (el < secs) {
        rounds++;
        FILE* pf = fopen(prog, "w");
        if (pf) { fprintf(pf, "burn_ms round %ld launches %ld t=%.0fs", rounds, launches, el); fclose(pf); }
        /* issue to all streams concurrently, then sync all (mirrors NTT's per-mul
         * multi-stream dispatch + yield sync). Many short rounds = high launch rate. */
        for (int r = 0; r < 40; r++) {                   /* 40 * nstr launches per round, unsynced */
            for (int j = 0; j < nstr; j++) {
                hipLaunchKernelGGL(burn_kernel, dim3(grid), dim3(block), 0, s[j], d[j], kiter);
                launches++;
            }
        }
        for (int j = 0; j < nstr; j++) HC(hipStreamSynchronize(s[j]));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        if (rounds % 50 == 0) { printf("  round=%ld launches=%ld elapsed=%.1fs rate=%.0f/s\n",
                                       rounds, launches, el, launches / el); fflush(stdout); }
    }
    for (int j = 0; j < nstr; j++) { HC(hipStreamSynchronize(s[j])); hipFree(d[j]); hipStreamDestroy(s[j]); }
    printf("gpu_burn_ms DONE: %ld rounds, %ld launches, %.1fs, %.0f launches/s, no crash\n",
           rounds, launches, el, launches / el);
    return 0;
}
