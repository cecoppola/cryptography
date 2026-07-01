/* gpu_burn_xfer.hip — sustained H2D/D2H PCIe-transfer + compute stress.
 *
 * The compute, concurrency, launch-volume, and strided-HBM burns all ran 90s
 * clean. The remaining structural difference vs our NTT is per-multiply HOST<->
 * DEVICE transfers over PCIe (our ntt_bigint_mul uploads both operands and
 * downloads the result every mul — ~64MB of PCIe DMA per mul). This replicates
 * that: per round, H2D upload + strided kernel + D2H download on N streams,
 * sustained. The silent hard-lock signature (below the OS, no log) is consistent
 * with a PCIe/fabric hang, so this is the signature-matching hypothesis.
 *
 * Args: argv[1]=seconds(90) argv[2]=log_n(21) argv[3]=nstreams(4)
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void bfly_kernel(unsigned long long* __restrict__ data, int half, int n_half)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_half) return;
    int pos = i & (half - 1), u = (i - pos) * 2 + pos, v = u + half;
    unsigned long long a = data[u], b = data[v];
    data[u] = a + b; data[v] = a - b + 0x9e3779b97f4a7c15ULL;
}

#define HC(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    return 2; } } while (0)

int main(int argc, char** argv)
{
    double secs = (argc > 1) ? atof(argv[1]) : 90.0;
    int    logn = (argc > 2) ? atoi(argv[2]) : 21;
    int    nstr = (argc > 3) ? atoi(argv[3]) : 4;
    int    N = 1 << logn, n_half = N / 2, block = 256, grid = (n_half + block - 1) / block;
    size_t bytes = (size_t)N * sizeof(unsigned long long);

    const char* prog = getenv("CRASH_PROGRESS"); if (!prog) prog = "/tmp/crash_progress";
    printf("gpu_burn_xfer: target=%.0fs log_n=%d nstreams=%d xfer=%.0fMB/mul (H2D+D2H+compute)\n",
           secs, logn, nstr, (double)bytes * 3 / 1e6);

    hipStream_t* s = (hipStream_t*)malloc(nstr * sizeof(hipStream_t));
    unsigned long long **d = (unsigned long long**)malloc(nstr * sizeof(void*));
    unsigned long long **hin = (unsigned long long**)malloc(nstr * sizeof(void*));
    unsigned long long **hout = (unsigned long long**)malloc(nstr * sizeof(void*));
    for (int j = 0; j < nstr; j++) {
        HC(hipStreamCreate(&s[j]));
        HC(hipMalloc(&d[j], bytes));
        HC(hipHostMalloc(&hin[j], bytes));   /* pinned host bufs, like our staging */
        HC(hipHostMalloc(&hout[j], bytes));
        for (int k = 0; k < N; k++) hin[j][k] = 0x1111111111111111ULL + k;
    }

    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    long muls = 0; double el = 0.0;
    while (el < secs) {
        muls++;
        FILE* pf = fopen(prog, "w");
        if (pf) { fprintf(pf, "burn_xfer mul %ld t=%.0fs", muls, el); fclose(pf); }
        for (int j = 0; j < nstr; j++) {
            HC(hipMemcpyAsync(d[j], hin[j], bytes, hipMemcpyHostToDevice, s[j]));  /* H2D */
            for (int stage = 0; stage < logn; stage++)
                hipLaunchKernelGGL(bfly_kernel, dim3(grid), dim3(block), 0, s[j], d[j], 1 << stage, n_half);
            HC(hipMemcpyAsync(hout[j], d[j], bytes, hipMemcpyDeviceToHost, s[j]));  /* D2H */
        }
        for (int j = 0; j < nstr; j++) HC(hipStreamSynchronize(s[j]));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        if (muls % 50 == 0) { printf("  mul=%ld elapsed=%.1fs\n", muls, el); fflush(stdout); }
    }
    printf("gpu_burn_xfer DONE: %ld muls, %.1fs, no crash\n", muls, el);
    return 0;
}
