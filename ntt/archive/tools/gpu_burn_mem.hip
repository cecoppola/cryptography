/* gpu_burn_mem.hip — memory-bound strided-HBM sustained stress.
 *
 * The compute-bound burns (single + 4-stream) ran 90s clean, ruling out the
 * platform, concurrency, and launch rate. Our NTT differs in the WORK: large
 * N=2^21 buffers with heavily-strided HBM access (CT-DIT butterfly stages,
 * stride doubling each pass). This replicates that memory pattern WITHOUT the
 * NTT math: per round, run log_n butterfly-strided passes over N=2^21 uint64
 * buffers on N streams. If THIS hard-locks at minute scale (while compute-bound
 * burns did not), the trigger is the large strided HBM access pattern.
 *
 * Args: argv[1]=seconds(90) argv[2]=log_n(21) argv[3]=nstreams(4)
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* butterfly-strided read-modify-write, mirrors ntt_global_stage_kernel access. */
__global__ void bfly_kernel(unsigned long long* __restrict__ data, int half, int n_half)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_half) return;
    int pos = i & (half - 1);
    int u   = (i - pos) * 2 + pos;
    int v   = u + half;
    unsigned long long a = data[u], b = data[v];
    data[u] = a + b;
    data[v] = a - b + 0x9e3779b97f4a7c15ULL;   /* keep it honest, strided RMW */
}

#define HC(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    return 2; } } while (0)

int main(int argc, char** argv)
{
    double secs = (argc > 1) ? atof(argv[1]) : 90.0;
    int    logn = (argc > 2) ? atoi(argv[2]) : 21;
    int    nstr = (argc > 3) ? atoi(argv[3]) : 4;
    int    N      = 1 << logn;
    int    n_half = N / 2;
    int    block  = 256;
    int    grid   = (n_half + block - 1) / block;

    const char* prog = getenv("CRASH_PROGRESS");
    if (!prog) prog = "/tmp/crash_progress";

    printf("gpu_burn_mem: target=%.0fs log_n=%d N=%d nstreams=%d buf=%.0fMB/stream (strided HBM)\n",
           secs, logn, N, nstr, (double)N * 8 / 1e6);

    hipStream_t* s = (hipStream_t*)malloc(nstr * sizeof(hipStream_t));
    unsigned long long** d = (unsigned long long**)malloc(nstr * sizeof(void*));
    for (int j = 0; j < nstr; j++) {
        HC(hipStreamCreate(&s[j]));
        HC(hipMalloc(&d[j], (size_t)N * sizeof(unsigned long long)));   /* 16 MB at log_n=21 */
        HC(hipMemset(d[j], 1, (size_t)N * sizeof(unsigned long long)));
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long muls = 0, launches = 0;
    double el = 0.0;
    while (el < secs) {
        muls++;
        FILE* pf = fopen(prog, "w");
        if (pf) { fprintf(pf, "burn_mem mul %ld launches %ld t=%.0fs", muls, launches, el); fclose(pf); }
        /* one "mul" = log_n butterfly-strided passes per stream (like a forward NTT),
         * issued across nstr streams, then sync — mirrors our per-mul dispatch. */
        for (int j = 0; j < nstr; j++)
            for (int stage = 0; stage < logn; stage++) {
                hipLaunchKernelGGL(bfly_kernel, dim3(grid), dim3(block), 0, s[j],
                                   d[j], 1 << stage, n_half);
                launches++;
            }
        for (int j = 0; j < nstr; j++) HC(hipStreamSynchronize(s[j]));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        if (muls % 100 == 0) { printf("  mul=%ld launches=%ld elapsed=%.1fs\n", muls, launches, el); fflush(stdout); }
    }
    for (int j = 0; j < nstr; j++) { HC(hipStreamSynchronize(s[j])); hipFree(d[j]); hipStreamDestroy(s[j]); }
    printf("gpu_burn_mem DONE: %ld muls, %ld launches, %.1fs, no crash\n", muls, launches, el);
    return 0;
}
