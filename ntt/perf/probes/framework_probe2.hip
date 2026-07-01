// framework_probe2.hip — extends framework_probe with two curves missing
// from the original probe:
//
//   1. PCIe transfer-size curve (paged + pinned, H2D + D2H)
//      sizes 4 KB → 256 MB
//   2. Streams concurrency curve: 1 / 2 / 4 / 8 / 16 streams, each
//      doing a fixed total amount of work distributed across them
//
// Designed to finish in <30 s. No rocprof, no contention kernels.

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define HIPCK(x) do { hipError_t _hck=(x); if(_hck!=hipSuccess){fprintf(stderr,"HIP err %s @ %s:%d\n",hipGetErrorString(_hck),__FILE__,__LINE__);return 1;}}while(0)

__global__ void k_vecadd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

static double mono_ms() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1e3 + t.tv_nsec*1e-6;
}

int main() {
    HIPCK(hipFree(0));
    printf("=== framework_probe2 (gfx1030 / 6900 XT) ===\n");

    // ---- 1. PCIe transfer-size curve --------------------------------------
    {
        const size_t sizes[] = {
            4ull<<10, 16ull<<10, 64ull<<10, 256ull<<10,
            1ull<<20, 4ull<<20, 16ull<<20, 64ull<<20, 256ull<<20
        };
        const int NS = sizeof(sizes)/sizeof(sizes[0]);
        size_t max_sz = sizes[NS-1];

        char *h_paged  = (char*)malloc(max_sz);
        char *h_pinned;
        char *d_buf;
        HIPCK(hipHostMalloc(&h_pinned, max_sz));
        HIPCK(hipMalloc(&d_buf, max_sz));
        memset(h_paged, 0xAA, max_sz);
        memset(h_pinned, 0xAA, max_sz);

        printf("\nPCIe transfer-size curve (best-of-10, GB/s)\n");
        printf("%-10s %12s %12s %12s %12s\n",
               "Size", "H2D paged", "H2D pinned", "D2H paged", "D2H pinned");
        printf("%-10s %12s %12s %12s %12s\n",
               "----", "---------", "----------", "---------", "----------");

        for (int s=0; s<NS; s++) {
            size_t SZ = sizes[s];
            int reps = (SZ < (1ull<<20)) ? 50 : 10;
            double bw[4]={0,0,0,0};
            for (int dir=0; dir<4; dir++) {
                double best = 1e30;
                for (int r=0; r<reps; r++) {
                    double t0 = mono_ms();
                    if      (dir==0) HIPCK(hipMemcpy(d_buf, h_paged,  SZ, hipMemcpyHostToDevice));
                    else if (dir==1) HIPCK(hipMemcpy(d_buf, h_pinned, SZ, hipMemcpyHostToDevice));
                    else if (dir==2) HIPCK(hipMemcpy(h_paged,  d_buf, SZ, hipMemcpyDeviceToHost));
                    else             HIPCK(hipMemcpy(h_pinned, d_buf, SZ, hipMemcpyDeviceToHost));
                    HIPCK(hipDeviceSynchronize());
                    double dt = mono_ms()-t0;
                    if (dt < best) best = dt;
                }
                bw[dir] = SZ/(best*1e-3)/1e9;
            }
            const char *u; size_t v;
            if (SZ>=1ull<<20) { u="MB"; v=SZ>>20; } else { u="KB"; v=SZ>>10; }
            printf("%-7zu %-2s %12.2f %12.2f %12.2f %12.2f\n", v, u, bw[0], bw[1], bw[2], bw[3]);
        }
        free(h_paged);
        HIPCK(hipHostFree(h_pinned));
        HIPCK(hipFree(d_buf));
    }

    // ---- 2. Streams concurrency curve -------------------------------------
    {
        const int NEL_TOTAL = 64<<20;        // 64 M elements total work
        const int N_STREAMS_MAX = 16;
        size_t SZ_TOTAL = NEL_TOTAL * sizeof(float);
        float *a, *b, *c;
        HIPCK(hipMalloc(&a, SZ_TOTAL));
        HIPCK(hipMalloc(&b, SZ_TOTAL));
        HIPCK(hipMalloc(&c, SZ_TOTAL));
        HIPCK(hipMemset(a, 1, SZ_TOTAL));
        HIPCK(hipMemset(b, 1, SZ_TOTAL));

        hipStream_t st[N_STREAMS_MAX];
        for (int s=0;s<N_STREAMS_MAX;s++) HIPCK(hipStreamCreate(&st[s]));

        printf("\nStreams concurrency curve (total work = %d M elements vec-add)\n", NEL_TOTAL>>20);
        printf("%-10s %12s %12s\n", "Streams", "ms", "speedup vs 1");
        printf("%-10s %12s %12s\n", "-------", "--", "------------");

        const int blk = 256;
        double t1_ms = 0;
        for (int n_str : {1,2,4,8,16}) {
            int per_stream = NEL_TOTAL / n_str;
            int grid = (per_stream + blk - 1) / blk;

            // warmup
            for (int s=0; s<n_str; s++)
                k_vecadd<<<grid,blk,0,st[s]>>>(a + s*per_stream, b + s*per_stream, c + s*per_stream, per_stream);
            for (int s=0; s<n_str; s++) HIPCK(hipStreamSynchronize(st[s]));

            double best = 1e30;
            for (int r=0; r<10; r++) {
                double t0 = mono_ms();
                for (int s=0; s<n_str; s++)
                    k_vecadd<<<grid,blk,0,st[s]>>>(a + s*per_stream, b + s*per_stream, c + s*per_stream, per_stream);
                for (int s=0; s<n_str; s++) HIPCK(hipStreamSynchronize(st[s]));
                double dt = mono_ms()-t0;
                if (dt < best) best = dt;
            }
            if (n_str == 1) t1_ms = best;
            printf("%-10d %12.3f %12.2fx\n", n_str, best, t1_ms/best);
        }
        for (int s=0;s<N_STREAMS_MAX;s++) HIPCK(hipStreamDestroy(st[s]));
        HIPCK(hipFree(a)); HIPCK(hipFree(b)); HIPCK(hipFree(c));
    }

    printf("\n=== framework_probe2 DONE ===\n");
    return 0;
}
