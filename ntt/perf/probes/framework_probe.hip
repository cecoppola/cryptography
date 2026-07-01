// framework_probe.hip — single-binary microbenchmark covering items the
// 14-task A-E campaign did not cover. Runs in <30 s on a 6900 XT and
// produces a tabular line per measurement to stdout.
//
//   1. Empty-kernel launch overhead (1000 launches, mean / stddev us)
//   2. Memory model: hipMalloc / hipMallocManaged / hipHostMalloc
//      bandwidth at 64 MB for H2D, D2H, and host-side write
//   3. Multi-stream concurrency: 1 vs 4 streams, vec-add 16 MB each
//   4. hipEvent vs CLOCK_MONOTONIC: time the same null-kernel both ways,
//      report skew

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define HIPCK(x) do { hipError_t _hck=(x); if(_hck!=hipSuccess){fprintf(stderr,"HIP err %s @ %s:%d\n",hipGetErrorString(_hck),__FILE__,__LINE__);return 1;}}while(0)

__global__ void k_empty() { }
__global__ void k_vecadd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

static double mono_ms() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1e3 + t.tv_nsec*1e-6;
}

int main() {
    HIPCK(hipFree(0));   // force runtime init out of timing
    printf("=== framework_probe (gfx1030 / 6900 XT) ===\n");

    // ---- 1. empty kernel launch overhead ----------------------------------
    {
        const int N = 1000;
        hipEvent_t s,e;
        HIPCK(hipEventCreate(&s)); HIPCK(hipEventCreate(&e));
        // warmup
        for (int i=0;i<32;i++) k_empty<<<1,1>>>();
        HIPCK(hipDeviceSynchronize());
        HIPCK(hipEventRecord(s));
        for (int i=0;i<N;i++) k_empty<<<1,1>>>();
        HIPCK(hipEventRecord(e));
        HIPCK(hipEventSynchronize(e));
        float ms; HIPCK(hipEventElapsedTime(&ms, s, e));
        // also measure single launch with sync via wall clock
        double t0 = mono_ms();
        for (int i=0;i<N;i++) { k_empty<<<1,1>>>(); HIPCK(hipDeviceSynchronize()); }
        double t1 = mono_ms();
        printf("LAUNCH_OVERHEAD async_us=%.2f sync_us=%.2f n=%d\n",
               ms*1e3/N, (t1-t0)*1e3/N, N);
        HIPCK(hipEventDestroy(s)); HIPCK(hipEventDestroy(e));
    }

    // ---- 2. memory model bandwidth (64 MB) --------------------------------
    {
        const size_t SZ = 64ull<<20;
        float *h_pinned, *h_paged = (float*)malloc(SZ);
        float *d_buf;
        float *m_buf;
        HIPCK(hipHostMalloc(&h_pinned, SZ));
        HIPCK(hipMalloc(&d_buf, SZ));
        HIPCK(hipMallocManaged(&m_buf, SZ));
        memset(h_paged, 0xAA, SZ);
        memset(h_pinned, 0xAA, SZ);

        auto h2d = [&](const char* tag, float* host) -> int {
            // 5 reps, take min (best case bandwidth)
            double best=1e30;
            for (int r=0;r<5;r++){
                double t0=mono_ms();
                HIPCK(hipMemcpy(d_buf, host, SZ, hipMemcpyHostToDevice));
                HIPCK(hipDeviceSynchronize());
                double dt=mono_ms()-t0;
                if (dt<best) best=dt;
            }
            printf("MEMBW H2D %-13s %.2f GB/s (%.3f ms)\n", tag, SZ/(best*1e-3)/1e9, best);
            return 0;
        };
        auto d2h = [&](const char* tag, float* host) -> int {
            double best=1e30;
            for (int r=0;r<5;r++){
                double t0=mono_ms();
                HIPCK(hipMemcpy(host, d_buf, SZ, hipMemcpyDeviceToHost));
                HIPCK(hipDeviceSynchronize());
                double dt=mono_ms()-t0;
                if (dt<best) best=dt;
            }
            printf("MEMBW D2H %-13s %.2f GB/s (%.3f ms)\n", tag, SZ/(best*1e-3)/1e9, best);
            return 0;
        };
        h2d("paged", h_paged);
        h2d("pinned", h_pinned);
        d2h("paged", h_paged);
        d2h("pinned", h_pinned);

        // managed: time hipMemcpy from device-mode managed to managed
        // (covers the gfx1030 PCIe migration path)
        {
            HIPCK(hipMemset(m_buf, 0, SZ));   // touch on device
            double best=1e30;
            for (int r=0;r<5;r++){
                double t0=mono_ms();
                memcpy(h_paged, m_buf, SZ);   // CPU-side read — triggers migration
                double dt=mono_ms()-t0;
                if (dt<best) best=dt;
            }
            printf("MEMBW CPU_READ_MANAGED  %.2f GB/s (%.3f ms)\n",
                   SZ/(best*1e-3)/1e9, best);
        }

        free(h_paged);
        HIPCK(hipHostFree(h_pinned));
        HIPCK(hipFree(d_buf));
        HIPCK(hipFree(m_buf));
    }

    // ---- 3. stream concurrency: 1 stream vs 4 streams ---------------------
    {
        const int NEL = 16<<20;
        const size_t SZ = NEL * sizeof(float);
        float *a, *b, *c[4];
        HIPCK(hipMalloc(&a, SZ));
        HIPCK(hipMalloc(&b, SZ));
        for (int s=0;s<4;s++) HIPCK(hipMalloc(&c[s], SZ));
        HIPCK(hipMemset(a, 1, SZ));
        HIPCK(hipMemset(b, 1, SZ));

        hipStream_t st[4];
        for (int s=0;s<4;s++) HIPCK(hipStreamCreate(&st[s]));

        const int blk = 256, grid = (NEL+blk-1)/blk;

        // Warmup
        for (int s=0;s<4;s++) k_vecadd<<<grid,blk,0,st[s]>>>(a,b,c[s],NEL);
        for (int s=0;s<4;s++) HIPCK(hipStreamSynchronize(st[s]));

        // 1-stream baseline (4 sequential)
        double t0 = mono_ms();
        for (int s=0;s<4;s++) {
            k_vecadd<<<grid,blk,0,st[0]>>>(a,b,c[s],NEL);
        }
        HIPCK(hipStreamSynchronize(st[0]));
        double seq_ms = mono_ms()-t0;

        // 4-stream concurrent
        t0 = mono_ms();
        for (int s=0;s<4;s++) k_vecadd<<<grid,blk,0,st[s]>>>(a,b,c[s],NEL);
        for (int s=0;s<4;s++) HIPCK(hipStreamSynchronize(st[s]));
        double par_ms = mono_ms()-t0;

        printf("STREAMS sequential_ms=%.3f parallel4_ms=%.3f speedup=%.2fx\n",
               seq_ms, par_ms, seq_ms/par_ms);

        for (int s=0;s<4;s++){ HIPCK(hipStreamDestroy(st[s])); HIPCK(hipFree(c[s])); }
        HIPCK(hipFree(a)); HIPCK(hipFree(b));
    }

    // ---- 4. hipEvent vs CLOCK_MONOTONIC calibration -----------------------
    {
        const int N = 256;
        hipEvent_t s,e;
        HIPCK(hipEventCreate(&s)); HIPCK(hipEventCreate(&e));
        for (int i=0;i<32;i++){ k_empty<<<1,1>>>(); HIPCK(hipDeviceSynchronize()); }
        double tw0 = mono_ms();
        HIPCK(hipEventRecord(s));
        for (int i=0;i<N;i++){ k_empty<<<1,1>>>(); HIPCK(hipDeviceSynchronize()); }
        HIPCK(hipEventRecord(e));
        HIPCK(hipEventSynchronize(e));
        double tw1 = mono_ms();
        float ev_ms;
        HIPCK(hipEventElapsedTime(&ev_ms, s, e));
        double mono_total = tw1 - tw0;
        printf("CLOCK_CAL hipEvent_ms=%.3f mono_ms=%.3f delta_ms=%.3f delta_pct=%.2f%%\n",
               ev_ms, mono_total, mono_total-ev_ms, (mono_total-ev_ms)/ev_ms*100.0);
        HIPCK(hipEventDestroy(s)); HIPCK(hipEventDestroy(e));
    }

    printf("=== framework_probe DONE ===\n");
    return 0;
}
