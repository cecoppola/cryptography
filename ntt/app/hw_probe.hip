/*
 * hw_probe.hip — Hardware diagnostic for CRT-NTT deployment.
 *   (DEPLOYMENT_DIAGNOSTIC.md Appendix C; output transcribes into Appendix A.)
 *
 * Read-only hardware inventory + unified-memory correctness micro-tests.
 * Self-contained (no engine headers); links nothing. Builds for gfx942 (target)
 * and gfx1030 (dev baseline) so the two can be diffed.
 *
 * Build:  make hw_probe        # gfx942 — run on the MI300A
 *         make hw_probe-dev    # gfx1030 — runs on the dev box
 *   or:   hipcc -O2 -Wall -Wextra --offload-arch=gfx942 -o hw_probe app/hw_probe.hip
 * Run:    srun -p mi -N 1 -n 1 ./hw_probe       (MI300A)
 *         ./bin/hw_probe_dev                     (dev box; kernels are µs-scale, safe)
 *
 * SAFETY: the managed GPU-access test is gated on concurrentManagedAccess, so on a
 * non-XNACK GPU (e.g. gfx1030, where managed memory is system RAM and a GPU read
 * page-faults) it is SKIPPED rather than attempted — the probe never faults the box.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXDEV 8

#define HIP_CHECK(expr) do {                                                  \
    hipError_t _e = (expr);                                                   \
    if (_e != hipSuccess) {                                                   \
        fprintf(stderr, "hw_probe: HIP error %d (%s) at %s:%d: %s\n",         \
                (int)_e, hipGetErrorString(_e), __FILE__, __LINE__, #expr);   \
        exit(1);                                                              \
    }                                                                         \
} while (0)

/* Best-effort: returns 1 on success, 0 on failure (for capability probes that
 * are allowed to fail without aborting the whole run). */
#define HIP_TRY(expr) ((expr) == hipSuccess)

/* The subset of device facts the engine would select block/LDS sizing from.
 * Plain C struct so ntt_mul_init / main.hip could later link probe_device() and
 * replace the hardcoded BLOCK_SIZE / TT_DIM / LDS constants (the c-style
 * "query hardware at startup" rule the engine currently does not follow). */
typedef struct {
    int    valid;
    char   name[256];
    char   arch[256];
    int    warpSize;
    int    cuCount;             /* multiProcessorCount                 */
    int    maxThreadsPerBlock;
    int    maxThreadsPerCU;     /* maxThreadsPerMultiProcessor         */
    int    regsPerBlock;
    size_t sharedPerBlock;
    size_t sharedPerCU;         /* maxSharedMemoryPerMultiProcessor    */
    size_t totalGlobalMem;
    int    l2CacheSize;
    int    clockKHz;
    int    memClockKHz;
    int    memBusBits;
    int    integrated;          /* APU?                                */
    int    managedMemory;
    int    concurrentManagedAccess;
    int    pageableMemoryAccess;
    int    directManagedHost;   /* directManagedMemAccessFromHost      */
} DeviceFacts;

/* probe_device — fill DeviceFacts for one device. Returns 0 on success. */
static int probe_device(int dev, DeviceFacts* f)
{
    hipDeviceProp_t p;
    memset(f, 0, sizeof *f);
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) return -1;
    f->valid              = 1;
    snprintf(f->name, sizeof f->name, "%s", p.name);
    snprintf(f->arch, sizeof f->arch, "%s", p.gcnArchName);
    f->warpSize           = p.warpSize;
    f->cuCount            = p.multiProcessorCount;
    f->maxThreadsPerBlock = p.maxThreadsPerBlock;
    f->maxThreadsPerCU    = p.maxThreadsPerMultiProcessor;
    f->regsPerBlock       = p.regsPerBlock;
    f->sharedPerBlock     = p.sharedMemPerBlock;
    f->sharedPerCU        = p.maxSharedMemoryPerMultiProcessor;
    f->totalGlobalMem     = p.totalGlobalMem;
    f->l2CacheSize        = p.l2CacheSize;
    f->clockKHz           = p.clockRate;
    f->memClockKHz        = p.memoryClockRate;
    f->memBusBits         = p.memoryBusWidth;
    f->integrated         = p.integrated;
    f->managedMemory      = p.managedMemory;
    f->concurrentManagedAccess = p.concurrentManagedAccess;
    f->pageableMemoryAccess    = p.pageableMemoryAccess;
    f->directManagedHost  = p.directManagedMemAccessFromHost;
    return 0;
}

/* ─── tiny test kernels (µs-scale; not a benchmark) ─────────────────────────── */
__global__ static void k_inc(int* p, int n, int add)
{ int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] += add; }
__global__ static void k_fill(int* p, int n, int mul)
{ int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = i * mul; }

/* ─── property table (one column per device) ────────────────────────────────── */
static void print_table(const DeviceFacts* f, int n)
{
    printf("\nProperty table (one column per device):\n");
    printf("  %-34s", "property");
    for (int i = 0; i < n; i++) { char h[16]; snprintf(h, sizeof h, "dev%d", i); printf(" %13s", h); }
    printf("\n");
    #define ROW_D(label, field)  do { printf("  %-34s", label); for (int i = 0; i < n; i++) printf(" %13d",  f[i].field); printf("\n"); } while (0)
    #define ROW_ZU(label, field) do { printf("  %-34s", label); for (int i = 0; i < n; i++) printf(" %13zu", f[i].field); printf("\n"); } while (0)
    #define ROW_YN(label, field) do { printf("  %-34s", label); for (int i = 0; i < n; i++) printf(" %13s",  f[i].field ? "yes" : "no"); printf("\n"); } while (0)
    ROW_D ("warpSize",                       warpSize);
    ROW_D ("CUs (multiProcessorCount)",      cuCount);
    ROW_D ("maxThreadsPerBlock",             maxThreadsPerBlock);
    ROW_D ("maxThreadsPerCU",                maxThreadsPerCU);
    ROW_D ("regsPerBlock",                   regsPerBlock);
    ROW_ZU("sharedMemPerBlock (B)",          sharedPerBlock);
    ROW_ZU("sharedMemPerCU (B)",             sharedPerCU);
    ROW_ZU("totalGlobalMem (B)",             totalGlobalMem);
    ROW_D ("l2CacheSize (B)",                l2CacheSize);
    ROW_D ("clockRate (kHz)",                clockKHz);
    ROW_D ("memClockRate (kHz)",             memClockKHz);
    ROW_D ("memBusWidth (bits)",             memBusBits);
    ROW_YN("integrated (APU)",               integrated);
    ROW_YN("managedMemory",                  managedMemory);
    ROW_YN("concurrentManagedAccess",        concurrentManagedAccess);
    ROW_YN("pageableMemoryAccess",           pageableMemoryAccess);
    ROW_YN("directManagedMemAccessFromHost", directManagedHost);
    #undef ROW_D
    #undef ROW_ZU
    #undef ROW_YN
    printf("  %-34s %s\n", "LDS bank count", "(not in hipDeviceProp_t; use rocminfo — expect 32)");
}

/* ─── peer-access matrix (canAccessPeer, then actually enable) ───────────────── */
static void print_peer(int n)
{
    printf("\nPeer-access matrix (canAccessPeer / enable; '-' = self):\n      ");
    for (int j = 0; j < n; j++) printf("  dev%d ", j);
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("  dev%d", i);
        for (int j = 0; j < n; j++) {
            if (i == j) { printf("   -   "); continue; }
            int can = 0;
            (void)hipDeviceCanAccessPeer(&can, i, j);
            const char* tag = "no";
            if (can) {
                HIP_CHECK(hipSetDevice(i));
                hipError_t e = hipDeviceEnablePeerAccess(j, 0);
                tag = (e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled) ? "Y/ok" : "Y/!!";
            }
            printf(" %5s ", tag);
        }
        printf("\n");
    }
    HIP_CHECK(hipSetDevice(0));
}

/* ─── unified-memory micro-tests on device 0 (return # of failures) ─────────── */
/* Validates the !GFX1030_LOCAL managed path (gated on concurrentManagedAccess). */
static int test_managed(int concurrent_ma)
{
    const int N = 4096;
    int rc = 0, *p = NULL;
    if (!HIP_TRY(hipMallocManaged((void**)&p, (size_t)N * sizeof(int)))) {
        printf("  managed alloc:                      FAIL (hipMallocManaged)\n");
        return 1;
    }
    for (int i = 0; i < N; i++) p[i] = i;
    printf("  managed alloc + CPU r/w:            PASS\n");
    if (concurrent_ma) {
        k_inc<<<(N + 255) / 256, 256>>>(p, N, 1000);
        HIP_CHECK(hipDeviceSynchronize());
        int bad = 0;
        for (int i = 0; i < N; i++) if (p[i] != i + 1000) bad++;
        printf("  managed GPU kernel access:          %s\n", bad ? "FAIL" : "PASS");
        rc += (bad != 0);
    } else {
        printf("  managed GPU kernel access:          SKIPPED (concurrentManagedAccess=0 — unsupported)\n");
    }
    (void)hipFree(p);
    return rc;
}

/* Validates the zero-copy hipHostMalloc(NonCoherent) aliasing the Garner output uses. */
static int test_zerocopy(void)
{
    const int N = 4096;
    int rc = 0, *h = NULL, *d = NULL;
    if (!HIP_TRY(hipHostMalloc((void**)&h, (size_t)N * sizeof(int), hipHostMallocNonCoherent))) {
        printf("  hipHostMalloc(NonCoherent):         FAIL\n");
        return 1;
    }
    if (!HIP_TRY(hipHostGetDevicePointer((void**)&d, h, 0))) {
        printf("  hipHostGetDevicePointer:            FAIL\n");
        (void)hipHostFree(h);
        return 1;
    }
    printf("  zero-copy alias (host==device ptr): %s\n",
           (void*)d == (void*)h ? "YES (unified — engine aliases d_out=h_out)" : "no (separate ptrs)");
    k_fill<<<(N + 255) / 256, 256>>>(d, N, 7);
    HIP_CHECK(hipDeviceSynchronize());
    int bad = 0;
    for (int i = 0; i < N; i++) if (h[i] != i * 7) bad++;
    printf("  zero-copy GPU write -> CPU read:    %s\n", bad ? "FAIL" : "PASS");
    rc += (bad != 0);
    (void)hipHostFree(h);
    return rc;
}

int main(void)
{
    printf("=== hw_probe — CRT-NTT hardware diagnostic ===\n");
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 1) {
        fprintf(stderr, "hw_probe: no HIP devices visible\n");
        return 1;
    }
    if (n > MAXDEV) n = MAXDEV;
    printf("hipGetDeviceCount: %d\n", n);

    DeviceFacts f[MAXDEV];
    for (int i = 0; i < n; i++) {
        if (probe_device(i, &f[i]) != 0) {
            fprintf(stderr, "hw_probe: probe_device(%d) failed\n", i);
            return 1;
        }
        printf("Device %d: %s [%s]\n", i, f[i].name, f[i].arch);
    }

    print_table(f, n);
    print_peer(n);

    printf("\nUnified-memory micro-tests (device 0):\n");
    HIP_CHECK(hipSetDevice(0));
    int fails = test_managed(f[0].concurrentManagedAccess) + test_zerocopy();

    printf("\nTranscribe the table above into DEPLOYMENT_DIAGNOSTIC.md Appendix A.\n");
    printf("%s\n", fails ? "=== hw_probe: micro-tests FAILED ===" : "=== hw_probe: OK ===");
    return fails ? 1 : 0;
}
