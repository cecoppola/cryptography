/*
 * ntt_cross_verify.hip — CPU vs GPU cross-verification for the NTT kernels
 *
 * Purpose:   Run the same NTT inputs through both the CPU reference algorithm
 *            and the GPU kernel, then compare outputs element-wise. This is
 *            the Phase 3 exit-criterion test; run it on the 6900XT.
 * Tests:     (1) Known-value: delta input → all-ones output
 *            (2) Known-value: all-ones input → [n, 0, 0, ...] output
 *            (3) Round-trip:  INTT(NTT(a)) == a for random-ish input
 *            (4) Sweep:       parametric n and q from command line
 * Output:    formatted pass/fail table; exits 0 on full pass, 1 on any fail.
 * Build:     hipcc -O2 -Wall -Wextra --offload-arch=gfx1030 \
 *                  -o ntt_cross_verify ntt_cross_verify.hip
 * Run:       ./ntt_cross_verify [n] [q] [omega]
 *            defaults: n=256, q=3329, omega=17  (ML-KEM / FIPS 203)
 * Ref:       ntt_cpu.c (CPU algorithm); ntt_gpu.hip (GPU kernels).
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* ── ANSI ────────────────────────────────────────────────────────────────── */
#define WHT "\033[1;37m"
#define CYN "\033[1;36m"
#define GRN "\033[1;32m"
#define YLW "\033[1;33m"
#define RED "\033[1;31m"
#define RST "\033[0m"

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t _e = (call);                                             \
        if (_e != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s:%d — %s\n",                      \
                    __FILE__, __LINE__, hipGetErrorString(_e));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* ── Shared parameter struct ─────────────────────────────────────────────── */
typedef struct {
    uint64_t n;
    uint64_t q;
    uint64_t omega;
    uint64_t omega_inv;
    uint64_t n_inv;
    uint32_t log2_n;
} ntt_params_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * CPU REFERENCE (static; prefixed cpu_ to avoid linker conflicts)
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint64_t cpu_mod_pow(uint64_t base, uint64_t exp, uint64_t m)
{
    uint64_t r = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) r = (uint64_t)((__uint128_t)r * base % m);
        base = (uint64_t)((__uint128_t)base * base % m);
        exp >>= 1;
    }
    return r;
}

static void cpu_bit_reverse(uint64_t *a, uint64_t n, uint32_t log2_n)
{
    for (uint64_t i = 0; i < n; i++) {
        uint64_t rev = 0, x = i;
        for (uint32_t b = 0; b < log2_n; b++) { rev = (rev << 1) | (x & 1); x >>= 1; }
        if (i < rev) { uint64_t t = a[i]; a[i] = a[rev]; a[rev] = t; }
    }
}

/* cpu_ntt: in-place CT-DIT NTT using twiddle table tw[k] = root^k mod q.
 * Lazy reduction: additions deferred; multiplications via __uint128_t.
 * Final pass reduces all values to [0, q). */
static void cpu_ntt(uint64_t *a, const uint64_t *tw, const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    cpu_bit_reverse(a, n, p->log2_n);
    for (uint64_t len = 2; len <= n; len <<= 1) {
        uint64_t half = len >> 1, stride = n / len;
        for (uint64_t k = 0; k < n; k += len)
            for (uint64_t j = 0; j < half; j++) {
                uint64_t u = a[k+j];
                uint64_t t = (uint64_t)((__uint128_t)tw[j*stride] * a[k+j+half] % q);
                a[k+j]        = u + t;
                a[k+j+half]   = u + q - t;
            }
    }
    for (uint64_t i = 0; i < n; i++) a[i] %= q;
}

static void cpu_intt(uint64_t *a, const uint64_t *twi, const ntt_params_t *p)
{
    uint64_t n = p->n, q = p->q;
    cpu_bit_reverse(a, n, p->log2_n);
    for (uint64_t len = 2; len <= n; len <<= 1) {
        uint64_t half = len >> 1, stride = n / len;
        for (uint64_t k = 0; k < n; k += len)
            for (uint64_t j = 0; j < half; j++) {
                uint64_t u = a[k+j];
                uint64_t t = (uint64_t)((__uint128_t)twi[j*stride] * a[k+j+half] % q);
                a[k+j]      = u + t;
                a[k+j+half] = u + q - t;
            }
    }
    for (uint64_t i = 0; i < n; i++)
        a[i] = (uint64_t)((__uint128_t)(a[i] % q) * p->n_inv % q);
}

static uint64_t *cpu_alloc_twiddles(const ntt_params_t *p, uint64_t root)
{
    uint64_t half = p->n >> 1;
    uint64_t *tw = (uint64_t *)malloc(half * sizeof *tw);
    if (!tw) return NULL;
    tw[0] = 1;
    for (uint64_t k = 1; k < half; k++)
        tw[k] = (uint64_t)((__uint128_t)tw[k-1] * root % p->q);
    return tw;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GPU KERNELS (same as ntt_gpu.hip)
 * ═══════════════════════════════════════════════════════════════════════════ */

#define GPU_MAX_BLOCK 256
#define GPU_MIN_BLOCKS 1

__global__ __launch_bounds__(GPU_MAX_BLOCK, GPU_MIN_BLOCKS)
void cv_bitrev_kernel(uint64_t * __restrict__ a, uint64_t n, uint32_t log2_n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t rev = 0, x = i;
    for (uint32_t b = 0; b < log2_n; b++) { rev = (rev << 1) | (x & 1); x >>= 1; }
    if (i < rev) { uint64_t tmp = a[i]; a[i] = a[rev]; a[rev] = tmp; }
}

__global__ __launch_bounds__(GPU_MAX_BLOCK, GPU_MIN_BLOCKS)
void cv_stage_kernel(uint64_t * __restrict__ a,
                     const uint64_t * __restrict__ tw,
                     uint64_t n, uint64_t q, uint64_t half, uint64_t stride)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n >> 1) return;
    uint64_t len = half << 1;
    uint64_t k = (tid / half) * len, j = tid % half;
    uint64_t u = a[k+j], v = a[k+j+half];
    uint64_t t = (tw[j*stride] * (v % q)) % q;
    uint64_t s = u + t, d = u + q - t;
    a[k+j]       = s >= q ? s - q : s;
    a[k+j+half]  = d >= q ? d - q : d;
}

__global__ __launch_bounds__(GPU_MAX_BLOCK, GPU_MIN_BLOCKS)
void cv_scale_kernel(uint64_t * __restrict__ a, uint64_t n, uint64_t sc, uint64_t q)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = (a[i] * sc) % q;
}

/* gpu_ntt: run the GPU NTT on a host array; returns result in a_host. */
static void gpu_ntt(uint64_t *a_host, const uint64_t *tw_host,
                    const ntt_params_t *p, uint64_t scale, int bsz)
{
    uint64_t n = p->n, q = p->q, hn = n >> 1;
    uint64_t *da, *dtw;
    HIP_CHECK(hipMalloc((void **)&da,  n  * sizeof *da));
    HIP_CHECK(hipMalloc((void **)&dtw, hn * sizeof *dtw));
    HIP_CHECK(hipMemcpy(da,  a_host,  n  * sizeof *da,  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dtw, tw_host, hn * sizeof *dtw, hipMemcpyHostToDevice));

    int gn = (int)((n  + (uint64_t)bsz - 1) / (uint64_t)bsz);
    int gh = (int)((hn + (uint64_t)bsz - 1) / (uint64_t)bsz);

    hipLaunchKernelGGL(cv_bitrev_kernel, gn, bsz, 0, 0, da, n, p->log2_n);
    HIP_CHECK(hipGetLastError());
    for (uint64_t len = 2; len <= n; len <<= 1) {
        uint64_t half = len >> 1, stride = n / len;
        hipLaunchKernelGGL(cv_stage_kernel, gh, bsz, 0, 0, da, dtw, n, q, half, stride);
        HIP_CHECK(hipGetLastError());
    }
    if (scale > 1) {
        hipLaunchKernelGGL(cv_scale_kernel, gn, bsz, 0, 0, da, n, scale, q);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(a_host, da, n * sizeof *da, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(da));
    HIP_CHECK(hipFree(dtw));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PARAMETER INIT
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * root_is_primitive: omega must be a primitive n-th root of unity mod q
 * (omega^n == 1 and omega^(n/2) != 1). Without this every test here is
 * vacuous: delta->all-ones holds for ANY twiddles, and round-trip /
 * CPU-vs-GPU agree for ANY invertible omega (incl. omega=1) because the
 * CPU "reference" and GPU kernel share the same twiddle table and the
 * same butterfly structure — a wrong/non-primitive root passes silently.
 */
static int root_is_primitive(uint64_t n, uint64_t q, uint64_t omega)
{
    if (omega == 0) return 0;
    if (cpu_mod_pow(omega, n, q) != 1) return 0;
    if (cpu_mod_pow(omega, n >> 1, q) == 1) return 0;
    return 1;
}

static int params_init(ntt_params_t *p)
{
    if (!p->n || (p->n & (p->n-1))) return -1;
    if (p->q < 2) {
        fprintf(stderr, "error: q must be a prime >= 2\n");
        return -1;
    }
    if (p->q > 0xFFFFFFFFULL) {
        fprintf(stderr, "error: GPU implementation currently requires q < 2^32\n");
        return -1;
    }
    if (!root_is_primitive(p->n, p->q, p->omega)) {
        fprintf(stderr, "error: omega=%lu is not a primitive %lu-th root of "
                "unity mod q=%lu\n", p->omega, p->n, p->q);
        return -1;
    }
    p->omega_inv = cpu_mod_pow(p->omega, p->q - 2, p->q);
    p->n_inv     = cpu_mod_pow(p->n,     p->q - 2, p->q);
    uint64_t t   = p->n; p->log2_n = 0;
    while (t > 1) { t >>= 1; p->log2_n++; }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST FRAMEWORK
 * ═══════════════════════════════════════════════════════════════════════════ */

static int total_pass = 0, total_fail = 0;

static void print_result(const char *name, int pass, const char *detail)
{
    const char *tag  = pass ? GRN "PASS" RST : RED "FAIL" RST;
    printf("  │ %-36s │ %s │ %-26s │\n", name, tag, detail);
}

/*
 * test_known_delta: NTT([1, 0, 0, ..., 0]) must equal [1, 1, ..., 1] mod q.
 * Proof: NTT(δ)[k] = Σ δ[j] · ω^{jk} = ω^0 = 1 for all k.
 */
static void test_known_delta(const ntt_params_t *p,
                              const uint64_t *tw, int bsz)
{
    uint64_t n = p->n;
    uint64_t *cpu = (uint64_t *)calloc(n, sizeof *cpu);
    uint64_t *gpu = (uint64_t *)calloc(n, sizeof *gpu);
    if (!cpu || !gpu) { free(cpu); free(gpu); return; }
    cpu[0] = gpu[0] = 1;

    cpu_ntt(cpu, tw, p);
    gpu_ntt(gpu, tw, p, 0, bsz);

    int ok_cpu = 1, ok_gpu = 1, ok_match = 1;
    for (uint64_t i = 0; i < n; i++) {
        if (cpu[i] != 1) ok_cpu = 0;
        if (gpu[i] != 1) ok_gpu = 0;
        if (cpu[i] != gpu[i]) ok_match = 0;
    }

    char det[64];
    snprintf(det, sizeof det, "cpu=%s gpu=%s match=%s",
             ok_cpu?"ok":"FAIL", ok_gpu?"ok":"FAIL", ok_match?"ok":"FAIL");
    int pass = ok_cpu && ok_gpu && ok_match;
    print_result("delta → all-ones (known value)", pass, det);
    pass ? total_pass++ : total_fail++;

    free(cpu); free(gpu);
}

/*
 * test_known_allones: NTT([1,1,...,1]) must equal [n, 0, 0, ..., 0] mod q.
 * Proof: for k=0, sum = n. For k>0, geometric series with ω^k ≠ 1 sums to 0.
 */
static void test_known_allones(const ntt_params_t *p,
                                const uint64_t *tw, int bsz)
{
    uint64_t n = p->n, q = p->q;
    uint64_t *cpu = (uint64_t *)malloc(n * sizeof *cpu);
    uint64_t *gpu = (uint64_t *)malloc(n * sizeof *gpu);
    if (!cpu || !gpu) { free(cpu); free(gpu); return; }
    for (uint64_t i = 0; i < n; i++) cpu[i] = gpu[i] = 1;

    cpu_ntt(cpu, tw, p);
    gpu_ntt(gpu, tw, p, 0, bsz);

    uint64_t expected0 = n % q;
    int ok_cpu = (cpu[0] == expected0), ok_gpu = (gpu[0] == expected0), ok_match = 1;
    for (uint64_t i = 1; i < n; i++) {
        if (cpu[i] != 0) ok_cpu = 0;
        if (gpu[i] != 0) ok_gpu = 0;
        if (cpu[i] != gpu[i]) ok_match = 0;
    }

    char det[64];
    snprintf(det, sizeof det, "cpu=%s gpu=%s match=%s",
             ok_cpu?"ok":"FAIL", ok_gpu?"ok":"FAIL", ok_match?"ok":"FAIL");
    int pass = ok_cpu && ok_gpu && ok_match;
    print_result("all-ones → [n,0,...] (known value)", pass, det);
    pass ? total_pass++ : total_fail++;

    free(cpu); free(gpu);
}

/*
 * test_round_trip_cpu: INTT(NTT(a)) == a on CPU.
 */
static void test_round_trip_cpu(const ntt_params_t *p,
                                 const uint64_t *tw, const uint64_t *twi)
{
    uint64_t n = p->n;
    uint64_t *a = (uint64_t *)malloc(n * sizeof *a);
    uint64_t *orig = (uint64_t *)malloc(n * sizeof *orig);
    if (!a || !orig) { free(a); free(orig); return; }
    for (uint64_t i = 0; i < n; i++) orig[i] = a[i] = (i * 1234567891ULL + 42) % p->q;

    cpu_ntt(a, tw, p);
    cpu_intt(a, twi, p);

    int pass = 1;
    for (uint64_t i = 0; i < n; i++) if (a[i] != orig[i]) { pass = 0; break; }
    print_result("CPU round-trip INTT(NTT(a))==a", pass, pass ? "all elements match" : "MISMATCH");
    pass ? total_pass++ : total_fail++;
    free(a); free(orig);
}

/*
 * test_round_trip_gpu: INTT(NTT(a)) == a on GPU.
 */
static void test_round_trip_gpu(const ntt_params_t *p,
                                 const uint64_t *tw, const uint64_t *twi, int bsz)
{
    uint64_t n = p->n;
    uint64_t *a = (uint64_t *)malloc(n * sizeof *a);
    uint64_t *orig = (uint64_t *)malloc(n * sizeof *orig);
    if (!a || !orig) { free(a); free(orig); return; }
    for (uint64_t i = 0; i < n; i++) orig[i] = a[i] = (i * 1234567891ULL + 42) % p->q;

    gpu_ntt(a, tw,  p, 0,       bsz);
    gpu_ntt(a, twi, p, p->n_inv, bsz);

    int pass = 1;
    for (uint64_t i = 0; i < n; i++) if (a[i] != orig[i]) { pass = 0; break; }
    print_result("GPU round-trip INTT(NTT(a))==a", pass, pass ? "all elements match" : "MISMATCH");
    pass ? total_pass++ : total_fail++;
    free(a); free(orig);
}

/*
 * test_cross_validate: CPU NTT output == GPU NTT output for 3 distinct inputs.
 */
static void test_cross_validate(const ntt_params_t *p,
                                 const uint64_t *tw, int bsz,
                                 const char *label,
                                 uint64_t seed)
{
    uint64_t n = p->n;
    uint64_t *cpu = (uint64_t *)malloc(n * sizeof *cpu);
    uint64_t *gpu = (uint64_t *)malloc(n * sizeof *gpu);
    if (!cpu || !gpu) { free(cpu); free(gpu); return; }
    for (uint64_t i = 0; i < n; i++)
        cpu[i] = gpu[i] = (i * seed + 7919ULL) % p->q;

    cpu_ntt(cpu, tw, p);
    gpu_ntt(gpu, tw, p, 0, bsz);

    uint64_t mismatches = 0, first_bad = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (cpu[i] != gpu[i]) {
            if (!mismatches) first_bad = i;
            mismatches++;
        }
    }

    char det[64];
    if (mismatches)
        snprintf(det, sizeof det, "%lu mismatches (first@%lu)", mismatches, first_bad);
    else
        snprintf(det, sizeof det, "all %lu elements match", n);

    int pass = (mismatches == 0);
    print_result(label, pass, det);
    pass ? total_pass++ : total_fail++;

    free(cpu); free(gpu);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    ntt_params_t p;
    p.n     = (argc > 1) ? (uint64_t)atol(argv[1]) : 256;
    p.q     = (argc > 2) ? (uint64_t)atol(argv[2]) : 3329;
    p.omega = (argc > 3) ? (uint64_t)atol(argv[3]) : 17;

    printf("\n" WHT
           "╔══════════════════════════════════════════════════════════╗\n"
           "║        NTT / MI300A  —  CPU vs GPU Cross-Verify         ║\n"
           "╚══════════════════════════════════════════════════════════╝\n"
           RST "\n");

    if (params_init(&p) != 0) {
        /* params_init prints the specific reason; n-not-pow2 is the only
         * silent path (no stderr inside), so report it here. */
        if (!p.n || (p.n & (p.n - 1)))
            fprintf(stderr, "error: n=%lu is not a power of 2\n", p.n);
        return 1;
    }

    /* Query GPU and derive launch block size from runtime warpSize.
     * RDNA2 (6900XT) has warpSize=32; CDNA3 (MI300A) has warpSize=64.
     * 128 = 4 wavefronts on RDNA2 / 2 wavefronts on CDNA3 — both saturate
     * the SIMD dispatcher at modest occupancy. Cap at GPU_MAX_BLOCK so the
     * __launch_bounds__ contract on every kernel still holds. */
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int bsz = 4 * prop.warpSize;          /* RDNA2: 128, CDNA3: 256 */
    if (bsz > GPU_MAX_BLOCK) bsz = GPU_MAX_BLOCK;

    printf(CYN "  Device: " RST "%s  warp=%d  CUs=%d\n",
           prop.name, prop.warpSize, prop.multiProcessorCount);
    printf(CYN "  Params: " RST "n=%lu  q=%lu  omega=%lu\n\n",
           p.n, p.q, p.omega);

    uint64_t *tw  = cpu_alloc_twiddles(&p, p.omega);
    uint64_t *twi = cpu_alloc_twiddles(&p, p.omega_inv);
    if (!tw || !twi) { fprintf(stderr, "alloc failed\n"); return 1; }

    printf("  ┌──────────────────────────────────────┬──────┬────────────────────────────┐\n");
    printf("  │ " CYN "%-36s" RST " │ " CYN "%-4s" RST " │ " CYN "%-26s" RST " │\n",
           "Test", "Res.", "Detail");
    printf("  ├──────────────────────────────────────┼──────┼────────────────────────────┤\n");

    test_known_delta(&p, tw, bsz);
    test_known_allones(&p, tw, bsz);
    test_round_trip_cpu(&p, tw, twi);
    test_round_trip_gpu(&p, tw, twi, bsz);
    test_cross_validate(&p, tw, bsz, "cross-validate: seed 0x1 (arithmetic)", 1ULL);
    test_cross_validate(&p, tw, bsz, "cross-validate: seed 0xDEAD (random)",  0xDEADULL);
    test_cross_validate(&p, tw, bsz, "cross-validate: seed 0xCAFE (random)",  0xCAFEULL);

    printf("  └──────────────────────────────────────┴──────┴────────────────────────────┘\n\n");
    printf("  Results: " GRN "%d passed" RST "  " RED "%d failed" RST "\n\n",
           total_pass, total_fail);

    free(tw); free(twi);
    return total_fail ? 1 : 0;
}
