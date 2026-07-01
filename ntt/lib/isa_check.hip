/*
 * isa_check.hip — Verify v_lshl_add_u64 appears in reduce_p1 ISA for gfx942.
 *
 * Build (cross-compile from dev machine):
 *   hipcc -O3 --offload-arch=gfx942 --save-temps -o /dev/null isa_check.hip
 *
 * Verify:
 *   grep -c v_lshl_add_u64 isa_check-hip-amdgcn-amd-amdhsa-gfx942.s
 *   # Expected: > 0 (at least one fused shift-add instruction in reduce_p1 round 2)
 *
 * CPU self-test (runs locally, no GPU needed):
 *   hipcc -O3 --offload-arch=gfx942 -o isa_check isa_check.hip && ./isa_check
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include "primes.h"

/* GPU kernel: exercises reduce_p1 so the device ISA contains the function.  */
__global__ void check_reduce_p1_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t*       __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = reduce_p1((__uint128_t)a[i] * b[i]);
}

/* CPU host test: validates reduce_p1 against reference (__uint128_t %). */
static int cpu_selftest(void)
{
    int ok = 1;
    uint64_t cases[][2] = {
        { P1 - 1,            P1 - 1            },
        { UINT64_C(1) << 32, UINT64_C(1) << 32 },
        { UINT64_C(0xdeadbeefcafe), UINT64_C(0x123456789abc) },
        { 1, 0 },
        { 0, 1 },
    };
    int n = (int)(sizeof(cases) / sizeof(cases[0]));
    int i;
    for (i = 0; i < n; i++) {
        uint64_t a = cases[i][0] % P1;
        uint64_t b = cases[i][1] % P1;
        uint64_t ref = (uint64_t)((__uint128_t)a * b % P1);
        uint64_t got = reduce_p1((__uint128_t)a * b);
        if (got != ref) {
            fprintf(stderr, "  FAIL case %d: a=%llx b=%llx ref=%llx got=%llx\n",
                    i, (unsigned long long)a, (unsigned long long)b,
                    (unsigned long long)ref, (unsigned long long)got);
            ok = 0;
        }
    }
    if (ok) printf("[PASS] reduce_p1 CPU self-test (%d cases)\n", n);
    else    printf("[FAIL] reduce_p1 CPU self-test\n");
    return ok;
}

int main(void)
{
    printf("=== ISA check: reduce_p1 / v_lshl_add_u64 ===\n\n");

    if (!cpu_selftest()) return 1;

    printf("\nTo verify gfx942 ISA:\n");
    printf("  hipcc -O3 --offload-arch=gfx942 --save-temps -o /dev/null isa_check.hip\n");
    printf("  grep -c v_lshl_add_u64 isa_check-hip-amdgcn-amd-amdhsa-gfx942.s\n");
    printf("  (expected: > 0)\n");
    return 0;
}
