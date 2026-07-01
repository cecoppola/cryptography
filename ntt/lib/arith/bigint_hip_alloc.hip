/*
 * bigint_hip_alloc.hip — hipMallocManaged-backed BigInt allocation.
 *
 * Provides strong definitions of bigint_alloc_managed() and hip_free_limbs(),
 * overriding the weak fallbacks in bigint.c.  Link this object alongside
 * bigint.o when GPU managed-memory placement is required (compute_e,
 * lib NTT tests).
 *
 * On MI300A (CDNA3, gfx942): hipMallocManaged allocates in unified HBM3;
 * both CPU and GPU access the same physical memory with no PCIe copy.
 * On gfx1030 (RDNA2, no XNACK): managed memory is SYSTEM RAM and GPU kernel
 * reads of it page-fault, so the managed path is force-disabled there
 * (base_convert.c sets use_mgd=0 under GFX1030_LOCAL). This allocator is
 * therefore an MI300A-only fast path; gfx1030 uses plain bigint_alloc.
 */

#include "bigint.h"
#include <hip/hip_runtime.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#define HIP_CHECK(call) do {                                              \
    hipError_t _e = (call);                                               \
    if (_e != hipSuccess) {                                               \
        fprintf(stderr, "HIP error %s:%d: %s\n",                         \
                __FILE__, __LINE__, hipGetErrorString(_e));               \
        exit(1);                                                          \
    }                                                                     \
} while (0)

extern "C" {

/*
 * bigint_alloc_managed — allocate limbs in GPU-accessible managed memory.
 * Returns a BigInt with managed=1; bigint_free() calls hip_free_limbs().
 * Pre-allocate with sufficient cap: bigint_resize() aborts on managed BigInts.
 */
BigInt bigint_alloc_managed(int cap)
{
    BigInt a;
    assert(cap > 0);
    HIP_CHECK(hipMallocManaged(&a.limbs, (size_t)cap * sizeof(limb_t)));
    memset(a.limbs, 0, (size_t)cap * sizeof(limb_t));
    a.n_limbs = 0;
    a.cap     = cap;
    a.managed = 1;
    return a;
}

/* hip_free_limbs — release a hipMallocManaged limb array. */
void hip_free_limbs(limb_t *p)
{
    (void)hipFree(p);
}

} /* extern "C" */
