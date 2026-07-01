/* cu_probe.hip — verify ROC_GLOBAL_CU_MASK actually restricts which CUs run.
 *
 * Launches a trivial kernel with many workgroups; each wave reads the hardware
 * HW_ID1 register and marks its location (SE/SA/WGP). Counting distinct locations
 * shows how many physical WGPs the runtime actually scheduled onto. If the CU mask
 * works, this count drops when ROC_GLOBAL_CU_MASK reserves CUs; if it's ignored,
 * the count stays at the full device width regardless.
 *
 * The kernel is microseconds long and does no real work, so it cannot saturate the
 * GPU long enough to starve a display page flip — safe to run on the display GPU.
 *
 *   hipcc -O2 --offload-arch=gfx1030 -o tools/cu_probe tools/cu_probe.hip
 *   ./tools/cu_probe                                   # full device
 *   ROC_GLOBAL_CU_MASK=0xff ./tools/cu_probe           # should report fewer
 */
#include <hip/hip_runtime.h>
#include <stdio.h>

#define MAXLOC 2048

__global__ void probe(unsigned *__restrict__ used)
{
    unsigned hwid;
    /* gfx10 HW_ID1: WAVE_ID[4:0], SIMD_ID[9:8], then WGP/SA/SE in the upper bits.
     * Shifting out the low 10 bits yields a per-WGP location key. */
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hwid));
    unsigned loc = (hwid >> 10) & (MAXLOC - 1);
    atomicOr(&used[loc], 1u);
}

int main(void)
{
    unsigned *d_used;
    unsigned h_used[MAXLOC];
    if (hipMalloc(&d_used, sizeof h_used) != hipSuccess) { printf("hipMalloc failed\n"); return 1; }
    hipMemset(d_used, 0, sizeof h_used);

    /* Many small workgroups so the scheduler spreads across every available WGP. */
    probe<<<dim3(200000), dim3(64)>>>(d_used);
    if (hipDeviceSynchronize() != hipSuccess) { printf("kernel launch/sync failed\n"); return 1; }

    hipMemcpy(h_used, d_used, sizeof h_used, hipMemcpyDeviceToHost);
    int n = 0;
    for (int i = 0; i < MAXLOC; i++) if (h_used[i]) n++;
    printf("distinct WGP locations scheduled: %d  (gfx1030 full device = 40 WGPs / 80 CUs)\n", n);
    hipFree(d_used);
    return 0;
}
