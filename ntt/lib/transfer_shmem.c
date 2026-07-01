/*
 * transfer_shmem.c — Inter-node SHMEM transfers
 *
 * Compiled with: oshcc -O3
 * Pure C — no HIP device code. Contains only OpenSHMEM API calls.
 * Linked into crt_ntt via oshcc which provides libsma + PMI automatically.
 *
 * OpenSHMEM version: 1.5 (cray-shmem/12.0.0)
 * Runtime: srun -N <nodes> -n <pes> --ntasks-per-node=<ppn> ./crt_ntt
 *
 * shmem_broadcastmem() is the correct API on cray-shmem/12.0.0.
 * shmem_broadcast64() (legacy pSync API) is NOT available on this system.
 *
 * After shmem_broadcastmem() returns, every PE has the input in shmem_buf.
 * The caller then uses hipMemcpy (host->device) to move data into APU 0,
 * and broadcast_input() (main.hip) distributes intra-node.
 */

#include <shmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * shmem_distribute_input — broadcast input polynomial from root PE to all PEs
 *
 * shmem_buf:  shmem_malloc'd symmetric buffer (same logical address on all PEs).
 *             On the root PE (src_pe), must already contain the input data
 *             before this call.
 * d_dev0:     hipMalloc'd buffer on device 0 of this node. Filled after
 *             the broadcast completes. Caller must #include hip_runtime.h
 *             and call hipMemcpy themselves, or this file needs HIP — to keep
 *             this file pure C, the hipMemcpy is done by the caller (main.hip).
 *             This function only performs the SHMEM collective.
 * bytes:      number of bytes to transfer (must be multiple of sizeof(long)).
 * src_pe:     the PE number holding the master copy of the polynomial.
 *
 * shmem_broadcastmem() is a collective — all PEs must call it simultaneously.
 * It blocks until the broadcast is complete on all PEs. No separate barrier.
 */
void shmem_distribute_input_collective(
    void*  shmem_buf,
    size_t bytes,
    int    src_pe)
{
    /* shmem_broadcastmem: copies bytes from src_pe's shmem_buf into every
     * other PE's shmem_buf. On src_pe itself this is a no-op (src==dst). */
    shmem_broadcastmem(
        SHMEM_TEAM_WORLD,
        shmem_buf,          /* destination: symmetric buffer on this PE */
        shmem_buf,          /* source: same address — read from src_pe   */
        bytes,
        src_pe);
    /* shmem_broadcastmem is blocking and collectively synchronizing.
     * All PEs have valid data in shmem_buf when this returns.            */
}

/*
 * shmem_collect_output — gather 250-bit results from all PEs onto dst_pe
 *
 * Each PE writes its local_bytes of results into dst_pe's shmem_out_buf
 * at the offset corresponding to its PE number.
 *
 * shmem_out_buf must be shmem_malloc'd with size >= npes * local_bytes.
 * local_result may be any host pointer.
 * After srun completes, dst_pe has the full concatenated output.
 */
void shmem_collect_output(
    void*       shmem_out_buf,
    const void* local_result,
    size_t      local_bytes,
    int         dst_pe)
{
    int me = shmem_my_pe();
    size_t offset = (size_t)me * local_bytes;

    /* Non-blocking put: this PE's results -> dst_pe's buffer at [offset]. */
    shmem_putmem_nbi(
        (char*)shmem_out_buf + offset,
        local_result,
        local_bytes,
        dst_pe);

    /* Wait for this PE's put to complete before barrier. */
    shmem_quiet();

    /* All PEs synchronize: dst_pe now has complete data from all PEs. */
    shmem_barrier_all();
}
