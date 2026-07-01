#!/bin/bash
# =============================================================================
# mi300a_probe.sh — Environment and hardware diagnostics for NTT optimisation
#
# PURPOSE: Collect system information needed to tune the NTT kernel for the
#          MI300A. Run this on the target system and save the output:
#
#            bash mi300a_probe.sh 2>&1 | tee mi300a_results.txt
#
#          Then share mi300a_results.txt. Each section explains what is being
#          collected and why it matters for NTT performance.
#
# SECTIONS:
#   1. ROCm & HIP environment
#   2. GPU device properties (raw rocminfo)
#   3. Unified memory topology
#   4. CPU and NUMA layout
#   5. Memory system
#   6. Cray/HPE module environment
#   7. Compiler versions and flags
#   8. Available profiling tools
#   9. NTT binary self-report (runs our own hipGetDeviceProperties query)
#  10. Quick bandwidth smoke-test
# =============================================================================

HDR() { printf '\n\033[1;36m━━━━  %s  ━━━━\033[0m\n' "$*"; }
CMD() { printf '\033[1;33m$ %s\033[0m\n' "$*"; eval "$*" 2>&1 || true; printf '\n'; }

printf '\033[1;37m╔══════════════════════════════════════════════════════════╗\n'
printf '║         NTT / MI300A — Environment Probe                ║\n'
printf '╚══════════════════════════════════════════════════════════╝\033[0m\n'
printf 'Collected: %s\n' "$(date)"
printf 'Hostname:  %s\n' "$(hostname)"

# ─────────────────────────────────────────────────────────────────────────────
HDR "1. ROCm & HIP Environment"
# Needed to confirm ROCm version, offload arch support, and library paths.
# hipconfig --full shows everything hipcc sees at compile time.
# ─────────────────────────────────────────────────────────────────────────────

CMD "hipconfig --full"
CMD "hipconfig --version"
CMD "hipconfig --compiler"
CMD "echo ROCM_PATH=\$ROCM_PATH"
CMD "echo HIP_PATH=\$HIP_PATH"
CMD "ls /opt/rocm/bin 2>/dev/null | head -30"

# ─────────────────────────────────────────────────────────────────────────────
HDR "2. GPU Device Properties — rocminfo"
# This is the most important section. We need:
#   - Compute Unit count (target: 228 on MI300A)
#   - Wavefront size (target: 64 on CDNA3)
#   - LDS size per CU — determines how many NTT elements fit in shared memory
#     and how many stages we can fuse into one kernel launch
#   - Max wavefronts per CU — sets occupancy ceiling
#   - HBM clock and memory bandwidth — sets the roof for bandwidth-bound kernels
#   - L2 cache size — affects twiddle factor caching strategy
#   - Max work-group size — caps our block_size parameter
#   - Registers per CU — constrains __launch_bounds__ choices
# ─────────────────────────────────────────────────────────────────────────────

CMD "rocminfo"
CMD "rocm-smi"
CMD "rocm-smi --showallinfo"
CMD "rocm-smi --showmeminfo all"
CMD "rocm-smi --showclocks"
CMD "rocm-smi --showpids"

# ─────────────────────────────────────────────────────────────────────────────
HDR "3. Unified Memory Topology (MI300A-specific)"
# MI300A is an APU: CPU and GPU share the same HBM3. There is no PCIe copy.
# We need to confirm:
#   - Whether hipHostMalloc pointers are directly dereferenceable on device
#   - XNACK (retry on page fault) status — affects pointer sharing
#   - NUMA distance between CPU sockets and the GPU
#   - Whether large pages are available (2 MB / 1 GB hugepages)
# ─────────────────────────────────────────────────────────────────────────────

CMD "rocm-smi --showtoponuma"
CMD "rocm-smi --showtopo"
CMD "cat /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | grep -E '(name|simd_count|mem_banks|max_waves|lds_size|wave_front_size|array_count|simd_arrays|cu_per_simd|wavefronts|vendor|device|location)'"
CMD "cat /proc/sys/vm/nr_hugepages"
CMD "cat /proc/sys/vm/nr_overcommit_hugepages"
CMD "grep -i huge /proc/meminfo"
CMD "numactl --hardware"
CMD "numactl --show"

# ─────────────────────────────────────────────────────────────────────────────
HDR "4. CPU and System Topology"
# The CPU companion on MI300A is AMD EPYC (likely Genoa). We need:
#   - Core count and NUMA layout (affects host-side twiddle precomputation)
#   - Cache hierarchy (L1/L2/L3) — guides CPU-side buffer sizing
#   - CPU frequency — sets host computation speed reference
# ─────────────────────────────────────────────────────────────────────────────

CMD "lscpu"
CMD "lstopo --no-io 2>/dev/null || hwloc-info 2>/dev/null | head -60"
CMD "lspci 2>/dev/null | grep -iE '(amd|display|vga|gpu|accelerator)'"

# ─────────────────────────────────────────────────────────────────────────────
HDR "5. Memory System"
# Total HBM3 capacity and available memory determine maximum transform sizes.
# On MI300A: 128 GB HBM3 shared between CPU and GPU.
# Large-n NTT (n=2^20) needs 2 × n × 8 bytes = 16 MB per transform.
# ─────────────────────────────────────────────────────────────────────────────

CMD "cat /proc/meminfo"
CMD "free -h"
CMD "dmidecode --type memory 2>/dev/null | grep -E '(Size|Speed|Type|Manufacturer)' | head -40"

# ─────────────────────────────────────────────────────────────────────────────
HDR "6. Cray/HPE Module Environment"
# We need to know exactly which modules are loaded and available so the
# Makefile can reference the right compiler wrappers and library paths.
# In particular: PrgEnv-amd, rocm, cray-shmem, cray-libsci versions.
# ─────────────────────────────────────────────────────────────────────────────

CMD "module list"
CMD "module avail 2>&1"
CMD "echo CC=\$CC"
CMD "echo CXX=\$CXX"
CMD "echo FC=\$FC"
CMD "echo CRAY_CPU_TARGET=\$CRAY_CPU_TARGET"
CMD "echo CRAY_ACCEL_TARGET=\$CRAY_ACCEL_TARGET"
CMD "echo PE_OFFLOAD_ARCH=\$PE_OFFLOAD_ARCH"
CMD "echo MPICH_DIR=\$MPICH_DIR"
CMD "echo CRAY_LIBSCI_DIR=\$CRAY_LIBSCI_DIR"
CMD "echo ROCM_VERSION=\$ROCM_VERSION"
CMD "printenv | grep -iE '(cray|rocm|hip|amd|gpu|accel|prgenv)' | sort"

# ─────────────────────────────────────────────────────────────────────────────
HDR "7. Compiler Versions and Default Flags"
# We need to know what compiler wrapper cc maps to, what flags the Cray PE
# adds automatically, and whether --offload-arch is set by the environment
# (via craype-accel-amd-gfx942 module) or must be passed explicitly.
# ─────────────────────────────────────────────────────────────────────────────

CMD "cc --version"
CMD "hipcc --version"
CMD "CC --version 2>/dev/null || true"
CMD "cc -### /dev/null 2>&1 | head -20"          # shows flags cc adds implicitly
CMD "hipcc -### /dev/null 2>&1 | head -30"        # shows hipcc implicit flags
CMD "cc -x c /dev/null -dM -E 2>/dev/null | grep -iE '(AMD|GFX|CDNA|HIP|ROCM|VERSION)'"

# ─────────────────────────────────────────────────────────────────────────────
HDR "8. Profiling and Optimisation Tools"
# rocprof and omniperf are the primary tools for kernel profiling on MI300A.
# omniperf gives per-CU counters (memory bandwidth, LDS utilisation, VGPR use).
# We need to know what's available before writing the profiling harness.
# ─────────────────────────────────────────────────────────────────────────────

CMD "rocprof --version 2>/dev/null"
CMD "which rocprof 2>/dev/null"
CMD "which omniperf 2>/dev/null"
CMD "omniperf --version 2>/dev/null"
CMD "which roctracer 2>/dev/null"
CMD "which rocprofv2 2>/dev/null"
CMD "ls /opt/rocm/bin/roc* 2>/dev/null"
CMD "which perf 2>/dev/null && perf stat --version 2>/dev/null"

# ─────────────────────────────────────────────────────────────────────────────
HDR "9. NTT Binary Self-Report"
# Run our own binary — it calls hipGetDeviceProperties() and prints a
# formatted summary of the device. This cross-checks rocminfo output and
# confirms our binary is correctly targeting gfx942.
# Build first: make gpu-mi300a
# ─────────────────────────────────────────────────────────────────────────────

if [ -x "./ntt_gpu_mi300a" ]; then
    CMD "./ntt_gpu_mi300a 256 3329 17 1"
else
    printf '\033[1;33m  ntt_gpu_mi300a not found — run "make gpu-mi300a" first\033[0m\n'
fi

if [ -x "./ntt_cross_verify_mi300a" ]; then
    CMD "./ntt_cross_verify_mi300a 256 3329 17"
    CMD "./ntt_cross_verify_mi300a 256 8380417 1753"
else
    printf '\033[1;33m  ntt_cross_verify_mi300a not found — run "make verify-mi300a" first\033[0m\n'
fi

# ─────────────────────────────────────────────────────────────────────────────
HDR "10. Memory Bandwidth Smoke-Test"
# HBM3 bandwidth is the ceiling for bandwidth-bound kernels (bit-reversal,
# twiddle factor loads). Theoretical MI300A HBM3 peak: ~5.3 TB/s.
# rocm-bandwidth-test measures achieved copy bandwidth between host and device.
# For the NTT, we care about device→device (HBM to LDS) bandwidth most.
# ─────────────────────────────────────────────────────────────────────────────

CMD "which rocm-bandwidth-test 2>/dev/null"
if command -v rocm-bandwidth-test >/dev/null 2>&1; then
    CMD "rocm-bandwidth-test"
else
    printf '  rocm-bandwidth-test not in PATH — try: /opt/rocm/bin/rocm-bandwidth-test\n'
    CMD "/opt/rocm/bin/rocm-bandwidth-test 2>/dev/null"
fi

# ─────────────────────────────────────────────────────────────────────────────
printf '\n\033[1;37m━━━━  Probe complete. Share mi300a_results.txt  ━━━━\033[0m\n\n'
printf 'Suggested output: bash mi300a_probe.sh 2>&1 | tee mi300a_results.txt\n\n'
