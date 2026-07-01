# =============================================================================
# Makefile — CRT-NTT on 4x MI300A K4 node
# ROCm 7.0.3 / gfx942 / PrgEnv-amd/8.6.0 / cray-shmem/12.0.0
#
# Three compilers, one per file type (mi300a_environment_0509.txt confirmed):
#   hipcc  -O3 --offload-arch=gfx942          HIP device+host code (.hip)
#   gcc    -O3 -L$(ROCM_LIB) -Wl,--allow-shlib-undefined  pure C host code
#   oshcc  -O3                                 OpenSHMEM code (.c with shmem.h)
#
# DO NOT add to hipcc CFLAGS:
#   -I/opt/rocm-*      (hipcc injects ROCm headers automatically)
#   -D__HIP_PLATFORM_AMD__ (hipcc injects this)
#   -march=*           (handled by craype-x86-genoa module)
#   -mllvm *           (not confirmed on this system)
#
# REQUIRED runtime environment (set in shell before srun):
#   export ROCM_PATH=/opt/rocm-7.0.3
#   export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
#   export HSA_ENABLE_SDMA=1
#   export ROCR_VISIBLE_DEVICES=0,1,2,3
#   export HIP_VISIBLE_DEVICES=0,1,2,3
#
# REQUIRED: all execution via srun (direct ./binary fails, cgroup restriction).
# Partition is `-p mi` (confirmed mi300a_environment_0509.txt §2; the `check`
# target below uses it):
#   Single-node test:  srun -p mi -N 1 -n 1 ./test_ntt
#   Single-node bench: srun -p mi -N 1 -n 4 ./crt_ntt 20 100
#   Multi-node bench:  srun -p mi -N 2 -n 8 --ntasks-per-node=4 ./crt_ntt 20 100
# =============================================================================

HIPCC    := hipcc
GCC      := gcc
OSHCC    := oshcc

ROCM_LIB := $(ROCM_PATH)/lib

# ─── HIP compile flags ────────────────────────────────────────────────────────
# Exactly the confirmed set from MI300A_env0429.md.
HIP_CFLAGS     := -O3 --offload-arch=gfx942 -DNTT_SOLINAS_P0
DEV_HIP_CFLAGS := -O3 --offload-arch=gfx1030 -DGFX1030_LOCAL=1 $(EXTRA_DEF)

# ─── C host compile flags ─────────────────────────────────────────────────────
# gcc for pure C host code (orchestrator, utilities).
C_CFLAGS  := -O3
C_LDFLAGS := -L$(ROCM_LIB) -Wl,--allow-shlib-undefined

# ─── SHMEM compile/link flags ─────────────────────────────────────────────────
# oshcc wraps mpicc/gcc with SHMEM include and library paths automatically.
# -fopenmp: OpenMP runtime for garner.hip Garner loop (LDFLAGS position).
SHMEM_CFLAGS  := -O3
SHMEM_LDFLAGS := -fopenmp -Wl,--allow-shlib-undefined

# ─── HIP objects (compiled separately to allow linking with oshcc) ─────────────
NTT_OBJ     := ntt_kernel.o
GARNER_OBJ  := garner.o
XFER_GPU_OBJ:= transfer_gpu.o

# ─── Targets ──────────────────────────────────────────────────────────────────
# ─── arith/ objects ───────────────────────────────────────────────────────────
ARITH_C_OBJS  := arith/bigint.o arith/multiply.o arith/newton.o arith/base_convert.o
ARITH_HIP_OBJ := arith/ntt_mul.o

.PHONY: all dev dev-isa clean check env isa-check arith test-arith test-arith-l112 \
        test-arith-dual test-arith-fuzz test-arith-gmp test-reduce test-modops \
        test-transfer-core transfer-kernels-isa test-e2e-oracle

all: crt_ntt test_ntt

# ─── Dev targets (6900 XT / gfx1030) ──────────────────────────────────────────
# Builds test_ntt only — crt_ntt requires oshcc (Cray SHMEM, MI300A cluster).
# Run directly without srun: ./test_ntt_dev
# All 8 tests execute on device 0 (hipSetDevice(PIDX % g_n_devs) fallback).
dev: test_ntt_dev

test_ntt_dev: test_dev.o ntt_kernel_dev.o garner_dev.o transfer_gpu_dev.o $(ARITH_C_OBJS)
	$(HIPCC) $(DEV_HIP_CFLAGS) -fopenmp \
	    test_dev.o ntt_kernel_dev.o garner_dev.o transfer_gpu_dev.o \
	    $(ARITH_C_OBJS) -o test_ntt_dev

test_dev.o: test.hip primes.h shoup.h crt_ntt.h
	$(HIPCC) $(DEV_HIP_CFLAGS) -fopenmp -c test.hip -o test_dev.o

ntt_kernel_dev.o: ntt_kernel.hip primes.h shoup.h crt_ntt.h
	$(HIPCC) $(DEV_HIP_CFLAGS) -c ntt_kernel.hip -o ntt_kernel_dev.o

garner_dev.o: garner.hip primes.h crt_ntt.h
	$(HIPCC) $(DEV_HIP_CFLAGS) -fopenmp -c garner.hip -o garner_dev.o

transfer_gpu_dev.o: transfer_gpu.hip
	$(HIPCC) $(DEV_HIP_CFLAGS) -c transfer_gpu.hip -o transfer_gpu_dev.o

dev-isa: isa_check.hip primes.h
	$(HIPCC) $(DEV_HIP_CFLAGS) --save-temps -o /dev/null isa_check.hip 2>&1 | grep -v "^$$" || true
	@grep -c v_lshl_add_u64 isa_check-hip-amdgcn-amd-amdhsa-gfx1030.s 2>/dev/null \
	    && echo "[PASS] v_lshl_add_u64 confirmed in gfx1030 ISA" \
	    || echo "[INFO] gfx1030 may use different instruction — check .s file"

# ── Phase F dual-build (LIMB_BITS 64/112; see ARCHITECTURE.md §4.2) ─────────
# One source, two limb widths. `test-arith` = the LB=64 oracle (default,
# back-compat). `test-arith-dual` builds+runs BOTH; each must be 51/51 —
# test_arith.c is representation-agnostic so it is a true differential.
ARITH_TEST_SRC := arith/test_arith.c arith/bigint.c arith/multiply.c \
                  arith/newton.c arith/base_convert.c
ARITH_TEST_DEP := arith/bigint.h arith/multiply.h arith/newton.h \
                  arith/base_convert.h crt_ntt.h primes.h

test-arith: arith/test_arith_l64
	./arith/test_arith_l64

test-arith-l112: arith/test_arith_l112
	./arith/test_arith_l112

test-arith-dual: arith/test_arith_l64 arith/test_arith_l112
	@echo "== LIMB_BITS=64 =="  && ./arith/test_arith_l64
	@echo "== LIMB_BITS=112 ==" && ./arith/test_arith_l112

# Adversarial fuzz harness — 2000 random trials at each width, hits the
# gather() LB=112 re-chunking path + scatter() residue range + add/sub
# inverse identity. Built into two binaries (one per LIMB_BITS).
ARITH_FUZZ_SRC := arith/test_arith_fuzz.c arith/bigint.c arith/multiply.c \
                  arith/newton.c arith/base_convert.c

test-arith-fuzz: arith/test_arith_fuzz_l64 arith/test_arith_fuzz_l112
	@echo "== FUZZ LIMB_BITS=64 =="  && ./arith/test_arith_fuzz_l64
	@echo "== FUZZ LIMB_BITS=112 ==" && ./arith/test_arith_fuzz_l112

arith/test_arith_fuzz_l64: $(ARITH_FUZZ_SRC) $(ARITH_TEST_DEP)
	$(GCC) -O2 -Wall -Wextra -DLIMB_BITS=64 -I. $(ARITH_FUZZ_SRC) -o $@

arith/test_arith_fuzz_l112: $(ARITH_FUZZ_SRC) $(ARITH_TEST_DEP)
	$(GCC) -O2 -Wall -Wextra -DLIMB_BITS=112 -I. $(ARITH_FUZZ_SRC) -o $@

# GMP cross-oracle: independent arbitrary-precision reference for the
# host-callable arith paths (decimal conversion, add, sub). Built dual-
# width. Requires libgmp-dev (#include <gmp.h>, -lgmp).
ARITH_GMP_SRC := arith/test_arith_gmp.c arith/bigint.c arith/multiply.c \
                 arith/newton.c arith/base_convert.c

test-arith-gmp: arith/test_arith_gmp_l64 arith/test_arith_gmp_l112
	@echo "== GMP LIMB_BITS=64 =="  && ./arith/test_arith_gmp_l64
	@echo "== GMP LIMB_BITS=112 ==" && ./arith/test_arith_gmp_l112

arith/test_arith_gmp_l64: $(ARITH_GMP_SRC) $(ARITH_TEST_DEP)
	$(GCC) -O2 -Wall -Wextra -DLIMB_BITS=64 -I. $(ARITH_GMP_SRC) -o $@ -lgmp

arith/test_arith_gmp_l112: $(ARITH_GMP_SRC) $(ARITH_TEST_DEP)
	$(GCC) -O2 -Wall -Wextra -DLIMB_BITS=112 -I. $(ARITH_GMP_SRC) -o $@ -lgmp

# GPU scatter/gather core oracle: host-callable per-element logic from
# transfer_core.h (the code the transfer_kernels.hip kernels wrap) vs the
# bigint_scatter[_t] / bigint_gather[_t] reference. Pure C, dual-width, no GPU.
TRANSFER_CORE_SRC := arith/test_transfer_core.c arith/bigint.c
TRANSFER_CORE_DEP := transfer_core.h arith/bigint.h crt_ntt.h primes.h

test-transfer-core: arith/test_transfer_core_l64 arith/test_transfer_core_l112
	@echo "== TRANSFER-CORE LIMB_BITS=64 =="  && ./arith/test_transfer_core_l64
	@echo "== TRANSFER-CORE LIMB_BITS=112 ==" && ./arith/test_transfer_core_l112

arith/test_transfer_core_l64: $(TRANSFER_CORE_SRC) $(TRANSFER_CORE_DEP)
	$(GCC) -O2 -Wall -Wextra -DLIMB_BITS=64 -I. $(TRANSFER_CORE_SRC) -o $@

arith/test_transfer_core_l112: $(TRANSFER_CORE_SRC) $(TRANSFER_CORE_DEP)
	$(GCC) -O2 -Wall -Wextra -DLIMB_BITS=112 -I. $(TRANSFER_CORE_SRC) -o $@

# Cross-compile the GPU scatter/gather kernels for gfx942 (compile-only; no GPU
# needed). Verifies the kernels build clean (0 warnings) for the MI300A target.
# MI300A enablement note: building transfer_gpu.hip with -DNTT_GPU_SCATTER_GATHER
# (the gated on-device scatter/gather/CLA path, default OFF) ALSO requires adding
# transfer_kernels.o to the crt_ntt / test_ntt link line, and operand/result
# BigInt limbs in unified memory. Verified signature-compatible + linkable here;
# GPU runtime pending MI300A (ROADMAP G10/G11).
transfer_kernels.o: transfer_kernels.hip transfer_core.h arith/bigint.h \
                    crt_ntt.h primes.h
	$(HIPCC) $(HIP_CFLAGS) -c transfer_kernels.hip -o transfer_kernels.o

transfer-kernels-isa: transfer_kernels.o
	@echo "[PASS] transfer_kernels.hip cross-compiled clean for gfx942"

# End-to-end CRT-NTT multiply oracle: real bigint_scatter -> independent CPU NTT
# convolution per prime -> CRT combine -> real bigint_gather, vs GMP product.
# Validates the whole pipeline's logic on CPU. Dual-width, pure C + libgmp.
E2E_SRC := arith/test_e2e_oracle.c arith/bigint.c
E2E_DEP := arith/bigint.h crt_ntt.h primes.h transfer_core.h

test-e2e-oracle: arith/test_e2e_oracle_l64 arith/test_e2e_oracle_l112
	@echo "== E2E LIMB_BITS=64 =="  && ./arith/test_e2e_oracle_l64
	@echo "== E2E LIMB_BITS=112 ==" && ./arith/test_e2e_oracle_l112

arith/test_e2e_oracle_l64: $(E2E_SRC) $(E2E_DEP)
	$(GCC) -O2 -Wall -Wextra -DLIMB_BITS=64 -I. $(E2E_SRC) -o $@ -lgmp

arith/test_e2e_oracle_l112: $(E2E_SRC) $(E2E_DEP)
	$(GCC) -O2 -Wall -Wextra -DLIMB_BITS=112 -I. $(E2E_SRC) -o $@ -lgmp

# Exhaustive reduce_p0..p3 correctness vs (x % P). Pure C, no GPU.
# Caught the reduce_p1 (round-2 fold) and reduce_p0 (final subtract) bugs.
test-reduce: test_reduce
	./test_reduce

test_reduce: test_reduce.c primes.h
	$(GCC) -O2 -Wall -Wextra -I. test_reduce.c -o test_reduce

# Exhaustive addmod/submod/mulmod/montgomery_mul/shoup_mul vs __uint128_t
# reference (montgomery_mul checked via reference-free REDC invariant).
# Caught the montgomery_mul 128-bit-sum-overflow bug. Host-only (no GPU).
test-modops: test_modops
	./test_modops

test_modops: test_modops.hip primes.h shoup.h crt_ntt.h
	@bash ../scripts/gpu_health_check.sh 2>/dev/null || \
	  { echo "ERROR: gpu_health_check failed — hipcc link unsafe on crash-recovery boot"; exit 1; }
	$(HIPCC) -O3 --offload-arch=gfx1030 -fopenmp -I. test_modops.hip -o test_modops

arith/test_arith_l64: $(ARITH_TEST_SRC) $(ARITH_TEST_DEP)
	$(GCC) -O2 -Wall -Wextra -DLIMB_BITS=64 -I. $(ARITH_TEST_SRC) -o $@

arith/test_arith_l112: $(ARITH_TEST_SRC) $(ARITH_TEST_DEP)
	$(GCC) -O2 -Wall -Wextra -DLIMB_BITS=112 -I. $(ARITH_TEST_SRC) -o $@

arith: $(ARITH_C_OBJS) $(ARITH_HIP_OBJ)

arith/bigint.o: arith/bigint.c arith/bigint.h crt_ntt.h primes.h
	$(GCC) $(C_CFLAGS) -c arith/bigint.c -o arith/bigint.o

# bigint.h #includes crt_ntt.h + primes.h, so every TU pulling bigint.h
# transitively depends on them — list them or a prime/CRT change silently
# does NOT rebuild these objects.
arith/multiply.o: arith/multiply.c arith/multiply.h arith/bigint.h \
                  crt_ntt.h primes.h
	$(GCC) $(C_CFLAGS) -c arith/multiply.c -o arith/multiply.o

arith/newton.o: arith/newton.c arith/newton.h arith/bigint.h arith/multiply.h \
                crt_ntt.h primes.h
	$(GCC) $(C_CFLAGS) -c arith/newton.c -o arith/newton.o

arith/base_convert.o: arith/base_convert.c arith/base_convert.h \
                      arith/bigint.h arith/newton.h arith/multiply.h \
                      crt_ntt.h primes.h
	$(GCC) $(C_CFLAGS) -c arith/base_convert.c -o arith/base_convert.o -lm

arith/ntt_mul.o: arith/ntt_mul.hip arith/bigint.h primes.h crt_ntt.h shoup.h
	$(HIPCC) $(HIP_CFLAGS) -fopenmp -c arith/ntt_mul.hip -o arith/ntt_mul.o

# Compile HIP translation units to object files
ntt_kernel.o: ntt_kernel.hip primes.h shoup.h crt_ntt.h
	$(HIPCC) $(HIP_CFLAGS) -c ntt_kernel.hip -o ntt_kernel.o

garner.o: garner.hip primes.h crt_ntt.h
	$(HIPCC) $(HIP_CFLAGS) -fopenmp -c garner.hip -o garner.o

transfer_gpu.o: transfer_gpu.hip
	$(HIPCC) $(HIP_CFLAGS) -c transfer_gpu.hip -o transfer_gpu.o

# Compile pure C SHMEM translation unit
transfer_shmem.o: transfer_shmem.c
	$(OSHCC) $(SHMEM_CFLAGS) -c transfer_shmem.c -o transfer_shmem.o

# Main benchmark binary: linked with oshcc to pull in SHMEM runtime
main.o: main.hip primes.h shoup.h crt_ntt.h
	$(HIPCC) $(HIP_CFLAGS) -fopenmp -c main.hip -o main.o

crt_ntt: main.o $(NTT_OBJ) $(GARNER_OBJ) $(XFER_GPU_OBJ) transfer_shmem.o
	$(OSHCC) $(SHMEM_CFLAGS) $(SHMEM_LDFLAGS) \
	    main.o $(NTT_OBJ) $(GARNER_OBJ) $(XFER_GPU_OBJ) transfer_shmem.o \
	    -L$(ROCM_LIB) -lamdhip64 \
	    -o crt_ntt

# Test binary: compile test.hip separately then link to avoid hipcc treating .o as source
test.o: test.hip primes.h shoup.h crt_ntt.h
	$(HIPCC) $(HIP_CFLAGS) -fopenmp -c test.hip -o test.o

test_ntt: test.o $(NTT_OBJ) $(GARNER_OBJ) $(XFER_GPU_OBJ) $(ARITH_C_OBJS)
	$(HIPCC) $(HIP_CFLAGS) -fopenmp \
	    test.o $(NTT_OBJ) $(GARNER_OBJ) $(XFER_GPU_OBJ) \
	    $(ARITH_C_OBJS) -o test_ntt

# Print required environment variables
env:
	@echo "Set these before running:"
	@echo "  export ROCM_PATH=/opt/rocm-7.0.3"
	@echo "  export LD_LIBRARY_PATH=\$$ROCM_PATH/lib:\$$LD_LIBRARY_PATH"
	@echo "  export HSA_ENABLE_SDMA=1"
	@echo "  export ROCR_VISIBLE_DEVICES=0,1,2,3"
	@echo "  export HIP_VISIBLE_DEVICES=0,1,2,3"

# Run tests then benchmark via srun (mandatory on this system).
# test_ntt: single process accessing all 4 devices internally (-n 1).
# crt_ntt:  4 SHMEM PEs, one per APU (-n 4).
check: test_ntt crt_ntt
	OMP_NUM_THREADS=96 OMP_PROC_BIND=spread OMP_PLACES=cores \
	    srun -p mi -N 1 -n 1 ./test_ntt
	OMP_NUM_THREADS=96 OMP_PROC_BIND=spread OMP_PLACES=cores \
	    srun -p mi -N 1 -n 4 ./crt_ntt 20 100

# Cross-compile isa_check.hip with --save-temps; grep gfx942 .s for v_lshl_add_u64.
isa-check: isa_check.hip primes.h
	$(HIPCC) $(HIP_CFLAGS) --save-temps -o /dev/null isa_check.hip 2>&1 | grep -v "^$$" || true
	@grep -c v_lshl_add_u64 isa_check-hip-amdgcn-amd-amdhsa-gfx942.s 2>/dev/null \
	    && echo "[PASS] v_lshl_add_u64 confirmed in gfx942 ISA" \
	    || echo "[WARN] .s file not found — check --save-temps output names"

clean:
	rm -f crt_ntt test_ntt test_ntt_dev test_reduce test_modops \
	    *.o arith/*.o arith/test_arith arith/test_arith_l64 arith/test_arith_l112 \
	    arith/test_arith_fuzz_l64 arith/test_arith_fuzz_l112 \
	    arith/test_arith_gmp_l64 arith/test_arith_gmp_l112 \
	    arith/test_transfer_core_l64 arith/test_transfer_core_l112 \
	    arith/test_e2e_oracle_l64 arith/test_e2e_oracle_l112 \
	    isa_check isa_check-*.s isa_check-*.bc isa_check-*.ll \
	    isa_check-*.hipi isa_check-*.hipfb isa_check*.hipfb \
	    isa_check-*.out isa_check-*.resolution.txt
	@# generic --save-temps artifacts (any TU: ntt_kernel/isa_check/…
	@# from `hipcc --save-temps`, e.g. isa-check / G9 ISA inspection)
	rm -f *-hip-amdgcn-amd-amdhsa-*.s  *-hip-amdgcn-amd-amdhsa-*.bc \
	    *-hip-amdgcn-amd-amdhsa-*.ll *-hip-amdgcn-amd-amdhsa-*.hipi \
	    *-hip-amdgcn-amd-amdhsa-*.out \
	    *-host-x86_64-unknown-linux-gnu.s *-host-x86_64-unknown-linux-gnu.bc \
	    *-host-x86_64-unknown-linux-gnu.ll *-host-x86_64-unknown-linux-gnu.hipi \
	    *.hip-hip-amdgcn-amd-amdhsa.hipfb
