# =============================================================================
# lib3/compute_e/Makefile — compute Euler's number e to arbitrary precision.
#
# Depends on lib2/arith/ (B1-B4) and lib2/ (NTT engine).
# HIP objects are compiled with hipcc; pure C with gcc.
# Link with hipcc (pulls in amdhip64) because ntt_mul.hip is in the link.
#
# Usage:
#   make              — build compute_e binary
#   make clean        — remove all generated files
#   make check        — quick smoke test: 100 digits of e
# =============================================================================

HIPCC    := hipcc
GCC      := gcc
ROCM_LIB := $(ROCM_PATH)/lib

ARITH_DIR := ../../lib2/arith
CRT_DIR   := ../../lib2

# ─── Compile flags ────────────────────────────────────────────────────────────
HIP_CFLAGS     := -O3 --offload-arch=gfx942 -fopenmp
HIP_CFLAGS_DEV := -O3 --offload-arch=gfx1030 -DGFX1030_LOCAL=1 -fopenmp $(EXTRA_DEF)
C_CFLAGS       := -O3 -Wall -Wextra -fopenmp
C_CFLAGS_DEV   := $(C_CFLAGS) -DGFX1030_LOCAL=1

# ─── arith/ C objects ─────────────────────────────────────────────────────────
ARITH_C_SRC := \
    $(ARITH_DIR)/bigint.c \
    $(ARITH_DIR)/multiply.c \
    $(ARITH_DIR)/newton.c \
    $(ARITH_DIR)/base_convert.c

ARITH_C_OBJ := $(notdir $(ARITH_C_SRC:.c=.o))

# ─── arith/ HIP objects ───────────────────────────────────────────────────────
NTT_MUL_OBJ      := ntt_mul.o
BIGINT_HIP_OBJ   := bigint_hip_alloc.o

# ─── lib2/ HIP objects needed by ntt_mul.hip ────────────────────────────────
NTT_KERNEL_OBJ := ntt_kernel.o
GARNER_OBJ     := garner.o

# ─── compute_e C objects ──────────────────────────────────────────────────────
COMPUTE_E_OBJ := main.o mem_pool.o binary_split.o

ALL_OBJ := $(COMPUTE_E_OBJ) $(ARITH_C_OBJ) \
           $(NTT_MUL_OBJ) $(BIGINT_HIP_OBJ) $(NTT_KERNEL_OBJ) $(GARNER_OBJ)

.PHONY: all dev clean check

all: compute_e
dev: compute_e_dev

# ─── arith/ C objects ─────────────────────────────────────────────────────────
bigint.o: $(ARITH_DIR)/bigint.c $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS) -I$(CRT_DIR) -c $< -o $@

multiply.o: $(ARITH_DIR)/multiply.c $(ARITH_DIR)/multiply.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS) -I$(CRT_DIR) -c $< -o $@

newton.o: $(ARITH_DIR)/newton.c $(ARITH_DIR)/newton.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS) -I$(CRT_DIR) -c $< -o $@

base_convert.o: $(ARITH_DIR)/base_convert.c $(ARITH_DIR)/base_convert.h \
                $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS) -I$(CRT_DIR) -c $< -o $@ -lm

# ─── arith/ HIP objects ───────────────────────────────────────────────────────
ntt_mul.o: $(ARITH_DIR)/ntt_mul.hip $(CRT_DIR)/primes.h $(CRT_DIR)/crt_ntt.h \
           $(CRT_DIR)/shoup.h $(ARITH_DIR)/bigint.h
	$(HIPCC) $(HIP_CFLAGS) -I$(CRT_DIR) -c $(ARITH_DIR)/ntt_mul.hip -o $@

bigint_hip_alloc.o: $(ARITH_DIR)/bigint_hip_alloc.hip $(ARITH_DIR)/bigint.h
	$(HIPCC) $(HIP_CFLAGS) -I$(CRT_DIR) -c $(ARITH_DIR)/bigint_hip_alloc.hip -o $@

# ─── crtcode/ HIP objects ─────────────────────────────────────────────────────
ntt_kernel.o: $(CRT_DIR)/ntt_kernel.hip $(CRT_DIR)/primes.h $(CRT_DIR)/shoup.h \
              $(CRT_DIR)/crt_ntt.h
	$(HIPCC) $(HIP_CFLAGS) -c $(CRT_DIR)/ntt_kernel.hip -o $@

garner.o: $(CRT_DIR)/garner.hip $(CRT_DIR)/primes.h $(CRT_DIR)/crt_ntt.h
	$(HIPCC) $(HIP_CFLAGS) -c $(CRT_DIR)/garner.hip -o $@

# ─── compute_e application objects ───────────────────────────────────────────
main.o: main.c binary_split.h mem_pool.h \
        $(ARITH_DIR)/multiply.h $(ARITH_DIR)/newton.h $(ARITH_DIR)/base_convert.h \
        $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS) -I$(CRT_DIR) -c main.c -o $@ -lm

mem_pool.o: mem_pool.c mem_pool.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS) -I$(CRT_DIR) -c mem_pool.c -o $@

binary_split.o: binary_split.c binary_split.h mem_pool.h \
                $(ARITH_DIR)/multiply.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS) -I$(CRT_DIR) -c binary_split.c -o $@

# ─── Link: hipcc to pull in amdhip64 and OpenMP ──────────────────────────────
compute_e: $(ALL_OBJ)
	$(HIPCC) $(HIP_CFLAGS) -L$(ROCM_LIB) \
	    $(ALL_OBJ) \
	    -lm -lpthread -o compute_e

# ─── Dev build: gfx1030 (6900 XT local) ─────────────────────────────────────
DEV_OBJ := $(patsubst %.o,%_dev.o,$(ALL_OBJ))

bigint_dev.o: $(ARITH_DIR)/bigint.c $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -I$(CRT_DIR) -c $< -o $@

multiply_dev.o: $(ARITH_DIR)/multiply.c $(ARITH_DIR)/multiply.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -I$(CRT_DIR) -c $< -o $@

newton_dev.o: $(ARITH_DIR)/newton.c $(ARITH_DIR)/newton.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -I$(CRT_DIR) -c $< -o $@

base_convert_dev.o: $(ARITH_DIR)/base_convert.c $(ARITH_DIR)/base_convert.h \
                    $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -I$(CRT_DIR) -c $< -o $@ -lm

ntt_mul_dev.o: $(ARITH_DIR)/ntt_mul.hip $(CRT_DIR)/primes.h $(CRT_DIR)/crt_ntt.h \
               $(CRT_DIR)/shoup.h $(ARITH_DIR)/bigint.h
	$(HIPCC) $(HIP_CFLAGS_DEV) -I$(CRT_DIR) -c $(ARITH_DIR)/ntt_mul.hip -o $@

bigint_hip_alloc_dev.o: $(ARITH_DIR)/bigint_hip_alloc.hip $(ARITH_DIR)/bigint.h
	$(HIPCC) $(HIP_CFLAGS_DEV) -I$(CRT_DIR) -c $(ARITH_DIR)/bigint_hip_alloc.hip -o $@

ntt_kernel_dev.o: $(CRT_DIR)/ntt_kernel.hip $(CRT_DIR)/primes.h $(CRT_DIR)/shoup.h \
                  $(CRT_DIR)/crt_ntt.h
	$(HIPCC) $(HIP_CFLAGS_DEV) -c $(CRT_DIR)/ntt_kernel.hip -o $@

garner_dev.o: $(CRT_DIR)/garner.hip $(CRT_DIR)/primes.h $(CRT_DIR)/crt_ntt.h
	$(HIPCC) $(HIP_CFLAGS_DEV) -c $(CRT_DIR)/garner.hip -o $@

main_dev.o: main.c binary_split.h mem_pool.h \
            $(ARITH_DIR)/multiply.h $(ARITH_DIR)/newton.h $(ARITH_DIR)/base_convert.h \
            $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -I$(CRT_DIR) -c main.c -o $@ -lm

mem_pool_dev.o: mem_pool.c mem_pool.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -I$(CRT_DIR) -c mem_pool.c -o $@

binary_split_dev.o: binary_split.c binary_split.h mem_pool.h \
                    $(ARITH_DIR)/multiply.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -I$(CRT_DIR) -c binary_split.c -o $@

compute_e_dev: $(DEV_OBJ)
	$(HIPCC) $(HIP_CFLAGS_DEV) -L$(ROCM_LIB) \
	    $(DEV_OBJ) \
	    -lm -lpthread -o compute_e_dev

# ─── Smoke test ──────────────────────────────────────────────────────────────
# e = 2.7182818284 5904523536 0287471352 6624977572 4709369995 ...
check: compute_e
	srun -p mi -N 1 -n 1 ./compute_e -d 100

check-dev: compute_e_dev
	./compute_e_dev -d 100

# ─── Canonical test set (bug-hunt scaffolds removed 2026-05-16) ───────────────
# test_kernel_dev     — GPU NTT known-answer test (prime 0), no external deps
# test_polymul_sweep  — randomized full-pipeline differential vs CPU schoolbook
#                       + large-N invariants. RUN VIA the safe wrapper:
#                         ../../scripts/gpu_run.sh 400 ./test_polymul_sweep 15
test_ntt_kernel_dev.o: test_ntt_kernel.hip $(CRT_DIR)/primes.h $(CRT_DIR)/shoup.h
	$(HIPCC) $(HIP_CFLAGS_DEV) -I$(CRT_DIR) -c test_ntt_kernel.hip -o $@

test_kernel_dev: test_ntt_kernel_dev.o ntt_kernel_dev.o garner_dev.o
	@bash ../../scripts/gpu_health_check.sh 2>/dev/null || \
	  { echo "ERROR: gpu_health_check failed — hipcc link unsafe on crash-recovery boot"; exit 1; }
	$(HIPCC) $(HIP_CFLAGS_DEV) -L$(ROCM_LIB) $^ -lm -o $@

test_polymul_sweep_dev.o: test_polymul_sweep.hip $(ARITH_DIR)/bigint.h $(ARITH_DIR)/multiply.h
	$(GCC) $(C_CFLAGS) -I$(CRT_DIR) -x c -c test_polymul_sweep.hip -o $@

test_polymul_sweep: test_polymul_sweep_dev.o bigint_dev.o multiply_dev.o \
                    ntt_mul_dev.o ntt_kernel_dev.o garner_dev.o
	@bash ../../scripts/gpu_health_check.sh 2>/dev/null || \
	  { echo "ERROR: gpu_health_check failed — hipcc link unsafe on crash-recovery boot"; exit 1; }
	$(HIPCC) $(HIP_CFLAGS_DEV) -L$(ROCM_LIB) $^ -lm -o $@

# ─── Phase F.7 dual-build (mirrors lib2/Makefile test-arith-l64/l112) ────────
# One source tree, two limb widths. LB-dependent units (every bigint.h-
# including TU: bigint/multiply/newton/base_convert + ntt_mul.hip
# scatter/gather/limb_pack + main/mem_pool/binary_split) are rebuilt per
# -DLIMB_BITS into *_l<W>.o; ntt_kernel/garner are limb-agnostic (CRT
# residues are uint64_t) so the existing shared ntt_kernel_dev.o /
# garner_dev.o are reused at BOTH widths — same split as lib2's verified
# reference. compute_e decimal output MUST be identical at LB=64 and
# LB=112 (the math differential, F.7a). test_polymul_sweep @ LB=112
# directly exercises ntt_mul.hip on the GPU at 112-bit (F.7b).
# Built via recursive make so the per-width rule set is written once.
.PHONY: f7-build f7-polymul112 _ce_link

compute_e_dev_l64:   ; $(MAKE) _ce_link LB=64
compute_e_dev_l112:  ; $(MAKE) _ce_link LB=112
f7-build: compute_e_dev_l64 compute_e_dev_l112

F7_OBJS := main_l$(LB).o mem_pool_l$(LB).o binary_split_l$(LB).o \
           bigint_l$(LB).o multiply_l$(LB).o newton_l$(LB).o \
           base_convert_l$(LB).o ntt_mul_l$(LB).o \
           ntt_kernel_dev.o garner_dev.o

_ce_link: $(F7_OBJS)
	@bash ../../scripts/gpu_health_check.sh 2>/dev/null || \
	  { echo "ERROR: gpu_health_check failed — hipcc link opens /dev/kfd via ROCm OpenMP" \
	        "runtime and will crash a degraded driver. Power cycle required first."; exit 1; }
	$(HIPCC) $(HIP_CFLAGS_DEV) -L$(ROCM_LIB) $(F7_OBJS) -lm -o compute_e_dev_l$(LB)

bigint_l$(LB).o: $(ARITH_DIR)/bigint.c $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -DLIMB_BITS=$(LB) -I$(CRT_DIR) -c $< -o $@
multiply_l$(LB).o: $(ARITH_DIR)/multiply.c $(ARITH_DIR)/multiply.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -DLIMB_BITS=$(LB) -I$(CRT_DIR) -c $< -o $@
newton_l$(LB).o: $(ARITH_DIR)/newton.c $(ARITH_DIR)/newton.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -DLIMB_BITS=$(LB) -I$(CRT_DIR) -c $< -o $@
base_convert_l$(LB).o: $(ARITH_DIR)/base_convert.c $(ARITH_DIR)/base_convert.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -DLIMB_BITS=$(LB) -I$(CRT_DIR) -c $< -o $@ -lm
ntt_mul_l$(LB).o: $(ARITH_DIR)/ntt_mul.hip $(CRT_DIR)/primes.h $(CRT_DIR)/crt_ntt.h \
                  $(CRT_DIR)/shoup.h $(ARITH_DIR)/bigint.h
	$(HIPCC) $(HIP_CFLAGS_DEV) -DLIMB_BITS=$(LB) -I$(CRT_DIR) -c $(ARITH_DIR)/ntt_mul.hip -o $@
main_l$(LB).o: main.c binary_split.h mem_pool.h \
               $(ARITH_DIR)/multiply.h $(ARITH_DIR)/newton.h $(ARITH_DIR)/base_convert.h \
               $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -DLIMB_BITS=$(LB) -I$(CRT_DIR) -c main.c -o $@ -lm
mem_pool_l$(LB).o: mem_pool.c mem_pool.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -DLIMB_BITS=$(LB) -I$(CRT_DIR) -c mem_pool.c -o $@
binary_split_l$(LB).o: binary_split.c binary_split.h mem_pool.h \
                       $(ARITH_DIR)/multiply.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS_DEV) -DLIMB_BITS=$(LB) -I$(CRT_DIR) -c binary_split.c -o $@

# F.7b: test_polymul_sweep rebuilt at LB=112 (ntt_kernel/garner shared)
test_polymul_sweep_l112.o: test_polymul_sweep.hip $(ARITH_DIR)/bigint.h $(ARITH_DIR)/multiply.h
	$(GCC) $(C_CFLAGS) -DLIMB_BITS=112 -I$(CRT_DIR) -x c -c test_polymul_sweep.hip -o $@
f7-polymul112: test_polymul_sweep_l112.o bigint_l112.o multiply_l112.o \
               ntt_mul_l112.o ntt_kernel_dev.o garner_dev.o
	@bash ../../scripts/gpu_health_check.sh 2>/dev/null || \
	  { echo "ERROR: gpu_health_check failed — hipcc link unsafe on crash-recovery boot"; exit 1; }
	$(HIPCC) $(HIP_CFLAGS_DEV) -L$(ROCM_LIB) $^ -lm -o test_polymul_sweep_l112

# ─── Host-only differential test (no GPU, no hipcc) ──────────────────────────
# compute_e_host: links main/mem_pool/binary_split + lib2/arith C objects
# with a host NTT stub (ntt_stub.c) — plain gcc, no amdhip64. The very
# high -DBIGINT_MUL_THRESHOLD forces schoolbook multiply always, so the
# GPU NTT path is never dispatched at any digit count (the stub only
# satisfies the linker). test-host runs it and diffs the output against
# the known decimal expansion of e (OEIS A001113) — a fully independent
# oracle (e is a mathematical constant).
HOST_SCHOOLBOOK := -DBIGINT_MUL_THRESHOLD=100000000
HOST_OBJ := main_host.o mem_pool_host.o binary_split_host.o \
            bigint_host.o multiply_host.o newton_host.o base_convert_host.o \
            ntt_stub.o

main_host.o: main.c binary_split.h mem_pool.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS) $(HOST_SCHOOLBOOK) -I$(CRT_DIR) -c main.c -o $@
mem_pool_host.o: mem_pool.c mem_pool.h $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS) $(HOST_SCHOOLBOOK) -I$(CRT_DIR) -c mem_pool.c -o $@
binary_split_host.o: binary_split.c binary_split.h mem_pool.h
	$(GCC) $(C_CFLAGS) $(HOST_SCHOOLBOOK) -I$(CRT_DIR) -c binary_split.c -o $@
bigint_host.o: $(ARITH_DIR)/bigint.c $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS) $(HOST_SCHOOLBOOK) -I$(CRT_DIR) -c $< -o $@
multiply_host.o: $(ARITH_DIR)/multiply.c $(ARITH_DIR)/multiply.h
	$(GCC) $(C_CFLAGS) $(HOST_SCHOOLBOOK) -I$(CRT_DIR) -c $< -o $@
newton_host.o: $(ARITH_DIR)/newton.c $(ARITH_DIR)/newton.h
	$(GCC) $(C_CFLAGS) $(HOST_SCHOOLBOOK) -I$(CRT_DIR) -c $< -o $@
base_convert_host.o: $(ARITH_DIR)/base_convert.c $(ARITH_DIR)/base_convert.h
	$(GCC) $(C_CFLAGS) $(HOST_SCHOOLBOOK) -I$(CRT_DIR) -c $< -o $@
ntt_stub.o: ntt_stub.c $(ARITH_DIR)/bigint.h
	$(GCC) $(C_CFLAGS) $(HOST_SCHOOLBOOK) -I$(CRT_DIR) -c ntt_stub.c -o $@

compute_e_host: $(HOST_OBJ)
	$(GCC) $(HOST_OBJ) -lm -fopenmp -o compute_e_host

.PHONY: test-host test-binsplit
test-host: compute_e_host
	bash ../../scripts/test_compute_e.sh

# test-binsplit: adversarial binary_split() test vs an independent
# radix-2^30 bignum reference (26 cells). Pure host C. HOST_SCHOOLBOOK
# (very high -DBIGINT_MUL_THRESHOLD) forces schoolbook multiply always,
# so the GPU NTT path is never dispatched (ntt_stub only links).
test_binary_split: test_binary_split.c binary_split.c mem_pool.c ntt_stub.c \
                   binary_split.h mem_pool.h $(CRT_HDRS)
	$(GCC) $(C_CFLAGS) $(HOST_SCHOOLBOOK) -I$(CRT_DIR) -o $@ \
	    test_binary_split.c binary_split.c mem_pool.c ntt_stub.c \
	    $(ARITH_DIR)/bigint.c $(ARITH_DIR)/multiply.c \
	    $(ARITH_DIR)/newton.c $(ARITH_DIR)/base_convert.c -lm

test-binsplit: test_binary_split
	./test_binary_split

clean:
	rm -f compute_e compute_e_dev test_kernel_dev test_polymul_sweep \
	    compute_e_dev_l64 compute_e_dev_l112 test_polymul_sweep_l112 \
	    compute_e_host test_binary_split \
	    test_exact_dev test_seq_dev test_p1_isolate \
	    bigint_hip_alloc.o bigint_hip_alloc_dev.o *.o bsplit_*.txt

# ── Transitive header closure (2026-05-19 exhaustive audit) ──────────────────
# bigint.h #includes crt_ntt.h + primes.h (and multiply.h/newton.h pull
# bigint.h). Every object built from a TU that transitively includes bigint.h
# must rebuild when the prime table / CRT defs / generator fixes change. These
# recipe-less rules ADD that prerequisite to all object variants in ONE place
# (root + lib2 Makefiles carry the explicit per-rule form; this closes the
# same stale-object-runs-old-code class for lib3/compute_e). $(F7_OBJS)
# expands empty-LB harmlessly in the parent make and correctly under the
# recursive `_ce_link LB=..` sub-make.
CRT_HDRS := $(CRT_DIR)/crt_ntt.h $(CRT_DIR)/primes.h
$(ALL_OBJ) $(DEV_OBJ) $(F7_OBJS): $(CRT_HDRS)
test_polymul_sweep_dev.o test_polymul_sweep_l112.o: $(CRT_HDRS)
$(HOST_OBJ): $(CRT_HDRS)
