# =============================================================================
# Makefile — NTT / MI300A project (root)
#
# Builds the ref/ reference layer (single-prime CPU NTT + gfx942 parity GPU
# kernels) and orchestrates the host reliability gate and the gfx942 cross-
# compile proof. lib/ and app/ have their own Makefiles; this root delegates
# to them where noted (all-mi300a, clean).
#
#   make check        host reliability gate (CPU-only; runs scripts/check.sh)
#   make all-mi300a   gfx942 cross-compile proof (ref + lib device TUs + app)
#   make cpu-all      build+bench every ref CPU binary (scripts/cpu_testbench.sh)
#   make all          build the ref CPU binaries
#   make clean        clean root bin/ + lib + app   (clean-host: root bin/ only)
#
# Design/API: ARCHITECTURE.md   Baseline: PERFORMANCE.md   Tasks: ROADMAP.md
# MI300A build: module load PrgEnv-amd/8.6.0 rocm/7.0.3 craype-accel-amd-gfx942
# =============================================================================

# ── Compilers / flags ─────────────────────────────────────────────────────────
HIPCC  ?= hipcc
CC      = $(HIPCC)            # unify on hipcc for single-source heterogeneous build
HOSTCC ?= cc                 # pure-C host tests (build on a CPU-only box)
CFLAGS   ?=
HIPFLAGS ?=
ALL_CFLAGS   := -O2 -Wall -Wextra $(CFLAGS)
ALL_HIPFLAGS := -O2 -Wall -Wextra $(HIPFLAGS)
HOSTLINK      = $(HOSTCC) -O2 -Wall -Wextra $(CFLAGS)
ARCH_MI300A  := --offload-arch=gfx942 -DMFMA_TARGET=1   # MFMA_TARGET gates CDNA mfma intrinsics

# ── Layout ────────────────────────────────────────────────────────────────────
S := ref
B := bin
# ntt.h #includes ntt_moduli.h, so every C TU transitively depends on it —
# it MUST be a prerequisite or a prime-table change silently skips the rebuild.
HDR := $(S)/ntt.h $(S)/ntt_moduli.h

# ── Output binaries ───────────────────────────────────────────────────────────
CPU_BIN     := $(B)/ntt_cpu
STOK_BIN    := $(B)/ntt_stockham
BENCH_BIN   := $(B)/ntt_bench
POLYMUL_BIN := $(B)/ntt_polymul
NEGACYC_BIN := $(B)/ntt_polymul_negacyclic
MLKEM_BIN   := $(B)/ntt_mlkem
CPU_BINS    := $(CPU_BIN) $(STOK_BIN) $(BENCH_BIN) $(POLYMUL_BIN) $(NEGACYC_BIN) $(MLKEM_BIN)

MI300A_BINS := $(B)/ntt_gpu_mi300a $(B)/ntt_gpu_stockham_mi300a \
               $(B)/ntt_gpu_polymul_mi300a $(B)/ntt_cross_verify_mi300a \
               $(B)/ntt_gpu_determinism_mi300a

.PHONY: all all-mi300a cpu stockham bench polymul negacyclic mlkem \
        gpu-mi300a gpu-stok-mi300a gpu-polymul-mi300a verify-mi300a det-mi300a \
        hw_probe hw_probe-dev \
        test-curated test-ntt-rigor test-ntt-rigor-stok test-polymul-integ \
        test-negacyc-integ test-mlkem-kat test-mont-correctness test-params-boundary \
        verify-nonvacuous coverage coverage-all test-asan check cpu-all \
        clean clean-host

all: $(CPU_BINS)

$(B):
	mkdir -p $(B)

# ── all-mi300a: gfx942 cross-compile proof ────────────────────────────────────
# Cross-compiles every gfx942 TU (ref GPU binaries + lib device .o + app binary)
# and logs it; the MI300A operator diffs the log against their first PrgEnv-amd
# build to spot environment drift. (crt_ntt link needs oshcc/SHMEM, MI300A-only.)
all-mi300a:
	@mkdir -p results
	@LOG=results/all_mi300a_$$(date +%Y%m%d_%H%M%S).log; \
	  : > $$LOG; \
	  printf '\n\033[1;37m═══  all-mi300a — gfx942 cross-compile proof  ═══\033[0m\n\n' | tee -a $$LOG; \
	  printf '  log: %s\n\n' $$LOG; \
	  set -e; \
	  printf '  ref mi300a binaries ...\n' | tee -a $$LOG; \
	  $(MAKE) $(MI300A_BINS) >>$$LOG 2>&1; \
	  printf '  lib device TUs (gfx942 .o, SHMEM link skipped) ...\n' | tee -a $$LOG; \
	  for tu in ntt_kernel garner main; do \
	    $(HIPCC) -O3 -Wall -Wextra $(ARCH_MI300A) -Ilib -c lib/$$tu.hip \
	        -o $(B)/$${tu}_mi300a.o >>$$LOG 2>&1 \
	      && printf '    BUILD OK: lib/%s.hip\n' $$tu | tee -a $$LOG; \
	  done; \
	  $(HIPCC) -O3 -Wall -Wextra $(ARCH_MI300A) -Ilib -Ilib/arith \
	      -c lib/arith/ntt_mul.hip -o $(B)/ntt_mul_mi300a.o >>$$LOG 2>&1 \
	    && printf '    BUILD OK: lib/arith/ntt_mul.hip\n' | tee -a $$LOG; \
	  printf '  app/compute_e gfx942 ...\n' | tee -a $$LOG; \
	  $(MAKE) -C app/compute_e all >>$$LOG 2>&1 \
	    && printf '    BUILD OK: app/compute_e/compute_e\n' | tee -a $$LOG; \
	  printf '\n  \033[1;32mall-mi300a: every gfx942 TU built clean\033[0m\n' | tee -a $$LOG; \
	  printf '  log saved: %s\n\n' $$LOG | tee -a $$LOG

# ── ref CPU binaries  (static pattern: bin/ntt_X <- ref/ntt_X.c) ──────────────
$(CPU_BINS): $(B)/%: $(S)/%.c $(HDR) | $(B)
	$(CC) $(ALL_CFLAGS) -o $@ $<
	@printf '  BUILD OK: %s\n' $@

cpu: $(CPU_BIN)
stockham: $(STOK_BIN)
bench: $(BENCH_BIN)
polymul: $(POLYMUL_BIN)
negacyclic: $(NEGACYC_BIN)
mlkem: $(MLKEM_BIN)

# ── ref gfx942 GPU binaries  (static pattern: bin/ntt_X_mi300a <- ref/ntt_X.hip) ─
$(MI300A_BINS): $(B)/%_mi300a: $(S)/%.hip | $(B)
	$(HIPCC) $(ALL_HIPFLAGS) $(ARCH_MI300A) -o $@ $<
	@printf '  BUILD OK: %s (gfx942)\n' $@

gpu-mi300a:         $(B)/ntt_gpu_mi300a
gpu-stok-mi300a:    $(B)/ntt_gpu_stockham_mi300a
gpu-polymul-mi300a: $(B)/ntt_gpu_polymul_mi300a
verify-mi300a:      $(B)/ntt_cross_verify_mi300a
det-mi300a:         $(B)/ntt_gpu_determinism_mi300a

# ── hw_probe: hardware diagnostic (DEPLOYMENT_DIAGNOSTIC.md Appendix C) ────────
# Build gfx942 to run on the MI300A; build -dev (gfx1030) to run on the dev box.
hw_probe: app/hw_probe.hip | $(B)
	$(HIPCC) $(ALL_HIPFLAGS) $(ARCH_MI300A) -o $(B)/hw_probe app/hw_probe.hip
	@printf '  BUILD OK: %s (gfx942 — run on MI300A via srun)\n' $(B)/hw_probe
hw_probe-dev: app/hw_probe.hip | $(B)
	$(HIPCC) $(ALL_HIPFLAGS) --offload-arch=gfx1030 -o $(B)/hw_probe_dev app/hw_probe.hip
	@printf '  BUILD OK: %s (gfx1030 — runs on the dev box)\n' $(B)/hw_probe_dev

# ── ref host test suite  (pure C, GPU-free; rolled up by `make check`) ────────
# Each suite links its subject TU with -D<TU>_NO_MAIN; HOSTCC=cc builds on a
# CPU-only box. test_ntt_rigor doubles as a Stockham differential (rigor-stok).
# test_polymul_integ doubles as the negacyclic integ via -DINTEG_NEGACYCLIC.
test-curated: $(B)/test_curated_primes ; $<
$(B)/test_curated_primes: $(S)/test_curated_primes.c $(HDR) $(S)/ntt_cpu.c | $(B)
	$(HOSTLINK) -DNTT_CPU_NO_MAIN -o $@ $(S)/test_curated_primes.c $(S)/ntt_cpu.c

test-ntt-rigor: $(B)/test_ntt_rigor ; $<
$(B)/test_ntt_rigor: $(S)/test_ntt_rigor.c $(HDR) $(S)/ntt_cpu.c | $(B)
	$(HOSTLINK) -DNTT_CPU_NO_MAIN -o $@ $(S)/test_ntt_rigor.c $(S)/ntt_cpu.c

test-ntt-rigor-stok: $(B)/test_ntt_rigor_stok ; $<
$(B)/test_ntt_rigor_stok: $(S)/test_ntt_rigor.c $(HDR) $(S)/ntt_stockham.c | $(B)
	$(HOSTLINK) -DNTT_STOCKHAM_NO_MAIN -o $@ $(S)/test_ntt_rigor.c $(S)/ntt_stockham.c

test-polymul-integ: $(B)/test_polymul_integ ; $<
$(B)/test_polymul_integ: $(S)/test_polymul_integ.c $(HDR) $(S)/ntt_polymul.c | $(B)
	$(HOSTLINK) -DNTT_POLYMUL_NO_MAIN -o $@ $(S)/test_polymul_integ.c $(S)/ntt_polymul.c

test-negacyc-integ: $(B)/test_negacyc_integ ; $<
$(B)/test_negacyc_integ: $(S)/test_polymul_integ.c $(HDR) $(S)/ntt_polymul_negacyclic.c | $(B)
	$(HOSTLINK) -DINTEG_NEGACYCLIC -DNTT_NEGACYC_NO_MAIN -o $@ $(S)/test_polymul_integ.c $(S)/ntt_polymul_negacyclic.c

test-mlkem-kat: $(B)/test_mlkem_kat ; $<
$(B)/test_mlkem_kat: $(S)/test_mlkem_fips203_kat.c $(S)/ntt_mlkem.c | $(B)
	$(HOSTLINK) -DNTT_MLKEM_NO_MAIN -o $@ $(S)/test_mlkem_fips203_kat.c $(S)/ntt_mlkem.c

test-mont-correctness: $(B)/test_mont_correctness ; $<
$(B)/test_mont_correctness: $(S)/test_mont_correctness.c $(HDR) $(S)/ntt_cpu.c | $(B)
	$(HOSTLINK) -DNTT_CPU_NO_MAIN -o $@ $(S)/test_mont_correctness.c $(S)/ntt_cpu.c

# params-boundary: one binary per subject TU (q<2 SIGFPE + lazy-overflow guards)
PB_SRC := $(S)/test_params_boundary.c
test-params-boundary: $(B)/test_params_boundary_cpu $(B)/test_params_boundary_stok \
                      $(B)/test_params_boundary_pm $(B)/test_params_boundary_neg \
                      $(B)/test_params_boundary_bench
	$(B)/test_params_boundary_cpu && $(B)/test_params_boundary_stok && \
	$(B)/test_params_boundary_pm && $(B)/test_params_boundary_neg && \
	$(B)/test_params_boundary_bench
$(B)/test_params_boundary_cpu: $(PB_SRC) $(HDR) $(S)/ntt_cpu.c | $(B)
	$(HOSTLINK) -DNTT_CPU_NO_MAIN -DSUBJECT='"ntt_cpu"' -DSUBJECT_HAS_LAZY=0 -o $@ $(PB_SRC) $(S)/ntt_cpu.c
$(B)/test_params_boundary_stok: $(PB_SRC) $(HDR) $(S)/ntt_stockham.c | $(B)
	$(HOSTLINK) -DNTT_STOCKHAM_NO_MAIN -DSUBJECT='"ntt_stockham"' -DSUBJECT_HAS_LAZY=0 -o $@ $(PB_SRC) $(S)/ntt_stockham.c
$(B)/test_params_boundary_pm: $(PB_SRC) $(HDR) $(S)/ntt_polymul.c | $(B)
	$(HOSTLINK) -DNTT_POLYMUL_NO_MAIN -DSUBJECT='"ntt_polymul"' -DSUBJECT_HAS_LAZY=1 -o $@ $(PB_SRC) $(S)/ntt_polymul.c
$(B)/test_params_boundary_neg: $(PB_SRC) $(HDR) $(S)/ntt_polymul_negacyclic.c | $(B)
	$(HOSTLINK) -DNTT_NEGACYC_NO_MAIN -DSUBJECT='"ntt_negacyc"' -DSUBJECT_HAS_LAZY=1 -o $@ $(PB_SRC) $(S)/ntt_polymul_negacyclic.c
$(B)/test_params_boundary_bench: $(PB_SRC) $(HDR) $(S)/ntt_bench.c | $(B)
	$(HOSTLINK) -DNTT_BENCH_NO_MAIN -DSUBJECT='"ntt_bench"' -DSUBJECT_HAS_LAZY=0 -o $@ $(PB_SRC) $(S)/ntt_bench.c

# ── Host gates + coverage (delegate to scripts/) ──────────────────────────────
verify-nonvacuous: ; bash scripts/verify_nonvacuous.sh
coverage:          ; bash scripts/coverage.sh
coverage-all:      ; bash scripts/coverage_full.sh
test-asan:         ; bash scripts/test_asan.sh
check:             ; bash scripts/check.sh
cpu-all:           ; bash scripts/cpu_testbench.sh

# ── Clean ─────────────────────────────────────────────────────────────────────
# clean-host: root bin/ only — does NOT recurse into lib/app, so a running GPU
# session's in-place binaries survive a host-only `make check` (coverage.sh +
# test_asan.sh use this). 2026-05-23 lesson: full clean wiped GPU binaries mid-run.
clean-host:
	rm -rf $(B) .rocprofv3 __pycache__
	rm -f bench_*.txt bench_gpu_*.txt det_*.txt curated_primes_*.txt negac_sweep_*.txt \
	      mlkem_kat_*.txt negacyc_integ_*.txt ntt_rigor_*.txt polymul_integ_*.txt
	@printf '  Cleaned (root only; lib/app untouched).\n'

clean: clean-host
	-$(MAKE) -C lib clean >/dev/null 2>&1
	-$(MAKE) -C app/compute_e clean >/dev/null 2>&1
	@printf '  Cleaned (root + lib + app/compute_e).\n'
