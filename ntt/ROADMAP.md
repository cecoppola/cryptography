[1;37m╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                      N T T   /   M I 3 0 0 A   —   R O A D M A P                                       ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat ROADMAP.md   or   less -R ROADMAP.md[0m

  Phase status, MI300A objectives, and known open work. Forward-
  looking only — resolved sessions, edit history, and incident
  narratives are not tracked here (those live in git history).

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  1. PHASE STATUS[0m

  ┌───────┬──────────────────────────┬────────────────────────┐
  │ [1;36mPhase[0m │ [1;36mName[0m                     │ [1;36mState[0m                  │
  ├───────┼──────────────────────────┼────────────────────────┤
  │ 1     │ Design & Plan            │ [1;32mdone[0m                   │
  │ 2     │ CPU Reference            │ [1;32mdone[0m                   │
  │ 3     │ GPU Kernel (gfx1030 dev) │ [1;32mdone[0m                   │
  │ 4     │ MI300A Tuning & Deploy   │ [1;33mprep done; deploy open[0m │
  └───────┴──────────────────────────┴────────────────────────┘

  Phase 4 remaining work is gfx942 runtime only — all CPU/dev-side
  preparation is complete (cross-compile clean for gfx942 on every TU
  except the oshcc/SHMEM final link, which is MI300A-cluster-only).

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  2. CURRENT CAPABILITY[0m

  - Single-prime NTT (ref): CPU reference + GPU kernels for 14 curated
    primes; 11 are GPU-eligible (q < 2^32). Cyclic, negacyclic, ML-KEM.
  - 4-prime CRT-NTT engine (lib): full pipeline cross-compiles clean for
    gfx942 (0 warnings); verified end-to-end on 6900 XT single-device
    (test.hip 22/22). 4-APU orchestration links only on MI300A.
  - compute_e (app): Euler's number to N digits via binary splitting;
    dispatches to lib above schoolbook threshold; byte-identical to OEIS
    A001113 at d=100 000 on 6900 XT (LB=64==112). d up to 2.1M (log_n=20)
    verified rc=0 in 53.34s (2026-06-14, through gpu_run.sh, no crash) after
    base_convert acceleration cut decimal conversion 49.98s -> 6.53s at d=1.1M
    (BASE_CONVERT_MUL_THRESHOLD=512 on GFX1030_LOCAL raises the schoolbook/NTT
    crossover to avoid per-dispatch PCIe sync). Calibration sweep 1.1M/1.5M/2.1M
    confirmed the timing model (estimate matched actual to 1.3%) and that the
    log_n=20 per-pass ring window is flip_done-safe. A full d=32M (log_n=24) run
    completed in 3.22 h, OEIS-correct (2026-06-30).
  - 3-factor recursive 4-step NTT (2026-06-29): extends the engine to log_n=24
    (~39M digits) past the old log_n=23 two-factor ceiling (LDS-bound 2^11 sub-
    transform). GMP-verified at log_n=24 — all 16,777,184 product limbs match (2
    seeds, 8.4M-limb operands); regression at log_n=20/22/23 clean. Gated at
    log_n==24 (NTT_MAX_LOGN=24 to enable); 18-23 use the unchanged 2-factor path.
    Validated in a full 32M-digit compute_e run (3.22 h, OEIS-correct, 2026-06-30).
    Also needed for MI300A (~log_n=28-29 on 128GB needs the recursive split).
  - Reliability: `make check` 21/21 host suites + ASAN/UBSan + non-vacuity
    proof + 90% coverage floor (GPU-free). GPU correctness is the GMP oracle
    (test_gmp_oracle, log_n=20/22/23/24) run via scripts/gpu_run.sh.

  Key performance milestones (full session narratives in ARCHIVE.md §A3):
  - NTT compute 287.8→112.7 ms at d=100k on 6900 XT (2.55× vs CT-DIT baseline).
    Wins: Stockham 4-step DFT fix (I6), tiled transpose (I7), block sizing
    (I8), xtranspose fusion (I13), Hadamard-fused INTT (I14).
  - NttMulState: pre-alloc buffers (A2/A3), GPU Garner (A1), concurrent
    NTT(f)||NTT(g) streams (A4), constexpr LDS stride (C1, 16912→4208 B).
  - I4 (2026-06-01): Montgomery 8-byte CT-DIT twiddles for odd log_n > 10;
    restores correct size instead of rounding up to even. GPU gate PASS 06-06
    (check-gpu 10/10, d=200k verified).
  - C2 (2026-06-01): __launch_bounds__(512, 4) on stockham_kernel*<LOG_N_SUB>
    where STOK_STRIDE_C ≤ 1024; compiler occupancy hint. PMC captured 06-06
    (X3_PMC_GFX1030_20260606); gfx942 confirmation pending (G6+).
  - I15 (2026-05-31): transposed scatter/gather eliminates 2 transpose_sq GPU
    launches per multiply. GPU gate PASS 06-06 (pipeline oracle 39/39, check-gpu R2).
  - GPU scatter/gather cores (2026-06-24): per-element scatter/gather logic
    (transfer_core.h) kernelized in transfer_kernels.hip and host-verified vs
    bigint_scatter[_t]/gather[_t] over random+adversarial inputs at LIMB_BITS
    64/112 (make test-transfer-core; non-vacuity confirmed; current total below).
    Cross-compiles clean for gfx942 + gfx1030 (-Wall -Wextra). Pipeline
    enablement and 4-APU/APU0 perf are MI300A-gated (G10/G11; the gated wire-in
    itself is done — see below): moves scatter (steps 1+2) and gather (step 8)
    off the CPU; raises the all-4-APU runtime fraction from ~80% toward ~93%
    (PERFORMANCE §"4-APU posture").
  - Parallel carry-lookahead gather (2026-06-24): the gather phase-2 carry sweep
    is reformulated as an O(log n) generate/propagate prefix (transfer_core.h
    cla_cell/cla_compose/cla_finalize) so the GPU phase-2 is not a single-thread
    scan. Host-verified vs the sequential normalizer AND bigint_gather, including
    deliberately constructed long-carry chains (generate + B-1 propagate runs)
    that random inputs never hit; non-vacuity confirmed for the monoid and the
    propagate flag. The multi-level blocked scan (local scan + aggregate scan +
    exclusive-prefix apply) is host-modeled recursively and verified == the
    single-pass scan at block sizes forcing deep recursion, and == the
    sequential normalization end-to-end (make test-transfer-core, 12456
    checks/width; apply-order and exclusive-vs-inclusive bugs both caught).
    Decompose/finalize + single-block LDS scan + multi-block (block-scan / apply
    / multi-level launcher) kernels added to transfer_kernels.hip (clean
    gfx942/gfx1030); two scan levels cover log_n<=20, a 3rd level for log_n>20
    is the only remaining GPU bring-up piece (launcher falls back to the
    verified sequential normalizer there).
  - End-to-end CRT-NTT multiply oracle (2026-06-24): host-only pure-C pipeline
    (real bigint_scatter -> independent radix-2 NTT convolution per prime, with
    self-discovered roots -> CRT combine -> real bigint_gather) validated vs GMP
    over random + all-max inputs at both LIMB_BITS (make test-e2e-oracle, 870
    checks/width; non-vacuity confirmed). Cross-checks the primes, NTT roots,
    CRT headroom, scatter/gather base, and cyclic=linear sizing on CPU before
    any GPU bring-up — independent of the GPU kernels. A second mode validates
    the negacyclic path (main.hip pipeline_negacyclic): c = a*b mod (X^n+1) via
    psi pre-twist -> length-n cyclic NTT -> psi^{-k} untwist (psi^n = -1), per
    prime + CRT, vs a GMP schoolbook negacyclic convolution; non-vacuity
    confirmed (disabled untwist AND pure-cyclic both caught at every size).
  - Warnings audit (2026-06-24): every ref/lib/app TU compiles clean (0
    warnings) under -Wall -Wextra for gfx942 + gfx1030. Fixed: a dangling-else
    that silently misbound an else in arith/ntt_mul.hip's delta diagnostic;
    dead make_ctdit_flat (+ stale comment, retired by I4) in test.hip; dead Rinv
    in test_modops.hip; a missing stok_block helper that left ntt_kernel_cswitch
    .hip (G9 experiment) uncompilable.
  - Sanitizer + strict-conversion pass (2026-06-24): the new host suites
    (transfer-core, e2e-oracle) added to scripts/test_asan.sh and run clean
    under ASan + UBSan at both LIMB_BITS; fixed a real mpz double-init leak in
    the e2e oracle's crt_setup. transfer_core.h (the GPU-bound scatter/gather/
    CLA arithmetic) and bigint.c scatter/gather are -Wconversion/-Wsign-
    conversion clean at both widths (one benign int->uint64 modulo made
    explicit) — no implicit narrowing in the index or carry math.
  - Gated GPU scatter/gather/CLA kernels (2026-06-24): the on-device scatter
    (steps 1+2) and gather+CLA (step 8) kernels live in transfer_kernels.hip
    (the example call sites were in the retired transfer_gpu.hip, now
    archive/lib_dead/; re-wiring into arith/ntt_mul.hip is the G10/G11 task)
    behind NTT_GPU_SCATTER_GATHER (default OFF; auto-off on GFX1030_LOCAL), with
    gated device scratch + launcher decls. Default build is byte-identical (CPU
    path unchanged); all four arch×flag combos compile clean (-Wall -Wextra) and
    the launcher signatures link-resolve against transfer_kernels.o (0 unresolved
    refs). A core-routed integration trial in test-e2e-oracle runs a full
    multiply through the exact kernel cores (scatter_value + gather_acc_limb +
    gather_carry_normalize) vs GMP (870 checks/width; non-vacuity confirmed) —
    proving the cores compose correctly in the integrated data path. Enabling on
    MI300A = flag + link transfer_kernels.o + managed operands; then measure.


[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  3. OPEN TASKS  —  MI300A (gfx942) HARDWARE REQUIRED[0m

  ┌────┬────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┐
  │ [1;36mID[0m │ [1;36mTask[0m                                                       │ [1;36mNotes[0m                                                      │
  ├────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ G2 │ Full 4-APU polymul correctness (CRT vs CPU bigint)         │ random inputs; verify CRT result exact                     │
  │ G6 │ rocprofv3 PMC on gfx942 (HBM/VALU/LDS; compare to gfx1030) │ expect ~3× CU + ~2× occupancy ratio                        │
  │ G7 │ Kernel-driven P2P benchmark N ≥ 2^23 (vs SDMA > 16 MB)     │ peer_copy_kernel planned (G7/G10); measure on 4-APU        │
  │ G8 │ Shared twiddle on APU 0 + peer access                      │ saves 3/4 HBM at N = 2^24                                  │
  │ G9 │ C++ templates vs pure-C switch(pidx): runtime perf         │ DECIDED 2026-06-29: keep templates (A/B: tmpl ~7% faster,  │
  │ G6+│ C2 occupancy confirmation (minBlocksPerCU=4 on stockham*)  │ STOK_STRIDE_C≤1024 → 4 blk/CU fits; PMC must confirm lift  │
  │ I16│ P0 Solinas wall-clock: per-prime hybrid (Solinas P0 only)  │ FIXED 2026-06-29: preproc PIDX bug; GMP-verified; gfx942 d │
  │ M1 │ oshcc/SHMEM crt_ntt final link                             │ gated on MI300A module load                                │
  │ G10│ GPU scatter: kernel + gated pipeline wiring done           │ enable NTT_GPU_SCATTER_GATHER + managed operands           │
  │ G11│ GPU gather+CLA: kernel + gated wiring done                 │ enable flag + link transfer_kernels.o; measure             │
  └────┴────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  4. MI300A ACCEPTANCE CRITERIA[0m

  Phase 4 is complete when, on a 4-APU MI300A node:

  - [ ] The engine + demo build clean (zero warnings) under PrgEnv-amd / ROCm 7.0.3:
        `cd lib && make all` (crt_ntt + test_ntt) and `cd app/compute_e && make`.
  - [ ] lib/test_ntt passes 22/22.
  - [ ] 4-APU CRT polymul (G2) matches CPU bigint reference over 200k
        random inputs including all-(p-1) edges.
  - [ ] compute_e d = 10^6 runs to completion across all 4 APUs and
        matches OEIS A001113.
  - [ ] PMC capture (G6) confirms VALU / VMEM / LDS within the
        scaled-from-gfx1030 envelope (PERFORMANCE §3).
  - [x] G9-FINAL runtime A/B between templates and pure-C switch
        recorded (2026-06-29) — DECISION: KEEP TEMPLATES; the switch path is rejected.
        Forward-Stockham NTT wall-clock A/B on gfx1030 @1300MHz (bench_g9.hip, prime 0,
        300 iters, hipEvent-timed): template<PIDX> 0.3317 ms/NTT vs cswitch 0.3589 ms at
        log_n=20 (+7.6%), and 0.01918 vs 0.02056 ms at log_n=14 (+6.7%). Templates ~7%
        faster, consistent across sizes — confirming the ISA evidence (compile-time
        prime immediates vs a runtime switch). (Upper bound on the pure-dispatch delta:
        ntt_kernel_cswitch.hip is a STALE port — 12 kernels vs the engine's 64, pre-I13
        so it uses a non-fused cross_twiddle_kernel_cs — which also handicaps it slightly;
        but the leaf butterflies dominate, so most of the gap is dispatch.) Migration is
        further argued against by the maintenance cost of a parallel switch-engine (the
        cswitch file already fell out of sync). Templates stay (c-style.md §Language).
        Harness kept at lib/bench_g9.hip (not in the build); cswitch retained as the
        retired G9 experiment.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  5. FUTURE WORK  (post-Phase-4, not blocking deploy)[0m

  ┌────┬────────────────────────────────────────────────────────────────────────────────┐
  │ [1;36m#[0m  │ [1;36mItem[0m                                                                           │
  ├────┼────────────────────────────────────────────────────────────────────────────────┤
  │ R1 │ Recalibrated perf-per-watt model with measured MI300A power                    │
  │ R2 │ Multi-node (> 1 K4) code path (P2P or SHMEM across nodes)                      │
  │ R3 │ [1;32mDONE 2026-05-28[0m hipHostMalloc pinned scatter bufs (NUMA-local HBM, DMA H2D)  │
  │ R4 │ gfx942 ISA diff vs gfx1030 — DONE 2026-05-30 (§I12-ISA in OPT_LEDGER + below)  │
  │ R5 │ MFMA path for an adjacent algorithm that fits (galois INT8 GEMM / BKZ buckets) │
  │ R6 │ [1;32mDONE 2026-05-28[0m per-call log_n from input sizes; stage-bucketed prefix valid  │
  │ R7 │ I11 rect Stockham re-enable for odd log_n > 10 (faster than CT-DIT-Mont I4)    │
  │    │   (a) DONE 2026-06-06: oracle PASS at all odd 11/13/15/17/19 vs CT-DIT-Mont.    │
  │    │   (b) DONE 2026-06-07: CT-DIT baseline verified on gfx1030:                    │
  │    │       d=200k  log_n=17  CT-DIT-Mont   1.45s   962 dispatches  PASS 06-06     │
  │    │       d=500k  log_n=18  Stockham       2.24s  1789 dispatches  PASS 06-06     │
  │    │       d=1M    log_n=19  CT-DIT-Mont   10.32s  3348 dispatches  PASS 06-07     │
  │    │   (c) PENDING clean-boot run 2026-06-07: I11 re-enabled; kernel bounds        │
  │    │       verified; 1000-iter stress PASS. Three crashes today caused by           │
  │    │       multi-tenant GPU (Chrome+Steam+ROCm) + lockup_timeout=0 + soft-reboot   │
  │    │       chain — NOT kernel defects. Requires: grub fix applied, full power       │
  │    │       cycle, Steam+Chrome closed, then run compute_e -d 1000000 -f.           │
  │ R8 │ I15 transposed-scatter reintroduction (perf). Removed from the main            │
  │    │   path during the F6 engine consolidation (2026-06-27) to retire the           │
  │    │   duplicate engine; ntt_mul.hip kept (correct to log_n=23 + crash-safe         │
  │    │   yield). STUDY + reintroduce I15 (transpose_sq elimination) as part of        │
  │    │   a later general optimization pass; verify via test_gmp_oracle_dev.           │
  └────┴────────────────────────────────────────────────────────────────────────────────┘

  [1;36mR8 DONE 2026-06-29 (opt-in; NOT a win on gfx1030):[0m I15 reintroduced behind
  -DNTT_I15 (default OFF; default build byte-identical). Input: scatter_kernel_t writes the
  operand TRANSPOSED (scatter_dispatch selects it via M15>0 on the even-log_n 4-step),
  eliminating the input transpose_sq. Output: garner_kernel_t reads residues from the
  transposed positions and writes d_out natural, folding the un-transpose into the Garner
  pass and eliminating the output transpose_sq (bigint_gather unchanged). GMP-VERIFIED at
  log_n=20 AND 22 with the flag; default unchanged (log_n=22/24 PASS). BUT it is a PERF
  REGRESSION on gfx1030: compute_e -d 500000 (log_n=18) split 6.33s with I15 vs 6.05s
  default (+4.5%). Cause: the transposed scatter/Garner do UNCOALESCED strided global
  access (adjacent threads stride by M), whereas the transpose_sq they replace is LDS-tiled
  (coalesced) — naive transpose-elimination costs more than it saves on RDNA2. KEEP DEFAULT
  OFF. To make it a win: LDS-tile the transposed scatter/Garner, OR confirm the benefit on
  MI300A unified HBM. The correctness-verified building blocks are in place for either.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  6. KNOWN GAPS  (off the critical path)[0m

  - perf/probes/gnfs_sieve.hip: FIXED 2026-05-30. Two issues corrected:
    (1) TILE_SIZE reduced from 65536 to 64512 so sieve[TILE_SIZE+1]
    fits in the 64 KB LDS limit. (2) Tier-2 atomicAdd(uint8_t*) →
    word-granularity atomicCAS(uint32_t*) read-modify-write loop (ROCm
    7 does not provide byte atomics). Both gfx1030 and gfx942 build
    clean (0 errors). Off the NTT critical path; not wired into the
    default build; sieve semantics unchanged.
  - ML-DSA benchmarking (ω = 3073009 = 1753^2 mod q cyclic, ψ = 1753 negacyclic):
    the dev-box bench-mldsa-* convenience targets were removed in the Makefile
    slimming (2026-06-30); run the ref binaries directly with those parameters.
  - Stockham 4-step log_n=16 bug (launch_stockham_large): root cause
    was (N1*N2)/1024 block count formula, correct only when N1=N2=1024.
    Fixed 2026-05-28: row pass uses N2 blocks, col pass uses N1 blocks.
    Production path (ntt_bigint_mul) parity dispatch re-enabled.
    Verification pending on MI300A (G2 task).
