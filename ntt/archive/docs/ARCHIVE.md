[1;37m╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                      N T T   /   M I 3 0 0 A   —   A R C H I V E                                       ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat ARCHIVE.md   or   less -R ARCHIVE.md[0m

  Historical content removed from the active doc set during the
  2026-05-24 forward-looking restructuring. Kept in one place so the
  user can relocate it outside the repo. Nothing here is needed to
  build, run, or extend the project — git history is the canonical
  record. Sections below were sourced from STATUS.md, RESEARCH.md
  (§2, §3, §8), REFERENCE.md (§5–§10 of the Tests-and-Benchmarks
  append + §10 dated measurements), and the perf/results/*.md set.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  A1. RESOLVED DEVIATIONS  (STATUS.md history, chronological)[0m

  - 2026-05-16 to 05-19: 3 reducer/REDC bugs found by brute-force
    tests, exhaustively re-verified by lib2/test_reduce.c +
    test_modops.hip. Full narrative in PROJECT_BACKLOG §6–§9.
  - 3 silently-wrong-NTT: ntt_moduli.h Solinas-60 g=3→10 and q=7681
    g=3→17 (non-primitive roots); lib2/main.hip make_tw N/2→N-1
    twiddle heap-overflow. Fixed + verified (test-curated, full 14-
    prime table re-verified, 0-warn gfx942/gfx1030 cross-compile).
  - 5 vacuous-test / SIGFPE: lib1 GPU params_init lacked primitive-
    root check (4 files) + q = 0 SIGFPE → added root_is_primitive()
    and q guard. Prior T1/U1/U2 results valid (legit configs still pass).
  - 3 Makefile stale-object holes (root / lib2 / lib3): closed via
    header-closure deps, rebuild-proven. scripts/cpu_testbench exit-
    blindness fixed. gpu_run.sh S9 stderr-swallow + S7/S15 latents
    fixed. KA4/KA5 + mul_ref test refs corrected. compute_e gate,
    newton-doc, nodiscard + stale comments, dead ntt_*_mont decls
    — all fixed + 0-warn.
  - 2026-05-19 advisory bigint.h comments corrected: shl cap
    (bits + LIMB_BITS - 1) / LIMB_BITS; gather base 2^(LIMB_BITS·i);
    limbs_for_digits / 64 documented as a deliberate conservative
    bound (unused); print_hex documented as a 64-bit-word debug helper.
    Comment-only; test-arith-dual still 56/56 both widths.
  - 2026-05-19 / 05-21 test-suite hardening + repo consolidation:
    NEW host suites — test-ntt-rigor[-stok] (156/156 each, all-14-prime
    adversarial vs independent O(n²) __int128 DFT), test-polymul-integ
    / test-negacyc-integ (198/198, 192/192 + 2 math-skips), test-mlkem-
    kat (FIPS-203 KAT 6/6), test-params-boundary (5 subject TUs),
    lib2 test-arith-fuzz (12k random trials, both LIMB_BITS) +
    test-arith-gmp (GMP independent oracle, 1200 checks), lib3
    test-binsplit (26/26) + test-host (compute_e vs OEIS digits of e).
    Every new suite proven non-vacuous by fault-injection. FOUND+FIXED:
    negacyclic 2n-root gating treated a math-impossible cell as FAIL
    → now a skip; q ≥ 2 SIGFPE guard generalized to all 5 lib1
    ntt_params_init; ntt_stockham full per-butterfly reduction.
    Reliability infra: `make check` unified gate, scripts/ci.sh +
    .github/workflows/ci.yml, coverage floor 60→90%, permanent
    `make test-asan`. Repo tidy: root scripts → scripts/, Makefile
    HOSTLINK factored, lib1/lib3 READMEs added, README/CLAUDE/Makefile-
    header drift fixed.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  A2. INCIDENTS  (perf/crash_diag/INCIDENT.md, summarized)[0m

  Seven incidents, all root-caused and mitigated in code:
  §1–§4   2026-04-XX  early gpu_run.sh hardening (S1–S5: rocm-smi
                       health gate, INT→TERM→KILL escalation, per-job
                       timeout).
  §5      2026-05-17  hard host lock during the J3 sweep — log-silent
                       RDNA2 amdgpu hard-hang. New safeguards: S6 post-
                       idle cooldown; S7 log_n cap (def 16); S8 compute_e
                       gate 24→22; S9 flock GPU mutex; S10 pre-check
                       GPU% / VRAM% / stray-proc; S11 adaptive cooldown
                       (≥6 / 180 s × 3); S13 persistent ledger
                       perf/crash_diag/gpu_run_ledger.log; S15 crash-
                       aware guard.
  §6      2026-05-17  S9 bugfix: `exec 9>"$LOCK" 2>/dev/null` (no cmd)
                       made stderr→/dev/null permanent, hiding ALL
                       wrapped GPU stderr (errors + [ntt_prof]) until
                       2026-05-17; fixed to `{ exec 9>"$LOCK"; }
                       2>/dev/null` (scoped).
  §7      2026-05-23  rocprofv3 against bin/ntt_gpu_polymul_* on RDNA2
                       is unsafe unless --bench-only is passed (skips
                       the binary's hardcoded selftest + 4×4 negacyclic
                       sweep). gpu_run.sh enforces this via S16 (refuses
                       unfiltered profilers) and S17 (refuses rocprofv3
                       against polymul without --bench-only).

  All safeguards live in scripts/gpu_run.sh; the header there is the
  current authoritative list.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  A3. SESSION CAMPAIGNS  (STATUS.md "history" block, condensed)[0m

  - 2026-05-17 6900XT campaign J0–J6 + audits T1–T6 / F-PROF /
    U1 / U2; F.7 / F.8 closed; INCIDENT §5 host-lock + S6–S15
    safeguards. → PROJECT_BACKLOG §6–§8, INCIDENT, COMPUTE_E_6900XT,
    MEGA_REPORT.
  - 2026-05-18 exhaustive G1–G4 line-by-line audit; 3 silently-wrong-
    NTT + 5 SIGFPE/vacuous + Makefile holes fixed; CRT plan promoted
    to RESEARCH §7 (now ARCHITECTURE §4). → PROJECT_BACKLOG §9.
  - 2026-05-19 / 05-21 test-suite hardening + repo consolidation.
    → PROJECT_BACKLOG §10.
  - 2026-05-23 6900XT re-verification + bench/ISA/probe refresh
    (Phase 0-4): all post-2026-05-17 code changes GPU-re-verified
    green; G9-PREP gfx1030 ISA A/B captured. → PROJECT_BACKLOG §11.
  - 2026-05-23 final perf + stress data (Groups A–D): cross-verify
    extended to n = 8192 / 16384 (12/12 cells, 84/84 sub); 43-cell
    CT-DIT / Stockham / polymul bench grid + compute_e d-scaling
    (124/132 + 8/8); rocprofv3 PMC refresh (C1 stockham + C2 polymul
    --bench-only); D1 long-soak determinism @ rep=200 9/9; D2
    thermal/power telemetry (peak 38°C / 48W, no throttling). Two GPU
    hangs during rocprofv3 + polymul work, root-caused and mitigated
    via S16 / S17 + --bench-only binary flag. → PROJECT_BACKLOG §12,
    INCIDENT §7.
  - 2026-05-24 doc handoff restructuring: STATUS / REFERENCE / RESEARCH
    consolidated → README + ARCHITECTURE + PERFORMANCE + ROADMAP;
    everything chronological moved here.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  A4. PHASE F  (LIMB_BITS migration: 64 → 112)[0m

  Spec lived in perf/results/PHASE_F_DESIGN.md; bench in PHASE_F_BENCH.md.
  Outcome retained in ARCHITECTURE §4.2 (headroom math) and PERFORMANCE
  §2 (compute_e d-scaling table). Chronology, kept here only:

  F.1–F.6b  bigint / multiply / newton dual-build limb-generic
            (LIMB_BITS 64/112); test_arith 56/56 at BOTH widths
            including direct test_gather (closes the LB=112 limb_pack
            gap — gather was GPU-path-only); base_convert.c audited
            generic; ntt_mul.hip 8-step HIP_CHECK-hardened
            log_n ∈ [3, 22].

  F.7       lib3 dual-build (recursive-make f7-build →
            compute_e_dev_l64 / l112 + f7-polymul112, mirrors lib2).
            F.7a compute_e byte-identical e at LB=64 ≡ LB=112 for
            d = 100..100 000. F.7b test_polymul_sweep 726/726 + 7/7 at
            LB=112 AND LB=64 — its mul_ref was 64-bit-hardcoded
            (base-2^64 schoolbook); rewritten limb-generic via
            bigint.h mul_limb / LIMB_MASK (LB=64 regression preserved).

  F.8       LB=112 up to 1.48× faster (fewer limbs → smaller NTT),
            byte-identical; LB=64 kept as oracle.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  A5. G9 PREP  (C++ templates vs pure-C switch)[0m

  Detail lived in perf/results/G9_CPP_VS_C.md. Outcome: keep templates
  pending the gfx942 runtime A/B (G9-FINAL, in ROADMAP §3).

  gfx1030 ISA A/B (cross-compile, 2026-05-23):
    templates: 425 KB / 45 kernels / 101 s_cbranch / max 19 SGPR / max 28 VGPR
    cswitch:   169 KB / 12 kernels / 194 s_cbranch / max 45 SGPR / max 26 VGPR

  Pure-C lib2/ntt_kernel_cswitch.hip authored; gfx942 --save-temps ISA
  diff favours keeping templates (PRIMES[] loads avoided; +18 SGPR
  branch tree added inside butterfly; ~1.7× larger kernel body). FINAL
  decision still gated on gfx942 runtime perf A/B on MI300A hardware.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  A6. RETIRED LIB1 VARIANTS  (10 algorithm studies)[0m

  Removed from the build / test set in the 2026-05-17 consolidation
  after each was either subsumed by the Stockham / Shoup choice in
  lib2 or proved a negative result. Results in MEGA_REPORT.md; sources
  not retained.

    ntt_mont        — Montgomery REDC butterfly variant
    ntt_lazy        — lazy/deferred reduction CPU variant
    ntt_plantard    — Plantard reduction variant
    ntt_shoup       — Shoup pre-mul CPU variant (kernel retained in lib2)
    ntt_gs_dif      — Gentleman-Sande DIF variant
    ntt_radix4      — radix-4 butterfly variant
    ntt_goldilocks  — q = 2^64 - 2^32 + 1 specialised reductions
    ntt_fourstep    — four-step matrix NTT (CPU)
    ntt_fourstep_gpu — four-step on GPU (subsumed by lib2/ntt_kernel_fourstep)
    ntt_shmem       — multi-process shared-memory NTT (replaced by lib2 SHMEM)

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  A7. CAMPAIGN BENCHMARK GRID  (perf/results/MEGA_REPORT.md)[0m

  The 14-task Tier A–E benchmark campaign on gfx1030. Headline tier
  table preserved here; full per-tier writeups are in git history.

  ┌───────────────────────────────┬─────────────────────────────────────────────────┐
  │ [1;36mTier[0m                          │ [1;36mCoverage[0m                                        │
  ├───────────────────────────────┼─────────────────────────────────────────────────┤
  │ A1 — n × q sweep              │ 77 (n,q) cells × 5 Stockham variants = 361 rows │
  │ A2 — Batch scaling            │ 16 batch sizes × 3 n values, q = 998244353      │
  │ A3 — Four-step large-N        │ 7 n × 4 moduli; n = 16384 … 1048576             │
  │ B1 — Random-seed stress       │ 200 invocations × 6 (n,q,ω) triples             │
  │ B2 — Thermal soak             │ 900 s sustained NTT load, 1 Hz rocm-smi         │
  │ C1 — Full PMC sweep           │ VALU / SALU / VMEM / LDS counters per kernel    │
  │ C2 — HIP-trace timelines      │ ns-resolution per launch / memcpy / sync        │
  │ C3 — Roofline                 │ AI vs (compute-peak / mem-peak)                 │
  │ D1 — GPU-vs-CPU crossover     │ same n, q on Stockham CPU and GPU fused/batched │
  │ D2 — Batched polymul          │ fused_polymul_batched_kernel, grid = batch      │
  │ D3 — Memcpy vs compute split  │ parsed from C2 trace                            │
  │ D4 — Block-size sweep         │ 5 block sizes × 3 n                             │
  │ E1 — n = 4096 hybrid          │ separate / fused / batched_D2 paths             │
  │ E2 — Cross-verify all q < 2³² │ 11 moduli × 4 n × 7 tests                       │
  └───────────────────────────────┴─────────────────────────────────────────────────┘

  Post-G7 follow-ups (gfx1030 session 2026-04-19/20):
    G1 Cross-verify CT-DIT + Stockham GPU vs CPU ......... 7/7 PASS
    G2 GPU selftest fourstep kernel ...................... PASS at n = 65536, 2²⁰
    G3 LDS bank-conflict avoidance via XOR swizzle ....... 5.07× @ n=256, 1.83× @ 2048
    G4 Multi-stage LDS kernel fusion (Merge-NTT) ......... 4.99× vs CT-DIT
    G5 Thread coarsening / register blocking ............. done (negative result)
    G6 (skipped — superseded by D2 batched path)
    G7 Fused NTT+Hadamard+INTT polymul single kernel ..... 9.80× vs separate launch

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  A8. CLOSED INVESTIGATIONS  (RESEARCH §2)[0m

  A1 — Atomic-reduction audit (perf/results/framework/A1_REDUCTION_AUDIT.md):
       every atomicAdd in ntt/bkz/galois audited; none matched the
       per-thread → single-counter anti-pattern (8.91×). No code change.
  A2 — MFMA applicability (perf/results/framework/A2_MFMA_EVALUATION.md):
       MFMA outputs fp32 / fp64 / int32 without modular reduction;
       24-bit fp32 mantissa cannot hold products for q ≥ 14 bits.
       Modular NTT cannot use MFMA. Adjacent algorithms (BKZ INT8, FP
       conv) can — out of scope. Probe at src/mfma_probe.hip mirror.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  A9. GNFS / RSA-1024 FEASIBILITY  (RESEARCH §8)[0m

  Exploratory analysis for perf/probes/gnfs_sieve.hip — a line-sieve
  probe that is NOT wired into the build and NOT on the NTT critical
  path. The full ~530-line L-function derivation, tuning guide, and
  per-step analysis were intentionally not carried into the canonical
  docs; the probe header is the live artifact.

  Headline (1024-bit sieve, central estimate; node = 4 MI300A APUs):
    1 024 MI300As ≈ 1.6 yr · 2 048 ≈ 9.7 mo · 4 096 ≈ 4.9 mo · 8 192 ≈ 73 d.
    Matrix step adds ~5–15%. GNFS scaling: RSA-250 → 1024-bit ~ 194×.

  Complementarity: NTT is compute-bound (INT64 modmul); GNFS sieve is
  HBM-atomic-bound — they stress disjoint MI300A subsystems.
  Citations: Boudot CRYPTO 2020 (RSA-250); Kleinjung CRYPTO 2010
  (RSA-768); Lenstra ASIACRYPT 2003; Bernstein CHES 2014; Peng
  arXiv:2508.12743 (MI300A mem).

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  A10. EARLY UNIT TEST CHECKLISTS  (REFERENCE Tests append)[0m

  Retained for historical interest; current authoritative test status
  comes from `make check` (16 host suites + ASAN/UBSan + non-vacuity
  proof + 90% coverage floor) and `make check-gpu`.

  U1–U9 (ntt_cpu.c, CPU reference): mod_pow / mod_inv / twiddle len /
        bit_reverse_perm / delta / all-ones / round-trips ML-KEM and
        ML-DSA — all PASS.
  S1–S6 (ntt_stockham.c): round-trip vs CT-DIT, impulse, both ML-KEM
        and ML-DSA — all PASS. Benchmark (5950X): ~241k NTT/s @ n=256.
  P1–P8 (ntt_polymul.c): identity, commutativity, associativity,
        X^(n/2)² ≡ X^n ≡ 1, schoolbook diff, throughput — all PASS.
  GPU (6900 XT, 2026-04-26 re-validation): ntt_gpu / ntt_gpu_stockham
        / ntt_gpu_polymul / ntt_gpu_fourstep / ntt_cross_verify —
        22/22 PASS.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

  [1;33mEnd of archive. Move this file outside the repo when ready.[0m

  [1;35m  NOTE 2026-06-01:[0m GPU_SESSION_PLAN.md and GPU_TASKS_6900XT.md
  (untracked) documented the completed 2026-05-29 6900XT session.
  Content consolidated into ROADMAP.md §2. Both files deleted 2026-06-01.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  A3. ROADMAP §2 SESSION NARRATIVES  (archived 2026-06-01)[0m

  The following entries were removed from ROADMAP.md §2 to keep that
  section concise. See git log for full diffs per session.

  MI300A-targeted GPU work (2026-05-25):
    GPU Garner kernel replaces 96-thread CPU OpenMP Garner; GPU scatter
    kernel replaces 4-thread CPU OpenMP scatter; zero-copy via GFX1030_LOCAL
    compile gate eliminates the d_out→h_out D2H on MI300A. Measured ~30%
    speedup at compute_e d=10 000 on the 6900XT alone; unified-HBM zero-copy
    win activates only on MI300A. See ARCHITECTURE §4.4b/§4.4c.

  5950X CPU work (2026-05-28):
    Stockham log_n=16 block-count bug fixed in launch_stockham_large/intt
    (was (N1*N2)/1024; now N2/N1 per pass); K4 topology parameterized via
    g_n_devs; ntt_bigint_mul R6 per-call log_n; R3 hipHostMalloc pinned
    scatter buffers; odd-log2_n coverage gap in ntt_stockham.c closed;
    Montgomery NTT correctness test added (33/33 PASS over 11 primes × 3
    sizes). make check 17/17.

  6900XT GPU session (2026-05-29):
    Shoup 32-bit precomputed-quotient butterfly in lib1/ntt_gpu_stockham.hip
    (3.6× speedup: n=256 32k→197k NTT/s); PMC compute-bound (4.2% VMEM);
    R8 determinism 11 primes (33/33 PASS); compute_e d=100,000 measured
    (0.45s); large-n baselines (n=16384 ~19k, n=65536 ~13k NTT/s); G9 ISA
    preliminary (templates 2.4× fewer branches vs cswitch). check-gpu 10/10.

  compute_e GPU-time safety gate (2026-05-29):
    compute_e -d 1000000 crashed the dev box (gfx1030 display TDR at 10 s).
    main.c now refuses runs > 8 s unless -f/--force. MI300A has no such limit.

  Stockham 4-step DFT fix + production swap (2026-05-29, MEASURED WIN):
    The lib2 4-step Stockham was NOT a DFT (both passes DFT'd the low index).
    FIXED with Bailey transpose structure. NTT compute 287.8→154.0 ms (1.87x)
    at d=100k on gfx1030; cumulative path to 112.7 ms (2.55x). Signed off:
    make check 17/17 + check-gpu 10/10 PASS.

  4-step kernel fusion (2026-05-30, MEASURED WIN):
    I13 fuses transpose+cross-twiddle; I14 folds Hadamard into INTT first
    load. NTT compute 126.4→112.7 ms (-11%). I12 (Solinas) REVERTED (+3.8%
    RDNA3). gfx942 ISA: Solinas P0 -9 VALU, P1 +9, P2 +17, P3 +17 → I16 proposed.

  GPU crash root cause fixed (2026-05-29):
    gfx1030 display TDR. (a) gpu_run.sh clamps to 8 s; (b) kernel cmdline
    raises compute TDR to 10 min. compute_e -d 1000000 now completes (10.25 s).

  I14 wired into ntt_bigint_mul (2026-05-31):
    launch_stockham_intt_hadamard replaces separate Hadamard + INTT launchers.
    4 standalone hadamard_kernel launches eliminated per multiply.

  CPU VM autonomous loop (2026-06-01, make check 17/17):
    D1 (dead d_tw_cross_inv removed, ~4 MB HBM + leak), A2 (n_inv_table at
    init), A3 (h_gather_tmp pre-alloc), A1 (GPU Garner wired), C1 (constexpr
    STOK_STRIDE_C, LDS 16912→4208 B), A4 (concurrent NTT(f)||NTT(g) streams).

  Transposed scatter/gather (2026-05-31, I15, CPU-done):
    bigint_scatter_t/bigint_gather_t fold 2 GPU transpose_sq launches into
    CPU. Host make check 17/17. GPU correctness gate pending.

[1;37m══ Archived from STATUS.md 2026-06-29 (resolved/superseded; lean-navigator rule) ══[0m

## Completed This Loop (2026-06-06, GPU session on gfx1030)

- PHASE 1: make check-gpu 10/10 PASS (R2-R11).
- PHASE 2: Pipeline profile. PHASE 4: group_b_bench. PMC C1/C2. PHASE 6 R7/I11 oracle PASS.
- GPU SAFETY HARDENING PASS (2026-06-06, second pass):
  Three crashes investigated. Root causes and all fixes:

  CRASH 1 (PMC): rocprofv3 PMC on display-sharing GPU corrupted amdgpu driver state after
    clean exit. Fixes: hard guard in group_c_pmc.sh (GPU_RUN_ALLOW_LONG required);
    S19 in gpu_run.sh blocking rocprofv3 without headless mode.
  CRASH 2 (hipcc): Corrupted driver from crash 1 survived soft reboot; hipcc device-open
    re-triggered hang. Fix: gpu_health_check.sh wired into check.sh, check_gpu.sh,
    gpu_session.sh, group_b_bench.sh, group_d_stress.sh, gpu_headless_run.sh.
  CRASH 3 (rocm-smi hang): smi_busy() called bare rocm-smi with no timeout; hung on
    corrupted driver; display manager stayed stopped; machine froze. Fix: timeout 5
    rocm-smi in smi_busy() with rc=124 treated as unavailable. Applied in gpu_run.sh,
    group_d_stress.sh (3 sites), run_full_sweep.sh.

  CRASH 4 (2026-06-06, headless incompatible): User ran d=500000 headless run.
    `systemctl stop gdm3` itself triggered an immediate amdgpu kernel panic on
    bare-metal 6900 XT (<1 second crash). This is a driver-level incompatibility:
    releasing the display from amdgpu while Gnome is active causes a kernel panic
    on this specific GPU+driver version. Confirmed 3 times total (both d=200k and
    d=500k headless attempts).
    Fix: gpu_headless_run.sh DEPRECATED for this hardware. All long runs now use:
      GPU_RUN_ALLOW_LONG=1 scripts/gpu_run.sh <timeout> <binary> -f
    ROCm Compute ring TDR is 600s (not 10s), so headless mode is not needed.

  CRASH 5 (2026-06-06, SIGPIPE chain): d=1000000 piped to head -15; SIGPIPE killed
    binary without GPU cleanup; dirty driver state; 21s later new run launched on
    dirty GPU → kernel panic severe enough to require BIOS reset and hard restart.
    Fixes applied this session:
    a. lib3/compute_e/main.c: added signal(SIGPIPE, gpu_cleanup_handler) so any
       broken-pipe exit runs hipDeviceReset before the process exits.
    b. gpu_run.sh post-check: skip sysfs poll when rc >= 130 (signal kill) since
       gpu_cleanup_handler already reset the device.

  CRASH 6 (2026-06-06, rocminfo in health check — BIOS reset level): After the
    BIOS reset and fresh boot, agent ran gpu_health_check.sh --verbose which called
    `timeout 5 rocminfo`. rocminfo opened /dev/kfd and issued HSA discovery
    commands. Despite returning exit 0, the async GPU ring commands completed later
    and triggered another kernel panic. This caused another BIOS-reset-level crash.
    ROOT CAUSE: rocminfo is NOT safe on a post-panic fragile driver. Even a clean
    return from rocminfo does not mean the driver is stable.
    Fixes:
    a. gpu_health_check.sh: removed rocminfo probe entirely. Now uses ONLY:
       lsmod, dmesg tail, /dev/dri/renderD* check, sysfs power_state/vendor.
       No device opens of any kind.
    b. gpu_run.sh smi_busy(): replaced rocm-smi with sysfs gpu_busy_percent read
       (/sys/class/drm/card0/device/gpu_busy_percent). No device open.
    RULE: After any crash, NEVER run rocminfo/rocm-smi until the machine has had
    a clean power-cycle boot and been stable for several minutes. Only sysfs is safe.

  Additional fixes (deeper audit 2026-06-06):
  - ntt_stub.c: added ntt_gpu_emergency_reset() no-op (host-only link was broken)
  - gpu_health_check.sh: expanded dmesg scan from 60→200 lines; added drm/fault patterns
  - Ledger NUL bytes: stripped from gpu_run_ledger.log; S15 now strips NULs before grep
  - Signal handler (main.c): added alarm(8)/SIGALRM=SIG_DFL so emergency reset cannot
    hang indefinitely (send_sigterm=0 means driver won't signal us on GPU reset; alarm
    ensures we exit within 8s if hipDeviceSynchronize blocks after TDR)
  - Makefile test-gpu / cross-verify: now wrapped via scripts/gpu_run.sh (were bare calls)
  - group_c_pmc.sh: added post-rocprofv3 health check after each capture; aborts if
    driver corrupted before launching next PMC capture
  - gpu_headless_run.sh: added gpu_health_check.sh gate after DM stop, before gpu_run.sh


## CRASH 13 (2026-06-08) — Root Cause and Fix

ROOT CAUSE (confirmed 2026-06-08 session, corrected from initial GfxOff theory):
  I11 (rect Bailey 4-step for odd log_n > 10) was marked DISABLED in OPT_LEDGER
  but the production dispatch code in lib2/arith/ntt_mul.hip was NEVER REVERTED.
  fwd_ntt_dispatch() and inv_ntt_dispatch() else branches still called
  launch_stockham_ntt_rect / launch_stockham_intt_rect for odd log_n > 10.
  At d=200000, log_n=17 (odd > 10): 962 dispatches through the broken path.
  The GPU fault produced an AllowGfxOff SMU timeout (0xFFFFFFFF) as a SYMPTOM
  of corrupted SMU state — the GfxOff error was the fire alarm, not the fire.
  Small runs (d=1000/10000/100000) use log_n<=13 (≤10 or even) → Stockham only,
  so I11 was never triggered and they always passed. Only odd log_n > 10 (17, 19...)
  hit the broken path.

FIX (applied 2026-06-08, ntt_mul.hip):
  fwd_ntt_dispatch() else: now calls launch_ntt<PIDX>() (CT-DIT).
  inv_ntt_dispatch() else: now calls launch_intt<PIDX>() (CT-DIT inverse).
  Added forward declaration for launch_intt<PIDX>() which was missing.
  OPT_LEDGER I11 entry updated with confirmed production fault history.

VERIFIED (2026-06-08, crash boot, GPU_RUN_FORCE=1):
  d=200000 → PASS rc=0 (1.47s, 962 dispatches, log_n=17, CT-DIT path active).
  Machine survived. This is the first successful d=200000 run.


## CRASH 14 (2026-06-08) — Root Cause and Fix

ROOT CAUSE (confirmed 2026-06-08):
  d=600000 (log_n=19, odd>10, CT-DIT) ran ~5 seconds of computation successfully,
  then crashed on HIP teardown. Root cause: gpu_preflight.sh had a string-format bug
  that caused the pp_features GFXOFF-disable write to always fail silently.

  Bug: _pp_high was extracted with `grep -o '0x.*'` which kept the "0x" prefix.
  Combined write produced "0x0x00003763a25f7dff" — kernel's kstrtou64 rejected it.
  Error was suppressed by 2>/dev/null + || branch, so the script logged a WARNING
  but GFXOFF stayed enabled on ALL boots since the service was created.

  Consequence: when d=600000 completed and the HIP context closed, the KFD driver
  called AllowGfxOff → SMU firmware/driver mismatch (driver v0x40 vs firmware v0x41)
  returned 0xFFFFFFFF (timeout) → machine hard-freeze, power cycle required.

  Evidence: journal at 19:43:34 showed "WARNING: could not write pp_features
  (GFXOFF still enabled)". Baseline showed "GFXOFF (20) : enabled" after service ran.
  Telemetry: GPU at 96-98% through T+5s, no T+6s entry → crash at teardown.
  AER errors: 0 throughout (not a hardware fault — pure SMU protocol error).

  Why d=200000 and d=300000 passed: AllowGfxOff timeout is nondeterministic;
  short runs may complete teardown before the SMU command times out.

FIX (applied 2026-06-08, scripts/gpu_preflight.sh):
  Changed `grep -o '0x.*'` to `grep -o '[0-9a-fA-F]*$'` for both _pp_high and
  _pp_low extraction — strips the "0x" prefix so write is "0x<high><low>".
  Updated Python to use `int('${_pp_low}',16)` for explicit hex parsing.
  Added post-write verification by re-reading pp_features for GFXOFF status.
  Test write confirmed: "0x00003763a36f7dff" → write OK, GFXOFF: disabled.

  gpu_run_ledger.log corrected: d=600000 entry changed from rc=SIGKILL / "NOT a GPU
  crash" to rc=CRASH / CRASH 14 (the SIGKILL was the kernel killing processes during
  the machine hard-freeze, not a timeout from gpu_run.sh).


## CRASH 15 (2026-06-08) — Second consecutive d=600000 run

ROOT CAUSE: KFD driver state from first session's teardown not fully cleared.
  First d=600000 (23:37:43-23:37:53): rc=0, GFXOFF disabled, computation clean.
  Second d=600000 (23:39:11, 78s later): crashed at T+0-1s — 0% GPU busy,
  730MB VRAM (no allocation), 0 AER — crash during hipSetDevice/KFD open
  before any kernel dispatch. Same mechanism as 2026-06-07 back-to-back crashes.
  S6 30s cooldown is insufficient between consecutive full HIP sessions on a
  display-sharing gfx1030. One GPU job per boot session is the safe rule.

FIX: No code change. Operational rule: never run two HIP sessions back-to-back
  on this machine. The GFXOFF fix (CRASH 14) is confirmed working — d=600000
  PASSED on the first attempt on a clean boot with GFXOFF disabled.


## flip_done Crash Class — RESOLVED 2026-06-13

Root cause confirmed via netconsole: with cwsr_enable=0, the top-level binary-split
NTT mul at d>=1M held the GFX ring for >5s, causing DRM flip_done timeout → GPU
reset → hard hang. Captured in crash_2026-06-13_15-04-24.224/ on receiver.

Fix (two-layer):
  System:  cwsr_enable=1 in /etc/modprobe.d/amdgpu-cwsr.conf (safe — GFXOFF is
           permanently disabled so the AllowGfxOff SMU message is never sent;
           CRASHes 12-19 cannot recur). Propagated via update-initramfs + shutdown.
  Code:    lib2/arith/ntt_mul.hip — added mid-dispatch sync+yield between forward
           and inverse NTT passes; EVERY default in gpu_run.sh changed from 5 to 1.
           Bounds max ring-busy window to max(fwd_NTT_time, inv_NTT_time).

Verified 2026-06-13: compute_e_dev_l64 -d 1100000 completed rc=0 in 69.50s wall
  (split 14.43s GPU, base_convert 49.98s CPU). Full GNOME desktop + VS Code attached.
  AER errors: 0. Post-check: GPU idle. No crash.


## base_convert Acceleration — VERIFIED 2026-06-14

BASE_CONVERT_MUL_THRESHOLD=512 on GFX1030_LOCAL (lib2/arith/base_convert.c): raise
  the schoolbook/NTT crossover during BFS base conversion so small nodes use CPU
  schoolbook instead of paying a ~5-20ms PCIe hipStreamSynchronize round-trip per
  GPU dispatch. Runtime threshold via bigint_mul_set/get_threshold (multiply.c);
  base_convert raises it around dc_convert_bfs, restores after. binary_split (always
  large operands) unaffected.

Two build bugs found+fixed this session (both required for the path to work):
  a. lib3/compute_e/Makefile: C_CFLAGS lacked -DGFX1030_LOCAL=1, so the GFX1030_LOCAL
     guards in base_convert.c were inactive → managed-memory path → GPU page fault
     (R11). Fix: C_CFLAGS_DEV := $(C_CFLAGS) -DGFX1030_LOCAL=1 on all F7 objects.
  b. Makefile:485: $(GPOLYMUL_6900XT) built with $(ARCH_MI300A) (gfx942) → "No
     compatible code objects for gfx1030" runtime crash (R9). Fix: $(ARCH_6900XT).

Verified 2026-06-14: compute_e_dev_l64 -d 1100000 rc=0 in 26.02s wall
  (split 14.44s, division 0.53s, base_convert 6.53s, 1837 dispatches). Leading 60
  digits match OEIS A001113. Desktop + VS Code attached. AER 0. GPU idle. No crash.
  base_convert: 49.98s → 6.53s (7.65x). Total: 69.50s → 26.02s (2.67x).
  check-gpu 2026-06-14: R2-R11 all PASS (R9 16/16, R11 d=10000 dual-LB rc=0/0).


## d=2.1M Verified — 2026-06-14 (largest dev-box run, flip_done-safe)

Calibration sweep then d=2.1M run, all through gpu_run.sh (desktop + VS Code
attached, 110s cap, cwsr=1 + 5ms mid-dispatch yield, netconsole watch on littleblue):
  d=1.1M log_n=19  1837 disp  split 14.46s  basecvt 6.48s  total 26.04s  rc=0
  d=1.5M log_n=20  2831 disp  split 31.59s  basecvt 7.72s  total 45.24s  rc=0
  d=2.1M log_n=20  3319 disp  split 32.13s  basecvt 14.61s total 53.34s  rc=0
Per-dispatch 7.87ms (log_n=19) vs 11.16ms (log_n=20, N=1048576). d=1.5M is the
same transform/per-pass ring window as d=2.1M; both clean (no flip_done, no wedge,
GPU idle, netconsole silent of faults). Estimate (~54s) matched actual 53.34s to
1.3%. e correct to leading digits (OEIS A001113). d=2.1M needs -f (binary's
internal est-GPU guard is ~3.6x pessimistic: est 159s vs 32s actual split).
This retires CRASH 22 (which was a 360s d=2.1M run that BYPASSED gpu_run.sh);
d=2.1M is now proven safe THROUGH the wrapper within the cap.


## CRASH 25 — 2026-06-14 — SMU thermal poll during d=3M (log_n=21)

Hard freeze ~10:55:47, ~32s into a d=3M run (log_n=21, largest ever), mid-binary_split.
Power cycle required. ROOT CAUSE (leading, confounded): a 20s software thermal poll
(hwmon temp1/2/3_input = TransferTableSmu2Dram SMU queries) running CONCURRENT with
peak GFX-ring saturation wedged the SMU — the CRASH 23 mechanism. Evidence: temps
benign (62C junction, not thermal); SMU-free telemetry nominal (VRAM/rt/AER/PCIe) to
the last sample 10:55:47 then nothing; netconsole captured ONLY the LAUNCH marker then
silence (instant-SMU-wedge signature, NOT flip_done which logs — cf CRASH 20). The poll
was the only SMU traffic introduced; d=2.1M/log_n=20 ran clean 3x the same day without
it. CONFOUND: log_n=21 (2x per-pass ring window) was also new and untested — a pure
sustained-load hang is possible, but the netconsole silence argues against flip_done.

RESOLUTION: in-run software SMU thermal poll ABANDONED. gpu_run telemetry stays SMU-free.
Thermal protection = hardware trip (temp1_crit/emergency, firmware-enforced). See memory
[[feedback-no-smu-thermal-poll-during-compute]]. Process errors: (1) validated the read
on an IDLE GPU, not during compute (wrong hazard regime); (2) changed two variables at
once (poll + biggest-ever size). Disambiguation (cold boot, supervised): rerun d=3M
log_n=21 WITHOUT poll — survives => poll caused it; crashes with flip_done => load did.


## CRASH 27 — 2026-06-14 — log_n=21 CAPTURED with full instrumentation

Built a crash-capture stack (kdump+hardlockup_panic, kmsg heartbeat→littleblue, amdgpu
fence-trend, netconsole@loglevel7; scripts/crash_hunt_run.sh; see memory
[[reference-crash-hunt-instrumentation]]). Triggered via microbench 2500 log_n=21 muls.
CAPTURED: froze at mul 1549/2500 (~56s in, predicted regime). Two-stage — GPU mul #1549
wedged (progress stalled), heartbeat beat ~750ms more, then TOTAL lock (heartbeat+journald
+netconsole stop together). NOT OOM (59.5GB free), NOT CPU (load 1.73), NOT leak. ZERO
kernel printk at loglevel 7; NO kdump (hardlockup_panic didn't fire), NO amdgpu TDR/reset,
NO NMI panic — lock is BELOW all kernel logging/recovery. = total hardware/fabric-level
hang (CPU access to wedged GPU never returns → SoC lock). NOT software-recoverable. Local
run.log lost buffered stdout at iter=906; remote heartbeat captured mul 1549 (validates the
off-box receiver design). CONFIRMS the log_n=20 ceiling is a HARDWARE limit; log_n>=21 = MI300A.
[SUPERSEDED 2026-06-16: NOT a hardware limit — fixed by pinning sclk 1300MHz; log_n>=21 runs here. See CEILING note above.]
Box cold-cycled (boot 16:14:20); panic knobs+loglevel reset on reboot.


## CRASH 26 — 2026-06-14 — d=3M log_n=21 (no poll) — ROOT CAUSE CORRECTED

Disambiguation run: re-ran d=3M/log_n=21 WITHOUT the thermal poll. It CRASHED AGAIN
(silent freeze ~64s in, mid-binary_split, benign/no faults) — vs ~32s with the poll.
CONCLUSION: the thermal poll was NOT the root cause of CRASH 25; it was only an
ACCELERANT (64s->32s). ROOT CAUSE = log_n=21 (N=2097152) GPU load on the display GPU:
a single NTT pass ring-window is ~2x the proven log_n=20 window, long enough that over
tens of seconds of binary_split muls a display page-flip misses flip_done -> amdgpu
reset -> hard hang (silent; saturation blocks netpoll/journald). Earlier CRASH 25
diagnosis (poll = prime suspect) was WRONG, corrected by this controlled rerun.

CEILING [SUPERSEDED 2026-06-16 — DO NOT FOLLOW THE log_n<=20 CAP BELOW]: the real fix
was a clock change, not a code/MI300A limit. Pinning sclk to 1300 MHz (down from the
1825 default that hard-locks under sustained load) makes log_n>=21 RUN STABLY on this
box. With 1300 MHz: log_n=22 GMP-verified, log_n=23 (d=19M) completed correct (~3.7h);
log_n=24 (3-factor recursive 4-step NTT) ACHIEVED 2026-06-29 — full multiply GMP-verified,
all 16,777,184 product limbs match (2 seeds, 8.4M-limb operands, ~39M digits); regression
20/22/23 clean. The display/flip_done framing here
was refuted (see GPU_CRASH_RECIPE.md §2 + [[log-n-21-ceiling-display-gpu]] + session
2ce85f8f). Run large jobs at 1300 MHz, verified before launch.
(Superseded original note:) cap dev-box compute_e at log_n<=20; log_n>=21 not safe / needs
MI300A or intra-pass yields. — wrong; the fix was the 1300 MHz pin.
Recovery: cold cycle done (boot 11:24:42); CRASH_MARKER present; health correctly blocks.

RECOVERY CONFIRMED 2026-06-14 11:16: after cold cycle (10:59:28), CRASH_MARKER cleared
(post-inspection), check 0b auto-passed (uptime>300s + 0 GPU errors), S15 overridden
deliberately (GPU_RUN_FORCE=1, user-authorized, after root-cause investigation). Re-ran
d=2.1M (log_n=20, NO thermal poll, no SMU reads): rc=0, total 53.65s (split 32.13s,
basecvt 14.63s, 3319 disp) — matches pre-crash 53.34s. netconsole silent of faults, GPU
idle, health rc=0, ledger shows clean EXIT. The box works again; the thermal poll was
the differentiator (its removal restored stability). d=3M/log_n=21 sans-poll remains the
optional supervised disambiguation if ever wanted.
