[1;37m╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                  N T T   /   M I 3 0 0 A   —   P E R F O R M A N C E                                   ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat PERFORMANCE.md   or   less -R PERFORMANCE.md[0m

  Current measured baseline per hardware target. One number per
  (kernel × n) cell, picked as the best of the most recent measurement
  pass. Forecasts for unmeasured targets are clearly labelled. Update
  this file when a kernel changes; do not accrete dated tables.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  1. AMD Ryzen 5950X (dev VM, single-threaded)  —  CPU REFERENCE[0m

  Single-threaded; ref CPU paths + app compute_e CPU path.
  Refreshed [1;36m2026-06-01[0m (100 000 iters, cpu_testbench.sh). All 50 checks PASS.

  [1;36m  ref NTT throughput at n ≤ 1024 (best of CT-DIT / Stockham / Montgomery):[0m

  ┌──────────────┬───────────┬──────┬────────────────┐
  │ [1;36mPrime[0m         │ [1;36mForm[0m       │ [1;36mmax n[0m │ [1;36mBest NTT/s[0m      │
  ├──────────────┼───────────┼──────┼────────────────┤
  │ Fermat-8      │ 2^8+1      │  256  │ 273 149 (CT-DIT) │
  │ ML-KEM        │ 13·2^8+1   │  256  │ 146 407 (Mont)   │
  │ FALCON        │ 3·2^12+1  │  1024 │  32 783 (Mont)   │
  │ ML-DSA        │ 2^23-2^13 │  1024 │  32 637 (Mont)   │
  │ Goldilocks    │ 2^64-2^32 │  1024 │  25 402 (Stok)   │
  └──────────────┴───────────┴──────┴────────────────┘

  [1;36m  app compute_e (CPU-only path, 5950X VM, 2026-06-01):[0m

  ┌─────────┬───────────┬────────────┐
  │ [1;36mdigits[0m   │ [1;36mwall time[0m │ [1;36mnotes[0m      │
  ├─────────┼───────────┼────────────┤
  │   100    │   < 1 ms  │ schoolbook │
  │  1 000  │   2 ms   │ schoolbook │
  │ 10 000  │  147 ms  │ NTT path   │
  │ 100 000 │  13.2 s  │ NTT path   │
  └─────────┴───────────┴────────────┘

  The 13.2 s figure is CPU-only (no GPU dispatch). The GPU path at d=100k
  ran in 0.47 s on gfx1030 (last measured 2026-05-29 pre-I13); the
  post-I13+I14+I15+A1+C1+A4 GPU baseline measured 06-06: 0.40 s (LB=64) / 0.37 s (LB=112).

  [1;36m  lib CPU pipeline overhead (scatter/gather, 2026-06-01, N=2^16):[0m

  ┌─────────────────────────┬───────────┬───────────────────┐
  │ [1;36mOperation[0m               │ [1;36mCost[0m      │ [1;36mNotes[0m             │
  ├─────────────────────────┼───────────┼───────────────────┤
  │ scatter (linear)        │ 0.11 ms   │ baseline         │
  │ scatter_t (transposed)  │ 0.17 ms   │ +0.06 ms (I15)   │
  │ gather  (linear)        │ 0.27 ms   │ baseline         │
  │ gather_t (reorder)      │ 0.45 ms   │ +0.18 ms (I15)   │
  │ modpow (n_inv, 1 prime) │ 0.48 µs  │ A2: 4× per call │
  └─────────────────────────┴───────────┴───────────────────┘

  I15 CPU overhead: scatter_t adds +0.06 ms and gather_t adds +0.18 ms vs
  the linear paths (extra O(N) reorder). This is traded against eliminating
  two GPU transpose_sq launches (~19.7% of NTT compute at d=100k). Net win
  on GPU is large; the CPU overhead is negligible relative to GPU compute.
  A2 modpow elimination saves ~1 ms at d=100k (561 calls × 4 primes × 0.48 µs).

  Note: dev box is a VM — bare-metal 5950X will be faster. These are
  representative; order-of-magnitude comparisons are reliable.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  2. AMD Radeon RX 6900 XT (gfx1030, RDNA2)  —  DEV TARGET[0m

  Dev hardware. Drives gpu_run.sh-wrapped binaries. All cells below
  measured over ≥ 10k iterations on the 14-prime GPU-eligible subset
  (q < 2^32, 11 primes); throughput is the median across primes unless
  a per-prime split is called out.

[1;36m  CT-DIT (ref/ntt_gpu.hip)  — NTT/s, uniform across primes:[0m

    n=256      ~33k        n=1024     ~27k        n=4096   ~22.5k
    Memory-bound (PMC 4.2% VMEM stalls); Shoup reduction has no effect.

[1;36m  Stockham LDS-fused (ref/ntt_gpu_stockham.hip)  — NTT/s:[0m

    n=256     ~197-202k    n=1024    ~158k         n=2048   ~113-115k  (Shoup 2026-05-29)
    n=16384   ~19k (4-step)            n=65536    ~13k (4-step)
    Shoup 32-bit butterfly replaces % q: ~3.6x speedup vs 2026-05-25 baseline.
    Pre-Shoup (2026-05-25): n=256 ~32-53k. Batched b=40 n=256: ~7M NTT/s.

[1;36m  Fused polymul (ref/ntt_gpu_polymul.hip)  — polymul/s:[0m

    n=256     ~98-100k     n=1024    ~67k         n=2048   ~44k  (11-prime median)
    CRT-hi/FHE-RNS (large primes): n=256 ~83-84k, n=1024 ~43-44k (2026-06-06).

[1;36m  compute_e d-scaling (app/compute_e, end-to-end):[0m

  ┌─────────┬────────┬────────┬─────────┐
  │ [1;36md[0m       │ [1;36mLB=64[0m  │ [1;36mLB=112[0m │ [1;36mspeedup[0m │
  ├─────────┼────────┼────────┼─────────┤
  │ 100     │ 0.14 s │ 0.14 s │ 1.00×   │
  │ 1 000   │ 0.16 s │ 0.17 s │ 0.94×   │
  │ 10 000  │ 0.24 s │ 0.22 s │ [1;32m1.09×[0m   │
  │ 100 000   │ 0.40 s │ 0.37 s │ [1;32m1.08×[0m   │
  │ 1 000 000 │ 7.72 s │   —    │    —    │
  │ 1 100 000 │ 26.02 s│   —    │    —    │
  │ 1 500 000 │ 45.24 s│   —    │    —    │
  │ 2 100 000 │ 53.34 s│   —    │    —    │
  └───────────┴────────┴────────┴─────────┘
  (Stockham 2026-05-29: d=100000 0.57/0.45 s CT-DIT -> 0.44/0.38 s, -23%/-16%;
   2026-06-06 bench sweep (bn3kngwrt): 0.40/0.37 s LB=64/112.
   d=1000000 measured 2026-06-07, clean boot + TDR enabled, LB=64, log_n=19,
   3482 dispatches; split 2.10s, 10^D 0.01s, div 0.10s, output 4.32s, rc=0.
   d=1100000 measured 2026-06-14, LB=64, log_n=19, 1837 dispatches, desktop+VS
   Code attached, flip_done-safe config (cwsr=1 + 5ms mid-dispatch yield):
   split 14.44s, div 0.53s, base_convert 6.53s, total 26.02s, rc=0. The yield
   (display-safety, every mul) inflates split vs the 2026-06-07 no-yield run;
   base_convert accel (BASE_CONVERT_MUL_THRESHOLD=512 on GFX1030_LOCAL) cut the
   decimal conversion 49.98s -> 6.53s (7.65x) vs the 2026-06-13 pre-accel run.
   smaller / odd-log_n d use CT-DIT/schoolbook and are ~unchanged.)

  Calibration sweep 2026-06-14 (LB=64, desktop attached, accelerated+capped config):
   d=1.1M  log_n=19  1837 disp  split 14.46s  basecvt 6.48s  total 26.04s  rc=0
   d=1.5M  log_n=20  2831 disp  split 31.59s  basecvt 7.72s  total 45.24s  rc=0
   d=1.8M  log_n=20  (binary self-aborted at its 120s est-GPU guard; needs -f)
   Per-dispatch: 7.87ms (log_n=19) vs 11.16ms (log_n=20, N=1048576). d=1.5M is
   already log_n=20 — the same transform/per-pass ring window as d=2.1M — and ran
   clean (no flip_done, no wedge), validating the log_n=20 GPU path on the display
   GPU. d=2.1M MEASURED 2026-06-14 (426229 terms, log_n=20, 3319 disp): split
   32.13s, div 0.66s, basecvt 14.61s, total 53.34s, rc=0, no crash (netconsole
   silent of faults, GPU idle post-run); pre-run extrapolation was ~54s (matched
   to 1.3%). Largest dev-box run to date. Needs -f (internal est-GPU guard ~3.6x
   pessimistic: est 159s vs 32s actual split). Validates the log_n=20 path
   through gpu_run.sh within the 110s cap.

[1;36m  Pipeline breakdown (NTT_PROFILE=2, gfx1030, 2026-06-06):[0m

    d=10 000  (log_n=13, CT-DIT-Mont, 149 dispatches, total 0.24 s):
      scatter 3.8 ms(4%)  gpu_pipe 70.1 ms(82%)  garner 10.7 ms(13%)  gather 0.4 ms(1%)
      gpu_pipe: upload 12.2 ms  +  compute 61.5 ms  +  download 0.2 ms  (transfers 17%)
    d=100 000 (log_n=16, Stockham, 561 dispatches, total 0.40 s):
      scatter 15.6 ms(7%)  gpu_pipe 140.2 ms(62%)  garner 65.4 ms(29%)  gather 5.1 ms(2%)
      gpu_pipe: upload 56.3 ms  +  compute 98.7 ms  +  download 0.8 ms  (transfers 37%)
    Note: garner (29%) and upload (37% of pipe) are GFX1030_LOCAL=1 artifacts;
    on MI300A unified memory both vanish (HBM3, zero-copy). Compute = floor.

[1;36m  PMC (rocprofv3, n=2048, q=998244353):[0m

    Stockham LDS-fused (n=2048, q=998244353, 2026-06-06):
      30.0M VALU / 24.3K LDS / 1.47M VMEM-cyc / 13.4K waves
      per-wave: 2229 VALU, 1.8 LDS, 110 VMEM-cyc; COMPUTE-BOUND
      Hot kernel (OTF): 21.3M VALU / 3264 waves = 6518 VALU/wave
    Polymul fused (n=2048, ITERS=50 --bench-only, 2026-06-06):
      12.9M VALU / 12.3K LDS / 715K VMEM-cyc / 4192 waves
      per-wave: 3078 VALU, 2.9 LDS, 171 VMEM-cyc
    Raw: perf/results/X3_PMC_GFX1030_20260606_184020.md

    Re-capture 2026-06-29 (ROCm 7.0.3 / rocprofv3 1.0.0; same config; box did
    NOT crash via the gpu_run.sh-wrapped path, DM up): Stockham 2761 VALU / 2.2
    LDS / 150 VMEM-cyc per wave (OTF hot kernel 7638 VALU/wave); Polymul 1595
    VALU / 1.0 LDS / 103 VMEM-cyc per wave — BOTH still COMPUTE-BOUND. Wave
    counts match 06-06 exactly and the ref kernels are unchanged (mtimes
    05-25/30), so the absolute VALU shift (Stockham +24%, OTF +17%, Polymul
    -48%) is a rocprofv3/ROCm VERSION effect, not a code change — trust the
    VALU-bound classification, treat absolute VALU magnitudes as version-
    dependent. Raw: perf/results/X3_PMC_GFX1030_20260629_181807.md.
    (Post-PMC rocm-smi shows GPU%=99% but at 42 W idle / clean dmesg — a cosmetic
    stuck-counter artifact, not a hang; check POWER not GPU% after a PMC run.)

[1;36m  Thermal / power under 60 s sustained ML-KEM bench:[0m

    peak 40 °C / 48 W / SCLK 2.57 GHz; GPU-busy 3-5%; no thermal throttling.

[1;36m  Determinism (200-repeat long-soak):[0m

    50-iter x 11 GPU-eligible primes x 3 sub-tests = 33/33 PASS (2026-05-29).
    200-iter x 11 primes x 3 sub-tests = 33/33 PASS (2026-05-29, S2 soak done).

[1;36m  I11 rect Stockham oracle (2026-06-06):[0m

    Rectangular Bailey 4-step (launch_stockham_ntt_rect) certified vs
    CT-DIT-Mont (I4) at ALL odd log_n: 11/13/15/17/19 = 5/5 PASS.
    Production dispatch in ntt_bigint_mul remains disabled (ROADMAP R7
    requires bounded headless compute_e d=200k/500k/1M before re-enable).
    Kernel arithmetic is correct; the original production-scale crash
    (compute_e d=1e6, log_n=19, 2026-05-27) may have been TDR/OOM, not
    a kernel arithmetic bug.

[1;36m  Cross-verify (R1 --full):[0m

    49 cells (11 GPU-eligible primes x n in {256..16384}) x 7 sub-tests = PASS.
    R10 extended to n=16384 (log_n=14 even); all 4-step cells PASS.

[1;36m  GPU Garner + GPU scatter end-to-end (compute_e d=10 000):[0m

    Pre-Phase-2/3:       LB=64  0.38 s   LB=112  0.32 s
    Post-Phase-2/3:      LB=64  0.24 s   LB=112  0.22 s   (~30% faster)

  [1;36m  Stockham swap (2026-05-29, measured; production ntt_bigint_mul):[0m
    The lib 4-step Stockham was found NOT to be a DFT (roundtrips but
    pointwise-multiply != convolution); FIXED via Bailey transpose + reverse-
    order inverse (ntt_kernel.hip), verified by test_ntt_convolution (all
    primes, 4-step sizes). ntt_bigint_mul now dispatches Stockham for even
    log_n and log_n<=10; CT-DIT for odd log_n>10. Plus a tiled LDS transpose
    (I7) and work-sized kernel launches (I8, stok_block — the fixed 512-thread
    launch left up to 75% of wave64 idle at log_n=16). Measured at d=100000
    (log_n=16, gfx1030), NTT_PROFILE=2, vs the CT-DIT baseline:
      NTT compute (NTT+Had+INTT):  287.8 ms -> 112.7 ms   (2.55x, -61%)
      engine total:                405.1 ms -> ~228  ms   (-44%)
    Incremental: Stockham swap 154.0 -> tiled transpose 144.1 -> I8 126.4
      -> I13 fused transpose+X 114.6 -> I14 fused Hadamard 112.7 ms.
    Digits EXACT and deterministic (100000/100000 identical across 2 runs).
    Verified correct at d=300000 (log_n=18, Stockham) and d=1000000
    (log_n=19, CT-DIT fallback) — both digit-exact. Note: odd log_n>10 still
    uses CT-DIT (Stockham 4-step needs a square split); a radix that covers
    odd sizes would extend the win there. After I13+I14+I15 the remaining GPU kernel mix at log16 is
    stockham butterflies + xtranspose (fused cross-twiddle); the two
    standalone transpose_sq calls (1st-T fwd + last-T inv) are eliminated
    by bigint_scatter_t / bigint_gather_t (I15). Wall-clock impact of I15
    pending GPU re-measure. garner (GPU kernel + the
    GFX1030_LOCAL d_out->h_out D2H, 65 ms) is a dev-only artifact that
    vanishes on MI300A unified memory.
    MI300A zero-copy gate (#1+#2, GFX1030_LOCAL=0) adds ~10% on top by
    skipping the GPU Garner d_out→h_out D2H; cannot be measured here.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  3. AMD Instinct MI300A (gfx942, CDNA3)  —  TARGET[0m

  Production target. Measured numbers populate here as Phase 4 runs;
  the table below is the [1;33mforecast / acceptance-target[0m informed by gfx1030
  measurements scaled by the CU ratio (2.85×) and the gfx942 occupancy
  factor (1.78–3.20× same kernel) — see ARCHITECTURE §10.

  ┌─────────────────────────────────────────┬──────┬─────────────────┬────────────┐
  │ [1;36mKernel[0m                                  │ [1;36mn[0m    │ [1;36mForecast[0m        │ [1;36mStatus[0m     │
  ├─────────────────────────────────────────┼──────┼─────────────────┼────────────┤
  │ Stockham LDS-fused (per APU)            │ 256  │ ~1.5M NTT/s     │ [1;33munmeasured[0m │
  │ Stockham LDS-fused (per APU)            │ 4096 │ ~250k NTT/s     │ [1;33munmeasured[0m │
  │ Fused polymul (per APU)                 │ 2048 │ ~400k polymul/s │ [1;33munmeasured[0m │
  │ 4-APU CRT-NTT pipeline (lib, log_n=20) │ 2^20 │ [1;33mmeasure[0m         │ [1;33munmeasured[0m │
  │ compute_e (app, d=10^6, all 4 APUs)    │ —    │ [1;33mmeasure[0m         │ [1;33munmeasured[0m │
  └─────────────────────────────────────────┴──────┴─────────────────┴────────────┘

[1;36m  MI300A hardware fixed points (from ~/MI300A_TARGET_ENVIRONMENT.md):[0m

    HBM3 effective         4.0 TB/s (76% of 5.3 spec)
    LDS                    5.0 TB/s
    FP64 DGEMM            87.6 TF/s
    SDMA                   3.98 TB/s
    P2P all-to-all sust.   650 GB/s    (60.2% of 978 burst)
    P2P unidir / link     66.63 GB/s   (use for broadcast_input)
    Atomic aggregate      912 GOPS/s
    INT32                14001 GOPS/s  (Shoup dep-chain floor ~745 GIOPS, ~37% util)

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  4. METHODOLOGY[0m

  All GPU numbers gathered through scripts/gpu_run.sh (S1-S17
  safeguards: single-job flock, rocm-smi health gate, INT→TERM→KILL
  signal escalation, crash-aware guard, --bench-only enforcement for
  rocprofv3+polymul on RDNA2).

  Run driver:
    bash scripts/gpu_run.sh <secs> <binary> [args]   # single guarded GPU run
    (the gpu_session.sh / group_*.sh orchestration drivers + PMC capture are
     in archive/scripts_devbox/ — dev-box-only, restored locally as needed;
     not part of the delivered tree.)

  Raw per-run logs live under perf/results/ (gitignored). This file
  consolidates the current best numbers; do not duplicate raw tables here.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

  [1;33mBaseline dates: §1 CPU 2026-05-24 (dev VM) · §2 6900 XT 2026-05-25 (post Phase 2/3/4: GPU Garner + GPU scatter + zero-copy gate) · §3 MI300A unmeasured.[0m
