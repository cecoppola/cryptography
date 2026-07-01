[1;37m╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                 N T T   /   M I 3 0 0 A   —   A R C H I T E C T U R E                                  ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat ARCHITECTURE.md   or   less -R ARCHITECTURE.md[0m

  Technical reference for the NTT stack: layered design, NTT
  mathematics, supported moduli, the 4-prime CRT-NTT engine, GPU
  kernel conventions, and MI300A optimization rules. Cross-project
  hardware facts live in ~/MI300A_TARGET_ENVIRONMENT.md and
  ~/HIP_6900XT_KNOWLEDGE.md.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  1. LAYERED DESIGN[0m

  Three layers, each tested independently and stacked at runtime.

  ┌───────┬────────────────────────────────────────────────────────┬─────────────────────┐
  │ [1;36mLayer[0m │ [1;36mRole[0m                                                   │ [1;36mBuilds without GPU?[0m │
  ├───────┼────────────────────────────────────────────────────────┼─────────────────────┤
  │ ref  │ Single-prime reference NTT (CPU + GPU parity kernels)  │ yes                 │
  │ lib  │ 4-prime CRT-NTT engine (~255-bit composite, MI300A K4) │ host only           │
  │ app  │ Application: compute_e (Euler's number to N digits)    │ host only           │
  └───────┴────────────────────────────────────────────────────────┴─────────────────────┘

  Each lib has its own README with file map and entry points.
  ref is the algorithm proving ground; lib is the production engine;
  app is the end-to-end consumer that exercises the full stack.

  lib exposes TWO entry points on one shared engine:
    - crt_ntt (main.hip)      OpenSHMEM-distributed polynomial-multiply
                              pipeline; multi-node K4. The product (§4.3).
    - arith/ (ntt_bigint_mul) single-node big-integer multiply API
                              (+ Newton division, decimal conv) layered on
                              the engine; GPU scatter/NTT/Garner + host gather,
                              intra-node multi-APU, NOT OpenSHMEM. Consumed by app/.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  2. NTT FUNDAMENTALS[0m

  The Number-Theoretic Transform is the discrete Fourier transform over a
  finite ring Z/qZ:

      X[k] = Σ_{j=0}^{n-1}  x[j] · ω^{jk}   (mod q)

  ω is an n-th primitive root of unity in Z/qZ; the inverse uses ω^{-1} and a
  final scale by n^{-1} mod q. Convolution becomes pointwise multiplication in
  NTT space — the basis for fast polynomial multiplication used in ML-KEM,
  ML-DSA, Falcon, and other lattice cryptography.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  3. SUPPORTED MODULI[0m

  ntt_moduli.h dispatches to the fastest reduction routine per modulus.

  ┌───────────────────────────────────┬──────┬─────────────────────┬──────────────────────────┐
  │ [1;36mq[0m                                 │ [1;36mBits[0m │ [1;36mReduction[0m           │ [1;36mApplication[0m              │
  ├───────────────────────────────────┼──────┼─────────────────────┼──────────────────────────┤
  │ 3329 (ML-KEM)                     │ 12   │ reduce_generic      │ FIPS 203 / Kyber         │
  │ 12289 (Falcon, NTRU)              │ 14   │ reduce_generic      │ FIPS 206 / Falcon        │
  │ 8380417 (ML-DSA)                  │ 23   │ reduce_dilithium    │ FIPS 204 / Dilithium     │
  │ 7681                              │ 13   │ reduce_generic      │ small Solinas-style      │
  │ 40961                             │ 16   │ reduce_generic      │ small Proth              │
  │ 167772161 (CRT triple-low)        │ 28   │ reduce_generic      │ low-residue NTT (CRT)    │
  │ 469762049                         │ 29   │ reduce_generic      │ NTT general              │
  │ 998244353 (CRT triple-high)       │ 30   │ reduce_generic      │ Goldilocks-adjacent      │
  │ 2013265921                        │ 31   │ reduce_generic      │ NTT general              │
  │ 2305843009213693953 (Mersenne)    │ 61   │ reduce_generic 128b │ large-q                  │
  │ 1152921504606584833 (Solinas-60)  │ 60   │ reduce_solinas_60   │ exact 2-pass; no 128 div │
  │ 2287828610704211969 (Solinas-61)  │ 61   │ reduce_generic      │                          │
  │ 18446744069414584321 (Goldilocks) │ 64   │ reduce_goldilocks   │ shift-only NTT           │
  │ Fermat F8 (2^256+1)               │ 257  │ reduce_fermat8      │ FFT-style                │
  └───────────────────────────────────┴──────┴─────────────────────┴──────────────────────────┘

  Primitive roots (ω) for the common (q, n) pairs:

  ┌──────────────────────┬────────┬──────────────────────────────────────────┐
  │ [1;36mModulus q[0m            │ [1;36mn[0m      │ [1;36mω[0m                                        │
  ├──────────────────────┼────────┼──────────────────────────────────────────┤
  │ 3329                 │ 256    │ 17                                       │
  │ 8380417              │ 256    │ 1753 fwd basic NTT / 3073009 cyclic ω²   │
  │ 998244353            │ varies │ 3 (primitive root mod q; raise to power) │
  │ 18446744069414584321 │ 64     │ 7 (Goldilocks coset eval; ψ²=ω)          │
  └──────────────────────┴────────┴──────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  4. 4-PRIME CRT-NTT ENGINE  (lib — canonical plan)[0m

  Multiply very large integers / polynomials by splitting the work
  over four ~64-bit NTT-friendly primes (one length-n number-theoretic
  transform per prime — one per MI300A APU on a K4 node), doing the
  convolution in each residue, then Garner-CRT reconstructing the true
  product modulo the ~255-bit composite. This section is the single
  source of truth for the 4-prime plan; lib/primes.h is the code-side
  reference.

[1;36m  4.1 The four primes (lib/primes.h)[0m

  ┌───────┬────────────────────┬──────────────────────────┬─────────┬─────────────────────┐
  │ [1;36mprime[0m │ [1;36mvalue (hex)[0m        │ [1;36mform[0m                     │ [1;36mmax NTT[0m │ [1;36mreduce[0m              │
  ├───────┼────────────────────┼──────────────────────────┼─────────┼─────────────────────┤
  │ P0    │ 0xffffffff00000001 │ 2^64-2^32+1 (Goldilocks) │ 2^32    │ reduce_p0 (Solinas) │
  │ P1    │ 0xffffffffff000001 │ 2^64-2^24+1              │ 2^24    │ reduce_p1 (2-pass)  │
  │ P2    │ 0xfffffffc00000001 │ 2^64-2^34+1              │ 2^34    │ reduce_p2           │
  │ P3    │ 0xffffff0000000001 │ 2^64-2^40+1              │ 2^40    │ reduce_p3           │
  └───────┴────────────────────┴──────────────────────────┴─────────┴─────────────────────┘

  Reduction principle: for q = 2^a-2^b+1, 2^a ≡ 2^b-1 (mod q); a
  128-bit product folds in 1-2 Solinas passes plus one conditional
  subtract — no 128-bit division. Each reduce_pK is exact over the
  full 128-bit domain (verified by lib/test_reduce). Shoup pre-mul
  (lib/shoup.h) handles twiddle multiplies in the butterfly.

[1;36m  4.2 Size / headroom[0m

  Composite Q = P0·P1·P2·P3 ≈ [1;36m2^255[0m. Effective transform size =
  min(2^32, 2^24, 2^34, 2^40) = [1;36m2^24[0m (P1-limited). A length-n cyclic
  convolution of LIMB_BITS-wide inputs has coefficients < n·(q-1)^2;
  for n ≤ 2^24 and 112-bit input limbs this is < 2^248. Q ≈ 2^255
  covers it with ~7 bits of headroom — no coefficient ever ≥ Q, so the
  CRT result is the exact integer product (no wrap).

[1;36m  4.3 Pipeline (lib/main.hip)[0m

  log_n > 10 — Bailey 4-step Stockham path (production for large n):
  1  bigint_scatter_t: scatter operands transposed (h[c*M+r]=limbs[r*M+c]%p)
     — pre-applies the 1st transpose_sq so the GPU can skip it (I15)
  2  H2D upload to 4 device residue arrays (d_a on stream, d_b on stream_b, A4)
  3  NTT(f) on stream[i] || NTT(g) on stream_b[i] concurrently (A4);
     hipStreamWaitEvent gates step 4+5 on both NTTs completing
  4+5 launch_stockham_intt_hadamard: Hadamard fused into 1st INTT
     butterfly load (I14) + inverse 4-step NTT; final transpose_sq
     eliminated — result stays in transposed layout on device (I15)
  6  Sync all INTT streams (no D2H of residues — GPU Garner reads directly, A1)
  7  garner_reconstruct_gpu: GPU Garner CRT on device 0, reads d_a[0..3] via
     peer access, writes directly to pinned h_out (A1; replaces CPU OpenMP)
  8  bigint_gather_t: untranspose on read, reconstruct result bigint (I15)

  log_n <= 10 — single-block Stockham (small n):
  1  bigint_scatter (linear)  2  H2D (d_a/d_b concurrent, A4)
  3  NTT(f)||NTT(g)  4+5 launch_stockham_intt_hadamard
  6  Sync  7  garner_reconstruct_gpu (A1)  8  bigint_gather (linear)

  Cyclic by default; negacyclic (mod X^n+1) via a 2n-th-root psi
  twist/untwist (main.hip pipeline_negacyclic).

[1;36m  4.4 Why Garner (not explicit CRT)[0m

  Each residue v_i stays in [0, p_i); the mixed-radix Garner
  accumulator needs only ceil(log2 ∏p_i) bits (64/128/192/256) vs a
  full 2·255-bit explicit-CRT multiply-accumulate. 12 precomputed M^-1
  constants (~96 B HBM, data-independent). garner.hip verified exact
  vs exact-bignum CRT over 200k vectors including all-(p-1) edges.

  [1;36mImplementation note (2026-05-25):[0m garner_reconstruct exists in two
  forms — a CPU OpenMP version (kept for the host-only test harness)
  and a GPU kernel garner_reconstruct_gpu (production path). The GPU
  variant runs ~110 ALU ops per coefficient × N coefficients in
  parallel on device 0, reading the 3 peer residue arrays via Infinity
  Fabric (or device-local on a single-device build). ~25× faster than
  the 96-thread CPU Garner at log_n=20.

[1;36m  4.4b GPU scatter / gather (lib/arith/bigint.c)[0m

  bigint_scatter_t (or bigint_scatter for log_n ≤ 10) reduces each
  limb mod the prime and writes in transposed order (I15), eliminating
  the first GPU transpose_sq. bigint_gather_t untransposes the Garner
  U256 output on the CPU (I15), eliminating the last GPU transpose_sq.
  Both run on the host by default. A gated on-device path
  (transfer_kernels.hip, -DNTT_GPU_SCATTER_GATHER) moves them to the GPU
  on MI300A; gather's carry chain — once the reason it stayed on CPU — is
  parallelized there via carry-lookahead (transfer_core.h). Host-verified
  (test_transfer_core, test_e2e_oracle); GPU runtime pending (ROADMAP G10/G11).

[1;36m  4.4c Zero-copy via GFX1030_LOCAL gate[0m

  On MI300A (unified HBM3), the GPU Garner output buffer h_out is
  allocated with hipHostMalloc(..., hipHostMallocNonCoherent); the
  GPU kernel writes directly into host-visible memory and the
  d_out→h_out D2H copy disappears. On the 6900XT dev path (discrete
  VRAM over PCIe) the standard malloc + hipMemcpyAsync path is
  retained. Both code paths live in the same source; the
  GFX1030_LOCAL compile flag selects between them (lib/Makefile
  ARCH_6900XT). See lib/main.hip.

[1;36m  4.5 4-APU / MI300A posture[0m

  Pin each prime's residue in its APU's local HBM3 (cross-domain
  ~50% BW); hipDeviceEnablePeerAccess between exchanging pairs;
  kernel-driven peer copy (~104 GB/s) beats SDMA (~90) for ≥16 MB
  (n ≥ 2^21); lead-APU (Garner gather) role rotatable. MFMA does
  not fit any step (see §10). main.hip's 4-APU orchestration uses
  Cray OpenSHMEM (transfer_shmem.c, oshcc) — links only on the
  MI300A cluster.

  K4 topology parameterization (2026-05-28): The 4 NTT primes are
  always distinct (CRT design); physical device count g_n_devs is
  detected at runtime via hipGetDeviceCount and capped at 4. Prime i
  runs on device ctx[i].dev_id = i % g_n_devs, so a single-device
  dev build routes all 4 primes through device 0 while the MI300A
  K4 path maps each prime 1-to-1. No hardcoded hipSetDevice(0..3)
  remains in main.hip.

  R3 NUMA-aware scatter allocation (2026-05-28): scatter input
  buffers h_a[i] / h_b[i] and h_out are allocated with
  hipHostMalloc(..., hipHostMallocNonCoherent) after hipSetDevice to
  the owning APU, placing them in that APU's HBM domain. Enables
  DMA-direct H2D without a system-memory bounce and improves NUMA
  locality on MI300A.

  R6 per-call log_n (2026-05-28): ntt_bigint_mul computes the
  minimum valid log_n from the actual operand limb counts rather
  than always using the max init-time size. Stage-bucketed twiddle
  tables are prefix-valid for all sub-sizes (CT-DIT stage s has 2^s
  entries, total = N-1), so no re-allocation is needed. Reduces
  scatter, H2D, NTT/INTT, and D2H work for small operands.

  Stockham 4-step block-count fix (2026-05-28): launch_stockham_
  large and launch_stockham_intt (ntt_kernel.hip and the G9 sibling
  ntt_kernel_cswitch.hip) previously used (N1*N2)/1024 for both
  passes. Correct values: row pass = N2 blocks (N2 independent
  sub-NTTs of size N1), col pass = N1 blocks (N1 independent
  sub-NTTs of size N2). The old formula was only correct for
  log_n=20 (N1=N2=1024).

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  5. SHOUP PRECOMPUTED-QUOTIENT MULTIPLICATION[0m

  Used in every butterfly in lib/ntt_kernel.hip. Avoids integer
  division in the hot path (~7 gfx942 instructions vs ~24 naive).

  [1;36mPrecomputation[0m (once per twiddle, during table init):
      w_inv = floor(w * 2^64 / p)

  [1;36mButterfly multiply[0m (w * x mod p):
      hi  = mulhi_u64(w_inv, x)      -- high 64 bits of w_inv * x
      t   = w * x - p * hi           -- exact result; see Shoup 1992
      if t >= p: t -= p              -- at most one correction step

  hi approximates floor(w*x/p) to within 1, so the subtraction is exact
  with at most one conditional reduction. The only expensive op is
  v_mul_hi_u64 (one instruction on gfx942). Defined in lib/shoup.h;
  twiddle tables store (w, w_inv) pairs precomputed by shoup_precompute()
  during the ntt_mul init path.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  6. STOCKHAM LDS PADDING  (STOK_PAD / STOK_STRIDE)[0m

  The Stockham-fused kernel (lib/ntt_kernel.hip) keeps butterfly
  intermediates in LDS ping-pong buffers. Without padding, threads in
  the same wavefront hit the same LDS bank, halving throughput.

  Defined in lib/crt_ntt.h:
      #define STOK_PAD(i)   ((i) + ((i) >> 5))
      #define STOK_STRIDE   1057     /* smallest prime > 1024 + 32 */

  STOK_PAD shifts element i by i/32 positions, spreading 32-element
  groups across distinct banks. STOK_STRIDE is the LDS row stride for
  the ping-pong buffer; a prime stride eliminates all aliasing patterns
  for n ≤ 2^10. For n > 2^10 the kernel falls back to the 4-step
  algorithm (lib/ntt_kernel_fourstep.hip) which uses global-memory
  tiles instead.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  7. COMPILE-TIME ARCH FLAGS  (GFX1030_LOCAL / MFMA_TARGET)[0m

  [1;36mGFX1030_LOCAL=1[0m  (set in lib/Makefile ARCH_6900XT)
    Enables RDNA2 workarounds when compiling for the 6900 XT dev machine:
      * Skips t_atomic benchmark — RDNA2 watchdog hangs the gfx ring without it
      * Swaps hipMalloc to hipMallocManaged in t_fabric — plain hipMalloc
        segfaults under rocprofv3 on RDNA2
      * Enables warpSize=32 paths (CDNA3 wave is 64; RDNA2 is 32)
    Both workarounds are safe to leave OFF on gfx942 (default).

  [1;36mMFMA_TARGET=1[0m  (set in lib/Makefile ARCH_MI300A)
    Reserved for future MFMA (matrix) intrinsic kernel paths on MI300A.
    Currently no production kernel uses MFMA — NTT modular arithmetic
    does not fit the accumulator model (see §10). The flag gates
    experimental paths without modifying shared kernel code; production
    kernels compile correctly with MFMA_TARGET=0 (default).

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  8. MI300A OPTIMIZATION CHECKLIST[0m

  ┌────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ [1;36mConcern[0m            │ [1;36mRule[0m                                                                                                                                     │
  ├────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Wavefront size     │ 64 threads. Block sizes must be multiples of 64.                                                                                         │
  │ Memory coalescing  │ 128-byte cache lines; sequential strongly preferred over strided.                                                                        │
  │ LDS                │ 64 KB per CU @ 5 TB/s. Per-block use ≤ 64 KB.                                                                                            │
  │ Register pressure  │ 512 VGPRs/SIMD × 4 SIMDs/CU. Most NTT kernels at < 50 VGPRs hit the max-waves ceiling, not the VGPR ceiling.                             │
  │ Atomics            │ 912 GOPS/s baseline. Use for synchronization, not high-throughput counting; LDS-reduction is ~8.9× faster on RDNA2 and similar on CDNA3. │
  │ NUMA               │ 4 NUMA domains per node; cross-domain BW ~50% of local. Pin tasks to APU sockets.                                                        │
  │ P2P transfers      │ Kernel-driven > SDMA for > 16 MB transfers.                                                                                              │
  │ Unified memory     │ hipMalloc is zero-copy on MI300A (unified HBM3); hipMallocManaged adds XNACK overhead.                                                   │
  │ Twiddle storage    │ Precompute on CPU in Montgomery form; HBM table; LDS-cache for n ≤ LDS-fit.                                                              │
  │ Reduction strategy │ Lazy/deferred mod when 64-bit headroom permits.                                                                                          │
  │ Hardware constants │ Query at startup via hipGetDeviceProperties.                                                                                             │
  └────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  9. ADVANCED HIP FEATURES  (CDNA3-relevant)[0m

  [1;36mLoop unrolling and ILP[0m
    Use #pragma unroll N on butterfly inner loops to expose independent
    loads and let the compiler issue them in parallel. Each unroll
    factor increases VGPR pressure and lowers occupancy — trade off.

  [1;36mAsynchronous memory and s_waitcnt[0m
    AMDGPU memory loads are asynchronous; the compiler emits
    s_waitcnt vmcnt(0) when it must wait. Inspect
    __builtin_amdgcn_s_waitcnt only if the profiler shows memory stalls.

  [1;36m__launch_bounds__(MAX_THREADS, MIN_BLOCKS)[0m
    Constrains VGPR allocation and gives the compiler an occupancy
    target. Apply to every __global__ kernel in this project.

  [1;36mLDS bank-conflict avoidance[0m
    Pad shared arrays by 1 element to avoid 32-way conflicts:
       __shared__ uint64_t tile[TILE][TILE + 1];
    The Stockham-fused kernel uses LDS_PAD(i) = i + (i >> 5).

  [1;36mMFMA matrix instructions[0m
    CDNA-only (gfx908+). Accumulators in fp32 / fp64 / int32; NTT
    modular arithmetic does not fit (see §10). Probe at
    ~/mi300a-dev/mfma_probe.hip.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  10. DESIGN RATIONALE  (measured assumptions)[0m

  These are the load-bearing facts behind the kernel and
  data-movement choices. All measured on a 6900 XT (gfx1030) dev box
  unless noted; the MI300A scaling factors are forecast inputs for
  Phase 4. See PERFORMANCE.md for the corresponding numeric baseline.

  ┌────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ [1;36m#[0m  │ [1;36mFact[0m                                                                                                                                                                                                              │
  ├────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 1  │ GPU kernel fusion (NTT+Hadamard+INTT in one launch) is the highest-value optimization: 9.80× vs separate 4-launch (gfx1030).                                                                                      │
  │ 2  │ LDS-fused single-kernel Stockham beats multi-launch 4.99× @n=256, 1.83× @n=2048. Implemented as stockham_lds_fused_kernel.                                                                                        │
  │ 3  │ MFMA does not apply to modular NTT: 24-bit fp32 mantissa cannot hold products for q ≥ 14 bits; int32 accumulator cannot hold 64-bit modular products. Adjacent algorithms (BKZ INT8, FP conv) can — out of scope. │
  │ 4  │ Per-thread atomicAdd to one counter loses 8.91× vs LDS-reduce on RDNA2. Codebase avoids this pattern.                                                                                                             │
  │ 5  │ gfx942 has 1.78–3.20× higher occupancy than gfx1030 same kernel, on top of 2.85× CU ratio.                                                                                                                        │
  │ 6  │ HBM3 effective BW on MI300A = 4.0 TB/s (76% of 5.3 spec). Use 4.0 in roofline models.                                                                                                                             │
  │ 7  │ Streams concurrency flat on RDNA2 (mem-saturated by 1). MI300A should scale to the memory ceiling.                                                                                                                │
  │ 8  │ Stockham (no bit-reversal) beats CT-DIT ~20% on CPU; larger on GPU.                                                                                                                                               │
  │ 9  │ Twiddles precomputed in HBM cheaper than recompute for n > LDS-fit; LDS-cached for small n.                                                                                                                       │
  │ 10 │ Atomic latency 100 ns/op single-thread (6900XT); MI300A 912 GOPS aggregate. Use atomics as sync primitives, not counters.                                                                                         │
  │ 11 │ INT32 MI300A 14,001 Gops/s; Shoup butterfly ~7 instr → 745 GIOPS = dep-chain floor (~37% utilization).                                                                                                            │
  │ 12 │ Zero-copy (CPU reads GPU HBM, unified mem) 371.3 GB/s. Garner reads 4N·8 B; ~0.17 ms @ N=2^20.                                                                                                                    │
  │ 13 │ Sustained all-to-all P2P 650 GB/s (60.2%), not 978 (burst). Use 650 for sustained multi-NTT.                                                                                                                      │
  │ 14 │ P2P unidir SDMA/link = 66.63 GB/s (74% of 90 spec; 90 is DUPLEX). Use 66.63 for broadcast_input().                                                                                                                │
  │ 15 │ MI300A binaries MUST launch via srun (cgroup device-access).                                                                                                                                                      │
  └────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  11. PROFILING TOOLS  (AMD)[0m

  ┌──────────────────────────────┬─────────────────────────────────────────────────────┐
  │ [1;36mTool[0m                         │ [1;36mUse[0m                                                 │
  ├──────────────────────────────┼─────────────────────────────────────────────────────┤
  │ rocminfo                     │ list architectural properties of every agent        │
  │ rocm-smi                     │ live telemetry: temp / power / clocks / topology    │
  │ rocm-smi --showtopo          │ APU-to-APU peer link map                            │
  │ rocprofv3 --stats            │ kernel timing summary                               │
  │ rocprofv3 --kernel-trace     │ per-launch trace (ns granularity)                   │
  │ rocprofv3 --hip-trace        │ HIP API trace (memcpy / sync / launch)              │
  │ rocprofv3 (PMC counters)     │ VALU / SALU / VMEM / LDS / cache; gfx942 full set   │
  │ llvm-objdump --mcpu=...      │ disassemble a HIP kernel ELF                        │
  │ llvm-readelf --notes         │ extract AMDGPU metadata (VGPR/SGPR/AGPR/LDS counts) │
  │ /opt/rocm-7.0.3/lib/llvm/bin │ canonical path for these tools (not /opt/rocm/bin/) │
  └──────────────────────────────┴─────────────────────────────────────────────────────┘

[1;31m  ●[0m NEVER SIGKILL a running rocprofv3. SIGTERM only; wait for clean exit.
    SIGKILL leaves stuck GPU queues that the kernel driver later resets,
    blackscreening the desktop on display+compute cards (e.g. 6900 XT).
  [1;31m●[0m NEVER loop rocprofv3 across binaries unsupervised. Cumulative load
    can trigger a reset even when no individual call would. Insert ≥ 8 s
    cooldown plus a rocm-smi --showtemp health check between calls.
  Launch all GPU binaries via scripts/gpu_run.sh — it enforces both rules
  plus a single-job flock and a crash-aware guard. See scripts/gpu_run.sh
  header for the full safeguard list.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  12. REFERENCES[0m

  ┌───────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ [1;36mTopic[0m             │ [1;36mKey citations[0m                                                                                         │
  ├───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ NTT algorithms    │ Cooley & Tukey 1965; Stockham 1966; Pollard 1971; Bailey 1989; Harvey 2014; Plantard 2021; Shoup 1995 │
  │ Modular reduction │ Barrett 1986; Montgomery 1985; Solinas 1999                                                           │
  │ Post-quantum      │ FIPS 203 (ML-KEM); FIPS 204 (ML-DSA); FIPS 206 (Falcon)                                               │
  │ AMD CDNA3/MI300A  │ MI300 ISA (2024); CDNA3 white paper (2023); ROCm 7 docs                                               │
  │ HIP / GPU         │ AMD HIP Programming Guide; chipsandcheese CDNA/RDNA deep-dives                                        │
  └───────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  13. SEE ALSO[0m

  README.md                       project entry, build/run quickstart
  PERFORMANCE.md                  current measured baselines per target
  ROADMAP.md                      Phase status, open MI300A tasks, future work
  ref/README.md, lib/README.md, app/README.md      per-layer overviews
  ~/MI300A_TARGET_ENVIRONMENT.md  hardware fixed points + measured baselines
  ~/HIP_6900XT_KNOWLEDGE.md       6900 XT dev-side architectural lessons
  ref/ntt_moduli.h               source-of-truth for modulus dispatch
  lib/primes.h                   source-of-truth for the four CRT primes
