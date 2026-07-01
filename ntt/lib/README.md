[1;37m╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗[0m
[1;37m║                               L I B   —   4 - P R I M E   C R T - N T T   E N G I N E                                ║[0m
[1;37m╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat lib/README.md   or   less -R lib/README.md[0m

  lib is the core distributed NTT engine for the MI300A K4 node. It performs
  large polynomial multiplication modulo a ~255-bit composite using Chinese
  Remainder Theorem (CRT): one 64-bit residue NTT per APU, Garner reconstruction
  on CPU, result as a U256 (256-bit little-endian integer).
  All binaries require MI300A hardware and must be launched via srun.

  TWO ENTRY POINTS, one shared engine (do not re-fork — see F6):
    - crt_ntt (main.hip): the distributed polynomial-multiply pipeline,
      OpenSHMEM-collective across the K4 node (multi-node capable). The product.
    - arith/ (ntt_bigint_mul): a single-node big-integer multiply API
      (bigint x bigint -> bigint) layered on the same engine, with Newton
      division + decimal conversion. GPU scatter/NTT/Garner with a host-side
      gather; intra-node multi-APU (round-robin over local devices), NOT
      OpenSHMEM. This is what app/ consumes.

  CANONICAL PLAN: the authoritative, consolidated 4-prime ~255-bit
  CRT-NTT plan (primes, size/headroom math, pipeline, Garner
  rationale, 4-APU posture) is ARCHITECTURE.md §4. This file is the
  lib-local build/usage reference; ARCHITECTURE.md §4 wins on conflict.

[1;37m════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  1. ARCHITECTURE[0m

  The composite modulus is Q = P0 * P1 * P2 * P3 ≈ 2^255.
  Each APU (device 0-3) owns one residue lane:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ [1;36mDevice[0m │ [1;36mPrime[0m           │ [1;36mValue[0m   │ [1;36mMax NTT[0m │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ 0        │ P0 = 2^64 - 2^32 + 1       │ Goldilocks         │ 2^32       │
  │ 1        │ P1 = 2^64 - 2^24 + 1       │ prim root 43       │ 2^24       │
  │ 2        │ P2 = 2^64 - 2^34 + 1       │ prim root 10       │ 2^34       │
  │ 3        │ P3 = 2^64 - 2^40 + 1       │ prim root 19       │ 2^40       │
  └─────────────────────────────────────────────────────────────────────────┘

  Effective CRT size: min(2^32, 2^24, 2^34, 2^40) = 2^24 (limited by P1).
  For n <= 2^24, Q ≈ 2^255 comfortably covers products up to n * (q-1)^2 < 2^248
  for 112-bit inputs.

[1;37m════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  2. FILE MAP[0m

  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ [1;36mFile[0m              │ [1;36mRole[0m                                                                          │
  ├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ primes.h                     │ Prime constants, Solinas reduction functions, Montgomery mul                             │
  │ shoup.h                      │ ShoupPair struct + shoup_mul<PIDX>: precomputed-quotient butterfly mul                   │
  │ crt_ntt.h                    │ Shared types: U256, GarnerConsts, STOK_PAD/STOK_STRIDE macros                            │
  │ ntt_kernel.hip               │ GPU kernels: CT-DIT, 4-step, Stockham LDS-fused, Hadamard, twist/untwist                 │
  │ garner.hip                   │ Garner CRT reconstruction — GPU kernel (production) + CPU OpenMP (host test path)        │
  │ transfer_shmem.c             │ OpenSHMEM inter-node collective distribute/collect                                       │
  │ main.hip                     │ Pipeline: init/pipeline/benchmark/CLI + broadcast_input() SDMA APU0->1,2,3               │
  │ test.hip                     │ 8-test correctness suite (runs on 6900 XT single-device too)                             │
  │ isa_check.hip                │ ISA probe: verifies v_lshl_add_u64 in gfx942 reduce_p1                                   │
  │ arith/bigint.c/h             │ 256+ bit integers: alloc, add, sub, scatter/gather, limb ops                             │
  │ arith/multiply.c/h           │ BigInt multiply: schoolbook for small; NTT-via-ntt_mul for large                         │
  │ arith/newton.c/h             │ Newton reciprocal: 53-bit float seed -> full-precision iterate                           │
  │ arith/base_convert.c/h       │ D&C decimal conversion using cached pow10 + Newton reciprocals                           │
  │ transfer_core.h              │ Per-element scatter/gather core logic; host+device inline (kernels + oracle share it)    │
  │ transfer_kernels.hip         │ GPU scatter/gather/CLA kernels + launchers (host-verified; gated pipeline wire-in done)  │
  │ arith/test_transfer_core.c   │ Host oracle: cores vs bigint_scatter[_t]/gather[_t]; dual-width, no GPU                  │
  │ arith/test_e2e_oracle.c      │ Host oracle: CRT-NTT multiply (cyclic) + negacyclic mod X^n+1 vs GMP; dual-width         │
  └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

[1;37m════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  3. MEMORY MODEL[0m

  lib uses hipMalloc (NOT hipMallocManaged) for all device arrays.
  On MI300A, hipMalloc pointers are in the shared HBM3 pool and are
  accessible from both CPU and GPU via Infinity Fabric without memcpy.
  hipMallocManaged adds XNACK page-fault overhead; hipMalloc is faster.

  Output U256 buffer (compile-time gated via GFX1030_LOCAL):
    GFX1030_LOCAL=1  → standard host malloc + device d_out + D2H copy
    GFX1030_LOCAL=0  → hipHostMalloc(HostMallocNonCoherent); h_out and
                       d_out alias the same unified buffer, GPU writes
                       land directly in host-visible memory, no D2H.

  Memory layout per APU (N = 2^log_n elements):
    d_f, d_g   : N x uint64_t = 8N bytes residue input arrays (per device)
    d_tw       : N/2 x ShoupPair = 16*(N/2) bytes forward twiddles
    d_tw_inv   : N/2 x ShoupPair inverse twiddles
    d_tw_cross : N/2 x ShoupPair 4-step inter-pass twiddles
    d_tw_stok  : flat Stockham table (same count as d_tw)

  For N=2^24: 8*2^24 = 128 MB per residue array. Total per APU ~1 GB.
  Four APUs x ~1 GB = ~4 GB total NTT workspace. Well within 96 GB HBM3.

[1;37m════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  4. BUILD[0m

  Load modules (Cray PE on MI300A node):
    module load PrgEnv-amd/8.6.0 rocm/7.0.3 craype-accel-amd-gfx942 \
                cray-mpich/9.0.1 cray-shmem/12.0.0 gcc/14.3.0

  Build targets:
    cd lib
    make all          # builds crt_ntt + test_ntt
    make isa-check    # compiles isa_check.hip --save-temps; greps for v_lshl_add_u64
    make arith        # builds arith/ objects only
    make clean

  Host-only verification (no GPU; runnable on the CPU dev box):
    make test-transfer-core    # scatter/gather + carry-lookahead cores vs bigint ref (l64+l112)
    make test-e2e-oracle       # full scatter->NTT->CRT->gather pipeline vs GMP (l64+l112)
    make transfer-kernels-isa  # cross-compile transfer_kernels.hip for gfx942 (0 warnings)

  Runtime environment (set before srun):
    export ROCM_PATH=/opt/rocm-7.0.3
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
    export HSA_ENABLE_SDMA=1
    export ROCR_VISIBLE_DEVICES=0,1,2,3
    export HIP_VISIBLE_DEVICES=0,1,2,3

[1;37m════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  5. RUNNING[0m

  [1;31mDirect ./binary fails — all execution must go through srun.[0m

    srun -p mi300a -N 1 -n 1 ./test_ntt            # 8-test correctness suite
    srun -p mi300a -N 1 -n 1 ./crt_ntt 20 100      # log_n=20, 100 trials
    srun -p mi300a -N 1 -n 1 ./crt_ntt 20 100 --stockham
    srun -p mi300a -N 1 -n 1 ./crt_ntt 20 100 --montgomery
    srun -p mi300a -N 2 -n 2 --ntasks-per-node=1 ./crt_ntt 22 50  # multi-node

  OMP flags for Garner reconstruction:
    OMP_NUM_THREADS=96 OMP_PROC_BIND=spread OMP_PLACES=cores srun ...

[1;37m════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  6. PIPELINE FLOW (single node, N-point polynomial multiply f*g)[0m

  1. Upload   hipMemcpyAsync host -> APU 0 HBM (f and g together).
              On MI300A: zero-copy via hipHostMalloc'd buffers when the
              caller opts in (GFX1030_LOCAL=0 build).
  2. Broadcast SDMA hipMemcpyPeerAsync APU 0 -> APUs 1,2,3 for ≤16 MB
              transfers; kernel-driven peer copy (peer_copy_kernel — planned, G7)
              for ≥16 MB (~104 GB/s vs 66.6 SDMA).
  3. NTT(f) || NTT(g) via launch_stockham_ntt per APU. For log_n > 10
              (4-step Stockham), the initial transpose_sq is eliminated
              because scatter wrote in transposed order (I15). For
              log_n ≤ 10, single-block Stockham.
  4+5. Hadamard + INTT  launch_stockham_intt_hadamard fuses the
              pointwise Hadamard into the first INTT butterfly load
              (I14). The final transpose_sq is eliminated — output
              stays in transposed order for gather to untranspose (I15).
  6. Sync      hipStreamSynchronize × 4
  7. Garner    GPU garner_reconstruct_gpu on device 0 reads all four
              residue arrays (device-local or peer-mapped) and writes
              U256 result. On MI300A with hipHostMalloc'd output the
              host sees the result directly; on 6900XT a final D2H
              copies device d_out to host h_out.

  Negacyclic variant (mod X^N + 1): adds launch_twist before NTT and
  launch_untwist after INTT using precomputed psi (2N-th root of unity).

[1;37m════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  7. KEY ALGORITHM DETAILS[0m

  [1;36mShoup precomputed-quotient multiplication (butterfly hot path)[0m
    For each twiddle w mod p, precompute: w_inv = floor(w * 2^64 / p).
    Butterfly: t = w*x - p * hi64(w_inv * x).
    hi64() is a single mulhi instruction; no 128-bit divide in the loop.
    Cost: ~7 gfx942 instructions vs ~24 for naive __uint128_t % p.

  [1;36mStockham LDS-fused kernel (alternative to CT-DIT for n <= 2^10)[0m
    All butterfly stages execute in a single kernel launch using LDS ping-pong.
    STOK_PAD(i) = i + (i >> 5) avoids LDS bank conflicts every 32 elements.
    STOK_STRIDE = 1057 (= 1024 + 32 + 1) is the padded row width.
    For n > 2^10: falls back to multi-launch 4-step matrix NTT.

  [1;36mGarner reconstruction (CPU after INTT)[0m
    Given residues r0..r3 mod P0..P3, recovers x in [0, Q) as U256:
      a0 =  r0
      a1 = (r1 - a0) * inv(P0, P1)                              mod P1
      a2 = ((r2 - a0 - a1*P0) * inv(P0,P2)) * inv(P1,P2)       mod P2
      a3 = (((r3 - a0 - a1*P0 - a2*P0*P1) * inv(P0,P3))...) * inv(P2,P3) mod P3
      x  = a0 + a1*P0 + a2*P0*P1 + a3*P0*P1*P2   (256-bit integer)
    GarnerConsts precomputes all inverses and multi-precision products once.
    OpenMP parallelism: each coefficient is independent.

  [1;36mSolinas reduction (all four primes have form 2^64 - 2^b + 1)[0m
    For 128-bit product x = x_hi*2^64 + x_lo:
      x mod p ≡ x_lo + x_hi*(2^b - 1)  (since 2^64 ≡ 2^b - 1 mod p)
    Iterates until result fits in 64 bits. No division.
    v_lshl_add_u64 (gfx942 ISA) encodes the 2^b shift+add in one instruction.

[1;37m════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  8. DEVIATIONS FROM CLAUDE.md "C ONLY" RULE[0m

  lib uses C++ templates (template<int PIDX>) throughout ntt_kernel.hip,
  garner.hip, and main.hip. This is intentional: compile-time PIDX bakes the
  prime index as an ISA immediate, eliminating runtime branching in the
  butterfly inner loop and enabling full __forceinline__ expansion.
  Equivalent C would require function pointers that defeat inlining.
  Host code in arith/ and transfer_shmem.c is pure C.

[1;37m════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  9. OPEN TASKS (require MI300A)[0m

  G2  Full polymul correctness test (random inputs, verify CRT result is correct)
  G6  rocprofv3 PMC: HBM utilization, ALU utilization, occupancy per kernel
  G7  Kernel-driven P2P benchmark at N >= 2^23 (peer_copy_kernel — planned, G7; verify ~104 vs 66.6 GB/s SDMA on 4-APU hw)
  G8  Shared twiddle table on device 0 + peer access (saves 3/4 HBM at N=2^24)
  G9  Stockham 4-step at log_n=16 — investigate the Newton-iteration
      regression that caused the production path to revert to CT-DIT
      (see ROADMAP §6 known gaps).

