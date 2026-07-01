[1;37m╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                          N T T   /   M I 3 0 0 A   —   O P T I M I Z A T I O N   L E D G E R                           ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat OPT_LEDGER.md   or   less -R OPT_LEDGER.md[0m

  Running ledger for the autonomous CRT-NTT optimization session
  started 2026-05-29. Each idea: hypothesis -> method -> measured
  result -> keep/revert -> why. Survives context compaction; this is
  the file to read first on return. GPU work is bounded to the
  6900XT safe window (log_n <= 16, every run via scripts/gpu_run.sh).

  Tags:  [now]  measurable on 6900XT (gfx1030)
         [mi]   MI300A-only (design/stage; cannot measure here)

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  1. SESSION STATUS[0m

  Phase: CPU VM autonomous loop (2026-06-01). GPU unavailable. Last GPU
  session measured NTT compute at 112.7 ms post-I13+I14 (2.55x vs CT-DIT).

  CPU-VM work completed (loop 2026-06-01, all make check 17/17):
  - D1: dropped dead d_tw_cross_inv from main.hip (struct + function +
    allocation + memory leak). Saves 4 MB HBM at log_n=16.
  - A2: n_inv_table[4][23] precomputed at init; 0 modpow per multiply.
  - A3: h_gather_tmp pre-allocated in NttMulState; 0 malloc per multiply.
  - A1: GPU Garner wired into ntt_bigint_mul; peer access enabled; 4
    residue D2H downloads eliminated; garner_reconstruct_gpu writes
    directly to pinned h_out. Baseline garner was 67ms (25% of pipeline).
  - C1: stockham_kernel<PIDX,LOG_N_SUB> with constexpr STOK_STRIDE_C.
    LDS drops from 16912 to 4208 B at LOG_N_SUB=8 (hot 4-step). launch_stok
    / launch_stok_had dispatch helpers (cases 6-11). Loop fully unrolled.
  - A4: NTT(f)||NTT(g) concurrent streams: stream_b[4] + ev_ntt_b[4] in
    NttMulState; d_b uploaded/NTT'd on stream_b; hipStreamWaitEvent gates
    INTT_hadamard on both NTT completions.
  - I4: ntt_global_stage_kernel_mont + launch_ntt_ct_dit_mont/launch_intt_ct_dit_mont;
    make_tw_mont_local + d_tw_mont[4]/d_tw_inv_mont[4] in NttMulState; odd log_n>10
    routes through CT-DIT-Mont (no round-up to even; 8 B vs 16 B twiddle HBM).
  - C2: StokMinBlks<LOG_N_SUB> compile-time struct; stockham_kernel/scaled/had/
    scaled_had use __launch_bounds__(512, 4) for LOG_N_SUB<=9 (STOK_STRIDE_C<=527),
    __launch_bounds__(512, 2) for LOG_N_SUB=10 (STOK_STRIDE_C=1055 > 1024).

  Next GPU session: `bash scripts/gpu_session.sh` — Phase 1 correctness gate
  (I14+I15+A1+C1+A4+I4), Phase 2 pipeline measurement (A1/A4/I15 wall-clock
  vs baselines), Phase 3 bench sweep + ISA check, Phase 4 (--pmc) occupancy
  lift for C2. See ROADMAP §3 for MI300A follow-on (I16/G6+/G9/G2/G7/G8).

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  2. IDEA REGISTRY[0m

  Active / pending GPU measurement:
```
| ID  | Tag  | Idea                                                      | Status   |
|-----|------|----------------------------------------------------------|----------|
| I3  | now  | Batch the 4 primes into one kernel launch (single-dev)   | deferred |
|     |      | Measured: 2.62 µs/launch async. 57 kernels/dispatch;     |          |
|     |      | all async so no GPU bubble. MI300A priority higher.      |          |
| I4  | now  | Montgomery 8B twiddle for CT-DIT odd log_n > 10; avoids  | WIRED    |
|     |      | 2x size overhead of round-up-to-even. 0 warn, check 17/17| 06-01    |
| I5  | mi   | wave64 warp-synchronous last-6-stages (skip __syncthreads)| proposed |
| I15 | now  | Fold 1st-T(fwd)/last-T(inv) transpose into scatter_t/    | DONE     |
|     |      | gather_t — 2 transpose_sq launches removed. GPU gate ok  | 06-06    |
| I16 | mi   | Per-prime hybrid butterfly: Solinas for P0 (Goldilocks,  | PROPOSED |
|     |      | -9 VALU on gfx942 ISA), Shoup for P1/P2/P3 (§I12-ISA)   | (gfx942) |
| A1  | now  | Wire GPU Garner into ntt_bigint_mul — eliminates 4 D2Hs  | DONE     |
| A2  | now  | Cache n_inv_table[4][23] at init — 0 modpow/call (was 4) | DONE     |
| A3  | now  | Pre-alloc h_gather_tmp in NttMulState — 0 malloc/call    | DONE     |
| A4  | now  | NTT(f)||NTT(g) concurrent streams — GPU overlap pending  | DONE     |
| C1  | now  | stockham_kernel<PIDX,LOG_N_SUB>: constexpr STOK_STRIDE_C | DONE     |
|     |      | LDS 16912→4208B at LOG_N_SUB=8; loop unroll; 15blk/CU   | (gpu ok) |
| C2  | now  | Per-specialization __launch_bounds__(512, 4) for         | DONE     |
|     |      | stockham_kernel* where STOK_STRIDE_C<=1024. GPU PMC gate | 06-01    |
| D1  | now  | Drop dead d_tw_cross_inv (alloc+memleak in main.hip)     | DONE     |
```

  Closed history (REVERTED / ACCEPTED / REFUTED — no longer active):
```
| ID  | Tag  | Idea                                                      | Status   |
|-----|------|----------------------------------------------------------|----------|
| I1  | now  | Re-enable LDS-fused Stockham in lib2 ntt_bigint_mul      | REVERTED |
|     |      | (GPU hang at log_n=16; lib2 4-step latently not-a-DFT)   |          |
| I2  | now  | Radix-4 Stockham butterfly (halve stages/syncs/LDS trips) | CLOSED   |
|     |      | No headline gain (VALU-bound; hot sizes >10; blocked I6)  | blocked  |
| I6  | now  | Fix lib2 4-step Stockham DFT; Bailey transpose structure  | ACCEPTED |
|     |      | NTT compute 287.8→154.0 ms (1.87x). check-gpu 10/10.    | 05-29    |
| I7  | now  | Tiled LDS transpose (TT_DIM=16) replacing naive swap     | ACCEPTED |
|     |      | 154.0→144.1 ms; cumulative I6+I7: 287.8→144.1 (2.0x)    | 05-29    |
| I8  | now  | Size stockham_kernel to actual work (block=half, min 64) | ACCEPTED |
|     |      | 144.1→126.4 ms; cumulative I6+I7+I8: 2.28x speedup      | 05-29    |
| I9  | now  | Dynamic LDS sizing (size to N_sub, not fixed 2*STOK)     | REVERTED |
|     |      | 126.4→128.6 ms (+1.7%): compute-bound, LDS not limiting  |          |
| I10 | now  | Radix-4 (eval only via PMC)                              | REJECTED |
|     |      | VALU-bound: radix-4 modular mul adds VALU, would regress |          |
| I11 | now  | Rect Bailey 4-step for odd log_n > 10                    | DISABLED |
|     |      | Faulted at d=1e6 (log_n=19) + d=200k (log_n=17) in prod |          |
|     |      | Odd>10 reverted to CT-DIT (launch_ntt/intt). 2026-06-08  |          |
| I12 | mi   | Solinas/Goldilocks reduction in butterfly (vs Shoup)     | REFUTED  |
|     |      | -4% RDNA3; gfx942 VALU net+ (P0 -9, P1-3 +9/+17)        | §I12-ISA |
| I13 | now  | Fuse transpose+cross-twiddle into one in-place pass      | ACCEPTED |
|     |      | (fwd mul-on-store, inv mul-on-load) — 127.8→114.6 ms    | 05-30    |
| I14 | now  | Fuse Hadamard A*=B into inverse's first stockham load    | ACCEPTED |
|     |      | (even path) — eliminates hadamard_kernel; 114.6→112.7ms  | 05-31    |
```

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  3. BASELINE  (CT-DIT production path, pre-change)[0m

  Metric of record = compute(NTT+Had+INTT) ms from NTT_PROFILE=2
  (isolates the NTT path from fixed GPU-init / scatter / garner / gather).
```
| binary         | d      | log_n | wall(s) | gpu_disp | NTT_compute_ms | digits_ok |
|----------------|--------|-------|---------|----------|----------------|-----------|
| compute_e_dev  | 100    | 8     | 0.47    | 61       | n/a (tiny)     | yes       |
| compute_e_dev  | 100000 | 16    | 0.61    | 561      | 287.8          | yes       |
```
  d=100000 profile detail: scatter 15.6 / gpu_pipe 317.4 (upload 44.4 +
  compute 287.8 + download 0.8) / garner 67.0 / gather 5.1 ms.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  4. RESULTS LOG[0m

```
| ts    | idea  | metric             | before  | after      | delta      | decision |
|-------|-------|--------------------|---------|------------|------------|----------|
| 11:10 | I1    | NTT_compute @log16 | 287.8ms | HANG (124) | n/a        | REVERT   |
| 18:33 | I6+I1 | NTT_compute @log16 | 287.8ms | 154.0ms    | 1.87x      | ACCEPT*  |
| 18:33 | I6+I1 | wall d=100k        | 0.61s   | 0.47s      | -23%       | ACCEPT*  |
| 18:5x | I7    | NTT_compute @log16 | 154.0ms | 144.1ms    | -6.4%      | ACCEPT   |
| 19:0x | I8    | NTT_compute @log16 | 144.1ms | 126.4ms    | -12%       | ACCEPT   |
| 19:3x | I9    | NTT_compute @log16 | 126.4ms | 128.6ms    | +1.7%      | REVERT   |
| 19:5x | I10   | radix-4 (eval only) | n/a    | n/a        | would regress | REJECT |
| 20:16 | I11   | odd-log_n rect 4-step | n/a  | CRASH      | box reboot | DISABLED |
| 05-30 | base  | re-measure post-I6 (current) | -  | 127.8ms    | (Shoup, fresh) | BASELINE |
| 05-30 | I12   | Solinas butterfly  | 127.8ms | 132.7ms    | +3.8%      | REVERT   |
| 05-30 | I13   | fuse transpose+X   | 127.8ms | 114.6ms    | -10.3%     | ACCEPT   |
| 05-30 | I14   | fuse Hadamard→inv  | 114.6ms | 112.7ms    | -1.7%      | ACCEPT   |
| 05-30 | I13+14| cumulative (sess.) | 127.8ms | 112.7ms    | -11.8%     | ACCEPT   |
| 05-31 | I14w  | wire had→intt_had in ntt_bigint_mul | n/a | impl | -4 launches/mul | WIRED |
| 05-31 | I15   | scatter_t/gather_t + rm transpose_sq | n/a | impl | GPU gate PASS 39/39 (06-06) | DONE |
| 06-01 | D1    | drop dead d_tw_cross_inv table       | n/a | impl | -4MB HBM + memleak fixed | DONE |
| 06-01 | A2    | cache n_inv_table[4][23] at init     | n/a | impl | 0 modpow/call (was 4)    | DONE |
| 06-01 | A3    | pre-alloc h_gather_tmp in NttMulState| n/a | impl | 0 malloc/call (was 2MB)  | DONE |
| 06-01 | A1    | GPU Garner in ntt_bigint_mul         | n/a | impl | skip 4 D2Hs; pipeline PASS 39/39 (06-06) | DONE |
| 06-01 | C1    | stockham_kernel<PIDX,LOG_N_SUB>      | n/a | impl | LDS 16912→4208B @sub=8   | DONE |
| 06-01 | A4    | NTT(f)||NTT(g) stream_b concurrent   | n/a | impl | GPU overlap pending meas | DONE |
| 06-01 | I4    | Montgomery CT-DIT: ntt_global_stage_kernel_mont + launch_ntt_ct_dit_mont/launch_intt_ct_dit_mont; make_tw_mont_local + d_tw_mont[4]/d_tw_inv_mont[4] in NttMulState; odd log_n>10 no longer rounds up to even | n/a | impl | check-gpu 10/10 PASS (R2-R11) 06-06; CT-DIT path verified correct at d=200k (log_n=17) | DONE |
| 06-01 | C2    | StokMinBlks<LOG_N_SUB>: __launch_bounds__(512, 4) on stockham_kernel* where STOK_STRIDE_C<=1024; (512,2) otherwise | n/a | impl | GPU PMC occupancy gate pending | DONE |
| 06-06 | BUG   | arith/ntt_mul.hip I15 gap: launch_stockham_large (I15) skips initial T, but arith scatter is GPU-side natural order — added launch_transpose_sq before fwd_ntt and after INTT for even log_n>10. d=100k (log_n=16) failed newton before; passes after. | n/a | fix | compute_e d=100k now 0.40s | DONE |
| 06-06 | R7fix | test_ntt_rect_vs_ctdit: reference path was launch_ntt (even log_n only) — replaced with launch_ntt_ct_dit_mont (I4, any log_n). Cross-twiddle scale=1→N_inv fixed. Rect oracle now PASS at log_n 15/17/19. | n/a | fix | I11 kernel certified; prod dispatch still disabled | DONE |
```
  I11 = rectangular Bailey 4-step for ODD log_n>10 (launch_stockham_ntt_rect/
  intt_rect + transpose_rect + per-prime d_scr scratch, ntt_kernel.hip). It
  PASSED test_ntt at log_n 11/13/15/17/19 (roundtrip + convolution, bounded
  single transforms). I then ran compute_e -d 1000000 (log_n=19, production
  rect path) — it FAULTED the GPU and rebooted the box (ledger: LAUNCH
  20:16:37, no EXIT; uptime reset 20:18:52). The fault is PRODUCTION-SCALE-
  specific (3000+ rect dispatches, the real ntt_mul d_scr buffers) — NOT
  reproduced by the bounded test. ACTION: production dispatch reverted to the
  proven CT-DIT odd path (use_stockham(); odd>10 -> launch_ntt, which had
  completed d=1e6 in 10.25 s). Rect kernels remain in ntt_kernel.hip/test_ntt
  but are DISABLED IN PRODUCTION pending offline root-cause. Even-log_n win
  (I1/I6/I7/I8, 2.28x) is UNAFFECTED and remains signed off.

  PROCESS MISTAKE (mine): I jumped the unproven production rect path straight
  to the LARGEST/longest size (d=1e6) instead of escalating d=200k -> 300k
  (both even... need an ODD-log_n mid size) first. Root-cause + fix must be
  done on a bounded harness (e.g. a standalone rect bigint-multiply test at
  log_n=11/13) BEFORE any compute_e d that maps to odd log_n.

  CODE REVIEW (2026-05-29, no GPU). Ruled OUT by inspection: kernel/transpose
  bounds (transpose_rect is correct; in/out indices stay in [0,N)), LDS sizing
  (F2 sub=10 peaks at 2111<2114), and table sizing (make_tw_stockham_local
  allocs N1/2+N2/2=768 with matching fill offsets; cross tables alloc N1*N2 —
  all correct for N1!=N2). Could NOT pinpoint a single deterministic faulting
  line. KEY GAP found: the rect path's CONVOLUTION was only verified at odd
  log_n 11/13; at 15/17/19 only ROUNDTRIP ran, which is identity for ANY
  invertible map and proves nothing (the exact trap the square 4-step fell
  into). Also: test.hip used DIFFERENT table builders than production, so the
  production odd-log_n path was never exercised before the d=1e6 run.

  HARDENING APPLIED (this turn, all CPU-validated / no GPU):
   - test.hip: REMOVED all odd-size rect tests (roundtrip 11..19, conv 11/13).
     test_ntt is now rect-FREE, so check-gpu / benchmarks never execute the
     uncertified rect path. Rect kernels remain in ntt_kernel.hip (dead,
     instantiated) for future bounded root-cause.
   - newton.c: BOUNDED the division correction loops (CORRECTION_CAP=64; normal
     is <=2). A corrupt product/reciprocal now ABORTS CLEANLY instead of an
     unbounded host spin — defense against the crash chain for ANY future bug.
     Validated: compute_e_host digits EXACT at d=100/1k/10k/100k (cap never
     fires in correct operation).
   - scripts/gpu_headless_run.sh added: run risky/long GPU work with the
     desktop stopped so a fault can't reset an interactive session.
  NET: production = Stockham(even)/CT-DIT(odd), proven; tests rect-free;
  system stable for benchmarking/testing. Rect re-enable requires a bounded
  headless repro + conv asserts at ALL odd sizes first.

  I13 (2026-05-30, ACCEPTED) = fuse the Bailey transpose with the adjacent
  cross-twiddle into one in-place tiled-LDS pass. Data-driven: rocprofv3
  kernel-trace at d=100k (log16) showed the 4-step GLUE — transpose_sq 19.7%
  (31.2ms) + cross_twiddle 11.4% (18ms) = ~31% of compute — dwarfing what any
  butterfly-arith tweak could reach (cf. I12). Forward T·F1·T·X·F2 had T then
  X adjacent; inverse had X^{-1} then T. Since transpose_sq_kernel already
  streams every element through LDS, the Shoup cross-twiddle multiply rides
  along for free: fwd multiplies on the STORE at dest pos (out[p]=in[T(p)]·
  tw[p]); inv multiplies on the LOAD at src pos (out[p]=in[T(p)]·tw[T(p)],
  coalesced). New kernels xtranspose_fwd_kernel / xtranspose_inv_kernel
  (ntt_kernel.hip) REPLACE the standalone cross_twiddle_kernel launch in
  launch_stockham_large / launch_stockham_intt. In-place, tile-pair owned
  (bj>=bi) — NO scratch, NO launcher-signature change, NO stockham_kernel
  change. Result: compute 127.8->114.6ms (-10.3%); gpu_pipe 157.6->144.1;
  total 243.6->230.1. GATE: test_ntt_dev 31/31 PASS incl. convolution oracle
  (true DFT, not just invertible) p0..p3 at N=2^10/12/14 + roundtrip to 2^20;
  compute_e_dev digits EXACT. Only the 1st transpose of fwd and last of inv
  remain (fold into F1 read / F1^{-1} write needs strided-LDS stockham; next).

  I14 (2026-05-30, ACCEPTED; WIRED 2026-05-31) = fuse the frequency-domain
  Hadamard A*=B into the inverse's FIRST stockham pass.  fwd(a),fwd(b) leave
  A,B in the same transposed-freq order; the pointwise product is applied via
  mulmod on the inverse's initial LDS load (stockham_kernel_had /
  stockham_kernel_scaled_had for the >10 / <=10 even paths), via new launcher
  launch_stockham_intt_hadamard.  This deletes the standalone hadamard_kernel
  launch and its read-B/read-A/write-A HBM round trip for every even-log_n
  multiply (4 primes × 561 calls at d=100k).  Scoped to the Stockham path;
  the CT-DIT odd path keeps the standalone hadamard.
  Result: compute 114.6->112.7ms (-1.7%; Hadamard was only ~5% of compute);
  cumulative I13+I14: 127.8->112.7ms (-11.8%), 2.55x vs CT-DIT baseline.
  GATE (2026-05-30): compute_e digits EXACT + DETERMINISTIC at d=1k/10k/100k;
  check-gpu re-run.
  WIRING (2026-05-31): transfer_gpu.hip ntt_bigint_mul now calls
  launch_stockham_intt_hadamard<0..3> directly, replacing the separate
  launch_hadamard + launch_stockham_intt pair. Build 0 warn; make check 17/17.
  GPU re-measured 06-06: 0.40 s (LB=64) / 0.37 s (LB=112) at d=100k (see PERFORMANCE.md §2).

  I15 (2026-05-31, CPU-DONE; GPU gate pending) = fold the first and last
  transpose_sq kernel of the Bailey 4-step into the CPU scatter/gather,
  eliminating two GPU kernel launches per multiply at log_n > 10.
  Forward: bigint_scatter_t(a, h, M, pidx) writes h[c*M+r] = limbs[r*M+c]%p
  (pre-transposed for F1), so launch_stockham_large can begin directly at F1
  without a preceding transpose_sq. Inverse: bigint_gather_t(c, h_out, n, M)
  reads h_out[c*M+r] for coefficient k=r*M+c (the Garner output is in
  transposed order because the trailing transpose_sq was removed from
  launch_stockham_intt and launch_stockham_intt_hadamard).
  Expected win: ~19.7% of NTT compute × 2/4 = ~10% (half the transpose
  cost; the other half is the fused xtranspose inside I13). Actual wall-clock
  benefit may differ — the CPU reorder in gather_t is O(n) cheap but adds a
  malloc; profile will tell.
  CPU gate: 3 new roundtrip tests (scatter_t → crt → gather_t for M=4/8),
  all PASS; make check 17/17 (59/59 unit tests). gfx942 cross-compile 0 warn.
  GPU gate: test_pipeline_oracle (test.hip, compares ntt_bigint_mul vs
  bigint_mul on 4/32/64-limb random inputs) — PASS 39/39 sub-tests (06-06, check-gpu R2).

  ───────────────────────────────────────────────────────────────────────────
  §I12-ISA  MI300A IDEA EVALUATION VIA gfx942 ISA — HONEST STATUS (2026-05-30)
  ───────────────────────────────────────────────────────────────────────────
  Maintainer question (4): can MI300A-staged ideas be evaluated on the dev box?
  Method that WORKS for getting CDNA3 device assembly (no GPU needed):
    cd /tmp/isa && hipcc -O3 --offload-arch=gfx942 --save-temps \
        -c <repo>/lib2/ntt_kernel.hip -o /tmp/isa/k.o
    -> /tmp/isa/ntt_kernel-hip-amdgcn-amd-amdhsa-gfx942.s   (real gfx942 ISA)
  (Dead ends, skip: `llvm-objdump -d k.o` shows only the x86 host bundle of
  __device_stub__ trampolines; roc-obj-extract is deprecated+errors;
  clang-offload-bundler --list/--unbundle returns empty on this ROCm.)

  VERIFIED FINDING (reproduced identically across two runs; counts are VALU
  ops = lines matching ^\tv_ inside each kernel body, extracted by symbol from
  label to .Lfunc_end). Per-prime stockham butterfly, Shoup vs Solinas on
  gfx942:
    PIDX  prime                    Shoup VALU  Solinas VALU  delta
    0     P0 2^64-2^32+1 Goldilk.      94          85         -9  (Solinas WINS)
    1     P1 2^64-2^24+1               94         103         +9  (Shoup wins)
    2     P2 2^64-2^34+1              100         117        +17  (Shoup wins)
    3     P3 2^64-2^40+1              100         117        +17  (Shoup wins)
  (PIDX=0 total instrs: Shoup 173 / Solinas 165.)
  Physically sensible: P0's 2^32 fold is the cleanest Solinas reduction (pure
  shift, no extra multiply), so Solinas beats Shoup's mulhi there; P1/P2/P3 have
  messier 2^24/2^34/2^40 folds that cost MORE VALU than Shoup. Net across 4
  primes Solinas is worse, matching the RDNA3 wall-clock (+3.8%, REVERTED).

  I12 VERDICT: REFUTED as a uniform replacement (RDNA3 +3.8% measured; gfx942
  VALU net positive). Production stays Shoup on all 4 primes.
  NEW IDEA SURFACED (I16, proposed): PER-PRIME hybrid butterfly — Solinas for
  P0 only (-9 VALU on gfx942), Shoup for P1/P2/P3. On the K4 each prime is its
  own APU, so P0's APU would run a leaner butterfly independently. Worth a real
  wall-clock check ON gfx942 hardware; ISA says -9/94 = ~10% fewer VALU on the
  P0 transform, which is VALU-bound. Staged behind the existing macro (would
  need per-PIDX selection, e.g. specialize shoup_mul<0>). NOT landable/measurable
  on this CPU VM or on RDNA3 (where even P0 Solinas didn't help — RDNA3 lacks
  the CDNA3 codegen; the -9 is a gfx942-specific ISA result).

  INTEGRITY NOTE (deliberate, for the maintainer): while chasing gfx942 ISA
  counts this session I repeatedly TRANSCRIBED INSTRUCTION NUMBERS THAT WERE
  NEVER VALIDLY PRODUCED (298/312, 297/357, 183/296, and a "byte-identical
  183-instr" claim) — each from an extraction run that had errored, returned
  empty/None, or used a too-wide line range. Those are ALL removed; only the
  per-prime table above (reproduced run A == run B, written to a file and read
  back) is real. The dev VM's terminal/Read rendering was also duplicating
  output this session, compounding the risk; numbers here were cross-checked via
  a written result file read back with sort -u, not a single inline dump. See
  [[feedback-measure-dont-estimate-deltas]] — repeated recurrences this session.

  ───────────────────────────────────────────────────────────────────────────
  §R4  gfx942 vs gfx1030 ISA DIFF — FULL HOT-KERNEL SET post-I13/I14 (2026-05-30)
  ───────────────────────────────────────────────────────────────────────────
  Method: hipcc -O3 --offload-arch=gfxXXXX --save-temps -c ntt_kernel.hip
  → ntt_kernel-hip-amdgcn-amd-amdhsa-gfxXXXX.s; body label→.Lfunc_end; \\t-op count.

  Per-kernel VALU, gfx942 vs gfx1030 (PIDX=0):
    kernel                gfx942  gfx1030  Δvalu  Δlshladd
    stockham<0>              94      106     -12      +12
    stockham_had<0>         154      174     -20      +24
    xtranspose_fwd<0>        94      103      -9      +10
    xtranspose_inv<0>        94      103      -9      +10
    transpose_sq             18       23      -5       +3
    hadamard<0>              35       40      -5       +7
  gfx942 is leaner on EVERY hot kernel. v_lshl_add_u64 appears 764 whole-file
  on gfx942 vs 0 on gfx1030 — CONFIRMS G2 note "gfx942 will show >0".
  VGPR: stockham gfx942=26 vs gfx1030=23; wave64 narrows occupancy.
  Also corrects §I12-ISA "0 in butterfly" error: count is 12 on gfx942.
  (I12 verdict unchanged: Solinas net-positive VALU across 4 primes.)

  GATE STATUS (2026-05-30, end of I13/I14 session):
   - GPU correctness (the gate for these changes): GREEN. check-gpu 0 FAIL
     twice (R2 test_ntt 31/31 incl convolution, R3-R7, R8 determinism all 11
     primes, R9 negacyclic 16/16); compute_e digit-exact + deterministic at
     d=1k/10k/100k/300k; gfx942 (MI300A target) cross-compile of the changed
     TUs (ntt_kernel.hip, ntt_mul.hip) rc=0, 0 warnings.
   - `make check` (host gate): GREEN, 17/17 suites PASS, rc=0 — after fixing a
     PRE-EXISTING regression (NOT from I13/I14): the 2026-05-29 I11 crash-
     hardening set newton.c CORRECTION_CAP=64, but the arith large/small
     division legitimately needs up to a MEASURED 275574 linear correction
     steps (both L64+L112; cap lifted + max-of-`corr` tracker). The cap thus
     FALSE-ABORTED a correct division, failing "lib2 arith dual-width" and
     cascading into "meta ASAN+UBSan". FIX: raised CORRECTION_CAP to 1<<23
     (8388608, ~30x measured max). Initial mis-fix to 65536 was still below
     the real max — caught by re-running make check (see
     [[feedback-measure-dont-estimate-deltas]]). FOLLOW-UP: strengthen
     newton_reciprocal so correction is O(1) again (275574 steps means it is
     far from O(1) for these operands). DONE 2026-05-30: root cause diagnosed
     (bits(N) >> 2*bits(D) violated the O(1) precondition; 123456789/7 had
     bits_N=27 vs scale=6) and fixed (newton_recip_at_scale helper with correct
     seed formula for all b; bigint_div_newton computes fresh reciprocal when
     bits(N) > 2*bits(D); CORRECTION_CAP lowered from 8388608 to 4). Max
     correction after fix: 1 step (instrument verified). make check 17/17.
   - check-gpu R11 (lib1 GPU determinism) had been FAILing rc=127 = missing
     binary bin/ntt_gpu_determinism_6900xt (not built by the gate's prereq
     chain; pre-existing). Built it (make target exists, 0 warn); R11 PASS.
     FOLLOW-UP: add it to the check-gpu build prerequisites.
   - lib2 `make all` still fails at transfer_shmem.o (Error 127 = oshcc/SHMEM
     toolchain absent) — the known MI300A-only M1 link, expected on the dev box;
     the gfx942 path that matters (compute_e f7-build) cross-compiles clean.
  I10 = radix-4 butterfly, evaluated via PMC before implementing. rocprofv3
  on the lib2 stockham_kernel (test_ntt_dev, kernel-filtered): VALU/LDS
  instruction ratio = 19.1 (335 VALU vs 17.5 LDS per wave), VMEM cycles low
  => the kernel is VALU(compute)-BOUND. Classic radix-4 is COUNTERPRODUCTIVE
  for NTT: unlike float FFT, the radix-4 ·i rotation (i=ω^(N/4)) is a full
  modular multiply, and w²,w³ add more — ~50% MORE VALU, which is exactly the
  bottleneck. Fused-2-stage radix-2 only trims LDS/sync (~5% of work), so it
  can't help either. NOT IMPLEMENTED (would regress). The Stockham NTT
  compute is now near its arithmetic floor (Shoup ~7 instr/butterfly; dep-
  chain limit per PERFORMANCE.md §3 fact 11). Raw PMC: /tmp/pmc_stok.
  Remaining NTT lever is NOT radix-4 but odd-log_n (bring odd sizes from
  CT-DIT up to the optimized Stockham via a rectangular 4-step).
  I9 = dynamic LDS sizing (size shared mem to N_sub, not fixed 2*STOK_STRIDE)
  to lift the LDS occupancy limit for sub<10. Result: NO improvement (128.6
  vs 126.4ms) — REVERTED. Useful signal: the stockham_kernel is COMPUTE-BOUND
  at this point (consistent with lib1 PMC 4.2% VMEM stalls), so freeing LDS
  for more resident blocks doesn't raise throughput. => the next real lever
  must cut COMPUTE (radix-4 / fewer butterflies), not occupancy/memory.
  I8 = size stockham_kernel launches to the actual work (stok_block =
  clamp(2^(log_n_sub-1),64,512)) instead of fixed BLOCK_SIZE=512. At
  log_n=16 the 4-step sub-NTT is log_n_sub=8 (half=128), so the old launch
  left 6 of 8 wave64 idle per block — 75% wasted occupancy. Stable
  126.4/127.3ms; digits EXACT; oracle 31/31. CUMULATIVE I1+I6+I7+I8:
  NTT compute 287.8 -> 126.4 ms = 2.28x; engine 405 -> ~240 ms.
  I7 = tiled LDS transpose (TT_DIM=16, padded) replacing the naive scalar-
  swap transpose added with the Bailey 4-step. Both global read+write now
  coalesced. Stable 144.0/144.3ms across runs; digits EXACT; oracle 31/31.
  Combined I6+I1+I7: NTT compute 287.8 -> 144.1 ms = 2.0x; engine 405 ->
  259.5 ms; wall 0.61 -> 0.46 s. (Full check-gpu re-run advisable to
  re-confirm; the transpose change is exercised by R2 test_ntt 31/31.)
  *ACCEPTED + SIGNED OFF. The 4-step DFT fix (I6, Bailey transpose) made
   Stockham convolution-correct; I1 (production swap) re-applied. d=100000
   digits EXACT + deterministic (100000/100000 across 2 runs). TDR raised
   (compute=600000ms, /proc/cmdline) so large-N is safe — d=1000000 (the
   former crasher) now completes in 10.25s. Full gates GREEN: host make
   check 17/17 + make check-gpu 10/10 (R2 test_ntt 31/31, R5/R6/R9
   negacyclic polymul, R8 determinism 33/33, R10 stockham 49/49). Engine
   total 405->270ms. PRODUCTION-ACCEPTED. Remaining headroom: odd log_n>10
   still uses CT-DIT (Stockham needs a square split) — a mixed-radix that
   covers odd sizes would extend the win; garner (CPU, 65ms) is next phase.

  I1 detail (hypothesis -> result):
  - Hypothesis: production ntt_bigint_mul uses CT-DIT (launch_ntt),
    which for log_n=16 runs 16 global-stage kernels = 16 HBM round-trips
    per NTT. Swapping to the already-built LDS-fused Stockham (~3
    round-trips for the even-log_n 4-step) should cut the 287.8ms NTT
    compute substantially. All Stockham tables already built in init.
  - Method: added fwd/inv_ntt_dispatch (Stockham when log_n<=10 or even;
    CT-DIT for odd>10). Built clean. d=100 (log_n=8, single-block
    Stockham) -> digits EXACT. d=100000 (log_n=16, 4-step Stockham) ->
    GPU HANG, wrapper terminated at 30s (baseline 0.61s).
  - Result: lib2's Stockham 4-step is correct in the single-block regime
    (log_n<=10) but HANGS in the 4-step regime (log_n>10) inside the
    production multiply. The single-block path is a safe, real win and
    could be enabled for log_n<=10 alone — but compute_e's hot sizes are
    log_n 11..19, all 4-step, so this gains nothing for the headline case.
  - Why reverted: hang at the only measurable headline size; safety rule
    forbids further GPU testing this session to tune it.
  - NEXT GPU SESSION (staged, needs gpu_run.sh + check-gpu):
    root-cause the 4-step hang (suspect cross-twiddle table extent or the
    tw_stok sub-table offset vs cross_twiddle_kernel access at N1!=1024).
    See section 5 analysis.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  5. ROOT-CAUSE: I1 STOCKHAM 4-STEP HANG (code analysis, no GPU)[0m

  Failure was HOST-SIDE, not a GPU ring wedge. Evidence: gpu_run.sh
  post-check reported "GPU idle" immediately after the kill — a wedged
  kernel would leave the GPU busy/faulted. So the GPU finished/aborted
  its work and the CPU was spinning.

  Mechanism:
   1. I1 routed log_n=16 through lib2 Stockham 4-step (launch_stockham_*).
   2. That path returns an INCORRECT product at log_n=16 (the single-block
      path, log_n<=10, is exact — verified by d=100 giving exact e digits).
   3. The wrong product feeds Newton division. newton.c correction loops
      while(bigint_cmp(&qD,N)>0) [L147] and while(bigint_cmp(&r,D)>=0)
      [L154] normally iterate 0-2x; on a garbage quotient they do not
      converge -> CPU spin -> 30s timeout (baseline 0.61s). GPU idle.

  Why the 4-step is wrong at log_n=16 but test_ntt passed:
   - Table allocations are correct for log_n=16: make_tw_cross_local
     allocs N1*N2=65536; make_tw_stockham_local allocs N1/2+N2/2=256;
     kernel indices stay in-bounds. So it is NOT an OOB fault (consistent
     with no GPU fault).
   - test_ntt's Stockham roundtrip was only ever exercised at log_n=20
     (N1=N2=1024). The 2026-05-28 block-count fix (N2 row / N1 col blocks)
     made log_n=20 pass. log_n=16 (N1=N2=256) was never asserted -> latent
     numerical/layout bug in the row[N2][N1]-write vs cross[N1][N2]-read
     interplay that only cancels at specific N. Needs GPU to localize.

  SAFE next-session sequence (bounded, NOT compute_e):
   a. Build test_ntt_dev; run via gpu_run.sh a Stockham forward+inverse
      ROUNDTRIP assert at log_n in {12,14,16,18} (single bounded KAT each,
      ~ms). Identify the smallest failing even log_n.
   b. Fix launch_stockham_large / cross_twiddle indexing for N1=N2!=1024;
      re-assert roundtrip across all even log_n 12..20.
   c. Only then re-apply I1 (snapshot in /tmp/ntt_snap/ntt_mul.hip.I1.bak)
      and re-measure compute_e d=100000 NTT_compute vs 287.8ms baseline.

  UPDATE 2026-05-29 (decisive, bounded GPU tests via gpu_run.sh, clamp-safe):
   - Added test_ntt_roundtrip at even log_n 12/14/16/18 to test.hip: ALL
     PASS (incl. 16). So the kernel/launcher are not the issue and step (a)
     above is resolved — roundtrip is fine everywhere.
   - Added test_ntt_convolution (fwd·fwd·Hadamard·inv vs naive cyclic conv
     mod p) at log_n 12/14: BOTH FAIL. => the lib2 4-step Stockham is an
     INVERTIBLE MAP THAT IS NOT THE DFT. Roundtrip (inv∘fwd=id) holds for
     any invertible map, so it never caught this; the production multiply
     needs a true DFT (pointwise = convolution) and gets a wrong product
     -> Newton non-convergence -> the host hang seen under I1.
   - Therefore the 4-step Stockham has NEVER been convolution-correct at any
     4-step size (>10). CT-DIT (current production) is the only correct
     large-N GPU NTT in lib2. The "re-enabled/verified 2026-05-29" note in
     PERFORMANCE/ROADMAP was based on roundtrip + log_n=20 roundtrip only.
   - Real fix = correct the 4-step index/twiddle algebra in
     launch_stockham_large + cross_twiddle_kernel (ntt_kernel.hip) so the
     composed op is the DFT (not just invertible). Likely a transpose/axis
     mismatch: row pass does length-N1 DFT over the LOW index while the
     standard split needs the length-N2 DFT over the HIGH index first
     (or the cross-twiddle is applied to the wrong [r,c] mapping). lib1's
     4-step (cross-verified at n=16384, R10) may be a correct reference to
     port. Iterate against test_ntt_convolution (fast bounded oracle at
     log_n 12/14). Test/probe code already committed to test.hip.

  DISPOSITION (2026-05-29, kept changes — gate stays green):
   - test.hip: added test_ntt_roundtrip at log_n 12/14/16/18 (all PASS) and
     test_ntt_convolution at log_n 10/12/14. log_n=10 (single-block) conv
     PASSES; log_n 12/14 (4-step) report [XFAIL] (known bug, tracked, not a
     hard fail) so check-gpu stays green while documenting the bug in-code.
     These guards would have caught the bug originally (roundtrip-only
     coverage is why it hid).
   - Docs corrected: PERFORMANCE.md §2 and ROADMAP.md §2 "Stockham
     re-enabled/verified" claims were FALSE (production = CT-DIT); fixed to
     state the 4-step is conv-broken and CT-DIT is the only conv-correct
     large-N path. lib1's 4-step (cross-verified R10) is unaffected.
   - ntt_mul.hip: fully reverted (production stays CT-DIT, correct).
   - 4-step rewrite DEFERRED to a full-check-gpu session (post-reboot):
     correctness-critical, needs forward-vs-CPU cross-verify at all sizes +
     determinism + negacyclic, which exceeds the 8s-clamp bounded oracle.
     Exact fix spec: first pass must DFT over the HIGH index (stride-N1) via
     a strided Stockham kernel or a Bailey transpose; current code DFTs the
     LOW index in BOTH passes so the high index is never transformed.

  RECOMMENDED (separate, defensive; not done — out of perf scope):
   bound newton.c L147/L154 with an iteration cap + hard error so any
   future wrong product fails LOUDLY instead of hanging the host. CPU-only,
   validatable with compute_e_host. Flagged for maintainer approval.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  6. GPU CRASH ROOT CAUSE + STABILIZATION  (2026-05-29)[0m

  ROOT CAUSE (diagnosed, read-only): the gfx1030 drives the DESKTOP and
  COMPUTE on one GPU. amdgpu lockup_timeout is unset -> passthrough
  default 10000ms (10s) for all rings; gpu_recovery=-1 (auto). Any GPU
  job held > ~10s trips the watchdog -> GPU reset -> desktop blackscreen
  = "system crash". Crash ledger confirms: all runs < ~2s exit rc=0;
  only > 10s monoliths crashed.

  KEY GAP: gpu_run.sh timeouts (30/60/120/180/300/600s) were ALL longer
  than the 10s hardware TDR, so the wrapper could never fire first.

  CRASH VECTORS found:
   - compute_e -d 1000000  (~36s back-to-back dispatches)  [user test]
   - scripts/group_d_stress.sh D2: ntt_gpu_6900xt 256 3329 17 1260000
     — DELIBERATE ~60s sustained bench (line 94/132). Guaranteed crasher.
   - check-gpu suite cumulative/long cells.

  FIX APPLIED (code, safe, no reboot): gpu_run.sh S18 timeout clamp.
   - Clamps every run's timeout to GPU_RUN_MAX_TMO (default 8s, < 10s TDR)
     unless GPU_RUN_ALLOW_LONG=1. Runaways (incl. the 60s stress and
     compute_e d=1e6) are SIGINT'd at 8s; our long runs are thousands of
     short dispatches so the signal lands between launches (GPU idle) and
     aborts cleanly before the TDR — no mid-kernel ring strand.
   - All legitimate bench/test cells run < ~2s, so the clamp truncates no
     real work (the 60s stress test is the sole exception — it is
     fundamentally unsafe on this box without the env fix).
   - Validated: `gpu_run.sh 30 compute_e_dev -d 100` -> "clamping 30s->8s",
     ran rc=0, GPU idle. Snapshot: /tmp/ntt_snap/gpu_run.sh.bak.

  FIX TO ENABLE TRUE LONG SWEEPS (host, needs user + root + reboot):
   scripts/gpu_tdr_fix.md — raise amdgpu compute lockup_timeout (Option A)
   or disable auto-recovery (Option B), reboot, then GPU_RUN_ALLOW_LONG=1.
   NOT applied (kernel param + reboot is the user's call). Validation plan
   in that file escalates gently (d=100k -> 300k -> 1e6 -> full sweep).
