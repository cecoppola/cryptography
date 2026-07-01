# DEPLOYMENT DIAGNOSTIC — MI300A hardware mapping runbook

> For the agent bringing this library up on the real MI300A. Plain markdown by
> convention (agent-executed; see AGENT_BRIEF.md). Read AGENT_BRIEF.md first.
> First version 2026-06-30 — refine as the hardware answers come back.

## How to use this guide
The engine was developed and tuned on a **6900XT (gfx1030, RDNA2, discrete VRAM,
display-shared)**. It carries three kinds of baggage that this runbook exists to fix:
1. **gfx1030-calibrated constants** (block sizes, occupancy hints, thresholds, a timing model),
2. **display-safety machinery** that is pointless and may slow you down on a compute-only APU,
3. **gfx942 + unified-HBM + 4-APU paths that have never run on real hardware.**

**Golden rules (do not skip):**
- **Correctness before performance.** Finish Phase 2 (oracles green on gfx942) before you
  trust or tune a single number. A fast wrong answer is the worst outcome.
- **Measure, then change.** Fill the Appendix-A fact sheet from real queries; never tune from a guess.
- **One change at a time, re-verified by the GMP oracle** (`test_gmp_oracle`). Record the before/after.
- **Persist every measured value and decision** in Appendix A + a per-phase log; future re-tunes depend on it.

**Each entry below has the same shape:**
- **WHAT** — the code item and where it lives (`file:symbol`), and why it's hardware-dependent.
- **GET** — the exact tool/command/probe that produces the number you need.
- **USE** — the decision it drives and the concrete code/flag change to make.

Work the phases in order. Check the boxes. Appendix D is the full `file:constant` index
(every hardware-touched item) so you can confirm nothing was missed.

---

## Phase 0 — Environment & build bring-up
- [ ] **WHAT:** exact target arch — `ARCH_MI300A := --offload-arch=gfx942` (root `Makefile`, `lib/Makefile`).
  **GET:** `rocminfo | grep -m1 'gfx'`; `amdgpu-arch`; `offload-arch`. Note the **xnack/sramecc** suffix.
  **USE:** if the node reports e.g. `gfx942:sramecc+:xnack-`, append it to every `--offload-arch` so codegen matches the silicon (affects ECC + page-fault behavior).
- [ ] **WHAT:** Cray PE toolchain — `PrgEnv-amd/8.6.0 rocm/7.0.3 craype-accel-amd-gfx942 cray-shmem/12.0.0 cray-mpich/9.0.1 gcc/14.3.0`.
  **GET:** `module avail`, `module list`, `hipcc --version`, `oshcc --version`, `sinfo`, `scontrol show node <node>`.
  **USE:** reconcile with the recorded module set in `scripts/setup_mi300a.sh` (+ README build-env section); fix any version drift before building.
- [ ] **WHAT:** the gfx942 cross-compile proof — `make all-mi300a`.
  **GET:** run it **on the node** with the deployed toolchain; diff the `results/all_mi300a_*.log` against the dev-box log.
  **USE:** it was built for exactly this drift check; any new warning/error is an environment or codegen difference to resolve first.
- [ ] **WHAT:** **M1** — the `oshcc`/SHMEM `crt_ntt` final link (the only build step that can't be done off-target).
  **GET:** `cd lib && make all` (needs `oshcc` + `cray-shmem`).
  **USE:** confirms the SHMEM distribution links; gate for any 4-APU/multi-node work.

## Phase 1 — Hardware inventory (fill Appendix A once; everything downstream reads it)
- [ ] **WHAT:** per-device compute/memory geometry the engine assumes.
  **GET:** build & run **`hw_probe`** (Appendix C) — dumps `hipDeviceProp_t` for all devices; cross-check with
  `rocminfo` (CU count, wavefront, LDS, max waves/CU), `rocm-smi --showmeminfo vram --showclocks`.
  **USE:** record CU count, **wavefrontSize (expect 64)**, LDS/CU (expect 64 KB), LDS banks (expect 32),
  VGPR file, max waves & blocks/CU, total HBM, L2 size into Appendix A. These feed Phases 4–6.
- [ ] **WHAT:** fabric / multi-APU topology.
  **GET:** `rocm-smi --showtopo --showtopoweight --showbus`; `rocm-bandwidth-test` (H2D/D2H + **P2P all-to-all**);
  `hipDeviceCanAccessPeer` 4×4 in `hw_probe`.
  **USE:** record the peer-access matrix + per-link and all-to-all bandwidth; drives the broadcast/Garner choices (Phase 6).
- [ ] **WHAT:** CPU/NUMA layout (for the OpenMP Garner fallback + scatter placement).
  **GET:** `numactl --hardware`, `lscpu`, `lstopo`.
  **USE:** record cores/APU (expect 24) + which cores map to which APU's HBM; drives `OMP_PLACES`/`OMP_PROC_BIND` (Phase 6).
- [ ] **WHAT:** unified-memory behavior.
  **GET:** in `hw_probe`, test `hipMallocManaged` + GPU read/write + `hipHostMalloc(NonCoherent)` aliasing; check
  `/sys/module/amdgpu/parameters/*` for XNACK.
  **USE:** confirm managed/host allocations land in unified HBM and the GPU reads them without fault — the premise of the entire `!GFX1030_LOCAL` path (Phase 2 validates correctness; Phase 5 tunes it).

## Phase 2 — Correctness on gfx942 (HARD GATE — nothing perf-related before this is green)
- [ ] **WHAT:** modular arithmetic on the new codegen — `primes.h` reductions, `montgomery_mul`, `shoup_mul`, `addmod/submod`.
  **GET:** build+run `lib/test_modops.hip` (4/4) and the host `test_reduce` (the reductions are exhaustively
  host-tested, but gfx942 device codegen of `__uint128_t` is new).
  **USE:** any mismatch = a codegen bug; do not proceed. This is where a bad `--offload-arch` or compiler regression shows up.
- [ ] **WHAT:** **Solinas-P0** default (`NTT_SOLINAS_P0`, `shoup.h`) — bit-correctness on gfx942.
  **GET:** `test_modops` with the flag (it's the gfx942 default); `test_gmp_oracle` at several log_n.
  **USE:** confirm P0 via the Solinas/Goldilocks path equals the reference; recall this flag once silently corrupted
  every prime (the preprocessor-vs-template bug) — verify, don't assume.
- [ ] **WHAT:** full multiply correctness — `ntt_bigint_mul` and `crt_ntt`.
  **GET:** `test_gmp_oracle <log_n> <seed>` at **20/22/23/24** (checks every product limb vs GMP);
  `test_ntt` (22/22); `test_e2e_oracle`.
  **USE:** green = the engine is arithmetically correct on gfx942. Only now are perf numbers meaningful.
- [ ] **WHAT:** the unified-HBM **zero-copy path** (`!GFX1030_LOCAL`) — never executed on real MI300A.
  **GET:** build the engine **without** `GFX1030_LOCAL` (the gfx942 default), run `test_gmp_oracle` + a small `compute_e`.
  **USE:** confirm `hipHostMalloc(NonCoherent)` aliasing of `h_out`/`d_out` and managed operands produce correct
  results with no D2H. If wrong, fix before relying on zero-copy.
- [ ] **WHAT:** **G2** — 4-APU CRT polymul vs CPU bigint.
  **GET:** the 4-APU `crt_ntt` path over ~200k random + all-(p-1) edge inputs.
  **USE:** exactness across all 4 residue lanes + Garner is the multi-APU correctness gate.

## Phase 3 — Neutralize dev-box baggage (free throughput; no display on MI300A)
- [ ] **WHAT:** low-priority compute streams — `NTT_STREAM_LOWPRIO` (default 1, `ntt_mul.hip` init).
  **GET:** read the init; A/B a representative multiply with the env 1 vs 0.
  **USE:** this exists to let the gfx1030 compositor preempt NTT kernels; on MI300A it can only *hurt*. Set default
  off for gfx942 (or `NTT_STREAM_LOWPRIO=0`), confirm no regression, make it the deployed default.
- [ ] **WHAT:** display-yield — `NTT_DISPLAY_YIELD_US` / `_EVERY` (`ntt_mul.hip` mid+end-of-mul sleeps).
  **GET:** confirm default (0 = off when unset).
  **USE:** keep it 0 on MI300A (it inserts `usleep` + stream syncs purely for flip_done safety). Document that the
  deploy must not set it.
- [ ] **WHAT:** the `compute_e` safety gate + timing model — `SAFE_GPU_SECS = 120`, `est_gpu` (`app/compute_e/main.c`).
  **GET:** read the model (1.1 ms/dispatch calibrated at log_n=16 **on gfx1030**, scaling `2^(log_n-16)·(log_n/16)`).
  **USE:** the 120 s TDR-derived gate does not apply (dedicated compute queue) — raise it far or bypass with `-f`;
  **re-calibrate `ms_per_dispatch` on gfx942** (Phase 5) so the estimate is meaningful.
- [ ] **WHAT:** clock pin / `gpu_run.sh` S-guards / cwsr / GFXOFF — all gfx1030 display-sharing safety.
  **GET:** n/a (these are dev-box host scripts, in `archive/scripts_devbox/`, gitignored).
  **USE:** do **not** port them to MI300A. Document that the MI300A run path is plain `srun` (no wrapper).

## Phase 4 — ISA & occupancy audit → re-tune launch geometry
- [ ] **WHAT:** hot-kernel ISA — `stockham_kernel`, `transpose_sq_kernel`, `xtranspose_*`, `garner_kernel`, `scatter_kernel`.
  **GET:** `hipcc -O3 --offload-arch=gfx942 --save-temps -c <tu>.hip` → inspect the `*.s` with `llvm-objdump`;
  seed: `lib/isa_check.hip` (verifies `v_lshl_add_u64` in reduce_p1). Read **VGPR count, LDS bytes, scratch/spills**.
  **USE:** confirm the expected instruction selection (the Goldilocks fold, fused shift-adds, no scratch spills). A
  spill or unexpected VALU count is a tuning signal.
- [ ] **WHAT:** PMC profile vs the gfx1030 baseline — **G6**.
  **GET:** `rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_LDS SQ_INST_CYCLES_VMEM SQ_WAVES` (+ occupancy, L2 hit, HBM BW)
  on the hot kernels. Compare per-wave to `archive/.../X3_PMC_GFX1030_*` (trust the bottleneck *class*, not absolute magnitudes).
  **USE:** classify each hot kernel (VALU-bound vs LDS vs VMEM/HBM) and read achieved occupancy.
- [ ] **WHAT:** occupancy hints — `NTT_LAUNCH_BOUNDS = __launch_bounds__(512, 2)` and `StokMinBlks<>` (`ntt_kernel.hip`).
  **GET:** from Phase-1 VGPR file + LDS/CU and the Phase-4 measured VGPR/LDS per kernel, compute the real
  blocks/CU ceiling; confirm with rocprofv3 occupancy.
  **USE:** raise `minBlocksPerCU` 2→4 where VGPR+LDS allow (the C2/G6+ candidate), re-measure; keep whichever wins.
- [ ] **WHAT:** block geometry — `BLOCK_SIZE = 512`, transpose `TT_DIM = 16`, the `__launch_bounds__(256, …)`
  kernels, and the stockham kernels' `__launch_bounds__((LOG_N_SUB>=11 ? 1024 : 512), StokMinBlks<>)` (so blocks
  jump to **1024 threads at LOG_N_SUB≥11**, i.e. log_n≥22 — the fix that unbroke log_n=22).
  **GET:** wavefrontSize + `maxThreadsPerBlock` (Appendix A; confirm ≥1024) + occupancy sweeps (build variants, time with `hipEvent`).
  **USE:** confirm 512/256/1024 are wave64-optimal; **ideally replace the hardcoded geometry with `hipDeviceProp_t`-driven
  selection** (the c-style rule: query `warpSize`/`sharedMemPerBlock`/`regsPerBlock` at init and pick block/tile size).
- [ ] **WHAT:** `MFMA_TARGET` (`-DMFMA_TARGET=1`, gfx942) gates `__builtin_amdgcn_mfma_*` matrix intrinsics.
  **GET:** `grep -rn 'mfma\|MFMA_TARGET' lib/ app/` — currently returns **nothing in source** (only the Makefile define).
  **USE:** the integer NTT does **not** use MFMA; the flag is a no-op placeholder (reserved for R5, an adjacent
  MFMA algorithm). Do **not** spend time "enabling MFMA" for the NTT — leave the flag, ignore it.
- [ ] **WHAT:** per-prime reduction choice — Shoup (P1–P3) vs Montgomery vs Solinas-P0 (`shoup.h`, `primes.h`).
  **GET:** the Phase-4 PMC: is the butterfly VALU-bound or HBM-bound on gfx942? Micro-bench each variant per prime.
  **USE:** confirm Solinas-P0 is an actual VALU win (it was only ISA-analyzed); on gfx942's higher HBM BW,
  Montgomery (half the twiddle storage) may beat Shoup on bandwidth-bound sizes — choose per prime, GMP-verify.

## Phase 5 — Memory model & threshold re-tuning
- [ ] **WHAT:** schoolbook↔NTT crossover — `BIGINT_MUL_THRESHOLD` (`lib/arith/multiply.h`) and
  `BASE_CONVERT_MUL_THRESHOLD` (=512 under `GFX1030_LOCAL`, mirrors the former otherwise; `base_convert.c`).
  **GET:** sweep operand sizes around the threshold on gfx942, timing schoolbook vs `ntt_bigint_mul`.
  **USE:** the crossover shifts with the faster GPU + cheap unified-memory sync (no PCIe round-trip) — set both
  thresholds from the measured crossover; the 512 raise was a PCIe-sync workaround that likely no longer applies.
- [ ] **WHAT:** managed-memory placement — `bigint_alloc_managed` (`bigint_hip_alloc.hip`), `use_mgd` (`base_convert.c`).
  **GET:** Phase-1 unified-memory test + timing managed vs explicit alloc at representative sizes.
  **USE:** on MI300A managed = unified HBM (works); set `use_mgd` thresholds so the BFS uses unified operands where it helps.
- [ ] **WHAT:** the `est_gpu` timing model (`app/compute_e/main.c`).
  **GET:** measure actual ms/dispatch on gfx942 at a few log_n.
  **USE:** replace the gfx1030 `1.1 ms`/scaling constants so the estimate (and any gate) is accurate on-target.
- [ ] **WHAT:** the size ceiling — `NTT_MAX_LOGN` default cap = 22 (`ntt_mul.hip`, `main.c`); VRAM ≈ 400·N bytes.
  **GET:** total HBM from Appendix A; compute the max N that fits (128 GB ⇒ log_n ≈ 28–29); confirm twiddle tables fit.
  **USE:** raise the default cap for MI300A. **The 3-factor is hardcoded for log_n==24 only** — for 25–29, generalize
  the recursion (asymmetric/odd outer splits, possibly a 4th factor; consider G8 shared-twiddle to fit tables). GMP-verify each new size.

## Phase 6 — 4-APU & fabric
- [ ] **WHAT:** input broadcast — `broadcast_input` SDMA `hipMemcpyPeerAsync` APU0→1,2,3 (`main.hip`).
  **GET:** Phase-1 P2P bandwidth; A/B SDMA vs a kernel-driven copy at N≥2^23 (**G7**).
  **USE:** pick SDMA below and kernel-P2P above the measured crossover (~16 MB on the design baseline; confirm).
- [ ] **WHAT:** peer access for cross-APU Garner. The GPU Garner on device 0 reads the other 3 APUs' residue
  lanes. Peer access is enabled in `lib/main.hip` (the `crt_ntt` init, `hipDeviceEnablePeerAccess`) — but
  **NOT in `lib/arith/ntt_mul.hip`** (the arith/`compute_e` engine), which on a 4-APU node maps the 4 primes
  to devices 0–3 yet never enables peer access. **This is a real deployment gap, not just a tune.**
  **GET:** the peer-access matrix (Phase 1); run `compute_e` with 4 APUs visible and watch for a Garner page-fault.
  **USE:** **add `hipDeviceEnablePeerAccess` to `ntt_mul_init_impl`** (mirror the `main.hip` loop) before any 4-APU
  arith run, OR pin the arith engine to one APU — decide which. Then confirm cross-APU reads work over Infinity
  Fabric and consider **G8** (shared twiddle on APU 0 + peer access) to save 3/4 of twiddle HBM at N=2^24.
- [ ] **WHAT:** gated on-device scatter/gather — `NTT_GPU_SCATTER_GATHER` + `transfer_kernels.hip` (**G10/G11**).
  **GET:** build with the flag + link `transfer_kernels.o` + managed operands; correctness via `test_e2e_oracle`'s
  core-routed trial; then measure.
  **USE:** the call sites were in the retired engine — re-wire them into `ntt_mul.hip`, then decide enable vs the
  current host gather by the measured all-4-APU runtime fraction (target ~80%→~93%).
- [ ] **WHAT:** OpenMP Garner fallback NUMA — `garner_reconstruct` (`garner.hip`), `OMP_PROC_BIND`/`OMP_PLACES`.
  **GET:** Phase-1 NUMA map.
  **USE:** if the CPU Garner path is used, bind 24 threads/APU spread; confirm the **GPU** Garner is the production path on-target.
- [ ] **WHAT:** multi-node SHMEM (**R2**) — `transfer_shmem.c` distribute/collect, if a K4 spans >1 node.
  **GET:** `srun -N 2 ... ./crt_ntt`; libfabric/OFI status.
  **USE:** validate the collective across nodes before claiming multi-node scaling.

## Phase 7 — Sustained run & acceptance
- [ ] **WHAT:** thermal/throttle under hours-long load.
  **GET:** `rocm-smi --showtemp --showpower --showclocks` sampled during a large sustained run (without the dev-box
  SMU-poll prohibitions — those were a gfx1030 crash class).
  **USE:** confirm no clock throttling at sustained load; if throttled, that bounds achievable throughput.
- [ ] **WHAT:** the ROADMAP §4 acceptance suite.
  **GET:** clean build under PrgEnv-amd; `lib/test_ntt` 22/22; G2 exact; `compute_e d=1e6` across 4 APUs == OEIS A001113; PMC within the scaled envelope.
  **USE:** all green ⇒ Phase 4 (deploy) complete.

---

## Appendix A — Hardware fact sheet (fill from Phase 1, then reference everywhere)
```
gfx arch (full, incl. xnack/sramecc): ____________
APUs visible (hipGetDeviceCount):     ____   wavefrontSize: ____  (expect 64)
CUs per APU: ____   max waves/CU: ____   max blocks/CU: ____   VGPRs/CU (file): ____
LDS per CU: ______ B (expect 65536)   LDS banks: ____ (expect 32)   L2: ______
HBM total (node / per-APU): ______ / ______   HBM BW measured: ______ TB/s
P2P peer matrix (4x4 canAccessPeer):  __________   per-link BW: ______  all-to-all: ______ TB/s
SDMA engines: ____   H2D/D2H BW: ______
CPU cores/APU: ____ (expect 24)  total: ____  NUMA nodes: ____  core→APU map: __________
XNACK: ____  managed=unified-HBM verified: Y/N   hipHostMalloc NonCoherent zero-copy: Y/N
sclk/mclk range: ______   default perf level: ______   power cap: ______   sustained-throttle: Y/N
ROCm: ____  HIP/amdclang: ____  KFD/amdgpu driver+fw: ____
```

**Dev-box reference (gfx1030 6900XT, captured 2026-06-30 via `hw_probe-dev` — for DIFFING, not the target):**
warpSize **32** · CUs 40 · maxThreads/block 1024 · maxThreads/CU 2048 · regs/block 65536 ·
LDS/block 65536 · LDS/CU 65536 · HBM 17.16 GB · L2 4 MB · sclk 2660 MHz · mclk 1000 MHz · bus 256-bit ·
integrated **no** · managed yes · concurrentManagedAccess **yes** · pageable no · directManagedHost no ·
micro-tests managed+zero-copy **all PASS**. On MI300A expect: wavefront **64**, integrated **yes**, far
larger HBM, and **4 visible APUs with a populated peer matrix** (the dev box is single-device).

## Appendix B — Tool cheat-sheet (what each tells you)
- `rocminfo` — agent props (gfx ID, CU count, wavefront, LDS, cache).
- `rocm-smi` — `--showmeminfo vram` (capacity/used) · `--showtopo[weight]` (P2P links) · `--showbus` · `--showclocks` · `--showpower --showtemp`.
- `rocm-bandwidth-test` — H2D/D2H + P2P all-to-all bandwidth (feeds broadcast/Garner choices).
- `amd-smi` — newer static+dynamic interface (overlaps rocm-smi).
- `hw_probe` (Appendix C) — the authoritative `hipDeviceProp_t` dump the engine should select from.
- `rocprofv3 --pmc …` — VALU/LDS/VMEM/waves + occupancy + cache/HBM counters (bottleneck classification).
- `hipcc --save-temps` + `llvm-objdump` — gfx942 ISA (instruction selection, VGPR, spills); `rocgdb` for kernel bugs.
- `numactl --hardware` / `lscpu` / `lstopo` — CPU/NUMA topology for OpenMP + scatter placement.
- `module`/`sinfo`/`scontrol`/`srun --version`/`oshcc --version` — Cray PE + Slurm.
- `dmesg` / `journalctl -k` — amdgpu/KFD faults, ECC, throttle events.
- `ldd` / `readelf -d` on the built binary — link deps (amdhip64, libsma/SHMEM, OpenMP); catches a missing GTL/SHMEM shim or a ROCm-6-vs-7 mismatch.

## Appendix C — `hw_probe` (the file exists: `app/hw_probe.hip`)
Built and ready: **`make hw_probe`** (gfx942, run on the MI300A via `srun`) or **`make hw_probe-dev`**
(gfx1030, runs on the dev box for a baseline to diff). Both compile clean `-Wall -Wextra`. It prints,
per device, a fixed-width table from `hipGetDeviceProperties` (arch, warpSize, CUs, maxThreads/block &
/CU, regs/block, LDS/block & /CU, total HBM, L2, clocks, and the unified-memory flags:
managed / concurrentManagedAccess / pageable / directManagedHost), the `hipDeviceCanAccessPeer` matrix
(with an actual enable), and two correctness micro-tests — managed GPU access (**gated on
`concurrentManagedAccess` so it cannot fault a non-XNACK GPU**) and `hipHostMalloc(NonCoherent)`
zero-copy aliasing. Its output transcribes straight into Appendix A. The collection lives in a reusable
`probe_device(int, DeviceFacts*)` so the engine can later call it at init and select `BLOCK_SIZE`/
`TT_DIM`/LDS sizing from it instead of the hardcoded constants (Phase 4 USE). LDS bank count isn't in
`hipDeviceProp_t` — get it from `rocminfo` (expect 32).

## Appendix D — Code-item index (every hardware-touched constant, for grep coverage)
| Item | Location | Phase |
|---|---|---|
| `--offload-arch=gfx942`, `MFMA_TARGET` | Makefile, lib/Makefile | 0, 4 |
| `BLOCK_SIZE=512`, `LDS_STRIDE=33`, `LDS_TOTAL` | lib/ntt_kernel.hip | 4 |
| `NTT_LAUNCH_BOUNDS(512,2)`, `StokMinBlks<>`, `STOK_STRIDE_C` | lib/ntt_kernel.hip | 4 |
| `TT_DIM=16`, transpose `__launch_bounds__(256,4)` | lib/ntt_kernel.hip | 4 |
| `__launch_bounds__(256,8)` garner/scatter | lib/garner.hip, lib/arith/ntt_mul.hip | 4 |
| `CLA_SCAN_BLOCK=1024` | lib/transfer_kernels.hip | 4, 6 |
| `STOK_PAD/STOK_STRIDE` (32-bank) | lib/crt_ntt.h | 4 |
| `NTT_SOLINAS_P0`, Shoup/Mont/Solinas | lib/shoup.h, lib/primes.h | 2, 4 |
| `__uint128_t` reductions, `P_INV` | lib/primes.h | 2, 4 |
| `NTT_MAX_LOGN` cap=22, 3-factor==24 | lib/arith/ntt_mul.hip, app/compute_e/main.c, lib/ntt_kernel.hip | 5 |
| `BIGINT_MUL_THRESHOLD` | lib/arith/multiply.h | 5 |
| `BASE_CONVERT_MUL_THRESHOLD=512`, `use_mgd` | lib/arith/base_convert.c | 5 |
| managed alloc | lib/arith/bigint_hip_alloc.hip | 1, 2, 5 |
| `GFX1030_LOCAL` zero-copy branch | lib/arith/ntt_mul.hip, lib/main.hip | 2, 5 |
| `n_devs` (ntt_mul.hip); peer access + `broadcast_input` (main.hip ONLY — arith engine lacks peer enable) | lib/arith/ntt_mul.hip, lib/main.hip | 1, 6 |
| `NTT_STREAM_LOWPRIO`, `NTT_DISPLAY_YIELD_US` | lib/arith/ntt_mul.hip | 3 |
| `SAFE_GPU_SECS=120`, `est_gpu` model | app/compute_e/main.c | 3, 5 |
| OpenMP Garner 96-thread/NUMA | lib/garner.hip | 1, 6 |
| SHMEM distribute/collect | lib/transfer_shmem.c | 0, 6 |
| `NTT_GPU_SCATTER_GATHER` (G10/G11) | lib/transfer_kernels.hip, transfer_core.h | 6 |
| `isa_check` (`v_lshl_add_u64`) | lib/isa_check.hip | 4 |
