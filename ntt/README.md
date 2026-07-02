# NTT / MI300A

High-performance modular **Number-Theoretic Transform** for the AMD MI300A. Pure C / HIP;
develops on a 6900 XT (gfx1030), cross-compiles for MI300A (gfx942). The deliverable
(`lib/`) is a **4-prime CRT-NTT engine for big-integer multiplication** on a K4 node of
MI300As, demonstrated by `app/compute_e` (Euler's number to N digits). `ref/` is a
single-prime reference (ML-KEM / ML-DSA / Goldilocks NTTs).

> **Picking this project up fresh? Read [AGENT_BRIEF.md](AGENT_BRIEF.md) first** — one dense
> file with the whole picture (goal, state, mental model, objectives, landmines, key data).

## At a glance

| Field | Value |
|---|---|
| Goal | Highest-throughput modular NTT for the MI300A APU |
| Languages | C (host), HIP (device); no C++ in source (one templated per-prime exception) |
| Toolchain | ROCm 7.0.3 / hipcc 7.0.51831 / AMD clang 20 |
| Dev hardware | AMD Radeon RX 6900 XT (gfx1030, RDNA2) |
| Target hardware | AMD Instinct MI300A (gfx942, CDNA3, wave 64, HBM3 unified memory) |
| Status | Delivery-ready: gfx942 cross-compile clean; log_n=24 (32M digits) validated |

## Documents

| Document | Read when |
|---|---|
| [AGENT_BRIEF.md](AGENT_BRIEF.md) | **fresh-agent orientation — read first** |
| README.md (this file) | first contact with the project |
| [ARCHITECTURE.md](ARCHITECTURE.md) | design + APIs + moduli + CRT engine + MI300A optimization |
| [PERFORMANCE.md](PERFORMANCE.md) | current measured baselines per hardware target |
| [ROADMAP.md](ROADMAP.md) | phase status, open MI300A tasks, future work |
| [DEPLOYMENT_DIAGNOSTIC.md](DEPLOYMENT_DIAGNOSTIC.md) | hardware-mapping runbook for MI300A bring-up |
| [STATUS.md](STATUS.md) | current state, API surface, next step |
| [CLAUDE.md](CLAUDE.md) | agent guidance (agent-driven sessions) |
| [ref/README.md](ref/README.md) · [lib/README.md](lib/README.md) · [app/README.md](app/README.md) | per-layer overviews |
| `~/MI300A_TARGET_ENVIRONMENT.md` · `~/HIP_6900XT_KNOWLEDGE.md` | cross-project references |

> The other canonical docs use an ANSI-boxed terminal style — view them with `less -R`.
> This README is GitHub-flavored markdown so it renders on the repo page.

## Quick build

**Local dev box (6900 XT / gfx1030):**
```sh
make all          # build the ref CPU binaries
make check        # GPU-free host reliability gate (21 suites)
make all-mi300a   # gfx942 cross-compile proof of every TU (compile-only)
```

**On the MI300A (Cray PE):**
```sh
module load PrgEnv-amd/8.6.0 rocm/7.0.3 craype-accel-amd-gfx942 \
            cray-mpich/9.0.1 cray-shmem/12.0.0 gcc/14.3.0
cd lib && make all                # crt_ntt (engine, oshcc/SHMEM) + test_ntt
cd ../app/compute_e && make       # the compute_e demonstrator

srun -p mi -N 1 -n 1 ./lib/test_ntt                            # correctness suite (22/22)
srun -p mi -N 1 -n 4 --ntasks-per-node=4 ./lib/crt_ntt 20 100  # 4-APU benchmark
```
Direct `./binary` execution fails on the MI300A — the cgroup forces all jobs through `srun`.

## Source layout

```
ref/       single-prime CPU reference NTT + GPU parity kernels (NOT shipped)
lib/       THE deliverable — 4-prime CRT-NTT engine + arith/ big-integer API (gfx942 K4)
app/       applications/demonstrators (compute_e: e to N digits; hw_probe)
scripts/   host test/CI gates + MI300A setup/acceptance
perf/      local-only bench/results/probes               (gitignored)
archive/   retired/dev-only material: docs, lib_dead, dev-box scripts (gitignored)
bin/       build artifacts                                (gitignored)
```

## Key make targets

| Target | What it does |
|---|---|
| `make all` | build the ref CPU binaries |
| `make all-mi300a` | gfx942 cross-compile proof of every TU |
| `make check` | GPU-free reliability gate — 21 host suites + ASAN/UBSan + fault-injection proof + 90% coverage floor |
| `make coverage-all` | gcov across ref + lib + app (80% per-TU floor) |
| `make cpu-all` | build + selftest + bench every ref CPU binary |
| `make gpu-mi300a` · `gpu-stok-mi300a` · `gpu-polymul-mi300a` · `verify-mi300a` | build the ref gfx942 GPU binaries |
| `make hw_probe` / `hw_probe-dev` | build the hardware-diagnostic probe (gfx942 / gfx1030) |
| `make clean` | clean `bin/` + lib + app |

Dev-box GPU runs go through `scripts/gpu_run.sh` (restored local-only; gitignored).

Arch flags: `--offload-arch=gfx942 -DMFMA_TARGET=1` (MI300A, root Makefile `ARCH_MI300A`) and
`--offload-arch=gfx1030 -DGFX1030_LOCAL=1` (dev box, lib/app Makefiles `DEV_HIP_CFLAGS`).

## Build environment (MI300A side)

Cray PE modules — confirmed on `baryon-cn0063`, 2026-05-09:

```
PrgEnv-amd/8.6.0  amd/7.0.3  rocm/7.0.3  craype/2.7.35
craype-x86-genoa  craype-accel-amd-gfx942  craype-network-ofi
cray-mpich/9.0.1  cray-shmem/12.0.0  cray-pmi/6.1.16
cray-libsci/25.09.0  cray-libsci_acc/25.09.2  cray-dsmml/0.4.0
libfabric/1.22.0  perftools-base/25.09.0  gcc/14.3.0
llvm/18.1.1 (system — DO NOT use for HIP)
```

**Caveats**
- `cray-mpich/9.0.1` is built against ROCm 6.0 — avoid MPI in GPU paths.
- Never wildcard-link ROCm (`-l*`) — `perftools-base` intercepts via `rocprof-sys`; list every `-l` explicitly.
- Direct `./binary` execution fails on the MI300A — the cgroup forces all jobs through `srun`.

**srun patterns**
```sh
srun -p mi -N 1 -n 1 ./bin/binary                          # single APU
srun -p mi -N 1 -n 4 --ntasks-per-node=4 ./bin/binary      # all 4 APUs
srun -p mi -N 2 -n 8 --ntasks-per-node=4 ./bin/binary      # two nodes
```

**Measured MI300A baselines:** HBM 4.0 TB/s · LDS 5.0 TB/s · FP64 DGEMM 87.6 TF/s ·
SDMA 3.98 TB/s · P2P all-to-all 4.62 TB/s · Atomic 912 GOPS/s.

**Troubleshooting**
- `perftools-base` interference → use explicit `-l` flags, never `-l*`.
- `libamdhip64.so.6` not found → the MPI GTL shim expects ROCm 6; build without MPI.
- HIP fat-binary non-deterministic → compare the disassembly hash, not the binary hash.

## Dev-box GPU safety (6900 XT only — not applicable to MI300A)

The dev GPU drives the display, so compute and graphics share one card:
- Never `SIGKILL` `rocprofv3` — it leaves stuck queues and a delayed reset.
- Never loop `rocprofv3` across binaries unsupervised.
- A GPU reset blackscreens the desktop; sustained compute can starve page-flips → hard hang.
  All dev-box GPU work goes through `scripts/gpu_run.sh` (1300 MHz clock pin, display-yield, S-guards).
- Build with `-Wall -Wextra`; fix every warning before marking a unit done.

None of this applies to the MI300A (dedicated compute queue, no display).

## Algorithm choices

Per-segment rationale and measured assumptions: see [ARCHITECTURE.md](ARCHITECTURE.md) §10.

| Seg | Choice |
|---|---|
| Transform | Cooley-Tukey DIT (CPU ref) + LDS-fused Stockham 4-step (GPU primary) + recursive 3-factor (log_n=24) |
| Multiply | Shoup precomputed-quotient (P1–P3) · Solinas/Goldilocks (P0, gfx942) · lazy reduction (CPU) |
| Bit-reversal | eliminated on the Stockham path; a separate kernel on the CT-DIT path |
| Twiddles | precomputed in HBM; LDS-cached for n ≤ 4096 |
| Memory | fused multi-stage LDS kernel + MI300A unified-HBM zero-copy output |
| Reconstruct | GPU Garner CRT → U256; host gather to the product big-integer |
