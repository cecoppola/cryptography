# AGENT BRIEF — NTT / MI300A  (read this first)

> Plain markdown on purpose: this file is for a fresh agent (LLM) to read, not for
> `less -R`. The canonical docs (README/ARCHITECTURE/ROADMAP/STATUS/PERFORMANCE +
> per-layer READMEs) keep the ANSI-boxed house style; this one optimizes for tokens.
> Last updated 2026-06-30.

## 0. TL;DR (orientation in 8 lines)
- **What:** highest-performance C/HIP **4-prime CRT Number-Theoretic Transform** for
  **big-integer multiplication**, targeting a **K4 node (4 APUs) of AMD MI300A**.
- **The product:** `lib/` (the engine). **The demo:** `app/compute_e` (Euler's *e* to
  N digits). **The reference:** `ref/` (single-prime NTT; proving ground, NOT shipped).
- **State: delivery-ready.** All dev-side work is done. log_n=24 validated end-to-end:
  a full `compute_e` to **32,000,000 digits** ran in **3.22 h** on the dev GPU,
  OEIS-correct (2026-06-30). Every TU compiles `-Wall -Wextra` clean; `make check` 17/17.
- **What's left: MI300A-target only** — the OpenSHMEM final link, 4-APU runtime + perf,
  enabling GPU scatter/gather, and gfx942 perf confirmation. See §5.
- **Hardware:** dev = 6900XT (gfx1030, RDNA2) + 5950X; target = MI300A (gfx942, CDNA3,
  wave64, unified HBM3), Cray PE, `srun`, `oshcc`/SHMEM.
- **Read next:** this file → ARCHITECTURE.md §4 (engine) → ROADMAP.md §3–4 (open tasks +
  acceptance) → lib/README.md. Hard-won facts live in the memory dir (§8).

## 1. First moves (confirm the tree is healthy, ~10 min, GPU-free)
```
make check            # 17 host suites + ASAN/UBSan + non-vacuity + 90% coverage; must be 17/17
make all-mi300a       # gfx942 cross-compile proof of every TU; must end "every gfx942 TU built clean"
```
Both are CPU-only/compile-only (no GPU execution) and safe anywhere. If either fails,
fix that before anything else. Do NOT run GPU binaries casually on the dev box — see §6.

## 2. Mental model (the whole thing on one screen)
Big-integer multiply via CRT-NTT: treat each BigInt's limbs as a polynomial evaluated at
x=2^64; the product is a cyclic convolution computed by an NTT over **4 Goldilocks-class
primes**; each ~248-bit product coefficient is reconstructed by **Garner CRT → U256**, then
carry-propagated back to limbs. Q = P0·P1·P2·P3 ≈ 2^255 covers products up to n·(q-1)^2.

**Two entry points share one engine — do NOT re-fork them** (a duplicate engine was
consolidated in "F6"; the retired copy is in `archive/lib_dead/transfer_gpu.hip`):
- **`crt_ntt`** (`lib/main.hip`): the distributed **polynomial**-multiply pipeline,
  **OpenSHMEM-collective across the K4 node**, `srun`-launched. *This is the product.*
- **`arith` / `ntt_bigint_mul`** (`lib/arith/ntt_mul.hip`): a **single-node big-integer**
  multiply API (+ Newton division + decimal conversion). *This is what `app/compute_e` uses.*
  Both use GPU NTT + **GPU Garner**; arith does a host-side gather (`bigint_gather`).

**NTT dispatch by log_n** (`fwd_ntt_dispatch` in ntt_mul.hip):
| log_n | path |
|---|---|
| ≤ 10 | single LDS Stockham block (any parity) |
| even, 12–22 | 2-factor LDS-fused Stockham 4-step (transpose-bracketed) |
| == 24 | recursive **3-factor** 4-step (outer 12+12, inner 6+6 leaves) |
| odd, > 10 | CT-DIT, Shoup butterflies (`launch_ntt`); the I11 rect-Stockham path is disabled, and the I4 Montgomery CT-DIT kernel is dormant (not wired) |

**arith pipeline:** H2D limbs → GPU scatter (per prime) → 4× forward NTT → Hadamard fused
into the inverse's first pass → GPU Garner → host `bigint_gather`.

## 3. Repo layout & key files
```
ref/   single-prime CPU reference NTT + gfx parity kernels  (NOT shipped)
lib/   THE deliverable — 4-prime CRT-NTT engine + arith/ bigint API
app/   applications/demonstrators (compute_e)
scripts/  host test/CI gates + MI300A setup/acceptance
gitignored (local-only, NEVER commit): archive/ (retired+devbox), perf/ (bench data), .claude/
```
Deliverable closure (what compiles into the product):
- Engine: `lib/{ntt_kernel.hip, garner.hip, main.hip, primes.h, shoup.h, crt_ntt.h}`
- Bigint API: `lib/arith/{ntt_mul.hip, bigint, multiply, newton, base_convert}.{c,h}`, `bigint_hip_alloc.hip`
- Multi-node distribution: `lib/transfer_shmem.c` (OpenSHMEM), `broadcast_input()` in `main.hip`
- MI300A-future (gated, not yet wired): `lib/transfer_kernels.hip`, `transfer_core.h`
- Demo: `app/compute_e/{main.c, binary_split.c, mem_pool.c}` + tests
- Oracles (run on new silicon before trusting it): `app/compute_e/test_gmp_oracle.c`,
  `lib/arith/{test_arith_gmp,test_e2e_oracle}.c`, `lib/test_modops.hip`

## 4. Build & verify
- **Dev box:** `make all` (ref CPU), `make check` (host gate), `make all-mi300a` (gfx942 proof).
- **MI300A:** `module load PrgEnv-amd/8.6.0 rocm/7.0.3 craype-accel-amd-gfx942 cray-shmem/12.0.0 ...`
  then `cd lib && make all` (builds `crt_ntt` via `oshcc`/SHMEM + `test_ntt`) and
  `cd app/compute_e && make`. Run via `srun -p mi -N 1 -n 1 ./lib/test_ntt` (correctness
  suite, 22/22), `srun -p mi -N 1 -n 4 --ntasks-per-node=4 ./lib/crt_ntt 20 100` (4-APU).
  Direct `./binary` fails on the MI300A (cgroup forces `srun`).
- **Correctness oracle:** `test_gmp_oracle <log_n> <seed>` checks every product limb vs GMP;
  validated at log_n 20/22/23/24.
- **Dev-box high-N GPU run** (always through the safety wrapper; background it):
  `NTT_MAX_LOGN=24 GPU_RUN_ALLOW_LONG=1 GPU_RUN_ALLOW_DESKTOP=1 bash scripts/gpu_run.sh \`
  `<timeout_s> app/compute_e/compute_e_dev -d <digits> -f` — `-f` overrides the safety gate.
  The printed `est_gpu` over-estimates ~4×, so actual ≈ `est_gpu`/4; set `<timeout_s>` to
  ~1.5× that actual (the 32M run: est ~46000 s, actual ~11600 s = 3.22 h, 5 h timeout).
- The lib/app Makefiles enforce `-Wall -Wextra`; the tree is warning-clean — keep it that way.

## 5. Objectives going forward — to deploy (all MI300A-hardware-gated)
| ID | Task |
|---|---|
| M1 | `oshcc`/SHMEM `crt_ntt` final link (only links on the MI300A cluster) |
| G2 | 4-APU CRT polymul correctness vs CPU bigint over ~200k random + all-(p-1) edges |
| G6 | rocprofv3 PMC on gfx942 (HBM/VALU/LDS) vs the gfx1030 envelope |
| G7/G8 | kernel-driven P2P benchmark; shared twiddle on APU 0 + peer access |
| G10/G11 | enable on-device scatter/gather (`-DNTT_GPU_SCATTER_GATHER` + `transfer_kernels.o` + managed operands); re-wire the call sites into `ntt_mul.hip`, then measure |
| — | confirm the Solinas-P0 VALU win on gfx942; push log_n to ~28–29 on 128 GB HBM (the 3-factor recursion already supports it) |

**Acceptance (ROADMAP §4):** clean build under PrgEnv-amd; `lib/test_ntt` 22/22; G2 exact;
`compute_e d=1e6` across 4 APUs matches OEIS A001113; PMC within the scaled envelope.

## 6. Landmines (hard-won — these will bite a fresh agent)
- **Preprocessor vs template params:** `#if (PIDX==0)` is evaluated by the preprocessor where
  `PIDX` is undefined (==0) → **always true** → silently routed every prime through `mulmod<0>`
  (the "Solinas-P0" corruption). Per-prime selection on a template param MUST be a compile-time
  `if (PIDX==0)`, never `#if`/`#elif`. (shoup.h documents the fix.)
- **GPU safety is DEV-BOX-ONLY and real:** the 6900XT drives the display; sustained compute can
  starve atomic page-flips → DRM `flip_done` timeout → hard machine hang. Mitigations on the box:
  1300 MHz clock pin, per-dispatch display-yield (`NTT_DISPLAY_YIELD_US`), `cwsr_enable=1`,
  GFXOFF disabled. **ALWAYS launch GPU work through `scripts/gpu_run.sh`** (S-guards, ledger,
  health check); **never bypass it** — bypassing has crashed the box. `gpu_run.sh` + deps were
  restored to `scripts/` as **gitignored local-only** (the rest of the GPU-safety stack is in
  `archive/scripts_devbox/`). **NONE of this applies to the MI300A** (dedicated compute, no display).
- **log_n=24 needs `NTT_MAX_LOGN=24`** (the engine default cap is 22 for safety). 23 also needs it.
- **Runtime ≈ engine estimate × 0.25.** The printed `est_gpu` is ~4× conservative (it printed
  ~46,000 s for the 32M run that actually took ~11,600 s). Budget timeouts at ~¼ of the estimate
  with margin; `compute_e` does not checkpoint — a too-short timeout wastes the whole run.
- **Never commit `archive/`, `perf/`, or `.claude/`** (gitignored). `perf/backups/` holds engine
  backups — don't delete. `git add -A` will try to stage `archive/` — don't let it.
- **Anchor cleanup `rm`s to explicit extensions** (`*.o`, named binaries), never broad globs like
  `test_*` (that once deleted 5 source files; recovered from the index).
- **The estimate gate** (`compute_e` SAFE_GPU_SECS=120) blocks long runs without `-f`; raise the
  cap and timeout deliberately, not reflexively.

## 7. Key data (worth memorizing)
**The four primes** (`lib/primes.h`): Q = ∏ ≈ 2^255; effective CRT NTT size = min = 2^24 (P1-limited).
| idx | prime (2^64 − 2^k + 1) | prim root | b (k) | max NTT |
|---|---|---|---|---|
| P0 | 0xffffffff00000001 (Goldilocks) | 7 | 32 | 2^32 |
| P1 | 0xffffffffff000001 | 43 | 24 | 2^24 |
| P2 | 0xfffffffc00000001 | 10 | 34 | 2^34 |
| P3 | 0xffffff0000000001 | 19 | 40 | 2^40 |

**digits → log_n** (compute_e on this engine): 500k→18, 1M→19, 2M→20, 8M→22, 16M→23, **32M→24**.
**VRAM ≈ 400·N bytes** (N=2^log_n): measured 2.0 GB @log_n=22, 3.75 GB @23, ~6.9 GB @24 (fits 16 GB).
**Butterfly multiply:** Shoup (precomputed quotient) for P1–P3; Solinas/Goldilocks for P0 on
gfx942 (`-DNTT_SOLINAS_P0`, default on gfx942). Reductions are full `[0,p)`.

## 8. Where to look
- **Docs:** README (entry/build) · ARCHITECTURE (design/moduli/CRT/§4 engine/§10 rationale) ·
  PERFORMANCE (measured baselines) · ROADMAP (phases/open tasks/acceptance) · STATUS (current
  state/API surface/next step) · lib|app|ref/README.md (per-layer).
- **Memory (hard-won, machine-checked facts):** `~/.claude/projects/-home-machinus-ntt/memory/`
  with `MEMORY.md` index — crash classes, the clock-pin, the Solinas bug, the 3-factor saga,
  the rename. Recalled memories appear in `<system-reminder>` blocks; verify a named file/flag
  still exists before relying on it.
- **Cross-project refs:** `~/MI300A_TARGET_ENVIRONMENT.md` (definitive target) ·
  `~/HIP_6900XT_KNOWLEDGE.md` (dev-side). The Cray module set is embedded in
  `scripts/setup_mi300a.sh` + README's build-env section.
- **History:** git log + `archive/docs/{ARCHIVE,OPT_LEDGER,GPU_CRASH_RECIPE}.md`.

## 9. Working norms (from CLAUDE.md + .claude/rules/c-style.md)
- C only (no C++ in kernels/host); the SOLE exception is the `template<int PIDX>` per-prime
  dispatch — keep it, don't add other C++. Use `__restrict__` in `.hip`.
- Hardware constants come from `hipGetDeviceProperties` at runtime — none hardcoded.
- Be token-efficient; prefer running commands over reading bytes. Source-of-truth order:
  code → STATUS.md → design docs.
- Build `-Wall -Wextra` clean before "done." Persist test/bench results to files; sweep ranges.
- Canonical `.md` files use the ANSI-boxed house style (see CLAUDE.md §6; `pretty_md.py`
  is local-only — replicate if absent).
- Commit only when the user asks; end commit messages with the Co-Authored-By line.
