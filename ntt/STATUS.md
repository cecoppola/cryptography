
## Phase Progress

Phase 4 — MI300A Tuning & Deploy. ALL dev-side work complete (through 2026-06-30);
the remainder is MI300A-target-only (4-APU SHMEM link M1, G6-G11, gfx942 runtime).

Repo reorganized for MI300A delivery (2026-06-30): lib1→ref, lib2→lib, lib3→app;
retired/dev-only material moved to archive/ (docs, lib_dead, scripts_devbox, tools,
logs). gfx942 cross-compile (`make all-mi300a`) re-verified clean after the move.

2026-06-30 milestones: log_n=24 validated end-to-end — a full compute_e to 32,000,000
digits completed in 3.22 h on the 6900XT, OEIS-correct (the 3-factor real-workload
proof, not just the oracle). Makefiles slimmed (597->185-line root); every source TU
compiles clean under -Wall -Wextra; source comments + MD docs audited for accuracy.

Recent work (2026-07-02):
- Review-driven hardening + 4 new host suites (garner CRT, mem_pool unit, arith
  alias-safety, newton div vs GMP) + 1 GPU death-test. make check 17→21/21.
- bigint_shr bug found+fixed by the alias suite: c==a with a limb-aligned shift did an
  overlapping memcpy (UB) despite claiming alias-safety — now memmove (ASan-clean, LB 64/112).
- newton fuzz vs GMP covers both div regimes (bits_N<=2b and the >2b fresh-reciprocal
  path) + reciprocal cross-check; corrected newton.c's stale "exact floor" comment
  (it's a tight lower bound the division correction absorbs; exact only for large Q).
- mem_pool.c rewritten no-op→recycling slab pool; ntt_bigint_mul gained a cyclic-wrap
  size guard (GPU death-test test_wrap_guard: oversized→exit(1), valid→exit(0), verified
  on gfx1030 via gpu_run.sh).

Recent work (2026-06-28 → 06-29):
- 3-factor recursive 4-step NTT → log_n=24 (~39M digits), GMP-verified. Fixes: BUG#1
  inverse inner cross-twiddle (conjugate); BUG#2 inner+outer 4-steps needed natural-I/O
  transposes (T_init/T_final) — masked by sparse/single-block inputs.
- Solinas-P0 correctness bug found+FIXED: `#elif (PIDX==0)` is a PREPROCESSOR test where
  the template param is undefined(==0) → ALL primes used mulmod<0>. Fixed with compile-time
  `if(PIDX==0)`; GMP-verified; restored as the gfx942 default.
- F6: single arith/ntt_mul.hip engine (transfer_gpu.hip retired). G9: KEEP templates
  (A/B ~7% faster than runtime switch). R8/I15: reintroduced opt-in (-DNTT_I15, default
  OFF) — GMP-verified but +4.5% slower on gfx1030 (uncoalesced vs LDS-tiled transpose).
- Hardening: preprocessor-on-template scan (clean) + non-vacuity gate green (+test-modops).
- PMC re-captured 2026-06-29 (no crash, gpu_run.sh path): hot kernels compute-bound.
- Resolved crash narratives + crash-avoidance recipe now in archive/docs/ (ARCHIVE.md,
  GPU_CRASH_RECIPE.md) — dev-box 6900XT only, not relevant to the MI300A target.

## Component Status

| Component         | State                                                          |
|-------------------|----------------------------------------------------------------|
| ref              | done — 14 primes, Shoup butterfly                              |
| lib NTT          | done — 2-factor (log_n<=23) + 3-factor (log_n=24, gated); I13/I14/C1 |
| lib pipeline     | done — A1+A2+A3+A4+D1; single engine since F6                  |
| lib arith        | done — newton O(1); I15 transposed scatter/Garner opt-in (-DNTT_I15) |
| app compute_e    | done — log_n=24 GMP-verified; 32M-digit run completed 3.22h, OEIS-correct (2026-06-30) |
| make check        | 21/21 host PASS (+lib garner CRT, +app mem_pool unit, +lib arith alias-safety, +lib newton div vs GMP; 2026-07-02); non-vacuity gate 12/12 NON-VACUOUS (+test-modops) |
| GMP oracle        | test_gmp_oracle: log_n=20/22/23/24 vs GMP all PASS (run via gpu_run.sh) |
| gfx942 xcompile   | clean (0 warn), incl. -DNTT_SOLINAS_P0 default                 |
| PMC (C1/C2)       | re-captured 2026-06-29 (compute-bound); X3_PMC_GFX1030_20260629_181807.md |

## GPU Safety (dev-box only — archived)

The full GPU-safety architecture, dev-box TDR/hardware parameters, and the resolved
CRASH 8-22 narratives now live in archive/docs/GPU_CRASH_RECIPE.md (6900XT display-
sharing class — not relevant to the MI300A compute-only target). Dev-box GPU runs go
through scripts/gpu_run.sh (restored local-only from archive/scripts_devbox/).

## API Surface

Key functions in lib/arith/ntt_mul.hip (single CRT-NTT engine since F6, 2026-06-27):
- ntt_mul_init_impl(int max_log_n) — allocates NttMulState, precomputes n_inv + twiddles;
  enables all-pairs peer access when n_devs>1 (2026-07-02, mirrors main.hip; no-op on the
  single-GPU dev box) so the cross-device scatter-reads-d_in and device-0 Garner-reads-d_a[1..3]
  don't fault on 4-APU. Runtime-validated on MI300A only (DEPLOYMENT_DIAGNOSTIC P6 resolved in code)
- ntt_bigint_mul(BigInt *c, *a, *b) — full CRT-NTT multiply pipeline (A1/A4 concurrent);
  crash-safety display yield + env log_n cap; GMP-verified to log_n=24 (3-factor path).
  Routes by log_n: even 12-22 → 2-factor Stockham; odd>10 → CT-DIT (Shoup); ==24 → 3-factor.
  Guards n_a+n_b-1 <= N (2026-07-02): aborts loudly if operands would overflow the fixed
  size-N cyclic convolution (silent-wrap footgun); compute_e sizes max_log_n with 2x margin.
- ntt_mul_teardown_impl() — frees all GPU state including the 3-factor + I15 buffers.
  (lib test_ntt links arith/ntt_mul{,_dev}.o; app compute_e already used this engine.)
- I15 opt-in (-DNTT_I15): scatter_kernel_t + garner_reconstruct_gpu_t eliminate the two
  even-log_n transpose_sq passes (GMP-verified; default OFF — slower on gfx1030).

Key kernels in lib/ntt_kernel.hip:
- stockham_kernel<PIDX, LOG_N_SUB>(data, tw) — C1 hot Stockham butterfly (constexpr STOK_STRIDE_C)
- launch_transpose_sq(data, M, stream) — in-place M×M uint64_t transpose (now exported)
- launch_ntt_ct_dit_mont<PIDX>(data, tw_mont, log_n, stream) — I4 CT-DIT-Mont
- launch_intt_ct_dit_mont<PIDX>(data, tw_inv_mont, n_inv, log_n, stream) — I4 inverse

GPU Garner in lib/garner.hip:
- garner_reconstruct_gpu(r0, r1, r2, r3, d_out, N, g, stream) — N parallel threads on device 0

## Deviations from Design Doc

Engine fork — RESOLVED (F6, 2026-06-27). ntt_bigint_mul was defined twice
(transfer_gpu.hip + arith/ntt_mul.hip, diverged). Consolidated to the single
arith/ntt_mul.hip engine: lib test_ntt/test_ntt_dev now link arith/ntt_mul{,_dev}.o;
crt_ntt was nm-verified to need nothing from transfer_gpu.o and dropped it.
transfer_gpu.hip retired (now in archive/lib_dead/) — kept only as the I15
transposed-scatter + gated GPU scatter/gather/CLA reference for the later R8
reintroduction. Verified: test_ntt_dev PASS @1300MHz (gfx1030), test_ntt builds
(gfx942). crt_ntt full link is MI300A-only (oshcc/SHMEM) — pending on target.

Resolved since: G9 DECIDED (keep templates, 2026-06-29); I16 Solinas-P0 FIXED (preprocessor
PIDX bug) + restored gfx942 default. Remaining MI300A-target-only: G6/G7/G8/M1 (4-APU + gfx942
runtime), G10/G11 (enable GPU scatter/gather on unified memory).

## Next Step

The dev-box phase is complete. The log_n=24 ceiling is validated end-to-end: a full
compute_e to 32,000,000 digits completed in 3.22 h on the 6900XT (2026-06-30), OEIS-
correct, via the 3-factor recursive 4-step path (NTT_MAX_LOGN=24) — the real-workload
proof beyond the GMP oracle. The repo is delivery-ready: reorganized (ref/lib/app +
archive/), Makefiles slimmed, every source TU warning-clean, comments + MD docs audited.

Only MI300A-target work remains: M1 SHMEM crt_ntt final link; G6/G7/G8 4-APU + occupancy
on real gfx942; G10/G11 enable GPU scatter/gather on unified memory; confirm the
Solinas-P0 VALU win + larger log_n (28-29) on 128 GB HBM. Ship path: cross-compile gfx942
(clean) → deploy MI300A → run the acceptance suite (ROADMAP §4).
