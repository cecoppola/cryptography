[1;37m╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                  N T T   /   M I 3 0 0 A   —   C O D E   R E V I E W                                   ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat REVIEW.md   or   less -R REVIEW.md[0m

  Total review of the NTT/MI300A codebase (lib1 single-prime · lib2 4-prime
  CRT engine · lib3 compute_e; ~21.3k LOC), conducted on the CPU dev box
  (no GPU). Phases P0-P7: baseline, build hygiene, host correctness, per-layer
  read, stability, goal alignment, organization. 12 findings; 10 fixed here,
  2 flagged for the MI300A session, 1 doc-decision, 1 observation. Conducted
  2026-06-24. Verification: all host suites + ASan/UBSan green; whole-tree
  -Wall -Wextra clean (gfx942+gfx1030); non-vacuity gate 11/11.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  1. FINDINGS[0m

  ┌─────┬──────┬─────────┬─────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │ [1;36mID[0m  │ [1;36mSev[0m  │ [1;36mStatus[0m  │ [1;36mArea[0m        │ [1;36mSummary[0m                                                                               │
  ├─────┼──────┼─────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ F2  │ [1;31mHIGH[0m │ [1;32mFIXED[0m   │ build/test  │ compute_e_host ran GPU guard -> host OEIS differential could not run; gated out       │
  │ F3  │ [1;31mHIGH[0m │ [1;31mGPU[0m     │ build/test  │ test_gmp_oracle (per-limb vs GMP) had no build rule; wired in (run on GPU)            │
  │ F6  │ [1;31mHIGH[0m │ [1;31mGPU[0m     │ structure   │ ntt_bigint_mul defined twice (lib2/lib3), DIVERGED (I15, odd-log_n algo); consolidate │
  │ F1  │ [1;33mMED [0m │ [1;32mFIXED[0m   │ correctness │ ntt_bigint_mul had no header prototype (implicit-decl); declared in multiply.h        │
  │ F4  │ [1;33mMED [0m │ [1;32mFIXED[0m   │ test infra  │ non-vacuity gate covered only lib1; extended to lib2/lib3 (11/11)                     │
  │ F8  │ [1;33mMED [0m │ [1;32mFIXED[0m   │ consistency │ c-style "no templates" vs 93 template<PIDX> sites + G9; rule aligned                  │
  │ F12 │ [1;33mMED [0m │ [1;33mPROPOSE[0m │ docs        │ STATUS.md non-canonical kitchen-sink; STATUS-vs-ROADMAP entry-point conflict          │
  │ F5  │ [1;36mLOW [0m │ [1;32mFIXED[0m   │ test infra  │ no direct bigint_mul vs GMP oracle; added (200/200 x2)                                │
  │ F3b │ [1;36mLOW [0m │ [1;32mFIXED[0m   │ docs        │ stale "gmp.h not installed" comment                                                   │
  │ F7  │ [1;36mLOW [0m │ [1;32mFIXED[0m   │ docs        │ main.c annotated emergency_reset as hipDeviceReset (it omits it; CRASH 14)            │
  │ F9  │ [1;36mLOW [0m │ [1;32mFIXED[0m   │ naming      │ use_ct_dit dispatches I11 rect, not CT-DIT; clarified                                 │
  │ F10 │ [1;36mLOW [0m │ [1;32mFIXED[0m   │ docs        │ STATUS.md "no deviations" omitted the engine fork; updated                            │
  │ F11 │ OBS  │ [1;36mOBSERVE[0m │ vcs         │ core engine + docs untracked in git (mid-migration); user handles VC                  │
  └─────┴──────┴─────────┴─────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  2. PRIORITIZED REMEDIATION    (what remains)[0m

  [1;31m1. F6 — consolidate the two engines[0m (GPU-verified refactor). The single
     biggest structural issue: transfer_gpu.hip (lib2) and arith/ntt_mul.hip
     (lib3) are two ~800-LOC ntt_bigint_mul orchestrations that have diverged —
     lib2 has I15 transposed scatter + I11 rect for odd log_n; lib3 lacks I15
     and keeps I11 disabled (CT-DIT). A fix to one won't reach the other.
     Action: pick one engine, delete the duplicate, re-verify compute_e d=1e6
     + lib2 test_ntt 22/22 on GPU.
  [1;31m2. F3 — run test_gmp_oracle on GPU[0m (now buildable): per-limb GPU-multiply
     vs GMP; sweep log_n toward the ceiling to pin any log_n=22+ correctness wall.
  [1;33m3. F12 — doc-architecture decision[0m: make ROADMAP the sole navigator and
     slim STATUS.md (CRASH narratives -> ARCHIVE, hw params -> PERFORMANCE), or
     drop STATUS.md. Resolves the STATUS-vs-ROADMAP entry-point conflict.
  [1;36m4. test-modops non-vacuity[0m on gfx1030 — the one host suite not yet in the
     non-vacuity gate (it is a .hip needing a GPU).

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  3. GPU-GATED BACKLOG    (MI300A / dev-GPU session)[0m

  - F6 engine consolidation + re-verify (compute_e d=1e6, test_ntt 22/22).
  - F3 test_gmp_oracle run + log_n sweep.
  - test-modops fault-injection (extend verify_nonvacuous once on GPU).
  - Existing ROADMAP Phase-4 criteria: G2 4-APU polymul, G6 PMC, G9 templates
    A/B, 4-APU compute_e vs OEIS, make all under PrgEnv-amd.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  4. POSITIVE CONFIRMATIONS    (no action)[0m

  [1;32mCorrectness[0m: all host suites pass (lib1+lib2+lib3) and are ASan+UBSan
     clean; empirically non-vacuous (gate now 11/11, lib1+lib2+lib3). Arithmetic
     is -Wconversion clean; differential oracles vs GMP + OEIS independent.
  [1;32mStability[0m: every hipMalloc HIP_CHECK-wrapped; hipDeviceReset reachable
     only from the signal handler and even there omitted (CRASH 14); gpu_run.sh
     guard stack (flock mutex, log_n cap, crash-aware, SMU-wedge/crash-boot) is
     comprehensive and consistent with the documented crash history.
  [1;32mBuild[0m: whole-tree -Wall -Wextra clean on gfx942 AND gfx1030.
  [1;32mParameterization[0m: warpSize runtime-queried; log_n per-call; q fixed by
     CRT design (inherent). Follows the c-style parameterization rule.
