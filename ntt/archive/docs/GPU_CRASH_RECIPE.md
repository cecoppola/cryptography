[1;37m╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                     G P U   C R A S H   R E C I P E   —   R X   6 9 0 0   X T   ( g f x 1 0 3 0 )                      ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat GPU_CRASH_RECIPE.md   or   less -R GPU_CRASH_RECIPE.md[0m

  How to keep compute stable on the 6900XT dev box and what to do if the
  hard-freeze crashes return. Written 2026-06-10 after CRASHes 12-22; clock updated 2026-06-16 (§2). When a
  two-run d=600000 test finally passed. This is the DISTILLED recipe; the full
  blow-by-blow lives in perf/crash_diag/gpu_run_ledger.log and the agent memory
  note [[crash21-hardware-marginal-regression]]. Read this BEFORE touching any
  amdgpu power/SMU setting or running compute on this card.

  SOURCES FOR THE 1300 MHz FIX (later than the 2026-06-10 SMU-class notes — the
  clock-regime wedge was found and fixed afterward, then we stopped discussing it):
    - agent memory [[log-n-21-ceiling-display-gpu]]  (V/f map: 1000 stable, 1500/1825 wedge)
    - agent memory [[log-n-22-ntt-correctness-bug]]  (log_n=22 ceiling, GMP-verified)
    - SESSION TRANSCRIPT (where 1300 was confirmed + applied; richest context):
        ~/.claude/projects/-home-machinus-ntt/2ce85f8f-31cf-4ff2-ae97-8684ad2c42c5.jsonl
      (51 MB; the log_n=22 fix + 1300 MHz binary-search + 3-factor-NTT-to-log_n=24 session).
    Cite/read these — NOT the older 1825-era notes — for the current clock regime.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  1. ONE-LINE DIAGNOSIS[0m

  Every hard-freeze on this card is the [1;37mSMU message layer wedging[0m (responses of
  [1;36m0xFFFFFFFF[0m to GfxOff / TransferTableSmu2Dram). The KFD compute path itself has
  NEVER faulted. The fix is to send the SMU as little message traffic as
  possible: [1;32mno clock ramps, no GFXOFF, no metrics polling[0m. It is NOT a power
  transient, NOT a dead card, and is NOT fixed by driver/ROCm/firmware swaps.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  2. THE WORKING CONFIGURATION  (all four must hold)[0m

  ┌───┬────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────┐
  │ [1;36m#[0m │ [1;36mRequirement[0m            │ [1;36mHow it is set / verified[0m                                                                   │
  ├───┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 1 │ GFXOFF disabled        │ ppfeaturemask=0xffe77fff in grub+modprobe.d (bit15 clear). gpu_fix_gfxoff.sh               │
  │ 2 │ OverDrive enabled      │ same mask, bit14 set — needed for the exact static pin (interface pp_od_clk_voltage)       │
  │ 3 │ Static clock pin       │ sclk OD min=max=1300, mclk pinned max. gpu_static_clock.sh; auto at boot via gpu_preflight │
  │ 4 │ Zero SMU sysfs polling │ NOTHING reads hwmon temp/freq/power OR gpu_busy_percent in a loop — all SMU-backed here    │
  └───┴────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────┘

  [1;33mWhy each:[0m (1) GfxOff transitions wedge the SMU. (2) OD is the only RDNA2
  interface that pins an EXACT MHz (pp_dpm_sclk's middle level is a floating
  'current freq' readout — index-pinning it landed 2565MHz, not 1825). (3) the
  idle->boost ramp at launch is the wedge window; min=max removes the ramp.
  (4) reading hwmon temp/freq/power AND gpu_busy_percent each issues
  TransferTableSmu2Dram (gpu_busy_percent is SMU-metrics-backed on this ASIC —
  CRASH 23). The debug telemetry harness was CAUSING crashes: ~150 such reads
  per run (1Hz telemetry + 5s postwatch + post-check) wedged longer jobs.
  Treat EVERY hwmon/busy/clock sysfs read as an SMU poke. Also CLOSE desktop
  GPU monitors (gnome-system-monitor etc.) — they poll metrics too.

[1;31m  CRITICAL — CLOCK VALUE (updated 2026-06-16, large-run regime):[0m
  The pin is [1;36m1300 MHz[0m, NOT 1825. This is a SECOND, distinct failure mode from
  the SMU wedge above: at [1;31m1825 MHz under sustained 64-bit integer multiply[0m the
  GFX ring silently stops, fences stop signalling, the hung-task watchdog fires,
  and the box hard-locks. Voltage does not fix it — it is a clock-regime
  stability ceiling. [1;32m1300 MHz proven ~2 h; 1200 MHz proven 88 min[0m. 1825 is OK
  only for SHORT/SMALL cells (the check-gpu gate). For ANY sustained/large run
  (compute_e log_n>=20, big test_gmp_oracle probes) the clock MUST be 1300, and
  it can drift back to 1825 without a reboot — VERIFY it right before launch.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  3. APPLY FROM SCRATCH  (e.g. after a fresh OS / mask reset)[0m

  [1;36mStep 1[0m  sudo bash scripts/gpu_fix_gfxoff.sh      # mask 0xffe77fff -> grub+modprobe.d
  [1;36mStep 2[0m  FULL COLD POWER CYCLE                    # (shutdown, off 10s, on)
  [1;36mStep 3[0m  verify:  grep -i gfxoff /sys/class/drm/card1/device/pp_features  -> disabled
  [1;36mStep 4[0m  verify:  ls /sys/class/drm/card1/device/pp_od_clk_voltage         -> present
  [1;36mStep 5[0m  sudo bash scripts/gpu_static_clock.sh 1300   # OD min=max pin (boot does this too)
  [1;36mStep 6[0m  run:  GPU_RUN_ALLOW_DESKTOP=1 COMPUTE_E_ALLOW_VSCODE=1 \
            bash scripts/gpu_run.sh 90 lib3/compute_e/compute_e_dev_l64 -d 600000
  The boot preflight (gpu_preflight.sh §5b) re-applies the pin every boot, so
  steps 5 is normally automatic; gpu_run S27 refuses to launch without it.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  4. HARD RULES  (violating any of these has crashed the box)[0m

  ┌─────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────┐
  │ [1;36mRule[0m                                    │ [1;36mReason[0m                                                                          │
  ├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ [1;31mNEVER set amdgpu.dpm=0[0m                  │ Sienna Cichlid fails driver probe (-95, smu fw load) -> unbootable/black screen │
  │ [1;31mNEVER warm-reboot after a hang[0m          │ SMU stays wedged across warm reset; only a COLD power cycle clears it           │
  │ [1;31mNEVER read SMU-metrics during fragility[0m │ hwmon temp/freq/power + pp_features reads can wedge/D-state-hang a sick SMU     │
  │ [1;31mNEVER grub-edit by append[0m               │ stacked amdgpu params (esp. stale dpm=0) broke boot — scripts REPLACE the set   │
  │ [1;31mNEVER run on a crash-recovery boot[0m      │ gpu_health_check 0-SMU probe blocks this; a cold cycle is required first        │
  └─────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  5. IF THE CRASHES RETURN  (diagnostic order)[0m

  [1;37mFirst confirm the config is intact[0m (most regressions are a lost setting):
    cat /sys/class/drm/card1/device/power_dpm_force_performance_level   # -> manual
    grep -i gfxoff /sys/class/drm/card1/device/pp_features              # -> disabled
    grep '\*' /sys/class/drm/card1/device/pp_dpm_sclk                   # -> 1300Mhz *
  [1;37mThen look for a NEW SMU-metrics reader[0m (the usual culprit):
    a desktop GPU monitor (mission-center/corectrl/psensor), a re-introduced
    telemetry field, or rocprofv3/PMC (NEVER run PMC autonomously — separate rule).
  [1;37mCapture evidence[0m: the crash boot's journal tail shows the SMU 0xFFFFFFFF
    flood; if absent it was lost to the freeze (journald async). The post-run
    watch log (run_*_postwatch.log, fsynced) survives delayed freezes.
  [1;37mOnly if config is intact AND no new poller AND it still wedges[0m on a single
    pinned run -> the SMU silicon has degraded further -> RMA/replace the card.
    (dpm=0 is NOT an escape hatch here — it does not boot.)

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  6. KEY FILES[0m

  ┌────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
  │ [1;36mFile[0m                               │ [1;36mRole[0m                                                                   │
  ├────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ scripts/gpu_fix_gfxoff.sh          │ writes ppfeaturemask=0xffe77fff (GFXOFF off + OD on), grub+modprobe.d  │
  │ scripts/gpu_static_clock.sh        │ OD static pin sclk=1300 min=max + mclk max; "auto" reverts                     │
  │ scripts/gpu_preflight.sh           │ §5b auto-applies the pin each boot (root service amdgpu-preflight)     │
  │ scripts/gpu_run.sh                 │ S23 cmdline GFXOFF check (SMU-silent), S27 requires perf=manual        │
  │ scripts/gpu_health_check.sh        │ 0-SMU detector: blocks runs when SMU EPERMs/hangs (wedged boot)        │
  │ scripts/gpu_dpm_off.sh             │ DISABLED (apply refuses) — dpm=0 unbootable here; kept for "undo" only │
  │ perf/crash_diag/gpu_run_ledger.log │ full crash-by-crash history (local, not in git)                        │
  └────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  7. SCALING d SAFELY  (the governing invariant)[0m

  [1;37mSMU messages per session must be O(1) in d, never O(run-duration).[0m
  The card tolerates only a handful of SMU messages; any per-second polling
  loop spends that budget faster on longer (larger-d) runs and wedges it —
  that is exactly how d=1.1M crashed after d=600k passed (CRASH 23).
  [1;32mInvariant now enforced:[0m no run-path loop reads any SMU node; per-run SMU
  traffic = one health-probe read + KFD SetWorkloadMask (2/context, fixed).
  So d=10M should be no more SMU-risky than d=600k. When raising d:
    - close ALL desktop GPU monitors and stop power-profiles-daemon (S28 gates this)
    - step up gradually (600k -> 800k -> 1.1M -> 1.5M ...), verify each before the
      next; NEVER jump size on a thin pass (2 good runs is not proof)
    - the ONLY residual SMU traffic (KFD workload-mask + display events) is fixed-
      count; if a single pinned run still wedges, it is hardware -> RMA.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  8. PERFORMANCE NOTE[0m

  The 1300MHz static pin (down from a 1825 attempt that hard-locked sustained
  runs) costs [1;32m~0.7s per d=600000 run[0m vs full 2.5GHz boost
  (10.25s vs ~9.5s) — negligible. This card is DEV hardware for correctness
  validation; the project's performance target is the MI300A, which has a
  dedicated compute queue and none of this SMU fragility. Do not spend effort
  recovering the lost boost on the 6900XT.
