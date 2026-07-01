#!/usr/bin/env python3
"""One-shot generator for GPU_CRASH_RECIPE.md (boxed ANSI format per CLAUDE.md §6).
Run once: python3 scripts/gen_gpu_crash_recipe.py ; thereafter edit the .md in place."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretty_md import box, hint, divider, section, tablize, cyan, green, red, yel, bold, write

blocks = []

blocks.append(box('G P U   C R A S H   R E C I P E   —   R X   6 9 0 0   X T   ( g f x 1 0 3 0 )'))
blocks.append(hint('GPU_CRASH_RECIPE.md'))

blocks.append(
"  How to keep compute stable on the 6900XT dev box and what to do if the\n"
"  hard-freeze crashes return. Written 2026-06-10 after CRASHes 12-22, when a\n"
"  two-run d=600000 test finally passed. This is the DISTILLED recipe; the full\n"
"  blow-by-blow lives in perf/crash_diag/gpu_run_ledger.log and the agent memory\n"
"  note [[crash21-hardware-marginal-regression]]. Read this BEFORE touching any\n"
"  amdgpu power/SMU setting or running compute on this card.")

blocks.append(divider())
blocks.append(section('1. ONE-LINE DIAGNOSIS'))
blocks.append(
"  Every hard-freeze on this card is the " + bold('SMU message layer wedging') + " (responses of\n"
"  " + cyan('0xFFFFFFFF') + " to GfxOff / TransferTableSmu2Dram). The KFD compute path itself has\n"
"  NEVER faulted. The fix is to send the SMU as little message traffic as\n"
"  possible: " + green('no clock ramps, no GFXOFF, no metrics polling') + ". It is NOT a power\n"
"  transient, NOT a dead card, and is NOT fixed by driver/ROCm/firmware swaps.")

blocks.append(divider())
blocks.append(section('2. THE WORKING CONFIGURATION', '(all four must hold)'))
blocks.append(tablize(
    ['#', 'Requirement', 'How it is set / verified'],
    [
        ['1', 'GFXOFF disabled', 'ppfeaturemask=0xffe77fff in grub+modprobe.d (bit15 clear). gpu_fix_gfxoff.sh'],
        ['2', 'OverDrive enabled', 'same mask, bit14 set — needed for the exact static pin (interface pp_od_clk_voltage)'],
        ['3', 'Static clock pin', 'sclk OD min=max=1825, mclk pinned max. gpu_static_clock.sh; auto at boot via gpu_preflight'],
        ['4', 'Zero SMU sysfs polling', 'NOTHING reads hwmon temp/freq/power OR gpu_busy_percent in a loop — all SMU-backed here'],
    ]))
blocks.append(
"  " + yel('Why each:') + " (1) GfxOff transitions wedge the SMU. (2) OD is the only RDNA2\n"
"  interface that pins an EXACT MHz (pp_dpm_sclk's middle level is a floating\n"
"  'current freq' readout — index-pinning it landed 2565MHz, not 1825). (3) the\n"
"  idle->boost ramp at launch is the wedge window; min=max removes the ramp.\n"
"  (4) reading hwmon temp/freq/power AND gpu_busy_percent each issues\n"
"  TransferTableSmu2Dram (gpu_busy_percent is SMU-metrics-backed on this ASIC —\n"
"  CRASH 23). The debug telemetry harness was CAUSING crashes: ~150 such reads\n"
"  per run (1Hz telemetry + 5s postwatch + post-check) wedged longer jobs.\n"
"  Treat EVERY hwmon/busy/clock sysfs read as an SMU poke. Also CLOSE desktop\n"
"  GPU monitors (gnome-system-monitor etc.) — they poll metrics too.")

blocks.append(divider())
blocks.append(section('3. APPLY FROM SCRATCH', '(e.g. after a fresh OS / mask reset)'))
blocks.append(
"  " + cyan('Step 1') + "  sudo bash scripts/gpu_fix_gfxoff.sh      # mask 0xffe77fff -> grub+modprobe.d\n"
"  " + cyan('Step 2') + "  FULL COLD POWER CYCLE                    # (shutdown, off 10s, on)\n"
"  " + cyan('Step 3') + "  verify:  grep -i gfxoff /sys/class/drm/card1/device/pp_features  -> disabled\n"
"  " + cyan('Step 4') + "  verify:  ls /sys/class/drm/card1/device/pp_od_clk_voltage         -> present\n"
"  " + cyan('Step 5') + "  sudo bash scripts/gpu_static_clock.sh 1825   # OD min=max pin (boot does this too)\n"
"  " + cyan('Step 6') + "  run:  GPU_RUN_ALLOW_DESKTOP=1 COMPUTE_E_ALLOW_VSCODE=1 \\\n"
"            bash scripts/gpu_run.sh 90 lib3/compute_e/compute_e_dev_l64 -d 600000\n"
"  The boot preflight (gpu_preflight.sh §5b) re-applies the pin every boot, so\n"
"  steps 5 is normally automatic; gpu_run S27 refuses to launch without it.")

blocks.append(divider())
blocks.append(section('4. HARD RULES', '(violating any of these has crashed the box)'))
blocks.append(tablize(
    ['Rule', 'Reason'],
    [
        [red('NEVER set amdgpu.dpm=0'), 'Sienna Cichlid fails driver probe (-95, smu fw load) -> unbootable/black screen'],
        [red('NEVER warm-reboot after a hang'), 'SMU stays wedged across warm reset; only a COLD power cycle clears it'],
        [red('NEVER read SMU-metrics during fragility'), 'hwmon temp/freq/power + pp_features reads can wedge/D-state-hang a sick SMU'],
        [red('NEVER grub-edit by append'), 'stacked amdgpu params (esp. stale dpm=0) broke boot — scripts REPLACE the set'],
        [red('NEVER run on a crash-recovery boot'), 'gpu_health_check 0-SMU probe blocks this; a cold cycle is required first'],
    ]))

blocks.append(divider())
blocks.append(section('5. IF THE CRASHES RETURN', '(diagnostic order)'))
blocks.append(
"  " + bold('First confirm the config is intact') + " (most regressions are a lost setting):\n"
"    cat /sys/class/drm/card1/device/power_dpm_force_performance_level   # -> manual\n"
"    grep -i gfxoff /sys/class/drm/card1/device/pp_features              # -> disabled\n"
"    grep '\\*' /sys/class/drm/card1/device/pp_dpm_sclk                   # -> 1825Mhz *\n"
"  " + bold('Then look for a NEW SMU-metrics reader') + " (the usual culprit):\n"
"    a desktop GPU monitor (mission-center/corectrl/psensor), a re-introduced\n"
"    telemetry field, or rocprofv3/PMC (NEVER run PMC autonomously — separate rule).\n"
"  " + bold('Capture evidence') + ": the crash boot's journal tail shows the SMU 0xFFFFFFFF\n"
"    flood; if absent it was lost to the freeze (journald async). The post-run\n"
"    watch log (run_*_postwatch.log, fsynced) survives delayed freezes.\n"
"  " + bold('Only if config is intact AND no new poller AND it still wedges') + " on a single\n"
"    pinned run -> the SMU silicon has degraded further -> RMA/replace the card.\n"
"    (dpm=0 is NOT an escape hatch here — it does not boot.)")

blocks.append(divider())
blocks.append(section('6. KEY FILES'))
blocks.append(tablize(
    ['File', 'Role'],
    [
        ['scripts/gpu_fix_gfxoff.sh', 'writes ppfeaturemask=0xffe77fff (GFXOFF off + OD on), grub+modprobe.d'],
        ['scripts/gpu_static_clock.sh', 'OD static pin sclk=1825 min=max + mclk max; "auto" reverts'],
        ['scripts/gpu_preflight.sh', '§5b auto-applies the pin each boot (root service amdgpu-preflight)'],
        ['scripts/gpu_run.sh', 'S23 cmdline GFXOFF check (SMU-silent), S27 requires perf=manual'],
        ['scripts/gpu_health_check.sh', '0-SMU detector: blocks runs when SMU EPERMs/hangs (wedged boot)'],
        ['scripts/gpu_dpm_off.sh', 'DISABLED (apply refuses) — dpm=0 unbootable here; kept for "undo" only'],
        ['perf/crash_diag/gpu_run_ledger.log', 'full crash-by-crash history (local, not in git)'],
    ]))

blocks.append(divider())
blocks.append(section('7. SCALING d SAFELY', '(the governing invariant)'))
blocks.append(
"  " + bold('SMU messages per session must be O(1) in d, never O(run-duration).') + "\n"
"  The card tolerates only a handful of SMU messages; any per-second polling\n"
"  loop spends that budget faster on longer (larger-d) runs and wedges it —\n"
"  that is exactly how d=1.1M crashed after d=600k passed (CRASH 23).\n"
"  " + green('Invariant now enforced:') + " no run-path loop reads any SMU node; per-run SMU\n"
"  traffic = one health-probe read + KFD SetWorkloadMask (2/context, fixed).\n"
"  So d=10M should be no more SMU-risky than d=600k. When raising d:\n"
"    - close ALL desktop GPU monitors and stop power-profiles-daemon (S28 gates this)\n"
"    - step up gradually (600k -> 800k -> 1.1M -> 1.5M ...), verify each before the\n"
"      next; NEVER jump size on a thin pass (2 good runs is not proof)\n"
"    - the ONLY residual SMU traffic (KFD workload-mask + display events) is fixed-\n"
"      count; if a single pinned run still wedges, it is hardware -> RMA.")

blocks.append(divider())
blocks.append(section('8. PERFORMANCE NOTE'))
blocks.append(
"  The 1825MHz static pin costs " + green('~0.7s per d=600000 run') + " vs full 2.5GHz boost\n"
"  (10.25s vs ~9.5s) — negligible. This card is DEV hardware for correctness\n"
"  validation; the project's performance target is the MI300A, which has a\n"
"  dedicated compute queue and none of this SMU fragility. Do not spend effort\n"
"  recovering the lost boost on the 6900XT.")

write(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'GPU_CRASH_RECIPE.md'), blocks)
print('wrote GPU_CRASH_RECIPE.md')
