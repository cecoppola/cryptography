#!/usr/bin/env bash
# gpu_run.sh — MANDATORY wrapper for every GPU binary run on the 6900XT.
#
# Why: killing a HIP process mid-kernel strands the amdgpu ring → GPU reset
# (this crashed the dev box repeatedly on 2026-05-16). This wrapper:
#   1. PRE-CHECK : refuses to launch if the GPU is already busy/faulted
#                  (checked via sysfs — no device open, no rocm-smi).
#   2. RUN       : runs the binary with a timeout that escalates GENTLY —
#                  SIGINT → SIGTERM → (last resort) SIGKILL, with grace gaps
#                  so the process can finish/abort its kernels cleanly.
#   3. POST-CHECK: after exit, polls sysfs until GPU% is idle again;
#                  if it stays busy, reports UNHEALTHY and exits 99 so the
#                  caller STOPS instead of launching the next run.
#   NOTE: rocm-smi / rocminfo are NOT used — they open /dev/kfd which can
#         crash a post-panic fragile driver (gfx1030 confirmed 2026-06-06).
#
# Usage:  scripts/gpu_run.sh <timeout_secs> <binary> [args...]
# Exit  : program's own code, 124 if it had to be terminated, 99 if a guard
#         tripped or the GPU did not return to a healthy idle state.
#
# Layered safeguards:
#   S1 wrapper · S6 post-idle cooldown · S7 log_n cap · S9 GPU mutex
#   (one job machine-wide; 2 agents share this box) · S10 hardened
#   pre-check (GPU% + VRAM% + stray HIP procs) · S11 adaptive cooldown
#   (burst throttle) · S13 persistent run ledger (survives reboot) ·
#   S15 crash-aware guard (refuse to blindly repeat a job that never
#   returned). Env: GPU_RUN_COOLDOWN, GPU_RUN_MAX_LOGN, GPU_RUN_LOCK_WAIT,
#   GPU_RUN_FORCE=1 (override S15).
set -u

TMO="${1:?usage: gpu_run.sh <timeout_secs> <binary> [args...]}"; shift
BIN="${1:?binary required}"; shift || true

# ── S16+S17 (INCIDENT.md §7: 2026-05-23 rocprofv3 hard hang) ─────────────────
# Profiler tooling (rocprofv3, hipprof, omniperf) spawns a GPU binary
# internally. The wrapper protects the spawner — but if the underlying
# binary runs a long internal sweep WHILE every kernel is instrumented,
# the sustained load can still hard-lock RDNA2.
#
# S16: require a kernel filter (any of --kernel-include-regex /
#      --kernel-trace / --kernel-rename) so instrumentation cost is
#      bounded. Verified by host TEST 1 (rejects unfiltered).
# S17: even with a kernel filter, the polymul binary's hardcoded
#      selftest + 4×4 negacyclic sweep is too much sustained load under
#      PMC on RDNA2 (verified: hung the box twice on 2026-05-23, the
#      second time with S16 + --kernel-include-regex + ITERS=100).
#      Require argv to include `--bench-only` (added to the polymul
#      binary 2026-05-23) so the binary skips the selftest + sweep
#      and only the bounded run_benchmark section runs.
case "$(basename "$BIN")" in
  rocprofv3|hipprof|omniperf)
    has_filter=0; profiled_bin=""
    for a in "$@"; do
      case "$a" in
        --kernel-include-regex*|--kernel-trace*|--kernel-rename*) has_filter=1 ;;
        bin/ntt_gpu_polymul_*|*/bin/ntt_gpu_polymul_*) profiled_bin="$a" ;;
      esac
    done
    if [ "$has_filter" = "0" ] && [ "${GPU_RUN_FORCE:-0}" != "1" ]; then
      echo "[gpu_run] ABORT (S16): $(basename "$BIN") invoked without"
      echo "          --kernel-include-regex / --kernel-trace. Heavy"
      echo "          instrumented sweeps can hard-lock RDNA2 (INCIDENT §7)."
      echo "          Set GPU_RUN_FORCE=1 to override deliberately."
      exit 99
    fi
    if [ -n "$profiled_bin" ] && [ "${GPU_RUN_FORCE:-0}" != "1" ]; then
      bench_only=0
      for a in "$@"; do
        [ "$a" = "--bench-only" ] && bench_only=1
      done
      if [ "$bench_only" = "0" ]; then
        echo "[gpu_run] ABORT (S17): $(basename "$BIN") -- ${profiled_bin}"
        echo "          requires the polymul binary's --bench-only flag."
        echo "          The polymul binary runs a hardcoded selftest +"
        echo "          4x4 negacyclic sweep before run_benchmark; under"
        echo "          PMC instrumentation on RDNA2 this hard-hangs the"
        echo "          GPU (INCIDENT §7, two reproductions 2026-05-23)."
        echo "          Pass --bench-only to skip the sweep. Override:"
        echo "          GPU_RUN_FORCE=1 (deliberate)."
        exit 99
      fi
    fi
    ;;
esac

# ── S6/S7 (INCIDENT.md §5: 2026-05-17 hard host lock) ─────────────────────────
# S6: mandatory post-idle COOLDOWN before exit so consecutive heavy GPU runs
#     cannot fire back-to-back (the sustained-load pattern that hard-locked the
#     6900XT). 30s default: after hipDeviceReset and process exit, the ROCm HSA
#     runtime runs hsa_shut_down() async on the compute ring; a second run
#     opening /dev/kfd before that completes races with the KFD driver cleanup
#     and crashes the machine (confirmed 2026-06-07/08).
#     Override: GPU_RUN_COOLDOWN=<secs> (default 30; 0 disables).
# S7: dev-box transform-size CAP. Only the *_polymul_sweep binary takes
#     max_log_n as its first positional arg, and its `bench` mode internally
#     sweeps to log_n=18. Reject log_n > GPU_RUN_MAX_LOGN (default 16) and
#     reject `bench` here. compute_e (-d N) / lib1 (n q omega iters) don't take
#     log_n positionally so they are untouched (S8 caps compute_e at log_n 22).
GPU_RUN_COOLDOWN="${GPU_RUN_COOLDOWN:-30}"
GPU_RUN_MAX_LOGN="${GPU_RUN_MAX_LOGN:-16}"
# S7b: microbench_window / test_polymul_sweep hard-cap at log_n<=20 on gfx1030.
# log_n=21 crashed 4x (CRASH 25-28); root cause is pp_dpm_sclk reads from
# crash_collect.sh triggering TransferTableSmu2Dram → 0xFFFFFFFF while GPU is
# saturated. Even without that, CRASH 26 (no collector) died at 64s — driver's
# own periodic SMU export may also fire. Hard-block here; override only on MI300A.
case "$BIN" in
  *microbench_window*|*test_polymul_sweep*)
    for a in "$@"; do
      case "$a" in
        ''|*[!0-9]*) : ;;
        *) if [ "$a" -ge 21 ] 2>/dev/null; then
             echo "[gpu_run] ABORT (S7b): log_n=$a >= 21 on gfx1030."
             echo "          log_n>=21 (N>=2M) hard-crashes this box — all 4 attempts"
             echo "          failed (CRASH 25-28). Run log_n>=21 on gfx942/MI300A."
             echo "          Override: GPU_RUN_ALLOW_LN21=1 (research only, expect crash)."
             [ "${GPU_RUN_ALLOW_LN21:-0}" = "1" ] || exit 99
           fi ;;
      esac
    done ;;
esac
case "$BIN" in
  *polymul_sweep*)
    for a in "$@"; do
      if [ "$a" = "bench" ] || [ "$a" = "b" ]; then
        echo "[gpu_run] ABORT (S7): polymul_sweep 'bench' mode sweeps to"
        echo "          log_n=18 — not allowed on the 6900XT dev box."
        echo "          Run the log_n>=18 throughput curve on gfx942/MI300A."
        exit 99
      fi
      case "$a" in
        ''|*[!0-9]*) : ;;                       # not a bare integer: skip
        *) # fail-safe: an all-digit arg that is NOT provably <= cap is
           # rejected. `! [ a -le cap ]` -> true both when a>cap (rc 1) and
           # when a overflows `[` integer parsing (rc 2, e.g. a 20-digit
           # value) — the old `-gt` form silently let oversized args through.
           if ! [ "$a" -le "$GPU_RUN_MAX_LOGN" ] 2>/dev/null; then
             echo "[gpu_run] ABORT (S7): max_log_n arg '$a' not <= cap"
             echo "          $GPU_RUN_MAX_LOGN. Raise GPU_RUN_MAX_LOGN only on"
             echo "          gfx942; the 6900XT hard-locks under large sweeps."
             exit 99
           fi ;;
      esac
    done ;;
esac

# ── Paths (resolved from this script's own location, CWD-independent) ─────────
SDIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(dirname "$SDIR")"
CDIR="$ROOT/perf/crash_diag"
LEDGER="$CDIR/gpu_run_ledger.log"
LOCK="$CDIR/.gpu_run.lock"
mkdir -p "$CDIR" 2>/dev/null || true
now() { date '+%Y-%m-%d %H:%M:%S'; }

amdgpu_card() {  # find the DRM card node for the amdgpu device (card0 or card1, etc.)
  for c in /sys/class/drm/card[0-9]*/device/gpu_busy_percent; do
    [ -r "$c" ] && { echo "${c%/device/gpu_busy_percent}"; return; }
  done
}

smi_busy() {  # echoes "1" if GPU looks busy, "0" if idle, "X" if can't determine
  # Use sysfs ONLY — no rocm-smi, no rocminfo, no device open.
  # rocm-smi/rocminfo open /dev/kfd and the render node; on a post-crash
  # fragile amdgpu driver, even a "successful" query can leave HSA async
  # commands on the ring that complete later and trigger a kernel panic.
  # Confirmed fatal on gfx1030 2026-06-06 (BIOS-reset-level crash).
  local CARD; CARD=$(amdgpu_card)
  local BUSY_PCT="${CARD}/device/gpu_busy_percent"
  if [ -r "$BUSY_PCT" ]; then
    local pct; pct=$(cat "$BUSY_PCT" 2>/dev/null)
    case "$pct" in
      ''|*[!0-9]*) echo X; return ;;
    esac
    [ "$pct" -ge 10 ] && { echo 1; return; }
    echo 0; return
  fi
  # Fallback: no sysfs → unknown.
  echo X
}

stray_hip() {  # echoes a stray GPU-binary process if one is already running.
  # Match on comm (the executable name) NOT args, else the wrapper's own
  # command line ("gpu_run.sh ... test_polymul_sweep") false-positives.
  # Exclude our own pid/ppid; comm is 15-char truncated so use a prefix set.
  ps -eo pid=,ppid=,comm= 2>/dev/null | awk -v me="$$" -v pp="${PPID:-0}" '
    {
      pid=$1; ppid=$2; c=$3
      if (pid==me || pid==pp || ppid==me) next
      if (c ~ /^(compute_e|test_polymul|test_kernel|test_ntt|ntt_gpu|ntt_cross|ntt_mul|isa_check|framework_pro)/) {
        print pid, c; exit
      }
    }'
}

# ── S9: GPU mutex — only one GPU job machine-wide (2 agents share this box) ───
# NOTE: `exec 9>FILE 2>/dev/null` (no command) makes BOTH redirections
# PERMANENT for the whole script — the stray 2>/dev/null silently swallowed
# every child's stderr (incl. NTT_PROFILE output and error messages) until
# 2026-05-17. Scope the suppression to the fd-9 open ONLY via a brace group
# so child stderr stays visible.
{ exec 9>"$LOCK"; } 2>/dev/null || true
if command -v flock >/dev/null 2>&1; then
  if ! flock -w "${GPU_RUN_LOCK_WAIT:-180}" 9; then
    echo "[gpu_run] ABORT (S9): another GPU job holds the lock after"
    echo "          ${GPU_RUN_LOCK_WAIT:-180}s. Concurrent GPU use is a"
    echo "          known hang trigger — not launching."
    exit 99
  fi
else
  echo "[gpu_run] WARNING: flock absent; S9 mutex inactive."
fi

# ── S15: crash-aware guard (AFTER the mutex, so a still-running concurrent ────
# job is not misread as a crash: once we hold the lock, any properly-wrapped
# prior run has written its EXIT. A dangling LAUNCH then means the wrapper
# itself died without writing EXIT — i.e. the box hard-hung. The 2026-05-17
# failure left NO usable kernel log; this ledger is the ground truth.
if [ -s "$LEDGER" ]; then
  # Strip NUL bytes before grep: a crash mid-write can leave NUL-padded sparse
  # blocks in the ledger. grep on a file with NULs prints "binary file matches"
  # to stderr and skips content, silently defeating the dangling-LAUNCH check.
  last="$(tr -d '\0' < "$LEDGER" | grep -E '^[0-9].* (LAUNCH|EXIT) ' | tail -1)"
  # Field-anchored parse (NOT a substring match): ledger line is
  #   <date> <time> LAUNCH|EXIT[ rc=N] <bin> <args...>
  # so field 3 is the kind and (for LAUNCH) field 4 the binary. The old
  # `case *" LAUNCH "*` misclassified any EXIT line whose args contained
  # the literal word LAUNCH. `read` (no `set --`) leaves the script's "$@"
  # — needed for the real launch below — untouched.
  read -r _ld _lt lkind lbin _lrest <<<"$last"
  if [ "$lkind" = "LAUNCH" ]; then
      echo "[gpu_run] S15 WARNING: previous GPU job has no EXIT in the ledger:"
      echo "          $last"
      echo "          => the box likely hard-hung on that run."
      if [ "$lbin" = "$BIN" ] && [ "${GPU_RUN_FORCE:-0}" != "1" ]; then
        echo "[gpu_run] ABORT (S15): refusing to re-run the same binary that"
        echo "          did not return ($BIN). Investigate first, or set"
        echo "          GPU_RUN_FORCE=1 to override deliberately."
        exit 99
      fi
      echo "[gpu_run] S15: proceeding (different binary or FORCE set)."
  fi
fi

# ── S23: SMU policy gate (rewritten 2026-06-10, CRASH 22) ─────────────────────
# The old S23/S23b read freq1_input (SMU metrics-table export) and pp_features
# (SMU GetEnabledFeatures query) — i.e. the pre-checks themselves fired SMU
# messages. CRASH 22 hard-froze DURING this pre-check burst, before the binary
# ever launched: on this card's failing SMU, *reading* SMU state is a crash
# vector. The rewritten gate sends ZERO SMU traffic:
#   dpm=0  → SMU never initialized; nothing to check; all SMU guards moot.
#   dpm on → verify GFXOFF is suppressed via the KERNEL CMDLINE ppfeaturemask
#            (bit 15 = PP_GFXOFF_MASK), not by querying the SMU. The cmdline is
#            the ground truth the driver itself uses (adev->pm.pp_feature is
#            write-once from this param at init).
_dpm=$(cat /sys/module/amdgpu/parameters/dpm 2>/dev/null || echo -1)
if [ "$_dpm" = "0" ]; then
  echo "[gpu_run] S23: amdgpu.dpm=0 — SMU/powerplay disabled; no SMU traffic possible. Skipping SMU guards."
else
  _ppmask=$(tr ' ' '\n' < /proc/cmdline | grep -o 'amdgpu.ppfeaturemask=0x[0-9a-fA-F]*' | cut -d= -f2)
  if [ -n "$_ppmask" ] && [ $(( _ppmask & 0x8000 )) -eq 0 ] 2>/dev/null; then
    echo "[gpu_run] S23: ppfeaturemask=$_ppmask (bit15 clear) — GFXOFF disabled at init; no SMU read needed."
  else
    echo "[gpu_run] ABORT (S23): GFXOFF is not disabled by kernel cmdline (ppfeaturemask=${_ppmask:-unset})."
    echo "          GfxOff SMU transitions hard-hang this card (CRASH 13-21)."
    echo "          Fix: sudo bash scripts/gpu_fix_gfxoff.sh   (then COLD power cycle)"
    echo "          Or eliminate ALL SMU traffic: sudo bash scripts/gpu_dpm_off.sh"
    exit 99
  fi

  # ── S27: static-clock gate (CRASH 22, 2026-06-10) ──────────────────────────
  # Dynamic re-clocking (idle<->boost ramps at compute launch) is the transition
  # window where the failing SMU wedges. Require the static base-clock pin
  # (perf level 'manual', set by gpu_preflight at boot or gpu_static_clock.sh)
  # before launching. Override: GPU_RUN_ALLOW_DYNAMIC=1.
  _perf_now=$(cat "$(amdgpu_card)/device/power_dpm_force_performance_level" 2>/dev/null)
  if [ "$_perf_now" != "manual" ] && [ "${GPU_RUN_ALLOW_DYNAMIC:-0}" != "1" ]; then
    echo "[gpu_run] ABORT (S27): perf level is '${_perf_now:-unknown}', not 'manual' (static pin)."
    echo "          Dynamic clock ramps at launch are the documented wedge window."
    echo "          Apply pin: sudo bash scripts/gpu_static_clock.sh 1300"
    echo "          Override (accept ramp risk): GPU_RUN_ALLOW_DYNAMIC=1"
    exit 99
  elif [ "$_perf_now" = "manual" ]; then
    echo "[gpu_run] S27: static clock pin active ($(grep '\*' "$(amdgpu_card)/device/pp_dpm_sclk" 2>/dev/null | tr -d '\n'))."
  fi
fi

# ── S24: PCIe AER baseline snapshot and error check ──────────────────────────
# PCIe Advanced Error Reporting counters reveal hardware-level transmission
# errors.  Non-zero counts before launch indicate a pre-existing PCIe fault
# that may worsen under compute load.  We record the baseline and check for
# increases in the post-run telemetry CSV.
_pci="/sys/bus/pci/devices/0000:0a:00.0"
_aer_cor_pre=$(grep "TOTAL_ERR_COR"     "${_pci}/aer_dev_correctable" 2>/dev/null | awk '{print $NF}' || echo "X")
_aer_non_pre=$(grep "TOTAL_ERR_NONFATAL" "${_pci}/aer_dev_nonfatal"    2>/dev/null | awk '{print $NF}' || echo "X")
_aer_fat_pre=$(grep "TOTAL_ERR_FATAL"    "${_pci}/aer_dev_fatal"        2>/dev/null | awk '{print $NF}' || echo "X")
echo "[gpu_run] S24: PCIe AER pre-launch — correctable=${_aer_cor_pre} nonfatal=${_aer_non_pre} fatal=${_aer_fat_pre}"
if [ "${_aer_non_pre:-0}" != "0" ] && [ "${_aer_non_pre}" != "X" ] 2>/dev/null; then
  echo "[gpu_run] WARNING (S24): PCIe non-fatal AER errors already present before launch."
  echo "          This may indicate a hardware or power issue. Proceeding but risk elevated."
fi
if [ "${_aer_fat_pre:-0}" != "0" ] && [ "${_aer_fat_pre}" != "X" ] 2>/dev/null; then
  echo "[gpu_run] ABORT (S24): PCIe FATAL AER errors present before launch — unsafe."
  exit 99
fi

# ── S25: preflight ready marker check ────────────────────────────────────────
# amdgpu-preflight.service writes /var/run/gpu_ready after the 120s SMU
# stabilisation wait and baseline snapshot.  If the marker is absent the GPU
# has not been through the stabilisation sequence this boot.
if [ ! -f /var/run/gpu_ready ] && [ "${GPU_RUN_SKIP_PREFLIGHT:-0}" != "1" ]; then
  _up=$(awk '{print int($1)}' /proc/uptime 2>/dev/null || echo 0)
  if [ "$_up" -lt 300 ] 2>/dev/null; then
    echo "[gpu_run] WARNING (S25): /var/run/gpu_ready absent and uptime=${_up}s < 300s."
    echo "          amdgpu-preflight.service may not have completed yet."
    echo "          Consider waiting before launching GPU compute."
  fi
fi

# ── S26: kill lingering pollers ───────────────────────────────────────────────
# A prior run's postwatch/telemetry must not overlap this run's compute.
for _stale in $(pgrep -f "gpu_run_postwatch|run_.*_postwatch|gpu_telemetry.sh" 2>/dev/null); do
  kill "$_stale" 2>/dev/null && echo "[gpu_run] S26: killed lingering poller pid=$_stale"
done
_pwpf="${LEDGER%/*}/postwatch.pid"
[ -f "$_pwpf" ] && { kill -- -"$(cat "$_pwpf" 2>/dev/null)" 2>/dev/null; rm -f "$_pwpf"; }

echo "[gpu_run] pre-check ..."
# Full health check — includes crash-boot detection (check 0: previous boot must
# have ended with a clean systemd shutdown). smi_busy() below only reads sysfs
# and cannot detect driver degradation from a soft reboot after a hard crash.
if ! bash "$(dirname "$0")/gpu_health_check.sh" 2>&1; then
  if [ "${GPU_RUN_FORCE:-0}" = "1" ]; then
    echo "[gpu_run] WARNING: gpu_health_check failed but GPU_RUN_FORCE=1 — proceeding."
  else
    echo "[gpu_run] ABORT: gpu_health_check.sh failed — unsafe to launch GPU binary."
    exit 99
  fi
fi
# NOTE: the old gpu_busy_percent "is GPU already busy?" pre-read was REMOVED
# (CRASH 23) — gpu_busy_percent is SMU-metrics-backed here, so even this single
# pre-check read is an avoidable SMU poke. "Is another job running?" is answered
# by stray_hip (below, /proc scan, non-SMU) and the KFD-proc / VRAM checks. The
# health-check 0-SMU probe already did the one diagnostic SMU read for this run.

# ── S28: static-pin re-verify ────────────────────────────────────────────────
# A profile change can reset the static clock pin between S27 and launch.
_perf_chk=$(cat "$(amdgpu_card)/device/power_dpm_force_performance_level" 2>/dev/null)
if [ "$_perf_chk" != "manual" ] && [ "${GPU_RUN_ALLOW_DYNAMIC:-0}" != "1" ]; then
  echo "[gpu_run] ABORT (S28): static pin lost (perf='${_perf_chk}', expected 'manual')."
  echo "          Re-apply: sudo bash scripts/gpu_static_clock.sh 1300"
  exit 99
fi

# ── S29: package power-state gate (idle D3hot wedge, 2026-06-11) ───────────────
# Every idle crash starts with the audio function failing to wake from D3hot.
# Both package functions must be pinned to D0 (power/control=on) so no suspend/
# wake transition can occur. A 'suspended' audio function here is the documented
# wedge precursor — refuse to launch until it is pinned.
_aud=/sys/bus/pci/devices/0000:0a:00.1
_aud_ctrl=$(cat "$_aud/power/control" 2>/dev/null)
if [ "$_aud_ctrl" != "on" ] && [ "${GPU_RUN_ALLOW_PKG_PM:-0}" != "1" ]; then
  echo "[gpu_run] ABORT (S29): audio fn 0a:00.1 runtime PM not pinned (control='${_aud_ctrl}')."
  echo "          Its D3hot->D0 wake failure is the recurring idle-wedge trigger."
  echo "          Fix: sudo bash scripts/gpu_no_pkg_powersave.sh"
  exit 99
fi

sp="$(stray_hip)"
if [ -n "$sp" ]; then
  echo "[gpu_run] ABORT (S10): a GPU binary is already running:"
  echo "          $sp"
  echo "          Refusing to add concurrent GPU load."
  exit 99
fi

# ── S20: multi-tenant GPU contention BLOCK ───────────────────────────────────
# Any Electron app or compositor that holds renderD128 can submit to the GFX
# ring and freeze the display when compute hangs it.
#
# Root cause history: CRASH 20 was diagnosed as the CWSR/MES bug in ROCm 7.x
# on gfx1030 (cwsr_enable=1 default, `MES failed to respond` in kernel log),
# NOT renderD128 sharing per se.  The fix — cwsr_enable=1 with GFXOFF disabled
# + mid-dispatch yield in ntt_mul.hip — allows compute to be preempted so
# display page-flips can proceed.  Xwayland and mutter are mandatory desktop
# processes that cannot be closed without ending the session; blocking them
# would make all desktop GPU runs impossible.
#
# CRASH 22 (2026-06-13) was caused by bypassing gpu_run.sh ENTIRELY and running
# d=2.1M (>360s) without the S18 time cap — not by the desktop override.
# GPU_RUN_ALLOW_DESKTOP=1 is safe when S18 enforces the 110s cap.
#
# Override tiers (Chrome/Steam are never allowed — close them first):
#   GPU_RUN_ALLOW_DESKTOP=1  all desktop processes present (Xwayland, mutter,
#                             VS Code) — safe with S18 cap + cwsr_enable=1
#   GPU_RUN_ALLOW_VSCODE=1   only when ALL holders are VS Code processes
_rd_pids=$(lsof -t /dev/dri/renderD128 2>/dev/null | sort -u || true)
if [ -n "$_rd_pids" ]; then
  _rd_list=""
  _all_code=1
  _has_chrome=0
  for _p in $_rd_pids; do
    _c=$(ps -o comm= -p "$_p" 2>/dev/null || echo "?")
    _rd_list="${_rd_list}    pid=${_p} (${_c})\n"
    [ "$_c" = "code" ] || _all_code=0
    case "$_c" in chrome|chromium|steam|steam_*) _has_chrome=1;; esac
  done
  if [ "$_has_chrome" -eq 1 ]; then
    echo "[gpu_run] ABORT (S20): Chrome or Steam holds renderD128 — close it first."
    printf '%b' "$_rd_list"
    exit 99
  elif [ "${GPU_RUN_ALLOW_DESKTOP:-0}" = "1" ]; then
    echo "[gpu_run] S20 WARNING: desktop processes hold renderD128:"
    printf '%b' "$_rd_list"
    echo "  Proceeding because GPU_RUN_ALLOW_DESKTOP=1 (safe with S18 cap + cwsr_enable=1)."
  elif [ "$_all_code" -eq 1 ] && [ "${GPU_RUN_ALLOW_VSCODE:-0}" = "1" ]; then
    echo "[gpu_run] S20 WARNING: VS Code holds renderD128."
    echo "  Proceeding ONLY because GPU_RUN_ALLOW_VSCODE=1 was set explicitly."
  else
    echo "[gpu_run] ABORT (S20): processes hold renderD128. Holders:"
    printf '%b' "$_rd_list"
    echo "  Desktop session: set GPU_RUN_ALLOW_DESKTOP=1 (safe with S18 cap + cwsr_enable=1)."
    echo "  VS Code only: set GPU_RUN_ALLOW_VSCODE=1."
    exit 99
  fi
fi

# ── S30: external SMU-reader preflight (2026-06-15, CRASH 26 analysis) ────────
# The confirmed crash mechanism is: ANY read of an SMU-metrics-backed sysfs node
# (hwmon temp/freq/power, gpu_busy_percent, pp_dpm_sclk/mclk, mem_busy_percent)
# while the GFX ring is busy issues TransferTableSmu2Dram on this fw/driver-
# mismatched SMU (fw 0x41 / driver 0x40) → 0xFFFFFFFF → hard hang. We removed
# every in-repo during-load reader, but a THIRD-PARTY reader waking mid-run is
# the prime suspect for CRASH 26 (d=3M froze at 64s with no tooling reader).
# Block known interactive GPU monitors outright (just close them); warn about
# daemons whose timer may fire during a long run.
_smu_readers=$(pgrep -fa 'gnome-system-monitor|mission-cent|psensor|radeontop|nvtop|corectrl|^lact|zenmonitor|gkrellm|mangohud|nvtop|amdgpu_top|btop' 2>/dev/null \
  | grep -ivE 'gpu_run|pgrep' || true)
if [ -n "$_smu_readers" ]; then
  echo "[gpu_run] ABORT (S30): a GPU-sensor app is running and will poll SMU during the run:"
  echo "$_smu_readers" | sed 's/^/    /'
  echo "  These read hwmon/gpu_busy_percent at 1+ Hz → TransferTableSmu2Dram → hard hang."
  echo "  Close them, then re-run. (Override: GPU_RUN_ALLOW_SMU_READERS=1 — NOT recommended.)"
  [ "${GPU_RUN_ALLOW_SMU_READERS:-0}" = "1" ] || exit 99
fi
# Daemon warning: fwupd's refresh timer can wake mid-run and re-enumerate the
# GPU. For long runs, quiesce it:  sudo systemctl stop fwupd.service
if systemctl is-active --quiet fwupd-refresh.timer 2>/dev/null; then
  echo "[gpu_run] S30 NOTE: fwupd-refresh.timer is active; for runs > a few minutes"
  echo "          consider: sudo systemctl stop fwupd.service  (prevents a mid-run GPU probe)."
fi

# ── S19: Block rocprofv3/rocprof on display-sharing GPU (2026-06-06 incident) ─
# rocprofv3 hardware PMC instrumentation hard-crashed this box twice even when
# all profiled kernels exited cleanly (rc=0).  The amdgpu driver's state is
# corrupted by PMC session teardown; subsequent GPU opens (hipcc, X11) then
# trigger a hang.  On a display-sharing GPU this kills the desktop.
# Allowed only under gpu_headless_run.sh (which sets GPU_RUN_ALLOW_LONG=1).
case "$BIN" in
  rocprofv3*|rocprof*)
    if [ "${GPU_RUN_ALLOW_LONG:-0}" != "1" ]; then
      echo "[gpu_run] S19: REFUSED rocprofv3/rocprof on display-sharing GPU." >&2
      echo "  PMC hard-crashed this machine twice (05-23, 06-06).  Use:" >&2
      echo "    sudo scripts/gpu_headless_run.sh <timeout> $BIN $*" >&2
      exit 3
    fi
    ;;
esac

# ── S18: TDR-aware timeout clamp (display-shared GPU, 2026-05-29 incident) ────
# This box's gfx1030 drives the desktop AND compute on a single GPU. The
# amdgpu hardware watchdog (lockup_timeout) resets the GPU when any job holds
# it past the timeout. S18 reads the live kernel parameter to decide whether
# clamping is still necessary.
#
# 2026-06-08 fix: clamp is derived from the live GFX TDR (lockup_timeout[0]) with
# a 10s safety margin.  The old hardcoded 8s was written for lockup_timeout=10000ms
# and was never updated when TDR was raised to 120000ms for the I9 fix.  Compute
# jobs run on the COMPUTE ring (TDR=600s); we use GFX TDR as the reference because
# on gfx1030 (unified) the GFX watchdog can still affect the compositor.  With
# lockup_timeout=120000ms the clamp becomes 110s, allowing jobs like d=600000
# (~14s) to complete without a premature SIGINT (CRASH 14, 2026-06-08).
_live_tdr=$(cat /sys/module/amdgpu/parameters/lockup_timeout 2>/dev/null | cut -d, -f1 || echo 10000)
_gfx_tmo_s=$(( ${_live_tdr:-10000} / 1000 ))
_tmo_safe=$(( _gfx_tmo_s > 20 ? _gfx_tmo_s - 10 : 8 ))
GPU_RUN_MAX_TMO="${GPU_RUN_MAX_TMO:-${_tmo_safe}}"
if [ "${GPU_RUN_ALLOW_LONG:-0}" != "1" ] \
   && [ "${_live_tdr:-0}" != "0" ] \
   && [ "$TMO" -gt "$GPU_RUN_MAX_TMO" ] 2>/dev/null; then
  echo "[gpu_run] S18: clamping timeout ${TMO}s -> ${GPU_RUN_MAX_TMO}s"
  echo "          (display-shared gfx1030; GFX TDR=${_live_tdr}ms → safe limit=${GPU_RUN_MAX_TMO}s)."
  TMO="$GPU_RUN_MAX_TMO"
elif [ "${_live_tdr:-0}" = "0" ] && [ "${GPU_RUN_ALLOW_LONG:-0}" != "1" ]; then
  echo "[gpu_run] S18: lockup_timeout param unset/0 — kernel defaults active (gfx 10s, compute 60s); no clamp applied."
else
  echo "[gpu_run] S18: timeout ${TMO}s within safe limit ${GPU_RUN_MAX_TMO}s (GFX TDR=${_live_tdr}ms)."
fi

# ── VRAM baseline for post-check ─────────────────────────────────────────────
# Capture VRAM used before launch. After exit, gpu_busy_percent reflects the
# GFX ring only; the ROCm compute ring may still be draining HBM allocations.
# A VRAM check catches lingering device memory that gpu_busy_percent misses.
_vram_node="$(amdgpu_card)/device/mem_info_vram_used"
_vram_baseline=$(cat "$_vram_node" 2>/dev/null || echo "X")

# ── Pre-launch logging setup ──────────────────────────────────────────────────
_ts=$(date '+%Y%m%d_%H%M%S')
_logbase="${LEDGER%/*}/run_${_ts}_$(basename "$BIN")"
_telem_log="${_logbase}_telemetry.csv"
_klog="${_logbase}_kernel.log"
_crash_marker="${LEDGER%/*}/CRASH_MARKER"

# Telemetry: 1Hz sysfs poller — GPU state to CSV (safe, no device open).
_telem_pid=""
if [ "${GPU_RUN_TELEMETRY:-1}" = "1" ]; then
    bash "$(dirname "$0")/gpu_telemetry.sh" "$_telem_log" &
    _telem_pid=$!
    echo "[gpu_run] telemetry: pid=$_telem_pid → $_telem_log"
fi

# Kernel log capture: follow kernel ring buffer to a file in real time.
# Captures amdgpu/SMU/KFD messages as they happen. NOTE: lines land in page
# cache, not on disk — a hard freeze can still lose the tail; the post-run
# watch below fsyncs its samples for that reason.
_klog_pid=""
if [ "${GPU_RUN_KLOG:-1}" = "1" ]; then
    journalctl -kf --no-tail --output=short-precise 2>/dev/null >> "$_klog" &
    _klog_pid=$!
    echo "[gpu_run] kernel log: pid=$_klog_pid → $_klog"
fi

# AMD HIP runtime error logging: ROCm will print warnings/errors to stderr.
export AMD_LOG_LEVEL="${AMD_LOG_LEVEL:-2}"   # 0=off 1=error 2=warn 3=info 4=debug

# ── CU reservation for the display compositor ────────────────────────────────
# On a display-sharing GPU (gfx1030), saturating ALL CUs with non-preemptible
# compute (cwsr_enable=0) can starve the compositor's render so a page flip never
# completes → "flip_done timed out" → hard hang (captured 2026-06-12). Reserve a
# few CUs for the display via ROC_GLOBAL_CU_MASK. Default GPU_RUN_RESERVE_CU=4;
# gpu_cu_mask.sh queries the CU count and reserves nothing on a headless GPU
# (MI300A target), so this is a no-op off the dev box. Caller override respected.
if [ -z "${HSA_CU_MASK:-}" ]; then
  _cumask="$(bash "$(dirname "$0")/gpu_cu_mask.sh" "${GPU_RUN_RESERVE_CU:-4}" 2>/dev/null || true)"
  if [ -n "$_cumask" ]; then
    export HSA_CU_MASK="$_cumask"
    echo "[gpu_run] CU reservation: HSA_CU_MASK=$_cumask (reserved ${GPU_RUN_RESERVE_CU:-4} CUs for display)"
  fi
fi

# ── Cooperative display yield (gfx1030 only) ─────────────────────────────────
# The root fix for flip_done crashes is cwsr_enable=1 + GFXOFF-disabled
# (modprobe.d/amdgpu-cwsr.conf + amdgpu-tdr.conf, active after initramfs rebuild).
# With preemption enabled the compositor can flip mid-compute.
#
# This yield is the code-layer defense-in-depth: ntt_mul.hip inserts a
# hipStreamSynchronize+usleep (a) between forward and inverse NTT passes
# (mid-dispatch) and (b) after each complete mul (end-of-mul).  Together they
# bound the max ring-busy window to max(fwd_NTT_time, inv_NTT_time) regardless
# of cwsr state -- confirmed necessary after the 2026-06-13 crash at d=1.1M
# where a single forward NTT held the ring for >5s and timed out DRM flip_done.
#
# EVERY=1 guarantees a yield window after every single mul call (not batched).
# Only activated on display-sharing GPUs (HSA_CU_MASK non-empty = desktop GPU).
if [ -n "${HSA_CU_MASK:-}" ]; then
  export NTT_DISPLAY_YIELD_US="${NTT_DISPLAY_YIELD_US:-5000}"
  export NTT_DISPLAY_YIELD_EVERY="${NTT_DISPLAY_YIELD_EVERY:-1}"
  echo "[gpu_run] display yield: ${NTT_DISPLAY_YIELD_US}us every ${NTT_DISPLAY_YIELD_EVERY} muls, mid+end-of-dispatch (flip_done guard)"
fi

# Crash marker: written before LAUNCH, deleted after clean EXIT.
# If it survives reboot, the previous compute crashed without cleanup.
# gpu_health_check.sh checks for this file on next boot.
{
    echo "binary:  $BIN $*"
    echo "started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "telem:   $_telem_log"
    echo "klog:    $_klog"
    echo "uptime:  $(cat /proc/uptime)"
} > "$_crash_marker"

echo "[gpu_run] launch: $BIN $* (timeout ${TMO}s, gentle escalation)"
echo "$(now) LAUNCH $BIN $*" >> "$LEDGER"
# Write a KERN_ERR marker so netconsole timestamps the exact compute start.
echo "<0>gpu_run: LAUNCH $BIN $* d=$(echo "$*" | grep -o '\-d [0-9]*' | tr -d ' ')" \
  | sudo tee /dev/kmsg 2>/dev/null || true
# --signal=INT first, then default TERM, then KILL — with kill-after grace.
timeout --signal=INT --kill-after=20 "${TMO}" "$BIN" "$@"
rc=$?
echo "$(now) EXIT rc=$rc $BIN $*" >> "$LEDGER"

# Clean exit: remove crash marker (crashed runs leave it in place).
[ "$rc" -eq 0 ] && rm -f "$_crash_marker"

# Stop background loggers.
for _pid in "$_telem_pid" "$_klog_pid"; do
    [ -n "${_pid:-}" ] && { kill "$_pid" 2>/dev/null; wait "$_pid" 2>/dev/null || true; }
done
# Force run logs + ledger to disk: a hard freeze loses unsynced page cache.
sync -d "$_klog" "$_telem_log" "$LEDGER" 2>/dev/null || sync
echo "[gpu_run] logs: telemetry=$_telem_log  kernel=$_klog"

# ── Post-run delayed-crash watch ──────────────────────────────────────────────
# CRASH 12 (2026-06-07) and CRASH 20 (2026-06-09) hard-froze the machine
# MINUTES AFTER a clean rc=0 exit and passing post-checks, and journald's
# async buffer lost the final minutes — neither crash left kernel evidence.
# Keep a detached watcher following the kernel ring and sampling GPU power
# state after exit, fsyncing every sample, so a delayed freeze leaves
# on-disk evidence. A subsequent run supersedes any still-running watcher.
_postwatch_secs="${GPU_RUN_POSTWATCH:-600}"
if [ "$_postwatch_secs" -gt 0 ] 2>/dev/null; then
  _pw="${_logbase}_postwatch.log"
  _pw_pidfile="${LEDGER%/*}/postwatch.pid"
  [ -f "$_pw_pidfile" ] && kill -- -"$(cat "$_pw_pidfile" 2>/dev/null)" 2>/dev/null
  setsid bash -c '
    # gpu_run_postwatch: delayed-crash evidence sampler. Reads ONLY cheap sysfs
    # (gpu_busy_percent + PCI runtime_status). Does NOT read pp_dpm_sclk — that
    # triggers an SMU metrics query, and an out-of-run poller firing SMU queries
    # was a self-inflicted concurrent-load variable (CRASH 21 analysis).
    pw="$1"; secs="$2"; card="$3"; pci="$4"
    audio_rt="${pci%.0}.1/power/runtime_status"
    # CRASH 23 (2026-06-10): gpu_busy_percent is SMU-metrics-backed on Sienna
    # Cichlid (= TransferTableSmu2Dram). Polling it here every 5s for 600s fired
    # ~120 SMU pokes per run AFTER the run — a major self-inflicted wedge source.
    # The watch now samples ONLY non-SMU sysfs: PCI runtime_status (D-state) and
    # the kernel journal follow, which is what actually reveals a delayed wedge.
    journalctl -kf --no-tail --output=short-precise >> "$pw" 2>/dev/null &
    jp=$!
    end=$(( $(date +%s) + secs ))
    while [ "$(date +%s)" -lt "$end" ]; do
      printf "%s gpu_rt=%s audio_rt=%s\n" \
        "$(date "+%H:%M:%S")" \
        "$(cat "$pci/power/runtime_status" 2>/dev/null || echo X)" \
        "$(cat "$audio_rt" 2>/dev/null || echo X)" >> "$pw"
      sync -d "$pw" 2>/dev/null || sync
      sleep 5
    done
    kill "$jp" 2>/dev/null
  ' gpu_run_postwatch "$_pw" "$_postwatch_secs" "$(amdgpu_card)" "$_pci" </dev/null >/dev/null 2>&1 9>&- &
  echo "$!" > "$_pw_pidfile"
  echo "[gpu_run] post-run watch: ${_postwatch_secs}s → $_pw (delayed-crash evidence)"
fi

# PCIe AER post-launch delta: non-zero increase = PCIe error during compute.
_aer_cor_post=$(grep "TOTAL_ERR_COR"      "${_pci}/aer_dev_correctable" 2>/dev/null | awk '{print $NF}' || echo "X")
_aer_non_post=$(grep "TOTAL_ERR_NONFATAL" "${_pci}/aer_dev_nonfatal"    2>/dev/null | awk '{print $NF}' || echo "X")
_aer_fat_post=$(grep "TOTAL_ERR_FATAL"    "${_pci}/aer_dev_fatal"        2>/dev/null | awk '{print $NF}' || echo "X")
if [ "${_aer_cor_post}" != "X" ] && [ "${_aer_cor_pre}" != "X" ] \
   && [ "$_aer_cor_post" -gt "$_aer_cor_pre" ] 2>/dev/null; then
  echo "[gpu_run] WARNING: PCIe correctable AER errors increased during run: ${_aer_cor_pre}→${_aer_cor_post}"
fi
if [ "${_aer_non_post}" != "X" ] && [ "${_aer_non_pre}" != "X" ] \
   && [ "$_aer_non_post" -gt "$_aer_non_pre" ] 2>/dev/null; then
  echo "[gpu_run] WARNING: PCIe NON-FATAL AER errors increased during run: ${_aer_non_pre}→${_aer_non_post}"
fi
if [ "${_aer_fat_post}" != "X" ] && [ "${_aer_fat_pre}" != "X" ] \
   && [ "$_aer_fat_post" -gt "$_aer_fat_pre" ] 2>/dev/null; then
  echo "[gpu_run] CRITICAL: PCIe FATAL AER errors during run: ${_aer_fat_pre}→${_aer_fat_post}"
  echo "          This is a hardware-level error. Do not run another GPU job."
fi
[ $rc -eq 124 ] && echo "[gpu_run] NOTE: binary timed out and was signalled (rc=124)."

# Post-check: give the driver a moment, then poll sysfs for idle.
# SKIP if binary was killed by a signal (rc >= 130): the binary's gpu_cleanup_handler
# already ran hipDeviceReset; sysfs is safe to read but the GPU% may momentarily
# spike during device reset — skip entirely to avoid false UNHEALTHY.
if [ "$rc" -ge 130 ] 2>/dev/null; then
  echo "[gpu_run] NOTE: binary killed by signal (rc=$rc); skipping post-check."
  echo "          gpu_cleanup_handler ran hipDeviceReset before exit."
  ok=1
else
  # Post-run idle detection uses ONLY non-SMU signals: VRAM return-to-baseline
  # (below) + KFD-proc drain (further below). The old gpu_busy_percent poll loop
  # was removed (CRASH 23): on Sienna Cichlid gpu_busy_percent is SMU-metrics-
  # backed, so polling it 10x after every run was itself an SMU-wedge source.
  # VRAM/KFD are better idle signals (they track the compute ring, not GFX) and
  # touch no SMU. A brief settle before the VRAM check:
  sleep 2
  # gpu_busy_percent reflects the GFX ring; the ROCm HSA runtime's async queue
  # teardown (hsa_shut_down) continues on the compute ring after GFX shows idle.
  # A second run starting before hsa_shut_down completes can conflict with the
  # previous run's cleanup and crash the machine (confirmed 2026-06-07).
  # Mandatory extra wait after idle is declared.
  sleep 5

  # VRAM return-to-baseline check: poll until mem_info_vram_used drops within
  # 200 MB of the pre-launch baseline.  Catches HBM allocations still held by
  # the ROCm compute ring after GFX reports idle (hipDeviceReset() is
  # synchronous, but hsa_shut_down runs async on the compute ring and may hold
  # device memory slightly longer than the GFX ring shows idle).
  if [ "$_vram_baseline" != "X" ] && [ -r "$_vram_node" ]; then
    _thresh=$(( _vram_baseline + 209715200 ))  # baseline + 200 MB
    for _vi in 1 2 3 4 5 6; do
      _vram_now=$(cat "$_vram_node" 2>/dev/null || echo "$_thresh")
      if [ "$_vram_now" -le "$_thresh" ] 2>/dev/null; then break; fi
      echo "[gpu_run] post-check: VRAM still elevated (${_vram_now} vs baseline ${_vram_baseline}); waiting 5s..."
      sleep 5
    done
    _vram_now=$(cat "$_vram_node" 2>/dev/null || echo "X")
    if [ "$_vram_now" != "X" ] && [ "$_vram_now" -gt "$_thresh" ] 2>/dev/null; then
      echo "[gpu_run] WARNING: VRAM did not return to baseline after 35s (now=${_vram_now}, baseline=${_vram_baseline})."
      echo "          Device memory may still be held. Proceeding but next run risk elevated."
    fi
  fi
fi
echo "[gpu_run] post-check OK (GPU idle). program rc=$rc"
# ── KFD proc cleanup poll ─────────────────────────────────────────────────────
# KFD destroys queues and releases PASIDs via an async workqueue after the HIP
# process exits.  If the next process opens /dev/kfd before this completes, the
# PASID allocation can conflict → hipSetDevice() crash on the second consecutive
# run (confirmed CRASH 15, 2026-06-08).  Poll until /sys/class/kfd/kfd/proc/ is
# empty — the kernel removes the entry only once full cleanup is done.
_kfd_proc="/sys/class/kfd/kfd/proc"
if [ -d "$_kfd_proc" ]; then
  _kfd_wait=0
  while [ "$(ls "$_kfd_proc" 2>/dev/null | wc -l)" -gt 0 ]; do
    _kfd_wait=$(( _kfd_wait + 1 ))
    if [ "$_kfd_wait" -ge 60 ]; then
      echo "[gpu_run] WARNING: KFD proc not empty after 60s — proceeding anyway"
      break
    fi
    sleep 1
  done
  [ "$_kfd_wait" -gt 0 ] && echo "[gpu_run] KFD proc drain: ${_kfd_wait}s"
fi
# ── S11: adaptive cooldown — if many runs clustered recently, lengthen it ─────
# (sustained back-to-back load is the documented hard-lock pattern). Counts
# LAUNCH events in the last ~3 min from the ledger; >=6 => cooldown x3.
cd="$GPU_RUN_COOLDOWN"
if [ -s "$LEDGER" ] && command -v date >/dev/null 2>&1; then
  cutoff="$(date -d '180 seconds ago' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
  if [ -n "$cutoff" ]; then
    rec="$(awk -v c="$cutoff" '/ LAUNCH /{ if (substr($0,1,19) >= c) n++ } END{print n+0}' "$LEDGER")"
    if [ "${rec:-0}" -ge 6 ] 2>/dev/null; then
      cd=$(( GPU_RUN_COOLDOWN * 3 ))
      echo "[gpu_run] S11: $rec runs in last 180s — extending cooldown to ${cd}s."
    fi
  fi
fi
if [ "$cd" -gt 0 ] 2>/dev/null; then
  echo "[gpu_run] S6 cooldown ${cd}s (no back-to-back heavy runs) ..."
  sleep "$cd"
fi
exit $rc
