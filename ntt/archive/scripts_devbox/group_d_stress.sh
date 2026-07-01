#!/bin/bash
# group_d_stress.sh — long-soak determinism + thermal/power telemetry.
#
# D1  3-prime long-soak determinism @ repeats=200 (R8 baseline was 50)
# D2  60-second sustained bench + sysfs telemetry (temp/power/freq)
#
# All GPU launches via scripts/gpu_run.sh. Telemetry polling reads sysfs
# ONLY — no rocm-smi/rocminfo calls (they open /dev/kfd which can crash
# a fragile amdgpu driver even on a "clean" read; confirmed BIOS-reset
# level crash 2026-06-06). sysfs reads never open the GPU device.
set -u
cd "$(dirname "$0")/.."

# sysfs_gpu_stats — read GPU telemetry without opening the device.
# Outputs a single tab-separated line: temp_C power_W sclk_MHz mclk_MHz gpu_pct vram_pct
# Any unavailable field is printed as "-".
sysfs_gpu_stats() {
  # Find amdgpu DRM card dynamically (index varies: card0 or card1, etc.)
  local CARD
  CARD=$(for c in /sys/class/drm/card[0-9]*/device/gpu_busy_percent; do
    [ -r "$c" ] && { echo "${c%/device/gpu_busy_percent}/device"; break; }; done)
  [ -z "$CARD" ] && CARD="/sys/class/drm/card1/device"  # fallback for gfx1030
  local HWMON; HWMON=$(ls /sys/class/hwmon/ 2>/dev/null | head -1)
  HWMON="/sys/class/hwmon/${HWMON}"

  local temp="-" pwr="-" sclk="-" mclk="-" gpu_p="-" vram_p="-"

  # SMU-METRICS GUARD (default OFF). Every node below — temp*_input,
  # power1_average, pp_dpm_sclk/mclk, gpu_busy_percent, mem_busy_percent — is
  # served by the SMU metrics table (msg TransferTableSmu2Dram, index 18). On
  # this 6900XT the SMU fw/driver interface mismatch (fw 0x41 / driver 0x40)
  # makes that message return 0xFFFFFFFF and HARD-HANG the machine when it lands
  # while the GFX ring is busy. This sampler runs at 1 Hz DURING a sustained
  # bench (the worst case), so reading these nodes here actively provokes the
  # crash. Disabled by default; set GPU_TELEMETRY_SMU=1 only on a stack with a
  # matched SMU interface. See scripts/gpu_telemetry.sh, CRASH 21/25/27/28.
  if [ "${GPU_TELEMETRY_SMU:-0}" = "1" ]; then
    local tf="${HWMON}/temp1_input"
    [ -r "$tf" ] && temp=$(awk '{printf "%.0f", $1/1000}' "$tf" 2>/dev/null)
    local pf="${HWMON}/power1_average"
    [ -r "$pf" ] && pwr=$(awk '{printf "%.1f", $1/1000000}' "$pf" 2>/dev/null)
    local sf="${CARD}/pp_dpm_sclk"
    [ -r "$sf" ] && sclk=$(grep '\*' "$sf" 2>/dev/null | grep -oE '[0-9]+Mhz' | tr -d 'Mhz')
    local mf="${CARD}/pp_dpm_mclk"
    [ -r "$mf" ] && mclk=$(grep '\*' "$mf" 2>/dev/null | grep -oE '[0-9]+Mhz' | tr -d 'Mhz')
    local gf="${CARD}/gpu_busy_percent"
    [ -r "$gf" ] && gpu_p=$(cat "$gf" 2>/dev/null)
    local vf="${CARD}/mem_busy_percent"
    [ -r "$vf" ] && vram_p=$(cat "$vf" 2>/dev/null)
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$temp" "$pwr" "$sclk" "$mclk" "$gpu_p" "$vram_p"
}

# GPU health gate — corrupted amdgpu driver state can hang GPU binaries even
# before a kernel is dispatched (device open hangs on gfx1030 post-crash).
if scripts/gpu_health_check.sh 2>/dev/null; then
  :
else
  rc=$?
  [ "$rc" -eq 1 ] && { echo "[group_d_stress] GPU driver unhealthy — power cycle required." >&2; exit 1; }
fi

TS=$(date +%Y%m%d_%H%M%S)
MD=perf/results/GFX1030_STRESS_${TS}.md
LOG=results/groupD_${TS}.log
TELEMETRY=results/groupD_telemetry_${TS}.tsv
mkdir -p perf/results results
: > "$LOG"

cat > "$MD" << EOF
# gfx1030 (6900 XT) — long-soak determinism + thermal/power telemetry

Captured ${TS} on AMD Radeon RX 6900 XT (gfx1030), ROCm 7.0.3.
Driver: scripts/group_d_stress.sh. Raw: ${LOG} + ${TELEMETRY}.

EOF

# ── D1 long-soak determinism, 3 primes × repeats=200 ────────────────────────
echo "## D1 — Long-soak determinism (repeats=200, vs R8 baseline 50)" >>"$MD"
echo "" >>"$MD"
echo "| config | R1 fwd-repeat | R2 round-trip | R3 CT-DIT vs Stockham |" >>"$MD"
echo "|---|---|---|---|" >>"$MD"

declare -a cells=(
    "256 3329 17"
    "256 8380417 3073009"
    "4096 12289 41"
)
labels=("ML-KEM (n=256)" "ML-DSA cyclic (n=256)" "n=4096 q=12289")

D1_FAIL=0
for i in "${!cells[@]}"; do
    cfg="${cells[$i]}"; lab="${labels[$i]}"
    echo "===== D1 $lab cfg=$cfg repeats=200 =====" >>"$LOG"
    out=$(scripts/gpu_run.sh 180 bin/ntt_gpu_determinism_6900xt $cfg 200 2>&1)
    rc=$?
    if [ "$rc" -eq 99 ]; then sleep 30; out=$(scripts/gpu_run.sh 180 bin/ntt_gpu_determinism_6900xt $cfg 200 2>&1); rc=$?; fi
    echo "$out" >>"$LOG"
    plain=$(echo "$out" | sed 's/\x1b\[[0-9;]*m//g')
    r1=$(echo "$plain" | grep -E 'R1 fwd' | grep -oE 'PASS|FAIL' | head -1)
    r2=$(echo "$plain" | grep -E 'R2 round-trip' | grep -oE 'PASS|FAIL' | head -1)
    r3=$(echo "$plain" | grep -E 'R3 CT-DIT vs' | grep -oE 'PASS|FAIL' | head -1)
    [ -z "$r1" ] && r1="-"; [ -z "$r2" ] && r2="-"; [ -z "$r3" ] && r3="-"
    [ "$r1" = "FAIL" -o "$r2" = "FAIL" -o "$r3" = "FAIL" -o "$rc" -ne 0 ] && D1_FAIL=1
    echo "| $lab cfg=\`$cfg\` | $r1 | $r2 | $r3 |" >>"$MD"
done
echo "" >>"$MD"

# ── D2 thermal/power/freq telemetry during 60s sustained bench ─────────────
echo "===== D2 telemetry capture starting =====" >>"$LOG"

# Pre-flight: GPU idle?
pre=$(sysfs_gpu_stats)
echo "[D2] pre-bench sysfs: $pre" >>"$LOG"

# Start telemetry polling in the background. 1 Hz, 70 samples max
# (covers a 60s bench with margin). Format: TSV with timestamp + key cols.
printf "ts\ttemp_C\tpower_W\tsclk_MHz\tmclk_MHz\tgpu_pct\tvram_pct\n" > "$TELEMETRY"
(
    end=$((SECONDS + 75))
    while [ $SECONDS -lt $end ]; do
        line=$(sysfs_gpu_stats)
        # sysfs_gpu_stats output: temp_C \t power_W \t sclk_MHz \t mclk_MHz \t gpu_pct \t vram_pct
        temp=$(echo "$line" | cut -f1)
        pwr=$(echo "$line"  | cut -f2)
        sclk=$(echo "$line" | cut -f3)
        mclk=$(echo "$line" | cut -f4)
        gpu_p=$(echo "$line" | cut -f5)
        vram_p=$(echo "$line" | cut -f6)
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$(date '+%H:%M:%S')" "$temp" "$pwr" "$sclk" "$mclk" "$gpu_p" "$vram_p" >> "$TELEMETRY"
        sleep 1
    done
) &
TELE_PID=$!
echo "[D2] telemetry polling started PID=$TELE_PID" >>"$LOG"

# 60s sustained bench. ML-KEM at 21k NTT/s → ~60s for 1.3M iters. Wrap
# with generous timeout (120s) — actual wall time should be ~55-65s.
echo "[D2] sustained bench launching (ntt_gpu_6900xt 256 3329 17 1260000)" >>"$LOG"
out=$(scripts/gpu_run.sh 120 bin/ntt_gpu_6900xt 256 3329 17 1260000 2>&1)
BENCH_RC=$?
echo "$out" >>"$LOG"

# Stop telemetry; let final sample land.
kill "$TELE_PID" 2>/dev/null
wait "$TELE_PID" 2>/dev/null
echo "[D2] telemetry polling stopped" >>"$LOG"

post=$(sysfs_gpu_stats)
echo "[D2] post-bench sysfs: $post" >>"$LOG"

# Summarize telemetry into the MD.
python3 << PYEOF >> "$MD"
import csv
ts, temps, pwrs, sclks, mclks, gpus, vrams = [], [], [], [], [], [], []
with open("$TELEMETRY") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        try:
            t = float(row["temp_C"]) if row["temp_C"] else None
            p = float(row["power_W"]) if row["power_W"] else None
            s = int(row["sclk_MHz"]) if row["sclk_MHz"] else None
            m = int(row["mclk_MHz"]) if row["mclk_MHz"] else None
            g = int(row["gpu_pct"])  if row["gpu_pct"]  else None
            v = int(row["vram_pct"]) if row["vram_pct"] else None
        except ValueError:
            continue
        ts.append(row["ts"]); temps.append(t); pwrs.append(p); sclks.append(s); mclks.append(m); gpus.append(g); vrams.append(v)

def stat(lst, fmt="{:.1f}"):
    xs = [x for x in lst if x is not None]
    if not xs: return ("-","-","-")
    return (fmt.format(min(xs)), fmt.format(sum(xs)/len(xs)), fmt.format(max(xs)))

print(f"## D2 — Thermal / power / frequency under sustained bench")
print()
print(f"Bench: \`bin/ntt_gpu_6900xt 256 3329 17 1260000\` (~60s sustained ML-KEM cyclic NTT).")
print(f"Samples: {len(ts)} (1 Hz polling).  Raw TSV: {''}\`$TELEMETRY\`.")
print()
print("| Metric | min | mean | max |")
print("|---|---:|---:|---:|")
mn,me,mx = stat(temps); print(f"| Temp (°C)  | {mn} | {me} | {mx} |")
mn,me,mx = stat(pwrs);  print(f"| Power (W)  | {mn} | {me} | {mx} |")
mn,me,mx = stat(sclks, fmt="{:.0f}"); print(f"| SCLK (MHz) | {mn} | {me} | {mx} |")
mn,me,mx = stat(mclks, fmt="{:.0f}"); print(f"| MCLK (MHz) | {mn} | {me} | {mx} |")
mn,me,mx = stat(gpus,  fmt="{:.0f}"); print(f"| GPU busy % | {mn} | {me} | {mx} |")
mn,me,mx = stat(vrams, fmt="{:.0f}"); print(f"| VRAM %     | {mn} | {me} | {mx} |")
print()
PYEOF

cat >> "$MD" << EOF

---

## Methodology / Safety

D1: each cell wrapped via scripts/gpu_run.sh; rc=99 retry-once.
D2: bench wrapped via scripts/gpu_run.sh; sysfs telemetry polled
once per second in a sibling shell (sysfs only, no device open,
no wrapper conflict).
EOF

echo
echo "Group D done."
echo "  MD:        $MD"
echo "  log:       $LOG"
echo "  telemetry: $TELEMETRY"
echo "  D1 rc:     $([ "$D1_FAIL" -eq 0 ] && echo OK || echo FAIL)"
echo "  D2 rc:     $BENCH_RC"
[ "$D1_FAIL" -eq 0 ] && [ "$BENCH_RC" -eq 0 ] && exit 0 || exit 1
