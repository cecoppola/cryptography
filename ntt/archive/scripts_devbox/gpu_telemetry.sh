#!/bin/bash
# gpu_telemetry.sh — continuous GPU sysfs telemetry logger
#
# Polls all critical GPU state nodes every second from sysfs only.
# No /dev/kfd, no device open, no rocm-smi.  Safe to run at any time.
#
# After a crash the last rows of the CSV show exact GPU state in the
# second(s) before the freeze.  PCIe AER counters reveal hardware errors
# that would otherwise be invisible.  Clock frequencies reveal SMU failures
# (freq=0 or freq at minimum means SMU is not responding to power requests).
#
# Usage:
#   bash scripts/gpu_telemetry.sh <logfile.csv> &
#   PID=$!; ...; kill $PID
#
# Columns:
#   timestamp | busy_pct | vram_used_mb | vram_total_mb |
#   temp_edge_c | temp_junc_c | temp_mem_c |
#   sclk_mhz | mclk_mhz | power_w |
#   perf_level | runtime_status |
#   pcie_speed | pcie_width |
#   aer_correctable | aer_nonfatal | aer_fatal |
#   kfd_gpu_id | notes

set -uo pipefail

LOGFILE="${1:?Usage: gpu_telemetry.sh <logfile.csv>}"
CARD="/sys/class/drm/card1/device"
PCI="/sys/bus/pci/devices/0000:0a:00.0"
# hwmon index varies after driver reinstalls; find by name file
HW=$(for d in "${CARD}"/hwmon/hwmon*/; do
  [ "$(cat "${d}name" 2>/dev/null)" = "amdgpu" ] && echo "${d%/}" && break
done)
HW="${HW:-${CARD}/hwmon/hwmon0}"
KFD="/sys/class/kfd/kfd/topology/nodes/1"
INTERVAL="${GPU_TELEM_INTERVAL:-1}"

mkdir -p "$(dirname "$LOGFILE")"

# SMU-metrics polling guard. The hwmon temp/freq/power nodes are served by the
# SMU metrics table (SMU msg TransferTableSmu2Dram, index 18). On this 6900XT
# the SMU firmware↔driver interface mismatch (fw if 0x41 / driver 0x40) makes
# that message return 0xFFFFFFFF and WEDGE the SMU, hard-hanging the machine —
# the first logged symptom of every recent crash is "Failed to export SMU
# metrics table". Polling those nodes at 1 Hz was therefore actively provoking
# the crash, not just observing it. Default OFF; set GPU_TELEMETRY_SMU=1 to
# restore (only on a stack with a matched SMU interface). CRASH 21, 2026-06-10.
SMU_METRICS="${GPU_TELEMETRY_SMU:-0}"
# Reads are timeout-guarded: a wedged SMU turns these into uninterruptible
# kernel sleeps; `timeout --signal=KILL` bounds the damage.
rd()  { timeout -s KILL 2 cat "$1" 2>/dev/null || echo "X"; }
mhz() { local v; v=$(rd "$1"); [ "$v" = "X" ] && echo "X" || echo $(( v / 1000000 )); }
mw()  { local v; v=$(rd "$1"); [ "$v" = "X" ] && echo "X" || echo $(( v / 1000000 )); }
mc()  { local v; v=$(rd "$1"); [ "$v" = "X" ] && echo "X" || echo $(( v / 1000 )); }
mb()  { local v; v=$(rd "$1"); [ "$v" = "X" ] && echo "X" || echo $(( v / 1048576 )); }

# Extract a single named counter from an AER file (e.g. "TOTAL_ERR_FATAL")
aer_total() {
    grep "$2" "$1" 2>/dev/null | awk '{print $NF}' || echo "X"
}

{
    echo "# gpu_telemetry started $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid=$$"
    echo "# interval=${INTERVAL}s  card=${CARD}  pci=${PCI}"
    echo "timestamp,busy_pct,vram_used_mb,vram_total_mb,temp_edge_c,temp_junc_c,temp_mem_c,sclk_mhz,mclk_mhz,power_w,perf_level,runtime_status,pcie_speed,pcie_width,aer_correctable,aer_nonfatal,aer_fatal,kfd_gpu_id"
} | tee -a "$LOGFILE"

trap 'echo "# gpu_telemetry stopped $(date -u +%H:%M:%S)" >> "$LOGFILE"; exit 0' TERM INT

while true; do
    ts=$(date -u '+%Y-%m-%d %H:%M:%S')

    # gpu_busy_percent is ALSO SMU-metrics-table-backed on Sienna Cichlid
    # (amdgpu_get_gpu_busy_percent -> SMU GetMetrics = TransferTableSmu2Dram).
    # Polling it at 1Hz wedged the SMU on the longer d=1100000 run (CRASH 23,
    # 2026-06-10) even after temp/freq/power were already gated off. It is now
    # under the same SMU_METRICS guard — telemetry during a run touches NO
    # SMU-backed node. vram/runtime_status/pcie/aer below are non-SMU sysfs.
    vu=$(mb    "${CARD}/mem_info_vram_used")
    vt=$(mb    "${CARD}/mem_info_vram_total")
    if [ "$SMU_METRICS" = "1" ]; then
        busy=$(rd "${CARD}/gpu_busy_percent")
        te=$(mc    "${HW}/temp1_input")
        tj=$(mc    "${HW}/temp2_input")
        tm=$(mc    "${HW}/temp3_input")
        sc=$(mhz   "${HW}/freq1_input")
        mc_=$(mhz  "${HW}/freq2_input")
        pw=$(mw    "${HW}/power1_average")
    else
        busy=- te=- tj=- tm=- sc=- mc_=- pw=-
    fi
    pl=$(rd    "${CARD}/power_dpm_force_performance_level")
    rs=$(rd    "${CARD}/power/runtime_status")
    ps=$(rd    "${CARD}/current_link_speed" | tr -d '\n')
    pw_=$(rd   "${CARD}/current_link_width")
    ac=$(aer_total "${PCI}/aer_dev_correctable" "TOTAL_ERR_COR")
    an=$(aer_total "${PCI}/aer_dev_nonfatal"    "TOTAL_ERR_NONFATAL")
    af=$(aer_total "${PCI}/aer_dev_fatal"       "TOTAL_ERR_FATAL")
    kg=$(rd    "${KFD}/gpu_id")

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$ts" "$busy" "$vu" "$vt" "$te" "$tj" "$tm" \
        "$sc" "$mc_" "$pw" "$pl" "$rs" "$ps" "$pw_" \
        "$ac" "$an" "$af" "$kg" \
        >> "$LOGFILE"

    sleep "$INTERVAL"
done
