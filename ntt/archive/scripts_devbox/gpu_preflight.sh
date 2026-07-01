#!/bin/bash
# gpu_preflight.sh — boot-time GPU initialization and baseline snapshot
#
# Run by amdgpu-preflight.service after graphical target.
# 1. Waits 120s for SMU firmware to stabilize after boot.
# 2. Writes amdgpu debug_mask for SMU/power-mgmt kernel logging.
# 3. Captures a full GPU baseline snapshot to a dated file.
# 4. Writes a "GPU ready" marker that gpu_health_check.sh reads.
#
# All actions are sysfs-only.  No device opens, no /dev/kfd.

CARD="/sys/class/drm/card1/device"
PCI="/sys/bus/pci/devices/0000:0a:00.0"
# hwmon index varies after driver changes — detect by name
HW=$(for d in "${CARD}"/hwmon/hwmon*/; do
  [ "$(cat "${d}name" 2>/dev/null)" = "amdgpu" ] && echo "${d%/}" && break
done)
HW="${HW:-${CARD}/hwmon/hwmon0}"
KFD="/sys/class/kfd/kfd/topology/nodes/1"
BASELINE_DIR="/var/log/gpu_preflight"
READY_MARKER="/var/run/gpu_ready"
DEBUG_MASK="0x306"    # driver(0x2)+KMS(0x4)+SMU/power(0x300) kernel log verbosity
WAIT_S="${GPU_PREFLIGHT_WAIT:-120}"

mkdir -p "$BASELINE_DIR"
log() { echo "[gpu_preflight $(date '+%H:%M:%S')] $*"; }
rd()  { cat "$1" 2>/dev/null || echo "UNAVAIL"; }
mhz() { local v; v=$(rd "$1"); [ "$v" = "UNAVAIL" ] && echo "UNAVAIL" || echo "$(( v/1000000 )) MHz"; }
mw()  { local v; v=$(rd "$1"); [ "$v" = "UNAVAIL" ] && echo "UNAVAIL" || echo "$(( v/1000000 )) W"; }
mc_() { local v; v=$(rd "$1"); [ "$v" = "UNAVAIL" ] && echo "UNAVAIL" || echo "$(( v/1000 )) C"; }
mb()  { local v; v=$(rd "$1"); [ "$v" = "UNAVAIL" ] && echo "UNAVAIL" || echo "$(( v/1048576 )) MB"; }

# ── 1. Write debug_mask immediately (before waiting) ─────────────────────────
log "setting amdgpu debug_mask=${DEBUG_MASK}"
echo "$DEBUG_MASK" > /sys/module/amdgpu/parameters/debug_mask 2>/dev/null \
    && log "debug_mask OK" \
    || log "WARNING: could not write debug_mask"

# ── 2. Wait for SMU stabilisation ────────────────────────────────────────────
log "waiting ${WAIT_S}s for SMU firmware to stabilise..."
sleep "$WAIT_S"
log "wait done."

# ── 3. Disable GFXOFF pp_feature (bit 20) ────────────────────────────────────
# The SMU driver/firmware interface version mismatch (driver v0x40, firmware v0x41
# on Sienna Cichlid / RX 6900 XT with amdgpu-dkms 6.10.5) causes both
# DisallowGfxOff and AllowGfxOff commands to return 0xFFFFFFFF (timeout) when
# the GPU transitions in/out of GfxOff during HIP init and teardown.  This
# hard-hangs the machine (confirmed CRASHes 12+13, 2026-06-07/08).
#
# PRIMARY FIX (2026-06-09, CRASHes 15-17): ppfeaturemask=0xffe7bfff in GRUB and
# modprobe.d clears PP_GFXOFF_MASK (bit 20) in adev->pm.pp_feature at module
# load.  smu_v11_0_gfx_off_control() checks this cached value and returns 0
# immediately without ever calling AllowGfxOff/DisallowGfxOff (smu_v11_0.c:1113).
# When this module parameter is in effect the pp_features sysfs write below is
# REDUNDANT — and it sends an SMU SetAllFeatures command that is unnecessary and
# risky on a crash-booted machine whose SMU state may be degraded.
# Skip the write when ppfeaturemask already has bit 20 clear.
PP_FEAT="${CARD}/pp_features"
_ppfeat_param=$(cat /sys/module/amdgpu/parameters/ppfeaturemask 2>/dev/null | tr -d '[:space:]')
_ppfeat_dec=$(python3 -c "print(int('${_ppfeat_param}',16))" 2>/dev/null || echo 0)
_gfxoff_bit=$(python3 -c "print(1 if (${_ppfeat_dec} & (1<<20)) else 0)" 2>/dev/null || echo 1)
if [ "$_gfxoff_bit" = "0" ]; then
    log "GFXOFF: ppfeaturemask=${_ppfeat_param} already has bit 20 clear — driver will never call AllowGfxOff/DisallowGfxOff; skipping pp_features SMU write."
elif [ -w "$PP_FEAT" ]; then
    _pp_raw=$(cat "$PP_FEAT" 2>/dev/null)           # "features high: 0xHHHH low: 0xLLLL"
    # Extract without "0x" prefix so the combined write is "0x<high><low>" (not "0x0x...")
    _pp_high=$(echo "$_pp_raw" | grep -o 'high: 0x[0-9a-fA-F]*' | grep -o '[0-9a-fA-F]*$')
    _pp_low=$(echo  "$_pp_raw" | grep -o 'low: 0x[0-9a-fA-F]*'  | grep -o '[0-9a-fA-F]*$')
    if [ -n "$_pp_high" ] && [ -n "$_pp_low" ]; then
        # Clear bit 20 (GFXOFF) in the low word; write combined 64-bit value
        _new_low=$(python3 -c "print(hex(int('${_pp_low}',16) & ~(1<<20)))")
        _new_low_hex=$(printf '%08x' "$(python3 -c "print(int('${_pp_low}',16) & ~(1<<20))")")
        _new_combined="${_pp_high}${_new_low_hex}"
        if echo "0x${_new_combined}" > "$PP_FEAT" 2>/dev/null; then
            _verify=$(cat "$PP_FEAT" 2>/dev/null | grep -i 'gfxoff.*disabled' || true)
            if [ -n "$_verify" ]; then
                log "GFXOFF disabled: pp_features low 0x${_pp_low} -> ${_new_low} (verified)"
            else
                log "WARNING: wrote pp_features but GFXOFF still shows enabled in feature list"
            fi
        else
            log "WARNING: could not write pp_features (GFXOFF still enabled)"
        fi
    else
        log "WARNING: could not parse pp_features — GFXOFF may still be enabled"
    fi
else
    log "WARNING: pp_features not writable — GFXOFF may still be enabled"
fi

# ── 5. Capture baseline snapshot ─────────────────────────────────────────────
BASELINE="${BASELINE_DIR}/baseline_$(date '+%Y%m%d_%H%M%S').txt"
{
    echo "=== GPU PREFLIGHT BASELINE $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
    echo "uptime:              $(cat /proc/uptime)"
    echo ""
    echo "--- amdgpu driver ---"
    echo "debug_mask:          $(rd /sys/module/amdgpu/parameters/debug_mask)"
    echo "lockup_timeout:      $(rd /sys/module/amdgpu/parameters/lockup_timeout)"
    echo "gpu_recovery:        $(rd /sys/module/amdgpu/parameters/gpu_recovery)"
    echo "runpm:               $(rd /sys/module/amdgpu/parameters/runpm)"
    echo ""
    echo "--- power / performance ---"
    echo "perf_level:          $(rd ${CARD}/power_dpm_force_performance_level)"
    echo "dpm_state:           $(rd ${CARD}/power_dpm_state)"
    echo "runtime_status:      $(rd ${CARD}/power/runtime_status)"
    echo "clkpm (PCIe):        $(rd ${PCI}/link/clkpm)"
    echo "GFXOFF feature:      $(cat ${CARD}/pp_features 2>/dev/null | grep -o 'GFXOFF.*' || echo 'UNAVAIL')"
    echo ""
    # SMU-metrics reads (freq/temp/power via hwmon) issue TransferTableSmu2Dram,
    # the SMU message that wedges this fw/driver-mismatched SMU (CRASH 21). They
    # run at EVERY boot via amdgpu-preflight.service and are a self-inflicted SMU
    # poke on fragile hardware. Default OFF; set GPU_PREFLIGHT_SMU=1 to restore.
    if [ "${GPU_PREFLIGHT_SMU:-0}" = "1" ]; then
        echo "--- clocks (SMU liveness) ---"
        echo "sclk:                $(mhz ${HW}/freq1_input)"
        echo "mclk:                $(mhz ${HW}/freq2_input)"
        echo ""
        echo "--- thermals ---"
        echo "temp_edge:           $(mc_ ${HW}/temp1_input)"
        echo "temp_junction:       $(mc_ ${HW}/temp2_input)"
        echo "temp_mem:            $(mc_ ${HW}/temp3_input)"
        echo "power:               $(mw  ${HW}/power1_average)"
        echo "power_cap:           $(mw  ${HW}/power1_cap)"
    else
        echo "--- clocks/thermals/power: SKIPPED (GPU_PREFLIGHT_SMU=0; SMU-poke guard) ---"
    fi
    echo ""
    echo "--- VRAM ---"
    echo "vram_used:           $(mb  ${CARD}/mem_info_vram_used)"
    echo "vram_total:          $(mb  ${CARD}/mem_info_vram_total)"
    echo ""
    echo "--- PCIe ---"
    echo "link_speed:          $(rd  ${CARD}/current_link_speed)"
    echo "link_width:          $(rd  ${CARD}/current_link_width)"
    echo "max_link_speed:      $(rd  ${PCI}/max_link_speed)"
    echo "max_link_width:      $(rd  ${PCI}/max_link_width)"
    echo ""
    echo "--- PCIe AER counters (should all be 0 on clean boot) ---"
    echo "aer_correctable:"
    cat "${PCI}/aer_dev_correctable" 2>/dev/null | sed 's/^/  /'
    echo "aer_nonfatal:"
    cat "${PCI}/aer_dev_nonfatal"    2>/dev/null | sed 's/^/  /'
    echo "aer_fatal:"
    cat "${PCI}/aer_dev_fatal"       2>/dev/null | sed 's/^/  /'
    echo ""
    echo "--- KFD ---"
    echo "kfd_gpu_id:          $(rd  ${KFD}/gpu_id)"
    echo ""
    echo "--- amdgpu driver version ---"
    modinfo amdgpu 2>/dev/null | grep -E "^version|^filename|srcversion" || true
    echo ""
    echo "--- recent kernel GPU messages ---"
    journalctl -b 0 -k --no-pager -n 50 2>/dev/null \
        | grep -iE "amdgpu|smu|gfxoff|kfd|drm" || echo "(none)"
} | tee "$BASELINE"

log "baseline written: $BASELINE"

# ── 5b. Static base-clock pin (CRASH 22 mitigation, 2026-06-10) ───────────────
# Pin sclk to base (1300; 1825 hard-locks sustained runs, 2026-06-16) + mclk max,
# via 'manual' perf level. One SMU setup
# burst here at boot, then no dynamic re-clocking ever: no idle<->boost ramps
# (the transition window where the failing SMU wedges). Skipped under dpm=0
# (no DPM => nothing to pin) or GPU_PREFLIGHT_STATIC=0.
_dpm_param=$(cat /sys/module/amdgpu/parameters/dpm 2>/dev/null || echo -1)
if [ "${GPU_PREFLIGHT_STATIC:-1}" = "1" ] && [ "$_dpm_param" != "0" ]; then
    if bash "$(dirname "$0")/gpu_static_clock.sh" "${GPU_PREFLIGHT_STATIC_MHZ:-1300}" >> "$BASELINE_DIR/static_clock.log" 2>&1; then
        log "static clock pinned: $(grep '\*' ${CARD}/pp_dpm_sclk 2>/dev/null | tr -d '\n') (see static_clock.log)"
    else
        log "WARNING: static clock pin FAILED — dynamic clocking still active (see static_clock.log)"
    fi
fi

# ── 6. Write ready marker ─────────────────────────────────────────────────────
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') baseline=${BASELINE}" > "$READY_MARKER"
log "GPU ready marker written: $READY_MARKER"
