#!/usr/bin/env bash
# gpu_test_logging.sh — exercise the logging features that work WITHOUT a reboot,
# in the current session. (ramoops's actual RAM capture needs the reserve_mem
# reboot and is NOT tested here — only its setup was dry-verified separately.)
set -u
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TR=/sys/kernel/tracing; [ -d "$TR" ] || TR=/sys/kernel/debug/tracing
pass=0; fail=0
ok(){ echo "  PASS: $1"; pass=$((pass+1)); }
no(){ echo "  FAIL: $1"; fail=$((fail+1)); }

echo "[1] kmsg O_SYNC logger — capture + durability across kill -9"
F=/tmp/logtest_kmsg.log
"$HERE/tools/kmsg_sync_logger" "$F" & P=$!
sleep 1; MSG="LOGTEST_$RANDOM$RANDOM"; echo "$MSG" > /dev/kmsg; sleep 0.3; kill -9 "$P" 2>/dev/null
if grep -q "$MSG" "$F" 2>/dev/null; then ok "captured '$MSG' and it survived kill -9"; else no "message not captured"; fi
rm -f "$F"

echo "[2] drm.debug verbosity (DRIVER|KMS|ATOMIC=0x16)"
_old=$(cat /sys/module/drm/parameters/debug)
echo 0x16 > /sys/module/drm/parameters/debug 2>/dev/null
_set=$(cat /sys/module/drm/parameters/debug)
sleep 3
_n=$(journalctl -k --since "4 seconds ago" 2>/dev/null | grep -ciE '\[drm:|drm_atomic|crtc|\bplane\b|vblank|flip')
echo "  drm.debug now = $_set ; DRM debug lines logged in 3s = $_n"
if [ "$_set" = "0x16" ]; then ok "drm.debug accepts 0x16"; else no "could not set drm.debug"; fi
[ "${_n:-0}" -gt 0 ] && echo "  (and it IS producing logs from live display activity ✓)" \
                      || echo "  (no DRM lines — screen was static; flag is set, will log under load)"
echo "$_old" > /sys/module/drm/parameters/debug 2>/dev/null

echo "[3] ftrace GPU event capture (gpu_scheduler + dma_fence) via a cu_probe dispatch"
if [ -d "$TR/events/gpu_scheduler" ] || [ -d "$TR/events/dma_fence" ]; then
  echo 1 > "$TR/events/gpu_scheduler/enable" 2>/dev/null || true
  echo 1 > "$TR/events/dma_fence/enable" 2>/dev/null || true
  : > "$TR/trace"; echo 1 > "$TR/tracing_on"
  env -u HSA_CU_MASK "$HERE/tools/cu_probe" >/dev/null 2>&1
  sleep 1
  _ev=$(grep -cE 'gpu_scheduler|dma_fence|drm_sched|fence_' "$TR/trace" 2>/dev/null)
  echo "  GPU trace events captured = $_ev"
  [ "${_ev:-0}" -gt 0 ] && ok "ftrace captured GPU scheduler/fence events" || no "no GPU events in trace ring"
  echo 0 > "$TR/tracing_on"; echo 0 > "$TR/events/gpu_scheduler/enable" 2>/dev/null || true
  echo 0 > "$TR/events/dma_fence/enable" 2>/dev/null || true
else
  echo "  (gpu_scheduler/dma_fence trace events not present on this kernel)"; no "trace event dirs missing"
fi

echo "[4] gpu_verbose_capture.sh on/off"
bash "$HERE/scripts/gpu_verbose_capture.sh"     >/dev/null 2>&1; _d1=$(cat /sys/module/drm/parameters/debug)
bash "$HERE/scripts/gpu_verbose_capture.sh" off >/dev/null 2>&1; _d2=$(cat /sys/module/drm/parameters/debug)
{ [ "$_d1" != "0" ] && [ "$_d2" = "0" ]; } && ok "verbose ON set drm.debug=$_d1, OFF restored 0" || no "on/off mismatch (on=$_d1 off=$_d2)"

echo
echo "==== logging self-test: $pass passed, $fail failed ===="
echo "NOTE: ramoops RAM capture is NOT exercised here (needs the reserve_mem reboot);"
echo "      its grub edit was dry-verified safe + idempotent separately."