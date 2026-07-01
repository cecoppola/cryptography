#!/usr/bin/env bash
# gpu_wdog_test.sh — health test for the 'watchdog' daemon (test-binary).
#
# The daemon runs this every `interval` seconds and pets /dev/watchdog ONLY if it
# returns 0. Returning non-zero (or hanging past test-timeout) makes the daemon
# STOP petting -> the SP5100 hardware-resets the board ~watchdog-timeout later.
#
# This catches the "GPU wedged but CPU still limping" failure that systemd's
# RuntimeWatchdog misses (CRASH, 2026-06-10): PID1 stays alive and keeps petting,
# so a dumb timer never fires. We instead detect the wedge directly from the
# kernel ring buffer — NO GPU/SMU sysfs access, so the test itself adds zero SMU
# traffic. If journald is itself wedged, this read hangs -> test-timeout -> reset,
# which is also the desired outcome.
#
# Exit 0 = healthy (pet). Exit 1 = wedge detected (stop petting -> reset).
set -u

# Signatures of the GPU/SMU/package wedge seen on this card.
SIGS='flip_done timed out|TransferTableSmu2Dram|Unable to change power state from D3hot to D0|amdgpu.*ring [a-z0-9_.]* timeout|amdgpu.*GPU reset begin|amdgpu.*Failed to export SMU|smu.*response:0xFFFFFFFF|RLC.*timed out'

# LATCH: once a wedge signature is EVER seen this boot, fail permanently. The
# old "last 25 seconds" window had a hole (CRASH, 2026-06-11): the D3hot message
# logs ONCE, the limping system then goes silent, the window expires, the test
# wrongly reports healthy, and petting resumes -> no reset. A tmpfs latch file
# closes that hole: after detection the test fails forever until the reboot
# clears /run.
LATCH=/run/gpu_wdog.wedged
[ -e "$LATCH" ] && exit 1

if timeout -s KILL 8 journalctl -k --since "120 seconds ago" --no-pager 2>/dev/null \
     | grep -qE "$SIGS"; then
  : > "$LATCH"
  logger -t gpu_wdog "WEDGE SIGNATURE detected — latched; withholding watchdog pet to force reset"
  exit 1
fi
exit 0
