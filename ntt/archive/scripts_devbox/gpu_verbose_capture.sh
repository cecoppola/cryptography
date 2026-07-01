#!/usr/bin/env bash
# gpu_verbose_capture.sh — turn ON extra DRM/atomic logging + GPU event tracing
# just before a run so netconsole captures leading context (which CRTC/plane/fence
# stalled), not just the one-line "flip_done timed out".
#
# drm.debug=0x16 emits at KERN_DEBUG (priority 7). The printk console_loglevel is
# raised to 8 so those messages actually reach netconsole (the receiver on littleblue
# captures everything). Without this, verbose DRM lines go only to the local ring
# buffer and are lost on cold power-off.
#
# ftrace_dump_on_oops only fires on a kernel Oops/panic — NOT on a hard hang
# (flip_done wedge). The ftrace ring is therefore volatile on this crash class;
# crash_collect.sh (called by the receiver via SSH while the box may still respond)
# is the mechanism for pulling it post-crash.
#
#   sudo bash scripts/gpu_verbose_capture.sh        # ON  (before a run)
#   sudo bash scripts/gpu_verbose_capture.sh off     # OFF (restore quiet)
#
set -u
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }
TR=/sys/kernel/tracing; [ -d "$TR" ] || TR=/sys/kernel/debug/tracing

if [ "${1:-}" = "off" ]; then
  echo 0 > /sys/module/drm/parameters/debug 2>/dev/null || true
  # Restore default console loglevel (4 = KERN_WARNING and above).
  echo "4 4 1 7" > /proc/sys/kernel/printk 2>/dev/null || true
  [ -d "$TR" ] && { echo 0 > "$TR/tracing_on" 2>/dev/null
    echo 0 > "$TR/events/drm/enable" 2>/dev/null
    echo 0 > "$TR/events/gpu_scheduler/enable" 2>/dev/null
    echo 0 > "$TR/events/dma_fence/enable" 2>/dev/null; }
  sysctl -wq kernel.ftrace_dump_on_oops=0 2>/dev/null || true
  echo "verbose capture OFF. printk loglevel restored to 4."
  exit 0
fi

# Raise printk console_loglevel to 8 (KERN_DEBUG+1) so drm.debug messages reach
# netconsole. Only the current loglevel changes; default/min/boot stay at 4/1/7.
echo "8 4 1 7" > /proc/sys/kernel/printk 2>/dev/null \
  && echo "printk loglevel -> 8 (KERN_DEBUG visible on netconsole)" \
  || echo "WARN: couldn't set printk loglevel"

# DRM verbosity: DRIVER(0x2)|KMS(0x4)|ATOMIC(0x10) = 0x16 — the flip/atomic path,
# without the full 0x1ff firehose.
echo 0x16 > /sys/module/drm/parameters/debug 2>/dev/null \
  && echo "drm.debug=0x16 (DRIVER|KMS|ATOMIC)" || echo "WARN: couldn't set drm.debug"

# GPU event trail into the ftrace ring. Not auto-dumped on hard hang, but
# crash_collect.sh (called by receiver SSH) can pull it if the CPU is still alive.
if [ -d "$TR" ]; then
  echo 8192 > "$TR/buffer_size_kb" 2>/dev/null || true
  echo 1 > "$TR/events/drm/enable" 2>/dev/null || true
  echo 1 > "$TR/events/gpu_scheduler/enable" 2>/dev/null || true
  echo 1 > "$TR/events/dma_fence/enable" 2>/dev/null || true
  echo 1 > "$TR/tracing_on" 2>/dev/null || true
  echo "ftrace events: drm + gpu_scheduler + dma_fence (8MB ring)"
fi
sysctl -wq kernel.ftrace_dump_on_oops=1 2>/dev/null || true

echo "verbose capture ON. netconsole receiver on littleblue will capture DRM debug stream."
echo "On crash: cold power-off, then check ~/crash-captures/ on littleblue."
