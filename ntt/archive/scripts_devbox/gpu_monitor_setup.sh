#!/usr/bin/env bash
# gpu_monitor_setup.sh — arm kernel-level capture so a GPU hard-hang produces a
# durable STACK TRACE + GPU event trail, not just a frozen screen.
#
# Strategy: the freeze is a CPU stuck on a GPU MMIO (or a stuck kernel task). The
# lockup/hung-task detectors fire an NMI/panic a few seconds later; the panic
# prints a stack trace (WHERE it hung) + dumps the ftrace ring (the GPU op
# sequence right before the hang) to the kernel log, which kmsg_sync_logger
# fdatasync's to disk. Everything here is RUNTIME (sysctl + tracefs) — no reboot.
#
#   sudo bash scripts/gpu_monitor_setup.sh         # arm
#   sudo bash scripts/gpu_monitor_setup.sh off      # restore defaults
#
set -u
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }
TR=/sys/kernel/tracing; [ -d "$TR" ] || TR=/sys/kernel/debug/tracing
sw() { sysctl -wq "$1=$2" 2>/dev/null || true; }

if [ "${1:-}" = "off" ]; then
  sw kernel.softlockup_panic 0; sw kernel.hardlockup_panic 0
  sw kernel.hung_task_panic 0; sw kernel.panic_on_oops 0
  sw kernel.ftrace_dump_on_oops 0; sw kernel.watchdog_thresh 10
  [ -d "$TR" ] && { echo 0 > "$TR/tracing_on" 2>/dev/null; echo nop > "$TR/current_tracer" 2>/dev/null
    echo 0 > "$TR/events/gpu_scheduler/enable" 2>/dev/null; echo 0 > "$TR/events/dma_fence/enable" 2>/dev/null; }
  echo "monitor capture disarmed (defaults restored)."
  exit 0
fi

# 1. Oops/hung-task handling — DELIBERATELY CONSERVATIVE.
#    softlockup_panic / hardlockup_panic at a 10s threshold panic on a transient
#    stall; combined with a high-priority logger that itself stalled a CPU, this
#    REBOOTED the box (2026-06-12). They are intentionally NOT set here anymore.
#    hung_task_panic stays at a long 60s timeout for genuine D-state wedges only.
sw kernel.hung_task_panic 1
sw kernel.hung_task_timeout_secs 60
sw kernel.panic_on_oops 1
sw kernel.panic 20                  # reboot 20s AFTER an oops (logger captures first)
sw kernel.ftrace_dump_on_oops 1     # dump the GPU event ring into the oops log

# 2. Max kernel verbosity so faults reach the log with full detail.
sw kernel.printk "8 4 1 7"
echo 1 > /sys/module/printk/parameters/ignore_loglevel 2>/dev/null || true

# 3. ftrace the GPU job/fence lifecycle (low-volume, high-signal). On panic the
#    ring is dumped, showing the last scheduler/fence events before the wedge.
if [ -d "$TR" ]; then
  echo 16384 > "$TR/buffer_size_kb" 2>/dev/null || true
  echo 1 > "$TR/events/gpu_scheduler/enable" 2>/dev/null || true
  echo 1 > "$TR/events/dma_fence/enable" 2>/dev/null || true
  # amdgpu VM-fault tracepoint if this kernel exposes it (names vary):
  for ev in amdgpu_vm_fault amdgpu/amdgpu_vm_fault amdgpu_iv; do
    echo 1 > "$TR/events/$ev/enable" 2>/dev/null && break
  done
  echo 1 > "$TR/tracing_on" 2>/dev/null || true
  echo "  ftrace armed: gpu_scheduler + dma_fence (buf 16MB), dump-on-oops on"
else
  echo "  (tracefs not found — skipping ftrace; lockup-panic stack still armed)"
fi

echo "DONE. Lockup/hung-task -> panic+stack, ftrace dump-on-oops, verbose printk."
echo "NOTE: on freeze, WAIT ~20-25s for the panic to fire & be captured BEFORE the"
echo "      cold reboot — the stack trace lands in the kmsg durable log in that window."
