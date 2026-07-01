#!/usr/bin/env bash
# crash_hunt_run.sh — run a GPU workload under full crash instrumentation.
#
# Stack:
#   1. kdump + hardlockup/softlockup_panic (armed separately) → vmcore+dmesg to
#      /var/crash on a panic, survives the cold-cycle.
#   2. kmsg heartbeat (4 Hz) → netconsole: last beat = freeze moment + state.
#   3. littleblue silence-watch: independent freeze timestamp on the receiver.
#   4. periodic amdgpu fence/sched/ring snapshots (crash_collect.sh): the last
#      snapshot before the freeze shows whether the GPU ring stalled.
#   5. console_loglevel=7 so every kmsg reaches netconsole.
#
# Usage: scripts/crash_hunt_run.sh <label> -- <gpu_run.sh args...>
#   e.g. scripts/crash_hunt_run.sh ln21_400 -- 200 lib3/compute_e/microbench_window 21 400
set -u
LABEL="${1:?need label}"; shift
[ "${1:-}" = "--" ] && shift
TS=$(date +%Y%m%d_%H%M%S)
DIR="results/crash_hunt/${LABEL}_${TS}"; mkdir -p "$DIR"
RECV=192.168.50.197
PROG=/tmp/crash_progress; : > "$PROG"

km() { printf '<4>CRASHHUNT %s\n' "$*" | sudo -n tee /dev/kmsg >/dev/null 2>&1; }
# crash_collect.sh must be invoked by the EXACT absolute path the NOPASSWD rule
# permits, else sudo demands a password.
CC="sudo -n /usr/bin/bash /home/machinus/ntt/scripts/crash_collect.sh"

echo "[crash_hunt] label=$LABEL dir=$DIR"
# Self-arm panic-on-lockup (runtime sysctls reset to 0 on reboot). sysctl -w is
# blocked but `tee` is NOPASSWD, so write /proc directly. Converts a silent hard
# lock into a panic → kdump vmcore in /var/crash + auto-reboot (panic=20).
echo 1 | sudo -n tee /proc/sys/kernel/hardlockup_panic >/dev/null 2>&1
echo 1 | sudo -n tee /proc/sys/kernel/softlockup_panic >/dev/null 2>&1
echo "7 4 1 7" | sudo -n tee /proc/sys/kernel/printk >/dev/null 2>&1   # loglevel 7 → all kmsg to netconsole
echo "[crash_hunt] kdump armed=$(cat /sys/kernel/kexec_crash_loaded 2>/dev/null) hardlockup_panic=$(cat /proc/sys/kernel/hardlockup_panic) softlockup_panic=$(cat /proc/sys/kernel/softlockup_panic) panic=$(cat /proc/sys/kernel/panic)s"

# pre-run amdgpu/system snapshot
$CC > "$DIR/pre_snapshot.txt" 2>&1 || true

# receiver silence-watch (independent freeze timestamp)
ssh "$RECV" 'pkill -f netconsole_silence_watch.py 2>/dev/null; sleep 0.3; setsid python3 ~/netconsole_silence_watch.py ~/netconsole-gpu.log ~/crash_silence.log 2.0 </dev/null >/dev/null 2>&1 &' >/dev/null 2>&1; sleep 0.5

# heartbeat
bash scripts/crash_heartbeat.sh "$PROG" 0.25 &
HB=$!

# periodic fence/ring snapshot (last one before freeze localizes a ring stall)
( n=0; while :; do n=$((n+1));
    { echo "=== snap $n $(date +%H:%M:%S) ==="; $CC 2>/dev/null \
        | sed -n '/DRM FENCE/,/FTRACE/p'; } >> "$DIR/fence_trend.txt" 2>&1
    sleep 5; done ) &
SNAP=$!

km "RUN START $LABEL :: $*"
echo "[crash_hunt] launching: gpu_run.sh $*"
bash scripts/gpu_run.sh "$@" > "$DIR/run.log" 2>&1
RC=$?
km "RUN END $LABEL rc=$RC"

kill "$HB" "$SNAP" 2>/dev/null; wait "$HB" "$SNAP" 2>/dev/null
ssh "$RECV" 'pkill -f netconsole_silence_watch.py 2>/dev/null' >/dev/null 2>&1

# post-run: new crash dumps?
$CC > "$DIR/post_snapshot.txt" 2>&1 || true
echo "[crash_hunt] /var/crash:"; sudo -n ls -t /var/crash 2>/dev/null | head
echo "[crash_hunt] rc=$RC  artifacts in $DIR"
echo "[crash_hunt] if the box froze+rebooted: read dump via  sudo ls /var/crash ; sudo cat /var/crash/*/dmesg.* | tail -200"
exit "$RC"
