#!/usr/bin/env bash
# gpu_recovery_watchdog.sh — auto-RESET a GPU/SMU hard-freeze instead of leaving
# a brick that needs a manual power-button hold.
#
# It does NOT prevent the SMU wedge. It guarantees an automatic reboot, via a
# HEALTH-GATED hardware watchdog:
#   - sp5100_tco (B550 SP5100 TCO) provides /dev/watchdog (independent silicon).
#   - the 'watchdog' DAEMON pets it only while scripts/gpu_wdog_test.sh passes;
#     that test trips on the GPU-wedge kernel-log signatures (flip_done timeout,
#     SMU 0xFFFFFFFF, D3hot->D0 fail). When the wedge appears, the daemon stops
#     petting -> the board resets ~30s later.
#
# WHY NOT systemd RuntimeWatchdog: it pets from PID1's loop, which STAYS ALIVE
# during a "GPU wedged but CPU limping" hang (CRASH 2026-06-10) -> it kept
# petting -> never reset. The daemon's health test catches that case; systemd's
# dumb timer does not. So we DISABLE systemd RuntimeWatchdog and let the daemon
# own /dev/watchdog.
#
# CAVEAT: a watchdog reset is a WARM reset; this card's SMU does NOT clear on warm
# reboot. After auto-recovery the MACHINE is usable but the GPU's SMU stays
# wedged and gpu_health_check.sh blocks GPU work until a manual COLD power cycle.
#
#   sudo apt-get install -y watchdog      # one-time
#   sudo bash scripts/gpu_recovery_watchdog.sh        # enable
#   sudo bash scripts/gpu_recovery_watchdog.sh off    # disable
#
set -eu
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
WDDROPIN=/etc/systemd/system.conf.d/watchdog.conf
SYSCTL=/etc/sysctl.d/99-gpu-recovery.conf
WDCONF=/etc/watchdog.conf
INITRAMFS_MODS=/etc/initramfs-tools/modules
TEST=/usr/local/sbin/gpu_wdog_test.sh

if [ "${1:-}" = "off" ]; then
  systemctl disable --now watchdog.service 2>/dev/null || true
  rm -f "$WDDROPIN" "$SYSCTL" "$TEST"
  sed -i '/^sp5100_tco$/d' "$INITRAMFS_MODS" 2>/dev/null || true
  update-initramfs -u >/dev/null 2>&1 || true
  systemctl daemon-reexec
  echo "recovery watchdog disabled (reboot to fully release the device)."
  exit 0
fi

# 1. HW watchdog module: load now (explicit modprobe bypasses Ubuntu's kernel
#    blacklist) + force into initramfs so /dev/watchdog exists early at boot.
modprobe sp5100_tco 2>/dev/null || true
[ -e /dev/watchdog ] && echo "sp5100_tco loaded: /dev/watchdog present" \
  || echo "WARNING: /dev/watchdog absent — enable 'Watchdog Timer' in BIOS."
grep -qxF sp5100_tco "$INITRAMFS_MODS" 2>/dev/null || echo sp5100_tco >> "$INITRAMFS_MODS"
update-initramfs -u >/dev/null 2>&1 || true

# 2. DISABLE systemd RuntimeWatchdog so the daemon can own /dev/watchdog.
mkdir -p "$(dirname "$WDDROPIN")"
printf '[Manager]\nRuntimeWatchdogSec=0\n' > "$WDDROPIN"
systemctl daemon-reexec

# 3. The health-gated watchdog daemon.
if ! command -v watchdog >/dev/null 2>&1 && ! dpkg -l watchdog 2>/dev/null | grep -q '^ii'; then
  echo
  echo "!! The 'watchdog' daemon is NOT installed. Run first:"
  echo "     sudo apt-get install -y watchdog"
  echo "   then re-run this script. (sysctls below are applied regardless.)"
else
  install -m 0755 "$HERE/gpu_wdog_test.sh" "$TEST"
  # Make the daemon load sp5100_tco itself (its ExecStartPre runs `modprobe
  # $watchdog_module`). Explicit modprobe bypasses Ubuntu's kernel blacklist,
  # which the initramfs/modules-load path does NOT reliably do (sp5100_tco
  # failed to load this way on the 6.17 kernel — /dev/watchdog was absent).
  if grep -q '^watchdog_module=' /etc/default/watchdog 2>/dev/null; then
    sed -i 's/^watchdog_module=.*/watchdog_module="sp5100_tco"/' /etc/default/watchdog
  else
    echo 'watchdog_module="sp5100_tco"' >> /etc/default/watchdog
  fi
  cat > "$WDCONF" <<EOF
# GPU-wedge health-gated watchdog (scripts/gpu_recovery_watchdog.sh)
watchdog-device = /dev/watchdog
watchdog-timeout = 30
interval = 5
test-binary = $TEST
test-timeout = 10
realtime = yes
priority = 1
EOF
  systemctl enable watchdog.service
  systemctl restart watchdog.service   # restart so it re-opens /dev/watchdog now
  sleep 1
  if journalctl -b 0 -u watchdog --no-pager 2>/dev/null | tail -5 | grep -q "cannot open /dev/watchdog"; then
    echo "WARNING: daemon still cannot open /dev/watchdog — check 'lsmod | grep sp5100'"
  else
    echo "watchdog daemon armed (device $(ls /dev/watchdog 2>/dev/null), health test = $TEST)"
  fi
fi

# 4. Secondary path: panic-on-detectable-hang -> auto-reboot + kdump. Lowered
#    hung_task_timeout so a wedged-SMU uninterruptible task triggers reset fast.
cat > "$SYSCTL" <<'EOF'
kernel.panic = 20
kernel.panic_on_oops = 1
kernel.hung_task_panic = 1
kernel.hung_task_timeout_secs = 30
EOF
sysctl -p "$SYSCTL" >/dev/null 2>&1 || true

echo
echo "DONE. Verify:  systemctl status watchdog --no-pager ; wdctl"
echo "REMINDER: auto-reset is WARM — GPU SMU stays wedged until a COLD cycle;"
echo "  gpu_health_check.sh blocks GPU work until then."
