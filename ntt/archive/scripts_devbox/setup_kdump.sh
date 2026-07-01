#!/usr/bin/env bash
# setup_kdump.sh — one-time root setup for kernel crash dump capture + analysis.
# Run once with: sudo bash scripts/setup_kdump.sh
# After setup, run: sudo reboot   (crashkernel= takes effect on next boot)
# After reboot, verify: sudo kdump-config show

set -euo pipefail

SUDOERS_FILE="/etc/sudoers.d/claude-kdump"
KDUMP_CFG="/etc/default/kdump-tools"
USER="${SUDO_USER:-machinus}"

log() { echo "[setup_kdump] $*"; }

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must run as root (sudo bash scripts/setup_kdump.sh)" >&2
    exit 1
fi

# ── 1. Configure kdump-tools ─────────────────────────────────────────────────
log "Configuring $KDUMP_CFG ..."
cat > "$KDUMP_CFG" << 'EOF'
# kdump-tools configuration — managed by scripts/setup_kdump.sh

USE_KDUMP=1

# Use the installed crash-kernel image built by linux-crashdump.
# If /var/lib/kdump/ is empty, kdump-config will fall back to the running
# kernel (which is relocatable on Ubuntu 6.8.x).
KDUMP_KERNEL=/var/lib/kdump/vmlinuz
KDUMP_INITRD=/var/lib/kdump/initrd.img

# Save dumps to /var/crash (default); keep last 5.
KDUMP_COREDIR="/var/crash"
KDUMP_NUM_DUMPS=5

# Fast lz4 compression; -d 31 = kernel pages only (strips user/free pages).
# Typical kernel-only vmcore is 200-600 MB even on 64 GB machines.
MAKEDUMP_ARGS="-c -d 31"
KDUMP_COMPRESSION=lz4

# Dump the dmesg ring buffer separately (fast, useful even if makedumpfile fails).
KDUMP_DUMP_DMESG=1

# On capture failure, drop to a shell so the crash kernel can be diagnosed
# rather than silently rebooting and losing state.
KDUMP_FAIL_CMD="echo 'makedumpfile FAILED — dropping to shell'; /bin/bash; reboot -f"
EOF
log "  done."

# ── 2. Verify the grub.d snippet is in place (installed by linux-crashdump) ──
GRUB_SNIPPET="/etc/default/grub.d/kdump-tools.cfg"
if [ -f "$GRUB_SNIPPET" ]; then
    log "crashkernel grub.d snippet: $GRUB_SNIPPET (already present)"
    grep crashkernel "$GRUB_SNIPPET" || true
else
    # Install it ourselves — formula from Ubuntu's linux-crashdump package.
    log "Installing crashkernel grub.d snippet ..."
    cat > "$GRUB_SNIPPET" << 'EOF'
GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT crashkernel=2G-4G:320M,4G-32G:512M,32G-64G:1024M,64G-128G:2048M,128G-:4096M"
EOF
fi

# ── 3. Update grub and initramfs ─────────────────────────────────────────────
log "Running update-grub ..."
update-grub 2>&1 | grep -E "Generating|Found|Warning|Error" || true
log "Running update-initramfs -u ..."
update-initramfs -u 2>&1 | grep -E "update-initramfs|error|warn" || true

# ── 4. Add NOPASSWD sudoers rules for autonomous crash analysis ───────────────
log "Installing sudoers rules -> $SUDOERS_FILE ..."
cat > "$SUDOERS_FILE" << EOF
# Passwordless rules for autonomous kdump setup and crash analysis.
# Installed by scripts/setup_kdump.sh — edit that script, not this file.

# kdump control
$USER ALL=(root) NOPASSWD: /usr/sbin/kdump-config
$USER ALL=(root) NOPASSWD: /bin/systemctl start kdump-tools
$USER ALL=(root) NOPASSWD: /bin/systemctl stop kdump-tools
$USER ALL=(root) NOPASSWD: /bin/systemctl restart kdump-tools
$USER ALL=(root) NOPASSWD: /bin/systemctl status kdump-tools

# Crash analysis: read vmcore files
$USER ALL=(root) NOPASSWD: /usr/bin/crash
$USER ALL=(root) NOPASSWD: /usr/bin/makedumpfile

# Read crash dumps directory
$USER ALL=(root) NOPASSWD: /bin/ls /var/crash
$USER ALL=(root) NOPASSWD: /bin/ls /var/crash/*
$USER ALL=(root) NOPASSWD: /bin/cat /var/crash/*/dmesg.*
$USER ALL=(root) NOPASSWD: /bin/find /var/crash -type f

# pp_features write (used by gpu_preflight.sh and gpu_run.sh S23b)
$USER ALL=(root) NOPASSWD: /usr/bin/tee /sys/class/drm/card*/device/pp_features
EOF
chmod 440 "$SUDOERS_FILE"
visudo -c -f "$SUDOERS_FILE" && log "  sudoers syntax OK." || {
    log "ERROR: sudoers syntax check failed — removing $SUDOERS_FILE"
    rm -f "$SUDOERS_FILE"; exit 1
}

# ── 5. Enable and start kdump-tools service ───────────────────────────────────
log "Enabling kdump-tools.service ..."
systemctl enable kdump-tools 2>/dev/null || true
# Don't start yet — no crashkernel= memory reserved until next boot.
log "  (service start deferred until after reboot with crashkernel= active)"

# ── 6. Summary ───────────────────────────────────────────────────────────────
cat << 'SUMMARY'

╔══════════════════════════════════════════════════════════════════╗
║                    kdump SETUP COMPLETE                          ║
╚══════════════════════════════════════════════════════════════════╝

Next step:
  sudo reboot       ← boots with crashkernel= reserved; kdump-tools
                       will load the crash kernel automatically.

After reboot, verify with:
  sudo kdump-config show
  grep crashkernel /proc/cmdline   ← must be present

On next crash:
  vmcore  → /var/crash/<timestamp>/dump.*.gz
  dmesg   → /var/crash/<timestamp>/dmesg.*

To analyze a dump (no sudo needed for crash invocation itself):
  sudo kdump-config show   ← find the vmcore path
  sudo crash /usr/lib/debug/boot/vmlinux-$(uname -r) /var/crash/<ts>/dump.*

SUMMARY
