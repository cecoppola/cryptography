#!/bin/bash
# setup_logging.sh — apply all GPU crash diagnostic configuration
# Run once with sudo: sudo bash scripts/setup_logging.sh

set -e

echo "[setup] creating journald drop-in..."
mkdir -p /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/gpu-crash-diag.conf << 'CONF'
[Journal]
Storage=persistent
SyncIntervalSec=1s
RateLimitIntervalSec=0
RateLimitBurst=0
CONF
echo "[setup] restarting journald..."
systemctl restart systemd-journald

echo "[setup] creating amdgpu debug modprobe config..."
cat > /etc/modprobe.d/amdgpu-debug.conf << 'CONF'
# amdgpu verbose logging for GPU crash diagnosis (2026-06-08)
# debug_mask 0x306 = driver(0x2) + KMS(0x4) + SMU/power-mgmt(0x300)
# Captures DisallowGfxOff sequences and power state transitions in kernel log.
# Takes effect on next module load (reboot or modprobe reload).
options amdgpu debug_mask=0x306
CONF

echo "[setup] installing preflight service..."
cp /home/machinus/ntt/scripts/amdgpu-preflight.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable amdgpu-preflight.service

echo "[setup] disabling PCIe link clock power management..."
echo 0 > /sys/bus/pci/devices/0000:0a:00.0/link/clkpm
cat > /etc/udev/rules.d/99-amdgpu-clkpm.conf << 'CONF'
SUBSYSTEM=="pci", ATTR{vendor}=="0x1002", ATTR{device}=="0x73bf", ATTR{link/clkpm}="0"
CONF

echo "[setup] installing sudoers rule for GFXOFF control..."
# gpu_run.sh S23b needs to re-disable GFXOFF (bit 20 of pp_features) if the
# preflight service was skipped or if something re-enabled GfxOff between boot
# and a compute run.  The rule allows writing to pp_features via tee only.
cat > /etc/sudoers.d/gpu-gfxoff << 'CONF'
# Allow machinus to disable GFXOFF in amdgpu pp_features without a password.
# Required by scripts/gpu_run.sh S23b to prevent AllowGfxOff/DisallowGfxOff
# SMU timeouts on RX 6900 XT (driver v0x40 vs firmware v0x41 mismatch, 2026-06-08).
machinus ALL=(root) NOPASSWD: /usr/bin/tee /sys/class/drm/card1/device/pp_features
CONF
chmod 0440 /etc/sudoers.d/gpu-gfxoff

echo "[setup] verifying..."
echo "  journald drop-in:  $(grep -c Storage /etc/systemd/journald.conf.d/gpu-crash-diag.conf) Storage line(s)"
echo "  amdgpu-debug.conf: $(grep -c debug_mask /etc/modprobe.d/amdgpu-debug.conf) debug_mask line(s)"
echo "  preflight service: $(systemctl is-enabled amdgpu-preflight.service)"
echo "  clkpm:             $(cat /sys/bus/pci/devices/0000:0a:00.0/link/clkpm)"
echo "  sudoers gpu-gfxoff:$(visudo -c -f /etc/sudoers.d/gpu-gfxoff 2>&1 | head -1)"
echo "[setup] done. GFXOFF disable takes effect after next power cycle (preflight service)."
