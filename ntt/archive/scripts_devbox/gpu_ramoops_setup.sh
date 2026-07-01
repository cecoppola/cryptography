#!/usr/bin/env bash
# gpu_ramoops_setup.sh — set up a RAM-backed console log (ramoops/pstore) that
# survives the instant total freeze, so the cause (e.g. "flip_done timed out")
# is captured even when the SSD logger can't write (the dying PCIe bus blocks the
# fsync). RAM writes need no bus/IO, and RAM contents survive a WARM reset.
#
# WORKFLOW after this is applied (one reboot to reserve the RAM region):
#   1. run the workload as usual (ramoops captures the kernel console to RAM)
#   2. on a freeze -> WARM reboot (reset button / `sudo reboot`, NOT power-off!)
#   3. read the captured console:  cat /sys/fs/pstore/console-ramoops-0
#   4. THEN cold-cycle for a clean SMU before the next GPU run
#
# Uses reserve_mem= (kernel >=6.12) so NO physical address is hardcoded (safe).
#
#   sudo bash scripts/gpu_ramoops_setup.sh        # apply, then reboot once
#   sudo bash scripts/gpu_ramoops_setup.sh off     # remove
#
set -eu
[ "$(id -u)" -eq 0 ] || { echo "must run as root (sudo)"; exit 1; }
GRUB=/etc/default/grub
MODCONF=/etc/modprobe.d/ramoops.conf
LOADCONF=/etc/modules-load.d/ramoops.conf

if [ "${1:-}" = "off" ]; then
  rm -f "$MODCONF" "$LOADCONF"
  sed -i -E '/^GRUB_CMDLINE_LINUX_DEFAULT=/ s/ reserve_mem=[^ "]*//g' "$GRUB"
  update-grub
  echo "ramoops removed. reboot to release the reserved RAM."
  exit 0
fi

# 1. Reserve a 16 MiB RAM region named 'ramoops' (address-free; kernel picks it).
sed -i -E '/^GRUB_CMDLINE_LINUX_DEFAULT=/ {
  s/ reserve_mem=[^ "]*//g
  s/"$/ reserve_mem=16M:4096:ramoops"/
}' "$GRUB"

# 2. ramoops module params: an 8 MiB console ring (captures ALL printk incl. the
#    always-logged *ERROR* flip_done line), plus oops/ftrace/pmsg areas. Powers of 2.
cat > "$MODCONF" <<'EOF'
options ramoops mem_name=ramoops console_size=0x800000 record_size=0x40000 ftrace_size=0x100000 pmsg_size=0x40000 ecc=1 dump_oops=1
EOF

# 3. Load ramoops at boot (post-boot is fine: the crash is long after boot, and
#    ramoops registers as a console as soon as it loads).
echo ramoops > "$LOADCONF"

update-grub
echo
echo "== staged =="
grep -o 'reserve_mem=[^ "]*' "$GRUB"
echo "modprobe.d: $(cat "$MODCONF")"
echo
echo "DONE. Reboot ONCE to reserve the RAM region and load ramoops, then verify:"
echo "  cat /sys/module/pstore/parameters/backend        # expect: ramoops"
echo "  ls -l /dev/pmsg0 /sys/fs/pstore/                  # ramoops present"
echo "  dmesg | grep -i ramoops                           # 'using ... at 0x...' = active"
