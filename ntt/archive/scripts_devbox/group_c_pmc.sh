#!/bin/bash
# group_c_pmc.sh — rocprofv3 PMC capture, GPU-safety-wrapper-respecting.
#
# C1  Stockham LDS-fused kernel @ n=2048 q=998244353 (CRT-hi)
# C2  Polymul fused kernel      @ n=2048 q=998244353
#
# Counters match the 2026-05-15 X3 baseline for direct comparison:
#   SQ_INSTS_VALU SQ_INSTS_LDS SQ_INST_CYCLES_VMEM SQ_WAVES
#
# 2026-05-23 incident: the previous version of this script invoked
# rocprofv3 directly, bypassing scripts/gpu_run.sh. Combined with the
# polymul binary's hardcoded internal negacyclic+cyclic sweep (many
# instrumented kernel launches in tight succession), this reproduced
# the INCIDENT §5 RDNA2 hard-hang signature and hard-locked the box.
#
# This version:
#   (a) wraps EVERY rocprofv3 launch via scripts/gpu_run.sh per the
#       GPU-SAFETY rule (S1-S15 + post-check),
#   (b) reduces ITERS 1000 -> 100 (PMC counters dominate launch cost,
#       so 100 iters give a stable per-wave reading),
#   (c) uses --kernel-include-regex to limit instrumentation to the
#       perf-critical kernel ONLY (the binary still runs its full
#       internal sweep, but PMC capture only attaches to the target
#       kernel, sharply reducing instrumented dispatch count),
#   (d) extends inter-launch cooldown so the GPU returns to idle
#       between C1 and C2.
set -u
cd "$(dirname "$0")/.."

# ── HARD GUARD: PMC crashed the box twice (2026-05-23, 2026-06-06) on ─────────
# display-sharing gfx1030.  rocprofv3 hardware instrumentation corrupts the
# amdgpu driver state even when all instrumented kernels exit cleanly.
# SUPERVISED-ONLY (rocprofv3 PMC can leave residual driver state). Set
# GPU_RUN_ALLOW_LONG=1 to acknowledge. Each rocprofv3 is wrapped in gpu_run.sh
# (DM stays UP) — this is the path that SUCCEEDED 2026-06-06 and 2026-06-29.
# Do NOT use gpu_headless_run.sh: `systemctl stop gdm3` panics the display-
# sharing gfx1030 (that was the 06-06 crash). See the feedback-pmc memory.
if [ "${GPU_RUN_ALLOW_LONG:-0}" != "1" ]; then
  echo "[group_c_pmc] REFUSED: GPU_RUN_ALLOW_LONG is not set (SUPERVISED-ONLY)." >&2
  echo "  rocprofv3 PMC on display-sharing gfx1030 is risky — run only with the" >&2
  echo "  user present to cold-cycle if it wedges. Safe path (DM stays up):" >&2
  echo "    GPU_RUN_ALLOW_LONG=1 scripts/group_c_pmc.sh" >&2
  echo "  Do NOT use gpu_headless_run.sh here — stopping gdm3 panics this GPU." >&2
  exit 3
fi

TS=$(date +%Y%m%d_%H%M%S)
MD=perf/results/X3_PMC_GFX1030_${TS}.md
PMCDIR=results/pmc_${TS}
LOG=results/groupC_${TS}.log
mkdir -p "$PMCDIR" perf/results
: > "$LOG"

COUNTERS="SQ_INSTS_VALU SQ_INSTS_LDS SQ_INST_CYCLES_VMEM SQ_WAVES"
# 2026-05-23 incident #2: even with the kernel filter, the polymul
# binary's selftest+sweep crashed at ITERS=100. C2 now uses the
# binary's new --bench-only flag (see ntt_gpu_polymul.hip main());
# ITERS reduced 100→50 belt-and-suspenders. C1 stockham is safe at
# ITERS=100 (already verified twice).
N=2048; Q=998244353; OMEGA=584193783
ITERS_C1=100; ITERS_C2=50

cat > "$MD" << EOF
# gfx1030 (6900 XT) — rocprofv3 PMC capture (refresh)

Captured ${TS} on AMD Radeon RX 6900 XT (gfx1030), ROCm 7.0.3.
Driver: scripts/group_c_pmc.sh (post-2026-05-23-incident hardened).
Raw output: \`${PMCDIR}/\`. Counter set: \`${COUNTERS}\` — same as the
2026-05-15 X3 baseline for direct per-wave comparison.

Kernel config: n=${N} q=${Q} (CRT-hi) ω=${OMEGA}.
C1 stockham ITERS=${ITERS_C1}, C2 polymul ITERS=${ITERS_C2} with --bench-only.
PMC instrumentation is restricted via \`--kernel-include-regex\` to
the target kernel only, sharply limiting instrumented dispatch count
versus the binary's internal sweep.

Every rocprofv3 invocation runs under scripts/gpu_run.sh (GPU-SAFETY
rule, S1-S15 safeguards + post-check).

EOF

# ── helper: run rocprofv3 via gpu_run.sh, with kernel filter ────────────────
pmc_capture() {
    local label="$1" prefix="$2" kernel_re="$3"; shift 3
    echo "===== $label =====" >>"$LOG"
    # gpu_run.sh wraps rocprofv3; the inner binary still respects S10
    # stray-hip detection (it's spawned by rocprofv3 under our lock).
    # --output-format csv: modern rocprofv3 defaults to SQLite (.db) which
    # the parse step below doesn't read.
    scripts/gpu_run.sh 300 rocprofv3 --pmc $COUNTERS \
        --kernel-include-regex "$kernel_re" \
        --output-format csv \
        -d "$PMCDIR" -o "$prefix" -- "$@" >>"$LOG" 2>&1
    local rc=$?
    echo "[$label] gpu_run.sh+rocprofv3 rc=$rc" >>"$LOG"
    # Post-PMC health check: rocprofv3 can corrupt amdgpu driver state even
    # after a clean exit (rc=0). Two confirmed crashes on gfx1030 (2026-05-23,
    # 2026-06-06). If the driver is now unhealthy, abort before the next capture
    # rather than launching C2 on a corrupted driver.
    echo "[group_c] post-PMC health check ($label) ..." >>"$LOG"
    if ! scripts/gpu_health_check.sh >>"$LOG" 2>&1; then
        local hrc=$?
        echo "[group_c] ABORT after $label: gpu_health_check.sh returned $hrc." | tee -a "$LOG" >&2
        echo "  rocprofv3 corrupted the amdgpu driver. Power cycle required." | tee -a "$LOG" >&2
        return 1
    fi
    return $rc
}

# ── C1 Stockham LDS-fused ───────────────────────────────────────────────────
pmc_capture "C1 Stockham LDS-fused" "c1_stockham" \
    "stockham_lds_fused|stockham.*lds" \
    bin/ntt_gpu_stockham_6900xt $N $Q $OMEGA $ITERS_C1
C1_RC=$?

# Extra cooldown between heavy PMC captures (per the 2026-05-23 lesson).
echo "[group_c] inter-capture cooldown 30s" >>"$LOG"
sleep 30

# ── C2 Polymul fused — REQUIRES --bench-only (S17 enforces) ─────────────────
pmc_capture "C2 Polymul fused" "c2_polymul" \
    "fused_polymul|polymul_fused|fused.*kernel" \
    bin/ntt_gpu_polymul_6900xt $N $Q $OMEGA $ITERS_C2 --bench-only
C2_RC=$?

echo "C1_RC=$C1_RC  C2_RC=$C2_RC" >>"$LOG"

# ── Parse + tabulate ────────────────────────────────────────────────────────
PMCDIR_EXPORT="$PMCDIR" COUNTERS_EXPORT="$COUNTERS" python3 << 'PYEOF' >> "$MD" 2>>"$LOG"
import csv, glob, os, re
from collections import defaultdict

PMCDIR = os.environ["PMCDIR_EXPORT"]

# rocprofv3 csv output: each row is one (kernel-dispatch, counter) sample.
# Schema: Counter_Name / Counter_Value / Kernel_Name plus identifiers.
def find_csv(prefix):
    # Match the counter_collection file specifically — that's the one with
    # PMC samples. (agent_info, kernel_trace, etc. share the same prefix.)
    return sorted(glob.glob(f"{PMCDIR}/{prefix}*counter_collection.csv"))

def aggregate(csvs, counter_names):
    totals = defaultdict(float)
    by_kernel = defaultdict(lambda: defaultdict(float))
    counter_set = set(counter_names)
    for p in csvs:
        with open(p) as f:
            r = csv.DictReader(f)
            for row in r:
                kn = row.get("Kernel_Name") or "?"
                c  = row.get("Counter_Name")
                v  = row.get("Counter_Value")
                if c not in counter_set or v is None: continue
                try: vf = float(v)
                except ValueError: continue
                totals[c] += vf
                by_kernel[kn][c] += vf
    return totals, by_kernel

counters = os.environ["COUNTERS_EXPORT"].split()
for label, prefix, header in [
    ("C1 Stockham LDS-fused kernel", "c1_stockham", "C1 — Stockham LDS-fused (lib1/ntt_gpu_stockham.hip)"),
    ("C2 Polymul fused kernel",      "c2_polymul",  "C2 — Polymul fused (lib1/ntt_gpu_polymul.hip)"),
]:
    csvs = find_csv(prefix)
    print(f"\n## {header}\n")
    if not csvs:
        print("_no CSV emitted; see groupC log for rocprofv3 failure_\n")
        continue
    totals, by_kernel = aggregate(csvs, counters)
    waves = totals.get("SQ_WAVES", 0)
    print(f"Raw: {', '.join(os.path.basename(p) for p in csvs)}\n")
    print("| Counter | Total | Per Wave |\n|---|---:|---:|")
    for c in counters:
        v = totals.get(c, 0)
        pw = (v/waves) if waves else 0
        print(f"| {c} | {int(v):,} | {pw:.1f} |")
    print()
    if by_kernel:
        print("### Per-kernel breakdown\n")
        print("| Kernel | " + " | ".join(counters) + " |")
        print("|---" + ("|---:" * len(counters)) + "|")
        for k in sorted(by_kernel)[:12]:
            short = k if len(k) <= 60 else k[:57]+"..."
            cols = " | ".join(f"{int(by_kernel[k].get(c,0)):,}" for c in counters)
            print(f"| `{short}` | {cols} |")
        print()
PYEOF

cat >> "$MD" << 'EOF'

---

## Comparison to 2026-05-15 X3 baseline

The X3 baseline ran at n=1024 omega=3, 100 iters. This refresh is at
n=2048 omega=584193783 (CRT-hi primitive 2048-th root), 100 iters,
same counter set. Absolute totals scale with n; per-wave ratios are
directly comparable.

See PERFORMANCE.md §2 for current gfx1030 PMC numbers.
EOF

echo
echo "Group C done."
echo "  MD:  $MD"
echo "  CSV: $PMCDIR/"
echo "  log: $LOG"
[ "$C1_RC" -eq 0 ] && [ "$C2_RC" -eq 0 ] && exit 0 || exit 1
