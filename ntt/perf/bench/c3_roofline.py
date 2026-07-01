#!/usr/bin/env python3
"""C3 — Roofline positioning for 6900XT (gfx1030).

gfx1030 rocprof v1 does not support TCC_HIT/TCC_EA_RDREQ — the only hardware
counters we can capture are GRBM_GUI_ACTIVE and SQ_INSTS_{VALU,LDS}. So the
"roofline" here is built from kernel durations (rocprof --stats) plus the
analytic byte/op counts per butterfly for a radix-2 Stockham NTT.

Peak numbers (RDNA2 6900XT, documented):
  HBM bandwidth        : 512 GB/s raw, ~420 GB/s effective
  64-bit mulmod peak   : ~256 GIOPS (conservative — no single public figure)
"""
import csv, os, glob, math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R    = lambda *p: os.path.join(ROOT, *p)

PEAK_GIOPS = 256.0
PEAK_BW_GB = 420.0

# Analytic per-butterfly counts for a radix-2 Stockham NTT stage:
#   1 mulmod (twiddle * v)   → ~6 dependent 64-bit mul/add
#   1 add (u + wv)           → 1
#   1 sub (u - wv + q)       → 1
#   2 LDS reads + 2 LDS writes → 32 B each way
OPS_PER_BUTTERFLY   = 8           # effective 64-bit ops per butterfly
BYTES_PER_BUTTERFLY = 32          # two LDS reads + two LDS writes

def stage_count(n):        return int(math.log2(n))
def butterflies(n):        return (n//2) * stage_count(n)

def load_stats(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return list(csv.DictReader(f))

# Pull per-kernel durations from the --stats CSVs (one run each for stockham/polymul)
kernels = {}
for p in glob.glob(R("results/rocprof/c1_stats_*.stats.csv")):
    for r in load_stats(p):
        name  = r['Name'].split('(')[0].strip('"')
        calls = int(r['Calls'])
        ns    = int(r['TotalDurationNs'])
        if name not in kernels: kernels[name] = {'calls':0, 'ns':0}
        kernels[name]['calls'] += calls
        kernels[name]['ns']    += ns

# n used for the stats run (from c1_pmc_sweep.sh: n=256 for stockham, n=256 for polymul)
N_USED = 256
BF    = butterflies(N_USED)
VALU  = BF * OPS_PER_BUTTERFLY
LDS_B = BF * BYTES_PER_BUTTERFLY
HBM_B = N_USED * 8 * 2          # one read + one write per NTT

out = [
    "# C3 — Roofline positioning (gfx1030 / 6900 XT)",
    "",
    f"Peak assumed: {PEAK_GIOPS:.0f} GIOPS (64-bit mulmod chain), "
    f"{PEAK_BW_GB:.0f} GB/s HBM effective.",
    "",
    f"Analytic counts for n={N_USED} Stockham NTT: "
    f"{BF} butterflies, {VALU} VALU-ops, {LDS_B/1024:.1f} KB LDS traffic, "
    f"{HBM_B} B HBM traffic (1 R + 1 W of the coefficient vector).",
    "",
    "| kernel | calls | total ns | avg ns | per-NTT GIOPS | HBM GB/s | AI (ops/B HBM) | limit |",
    "|---|---|---|---|---|---|---|---|",
]

for k, d in sorted(kernels.items(), key=lambda kv:-kv[1]['ns']):
    avg_ns = d['ns'] / max(1, d['calls'])
    # per-NTT GIOPS = analytic VALU ops / avg kernel time
    giops  = VALU / (avg_ns * 1e-9) / 1e9
    gbs    = HBM_B / (avg_ns * 1e-9) / 1e9
    ai     = VALU / HBM_B
    limit  = "compute" if giops/PEAK_GIOPS > gbs/PEAK_BW_GB else "memory"
    out.append(f"| `{k}` | {d['calls']} | {d['ns']} | {avg_ns:.0f} | "
               f"{giops:.1f} | {gbs:.1f} | {ai:.2f} | {limit} |")

out += [
    "",
    "Notes:",
    "- Full hardware counters (TCC_HIT / TCC_EA_RDREQ / SQ_LDS_BANK_CONFLICT) are not "
    "supported by rocprof v1 on gfx1030; only GRBM_* and SQ_INSTS_{VALU,LDS} work. "
    "The above uses analytic op/byte counts, not measured traffic.",
    "- On CDNA3 (MI300A) all counters above are available; re-run `bench/c1_pmc_sweep.sh` "
    "on MI300A to replace the analytic column with measured values.",
]

dst = R("results/roofline/c3_roofline.md")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, 'w') as f: f.write("\n".join(out) + "\n")
print(f"wrote {dst}")
