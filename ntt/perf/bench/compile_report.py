#!/usr/bin/env python3
"""Compile CSV data into MEGA_REPORT.md appendix tables."""
import csv, os, sys, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = lambda *p: os.path.join(ROOT, *p)

def read_csv(path):
    if not os.path.exists(path): return [], []
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows: return [], []
    return rows[0], rows[1:]

def md_table(header, rows, sig=0):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"]*len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)

sections = []

# A1: pivot by variant, show best NTT/s at each (n,q)
h, rows = read_csv(R("results/sweep/sweep_nq.csv"))
if rows:
    by = {}
    for r in rows:
        n, q, var, ntts, _ = r
        by.setdefault((n, q), {})[var] = ntts
    variants = ["multi-launch (global)", "LDS-fused (padded)",
                "LDS-fused reg2 (G5)", "LDS-fused OTF (G6)",
                "batched b=40 (correctness PASS)"]
    short = ["multi", "fused", "reg2", "OTF", "batch40"]
    hdr = ["n", "q"] + short
    tab = []
    for (n, q), d in sorted(by.items(), key=lambda kv:(int(kv[0][1]), int(kv[0][0]))):
        tab.append([n, q] + [d.get(v, "-") for v in variants])
    sections.append(("A1 — n×q throughput sweep (NTT/s)", md_table(hdr, tab)))

# A2: pivot
h, rows = read_csv(R("results/sweep/batch_scaling.csv"))
if rows:
    by = {}
    for r in rows:
        n, q, b, ntts, nspb = r
        by.setdefault(n, {})[int(b)] = ntts
    batches = sorted({int(r[2]) for r in rows})
    hdr = ["batch"] + [f"n={n}" for n in sorted(by, key=int)]
    tab = []
    for b in batches:
        tab.append([str(b)] + [by[n].get(b, "-") for n in sorted(by, key=int)])
    sections.append(("A2 — Batch-scaling NTT/s at q=998244353", md_table(hdr, tab)))

# A3
h, rows = read_csv(R("results/sweep/fourstep_largeN.csv"))
if rows:
    by = {}
    for r in rows:
        n, q, ntts, ms = r
        by.setdefault(q, {})[int(n)] = (ntts, ms)
    ns = sorted({int(r[0]) for r in rows})
    hdr = ["n"] + [f"q={q} ntts" for q in sorted(by, key=int)]
    tab = []
    for n in ns:
        tab.append([str(n)] + [by[q].get(n, ("-","-"))[0] for q in sorted(by, key=int)])
    sections.append(("A3 — Four-step large-N NTT/s", md_table(hdr, tab)))

# B1
h, rows = read_csv(R("results/stress/b1_stress.csv"))
if rows:
    sections.append(("B1 — Random-seed correctness stress (pass/fail out of runs)",
                     md_table(h, rows)))

# B2 — summarize min/max/mean over time
h, rows = read_csv(R("results/thermal/b2_thermal.csv"))
if rows:
    fused = [int(r[1]) for r in rows if r[1] not in ("NA","")]
    temps = [float(r[3]) for r in rows if r[3] not in ("NA","")]
    if fused and temps:
        tab = [
            ["duration (s)", rows[-1][0]],
            ["fused NTT/s min",  str(min(fused))],
            ["fused NTT/s max",  str(max(fused))],
            ["fused NTT/s mean", f"{sum(fused)/len(fused):.0f}"],
            ["temp edge °C min", f"{min(temps):.1f}"],
            ["temp edge °C max", f"{max(temps):.1f}"],
            ["temp edge °C mean",f"{sum(temps)/len(temps):.1f}"],
        ]
        sections.append(("B2 — 15-min thermal loop summary",
                         md_table(["metric","value"], tab)))

# D1
h, rows = read_csv(R("results/sweep/d1_crossover.csv"))
if rows:
    sections.append(("D1 — GPU vs CPU crossover", md_table(h, rows)))

# D4
h, rows = read_csv(R("results/sweep/d4_block_size.csv"))
if rows:
    sections.append(("D4 — Block-size sweep (multi-launch kernel)",
                     md_table(h, rows)))

# E1
h, rows = read_csv(R("results/sweep/e1_n4096.csv"))
if rows:
    sections.append(("E1 — n=4096 polymul", md_table(h, rows)))

# D3
h, rows = read_csv(R("results/sweep/d3_memcpy_breakdown.csv"))
if rows:
    sections.append(("D3 — Memcpy vs compute (Stockham, from C2 trace)",
                     md_table(h, rows)))

# E2 — summarize pass counts
p = R("results/sweep/cross_verify_all.txt")
if os.path.exists(p):
    txt = open(p).read()
    # strip ANSI
    import re
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    passed = txt.count("7 passed")
    failed = sum(int(m.group(1)) for m in re.finditer(r"(\d+) failed", txt))
    blocks = txt.count("=== n=")
    sections.append(("E2 — Cross-verify all moduli",
                     md_table(["metric","value"],
                              [["configurations", str(blocks)],
                               ["configs with 7/7 pass", str(passed)],
                               ["total failed tests", str(failed)]])))

# C1 stats
sp = R("results/rocprof/c1_stats_stockham.csv.stats.csv")
if not os.path.exists(sp):
    sp = R("results/rocprof/c1_stats_stockham.stats.csv")
if os.path.exists(sp):
    with open(sp) as f:
        rows = list(csv.reader(f))
    if rows:
        sections.append(("C1 — Stockham kernel timing (rocprof --stats)",
                         md_table(rows[0], rows[1:10])))

# Write back into MEGA_REPORT.md
report_path = R("results/MEGA_REPORT.md")
base = open(report_path).read() if os.path.exists(report_path) else ""
# Strip any prior auto appendix
marker = "## Appendix: raw numbers"
if marker in base:
    base = base.split(marker)[0].rstrip() + "\n"
with open(report_path, "w") as f:
    f.write(base)
    f.write("\n## Appendix: raw numbers\n\n")
    for title, tbl in sections:
        f.write(f"### {title}\n\n{tbl}\n\n")
print(f"wrote {report_path} with {len(sections)} sections")
