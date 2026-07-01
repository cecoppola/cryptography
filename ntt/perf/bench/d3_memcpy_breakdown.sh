#!/usr/bin/env bash
# D3: memcpy vs compute breakdown from the C2 hip-trace results.
set -u
cd /home/machinus/ntt
OUT=results/sweep/d3_memcpy_breakdown.csv

python3 - "$OUT" <<'PYEOF'
import sys, csv, os, glob
dst = sys.argv[1]
def read_stats(p):
    if not os.path.exists(p): return []
    with open(p) as f:
        rdr = csv.DictReader(f)
        return list(rdr)

# Union of kernel stats and hip api stats for stockham + polymul + fourstep.
totals = {}  # category -> ns
def bucket(name, ns):
    n = name.lower()
    if 'memcpy' in n or 'copybuffer' in n:         cat = 'memcpy'
    elif 'hiplaunch' in n:                         cat = 'launch'
    elif 'synchronize' in n or 'hipstream' in n:   cat = 'sync'
    elif 'hipmalloc' in n or 'hipfree' in n:       cat = 'alloc'
    elif 'hipgetdev' in n or 'hipevent' in n:      cat = 'query'
    elif 'stockham_lds_batched' in n:              cat = 'kernel_batched'
    elif 'stockham_lds_fused_otf' in n:            cat = 'kernel_otf'
    elif 'stockham_lds_fused_reg2' in n:           cat = 'kernel_reg2'
    elif 'stockham_lds_fused_coarse' in n:         cat = 'kernel_coarse'
    elif 'stockham_lds_fused' in n:                cat = 'kernel_fused'
    elif 'stockham_stage' in n:                    cat = 'kernel_stage'
    elif 'reduce_kernel' in n:                     cat = 'kernel_reduce'
    elif 'polymul' in n:                           cat = 'kernel_polymul'
    elif 'fourstep' in n or 'transpose' in n:      cat = 'kernel_fourstep'
    elif 'pm_' in n:                               cat = 'kernel_polymul_sep'
    else:                                          cat = 'other'
    totals[cat] = totals.get(cat, 0) + ns

for ws in glob.glob('results/trace/c2_*_trace.stats.csv'):
    for r in read_stats(ws):
        try: bucket(r['Name'], int(r['TotalDurationNs']))
        except: pass
for ws in glob.glob('results/trace/c2_*_trace.hip_stats.csv'):
    for r in read_stats(ws):
        try: bucket(r['Name'], int(r['TotalDurationNs']))
        except: pass
for ws in glob.glob('results/trace/c2_*_trace.copy_stats.csv'):
    for r in read_stats(ws):
        try: bucket(r['Name'], int(r['TotalDurationNs']))
        except: pass

tot = sum(totals.values()) or 1
with open(dst, 'w') as f:
    f.write('category,total_ns,pct_of_total\n')
    for k,v in sorted(totals.items(), key=lambda kv:-kv[1]):
        f.write(f'{k},{v},{100.0*v/tot:.2f}\n')
    f.write(f'_total,{tot},100.00\n')
PYEOF
echo "DONE D3"
