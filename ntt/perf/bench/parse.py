#!/usr/bin/env python3
"""Parse NTT binary stdout and extract rows of NTT/s values.

Reads stdout from stdin, strips ANSI, picks key metrics. Prints CSV-ish tokens.
"""
import sys, re

ansi = re.compile(r'\x1b\[[0-9;]*m')

def clean(s): return ansi.sub('', s)

def parse_stockham(text):
    text = clean(text)
    out = {}
    # table rows like "│ multi-launch (global)    │        33295 │        29.33 │"
    for line in text.splitlines():
        m = re.match(r'\s*\u2502\s*(.*?)\s*\u2502\s*([0-9]+)\s*\u2502\s*([0-9.]+)\s*\u2502', line)
        if m:
            label = m.group(1).strip()
            ntts = int(m.group(2))
            nspb = float(m.group(3))
            out[label] = (ntts, nspb)
    return out

def parse_fourstep(text):
    text = clean(text)
    m = re.search(r'NTT throughput:\s*(\d+)\s*/s', text)
    t = re.search(r'Time per NTT:\s*([0-9.]+)\s*ms', text)
    return (int(m.group(1)) if m else None, float(t.group(1)) if t else None)

def parse_polymul(text):
    text = clean(text)
    out = {}
    # look for "polymul/s" rows or benchmark rows
    for line in text.splitlines():
        m = re.search(r'(separate|fused|cyclic|negacyclic).*?([0-9]+)\s*polymul/s', line, re.I)
        if m: out[m.group(1).lower()] = int(m.group(2))
    # also tabled "│ fused (1-kernel)         │    ...    │    98433 │"
    for line in text.splitlines():
        m = re.match(r'\s*\u2502\s*([a-zA-Z \-\(\)0-9]+?)\s*\u2502\s*([0-9]+)\s*\u2502', line)
        if m:
            label = m.group(1).strip().lower()
            if 'polymul' in label or 'fused' in label or 'separate' in label:
                out[label] = int(m.group(2))
    return out

if __name__ == "__main__":
    mode = sys.argv[1]
    text = sys.stdin.read()
    if mode == "stockham":
        for k,(v,ns) in parse_stockham(text).items():
            print(f"{k},{v},{ns}")
    elif mode == "fourstep":
        n, ms = parse_fourstep(text)
        print(f"{n},{ms}")
    elif mode == "polymul":
        for k,v in parse_polymul(text).items():
            print(f"{k},{v}")
