#!/usr/bin/env python3
"""Compute primitive n-th root of unity. Exits non-zero if n exceeds max_log2_n."""
import sys

MODULI = {
    257:        (3,   8),
    3329:       (3,   8),
    7681:       (3,   9),
    12289:      (11,  12),
    40961:      (3,   13),
    65537:      (3,   16),
    8380417:    (10,  13),
    167772161:  (3,   25),
    469762049:  (3,   26),
    998244353:  (3,   23),
    2013265921: (31,  27),
}

def modpow(b, e, m):
    r = 1; b %= m
    while e > 0:
        if e & 1: r = r * b % m
        b = b * b % m; e >>= 1
    return r

q = int(sys.argv[1]); n = int(sys.argv[2])
if q not in MODULI: sys.exit(2)
g, ml2n = MODULI[q]
if n > (1 << ml2n) or (n & (n - 1)) or n < 2: sys.exit(3)
print(modpow(g, (q - 1) // n, q))
