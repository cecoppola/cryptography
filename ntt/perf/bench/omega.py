#!/usr/bin/env python3
"""Compute primitive n-th root of unity ω in Z/qZ.

Usage: python3 omega.py <q> <n>  -> prints omega
"""
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
    r = 1
    b %= m
    while e > 0:
        if e & 1: r = r * b % m
        b = b * b % m
        e >>= 1
    return r

def omega_for(q, n):
    g, ml2n = MODULI[q]
    max_n = 1 << ml2n
    if n > max_n or (n & (n - 1)):
        return None
    return modpow(g, (q - 1) // n, q)

if __name__ == "__main__":
    q = int(sys.argv[1]); n = int(sys.argv[2])
    w = omega_for(q, n)
    if w is None: sys.exit(1)
    print(w)
