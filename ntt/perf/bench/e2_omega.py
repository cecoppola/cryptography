#!/usr/bin/env python3
"""
e2_omega.py — compute a primitive n-th root of unity omega mod q.

Standalone (no hardcoded omega tables): factors q-1, finds a generator g,
then omega = g^((q-1)/n) mod q. Verifies omega^n == 1 and omega^(n/2) != 1
(i.e. omega is a *primitive* n-th root). Also requires n | (q-1); if n does
not divide q-1 there is no n-th root and the script exits non-zero.

Usage:  e2_omega.py <q> <n>
Exit :  0 prints omega; 2 = n not power of 2 / n<2; 3 = n does not divide
        q-1 (no n-th root); 4 = primitivity check failed.

Used by perf/bench/e2_cross_verify_all.sh to generate the (q,n,omega) grid
for the lib1 CPU-vs-GPU cross-verify sweep over the 14-prime curated set.
"""
import sys


def modpow(b, e, m):
    r = 1
    b %= m
    while e > 0:
        if e & 1:
            r = r * b % m
        b = b * b % m
        e >>= 1
    return r


def factorize(x):
    """Return the set of distinct prime factors of x (trial division;
    x = q-1 here is smooth for all NTT-friendly primes so this is fast)."""
    f = set()
    d = 2
    while d * d <= x:
        while x % d == 0:
            f.add(d)
            x //= d
        d += 1 if d == 2 else 2
    if x > 1:
        f.add(x)
    return f


def find_generator(q):
    """Smallest generator g of the multiplicative group mod q (q prime)."""
    phi = q - 1
    facs = factorize(phi)
    g = 2
    while g < q:
        if all(modpow(g, phi // p, q) != 1 for p in facs):
            return g
        g += 1
    raise RuntimeError("no generator found")


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: e2_omega.py <q> <n>\n")
        sys.exit(1)
    q = int(sys.argv[1])
    n = int(sys.argv[2])

    if n < 2 or (n & (n - 1)) != 0:
        sys.stderr.write("n must be a power of 2 >= 2\n")
        sys.exit(2)

    # n-th root of unity exists iff n | (q-1).
    if (q - 1) % n != 0:
        sys.stderr.write("n does not divide q-1: no n-th root\n")
        sys.exit(3)

    g = find_generator(q)
    omega = modpow(g, (q - 1) // n, q)

    # Primitivity: omega^n == 1 and omega^(n/2) != 1.
    if modpow(omega, n, q) != 1 or modpow(omega, n // 2, q) == 1:
        sys.stderr.write("primitivity check failed\n")
        sys.exit(4)

    print(omega)


if __name__ == "__main__":
    main()
