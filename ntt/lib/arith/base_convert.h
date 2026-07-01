#pragma once
/*
 * base_convert.h — Divide-and-conquer decimal conversion for large BigInts.
 *
 * Algorithm: binary splitting on the digit count.
 *   bigint_to_decimal(N, n_digits) finds level L = ceil(log2(n_digits)),
 *   then recursively splits at 10^(2^(L-1)) until 1-digit base cases.
 *   Total work: O(M(n) * log n) where M(n) is cost of one n-digit multiply.
 *
 * Cache: base_convert_init(max_level) precomputes pow10[k] = 10^(2^k) and
 *   their Newton reciprocals for k = 0..max_level.  Must be called once.
 *   For 10^7 digits, max_level = 24 (2^24 > 10^7).
 */

#include "bigint.h"

/* Precompute pow10[k] = 10^(2^k) and Newton reciprocals for k = 0..max_level.
 * Must be called before bigint_to_decimal() or bigint_write_decimal(). */
void base_convert_init(int max_level);
void base_convert_teardown(void);

/* Allocate and return null-terminated decimal string for N.  Caller frees.
 * Returns NULL if cache is not large enough (call base_convert_init again). */
char *bigint_to_decimal(const BigInt *N);

/* Write decimal digits of N to file descriptor fd (no null terminator).
 * Returns number of digits written, or -1 on error. */
long bigint_write_decimal(const BigInt *N, int fd);

/* Compute out = 10^n using the precomputed pow10 cache (binary-rep multiply).
 * base_convert_init(ceil(log2(n))) must have been called first. */
void base_convert_pow10_exact(BigInt *out, long n);
