#pragma once
/*
 * binary_split.h — Binary splitting to compute e = sum_{k=0}^{N} 1/k!.
 *
 * split(a, b) computes (P, B) for the half-open range [a, b), meaning the
 * partial sum sum_{k=a}^{b-1} 1/k!.
 *
 * Invariant:   sum_{k=a}^{b-1} 1/k! = P / (b-1)!
 * where B = product_{k=a}^{b-1} k = (b-1)! / (a-1)!  (partial factorial).
 *
 * Leaf  (b - a == 1, k = a):  P = 1, B = a.
 *   Verify: P/B * (a-1)! = 1/a! ✓
 *
 * Merge at m = (a+b)/2:
 *   B  = B_L * B_R
 *   P  = P_L * B_R + P_R
 *
 * At root (a=1, b=N+1):
 *   B_root = N!,  P_root / B_root = sum_{k=1}^{N} 1/k!
 *   e = 1 + P_root / B_root   (the 1/0! = 1 term is added by the caller)
 *
 * See binary_split.c for N_TERMS selection and full algorithm.
 */

#include "../../lib/arith/bigint.h"
#include "mem_pool.h"

typedef struct {
    BigInt P;   /* partial-sum numerator          */
    BigInt B;   /* partial factorial = (b-1)!/(a-1)! */
} SplitResult;

/*
 * Compute (P, B) for sum_{k=a}^{b-1} 1/k!.
 * out->P and out->B are allocated inside; caller must free with split_free().
 * pool may be NULL (falls back to bigint_alloc/free directly).
 */
void binary_split(SplitResult *out, long a, long b, MemPool *pool);

/* Free both BigInts in a SplitResult. */
void split_free(SplitResult *s);

/*
 * Estimate the number of terms needed so that the tail sum_{k>N} 1/k!
 * is less than 10^(-n_digits).  Returns N rounded up to a safe margin.
 */
long split_terms_needed(long n_digits);
