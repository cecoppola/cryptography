#pragma once
/*
 * multiply.h — Hybrid BigInt multiplication dispatcher.
 *
 * Below BIGINT_MUL_THRESHOLD limbs: O(n²) schoolbook (56-bit sub-limb split).
 * Above threshold: NTT-based multiply via the 4-APU CRT pipeline.
 *
 * Call ntt_mul_init() once before using bigint_mul() with large operands.
 * Call ntt_mul_teardown() at program exit to release GPU resources.
 */

#include "bigint.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Compile-time lower bound for the NTT crossover (limbs per operand).
 * Overridable via -DBIGINT_MUL_THRESHOLD=N.  The actual runtime crossover
 * is held in a module variable (see bigint_mul_get/set_threshold) so that
 * callers can temporarily raise it without a rebuild — e.g. base_convert
 * uses a larger value on PCIe GPUs where hipStreamSynchronize is expensive. */
#ifndef BIGINT_MUL_THRESHOLD
#define BIGINT_MUL_THRESHOLD  64
#endif

/* Runtime threshold accessors.  The initial value equals BIGINT_MUL_THRESHOLD.
 * Not thread-safe for concurrent set calls; designed for single-threaded
 * adjustment around a parallel region (set before, restore after). */
int  bigint_mul_get_threshold(void);
void bigint_mul_set_threshold(int t);

/* Initialise the NTT multiply engine for operands up to max_limbs limbs.
 * max_log_n is ceil(log2(2 * max_limbs)); must be <= 24 for our prime set. */
void ntt_mul_init(int max_log_n);
void ntt_mul_teardown(void);

/* Returns the number of times ntt_bigint_mul dispatched to the GPU since
 * ntt_mul_init.  Used by compute_e to confirm GPU is actually invoked. */
unsigned long ntt_get_gpu_dispatch_count(void);

/* NTT-based multiply: the >= threshold path bigint_mul dispatches to. Defined
 * in arith/ntt_mul.hip; requires ntt_mul_init first.
 * Declared here so every caller (multiply.c, microbench_window.c, the host
 * differential harness) shares one prototype instead of a local extern. */
void ntt_bigint_mul(BigInt *c, const BigInt *a, const BigInt *b);

/* c = a * b.  c is resized as needed.  Thread-safe only if ntt_mul_init
 * has been called from the same thread before any parallel bigint_mul call. */
void bigint_mul(BigInt *c, const BigInt *a, const BigInt *b);

#ifdef __cplusplus
}
#endif
