#pragma once
/*
 * newton.h — Newton-Raphson reciprocal for large-integer division.
 *
 * newton_reciprocal() computes x = floor(2^(2*b) / Q), b = bigint_bits(Q).
 * Seed from a native __uint128_t computation (no GMP); iterations via
 * bigint_mul(), which dispatches to GPU-NTT for large operands automatically.
 *
 * bigint_div_newton() uses the precomputed reciprocal to compute floor(N/D)
 * in O(M(n)) time.  Caller must precompute recip = newton_reciprocal(D)
 * once per distinct divisor D.
 */

#include "bigint.h"
#include "multiply.h"

/* Compute x ≈ floor(2^(2*b) / Q), b = bigint_bits(Q).
 * x is resized as needed.  The result is exact OR underestimates the
 * true floor by exactly 1 (it is never an overestimate): the Newton
 * increment underflows to 0 one step early when Q divides 2^(2*b)
 * (e.g. Q a power of two).  bigint_div_newton's correction loop is
 * designed to absorb this ≤1 underestimate. */
void newton_reciprocal(BigInt *x, const BigInt *Q);

/* Compute q = floor(N / D).
 * recip must equal floor(2^(2*b) / D) where b = bigint_bits(D).
 * At most two correction steps are applied to guarantee an exact result. */
void bigint_div_newton(BigInt *q, const BigInt *N, const BigInt *D,
                       const BigInt *recip);
