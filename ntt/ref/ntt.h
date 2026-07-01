/*
 * ntt.h — shared interface for all NTT implementations in this project.
 *
 * Include this header in every implementation file (.c or .hip) to ensure
 * the parameter struct and public function signatures stay consistent.
 * Each binary links against exactly one implementation of ntt_forward/
 * ntt_inverse (ntt_cpu.c or ntt_gpu.hip).
 *
 * Compatibility: C99 and C++ (hipcc). Uses extern "C" guard for C++ callers.
 */

#ifndef NTT_H
#define NTT_H

#include <stdint.h>
#include <stdlib.h>   /* size_t, malloc */
#include "ntt_moduli.h"  /* reduce_fn_t, addmod, submod, NTT_MODULI table  */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Parameter struct ────────────────────────────────────────────────────────
 * All transform parameters in one struct. Passed explicitly to every
 * function; no global mutable state. Filled by ntt_params_init().
 * ntt_params_init() looks up 'q' in NTT_MODULI and sets 'reduce' to the
 * fastest available reduction function; falls back to reduce_generic.
 */
typedef struct {
    uint64_t    n;          /* transform length; must be a power of 2        */
    uint64_t    q;          /* NTT-friendly prime modulus                     */
    uint64_t    omega;      /* primitive n-th root of unity mod q             */
    uint64_t    omega_inv;  /* modular inverse of omega (for INTT twiddles)   */
    uint64_t    n_inv;      /* modular inverse of n mod q (INTT final scale)  */
    uint32_t    log2_n;     /* log2(n); precomputed                           */
    reduce_fn_t reduce;     /* fast modular reduction: reduce(a*b, q) → [0,q)*/
} ntt_params_t;

/* ── Parameter initialisation ────────────────────────────────────────────────
 * ntt_params_init: compute derived fields (omega_inv, n_inv, log2_n).
 * Caller must set n, q, omega before calling. Uses Fermat's little theorem.
 * Returns 0 on success; -1 if n is not a power of 2.
 */
int ntt_params_init(ntt_params_t *p);

/* ── Twiddle factor allocation ───────────────────────────────────────────────
 * All functions return heap-allocated arrays; caller must free().
 * Returns NULL on allocation failure.
 *
 * ntt_alloc_twiddles:      tw[k] = omega^k mod q,     k in [0, n/2)
 * ntt_alloc_twiddles_inv:  tw[k] = omega_inv^k mod q, k in [0, n/2)
 */
uint64_t *ntt_alloc_twiddles(const ntt_params_t *p);
uint64_t *ntt_alloc_twiddles_inv(const ntt_params_t *p);

/* ── Core NTT — lazy reduction (ntt_cpu.c / ntt_gpu.hip) ────────────────────
 * ntt_forward: in-place CT-DIT NTT over Z_q.
 *   Input:  a[0..n-1] in natural order, values in [0, q).
 *   Output: NTT(a), values in [0, q).
 *   twiddles: from ntt_alloc_twiddles(p).
 *
 * ntt_inverse: in-place INTT. Applies forward NTT with omega_inv twiddles,
 *   then scales by n^{-1} mod q.
 *   twiddles_inv: from ntt_alloc_twiddles_inv(p).
 */
void ntt_forward(uint64_t *a, const uint64_t *twiddles,     const ntt_params_t *p);
void ntt_inverse(uint64_t *a, const uint64_t *twiddles_inv, const ntt_params_t *p);

/* ── Polynomial multiplication via NTT ───────────────────────────────────────
 * polymul_ntt:            cyclic convolution mod (X^n - 1).
 * polymul_ntt_negacyclic: negacyclic convolution mod (X^n + 1) via
 *                         twisted-NTT; psi is a 2n-th root of unity.
 * alloc_twist / alloc_twist_inv: twist[i] = psi^i / psi^-i mod q (length n).
 * c may alias f or g. tw/twi from ntt_alloc_twiddles[_inv].
 */
void polymul_ntt(const uint64_t * restrict f,
                 const uint64_t * restrict g,
                       uint64_t * restrict c,
                 const uint64_t *tw, const uint64_t *twi,
                 const ntt_params_t *p);

void polymul_ntt_negacyclic(const uint64_t * restrict f,
                            const uint64_t * restrict g,
                                  uint64_t * restrict c,
                            const uint64_t *tw, const uint64_t *twi,
                            const uint64_t *twist, const uint64_t *twist_inv,
                            const ntt_params_t *p);

uint64_t *alloc_twist(uint64_t n, uint64_t q, uint64_t psi);
uint64_t *alloc_twist_inv(uint64_t n, uint64_t q, uint64_t psi);

/* (The Montgomery-butterfly variant ntt_mont.c — and its twiddle
 * allocators — were retired 2026-05-17; see STATUS.md. No Montgomery
 * declarations remain in this header.) */

#ifdef __cplusplus
}
#endif

#endif /* NTT_H */
