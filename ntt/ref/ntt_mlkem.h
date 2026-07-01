/*
 * ntt_mlkem.h — ML-KEM (FIPS-203) specific 7-layer NTT over Z_3329.
 *
 * Distinct from ntt.h (generic n-th root, 8+ layers). ML-KEM uses
 * n=256 with a 2n-th root psi=17 and 7 NTT layers (incomplete NTT).
 */

#ifndef NTT_MLKEM_H
#define NTT_MLKEM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialise zeta / gamma tables (idempotent). Call once before any other
 * mlkem_* function. */
void mlkem_init_tables(void);

/* Read-only accessors for the precomputed tables (length 128 each). */
const uint64_t *mlkem_get_zeta_table(void);
const uint64_t *mlkem_get_gamma_table(void);

/* In-place 7-layer NTT / INTT on length-256 polynomials over Z_3329. */
void mlkem_ntt(uint64_t *f);
void mlkem_intt(uint64_t *f);

/* Pointwise polynomial multiplication in NTT domain: h = f * g mod (X^256+1). */
void mlkem_polymul(const uint64_t *f, const uint64_t *g, uint64_t *h);

#ifdef __cplusplus
}
#endif

#endif /* NTT_MLKEM_H */
