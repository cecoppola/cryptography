/* test_gmp_oracle.c — TRUE differential reference for the GPU NTT bigint multiply.
 *
 * Purpose: ntt_bigint_mul (GPU CRT-NTT) is compared against GMP's mpz_mul for a
 * single full-size multiply at a chosen operand limb-count. This is the missing
 * rigorous oracle for diagnosing/fixing the log_n=22 correctness bug: it checks
 * EVERY product limb (not just leading digits or commutativity) against a
 * trusted independent implementation, in well under a second.
 *
 * The few __gmpz_* prototypes are declared directly (rather than #include
 * <gmp.h>) so this oracle builds even where the GMP dev header is absent;
 * libgmp.so provides the symbols. mp_limb_t is 64-bit (unsigned long, LP64)
 * matching GMP's build here and our LIMB_BITS=64 layout, so mpz_import/export
 * with size=8, order=-1, endian=0 maps 1:1 to BigInt.limbs.
 *
 * Usage:  test_gmp_oracle <log_n> [seed]   (arg1 = log_n to test; default 22)
 *   operand_limbs : size of each random operand a,b (product needs ~2x → sets
 *                   the engine's log_n). Sweep this to find the correctness wall.
 * Output: PASS, or FAIL with first-diff index, differing-limb count, and where
 *         in the product the corruption begins (fraction).
 */
#include "bigint.h"
#include "multiply.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void ntt_bigint_mul(BigInt *c, const BigInt *a, const BigInt *b);

/* ── Minimal GMP C-ABI declarations (no gmp.h needed; link -lgmp) ─────────── */
typedef unsigned long mp_limb_t;                 /* 64-bit on this platform   */
typedef struct { int _mp_alloc, _mp_size; mp_limb_t *_mp_d; } __mpz_struct;
typedef __mpz_struct mpz_t[1];
extern void  __gmpz_init(__mpz_struct *);
extern void  __gmpz_clear(__mpz_struct *);
extern void  __gmpz_import(__mpz_struct *, size_t, int, size_t, int, size_t, const void *);
extern void *__gmpz_export(void *, size_t *, int, size_t, int, size_t, const __mpz_struct *);
extern void  __gmpz_mul(__mpz_struct *, const __mpz_struct *, const __mpz_struct *);

/* xorshift64 PRNG for deterministic random limbs. */
static uint64_t rng_state;
static uint64_t xr(void) {
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng_state = x;
}

int main(int argc, char **argv)
{
    /* ntt_bigint_mul uses a FIXED N=2^log_n set by ntt_mul_init (ntt_bigint_mul
     * reads log_n = g_nms.log_n). So we init at the log_n under test and size
     * operands to ~N/2 limbs each (product ~N, must stay <= N to avoid the cyclic
     * NTT wrapping). arg1 = log_n to test (default 22 = the production cap). */
    int log_n  = (argc > 1) ? atoi(argv[1]) : 22;
    rng_state  = (argc > 2) ? (uint64_t)strtoull(argv[2], NULL, 10) : 0x123456789abcdefULL;
    if (rng_state == 0) rng_state = 1;
    int N      = 1 << log_n;
    int nlimbs = N / 2 - 16;     /* operands ~half N; product ~N-32 < N (no wrap) */
    /* Optional argv[3]: override operand limb count (e.g. small operands at a
     * large log_n to isolate transform-size bugs from operand-size bugs). */
    if (argc > 3) { int ov = atoi(argv[3]); if (ov > 0 && ov < nlimbs) nlimbs = ov; }

    ntt_mul_init(log_n);
    printf("test_gmp_oracle: log_n=%d N=%d operand_limbs=%d\n", log_n, N, nlimbs);

    BigInt a = bigint_alloc(nlimbs + 8);
    BigInt b = bigint_alloc(nlimbs + 8);
    BigInt c = bigint_alloc(2 * nlimbs + 16);
    for (int i = 0; i < nlimbs; i++) { a.limbs[i] = xr(); b.limbs[i] = xr(); }
    a.limbs[nlimbs - 1] |= 1ULL << 63;   /* force top limb nonzero (exact size) */
    b.limbs[nlimbs - 1] |= 1ULL << 63;
    a.n_limbs = nlimbs;
    b.n_limbs = nlimbs;

    /* GPU multiply. */
    ntt_bigint_mul(&c, &a, &b);

    /* GMP reference. */
    mpz_t ga, gb, gc;
    __gmpz_init(ga); __gmpz_init(gb); __gmpz_init(gc);
    __gmpz_import(ga, (size_t)a.n_limbs, -1, sizeof(uint64_t), 0, 0, a.limbs);
    __gmpz_import(gb, (size_t)b.n_limbs, -1, sizeof(uint64_t), 0, 0, b.limbs);
    __gmpz_mul(gc, ga, gb);

    size_t ref_n = 0;
    uint64_t *ref = (uint64_t *)malloc((size_t)(2 * nlimbs + 16) * sizeof(uint64_t));
    memset(ref, 0, (size_t)(2 * nlimbs + 16) * sizeof(uint64_t));
    __gmpz_export(ref, &ref_n, -1, sizeof(uint64_t), 0, 0, gc);

    /* Compare limb-by-limb (trim both to significant length). */
    int gpu_n = c.n_limbs;
    int maxn  = (gpu_n > (int)ref_n) ? gpu_n : (int)ref_n;
    int first_diff = -1, ndiff = 0;
    for (int i = 0; i < maxn; i++) {
        uint64_t g = (i < gpu_n) ? (uint64_t)c.limbs[i] : 0;
        uint64_t r = (i < (int)ref_n) ? ref[i] : 0;
        if (g != r) { if (first_diff < 0) first_diff = i; ndiff++; }
    }

    if (first_diff < 0 && gpu_n == (int)ref_n) {
        printf("  PASS: all %d product limbs match GMP. (gpu_n=ref_n=%d)\n", maxn, gpu_n);
    } else {
        printf("  FAIL: gpu_n=%d ref_n=%zu  first_diff@limb %d (%.1f%% into product)  diff_limbs=%d/%d\n",
               gpu_n, ref_n, first_diff, maxn ? 100.0 * first_diff / maxn : 0.0, ndiff, maxn);
        if (first_diff >= 0)
            printf("        @%d: gpu=0x%016llx  gmp=0x%016llx\n", first_diff,
                   (unsigned long long)(first_diff < gpu_n ? (uint64_t)c.limbs[first_diff] : 0),
                   (unsigned long long)(first_diff < (int)ref_n ? ref[first_diff] : 0));
    }

    __gmpz_clear(ga); __gmpz_clear(gb); __gmpz_clear(gc);
    free(ref);
    bigint_free(&a); bigint_free(&b); bigint_free(&c);
    ntt_mul_teardown();
    return (first_diff < 0 && gpu_n == (int)ref_n) ? 0 : 1;
}
