/*
 * bigint.c — Arbitrary-precision unsigned integer (variable-width limbs).
 *
 * Supports LIMB_BITS=64 (default) and LIMB_BITS=112 (Phase F upgrade).
 * Most arithmetic functions (add/sub/shl/shr/mul/div) use LIMB_BITS
 * generically. bigint_gather has explicit #if LIMB_BITS == 64 / 112
 * paths because the U256→limb re-packing is width-specific.
 * Both widths pass the full dual-width test suite (lib/arith/test_arith.c).
 */

#include "bigint.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* ─── Allocation ─────────────────────────────────────────────────────────── */

BigInt bigint_alloc(int cap)
{
    BigInt a;
    assert(cap > 0);
    a.limbs   = (limb_t *)calloc((size_t)cap, sizeof(limb_t));
    if (!a.limbs) { fprintf(stderr, "bigint_alloc: OOM\n"); exit(1); }
    a.n_limbs = 0;
    a.cap     = cap;
    a.managed = 0;
    return a;
}

/* hip_free_limbs: weak fallback; overridden by bigint_hip_alloc.o when linked. */
__attribute__((weak)) void hip_free_limbs(limb_t *p) { free(p); }

void bigint_free(BigInt *a)
{
    if (a->managed)
        hip_free_limbs(a->limbs);
    else
        free(a->limbs);
    a->limbs   = NULL;
    a->n_limbs = 0;
    a->cap     = 0;
    a->managed = 0;
}

void bigint_resize(BigInt *a, int new_cap)
{
    if (new_cap <= a->cap) return;
    if (a->managed) {
        /* Managed BigInts must be pre-allocated with sufficient cap.
         * hipRealloc does not exist; growing would require alloc+copy+free. */
        fprintf(stderr, "bigint_resize: managed BigInt cap=%d insufficient "
                "for new_cap=%d — pre-allocate larger\n", a->cap, new_cap);
        abort();
    }
    a->limbs = (limb_t *)realloc(a->limbs, (size_t)new_cap * sizeof(limb_t));
    if (!a->limbs) { fprintf(stderr, "bigint_resize: OOM\n"); exit(1); }
    memset(a->limbs + a->cap, 0, (size_t)(new_cap - a->cap) * sizeof(limb_t));
    a->cap = new_cap;
}

/* bigint_alloc_managed: weak fallback using calloc + managed=1.
 * Overridden by bigint_hip_alloc.o (hipMallocManaged) when linked. */
__attribute__((weak)) BigInt bigint_alloc_managed(int cap)
{
    BigInt a = bigint_alloc(cap);
    a.managed = 1;
    return a;
}

/* ─── Internal helpers ───────────────────────────────────────────────────── */

static void trim(BigInt *a)
{
    while (a->n_limbs > 0 && a->limbs[a->n_limbs - 1] == 0)
        a->n_limbs--;
}

/* ─── Assignment ─────────────────────────────────────────────────────────── */

void bigint_zero(BigInt *a)
{
    memset(a->limbs, 0, (size_t)a->cap * sizeof(limb_t));
    a->n_limbs = 0;
}

void bigint_set_u64(BigInt *a, uint64_t v)
{
    bigint_zero(a);
    if (v) { a->limbs[0] = v; a->n_limbs = 1; }
}

void bigint_copy(BigInt *dst, const BigInt *src)
{
    bigint_resize(dst, src->n_limbs > 0 ? src->n_limbs : 1);
    memcpy(dst->limbs, src->limbs, (size_t)src->n_limbs * sizeof(limb_t));
    if (dst->cap > src->n_limbs)
        memset(dst->limbs + src->n_limbs, 0,
               (size_t)(dst->cap - src->n_limbs) * sizeof(limb_t));
    dst->n_limbs = src->n_limbs;
}

/* ─── Comparison ─────────────────────────────────────────────────────────── */

int bigint_is_zero(const BigInt *a) { return a->n_limbs == 0; }

int bigint_cmp(const BigInt *a, const BigInt *b)
{
    if (a->n_limbs != b->n_limbs)
        return a->n_limbs > b->n_limbs ? 1 : -1;
    for (int i = a->n_limbs - 1; i >= 0; i--) {
        if (a->limbs[i] != b->limbs[i])
            return a->limbs[i] > b->limbs[i] ? 1 : -1;
    }
    return 0;
}

/* ─── Addition ───────────────────────────────────────────────────────────── */

void bigint_add(BigInt *c, const BigInt *a, const BigInt *b)
{
    int na = a->n_limbs, nb = b->n_limbs;
    int nc = (na > nb ? na : nb) + 1;
    bigint_resize(c, nc);
    /* No bigint_zero: loop writes every limb before reading it (alias-safe). */

    limb_t carry = 0;
    int i;
    for (i = 0; i < nc - 1; i++) {
        limb_t ai = i < na ? a->limbs[i] : 0;
        limb_t bi = i < nb ? b->limbs[i] : 0;
        unsigned __int128 s = (unsigned __int128)ai + bi + carry;
        c->limbs[i] = (limb_t)(s & LIMB_MASK);
        carry = (limb_t)(s >> LIMB_BITS);
    }
    c->limbs[i] = carry;
    c->n_limbs = nc;
    trim(c);
}

/* ─── Subtraction (requires a >= b) ─────────────────────────────────────── */

void bigint_sub(BigInt *c, const BigInt *a, const BigInt *b)
{
    int na = a->n_limbs;
    bigint_resize(c, na > 0 ? na : 1);
    /* No bigint_zero: loop writes every limb before reading it (alias-safe). */

    limb_t borrow = 0;
    for (int i = 0; i < na; i++) {
        unsigned __int128 ai = a->limbs[i];
        unsigned __int128 bi = i < b->n_limbs ? b->limbs[i] : 0;
        unsigned __int128 d  = ai - bi - borrow;
        /* u128 promotion makes bi+borrow exact, so the old 64-bit-only
         * `bi == UINT64_MAX` overflow clause is no longer needed. */
        borrow = (ai < bi + borrow) ? 1 : 0;
        c->limbs[i] = (limb_t)(d & LIMB_MASK);
    }
    /* assert(borrow == 0): caller guarantees a >= b */
    c->n_limbs = na;
    trim(c);
}

/* ─── Scalar add in-place ────────────────────────────────────────────────── */

void bigint_add_u64(BigInt *a, uint64_t v)
{
    bigint_resize(a, a->n_limbs + 2);
    limb_t carry = v;
    int i = 0;
    while (carry) {
        unsigned __int128 s = (unsigned __int128)a->limbs[i] + carry;
        a->limbs[i] = (limb_t)(s & LIMB_MASK);
        carry = (limb_t)(s >> LIMB_BITS);
        i++;
        if (i > a->n_limbs) a->n_limbs = i;
    }
    /* Redundant: if v>0, carry=v, the while loop runs ≥1 iteration and sets
     * n_limbs via the `if (i > a->n_limbs)` branch above. Never reachable. */
    trim(a);
}

/* ─── Shifts ─────────────────────────────────────────────────────────────── */

void bigint_shl(BigInt *c, const BigInt *a, int bits)
{
    int limb_shift = bits / LIMB_BITS;
    int bit_shift  = bits % LIMB_BITS;
    int nc = a->n_limbs + limb_shift + 1;
    bigint_resize(c, nc);
    bigint_zero(c);

    if (bit_shift == 0) {
        memcpy(c->limbs + limb_shift, a->limbs,
               (size_t)a->n_limbs * sizeof(limb_t));
    } else {
        for (int i = 0; i < a->n_limbs; i++) {
            c->limbs[i + limb_shift]     |= (a->limbs[i] << bit_shift) & LIMB_MASK;
            c->limbs[i + limb_shift + 1] |= a->limbs[i] >> (LIMB_BITS - bit_shift);
        }
    }
    c->n_limbs = nc;
    trim(c);
}

void bigint_shr(BigInt *c, const BigInt *a, int bits)
{
    int limb_shift = bits / LIMB_BITS;
    int bit_shift  = bits % LIMB_BITS;
    if (limb_shift >= a->n_limbs) { bigint_zero(c); return; }

    int nc = a->n_limbs - limb_shift;
    bigint_resize(c, nc > 0 ? nc : 1);
    /* No bigint_zero: the else-loop reads higher-index limbs before writing
     * lower ones (alias-safe when c==a; reads always lead writes in index).
     * The bit_shift==0 branch MUST use memmove, not memcpy: when c==a and
     * limb_shift>0 the source (a->limbs+limb_shift) overlaps the destination
     * (c->limbs), which is undefined behaviour for memcpy (caught by ASan's
     * memcpy-param-overlap). memmove copies down-safely and is identical to
     * memcpy when the buffers do not alias. */
    if (bit_shift == 0) {
        memmove(c->limbs, a->limbs + limb_shift,
                (size_t)nc * sizeof(limb_t));
    } else {
        for (int i = 0; i < nc; i++) {
            c->limbs[i] = a->limbs[i + limb_shift] >> bit_shift;
            if (i + limb_shift + 1 < a->n_limbs)
                c->limbs[i] |= (a->limbs[i + limb_shift + 1]
                                << (LIMB_BITS - bit_shift)) & LIMB_MASK;
        }
    }
    c->n_limbs = nc;
    trim(c);
}

/* ─── Scalar sub in-place ────────────────────────────────────────────────── */

void bigint_sub_u64(BigInt *a, uint64_t v)
{
    limb_t borrow = v;
    for (int i = 0; i < a->n_limbs && borrow; i++) {
        limb_t old = a->limbs[i];
        a->limbs[i]  = (limb_t)((old - borrow) & LIMB_MASK);
        borrow = (old < borrow) ? 1 : 0;
    }
    while (a->n_limbs > 0 && a->limbs[a->n_limbs - 1] == 0)
        a->n_limbs--;
}

/* ─── Scalar multiply ────────────────────────────────────────────────────── */

void bigint_mul_u64(BigInt *c, const BigInt *a, uint64_t v)
{
    int nc = a->n_limbs + 1;
    bigint_resize(c, nc);
    bigint_zero(c);
    /* v (<=2^64-1) fits a limb at both LB=64 and 112; mul_limb keeps
     * the 112-bit product within __uint128_t (F.4). */
    limb_t carry = 0;
    for (int i = 0; i < a->n_limbs; i++) {
        limb_t r0, r1;
        mul_limb(a->limbs[i], (limb_t)v, &r0, &r1);
        unsigned __int128 acc = (unsigned __int128)r0 + carry;
        c->limbs[i] = (limb_t)(acc & LIMB_MASK);
        carry = (limb_t)(r1 + (acc >> LIMB_BITS));
    }
    c->limbs[a->n_limbs] = carry;
    c->n_limbs = nc;
    trim(c);
}

/* ─── NTT scatter ────────────────────────────────────────────────────────── */
/*
 * Reduce each limb mod PRIMES[pidx] and write to coeffs[]. Zero-pad
 * from a->n_limbs to n_coeffs.
 *
 * F.5: a 112-bit limb can exceed p, so reduce with (__uint128_t)limb
 * % p — pure standard-dialect C (__uint128_t is the codebase's base
 * type, used throughout bigint.c). At LB=64 this equals the original
 * `limb % p` (limb < 2^64). The Solinas reduce_pK fast path is in
 * primes.h with HIP/C++ qualifiers (__host__ __device__ __force-
 * inline__) and is deliberately NOT pulled into pure-C bigint.c;
 * revisit in F.8 only if profiling shows the u128%u64 divide matters
 * at 112-bit. Loop structure unchanged from the original.
 */
void bigint_scatter(const BigInt *a, uint64_t *coeffs, int n_coeffs, int pidx)
{
    const uint64_t p = PRIMES[pidx];
    int i;
    for (i = 0; i < a->n_limbs && i < n_coeffs; i++)
        coeffs[i] = (uint64_t)((__uint128_t)a->limbs[i] % p);
    for (; i < n_coeffs; i++)
        coeffs[i] = 0;
}

/* ─── NTT transposed scatter (I15) ──────────────────────────────────────── */
/*
 * Write coefficients in transposed order: coeffs[c*M + r] = limbs[r*M + c] % p.
 * This pre-applies the first transpose_sq of the Bailey 4-step so the GPU
 * forward NTT can begin directly at F1, saving one transpose kernel launch.
 * N = M*M must equal the NTT size (even log_n > 10 only).
 */
void bigint_scatter_t(const BigInt *a, uint64_t *coeffs, int M, int pidx)
{
    const uint64_t p = PRIMES[pidx];
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < M; c++) {
            int src = r * M + c;
            uint64_t v = (src < a->n_limbs)
                       ? (uint64_t)((__uint128_t)a->limbs[src] % p)
                       : 0;
            coeffs[c * M + r] = v;
        }
    }
}

/* ─── NTT gather ─────────────────────────────────────────────────────────── */
/*
 * Reconstruct BigInt from n_coeffs Garner U256 polynomial coefficients:
 *
 *   value = sum_{i=0}^{n_coeffs-1} coeffs[i] * 2^(LIMB_BITS*i)
 *
 * Each coeffs[i] is a U256 (four 64-bit words, little-endian, < 2^255).
 * LIMB_BITS*i is exactly limb i's bit offset, so coeff i is limb-
 * ALIGNED — but its re-chunking is width-specific: at LB=64 the 4
 * words ARE 4 limbs; at LB=112 a 256-bit value becomes three 112-bit
 * limbs. Inverse of bigint_scatter; defines the polynomial base
 * (PHASE_F_DESIGN.md F.6 — a genuine rewrite, hence explicit per-width
 * paths rather than a fragile generic bit-shifter). O(n_coeffs).
 *
 * a must have cap >= n_coeffs + 4 to absorb carry overflow.
 */
void bigint_gather(BigInt *a, const U256 *coeffs, int n_coeffs)
{
    bigint_resize(a, n_coeffs + 4);
    bigint_zero(a);
    a->n_limbs = n_coeffs + 4;

#if LIMB_BITS == 64
    /* Original 64-bit path, verbatim: each word is one limb; coeff i
     * spans limbs [i, i+4). Kept identical so F.6 is a no-op at LB=64. */
    for (int i = 0; i < n_coeffs; i++) {
        uint64_t carry = 0;
        for (int w = 0; w < 4; w++) {
            int pos = i + w;
            if (pos >= a->cap) break;
            unsigned __int128 s = (unsigned __int128)a->limbs[pos]
                                + coeffs[i].w[w] + carry;
            a->limbs[pos] = (uint64_t)s;
            carry = (uint64_t)(s >> 64);
        }
        for (int j = i + 4; carry && j < a->cap; j++) {
            unsigned __int128 s = (unsigned __int128)a->limbs[j] + carry;
            a->limbs[j] = (uint64_t)s;
            carry = (uint64_t)(s >> 64);
        }
    }
#elif LIMB_BITS == 112
    /* 256-bit V = w0 | w1<<64 | w2<<128 | w3<<192 re-chunked into three
     * 112-bit limbs, added at the limb-aligned offset i:
     *   L0 = bits[0,112)   = w0 | (w1 & (2^48-1))<<64
     *   L1 = bits[112,224) = w1>>48 | w2<<16 | (w3 & (2^32-1))<<80
     *   L2 = bits[224,256) = w3>>32                       (<= 32 bits)
     */
    for (int i = 0; i < n_coeffs; i++) {
        unsigned __int128 w0 = coeffs[i].w[0];
        unsigned __int128 w1 = coeffs[i].w[1];
        unsigned __int128 w2 = coeffs[i].w[2];
        unsigned __int128 w3 = coeffs[i].w[3];
        unsigned __int128 part[3];
        part[0] =  w0 | ((w1 & (((unsigned __int128)1 << 48) - 1)) << 64);
        part[1] = (w1 >> 48) | (w2 << 16)
                | ((w3 & (((unsigned __int128)1 << 32) - 1)) << 80);
        part[2] =  w3 >> 32;
        unsigned __int128 carry = 0;
        for (int t = 0; t < 3; t++) {
            int pos = i + t;
            if (pos >= a->cap) break;
            unsigned __int128 s = (unsigned __int128)a->limbs[pos]
                                + part[t] + carry;
            a->limbs[pos] = (limb_t)(s & LIMB_MASK);
            carry = s >> LIMB_BITS;
        }
        for (int j = i + 3; carry && j < a->cap; j++) {
            unsigned __int128 s = (unsigned __int128)a->limbs[j] + carry;
            a->limbs[j] = (limb_t)(s & LIMB_MASK);
            carry = s >> LIMB_BITS;
        }
    }
#else
#error "bigint_gather: unsupported LIMB_BITS (expected 64 or 112)"
#endif
    trim(a);
}

/* ─── NTT transposed gather (I15) ───────────────────────────────────────── */
/*
 * Undo the I15 transposed-scatter layout: coefficient k = r*M + c is stored
 * at coeffs[c*M + r].  Reorder into a temporary linear array then delegate
 * to bigint_gather.  The reorder is O(n_coeffs) and cheaper than the GPU
 * transpose kernel it replaces.  M must equal 2^(log_n/2) from scatter time.
 */
void bigint_gather_t(BigInt *a, const U256 *coeffs, int n_coeffs, int M)
{
    U256 *tmp = (U256 *)malloc((size_t)n_coeffs * sizeof(U256));
    if (!tmp) { fprintf(stderr, "bigint_gather_t: OOM\n"); exit(1); }
    for (int k = 0; k < n_coeffs; k++) {
        int r = k / M, c = k % M;
        tmp[k] = coeffs[c * M + r];
    }
    bigint_gather(a, tmp, n_coeffs);
    free(tmp);
}

/* ─── Utility ────────────────────────────────────────────────────────────── */

int bigint_bits(const BigInt *a)
{
    if (a->n_limbs == 0) return 0;
    int top = a->n_limbs - 1;
    limb_t w = a->limbs[top];
    int b = LIMB_BITS;
    while (b > 0 && !(w >> (b - 1))) b--;
    return top * LIMB_BITS + b;
}

void bigint_print_hex(const BigInt *a, const char *label)
{
    if (label) fprintf(stderr, "%s: ", label);
    if (a->n_limbs == 0) { fprintf(stderr, "0\n"); return; }
    for (int i = a->n_limbs - 1; i >= 0; i--)
        fprintf(stderr, "%016llx", (unsigned long long)a->limbs[i]);
    fprintf(stderr, "\n");
}
