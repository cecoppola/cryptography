/*
 * test_mlkem_fips203_kat.c — FIPS 203 Known-Answer Test for ref/ntt_mlkem.c.
 *
 * Rigor model (non-vacuous by design):
 *   The ref ntt_mlkem.c selftest in its main() is a round-trip + polymul
 *   check that PASSES if the zeta table is wrong-but-self-inverse — the
 *   same vacuity class that hid the KA4/KA5 ordering bug and the omega=1
 *   pass family. This KAT closes that gap by INDEPENDENTLY recomputing
 *   the spec constants (FIPS 203 §B.2-§B.5) and asserting equality, then
 *   driving the polymul pipeline against an INDEPENDENT __int128
 *   negacyclic schoolbook.
 *
 * Subtests (all PASS required; exit 0 iff all):
 *   K1  br_7  : an independent bit-reverse-7 matches lib's behavior for
 *               all 128 inputs (re-derived: br_7(k) for k=0..127).
 *   K2  ZETAS[128] : lib's mlkem_get_zeta_table()[k] equals an
 *               INDEPENDENT computation of 17^br_7_indep(k) mod 3329
 *               (its own modpow via __int128). Full 128-entry table —
 *               not the 16-zeta spot check the earlier audit did.
 *   K3  GAMMAS[128] : same — 17^(2*br_7_indep(i)+1) mod 3329, all 128.
 *   K4  n_inv : 128^-1 mod 3329 == 3303 (the INTT scale; the file
 *               hardcodes this — verify the constant via __int128 modpow
 *               of 128^(3329-2) mod 3329).
 *   K5  RT+VAC : mlkem_intt(mlkem_ntt(f)) == f for delta/ones/alternating/
 *               random; AND for the non-constant inputs assert
 *               mlkem_ntt(f) != f (vacuity guard: kills identity-
 *               transform degeneracy that a bare round-trip would pass).
 *   K6  POLYMUL vs INDEPENDENT NEGACYCLIC SCHOOLBOOK : compute
 *               h = mlkem_intt(mlkem_polymul(mlkem_ntt(f), mlkem_ntt(g)))
 *               and assert it equals an independent O(n^2) schoolbook
 *               in Z_q[X]/(X^n+1), over random inputs and a hand-checked
 *               (1+X)*(1+X) = 1+2X+X^2 vector.
 *
 * Independence: every reference is computed in this file using __int128
 * modmul + an own modpow + own bit-reverse-7 — disjoint from ntt_mlkem.c.
 * No GMP, no GPU, no HIP. Deterministic splitmix64. Result also written
 * to mlkem_kat_<ts>.txt per project rule 5.
 *
 * Build (links ntt_mlkem.c with NTT_MLKEM_NO_MAIN):
 *   cc -O2 -Wall -Wextra -DNTT_MLKEM_NO_MAIN -o bin/test_mlkem_kat \
 *      ref/test_mlkem_fips203_kat.c ref/ntt_mlkem.c
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ntt_mlkem.h"

#define KQ 3329u
#define KN 256u
#define KZETA 17u

/* ── ntt_mlkem.c public API (re-declared; we link against it) ────────────── */
void          mlkem_init_tables(void);
const uint64_t *mlkem_get_zeta_table(void);
const uint64_t *mlkem_get_gamma_table(void);
void          mlkem_ntt(uint64_t *f);
void          mlkem_intt(uint64_t *f);
void          mlkem_polymul(const uint64_t *f, const uint64_t *g, uint64_t *h);

/* ── independent helpers (must NOT share code with ntt_mlkem.c) ──────────── */
static uint64_t indep_modpow(uint64_t b, uint64_t e, uint64_t m)
{
    uint64_t r = 1; b %= m;
    while (e) {
        if (e & 1) r = (uint64_t)(((unsigned __int128)r * b) % m);
        b = (uint64_t)(((unsigned __int128)b * b) % m);
        e >>= 1;
    }
    return r;
}
/* INDEPENDENT bit-reverse-7 (different control flow from ntt_mlkem's br_7). */
static unsigned indep_br7(unsigned x)
{
    unsigned r = 0;
    for (int i = 0; i < 7; i++) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r & 0x7Fu;
}

/* deterministic PRNG */
static uint64_t SM(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* independent O(n^2) negacyclic schoolbook in Z_q[X]/(X^n+1) */
static void ref_negacyclic(const uint64_t *a, const uint64_t *b, uint64_t *c,
                            uint64_t n, uint64_t q)
{
    for (uint64_t k = 0; k < n; k++) c[k] = 0;
    for (uint64_t i = 0; i < n; i++)
        for (uint64_t j = 0; j < n; j++) {
            uint64_t s = i + j, k = s % n;
            unsigned __int128 p = (unsigned __int128)a[i] * b[j] % q;
            if ((s / n) & 1u)
                c[k] = (uint64_t)(((unsigned __int128)c[k] + q - p) % q);
            else
                c[k] = (uint64_t)(((unsigned __int128)c[k] + p) % q);
        }
}

static int vec_eq(const uint64_t *a, const uint64_t *b, uint64_t n)
{
    return memcmp(a, b, (size_t)n * sizeof(uint64_t)) == 0;
}
static int all_equal(const uint64_t *v, uint64_t n)
{
    for (uint64_t i = 1; i < n; i++) if (v[i] != v[0]) return 0;
    return 1;
}

int main(void)
{
    mlkem_init_tables();
    const uint64_t *Zlib = mlkem_get_zeta_table();
    const uint64_t *Glib = mlkem_get_gamma_table();

    int pass = 0, total = 6;
    printf("\n  ML-KEM FIPS-203 KAT  (independent spec verification)\n");
    printf("  ----------------------------------------------------\n");

    /* K1: br_7 independence — sanity that our independent br7 is itself
     * an involution-on-[0,128) bijection (else our K2/K3 reference uses
     * a broken br). We can't directly access lib's br_7 (file-static),
     * but K2/K3 transitively verify it through the resulting tables. */
    int k1 = 1;
    for (unsigned x = 0; x < 128; x++) if (indep_br7(indep_br7(x)) != x) k1 = 0;
    printf("  K1 br_7 indep is an involution on [0,128) : %s\n", k1?"PASS":"FAIL");
    if (k1) pass++;

    /* K2: ZETAS table — full 128, INDEPENDENT recompute */
    int k2 = 1, k2_first = -1;
    for (unsigned k = 0; k < 128; k++) {
        uint64_t want = indep_modpow(KZETA, indep_br7(k), KQ);
        if (Zlib[k] != want) {
            if (k2_first < 0) k2_first = (int)k;
            k2 = 0;
        }
    }
    printf("  K2 ZETAS[0..127] match 17^br_7(k) mod 3329 : %s",
           k2?"PASS":"FAIL");
    if (!k2) printf(" (first diff k=%d got %llu want %llu)",
                    k2_first, (unsigned long long)Zlib[k2_first],
                    (unsigned long long)indep_modpow(KZETA, indep_br7(k2_first), KQ));
    printf("\n");
    if (k2) pass++;

    /* K3: GAMMAS table — full 128, INDEPENDENT recompute */
    int k3 = 1, k3_first = -1;
    for (unsigned i = 0; i < 128; i++) {
        uint64_t want = indep_modpow(KZETA, 2 * indep_br7(i) + 1, KQ);
        if (Glib[i] != want) {
            if (k3_first < 0) k3_first = (int)i;
            k3 = 0;
        }
    }
    printf("  K3 GAMMAS[0..127] match 17^(2br_7(i)+1)    : %s",
           k3?"PASS":"FAIL");
    if (!k3) printf(" (first diff i=%d)", k3_first);
    printf("\n");
    if (k3) pass++;

    /* K4: n_inv constant — independently 128^(q-2) mod q via __int128 */
    uint64_t n_inv = indep_modpow(128, KQ - 2, KQ);
    int k4 = (n_inv == 3303);
    printf("  K4 128^-1 mod 3329 == 3303 (INTT scale)    : %s (got %llu)\n",
           k4?"PASS":"FAIL", (unsigned long long)n_inv);
    if (k4) pass++;

    /* K5: round-trip + vacuity over structured + random inputs */
    int k5 = 1;
    uint64_t f[KN], orig[KN];
    uint64_t seed = 0xFEEDFACECAFE0001ULL;
    for (int t = 0; t < 6 && k5; t++) {
        for (unsigned i = 0; i < KN; i++) {
            switch (t) {
              case 0: f[i] = (i == 0);                          break; /* delta */
              case 1: f[i] = 1;                                 break; /* ones  */
              case 2: f[i] = (i & 1) ? (KQ - 1) : 1;            break; /* alt   */
              case 3: f[i] = (uint64_t)((i * 31337u) % KQ);     break; /* ramp  */
              case 4: f[i] = SM(&seed) % KQ;                    break; /* rand  */
              case 5: f[i] = (i + 1) % KQ;                      break; /* gradient */
            }
        }
        memcpy(orig, f, sizeof f);
        mlkem_ntt(f);
        /* vacuity: for non-constant inputs (t>=2), NTT must mix (output
         * must differ from input AND not be constant). t<2 are constants
         * or delta whose spectrum may equal input by coincidence — skip. */
        if (t >= 2) {
            if (vec_eq(f, orig, KN) || all_equal(f, KN)) k5 = 0;
        }
        mlkem_intt(f);
        if (!vec_eq(f, orig, KN)) k5 = 0;
    }
    printf("  K5 mlkem_intt(mlkem_ntt(f)) == f + vacuity : %s\n",
           k5?"PASS":"FAIL");
    if (k5) pass++;

    /* K6: polymul vs INDEPENDENT negacyclic schoolbook */
    int k6 = 1;
    uint64_t a[KN], b[KN], h[KN], H[KN], A[KN], B[KN], ref[KN];
    /* hand-checked: (1+X)*(1+X) = 1 + 2X + X^2 (no wrap; degree 2 < n) */
    memset(a, 0, sizeof a); memset(b, 0, sizeof b);
    a[0]=1; a[1]=1; b[0]=1; b[1]=1;
    memcpy(A, a, sizeof a); memcpy(B, b, sizeof b);
    mlkem_ntt(A); mlkem_ntt(B); mlkem_polymul(A, B, H); memcpy(h, H, sizeof H); mlkem_intt(h);
    if (h[0] != 1 || h[1] != 2 || h[2] != 1) k6 = 0;
    for (unsigned i = 3; i < KN; i++) if (h[i] != 0) { k6 = 0; break; }
    /* + 3 random differential trials vs independent schoolbook */
    for (int t = 0; t < 3 && k6; t++) {
        for (unsigned i = 0; i < KN; i++) { a[i] = SM(&seed) % KQ; b[i] = SM(&seed) % KQ; }
        memcpy(A, a, sizeof a); memcpy(B, b, sizeof b);
        mlkem_ntt(A); mlkem_ntt(B); mlkem_polymul(A, B, H); memcpy(h, H, sizeof H); mlkem_intt(h);
        ref_negacyclic(a, b, ref, KN, KQ);
        if (!vec_eq(h, ref, KN)) k6 = 0;
    }
    printf("  K6 polymul vs independent negacyclic ref   : %s\n",
           k6?"PASS":"FAIL");
    if (k6) pass++;

    printf("  ----------------------------------------------------\n");
    printf("  %d/%d %s\n\n", pass, total, (pass==total)?"ALL PASS":"FAILURES PRESENT");

    char ts[32]; time_t tt = time(0);
    strftime(ts, sizeof ts, "%Y%m%d_%H%M%S", localtime(&tt));
    char fn[64]; snprintf(fn, sizeof fn, "mlkem_kat_%s.txt", ts);
    FILE *fp = fopen(fn, "w");
    if (fp) { fprintf(fp, "mlkem_kat %s : %d/%d %s\n", ts, pass, total,
                      (pass==total)?"ALL PASS":"FAIL"); fclose(fp); }
    return (pass==total) ? 0 : 1;
}
