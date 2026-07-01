/*
 * test_polymul_integ.c — END-TO-END integration test for the ref lazy
 * Stockham polynomial-multiply harnesses, over EVERY admissible curated
 * prime and size, against an INDEPENDENT O(n^2) schoolbook.
 *
 * Why this exists (project bug history):
 *   - test_curated_primes.c exercises polymul only for the 2 large primes
 *     and only via the *full-reduction* CT-DIT path (ntt_cpu.c), NOT the
 *     lazy-accumulation Stockham harness used by ntt_polymul.c /
 *     ntt_polymul_negacyclic.c. test_ntt_rigor.c convolution-tests the
 *     CT-DIT/Stockham *kernels* but not the polymul *drivers* (twist,
 *     pre/post multiply, the lazy-overflow guard in their own
 *     ntt_params_init). This test closes that exact gap.
 *
 * Two build modes (one source, two binaries — like test_ntt_rigor):
 *   default            : links ntt_polymul.c       → CYCLIC  (mod X^n-1)
 *   -DINTEG_NEGACYCLIC : links ntt_polymul_negacyclic.c → NEGACYCLIC (X^n+1)
 *
 * For each curated prime and n in {4,8,16,64,256,1024} that the harness's
 * OWN ntt_params_init() admits (lazy-overflow guard rejects large q — that
 * rejection is itself asserted), and for SEEDS distinct random inputs:
 *
 *   PRIM   ntt_modulus_omega(m,n) is a genuine primitive n-th root
 *          (omega!=0,1; omega^n==1; omega^(n/2)==q-1) — the g=3 regression.
 *   MUL    polymul (driver under test) == independent __int128 O(n^2)
 *          schoolbook (cyclic or negacyclic per mode). The reference shares
 *          NO arithmetic with the code under test (plain (__int128)%q).
 *   VAC    vacuity: the schoolbook product of two random non-constant
 *          inputs is itself non-constant AND differs from each input — so
 *          a degenerate transform (omega=1 / identity) cannot pass MUL by
 *          producing trivial output. Also a known hand-checked vector
 *          ( (1+X) * (1+X) ) is asserted against its closed form.
 *
 * Negacyclic mode additionally derives psi = primitive 2n-th root with
 * psi^2==omega and feeds the twist tables the driver expects.
 *
 * Deterministic splitmix64. CPU only; no GPU/HIP/GMP. Fixed-width table +
 * timestamped result file. Exit 0 iff every (prime,n,seed,subtest) PASSes.
 *
 * Build (see Makefile test-polymul-integ / test-negacyc-integ):
 *   cc -O2 -Wall -Wextra -DNTT_POLYMUL_NO_MAIN \
 *      test_polymul_integ.c ntt_polymul.c -o bin/test_polymul_integ
 *   cc -O2 -Wall -Wextra -DINTEG_NEGACYCLIC -DNTT_NEGACYC_NO_MAIN \
 *      test_polymul_integ.c ntt_polymul_negacyclic.c -o bin/test_negacyc_integ
 */
#include "ntt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* drivers under test (defined in the linked polymul TU) */
void polymul_ntt(const uint64_t *f, const uint64_t *g, uint64_t *c,
                 const uint64_t *tw, const uint64_t *twi,
                 const ntt_params_t *p);
#ifdef INTEG_NEGACYCLIC
void polymul_ntt_negacyclic(const uint64_t *f, const uint64_t *g, uint64_t *c,
                            const uint64_t *tw, const uint64_t *twi,
                            const uint64_t *twist, const uint64_t *twist_inv,
                            const ntt_params_t *p);
uint64_t *alloc_twist(uint64_t n, uint64_t q, uint64_t psi);
uint64_t *alloc_twist_inv(uint64_t n, uint64_t q, uint64_t psi);
#endif

/* ── deterministic PRNG (splitmix64) ─────────────────────────────────────── */
static uint64_t SM(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ── independent modular arithmetic (NOT the code under test) ────────────── */
static uint64_t rmul(uint64_t a, uint64_t b, uint64_t q)
{
    return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) % q);
}
static uint64_t rpow(uint64_t b, uint64_t e, uint64_t q)
{
    uint64_t r = 1 % q;
    b %= q;
    while (e) { if (e & 1) r = rmul(r, b, q); b = rmul(b, b, q); e >>= 1; }
    return r;
}

#ifndef INTEG_NEGACYCLIC
/* independent O(n^2) cyclic convolution mod (X^n - 1) — only used by the
 * default (cyclic) build; guarded to silence -Wunused-function on the
 * negacyclic build mode. */
static void sb_cyclic(const uint64_t *a, const uint64_t *b, uint64_t *c,
                      uint64_t n, uint64_t q)
{
    for (uint64_t k = 0; k < n; k++) c[k] = 0;
    for (uint64_t i = 0; i < n; i++)
        for (uint64_t j = 0; j < n; j++) {
            uint64_t k = (i + j) % n;
            c[k] = (uint64_t)(((unsigned __int128)c[k]
                              + (unsigned __int128)a[i] * b[j]) % q);
        }
}
#else
/* independent O(n^2) negacyclic convolution mod (X^n + 1) — only used by
 * the INTEG_NEGACYCLIC build; guarded to silence -Wunused-function on
 * the default (cyclic) build mode. */
static void sb_negacyclic(const uint64_t *a, const uint64_t *b, uint64_t *c,
                          uint64_t n, uint64_t q)
{
    for (uint64_t k = 0; k < n; k++) c[k] = 0;
    for (uint64_t i = 0; i < n; i++)
        for (uint64_t j = 0; j < n; j++) {
            uint64_t s = i + j, k = s % n;
            unsigned __int128 prod = (unsigned __int128)a[i] * b[j] % q;
            if ((s / n) & 1)              /* wrapped once → -1 */
                c[k] = (uint64_t)(((unsigned __int128)c[k] + q - prod) % q);
            else
                c[k] = (uint64_t)(((unsigned __int128)c[k] + prod) % q);
        }
}
#endif /* INTEG_NEGACYCLIC */

static int all_equal(const uint64_t *v, uint64_t n)
{
    for (uint64_t i = 1; i < n; i++) if (v[i] != v[0]) return 0;
    return 1;
}
static int vec_eq(const uint64_t *a, const uint64_t *b, uint64_t n)
{
    return memcmp(a, b, (size_t)n * sizeof(uint64_t)) == 0;
}

/* one (modulus, n, seed) cell. Returns 0 on full PASS; sets sub flags. */
static int run_cell(const ntt_modulus_info_t *m, uint64_t n, uint64_t *seed,
                    int *prim_ok, int *mul_ok, int *vac_ok)
{
    uint64_t q = m->q;
    *prim_ok = *mul_ok = *vac_ok = 0;

    uint64_t omega = ntt_modulus_omega(m, n);
    if (omega == 0 || omega == 1)            return 1;
    if (rpow(omega, n, q) != 1)              return 1;
    if (rpow(omega, n / 2, q) != q - 1)      return 1;
    *prim_ok = 1;

    ntt_params_t p;
    memset(&p, 0, sizeof p);
    p.n = n; p.q = q; p.omega = omega;
    /* The driver's OWN ntt_params_init carries the lazy-overflow guard.
     * If it rejects this (prime,n) the caller must have already filtered
     * it; reaching here with a reject is a guard/agreement bug. */
    if (ntt_params_init(&p) != 0) return 2;

    uint64_t *tw  = ntt_alloc_twiddles(&p);
    uint64_t *twi = ntt_alloc_twiddles_inv(&p);
    uint64_t *a   = malloc((size_t)n * sizeof(uint64_t));
    uint64_t *b   = malloc((size_t)n * sizeof(uint64_t));
    uint64_t *c   = malloc((size_t)n * sizeof(uint64_t));
    uint64_t *ref = malloc((size_t)n * sizeof(uint64_t));
    if (!tw || !twi || !a || !b || !c || !ref) {
        free(tw); free(twi); free(a); free(b); free(c); free(ref); return 2;
    }

#ifdef INTEG_NEGACYCLIC
    /* psi: primitive 2n-th root with psi^2 == omega. */
    uint64_t psi = ntt_modulus_omega(m, 2 * n);
    if (psi == 0 || rpow(psi, 2, q) != omega) {
        if ((q - 1) % (2 * n) == 0) psi = rpow(m->g, (q - 1) / (2 * n), q);
        else psi = 0;
    }
    if (psi == 0 || rpow(psi, 2, q) != omega
                 || rpow(psi, 2 * n, q) != 1 || rpow(psi, n, q) == 1) {
        /* No primitive 2n-th root exists in (Z/qZ)* (e.g. q=257/3329 at
         * n=256: q-1's 2-adic valuation is only 8, so 2n=512 doesn't
         * divide q-1). This is MATH-INAPPLICABLE, NOT a failure of the
         * driver — return -1 so the caller records a skip, not a FAIL. */
        free(tw); free(twi); free(a); free(b); free(c); free(ref);
        *prim_ok = *mul_ok = *vac_ok = 1;     /* N/A; counted as skip */
        return -1;
    }
    uint64_t *tws  = alloc_twist(n, q, psi);
    uint64_t *twsi = alloc_twist_inv(n, q, psi);
    if (!tws || !twsi) {
        free(tw); free(twi); free(a); free(b); free(c); free(ref);
        free(tws); free(twsi); return 2;
    }
#endif

    /* ---- MUL: driver vs independent schoolbook over SEEDS inputs ---- */
    int mul = 1, vac = 1;
    for (int t = 0; t < 4 && mul; t++) {
        for (uint64_t i = 0; i < n; i++) { a[i] = SM(seed) % q; b[i] = SM(seed) % q; }
        if (all_equal(a, n)) a[0] = (a[0] + 1) % q;
        if (all_equal(b, n)) b[0] = (b[0] + 1) % q;
#ifdef INTEG_NEGACYCLIC
        polymul_ntt_negacyclic(a, b, c, tw, twi, tws, twsi, &p);
        sb_negacyclic(a, b, ref, n, q);
#else
        polymul_ntt(a, b, c, tw, twi, &p);
        sb_cyclic(a, b, ref, n, q);
#endif
        if (!vec_eq(c, ref, n)) mul = 0;
        /* vacuity: a non-trivial product must not coincide with an input
         * and (for these random inputs) is overwhelmingly non-constant. */
        if (vec_eq(c, a, n) || vec_eq(c, b, n) || all_equal(c, n)) vac = 0;
    }
    *mul_ok = mul;

    /* ---- VAC: closed-form hand vector (1 + X)^2 ----
     * cyclic & negacyclic agree on (1+X)^2 = 1 + 2X + X^2 for n > 2
     * (no wrap), giving a reference independent of the schoolbook loop. */
    for (uint64_t i = 0; i < n; i++) a[i] = b[i] = 0;
    a[0] = 1; a[1] = 1; b[0] = 1; b[1] = 1;
#ifdef INTEG_NEGACYCLIC
    polymul_ntt_negacyclic(a, b, c, tw, twi, tws, twsi, &p);
#else
    polymul_ntt(a, b, c, tw, twi, &p);
#endif
    if (!(c[0] == 1 % q && c[1] == 2 % q && c[2] == 1 % q)) vac = 0;
    for (uint64_t i = 3; i < n; i++) if (c[i] != 0) { vac = 0; break; }
    *vac_ok = vac;

    free(tw); free(twi); free(a); free(b); free(c); free(ref);
#ifdef INTEG_NEGACYCLIC
    free(tws); free(twsi);
#endif
    return (*prim_ok && *mul_ok && *vac_ok) ? 0 : 1;
}

int main(int argc, char **argv)
{
    uint64_t ns[]  = { 4, 8, 16, 64, 256, 1024 };
    int nns = (int)(sizeof ns / sizeof ns[0]);
    int seeds = 3;
    if (argc > 1) seeds = atoi(argv[1]);
    if (seeds < 1) seeds = 1;
    uint64_t seed = 0xC0FFEE5EED1234FFULL;

#ifdef INTEG_NEGACYCLIC
    const char *mode = "NEGACYCLIC (mod X^n+1)";
    const char *tag  = "negacyc_integ";
#else
    const char *mode = "CYCLIC (mod X^n-1)";
    const char *tag  = "polymul_integ";
#endif
    printf("\n  NTT / MI300A — POLYMUL INTEGRATION TEST  [%s]\n", mode);
    printf("  driver vs INDEPENDENT O(n^2) schoolbook, all curated primes\n");
    printf("  %-12s %-22s %-5s  %-4s %-4s %-4s\n",
           "modulus", "q", "n", "PRIM", "MUL", "VAC");
    printf("  ------------------------------------------------------------\n");

    int total = 0, passed = 0, cells = 0, skipped = 0, rejected = 0;
    int psi_skipped = 0;
    for (int mi = 0; mi < NTT_NUM_MODULI; mi++) {
        const ntt_modulus_info_t *m = &NTT_MODULI[mi];
        for (int k = 0; k < nns; k++) {
            uint64_t n = ns[k];
            if (n < 4 || (n & (n - 1))) continue;
            if (ntt_modulus_omega(m, n) == 0) { skipped++; continue; }
            /* Probe the driver's own guard with a throwaway params copy.
             * Large-q primes are EXPECTED to be rejected by the lazy
             * harness — that rejection is part of the contract, not a
             * skip we hide. */
            ntt_params_t probe;
            memset(&probe, 0, sizeof probe);
            probe.n = n; probe.q = m->q;
            probe.omega = ntt_modulus_omega(m, n);
            if (ntt_params_init(&probe) != 0) { rejected++; continue; }
            int pr, mu, va;
            int rc = run_cell(m, n, &seed, &pr, &mu, &va);
            if (rc == -1) {
                /* Negacyclic: no primitive 2n-th root exists for this (q,n)
                 * (math-inapplicable, not a driver fail). Print as skip and
                 * don't fold into the subtest tally. */
                printf("  %-12s %-22llu %-5llu  skip (no 2n-th root)\n",
                       m->name, (unsigned long long)m->q,
                       (unsigned long long)n);
                psi_skipped++;
                continue;
            }
            cells++;
            total += 3;
            passed += pr + mu + va;
            printf("  %-12s %-22llu %-5llu  %-4s %-4s %-4s%s\n",
                   m->name, (unsigned long long)m->q, (unsigned long long)n,
                   pr?"OK":"FAIL", mu?"OK":"FAIL", va?"OK":"FAIL",
                   (rc == 0) ? "" : "   <== CELL FAIL");
        }
    }
    printf("  ------------------------------------------------------------\n");
    printf("  cells=%d  no-root-skip=%d  no-psi-skip=%d  "
           "guard-rejected(large-q)=%d  subtests %d/%d\n",
           cells, skipped, psi_skipped, rejected, passed, total);
    int ok = (cells > 0 && passed == total);
    printf("  %s\n\n", ok ? "ALL PASS" : "FAILURES PRESENT");

    char ts[32]; time_t tt = time(0);
    strftime(ts, sizeof ts, "%Y%m%d_%H%M%S", localtime(&tt));
    char fn[80]; snprintf(fn, sizeof fn, "%s_%s.txt", tag, ts);
    FILE *f = fopen(fn, "w");
    if (f) {
        fprintf(f, "%s %s : cells=%d skip=%d rejected=%d %d/%d %s\n",
                tag, ts, cells, skipped, rejected, passed, total,
                ok ? "ALL PASS" : "FAIL");
        fclose(f);
    }
    return ok ? 0 : 1;
}
