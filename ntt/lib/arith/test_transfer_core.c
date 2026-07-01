/*
 * test_transfer_core.c — Host oracle for the GPU scatter / gather core logic.
 *
 * The GPU kernels in transfer_kernels.hip are one-thread-per-output-element
 * wrappers around the host+device inline functions in transfer_core.h. This
 * test calls those SAME functions in a host loop (exactly reproducing what each
 * GPU thread computes) and checks the result against the established CPU
 * reference in bigint.c:
 *
 *   scatter:  scatter_value() over all output indices   == bigint_scatter[_t]
 *   gather :  gather_acc_limb() + gather_carry_normalize == bigint_gather[_t]
 *   gather (CLA): gather_acc_limb() + parallel carry-lookahead (cla_cell /
 *             cla_compose Hillis-Steele scan / cla_finalize) == bigint_gather[_t],
 *             and its scanned carries == the sequential carry recurrence.
 *   gather (CLA multi-level): the recursive blocked scan (GPU multi-block path,
 *             local scan + aggregate scan + exclusive-prefix apply) == the
 *             single-pass scan and the sequential normalization, at block sizes
 *             forcing deep recursion.
 *
 * Verifying the per-element cores on the host means the only thing left to
 * prove on MI300A is that the kernels launch and are fast — the arithmetic is
 * hardware-independent and is fully exercised here at both LIMB_BITS (64/112)
 * over randomized and adversarial (all-ones, zero-pad, single-coeff) inputs.
 *
 * Built dual-width by lib/Makefile (test-transfer-core), pure C (gcc), no GPU.
 *
 * Non-vacuity: every CHECK is a differential against an independent reference
 * over random inputs; a vacuous pass would require both implementations to be
 * wrong identically. Confirmed to FAIL when xpose_index or gather_part is
 * perturbed (manual perturbation during development).
 */

#include "../transfer_core.h"   /* cores + (via bigint.h) bigint_*, U256, PRIMES */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GRN "\033[1;32m"
#define RED "\033[1;31m"
#define CYN "\033[1;36m"
#define RST "\033[0m"

static int fails = 0;
static int checks = 0;
#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { fails++; printf("  " RED "FAIL" RST " "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

/* Deterministic xorshift PRNG — reproducible failures. */
static uint64_t rng_state = 0xC0FFEE123456789ULL;
static uint64_t rng64(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* Random limb in [0, 2^LIMB_BITS). */
static limb_t rand_limb(void)
{
    limb_t v = (limb_t)rng64();
#if LIMB_BITS == 112
    v |= ((limb_t)rng64()) << 64;
    v &= LIMB_MASK;
#endif
    return v;
}

static void fill_random_bigint(BigInt *a, int n_limbs)
{
    bigint_zero(a);
    for (int i = 0; i < n_limbs; i++) a->limbs[i] = rand_limb();
    a->n_limbs = n_limbs;
}

static void fill_random_u256(U256 *c, int all_ones)
{
    for (int w = 0; w < 4; w++) c->w[w] = all_ones ? UINT64_MAX : rng64();
}

/* ─── Parallel carry-lookahead (CLA) gather normalization (host model) ────────
 * Reproduces the GPU phase-2 CLA path: decompose each phase-1 partial sum into
 * a (generate, propagate) cell + pre-carry digit, run a Hillis-Steele inclusive
 * prefix scan over cla_compose (the exact O(log n) GPU pattern) to get every
 * thin carry-in, then finalize each limb in parallel. Cross-checks the scanned
 * carries against the sequential carry recurrence (*scan_ok) so a monoid/scan
 * bug is caught independently of the limb comparison. Returns the final thin
 * carry out of the top limb (0 on a correctly sized buffer). */
static uint64_t cla_normalize(const uint64_t *acc_lo, const uint64_t *acc_hi,
                              limb_t *out, int n_out, int *scan_ok)
{
    ClaPair *cell = (ClaPair *)malloc((size_t)n_out * sizeof(ClaPair));
    ClaPair *pref = (ClaPair *)malloc((size_t)n_out * sizeof(ClaPair));
    ClaPair *tmp  = (ClaPair *)malloc((size_t)n_out * sizeof(ClaPair));
    limb_t  *res  = (limb_t  *)malloc((size_t)n_out * sizeof(limb_t));

    for (int j = 0; j < n_out; j++) {                 /* decompose (parallel) */
        uint64_t g_prev = (j == 0) ? 0 : cla_high(acc_lo[j-1], acc_hi[j-1]);
        cell[j] = cla_cell(acc_lo[j], acc_hi[j], g_prev, &res[j]);
        pref[j] = cell[j];
    }
    for (int d = 1; d < n_out; d <<= 1) {             /* Hillis-Steele scan   */
        memcpy(tmp, pref, (size_t)n_out * sizeof(ClaPair));
        for (int j = d; j < n_out; j++) pref[j] = cla_compose(tmp[j-d], tmp[j]);
    }

    *scan_ok = 1;
    uint64_t cin_seq = 0;                             /* sequential reference */
    for (int j = 0; j < n_out; j++) {
        uint64_t cin_scan = (j == 0) ? 0 : pref[j-1].g;
        if (cin_scan != cin_seq) *scan_ok = 0;
        out[j] = cla_finalize(res[j], cin_scan);       /* finalize (parallel)  */
        cin_seq = cell[j].g | (cell[j].p & cin_seq);   /* carry out of limb j  */
    }
    free(cell); free(pref); free(tmp); free(res);
    return cin_seq;
}

/* ─── Multi-level blocked prefix scan (host model of the GPU's segmented scan) ─
 * The single-block LDS scan only covers n_out <= one block. At production sizes
 * the GPU scans in levels: (a) each block inclusive-scans its own cells in LDS
 * and emits its aggregate; (b) the aggregate array is scanned the same way
 * (recursively, until it fits one block); (c) each block composes the exclusive
 * prefix of all earlier blocks (from the LOWER side — the monoid is NOT
 * commutative) into its local results. This recursive model reproduces that at
 * any depth, so verifying it against the single-pass scan validates the scheme
 * at every scale. BLK is the block size (kept small in tests to force deep
 * recursion). */

/* Sequential inclusive scan — the ground-truth global prefix. */
static void scan_ref(const ClaPair *cell, ClaPair *pref, int n)
{
    if (n <= 0) return;
    pref[0] = cell[0];
    for (int j = 1; j < n; j++) pref[j] = cla_compose(pref[j-1], cell[j]);
}

/* Recursive blocked inclusive scan; result identical to scan_ref(cell,pref,n). */
static void blocked_scan(const ClaPair *cell, ClaPair *pref, int n, int BLK)
{
    if (n <= BLK) { scan_ref(cell, pref, n); return; }
    int B = (n + BLK - 1) / BLK;
    /* calloc: agg/aggpre are fully written below before use; zero-init only
     * silences a GCC -Wmaybe-uninitialized false positive across the recursion. */
    ClaPair *agg    = (ClaPair *)calloc((size_t)B, sizeof(ClaPair));
    ClaPair *aggpre = (ClaPair *)calloc((size_t)B, sizeof(ClaPair));
    for (int b = 0; b < B; b++) {                    /* (a) local block scans  */
        int s = b * BLK, e = s + BLK; if (e > n) e = n;
        scan_ref(cell + s, pref + s, e - s);
        agg[b] = pref[e - 1];                        /* block aggregate        */
    }
    blocked_scan(agg, aggpre, B, BLK);               /* (b) scan aggregates    */
    for (int b = 1; b < B; b++) {                    /* (c) apply excl. prefix */
        ClaPair bp = aggpre[b - 1];                  /* prefix of blocks [0,b) */
        int s = b * BLK, e = s + BLK; if (e > n) e = n;
        for (int j = s; j < e; j++) pref[j] = cla_compose(bp, pref[j]);
    }
    free(agg); free(aggpre);
}

/* CLA normalize using the multi-level blocked scan (GPU multi-block path). Must
 * yield the same limbs as cla_normalize / gather_carry_normalize. Cross-checks
 * the blocked-scan carries against the sequential recurrence (*scan_ok). */
static uint64_t cla_normalize_blocked(const uint64_t *acc_lo, const uint64_t *acc_hi,
                                      limb_t *out, int n_out, int BLK, int *scan_ok)
{
    ClaPair *cell = (ClaPair *)malloc((size_t)n_out * sizeof(ClaPair));
    ClaPair *pref = (ClaPair *)malloc((size_t)n_out * sizeof(ClaPair));
    limb_t  *res  = (limb_t  *)malloc((size_t)n_out * sizeof(limb_t));

    for (int j = 0; j < n_out; j++) {
        uint64_t g_prev = (j == 0) ? 0 : cla_high(acc_lo[j-1], acc_hi[j-1]);
        cell[j] = cla_cell(acc_lo[j], acc_hi[j], g_prev, &res[j]);
    }
    blocked_scan(cell, pref, n_out, BLK);

    *scan_ok = 1;
    uint64_t cin_seq = 0;
    for (int j = 0; j < n_out; j++) {
        uint64_t cin_scan = (j == 0) ? 0 : pref[j-1].g;
        if (cin_scan != cin_seq) *scan_ok = 0;
        out[j] = cla_finalize(res[j], cin_scan);
        cin_seq = cell[j].g | (cell[j].p & cin_seq);
    }
    free(cell); free(pref); free(res);
    return cin_seq;
}

/* Verify the multi-level blocked scan reproduces `expect` (a previously-
 * validated normalization) at several block sizes, exercising deep recursion. */
static void check_blocked_equals(const uint64_t *acc_lo, const uint64_t *acc_hi,
                                 const limb_t *expect, uint64_t expect_carry,
                                 int n_out, const char *label)
{
    limb_t *b = (limb_t *)malloc((size_t)n_out * sizeof(limb_t));
    int blks[] = { 2, 4, 16 };
    for (size_t bi = 0; bi < sizeof(blks)/sizeof(blks[0]); bi++) {
        int BLK = blks[bi], scan_ok = 1;
        uint64_t c = cla_normalize_blocked(acc_lo, acc_hi, b, n_out, BLK, &scan_ok);
        CHECK(scan_ok, "%s blocked BLK=%d: scan != recurrence", label, BLK);
        CHECK(c == expect_carry, "%s blocked BLK=%d: carry %llu != %llu",
              label, BLK, (unsigned long long)c, (unsigned long long)expect_carry);
        int ok = 1, bad = -1;
        for (int j = 0; j < n_out; j++) if (b[j] != expect[j]) { ok = 0; bad = j; break; }
        CHECK(ok, "%s blocked BLK=%d: limb %d mismatch", label, BLK, bad);
    }
    free(b);
}

/* Pack a desired (base-B residue `low`, local-high `ghi`) pair for limb j into
 * the split 128-bit partial-sum arrays, respecting the phase-1 invariant
 * acc < 4B (ghi in {0,1,2,3}). Lets the CLA tests build exact carry patterns. */
static void set_acc(uint64_t *acc_lo, uint64_t *acc_hi, int j,
                    limb_t low, uint64_t ghi)
{
#if LIMB_BITS == 64
    acc_lo[j] = (uint64_t)low;
    acc_hi[j] = ghi;
#else /* 112 */
    acc_lo[j] = (uint64_t)low;                          /* bits 0..63          */
    acc_hi[j] = (uint64_t)(low >> 64) | (ghi << 48);    /* bits 64..111 | ghi  */
#endif
}

/* Compare the CLA normalizer against the sequential normalizer on an arbitrary
 * (phase-1-valid) partial-sum array. The sequential gather_carry_normalize is
 * the definition of correct carry normalization; the two use entirely different
 * carry algorithms (ripple vs generate/propagate prefix), so agreement on
 * adversarial carry chains is strong evidence. */
static void check_cla_acc(uint64_t *acc_lo, uint64_t *acc_hi, int n_out,
                          const char *label)
{
    limb_t *seq = (limb_t *)malloc((size_t)n_out * sizeof(limb_t));
    limb_t *cla = (limb_t *)malloc((size_t)n_out * sizeof(limb_t));
    uint64_t seq_carry = gather_carry_normalize(acc_lo, acc_hi, seq, n_out);
    int scan_ok = 1;
    uint64_t cla_carry = cla_normalize(acc_lo, acc_hi, cla, n_out, &scan_ok);
    CHECK(scan_ok, "cla-direct %s: scan != recurrence", label);
    CHECK(seq_carry == cla_carry, "cla-direct %s: final carry seq=%llu cla=%llu",
          label, (unsigned long long)seq_carry, (unsigned long long)cla_carry);
    int ok = 1, bad = -1;
    for (int j = 0; j < n_out; j++) if (seq[j] != cla[j]) { ok = 0; bad = j; break; }
    CHECK(ok, "cla-direct %s: limb %d seq != cla", label, bad);
    check_blocked_equals(acc_lo, acc_hi, seq, seq_carry, n_out, label);
    free(seq); free(cla);
}

/* Adversarial CLA carry-chain coverage: deliberately builds the long-carry-
 * propagation cases that random U256 inputs never hit (res==B-1 runs receiving
 * a generated carry). This is the scenario carry-lookahead exists for. */
static void test_cla_direct(void)
{
    int n = 64;
    uint64_t *lo = (uint64_t *)calloc((size_t)n, sizeof(uint64_t));
    uint64_t *hi = (uint64_t *)calloc((size_t)n, sizeof(uint64_t));

    /* Max propagation: a generated carry ripples through a long run of B-1
     * residues. limb0 feeds ghi=1 into limb1 (low=B-1 -> t=B -> generate);
     * limbs 2..n-2 are B-1 (propagate); limb n-1 absorbs. */
    for (int j = 0; j < n; j++) set_acc(lo, hi, j, (limb_t)0, 0);
    set_acc(lo, hi, 0, (limb_t)0,         1);
    set_acc(lo, hi, 1, (limb_t)LIMB_MASK, 0);
    for (int j = 2; j < n - 1; j++) set_acc(lo, hi, j, (limb_t)LIMB_MASK, 0);
    check_cla_acc(lo, hi, n, "max-prop");

    /* Two independent generate+propagate segments. */
    for (int j = 0; j < n; j++) set_acc(lo, hi, j, (limb_t)0, 0);
    set_acc(lo, hi, 4,  (limb_t)0,         1);
    set_acc(lo, hi, 5,  (limb_t)LIMB_MASK, 0);
    set_acc(lo, hi, 6,  (limb_t)LIMB_MASK, 0);
    set_acc(lo, hi, 7,  (limb_t)LIMB_MASK, 0);
    set_acc(lo, hi, 20, (limb_t)0,         1);
    set_acc(lo, hi, 21, (limb_t)LIMB_MASK, 0);
    set_acc(lo, hi, 22, (limb_t)LIMB_MASK, 0);
    check_cla_acc(lo, hi, n, "two-segment");

    /* Every limb ghi=3 + B-1 residue: maximal local-high folding with full
     * propagate run (stresses the t = B-1 + 3 boundary on every limb). */
    for (int j = 0; j < n - 1; j++) set_acc(lo, hi, j, (limb_t)LIMB_MASK, 3);
    set_acc(lo, hi, n - 1, (limb_t)0, 0);
    check_cla_acc(lo, hi, n, "ghi3-allones");
    free(lo); free(hi);

    /* Randomized biased soak: low frequently B-1 (propagate) with random ghi
     * and occasional generators — many carry chains of varied length. */
    for (int t = 0; t < 300; t++) {
        int m = 8 + (int)(rng64() % 200);
        uint64_t *al = (uint64_t *)malloc((size_t)m * sizeof(uint64_t));
        uint64_t *ah = (uint64_t *)malloc((size_t)m * sizeof(uint64_t));
        for (int j = 0; j < m; j++) {
            uint64_t r = rng64() % 10, ghi = rng64() % 4;
            limb_t low = (r < 6) ? (limb_t)LIMB_MASK
                       : (r < 8) ? (limb_t)0
                                 : rand_limb();
            set_acc(al, ah, j, low, ghi);
        }
        for (int j = m - 4 >= 0 ? m - 4 : 0; j < m; j++) set_acc(al, ah, j, (limb_t)0, 0);
        check_cla_acc(al, ah, m, "biased-soak");
        free(al); free(ah);
    }
}

/* Abstract multi-level scan check: blocked_scan must equal the single-pass
 * scan_ref over arbitrary monoid inputs, at block sizes small enough to force
 * deep recursion (BLK=2, n=1000 is ~10 levels). Independent of the gather data
 * path — isolates the scan orchestration (block boundaries, exclusive-prefix
 * apply order, multi-level composition). */
static void test_blocked_scan(void)
{
    int ns[]   = { 1, 2, 5, 8, 17, 64, 100, 257, 500, 1000 };
    int blks[] = { 2, 3, 4, 7, 16, 64 };
    for (size_t ni = 0; ni < sizeof(ns)/sizeof(ns[0]); ni++) {
        int n = ns[ni];
        ClaPair *cell = (ClaPair *)malloc((size_t)n * sizeof(ClaPair));
        ClaPair *ref  = (ClaPair *)malloc((size_t)n * sizeof(ClaPair));
        ClaPair *got  = (ClaPair *)malloc((size_t)n * sizeof(ClaPair));
        for (int t = 0; t < 20; t++) {
            for (int j = 0; j < n; j++) {
                cell[j].g = (unsigned char)(rng64() & 1);
                cell[j].p = (unsigned char)(rng64() & 1);
            }
            scan_ref(cell, ref, n);
            for (size_t bi = 0; bi < sizeof(blks)/sizeof(blks[0]); bi++) {
                int BLK = blks[bi];
                blocked_scan(cell, got, n, BLK);
                int ok = 1, bad = -1;
                for (int j = 0; j < n; j++)
                    if (got[j].g != ref[j].g || got[j].p != ref[j].p) { ok = 0; bad = j; break; }
                CHECK(ok, "blocked_scan n=%d BLK=%d: differs at %d", n, BLK, bad);
            }
        }
        free(cell); free(ref); free(got);
    }
}

/* ─── Scatter: cores vs bigint_scatter (linear) / bigint_scatter_t (I15) ──────
 * Reproduces scatter_kernel for every output index and compares the whole
 * coefficient array to the reference. transposed selects the layout. */
static void test_scatter(int n_limbs, int n_coeffs, int M, int transposed)
{
    BigInt a = bigint_alloc(n_limbs + 4);
    fill_random_bigint(&a, n_limbs);

    uint64_t *ref  = (uint64_t *)malloc((size_t)n_coeffs * sizeof(uint64_t));
    uint64_t *mine = (uint64_t *)malloc((size_t)n_coeffs * sizeof(uint64_t));

    for (int pidx = 0; pidx < 4; pidx++) {
        if (transposed) bigint_scatter_t(&a, ref, M, pidx);
        else            bigint_scatter(&a, ref, n_coeffs, pidx);

        for (int j = 0; j < n_coeffs; j++) {           /* one GPU thread / coeff */
            int src = xpose_index(j, M, transposed);
            mine[j] = scatter_value(a.limbs, a.n_limbs, src, pidx);
        }
        CHECK(memcmp(ref, mine, (size_t)n_coeffs * sizeof(uint64_t)) == 0,
              "scatter%s nlimbs=%d ncoeffs=%d M=%d p%d",
              transposed ? "_t" : "", n_limbs, n_coeffs, M, pidx);
    }
    free(ref); free(mine); bigint_free(&a);
}

/* ─── Gather: cores vs bigint_gather (linear) / bigint_gather_t (I15) ─────────
 * Reproduces gather_accum_kernel (phase 1) + gather_carry_kernel (phase 2) and
 * compares the normalized limb array to the reference BigInt. n_store is the
 * coefficient-array length (= N for transposed so xpose covers all slots). */
static void test_gather(int n_coeffs, int n_store, int M, int transposed,
                        int all_ones)
{
    U256 *coeffs = (U256 *)malloc((size_t)n_store * sizeof(U256));
    for (int i = 0; i < n_store; i++) fill_random_u256(&coeffs[i], all_ones);

    /* Reference. */
    BigInt ref = bigint_alloc(n_coeffs + 8);
    if (transposed) bigint_gather_t(&ref, coeffs, n_coeffs, M);
    else            bigint_gather(&ref, coeffs, n_coeffs);

    /* Emulated GPU path. */
    int n_out = n_coeffs + 4;
    uint64_t *acc_lo = (uint64_t *)malloc((size_t)n_out * sizeof(uint64_t));
    uint64_t *acc_hi = (uint64_t *)malloc((size_t)n_out * sizeof(uint64_t));
    limb_t   *mine   = (limb_t   *)malloc((size_t)n_out * sizeof(limb_t));

    for (int j = 0; j < n_out; j++) {                  /* phase 1: thread / limb */
        __uint128_t acc = gather_acc_limb(coeffs, n_coeffs, j, M, transposed);
        acc_lo[j] = (uint64_t)acc;
        acc_hi[j] = (uint64_t)(acc >> 64);
    }
    uint64_t carry = gather_carry_normalize(acc_lo, acc_hi, mine, n_out); /* ph2 */

    CHECK(carry == 0, "gather%s ncoeffs=%d M=%d ones=%d: nonzero top carry",
          transposed ? "_t" : "", n_coeffs, M, all_ones);

    int ok = 1, bad = -1;
    for (int j = 0; j < n_out; j++) {
        limb_t expect = (j < ref.n_limbs) ? ref.limbs[j] : (limb_t)0;
        if (mine[j] != expect) { ok = 0; bad = j; break; }
    }
    CHECK(ok, "gather%s ncoeffs=%d M=%d ones=%d: limb %d mismatch",
          transposed ? "_t" : "", n_coeffs, M, all_ones, bad);

    /* Parallel carry-lookahead path: must equal both the reference and the
     * sequential normalizer, and its scan must match the carry recurrence. */
    limb_t *cla = (limb_t *)malloc((size_t)n_out * sizeof(limb_t));
    int scan_ok = 1;
    uint64_t cla_carry = cla_normalize(acc_lo, acc_hi, cla, n_out, &scan_ok);
    CHECK(scan_ok, "gather%s ncoeffs=%d M=%d ones=%d: CLA scan != recurrence",
          transposed ? "_t" : "", n_coeffs, M, all_ones);
    CHECK(cla_carry == 0, "gather%s ncoeffs=%d M=%d ones=%d: CLA nonzero top carry",
          transposed ? "_t" : "", n_coeffs, M, all_ones);
    int ok2 = 1, bad2 = -1;
    for (int j = 0; j < n_out; j++) {
        limb_t expect = (j < ref.n_limbs) ? ref.limbs[j] : (limb_t)0;
        if (cla[j] != expect || cla[j] != mine[j]) { ok2 = 0; bad2 = j; break; }
    }
    CHECK(ok2, "gather%s ncoeffs=%d M=%d ones=%d: CLA limb %d mismatch",
          transposed ? "_t" : "", n_coeffs, M, all_ones, bad2);

    /* Multi-level blocked scan must reproduce the sequential normalization. */
    {
        char lbl[64];
        snprintf(lbl, sizeof(lbl), "gather%s nc=%d M=%d ones=%d",
                 transposed ? "_t" : "", n_coeffs, M, all_ones);
        check_blocked_equals(acc_lo, acc_hi, mine, 0, n_out, lbl);
    }

    free(coeffs); free(acc_lo); free(acc_hi); free(mine); free(cla);
    bigint_free(&ref);
}

int main(void)
{
    printf(CYN "=== transfer_core host oracle  (LIMB_BITS=%d, COEFF_LIMB_SPAN=%d) ==="
           RST "\n", LIMB_BITS, COEFF_LIMB_SPAN);

    /* ─ Scatter, linear path (log_n <= 10 / CT-DIT): assorted sizes ─ */
    printf("\n" CYN "--- scatter (linear) ---" RST "\n");
    int lin_nc[] = { 1, 2, 7, 64, 100, 1024 };
    for (size_t i = 0; i < sizeof(lin_nc)/sizeof(lin_nc[0]); i++) {
        int nc = lin_nc[i];
        test_scatter(/*n_limbs*/ nc / 2 + 1, nc, /*M*/ 0, /*transposed*/ 0); /* zero-pad */
        test_scatter(nc, nc, 0, 0);                                          /* exact   */
        test_scatter(nc + 5, nc, 0, 0);                                      /* truncate*/
    }

    /* ─ Scatter, I15 transposed path (even log_n > 10): N = M*M ─ */
    printf(CYN "--- scatter_t (I15 transposed) ---" RST "\n");
    int Ms[] = { 2, 4, 8, 16, 32 };
    for (size_t i = 0; i < sizeof(Ms)/sizeof(Ms[0]); i++) {
        int M = Ms[i], N = M * M;
        test_scatter(/*n_limbs*/ N / 2 + 3, N, M, /*transposed*/ 1);
        test_scatter(N, N, M, 1);
    }

    /* ─ Gather, linear path: random + adversarial all-ones ─ */
    printf(CYN "--- gather (linear) ---" RST "\n");
    for (size_t i = 0; i < sizeof(lin_nc)/sizeof(lin_nc[0]); i++) {
        int nc = lin_nc[i];
        test_gather(/*n_coeffs*/ nc, /*n_store*/ nc, /*M*/ 0, /*transposed*/ 0, 0);
        test_gather(nc, nc, 0, 0, /*all_ones*/ 1);
    }

    /* ─ Gather, I15 transposed path: n_coeffs = N so xpose covers all slots ─ */
    printf(CYN "--- gather_t (I15 transposed) ---" RST "\n");
    for (size_t i = 0; i < sizeof(Ms)/sizeof(Ms[0]); i++) {
        int M = Ms[i], N = M * M;
        test_gather(/*n_coeffs*/ N, /*n_store*/ N, M, /*transposed*/ 1, 0);
        test_gather(N, N, M, 1, /*all_ones*/ 1);
    }

    /* ─ Randomized soak: many trials at varied sizes ─ */
    printf(CYN "--- randomized soak (400 trials) ---" RST "\n");
    for (int t = 0; t < 200; t++) {
        int nc = 1 + (int)(rng64() % 512);
        test_scatter(1 + (int)(rng64() % 600), nc, 0, 0);
        test_gather(nc, nc, 0, 0, 0);
    }
    for (int t = 0; t < 200; t++) {
        int M = 2 + (int)(rng64() % 30), N = M * M;
        test_scatter(1 + (int)(rng64() % (uint64_t)(N + 10)), N, M, 1);
        test_gather(N, N, M, 1, 0);
    }

    /* ─ Adversarial CLA carry chains (generate + long propagate runs) ─ */
    printf(CYN "--- CLA carry-lookahead, adversarial chains ---" RST "\n");
    test_cla_direct();

    /* ─ Multi-level blocked prefix scan (GPU multi-block path) ─ */
    printf(CYN "--- CLA multi-level blocked scan ---" RST "\n");
    test_blocked_scan();

    printf("\n");
    if (fails == 0)
        printf(GRN "=== All %d checks passed ===" RST "\n", checks);
    else
        printf(RED "=== %d/%d checks FAILED ===" RST "\n", fails, checks);
    return fails ? 1 : 0;
}
