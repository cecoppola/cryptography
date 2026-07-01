/*
 * base_convert.c — BFS divide-and-conquer decimal conversion.
 *
 * Cache structure: pow10_cache[k] = 10^(2^k), recip_cache[k] = Newton reciprocal.
 * At level L, an N with 2^L decimal digits is split as:
 *   (q, r) = divmod(N, pow10_cache[L-1])    (q has upper 2^(L-1) digits, r lower)
 * Both halves are processed at level L-1, yielding exactly 2^(L-1) characters each.
 *
 * dc_convert_bfs() processes the D&C tree level-at-a-time (BFS).  Nodes above
 * BASE_CONVERT_MUL_THRESHOLD limbs are allocated in managed memory and dispatched
 * to the GPU NTT; nodes at or below the threshold use CPU schoolbook.
 *
 * BASE_CONVERT_MUL_THRESHOLD mirrors BIGINT_MUL_THRESHOLD on MI300A (unified
 * HBM3, cheap sync) but is raised on GFX1030_LOCAL where each
 * hipStreamSynchronize round-trip costs ~5-20 ms over PCIe — schoolbook is
 * faster than the GPU for small operands on that architecture.
 * The threshold is injected into bigint_mul's runtime slot around the BFS so
 * the binary_split path (always large operands) is unaffected.
 */

#include "base_convert.h"
#include "newton.h"
#include "multiply.h"   /* BIGINT_MUL_THRESHOLD, bigint_mul_get/set_threshold */

/* Architecture-appropriate NTT crossover for the base conversion BFS. */
#ifndef BASE_CONVERT_MUL_THRESHOLD
#  ifdef GFX1030_LOCAL
#    define BASE_CONVERT_MUL_THRESHOLD  512
#  else
#    define BASE_CONVERT_MUL_THRESHOLD  BIGINT_MUL_THRESHOLD
#  endif
#endif
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_LEVELS 30

static BigInt pow10_cache[MAX_LEVELS];
static BigInt recip_cache[MAX_LEVELS];
static int    cache_levels = 0;   /* number of valid entries */

/* ─── Init / teardown ────────────────────────────────────────────────────── */

void base_convert_init(int max_level)
{
    if (max_level >= MAX_LEVELS) {
        fprintf(stderr, "base_convert_init: max_level %d clamped to %d\n",
                max_level, MAX_LEVELS - 1);
        max_level = MAX_LEVELS - 1;
    }

    /* Level 0: pow10[0] = 10, recip[0] = floor(2^(2*b) / 10). */
    pow10_cache[0] = bigint_alloc(2);
    bigint_set_u64(&pow10_cache[0], 10);
    recip_cache[0] = bigint_alloc(4);
    newton_reciprocal(&recip_cache[0], &pow10_cache[0]);

    for (int k = 1; k <= max_level; k++) {
        /* pow10[k] = pow10[k-1]^2 */
        int cap = pow10_cache[k - 1].n_limbs * 2 + 4;
        pow10_cache[k] = bigint_alloc(cap);
        bigint_mul(&pow10_cache[k], &pow10_cache[k - 1], &pow10_cache[k - 1]);

        int rcap = pow10_cache[k].n_limbs * 2 + 4;
        recip_cache[k] = bigint_alloc(rcap);
        newton_reciprocal(&recip_cache[k], &pow10_cache[k]);
    }
    cache_levels = max_level + 1;
}

void base_convert_teardown(void)
{
    for (int k = 0; k < cache_levels; k++) {
        bigint_free(&pow10_cache[k]);
        bigint_free(&recip_cache[k]);
    }
    cache_levels = 0;
}

/* ─── BFS conversion ─────────────────────────────────────────────────────── */

/* One node in the BFS frontier: the BigInt for this sub-problem (managed
 * memory) and the start of its output character range. */
typedef struct { BigInt N; char *buf; } BFSNode;

/*
 * dc_convert_bfs — BFS (level-at-a-time) divide-and-conquer decimal conversion.
 *
 * Processes 2^(level-lv) independent nodes at BFS depth (level-lv) in a
 * single pass.  All intermediate q/r BigInts are managed (hipMallocManaged)
 * so ntt_bigint_mul's scatter kernel reads them directly without an H2D copy.
 * OpenMP parallelises nodes within each BFS level; GPU NTT calls serialise
 * on ntt_gpu_lock inside multiply.c.
 *
 * Fills buf[0..2^level-1] with exactly 2^level decimal characters (zero-padded).
 * N_cpu is the initial CPU-memory BigInt; copied to managed memory on entry.
 */
static void dc_convert_bfs(char *buf, int level, const BigInt *N_cpu)
{
    /* Seed: copy input to managed memory when large enough to hit the NTT path.
     * GFX1030_LOCAL: hipMallocManaged allocates in system RAM; GPU access causes
     * page faults on RDNA2 (no XNACK / on-demand migration on PCIe).  Disable
     * managed allocation and let ntt_bigint_mul use the H2D staging buffer.
     * MI300A (unified HBM3): managed allocation lives in shared physical memory,
     * so the GPU reads it directly without any copy. */
#ifdef GFX1030_LOCAL
    int root_mgd = 0;
#else
    int root_mgd = (N_cpu->n_limbs > BASE_CONVERT_MUL_THRESHOLD);
#endif
    BigInt N0 = root_mgd ? bigint_alloc_managed(N_cpu->n_limbs + 8)
                         : bigint_alloc(N_cpu->n_limbs + 8);
    bigint_copy(&N0, N_cpu);

    long count = 1;
    BFSNode *nodes = (BFSNode *)malloc(sizeof(BFSNode));
    if (!nodes) { fprintf(stderr, "dc_convert_bfs: OOM\n"); exit(1); }
    nodes[0] = (BFSNode){ N0, buf };

    for (int lv = level; lv >= 1; lv--) {
        int  lv1  = lv - 1;
        int  half = 1 << lv1;   /* chars per output half */
        long next_count = count * 2;
        BFSNode *next = (BFSNode *)malloc((size_t)next_count * sizeof(BFSNode));
        if (!next) { fprintf(stderr, "dc_convert_bfs: OOM\n"); exit(1); }

        /* Independent splits — parallelize across the BFS frontier.
         * GPU NTT calls inside bigint_mul are serialised by ntt_gpu_lock;
         * alloc/shr/sub/free run concurrently across threads. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (long i = 0; i < count; i++) {
            const BigInt *N   = &nodes[i].N;
            int           nlm = N->n_limbs;
            int           plm = pow10_cache[lv1].n_limbs;

            /* q and r: managed only when N is large enough to go through the
             * NTT path (nlm > BIGINT_MUL_THRESHOLD).  Leaf-level nodes are
             * tiny (≤ 2 limbs); schoolbook handles them and managed memory
             * would just add hipMallocManaged overhead with no H2D benefit.
             * qD is a CPU temp only used for r = N - q*pow10. */
#ifdef GFX1030_LOCAL
            int use_mgd = 0;   /* PCIe: managed = system RAM, GPU faults on access */
#else
            int use_mgd = (nlm > BASE_CONVERT_MUL_THRESHOLD);
#endif
            BigInt q  = use_mgd ? bigint_alloc_managed(nlm + 8) : bigint_alloc(nlm + 8);
            BigInt qD = bigint_alloc(nlm + plm + 8);
            BigInt r  = use_mgd ? bigint_alloc_managed(nlm + 8) : bigint_alloc(nlm + 8);

            bigint_div_newton(&q, N, &pow10_cache[lv1], &recip_cache[lv1]);
            bigint_mul(&qD, &q, &pow10_cache[lv1]);
            bigint_sub(&r, N, &qD);

            bigint_free(&nodes[i].N);
            bigint_free(&qD);

            next[2 * i]     = (BFSNode){ q, nodes[i].buf };
            next[2 * i + 1] = (BFSNode){ r, nodes[i].buf + half };
        }

        free(nodes);
        nodes = next;
        count = next_count;
    }

    /* Base: each node holds a single decimal digit. */
    for (long i = 0; i < count; i++) {
        nodes[i].buf[0] = '0' + (char)(nodes[i].N.n_limbs > 0
                                        ? (int)(nodes[i].N.limbs[0] % 10) : 0);
        bigint_free(&nodes[i].N);
    }
    free(nodes);
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

char *bigint_to_decimal(const BigInt *N)
{
    if (bigint_is_zero(N)) {
        char *s = malloc(2);
        if (!s) return NULL;
        s[0] = '0'; s[1] = '\0';
        return s;
    }

    /* Estimate decimal digit count: ceil(bits * log10(2)) + 2 guard digits. */
    int bits     = bigint_bits(N);
    long n_digits = (long)bits * 30103L / 100000L + 2;

    /* Find smallest level L such that 2^L >= n_digits. */
    int level = 0;
    while ((1L << level) < n_digits) level++;

    if (level >= cache_levels) {
        fprintf(stderr, "base_convert: need level %d but cache has %d levels "
                "(call base_convert_init(%d))\n", level, cache_levels, level);
        return NULL;
    }

    long buf_size = 1L << level;
    char *buf = malloc((size_t)buf_size + 1);
    if (!buf) return NULL;

    /* Raise the NTT threshold for base conversion: schoolbook is faster than
     * a GPU dispatch for small nodes on PCIe hardware.  The binary_split path
     * always uses large operands and is unaffected by this temporary change. */
    int saved_threshold = bigint_mul_get_threshold();
    bigint_mul_set_threshold(BASE_CONVERT_MUL_THRESHOLD);
    dc_convert_bfs(buf, level, N);
    bigint_mul_set_threshold(saved_threshold);
    buf[buf_size] = '\0';

    /* Strip leading zeros, keeping at least one digit. */
    char *start = buf;
    while (*start == '0' && *(start + 1) != '\0') start++;

    long len = buf_size - (start - buf);
    memmove(buf, start, (size_t)(len + 1));
    return buf;
}

/* ─── Power-of-10 helper ─────────────────────────────────────────────────── */

void base_convert_pow10_exact(BigInt *out, long n)
{
    /* out = 10^n using the binary representation of n and pow10_cache[k] = 10^(2^k). */
    bigint_resize(out, 4);
    bigint_set_u64(out, 1);

    BigInt tmp = bigint_alloc(8);
    int k = 0;
    long remaining = n;
    while (remaining > 0) {
        if (k >= cache_levels) {
            fprintf(stderr, "base_convert_pow10_exact: cache too small for 10^%ld "
                    "(call base_convert_init with max_level >= %d)\n", n, k);
            bigint_free(&tmp);
            return;
        }
        if (remaining & 1) {
            bigint_mul(&tmp, out, &pow10_cache[k]);
            bigint_copy(out, &tmp);
        }
        remaining >>= 1;
        k++;
    }
    bigint_free(&tmp);
}

long bigint_write_decimal(const BigInt *N, int fd)
{
    char *s = bigint_to_decimal(N);
    if (!s) return -1;
    long len = (long)strlen(s);
    if (write(fd, s, (size_t)len) < 0) {
        free(s);
        return -1;
    }
    free(s);
    return len;
}
