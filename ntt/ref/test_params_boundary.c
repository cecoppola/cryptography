/*
 * test_params_boundary.c — boundary unit test for ntt_params_init guards.
 *
 * Covers two guard classes generalised across all 5 ref init functions:
 *   (a) SIGFPE-prevention: q < 2  → return -1   (all 5 TUs)
 *   (b) lazy-overflow:     q > UINT64_MAX/(2*n) → return -1
 *       (ntt_polymul.c + ntt_polymul_negacyclic.c only — those two run
 *       a custom unreduced Stockham butterfly. ntt_cpu/ntt_stockham/
 *       ntt_bench use addmod/submod per step and don't need it.)
 *
 * Built once per subject TU (5 binaries):
 *   cc -DNTT_CPU_NO_MAIN  test_params_boundary.c ntt_cpu.c
 *   cc -DNTT_STOCKHAM_NO_MAIN test_params_boundary.c ntt_stockham.c
 *   ... etc.
 *
 * The build is parameterised by -DSUBJECT="..." (label only) and
 * -DSUBJECT_HAS_LAZY=0/1 (whether the lazy-overflow guard exists).
 *
 * Build:  see Makefile targets test-params-boundary-{cpu,stok,pm,neg,bench}
 */

#include <stdio.h>
#include <stdint.h>
#include "ntt.h"

#ifndef SUBJECT
#define SUBJECT "?"
#endif
#ifndef SUBJECT_HAS_LAZY
#define SUBJECT_HAS_LAZY 0
#endif

#define GRN "\033[1;32m"
#define RED "\033[1;31m"
#define CYN "\033[1;36m"
#define RST "\033[0m"

static int fails = 0;

static void check(const char *label, int got, int want)
{
    int ok = (got == want);
    fails += !ok;
    printf("  %-40s got=%-3d want=%-3d %s%s%s\n",
           label, got, want,
           ok ? GRN : RED, ok ? "PASS" : "FAIL", RST);
}

int main(void)
{
    printf("\n" CYN "  ── ntt_params_init boundary test (subject: %s, lazy-guard=%d) ──" RST "\n",
           SUBJECT, SUBJECT_HAS_LAZY);

    ntt_params_t p;

    /* T1: q=0 — must reject (h_mod_pow base%=0 SIGFPE prevention). */
    p = (ntt_params_t){ .n = 2, .q = 0, .omega = 1 };
    check("T1 q=0 (SIGFPE prevention)", ntt_params_init(&p), -1);

    /* T2: q=1 — degenerate; must reject. */
    p = (ntt_params_t){ .n = 2, .q = 1, .omega = 0 };
    check("T2 q=1 (degenerate)", ntt_params_init(&p), -1);

    /* T3: q=2, n=2, omega=1 — minimum admissible; must accept. */
    p = (ntt_params_t){ .n = 2, .q = 2, .omega = 1 };
    check("T3 q=2 (minimum admissible)", ntt_params_init(&p), 0);

    /* T4: n=0 — invalid (power-of-2 guard); must reject. */
    p = (ntt_params_t){ .n = 0, .q = 3329, .omega = 17 };
    check("T4 n=0 (power-of-2 guard)", ntt_params_init(&p), -1);

    /* T5: n=3 (not power of 2) — must reject. */
    p = (ntt_params_t){ .n = 3, .q = 3329, .omega = 17 };
    check("T5 n=3 (not power of 2)", ntt_params_init(&p), -1);

    /* T6: ML-KEM canonical (n=256, q=3329, omega=17) — must accept. */
    p = (ntt_params_t){ .n = 256, .q = 3329, .omega = 17 };
    check("T6 ML-KEM canonical (n=256,q=3329)", ntt_params_init(&p), 0);

#if SUBJECT_HAS_LAZY
    /* Lazy-overflow boundary tests — only for subjects that carry the
     * q > UINT64_MAX/(2n) guard. Boundary value is exact integer division. */
    const uint64_t N = 64;
    const uint64_t BOUND = UINT64_MAX / (2 * N);   /* admissible max */

    /* T7: q = BOUND, n=64 — exactly at the limit; must accept.
     * (h_mod_pow with such a large q is O(log q) ≈ 58 squares, fine.) */
    p = (ntt_params_t){ .n = N, .q = BOUND, .omega = 1 };
    check("T7 q == UINT64_MAX/(2n) (at limit)", ntt_params_init(&p), 0);

    /* T8: q = BOUND + 1, n=64 — one over the limit; must reject.
     * This is the bug class that would let lazy accumulation overflow
     * uint64_t silently — guard MUST catch it. */
    p = (ntt_params_t){ .n = N, .q = BOUND + 1, .omega = 1 };
    check("T8 q == UINT64_MAX/(2n)+1 (one over)", ntt_params_init(&p), -1);

    /* T9: q = UINT64_MAX, n=64 — far over; must reject. */
    p = (ntt_params_t){ .n = N, .q = UINT64_MAX, .omega = 1 };
    check("T9 q = UINT64_MAX (far over)", ntt_params_init(&p), -1);
#else
    /* Subjects without the lazy-overflow guard: a large q must still be
     * ACCEPTED by init (no extra check), because their butterfly does
     * addmod/submod per step and is safe at any q < 2^64. Verify that
     * fact — if a stale comment ever claimed otherwise, this catches it. */
    p = (ntt_params_t){ .n = 64, .q = 18446744069414584321ULL /* Goldilocks */,
                        .omega = 2 /* any nonzero */ };
    check("T7 q=Goldilocks (no lazy guard, accept)", ntt_params_init(&p), 0);
#endif

    printf("\n  %s%d failures%s\n", fails ? RED : GRN, fails, RST);
    return fails ? 1 : 0;
}
