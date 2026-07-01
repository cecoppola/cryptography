/*
 * ntt_stub.c — host-only NTT stubs for the compute_e CPU differential test.
 *
 * multiply.c dispatches to ntt_bigint_mul() only when an operand reaches
 * BIGINT_MUL_THRESHOLD (64 limbs ≈ 1233 decimal digits). The host
 * differential test runs compute_e at digit counts (<=200) for which
 * every multiply provably stays on the schoolbook path, so these stubs
 * are never reached. If one IS reached, abort loudly — that means the
 * test's digit cap was set too high, not that the stub is acceptable.
 *
 * Same stub convention as lib/arith/test_arith.c. Lets compute_e link
 * with plain gcc (no hipcc, no amdhip64, no GPU runtime dependency).
 */

#include "../../lib/arith/bigint.h"
#include <stdio.h>
#include <stdlib.h>

void ntt_bigint_mul(BigInt *c, const BigInt *a, const BigInt *b)
{
    (void)c; (void)a; (void)b;
    fprintf(stderr,
        "FAIL: ntt_bigint_mul reached — operand exceeded the schoolbook\n"
        "      threshold; lower the host differential test's digit cap.\n");
    exit(2);
}

void ntt_mul_init_impl(int max_log_n) { (void)max_log_n; }
void ntt_mul_teardown_impl(void) {}
unsigned long ntt_get_gpu_dispatch_count(void) { return 0; } /* CPU-only stub */
void ntt_gpu_emergency_reset(void) {}  /* no-op: no GPU in host build */
