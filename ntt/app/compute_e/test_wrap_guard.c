/*
 * test_wrap_guard.c — Death-test for ntt_bigint_mul's cyclic-wrap size guard.
 *
 * ntt_bigint_mul runs a fixed size-N cyclic convolution; it aborts (exit(1)) if
 * the operands would overflow it (n_a + n_b - 1 > N), because otherwise the top
 * product coefficients fold back mod x^N-1 and the result is silently wrong.
 * This test verifies BOTH directions via fork():
 *   - negative: oversized operands MUST make the child exit(1) (guard fires);
 *   - positive: valid operands MUST let the child exit(0) (guard never over-fires).
 * The guard is pure host logic reached right after init, so the negative child
 * launches no kernels; the positive child runs one small (log_n=12) multiply.
 *
 * GPU test — requires a device. Not in the GPU-free `make check`; run via the
 * dev GPU suite (gpu_run.sh) like the other engine tests. Links the _dev engine
 * objects (mirrors test_3factor_roundtrip).
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include "arith/bigint.h"

extern void ntt_mul_init_impl(int max_log_n);
extern void ntt_mul_teardown_impl(void);
extern void ntt_bigint_mul(BigInt *c, const BigInt *a, const BigInt *b);

#define GUARD_MISSED 42   /* distinct child code: mul returned without aborting */

/* Build a BigInt with `n` nonzero significant limbs (top limb nonzero so it
 * survives trimming, keeping n_limbs == n as the guard reads it). */
static BigInt make_operand(int n)
{
    BigInt a = bigint_alloc(n + 4);
    for (int i = 0; i < n; i++) a.limbs[i] = (limb_t)(i + 1);
    a.n_limbs = n;
    return a;
}

/* Run one multiply in this (child) process; never returns for the guard case. */
static void child_multiply(int log_n, int n_a, int n_b)
{
    ntt_mul_init_impl(log_n);              /* opens the HIP context */
    BigInt a = make_operand(n_a);
    BigInt b = make_operand(n_b);
    BigInt c = bigint_alloc((1 << log_n) + 8);
    ntt_bigint_mul(&c, &a, &b);            /* guard exit(1)s here when oversized */
    /* Reached only when the guard did NOT fire (valid size, or guard broken). */
    bigint_free(&a); bigint_free(&b); bigint_free(&c);
    ntt_mul_teardown_impl();
    _exit(0);
}

/* Fork the multiply and return the child's exit status (or -1 if it did not
 * exit cleanly, e.g. killed by a signal). */
static int run_child(int log_n, int n_a, int n_b)
{
    fflush(stdout); fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(2); }
    if (pid == 0) {
        child_multiply(log_n, n_a, n_b);
        _exit(GUARD_MISSED);   /* child_multiply returns only on the valid path */
    }
    int st = 0;
    if (waitpid(pid, &st, 0) < 0) { perror("waitpid"); exit(2); }
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

int main(void)
{
    int fails = 0;
    int log_n = 12;                 /* N = 4096 */
    int N = 1 << log_n;

    /* Negative: n_a + n_b - 1 = 5999 > 4096 -> guard must exit(1). */
    int neg = run_child(log_n, 3000, 3000);
    if (neg == 1) {
        printf("  [ok]   oversized operands -> guard aborted (exit 1)\n");
    } else {
        printf("  [FAIL] oversized operands -> child exit %d (expected 1; "
               "%d = guard missed / cyclic wrap not caught)\n", neg, GUARD_MISSED);
        fails++;
    }

    /* Positive: n_a + n_b - 1 = 1999 <= 4096 -> multiply completes, exit(0). */
    int pos = run_child(log_n, 1000, 1000);
    if (pos == 0) {
        printf("  [ok]   valid operands (n_a+n_b-1=1999 <= N=%d) -> completed\n", N);
    } else {
        printf("  [FAIL] valid operands -> child exit %d (expected 0; guard "
               "over-fired or multiply failed)\n", pos);
        fails++;
    }

    printf("test_wrap_guard: %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
