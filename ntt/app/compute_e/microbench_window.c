/* microbench_window.c — single full-size NTT-mul timing harness.
 *
 * Purpose: measure the wall-clock duration of ONE full-size ntt_bigint_mul at a
 * given log_n, to characterise the uninterrupted GPU compute window (the
 * forward/inverse NTT phase) that drives the flip_done / queue-preemption
 * deadline on the display GPU.  Run with NTT_PROFILE=2 to also print the
 * scatter / gpu-pipe(=fwd+inv) / garner / gather breakdown at teardown.
 *
 * Inputs:  argv[1] = log_n (3..22), argv[2] = iters (default 5).
 * Outputs: per-iteration mul wall time (ms) to stdout; profiler breakdown to
 *          stderr at teardown (when NTT_PROFILE>=1).
 * NOT a correctness test — operands are arbitrary full-N limb fills.
 */
#include "bigint.h"
#include "multiply.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static double ms_since(struct timespec a, struct timespec b)
{
    return (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

int main(int argc, char **argv)
{
    int log_n    = (argc > 1) ? atoi(argv[1]) : 20;
    int iters    = (argc > 2) ? atoi(argv[2]) : 5;
    int sleep_ms = (argc > 3) ? atoi(argv[3]) : 0;   /* inter-mul sleep, 0=none */
    int N        = 1 << log_n;

    printf("microbench_window: log_n=%d N=%d iters=%d sleep_ms=%d (NTT_PROFILE=%s)\n",
           log_n, N, iters, sleep_ms, getenv("NTT_PROFILE") ? getenv("NTT_PROFILE") : "0");

    ntt_mul_init(log_n);

    BigInt a = bigint_alloc(N + 8);
    BigInt b = bigint_alloc(N + 8);
    BigInt c = bigint_alloc(2 * N + 16);
    /* Full-N pseudo-random nonzero fill so the NTT runs at full size. */
    for (int i = 0; i < N; i++) {
        a.limbs[i] = (limb_t)(0x9e3779b97f4a7c15ULL * (unsigned)(i + 1));
        b.limbs[i] = (limb_t)(0xc2b2ae3d27d4eb4fULL * (unsigned)(i + 3));
    }
    a.n_limbs = N;
    b.n_limbs = N;

    /* Progress file for the crash heartbeat (CRASH_PROGRESS env, default /tmp). */
    const char *progpath = getenv("CRASH_PROGRESS");
    if (!progpath) progpath = "/tmp/crash_progress";

    double best = 1e30, sum = 0.0;
    for (int it = 0; it < iters; it++) {
        /* publish progress BEFORE the mul so the last heartbeat names the mul
         * that was in flight at the freeze. */
        FILE *pf = fopen(progpath, "w");
        if (pf) { fprintf(pf, "ln%d mul %d/%d", log_n, it, iters); fclose(pf); }
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        ntt_bigint_mul(&c, &a, &b);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = ms_since(t0, t1);
        if (ms < best) best = ms;
        sum += ms;
        printf("  log_n=%d iter=%d  mul_wall=%.2f ms\n", log_n, it, ms);
        fflush(stdout);
        if (sleep_ms > 0) usleep((useconds_t)sleep_ms * 1000);
    }
    printf("log_n=%d  best_mul=%.2f ms  avg_mul=%.2f ms  (window ~= compute/2 from profiler)\n",
           log_n, best, sum / iters);

    bigint_free(&a);
    bigint_free(&b);
    bigint_free(&c);
    ntt_mul_teardown();
    return 0;
}
