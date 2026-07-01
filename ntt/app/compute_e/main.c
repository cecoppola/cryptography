/*
 * compute_e/main.c — Compute decimal digits of e to arbitrary precision.
 *
 * Algorithm:
 *   1. Estimate N_TERMS such that sum_{k>N} 1/k! < 10^(-D-5) (guard digits).
 *   2. binary_split(1, N_TERMS+1) gives (P, B) where sum_{k=1}^{N} 1/k! = P/B.
 *      B = N! (partial factorial product), e - 1 = P/B, e = (B+P)/B.
 *   3. Compute TEN_D = 10^D using cached powers of 10.
 *   4. result = floor((B+P) * TEN_D / B) via Newton reciprocal division.
 *   5. Convert result to decimal and print with "2." prefix.
 *
 * The first digit of e is always 2; result should equal 2718281828... * 10^(D-1).
 *
 * Usage:  compute_e -d <digits>
 * Default: 1000 digits.
 *
 * GPU NTT is used automatically for large operands (>= BIGINT_MUL_THRESHOLD limbs).
 * Call srun on MI300A before invoking.
 */

#include "binary_split.h"
#include "mem_pool.h"
#include "../../lib/arith/multiply.h"
#include "../../lib/arith/newton.h"
#include "../../lib/arith/base_convert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <signal.h>

/* Provided by ntt_mul.hip (compiled with hipcc); synchronizes and resets all
 * GPU devices.  Called from the signal handler so that in-flight HIP kernels
 * do not leave the command ring in an incomplete state on display-sharing
 * gfx1030 — an incomplete ring causes the driver to hang on the next open. */
extern void ntt_gpu_emergency_reset(void);

/* ─── GPU cleanup on signal ────────────────────────────────────────────────
 * SIGINT is sent by gpu_run.sh's timeout wrapper to abort a long run.
 * Without cleanup, in-flight kernels leave the GPU ring incomplete, and the
 * amdgpu driver can hard-hang the machine when it tries to recover.
 * SIGKILL cannot be caught — power cycle is the only recovery for that case.
 *
 * Hard deadline: amdgpu send_sigterm=0 means the driver does NOT signal our
 * process when it resets the GPU after a TDR. hipDeviceSynchronize() inside
 * ntt_gpu_emergency_reset() can therefore block indefinitely if the driver is
 * mid-reset when we call it. We set alarm(8) with SIGALRM=SIG_DFL so if the
 * reset call hangs, the process is terminated by SIGALRM within 8 seconds
 * rather than waiting for the 20-second SIGKILL from gpu_run.sh's kill-after.
 */
static void gpu_cleanup_handler(int sig)
{
    (void)sig;
    signal(SIGALRM, SIG_DFL);   /* default SIGALRM = terminate immediately   */
    alarm(8);                    /* hard deadline: exit within 8s no matter what */
    ntt_gpu_emergency_reset();   /* hipDeviceSynchronize only — NOT hipDeviceReset
                                  * (CRASH 14: reset + KFD teardown hard-crashes) */
    alarm(0);                    /* cancel alarm if reset completed cleanly   */
    _exit(130);
}

/* ─── Timing helpers ─────────────────────────────────────────────────────── */

static double wall_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ─── Max NTT transform size needed for operands of n_limbs limbs ─────────── */

static int log2_for_limbs(long n_limbs)
{
    int log2n = 1;
    while ((1L << log2n) < 2 * n_limbs) log2n++;
    return log2n;
}

/* ─── Main ───────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    signal(SIGINT,  gpu_cleanup_handler);
    signal(SIGTERM, gpu_cleanup_handler);
    signal(SIGPIPE, gpu_cleanup_handler); /* pipe closed (e.g. | head): clean GPU before exit */

    /* gfx1030 shares GPU rings with any Electron app (Chrome, VS Code, etc.).
     * Concurrent GPU access caused machine crashes even with TDR enabled
     * (2026-06-07: initially blamed Chrome, root-caused to VS Code's Electron
     * GPU compositor — same --type=gpu-process mechanism, different binary name).
     * Check before opening any HIP context.  Use popen so no device is opened.
     *
     * The former "VS Code-only + lockup_timeout >= 60 s is acceptable" exception
     * was REFUTED by CRASH 20 (2026-06-09): under exactly those conditions a
     * d=800000 launch froze the display/GFX pipeline (compute completed
     * underneath; machine dead at the seat).  Default is now hard abort on ANY
     * gpu-process; set COMPUTE_E_ALLOW_VSCODE=1 to accept the risk explicitly.
     *
     * Skipped in the pure-host build (COMPUTE_E_HOST): ntt_stub never opens a
     * HIP context, so the compositor cannot be starved — and gating it lets the
     * host differential-vs-OEIS test run on a dev box with VS Code open. */
#ifndef COMPUTE_E_HOST
    {
        /* Count --type=gpu-process instances and check for VS Code specifically. */
        FILE *fp = popen(
            "pgrep -af -- --type=gpu-process 2>/dev/null | grep -vc grep",
            "r");
        int n_gpu = 0;
        if (fp) {
            if (fscanf(fp, "%d", &n_gpu) < 1) n_gpu = 0;
            pclose(fp);
        }
        if (n_gpu > 0) {
            /* Check if it is VS Code (exactly one process, binary named "code"). */
            int vscode_only = 0;
            if (n_gpu == 1) {
                FILE *wp = popen(
                    "pgrep -af -- --type=gpu-process 2>/dev/null | grep -c '/code '",
                    "r");
                if (wp) {
                    int m = 0;
                    if (fscanf(wp, "%d", &m) < 1) m = 0;
                    pclose(wp);
                    vscode_only = (m >= 1);
                }
            }

            const char *allow = getenv("COMPUTE_E_ALLOW_VSCODE");
            if (vscode_only && allow && allow[0] == '1') {
                fprintf(stderr,
                    "compute_e: WARNING — VS Code GPU compositor is running.\n"
                    "  Proceeding ONLY because COMPUTE_E_ALLOW_VSCODE=1 is set.\n"
                    "  This configuration froze the display at compute launch"
                    " (CRASH 20, 2026-06-09).\n");
            } else {
                fprintf(stderr,
                    "compute_e: ABORT — Electron GPU compositor is running (%d process%s).\n"
                    "  A compositor sharing renderD128 froze the display at compute\n"
                    "  launch even with TDR >= 60 s (CRASH 20, 2026-06-09).\n"
                    "  Restart VS Code via ~/.local/bin/code (adds --disable-gpu),\n"
                    "  or close Chrome/Steam, then retry.\n",
                    n_gpu, n_gpu == 1 ? "" : "es");
                return 1;
            }
        }
    }
#endif /* !COMPUTE_E_HOST */

    long n_digits = 1000;
    int  force    = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            n_digits = atol(argv[++i]);
            if (n_digits < 1) { fprintf(stderr, "digits must be >= 1\n"); return 1; }
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--force") == 0) {
            force = 1;
        } else {
            fprintf(stderr, "Usage: %s [-d digits] [-f/--force]\n", argv[0]);
            return 1;
        }
    }

    /* Extra guard digits to absorb rounding errors. */
    long n_guard   = 20;
    long n_total   = n_digits + n_guard;
    long n_terms   = split_terms_needed(n_total);

    printf("compute_e: %ld digits, %ld terms, %ld guard\n",
           n_digits, n_terms, n_guard);
    fflush(stdout);

    /* ── Initialise NTT multiply engine ───────────────────────────────────── */
    /* Estimate max operand size: B = N_TERMS! ≈ exp(N*lnN - N) bits.         */
    double ln_B   = (double)n_terms * (log((double)n_terms) - 1.0);
    long B_bits   = (long)(ln_B / log(2.0)) + 64;
    long B_limbs  = B_bits / 64 + 4;
    /* TEN_D has n_total * log2(10) + 2 bits ≈ n_total * 3.32 bits.          */
    long T_limbs  = (long)(n_total * 3.33) / 64 + 4;
    /* Worst-case intermediate NTT product is in newton_reciprocal:
     *   x (≈ 2*B_limbs) × e (≈ 2*B_limbs) → product ≈ 4*B_limbs limbs.
     * The final (B+P)*TEN_D ≈ B_limbs + T_limbs is smaller.                */
    long max_limbs = 4 * B_limbs;
    if (3 * B_limbs + T_limbs > max_limbs) max_limbs = 3 * B_limbs + T_limbs;
    int  max_log_n = log2_for_limbs(max_limbs);
    /* Gate must match the engine's cap (ntt_mul_init). That cap defaults to 22
     * but is env-overridable via NTT_MAX_LOGN (log_n=22 fixed 2026-06-16,
     * log_n=23 GMP-verified, log_n=24 added via the recursive 3-factor 4-step
     * and validated in a 32M-digit run 2026-06-30). Read the same env here so
     * the host gate and the engine agree. */
    int engine_cap = 22;
    { const char *e = getenv("NTT_MAX_LOGN");
      if (e) { int v = atoi(e); if (v >= 3 && v <= 26) engine_cap = v; } }
    if (max_log_n > engine_cap) {
        fprintf(stderr, "compute_e: operands too large for 4-prime NTT "
                "(need log_n=%d, engine max=%d; set NTT_MAX_LOGN to raise).\n",
                max_log_n, engine_cap);
        return 1;
    }

    /* ── GPU runtime estimate and safety gate ────────────────────────────────
     * ntt_bigint_mul always dispatches at max_log_n (N=2^max_log_n) regardless
     * of actual operand size.  The binary_split tree above the schoolbook
     * threshold (64 limbs) has ~4*B_limbs/64 nodes; each calls bigint_mul twice.
     * Calibrated from measured d=100 000 baseline (0.45 s, log_n=16, ~400
     * dispatches → ~1.1 ms/dispatch).  Scales as 2^(log_n-16) * (log_n/16).  */
    long   est_dispatches = 4L * B_limbs / 64 + 60; /* +60: init/post/newton  */
    double ms_per_dispatch;
    if (max_log_n >= 16)
        ms_per_dispatch = 1.1 * pow(2.0, max_log_n - 16) * ((double)max_log_n / 16.0);
    else
        ms_per_dispatch = 1.1;
    double est_gpu_secs = (double)est_dispatches * ms_per_dispatch / 1000.0;

    printf("  log_n=%d  est_dispatches=~%ld  est_gpu=~%.1f s\n",
           max_log_n, est_dispatches, est_gpu_secs);
    fflush(stdout);

    /* gfx1030 is a display GPU sharing GPU resources with the compositor.
     * With TDR enabled (lockup_timeout=10000,600000,600000,600000,
     * gpu_recovery=1), the compute ring watchdog is 600 s — long runs
     * auto-recover on hang instead of hard-freezing the machine.
     * 120 s covers d~3M; beyond that use -f explicitly.
     * The MI300A has a dedicated compute queue — no such constraint.         */
#define SAFE_GPU_SECS 120.0
    if (est_gpu_secs > SAFE_GPU_SECS && !force) {
        fprintf(stderr,
            "compute_e: estimated GPU time %.1f s exceeds safe limit (%.0f s).\n"
            "  d=%ld, log_n=%d, ~%ld dispatches each at N=%ld.\n"
            "  On a display GPU (gfx1030) this risks a GPU hang / system crash.\n"
            "  Use -f / --force to override, or run on MI300A for large d.\n",
            est_gpu_secs, SAFE_GPU_SECS,
            n_digits, max_log_n, est_dispatches, 1L << max_log_n);
        return 1;
    }

    double t0 = wall_sec();
    printf("  ntt_mul_init(log_n=%d) ...\n", max_log_n);
    fflush(stdout);
    ntt_mul_init(max_log_n);

    /* ── Initialise base-conversion cache ────────────────────────────────── */
    /* Need pow10[k] = 10^(2^k) up to the level that covers n_total digits.  */
    int bc_level = 0;
    while ((1L << bc_level) < n_total) bc_level++;

    printf("  base_convert_init(max_level=%d) ...\n", bc_level);
    fflush(stdout);
    base_convert_init(bc_level);

    /* ── Binary split ─────────────────────────────────────────────────────── */
    printf("  binary_split(1, %ld) ...\n", n_terms + 1);
    fflush(stdout);
    double t1 = wall_sec();
    MemPool *pool = mem_pool_create();
    SplitResult sr;
    binary_split(&sr, 1L, n_terms + 1L, pool);
    mem_pool_destroy(pool);
    printf("  split done: %.2f s\n", wall_sec() - t1);
    fflush(stdout);

    /* sr.B = N_TERMS!,  sr.P / sr.B = sum_{k=1}^{N} 1/k!                   */
    /* e = (sr.B + sr.P) / sr.B                                              */
    BigInt numerator = bigint_alloc(sr.B.n_limbs + 2);
    bigint_add(&numerator, &sr.B, &sr.P);   /* numerator = B + P             */

    /* ── TEN_D = 10^n_total ──────────────────────────────────────────────── */
    printf("  computing 10^%ld ...\n", n_total);
    fflush(stdout);
    double t2 = wall_sec();
    BigInt ten_D = bigint_alloc((long)(n_total * 3.33 / 64) + 8);
    base_convert_pow10_exact(&ten_D, n_total);
    printf("  10^%ld done: %.2f s\n", n_total, wall_sec() - t2);
    fflush(stdout);

    /* ── result = floor(numerator * TEN_D / B) ───────────────────────────── */
    printf("  computing result = floor((B+P)*10^D / B) ...\n");
    fflush(stdout);
    double t3 = wall_sec();

    BigInt recip_B = bigint_alloc(sr.B.n_limbs * 2 + 4);
    newton_reciprocal(&recip_B, &sr.B);

    /* num = numerator * TEN_D */
    long num_cap = numerator.n_limbs + ten_D.n_limbs + 4;
    BigInt num = bigint_alloc((int)num_cap);
    bigint_mul(&num, &numerator, &ten_D);

    /* result = floor(num / B) */
    BigInt result = bigint_alloc((int)(num.n_limbs + 4));
    bigint_div_newton(&result, &num, &sr.B, &recip_B);
    printf("  division done: %.2f s\n", wall_sec() - t3);
    fflush(stdout);

    /* ── Base conversion — GPU still alive ──────────────────────────────── */
    /* bigint_to_decimal uses GPU NTT (via bigint_mul / newton_reciprocal) and
     * reads pow10_cache which base_convert_teardown frees.  Both must be done
     * before any teardown.  The result is a malloc'd string in CPU RAM.      */
    double t4 = wall_sec();
    printf("  base conversion ...\n");
    fflush(stdout);

    char *dec = bigint_to_decimal(&result);
    if (!dec) { fprintf(stderr, "base conversion failed\n"); return 1; }
    printf("  base conversion done: %.2f s\n", wall_sec() - t4);
    fflush(stdout);

    unsigned long gpu_calls = ntt_get_gpu_dispatch_count();

    /* ── GPU teardown before terminal output ─────────────────────────────── */
    /* dec[] is in CPU RAM.  Tear down the GPU NOW, before fwrite() sends 1M+
     * characters to stdout.  On gfx1030 the terminal (xterm.js/Electron)
     * renders output via WebGL on renderD128 — the same node as our HIP
     * compute.  ntt_mul_teardown() frees all VRAM and destroys all streams;
     * after it returns, no HIP allocations remain and the compute ring is
     * idle.  Do NOT call hipDeviceReset() here — it forcibly resets the HIP
     * context before the ROCm atexit (hsa_shut_down) can clean up, causing a
     * KFD driver race that crashes the machine on display-sharing GPUs
     * (confirmed 2026-06-08).  Let the HIP/HSA runtime shut down naturally
     * via its atexit handler after main() returns.  */
    base_convert_teardown();
    ntt_mul_teardown();
    signal(SIGINT,  SIG_DFL); /* restore defaults: all VRAM freed, no GPU     */
    signal(SIGTERM, SIG_DFL); /* cleanup needed if interrupted during I/O     */
    signal(SIGPIPE, SIG_DFL);

    long dec_len = (long)strlen(dec);
    if (dec_len < n_digits) {
        fprintf(stderr, "warning: fewer digits than expected (%ld < %ld)\n",
                dec_len, n_digits);
        n_digits = dec_len;
    }

    double t5 = wall_sec();
    printf("  output ...\n");
    fflush(stdout);
    printf("\ne = ");
    size_t wr = 0;
    wr += fwrite(dec, 1, 1, stdout);
    wr += fwrite(".", 1, 1, stdout);
    wr += fwrite(dec + 1, 1, (size_t)(n_digits - 1), stdout);
    wr += fwrite("\n", 1, 1, stdout);
    if (wr != (size_t)(n_digits + 2)) {
        fprintf(stderr, "compute_e: short write (%zu of %zu)\n",
                wr, (size_t)(n_digits + 2));
        return 1;
    }
    printf("  output: %.2f s\n", wall_sec() - t5);

    free(dec);
    bigint_free(&sr.P);
    bigint_free(&sr.B);
    bigint_free(&numerator);
    bigint_free(&ten_D);
    bigint_free(&recip_B);
    bigint_free(&num);
    bigint_free(&result);

    printf("total: %.2f s\n", wall_sec() - t0);
    if (gpu_calls > 0)
        printf("gpu_dispatches: %lu\n", gpu_calls);
    return 0;
}
