#include <stdio.h>
#include <stdlib.h>

/* Link against ntt_mul (provides ntt_mul_init_impl, ntt_mul_test_3factor_roundtrip). */
extern void ntt_mul_init_impl(int max_log_n);
extern int  ntt_mul_test_3factor_roundtrip(void);
extern int  ntt_mul_test_3factor_convolution(void);
extern void ntt_mul_teardown_impl(void);

int main(void)
{
    printf("3-factor tests: (1) NTT->INTT roundtrip, (2) cyclic convolution  (log_n=24)\n");
    fflush(stdout);

    ntt_mul_init_impl(24);

    int rt = ntt_mul_test_3factor_roundtrip();
    printf(rt == 0 ? "PASS roundtrip: all 2^24 match\n" : "FAIL roundtrip: %d mismatches\n", rt);
    fflush(stdout);

    int cv = ntt_mul_test_3factor_convolution();
    printf(cv == 0 ? "PASS convolution: matches hand-computed cyclic conv\n"
                   : "FAIL convolution: %d mismatches\n", cv);

    ntt_mul_teardown_impl();
    return (rt || cv) ? 1 : 0;
}
