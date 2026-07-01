/*
 * test_reduce.c — Exhaustive-ish correctness test for reduce_p0..p3.
 *
 * Compares each reduce_pK(x) against the exact reference (x % PK) for
 * x in [0, (PK-1)^2] (the full Hadamard product domain). Covers:
 *   - 5M uniform-random 128-bit products a*b with a,b in [0,PK)
 *   - targeted corners: a,b in {0,1,2,PK-1,PK-2, near 2^32/2^24/2^34/2^40,
 *     and values engineered so the round-2 fold lands within ±2^20 of 2^64}
 *
 * Pure C (gcc), no GPU. reduce_pK come from primes.h (host path).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "primes.h"

static uint64_t splitmix64(uint64_t *s){
    uint64_t z=(*s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
    z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}

static uint64_t red(int pidx,__uint128_t x){
    switch(pidx){case 0:return reduce_p0(x);case 1:return reduce_p1(x);
        case 2:return reduce_p2(x);default:return reduce_p3(x);}
}
static int check(int pidx,uint64_t P,uint64_t a,uint64_t b,long *bad){
    __uint128_t x=(__uint128_t)a*b;
    uint64_t got=red(pidx,x);
    uint64_t exp=(uint64_t)(x % P);
    if(got!=exp){
        if(*bad<8)
            fprintf(stderr,"  P%d FAIL a=%016llx b=%016llx got=%016llx exp=%016llx\n",
                pidx,(unsigned long long)a,(unsigned long long)b,
                (unsigned long long)got,(unsigned long long)exp);
        (*bad)++; return 1;
    }
    return 0;
}

int main(void){
    uint64_t Ps[4]={P0,P1,P2,P3};
    int fail=0;
    for(int t=0;t<4;t++){
        uint64_t P=Ps[t]; long bad=0;
        /* corner operands */
        uint64_t corners[]={0,1,2,3,P-1,P-2,P-3,
            (1ULL<<32),(1ULL<<32)-1,(1ULL<<24),(1ULL<<24)-1,
            (1ULL<<34),(1ULL<<34)-1,(1ULL<<40),(1ULL<<40)-1,
            0xffffffffffffffffULL%P,0x8000000000000000ULL%P,
            P-(1ULL<<24),P-(1ULL<<32),P-(1ULL<<40),
            0x00000000ffffffffULL,0xffffffff00000000ULL%P};
        int nc=(int)(sizeof(corners)/sizeof(corners[0]));
        for(int i=0;i<nc;i++)for(int j=0;j<nc;j++)
            check(t,P,corners[i],corners[j],&bad);
        /* engineered: pick a,b so hi*Cx+lo round-2 fold straddles 2^64.
         * Sweep a across a band, b = P-1, plus a*b where product hi ~ P-1. */
        for(uint64_t a=0;a<(1ULL<<20);a++){
            check(t,P,a,P-1,&bad);
            check(t,P,P-1,(P-1)-a,&bad);
            check(t,P,(P-1)-a,P-2,&bad);
        }
        /* uniform random */
        uint64_t s=0xD1B54A32D192ED03ULL ^ (uint64_t)t;
        for(long n=0;n<5000000;n++){
            uint64_t a=splitmix64(&s)%P, b=splitmix64(&s)%P;
            check(t,P,a,b,&bad);
        }
        printf("P%d (%016llx): %s  (%ld mismatches)\n",
            t,(unsigned long long)P,bad?"FAIL":"PASS",bad);
        if(bad)fail=1;
    }
    printf("%s\n",fail?"=== REDUCE TEST FAILED ===":"=== all reduce tests PASS ===");
    return fail;
}
