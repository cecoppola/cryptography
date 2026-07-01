/*
 * test_polymul_sweep.hip — Randomized differential stress test for the full
 * 4-prime CRT-NTT bigint multiply pipeline on the 6900XT.
 *
 * For each (na, nb) operand size in a sweep that crosses every log_n
 * boundary the engine uses (n_coeffs = na+nb spanning 2^8..2^15), and for
 * several random trials, compare:
 *     got = bigint_mul(a, b)            (GPU CRT-NTT: scatter→NTT→Hadamard
 *                                        →INTT→Garner→gather, all 4 primes)
 *     exp = schoolbook(a, b)            (CPU __uint128_t carry chain)
 * Sizes >64 limbs force the NTT path (BIGINT_MUL_THRESHOLD=64); the small
 * sizes also exercise the schoolbook fallback as a control.
 *
 * For large sizes where O(n^2) schoolbook is too slow, instead assert two
 * cheap structural invariants that every one of this session's three bugs
 * (reduce_p0/p1, montgomery_mul) violated:
 *   (I1) commutativity:  a*b == b*a
 *   (I2) degree bound:   limbs beyond na+nb-1 are zero (no scrambled tail)
 *
 * Deterministic PRNG (splitmix64) so any failure is reproducible.
 */
#include "../../lib/arith/bigint.h"
#include "../../lib/arith/multiply.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t SM(uint64_t *s){
    uint64_t z=(*s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
    z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}
static void randbig(BigInt *x,int n,uint64_t *s){
    bigint_resize(x,n+4); bigint_zero(x); x->n_limbs=n;
    for(int i=0;i<n;i++) x->limbs[i]=SM(s);
    if(n>0 && x->limbs[n-1]==0) x->limbs[n-1]=1;     /* keep it n limbs */
    while(x->n_limbs>0 && x->limbs[x->n_limbs-1]==0) x->n_limbs--;
}
/* Independent O(n^2) schoolbook in base 2^LIMB_BITS — the differential
 * reference for the GPU CRT-NTT bigint_mul. Limb-generic: uses bigint.h's
 * width-correct mul_limb (lib test_arith 56/56 verified at LB=64 AND 112,
 * and from the representation layer, not the multiply path under test) for
 * the limb*limb -> (lo:hi) product, then a column accumulator wide enough
 * for both widths. For the sweep's max ~2049-limb operands a column gets
 * < ~2^124 (lo+hi over <=na products, plus carry), which fits an
 * unsigned __int128 at LIMB_BITS = 64 and 112 alike. */
static void mul_ref(BigInt *c,const BigInt *a,const BigInt *b){
    int na=a->n_limbs,nb=b->n_limbs,nc=na+nb;
    bigint_resize(c,nc+1); bigint_zero(c); c->n_limbs=nc;
    unsigned __int128 *col=(unsigned __int128*)calloc((size_t)nc+1,
                                                      sizeof(*col));
    for(int i=0;i<na;i++)for(int j=0;j<nb;j++){
        limb_t lo,hi; mul_limb(a->limbs[i],b->limbs[j],&lo,&hi);
        col[i+j]   += (unsigned __int128)lo;
        col[i+j+1] += (unsigned __int128)hi;
    }
    unsigned __int128 cr=0;
    for(int k=0;k<nc;k++){
        unsigned __int128 v=col[k]+cr;
        c->limbs[k]=(limb_t)(v & LIMB_MASK);
        cr=v>>LIMB_BITS;
    }
    free(col);
    while(c->n_limbs>0 && c->limbs[c->n_limbs-1]==0) c->n_limbs--;
}
static int eq(const BigInt*x,const BigInt*y){
    if(x->n_limbs!=y->n_limbs) return 0;
    for(int i=0;i<x->n_limbs;i++) if(x->limbs[i]!=y->limbs[i]) return 0;
    return 1;
}

int main(int argc,char**argv){
    /* GPU-SAFETY: size the engine to the LARGEST n_coeffs this test uses,
     * not a blanket 2^20.  Large-N section caps na+nb ≤ ~2^16, so log_n=16
     * (N=65536) is sufficient; allow an override but clamp to [12,18] so a
     * stray arg can never request a multi-GiB twiddle alloc.  ntt_mul_init
     * itself also rejects anything outside [3,22] (2026-05-16 hardening). */
    int max_log_n = (argc>1)?atoi(argv[1]):16;
    if (max_log_n < 12) max_log_n = 12;
    if (max_log_n > 18) max_log_n = 18;
    uint64_t s=0x243F6A8885A308D3ULL;

    /* B: per-size engine throughput curve. Re-inits the engine at each log_n
     * so every point uses its natural transform size (the engine otherwise
     * uses one fixed N for all multiplies). `./test_polymul_sweep <x> bench`. */
    if (argc>2 && argv[2][0]=='b') {
        int Ls[]={10,12,14,16,18};
        printf("# log_n,N,limbs,iters,ms_per_mul,GBps_moved_incl_compute\n");
        for (int li=0; li<5; li++) {
            int L=Ls[li];
            ntt_mul_init(L);
            int Nn=1<<L, n=Nn/2-8;
            BigInt a=bigint_alloc(n+4), b=bigint_alloc(n+4), g=bigint_alloc(2*n+8);
            randbig(&a,n,&s); randbig(&b,n,&s);
            int it = L<=12?20:(L<=15?8:4);
            struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
            for (int k=0;k<it;k++) bigint_mul(&g,&a,&b);
            clock_gettime(CLOCK_MONOTONIC,&t1);
            double ms=((t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)*1e-6)/it;
            double gb=4.0*3.0*(double)Nn*8.0/1e9;        /* 4 primes×(2 H2D+1 D2H)×N×8B */
            printf("%d,%d,%d,%d,%.4f,%.1f\n",L,Nn,n,it,ms,gb/(ms*1e-3));
            bigint_free(&a);bigint_free(&b);bigint_free(&g);
            ntt_mul_teardown();
        }
        return 0;
    }

    ntt_mul_init(max_log_n);
    long pass=0,fail=0;

    /* Differential vs schoolbook: sizes crossing log_n 8..12 boundaries.   */
    int diff_sizes[]={50,65,66,80,100,127,128,129,200,256,257,400,512,513,
                      700,1000,1024,1025,1500,2000,2048,2049};
    int nd=(int)(sizeof(diff_sizes)/sizeof(diff_sizes[0]));
    printf("== differential (GPU bigint_mul vs CPU schoolbook) ==\n");
    for(int ia=0;ia<nd;ia++)for(int ib=0;ib<nd;ib++){
        if((ia+ib)&1) continue;                 /* ~half the grid: keep it bounded */
        int na=diff_sizes[ia], nb=diff_sizes[ib];
        for(int tr=0;tr<3;tr++){
            BigInt a=bigint_alloc(na+4),b=bigint_alloc(nb+4);
            BigInt g=bigint_alloc(na+nb+8),e=bigint_alloc(na+nb+8);
            randbig(&a,na,&s); randbig(&b,nb,&s);
            bigint_mul(&g,&a,&b); mul_ref(&e,&a,&b);
            if(eq(&g,&e)) pass++; else { fail++;
                if(fail<=8){ int d=-1;
                    int lim=g.n_limbs>e.n_limbs?g.n_limbs:e.n_limbs;
                    for(int i=0;i<lim;i++){uint64_t gv=i<g.n_limbs?g.limbs[i]:0,
                        ev=i<e.n_limbs?e.limbs[i]:0; if(gv!=ev){d=i;break;}}
                    fprintf(stderr,"  FAIL na=%d nb=%d tr=%d gn=%d en=%d firstdiff@%d\n",
                            na,nb,tr,g.n_limbs,e.n_limbs,d);} }
            bigint_free(&a);bigint_free(&b);bigint_free(&g);bigint_free(&e);
        }
    }
    printf("  differential: %ld pass, %ld fail\n",pass,fail);

    /* Large-N structural invariants (schoolbook too slow): commutativity +
     * degree bound.  GPU-SAFETY: pair sizes so na+nb stays < 2^max_log_n
     * (engine uses fixed N=2^log_n and truncates n_coeffs>N). With the
     * default log_n=16 (N=65536) every pair below is ≤ ~60000.            */
    long ip=0,iff=0;
    int Ncap = 1 << max_log_n;
    int big[]={3000,5000,8000,12000,18000,24000,28000};
    int nb2=(int)(sizeof(big)/sizeof(big[0]));
    printf("== large-N invariants (commutativity + degree bound, N=%d) ==\n",Ncap);
    for(int i=0;i<nb2;i++){
        int na=big[i], nb=big[(nb2-1)-i];
        if(na+nb >= Ncap-8) continue;        /* never exceed the transform */
        BigInt a=bigint_alloc(na+4),b=bigint_alloc(nb+4);
        BigInt ab=bigint_alloc(na+nb+8),ba=bigint_alloc(na+nb+8);
        randbig(&a,na,&s); randbig(&b,nb,&s);
        bigint_mul(&ab,&a,&b); bigint_mul(&ba,&b,&a);
        int ok = eq(&ab,&ba) && ab.n_limbs<=na+nb;
        /* tail beyond na+nb-1 must not exist (trimmed) */
        if(ok) ip++; else { iff++;
            fprintf(stderr,"  FAIL na=%d nb=%d comm=%d abn=%d ban=%d cap=%d\n",
                na,nb,eq(&ab,&ba),ab.n_limbs,ba.n_limbs,na+nb); }
        bigint_free(&a);bigint_free(&b);bigint_free(&ab);bigint_free(&ba);
    }
    printf("  large-N: %ld pass, %ld fail\n",ip,iff);

    ntt_mul_teardown();
    long tf=fail+iff;
    printf("%s  (diff %ld/%ld, large %ld/%ld)\n",
        tf? "=== POLYMUL SWEEP FAILED ===":"=== POLYMUL SWEEP PASS ===",
        pass,pass+fail,ip,ip+iff);
    return tf?1:0;
}
