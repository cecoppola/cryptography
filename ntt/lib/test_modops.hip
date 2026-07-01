/*
 * test_modops.hip — Exhaustive correctness test for the templated modular
 * primitives that the NTT butterfly / Hadamard / Garner paths rely on:
 *   addmod<P>, submod<P>, mulmod<P>, montgomery_mul<P>, shoup_mul<P>
 * for all four primes P0..P3.
 *
 * Companion to test_reduce.c (which covers reduce_p0..p3). Compiled with
 * hipcc (C++ frontend for the templates) but runs entirely on the HOST —
 * no kernel launches, no GPU contention. Each primitive is checked against
 * a __uint128_t reference over: all-pairs corners, engineered sweeps that
 * straddle 2^64 / p (the class that hid the reduce_p0/p1 bugs), and
 * millions of uniform-random operands.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "primes.h"
#include "shoup.h"

static uint64_t sm64(uint64_t *s){
    uint64_t z=(*s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
    z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}

template<int P> static const char* pname(){return "?";}
template<> const char* pname<0>(){return "P0";}
template<> const char* pname<1>(){return "P1";}
template<> const char* pname<2>(){return "P2";}
template<> const char* pname<3>(){return "P3";}

template<int PIDX>
static long test_prime(void)
{
    const uint64_t p = PRIMES[PIDX];
    long bad = 0;
    /* Montgomery R mod p for the reference-free REDC invariant: R = 2^64. */
    uint64_t Rmodp = (uint64_t)(((__uint128_t)1 << 64) % p);

    uint64_t cs[] = {0,1,2,3, p-1,p-2,p-3, p/2, p/2+1,
        (uint64_t)((__uint128_t)1<<63)%p,
        (uint64_t)((1ULL<<32))%p, (uint64_t)((1ULL<<32)-1),
        (uint64_t)((1ULL<<40))%p, (uint64_t)((1ULL<<24))%p,
        (uint64_t)(-(__int128)p)%p,           /* 2^64 mod p region */
        p-(1ULL<<24), p-(1ULL<<32), p-(1ULL<<40)%p};
    int nc=(int)(sizeof(cs)/sizeof(cs[0]));

    #define CK_ADD(a,b) do{ uint64_t g=addmod<PIDX>((a),(b)); \
        uint64_t e=(uint64_t)(((__uint128_t)(a)+(b))%p); \
        if(g!=e){ if(bad<6)fprintf(stderr,"  %s addmod a=%016llx b=%016llx g=%016llx e=%016llx\n",\
            pname<PIDX>(),(unsigned long long)(a),(unsigned long long)(b),\
            (unsigned long long)g,(unsigned long long)e); bad++; } }while(0)
    #define CK_SUB(a,b) do{ uint64_t g=submod<PIDX>((a),(b)); \
        uint64_t e=(uint64_t)(((__uint128_t)(a)+p-(b))%p); \
        if(g!=e){ if(bad<6)fprintf(stderr,"  %s submod a=%016llx b=%016llx g=%016llx e=%016llx\n",\
            pname<PIDX>(),(unsigned long long)(a),(unsigned long long)(b),\
            (unsigned long long)g,(unsigned long long)e); bad++; } }while(0)
    #define CK_MUL(a,b) do{ uint64_t g=mulmod<PIDX>((a),(b)); \
        uint64_t e=(uint64_t)(((__uint128_t)(a)*(b))%p); \
        if(g!=e){ if(bad<6)fprintf(stderr,"  %s mulmod a=%016llx b=%016llx g=%016llx e=%016llx\n",\
            pname<PIDX>(),(unsigned long long)(a),(unsigned long long)(b),\
            (unsigned long long)g,(unsigned long long)e); bad++; } }while(0)
    /* Reference-free REDC invariant: g is the unique value in [0,p) with
     * g * (2^64 mod p) ≡ a*b (mod p).  Needs no modular inverse, so it cannot
     * mask or fake a montgomery_mul defect. */
    #define CK_MON(a,b) do{ uint64_t g=montgomery_mul<PIDX>((a),(b)); \
        __uint128_t lhs=(__uint128_t)g*Rmodp%p; \
        __uint128_t rhs=(__uint128_t)(a)*(b)%p; \
        if(g>=p || lhs!=rhs){ if(bad<6)fprintf(stderr,\
            "  %s mont a=%016llx b=%016llx g=%016llx (g*R)%%p=%016llx (a*b)%%p=%016llx\n",\
            pname<PIDX>(),(unsigned long long)(a),(unsigned long long)(b),\
            (unsigned long long)g,(unsigned long long)(uint64_t)lhs,\
            (unsigned long long)(uint64_t)rhs); bad++; } }while(0)
    #define CK_SHO(t_,w_) do{ ShoupPair sp=make_shoup((w_),p); \
        uint64_t g=shoup_mul<PIDX>((t_),sp.w,sp.w_inv); \
        uint64_t e=(uint64_t)(((__uint128_t)(w_)*(t_))%p); \
        if(g!=e){ if(bad<6)fprintf(stderr,"  %s shoup t=%016llx w=%016llx g=%016llx e=%016llx\n",\
            pname<PIDX>(),(unsigned long long)(t_),(unsigned long long)(w_),\
            (unsigned long long)g,(unsigned long long)e); bad++; } }while(0)

    /* All-pairs corners */
    for(int i=0;i<nc;i++)for(int j=0;j<nc;j++){
        uint64_t a=cs[i]%p, b=cs[j]%p;
        CK_ADD(a,b); CK_SUB(a,b); CK_MUL(a,b); CK_MON(a,b); CK_SHO(a,b);
    }
    /* Engineered: straddle 2^64 and straddle p. a near (2^64-p), b near p-1;
     * submod a<b with a tiny so a+p overflows; shoup t swept with w=p-1. */
    for(uint64_t k=0;k<(1ULL<<20);k++){
        uint64_t a1=(p-1)-k, b1=p-1;
        CK_ADD(a1,b1); CK_ADD(k,b1); CK_ADD(b1,k);
        CK_SUB(k,(p-1)-k%p); CK_SUB((p-1)-k,k); CK_SUB(k,p-1);
        CK_MUL(b1,(p-1)-k); CK_MUL(k,b1);
        CK_MON(b1,(p-1)-k); CK_MON(k,b1);
        CK_SHO((p-1)-k,b1); CK_SHO(k,b1); CK_SHO(b1,(p-1)-k);
    }
    /* Uniform random */
    uint64_t s=0xC2B2AE3D27D4EB4FULL ^ (uint64_t)(PIDX*0x9E3779B1u);
    for(long n=0;n<3000000;n++){
        uint64_t a=sm64(&s)%p, b=sm64(&s)%p;
        CK_ADD(a,b); CK_SUB(a,b); CK_MUL(a,b); CK_MON(a,b); CK_SHO(a,b);
    }
    printf("%s (%016llx): %s  (%ld mismatches)\n",
        pname<PIDX>(),(unsigned long long)p, bad?"FAIL":"PASS", bad);
    return bad;
}

int main(void)
{
    long f=0;
    f+=test_prime<0>();
    f+=test_prime<1>();
    f+=test_prime<2>();
    f+=test_prime<3>();
    printf("%s\n", f? "=== MODOPS TEST FAILED ===":"=== all modops tests PASS ===");
    return f?1:0;
}
