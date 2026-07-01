/* bench_g9.hip — G9 runtime A/B: C++ template<PIDX> dispatch vs pure-C
 * runtime switch(pidx). Times the forward Stockham NTT (launch_stockham_ntt vs
 * launch_stockham_ntt_cs) on the same buffer + twiddles, prime 0, even log_n.
 *
 * Build twice (each links ONLY its engine object — the launchers don't clash:
 * the cswitch ones carry a _cs suffix):
 *   template: hipcc ... bench_g9.hip ntt_kernel_dev.o            -o bench_g9_tmpl
 *   cswitch:  hipcc ... -DUSE_CSWITCH bench_g9.hip ntt_kernel_cswitch_dev.o -o bench_g9_cs
 *
 * CAVEAT: ntt_kernel_cswitch.hip is a STALE port (pre-I13: it uses a separate
 * cross_twiddle_kernel_cs, not the fused xtranspose). The two leaf Stockham
 * passes are the SAME algorithm and dominate the runtime, so this isolates the
 * dispatch cost well; the single non-fused transpose pass is a minor confound
 * that, if anything, handicaps the cswitch number. */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "primes.h"
#include "shoup.h"
#include "crt_ntt.h"

#define HIP_CHECK(e) do{ hipError_t _e=(e); if(_e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(_e)); exit(1);} }while(0)

#ifdef USE_CSWITCH
extern void launch_stockham_ntt_cs(int, uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
#  define LAUNCH(d,ts,tc,ln,s) launch_stockham_ntt_cs(0,(d),(ts),(tc),(ln),(s))
#  define VARIANT "cswitch  runtime switch(pidx)"
#else
template<int PIDX> void launch_stockham_ntt(uint64_t*, const ShoupPair*, const ShoupPair*, int, hipStream_t);
#  define LAUNCH(d,ts,tc,ln,s) launch_stockham_ntt<0>((d),(ts),(tc),(ln),(s))
#  define VARIANT "template<PIDX>  compile-time"
#endif

/* Twiddle builders copied verbatim from arith/ntt_mul.hip (dev arg dropped). */
static void fill_flat(ShoupPair* h, uint64_t p, uint64_t g, int log_m){
    int M=1<<log_m; uint64_t omega=modpow(g,(p-1)/(uint64_t)M,p), cur=1;
    for(int k=0;k<M/2;k++){ h[k]=make_shoup(cur,p); cur=(uint64_t)((__uint128_t)cur*omega%p);} }
static ShoupPair* make_tw_stok(uint64_t p, uint64_t g, int log_n){
    int a, n1, n2;
    if(log_n<=10){ a=(1<<log_n)/2; } else { n1=log_n/2; n2=log_n-n1; a=(1<<(n1-1))+(1<<(n2-1)); }
    ShoupPair* h=(ShoupPair*)malloc((size_t)a*sizeof(ShoupPair));
    if(log_n<=10){ fill_flat(h,p,g,log_n); } else { n1=log_n/2; n2=log_n-n1;
        fill_flat(h,p,g,n1); fill_flat(h+(1<<(n1-1)),p,g,n2); }
    ShoupPair* d; HIP_CHECK(hipMalloc(&d,(size_t)a*sizeof(ShoupPair)));
    HIP_CHECK(hipMemcpy(d,h,(size_t)a*sizeof(ShoupPair),hipMemcpyHostToDevice)); free(h); return d; }
static ShoupPair* make_tw_cross(uint64_t p, uint64_t g, int log_n){
    int n1=log_n/2, n2=log_n-n1, N1=1<<n1, N2=1<<n2;
    ShoupPair* h=(ShoupPair*)malloc((size_t)(N1*N2)*sizeof(ShoupPair));
    uint64_t omega=modpow(g,(p-1)/(uint64_t)(N1*N2),p);
    for(int r=0;r<N1;r++){ uint64_t rb=modpow(omega,(uint64_t)r,p), cur=1;
        for(int c=0;c<N2;c++){ h[r*N2+c]=make_shoup(cur,p); cur=(uint64_t)((__uint128_t)cur*rb%p);} }
    ShoupPair* d; HIP_CHECK(hipMalloc(&d,(size_t)(N1*N2)*sizeof(ShoupPair)));
    HIP_CHECK(hipMemcpy(d,h,(size_t)(N1*N2)*sizeof(ShoupPair),hipMemcpyHostToDevice)); free(h); return d; }

int main(int argc, char** argv){
    int log_n = argc>1 ? atoi(argv[1]) : 20;
    int iters = argc>2 ? atoi(argv[2]) : 300;
    size_t N = (size_t)1<<log_n;
    uint64_t p = PRIMES[0], g = PRIM_ROOT[0];
    HIP_CHECK(hipSetDevice(0));
    uint64_t* h=(uint64_t*)malloc(N*8); uint64_t s=0x123456789ULL;
    for(size_t i=0;i<N;i++){ s=s*6364136223846793005ULL+1; h[i]=s%p; }
    uint64_t* d; HIP_CHECK(hipMalloc(&d,N*8)); HIP_CHECK(hipMemcpy(d,h,N*8,hipMemcpyHostToDevice));
    ShoupPair* ts=make_tw_stok(p,g,log_n); ShoupPair* tc=make_tw_cross(p,g,log_n);
    hipStream_t st; HIP_CHECK(hipStreamCreate(&st));
    for(int w=0;w<20;w++) LAUNCH(d,ts,tc,log_n,st);          /* warmup */
    HIP_CHECK(hipStreamSynchronize(st));
    hipEvent_t e0,e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
    HIP_CHECK(hipEventRecord(e0,st));
    for(int it=0;it<iters;it++) LAUNCH(d,ts,tc,log_n,st);
    HIP_CHECK(hipEventRecord(e1,st)); HIP_CHECK(hipEventSynchronize(e1));
    float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,e0,e1));
    printf("G9 bench  variant=[%s]  log_n=%d  iters=%d  total=%.2f ms  per-NTT=%.5f ms\n",
           VARIANT, log_n, iters, ms, ms/iters);
    return 0;
}
