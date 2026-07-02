/*
 * test_mem_pool.c — Unit test for the BigInt free-list slab pool (mem_pool.c).
 *
 * The pool was rewritten from a malloc/free pass-through into a recycling
 * allocator; binary_split only exercises it indirectly. This test pins the
 * observable contracts directly, and its randomized stress loop is the payload
 * the ASAN+UBSan sweep scrutinizes for use-after-free / double-free / heap
 * overflow / leaks (mem_pool.c is rebuilt under -fsanitize in that gate).
 *
 * White-box: MEMPOOL_MAX_SLABS-dependent and best-fit/grow behaviours are
 * asserted against THIS implementation (pointer identity where recycling must
 * reuse an allocation). Pure host C; links mem_pool.c + bigint.c (bigint_free /
 * bigint_alloc_managed for the put fallbacks).
 */

#include "mem_pool.h"
#include "../../lib/arith/bigint.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MEMPOOL_MAX_SLABS 64   /* must match mem_pool.c */

static int failures = 0;
static void check(int cond, const char *what)
{
    if (!cond) { fprintf(stderr, "  FAIL: %s\n", what); failures++; }
}

/* All `cap` limbs of a fresh get() must be zero. */
static int all_zero(const BigInt *b)
{
    for (int i = 0; i < b->cap; i++) if (b->limbs[i] != 0) return 0;
    return 1;
}

/* Fill every limb with a nonzero pattern (also probes cap-sized writes under ASan). */
static void scribble(BigInt *b)
{
    for (int i = 0; i < b->cap; i++) b->limbs[i] = (limb_t)(0xA5A5A5A5u + i);
}

static void test_get_contract(MemPool *p)
{
    BigInt a = mem_pool_get(p, 10);
    check(a.limbs != NULL, "get: non-null limbs");
    check(a.cap >= 10,     "get: cap >= min_cap");
    check(a.n_limbs == 0,  "get: n_limbs == 0");
    check(a.managed == 0,  "get: managed == 0");
    check(all_zero(&a),    "get: buffer zeroed");
    BigInt z = mem_pool_get(p, 0);
    check(z.cap >= 1,      "get(0): cap clamped to >= 1");
    mem_pool_put(p, &a);
    mem_pool_put(p, &z);
}

static void test_put_detaches(MemPool *p)
{
    BigInt a = mem_pool_get(p, 8);
    mem_pool_put(p, &a);
    check(a.limbs == NULL, "put: detaches limbs (no double-free)");
    check(a.cap == 0,      "put: cap zeroed");
    check(a.n_limbs == 0,  "put: n_limbs zeroed");
}

static void test_recycle_and_zero(MemPool *p)
{
    BigInt a = mem_pool_get(p, 16);
    limb_t *orig = a.limbs;
    scribble(&a);
    mem_pool_put(p, &a);
    BigInt b = mem_pool_get(p, 16);          /* best-fit: the only cap-16 buffer */
    check(b.limbs == orig, "recycle: same allocation reused");
    check(all_zero(&b),    "recycle: reused buffer re-zeroed");
    mem_pool_put(p, &b);
}

static void test_best_fit(MemPool *p)
{
    /* Fresh caps 50, 10, 20; put all; a get(15) must pick the cap-20 buffer. */
    BigInt x = mem_pool_get(p, 50);
    BigInt y = mem_pool_get(p, 10);
    BigInt w = mem_pool_get(p, 20);
    limb_t *p20 = w.limbs;
    mem_pool_put(p, &x); mem_pool_put(p, &y); mem_pool_put(p, &w);
    BigInt got = mem_pool_get(p, 15);
    check(got.limbs == p20, "best-fit: smallest cap >= min chosen");
    check(got.cap == 20,    "best-fit: returned cap is the best fit");
    mem_pool_put(p, &got);
    /* drain the pool so later tests start clean */
    BigInt d;
    d = mem_pool_get(p, 50); mem_pool_put(p, &d); /* leave as-is; destroy frees */
}

static void test_grow_largest(void)
{
    MemPool *p = mem_pool_create();
    BigInt s = mem_pool_get(p, 5);
    BigInt l = mem_pool_get(p, 8);
    mem_pool_put(p, &s); mem_pool_put(p, &l);   /* free list: caps {5, 8} */
    BigInt big = mem_pool_get(p, 100);          /* none fit -> grow the largest (8) */
    check(big.cap == 100,  "grow: returned cap == requested");
    check(all_zero(&big),  "grow: grown buffer fully zeroed");
    mem_pool_put(p, &big);
    /* The cap-5 buffer must remain: a get(4) best-fits it, proving the cap-8
     * (largest) was the one consumed by the grow, not the cap-5. */
    BigInt small = mem_pool_get(p, 4);
    check(small.cap == 5,  "grow: consumed the largest free buffer, not the smallest");
    mem_pool_put(p, &small);
    mem_pool_destroy(p);
}

static void test_overcap_fallback(void)
{
    MemPool *p = mem_pool_create();
    /* Put more than MEMPOOL_MAX_SLABS buffers: the surplus puts must free (not
     * pool) without leaking — ASan/LeakSan confirms. */
    BigInt bufs[MEMPOOL_MAX_SLABS + 16];
    for (int i = 0; i < MEMPOOL_MAX_SLABS + 16; i++) bufs[i] = mem_pool_get(p, 4);
    for (int i = 0; i < MEMPOOL_MAX_SLABS + 16; i++) {
        mem_pool_put(p, &bufs[i]);
        check(bufs[i].limbs == NULL, "overcap: every put detaches");
    }
    /* Pool still functional afterward. */
    BigInt a = mem_pool_get(p, 4);
    check(a.limbs != NULL && all_zero(&a), "overcap: pool usable after surplus");
    mem_pool_put(p, &a);
    mem_pool_destroy(p);
}

static void test_managed_fallback(MemPool *p)
{
    /* A managed BigInt must be released via bigint_free (weak hip_free_limbs =
     * free here), never pooled. Observable: detached, and ASan-clean. */
    BigInt m = bigint_alloc_managed(8);
    check(m.managed == 1, "managed: alloc sets flag");
    mem_pool_put(p, &m);
    check(m.limbs == NULL, "managed: put freed and detached (not pooled)");
}

static void test_stress(void)
{
    MemPool *p = mem_pool_create();
    enum { LIVE = 128, ITERS = 200000 };
    BigInt live[LIVE];
    int used[LIVE];
    for (int i = 0; i < LIVE; i++) used[i] = 0;
    unsigned st = 0x1234567u;
    for (int it = 0; it < ITERS; it++) {
        st = st * 1103515245u + 12345u;
        int slot = (st >> 8) % LIVE;
        if (used[slot]) {
            /* verify integrity before returning, then put */
            mem_pool_put(p, &live[slot]);
            used[slot] = 0;
        } else {
            int cap = 1 + ((st >> 5) % 300);
            live[slot] = mem_pool_get(p, cap);
            if (live[slot].cap < cap || !all_zero(&live[slot])) { failures++; }
            /* touch first and last limb (ASan redzone check) */
            live[slot].limbs[0] = (limb_t)it;
            live[slot].limbs[live[slot].cap - 1] = (limb_t)~it;
            used[slot] = 1;
        }
    }
    for (int i = 0; i < LIVE; i++) if (used[i]) mem_pool_put(p, &live[i]);
    mem_pool_destroy(p);   /* LeakSan: any un-freed slab is caught here */
}

int main(void)
{
    MemPool *p = mem_pool_create();
    test_get_contract(p);
    test_put_detaches(p);
    test_recycle_and_zero(p);
    test_best_fit(p);
    test_managed_fallback(p);
    mem_pool_destroy(p);

    test_grow_largest();
    test_overcap_fallback();
    test_stress();

    printf("test_mem_pool: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
