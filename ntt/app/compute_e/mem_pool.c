/*
 * mem_pool.c — Free-list slab pool for BigInt limb buffers.
 *
 * binary_split churns thousands of short-lived BigInts as the recursion unwinds
 * (every internal node allocates B, P, and a P_L*B_R temporary). This pool
 * recycles their limb allocations so the hot path pays an O(1) list op instead
 * of a malloc/free pair per node.
 *
 * SINGLE-THREADED: binary_split recurses on one thread; its large multiplies go
 * through bigint_mul -> ntt_bigint_mul, which serialize on their own GPU mutex
 * and never touch this pool. No locking is therefore required.
 *
 * OWNERSHIP: mem_pool_get returns a plain malloc-family buffer (managed==0,
 * zeroed, matching the old bigint_alloc behaviour). Buffers that are never
 * returned via mem_pool_put — the root P/B results handed back to main.c — leave
 * the pool and are released by the caller's bigint_free (a standard free()), so
 * that stays correct. mem_pool_put recycles the buffer for a later get; the free
 * list is capped (MEMPOOL_MAX_SLABS) so both the best-fit scan cost and the
 * resident pool memory stay bounded on large runs — an over-cap put just frees.
 */

#include "mem_pool.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Cap on recycled buffers kept live. binary_split's simultaneously-free set is
 * ~O(recursion depth) = tens of buffers even at multi-million-term inputs, so a
 * small fixed cap captures essentially all the churn while keeping get() O(1). */
#define MEMPOOL_MAX_SLABS 64

typedef struct { limb_t *ptr; int cap; } Slab;

struct MemPool {
    Slab slab[MEMPOOL_MAX_SLABS];
    int  n_free;
};

MemPool *mem_pool_create(void)
{
    MemPool *p = (MemPool *)malloc(sizeof(MemPool));
    if (!p) { fprintf(stderr, "mem_pool_create: OOM\n"); exit(1); }
    p->n_free = 0;
    return p;
}

void mem_pool_destroy(MemPool *pool)
{
    if (!pool) return;
    for (int i = 0; i < pool->n_free; i++)
        free(pool->slab[i].ptr);
    free(pool);
}

/* Remove free-list entry i by swapping the last element into its slot. */
static void slab_remove(MemPool *pool, int i)
{
    pool->slab[i] = pool->slab[pool->n_free - 1];
    pool->n_free--;
}

BigInt mem_pool_get(MemPool *pool, int min_cap)
{
    if (min_cap < 1) min_cap = 1;

    BigInt a;
    a.n_limbs = 0;
    a.managed = 0;

    if (pool && pool->n_free > 0) {
        /* Best-fit: the smallest recycled buffer with cap >= min_cap avoids
         * handing a large buffer to a tiny request (which would then be a poor
         * fit for future reuse). */
        int best = -1;
        for (int i = 0; i < pool->n_free; i++)
            if (pool->slab[i].cap >= min_cap &&
                (best < 0 || pool->slab[i].cap < pool->slab[best].cap))
                best = i;

        if (best >= 0) {
            a.limbs = pool->slab[best].ptr;
            a.cap   = pool->slab[best].cap;
            slab_remove(pool, best);
            memset(a.limbs, 0, (size_t)a.cap * sizeof(limb_t));
            return a;
        }

        /* None large enough: grow the largest free buffer in place (reuse its
         * allocation instead of leaking it alongside a fresh calloc). */
        int big = 0;
        for (int i = 1; i < pool->n_free; i++)
            if (pool->slab[i].cap > pool->slab[big].cap) big = i;
        limb_t *grown = (limb_t *)realloc(pool->slab[big].ptr,
                                          (size_t)min_cap * sizeof(limb_t));
        if (!grown) { fprintf(stderr, "mem_pool_get: OOM\n"); exit(1); }
        slab_remove(pool, big);
        a.limbs = grown;
        a.cap   = min_cap;
        memset(a.limbs, 0, (size_t)a.cap * sizeof(limb_t));
        return a;
    }

    a.limbs = (limb_t *)calloc((size_t)min_cap, sizeof(limb_t));
    if (!a.limbs) { fprintf(stderr, "mem_pool_get: OOM\n"); exit(1); }
    a.cap = min_cap;
    return a;
}

void mem_pool_put(MemPool *pool, BigInt *b)
{
    if (!b || !b->limbs) return;

    /* The pool recycles only plain malloc'd buffers. A managed BigInt
     * (hipMallocManaged) must be released via bigint_free -> hip_free_limbs;
     * binary_split never routes one here, but stay correct if that changes.
     * A NULL pool (defensive) also falls back to the normal free path. */
    if (b->managed || !pool || pool->n_free >= MEMPOOL_MAX_SLABS) {
        bigint_free(b);
        return;
    }

    pool->slab[pool->n_free].ptr = b->limbs;
    pool->slab[pool->n_free].cap = b->cap;
    pool->n_free++;

    /* Detach so the caller cannot double-free the now-pooled buffer. */
    b->limbs   = NULL;
    b->n_limbs = 0;
    b->cap     = 0;
    b->managed = 0;
}
