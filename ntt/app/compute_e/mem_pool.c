/*
 * mem_pool.c — Thin pass-through pool (replace internals with slab if needed).
 *
 * All calls delegate directly to bigint_alloc / bigint_free.
 * The MemPool struct exists only to satisfy the typed API; swap in a real
 * allocator by replacing this file without changing any callers.
 */

#include "mem_pool.h"
#include <stdlib.h>
#include <stdio.h>

struct MemPool { int dummy; };

MemPool *mem_pool_create(void)
{
    MemPool *p = malloc(sizeof(MemPool));
    if (!p) { fprintf(stderr, "mem_pool_create: OOM\n"); exit(1); }
    return p;
}

void mem_pool_destroy(MemPool *pool)
{
    free(pool);
}

BigInt mem_pool_get(MemPool *pool, int min_cap)
{
    (void)pool;
    return bigint_alloc(min_cap > 0 ? min_cap : 1);
}

void mem_pool_put(MemPool *pool, BigInt *b)
{
    (void)pool;
    bigint_free(b);
}
