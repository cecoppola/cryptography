#pragma once
/*
 * mem_pool.h — BigInt scratch-buffer pool for binary splitting.
 *
 * Provides a get/put interface so callers avoid repeated malloc/free in the
 * recursion hot path.  The current implementation is a thin pass-through to
 * bigint_alloc/free; replace the internals with a slab allocator if malloc
 * overhead becomes measurable.
 *
 * Usage:
 *   MemPool *pool = mem_pool_create();
 *   BigInt tmp = mem_pool_get(pool, needed_cap);
 *   // ... use tmp ...
 *   mem_pool_put(pool, &tmp);
 *   mem_pool_destroy(pool);
 */

#include "../../lib/arith/bigint.h"

typedef struct MemPool MemPool;

MemPool *mem_pool_create(void);
void     mem_pool_destroy(MemPool *pool);

/* Return a BigInt with cap >= min_cap limbs (contents undefined). */
BigInt   mem_pool_get(MemPool *pool, int min_cap);

/* Return b to the pool; b->limbs may be reused by subsequent mem_pool_get. */
void     mem_pool_put(MemPool *pool, BigInt *b);
