#pragma once
/*
 * mem_pool.h — BigInt scratch-buffer pool for binary splitting.
 *
 * Provides a get/put interface so callers avoid repeated malloc/free in the
 * recursion hot path.  Implemented as a bounded free-list slab pool (see
 * mem_pool.c): put recycles a limb buffer, get reuses a best-fit one.
 * Single-threaded (binary_split recurses on one thread).
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

/* Return a zeroed BigInt with cap >= min_cap limbs, n_limbs=0, managed=0. */
BigInt   mem_pool_get(MemPool *pool, int min_cap);

/* Return b to the pool for reuse by a later mem_pool_get, and detach it
 * (b->limbs set NULL) so the caller cannot double-free the pooled buffer. */
void     mem_pool_put(MemPool *pool, BigInt *b);
