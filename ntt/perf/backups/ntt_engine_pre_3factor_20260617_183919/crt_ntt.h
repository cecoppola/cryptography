#pragma once
/*
 * crt_ntt.h — Shared types for the CRT-NTT pipeline
 *
 * Included by garner.hip and main.hip. Centralises U256 and GarnerConsts
 * to avoid duplicate-typedef link errors when both translation units are
 * compiled into the same binary.
 */

#include <stdint.h>

/* Stockham LDS padding — must match ntt_kernel.hip definitions. */
#define STOK_PAD(i)  ((i) + ((i) >> 5))
#define STOK_STRIDE  1057

/* 256-bit integer, little-endian 64-bit limbs. w[0] is least-significant. */
typedef struct { uint64_t w[4]; } U256;

/* Garner CRT precomputed constants (computed once at init, read-only after). */
typedef struct {
    /* Modular inverses (Fermat: a^{p-2} mod p) */
    uint64_t inv_p0_p1;    /* p0^{-1} mod p1 */
    uint64_t inv_p0_p2;    /* p0^{-1} mod p2 */
    uint64_t inv_p0_p3;    /* p0^{-1} mod p3 */
    uint64_t inv_p1_p2;    /* p1^{-1} mod p2 */
    uint64_t inv_p1_p3;    /* p1^{-1} mod p3 */
    uint64_t inv_p2_p3;    /* p2^{-1} mod p3 */
    /* Pre-reduced products (avoid % in hot loop) */
    uint64_t p0_mod_p2;    /* P0 mod P2 */
    uint64_t p0_mod_p3;    /* P0 mod P3 */
    uint64_t p1_mod_p3;    /* P1 mod P3 */
    uint64_t p0p1_mod_p3;  /* (P0*P1) mod P3 */
    /* Multi-precision products for 256-bit Garner assembly */
    U256 P0_256;            /* P0 as U256 */
    U256 P0P1_256;          /* P0*P1 as U256 (128-bit product) */
    U256 P0P1P2_256;        /* P0*P1*P2 as U256 (~192-bit product) */
} GarnerConsts;
