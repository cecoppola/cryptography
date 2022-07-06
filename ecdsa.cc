// Copyright 2022 C. E. Coppola
// This program generates a public and a private ECDSA key, and outputs them.
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <array>
#include <fstream>
#include <sstream>
#include <random>
#include <nettle/buffer.h>
#include <nettle/ecc-curve.h>
#include <nettle/ecdsa.h>
#include <nettle/sexp.h>
#include <nettle/yarrow.h>
#include <getopt.h>
#include <gmpxx.h>
using namespace std;

//Uses the ECDSA algorithm to generate keys.
int main(int argc, char **argv) {

  random_device rd;
  srand (rd());
  const uint8_t seed = rand();
  struct yarrow256_ctx yarrow256_ctx;
  size_t ranlength = 32;
  yarrow256_seed(&yarrow256_ctx, ranlength, &seed);

  //const struct ecc_curve* ecc = nettle_get_secp_521r1();
  const struct ecc_curve* ecc = nettle_get_secp_256r1();

  struct ecc_point pub;
  ecc_point_init(&pub, ecc);
  mpz_t x;
  mpz_init(x);
  mpz_t y;
  mpz_init(y);
  ecc_point_set(&pub, x, y);

  struct ecc_scalar key;
  ecc_scalar_init(&key, ecc);
  mpz_t z;
  mpz_init(z);
  ecc_scalar_set(&key, z);

  ecdsa_generate_keypair(&pub, &key, (void *) &yarrow256_ctx, (nettle_random_func *) yarrow256_random);

  ecc_point_get(&pub, x, NULL);
  gmp_printf("Abscissa = %66Zx\n", x);

  ecc_point_get(&pub, NULL, y);
  gmp_printf("Ordinate = %66Zx\n", y);

  ecc_scalar_get(&key, z);
  gmp_printf("Key =    %68Zx\n", z);

  ecc_point_clear(&pub);
  ecc_scalar_clear(&key);

}
