// Copyright 2022 C. E. Coppola
// This program generates a public and a private RSA 4096-bit key, and outputs them.
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
#include <nettle/rsa.h>
#include <nettle/sexp.h>
#include <nettle/yarrow.h>
#include <getopt.h>
#include <gmpxx.h>
using namespace std;

//Uses the RSA algorithm to generate keys.
int main(int argc, char **argv) {

  struct rsa_public_key pub;
  rsa_public_key_init(&pub);
  rsa_public_key_prepare(&pub);

  struct rsa_private_key key;
  rsa_private_key_init(&key);
  rsa_private_key_prepare(&key);

  random_device rd;
  srand (rd());
  const uint8_t seed = rand();
  struct yarrow256_ctx yarrow256_ctx;
  size_t ranlength = 32;
  yarrow256_seed(&yarrow256_ctx, ranlength, &seed);

  unsigned long key_size = DEFAULT_KEYSIZE;
  unsigned long key_e = 24;
  rsa_generate_keypair(&pub, &key, (void *) &yarrow256_ctx, (nettle_random_func *) yarrow256_random, NULL, NULL, key_size, key_e);

  mpz_class exponent;
  exponent = mpz_class(pub.e);
  string exponentstring = exponent.get_str();
  cout << "Exponent: " << exponentstring << "\n\n";

  mpz_class factor1;
  factor1 = mpz_class(key.p);
  string factor1string = factor1.get_str();
  cout << "Prime 1: " << factor1string << "\n\n";

  mpz_class factor2;
  factor2 = mpz_class(key.q);
  string factor2string = factor2.get_str();
  cout << "Prime 2: " << factor2string << "\n\n";

  mpz_class integer;
  integer = mpz_class(key.d);
  string integerstring = integer.get_str();
  cout << "Inverse: " << integerstring << "\n";

  rsa_public_key_clear(&pub);
  rsa_private_key_clear(&key);
}

