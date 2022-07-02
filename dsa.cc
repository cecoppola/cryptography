// Copyright 2022 C. E. Coppola
// This program generates a public and a private DSA key, and outputs them.
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
#include <nettle/dsa.h>
#include <nettle/sexp.h>
#include <nettle/yarrow.h>
#include <getopt.h>
#include <gmpxx.h>
using namespace std;

//Uses the DSA algorithm to generate keys.
int main(int argc, char **argv) {

  random_device rd;
  srand (rd());
  const uint8_t seed = rand();
  struct yarrow256_ctx yarrow256_ctx;
  size_t ranlength = 32;
  yarrow256_seed(&yarrow256_ctx, ranlength, &seed);

  unsigned int q_bits =  160;
  unsigned int p_bits = 1024;

  struct dsa_params param1;
  dsa_params_init(&param1);

  dsa_generate_params(&param1, (void *) &yarrow256_ctx, (nettle_random_func *) yarrow256_random, NULL, NULL, p_bits, q_bits);

  mpz_t pub;
  mpz_init(pub);
  mpz_t key;
  mpz_init(key);

  dsa_generate_keypair(&param1, pub, key, (void *) &yarrow256_ctx, (nettle_random_func *) yarrow256_random);

  mpz_class primeq;
  primeq = mpz_class(param1.q);
  string qstring = primeq.get_str();
  cout << "Prime 1: " << qstring << "\n\n";

  mpz_class primep;
  primep = mpz_class(param1.p);
  string pstring = primep.get_str();
  cout << "Prime 2: " << pstring << "\n\n";

  mpz_class generator;
  generator = mpz_class(param1.g);
  string gstring = generator.get_str();
  cout << "Generator: " << gstring << "\n";

}

