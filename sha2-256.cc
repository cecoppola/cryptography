// Copyright 2022 C. E. Coppola
// This program takes a message from standard input, hashes the message using SHA-2, with a 256-bit key, and prints the output.

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <nettle/sha2.h>
using namespace std;

//Prints the hash of the message in hexidecimal with spaces for readability.
static void display_hex(unsigned length, uint8_t *data) {
  for (int i = 0; i<length; i++) printf("%02x ", data[i]);
  printf("\n");
}

//Takes a message and hashes it using the SHA-2 algorithm.
int main(int argc, char **argv) {

  struct sha256_ctx ctx;
  sha256_init(&ctx);

  stringstream fileinput;
  fileinput.str("");
  fileinput << cin.rdbuf();
  string textin = fileinput.str();

  size_t textsize = textin.length();
  uint8_t message[textsize] = {0};

  for(int i=0; i<=textsize; i++) message[i] = textin[i];
  sha256_update (&ctx, SHA256_DIGEST_SIZE, message);

  uint8_t digest[SHA256_DIGEST_SIZE] = {0};
  sha256_digest(&ctx, SHA256_DIGEST_SIZE, digest);

  display_hex(SHA256_DIGEST_SIZE, digest);
 
  return 1; 
}

