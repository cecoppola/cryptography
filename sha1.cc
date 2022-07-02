// Copyright 2022 C. E. Coppola
// This program takes a message from standard input, hashes the message using SHA-1 and prints the output.

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <nettle/sha1.h>
using namespace std;

//Prints the hash of the message in hexidecimal with spaces for readability.
static void display_hex(unsigned length, uint8_t *data) {
  for (int i = 0; i<length; i++) printf("%02x ", data[i]);
  printf("\n");
}

//Takes a message and hashes it using the SHA-1 algorithm.
int main(int argc, char **argv) {

  struct sha1_ctx ctx;
  sha1_init(&ctx);

  stringstream fileinput;
  fileinput.str("");
  fileinput << cin.rdbuf();
  string textin = fileinput.str();

  size_t textsize = textin.length();
  uint8_t message[textsize] = {0};

  for(int i=0; i<=textsize; i++) message[i] = textin[i];
  sha1_update (&ctx, SHA1_DIGEST_SIZE, message);

  uint8_t digest[SHA1_DIGEST_SIZE] = {0};
  sha1_digest(&ctx, SHA1_DIGEST_SIZE, digest);

  display_hex(SHA1_DIGEST_SIZE, digest);
 
  return 1; 
}

