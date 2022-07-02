// Copyright 2022 C. E. Coppola
// This program generates a random key, encrypts a chosen text with Salsa20, and then decrypts it using the same key.

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <nettle/salsa20.h>
using namespace std;

//Generates a random key.
void salsakey(uint8_t* key, size_t max) {
  random_device rd;
  srand (rd());
  for(int i=0; i<max; i++) key[i] = rand()%256;
  return; 
}

//Encrypts and thendes decrypts an input text using Salsa20.
int main(int argc, char **argv) {

  uint8_t keysize = SALSA20_256_KEY_SIZE;
  uint8_t key[keysize] = {0};
  salsakey(key, keysize);
  printf("hex key: ");
  for (int i=0; i<keysize ; i++) {
    printf("%x", key[i]);
  }
  printf("\n");


  uint8_t blocksize = SALSA20_BLOCK_SIZE;

  stringstream fileinput;
  fileinput.str("");
  fileinput << cin.rdbuf();
  string textin = fileinput.str();

  size_t size = textin.length();
  unsigned char inputtext[size+1] = {0};

  for(int i=0; i<size; i++) {
    inputtext[i] = textin[i];
  }

  struct salsa20_ctx ctx;
  salsa20_256_set_key(&ctx, key);

  uint8_t nonce[SALSA20_NONCE_SIZE] = {1,2,3,4,5,6,7,8};
  salsa20_set_nonce(&ctx, nonce);

  uint8_t ciphertext[size] = {0};
  salsa20_crypt(&ctx, size, ciphertext, inputtext);

  unsigned char outputtext[size+1] = {0};
  salsa20_set_nonce(&ctx, nonce);
  salsa20_crypt(&ctx, size, outputtext, ciphertext);

  for(int i=0; i<size; i++) {
    printf("%c", outputtext[i]);
  }

  return 0; 
}

