// Copyright 2022 C. E. Coppola
// This program generates a random DES key, encrypts a chosen text, and then decrypts it using the same key.

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <nettle/des.h>
using namespace std;

//Generates a random key.
void deskey(uint8_t* key, size_t max) {
  random_device rd;
  srand (rd());
  for(int i=0; i<max; i++) key[i] = rand()%256;
  return; 
}

//Encrypts and thendes decrypts an input text using DES.
int main(int argc, char **argv) {

  uint8_t blocksize = DES_BLOCK_SIZE;
  uint8_t keysize = DES_KEY_SIZE;

  uint8_t key[blocksize +1] = {0};
  uint8_t keycopy[blocksize +1] = {0};

  deskey(key, keysize);
  for(int i = 0; i < keysize; i++) keycopy[i] = key[i];

  int pcheck = 1;
  pcheck = des_check_parity (keysize, key);
  if(pcheck == 0) des_fix_parity (keysize, key, keycopy);

  printf("hex key: ");
  for (int i=0; i<blocksize ; i++) {
    printf("%x", key[i]);
  }
  printf("\n");

  stringstream fileinput;
  fileinput.str("");
  fileinput << cin.rdbuf();
  string textin = fileinput.str();

  size_t size = textin.length();
  size_t blocks = (1+(size-(size%blocksize))/blocksize);

  unsigned char inputtext[blocks*(blocksize )+1] = {0};

  for(int i=0; i<blocks; i++) {
    for(int j=0; j<blocksize ; j++) {
      inputtext[i*blocksize+j] = textin[i*blocksize+j];
    }
  }

  struct des_ctx ctx;
  des_set_key(&ctx, key);

  uint8_t ciphertext[blocks*blocksize] = {0};
  des_encrypt(&ctx, blocks*blocksize, ciphertext, inputtext);

  unsigned char outputtext[blocks*(blocksize )+1] = {0};
  des_decrypt(&ctx, blocks*blocksize, outputtext, ciphertext);

  for(int i=0; i<blocks; i++) {
    for(int j=0; j<blocksize ; j++) {
      printf("%c", outputtext[i*blocksize+j]);
    }
  }

  return 0; 
}

