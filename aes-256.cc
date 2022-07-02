// Copyright 2022 C. E. Coppola
// This program generates a random 256-bit key, encrypts a chosen text with AES, and then decrypts it using the same key.

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <nettle/aes.h>
using namespace std;

//Generates a random key.
void aeskey(uint8_t* key, size_t max) {
  random_device rd;
  srand (rd());
  for(int i=0; i<max; i++) key[i] = rand()%256;
  return; 
}

//Encrypts and thendes decrypts an input text using AES.
int main(int argc, char **argv) {

  uint8_t blocksize = AES_BLOCK_SIZE;
  uint8_t key[blocksize +1] = {0};

  aeskey(key, blocksize );
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

  uint8_t ciphertext[blocks*blocksize] = {0};
  unsigned char outputtext[blocks*(blocksize )+1] = {0};
  unsigned char inputtext[blocks*(blocksize )+1] = {0};

  for(int i=0; i<blocks; i++) {
    for(int j=0; j<blocksize ; j++) {
      inputtext[i*blocksize+j] = textin[i*blocksize+j];
    }
  }

  struct aes256_ctx ctx;
  aes256_set_encrypt_key(&ctx, key);
  aes256_encrypt(&ctx, blocks*blocksize, ciphertext, inputtext);

  aes256_set_decrypt_key(&ctx, key);
  aes256_decrypt(&ctx, blocks*blocksize, outputtext, ciphertext);

  for(int i=0; i<blocks; i++) {
    for(int j=0; j<blocksize ; j++) {
      printf("%c", outputtext[i*blocksize+j]);
    }
  }

  return 0; 
}

