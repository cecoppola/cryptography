// Copyright 2022 C. E. Coppola
// This program generates a random RC4 key, encrypts a chosen text, and then decrypts it using the same key.

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <nettle/arcfour.h>
using namespace std;

//Generates a random key for RC4.
void rc4key(uint8_t* key, size_t max) {
  random_device rd;
  srand (rd());
  for(int i=0; i<max; i++) key[i] = rand()%256;
  return; 
}

//Encrypts and thendes decrypts an input text using RC4.
int main(int argc, char **argv) {

  int keysize = ARCFOUR_MAX_KEY_SIZE;
  uint8_t key[keysize +1] = {0};

  rc4key(key, keysize);

  printf("hex key: ");
  for (int i=0; i<keysize ; i++) {
    printf("%x", key[i]);
  }
  printf("\n");

  stringstream fileinput;
  fileinput.str("");
  fileinput << cin.rdbuf();
  string textin = fileinput.str();

  size_t size = textin.length();
  unsigned char inputtext[size+1] = {0};

  for(int i=0; i<size; i++) {
    inputtext[i] = textin[i];
  }

  struct arcfour_ctx ctx;

  uint8_t ciphertext[size] = {0};
  arcfour_set_key(&ctx, keysize, key);
  arcfour_crypt(&ctx, size, ciphertext, inputtext);

  unsigned char outputtext[size+1] = {0};
  arcfour_set_key(&ctx, keysize, key);
  arcfour_crypt(&ctx, size, outputtext, ciphertext);

  for(int i=0; i<size; i++) {
    printf("%c", outputtext[i]);
  }

  return 0; 
}

