This repository contains source code for ten programs which implement common cryptography algorithms. These programs are based on methods in the open source Nettle library (https://www.lysator.liu.se/~nisse/nettle/). You will need to install that library to compile these programs. I recommend the following compiler flags:  
$ make sha.cc -lgsl -lgslcblas -lm -lnettle -lhogweed -lgmp

The ten algorithms are:

Hashing  
-------  
MD5  
SHA1  
SHA2-256  
SHA3-512  

Symmetric Key  
-------------  
RC4  
DES  
AES  
Salsa20  

Public Key  
----------  
RSA  
DSA  

To execute the hashing functions, compile the program and use standard input:  
$ make sha-512  
$ ./sha3-512 < plaintext.txt  

To execute the symmetric key algorithms, compile the program and use standard input:  
$ make sha-512  
$ ./aes-256 < plaintext.txt  

The public key algorithms don't encrypt; they just demonstrate key generation. To execute them and output the keys, compile and run:  
$ make rsa-4096  
$ ./rsa-4096  
