This repository contains source code for ten programs which implement common crytography algorithms. These programs are based on methods in the Nettle library (https://www.lysator.liu.se/~nisse/nettle/).

The ten algorithms are:

Hash:
MD5
SHA
SHA2-256
SHA3-512

Symmetric Key:
RC4
DES
AES
Salsa20

Public Key:
RSA
DSA

To execute the hashing functions, compile the program and use standard input:
./sha3-512 < plaintext.txt

To execute the symmetric key algorithms, compile the program and use standard input:
./aes-256 < plaintext.txt

The public key algorithms don't encrypt; the just demonstrate key generation. To execute then, compile and run:
./rsa-4096
