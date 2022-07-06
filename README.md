This repository contains source code and precompiled binaries for utilities which implement common cryptography algorithms. These programs are built using functions from the open source Nettle library (https://www.lysator.liu.se/~nisse/nettle/). Here are suggested compilation flags:
> $ make aes-256 -lgsl -lgslcblas -lm -lnettle -lhogweed -lgmp 

The algorithms are:

**Hashing**  
- MD5  
- SHA1  
- SHA2-256  
- SHA3-512  

**Symmetric Cipher**
- RC4  
- DES  
- AES  
- Salsa20  

**Public Key**
- RSA  
- DSA  
- EcDSA

To execute the hashing functions, compile the program and use standard input:  
> $ ./sha3-512 < message.txt  

To execute the symmetric key algorithms, compile the program and use standard input:   
> $ ./aes-256 < message.txt  

The public key algorithms don't encrypt; they just demonstrate key generation. To execute them and output the keys, compile and run:   
> $ ./rsa-4096  
