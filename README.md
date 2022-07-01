This repository contains source code for ten programs which implement common cryptography algorithms. These programs are based on methods in the open source Nettle library (https://www.lysator.liu.se/~nisse/nettle/). You will need to install that library to compile these programs. Here are suggested flags:
> $ make aes-256 -lgsl -lgslcblas -lm -lnettle -lhogweed -lgmp 

The ten algorithms are:

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

To execute the hashing functions, compile the program and use standard input:  
> $ ./sha3-512 < message.txt  

To execute the symmetric key algorithms, compile the program and use standard input:   
> $ ./aes-256 < message.txt  

The public key algorithms don't encrypt; they just demonstrate key generation. To execute them and output the keys, compile and run:   
> $ ./rsa-4096  
