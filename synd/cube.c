#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <linux/random.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <inttypes.h>

#define bmask "%c%c%c%c%c%c%c%c"
#define bits(byte)             \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0') 

void printbits(uint64_t row) {
printf(""bmask""bmask""bmask""bmask""bmask""bmask""bmask""bmask"\n", 
  bits(row>>56), bits(row>>48), bits(row>>40), bits(row>>32),
  bits(row>>24), bits(row>>16), bits(row>>8),  bits(row>>0));
}

uint64_t trigen(gsl_rng *qrng) {
  uint64_t b = gsl_rng_uniform_int(qrng, 3);
  return 1 + b;
}

uint64_t* iidmap(gsl_rng *qrng, uint64_t k, uint64_t numdata) {
  uint64_t* errors = calloc(k, sizeof(uint64_t));
  uint64_t* locations = calloc(numdata, sizeof(uint64_t));
  for(uint64_t i = 0; i < numdata; i++) locations[i] = i;
  gsl_ran_choose(qrng, errors, k, locations, numdata, sizeof(uint64_t));
  uint64_t* derrors = calloc(k, sizeof(uint64_t));
  for(uint64_t j = 0; j < k; j++) derrors[j] = 2*errors[j];
  free(errors);
  free(locations);
  return derrors;
}

unsigned long fourseed() {
  time_t t;
  int pid_t = getpid();
  unsigned long rseed1 = pid_t * time(&t);
  srand(rseed1);
  unsigned char buf[4];
  int rv = syscall(SYS_getrandom, buf, sizeof(buf), 0);
  uint32_t bufint = 0;
  for (int i = 0; i < sizeof(buf); i++) bufint = (bufint << 8) | buf[i];
  uint32_t rsalt = rand();
  unsigned long rseed2 = pid_t*time(&t)*rsalt*bufint;
  printf("seed = % *li\n", 20, rseed2);
  return rseed2;
}

int main(int argc, char **argv) {
  
  uint64_t distance = atoi(argv[1]);
  double p_zero     = atof(argv[2]);
  double rho_odd    = atof(argv[3]);
  double rho_even   = atof(argv[4]);
  uint16_t blocks   = atoi(argv[5]);
  uint64_t width    = 2*distance - 1;
  uint64_t area     = width*width;
  uint64_t numdata  = (area+1)/2;

  int c;
  const char* optstring = "phb";
  int pflag = 0;
  int hflag = 0;
  int bflag = 0;
  while ((c = getopt(argc, argv, optstring)) != -1) {
    switch (c) {
      case 'p':
        pflag = 1;
        break;
      case 'h':
        hflag = 1;
        break;
      case 'b':
        bflag = 1;
        break;
    }
  }

  srand(fourseed());
  gsl_rng *qrng;
  qrng = gsl_rng_alloc(gsl_rng_taus2);
  gsl_rng_set(qrng, rand());

  uint8_t** qarray = calloc(width, sizeof(uint8_t*));
  for(int i = 0; i < width; i++) qarray[i] = calloc(width, sizeof(uint8_t));
  float** gamma = calloc(width, sizeof(float*));
  for(int i = 0; i < width; i++) gamma[i] = calloc(width, sizeof(float));

  FILE* Synfile ;
  Synfile = fopen("syntsynd.txt","w");
  FILE* bSynfile;
  bSynfile = fopen("syntsynd.bin","wb");
  uint64_t rowsyn = 0UL;
  uint64_t p2shift = 0;
  uint64_t p1bit = 0;
  uint64_t tbval = 0;
  uint64_t y = 0;

  //Loop over blocks
  for(uint64_t n = 0; n < blocks; n++) {
    //Loop over layers
    for(uint64_t t = 0; t < distance; t++) {
  
      //Reset interactions
      for(uint64_t i = 0; i < width; i++) { 
        for(uint64_t j = 0; j < width; j++) { 
          gamma[i][j] = 0.0;
        }
      }

      //Generate new uncorrelated distribution
      uint64_t k = gsl_ran_binomial(qrng, p_zero, numdata);
      uint64_t* errors = iidmap(qrng, k, numdata);
      uint64_t xcoord = 0; uint64_t ycoord = 0;
      for(int i = 0; i < k; i++) {
        div_t mod = div(errors[i], width);
        ycoord = mod.quot;
        xcoord = mod.rem;
        qarray[ycoord][xcoord] ^= trigen(qrng);
      }
      free(errors);

      //Compute the correlation matrix
      uint16_t gval = 0;
      for(uint64_t x = 0; x < width; x++) {
        for(uint64_t y = 0; y < width; y++) {
        gval = qarray[x][y];
        if(gval == 0) continue;
          else {
            if((x < (width-1)) && (y < (width-1))) gamma[x+1][y+1] += rho_odd;
            if((x < (width-1)) && (y > 1))         gamma[x+1][y-1] += rho_odd;
            if((x > 1)         && (x < (width-1))) gamma[x-1][y+1] += rho_odd;
            if((x > 1)         && (y > 1))         gamma[x-1][y-1] += rho_odd;
            if (x < (width-2))                     gamma[x+2][y+0] += rho_even;
            if((x < (width-2)) && (y < (width-2))) gamma[x+2][y+2] += rho_even;
            if                    (y < (width-2))  gamma[x+0][y+2] += rho_even;
            if((x > 2)         && (y < (width-2))) gamma[x-2][y+2] += rho_even;
            if (x > 2)                             gamma[x-2][y+0] += rho_even;
            if((x > 2)         && (y > 2))         gamma[x-2][y-2] += rho_even;
            if                    (y > 2)          gamma[x+0][y-2] += rho_even;
            if((x < (width-2)) && (y > 2))         gamma[x+2][y-2] += rho_even;
          }
        }
      }

      //Add correlations
      for(uint64_t x = 0; x < width; x++) {
        for(uint64_t y = 0; y < width; y++) {
          gval = gamma[x][y];
          if(gval == 0) continue;
          if (gsl_rng_uniform(qrng) < gval) { qarray[x][y] ^= trigen(qrng); }
        }
      }

      //Compute syndrome
      for(uint64_t x = 0; x < width; x++) {
        for(uint64_t y = 0; y < width; y++) {
          if ((x%2) ^ (y%2)) { 
            qarray[x][y] = 0;
            if(x > 0)         qarray[x][y] ^= qarray[x-1][y];
            if(x < (width-1)) qarray[x][y] ^= qarray[x+1][y];
            if(y > 0)         qarray[x][y] ^= qarray[x][y-1];
            if(y < (width-1)) qarray[x][y] ^= qarray[x][y+1];       
            if(y%2)           qarray[x][y] &= 1;
            if(x%2)           qarray[x][y] &= 2;
          }
        }
      }

      //Print syndrome
      uint64_t p = 0;
      if (pflag) {
        printf("\nSyndrome at (%3li,%3li):\n", n+1, t+1);
        for(uint64_t x = 0; x < width; x++) {
          for(uint64_t y2 = 0; y2 < width; y2++) {
            y = width - y2 - 1;
            p = x + y;
            if (p%2) {
              if     ( qarray[x][y] == 1) printf("\x1b[38;5;226mZ \x1B[0m");
              else if( qarray[x][y] == 2) printf("\x1b[38;5;226mX \x1B[0m");
              else if   (x%2)  printf("\x1b[38;5;36m%i \x1B[0m", qarray[x][y]);      
              else if (!(x%2)) printf("\x1b[38;5;31m%i \x1B[0m", qarray[x][y]);      
            }
            else {
              if( qarray[x][y] == 0) printf("\x1b[38;5;245mI \x1B[0m");         	
              if( qarray[x][y] == 1) printf("\x1b[38;5;160mZ \x1B[0m");
              if( qarray[x][y] == 2) printf("\x1b[38;5;160mX \x1B[0m");
              if( qarray[x][y] == 3) printf("\x1b[38;5;160mY \x1B[0m");
            }
          }
          printf("\x1B[0m\n");
        }
      }

      //Write syndrome
      if (hflag) {
        for(uint64_t y = 0; y < width; y++) {
          if (!(y%2)) fprintf(Synfile, " ");
          for(uint64_t x = 0; x < width; x++) {
            p = x + y;
            if (p%2) {
              fprintf(Synfile, "%i ", qarray[x][y]);
            }
          }
          fprintf(Synfile, "\n");
        }
        fprintf(Synfile, "\n");
      }

      //Write binary syndrome
      if (bflag) {
        for(uint64_t y = 0; y < width; y++) { 
          p1bit = y%2;
          p2shift = 0;
          rowsyn = 0;
          for(uint64_t x = 0; x < width; x++) {
            if ((p1bit+x)%2) {
              tbval = qarray[y][x];
              rowsyn += (tbval << (p2shift));
              p2shift += 2;
            }
          }
          if (p1bit) rowsyn >>= 1;
          else rowsyn <<= 1;
          size_t synwrite = fwrite(&rowsyn, sizeof(uint64_t), 1, bSynfile);
        }
      }

    }

    //Correct all errors
    for(uint64_t i = 0; i < width; i++) { 
      for(uint64_t j = 0; j < width; j++) { 
        qarray[i][j] = 0.0;
      }
    }

  }

  fclose(Synfile);
  fclose(bSynfile);
  for(int i = 0; i < width; i++) free(gamma[i]);
  free(gamma);
  for(int i = 0; i < width; i++) free(qarray[i]);
  free(qarray);
  gsl_rng_free(qrng);
  return 0;

}

