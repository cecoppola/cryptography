//#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <inttypes.h>

uint64_t trigen(gsl_rng *qrng) {
  uint64_t b = gsl_rng_uniform_int(qrng, 3);
  return 1 + b;
}

char** gridmap(gsl_rng *qrng, double p_zero, double rho_odd, double rho_even, uint64_t k, uint64_t width) {
  double wsum = 1.0 + rho_odd + rho_even;
  double p0 = p_zero/wsum;
  double p1 = rho_odd/wsum;
  double p2 = rho_even/wsum;
  uint64_t xcoord = 0; uint64_t ycoord = 0;
  uint64_t numdata = (width*width+1)/2;
  char gval = 0;
  char** qarray = calloc(width, sizeof(char*));
  for(int i = 0; i < width; i++) qarray[i] = calloc(width, sizeof(char));
  for(int n = 0; n < k; n++) {
    uint64_t errindex = 2*gsl_rng_uniform_int(qrng, numdata);
    div_t mod = div(errindex, width);
    xcoord = mod.quot; ycoord = mod.rem;
    qarray[xcoord][ycoord] ^= trigen(qrng);
    for(uint8_t x = 0; x < width; x++) {
      for(uint8_t y = 0; y < width; y++) {
        gval = qarray[x][y];
        if(gval == 0) continue;
        else {
          if (gsl_rng_uniform(qrng) < p1) { if(x > 1)         qarray[x-2][y] ^= trigen(qrng); }
          if (gsl_rng_uniform(qrng) < p1) { if(x < (width-2)) qarray[x+2][y] ^= trigen(qrng); }
          if (gsl_rng_uniform(qrng) < p1) { if(y > 1)         qarray[x][y-2] ^= trigen(qrng); }
          if (gsl_rng_uniform(qrng) < p1) { if(y < (width-2)) qarray[x][y+2] ^= trigen(qrng); }
          if (gsl_rng_uniform(qrng) < p2) { if(x > 3)         qarray[x-4][y] ^= trigen(qrng); }
          if (gsl_rng_uniform(qrng) < p2) { if(x < (width-4)) qarray[x+4][y] ^= trigen(qrng); }
          if (gsl_rng_uniform(qrng) < p2) { if(y > 3)         qarray[x][y-4] ^= trigen(qrng); }
          if (gsl_rng_uniform(qrng) < p2) { if(y < (width-4)) qarray[x][y+4] ^= trigen(qrng); }
          if (gsl_rng_uniform(qrng) < p2) { if((x < (width-2)) && (y < (width-2))) qarray[x+2][y+2] ^= trigen(qrng); }
          if (gsl_rng_uniform(qrng) < p2) { if((x > 1)         && (y > 1))         qarray[x-2][y-2] ^= trigen(qrng); }
          if (gsl_rng_uniform(qrng) < p2) { if((x < (width-2)) && (y > 1))         qarray[x+2][y-2] ^= trigen(qrng); }
          if (gsl_rng_uniform(qrng) < p2) { if((x > 1)         && (y < (width-2))) qarray[x-2][y+2] ^= trigen(qrng); }
        }
      }
    }
  }
  return qarray;
}

int main(int argc, char **argv) {
  uint8_t distance = atoi(argv[1]);
  double p_zero    = atof(argv[2]);
  double rho_odd   = atof(argv[3]);
  double rho_even  = atof(argv[4]);
  uint64_t width = 2*distance - 1;
  uint64_t area = width*width;
  time_t t;
  int pid_t = getpid();
  uint32_t rsalt = rand();
  unsigned long rseed = pid_t + time(&t) + rsalt;
  srand(rseed);
  gsl_rng *qrng;
  qrng = gsl_rng_alloc(gsl_rng_taus2);
  gsl_rng_set(qrng, rand());
  uint64_t numdata = (area+1)/2;
  char** qarray = calloc(width, sizeof(char*));
  for(int i = 0; i < width; i++) qarray[i] = calloc(width, sizeof(char));
  FILE* Synfile; Synfile = fopen("syntsynd.dat","w");

  for(int t = 0; t < distance; t++) {
    for(int i = 0; i < width; i++) { 
      for(int j = 0; j < width; j++) { 
        qarray[i][j] = 0;
      }
    }
    uint64_t k = gsl_ran_binomial(qrng, p_zero, numdata);
    qarray = gridmap(qrng, p_zero, rho_odd, rho_even, k, width);
    for(uint8_t x = 0; x < width; x++) {
      for(uint8_t y = 0; y < width; y++) {
        if ((x%2) ^ (y%2)) { 
          if(x > 0)         qarray[x][y] ^= qarray[x-1][y];
          if(x < (width-1)) qarray[x][y] ^= qarray[x+1][y];
          if(y > 0)         qarray[x][y] ^= qarray[x][y-1];
          if(y < (width-1)) qarray[x][y] ^= qarray[x][y+1];       
          if(y%2)           qarray[x][y] &= 1;
          if(x%2)           qarray[x][y] &= 2;
        }
      }
    }
    printf("\nRandom Syndrome Map %2i:\n", t+1);
    uint64_t p = 0;
    for(uint64_t x = 0; x < width; x++) {
      for(uint64_t y = 0; y < width; y++) {
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
    for(uint8_t t = 0; t < 1; t++) {
      for(uint8_t x = 0; x < width; x++) {
        if (!(x%2)) fprintf(Synfile, " ");
        for(uint8_t y = 0; y < width; y++) {
          p = x + y;
          if (p%2) {
            fprintf(Synfile, "%i ", qarray[x][y]);
          }
        }
        fprintf(Synfile, "\n");
      }
      fprintf(Synfile, "\n");
    }
  }

  for(int i = 0; i < width; i++) free(qarray[i]);
  free(qarray);
  fclose(Synfile);
  gsl_rng_free(qrng);
  return 0;
}

