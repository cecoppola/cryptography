// ./cube 22 .0013 .30 .15 10
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

uint64_t* iidmap(gsl_rng *qrng, double p_zero, uint64_t k, uint64_t numdata) {
  uint64_t* errors = calloc(k, sizeof(uint64_t));
  uint64_t* locations = calloc(numdata, sizeof(uint64_t));
  for(int i = 0; i < numdata; i++) locations[i] = i;
  gsl_ran_choose(qrng, errors, k, locations, numdata, sizeof(uint64_t));
  uint64_t* derrors = calloc(k, sizeof(uint64_t));
  for(int j = 0; j < k; j++) derrors[j] = 2*errors[j];
  return derrors;
}

int main(int argc, char **argv) {

  uint8_t distance = atoi(argv[1]);
  double p_zero    = atof(argv[2]);
  double rho_odd   = atof(argv[3]);
  double rho_even  = atof(argv[4]);
  uint16_t blocks  = atoi(argv[5]);
  uint64_t width = 2*distance - 1;
  uint64_t area = width*width;
  uint64_t numdata = (area+1)/2;

  time_t t;
  int pid_t = getpid();
  uint32_t rsalt = rand();
  unsigned long rseed = pid_t + time(&t) + rsalt;
  srand(rseed);
  gsl_rng *qrng;
  qrng = gsl_rng_alloc(gsl_rng_taus2);
  gsl_rng_set(qrng, rand());

  char** qarray = calloc(width, sizeof(char*));
  for(int i = 0; i < width; i++) qarray[i] = calloc(width, sizeof(char));
  float** gamma = calloc(width, sizeof(float*));
  for(int i = 0; i < width; i++) gamma[i] = calloc(width, sizeof(float));

  FILE* Synfile; Synfile = fopen("syntsynd.dat","w");

  //Loop over blocks
  for(int n = 0; n < blocks; n++) {
    //Loop over layers
    for(int t = 0; t < distance; t++) {
  
      //Reset interactions
      for(int i = 0; i < width; i++) { 
        for(int j = 0; j < width; j++) { 
          gamma[i][j] = 0.0;
        }
      }

      //Generate new uncorrelated distribution
      uint64_t k = gsl_ran_binomial(qrng, p_zero, numdata);
      uint64_t* errors = calloc(k, sizeof(uint64_t));
      uint64_t numdata = (width*width+1)/2;
      errors = iidmap(qrng, p_zero, k, numdata);
      uint64_t xcoord = 0; uint64_t ycoord = 0;
      for(int i = 0; i < k; i++) {
        div_t mod = div(errors[i], width);
        xcoord = mod.quot;
        ycoord = mod.rem;
        qarray[xcoord][ycoord] ^= trigen(qrng);
      }
      free(errors);

      //Compute the correlation matrix
      char gval = 0;
      for(uint8_t x = 0; x < width; x++) {
        for(uint8_t y = 0; y < width; y++) {
        gval = qarray[x][y];
        if(gval == 0) continue;
          else {
            if (x > 1)                             gamma[x-2][y] += rho_odd;
            if (x < (width-2))                     gamma[x+2][y] += rho_odd;
            if (y > 1)                             gamma[x][y-2] += rho_odd;
            if (y < (width-2))                     gamma[x][y+2] += rho_odd;
            if (x > 3)                             gamma[x-4][y]   += rho_even;
            if (x < (width-4))                     gamma[x+4][y]   += rho_even;
            if (y > 3)                             gamma[x][y-4]   += rho_even;
            if (y < (width-4))                     gamma[x][y+4]   += rho_even;
            if((x < (width-2)) && (y < (width-2))) gamma[x+2][y+2] += rho_even;
            if((x > 1)         && (y > 1))         gamma[x-2][y-2] += rho_even;
            if((x < (width-2)) && (y > 1))         gamma[x+2][y-2] += rho_even;
            if((x > 1)         && (y < (width-2))) gamma[x-2][y+2] += rho_even;
          }
        }
      }

      //Add correlations
      for(uint8_t x = 0; x < width; x++) {
        for(uint8_t y = 0; y < width; y++) {
          gval = gamma[x][y];
          if(gval == 0) continue;
          if (gsl_rng_uniform(qrng) < gval) { qarray[x][y] ^= trigen(qrng); }
        }
      }

      //Compute syndrome
      for(uint8_t x = 0; x < width; x++) {
        for(uint8_t y = 0; y < width; y++) {
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
      printf("\nSyndrome at (%3i,%3i):\n", n+1, t+1);
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

      //Write syndrome
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

    //Correct all errors
    for(int i = 0; i < width; i++) { 
      for(int j = 0; j < width; j++) { 
        qarray[i][j] = 0.0;
      }
    }

  }

  for(int i = 0; i < width; i++) free(gamma[i]);
  free(gamma);
  for(int i = 0; i < width; i++) free(qarray[i]);
  free(qarray);  fclose(Synfile);
  gsl_rng_free(qrng);
  return 0;
}

