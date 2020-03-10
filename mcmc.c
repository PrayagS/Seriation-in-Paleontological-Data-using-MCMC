/* $Id: mcmc.c,v 1.33 2005/12/20 15:54:08 kaip Exp $ */

/*
 * 
 * mcmc
 * Copyright (C) 2005  Kai Puolamaki <Kai.Puolamaki@iki.fi> 
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 *
 */

/******************************************************************* 
   Requires GSL - GNU Scientific Library - Version 1.6.
   http://www.gnu.org/software/gsl/
 *******************************************************************/

#define MCMCHAVEMAIN
/*#define MCMCDEBUG*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "mcmc.h"
#include "log.h"

gsl_rng *r;

#ifdef MCMCHAVEMAIN

/* 
 * Simple usage: 
 *  env GSL_RNG_SEED=42 ./mcmc < g10s10.txt
 */

main(int argc, char *argv[])
{

  FILE *f = fopen("mcmc_c.log", "a");
  log_set_fp(f);

  int i, tb = 1000, ts = 1000, manycd = 0;
  mcmc_model x;

  switch (argc)
  {
  case 1:
    break;
  case 4:
    if (sscanf(argv[1], "%d", &manycd) &&
        sscanf(argv[2], "%d", &tb) == 1 && tb >= 0 &&
        sscanf(argv[3], "%d", &ts) == 1 && ts >= 0)
      break;
  default:
    fprintf(stderr, "usage: %s [manycd Tburnin T]\n", argv[0]);
    exit(1);
  }

  mcmc_init();
  mcmc_readmodel(&x, stdin, manycd);

  /*
   * Randomize the initial condition.
   */
  mcmc_randomize(&x);

  // mcmc_consistent(&x);

  /*
   * Burn-in.
   */
  // for (i = 0; i < tb; i++)
  // {
  //   mcmc_sample(&x);
  // }

  /*
   * Sampling.
   */
  // for (i = 0; i < ts; i++)
  // {
  //   mcmc_sample(&x);
  //   mcmc_save(&x, stdout);
  // }

  // if (mcmc_consistent(&x))
  // {
  //   mcmc_print(&x, stderr);
  //   fprintf(stderr, "main: error.\n");
  //   exit(1);
  // }

  mcmc_freemodel(&x);
  mcmc_free();

  return 0;
}
/* end of main */
#endif

int mcmc_sample(mcmc_model *x)
/*
 * Basic sampling iteration consists of block of 10 samples.
 */
{
  int i, j;
  static int count = 0, cc = 0, cd = 0, cab = 0, cpi1 = 0, cpi20 = 0, cpi21 = 0, cpi3 = 0;
  gsl_vector_int *p;

  p = gsl_vector_int_alloc(x->N);

  for (i = 0; i < 10; i++)
  {
    cc += mcmc_samplec(x);
    cd += mcmc_sampled(x);
    cab += mcmc_sampleab(x);
    /*
     * Sampling of permutations is more difficult (lower acceptance
     * probability of the MH proposal). Therefore, we compensate by
     * trying several MH proposals for permutation. Ideally all parameters
     * should be sampled with a approximately similar acceptance 
     * probability.
     */
    cpi21 += mcmc_samplepi2(x, 1);
    for (j = 0; j < 5; j++)
    {
      cpi1 += mcmc_samplepi1(x);
      cpi20 += mcmc_samplepi2(x, 0);
      cpi3 += mcmc_samplepi3(x, p);
    }
  }
  count++;

  gsl_vector_int_free(p);

#ifdef MCMCDEBUG
  fprintf(stderr, "mcmc_sample: %f %f %f %f %f %f %f\n",
          cc / (10. * count), cd / (10. * count), cab / (20. * count * x->M),
          cpi1 / (10. * 5 * count), cpi20 / (10. * 5 * count), cpi21 / (10. * 5 * count),
          cpi3 / (10. * 5 * count));
  mcmc_consistent(x);
#endif

  return cc + cd + cab + cpi1 + cpi20 + cpi21 + cpi3;
}
/* end of mcmc_sample */

void mcmc_save(const mcmc_model *x, FILE *f)
/*
 * Saves model x to stream f in machine-readable format.
 * All parameters are in one space and tab separated line that
 * ends with a newline. 
 */
{
  int i;
  for (i = 0; i < x->M; i++)
    fprintf(f, "%d ", gsl_vector_int_get(x->a, i));
  fprintf(f, "\t");
  for (i = 0; i < x->M; i++)
    fprintf(f, "%d ", gsl_vector_int_get(x->b, i));
  fprintf(f, "\t");
  for (i = 0; i < x->N; i++)
    fprintf(f, "%d ", gsl_permutation_get(x->pi, i));
  fprintf(f, "\t");
  for (i = 0; i < x->M; i++)
    fprintf(f, "%.14f ", exp(gsl_vector_get(x->c, i)));
  fprintf(f, "\t");
  for (i = 0; i < x->M; i++)
    fprintf(f, "%.14f ", exp(gsl_vector_get(x->d, i)));
  fprintf(f, "\t%.14f\n", x->loglik);
}
/* end of mcmc_save */

void mcmc_print(const mcmc_model *x, FILE *f)
/*
 * Prints the model in human-readable format.
 */
{
  int i, j;

  fprintf(f, "N = %d  M = %d    c = %g  d = %g\n",
          x->N, x->M, exp(gsl_vector_get(x->c, 0)), exp(gsl_vector_get(x->d, 0)));
  for (i = 0; i < x->M; i++)
    fprintf(f, "%3d  a = %d  b = %d\n",
            i, gsl_vector_int_get(x->a, i), gsl_vector_int_get(x->b, i));
  fprintf(f, "pi  =");
  for (i = 0; i < x->N; i++)
    fprintf(f, " %d", gsl_permutation_get(x->pi, i));
  fprintf(f, "\nrpi =");
  for (i = 0; i < x->N; i++)
    fprintf(f, " %d", gsl_permutation_get(x->rpi, i));
  fprintf(f, "\nHard sites (pi(h)):");
  j = 0;
  for (i = 0; i < x->N; i++)
  {
    if (gsl_vector_int_get(x->h, i))
    {
      fprintf(f, " %d(%d)", i, gsl_permutation_get(x->pi, i));
      j += 1;
    }
  }
  fprintf(f, " (%d total)\n", j);
#ifdef MCMCPRINTDATAMATRIX
  fprintf(f, "Data matrix X:\n");
  for (i = 0; i < x->N; i++)
  {
    for (j = 0; j < x->M; j++)
      fprintf(f, "%d", gsl_matrix_int_get(x->X, i, j));
    fprintf(f, "\n");
  }
#endif
}
/* end of mcmc_print */

mcmc_model *mcmc_readmodel(mcmc_model *x, FILE *f, int manycd)
/*
 * Reads and initializes model from a standard format ASCII file.
 */
{
  int i, j, k, n, m;
  char s[MAXS];

  if (!fgets(s, MAXS, f))
  {
    fprintf(stderr, "mcmc_readmodel: read error.\n");
    exit(1);
  }

  if (sscanf(s, "%d %d", &n, &m) != 2 || n <= 0 || m <= 0)
  {
    fprintf(stderr, "mcmc_readmodel: read error at header.\n");
    exit(1);
  }

  x->N = n;
  x->M = m;
  x->nh = 0;

  x->manycd = manycd;

  x->h = gsl_vector_int_calloc(n);
  x->X = gsl_matrix_int_calloc(n, m);
  for (i = 0; i < n; i++)
  {
    if (!fgets(s, MAXS, f))
    {
      fprintf(stderr, "mcmc_readmodel: read error.\n");
      exit(1);
    }
    for (j = k = 0; j < m; j++)
    {
      // Keep going till valid data point occurs
      while (s[k] != '0' && s[k] != '1' && s[k] != '\0')
        k++;
      switch (s[k])
      {
      case '0':
        // set 0 in matrix at that point
        gsl_matrix_int_set(x->X, i, j, 0);
        k++;
        break;
      case '1':
        // set 1 in matrix at that point
        gsl_matrix_int_set(x->X, i, j, 1);
        k++;
        break;
      }
    }
    while (s[k] != '*' && s[k] != '\0')
      k++;
    // set hard ordering vector if '*' occurs
    if (s[k] == '*')
    {
      gsl_vector_int_set(x->h, i, 1);
      x->nh++; // increment hard sites counter
    }
  }

  // printf("%d", x->nh);

  x->pi = gsl_permutation_calloc(n);      // init permutation to identity
  x->rpi = gsl_permutation_alloc(n);      // init permutation but elements are undefined
  gsl_permutation_inverse(x->rpi, x->pi); // Inverse of permutation pi stored in rpi

  x->a = gsl_vector_int_alloc(m); // init birth sites of taxa
  x->b = gsl_vector_int_alloc(m); // init death sites of taxa
  mcmc_initab(x);

  x->c = gsl_vector_alloc(m); // log probability of false 1
  x->d = gsl_vector_alloc(m); // log probability of false 0

  // set predefined values of c and d for each taxa
  for (i = 0; i < x->M; i++)
  {
    gsl_vector_set(x->c, i, log(.01));
    gsl_vector_set(x->d, i, log(.3));
  }

  // init vectors of true 0s,1s and false 0s,1s for each taxa
  x->t0 = gsl_vector_int_alloc(m);
  x->f0 = gsl_vector_int_alloc(m);
  x->t1 = gsl_vector_int_alloc(m);
  x->f1 = gsl_vector_int_alloc(m);

  // Count number of true 0s,1s and false 0s,1s
  mcmc_count01(x);

  // Compute log likelihood of dataset (matrix)
  x->loglik = mcmc_logl(x);
  // printf("%lf", x->loglik);

  return x;
}
/* end of mcmc_readmodel */

void mcmc_initab(mcmc_model *x)
/*
 * Init a's and b's.
 */
{
  int n, m;

  for (m = 0; m < x->M; m++)
  {
    n = 0;
    /*
     * Keep going till matrix point is 0
     */
    while (n < x->N && !gsl_matrix_int_get(x->X, gsl_permutation_get(x->rpi, n), m))
      n++;
    if (n == x->N)
    {
      fprintf(stderr, "mcmc_initab: zero column at %d, continuing.\n", m);
      gsl_vector_int_set(x->a, m, 0);
      gsl_vector_int_set(x->b, m, x->N);
    }
    else
    {
      // nth site is the first site where taxa m first occured
      // set that as the birth site for taxa m in vector a
      gsl_vector_int_set(x->a, m, n);
      // Start from the end and find the death site
      n = x->N - 1;
      while (n >= 0 && !gsl_matrix_int_get(x->X, gsl_permutation_get(x->rpi, n), m))
        n--;
      // set that as the death site for taxa m in vector b
      gsl_vector_int_set(x->b, m, n + 1);
    }
  }
}
/* end of mcmc_initab */

void mcmc_randomize(mcmc_model *x)
/*
 * Shuffle permutation and adjust a's and b's to create random
 * initial condition. The order of the hard sites is preserved.
 */
{
  int i, j, k;
  int *p, *q;

  if (x->nh == 0)
  {
    // Shuffles array of elements using rng r
    gsl_ran_shuffle(r, x->pi->data, x->N, sizeof(size_t));
    gsl_permutation_inverse(x->rpi, x->pi);
    mcmc_count01(x);
    x->loglik = mcmc_logl(x);
    return;
  }
  else if (x->nh == x->N)
  {
    return;
  }

  if (!(p = malloc(x->N * sizeof(int))) || !(q = malloc(x->nh * sizeof(int))))
  {
    fprintf(stderr, "mcmc_randomize: out of memory.\n");
    exit(1);
  }
  for (i = 0; i < x->N; i++)
    p[i] = i;

  // printf("p\n");
  // for (size_t i = 0; i < x->N; i++)
  // {
  //   printf("%d ", p[i]);
  // }
  // printf("\n\n");

  /*
   * q will contain x->nh indices, in order.
   * q will have nh sites randomly taken from p which contains N sites
   */
  gsl_ran_choose(r, q, x->nh, p, x->N, sizeof(int));

  // printf("q\n");
  // for (size_t i = 0; i < x->nh; i++)
  // {
  //   printf("%d ", q[i]);
  // }
  // printf("\n\n");

  for (i = j = k = 0; i < x->N; i++)
  {
    if (i == q[j])
    {
      j++;
    }
    else
    {
      p[k++] = i;
    }
  }
  // printf("\n\n");

  // printf("p\n");
  // for (size_t i = 0; i < x->N; i++)
  // {
  //   printf("%d ", p[i]);
  // }
  // printf("\n\n");

  gsl_ran_shuffle(r, p, x->N - x->nh, sizeof(int));

  // printf("p\n");
  // for (size_t i = 0; i < x->N; i++)
  // {
  //   printf("%d ", p[i]);
  // }
  // printf("\n\n");

  for (i = j = k = 0; i < x->N; i++)
  {
    if (gsl_vector_int_get(x->h, i))
      x->pi->data[i] = q[j++];
    else
      x->pi->data[i] = p[k++];
  }

  if (gsl_permutation_valid(x->pi))
  {
    fprintf(stderr, "mcmc_randomize: invalid permutation.\n");
    exit(1);
  }
  gsl_permutation_inverse(x->rpi, x->pi);

  free(p);
  free(q);

  mcmc_initab(x);
  mcmc_count01(x);
  x->loglik = mcmc_logl(x);
}
/* end of mcmc_randomize */

void mcmc_init(void)
/*
 * Init everything necessary (see mcmc_free).
 */
{
  /*
   * Setup random number generator by defining its 
   * type (not used here) and
   * seed value (set using the environment var GSL_RNG_SEED) 
   */
  gsl_rng_env_setup();
  r = gsl_rng_alloc(gsl_rng_default); // Return instance of rng
}
/* end of mcmc_init */

void mcmc_free(void)
/*
 * Free everything necessary (see mcmc_init).
 */
{
  gsl_rng_free(r);
}
/* end of mcmc_free */

void mcmc_freemodel(mcmc_model *x)
/*
 * Free memory allocations associated with the model.
 */
{
  gsl_matrix_int_free(x->X);
  gsl_vector_int_free(x->a);
  gsl_vector_int_free(x->b);
  gsl_vector_free(x->c);
  gsl_vector_free(x->d);
  gsl_vector_int_free(x->t0);
  gsl_vector_int_free(x->t1);
  gsl_vector_int_free(x->f0);
  gsl_vector_int_free(x->f1);
  gsl_permutation_free(x->pi);
  gsl_permutation_free(x->rpi);
  gsl_vector_int_free(x->h);
}
/* end of mcmc_freemodel */

double mcmc_logl(const mcmc_model *x)
/*
 * Computes log-likelihood of the data. It is assumed the the numbers
 * of correct and wrong 0s and 1s (t0, f0, t1, f1) are known (they
 * can be calculated with mcmc_count01). All sampling functions maintain
 * the log-likelihood value inside the model structure.
 * 
 * Basically computes the log probability/likelihood as given on
 * top-right page 5.
 */
{
  int m;
  double loglik = 0., c, d;

  for (m = 0; m < x->M; m++)
  {
    c = gsl_vector_get(x->c, m);
    d = gsl_vector_get(x->d, m);
    loglik +=
        gsl_vector_int_get(x->t0, m) * log(1. - exp(c)) + gsl_vector_int_get(x->f0, m) * d + gsl_vector_int_get(x->t1, m) * log(1. - exp(d)) + gsl_vector_int_get(x->f1, m) * c;
  }

  return loglik;
}
/* end of mcmc_logl */

void mcmc_count01(mcmc_model *x)
/*
 * Computes the numbers of correct and wrong 0s and 1s (t0, f0, t1, f1).
 * All sampling functions maintain these numbers. This function should be
 * called when initializing a model or checking the sampling routines
 * for bugs.
 */
{
  int n, m;
  int t0, f0, t1, f1;

  x->t0a = x->f0a = x->t1a = x->f1a = 0;

  for (m = 0; m < x->M; m++)
  {
    t0 = f0 = t1 = f1 = 0;
    for (n = 0; n < x->N; n++)
    {
      // Condition for valid data point in the matrix
      if (gsl_vector_int_get(x->a, m) <= gsl_permutation_get(x->pi, n) &&
          gsl_permutation_get(x->pi, n) < gsl_vector_int_get(x->b, m))
      {
        // Condition for true 1
        if (gsl_matrix_int_get(x->X, n, m))
          t1++;
        // Condition for false 0
        else
          f0++;
      }
      else
      {
        // Condition for false 1
        if (gsl_matrix_int_get(x->X, n, m))
          f1++;
        // Condition for true 0
        else
          t0++;
      }
    }

    // Maintain global counters
    x->t0a += t0;
    x->f0a += f0;
    x->t1a += t1;
    x->f1a += f1;

    // Maintain taxa-wise vectors
    gsl_vector_int_set(x->t0, m, t0);
    gsl_vector_int_set(x->f0, m, f0);
    gsl_vector_int_set(x->t1, m, t1);
    gsl_vector_int_set(x->f1, m, f1);
  }
  // for (size_t i = 0; i < m; i++)
  // {
  //   printf("%d ", gsl_vector_int_get(x->t0, i));
  // }
  // printf("\n\n");
}
/* end of mcmc_count01 */

gsl_vector *mcmc_logtop(gsl_vector *p, int n)
/*
 * Takes a vector p of length n, exponentiates it and normalizes it
 * to a probability distribution (i.e., \sum p_i=1). The functions takes
 * care to avoid over- or underflows.
 */
{
  int i;
  double x, y, z;

  z = gsl_vector_get(p, 0);
  for (i = 1; i < n; i++)
    if (gsl_vector_get(p, i) > z)
      z = gsl_vector_get(p, i);
  x = 0.;
  for (i = 0; i < n; i++)
  {
    y = exp(GSL_MAX(LOGEPSILON, gsl_vector_get(p, i) - z));
    gsl_vector_set(p, i, y);
    x += y;
  }
  for (i = 0; i < n; i++)
    gsl_vector_set(p, i, gsl_vector_get(p, i) / x);

  return p;
}
/* end of mcmc_logtop */

double mcmc_samplebeta(double *x, double a, double b, double low, double high)
/* 
 * Take a sample from Beta distribution, with probability bound to [low,high].
 */
{
  double y;
  y = gsl_ran_beta(r, 1. + a, 1. + b);
  if (y > 0.)
  {
    y = log(y);
    if (low <= y && y <= high)
      *x = y;
  }
  return *x;
}
/* end of mcmc_samplebeta */

int mcmc_samplec(mcmc_model *x)
/*
 * Update c with a MH step. Returns the number of accepted samples 
 * (0 or 1).
 */
{
  int m;
  double y;

  if (x->manycd)
  {
    for (m = 0; m < x->M; m++)
    {
      y = gsl_vector_get(x->c, m);
      mcmc_samplebeta(&y, gsl_vector_int_get(x->f1, m), gsl_vector_int_get(x->t0, m), MINC, MAXC);
      gsl_vector_set(x->c, m, y);
    }
    return x->M;
  }
  else
  {
    y = gsl_vector_get(x->c, 0);
    mcmc_samplebeta(&y, x->f1a, x->t0a, MINC, MAXC);
    for (m = 0; m < x->M; m++)
      gsl_vector_set(x->c, m, y);
    return 1;
  }
}
/* end of mcmc_samplec */

int mcmc_sampled(mcmc_model *x)
/*
 * Update d with a MH step. Returns the number of accepted samples 
 * (0 or 1).
 */
{
  int m;
  double y;

  if (x->manycd)
  {
    for (m = 0; m < x->M; m++)
    {
      y = gsl_vector_get(x->d, m);
      mcmc_samplebeta(&y, gsl_vector_int_get(x->f0, m), gsl_vector_int_get(x->t1, m), MIND, MAXD);
      gsl_vector_set(x->d, m, y);
    }
    return x->M;
  }
  else
  {
    y = gsl_vector_get(x->d, 0);
    mcmc_samplebeta(&y, x->f0a, x->t1a, MIND, MAXD);
    for (m = 0; m < x->M; m++)
      gsl_vector_set(x->d, m, y);
    return 1;
  }
}
/* end of mcmc_sampled */

void mcmc_auxa(const gsl_vector_int *x, int b, double c, double d,
               int *a, int *t0, int *f0, int *t1, int *f1,
               gsl_vector *q,
               gsl_vector_int *dt0, gsl_vector_int *df0,
               gsl_vector_int *dt1, gsl_vector_int *df1)
/*
 * Auxiliary function that is used in Gibbs-sampling a(m) or b(m) by
 * mcmc_sampleab.
 */
{
  int i;
  double cc, dd;

  gsl_vector_set(q, *a, 0.);
  gsl_vector_int_set(dt0, *a, 0);
  gsl_vector_int_set(df0, *a, 0);
  gsl_vector_int_set(dt1, *a, 0);
  gsl_vector_int_set(df1, *a, 0);

  cc = log(1. - exp(c));
  dd = log(1. - exp(d));

  for (i = *a - 1; i >= 0; i--)
  {
    if (gsl_vector_int_get(x, i))
    {
      gsl_vector_int_set(dt0, i, gsl_vector_int_get(dt0, i + 1));
      gsl_vector_int_set(df0, i, gsl_vector_int_get(df0, i + 1));
      gsl_vector_int_set(dt1, i, gsl_vector_int_get(dt1, i + 1) + 1);
      gsl_vector_int_set(df1, i, gsl_vector_int_get(df1, i + 1) - 1);
    }
    else
    {
      gsl_vector_int_set(dt0, i, gsl_vector_int_get(dt0, i + 1) - 1);
      gsl_vector_int_set(df0, i, gsl_vector_int_get(df0, i + 1) + 1);
      gsl_vector_int_set(dt1, i, gsl_vector_int_get(dt1, i + 1));
      gsl_vector_int_set(df1, i, gsl_vector_int_get(df1, i + 1));
    }
  }
  for (i = *a + 1; i <= b; i++)
  {
    if (gsl_vector_int_get(x, i - 1))
    {
      gsl_vector_int_set(dt0, i, gsl_vector_int_get(dt0, i - 1));
      gsl_vector_int_set(df0, i, gsl_vector_int_get(df0, i - 1));
      gsl_vector_int_set(dt1, i, gsl_vector_int_get(dt1, i - 1) - 1);
      gsl_vector_int_set(df1, i, gsl_vector_int_get(df1, i - 1) + 1);
    }
    else
    {
      gsl_vector_int_set(dt0, i, gsl_vector_int_get(dt0, i - 1) + 1);
      gsl_vector_int_set(df0, i, gsl_vector_int_get(df0, i - 1) - 1);
      gsl_vector_int_set(dt1, i, gsl_vector_int_get(dt1, i - 1));
      gsl_vector_int_set(df1, i, gsl_vector_int_get(df1, i - 1));
    }
  }
  for (i = 0; i <= b; i++)
  {
    gsl_vector_set(q, i,
                   gsl_vector_int_get(dt0, i) * cc + gsl_vector_int_get(df0, i) * d + gsl_vector_int_get(dt1, i) * dd + gsl_vector_int_get(df1, i) * c);
  }

  *a = mcmc_randompick(mcmc_logtop(q, b + 1), b + 1);
  *t0 += gsl_vector_int_get(dt0, *a);
  *f0 += gsl_vector_int_get(df0, *a);
  *t1 += gsl_vector_int_get(dt1, *a);
  *f1 += gsl_vector_int_get(df1, *a);
}
/* end of mcmc_auxa */

int mcmc_randompick(const gsl_vector *p, int n)
/*
 * Pick one sample of a probability vector p of length n. Returns an
 * integer from interval [0,n[.
 */
{
  int i = 0;
  double x;
  x = gsl_rng_uniform(r) - gsl_vector_get(p, 0);
  while (x > 0. && i < n - 1)
  {
    x -= gsl_vector_get(p, ++i);
  }
  return i;
}
/* end of mcmc_randompick */

int mcmc_sampleab(mcmc_model *x)
/*
 * Sample all a's and b's with Gibbs method. Returns the number of changed
 * a's or b's (0...2M).
 */
{
  int m, t, count = 0;
  int t0, f0, t1, f1;
  double c, d;
  gsl_vector_int *v;
  gsl_vector *q;
  gsl_vector_int *dt0, *df0, *dt1, *df1;

  v = gsl_vector_int_alloc(x->N);

  q = gsl_vector_calloc(x->N + 1);
  dt0 = gsl_vector_int_alloc(x->N + 1);
  df0 = gsl_vector_int_alloc(x->N + 1);
  dt1 = gsl_vector_int_alloc(x->N + 1);
  df1 = gsl_vector_int_alloc(x->N + 1);

  for (m = 0; m < x->M; m++)
  {
    gsl_matrix_int_get_col(v, x->X, m);
    gsl_permute_vector_int(x->rpi, v);

    t = gsl_vector_int_get(x->a, m);
    t0 = gsl_vector_int_get(x->t0, m);
    f0 = gsl_vector_int_get(x->f0, m);
    t1 = gsl_vector_int_get(x->t1, m);
    f1 = gsl_vector_int_get(x->f1, m);
    c = gsl_vector_get(x->c, m);
    d = gsl_vector_get(x->d, m);
    mcmc_auxa(v, gsl_vector_int_get(x->b, m), c, d,
              &t, &t0, &f0, &t1, &f1,
              q, dt0, df0, dt1, df1);
    if (t != gsl_vector_int_get(x->a, m))
    {
      gsl_vector_int_set(x->a, m, t);
      count++;
    }

    gsl_vector_int_reverse(v);

    t = x->N - gsl_vector_int_get(x->b, m);
    mcmc_auxa(v, x->N - gsl_vector_int_get(x->a, m), c, d,
              &t, &t0, &f0, &t1, &f1,
              q, dt0, df0, dt1, df1);
    gsl_vector_int_set(x->t0, m, t0);
    gsl_vector_int_set(x->f0, m, f0);
    gsl_vector_int_set(x->t1, m, t1);
    gsl_vector_int_set(x->f1, m, f1);
    if (t != x->N - gsl_vector_int_get(x->b, m))
    {
      gsl_vector_int_set(x->b, m, x->N - t);
      count++;
    }
  }

  x->t0a = x->f0a = x->t1a = x->f1a = 0;
  for (m = 0; m < x->M; m++)
  {
    x->t0a += gsl_vector_int_get(x->t0, m);
    x->f0a += gsl_vector_int_get(x->f0, m);
    x->t1a += gsl_vector_int_get(x->t1, m);
    x->f1a += gsl_vector_int_get(x->f1, m);
  }

  x->loglik = mcmc_logl(x);
  gsl_vector_int_free(v);

  gsl_vector_free(q);
  gsl_vector_int_free(dt0);
  gsl_vector_int_free(df0);
  gsl_vector_int_free(dt1);
  gsl_vector_int_free(df1);

  return count;
}
/* end of mcmc_sampleab */

int mcmc_consistent(mcmc_model *x)
/*
 * Checks if the model is consistent, e.g., if the actual number of false
 * 0s and 1s matches the values stored in the model structure. Useful 
 * for debugging the sampling functions. The sampling functions should
 * maintain the model in a consistent state. Returns 1 and prints an
 * error message if there is a problem, otherwise returns 0.
 */
{
  int flag = 0, i, n, m, a, b, t0, f0, t1, f1;
  double loglik, delta;
  gsl_permutation *p;

  /*
   * Check a and b.
   */
  for (m = 0; m < x->M; m++)
  {
    a = gsl_vector_int_get(x->a, m);
    b = gsl_vector_int_get(x->b, m);
    if (!(0 <= a && a <= b && b <= x->N))
    {
      fprintf(stderr,
              "mcmc_consistent: error. a(%d) = %d  b(%d) = %d\n",
              m, a, m, b);
      flag = 1;
    }
  }

  /*
   * Check permutations pi and rpi and that hard sites are in correct 
   * order.
   */
  if (gsl_permutation_valid(x->pi) || gsl_permutation_valid(x->pi))
  {
    fprintf(stderr, "mcmc_consistent: invalid permutation.\n");
    flag = 1;
  }
  p = gsl_permutation_alloc(x->N);
  gsl_permutation_inverse(p, x->pi);
  for (n = 0; n < x->N; n++)
  {
    if (gsl_permutation_get(x->rpi, n) != gsl_permutation_get(p, n))
    {
      fprintf(stderr, "mcmc_consistent: rpi is not inverse of pi.\n");
      flag = 1;
      break;
    }
  }
  gsl_permutation_free(p);
  m = -1;
  i = 0;
  for (n = 0; n < x->N; n++)
  {
    if (gsl_vector_int_get(x->h, n))
    {
      i++;
      if (m >= 0 && gsl_permutation_get(x->pi, n) < m)
      {
        fprintf(stderr,
                "mcmc_consistent: hard site order is incorrect %d %d %d.\n",
                n, (int)gsl_permutation_get(x->pi, n), m);
        flag = 1;
      }
      m = gsl_permutation_get(x->pi, n);
    }
  }
  if (i != x->nh)
  {
    fprintf(stderr,
            "mcmc_consistent: incorrect number of hard sites %d %d.\n",
            i, x->nh);
    flag = 1;
  }

  t0 = x->t0a;
  f0 = x->f0a;
  t1 = x->t1a;
  f1 = x->f1a;
  loglik = x->loglik;
  mcmc_count01(x);
  x->loglik = mcmc_logl(x);
  delta = loglik - x->loglik;
  if (delta < 0.)
    delta = -delta;
  if (t0 != x->t0a || f0 != x->f0a || t1 != x->t1a || f1 != x->f1a || delta > 1e-8)
  {
    fprintf(stderr,
            "mcmc_consistent: inconsistent parameters.\n",
            "t0 %d %d\nf0 %d %d\nt1 %d %d\nf1 %d %d\nlogl %g %g\n",
            t0, x->t0a, f0, x->f0a, t1, x->t1a, f1, x->f1a, loglik, x->loglik);
    flag = 1;
  }

  return flag;
}
/* end of mcmc_consistent */

int mcmc_ininterval(int i, int a, int b, int inc1, int inc2)
/*
 * Auxiliary funtion to check whether i is within an interval limited
 * by a and b.
 */
{
  int r;

  if (a > b)
  {
    r = a;
    a = b;
    b = r;
  }
  if (inc1)
    r = (a <= i);
  else
    r = (a < i);
  if (r)
  {
    if (inc2)
      r = (i <= b);
    else
      r = (i < b);
  }

  return r;
}
/* end of mcmc_ininterval */

int mcmc_samplepi1(mcmc_model *x)
/*
 * Sample the permutation, pi, with MH method.
 * The proposal is to move a randomly selected row at position i
 * to position j, simultaneously moving all intervening rows and
 * limits (a's and b's), accordingly.
 */
{
  int n, m, i, j, ii, jj, a, b, ain, bin, t;
  int dt0, df0, dt1, df1;
  double delta, c, d;
  gsl_vector_int v;

  i = gsl_rng_uniform_int(r, x->N);
  j = gsl_rng_uniform_int(r, x->N - 1);
  if (j >= i)
    j++;
  if (i < j)
  {
    ii = i;
    jj = j;
  }
  else
  {
    ii = j;
    jj = i;
  }

  /*
   * Moving the site at position i to position j is forbidden, if
   * the site at i is a hard site _and_ if there is another hard
   * site within the interval defined by i and j.
   */
  if (gsl_vector_int_get(x->h, gsl_permutation_get(x->rpi, i)))
  {
    m = 0;
    for (n = ii; n <= jj; n++)
    {
      m += gsl_vector_int_get(x->h, gsl_permutation_get(x->rpi, n));
      if (m > 1)
        return 0;
    }
  }

  /*
   * Compute the changes in the number of true and false 0s and 1s
   * resulting from the proposed permutation.
   */
  v = gsl_matrix_int_row(x->X, gsl_permutation_get(x->rpi, i)).vector;
  delta = 0.;
  if (i < j)
  {
    for (m = 0; m < x->M; m++)
    {
      dt0 = df0 = dt1 = df1 = 0;
      a = gsl_vector_int_get(x->a, m);
      b = gsl_vector_int_get(x->b, m);
      ain = (ii < a && a <= jj + 1);
      bin = (ii < b && b <= jj + 1);
      if (ain && !bin)
      {
        if (gsl_vector_int_get(&v, m))
        {
          dt1++;
          df1--;
        }
        else
        {
          dt0--;
          df0++;
        }
      }
      else if (!ain && bin)
      {
        if (gsl_vector_int_get(&v, m))
        {
          dt1--;
          df1++;
        }
        else
        {
          dt0++;
          df0--;
        }
      }
      c = gsl_vector_get(x->c, m);
      d = gsl_vector_get(x->d, m);
      delta += dt0 * log(1. - exp(c)) + df0 * d + dt1 * log(1. - exp(d)) + df1 * c;
    }
  }
  else
  {
    for (m = 0; m < x->M; m++)
    {
      dt0 = df0 = dt1 = df1 = 0;
      a = gsl_vector_int_get(x->a, m);
      b = gsl_vector_int_get(x->b, m);
      ain = (ii <= a && a <= jj);
      bin = (ii <= b && b <= jj);
      if (!ain && bin)
      {
        if (gsl_vector_int_get(&v, m))
        {
          dt1++;
          df1--;
        }
        else
        {
          dt0--;
          df0++;
        }
      }
      else if (ain && !bin)
      {
        if (gsl_vector_int_get(&v, m))
        {
          dt1--;
          df1++;
        }
        else
        {
          df0--;
          dt0++;
        }
      }
      c = gsl_vector_get(x->c, m);
      d = gsl_vector_get(x->d, m);
      delta += dt0 * log(1. - exp(c)) + df0 * d + dt1 * log(1. - exp(d)) + df1 * c;
    }
  }

  /*
   * Accept the proposed permutation with the normal MH probability.
   */
  if (delta >= 0. || delta > log(gsl_rng_uniform_pos(r)))
  {
    /*
     * Change the permutation and move the a's and b's accordingly.
     */
    if (i < j)
    {
      for (m = 0; m < x->M; m++)
      {
        a = gsl_vector_int_get(x->a, m);
        b = gsl_vector_int_get(x->b, m);
        if (ii < a && a <= jj + 1)
          gsl_vector_int_set(x->a, m, a - 1);
        if (ii < b && b <= jj + 1)
          gsl_vector_int_set(x->b, m, b - 1);
      }
      t = x->rpi->data[i];
      for (n = i; n < j; n++)
        x->rpi->data[n] = x->rpi->data[n + 1];
      x->rpi->data[j] = t;
    }
    else
    {
      for (m = 0; m < x->M; m++)
      {
        a = gsl_vector_int_get(x->a, m);
        b = gsl_vector_int_get(x->b, m);
        if (ii <= a && a <= jj)
          gsl_vector_int_set(x->a, m, a + 1);
        if (ii <= b && b <= jj)
          gsl_vector_int_set(x->b, m, b + 1);
      }
      t = x->rpi->data[i];
      for (n = i; n > j; n--)
        x->rpi->data[n] = x->rpi->data[n - 1];
      x->rpi->data[j] = t;
    }
    /* 
     * Update also the auxiliary variables.
     */
    gsl_permutation_inverse(x->pi, x->rpi);
    x->loglik += delta;
    mcmc_count01(x);
    return 1;
  }

  return 0;
}
/* end of mcmc_samplepi1 */

int mcmc_samplepi2(mcmc_model *x, int swap)
/*
 * Sample the permutation, pi, with MH method.
 * The proposal is to reverse the order of rows and limits (a's and 
 * b's) within an interval defined by randomly selected i and j.
 * j=i+1 if swap=1 (swap neighbouring rows).
 */
{
  int i, j, n, m, a, b, inc1, inc2, ain, bin;
  int dt0, df0, dt1, df1;
  double delta, c, d;

  if (!swap)
  {
    i = gsl_rng_uniform_int(r, x->N);
    j = gsl_rng_uniform_int(r, x->N - 1);
    if (j >= i)
    {
      j++;
    }
    else
    {
      n = i;
      i = j;
      j = n;
    }
  }
  else
  {
    i = gsl_rng_uniform_int(r, x->N - 1);
    j = i + 1;
  }

  /*
   * The permutation is forbidden iff the interval [i,j] contains at 
   * least two hard sites.
   */
  m = 0;
  for (n = i; n <= j; n++)
  {
    m += gsl_vector_int_get(x->h, gsl_permutation_get(x->rpi, n));
    if (m > 1)
      return 0;
  }

  /*
   * We have several options to handle the a's and b's at the borders
   * of the interval defined by i and j. Randomly pick one method.
   */
  inc1 = gsl_rng_uniform_int(r, 2);
  inc2 = gsl_rng_uniform_int(r, 2);

  /*
   * Compute the changes in the number of true and false 0s and 1s
   * resulting from the proposed permutation.
   */
  delta = 0.;
  for (m = 0; m < x->M; m++)
  {
    dt0 = df0 = dt1 = df1 = 0;
    a = gsl_vector_int_get(x->a, m);
    b = gsl_vector_int_get(x->b, m);
    ain = mcmc_ininterval(a, i, j + 1, inc1, inc2);
    bin = mcmc_ininterval(b, i, j + 1, inc1, inc2);
    if (ain && !bin)
    {
      for (n = i; n < a; n++)
      {
        if (gsl_matrix_int_get(x->X, gsl_permutation_get(x->rpi, n), m))
        {
          dt1++;
          df1--;
        }
        else
        {
          dt0--;
          df0++;
        }
      }
      for (n = a; n <= j; n++)
      {
        if (gsl_matrix_int_get(x->X, gsl_permutation_get(x->rpi, n), m))
        {
          dt1--;
          df1++;
        }
        else
        {
          dt0++;
          df0--;
        }
      }
    }
    else if (!ain && bin)
    {
      for (n = i; n < b; n++)
      {
        if (gsl_matrix_int_get(x->X, gsl_permutation_get(x->rpi, n), m))
        {
          dt1--;
          df1++;
        }
        else
        {
          dt0++;
          df0--;
        }
      }
      for (n = b; n <= j; n++)
      {
        if (gsl_matrix_int_get(x->X, gsl_permutation_get(x->rpi, n), m))
        {
          dt1++;
          df1--;
        }
        else
        {
          dt0--;
          df0++;
        }
      }
    }
    c = gsl_vector_get(x->c, m);
    d = gsl_vector_get(x->d, m);
    delta += dt0 * log(1. - exp(c)) + df0 * d + dt1 * log(1. - exp(d)) + df1 * c;
  }

  /*
   * Accept the proposed permutation with the normal MH probability.
   */
  if (delta >= 0. || delta > log(gsl_rng_uniform_pos(r)))
  {
    /*
     * Update the a's and b's accordingly.
     */
    for (m = 0; m < x->M; m++)
    {
      a = gsl_vector_int_get(x->a, m);
      b = gsl_vector_int_get(x->b, m);
      ain = mcmc_ininterval(a, i, j + 1, inc1, inc2);
      bin = mcmc_ininterval(b, i, j + 1, inc1, inc2);
      if (ain && !bin)
      {
        gsl_vector_int_set(x->a, m, i + j + 1 - a);
      }
      else if (!ain && bin)
      {
        gsl_vector_int_set(x->b, m, i + j + 1 - b);
      }
      else if (ain && bin)
      {
        gsl_vector_int_set(x->b, m, i + j + 1 - a);
        gsl_vector_int_set(x->a, m, i + j + 1 - b);
      }
    }
    /*
     * Update the permutation...
     */
    for (n = i; n <= (i + j) / 2; n++)
    {
      m = x->rpi->data[n];
      x->rpi->data[n] = x->rpi->data[i + j - n];
      x->rpi->data[i + j - n] = m;
    }
    /*
     * ...and the auxiliary variables.
     */
    gsl_permutation_inverse(x->pi, x->rpi);
    x->loglik += delta;
    mcmc_count01(x);

    return 1;
  }

  return 0;
}
/* end of mcmc_samplepi2 */

int mcmc_samplepi3(mcmc_model *x, gsl_vector_int *p)
/*
 * Sample the permutation, pi, with MH method.
 * The proposal is to reverse the order of rows and limits (a's and 
 * b's) within an interval defined by randomly selected i and j.
 * Only the order of non-hard sites is swapped, hard sites are left
 * alone.
 */
{
  int i, j, n, m, a, b, nn, na, nb, inc1, inc2, ain, bin, wasalive, isalive;
  int dt0, df0, dt1, df1;
  double delta, c, d;

  if (x->N - x->nh < 2)
    return 0;

  n = gsl_rng_uniform_int(r, x->N - x->nh);
  m = gsl_rng_uniform_int(r, x->N - x->nh - 1);
  if (n <= m)
  {
    i = n;
    j = m + 1;
  }
  else
  {
    i = m;
    j = n;
  }

  n = 0;
  while (n <= i)
  {
    if (gsl_vector_int_get(x->h, gsl_permutation_get(x->rpi, n)))
    {
      i++;
      j++;
    }
    n++;
  }
  while (n <= j)
  {
    if (gsl_vector_int_get(x->h, gsl_permutation_get(x->rpi, n)))
      j++;
    n++;
  }
  n = i;
  m = j;
  while (n <= m)
  {
    if (gsl_vector_int_get(x->h, gsl_permutation_get(x->rpi, n)))
    {
      gsl_vector_int_set(p, n, n);
      n++;
    }
    else if (gsl_vector_int_get(x->h, gsl_permutation_get(x->rpi, m)))
    {
      gsl_vector_int_set(p, m, m);
      m--;
    }
    else
    {
      gsl_vector_int_set(p, n, m);
      gsl_vector_int_set(p, m, n);
      n++;
      m--;
    }
  }

  /*
   * We have several options to handle the a's and b's at the borders
   * of the interval defined by i and j. Randomly pick one method.
   */
  inc1 = gsl_rng_uniform_int(r, 2);
  inc2 = gsl_rng_uniform_int(r, 2);

  /*
   * Compute the changes in the number of true and false 0s and 1s
   * resulting from the proposed permutation.
   */
  delta = 0.;
  for (m = 0; m < x->M; m++)
  {
    dt0 = df0 = dt1 = df1 = 0;
    a = gsl_vector_int_get(x->a, m);
    b = gsl_vector_int_get(x->b, m);
    ain = mcmc_ininterval(a, i, j + 1, inc1, inc2);
    bin = mcmc_ininterval(b, i, j + 1, inc1, inc2);
    if (ain && !bin)
    {
      na = i + j + 1 - a;
      nb = b;
    }
    else if (!ain && bin)
    {
      na = a;
      nb = i + j + 1 - b;
    }
    else if (ain && bin)
    {
      na = i + j + 1 - b;
      nb = i + j + 1 - a;
    }
    else
    {
      na = a;
      nb = b;
    }
    for (n = i; n <= j; n++)
    {
      nn = gsl_vector_int_get(p, n);
      wasalive = (a <= n && n < b);
      isalive = (na <= nn && nn < nb);
      if (wasalive && !isalive)
      {
        if (gsl_matrix_int_get(x->X, gsl_permutation_get(x->rpi, n), m))
        {
          dt1--;
          df1++;
        }
        else
        {
          df0--;
          dt0++;
        }
      }
      else if (!wasalive && isalive)
      {
        if (gsl_matrix_int_get(x->X, gsl_permutation_get(x->rpi, n), m))
        {
          dt1++;
          df1--;
        }
        else
        {
          df0++;
          dt0--;
        }
      }
    }
    c = gsl_vector_get(x->c, m);
    d = gsl_vector_get(x->d, m);
    delta += dt0 * log(1. - exp(c)) + df0 * d + dt1 * log(1. - exp(d)) + df1 * c;
  }

  /*
   * Accept the proposed permutation with the normal MH probability.
   */
  if (delta >= 0. || delta > log(gsl_rng_uniform_pos(r)))
  {
    /*
     * Update the a's and b's accordingly.
     */
    for (m = 0; m < x->M; m++)
    {
      a = gsl_vector_int_get(x->a, m);
      b = gsl_vector_int_get(x->b, m);
      ain = mcmc_ininterval(a, i, j + 1, inc1, inc2);
      bin = mcmc_ininterval(b, i, j + 1, inc1, inc2);
      if (ain && !bin)
      {
        gsl_vector_int_set(x->a, m, i + j + 1 - a);
      }
      else if (!ain && bin)
      {
        gsl_vector_int_set(x->b, m, i + j + 1 - b);
      }
      else if (ain && bin)
      {
        gsl_vector_int_set(x->b, m, i + j + 1 - a);
        gsl_vector_int_set(x->a, m, i + j + 1 - b);
      }
    }
    /*
     * Update the permutation...
     */
    for (n = i; n <= j; n++)
    {
      a = gsl_vector_int_get(p, n);
      gsl_vector_int_set(p, n, gsl_permutation_get(x->rpi, a));
    }
    for (n = i; n <= j; n++)
      x->rpi->data[n] = gsl_vector_int_get(p, n);
    /*
     * ...and the auxiliary variables.
     */
    gsl_permutation_inverse(x->pi, x->rpi);
    x->loglik += delta;
    mcmc_count01(x);

    return 1;
  }

  return 0;
}
/* end of mcmc_samplepi3 */
