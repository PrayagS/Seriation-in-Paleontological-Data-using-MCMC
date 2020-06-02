/* $Id: mcmc.h,v 1.8 2005/12/20 15:54:08 kaip Exp $ */

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

#define MAXS 2000
#define LOGEPSILON (-32.236191301916641) /* log(1e-14) */
#define MINC (-6.9077552789821368)       /* log(.001)  */
#define MAXC (-2.3025850929940455)       /* log(.1)    */
#define MIND (-1.6094379124341003)       /* log(.2)    */
#define MAXD (-0.22314355131420971)      /* log(.8)    */

typedef struct
{
  int N, M;                          /* Numbers of sites and species */
  gsl_matrix_int *X;                 /* NxM data matrix */
  gsl_vector_int *a, *b;             /* Births and deaths of species */
  gsl_permutation *pi, *rpi;         /* Order of sites */
  gsl_vector_int *h;                 /* Hard sites */
  gsl_vector *c, *d;                 /* Log-probability of false 1 (c) and 0 (d) */
  int manycd;                        /* True if all genera have their own c/d. */
  double loglik;                     /* Auxiliary variables */
  gsl_vector_int *t0, *f0, *t1, *f1; /* Vectors containing no. of true 0s,1s anf false 0s,1s for each taxa */
  int t0a, f0a, t1a, f1a;            /* Total no. of true 0s,1s anf false 0s,1s */
  int nh;                            /* No. of hard ordered sites */
} mcmc_model;

mcmc_model *mcmc_readmodel(mcmc_model *x, FILE *f, int manycd);
int mcmc_sample(mcmc_model *x);
void mcmc_print(const mcmc_model *x, FILE *f);
void mcmc_save(const mcmc_model *x, FILE *f1, FILE *f2, FILE *f3);
void mcmc_init(void);
void mcmc_free(void);
void mcmc_freemodel(mcmc_model *x);
void mcmc_initab(mcmc_model *x);
void mcmc_randomize(mcmc_model *x);
double mcmc_logl(const mcmc_model *x);
void mcmc_count01(mcmc_model *x);
gsl_vector *mcmc_logtop(gsl_vector *p, int n);
int mcmc_samplec(mcmc_model *x);
int mcmc_sampled(mcmc_model *x);
void mcmc_auxa(const gsl_vector_int *x, int b, double c, double d,
               int *a, int *t0, int *f0, int *t1, int *f1,
               gsl_vector *q,
               gsl_vector_int *dt0, gsl_vector_int *df0,
               gsl_vector_int *dt1, gsl_vector_int *df1);
int mcmc_randompick(const gsl_vector *p, int n);
int mcmc_sampleab(mcmc_model *x);
int mcmc_consistent(mcmc_model *x);
int mcmc_ininterval(int i, int a, int b, int inc1, int inc2);
int mcmc_samplepi1(mcmc_model *x);
int mcmc_samplepi2(mcmc_model *x, int swap);
int mcmc_samplepi3(mcmc_model *x, gsl_vector_int *p);
double mcmc_samplebeta(double *x, double a, double b, double low, double high);
