#include <gsl/gsl_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include "log.h"

gsl_rng *r;

int main()
{
  FILE *f = fopen("logs.txt", "w");
  log_set_fp(f);
  gsl_rng_env_setup();
  r = gsl_rng_alloc(gsl_rng_default);

  gsl_matrix *X = gsl_matrix_int_calloc(3, 3);
  for (size_t i = 0; i < 3; i++)
  {
    for (size_t j = 0; j < 3; j++)
    {
      gsl_matrix_int_set(X, i, j, i + j);
    }
  }

  log_trace("HELLO WORLD");

  for (size_t i = 0; i < 3; i++)
  {
    for (size_t j = 0; j < 3; j++)
    {
      printf("%d ", gsl_matrix_int_get(X, i, j));
    }
    printf("\n");
  }

  printf("\n\nOG Permutation");

  gsl_permutation *p = gsl_permutation_calloc(3);
  for (size_t i = 0; i < 3; i++)
  {
    printf("%d ", gsl_permutation_get(p, i));
  }
  printf("\n\nShuffled Permutation");
  gsl_ran_shuffle(r, p->data, 3, sizeof(size_t));
  for (size_t i = 0; i < 3; i++)
  {
    printf("%d ", gsl_permutation_get(p, i));
  }

  printf("\n");

  gsl_permutation *ip = gsl_permutation_alloc(3); // init permutation but elements are undefined
  gsl_permutation_inverse(ip, p);
  for (size_t i = 0; i < 3; i++)
  {
    printf("%d ", gsl_permutation_get(p, i));
  }

  printf("\n");

  gsl_vector *v = gsl_vector_int_alloc(3);
  gsl_matrix_int_get_col(v, X, 1);
  for (size_t i = 0; i < 3; i++)
  {
    printf("%d ", gsl_vector_int_get(v, i));
  }

  printf("\n");

  gsl_permute_vector_int(ip, v);
  for (size_t i = 0; i < 3; i++)
  {
    printf("%d ", gsl_vector_int_get(v, i));
  }
}
