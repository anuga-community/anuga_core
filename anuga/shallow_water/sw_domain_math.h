#ifndef ANUGA_SHALLOW_WATER_SW_DOMAIN_MATH_H
#define ANUGA_SHALLOW_WATER_SW_DOMAIN_MATH_H
#include "math.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#ifdef USE_LIB_BLAS
#include <cblas.h>
#endif

#include "anuga_runtime.h"
#include "anuga_typedefs.h"

void anuga_daxpy(const anuga_int N, const double alpha, const double *X, const int incX, double *Y, const anuga_int incY)
{
#ifdef USE_LIB_BLAS
  // Use BLAS for optimized performance
  cblas_daxpy(N, alpha, X, incX, Y, incY);
  return;
  #else
#pragma omp parallel for simd schedule(static)
  for (anuga_int i = 0; i < N; i++)
  {
    Y[i*incY] += alpha * X[i*incX];
  }
  #endif
}

void anuga_dscal(const anuga_int N, const double alpha, double *X, const anuga_int incX)
{
    #ifdef USE_LIB_BLAS
    cblas_dscal(N, alpha, X, incX);
    return;
    #else
#pragma omp parallel for simd schedule(static)
  for (anuga_int i = 0; i < N; i++)
  {
    X[i*incX] *= alpha;
  }
  #endif
}
#endif // ANUGA_SHALLOW_WATER_SW_DOMAIN_MATH_H 