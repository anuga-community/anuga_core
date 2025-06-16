#include "math.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#if defined(__APPLE__)
// clang doesn't have openmp
#else
#include "omp.h"
#endif

#ifdef USE_LIB_BLAS
#include <cblas.h>
#endif

void anuga_daxpy(const int64_t N, const double alpha, const double *X, const int incX, double *Y, const int64_t incY)
{
#ifdef USE_LIB_BLAS
  // Use BLAS for optimized performance
  cblas_daxpy(N, alpha, X, incX, Y, incY);
  return;
  #else
#pragma omp parallel for simd schedule(static)
  for (int64_t i = 0; i < N; i++)
  {
    Y[i*incY] += alpha * X[i*incX];
  }
  #endif
}

void anuga_dscal(const int64_t N, const double alpha, double *X, const int64_t incX)
{
    #ifdef USE_LIB_BLAS
    cblas_dscal(N, alpha, X, incX);
    return;
    #else
#pragma omp parallel for simd schedule(static)
  for (int64_t i = 0; i < N; i++)
  {
    X[i*incX] *= alpha;
  }
  #endif
}