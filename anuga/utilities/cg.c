// C extension for the cg_solve module. Implements a c routine of the 
// conjugate gradient algorithm to solve Ax=b, using sparse matrix A in 
// the csr format.
//
// See the module cg_solve.py
//
// Padarn Wilson 2012
//
// Note Padarn 26/11/12: Currently the input matrix for the CG solve must
// be a Sparse_CSR matrix python object - defined in anuga/utilities/sparse.py
//
// Note Padarn 5/12/12: I have tried a few optimization modifications which
// didn't seem to save any time:
// -- conversion of the int arrays to long arrays (to save memory passing time)
// -- taking advantage of the symmetric quality of the matrix A to reduce the zAx loop
// -- specifying different 'chunk' sizes for the openmp loops
// -- using blas instead of own openmp loops
	
#include "math.h"
#include "stdio.h"
#include <stdint.h>
#include "anuga_typedefs.h"

#if defined(__APPLE__)
   // clang doesn't have openmp
#else
   #include "omp.h"
#endif



// Dot product of two double vectors: a.b
// @input N: anuga_int length of vectors a and b
//        a: first vector of doubles
//        b: second vector of double
// @return: double result of a.b 
double cg_ddot( anuga_int N, double *a, double *b)
{
  double ret = 0;
  anuga_int i;
  #pragma omp parallel for private(i) reduction(+:ret)
  for(i=0;i<N;i++)
  {
    ret+=a[i]*b[i];
  }
  return ret;

}

// In place multiplication of a double vector x by constant a: a*x
// @input N: anuga_int length of vector x
//        a: double scalar to multiply by
//        x: double vector to scale
void cg_dscal(anuga_int N, double a, double *x)
{
  anuga_int i;
  #pragma omp parallel for private(i)
  for(i=0;i<N;i++)
  {
    x[i]=a*x[i];
  }

}

// Copy of one vector to another - memory already allocated: y=x
// @input N: anuga_int length of vectors x and y
//        x: double vector to make copy of
//        y: double vector to copy into
void cg_dcopy( anuga_int N, double *x, double *y)
{
  anuga_int i;
  #pragma omp parallel for private(i)
  for(i=0;i<N;i++)
  {
    y[i]=x[i];
  }
}

// In place axpy operation: y = a*x + y
// @input N: anuga_int length of vectors x and y
//        a: double to multiply x by
//        x: first double vector
//        y: second double vector, stores result
void cg_daxpy(anuga_int N, double a, double *x, double *y)
{
  anuga_int i;
  #pragma omp parallel for private(i)
  for(i=0;i<N;i++)
  {
    y[i]=y[i]+a*x[i];
  }
}

// Sparse CSR matrix-vector product: z = A*x
// @input z: double vector to store the result
//        data: double vector with non-zero entries of A
//        colind: anuga_int vector of column indicies of non-zero entries of A
//        row_ptr: anuga_int vector giving index of rows for non-zero entires of A
//        x: double vector to be multiplied
//        M: length of vector x
void cg_zAx(double * z, double * data, anuga_int * colind, anuga_int * row_ptr, double * x, anuga_int M){

  
  
  anuga_int i, j, ckey;


     
    #pragma omp parallel for private(ckey,j,i)
    for (i=0; i<M; i++){
      z[i]=0;
      for (ckey=row_ptr[i]; ckey<row_ptr[i+1]; ckey++) {
        j = colind[ckey];
        z[i] += data[ckey]*x[j];
      }              
    }
  

}

// Diagonal matrix-vector product: z = D*x
// @input z: double vector to store the result
//        D: double vector of diagonal matrix
//        x: double vector to be multiplied
//        M: length of vector x
void cg_zDx(double * z, double * D, double * x, anuga_int M){

  
  anuga_int i;
   
    #pragma omp parallel for private(i)
    for (i=0; i<M; i++){
      z[i]=D[i]*x[i];              
    }
  

}

// Diagonal matrix-vector product: z = D*x
// @input z: double vector to store the result
//        D: double vector of diagonal matrix
//        x: double vector to be multiplied
//        M: length of vector x
void cg_zDinx(double * z, double * D, double * x, anuga_int M){

  
  anuga_int i;
   
    #pragma omp parallel for private(i)
    for (i=0; i<M; i++){
      z[i]=1.0/D[i]*x[i];              
    }
  

}



// Sparse CSR matrix-vector product and vector addition: z = a*A*x + y
// @input z: double vector to store the result
//        a: double to scale matrix-vector product by
//        data: double vector with non-zero entries of A
//        colind: anuga_int vector of column indicies of non-zero entries of A
//        row_ptr: anuga_int vector giving index of rows for non-zero entires of A
//        x: double vector to be multiplied
//        y: double vector to add
//        M: length of vector x
void cg_zaAxpy(double * z, double a, double * data, anuga_int * colind, anuga_int * row_ptr, double * x,
      double * y,anuga_int M){
  anuga_int i, j, ckey;
  #pragma omp parallel for private(ckey,j,i)
    for (i=0; i<M; i++ ){
      z[i]=y[i];

      for (ckey=row_ptr[i]; ckey<row_ptr[i+1]; ckey++) {
        j = colind[ckey];
        z[i] += a*data[ckey]*x[j];
      }              

  }

}

// Jacobi preconditioner for matrix, A, and right hand side, b. Mutiplies each row
// by one divided by the diagonal element of the matrix. If the diagonal 
// element is zero, does nothing (should nnot occur)
//        colind: anuga_int vector of column indicies of non-zero entries of A
//        row_ptr: anuga_int vector giving index of rows for non-zero entires of A
//        b: double vector specifying right hand side of equation to solve
//        M: length of vector b

anuga_int _jacobi_precon_c(double* data, 
                anuga_int* colind,
                anuga_int* row_ptr,
                double * precon,
                anuga_int M){


  anuga_int i, j, ckey;
  double diag;


     
  #pragma omp parallel for private(diag,ckey,j,i)
  for (i=0; i<M; i++){
    diag = 0;
    for (ckey=row_ptr[i]; ckey<row_ptr[i+1]; ckey++) {
      j = colind[ckey];
      if (i==j){
        diag = data[ckey];
      }
    }
    if (diag == 0){
      diag =1;
    }
    precon[i]=diag;
  }
  
  return 0;

}

// Conjugate gradient solve Ax = b for x, A given in Sparse CSR format
// @input data: double vector with non-zero entries of A
//        colind: anuga_int vector of column indicies of non-zero entries of A
//        row_ptr: anuga_int vector giving index of rows for non-zero entires of A
//        b: double vector specifying right hand side of equation to solve
//        x: double vector with initial guess and to store result
//        imax: maximum number of iterations
//        tol: error tollerance for stopping criteria
//        M: length of vectors x and b
// @return: 0 on success  
anuga_int _cg_solve_c(double* data, 
                anuga_int* colind,
                anuga_int* row_ptr,
                double * b,
                double * x,
                anuga_int imax,
                double tol,
                double a_tol,
                anuga_int M){

  anuga_int i = 1;
  double alpha,rTr,rTrOld,bt,rTr0;

  double * d = malloc(sizeof(double)*M);
  double * r = malloc(sizeof(double)*M);
  double * q = malloc(sizeof(double)*M);
  double * xold = malloc(sizeof(double)*M);

  cg_zaAxpy(r,-1.0,data,colind,row_ptr,x,b,M);
  cg_dcopy(M,r,d);

  rTr=cg_ddot(M,r,r);
  rTr0 = rTr;
  
  while((i<imax) && (rTr>pow(tol,2)*rTr0) && (rTr > pow(a_tol,2))){

    cg_zAx(q,data,colind,row_ptr,d,M);
    alpha = rTr/cg_ddot(M,d,q);
    cg_dcopy(M,x,xold);
    cg_daxpy(M,alpha,d,x);

    cg_daxpy(M,-alpha,q,r);
    rTrOld = rTr;
    rTr = cg_ddot(M,r,r);

    bt= rTr/rTrOld;

    cg_dscal(M,bt,d);
    cg_daxpy(M,1.0,r,d);

    i=i+1;

  }
  
  free(d);
  free(r);
  free(q);
  free(xold);

  if (i>=imax){
    return -1;
  }
  else{
    return 0;
  }
  

}          

// Conjugate gradient solve Ax = b for x, A given in Sparse CSR format,
// using a diagonal preconditioner M. 
// @input data: double vector with non-zero entries of A
//        colind: anuga_int vector of column indicies of non-zero entries of A
//        row_ptr: anuga_int vector giving index of rows for non-zero entires of A
//        b: double vector specifying right hand side of equation to solve
//        x: double vector with initial guess and to store result
//        imax: maximum number of iterations
//        tol: error tollerance for stopping criteria
//        M: length of vectors x and b
//        precon: diagonal preconditioner given as vector
// @return: 0 on success  
anuga_int _cg_solve_c_precon(double* data, 
                anuga_int* colind,
                anuga_int* row_ptr,
                double * b,
                double * x,
                anuga_int imax,
                double tol,
                double a_tol,
                anuga_int M,
                double * precon){

  anuga_int i = 1;
  double alpha,rTr,rTrOld,bt,rTr0;

  double * d = malloc(sizeof(double)*M);
  double * r = malloc(sizeof(double)*M);
  double * q = malloc(sizeof(double)*M);
  double * xold = malloc(sizeof(double)*M);
  double * rhat = malloc(sizeof(double)*M);
  double * temp = malloc(sizeof(double)*M);

  cg_zaAxpy(r,-1.0,data,colind,row_ptr,x,b,M);
  cg_zDinx(rhat,precon,r,M);
  cg_dcopy(M,rhat,d);

  rTr=cg_ddot(M,r,rhat);
  rTr0 = rTr;
  
  while((i<imax) && (rTr>pow(tol,2)*rTr0) && (rTr > pow(a_tol,2))){

    cg_zAx(q,data,colind,row_ptr,d,M);
    alpha = rTr/cg_ddot(M,d,q);
    cg_dcopy(M,x,xold);
    cg_daxpy(M,alpha,d,x);

    cg_daxpy(M,-alpha,q,r);
    cg_zDinx(rhat,precon,r,M);
    rTrOld = rTr;
    rTr = cg_ddot(M,r,rhat);

    bt= rTr/rTrOld;

    cg_dscal(M,bt,d);
    cg_daxpy(M,1.0,rhat,d);

    i=i+1;

  }
  free(temp);
  free(rhat);
  free(d);
  free(r);
  free(q);
  free(xold);

  if (i>=imax){
    return -1;
  }
  else{
    return 0;
  }
  

}       

// SSOR preconditioner for CG solver
// Computes M^{-1} * r for SSOR preconditioning where M = (D/omega + L) * (D/omega)^{-1} * (D/omega + U)
// For simplicity, stores the effective diagonal scaling factor for each row.
// omega is the relaxation parameter (typically 1.0-1.8)
//
// The SSOR preconditioner application: z = M^{-1} * r requires:
//   Forward sweep: (D/omega + L) * temp = r
//   Scaling: temp = (omega / (2 - omega)) * D^{-1} * temp
//   Backward sweep: (D/omega + U) * z = temp
// This reduces CG iterations by 2-5x compared to Jacobi for typical FEM matrices.
//
// @input data: double vector with non-zero entries of A
//        colind: anuga_int vector of column indices of non-zero entries of A
//        row_ptr: anuga_int vector giving index of rows for non-zero entries of A
//        r: double vector (right-hand side / residual)
//        z: double vector to store the result M^{-1} * r
//        omega: relaxation parameter (typically 1.0-1.8)
//        M: number of rows
void _ssor_apply_c(double* data, anuga_int* colind, anuga_int* row_ptr,
                   double* r, double* z, double omega, anuga_int M)
{
  anuga_int i, ckey, j;
  double diag_i;

  double * temp = malloc(sizeof(double)*M);
  double * diag = malloc(sizeof(double)*M);

  // Extract diagonal entries
  for (i = 0; i < M; i++) {
    diag[i] = 1.0;
    for (ckey = row_ptr[i]; ckey < row_ptr[i+1]; ckey++) {
      if (colind[ckey] == i) {
        diag[i] = data[ckey];
        break;
      }
    }
  }

  // Forward sweep: (D/omega + L) * temp = r
  for (i = 0; i < M; i++) {
    temp[i] = r[i];
    for (ckey = row_ptr[i]; ckey < row_ptr[i+1]; ckey++) {
      j = colind[ckey];
      if (j < i) {
        temp[i] -= data[ckey] * temp[j];
      }
    }
    temp[i] *= omega / diag[i];
  }

  // Diagonal scaling: temp = (omega / (2 - omega)) * D^{-1} * temp
  // Combined: temp[i] *= diag[i] * (2-omega) / omega
  // This is equivalent to: temp = (D/omega)^{-1} * D * temp * (2-omega)/omega
  // = D * (omega/(2-omega))^{-1} * D^{-1} * temp ... simplifies to scaling
  for (i = 0; i < M; i++) {
    temp[i] *= diag[i] * (2.0 - omega) / omega;
  }

  // Backward sweep: (D/omega + U) * z = temp
  for (i = M - 1; i >= 0; i--) {
    z[i] = temp[i];
    for (ckey = row_ptr[i]; ckey < row_ptr[i+1]; ckey++) {
      j = colind[ckey];
      if (j > i) {
        z[i] -= data[ckey] * z[j];
      }
    }
    z[i] *= omega / diag[i];
  }

  free(temp);
  free(diag);
}

// Conjugate gradient solve Ax = b for x, A given in Sparse CSR format,
// using SSOR preconditioning.
// @input data: double vector with non-zero entries of A
//        colind: anuga_int vector of column indices of non-zero entries of A
//        row_ptr: anuga_int vector giving index of rows for non-zero entries of A
//        b: double vector specifying right hand side of equation to solve
//        x: double vector with initial guess and to store result
//        imax: maximum number of iterations
//        tol: relative error tolerance for stopping criteria
//        a_tol: absolute error tolerance for stopping criteria
//        M: length of vectors x and b
//        omega: SSOR relaxation parameter (typically 1.0-1.8)
// @return: 0 on success, -1 if max iterations exceeded
anuga_int _cg_solve_c_ssor(double* data, 
                anuga_int* colind,
                anuga_int* row_ptr,
                double * b,
                double * x,
                anuga_int imax,
                double tol,
                double a_tol,
                anuga_int M,
                double omega)
{
  anuga_int i = 1;
  double alpha, rTr, rTrOld, bt, rTr0;

  double * d = malloc(sizeof(double)*M);
  double * r = malloc(sizeof(double)*M);
  double * q = malloc(sizeof(double)*M);
  double * z = malloc(sizeof(double)*M);

  // r = b - A*x
  cg_zaAxpy(r, -1.0, data, colind, row_ptr, x, b, M);

  // z = M^{-1} * r  (SSOR preconditioner)
  _ssor_apply_c(data, colind, row_ptr, r, z, omega, M);

  // d = z
  cg_dcopy(M, z, d);

  // rTr = r . z
  rTr = cg_ddot(M, r, z);
  rTr0 = rTr;

  while ((i < imax) && (rTr > pow(tol, 2) * rTr0) && (rTr > pow(a_tol, 2))) {

    // q = A * d
    cg_zAx(q, data, colind, row_ptr, d, M);

    // alpha = rTr / (d . q)
    alpha = rTr / cg_ddot(M, d, q);

    // x += alpha * d
    cg_daxpy(M, alpha, d, x);

    // r -= alpha * q
    cg_daxpy(M, -alpha, q, r);

    // z = M^{-1} * r  (SSOR preconditioner)
    _ssor_apply_c(data, colind, row_ptr, r, z, omega, M);

    rTrOld = rTr;

    // rTr = r . z
    rTr = cg_ddot(M, r, z);

    bt = rTr / rTrOld;

    // d = z + bt * d
    cg_dscal(M, bt, d);
    cg_daxpy(M, 1.0, z, d);

    i = i + 1;
  }

  free(d);
  free(r);
  free(q);
  free(z);

  if (i >= imax) {
    return -1;
  }
  else {
    return 0;
  }
}

// Persistent OpenMP conjugate gradient solve Ax = b for x, A given in Sparse CSR format.
// Uses a single parallel region to avoid repeated fork/join overhead.
// @input data: double vector with non-zero entries of A
//        colind: anuga_int vector of column indices of non-zero entries of A
//        row_ptr: anuga_int vector giving index of rows for non-zero entries of A
//        b: double vector specifying right hand side of equation to solve
//        x: double vector with initial guess and to store result
//        imax: maximum number of iterations
//        tol: relative error tolerance for stopping criteria
//        a_tol: absolute error tolerance for stopping criteria
//        M: length of vectors x and b
// @return: 0 on success, -1 if max iterations exceeded
anuga_int _cg_solve_c_persistent(double* data, 
                anuga_int* colind,
                anuga_int* row_ptr,
                double * b,
                double * x,
                anuga_int imax,
                double tol,
                double a_tol,
                anuga_int M)
{
  anuga_int i_count = 1;
  double alpha, rTr, rTrOld, bt, rTr0;
  anuga_int i;
  anuga_int ckey, j;

  double * d = malloc(sizeof(double)*M);
  double * r = malloc(sizeof(double)*M);
  double * q = malloc(sizeof(double)*M);

  #pragma omp parallel private(i, ckey, j)
  {
    // r = b - A*x
    #pragma omp for
    for (i = 0; i < M; i++) {
      r[i] = b[i];
      for (ckey = row_ptr[i]; ckey < row_ptr[i+1]; ckey++) {
        j = colind[ckey];
        r[i] -= data[ckey] * x[j];
      }
    }

    // d = r
    #pragma omp for
    for (i = 0; i < M; i++) {
      d[i] = r[i];
    }

    // rTr = r . r
    double local_rTr = 0.0;
    #pragma omp for reduction(+:local_rTr)
    for (i = 0; i < M; i++) {
      local_rTr += r[i] * r[i];
    }
    #pragma omp single
    {
      rTr = local_rTr;
      rTr0 = rTr;
    }

    while ((i_count < imax) && (rTr > pow(tol, 2) * rTr0) && (rTr > pow(a_tol, 2))) {

      // q = A * d
      #pragma omp for
      for (i = 0; i < M; i++) {
        q[i] = 0.0;
        for (ckey = row_ptr[i]; ckey < row_ptr[i+1]; ckey++) {
          j = colind[ckey];
          q[i] += data[ckey] * d[j];
        }
      }

      // dTq = d . q
      double local_dTq = 0.0;
      #pragma omp for reduction(+:local_dTq)
      for (i = 0; i < M; i++) {
        local_dTq += d[i] * q[i];
      }
      #pragma omp single
      {
        alpha = rTr / local_dTq;
      }

      // x += alpha*d, r -= alpha*q
      #pragma omp for
      for (i = 0; i < M; i++) {
        x[i] += alpha * d[i];
        r[i] -= alpha * q[i];
      }

      // rTr_new = r . r
      double local_rTr_new = 0.0;
      #pragma omp for reduction(+:local_rTr_new)
      for (i = 0; i < M; i++) {
        local_rTr_new += r[i] * r[i];
      }
      #pragma omp single
      {
        rTrOld = rTr;
        rTr = local_rTr_new;
        bt = rTr / rTrOld;
        i_count++;
      }

      // d = r + bt*d
      #pragma omp for
      for (i = 0; i < M; i++) {
        d[i] = r[i] + bt * d[i];
      }
    }
  }

  free(d);
  free(r);
  free(q);

  if (i_count >= imax) {
    return -1;
  }
  else {
    return 0;
  }
}

