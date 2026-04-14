#cython: wraparound=False, boundscheck=False, cdivision=True, profile=False, nonecheck=False, overflowcheck=False, cdivision_warnings=False, unraisable_tracebacks=False
import cython
from libc.stdint cimport int64_t

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern from "cg.c":
    int64_t _jacobi_precon_c(double* data, int64_t* colind, int64_t* row_ptr, double* precon, int64_t M)
    int64_t _cg_solve_c(double* data, int64_t* colind, int64_t* row_ptr, double* b, double* x, int64_t imax, double tol, double a_tol, int64_t M)
    int64_t _cg_solve_c_precon(double* data, int64_t* colind, int64_t* row_ptr, double* b, double* x, int64_t imax, double tol, double a_tol, int64_t M, double* precon)
    int64_t _cg_solve_c_ssor(double* data, int64_t* colind, int64_t* row_ptr, double* b, double* x, int64_t imax, double tol, double a_tol, int64_t M, double omega)
    int64_t _cg_solve_c_persistent(double* data, int64_t* colind, int64_t* row_ptr, double* b, double* x, int64_t imax, double tol, double a_tol, int64_t M)

def jacobi_precon_c(object csr_sparse, np.ndarray[double, ndim=1, mode="c"] precon not None):

    cdef int64_t M, err
    cdef np.ndarray[double, ndim=1, mode="c"] data
    cdef np.ndarray[int64_t, ndim=1, mode="c"] colind
    cdef np.ndarray[int64_t, ndim=1, mode="c"] row_ptr

    data = csr_sparse.data
    colind = csr_sparse.colind
    row_ptr = csr_sparse.row_ptr

    M = row_ptr.shape[0] - 1

    err = _jacobi_precon_c(&data[0], &colind[0], &row_ptr[0], &precon[0], M)

def cg_solve_c(object csr_sparse,\
                np.ndarray[double, ndim=1, mode="c"] x0 not None,\
                np.ndarray[double, ndim=1, mode="c"] b not None,\
                int64_t imax,\
                double tol,\
                double a_tol,\
                int64_t bcols):

    cdef int64_t M, err
    cdef np.ndarray[double, ndim=1, mode="c"] data
    cdef np.ndarray[int64_t, ndim=1, mode="c"] colind
    cdef np.ndarray[int64_t, ndim=1, mode="c"] row_ptr

    data = csr_sparse.data
    colind = csr_sparse.colind
    row_ptr = csr_sparse.row_ptr

    M = row_ptr.shape[0] - 1

    err = _cg_solve_c(&data[0],\
                    &colind[0],\
                    &row_ptr[0],\
                    &b[0],\
                    &x0[0],\
                    imax,\
                    tol,\
                    a_tol,\
                    M)

    return err

def cg_solve_c_precon(object csr_sparse,\
                    np.ndarray[double, ndim=1, mode="c"] x0 not None,\
                    np.ndarray[double, ndim=1, mode="c"] b not None,\
                    int64_t imax,\
                    double tol,\
                    double a_tol,\
                    int64_t bcols,\
                    np.ndarray[double, ndim=1, mode="c"] precon not None):

    cdef int64_t M, err
    cdef np.ndarray[double, ndim=1, mode="c"] data
    cdef np.ndarray[int64_t, ndim=1, mode="c"] colind
    cdef np.ndarray[int64_t, ndim=1, mode="c"] row_ptr

    data = csr_sparse.data
    colind = csr_sparse.colind
    row_ptr = csr_sparse.row_ptr

    M = row_ptr.shape[0] - 1

    err = _cg_solve_c_precon(&data[0],\
                            &colind[0],\
                            &row_ptr[0],\
                            &b[0],\
                            &x0[0],\
                            imax,\
                            tol,\
                            a_tol,\
                            M,\
                            &precon[0])

    return err


def cg_solve_c_ssor(object csr_sparse,\
                    np.ndarray[double, ndim=1, mode="c"] x0 not None,\
                    np.ndarray[double, ndim=1, mode="c"] b not None,\
                    int64_t imax,\
                    double tol,\
                    double a_tol,\
                    int64_t bcols,\
                    double omega=1.0):
    """Conjugate gradient solve with SSOR preconditioning.

    Parameters
    ----------
    csr_sparse : Sparse_CSR
        System matrix in CSR format.
    x0 : ndarray
        Initial guess; modified in-place with the solution.
    b : ndarray
        Right-hand side vector.
    imax : int
        Maximum number of CG iterations.
    tol : float
        Relative residual tolerance.
    a_tol : float
        Absolute residual tolerance.
    bcols : int
        Number of right-hand side columns (unused, kept for API consistency).
    omega : float, optional
        SSOR relaxation parameter (default 1.0; optimal values ~1.2-1.6).

    Returns
    -------
    int
        0 on convergence, -1 if maximum iterations reached.
    """
    cdef int64_t M, err
    cdef np.ndarray[double, ndim=1, mode="c"] data
    cdef np.ndarray[int64_t, ndim=1, mode="c"] colind
    cdef np.ndarray[int64_t, ndim=1, mode="c"] row_ptr

    data = csr_sparse.data
    colind = csr_sparse.colind
    row_ptr = csr_sparse.row_ptr

    M = row_ptr.shape[0] - 1

    err = _cg_solve_c_ssor(&data[0],\
                           &colind[0],\
                           &row_ptr[0],\
                           &b[0],\
                           &x0[0],\
                           imax,\
                           tol,\
                           a_tol,\
                           M,\
                           omega)

    return err


def cg_solve_c_persistent(object csr_sparse,\
                          np.ndarray[double, ndim=1, mode="c"] x0 not None,\
                          np.ndarray[double, ndim=1, mode="c"] b not None,\
                          int64_t imax,\
                          double tol,\
                          double a_tol,\
                          int64_t bcols):
    """Conjugate gradient solve with persistent OpenMP thread team.

    Uses a single parallel region to avoid repeated fork/join overhead, which
    reduces overhead for small-to-medium problems (N < 100K).

    Parameters
    ----------
    csr_sparse : Sparse_CSR
        System matrix in CSR format.
    x0 : ndarray
        Initial guess; modified in-place with the solution.
    b : ndarray
        Right-hand side vector.
    imax : int
        Maximum number of CG iterations.
    tol : float
        Relative residual tolerance.
    a_tol : float
        Absolute residual tolerance.
    bcols : int
        Number of right-hand side columns (unused, kept for API consistency).

    Returns
    -------
    int
        0 on convergence, -1 if maximum iterations reached.
    """
    cdef int64_t M, err
    cdef np.ndarray[double, ndim=1, mode="c"] data
    cdef np.ndarray[int64_t, ndim=1, mode="c"] colind
    cdef np.ndarray[int64_t, ndim=1, mode="c"] row_ptr

    data = csr_sparse.data
    colind = csr_sparse.colind
    row_ptr = csr_sparse.row_ptr

    M = row_ptr.shape[0] - 1

    err = _cg_solve_c_persistent(&data[0],\
                                 &colind[0],\
                                 &row_ptr[0],\
                                 &b[0],\
                                 &x0[0],\
                                 imax,\
                                 tol,\
                                 a_tol,\
                                 M)

    return err

    return err
