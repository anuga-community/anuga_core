#cython: wraparound=False, boundscheck=False, cdivision=True, language_level=3
"""Cython wrapper for ParMETIS_V3_PartKway.

Provides parallel graph partitioning via ParMETIS, callable from Python
using mpi4py communicators and numpy arrays.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t, int64_t

from mpi4py.libmpi cimport MPI_Comm
from mpi4py cimport MPI as mpi4py_MPI


cdef extern from "parmetis.h":
    ctypedef int32_t idx_t
    ctypedef float real_t
    int ParMETIS_V3_PartKway(
        idx_t *vtxdist, idx_t *xadj, idx_t *adjncy,
        idx_t *vwgt, idx_t *adjwgt, idx_t *wgtflag,
        idx_t *numflag, idx_t *ncon, idx_t *nparts,
        real_t *tpwgts, real_t *ubvec, idx_t *options,
        idx_t *edgecut, idx_t *part, MPI_Comm *comm)


def neighbours_to_csr(int64_t[:,::1] neighbours, int start, int end):
    """Convert a block of ANUGA's neighbours array to CSR format.

    Parameters
    ----------
    neighbours : int64 array, shape (N, 3)
        Full mesh neighbour array. neighbours[i, e] is the triangle
        adjacent to triangle i across edge e, or negative for boundary.
    start, end : int
        Local triangle range [start, end).

    Returns
    -------
    xadj : ndarray of int32, shape (n_local + 1,)
        CSR row pointers.
    adjncy : ndarray of int32
        CSR column indices (global triangle IDs).
    """
    cdef int n_local = end - start
    cdef int i, e, count
    cdef int64_t nb

    # Upper bound: 3 neighbours per triangle
    cdef int32_t *xadj_buf = <int32_t *>malloc((n_local + 1) * sizeof(int32_t))
    cdef int32_t *adjncy_buf = <int32_t *>malloc(3 * n_local * sizeof(int32_t))
    if xadj_buf == NULL or adjncy_buf == NULL:
        if xadj_buf != NULL:
            free(xadj_buf)
        if adjncy_buf != NULL:
            free(adjncy_buf)
        raise MemoryError("Failed to allocate CSR buffers")

    cdef int nnz = 0
    xadj_buf[0] = 0

    for i in range(n_local):
        count = 0
        for e in range(3):
            nb = neighbours[start + i, e]
            if nb >= 0:
                adjncy_buf[nnz] = <int32_t>nb
                nnz += 1
                count += 1
        xadj_buf[i + 1] = xadj_buf[i] + count

    # Copy to numpy arrays
    xadj_np = np.empty(n_local + 1, dtype=np.int32)
    adjncy_np = np.empty(nnz, dtype=np.int32)

    cdef int32_t[::1] xadj_view = xadj_np
    cdef int32_t[::1] adjncy_view = adjncy_np

    cdef int j
    for j in range(n_local + 1):
        xadj_view[j] = xadj_buf[j]
    for j in range(nnz):
        adjncy_view[j] = adjncy_buf[j]

    free(xadj_buf)
    free(adjncy_buf)

    return xadj_np, adjncy_np


def parmetis_part_kway(int32_t[::1] vtxdist,
                       int32_t[::1] xadj,
                       int32_t[::1] adjncy,
                       int nparts,
                       mpi4py_MPI.Comm comm):
    """Call ParMETIS_V3_PartKway collectively.

    Parameters
    ----------
    vtxdist : int32 array, shape (npes + 1,)
        Global vertex distribution across processors.
    xadj : int32 array, shape (n_local + 1,)
        CSR row pointers for local adjacency.
    adjncy : int32 array
        CSR column indices (global vertex IDs).
    nparts : int
        Number of partitions desired.
    comm : mpi4py.MPI.Comm
        MPI communicator (all ranks must call collectively).

    Returns
    -------
    edgecut : int
        Number of edge cuts in the partition.
    part : ndarray of int32, shape (n_local,)
        Partition assignment for each local vertex.
    """
    cdef int rank = comm.Get_rank()
    cdef int n_local = vtxdist[rank + 1] - vtxdist[rank]

    # Output array
    part_np = np.empty(n_local, dtype=np.int32)
    cdef int32_t[::1] part_view = part_np

    # ParMETIS parameters
    cdef idx_t wgtflag = 0      # no weights
    cdef idx_t numflag = 0      # C-style numbering
    cdef idx_t ncon = 1         # single constraint
    cdef idx_t nparts_c = <idx_t>nparts
    cdef idx_t edgecut = 0
    cdef idx_t options[3]
    options[0] = 0              # use defaults

    # Uniform target partition weights
    cdef real_t *tpwgts = <real_t *>malloc(nparts * sizeof(real_t))
    if tpwgts == NULL:
        raise MemoryError("Failed to allocate tpwgts")
    cdef int i
    for i in range(nparts):
        tpwgts[i] = 1.0 / <real_t>nparts

    # Imbalance tolerance
    cdef real_t ubvec = 1.05

    # Get C-level MPI_Comm handle
    cdef MPI_Comm c_comm = comm.ob_mpi

    cdef int ret
    ret = ParMETIS_V3_PartKway(
        &vtxdist[0], &xadj[0], &adjncy[0],
        NULL, NULL,             # vwgt, adjwgt
        &wgtflag, &numflag, &ncon, &nparts_c,
        tpwgts, &ubvec, options,
        &edgecut, &part_view[0], &c_comm)

    free(tpwgts)

    if ret != 1:
        raise RuntimeError(f"ParMETIS_V3_PartKway failed with return code {ret}")

    return int(edgecut), part_np
