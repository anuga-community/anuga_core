#cython: wraparound=False, boundscheck=False, cdivision=True, profile=False, nonecheck=False, initializedcheck=False, language_level=3
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint8_t
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy as c_memcpy
from cython.parallel cimport prange, parallel


def ghost_layer_bfs(int64_t[:,::1] neighbours not None,
                    int64_t tlower,
                    int64_t tupper,
                    int layer_width):
    """BFS ghost-layer expansion replacing the setdiff1d/unique/extract chain.

    Uses a uint8 state array (ntriangles bytes) for O(1) membership tests.
    No per-layer sort or set-difference is needed: each triangle is visited
    at most once.  Between layers the frontier is collected via np.where on
    the state array, which is a single fast C pass over ntriangles bytes.

    Parameters
    ----------
    neighbours : int64 array, shape (ntriangles, 3), C-contiguous
        Global neighbour array from Mesh.neighbours.  Boundary / ghost
        sentinel values are negative and are skipped.
    tlower, tupper : int64
        Half-open range [tlower, tupper) of triangle global IDs owned as
        full triangles by this processor.
    layer_width : int
        Number of ghost layers to expand (default in ANUGA is 2).

    Returns
    -------
    ghost_trianglemap : int64 ndarray, sorted
        Global triangle IDs of every ghost triangle for this processor.
    """
    cdef int64_t ntriangles = neighbours.shape[0]
    cdef int64_t t, nb, j
    cdef int64_t frontier_size
    cdef int edge, i

    # State array: 0 = unvisited, 1 = full, 2 = layer-0 ghost,
    #              3 = layer-1 ghost, ...
    # Only ntriangles *bytes* — 50 MB for a 50 M-triangle mesh.
    state_np = np.zeros(ntriangles, dtype=np.uint8)
    cdef uint8_t[::1] state = state_np

    with nogil:
        # Mark full triangles
        for t in range(tlower, tupper):
            state[t] = 1

        # Layer 0: immediate neighbours of the full region
        for t in range(tlower, tupper):
            for edge in range(3):
                nb = neighbours[t, edge]
                if nb >= 0 and state[nb] == 0:
                    state[nb] = 2

    # Collect the layer-0 frontier with np.where (fast C scan, result sorted).
    # For layer_width == 1 we skip this block entirely.
    cdef int64_t[::1] frontier
    for i in range(1, layer_width):
        frontier = np.where(state_np == <uint8_t>(i + 1))[0].astype(np.int64)
        frontier_size = frontier.shape[0]
        with nogil:
            for j in range(frontier_size):
                t = frontier[j]
                for edge in range(3):
                    nb = neighbours[t, edge]
                    if nb >= 0 and state[nb] == 0:
                        state[nb] = <uint8_t>(i + 2)

    # All ghost triangles have state > 1.  np.where returns sorted indices.
    return np.where(state_np > 1)[0].astype(np.int64)


def ghost_bnd_layer_classify(int64_t[:,::1] neighbours not None,
                              int64_t[::1] ghost_ids not None,
                              int64_t tlower,
                              int64_t tupper):
    """Classify ghost-triangle edges as ghost boundaries.

    For each ghost triangle, checks its three edges.  An edge is a ghost
    boundary if the neighbour on that edge is neither a full triangle
    (in [tlower, tupper)) nor another ghost triangle.  Negative neighbours
    (mesh boundary sentinels) also count as ghost boundaries.

    Uses a uint8 `in_proc` membership array for O(1) lookup instead of
    the O(n) isin() calls in the pure-Python version.

    Parameters
    ----------
    neighbours : int64 array, shape (ntriangles, 3), C-contiguous
    ghost_ids  : int64 array, shape (G,), sorted ghost triangle global IDs
    tlower, tupper : int64 — half-open range of full triangles [tlower, tupper)

    Returns
    -------
    tri_ids  : int64 ndarray  — global triangle ID for each ghost boundary edge
    edge_ids : int64 ndarray  — edge index (0, 1, or 2) for each entry
    """
    cdef int64_t ntriangles = neighbours.shape[0]
    cdef int64_t G = ghost_ids.shape[0]
    cdef int64_t g, t, nb, count
    cdef int edge

    # Mark all triangles in this processor (full + ghost) as in_proc.
    in_proc_np = np.zeros(ntriangles, dtype=np.uint8)
    cdef uint8_t[::1] in_proc = in_proc_np

    with nogil:
        for t in range(tlower, tupper):
            in_proc[t] = 1
        for g in range(G):
            in_proc[ghost_ids[g]] = 1

    # Upper bound: at most 3*G ghost boundary edges.
    tri_buf_np  = np.empty(3 * G, dtype=np.int64)
    edge_buf_np = np.empty(3 * G, dtype=np.int64)
    cdef int64_t[::1] tri_buf  = tri_buf_np
    cdef int64_t[::1] edge_buf = edge_buf_np

    count = 0
    with nogil:
        for g in range(G):
            t = ghost_ids[g]
            for edge in range(3):
                nb = neighbours[t, edge]
                # Ghost boundary: neighbour is outside the processor set,
                # including mesh-boundary sentinels (nb < 0).
                if nb < 0 or in_proc[nb] == 0:
                    tri_buf[count]  = t
                    edge_buf[count] = edge
                    count += 1

    return tri_buf_np[:count].copy(), edge_buf_np[:count].copy()


def ghost_layer_bfs_all(int64_t[:,::1] neighbours not None,
                         int64_t[::1] tlower_arr not None,
                         int64_t[::1] tupper_arr not None,
                         int layer_width):
    """BFS ghost-layer expansion for all P processors in parallel (OpenMP).

    Each OpenMP thread runs one processor's BFS independently, using a
    heap-allocated uint8 state array (calloc'd to zero).  Peak extra memory
    equals ``num_threads × ntriangles`` bytes — e.g. 8 threads × 50 MB for a
    50 M-triangle mesh.  When OpenMP is unavailable the loop degrades to serial.

    Parameters
    ----------
    neighbours : int64 array, shape (ntriangles, 3), C-contiguous
    tlower_arr, tupper_arr : int64 arrays, shape (P,)
        Half-open ranges [tlower, tupper) of full triangle IDs per processor.
    layer_width : int
        Number of ghost layers (default in ANUGA is 2).

    Returns
    -------
    list of P int64 ndarrays
        Each element contains the sorted global triangle IDs of every ghost
        triangle for that processor.
    """
    cdef int P = tlower_arr.shape[0]
    cdef int64_t ntriangles = neighbours.shape[0]
    cdef int p, i, edge
    cdef int64_t t, nb, tlo, tup
    cdef uint8_t* state_p   # thread-private inside prange

    # One state-array pointer per processor; each is malloc'd inside the prange
    # (at most num_threads are live simultaneously) and freed after extraction.
    cdef uint8_t** states = <uint8_t**>malloc(P * sizeof(uint8_t*))
    if states == NULL:
        raise MemoryError("ghost_layer_bfs_all: failed to allocate state pointer array")

    with nogil, parallel():
        for p in prange(P, schedule='dynamic'):
            tlo = tlower_arr[p]
            tup = tupper_arr[p]

            # calloc zeros the bytes — avoids an explicit memset.
            states[p] = <uint8_t*>calloc(ntriangles, sizeof(uint8_t))

            # Mark full triangles (state = 1)
            for t in range(tlo, tup):
                states[p][t] = 1

            # Layer 0: mark immediate neighbours outside the full region (state = 2)
            for t in range(tlo, tup):
                for edge in range(3):
                    nb = neighbours[t, edge]
                    if nb >= 0 and states[p][nb] == 0:
                        states[p][nb] = 2

            # Additional layers: scan the full state array for the current frontier.
            # O(layer_width × ntriangles) per processor — typically layer_width = 2.
            for i in range(1, layer_width):
                for t in range(ntriangles):
                    if states[p][t] == i + 1:
                        for edge in range(3):
                            nb = neighbours[t, edge]
                            if nb >= 0 and states[p][nb] == 0:
                                states[p][nb] = i + 2

    # Extract ghost IDs from each state array using numpy's fast C scan.
    # np.asarray wraps the malloc'd memory without copying; np.where does
    # a single O(ntriangles) C pass to collect indices where state > 1.
    results = []
    cdef uint8_t[::1] state_view
    for p in range(P):
        state_view = <uint8_t[:ntriangles:1]>states[p]
        results.append(np.where(np.asarray(state_view) > 1)[0].astype(np.int64))
        free(states[p])
    free(states)

    return results


def ghost_bnd_layer_classify_all(int64_t[:,::1] neighbours not None,
                                  ghost_id_list,
                                  int64_t[::1] tlower_arr not None,
                                  int64_t[::1] tupper_arr not None):
    """Classify ghost boundary edges for all processors in parallel (OpenMP).

    For each processor p, every edge of every ghost triangle is examined.  An
    edge is a ghost boundary if its neighbour is outside the processor set
    (neither full nor ghost) or is a mesh-boundary sentinel (nb < 0).

    Each thread uses a heap-allocated ``uint8 in_proc[ntriangles]`` membership
    array and writes results to a malloc'd output buffer (at most 3×G entries).

    Parameters
    ----------
    neighbours : int64 array, shape (ntriangles, 3), C-contiguous
    ghost_id_list : list of P int64 ndarrays (from ghost_layer_bfs_all)
    tlower_arr, tupper_arr : int64 arrays, shape (P,)

    Returns
    -------
    list of P (tri_ids, edge_ids) tuples of int64 ndarrays.
    """
    cdef int P = tlower_arr.shape[0]
    cdef int64_t ntriangles = neighbours.shape[0]
    cdef int p, edge
    cdef int64_t t, nb, g, G, tlo, tup
    cdef uint8_t* in_proc_p   # thread-private inside prange

    # Pre-extract raw C pointers to the ghost-ID arrays so they are
    # accessible inside nogil.  Arrays must remain alive for the prange duration.
    cdef int64_t** ghost_ptrs = <int64_t**>malloc(P * sizeof(int64_t*))
    cdef int64_t*  ghost_lens = <int64_t*>malloc(P * sizeof(int64_t))
    cdef int64_t[::1] garr
    for p in range(P):
        ghost_lens[p] = len(ghost_id_list[p])
        if ghost_lens[p] > 0:
            garr = ghost_id_list[p]
            ghost_ptrs[p] = &garr[0]
        else:
            ghost_ptrs[p] = NULL

    # Per-processor output buffers (malloc'd inside prange, freed after copy).
    cdef int64_t** tri_bufs  = <int64_t**>malloc(P * sizeof(int64_t*))
    cdef int64_t** edge_bufs = <int64_t**>malloc(P * sizeof(int64_t*))
    cdef int64_t*  counts    = <int64_t*>malloc(P * sizeof(int64_t))

    with nogil, parallel():
        for p in prange(P, schedule='dynamic'):
            tlo = tlower_arr[p]
            tup = tupper_arr[p]
            G   = ghost_lens[p]

            # Mark full and ghost triangles as in-processor.
            in_proc_p = <uint8_t*>calloc(ntriangles, sizeof(uint8_t))
            for t in range(tlo, tup):
                in_proc_p[t] = 1
            for g in range(G):
                in_proc_p[ghost_ptrs[p][g]] = 1

            # Allocate output (upper bound: 3 edges per ghost triangle).
            if G > 0:
                tri_bufs[p]  = <int64_t*>malloc(3 * G * sizeof(int64_t))
                edge_bufs[p] = <int64_t*>malloc(3 * G * sizeof(int64_t))
            else:
                tri_bufs[p]  = NULL
                edge_bufs[p] = NULL
            counts[p] = 0

            for g in range(G):
                t = ghost_ptrs[p][g]
                for edge in range(3):
                    nb = neighbours[t, edge]
                    if nb < 0 or in_proc_p[nb] == 0:
                        tri_bufs[p][counts[p]]  = t
                        edge_bufs[p][counts[p]] = edge
                        counts[p] += 1

            free(in_proc_p)

    # Copy results into numpy arrays (requires GIL) then free C buffers.
    results = []
    cdef int64_t cnt
    cdef int64_t[::1] tri_view, edge_view
    for p in range(P):
        cnt = counts[p]
        tri_arr  = np.empty(cnt, dtype=np.int64)
        edge_arr = np.empty(cnt, dtype=np.int64)
        if cnt > 0:
            tri_view  = tri_arr
            edge_view = edge_arr
            c_memcpy(&tri_view[0],  tri_bufs[p],  cnt * sizeof(int64_t))
            c_memcpy(&edge_view[0], edge_bufs[p], cnt * sizeof(int64_t))
        if tri_bufs[p] != NULL:
            free(tri_bufs[p])
            free(edge_bufs[p])
        results.append((tri_arr, edge_arr))

    free(ghost_ptrs)
    free(ghost_lens)
    free(tri_bufs)
    free(edge_bufs)
    free(counts)

    return results
