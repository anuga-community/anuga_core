#cython: wraparound=False, boundscheck=False, cdivision=True, profile=False, nonecheck=False, initializedcheck=False, language_level=3
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint8_t


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
