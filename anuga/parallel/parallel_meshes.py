"""parallel_meshes — structured rectangular meshes for parallel finite-volume
computations of the shallow water equations.

Each function returns the local sub-mesh for the calling MPI rank together with
the dictionaries that describe the ghost-layer communication pattern required
by :class:`~anuga.parallel.parallel_shallow_water.Parallel_domain`.

Strip decomposition
-------------------
The global domain is divided into vertical strips (columns of cells).  Rank *r*
owns exactly ``mm`` full columns (``_column_distribution`` distributes any
remainder round-robin).  Left and right ghost columns (one each) are appended
from the neighbouring ranks so that every rank can compute flux updates on its
full columns without extra communication during the time-step loop.

Ole Nielsen, Stephen Roberts, Duncan Gray, Christopher Zoppou
Geoscience Australia, 2005

Modified by Linda Stals, March 2006 (ghost boundaries)
Rewritten 2026 (correct strip decomposition, cross mesh, domain wrappers)
"""

import numpy as num

from .parallel_api import myid, numprocs, pypar_available


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _column_distribution(m_g, nprocs, rank):
    """Return *(i_start, mm)* — the global start column and the number of
    full columns owned by *rank*.

    Columns are distributed as evenly as possible; the first ``m_g %
    nprocs`` ranks each get one extra column.
    """
    col_per_rank = m_g // nprocs
    remainder    = m_g % nprocs
    i_start = rank * col_per_rank + min(rank, remainder)
    mm      = col_per_rank + (1 if rank < remainder else 0)
    return i_start, mm


def _reorder_full_first(elements, boundary, neighbours, neighbour_edges,
                        full_send_dict, ghost_recv_dict,
                        tris_per_col, m_local, left_offset, mm,
                        has_left_ghost, has_right_ghost):
    """Reorder triangles so full columns come first, ghost columns last.

    The parallel domain requires ``elements[:number_of_full_triangles]`` to
    contain only full (non-ghost) triangles.  The serial mesh builders produce
    columns in geometric order (0, 1, …, m_local-1), putting the ghost-left
    column first.  This helper reorders to:

        [full cols: left_offset … left_offset+mm-1]
        [ghost left col: 0        (if has_left_ghost) ]
        [ghost right col: m_local-1 (if has_right_ghost)]

    Parameters
    ----------
    elements, boundary, neighbours, neighbour_edges : mesh arrays/dict
    full_send_dict, ghost_recv_dict : communication dicts (indices remapped)
    tris_per_col : int — triangles per column (2*n_g or 4*n_g)
    m_local, left_offset, mm, has_left_ghost, has_right_ghost : geometry

    Returns
    -------
    Same types as inputs, with triangle indices remapped consistently.
    """
    Nt = m_local * tris_per_col

    # Build permutation: new_idx -> old_idx
    # Full columns come first
    full_cols  = list(range(left_offset, left_offset + mm))
    ghost_cols = []
    if has_left_ghost:
        ghost_cols.append(0)
    if has_right_ghost:
        ghost_cols.append(m_local - 1)

    perm = num.concatenate(
        [num.arange(c * tris_per_col, (c + 1) * tris_per_col)
         for c in full_cols + ghost_cols]
    ).astype(int)

    # Inverse permutation: old_idx -> new_idx
    inv_perm = num.empty(Nt, dtype=int)
    inv_perm[perm] = num.arange(Nt)

    # Reorder elements and neighbour_edges (per-triangle, no index remapping)
    new_elements       = elements[perm]
    new_neighbour_edges = neighbour_edges[perm]

    # Reorder neighbours (per-triangle) and remap the stored triangle indices
    new_neighbours = neighbours[perm].copy()
    mask = new_neighbours >= 0
    new_neighbours[mask] = inv_perm[new_neighbours[mask]]

    # Remap boundary dict keys
    new_boundary = {(inv_perm[tri], edge): tag
                    for (tri, edge), tag in boundary.items()}

    # Remap communication dict index arrays
    def _remap_dict(d):
        return {rank: [inv_perm[arr], inv_perm[arr]]
                for rank, (arr, _) in d.items()}

    new_full_send_dict  = _remap_dict(full_send_dict)
    new_ghost_recv_dict = _remap_dict(ghost_recv_dict)

    return (new_elements, new_boundary, new_neighbours, new_neighbour_edges,
            new_full_send_dict, new_ghost_recv_dict)


def _local_geometry(m_g, n_g, len1_g, len2_g, origin_g):
    """Return ``(i_start, mm, left_offset, m_local, delta1, delta2, ox, oy)``.

    Computes the strip-decomposition geometry for the calling rank.
    """
    i_start, mm = _column_distribution(m_g, numprocs, myid)

    has_left_ghost  = (myid > 0)
    has_right_ghost = (myid < numprocs - 1)
    left_offset  = 1 if has_left_ghost  else 0
    right_offset = 1 if has_right_ghost else 0
    m_local = mm + left_offset + right_offset

    delta1 = len1_g / m_g
    delta2 = len2_g / n_g

    # x-origin of the local mesh (leftmost node column, may be ghost)
    ox = origin_g[0] + (i_start - left_offset) * delta1
    oy = origin_g[1]

    return i_start, mm, left_offset, m_local, delta1, delta2, ox, oy


# ---------------------------------------------------------------------------
# parallel_rectangular_with_neighbours
# ---------------------------------------------------------------------------

def parallel_rectangular_with_neighbours(m_g, n_g, len1_g=1.0, len2_g=1.0,
                                          origin_g=(0.0, 0.0)):
    """Build the local portion of a rectangular (2-triangle/cell) mesh,
    including pre-computed neighbour connectivity.

    This is the primary rectangular mesh builder for parallel use.
    :func:`parallel_rectangle` is a thin wrapper that discards the neighbour
    arrays for callers that do not need them.

    Parameters
    ----------
    m_g : int
        Total number of cells in the x-direction (global).
    n_g : int
        Total number of cells in the y-direction (all ranks share every row).
    len1_g : float
        Total domain length in x (default 1.0).
    len2_g : float
        Total domain length in y (default 1.0).
    origin_g : tuple (float, float)
        Lower-left corner of the global domain (default (0, 0)).

    Returns
    -------
    points : ndarray, shape (Np, 2)
        Node coordinates for this rank's local mesh.
    elements : ndarray, shape (Nt, 3)
        Triangle vertex indices (into *points*).
    boundary : dict
        ``(triangle_index, edge_index) -> tag``.  Tags are ``'left'``,
        ``'right'``, ``'top'``, ``'bottom'``, or ``'ghost'``.
    neighbours : ndarray, shape (Nt, 3)
        Neighbour triangle index for each edge; -1 for boundary/ghost edges.
    neighbour_edges : ndarray, shape (Nt, 3)
        Reciprocal edge index in the neighbouring triangle; -1 for boundaries.
    full_send_dict : dict
        ``{neighbour_rank: [tri_indices, tri_indices]}``
    ghost_recv_dict : dict
        ``{neighbour_rank: [tri_indices, tri_indices]}``

    Notes
    -----
    The local mesh is constructed by calling :func:`rectangular_with_neighbours`
    on the strip ``m_local × n_g`` and then relabelling the boundary tags of
    ghost columns as ``'ghost'``.

    Triangle indexing: ``lower(i,j) = 2*(i*n_g + j)``,
    ``upper(i,j) = 2*(i*n_g + j) + 1``.
    """

    from anuga.abstract_2d_finite_volumes.mesh_factory import \
        rectangular_with_neighbours as _rect_wn

    i_start, mm, left_offset, m_local, delta1, delta2, ox, oy = \
        _local_geometry(m_g, n_g, len1_g, len2_g, origin_g)

    has_left_ghost  = (myid > 0)
    has_right_ghost = (myid < numprocs - 1)

    # Build local serial mesh — inherits all neighbour connectivity
    points, elements, boundary, neighbours, neighbour_edges = _rect_wn(
        m_local, n_g,
        len1=m_local * delta1,
        len2=len2_g,
        origin=(ox, oy),
    )

    # Retag ghost column boundaries
    # Rectangle boundary format (from rectangular_construct Cython):
    #   left  boundary: (2*j + 1,                  2) = 'left'   for j in range(n_g)
    #   right boundary: (2*((m-1)*n_g + j),         2) = 'right'  for j in range(n_g)
    for j in range(n_g):
        if has_left_ghost:
            boundary[(2 * j + 1, 2)] = 'ghost'
        if has_right_ghost:
            boundary[(2 * ((m_local - 1) * n_g + j), 2)] = 'ghost'

    # Ghost/full triangle index ranges (contiguous per column, pre-reorder)
    full_send_dict  = {}
    ghost_recv_dict = {}

    if has_left_ghost:
        arr_fl = num.arange(2 * left_offset * n_g,
                            2 * (left_offset + 1) * n_g, dtype=int)
        arr_gl = num.arange(0, 2 * n_g, dtype=int)
        full_send_dict [myid - 1] = [arr_fl, arr_fl]
        ghost_recv_dict[myid - 1] = [arr_gl, arr_gl]

    if has_right_ghost:
        fr_start = 2 * (left_offset + mm - 1) * n_g
        gr_start = 2 * (m_local - 1)          * n_g
        arr_fr = num.arange(fr_start, fr_start + 2 * n_g, dtype=int)
        arr_gr = num.arange(gr_start, gr_start + 2 * n_g, dtype=int)
        full_send_dict [myid + 1] = [arr_fr, arr_fr]
        ghost_recv_dict[myid + 1] = [arr_gr, arr_gr]

    # Reorder so full triangles come first (required by Parallel_domain)
    if has_left_ghost or has_right_ghost:
        elements, boundary, neighbours, neighbour_edges, \
            full_send_dict, ghost_recv_dict = \
            _reorder_full_first(elements, boundary, neighbours, neighbour_edges,
                                full_send_dict, ghost_recv_dict,
                                2 * n_g, m_local, left_offset, mm,
                                has_left_ghost, has_right_ghost)

    return points, elements, boundary, neighbours, neighbour_edges, \
           full_send_dict, ghost_recv_dict


# ---------------------------------------------------------------------------
# parallel_rectangle  (thin wrapper — backward compatible)
# ---------------------------------------------------------------------------

def parallel_rectangle(m_g, n_g, len1_g=1.0, len2_g=1.0,
                       origin_g=(0.0, 0.0)):
    """Build the local portion of a rectangular (2-triangle/cell) mesh.

    Returns the same data as :func:`parallel_rectangular_with_neighbours` but
    without the neighbour arrays.  Kept for backward compatibility.

    Returns
    -------
    points, elements, boundary, full_send_dict, ghost_recv_dict
    """
    points, elements, boundary, _neighbours, _neighbour_edges, \
        full_send_dict, ghost_recv_dict = \
        parallel_rectangular_with_neighbours(m_g, n_g, len1_g, len2_g,
                                             origin_g)
    return points, elements, boundary, full_send_dict, ghost_recv_dict


# ---------------------------------------------------------------------------
# parallel_rectangular_cross_with_neighbours
# ---------------------------------------------------------------------------

def parallel_rectangular_cross_with_neighbours(m_g, n_g, len1_g=1.0,
                                               len2_g=1.0,
                                               origin_g=(0.0, 0.0)):
    """Build the local portion of a rectangular cross (4-triangle/cell) mesh,
    including pre-computed neighbour connectivity.

    Each rectangular cell is divided into four triangles using a centre point
    (cross pattern), matching the serial
    :func:`~anuga.rectangular_cross_with_neighbours` mesh factory.

    Parameters
    ----------
    m_g : int
        Total number of cells in the x-direction (global).
    n_g : int
        Total number of cells in the y-direction.
    len1_g : float
        Total domain length in x (default 1.0).
    len2_g : float
        Total domain length in y (default 1.0).
    origin_g : tuple (float, float)
        Lower-left corner of the global domain (default (0, 0)).

    Returns
    -------
    points : ndarray, shape (Np, 2)
    elements : ndarray, shape (Nt, 3)
    boundary : dict
    neighbours : ndarray, shape (Nt, 3)
    neighbour_edges : ndarray, shape (Nt, 3)
    full_send_dict : dict
    ghost_recv_dict : dict

    Notes
    -----
    Triangle indexing within cell *(i, j)*:
    ``T(i, j, k) = 4*(i*n_g + j) + k``, where *k* = 0 (left), 1 (bottom),
    2 (right), 3 (top).

    Boundary edges (edge 1 of outer triangle):

    * Left  (i = 0):           ``(T(0, j, 0), 1)``
    * Right (i = m_local - 1): ``(T(m_local-1, j, 2), 1)``
    * Bottom (j = 0):          ``(T(i, 0, 1), 1)``
    * Top   (j = n_g - 1):     ``(T(i, n_g-1, 3), 1)``
    """

    from anuga.abstract_2d_finite_volumes.mesh_factory import \
        rectangular_cross_with_neighbours as _cross_wn

    i_start, mm, left_offset, m_local, delta1, delta2, ox, oy = \
        _local_geometry(m_g, n_g, len1_g, len2_g, origin_g)

    has_left_ghost  = (myid > 0)
    has_right_ghost = (myid < numprocs - 1)

    # Build local serial cross mesh — inherits all neighbour connectivity
    points, elements, boundary, neighbours, neighbour_edges = _cross_wn(
        m_local, n_g,
        len1=m_local * delta1,
        len2=len2_g,
        origin=(ox, oy),
    )

    # Retag ghost column boundaries
    # Cross boundary format (from rectangular_cross_construct Cython):
    #   left  boundary: (4*j + 0,                       1) = 'left'   j in range(n_g)
    #   right boundary: (4*((m-1)*n_g + j) + 2,         1) = 'right'  j in range(n_g)
    for j in range(n_g):
        if has_left_ghost:
            boundary[(4 * j, 1)] = 'ghost'
        if has_right_ghost:
            boundary[(4 * ((m_local - 1) * n_g + j) + 2, 1)] = 'ghost'

    # Ghost/full triangle index ranges (contiguous per column, pre-reorder)
    full_send_dict  = {}
    ghost_recv_dict = {}

    if has_left_ghost:
        arr_fl = num.arange(4 * left_offset * n_g,
                            4 * (left_offset + 1) * n_g, dtype=int)
        arr_gl = num.arange(0, 4 * n_g, dtype=int)
        full_send_dict [myid - 1] = [arr_fl, arr_fl]
        ghost_recv_dict[myid - 1] = [arr_gl, arr_gl]

    if has_right_ghost:
        fr_start = 4 * (left_offset + mm - 1) * n_g
        gr_start = 4 * (m_local - 1)          * n_g
        arr_fr = num.arange(fr_start, fr_start + 4 * n_g, dtype=int)
        arr_gr = num.arange(gr_start, gr_start + 4 * n_g, dtype=int)
        full_send_dict [myid + 1] = [arr_fr, arr_fr]
        ghost_recv_dict[myid + 1] = [arr_gr, arr_gr]

    # Reorder so full triangles come first (required by Parallel_domain)
    if has_left_ghost or has_right_ghost:
        elements, boundary, neighbours, neighbour_edges, \
            full_send_dict, ghost_recv_dict = \
            _reorder_full_first(elements, boundary, neighbours, neighbour_edges,
                                full_send_dict, ghost_recv_dict,
                                4 * n_g, m_local, left_offset, mm,
                                has_left_ghost, has_right_ghost)

    return points, elements, boundary, neighbours, neighbour_edges, \
           full_send_dict, ghost_recv_dict


# ---------------------------------------------------------------------------
# parallel_rectangular_cross  (thin wrapper — backward compatible)
# ---------------------------------------------------------------------------

def parallel_rectangular_cross(m_g, n_g, len1_g=1.0, len2_g=1.0,
                                origin_g=(0.0, 0.0)):
    """Build the local portion of a rectangular cross (4-triangle/cell) mesh.

    Returns the same data as
    :func:`parallel_rectangular_cross_with_neighbours` but without the
    neighbour arrays.

    Returns
    -------
    points, elements, boundary, full_send_dict, ghost_recv_dict
    """
    points, elements, boundary, _neighbours, _neighbour_edges, \
        full_send_dict, ghost_recv_dict = \
        parallel_rectangular_cross_with_neighbours(m_g, n_g, len1_g, len2_g,
                                                   origin_g)
    return points, elements, boundary, full_send_dict, ghost_recv_dict


# ---------------------------------------------------------------------------
# Local-to-global index maps
# ---------------------------------------------------------------------------

def _build_tri_l2g(tris_per_col, mm, n_g, i_start,
                   has_left_ghost, has_right_ghost):
    """Return *tri_l2g* — local triangle index → global triangle index.

    After *_reorder_full_first* the local triangle order is:
    ``[full cols 0..mm-1] [ghost-left col] [ghost-right col]``.

    Global numbering: column *c_g* owns triangles
    ``c_g * tris_per_col`` … ``(c_g+1) * tris_per_col - 1``.
    """
    n_full        = mm * tris_per_col
    n_ghost_left  = tris_per_col if has_left_ghost  else 0
    n_ghost_right = tris_per_col if has_right_ghost else 0
    Nt = n_full + n_ghost_left + n_ghost_right

    tri_l2g = num.empty(Nt, dtype=int)

    # Full columns (i_start … i_start+mm-1 globally)
    g_cols = num.arange(mm) + i_start
    row_tri = num.arange(tris_per_col)
    tri_l2g[:n_full] = (g_cols[:, None] * tris_per_col + row_tri[None, :]).ravel()

    # Ghost left (global column i_start-1)
    if has_left_ghost:
        tri_l2g[n_full:n_full + tris_per_col] = \
            (i_start - 1) * tris_per_col + row_tri

    # Ghost right (global column i_start+mm)
    if has_right_ghost:
        tri_l2g[n_full + n_ghost_left:] = \
            (i_start + mm) * tris_per_col + row_tri

    return tri_l2g


def _build_node_l2g_rect(m_g, n_g, m_local, i_start, left_offset):
    """Local node index → global node index for a rectangular mesh.

    Node ordering: ``I(i, j) = i * (n_g+1) + j``.
    Local column *i_local* corresponds to global column
    ``i_local + i_start - left_offset``.
    """
    i_locals  = num.arange(m_local + 1)
    i_globals = i_locals + i_start - left_offset
    j_range   = num.arange(n_g + 1)
    return (i_globals[:, None] * (n_g + 1) + j_range[None, :]).ravel()


def _build_node_l2g_cross(m_g, n_g, m_local, i_start, left_offset):
    """Local node index → global node index for a cross mesh.

    Corner ordering: ``I(i, j) = i * (n_g+1) + j``  (first ``(m+1)*(n+1)`` nodes).
    Centre ordering: ``C(i, j) = (m+1)*(n+1) + i*n + j`` (remaining ``m*n`` nodes).
    """
    i_locals  = num.arange(m_local + 1)
    i_globals = i_locals + i_start - left_offset
    j_range   = num.arange(n_g + 1)
    corners   = (i_globals[:, None] * (n_g + 1) + j_range[None, :]).ravel()

    # Centre nodes
    i_locals_c  = num.arange(m_local)
    i_globals_c = i_locals_c + i_start - left_offset
    j_range_c   = num.arange(n_g)
    global_centre_offset = (m_g + 1) * (n_g + 1)
    centres = global_centre_offset + \
              (i_globals_c[:, None] * n_g + j_range_c[None, :]).ravel()

    return num.concatenate([corners, centres])


# ---------------------------------------------------------------------------
# Domain wrappers
# ---------------------------------------------------------------------------

def parallel_rectangular_domain(m_g, n_g, len1_g=1.0, len2_g=1.0,
                                 origin_g=(0.0, 0.0)):
    """Return a :class:`~anuga.parallel.Parallel_domain` on a rectangular mesh.

    In serial (numprocs == 1 or mpi4py unavailable) returns a plain
    :class:`~anuga.Domain` built from :func:`~anuga.rectangular_domain`.

    Parameters
    ----------
    m_g, n_g : int
        Global cell counts in x and y.
    len1_g, len2_g : float
        Domain lengths in x and y (default 1.0).
    origin_g : tuple
        Lower-left corner (default (0, 0)).

    Returns
    -------
    Domain or Parallel_domain
    """

    if not pypar_available or numprocs == 1:
        from anuga.abstract_2d_finite_volumes.mesh_factory import \
            rectangular_with_neighbours
        from anuga.abstract_2d_finite_volumes.neighbour_mesh import Mesh
        from anuga.shallow_water.shallow_water_domain import Domain
        points, vertices, boundary, neighbours, neighbour_edges = \
            rectangular_with_neighbours(m_g, n_g,
                                        len1=len1_g, len2=len2_g,
                                        origin=origin_g)
        mesh = Mesh(points, vertices, boundary,
                    triangle_neighbours=neighbours,
                    triangle_neighbour_edges=neighbour_edges)
        return Domain(mesh)

    from .parallel_shallow_water import Parallel_domain
    from anuga.abstract_2d_finite_volumes.neighbour_mesh import Mesh

    points, elements, boundary, neighbours, neighbour_edges, \
        full_send_dict, ghost_recv_dict = \
        parallel_rectangular_with_neighbours(m_g, n_g, len1_g, len2_g,
                                             origin_g)

    mesh = Mesh(points, elements, boundary,
                triangle_neighbours=neighbours,
                triangle_neighbour_edges=neighbour_edges)

    i_start, mm = _column_distribution(m_g, numprocs, myid)
    has_left_ghost  = (myid > 0)
    has_right_ghost = (myid < numprocs - 1)
    left_offset  = 1 if has_left_ghost  else 0
    right_offset = 1 if has_right_ghost else 0
    m_local = mm + left_offset + right_offset

    number_of_full_triangles = 2 * mm * n_g
    number_of_full_nodes     = (mm + 1) * (n_g + 1)

    tri_l2g  = _build_tri_l2g(2 * n_g, mm, n_g, i_start,
                               has_left_ghost, has_right_ghost)
    node_l2g = _build_node_l2g_rect(m_g, n_g, m_local, i_start, left_offset)

    return Parallel_domain(
        mesh, None,
        full_send_dict=full_send_dict,
        ghost_recv_dict=ghost_recv_dict,
        number_of_full_nodes=number_of_full_nodes,
        number_of_full_triangles=number_of_full_triangles,
        number_of_global_triangles=2 * m_g * n_g,
        number_of_global_nodes=(m_g + 1) * (n_g + 1),
        tri_l2g=tri_l2g,
        node_l2g=node_l2g,
    )


def parallel_rectangular_cross_domain(m_g, n_g, len1_g=1.0, len2_g=1.0,
                                       origin_g=(0.0, 0.0)):
    """Return a :class:`~anuga.parallel.Parallel_domain` on a cross mesh.

    In serial (numprocs == 1 or mpi4py unavailable) returns a plain
    :class:`~anuga.Domain` built from :func:`~anuga.rectangular_cross_domain`.

    Parameters
    ----------
    m_g, n_g : int
        Global cell counts in x and y.
    len1_g, len2_g : float
        Domain lengths in x and y (default 1.0).
    origin_g : tuple
        Lower-left corner (default (0, 0)).

    Returns
    -------
    Domain or Parallel_domain
    """

    if not pypar_available or numprocs == 1:
        from anuga.extras import rectangular_cross_domain
        return rectangular_cross_domain(m_g, n_g,
                                        len1=len1_g, len2=len2_g,
                                        origin=origin_g)

    from .parallel_shallow_water import Parallel_domain
    from anuga.abstract_2d_finite_volumes.neighbour_mesh import Mesh

    points, elements, boundary, neighbours, neighbour_edges, \
        full_send_dict, ghost_recv_dict = \
        parallel_rectangular_cross_with_neighbours(m_g, n_g, len1_g, len2_g,
                                                   origin_g)

    mesh = Mesh(points, elements, boundary,
                triangle_neighbours=neighbours,
                triangle_neighbour_edges=neighbour_edges)

    i_start, mm = _column_distribution(m_g, numprocs, myid)
    has_left_ghost  = (myid > 0)
    has_right_ghost = (myid < numprocs - 1)
    left_offset  = 1 if has_left_ghost  else 0
    right_offset = 1 if has_right_ghost else 0
    m_local = mm + left_offset + right_offset

    number_of_full_triangles = 4 * mm * n_g
    number_of_full_nodes     = (mm + 1) * (n_g + 1) + mm * n_g

    tri_l2g  = _build_tri_l2g(4 * n_g, mm, n_g, i_start,
                               has_left_ghost, has_right_ghost)
    node_l2g = _build_node_l2g_cross(m_g, n_g, m_local, i_start, left_offset)

    return Parallel_domain(
        mesh, None,
        full_send_dict=full_send_dict,
        ghost_recv_dict=ghost_recv_dict,
        number_of_full_nodes=number_of_full_nodes,
        number_of_full_triangles=number_of_full_triangles,
        number_of_global_triangles=4 * m_g * n_g,
        number_of_global_nodes=(m_g + 1) * (n_g + 1) + m_g * n_g,
        tri_l2g=tri_l2g,
        node_l2g=node_l2g,
    )
