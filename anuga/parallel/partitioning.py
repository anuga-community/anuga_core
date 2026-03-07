

import numpy as num



def metis_partition(domain, n_procs):
    """
    Partition a mesh using METIS partitioning library.

    This function is a wrapper for the METIS partitioning routines in pymetis.
    The specific partitioning routine used depends on the installed METIS version:
    - METIS 4: Uses partMeshNodal
    - METIS 5: Uses part_mesh if available, otherwise part_graph

    Parameters
    ----------
    domain : object
        Domain object containing mesh information including number of triangles,
        nodes, triangle connectivity, and neighbor information.
    n_procs : int
        Number of processors to partition the mesh across.

    Returns
    -------
    epart_order : ndarray
        Integer array of indices that sort elements by partition assignment.
        Can be used to reorder triangles such that all triangles assigned to
        the same processor are contiguous.
    triangles_per_proc : ndarray
        Integer array of length n_procs where triangles_per_proc[i] is the
        number of triangles assigned to processor i.

    Raises
    ------
    ValueError
        If n_procs is greater than the number of triangles in the domain.
    AssertionError
        If partitioning results in any processor owning zero triangles.

    Notes
    -----
    The partitioning minimizes edge cuts to improve communication efficiency
    in parallel computations. For single processor (n_procs=1), returns trivial
    partitioning without invoking METIS.

    The function handles edge cases where METIS returns zero-dimensional arrays
    by flattening them to standard 1D arrays.
    """

    from pymetis import part_graph

    try:
        from pymetis import part_mesh
        metis_version = "5_part_mesh"
    except:
        metis_version = "5_part_graph"

    n_tri = domain.number_of_triangles

    if n_procs > n_tri:
        raise ValueError("Number of processors must be less than or equal to the number of triangles")  

    if n_procs == 1:
        epart_order = num.arange(n_tri, dtype=int)
        triangles_per_proc = [n_tri]
        return epart_order, triangles_per_proc

    # Use metis to partition the mesh. 
    # The partitioning routine used depends on the version of metis installed. 
    # If metis 4 is installed, partMeshNodal is used. If metis 5 is installed, 
    # part_mesh is used if available, otherwise part_graph

    if metis_version == 4:
        n_vert = domain.get_number_of_nodes()
        t_list2 = domain.triangles.copy()
        t_list = num.reshape(t_list2, (-1,))
        # The 1 here is for triangular mesh elements.
        edgecut, epart, npart = partMeshNodal(n_tri, n_vert, t_list, 1, n_procs)
        # print edgecut
        # print npart
        #print epart
        del edgecut
        del npart


    if metis_version == "5_part_mesh":

        objval, epart, npart = part_mesh(n_procs, domain.triangles)


    if metis_version == "5_part_graph":
        # build adjacency list
        # neighbours uses negative integer-indices to denote boudary edges.
        # pymetis totally cant handle that, so we have to delete these.
        neigh = domain.neighbours.tolist()
        for i in range(len(neigh)):
            if neigh[i][2] < 0:
                del neigh[i][2]
            if neigh[i][1] < 0:
                del neigh[i][1]
            if neigh[i][0] < 0:
                del neigh[i][0]

        cutcount, partvert = part_graph(n_procs, neigh)

        epart = partvert

    # Sometimes (usu. on x86_64), partMeshNodal returns an array of zero
    # dimensional arrays. Correct this.
    # TODO: Not sure if this can still happen with metis 5
    if type(epart[0]) == num.ndarray:
        epart_new = num.zeros(len(epart), int)
        epart_new[:] = epart[:][0]
        epart = epart_new
        del epart_new


    triangles_per_proc = num.bincount(epart)

    msg = "Partition created where at least one submesh has no triangles. "
    msg += "Try using a smaller number of mpi processes."
    assert num.all(triangles_per_proc > 0), msg

    #proc_sum = num.zeros(n_procs+1, int)
    #proc_sum[1:] = num.cumsum(triangles_per_proc)

    epart_order = num.argsort(epart, kind='mergesort')

    return epart_order, triangles_per_proc


#==============================================================
# Code for computing Morton (Z-order) codes for 2D points, 
# used for spatial locality-preserving ordering of mesh elements.
# This is based on the "Bit Twiddling Hacks" by Sean Eron Anderson:
# https://graphics.stanford.edu/~seander/bithacks.html#InterleaveTables
#==============================================================


def morton_partition(domain, n_procs):
    """
    Partition a mesh using Morton (Z-order) codes.

    This function computes Morton codes for the centroids of the triangles
    in the domain, sorts the triangles by these codes, and then divides them
    into contiguous blocks for each processor. This is a simple spatial
    partitioning method that can improve cache locality in parallel computations.

    Parameters
    ----------
    domain : object
        Domain object containing mesh information including number of triangles,
        nodes, triangle connectivity, and neighbor information.
    n_procs : int
        Number of processors to partition the mesh across.

    Returns
    -------
    epart_order : ndarray
        Integer array of indices that sort elements by Morton code order.
        Can be used to reorder triangles such that all triangles assigned to
        the same processor are contiguous.
    triangles_per_proc : ndarray
        Integer array of length n_procs where triangles_per_proc[i] is the
        number of triangles assigned to processor i.

    Raises
    ------
    ValueError
        If n_procs is greater than the number of triangles in the domain.
    """

    n_tri = domain.number_of_triangles

    if n_procs > n_tri:
        raise ValueError("Number of processors must be less than or equal to the number of triangles")  


    points = domain.centroid_coordinates
    order = morton_order_from_points(points)

    # Now we have an ordering of the triangles based on their centroids' Morton codes.
    # We can divide this ordering into contiguous blocks for each processor.

    triangles_per_proc = num.full(n_procs, n_tri // n_procs, dtype=int)
    remainder = n_tri % n_procs
    if remainder > 0:
        triangles_per_proc[:remainder] += 1

    assert triangles_per_proc.sum() == n_tri, "Total number of triangles must match after partitioning"

    return order, triangles_per_proc

def _part1by1_32(n):
    """Expand 16 lower bits so that there is one zero bit between each.

    Parameters
    ----------
    n : ndarray of uint32
        Input values. Only the lower 16 bits are used.

    Returns
    -------
    ndarray of uint32
        Values with bits of ``n`` separated by one zero bit, suitable
        for use in 2D Morton encoding.
    """
    n &= np.uint32(0x0000FFFF)
    n = (n | (n << 8)) & np.uint32(0x00FF00FF)
    n = (n | (n << 4)) & np.uint32(0x0F0F0F0F)
    n = (n | (n << 2)) & np.uint32(0x33333333)
    n = (n | (n << 1)) & np.uint32(0x55555555)
    return n


def morton2d_uint32(ix, iy):
    """Compute 2D Morton (Z-order) codes for integer coordinates.

    This uses a 32-bit encoder with 16 bits per axis, suitable for
    integer coordinates in the range [0, 2**16 - 1].

    Parameters
    ----------
    ix : array_like of uint32
        X coordinates mapped to integers in [0, 2**16 - 1].
    iy : array_like of uint32
        Y coordinates mapped to integers in [0, 2**16 - 1].

    Returns
    -------
    ndarray of uint32
        Morton code for each pair ``(ix[i], iy[i])``.
    """
    ix = ix.astype(np.uint32)
    iy = iy.astype(np.uint32)
    return _part1by1_32(ix) | (_part1by1_32(iy) << np.uint32(1))


def morton_order_from_points(points, grid_bits=16):
    """Return Morton ordering of triangles from point coordinates.

    The points are first normalized to the unit square based on their
    bounding box, then mapped to an integer grid with ``2**grid_bits``
    cells per axis. A 2D Morton code is computed for each point and
    used to derive a permutation that is spatially locality-preserving.

    Parameters
    ----------
    points : array_like, shape (N, 2)
        Points as ``(x, y)`` coordinates.
    grid_bits : int, optional
        Number of bits per axis for the integer grid. Must be
        ``<= 16`` for this implementation. The effective resolution
        per axis is ``2**grid_bits``. Default is 16.

    Returns
    -------
    order : ndarray of int, shape (N,)
        Indices that sort the points in Morton (Z-order) order.

    Notes
    -----
    This function is well suited for reordering per-element quantities
    in a finite-volume or finite-element code to improve cache
    locality. For example, for an ANUGA domain, you can pass
    ``domain.centroid_coordinates`` as ``points`` and then reorder
    per-triangle arrays with the returned permutation.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    if grid_bits < 1 or grid_bits > 16:
        raise ValueError("grid_bits must be in the range [1, 16]")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = maxs - mins

    # Avoid zero span in a degenerate direction.
    span[span == 0.0] = 1.0

    norm = (points - mins) / span

    max_int = np.uint32((1 << grid_bits) - 1)
    ix = np.minimum((norm[:, 0] * max_int).astype(np.uint32), max_int)
    iy = np.minimum((norm[:, 1] * max_int).astype(np.uint32), max_int)

    codes = morton2d_uint32(ix, iy)
    order = np.argsort(codes, kind="mergesort")
    return order
