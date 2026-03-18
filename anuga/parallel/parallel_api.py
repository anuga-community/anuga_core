"""Trying to lump parallel stuff into simpler interface


"""
import numpy as num

# The abstract Python-MPI interface
from anuga.utilities.parallel_abstraction import size, rank, get_processor_name
from anuga.utilities.parallel_abstraction import finalize, send, receive, reduce
from anuga.utilities.parallel_abstraction import isend, waitall
from anuga.utilities.parallel_abstraction import pypar_available, barrier

from anuga.parallel.sequential_distribute import sequential_distribute_dump
from anuga.parallel.sequential_distribute import sequential_distribute_load

# ANUGA parallel engine (only load if pypar can)
if pypar_available:
    from anuga.parallel.distribute_mesh  import send_submesh
    from anuga.parallel.distribute_mesh  import rec_submesh
    from anuga.parallel.distribute_mesh  import extract_submesh

    # Mesh partitioning using Metis
    from anuga.parallel.distribute_mesh import build_submesh
    from anuga.parallel.distribute_mesh import partition_mesh
    from anuga.parallel.distribute_mesh import ghost_commun_pattern
    from anuga.parallel.distribute_mesh import build_local_mesh

    from anuga.parallel.parallel_shallow_water import Parallel_domain



from anuga.abstract_2d_finite_volumes.neighbour_mesh import Mesh

#------------------------------------------------------------------------------
# Read in processor information
#------------------------------------------------------------------------------

numprocs = size()
myid = rank()
processor_name = get_processor_name()
#print 'I am processor %d of %d on node %s' %(myid, numprocs, processor_name)



def collect_value(value):

    value = value

    if myid == 0:
        for i in range(numprocs):
            if i == 0: continue
            val = receive(i)
            value = value + val
    else:
        send(value, 0)


    if myid == 0:
        for i in range(1,numprocs):
            send(value,i)
    else:
        value = receive(0)


    return value




def distribute(domain, verbose=False, debug=False, parameters = None):
    """Distribute the domain to all processes in a parallel computing environment.

    This function partitions a computational domain across multiple processes and
    creates parallel domain instances on each process. The master process (myid=0)
    handles the domain partitioning and distributes submeshes to worker processes,
    while worker processes receive their respective submesh data.

    Parameters
    ----------
    domain : Domain
        The computational domain to be distributed across processes.
    verbose : bool, optional
        If True, print detailed information during distribution process.
        Default is False.
    debug : bool, optional
        If True, enable debug mode for additional diagnostics.
        Default is False.
    parameters : dict, optional
        User-defined parameters to customize distribution behavior,
        particularly the size of ghost layers. Default is None.

    Returns
    -------
    Parallel_domain or Domain
        A Parallel_domain object containing the distributed submesh data and
        quantities specific to the current process. If mpi4py is not available
        or only one process is running, returns the original domain unchanged.

    Notes
    -----
    - Requires mpi4py for parallel communication
    - Only functions with multiple processes (numprocs > 1)
    - The master process (myid=0) partitions the domain and distributes data
    - All other processes receive submesh data from the master process
    - Boundary conditions are transferred as ghost boundaries
    - Domain attributes (flow algorithm, georeferencing, etc.) are preserved
      in the parallel domain instances
    """

    if not pypar_available or numprocs == 1 : return domain # Bypass


    if myid == 0:
        from .sequential_distribute import Sequential_distribute
        partition = Sequential_distribute(domain, verbose, debug, parameters)

        partition.distribute(numprocs)

        kwargs, points, vertices, boundary, quantities, boundary_map, \
                domain_name, domain_dir, domain_store, domain_store_centroids, \
                domain_minimum_storable_height, domain_minimum_allowed_height, \
                domain_flow_algorithm, domain_georef, \
                domain_quantities_to_be_stored, domain_smooth, domain_low_froude \
                 = partition.extract_submesh(0)

        requests = []
        for p in range(1, numprocs):

            tostore = partition.extract_submesh(p)

            requests.append(isend(tostore, p))

        waitall(requests)

    else:

        kwargs, points, vertices, boundary, quantities, boundary_map, \
            domain_name, domain_dir, domain_store, domain_store_centroids, \
            domain_minimum_storable_height, domain_minimum_allowed_height, \
            domain_flow_algorithm, domain_georef, \
            domain_quantities_to_be_stored, domain_smooth, domain_low_froude\
             = receive(0)

    #---------------------------------------------------------------------------
    # Now Create parallel domain
    #---------------------------------------------------------------------------
    parallel_domain = Parallel_domain(points, vertices, boundary, **kwargs)


    #------------------------------------------------------------------------
    # Copy in quantity data
    #------------------------------------------------------------------------
    for q in quantities:
        try:
            parallel_domain.set_quantity(q, quantities[q], location='centroids')
        except KeyError:
            #print 'Try to create quantity %s'% q
            from anuga import Quantity
            Q = Quantity(parallel_domain, name=q, register=True)
            parallel_domain.set_quantity(q, quantities[q], location='centroids')

    #------------------------------------------------------------------------
    # Transfer boundary conditions to each subdomain
    #------------------------------------------------------------------------
    boundary_map['ghost'] = None  # Add binding to ghost boundary
    parallel_domain.set_boundary(boundary_map)


    #------------------------------------------------------------------------
    # Transfer other attributes to each subdomain
    #------------------------------------------------------------------------

    parallel_domain.set_flow_algorithm(domain_flow_algorithm)
    parallel_domain.set_name(domain_name)
    parallel_domain.set_datadir(domain_dir)
    parallel_domain.set_store(domain_store)
    parallel_domain.set_low_froude(domain_low_froude)
    parallel_domain.set_store_centroids(domain_store_centroids)
    parallel_domain.set_minimum_storable_height(domain_minimum_storable_height)
    parallel_domain.set_minimum_allowed_height(domain_minimum_allowed_height)
    parallel_domain.geo_reference = domain_georef
    parallel_domain.set_quantities_to_be_stored(domain_quantities_to_be_stored)
    parallel_domain.smooth = domain_smooth

    return parallel_domain



def distribute_collaborative(domain, verbose=False, debug=False, parameters=None):
    """MPI-collaborative domain distribution — eliminates the rank-0 bottleneck.

    The standard :func:`distribute` function builds ALL P submeshes sequentially
    on rank 0 — O(P × N) work.  This function instead:

    1. Rank 0 partitions the mesh (METIS / Morton / Hilbert).
    2. All ranks receive the full mesh topology via MPI broadcast.
    3. Each rank independently builds its own ghost layer using the
       Cython BFS kernel (``ghost_layer_bfs`` / ``ghost_bnd_layer_classify``).
    4. One ``MPI_Allgather`` of ghost-communication arrays lets every rank
       derive its own ``full_commun`` table without further coordination.
    5. Each rank calls ``build_local_mesh`` locally.

    The only serial bottleneck remaining on rank 0 is the METIS partition call,
    which is O(N) and typically a small fraction of total setup time.

    Parameters
    ----------
    domain : Domain
        The full serial domain (only meaningful on rank 0; other ranks may
        pass any Domain object — it will not be read).
    verbose : bool, optional
    debug : bool, optional
    parameters : dict, optional
        Passed through to ``partition_mesh``.  Recognised extra keys:

        ``ghost_layer_width`` : int, default 2

    Returns
    -------
    Parallel_domain
        The local parallel domain for this MPI rank.
    """

    if not pypar_available:
        return domain

    from mpi4py import MPI
    comm  = MPI.COMM_WORLD
    myid  = comm.Get_rank()
    numprocs = comm.Get_size()

    if numprocs == 1:
        return domain

    if debug:
        verbose = True

    layer_width = (parameters or {}).get('ghost_layer_width', 2)

    # ── Step 1: rank 0 partitions ────────────────────────────────────────
    if myid == 0:
        new_mesh, tpp, quantities, _s2p, _p2s = partition_mesh(
            domain, numprocs, parameters=parameters, verbose=verbose)

        # Build dummy boundary_map from boundary tags (same as Sequential_distribute).
        # domain.boundary_map may be None if set_boundary() hasn't been called yet.
        bdmap = domain.boundary_map
        if bdmap is None:
            bdmap = {tag: None for tag in domain.get_boundary_tags()}
            domain.set_boundary(bdmap)

        domain_attrs = {
            'name':                     domain.get_name(),
            'dir':                      domain.get_datadir(),
            'store':                    domain.get_store(),
            'store_centroids':          domain.get_store_centroids(),
            'min_store_height':         domain.minimum_storable_height,
            'min_allowed_height':       domain.get_minimum_allowed_height(),
            'flow_algorithm':           domain.get_flow_algorithm(),
            'geo_reference':            domain.geo_reference,
            'quantities_to_be_stored':  domain.quantities_to_be_stored,
            'smooth':                   domain.smooth,
            'low_froude':               domain.low_froude,
            'boundary_map':             domain.boundary_map,
            'num_global_tri':           domain.number_of_triangles,
            'num_global_nodes':         domain.number_of_nodes,
        }
        quant_keys = list(quantities.keys())
    else:
        new_mesh = tpp = quantities = domain_attrs = quant_keys = None

    # ── Step 2: broadcast small metadata ─────────────────────────────────
    domain_attrs = comm.bcast(domain_attrs, root=0)
    tpp          = comm.bcast(tpp,          root=0)
    quant_keys   = comm.bcast(quant_keys,   root=0)

    # ── Step 3: broadcast mesh topology (large numpy arrays) ─────────────
    def _bcast_ndarray(arr_on_root):
        """Broadcast one numpy array from rank 0. arr_on_root is None on other ranks."""
        meta = (arr_on_root.shape, arr_on_root.dtype.str) if myid == 0 else None
        shape, dtype_str = comm.bcast(meta, root=0)
        buf = num.empty(shape, dtype=num.dtype(dtype_str))
        if myid == 0:
            buf[...] = arr_on_root
        comm.Bcast(buf, root=0)
        return buf

    nodes      = _bcast_ndarray(new_mesh.nodes      if myid == 0 else None)
    triangles  = _bcast_ndarray(new_mesh.triangles  if myid == 0 else None)
    neighbours = _bcast_ndarray(new_mesh.neighbours if myid == 0 else None)
    boundary   = comm.bcast(new_mesh.boundary if myid == 0 else None, root=0)

    # ── Step 4: broadcast quantities ─────────────────────────────────────
    # Each rank will extract its own full and ghost slices after the BFS.
    quant_all = {k: _bcast_ndarray(quantities[k] if myid == 0 else None)
                 for k in quant_keys}

    # ── Step 5: each rank builds its own ghost layer ──────────────────────
    cumsum  = num.concatenate([[0], num.cumsum(tpp)])
    tlower  = int(cumsum[myid])
    tupper  = int(cumsum[myid + 1])

    neighbours_c = num.ascontiguousarray(neighbours, dtype=num.int64)

    from .distribute_mesh_ext import ghost_layer_bfs, ghost_bnd_layer_classify

    ghost_ids            = ghost_layer_bfs(neighbours_c, tlower, tupper, layer_width)
    tri_ids, edge_ids    = ghost_bnd_layer_classify(neighbours_c, ghost_ids,
                                                    tlower, tupper)

    # ── Step 6: assemble local submesh_cell ──────────────────────────────
    # Full triangles and nodes
    full_triangles = triangles[tlower:tupper]           # (N_full, 3)
    full_node_ids  = num.unique(full_triangles.flat)
    full_nodes_col = num.concatenate(                   # (N_fn, 3): [gid, x, y]
        [full_node_ids.reshape(-1, 1), nodes[full_node_ids]], axis=1)

    # Ghost triangles and nodes
    ghost_tri_verts = triangles[ghost_ids]              # (G, 3)
    ghost_node_ids  = num.setdiff1d(num.unique(ghost_tri_verts.flat), full_node_ids)
    ghost_nodes_col = num.concatenate(                  # (G_n, 3): [gid, x, y]
        [ghost_node_ids.reshape(-1, 1), nodes[ghost_node_ids]], axis=1)
    ghost_tris_col  = num.concatenate(                  # (G, 4): [gid, v0, v1, v2]
        [ghost_ids.reshape(-1, 1), ghost_tri_verts], axis=1)

    # Full boundary: binary-search into the sorted global boundary dict
    sorted_bnd_keys    = sorted(boundary.keys(), key=lambda k: k[0])
    if sorted_bnd_keys:
        sorted_bnd_tids = num.array([k[0] for k in sorted_bnd_keys], dtype=int)
        lo = int(num.searchsorted(sorted_bnd_tids, tlower))
        hi = int(num.searchsorted(sorted_bnd_tids, tupper))
        full_boundary = {sorted_bnd_keys[i]: boundary[sorted_bnd_keys[i]]
                         for i in range(lo, hi)}
    else:
        full_boundary = {}

    # Ghost boundary: start with 'ghost', override with real tags where present
    ghost_boundary = {(int(t), int(e)): 'ghost'
                      for t, e in zip(tri_ids, edge_ids)}
    ghost_boundary.update((k, boundary[k])
                          for k in ghost_boundary.keys() & boundary.keys())

    # Ghost communication pattern: ghost_tris_col[:, 0] → owning processor
    tri_per_proc_ranges = cumsum[1:] - 1
    ghost_commun = ghost_commun_pattern(ghost_tris_col, myid, tri_per_proc_ranges)

    # ── Step 7: allgather ghost_commun → each rank derives full_commun ───
    # ghost_commun[p] has shape (G_p, 2): col0 = global_id, col1 = owner_proc.
    # After allgather, rank myid filters for entries where owner == myid.
    all_ghost_commun = comm.allgather(ghost_commun)

    full_commun = {}
    for q, gc in enumerate(all_ghost_commun):
        for row in gc:
            gid   = int(row[0])
            owner = int(row[1])
            if owner == myid:
                if gid not in full_commun:
                    full_commun[gid] = []
                full_commun[gid].append(q)

    # ── Step 8: build local mesh in GA format ────────────────────────────
    submesh_cell = {
        'ghost_layer_width': layer_width,
        'full_nodes':        full_nodes_col,
        'ghost_nodes':       ghost_nodes_col,
        'full_triangles':    full_triangles,
        'ghost_triangles':   ghost_tris_col,
        'full_boundary':     full_boundary,
        'ghost_boundary':    ghost_boundary,
        'ghost_commun':      ghost_commun,
        'full_commun':       full_commun,
        'full_quan':  {k: quant_all[k][tlower:tupper] for k in quant_keys},
        'ghost_quan': {k: quant_all[k][ghost_ids]     for k in quant_keys},
    }

    points, vertices, boundary_local, quantities_local, ghost_recv_dict, \
        full_send_dict, tri_map, node_map, tri_l2g, node_l2g, glw = \
        build_local_mesh(submesh_cell, tlower, tupper, numprocs)

    number_of_full_nodes     = len(full_nodes_col)
    number_of_full_triangles = tupper - tlower

    # ── Step 9: create Parallel_domain ───────────────────────────────────
    kwargs = {
        'full_send_dict':             full_send_dict,
        'ghost_recv_dict':            ghost_recv_dict,
        'number_of_full_nodes':       number_of_full_nodes,
        'number_of_full_triangles':   number_of_full_triangles,
        'geo_reference':              domain_attrs['geo_reference'],
        'number_of_global_triangles': domain_attrs['num_global_tri'],
        'number_of_global_nodes':     domain_attrs['num_global_nodes'],
        'processor':                  myid,
        'numproc':                    numprocs,
        's2p_map':                    None,
        'p2s_map':                    None,
        'tri_l2g':                    tri_l2g,
        'node_l2g':                   node_l2g,
        'ghost_layer_width':          glw,
    }

    parallel_domain = Parallel_domain(points, vertices, boundary_local, **kwargs)

    for q in quantities_local:
        try:
            parallel_domain.set_quantity(q, quantities_local[q], location='centroids')
        except KeyError:
            from anuga import Quantity
            Q = Quantity(parallel_domain, name=q, register=True)
            parallel_domain.set_quantity(q, quantities_local[q], location='centroids')

    boundary_map = domain_attrs['boundary_map'].copy()
    boundary_map['ghost'] = None
    parallel_domain.set_boundary(boundary_map)

    parallel_domain.set_name(domain_attrs['name'])
    parallel_domain.set_datadir(domain_attrs['dir'])
    parallel_domain.set_store(domain_attrs['store'])
    parallel_domain.set_store_centroids(domain_attrs['store_centroids'])
    parallel_domain.set_minimum_storable_height(domain_attrs['min_store_height'])
    parallel_domain.set_minimum_allowed_height(domain_attrs['min_allowed_height'])
    parallel_domain.set_flow_algorithm(domain_attrs['flow_algorithm'])
    parallel_domain.geo_reference = domain_attrs['geo_reference']
    parallel_domain.set_quantities_to_be_stored(domain_attrs['quantities_to_be_stored'])
    parallel_domain.smooth = domain_attrs['smooth']
    parallel_domain.set_low_froude(domain_attrs['low_froude'])

    return parallel_domain


def distribute_mesh(domain, verbose=False, debug=False, parameters=None):
    """ Distribute and send the mesh info to all the processors.
    Should only be run from processor 0 and will send info to the other
    processors.
    There should be a corresponding  rec_submesh called from all the other
    processors
    """

    if debug:
        verbose = True

    numprocs = size()


    # Subdivide the mesh
    if verbose: print('Subdivide mesh')
    new_mesh, triangles_per_proc, quantities, \
           s2p_map, p2s_map = \
           partition_mesh(domain, numprocs)


    # Build the mesh that should be assigned to each processor,
    # this includes ghost nodes and the communication pattern
    if verbose: print('Build submeshes')
    submesh = build_submesh(new_mesh, quantities, triangles_per_proc, parameters)

    if verbose:
        for p in range(numprocs):
            N = len(submesh['ghost_nodes'][p])
            M = len(submesh['ghost_triangles'][p])
            print('There are %d ghost nodes and %d ghost triangles on proc %d'\
                  %(N, M, p))

    #if debug:
    #    from pprint import pprint
    #    pprint(submesh)


    # Send the mesh partition to the appropriate processor
    if verbose: print('Distribute submeshes')
    for p in range(1, numprocs):
        send_submesh(submesh, triangles_per_proc, p2s_map, p, verbose)

    # Build the local mesh for processor 0
    points, vertices, boundary, quantities, \
            ghost_recv_dict, full_send_dict, \
            tri_map, node_map, tri_l2g, node_l2g, ghost_layer_width =\
              extract_submesh(submesh, triangles_per_proc, p2s_map, 0)



    # Keep track of the number full nodes and triangles.
    # This is useful later if one needs access to a ghost-free domain
    # Here, we do it for process 0. The others are done in rec_submesh.
    number_of_full_nodes = len(submesh['full_nodes'][0])
    number_of_full_triangles = len(submesh['full_triangles'][0])


    # Return structures necessary for building the parallel domain
    return points, vertices, boundary, quantities,\
           ghost_recv_dict, full_send_dict,\
           number_of_full_nodes, number_of_full_triangles, \
           s2p_map, p2s_map, tri_map, node_map, tri_l2g, node_l2g, \
           ghost_layer_width



def mpicmd(script_name='echo', numprocs=3):

    extra_options = mpi_extra_options()

    return "mpiexec -np %d  %s  python -m mpi4py %s" % (numprocs, extra_options, script_name)  

def mpi_extra_options():

    extra_options = '--oversubscribe'
    cmd = 'mpiexec -np 3 ' + extra_options + ' echo '

    #print(cmd)
    import subprocess
    result = subprocess.run(cmd.split(), capture_output=True)
    if result.returncode != 0:
        extra_options = ' '

    import platform
    if platform.system() == 'Windows':
        extra_options = ' '

    return extra_options