"""Trying to lump parallel stuff into simpler interface


"""

import numpy as num

from anuga import Domain

from anuga.parallel.distribute_mesh  import send_submesh
from anuga.parallel.distribute_mesh  import rec_submesh
from anuga.parallel.distribute_mesh  import extract_submesh

# Mesh partitioning using Metis
from anuga.parallel.distribute_mesh import build_submesh
from anuga.parallel.distribute_mesh import partition_mesh

from anuga.parallel.parallel_shallow_water import Parallel_domain



class Sequential_distribute:

    def __init__(self, domain, verbose=False, debug=False, parameters=None):

        if debug:
            verbose = True

        self.domain = domain
        self.verbose = verbose
        self.debug = debug
        self.parameters = parameters


    def distribute(self, numprocs=1):

        self.numprocs = numprocs

        domain = self.domain
        verbose = self.verbose
        debug = self.debug
        parameters = self.parameters

        # FIXME: Dummy assignment (until boundaries are refactored to
        # be independent of domains until they are applied)
        bdmap = {}
        for tag in domain.get_boundary_tags():
            bdmap[tag] = None

        domain.set_boundary(bdmap)


        self.domain_name = domain.get_name()
        self.domain_dir = domain.get_datadir()
        self.domain_store = domain.get_store()
        self.domain_store_centroids = domain.get_store_centroids()
        self.domain_minimum_storable_height = domain.minimum_storable_height
        self.domain_flow_algorithm = domain.get_flow_algorithm()
        self.domain_minimum_allowed_height = domain.get_minimum_allowed_height()
        self.domain_georef = domain.geo_reference
        self.domain_quantities_to_be_stored = domain.quantities_to_be_stored
        self.domain_smooth = domain.smooth
        self.domain_low_froude = domain.low_froude
        self.number_of_global_triangles = domain.number_of_triangles
        self.number_of_global_nodes = domain.number_of_nodes
        self.boundary_map = domain.boundary_map


        # Subdivide the mesh
        if verbose: print('sequential_distribute: Subdivide mesh')

        new_mesh, triangles_per_proc, quantities, \
               s2p_map, p2s_map = \
               partition_mesh(domain, numprocs, parameters=parameters, verbose=verbose)


        # Build the mesh that should be assigned to each processor,
        # this includes ghost nodes and the communication pattern
        if verbose: print('sequential_distribute: Build submeshes')
        if verbose: print('sequential_distribute: parameters: ',parameters)

        submesh = build_submesh(new_mesh, quantities, triangles_per_proc,
                                parameters=parameters, verbose=verbose)

        if verbose:
            for p in range(numprocs):
                N = len(submesh['ghost_nodes'][p])
                M = len(submesh['ghost_triangles'][p])
                print('sequential_distribute: There are %d ghost nodes and %d ghost triangles on proc %d'\
                      %(N, M, p))


        self.submesh = submesh
        self.triangles_per_proc = triangles_per_proc
        self.p2s_map =  p2s_map


    def extract_submesh(self, p=0):
        """Build the local mesh for processor p
        """

        submesh = self.submesh
        triangles_per_proc = self.triangles_per_proc
        p2s_map = self.p2s_map
        verbose = self.verbose
        debug = self.debug

        assert p>=0
        assert p<self.numprocs


        points, vertices, boundary, quantities, \
            ghost_recv_dict, full_send_dict, \
            tri_map, node_map, tri_l2g, node_l2g, ghost_layer_width =\
              extract_submesh(submesh, triangles_per_proc, p2s_map, p)


        number_of_full_nodes = len(submesh['full_nodes'][p])
        number_of_full_triangles = len(submesh['full_triangles'][p])


        if debug:
            import pprint
            print(50*"=")
            print('sequential_distribute: NODE_L2G')
            pprint.pprint(node_l2g)

            pprint.pprint(node_l2g[vertices[:,0]])

            print('sequential_distribute: VERTICES')
            pprint.pprint(vertices[:,0])
            # FIXME: new_triangles, new_nodes, original_triangles, tri_l2orig
            # are not available in this scope — these assertions are incomplete

            print('sequential_distribute: POINTS')
            pprint.pprint(points)

            print('sequential_distribute: TRI')
            pprint.pprint(tri_l2g)
            pprint.pprint(p2s_map[tri_l2g])

            print('NODES')
            pprint.pprint(node_map)
            pprint.pprint(node_l2g)

        #tri_l2orig = p2s_map[tri_l2g]

        s2p_map = None
        p2s_map = None

        #------------------------------------------------------------------------
        # Build the parallel domain for this processor using partion structures
        #------------------------------------------------------------------------

        if verbose:
            print('sequential_distribute: P%g, no_full_nodes = %g, no_full_triangles = %g' % (p, number_of_full_nodes, number_of_full_triangles))


        kwargs = {'full_send_dict': full_send_dict,
                'ghost_recv_dict': ghost_recv_dict,
                'number_of_full_nodes': number_of_full_nodes,
                'number_of_full_triangles': number_of_full_triangles,
                'geo_reference': self.domain_georef,
                'number_of_global_triangles':  self.number_of_global_triangles,
                'number_of_global_nodes':  self.number_of_global_nodes,
                'processor':  p,
                'numproc':  self.numprocs,
                's2p_map':  s2p_map,
                'p2s_map':  p2s_map, ## jj added this
                'tri_l2g':  tri_l2g, ## SR added this
                'node_l2g':  node_l2g,
                'ghost_layer_width':  ghost_layer_width}

        boundary_map = self.boundary_map
        domain_name = self.domain_name
        domain_dir = self.domain_dir
        domain_store = self.domain_store
        domain_store_centroids = self.domain_store_centroids
        domain_minimum_storable_height = self.domain_minimum_storable_height
        domain_minimum_allowed_height = self.domain_minimum_allowed_height
        domain_flow_algorithm = self.domain_flow_algorithm
        domain_georef = self.domain_georef
        domain_quantities_to_be_stored = self.domain_quantities_to_be_stored
        domain_smooth = self.domain_smooth
        domain_low_froude = self.domain_low_froude

        tostore = (kwargs, points, vertices, boundary, quantities, \
                   boundary_map, \
                   domain_name, domain_dir, domain_store, domain_store_centroids, \
                   domain_minimum_storable_height, \
                   domain_minimum_allowed_height, domain_flow_algorithm, \
                   domain_georef, domain_quantities_to_be_stored, domain_smooth, \
                   domain_low_froude)

        return tostore


    def _release_submesh_rank(self, p):
        """Null out submesh data for rank p to allow garbage collection.

        Called by sequential_distribute_dump after rank p's files have been
        written.  Once all arrays for rank p are released the GC can recover
        the memory before proceeding to rank p+1.
        """
        submesh = self.submesh
        for key in ('full_nodes', 'full_triangles', 'full_boundary',
                    'ghost_nodes', 'ghost_triangles', 'ghost_commun',
                    'ghost_boundary', 'ghost_layer_width', 'full_commun'):
            lst = submesh.get(key)
            if lst is not None and p < len(lst):
                lst[p] = None
        for k in submesh.get('full_quan', {}):
            submesh['full_quan'][k][p] = None
        for k in submesh.get('ghost_quan', {}):
            submesh['ghost_quan'][k][p] = None




def sequential_distribute_dump(domain, numprocs=1, verbose=False, partition_dir='.', debug=False, parameters = None):
    """ Distribute the domain, create parallel domain and pickle result
    """

    import gc
    import pickle
    from os.path import join

    partition = Sequential_distribute(domain, verbose, debug, parameters)

    if verbose: print('sequential_distribute_dump: Partitioning mesh to %d processes'%numprocs)
    partition.distribute(numprocs)

    # Make sure the partition_dir exists
    if partition_dir != '.':
        import os
        import errno
        try:
            os.makedirs(partition_dir, exist_ok=True)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    if verbose: print('sequential_distribute_dump: Dumping partitions to %s'%partition_dir)

    for p in range(0, numprocs):

        tostore = partition.extract_submesh(p)

        pickle_name = partition.domain_name + '_P%g_%g.pickle'% (numprocs,p)
        pickle_name = join(partition_dir,pickle_name)

        lst = list(tostore)

        # Write points and triangles to their own files
        num.save(pickle_name+".np1",tostore[1]) # num.save appends .npy to filename
        lst[1] = pickle_name+".np1.npy"
        num.save(pickle_name+".np2",tostore[2])
        lst[2] = pickle_name+".np2.npy"

        # Write each quantity to its own file
        for k in tostore[4]:
            num.save(pickle_name+".np4."+k, num.asarray(tostore[4][k]))
            lst[4][k] = pickle_name+".np4."+k+".npy"

        with open(pickle_name, 'wb') as f:
            pickle.dump(tuple(lst), f, protocol=pickle.HIGHEST_PROTOCOL)

        # Release this rank's submesh data so memory can be reclaimed before
        # processing the next rank.  Without this, all P subdomains' arrays
        # remain live throughout the entire dump loop.
        partition._release_submesh_rank(p)
        del tostore, lst

        # Run GC every rank — cyclic references inside submesh dicts are not
        # freed by refcounting alone and must be collected explicitly.
        gc.collect()

    gc.collect()
    return


def sequential_distribute_load(filename = 'domain', partition_dir = '.', verbose = False):


    from anuga import myid, numprocs

    from os.path import join

    pickle_name = filename+'_P%g_%g.pickle'% (numprocs,myid)
    pickle_name = join(partition_dir,pickle_name)

    return sequential_distribute_load_pickle_file(pickle_name, numprocs, verbose = verbose)


def sequential_distribute_load_pickle_file(pickle_name, np=1, verbose = False):
    """
    Open pickle files
    """

    f = open(pickle_name, 'rb')
    import pickle

    kwargs, points, vertices, boundary, quantities, boundary_map, \
                   domain_name, domain_dir, domain_store, domain_store_centroids, \
                   domain_minimum_storable_height, domain_minimum_allowed_height, \
                   domain_flow_algorithm, domain_georef, \
                   domain_quantities_to_be_stored, domain_smooth, \
                   domain_low_froude = pickle.load(f)
    f.close()

    # Note that quantities is a dictionary with quantity name keys and filenames of numpy arrays.
    # points and vertices are filenames of numpy arrays. These need to be loaded.
    for k in quantities:
        quantities[k] = num.load(quantities[k])
    points = num.load(points)
    vertices = num.load(vertices)

    #---------------------------------------------------------------------------
    # Create domain (parallel if np>1)
    #---------------------------------------------------------------------------
    if np>1:
        domain = Parallel_domain(points, vertices, boundary, **kwargs)
    else:
        domain = Domain(points, vertices, boundary, **kwargs)

    #------------------------------------------------------------------------
    # Copy in quantity data
    #------------------------------------------------------------------------
    for q in quantities:
        domain.set_quantity(q, quantities[q], location='centroids')


    #------------------------------------------------------------------------
    # Transfer boundary conditions to each subdomain
    #------------------------------------------------------------------------
    boundary_map['ghost'] = None  # Add binding to ghost boundary
    domain.set_boundary(boundary_map)


    #------------------------------------------------------------------------
    # Transfer other attributes to each subdomain
    #------------------------------------------------------------------------
    domain.set_name(domain_name)
    domain.set_datadir(domain_dir)
    domain.set_flow_algorithm(domain_flow_algorithm)
    domain.set_low_froude(domain_low_froude)
    domain.set_store(domain_store)
    domain.set_store_centroids(domain_store_centroids)
    domain.set_minimum_storable_height(domain_minimum_storable_height)
    domain.set_minimum_allowed_height(domain_minimum_allowed_height)
    domain.geo_reference = domain_georef
    domain.set_quantities_to_be_stored(domain_quantities_to_be_stored)
    domain.smooth = domain_smooth

    return domain


# ---------------------------------------------------------------------------
# Mesh-only partition save / load  (no quantities)
# ---------------------------------------------------------------------------

def _write_mesh_partition(fname, rank, numprocs,
                          points, vertices, boundary,
                          ghost_recv_dict, full_send_dict,
                          tri_l2g, node_l2g,
                          number_of_full_triangles, number_of_full_nodes,
                          number_of_global_triangles, number_of_global_nodes,
                          ghost_layer_width, geo_ref):
    """Write one rank's mesh partition to a NetCDF4 file."""
    import netCDF4

    with netCDF4.Dataset(fname, 'w', format='NETCDF4') as nc:

        # --- global scalar attributes ---
        nc.rank = rank
        nc.numprocs = numprocs
        nc.number_of_full_triangles = number_of_full_triangles
        nc.number_of_full_nodes = number_of_full_nodes
        nc.number_of_global_triangles = number_of_global_triangles
        nc.number_of_global_nodes = number_of_global_nodes
        nc.ghost_layer_width = ghost_layer_width
        geo_ref.write_NetCDF(nc)

        Nnodes = len(points)
        Ntri   = len(vertices)

        nc.createDimension('node',  Nnodes)
        nc.createDimension('tri',   Ntri)
        nc.createDimension('two',   2)
        nc.createDimension('three', 3)

        # --- mesh arrays ---
        v = nc.createVariable('points', 'f8', ('node', 'two'))
        v[:] = points
        v = nc.createVariable('vertices', 'i4', ('tri', 'three'))
        v[:] = vertices
        v = nc.createVariable('tri_l2g', 'i4', ('tri',))
        v[:] = tri_l2g
        v = nc.createVariable('node_l2g', 'i4', ('node',))
        v[:] = node_l2g

        # --- boundary: encode {(tri, edge): tag} as three parallel arrays ---
        keys      = list(boundary.keys())
        bnd_tris  = num.array([k[0] for k in keys], dtype='i4')
        bnd_edges = num.array([k[1] for k in keys], dtype='i4')
        bnd_tags  = [boundary[k] for k in keys]
        Nbnd = len(bnd_tris)

        max_tag_len = max((len(t) for t in bnd_tags), default=1)
        nc.createDimension('bnd',     Nbnd)
        nc.createDimension('tag_len', max_tag_len)
        v = nc.createVariable('boundary_tri', 'i4', ('bnd',))
        v[:] = bnd_tris
        v = nc.createVariable('boundary_edge', 'i4', ('bnd',))
        v[:] = bnd_edges
        tag_var = nc.createVariable('boundary_tag', 'S1', ('bnd', 'tag_len'))
        for i, tag in enumerate(bnd_tags):
            padded = tag.ljust(max_tag_len, '\x00')
            tag_var[i, :] = num.frombuffer(padded.encode('ascii'), dtype='S1')

        # --- send/recv communication: CSR encoding ---
        # full_send_dict / ghost_recv_dict: {rank: [local_indices, global_indices]}
        for prefix, comm_dict in (('send', full_send_dict),
                                  ('recv', ghost_recv_dict)):
            ranks_sorted  = sorted(comm_dict.keys())
            local_arrays  = [comm_dict[r][0] for r in ranks_sorted]
            global_arrays = [comm_dict[r][1] for r in ranks_sorted]
            offsets = num.zeros(len(ranks_sorted) + 1, dtype='i4')
            for i, arr in enumerate(local_arrays):
                offsets[i + 1] = offsets[i] + len(arr)
            local_packed  = num.concatenate(local_arrays).astype('i4') \
                if local_arrays else num.array([], dtype='i4')
            global_packed = num.concatenate(global_arrays).astype('i4') \
                if global_arrays else num.array([], dtype='i4')

            nr = len(ranks_sorted)
            nc.createDimension(f'{prefix}_nranks',       nr)
            nc.createDimension(f'{prefix}_nranks_plus1', nr + 1)
            nc.createDimension(f'{prefix}_total',        len(local_packed))
            v = nc.createVariable(f'{prefix}_ranks', 'i4',
                                  (f'{prefix}_nranks',))
            v[:] = num.array(ranks_sorted, dtype='i4') if ranks_sorted \
                else num.array([], dtype='i4')
            v = nc.createVariable(f'{prefix}_offsets', 'i4',
                                  (f'{prefix}_nranks_plus1',))
            v[:] = offsets
            lv = nc.createVariable(f'{prefix}_local',  'i4',
                                   (f'{prefix}_total',))
            gv = nc.createVariable(f'{prefix}_global', 'i4',
                                   (f'{prefix}_total',))
            if len(local_packed):
                lv[:] = local_packed
                gv[:] = global_packed


def _release_mesh_submesh_rank(submesh, p):
    """Null out submesh arrays for rank *p* to allow GC before next rank."""
    for key in ('full_nodes', 'full_triangles', 'full_boundary',
                'ghost_nodes', 'ghost_triangles', 'ghost_commun',
                'ghost_boundary', 'ghost_layer_width', 'full_commun'):
        lst = submesh.get(key)
        if lst is not None and p < len(lst):
            lst[p] = None


def sequential_mesh_dump(domain, numprocs, partition_dir='.', name=None,
                         verbose=False, parameters=None):
    """Partition a domain mesh and write one NetCDF4 file per rank.

    Saves mesh topology and halo structure only — no quantities.
    Suitable as an offline preprocessing step before large parallel runs.
    After loading with :func:`sequential_mesh_load` the caller sets initial
    conditions via ``domain.set_quantity()`` before evolving.

    Files are written to ``<partition_dir>/<name>_mesh_P<numprocs>_<rank>.nc``.

    Parameters
    ----------
    domain : Domain or Basic_mesh
        Source mesh.  Quantities present on the domain are ignored.
    numprocs : int
        Number of partitions to create.
    partition_dir : str or path-like
        Output directory, created if it does not exist.
    name : str, optional
        Base name for output files.  Defaults to ``domain.get_name()`` when
        available, otherwise ``'mesh'``.
    verbose : bool
    parameters : dict, optional
        Forwarded to :func:`~anuga.parallel.distribute_mesh.partition_mesh`
        and :func:`~anuga.parallel.distribute_mesh.build_submesh`.
        Recognised keys include ``'partition_scheme'`` (``'metis'``,
        ``'morton'``, or ``'hilbert'``), ``'ghost_layer_width'``, and
        ``'cache_dir'``.
    """
    import gc
    import os

    from anuga.coordinate_transforms.geo_reference import Geo_reference

    if name is None:
        name = domain.get_name() if hasattr(domain, 'get_name') else 'mesh'

    os.makedirs(partition_dir, exist_ok=True)

    geo_ref = getattr(domain, 'geo_reference', Geo_reference())
    number_of_global_triangles = domain.number_of_triangles
    number_of_global_nodes     = domain.number_of_nodes

    if verbose:
        print(f'sequential_mesh_dump: partitioning {number_of_global_triangles}'
              f' triangles across {numprocs} ranks')

    new_mesh, triangles_per_proc, _, _s2p_map, p2s_map = partition_mesh(
        domain, numprocs, parameters=parameters, verbose=verbose)

    if verbose:
        print('sequential_mesh_dump: building submeshes')

    submesh = build_submesh(new_mesh, {}, triangles_per_proc,
                            parameters=parameters, verbose=verbose)

    if verbose:
        for p in range(numprocs):
            N = len(submesh['ghost_nodes'][p])
            M = len(submesh['ghost_triangles'][p])
            print(f'sequential_mesh_dump: rank {p}: '
                  f'{len(submesh["full_triangles"][p])} full triangles, '
                  f'{M} ghost triangles, {N} ghost nodes')

    for p in range(numprocs):
        points, vertices, boundary, _quantities, ghost_recv_dict, \
            full_send_dict, _tri_map, _node_map, tri_l2g, node_l2g, \
            ghost_layer_width = \
            extract_submesh(submesh, triangles_per_proc, p2s_map, p)

        number_of_full_triangles = len(submesh['full_triangles'][p])
        number_of_full_nodes     = len(submesh['full_nodes'][p])

        fname = os.path.join(partition_dir,
                             f'{name}_mesh_P{numprocs}_{p}.nc')
        if verbose:
            print(f'sequential_mesh_dump: writing {fname}')

        _write_mesh_partition(
            fname, p, numprocs,
            points, vertices, boundary,
            ghost_recv_dict, full_send_dict,
            tri_l2g, node_l2g,
            number_of_full_triangles, number_of_full_nodes,
            number_of_global_triangles, number_of_global_nodes,
            ghost_layer_width, geo_ref)

        _release_mesh_submesh_rank(submesh, p)
        del points, vertices, boundary, tri_l2g, node_l2g
        del ghost_recv_dict, full_send_dict
        gc.collect()

    gc.collect()


def sequential_mesh_load(name, partition_dir='.', verbose=False):
    """Load this rank's mesh partition and return a bare :class:`Parallel_domain`.

    Reads the NetCDF4 file written by :func:`sequential_mesh_dump` for the
    calling MPI rank.  No quantities are set; call ``domain.set_quantity()``
    and ``domain.set_boundary()`` before evolving.

    Parameters
    ----------
    name : str
        Base name passed to :func:`sequential_mesh_dump`.
    partition_dir : str or path-like
        Directory containing the partition files.
    verbose : bool

    Returns
    -------
    Parallel_domain
        Domain with mesh topology and halo structure initialised.
        All quantities are zero; boundary conditions are unset (``None``).
    """
    import os
    import netCDF4

    from anuga import myid, numprocs
    from anuga.coordinate_transforms.geo_reference import Geo_reference

    fname = os.path.join(partition_dir, f'{name}_mesh_P{numprocs}_{myid}.nc')
    if verbose:
        print(f'sequential_mesh_load: rank {myid} reading {fname}')

    with netCDF4.Dataset(fname, 'r') as nc:

        # --- scalar metadata ---
        number_of_full_triangles   = int(nc.number_of_full_triangles)
        number_of_full_nodes       = int(nc.number_of_full_nodes)
        number_of_global_triangles = int(nc.number_of_global_triangles)
        number_of_global_nodes     = int(nc.number_of_global_nodes)
        ghost_layer_width          = int(nc.ghost_layer_width)
        geo_ref = Geo_reference(NetCDFObject=nc)

        # --- mesh arrays ---
        points   = num.array(nc['points'][:])
        vertices = num.array(nc['vertices'][:])
        tri_l2g  = num.array(nc['tri_l2g'][:])
        node_l2g = num.array(nc['node_l2g'][:])

        # --- boundary dict ---
        bnd_tris  = num.array(nc['boundary_tri'][:])
        bnd_edges = num.array(nc['boundary_edge'][:])
        raw_tags  = nc['boundary_tag'][:]
        bnd_tags  = [
            b''.join(row).decode('ascii').rstrip('\x00')
            for row in raw_tags
        ]
        boundary = {
            (int(bnd_tris[i]), int(bnd_edges[i])): bnd_tags[i]
            for i in range(len(bnd_tris))
        }

        # --- communication dicts (CSR → dict) ---
        def _read_comm_dict(nc, prefix):
            ranks   = list(nc[f'{prefix}_ranks'][:].astype(int))
            offsets = nc[f'{prefix}_offsets'][:].astype(int)
            local_  = num.array(nc[f'{prefix}_local'][:])
            global_ = num.array(nc[f'{prefix}_global'][:])
            d = {}
            for i, r in enumerate(ranks):
                s, e = offsets[i], offsets[i + 1]
                d[r] = [local_[s:e], global_[s:e]]
            return d

        full_send_dict  = _read_comm_dict(nc, 'send')
        ghost_recv_dict = _read_comm_dict(nc, 'recv')

    domain = Parallel_domain(
        points, vertices, boundary,
        full_send_dict=full_send_dict,
        ghost_recv_dict=ghost_recv_dict,
        number_of_full_nodes=number_of_full_nodes,
        number_of_full_triangles=number_of_full_triangles,
        number_of_global_triangles=number_of_global_triangles,
        number_of_global_nodes=number_of_global_nodes,
        processor=myid,
        numproc=numprocs,
        s2p_map=None,
        p2s_map=None,
        tri_l2g=tri_l2g,
        node_l2g=node_l2g,
        ghost_layer_width=ghost_layer_width,
        geo_reference=geo_ref,
    )

    # Register all boundary tags with None BCs (caller replaces these).
    boundary_map = {tag: None for tag in set(boundary.values())}
    boundary_map['ghost'] = None
    domain.set_boundary(boundary_map)

    return domain


# ---------------------------------------------------------------------------
# Uniform mesh refinement
# ---------------------------------------------------------------------------

# ANUGA edge convention: edge j is opposite vertex j in triangle (v0, v1, v2).
# Each parent boundary edge j maps to child boundary edges of two children.
_EDGE_CHILD_INHERITANCE = {
    0: ((1, 0), (2, 0)),  # edge 0 (v1-v2) → child1 edge0, child2 edge0
    1: ((0, 1), (2, 1)),  # edge 1 (v0-v2) → child0 edge1, child2 edge1
    2: ((0, 2), (1, 2)),  # edge 2 (v0-v1) → child0 edge2, child1 edge2
}


def uniform_refine_domain(domain, verbose=False):
    """Uniformly refine a sequential domain; return a quantity-free Domain.

    Each triangle (v0, v1, v2) is replaced by four children using red
    refinement (connect edge midpoints):

    - child 0 = (v0,  m01, m02)  corner near v0
    - child 1 = (m01, v1,  m12)  corner near v1
    - child 2 = (m02, m12, v2)   corner near v2
    - child 3 = (m01, m12, m02)  central

    where m01, m12, m02 are the midpoints of edges (v0,v1), (v1,v2), (v0,v2).
    Boundary tags are propagated to the two child edges that inherit each
    parent boundary edge.  Quantities from *domain* are not copied.

    Parameters
    ----------
    domain : Domain
        Sequential domain to refine.
    verbose : bool

    Returns
    -------
    Domain
        Refined domain with 4x triangles; no quantities set.
    """
    from anuga import Domain

    triangles = domain.triangles    # (N, 3) node indices
    nodes     = domain.mesh.nodes   # (M, 2) coordinates
    boundary  = domain.boundary     # {(tri, edge): tag}

    N = len(triangles)
    M = len(nodes)

    if verbose:
        print(f'uniform_refine_domain: {N} triangles, {M} nodes '
              f'→ {4*N} triangles')

    # Assign a midpoint node ID to every unique edge.
    edge_to_mid = {}
    for v0, v1, v2 in triangles:
        for a, b in ((v0, v1), (v1, v2), (v0, v2)):
            e = (min(a, b), max(a, b))
            if e not in edge_to_mid:
                edge_to_mid[e] = M + len(edge_to_mid)

    M_new     = M + len(edge_to_mid)
    new_nodes = num.zeros((M_new, 2))
    new_nodes[:M] = nodes
    for (la, lb), mid_id in edge_to_mid.items():
        new_nodes[mid_id] = (nodes[la] + nodes[lb]) * 0.5

    new_triangles = num.zeros((4 * N, 3), dtype='i4')
    for t, (v0, v1, v2) in enumerate(triangles):
        m01 = edge_to_mid[(min(v0, v1), max(v0, v1))]
        m12 = edge_to_mid[(min(v1, v2), max(v1, v2))]
        m02 = edge_to_mid[(min(v0, v2), max(v0, v2))]
        new_triangles[4*t + 0] = (v0,  m01, m02)
        new_triangles[4*t + 1] = (m01, v1,  m12)
        new_triangles[4*t + 2] = (m02, m12, v2)
        new_triangles[4*t + 3] = (m01, m12, m02)

    new_boundary = {}
    for (tri, edge_j), tag in boundary.items():
        for child_idx, child_edge in _EDGE_CHILD_INHERITANCE[edge_j]:
            new_boundary[(4*tri + child_idx, child_edge)] = tag

    geo_ref = getattr(domain, 'geo_reference', None)
    kwargs  = {'geo_reference': geo_ref} if geo_ref is not None else {}
    return Domain(new_nodes, new_triangles, new_boundary, **kwargs)


def _read_partition_nc(fname):
    """Read one partition NetCDF4 file; return a plain data dict."""
    import netCDF4
    from anuga.coordinate_transforms.geo_reference import Geo_reference

    with netCDF4.Dataset(fname, 'r') as nc:
        rank     = int(nc.rank)
        numprocs = int(nc.numprocs)
        N_full   = int(nc.number_of_full_triangles)
        M_full   = int(nc.number_of_full_nodes)
        N_global = int(nc.number_of_global_triangles)
        M_global = int(nc.number_of_global_nodes)
        glw      = int(nc.ghost_layer_width)
        geo_ref  = Geo_reference(NetCDFObject=nc)

        points   = num.array(nc['points'][:])
        vertices = num.array(nc['vertices'][:], dtype='i4')
        tri_l2g  = num.array(nc['tri_l2g'][:],  dtype='i4')
        node_l2g = num.array(nc['node_l2g'][:], dtype='i4')

        bnd_tris  = num.array(nc['boundary_tri'][:])
        bnd_edges = num.array(nc['boundary_edge'][:])
        raw_tags  = nc['boundary_tag'][:]
        bnd_tags  = [b''.join(row).decode('ascii').rstrip('\x00')
                     for row in raw_tags]
        boundary  = {(int(bnd_tris[i]), int(bnd_edges[i])): bnd_tags[i]
                     for i in range(len(bnd_tris))}

        def _read_comm(nc, prefix):
            ranks_  = list(nc[f'{prefix}_ranks'][:].astype(int))
            offsets = nc[f'{prefix}_offsets'][:].astype(int)
            local_  = num.array(nc[f'{prefix}_local'][:],  dtype='i4')
            global_ = num.array(nc[f'{prefix}_global'][:], dtype='i4')
            d = {}
            for i, r in enumerate(ranks_):
                s, e = offsets[i], offsets[i + 1]
                d[r] = [local_[s:e], global_[s:e]]
            return d

        full_send_dict  = _read_comm(nc, 'send')
        ghost_recv_dict = _read_comm(nc, 'recv')

    return dict(
        rank=rank, numprocs=numprocs,
        points=points, vertices=vertices, tri_l2g=tri_l2g, node_l2g=node_l2g,
        boundary=boundary, ghost_recv_dict=ghost_recv_dict,
        full_send_dict=full_send_dict,
        number_of_full_triangles=N_full, number_of_full_nodes=M_full,
        number_of_global_triangles=N_global, number_of_global_nodes=M_global,
        ghost_layer_width=glw, geo_ref=geo_ref,
    )


def _refine_one_level(input_name, numprocs, output_name,
                       input_dir, output_dir, verbose=False):
    """Refine all partition files by one level (each triangle → 4 children).

    Two-pass algorithm:

    1. Read all partition files and collect every unique global edge.
       Determine midpoint ownership: a midpoint is owned by the lowest-rank
       partition that has the edge in one of its full triangles.

    2. For each partition, compute the refined mesh arrays and write a new
       partition file.

    Parameters
    ----------
    input_name, output_name : str
        Base names for input/output partition files.
    numprocs : int
    input_dir, output_dir : str or path-like
    verbose : bool
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 1: collect global edges and assign midpoint ownership.
    # ------------------------------------------------------------------
    all_global_edges = set()
    edge_full_rank   = {}   # global_edge → lowest rank with it in a full tri
    M_global_in      = None
    N_global_in      = None

    for p in range(numprocs):
        fname = os.path.join(input_dir,
                             f'{input_name}_mesh_P{numprocs}_{p}.nc')
        if verbose:
            print(f'_refine_one_level: pass 1 reading {fname}')
        data = _read_partition_nc(fname)

        if M_global_in is None:
            M_global_in = data['number_of_global_nodes']
            N_global_in = data['number_of_global_triangles']

        vertices_p = data['vertices']    # (N_local, 3) local IDs
        node_l2g_p = data['node_l2g']   # (M_local,)  global IDs
        N_full_p   = data['number_of_full_triangles']

        # Vectorised: compute global IDs for every triangle vertex.
        vg = node_l2g_p[vertices_p]      # (N_local, 3) global IDs

        for col_a, col_b in ((0, 1), (1, 2), (0, 2)):
            ea = num.minimum(vg[:, col_a], vg[:, col_b])
            eb = num.maximum(vg[:, col_a], vg[:, col_b])
            all_global_edges.update(
                zip(ea.tolist(), eb.tolist()))

        # Full-triangle edges → candidate ownership for rank p.
        vg_f = vg[:N_full_p]
        for col_a, col_b in ((0, 1), (1, 2), (0, 2)):
            ea = num.minimum(vg_f[:, col_a], vg_f[:, col_b])
            eb = num.maximum(vg_f[:, col_a], vg_f[:, col_b])
            for ga, gb in zip(ea.tolist(), eb.tolist()):
                ge = (ga, gb)
                if ge not in edge_full_rank or edge_full_rank[ge] > p:
                    edge_full_rank[ge] = p

    sorted_edges       = sorted(all_global_edges)
    edge_to_mid_global = {ge: M_global_in + i
                          for i, ge in enumerate(sorted_edges)}
    M_global_new = M_global_in + len(sorted_edges)
    N_global_new = 4 * N_global_in

    if verbose:
        print(f'_refine_one_level: global tri {N_global_in} → {N_global_new}, '
              f'nodes {M_global_in} → {M_global_new}')

    # ------------------------------------------------------------------
    # Pass 2: refine each partition.
    # ------------------------------------------------------------------
    for p in range(numprocs):
        fname_in = os.path.join(input_dir,
                                f'{input_name}_mesh_P{numprocs}_{p}.nc')
        if verbose:
            print(f'_refine_one_level: pass 2 refining rank {p}')
        data = _read_partition_nc(fname_in)

        points     = data['points']
        vertices   = data['vertices']
        tri_l2g    = data['tri_l2g']
        node_l2g   = data['node_l2g']
        boundary   = data['boundary']
        ghost_recv = data['ghost_recv_dict']
        full_send  = data['full_send_dict']
        N_full     = data['number_of_full_triangles']
        M_full     = data['number_of_full_nodes']
        glw        = data['ghost_layer_width']
        geo_ref    = data['geo_ref']

        N_local = len(vertices)
        M_local = len(points)
        M_ghost = M_local - M_full

        # Collect all unique local edges and their global counterparts.
        local_edges_map = {}   # (la, lb) → (ga, gb)
        for col_a, col_b in ((0, 1), (1, 2), (0, 2)):
            la_arr = num.minimum(vertices[:, col_a], vertices[:, col_b])
            lb_arr = num.maximum(vertices[:, col_a], vertices[:, col_b])
            ga_arr = num.minimum(node_l2g[la_arr], node_l2g[lb_arr])
            gb_arr = num.maximum(node_l2g[la_arr], node_l2g[lb_arr])
            for k in range(N_local):
                le = (int(la_arr[k]), int(lb_arr[k]))
                if le not in local_edges_map:
                    local_edges_map[le] = (int(ga_arr[k]), int(gb_arr[k]))

        # Classify: full (owned by this rank p) vs ghost.
        full_mid_list  = []
        ghost_mid_list = []
        for le, ge in local_edges_map.items():
            if edge_full_rank.get(ge, -1) == p:
                full_mid_list.append((le, ge))
            else:
                ghost_mid_list.append((le, ge))

        full_mid_list.sort(key=lambda x: edge_to_mid_global[x[1]])
        ghost_mid_list.sort(key=lambda x: edge_to_mid_global[x[1]])

        n_full_mids  = len(full_mid_list)
        n_ghost_mids = len(ghost_mid_list)
        M_full_new   = M_full + n_full_mids
        M_local_new  = M_local + n_full_mids + n_ghost_mids
        N_full_new   = 4 * N_full

        # Local midpoint IDs:
        #   0..M_full-1           original full nodes (unchanged)
        #   M_full..M_full_new-1  new full midpoints
        #   M_full_new..M_full_new+M_ghost-1  original ghost nodes (shifted)
        #   M_full_new+M_ghost..  ghost-only midpoints
        edge_to_local_mid = {}
        for k, (le, _ge) in enumerate(full_mid_list):
            edge_to_local_mid[le] = M_full + k
        for k, (le, _ge) in enumerate(ghost_mid_list):
            edge_to_local_mid[le] = M_full_new + M_ghost + k

        # New points array (coordinates).
        new_points = num.zeros((M_local_new, 2))
        new_points[:M_full] = points[:M_full]
        for k, ((la, lb), _ge) in enumerate(full_mid_list):
            new_points[M_full + k] = (points[la] + points[lb]) * 0.5
        new_points[M_full_new: M_full_new + M_ghost] = points[M_full:]
        for k, ((la, lb), _ge) in enumerate(ghost_mid_list):
            new_points[M_full_new + M_ghost + k] = \
                (points[la] + points[lb]) * 0.5

        # New node_l2g.
        new_node_l2g = num.zeros(M_local_new, dtype='i4')
        new_node_l2g[:M_full] = node_l2g[:M_full]
        for k, (_le, ge) in enumerate(full_mid_list):
            new_node_l2g[M_full + k] = edge_to_mid_global[ge]
        new_node_l2g[M_full_new: M_full_new + M_ghost] = node_l2g[M_full:]
        for k, (_le, ge) in enumerate(ghost_mid_list):
            new_node_l2g[M_full_new + M_ghost + k] = edge_to_mid_global[ge]

        # New vertices (4 × N_local triangles).
        new_vertices = num.zeros((4 * N_local, 3), dtype='i4')
        for t in range(N_local):
            v0 = int(vertices[t, 0])
            v1 = int(vertices[t, 1])
            v2 = int(vertices[t, 2])
            # Remap original node IDs: ghost nodes shift by n_full_mids.
            rv0 = v0 if v0 < M_full else v0 + n_full_mids
            rv1 = v1 if v1 < M_full else v1 + n_full_mids
            rv2 = v2 if v2 < M_full else v2 + n_full_mids
            m01 = edge_to_local_mid[(min(v0, v1), max(v0, v1))]
            m12 = edge_to_local_mid[(min(v1, v2), max(v1, v2))]
            m02 = edge_to_local_mid[(min(v0, v2), max(v0, v2))]
            new_vertices[4*t + 0] = (rv0, m01, m02)
            new_vertices[4*t + 1] = (m01, rv1, m12)
            new_vertices[4*t + 2] = (m02, m12, rv2)
            new_vertices[4*t + 3] = (m01, m12, m02)

        # New tri_l2g.
        new_tri_l2g = (
            4 * tri_l2g[:, None] + num.arange(4, dtype='i4')[None, :]
        ).flatten().astype('i4')

        # Propagate boundary tags: each parent boundary edge → two children.
        new_boundary = {}
        for (tri_loc, edge_j), tag in boundary.items():
            for child_idx, child_edge in _EDGE_CHILD_INHERITANCE[edge_j]:
                new_boundary[(4 * tri_loc + child_idx, child_edge)] = tag

        # Expand communication dicts: each ghost/send entry → 4 children.
        def _expand_comm(d):
            out = {}
            for rank_r, (la_arr, ga_arr) in d.items():
                la = num.asarray(la_arr, dtype='i4')
                ga = num.asarray(ga_arr, dtype='i4')
                c  = num.arange(4, dtype='i4')
                out[rank_r] = [
                    (4 * la[:, None] + c[None, :]).flatten(),
                    (4 * ga[:, None] + c[None, :]).flatten(),
                ]
            return out

        new_ghost_recv = _expand_comm(ghost_recv)
        new_full_send  = _expand_comm(full_send)

        fname_out = os.path.join(output_dir,
                                 f'{output_name}_mesh_P{numprocs}_{p}.nc')
        _write_mesh_partition(
            fname_out, p, numprocs,
            new_points, new_vertices, new_boundary,
            new_ghost_recv, new_full_send,
            new_tri_l2g, new_node_l2g,
            N_full_new, M_full_new,
            N_global_new, M_global_new,
            glw, geo_ref)

        if verbose:
            print(f'_refine_one_level: wrote {fname_out} '
                  f'({N_full_new} full tri, {M_full_new} full nodes)')


def sequential_mesh_refine(name, numprocs, levels=1, output_name=None,
                            partition_dir='.', output_dir=None, verbose=False):
    """Uniformly refine pre-computed partition files by one or more levels.

    Each refinement level replaces every triangle with four children using
    red refinement (connect edge midpoints).  After *levels* refinements the
    mesh has 4^levels times as many triangles.

    The output files can be loaded with :func:`sequential_mesh_load`.

    Parameters
    ----------
    name : str
        Base name of the input partition files (written by
        :func:`sequential_mesh_dump`).
    numprocs : int
        Number of partitions.
    levels : int
        Number of refinement levels.  Default is 1.
    output_name : str, optional
        Base name for output files.  Defaults to *name*.
    partition_dir : str
        Directory containing the input partition files.
    output_dir : str, optional
        Directory for output files.  Defaults to *partition_dir*.
    verbose : bool
    """
    import os
    import shutil
    import tempfile

    if output_name is None:
        output_name = name
    if output_dir is None:
        output_dir = partition_dir
    if levels <= 0:
        return

    os.makedirs(output_dir, exist_ok=True)

    if levels == 1:
        _refine_one_level(name, numprocs, output_name,
                          partition_dir, output_dir, verbose=verbose)
        return

    # Multi-level: pipe each level's output into the next via temp dirs.
    tmp_dirs     = []
    current_name = name
    current_dir  = partition_dir
    try:
        for level in range(levels):
            is_last = (level == levels - 1)
            out_name = output_name if is_last else 'refined'
            out_dir  = output_dir  if is_last else tempfile.mkdtemp()
            if not is_last:
                tmp_dirs.append(out_dir)
            if verbose:
                print(f'sequential_mesh_refine: level {level + 1} / {levels}')
            _refine_one_level(current_name, numprocs, out_name,
                              current_dir, out_dir, verbose=verbose)
            current_name = out_name
            current_dir  = out_dir
    finally:
        for d in tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)


def create_parallel_mesh(domain, numprocs, refinement_levels=0, name=None,
                          partition_dir='.', verbose=False, parameters=None):
    """Partition a mesh and optionally refine it offline.

    Combines :func:`sequential_mesh_dump` and :func:`sequential_mesh_refine`
    into a single call.  A coarse mesh is partitioned on a single process,
    then each partition is uniformly refined *refinement_levels* times.  The
    resulting files are loaded at run-time with :func:`sequential_mesh_load`.

    Parameters
    ----------
    domain : Domain or Basic_mesh
        Sequential domain or lightweight :class:`Basic_mesh` to partition.
        Quantities present on a Domain are not stored in the partition files.
    numprocs : int
        Number of partitions.
    refinement_levels : int
        Number of uniform refinement levels.  0 means no refinement (just
        partition and dump).
    name : str, optional
        Base name for the partition files.  Defaults to
        ``domain.get_name()`` when available, otherwise ``'mesh'``.
    partition_dir : str
        Output directory for the final partition files.
    verbose : bool
    parameters : dict, optional
        Forwarded to :func:`sequential_mesh_dump`.

    Returns
    -------
    str
        The *name* used for the partition files (pass to
        :func:`sequential_mesh_load`).
    """
    import os
    import tempfile

    if name is None:
        name = domain.get_name() if hasattr(domain, 'get_name') else 'mesh'

    os.makedirs(partition_dir, exist_ok=True)

    if refinement_levels == 0:
        sequential_mesh_dump(domain, numprocs,
                              partition_dir=partition_dir, name=name,
                              verbose=verbose, parameters=parameters)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            sequential_mesh_dump(domain, numprocs,
                                  partition_dir=tmpdir, name=name,
                                  verbose=verbose, parameters=parameters)
            sequential_mesh_refine(name, numprocs,
                                    levels=refinement_levels,
                                    output_name=name,
                                    partition_dir=tmpdir,
                                    output_dir=partition_dir,
                                    verbose=verbose)
    return name
