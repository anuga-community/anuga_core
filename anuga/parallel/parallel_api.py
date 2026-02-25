"""Trying to lump parallel stuff into simpler interface


"""



from builtins import range
import numpy as num

# The abstract Python-MPI interface
from anuga.utilities.parallel_abstraction import size, rank, get_processor_name
from anuga.utilities.parallel_abstraction import finalize, send, receive, reduce
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
    """ Distribute the domain to all processes

    parameters allows user to change size of ghost layer
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

        for p in range(1, numprocs):

            tostore = partition.extract_submesh(p)

            send(tostore,p)

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

    #PETE: s2p_map (maps serial domain triangles to parallel domain triangles)
    #      sp2_map (maps parallel domain triangles to domain triangles)


    #new_mesh = Mesh(new_nodes, new_triangles, new_boundary)

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