"""Parallel Merimbula simulation using distribute_basic_mesh.

Demonstrates the mesh-first parallel workflow with a real unstructured mesh
read from a .tsh file.  Only rank 0 reads the file and builds a Basic_mesh;
all other ranks pass None.  After distribution each rank sets its own
initial conditions by interpolating the elevation data broadcast from rank 0.

Run with::

    mpiexec -np 4 python -u run_parallel_merimbula_basic_mesh.py

Produces per-rank SWW files merged into a single output file at the end.
"""

import os
import sys
import time

import numpy as np
from math import sin

import anuga
from anuga import (
    Reflective_boundary,
    Transmissive_n_momentum_zero_t_momentum_set_stage_boundary,
    myid, numprocs, finalize, barrier,
)
from anuga.abstract_2d_finite_volumes.basic_mesh import Basic_mesh
from anuga.abstract_2d_finite_volumes.pmesh2domain import pmesh_dict_to_tag_dict
from anuga.parallel.parallel_api import distribute_basic_mesh

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
mesh_filename = os.path.join(DATA_DIR, 'merimbula_10785_1.tsh')

# Stage pulse region (UTM coordinates relative to geo_reference origin)
x0 = 756000.0
x1 = 756500.0

yieldstep = 10
finaltime = 50

verbose = True

# ---------------------------------------------------------------------------
# Step 1: Rank 0 reads the mesh file and builds a Basic_mesh.
#         Other ranks do nothing at this stage.
# ---------------------------------------------------------------------------
t0 = time.time()

if myid == 0:
    from anuga.load_mesh.loadASCII import import_mesh_file

    if verbose:
        print('Reading mesh file:', mesh_filename)

    mesh_dict = import_mesh_file(mesh_filename)

    nodes       = mesh_dict['vertices']          # (M, 2) float
    triangles   = mesh_dict['triangles']         # (N, 3) int
    geo_ref     = mesh_dict['geo_reference']
    boundary    = pmesh_dict_to_tag_dict(mesh_dict)

    # Extract per-vertex elevation stored in the .tsh file.
    vertex_atts    = np.array(mesh_dict['vertex_attributes'])  # (M, n_atts)
    vertex_titles  = mesh_dict['vertex_attribute_titles']
    elev_idx       = vertex_titles.index('elevation')
    elevation_v    = vertex_atts[:, elev_idx].astype(float)    # (M,)

    bm = Basic_mesh(nodes, triangles,
                    boundary=boundary,
                    geo_reference=geo_ref)

    if verbose:
        print(f'Basic_mesh: {bm.number_of_triangles} triangles, '
              f'{bm.number_of_nodes} nodes')

    # Package global node coordinates + elevation for broadcast.
    bcast_data = {'nodes': nodes, 'elevation': elevation_v}
else:
    bm         = None
    bcast_data = None

t1 = time.time()
if myid == 0:
    print(f'Mesh read + Basic_mesh construction: {t1 - t0:.3f} s')

# ---------------------------------------------------------------------------
# Step 2: Broadcast elevation data so every rank can set quantities.
#
#         For large meshes (millions of nodes) this broadcast is still
#         O(M) -- consider using a SWW/DEM file and file_function instead
#         if rank-0 memory is the limiting constraint.
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    bcast_data = comm.bcast(bcast_data, root=0)
except ImportError:
    pass   # serial run: bcast_data already set on rank 0

global_nodes     = bcast_data['nodes']       # (M, 2)
global_elevation = bcast_data['elevation']   # (M,)

# Build a fast interpolator so each rank can evaluate elevation at any (x, y).
from scipy.interpolate import LinearNDInterpolator
elevation_func = LinearNDInterpolator(global_nodes, global_elevation)

# ---------------------------------------------------------------------------
# Step 3: Distribute the mesh to all ranks.
# ---------------------------------------------------------------------------
barrier()
if myid == 0 and verbose:
    print('DISTRIBUTING MESH')
    sys.stdout.flush()

domain = distribute_basic_mesh(bm, verbose=verbose)
domain.set_name('merimbula_basic_mesh')

t2 = time.time()
if myid == 0:
    print(f'distribute_basic_mesh: {t2 - t1:.3f} s')

# ---------------------------------------------------------------------------
# Step 4: Set quantities on each rank's local submesh.
# ---------------------------------------------------------------------------

# Elevation: interpolate from the global vertex data.
domain.set_quantity('elevation', elevation_func)

# Stage: flat water level of 1 m above local elevation, with a
# higher pulse (1 m extra) in the band x0 < x < x1.
def initial_stage(x, y):
    return 1.0 + 1.0 * ((x > x0) & (x < x1))

domain.set_quantity('stage', initial_stage)

domain.set_quantities_to_be_stored({
    'elevation':  1,
    'friction':   1,
    'stage':      2,
    'xmomentum':  2,
    'ymomentum':  2,
})

# ---------------------------------------------------------------------------
# Step 5: Boundary conditions.
# ---------------------------------------------------------------------------
Br  = Reflective_boundary(domain)
Bts = Transmissive_n_momentum_zero_t_momentum_set_stage_boundary(
          domain, lambda t: 10 * sin(t / 60))

domain.set_boundary({'exterior': Br, 'open': Bts})

# ---------------------------------------------------------------------------
# Step 6: Evolve.
# ---------------------------------------------------------------------------
if myid == 0 and verbose:
    print('EVOLVE')

barrier()
t3 = time.time()

for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
    if myid == 0:
        domain.print_timestepping_statistics()

evolve_time = time.time() - t3

barrier()

for p in range(numprocs):
    if myid == p:
        print(50 * '=')
        print(f'Rank {myid}')
        print(f'  Evolve time:        {evolve_time:.2f} s')
        print(f'  Communication:      {domain.communication_time:.2f} s')
        print(f'  Reduction comms:    {domain.communication_reduce_time:.2f} s')
        print(f'  Broadcast comms:    {domain.communication_broadcast_time:.2f} s')
        sys.stdout.flush()
    barrier()

# ---------------------------------------------------------------------------
# Step 7: Merge per-rank SWW files into one.
# ---------------------------------------------------------------------------
domain.sww_merge(delete_old=True)

if myid == 0:
    print(50 * '=')
    print(f'Number of triangles: {domain.number_of_global_triangles}')
    total = t2 - t0 + evolve_time
    print(f'Timings: read={t1-t0:.2f}s  dist={t2-t1:.2f}s  evolve={evolve_time:.2f}s')

finalize()
