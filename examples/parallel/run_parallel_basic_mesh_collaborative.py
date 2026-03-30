#########################################################
#
#  Example of running a simple parallel model using
#  distribute_basic_mesh_collaborative.
#
#  Like distribute_basic_mesh, only rank 0 builds the mesh
#  and quantities are set per-rank after distribution.
#  The collaborative variant uses shared memory within each
#  compute node so that the full topology is stored only once
#  per node rather than once per MPI rank.
#
#  To run in parallel on 4 processes:
#
#    mpiexec -np 4 python -u run_parallel_basic_mesh_collaborative.py
#
#########################################################

import time
import sys
import anuga

from anuga import Reflective_boundary
from anuga import myid, numprocs, finalize, barrier

from anuga.abstract_2d_finite_volumes.basic_mesh import rectangular_cross_basic_mesh
from anuga.parallel.parallel_api import distribute_basic_mesh_collaborative

import argparse

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
yieldstep = 0.005
finaltime = 0.015
sqrtN = 100
length = 2.0
width = 2.0

parser = argparse.ArgumentParser(
    description='Rectangular domain via distribute_basic_mesh_collaborative')
parser.add_argument('-ft', '--finaltime', type=float, default=finaltime,
                    help='final simulation time')
parser.add_argument('-ys', '--yieldstep', type=float, default=yieldstep,
                    help='yield step')
parser.add_argument('-sn', '--sqrtN', type=int, default=sqrtN,
                    help='grid resolution: sqrtN x sqrtN cells')
parser.add_argument('-gl', '--ghost_layer', type=int, default=2,
                    help='ghost layer width')
parser.add_argument('-ps', '--partition_scheme', type=str, default='metis',
                    help='partition scheme: metis, morton, or hilbert')
parser.add_argument('-sww', '--store_sww', action='store_true',
                    help='write SWW output files')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='verbose output')

args = parser.parse_args()

if myid == 0:
    print(args)

sqrtN = args.sqrtN
yieldstep = args.yieldstep
finaltime = args.finaltime
verbose = args.verbose
store_sww = args.store_sww

dist_params = {
    'ghost_layer_width': args.ghost_layer,
    'partition_scheme': args.partition_scheme,
}

# ---------------------------------------------------------------------------
# Step 1: Build mesh on rank 0 only
# ---------------------------------------------------------------------------
t0 = time.time()

if myid == 0:
    bm = rectangular_cross_basic_mesh(sqrtN, sqrtN,
                                      len1=length, len2=width,
                                      origin=(-length / 2, -width / 2))
    if verbose:
        print(f'Basic_mesh: {bm.number_of_triangles} triangles, '
              f'{bm.number_of_nodes} nodes')
else:
    bm = None

t1 = time.time()
if myid == 0:
    print(f'Basic_mesh creation time: {t1 - t0:.3f} s')
    print(f'Number of triangles: {bm.number_of_triangles}')

# ---------------------------------------------------------------------------
# Step 2: Distribute mesh to all ranks (shared-memory collaborative)
# ---------------------------------------------------------------------------
barrier()
if myid == 0:
    print('DISTRIBUTING MESH (collaborative / shared-memory)')
    sys.stdout.flush()

domain = distribute_basic_mesh_collaborative(
    bm, verbose=verbose, parameters=dist_params)
domain.set_name('sw_rectangle_basic_mesh_collab')

t2 = time.time()
if myid == 0:
    print(f'Distribute time: {t2 - t1:.3f} s')

# ---------------------------------------------------------------------------
# Step 3: Set quantities per-rank
# ---------------------------------------------------------------------------
domain.set_store(store_sww)
domain.set_flow_algorithm('DE0')

domain.set_quantity('elevation', lambda x, y: -1.0 - x)
domain.set_quantity('stage', 1.0)

# ---------------------------------------------------------------------------
# Step 4: Boundary conditions
# ---------------------------------------------------------------------------
R = Reflective_boundary(domain)
domain.set_boundary({'left': R, 'right': R, 'bottom': R, 'top': R})

# ---------------------------------------------------------------------------
# Step 5: Set a circular high-stage region
# ---------------------------------------------------------------------------
anuga.Set_stage(domain, center=(0.0, 0.0), radius=0.5, stage=2.0)()

barrier()

# ---------------------------------------------------------------------------
# Step 6: Evolve
# ---------------------------------------------------------------------------
t3 = time.time()

for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
    if myid == 0:
        domain.write_time()
        sys.stdout.flush()

evolve_time = time.time() - t3

if myid == 0:
    print(f'Evolve time: {evolve_time:.3f} s')

# ---------------------------------------------------------------------------
# Merge SWW output (one file per rank → single file)
# ---------------------------------------------------------------------------
if store_sww:
    domain.sww_merge(delete_old=True)

if myid == 0:
    print(80 * '=')
    print('np, ntri, mesh_time, dist_time, evolve_time')
    print(f'{numprocs}, {domain.number_of_global_triangles}, '
          f'{t1 - t0:.3f}, {t2 - t1:.3f}, {evolve_time:.3f}')

finalize()
