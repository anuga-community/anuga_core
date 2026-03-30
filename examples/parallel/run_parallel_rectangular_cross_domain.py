"""Parallel rectangular cross-mesh simulation using parallel_rectangular_cross_domain.

Demonstrates the direct parallel mesh workflow: every rank builds its own
local strip of the mesh simultaneously, without rank 0 ever holding the
full domain in memory.  This is simpler and more scalable than the
``distribute(domain)`` approach used in run_parallel_rectangular.py.

Physics
-------
Circular dam-break on a 2 m × 2 m flat domain.  A circular region of
radius 0.5 m centred on (0, 0) is initialised with stage = 2.0 m; the
rest of the domain has stage = 1.0 m.  All walls are reflective.

Run
---
Serial::

    python -u run_parallel_rectangular_cross_domain.py

Parallel (4 ranks)::

    mpiexec -np 4 python -u run_parallel_rectangular_cross_domain.py

Options::

    -n  / --ncells    cells per side (default 100)
    -ys / --yieldstep output interval in seconds (default 0.005)
    -ft / --finaltime stop time in seconds (default 0.05)
    -v  / --verbose   print per-rank statistics after evolve
    --no-sww          skip writing SWW output (faster benchmarking)
"""

import argparse
import sys
import time

import anuga
from anuga import (
    Reflective_boundary,
    myid, numprocs, finalize, barrier,
)
from anuga.parallel.parallel_meshes import parallel_rectangular_cross_domain


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Parallel rectangular cross-domain dam-break example')
parser.add_argument('-n',   '--ncells',    type=int,   default=100)
parser.add_argument('-ys',  '--yieldstep', type=float, default=0.005)
parser.add_argument('-ft',  '--finaltime', type=float, default=0.05)
parser.add_argument('-v',   '--verbose',   action='store_true')
parser.add_argument('--no-sww',            action='store_true',
                    dest='no_sww')
args = parser.parse_args()

N          = args.ncells
yieldstep  = args.yieldstep
finaltime  = args.finaltime
verbose    = args.verbose
store_sww  = not args.no_sww

length = 2.0
width  = 2.0

if myid == 0:
    print()
    print('Parallel rectangular cross-domain example')
    print(f'  Ranks        : {numprocs}')
    print(f'  Grid         : {N} x {N}  ({4*N*N} triangles global)')
    print(f'  Domain       : {length} m x {width} m')
    print(f'  Yieldstep    : {yieldstep} s')
    print(f'  Finaltime    : {finaltime} s')
    print(f'  Store SWW    : {store_sww}')
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 1: Every rank builds its own local mesh strip directly.
#         No rank-0 bottleneck — all ranks work simultaneously.
# ---------------------------------------------------------------------------
barrier()
t_build = time.time()

domain = parallel_rectangular_cross_domain(
    N, N,
    len1_g=length,
    len2_g=width,
    origin_g=(-length / 2, -width / 2),
)

domain.set_name('rectangular_cross_parallel')
domain.set_store(store_sww)
domain.set_flow_algorithm('DE0')

t_build = time.time() - t_build
if myid == 0:
    print(f'  Mesh build   : {t_build:.3f} s  '
          f'(all ranks simultaneously)')
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Step 2: Initial conditions — every rank sets quantities on its own strip.
# ---------------------------------------------------------------------------
domain.set_quantity('elevation', 0.0)
domain.set_quantity('friction',  0.016)

# Circular dam: stage = 2.0 inside radius 0.5, else 1.0
domain.set_quantity('stage',
    lambda x, y: 2.0 * ((x**2 + y**2) < 0.5**2)
               + 1.0 * ((x**2 + y**2) >= 0.5**2))

# ---------------------------------------------------------------------------
# Step 3: Boundary conditions.
# ---------------------------------------------------------------------------
Br = Reflective_boundary(domain)
domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

# ---------------------------------------------------------------------------
# Step 4: Evolve.
# ---------------------------------------------------------------------------
if myid == 0:
    print()
    print('Evolving ...')
    sys.stdout.flush()

barrier()
t_evolve = time.time()

for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
    if myid == 0:
        domain.print_timestepping_statistics()

t_evolve = time.time() - t_evolve

barrier()

# ---------------------------------------------------------------------------
# Step 5: Per-rank timing summary.
# ---------------------------------------------------------------------------
if verbose:
    for p in range(numprocs):
        if myid == p:
            print(50 * '-')
            print(f'Rank {myid}/{numprocs}')
            print(f'  Evolve time      : {t_evolve:.3f} s')
            print(f'  Communication    : {domain.communication_time:.3f} s')
            print(f'  Reduce comms     : {domain.communication_reduce_time:.3f} s')
            print(f'  Broadcast comms  : {domain.communication_broadcast_time:.3f} s')
            sys.stdout.flush()
        barrier()

# ---------------------------------------------------------------------------
# Step 6: Merge per-rank SWW files.
# ---------------------------------------------------------------------------
if store_sww:
    domain.sww_merge(delete_old=True)

if myid == 0:
    print()
    print(50 * '=')
    print(f'Global triangles : {domain.number_of_global_triangles}')
    print(f'Build time       : {t_build:.3f} s')
    print(f'Evolve time      : {t_evolve:.3f} s')
    print()

finalize()
