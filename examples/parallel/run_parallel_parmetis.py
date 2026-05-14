"""
Example: parallel simulation using ParMETIS graph partitioning.

ParMETIS minimises edge cuts across MPI ranks, which can reduce inter-rank
communication compared to space-filling-curve schemes (morton, hilbert) or
METIS.  It requires all ranks to participate in a collective partitioning
call before the domain is distributed.

Usage
-----
  python  run_parallel_parmetis.py          # serial (trivial 1-rank path)
  mpiexec -np 2 python run_parallel_parmetis.py
  mpiexec -np 4 python run_parallel_parmetis.py
  mpiexec -np 4 python run_parallel_parmetis.py --sqrtN 50 --finaltime 0.05

To compare partition schemes back-to-back:
  mpiexec -np 4 python run_parallel_parmetis.py --partition_scheme parmetis
  mpiexec -np 4 python run_parallel_rectangular.py -ps metis
"""

import sys
import time
import argparse
import numpy as np

import anuga
from anuga import rectangular_cross_domain, Reflective_boundary
from anuga import distribute, myid, numprocs, finalize, barrier
from anuga.parallel.partitioning import parmetis_available
from anuga.utilities.parallel_abstraction import global_except_hook

sys.excepthook = global_except_hook

# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Rectangular dam-break using ParMETIS partitioning')
parser.add_argument('--sqrtN', type=int, default=30,
                    help='Grid size: sqrtN x sqrtN cells (default 30)')
parser.add_argument('--yieldstep', type=float, default=0.02)
parser.add_argument('--finaltime', type=float, default=0.1)
parser.add_argument('--ghost_layer', type=int, default=2)
parser.add_argument('--partition_scheme', type=str, default='parmetis',
                    choices=['parmetis', 'metis', 'morton', 'hilbert'],
                    help='Partition scheme (default: parmetis)')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

if myid == 0:
    print(f"{'='*60}")
    print(f"Partition scheme : {args.partition_scheme}")
    print(f"MPI ranks        : {numprocs}")
    print(f"Grid             : {args.sqrtN} x {args.sqrtN}")
    if args.partition_scheme == 'parmetis':
        print(f"ParMETIS available: {parmetis_available()}")
    print(f"{'='*60}")

if args.partition_scheme == 'parmetis' and not parmetis_available():
    if myid == 0:
        print("ParMETIS is not installed. Install it with:")
        print("  conda install -c conda-forge parmetis")
        print("Falling back to metis.")
    args.partition_scheme = 'metis'

dist_params = {
    'partition_scheme': args.partition_scheme,
    'ghost_layer_width': args.ghost_layer,
}

# ---------------------------------------------------------------------------
# Build domain on rank 0 only
# ---------------------------------------------------------------------------
t0 = time.time()

if myid == 0:
    domain = rectangular_cross_domain(
        args.sqrtN, args.sqrtN,
        len1=10.0, len2=10.0,
        verbose=args.verbose)

    domain.set_flow_algorithm('DE0')
    domain.set_store(False)

    # Initial condition: dam-break (left half higher water level)
    domain.set_quantity('elevation', 0.0)
    domain.set_quantity('stage', lambda x, y: np.where(x < 5.0, 2.0, 1.0))
    domain.set_quantity('friction', 0.01)

    n_tri = domain.number_of_triangles
    print(f"Domain created: {n_tri} triangles  ({time.time()-t0:.2f} s)")
else:
    domain = None

barrier()

# ---------------------------------------------------------------------------
# Distribute with chosen partition scheme (parmetis is collective)
# ---------------------------------------------------------------------------
t1 = time.time()

domain = distribute(domain, verbose=args.verbose, parameters=dist_params)

barrier()
t2 = time.time()

if myid == 0:
    print(f"Distributed ({args.partition_scheme}): {t2-t1:.2f} s")

# ---------------------------------------------------------------------------
# Boundaries and evolve
# ---------------------------------------------------------------------------
Br = Reflective_boundary(domain)
domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

t3 = time.time()
for t in domain.evolve(yieldstep=args.yieldstep, finaltime=args.finaltime):
    if myid == 0 and args.verbose:
        domain.write_time()

barrier()
evolve_time = time.time() - t3

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
if myid == 0:
    print(f"Evolved to t={args.finaltime:.3f}:  {evolve_time:.2f} s")

# Gather per-rank triangle counts for balance report
from mpi4py import MPI
n_full = domain.number_of_full_triangles
all_counts = MPI.COMM_WORLD.gather(n_full, root=0)

if myid == 0:
    all_counts = np.array(all_counts)
    imbalance = all_counts.max() / all_counts.mean()
    print(f"\nPartition balance ({args.partition_scheme}):")
    print(f"  Triangles per rank : {all_counts}")
    print(f"  Imbalance ratio    : {imbalance:.3f}  (1.0 = perfect)")
    print(f"{'='*60}")

finalize()
