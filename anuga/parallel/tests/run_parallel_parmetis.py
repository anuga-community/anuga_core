"""
Demonstrate and smoke-test ParMETIS parallel mesh partitioning.

Two tests are performed:
  1. Direct API — calls parmetis_partition() collectively and validates the
     returned partition on rank 0.
  2. distribute() path — uses distribute(domain, parameters={'partition_scheme':
     'parmetis'}) to partition, distribute, and briefly evolve a domain.

Run:
  python  run_parallel_parmetis.py           # serial (n_procs=1 trivial path)
  mpiexec -np 2 python run_parallel_parmetis.py
  mpiexec -np 4 python run_parallel_parmetis.py
"""

import sys
import numpy as np

import anuga
from anuga import rectangular_cross_domain, Reflective_boundary
from anuga import distribute, myid, numprocs, finalize
from anuga.parallel.partitioning import parmetis_available, parmetis_partition
from anuga.utilities.parallel_abstraction import global_except_hook

sys.excepthook = global_except_hook

# ---------------------------------------------------------------------------
# Guard: exit cleanly if ParMETIS not installed
# ---------------------------------------------------------------------------
if not parmetis_available():
    if myid == 0:
        print("ParMETIS not available — skipping run_parallel_parmetis.py")
    finalize()
    sys.exit(0)

if myid == 0:
    print("ParMETIS available: True")
    print(f"Running with {numprocs} MPI rank(s)")

# ---------------------------------------------------------------------------
# Build a small test domain (same on every rank for rectangular meshes)
# ---------------------------------------------------------------------------
M, N = 20, 20
Lx, Ly = 10.0, 10.0

domain = rectangular_cross_domain(M, N, len1=Lx, len2=Ly)
domain.set_flow_algorithm('DE0')
domain.store = False
domain.set_quantity('elevation', 0.0)
domain.set_quantity('stage', lambda x, y: np.where(x < Lx / 2, 2.0, 1.0))

n_tri = domain.number_of_triangles

# ---------------------------------------------------------------------------
# Test 1: direct parmetis_partition() call
# ---------------------------------------------------------------------------
from mpi4py import MPI
comm = MPI.COMM_WORLD

# All ranks supply the full neighbours array (same for all on a rectangular mesh)
neighbours = np.ascontiguousarray(domain.neighbours, dtype=np.int64)

epart_order, tpp = parmetis_partition(neighbours, n_tri, numprocs, comm)

if myid == 0:
    assert epart_order is not None, "rank 0 must receive epart_order"
    assert tpp is not None,         "rank 0 must receive tpp"
    assert len(epart_order) == n_tri, \
        f"epart_order length {len(epart_order)} != n_tri {n_tri}"
    assert len(tpp) == numprocs, \
        f"tpp length {len(tpp)} != numprocs {numprocs}"
    assert tpp.sum() == n_tri, \
        f"tpp sum {tpp.sum()} != n_tri {n_tri}"
    assert np.all(tpp > 0), f"some rank has zero triangles: {tpp}"
    # epart_order must be a permutation of 0..n_tri-1
    assert np.array_equal(np.sort(epart_order), np.arange(n_tri)), \
        "epart_order is not a valid permutation"
    print(f"Test 1 PASSED — {n_tri} triangles partitioned into {numprocs} rank(s)")
    print(f"  triangles per rank: {tpp}")

comm.Barrier()

# ---------------------------------------------------------------------------
# Test 2: distribute() with partition_scheme='parmetis'
# ---------------------------------------------------------------------------
domain2 = rectangular_cross_domain(M, N, len1=Lx, len2=Ly)
domain2.set_flow_algorithm('DE0')
domain2.store = False
domain2.set_quantity('elevation', 0.0)
domain2.set_quantity('stage', lambda x, y: np.where(x < Lx / 2, 2.0, 1.0))

if numprocs > 1:
    domain2 = distribute(domain2, parameters={'partition_scheme': 'parmetis'})

Br = Reflective_boundary(domain2)
domain2.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

# Evolve for a few steps to confirm correctness
for t in domain2.evolve(yieldstep=0.05, finaltime=0.1):
    pass

if myid == 0:
    print("Test 2 PASSED — domain evolved to t=0.1 with parmetis partition")

finalize()
