"""
Multi-rank GPU halo-exchange test.

Verifies that multiprocessor_mode=2 (C-level MPI ghost exchange via
gpu_exchange_ghosts) produces results matching multiprocessor_mode=1
(Python MPI ghost exchange) when run in parallel.

This exercises the key correctness path: in mode=2 `update_ghosts()` returns
early and lets the C-level MPI calls handle halo exchange.  Any bug there
would cause the two modes to diverge.

mode=1 uses the OpenMP C extension kernels; mode=2 uses the GPU C kernels.
Both implement the same algorithm but with different code paths, so small
floating-point differences (< 1e-6) are expected and acceptable.
Ghost exchange errors produce O(1) differences and are easily caught.

Both modes call the same core_kernels.c functions for computation; the only
difference is ghost exchange (Python MPI vs C MPI).  After a correct build
the results should be bit-for-bit identical, so ATOL=1e-12 is used.  Ghost
exchange errors produce O(0.1-1.0) differences and are easily caught.
All ranks vote on the result via MPI allreduce before raising so that no
rank is left waiting.

Runs as pytest (launches mpirun subprocess, auto-marked slow because it lives
in anuga/parallel/tests/):

    pytest anuga/parallel/tests/test_parallel_sw_flow_gpu.py

Or directly via MPI (2 or 4 ranks):

    mpiexec -np 4 python -m mpi4py anuga/parallel/tests/test_parallel_sw_flow_gpu.py
"""

import os
import sys
import tempfile
import unittest

import numpy as num
import pytest

import anuga
from anuga import (Dirichlet_boundary, Reflective_boundary,
                   distribute, rectangular_cross_domain)
from anuga import myid, numprocs, barrier, finalize


# ---------------------------------------------------------------------------
# Check GPU extension is available (CPU_ONLY_MODE counts as available)
# ---------------------------------------------------------------------------

def gpu_available():
    try:
        from anuga.shallow_water.sw_domain_gpu_ext import init_gpu_domain  # noqa: F401
        return True
    except ImportError:
        return False


def gpu_has_mpi():
    try:
        from anuga.shallow_water.sw_domain_gpu_ext import gpu_has_mpi as _gpu_has_mpi
        return _gpu_has_mpi()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

M = 29
N = 29
YIELDSTEP = 0.25
FINALTIME = 1.0
GAUGE_POINTS = [[0.4, 0.5], [0.6, 0.5], [0.8, 0.5], [0.9, 0.5]]
ATOL = 1e-12


def topography(x, y):
    return -x / 2


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(mode, verbose=False):
    """Run a parallel DE0 (Euler) simulation with the given multiprocessor_mode.

    Parameters
    ----------
    mode : int
        1 = Python Euler loop + Python MPI ghost exchange (OpenMP kernels)
        2 = Python Euler loop + C-level MPI ghost exchange (GPU kernels)

    Returns
    -------
    gauge_values : list of lists
        Time-series at GAUGE_POINTS for triangles owned by this rank.
        Entry is [] for points not owned by this rank.
    tri_ids : list of int
        Local triangle index for each gauge point (-1 / -2 = not on this rank).
    """
    domain = rectangular_cross_domain(M, N)
    domain.set_quantity('elevation', topography)
    domain.set_quantity('friction', 0.0)
    domain.set_quantity('stage', expression='elevation')

    domain = distribute(domain, verbose=False)

    tmpdir = tempfile.mkdtemp()
    domain.set_name(f'gpu_halo_mode{mode}')
    domain.set_datadir(tmpdir)
    domain.set_quantities_to_be_stored(None)

    # Boundaries must be set BEFORE set_multiprocessor_mode(2)
    Br = Reflective_boundary(domain)
    Bd = Dirichlet_boundary([-0.2, 0., 0.])
    domain.set_boundary({'left': Br, 'right': Bd, 'top': Br, 'bottom': Br})

    domain.set_multiprocessor_mode(mode)

    # Find which gauge triangles this rank owns
    tri_ids = []
    for point in GAUGE_POINTS:
        try:
            k = domain.get_triangle_containing_point(point)
            tri_ids.append(k if domain.tri_full_flag[k] == 1 else -1)
        except Exception:
            tri_ids.append(-2)

    gauge_values = [[] for _ in GAUGE_POINTS]
    stage = domain.get_quantity('stage')

    for _ in domain.evolve(yieldstep=YIELDSTEP, finaltime=FINALTIME):
        for i, tid in enumerate(tri_ids):
            if tid > -1:
                gauge_values[i].append(float(stage.centroid_values[tid]))

    return gauge_values, tri_ids


# ---------------------------------------------------------------------------
# Comparison helper (runs on every rank)
# ---------------------------------------------------------------------------

def compare_modes(G1, ids1, G2, ids2, label=''):
    """Assert mode=1 and mode=2 gauge time-series match across all ranks.

    Uses MPI allreduce to find the global maximum difference so that all
    ranks raise (or pass) together — prevents one rank from hanging in a
    collective while another has already called MPI_Abort.
    """
    local_max_diff = 0.0
    worst_gauge = -1
    for i, (tid1, tid2) in enumerate(zip(ids1, ids2)):
        if tid1 > -1 and tid2 > -1:
            a1 = num.array(G1[i])
            a2 = num.array(G2[i])
            diff = float(num.max(num.abs(a1 - a2)))
            if diff > local_max_diff:
                local_max_diff = diff
                worst_gauge = i

    # Allreduce: every rank gets the global maximum difference
    try:
        from mpi4py import MPI
        global_max_diff = MPI.COMM_WORLD.allreduce(local_max_diff, op=MPI.MAX)
    except ImportError:
        global_max_diff = local_max_diff

    if global_max_diff > ATOL:
        raise AssertionError(
            f'[rank {myid}] {label}: '
            f'mode=1 vs mode=2 global max diff = {global_max_diff:.2e} '
            f'(atol={ATOL}, local worst gauge={worst_gauge})'
        )


# ---------------------------------------------------------------------------
# Pytest class — spawns MPI subprocess for each rank count
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not gpu_available(),
                    reason='GPU extension not available')
@pytest.mark.skipif(not gpu_has_mpi(),
                    reason='GPU extension built without C MPI (mpi.h not found at build time)')
@pytest.mark.skipif('mpi4py' not in sys.modules,
                    reason='requires mpi4py')
class Test_parallel_sw_gpu_halo(unittest.TestCase):

    def test_gpu_halo_2ranks(self):
        """2-rank: GPU halo exchange matches Python halo exchange."""
        cmd = anuga.mpicmd(os.path.abspath(__file__), numprocs=2)
        assert os.system(cmd) == 0

    def test_gpu_halo_4ranks(self):
        """4-rank: GPU halo exchange matches Python halo exchange."""
        cmd = anuga.mpicmd(os.path.abspath(__file__), numprocs=4)
        assert os.system(cmd) == 0


# ---------------------------------------------------------------------------
# __main__ entry point (executed by each MPI rank)
# ---------------------------------------------------------------------------

def assert_(condition, msg='Assertion Failed'):
    if not condition:
        raise AssertionError(msg)


if __name__ == '__main__':
    if numprocs == 1:
        # Single-process: run as a normal unittest (no MPI needed)
        runner = unittest.TextTestRunner()
        suite = unittest.TestLoader().loadTestsFromTestCase(
            Test_parallel_sw_gpu_halo)
        runner.run(suite)
    else:
        from anuga.utilities.parallel_abstraction import global_except_hook
        sys.excepthook = global_except_hook

        if myid == 0:
            print(f'\n=== GPU halo exchange test: {numprocs} ranks ===')

        # --- mode=1: Python RK2 + Python MPI ghost exchange ---
        barrier()
        if myid == 0:
            print('  Running mode=1 (Python RK2, Python MPI) ...')
        G1, ids1 = run_simulation(mode=1, verbose=False)
        barrier()

        # --- mode=2: C RK2 + C-level MPI ghost exchange ---
        if myid == 0:
            print('  Running mode=2 (C RK2, GPU/C MPI) ...')
        G2, ids2 = run_simulation(mode=2, verbose=False)
        barrier()

        # --- Compare on every rank ---
        compare_modes(G1, ids1, G2, ids2,
                      label=f'np={numprocs}')

        if myid == 0:
            print(f'  PASS: mode=1 and mode=2 agree (atol={ATOL})')

        finalize()
