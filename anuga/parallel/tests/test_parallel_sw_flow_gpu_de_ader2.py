"""
Multi-rank GPU halo-exchange test for DE_ader2 flow algorithm.

Verifies that multiprocessor_mode=2 (GPU ADER-2 step, GPU internal MPI ghost
exchange) produces results matching multiprocessor_mode=1 (CPU ADER-2 step,
Python MPI ghost exchange) when run in parallel.

The GPU ADER-2 path (mode=2) uses a two-flux-call scheme: a CFL flux call from
Q^n to determine the timestep, then the C-K predictor using that timestep, then
a second flux call from Q^{n+1/2}.  The Python ADER-2 path (mode=1) uses a
single-flux-call scheme: the C-K predictor uses _ader2_prev_dt from the previous
step.  Both are valid ADER-2 implementations but they differ by O(dt^3)
truncation terms per step, accumulating to ~O(dt^2) ≈ 1e-3 over the run.
The test therefore uses a relaxed tolerance (atol=1e-2) rather than machine zero.

Run via pytest (auto-marked slow):

    pytest anuga/parallel/tests/test_parallel_sw_flow_gpu_de_ader2.py

Or directly via MPI:

    mpiexec -np 4 python -m mpi4py anuga/parallel/tests/test_parallel_sw_flow_gpu_de_ader2.py
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


def gpu_available():
    try:
        from anuga.shallow_water.sw_domain_gpu_ext import init_gpu_domain  # noqa: F401
        return True
    except ImportError:
        return False


M = 29
N = 29
YIELDSTEP = 0.25
FINALTIME = 1.0
GAUGE_POINTS = [[0.4, 0.5], [0.6, 0.5], [0.8, 0.5], [0.9, 0.5]]
# mode=1 uses a single-flux-call scheme with _ader2_prev_dt from the previous
# step; mode=2 uses a two-flux-call scheme with the current CFL dt.  Both are
# valid ADER-2 implementations but differ by O(dt^3) terms per step, giving
# ~O(dt^2) accumulated error over the run (~1e-3 for dt≈0.05 over 1 s).
ATOL = 1e-2


def topography(x, y):
    return -x / 2


def run_simulation(mode, verbose=False):
    """Run a parallel DE_ader2 simulation with the given multiprocessor_mode.

    Parameters
    ----------
    mode : int
        1 = CPU ADER-2 step + Python MPI ghost exchange
        2 = GPU ADER-2 step + C-level MPI ghost exchange
    """
    domain = rectangular_cross_domain(M, N)
    domain.set_quantity('elevation', topography)
    domain.set_quantity('friction', 0.0)
    domain.set_quantity('stage', expression='elevation')

    domain = distribute(domain, verbose=False)

    domain.set_name(f'gpu_de_ader2_mode{mode}')
    domain.set_datadir(tempfile.mkdtemp())
    domain.set_flow_algorithm('DE_ader2')
    domain.set_quantities_to_be_stored(None)

    Br = Reflective_boundary(domain)
    Bd = Dirichlet_boundary([-0.2, 0., 0.])
    domain.set_boundary({'left': Br, 'right': Bd, 'top': Br, 'bottom': Br})

    domain.set_multiprocessor_mode(mode)

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


def compare_modes(G1, ids1, G2, ids2, label=''):
    for i, (tid1, tid2) in enumerate(zip(ids1, ids2)):
        if tid1 > -1 and tid2 > -1:
            a1 = num.array(G1[i])
            a2 = num.array(G2[i])
            if not num.allclose(a1, a2, atol=ATOL):
                max_diff = num.max(num.abs(a1 - a2))
                raise AssertionError(
                    f'[rank {myid}] {label} gauge {i}: '
                    f'mode=1 vs mode=2 max diff = {max_diff:.2e} (atol={ATOL})'
                )


@pytest.mark.skipif(not gpu_available(),
                    reason='GPU OpenMP extension not available')
@pytest.mark.skipif('mpi4py' not in sys.modules,
                    reason='requires mpi4py')
class Test_parallel_sw_gpu_de_ader2(unittest.TestCase):

    def test_gpu_de_ader2_2ranks(self):
        """2-rank DE_ader2: GPU halo exchange matches Python halo exchange."""
        cmd = anuga.mpicmd(os.path.abspath(__file__), numprocs=2)
        assert os.system(cmd) == 0

    def test_gpu_de_ader2_4ranks(self):
        """4-rank DE_ader2: GPU halo exchange matches Python halo exchange."""
        cmd = anuga.mpicmd(os.path.abspath(__file__), numprocs=4)
        assert os.system(cmd) == 0


def assert_(condition, msg='Assertion Failed'):
    if not condition:
        raise AssertionError(msg)


if __name__ == '__main__':
    if numprocs == 1:
        runner = unittest.TextTestRunner()
        suite = unittest.TestLoader().loadTestsFromTestCase(
            Test_parallel_sw_gpu_de_ader2)
        runner.run(suite)
    else:
        from anuga.utilities.parallel_abstraction import global_except_hook
        sys.excepthook = global_except_hook

        if myid == 0:
            print(f'\n=== GPU halo exchange test DE_ader2: {numprocs} ranks ===')

        barrier()
        if myid == 0:
            print('  Running mode=1 (CPU ADER-2, Python MPI) ...')
        G1, ids1 = run_simulation(mode=1, verbose=False)
        barrier()

        if myid == 0:
            print('  Running mode=2 (GPU ADER-2, GPU/C MPI) ...')
        G2, ids2 = run_simulation(mode=2, verbose=False)
        barrier()

        compare_modes(G1, ids1, G2, ids2, label=f'np={numprocs}')

        if myid == 0:
            print(f'  PASS: mode=1 and mode=2 agree (atol={ATOL})')

        finalize()
