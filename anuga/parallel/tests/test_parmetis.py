"""Tests for ParMETIS partitioning support.

Serial tests (no MPI needed):
  - parmetis_available() returns a bool
  - neighbours_to_csr() CSR conversion (several mesh topologies)
  - parmetis_partition() with n_procs=1 (trivial path, no ParMETIS call)

MPI smoke tests (marked slow, spawn mpiexec):
  - run_parallel_parmetis.py exits 0 on 2 and 4 ranks

Run serially:
  pytest anuga/parallel/tests/test_parmetis.py

Run in parallel (handled by the MPI smoke tests):
  mpiexec -np 4 python anuga/parallel/tests/run_parallel_parmetis.py
"""

import os
import sys
import subprocess
import unittest

import numpy as np
import pytest

try:
    import mpi4py
    MPI4PY_AVAILABLE = True
except ImportError:
    MPI4PY_AVAILABLE = False

from anuga.parallel.partitioning import parmetis_available

PARMETIS_AVAILABLE = parmetis_available()

path = os.path.dirname(os.path.abspath(__file__))
run_script = os.path.join(path, 'run_parallel_parmetis.py')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain_neighbours(n):
    """Build a linear chain of n triangles.

    Triangle k is adjacent to triangle k-1 (edge 0) and k+1 (edge 2);
    edge 1 is always a boundary edge.
    """
    nb = np.full((n, 3), -1, dtype=np.int64)
    for k in range(n):
        if k > 0:
            nb[k, 0] = k - 1
        if k < n - 1:
            nb[k, 2] = k + 1
    return nb


def _ring_neighbours(n):
    """Build a ring of n triangles (each has exactly 2 interior neighbours)."""
    nb = np.full((n, 3), -1, dtype=np.int64)
    for k in range(n):
        nb[k, 0] = (k - 1) % n
        nb[k, 2] = (k + 1) % n
    return nb


# ---------------------------------------------------------------------------
# parmetis_available()
# ---------------------------------------------------------------------------

class TestParmetisAvailable(unittest.TestCase):

    def test_returns_bool(self):
        result = parmetis_available()
        self.assertIsInstance(result, bool)


# ---------------------------------------------------------------------------
# neighbours_to_csr()
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PARMETIS_AVAILABLE, reason='ParMETIS not installed')
class TestNeighboursToCsr(unittest.TestCase):

    def setUp(self):
        from anuga.parallel.parmetis_ext import neighbours_to_csr
        self.neighbours_to_csr = neighbours_to_csr

    def _csr(self, nb, start, end):
        return self.neighbours_to_csr(
            np.ascontiguousarray(nb, dtype=np.int64), start, end)

    # --- output shape / dtype ---

    def test_xadj_length(self):
        nb = _chain_neighbours(6)
        xadj, adjncy = self._csr(nb, 0, 6)
        self.assertEqual(len(xadj), 7)  # n_local + 1

    def test_output_dtype_int32(self):
        nb = _chain_neighbours(4)
        xadj, adjncy = self._csr(nb, 0, 4)
        self.assertEqual(xadj.dtype, np.int32)
        self.assertEqual(adjncy.dtype, np.int32)

    def test_xadj_monotone(self):
        nb = _chain_neighbours(8)
        xadj, _ = self._csr(nb, 0, 8)
        self.assertTrue(np.all(np.diff(xadj) >= 0))

    # --- boundary edges excluded ---

    def test_all_boundary_gives_empty_adjncy(self):
        nb = np.full((3, 3), -1, dtype=np.int64)
        xadj, adjncy = self._csr(nb, 0, 3)
        self.assertEqual(len(adjncy), 0)
        self.assertTrue(np.all(xadj == 0))

    def test_boundary_edges_not_in_adjncy(self):
        nb = _chain_neighbours(5)
        xadj, adjncy = self._csr(nb, 0, 5)
        # All entries must be valid global triangle ids (0..4)
        self.assertTrue(np.all(adjncy >= 0))
        self.assertTrue(np.all(adjncy < 5))

    # --- chain mesh: full block ---

    def test_chain_full_block_xadj(self):
        # chain: tri 0 has 1 neighbour, tris 1..n-2 have 2, tri n-1 has 1
        n = 5
        nb = _chain_neighbours(n)
        xadj, adjncy = self._csr(nb, 0, n)
        expected_xadj = np.array([0, 1, 3, 5, 7, 8], dtype=np.int32)
        np.testing.assert_array_equal(xadj, expected_xadj)

    def test_chain_full_block_adjncy(self):
        nb = _chain_neighbours(4)
        xadj, adjncy = self._csr(nb, 0, 4)
        # tri 0: [1], tri 1: [0,2], tri 2: [1,3], tri 3: [2]
        expected = np.array([1, 0, 2, 1, 3, 2], dtype=np.int32)
        np.testing.assert_array_equal(adjncy, expected)

    def test_chain_total_edges(self):
        n = 10
        nb = _chain_neighbours(n)
        xadj, adjncy = self._csr(nb, 0, n)
        # chain has n-1 interior edges, each counted twice
        self.assertEqual(len(adjncy), 2 * (n - 1))

    # --- chain mesh: partial block ---

    def test_chain_partial_block_middle(self):
        # Extract local block [2, 4) from a 6-triangle chain
        nb = _chain_neighbours(6)
        xadj, adjncy = self._csr(nb, 2, 4)
        # tri 2 (local 0): neighbours [1, -, 3] → [1, 3]
        # tri 3 (local 1): neighbours [2, -, 4] → [2, 4]
        np.testing.assert_array_equal(xadj, [0, 2, 4])
        np.testing.assert_array_equal(adjncy, [1, 3, 2, 4])

    def test_chain_partial_block_start(self):
        nb = _chain_neighbours(5)
        xadj, adjncy = self._csr(nb, 0, 2)
        # tri 0: [1]; tri 1: [0, 2]
        np.testing.assert_array_equal(xadj, [0, 1, 3])
        np.testing.assert_array_equal(adjncy, [1, 0, 2])

    def test_chain_partial_block_end(self):
        nb = _chain_neighbours(5)
        xadj, adjncy = self._csr(nb, 3, 5)
        # tri 3: [2, 4]; tri 4: [3]
        np.testing.assert_array_equal(xadj, [0, 2, 3])
        np.testing.assert_array_equal(adjncy, [2, 4, 3])

    # --- ring mesh ---

    def test_ring_every_triangle_has_two_neighbours(self):
        n = 8
        nb = _ring_neighbours(n)
        xadj, adjncy = self._csr(nb, 0, n)
        # Every triangle in a ring has exactly 2 neighbours
        self.assertEqual(len(adjncy), 2 * n)
        np.testing.assert_array_equal(np.diff(xadj), 2)

    # --- single triangle ---

    def test_single_triangle_all_boundary(self):
        nb = np.array([[-1, -1, -1]], dtype=np.int64)
        xadj, adjncy = self._csr(nb, 0, 1)
        np.testing.assert_array_equal(xadj, [0, 0])
        self.assertEqual(len(adjncy), 0)

    def test_single_triangle_with_neighbours(self):
        # Single interior triangle: all three edges shared
        nb = np.array([[1, 2, 3]], dtype=np.int64)
        xadj, adjncy = self._csr(nb, 0, 1)
        np.testing.assert_array_equal(xadj, [0, 3])
        self.assertEqual(len(adjncy), 3)


# ---------------------------------------------------------------------------
# parmetis_partition() — serial (n_procs=1) path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MPI4PY_AVAILABLE, reason='mpi4py not available')
@pytest.mark.skipif(not PARMETIS_AVAILABLE, reason='ParMETIS not installed')
class TestParmetisPartitionSerial(unittest.TestCase):

    def setUp(self):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD   # size=1 when running serially

    def _partition(self, nb, n_tri):
        from anuga.parallel.partitioning import parmetis_partition
        return parmetis_partition(nb, n_tri, 1, self.comm)

    def test_returns_tuple_of_two(self):
        nb = _chain_neighbours(10)
        result = self._partition(nb, 10)
        self.assertEqual(len(result), 2)

    def test_epart_order_is_identity_permutation(self):
        n = 12
        nb = _chain_neighbours(n)
        epart_order, _ = self._partition(nb, n)
        np.testing.assert_array_equal(epart_order, np.arange(n))

    def test_tpp_is_array_of_n_tri(self):
        n = 8
        nb = _chain_neighbours(n)
        _, tpp = self._partition(nb, n)
        np.testing.assert_array_equal(tpp, [n])

    def test_epart_order_length(self):
        n = 20
        nb = _ring_neighbours(n)
        epart_order, _ = self._partition(nb, n)
        self.assertEqual(len(epart_order), n)

    def test_works_with_single_triangle(self):
        nb = np.array([[-1, -1, -1]], dtype=np.int64)
        epart_order, tpp = self._partition(nb, 1)
        np.testing.assert_array_equal(epart_order, [0])
        np.testing.assert_array_equal(tpp, [1])


# ---------------------------------------------------------------------------
# MPI smoke tests — spawn run_parallel_parmetis.py via mpiexec
# ---------------------------------------------------------------------------

def _extra_mpiexec_options():
    import platform
    if platform.system() == 'Windows':
        return ''
    result = subprocess.run(
        ['mpiexec', '-np', '2', '--oversubscribe', 'echo', 'ok'],
        capture_output=True)
    return '--oversubscribe' if result.returncode == 0 else ''


@pytest.mark.slow
@pytest.mark.skipif(not MPI4PY_AVAILABLE, reason='requires mpi4py')
@pytest.mark.skipif(not PARMETIS_AVAILABLE, reason='ParMETIS not installed')
class TestParmetisMPI(unittest.TestCase):
    """Smoke test: run_parallel_parmetis.py exits 0 under mpiexec."""

    def _run(self, nprocs):
        extra = _extra_mpiexec_options()
        cmd = (['mpiexec', '-np', str(nprocs)]
               + (extra.split() if extra else [])
               + [sys.executable, run_script])
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            raise AssertionError(
                f'mpiexec -np {nprocs} timed out after 120 s')
        if result.returncode != 0:
            raise AssertionError(
                f'mpiexec -np {nprocs} failed:\n'
                + result.stdout.decode()
                + result.stderr.decode())

    def test_mpi_2ranks(self):
        self._run(2)

    def test_mpi_4ranks(self):
        self._run(4)


if __name__ == '__main__':
    unittest.main()
