"""Test that distribute_collaborative() produces numerically identical results
to a sequential run of the shallow water domain.

WARNING: This assumes that the command to run jobs is mpiexec.
"""

import platform
import unittest
import numpy as num
import os
import subprocess
import sys

try:
    import mpi4py
except ImportError:
    pass

import pytest

verbose = False

path = os.path.dirname(__file__)  # Get folder where this script lives
run_filename = os.path.join(path, 'run_parallel_collaborative_domain.py')

# These must match the filenames used in the run script.
sequential_file = 'collaborative_domain_sequential.txt'
parallel_file   = 'collaborative_domain_parallel.txt'


@pytest.mark.skipif('mpi4py' not in sys.modules,
                    reason="requires the mpi4py module")
class Test_parallel_collaborative_domain(unittest.TestCase):

    def setUp(self):
        # ----------------------
        # First run sequentially
        # ----------------------
        cmd = 'python ' + run_filename
        result = subprocess.run(cmd.split(), capture_output=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise Exception(result.stderr)

        # Detect whether --oversubscribe is supported (OpenMPI only)
        extra_options = '--oversubscribe'
        result = subprocess.run(
            ('mpiexec -np 3 ' + extra_options + ' echo').split(),
            capture_output=True)
        if result.returncode != 0:
            extra_options = ''

        if platform.system() == 'Windows':
            extra_options = ''

        # --------------------
        # Then run in parallel
        # --------------------
        cmd = 'mpiexec -np 3 ' + extra_options + ' python ' + run_filename
        result = subprocess.run(cmd.split(), capture_output=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise Exception(result.stderr)

    def tearDown(self):
        for fname in (sequential_file, parallel_file):
            try:
                os.remove(fname)
            except OSError:
                pass

    def test_that_sequential_and_parallel_outputs_are_identical(self):
        with open(sequential_file) as fid_seq, open(parallel_file) as fid_par:
            seq_values = fid_seq.readlines()
            par_values = fid_par.readlines()

        N = len(seq_values)
        assert len(par_values) == N

        for i in range(N):
            seq_nums = [float(x) for x in seq_values[i].split()]
            par_nums = [float(x) for x in par_values[i].split()]
            assert num.allclose(seq_nums, par_nums)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    suite = unittest.TestLoader().loadTestsFromTestCase(
        Test_parallel_collaborative_domain)
    runner.run(suite)
