"""
Test that a channel draining through a Normal_depth_boundary gives the same
answer on one process and on three.
"""

import os
import platform
import subprocess
import sys

import pytest

try:
    import mpi4py  # noqa: F401
except ImportError:
    pass

from anuga.file.sww import sww_files_are_equal

run_filename = os.path.join(os.path.dirname(__file__),
                            'run_parallel_normal_depth.py')


def run(cmd, cwd):
    result = subprocess.run(cmd.split(), capture_output=True, cwd=cwd)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise Exception(result.stderr)


def mpiexec_options():
    if platform.system() == 'Windows':
        return ''
    result = subprocess.run('mpiexec -np 3 --oversubscribe echo'.split(),
                            capture_output=True)
    return '--oversubscribe' if result.returncode == 0 else ''


@pytest.mark.skipif('mpi4py' not in sys.modules,
                    reason="requires the mpi4py module")
@pytest.mark.parametrize('alg', ['DE0', 'DE1', 'DE_ader2'])
def test_that_sequential_and_parallel_outputs_are_identical(alg, tmp_path):
    cwd = str(tmp_path)
    run('%s %s %s' % (sys.executable, run_filename, alg), cwd)
    run('mpiexec -np 3 %s %s %s %s' % (mpiexec_options(), sys.executable,
                                       run_filename, alg), cwd)

    assert sww_files_are_equal(os.path.join(cwd, 's_normal_depth.sww'),
                               os.path.join(cwd, 'p_normal_depth.sww'))
