"""Tests for the parallel-run sww merge, including the workers option."""

import hashlib
import os
import tempfile
import unittest

import numpy as num
from netCDF4 import Dataset

from anuga.utilities.sww_merge import (_sww_merge_parallel_smooth,
                                       _sww_merge_parallel_non_smooth)


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _common_attrs(ds):
    ds.order = 2
    ds.xllcorner = 0.0
    ds.yllcorner = 0.0
    ds.zone = -1
    ds.false_easting = 500000
    ds.false_northing = 10000000
    ds.datum = 'wgs84'
    ds.projection = 'UTM'
    ds.units = 'm'
    ds.starttime = 0
    ds.description = 'sww merge test'


def _write_smooth_files(tmpdir, n_procs=3, nodes_per=400, n_steps=11):
    """Small synthetic smooth-format per-rank files (shared vertices)."""
    rng = num.random.default_rng(42)
    n_global_nodes = n_procs * nodes_per
    tris_per = int(nodes_per * 1.8)
    n_global_tris = n_procs * tris_per
    files = []
    for r in range(n_procs):
        n_ghost = nodes_per // 10
        n_pts = nodes_per + n_ghost
        node_l2g = num.concatenate([
            num.arange(r * nodes_per, (r + 1) * nodes_per),
            rng.choice(n_global_nodes, size=n_ghost, replace=False)
        ]).astype(num.int32)
        n_tris_ghost = tris_per // 10
        n_tris = tris_per + n_tris_ghost
        tri_l2g = num.concatenate([
            num.arange(r * tris_per, (r + 1) * tris_per),
            rng.choice(n_global_tris, size=n_tris_ghost, replace=False)
        ]).astype(num.int32)
        tri_full_flag = num.concatenate([
            num.ones(tris_per, num.int32), num.zeros(n_tris_ghost, num.int32)])
        volumes = rng.integers(0, nodes_per, size=(n_tris, 3)).astype(num.int32)

        path = os.path.join(tmpdir, 'sm_P%d_%d.sww' % (n_procs, r))
        ds = Dataset(path, 'w', format='NETCDF3_64BIT')
        ds.createDimension('number_of_volumes', n_tris)
        ds.createDimension('number_of_vertices', 3)
        ds.createDimension('number_of_points', n_pts)
        ds.createDimension('number_of_timesteps', None)
        ds.createDimension('numbers_in_range', 2)
        ds.number_of_global_triangles = n_global_tris
        ds.number_of_global_nodes = n_global_nodes
        _common_attrs(ds)
        ds.createVariable('time', 'f8', ('number_of_timesteps',))
        for name in ('x', 'y', 'elevation'):
            v = ds.createVariable(name, 'f4', ('number_of_points',))
            v[:] = rng.random(n_pts, dtype=num.float32)
        ds.createVariable('volumes', 'i4',
                          ('number_of_volumes', 'number_of_vertices'))[:] = volumes
        ds.createVariable('tri_l2g', 'i4', ('number_of_volumes',))[:] = tri_l2g
        ds.createVariable('node_l2g', 'i4', ('number_of_points',))[:] = node_l2g
        ds.createVariable('tri_full_flag', 'i4',
                          ('number_of_volumes',))[:] = tri_full_flag
        for q in ('stage', 'xmomentum', 'ymomentum'):
            ds.createVariable(q, 'f4', ('number_of_timesteps',
                                        'number_of_points'))
            ds.createVariable(q + '_range', 'f4',
                              ('numbers_in_range',))[:] = [0.0, 1.0]
        ds.createVariable('stage_c', 'f4', ('number_of_timesteps',
                                            'number_of_volumes'))
        ds.variables['time'][:] = num.arange(n_steps, dtype=num.float64)
        for q in ('stage', 'xmomentum', 'ymomentum'):
            ds.variables[q][:, :] = rng.random((n_steps, n_pts),
                                               dtype=num.float32)
        ds.variables['stage_c'][:, :] = rng.random((n_steps, n_tris),
                                                   dtype=num.float32)
        ds.close()
        files.append(path)
    return files


def _write_non_smooth_files(tmpdir, n_procs=3, tris_per=600, n_steps=9):
    """Small synthetic non-smooth per-rank files (3 vertices per triangle)."""
    rng = num.random.default_rng(7)
    n_global_tris = n_procs * tris_per
    files = []
    for r in range(n_procs):
        n_tris = tris_per + tris_per // 10
        n_pts = 3 * n_tris
        tri_l2g = num.concatenate([
            num.arange(r * tris_per, (r + 1) * tris_per),
            rng.choice(n_global_tris, size=n_tris - tris_per, replace=False)
        ]).astype(num.int32)
        tri_full_flag = num.concatenate([
            num.ones(tris_per, num.int32),
            num.zeros(n_tris - tris_per, num.int32)])

        path = os.path.join(tmpdir, 'ns_P%d_%d.sww' % (n_procs, r))
        ds = Dataset(path, 'w', format='NETCDF3_64BIT')
        ds.createDimension('number_of_volumes', n_tris)
        ds.createDimension('number_of_vertices', 3)
        ds.createDimension('number_of_points', n_pts)
        ds.createDimension('number_of_timesteps', None)
        ds.createDimension('numbers_in_range', 2)
        ds.number_of_global_triangles = n_global_tris
        ds.number_of_global_nodes = 3 * n_global_tris
        _common_attrs(ds)
        ds.createVariable('time', 'f8', ('number_of_timesteps',))
        for name in ('x', 'y', 'elevation'):
            v = ds.createVariable(name, 'f4', ('number_of_points',))
            v[:] = rng.random(n_pts, dtype=num.float32)
        ds.createVariable('volumes', 'i4',
                          ('number_of_volumes', 'number_of_vertices'))[:] = \
            num.arange(3 * n_tris, dtype=num.int32).reshape(-1, 3)
        ds.createVariable('tri_l2g', 'i4', ('number_of_volumes',))[:] = tri_l2g
        ds.createVariable('tri_full_flag', 'i4',
                          ('number_of_volumes',))[:] = tri_full_flag
        for q in ('stage', 'xmomentum', 'ymomentum'):
            ds.createVariable(q, 'f4', ('number_of_timesteps',
                                        'number_of_points'))
            ds.createVariable(q + '_range', 'f4',
                              ('numbers_in_range',))[:] = [0.0, 1.0]
        ds.variables['time'][:] = num.arange(n_steps, dtype=num.float64)
        for q in ('stage', 'xmomentum', 'ymomentum'):
            ds.variables[q][:, :] = rng.random((n_steps, n_pts),
                                               dtype=num.float32)
        ds.close()
        files.append(path)
    return files


class Test_sww_merge_workers(unittest.TestCase):
    """The workers option must reproduce the serial merge byte for byte."""

    def test_smooth_parallel_merge_identical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = _write_smooth_files(tmpdir)
            serial = os.path.join(tmpdir, 'serial.sww')
            par = os.path.join(tmpdir, 'par.sww')
            par_chunk = os.path.join(tmpdir, 'par_chunk.sww')
            _sww_merge_parallel_smooth(files, serial)
            _sww_merge_parallel_smooth(files, par, workers=2)
            _sww_merge_parallel_smooth(files, par_chunk, workers=2,
                                       chunk_size=3)
            self.assertEqual(_md5(serial), _md5(par))
            self.assertEqual(_md5(serial), _md5(par_chunk))

    def test_smooth_serial_chunked_identical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = _write_smooth_files(tmpdir)
            serial = os.path.join(tmpdir, 'serial.sww')
            chunked = os.path.join(tmpdir, 'chunked.sww')
            _sww_merge_parallel_smooth(files, serial)
            _sww_merge_parallel_smooth(files, chunked, chunk_size=4)
            self.assertEqual(_md5(serial), _md5(chunked))

    def test_non_smooth_parallel_merge_identical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = _write_non_smooth_files(tmpdir)
            serial = os.path.join(tmpdir, 'serial.sww')
            par = os.path.join(tmpdir, 'par.sww')
            _sww_merge_parallel_non_smooth(files, serial)
            _sww_merge_parallel_non_smooth(files, par, workers=2,
                                           chunk_size=2)
            self.assertEqual(_md5(serial), _md5(par))


if __name__ == '__main__':
    unittest.main()
