"""Benchmark: parallel elevation-loading strategies for distribute_basic_mesh.

Compares three approaches for setting elevation on a distributed mesh:

  A  — TSH with elevation embedded (rank 0 reads TSH, extracts vertex
       attributes, broadcasts nodes + elevation to all ranks).

  B1 — TSH mesh only (rank 0) + XYA file read by rank 0 and broadcast.

  B2 — TSH mesh only (rank 0) + XYA file read independently by every rank.

Each approach is timed from file-open through to domain.set_quantity so that
the comparison captures I/O, broadcast communication, and interpolation cost.
The full evolve loop is intentionally excluded; this is a setup benchmark.

Usage::

    mpiexec -np 4 python -u bench_parallel_elevation.py
    mpiexec -np 1 python -u bench_parallel_elevation.py   # serial baseline
"""

import os
import sys
import time

import numpy as np

import anuga
from anuga import (
    myid, numprocs, finalize, barrier,
    distribute_basic_mesh, basic_mesh_from_mesh_file, Geo_reference,
)

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR      = os.path.join(os.path.dirname(__file__), 'data')
MESH_FILE     = os.path.join(DATA_DIR, 'merimbula_10785_1.tsh')
ELEV_XYA_FILE = os.path.join(DATA_DIR, 'merimbula_bathymetry.xya')

georef = Geo_reference(zone=55, hemisphere='southern')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bcast(data):
    """Broadcast data from rank 0; no-op in serial."""
    if HAS_MPI:
        return comm.bcast(data, root=0)
    return data


def _barrier():
    barrier()


def _timed(label, t_start):
    """Print a timing line from rank 0."""
    if myid == 0:
        print(f'  {label:<45s} {time.time() - t_start:.4f} s')
        sys.stdout.flush()


def _section(title):
    if myid == 0:
        print()
        print('=' * 60)
        print(title)
        print('=' * 60)
        sys.stdout.flush()


def _make_elevation_func_from_nodes(nodes, elevation):
    """Build a LinearNDInterpolator from global node data."""
    from scipy.interpolate import LinearNDInterpolator
    return LinearNDInterpolator(nodes, elevation)


def _make_elevation_func_from_xya(xya_path):
    """Read an XYA CSV and return a LinearNDInterpolator."""
    from scipy.interpolate import LinearNDInterpolator
    data = np.loadtxt(xya_path, delimiter=',', skiprows=1)
    pts  = data[:, :2]
    elev = data[:, 2]
    return LinearNDInterpolator(pts, elev)


# ===========================================================================
# Approach A: elevation embedded in TSH, rank 0 reads + broadcasts
# ===========================================================================

def approach_A(verbose=False):
    """TSH file contains elevation vertex attribute; rank 0 reads and bcasts."""

    _section('Approach A: elevation from TSH vertex attributes (rank-0 bcast)')

    t0 = time.time()

    # --- Step 1: rank 0 reads mesh + elevation ---
    if myid == 0:
        bm = basic_mesh_from_mesh_file(MESH_FILE, verbose=verbose)
        if 'elevation' in bm.vertex_attribute_titles:
            idx = bm.vertex_attribute_titles.index('elevation')
            elev = bm.vertex_attributes[:, idx]
        else:
            raise RuntimeError('No elevation attribute in TSH file')
        bcast_data = {'nodes': bm.nodes, 'elevation': elev}
    else:
        bm         = None
        bcast_data = None

    _timed('rank-0 TSH read (mesh + elevation)', t0)

    # --- Step 2: broadcast elevation ---
    t1 = time.time()
    bcast_data = _bcast(bcast_data)
    _timed('broadcast nodes + elevation', t1)

    # --- Step 3: build interpolator ---
    t2 = time.time()
    elev_func = _make_elevation_func_from_nodes(
        bcast_data['nodes'], bcast_data['elevation'])
    _timed('build LinearNDInterpolator (all ranks)', t2)

    # --- Step 4: distribute ---
    _barrier()
    t3 = time.time()
    domain = distribute_basic_mesh(bm, verbose=False)
    domain.set_georeference(georef)
    _timed('distribute_basic_mesh', t3)

    # --- Step 5: set quantity ---
    t4 = time.time()
    domain.set_quantity('elevation', elev_func)
    _timed('set_quantity elevation', t4)

    _barrier()
    total = time.time() - t0
    if myid == 0:
        print(f'  {"TOTAL":<45s} {total:.4f} s')
    return domain


# ===========================================================================
# Approach B1: elevation from XYA, rank 0 reads + broadcasts
# ===========================================================================

def approach_B1(verbose=False):
    """XYA elevation file read by rank 0 and broadcast to all ranks."""

    _section('Approach B1: elevation from XYA, rank-0 reads + bcasts')

    t0 = time.time()

    # --- Step 1: rank 0 reads mesh (no elevation needed) ---
    if myid == 0:
        bm = basic_mesh_from_mesh_file(MESH_FILE, verbose=verbose)
    else:
        bm = None

    _timed('rank-0 TSH read (mesh only)', t0)

    # --- Step 2: rank 0 reads XYA, broadcasts ---
    t1 = time.time()
    if myid == 0:
        xya_data = np.loadtxt(ELEV_XYA_FILE, delimiter=',', skiprows=1)
    else:
        xya_data = None

    _timed('rank-0 XYA read', t1)

    t2 = time.time()
    xya_data = _bcast(xya_data)
    _timed('broadcast XYA data', t2)

    # --- Step 3: build interpolator ---
    t3 = time.time()
    from scipy.interpolate import LinearNDInterpolator
    elev_func = LinearNDInterpolator(xya_data[:, :2], xya_data[:, 2])
    _timed('build LinearNDInterpolator (all ranks)', t3)

    # --- Step 4: distribute ---
    _barrier()
    t4 = time.time()
    domain = distribute_basic_mesh(bm, verbose=False)
    domain.set_georeference(georef)
    _timed('distribute_basic_mesh', t4)

    # --- Step 5: set quantity ---
    t5 = time.time()
    domain.set_quantity('elevation', elev_func)
    _timed('set_quantity elevation', t5)

    _barrier()
    total = time.time() - t0
    if myid == 0:
        print(f'  {"TOTAL":<45s} {total:.4f} s')
    return domain


# ===========================================================================
# Approach B2: elevation from XYA, all ranks read independently
# ===========================================================================

def approach_B2(verbose=False):
    """XYA elevation file read independently by every rank (no broadcast)."""

    _section('Approach B2: elevation from XYA, all ranks read independently')

    t0 = time.time()

    # --- Step 1: rank 0 reads mesh ---
    if myid == 0:
        bm = basic_mesh_from_mesh_file(MESH_FILE, verbose=verbose)
    else:
        bm = None

    _timed('rank-0 TSH read (mesh only)', t0)

    # --- Step 2: all ranks read XYA independently ---
    _barrier()   # ensure mesh read completes before ranks hammer the file
    t1 = time.time()
    xya_data = np.loadtxt(ELEV_XYA_FILE, delimiter=',', skiprows=1)
    # Synchronise so we time the slowest rank
    _barrier()
    _timed('all-ranks XYA read (slowest rank)', t1)

    # --- Step 3: build interpolator ---
    t2 = time.time()
    from scipy.interpolate import LinearNDInterpolator
    elev_func = LinearNDInterpolator(xya_data[:, :2], xya_data[:, 2])
    _barrier()
    _timed('build LinearNDInterpolator (slowest rank)', t2)

    # --- Step 4: distribute ---
    t3 = time.time()
    domain = distribute_basic_mesh(bm, verbose=False)
    domain.set_georeference(georef)
    _timed('distribute_basic_mesh', t3)

    # --- Step 5: set quantity ---
    t4 = time.time()
    domain.set_quantity('elevation', elev_func)
    _timed('set_quantity elevation', t4)

    _barrier()
    total = time.time() - t0
    if myid == 0:
        print(f'  {"TOTAL":<45s} {total:.4f} s')
    return domain


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':

    if myid == 0:
        print()
        print('Parallel elevation benchmark')
        print(f'  Ranks     : {numprocs}')
        print(f'  Mesh file : {MESH_FILE}')
        print(f'  XYA file  : {ELEV_XYA_FILE}')

    # Run each approach; delete domain between runs to free memory
    domain_A  = approach_A()
    del domain_A
    _barrier()

    domain_B1 = approach_B1()
    del domain_B1
    _barrier()

    domain_B2 = approach_B2()
    del domain_B2
    _barrier()

    if myid == 0:
        print()
        print('Benchmark complete.')

    finalize()
