#!/usr/bin/env python3
"""Benchmark distribute_basic_mesh() -- the mesh-first parallel workflow.

Creates a Basic_mesh on rank 0 (no quantities), distributes it to all ranks,
then sets initial conditions on the local Parallel_domain.  Compares timing
and memory against the traditional distribute() approach where a full Domain
with quantities is built on rank 0 first.

Usage
-----
    mpirun -np 8 python scripts/benchmark_distribute_basic_mesh.py [options]

Options
-------
--size M       Grid size: 4*M*M triangles (rectangular_cross, default 500)
--reps R       Repetitions for timing (default 1)
--scheme S     Partition scheme: metis | morton | hilbert (default morton)
--interval T   Memory ticker interval in seconds (default 5.0)
--no-evolve    Skip the short evolve check (default: run 1 step)
"""

import argparse
import gc
import os
import statistics
import sys
import time

import numpy as num

# ---------------------------------------------------------------------------
# MPI
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI
    comm  = MPI.COMM_WORLD
    myid  = comm.Get_rank()
    nproc = comm.Get_size()
    mpi_available = True
except ImportError:
    comm  = None
    myid  = 0
    nproc = 1
    mpi_available = False


# ---------------------------------------------------------------------------
# Memory helpers (PSS preferred, RSS fallback)
# ---------------------------------------------------------------------------

def _rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _pss_mb():
    try:
        with open('/proc/self/smaps_rollup') as f:
            for line in f:
                if line.startswith('Pss:'):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        total = 0
        with open('/proc/self/smaps') as f:
            for line in f:
                if line.startswith('Pss:'):
                    total += int(line.split()[1])
        return total / 1024.0
    except OSError:
        return _rss_mb()


def _reduce_mem(val_mb, label):
    """Sum PSS across all ranks (reduce to rank 0).  Returns GiB on rank 0."""
    if mpi_available:
        total = comm.reduce(val_mb, op=MPI.SUM, root=0)
    else:
        total = val_mb
    return (total or 0.0) / 1024.0


# ---------------------------------------------------------------------------
# Timed sections
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self._t = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self._t


def _barrier_wall():
    """Barrier then return the wall time on rank 0."""
    if mpi_available:
        comm.Barrier()
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--size',     type=int,   default=500)
    p.add_argument('--reps',     type=int,   default=1)
    p.add_argument('--scheme',   type=str,   default='morton',
                   choices=['metis', 'morton', 'hilbert'])
    p.add_argument('--interval', type=float, default=5.0)
    p.add_argument('--no-evolve', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Basic_mesh creation
# ---------------------------------------------------------------------------

def make_basic_mesh(grid_size, scheme):
    """Rank 0: build Basic_mesh.  Others: return None."""
    if myid != 0:
        return None
    from anuga.abstract_2d_finite_volumes.basic_mesh import \
        rectangular_cross_basic_mesh
    bm = rectangular_cross_basic_mesh(grid_size, grid_size,
                                      len1=grid_size, len2=grid_size)
    return bm


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def time_distribute_basic_mesh(grid_size, scheme, reps):
    """Time the mesh-first distribute_basic_mesh() workflow."""
    from anuga.parallel.parallel_api import distribute_basic_mesh

    parameters = {'partition_scheme': scheme}

    times = []
    pss_vals = []

    for rep in range(reps):
        gc.collect()
        if mpi_available:
            comm.Barrier()

        bm = make_basic_mesh(grid_size, scheme)

        t0 = _barrier_wall()
        domain = distribute_basic_mesh(bm, parameters=parameters)
        t1 = _barrier_wall()
        times.append(t1 - t0)

        # Set initial conditions post-distribution
        domain.set_quantity('elevation', 0.0)
        domain.set_quantity('stage',     0.0)
        domain.set_quantity('friction',  0.03)

        pss_mb = _pss_mb()
        pss_gib = _reduce_mem(pss_mb, 'distribute_basic_mesh')
        pss_vals.append(pss_gib)

        del domain, bm
        gc.collect()
        if mpi_available:
            comm.Barrier()

    return times, pss_vals


def time_distribute_domain(grid_size, scheme, reps):
    """Time the traditional distribute() workflow for comparison."""
    from anuga import Domain
    from anuga.parallel.parallel_api import distribute

    parameters = {'partition_scheme': scheme}

    times = []
    pss_vals = []

    for rep in range(reps):
        gc.collect()
        if mpi_available:
            comm.Barrier()

        if myid == 0:
            from anuga.abstract_2d_finite_volumes.mesh_factory import \
                rectangular_cross_with_neighbours
            points, elements, boundary, neighbours, neighbour_edges = \
                rectangular_cross_with_neighbours(grid_size, grid_size,
                                                  len1=grid_size,
                                                  len2=grid_size)
            domain = Domain(points, elements, boundary)
            domain.set_quantity('elevation', 0.0)
            domain.set_quantity('stage',     0.0)
            domain.set_quantity('friction',  0.03)
        else:
            domain = None

        t0 = _barrier_wall()
        if myid == 0:
            domain = distribute(domain, parameters=parameters)
        else:
            domain = distribute(domain, parameters=parameters)
        t1 = _barrier_wall()
        times.append(t1 - t0)

        pss_mb = _pss_mb()
        pss_gib = _reduce_mem(pss_mb, 'distribute')
        pss_vals.append(pss_gib)

        del domain
        gc.collect()
        if mpi_available:
            comm.Barrier()

    return times, pss_vals


def run_evolve_check(grid_size, scheme):
    """Distribute a small mesh and run one evolve step to verify correctness."""
    from anuga.parallel.parallel_api import distribute_basic_mesh
    from anuga import Reflective_boundary

    bm = make_basic_mesh(grid_size, scheme)
    domain = distribute_basic_mesh(bm, parameters={'partition_scheme': scheme})

    domain.set_quantity('elevation', 0.0)
    domain.set_quantity('stage',     lambda x, y: num.where(
        (x > grid_size * 0.4) & (x < grid_size * 0.6), 1.0, 0.0))
    domain.set_quantity('friction',  0.03)

    Br = Reflective_boundary(domain)
    domain.set_boundary({
        'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

    t0 = time.perf_counter()
    for t in domain.evolve(yieldstep=0.01, finaltime=0.01):
        pass
    dt = time.perf_counter() - t0

    if myid == 0:
        print(f'  Evolve check: 1 step in {dt:.2f}s  -- domain OK')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    M        = args.size
    reps     = args.reps
    scheme   = args.scheme
    ntri     = 4 * M * M

    if myid == 0:
        W = 66
        print('=' * W)
        print('  distribute_basic_mesh() benchmark  -- mesh-first parallel workflow')
        print(f'  Mesh:       rectangular_cross {M}x{M}')
        print(f'  Triangles:  {ntri:,}')
        print(f'  MPI ranks:  {nproc}   Repetitions: {reps}')
        print(f'  Scheme:     {scheme}')
        print('=' * W)

    # -------------------------------------------------------------------
    # distribute_basic_mesh() benchmark
    # -------------------------------------------------------------------
    times_new, pss_new = time_distribute_basic_mesh(M, scheme, reps)

    # -------------------------------------------------------------------
    # Traditional distribute() benchmark for comparison
    # -------------------------------------------------------------------
    times_old, pss_old = time_distribute_domain(M, scheme, reps)

    # -------------------------------------------------------------------
    # Evolve check
    # -------------------------------------------------------------------
    if not args.no_evolve:
        small = min(M, 50)   # use a smaller mesh for the evolve check
        if myid == 0:
            print(f'\n  Running evolve check on {small}x{small} mesh ...')
        run_evolve_check(small, scheme)

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    if myid == 0:
        def fmt_t(ts):
            m = statistics.median(ts)
            return f'{m:7.3f}s'

        def fmt_p(ps):
            m = statistics.median(ps)
            return f'{m:7.2f} GiB'

        W = 32
        C = 18
        sep = '-' * (W + 2 + 2 * (C + 2))
        print()
        print(sep)
        print(f'  {"":>{W}}  {"distribute_basic_mesh()":>{C}}  {"distribute() + Domain":>{C}}')
        print(sep)
        print(f'  {"Wall time (median)":>{W}}  {fmt_t(times_new):>{C}}  {fmt_t(times_old):>{C}}')
        print(f'  {"Peak PSS (sum, GiB)":>{W}}  {fmt_p(pss_new):>{C}}  {fmt_p(pss_old):>{C}}')
        print(sep)

        speedup = statistics.median(times_old) / statistics.median(times_new)
        mem_saving_pct = 100.0 * (statistics.median(pss_old) -
                                   statistics.median(pss_new)) / \
                          max(statistics.median(pss_old), 1e-9)
        print(f'\n  distribute_basic_mesh() speedup:   {speedup:.2f}x faster')
        print(f'  Memory saving:               {mem_saving_pct:.1f}%')
        print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f'\n[rank {myid}] ERROR: {e}', flush=True)
        traceback.print_exc()
        if mpi_available:
            comm.Abort(1)
        sys.exit(1)
    if mpi_available:
        MPI.Finalize()
