"""Benchmark distribute() vs distribute_collaborative().

Runs both functions on a mesh and reports wall-clock time (max across
all ranks) and peak memory (sum and max across all ranks) for each.
A background thread on rank 0 prints a live memory ticker during each
timed call so you can watch allocation grow.

Run with:

    mpiexec -np <N> python scripts/benchmark_distribute.py [--reps <R>] [options]

Options
-------
--reps R      Number of timed repetitions per function (default: 3).
--mesh PATH   Path to a .tsh mesh file (uses merimbula by default).
--size M      Use a synthetic M×M rectangular mesh (overrides --mesh).
              Each cell is split into 4 triangles, giving 4*M*M triangles.
--interval S  Memory-ticker sample interval in seconds (default: 1.0).
"""

import argparse
import gc
import os
import shutil
import statistics
import tempfile
import threading
import time

from mpi4py import MPI

import anuga
from anuga.parallel.parallel_api import distribute, distribute_collaborative
from anuga.parallel.sequential_distribute import (
    sequential_distribute_dump, sequential_distribute_load)
from anuga import create_domain_from_file
from anuga.utilities.system_tools import get_pathname_from_package

comm  = MPI.COMM_WORLD
myid  = comm.Get_rank()
nproc = comm.Get_size()


# ── Memory sampling ───────────────────────────────────────────────────────────

def _rss_mb():
    """Return current RSS of this process in MiB (Linux /proc, no dependencies)."""
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0   # kB → MiB
    except OSError:
        pass
    return 0.0


def _pss_mb():
    """Return Proportional Set Size in MiB.

    PSS counts shared memory pages proportionally (divided by the number of
    processes that map them), so summing PSS across ranks gives the true
    physical-memory footprint of the job — unlike VmRSS which double-counts
    pages shared via MPI.Win.Allocate_shared.
    """
    try:
        with open('/proc/self/smaps_rollup') as fh:
            for line in fh:
                if line.startswith('Pss:'):
                    return int(line.split()[1]) / 1024.0   # kB → MiB
    except OSError:
        pass
    # Fallback: parse /proc/self/smaps (slower but always present)
    try:
        total = 0
        with open('/proc/self/smaps') as fh:
            for line in fh:
                if line.startswith('Pss:'):
                    total += int(line.split()[1])
        return total / 1024.0
    except OSError:
        pass
    return 0.0


class MemoryMonitor:
    """Background thread that samples this process's RSS every `interval` seconds.

    Use as a context manager::

        with MemoryMonitor(label='distribute()', interval=1.0) as mon:
            fn(domain)
        peak_mib = mon.peak_mb
    """

    def __init__(self, label='', interval=1.0, print_ticker=False):
        self.label        = label
        self.interval     = interval
        self.print_ticker = print_ticker
        self.peak_rss_mb  = 0.0
        self.peak_pss_mb  = 0.0
        self._stop        = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True)

    # Keep backward-compatible alias
    @property
    def peak_mb(self):
        return self.peak_rss_mb

    def __enter__(self):
        self.peak_rss_mb = _rss_mb()
        self.peak_pss_mb = _pss_mb()
        self._t0         = time.time()
        self._stop.clear()
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.wait(self.interval):
            rss = _rss_mb()
            pss = _pss_mb()
            if rss > self.peak_rss_mb:
                self.peak_rss_mb = rss
            if pss > self.peak_pss_mb:
                self.peak_pss_mb = pss
            if self.print_ticker:
                elapsed = time.time() - self._t0
                print(f'  [{self.label}]  t={elapsed:6.1f}s  '
                      f'rank-0 RSS={rss:,.0f} MiB  PSS={pss:,.0f} MiB',
                      flush=True)


# ── Domain construction ───────────────────────────────────────────────────────

def make_domain(mesh_filename=None, grid_size=None):
    """Build domain from file or synthetic rectangular mesh."""
    if grid_size is not None:
        pts, verts, bnd = anuga.rectangular_cross(grid_size, grid_size,
                                                   len1=1.0, len2=1.0)
        domain = anuga.Domain(pts, verts, bnd)
    else:
        domain = create_domain_from_file(mesh_filename)
    domain.set_quantity('stage', lambda x, y: 1.0 + 0.01 * x)
    return domain


# ── Timed run ─────────────────────────────────────────────────────────────────

def time_distribute(fn, mesh_filename, grid_size, reps, ticker_interval):
    """Run fn `reps` times; return (times, pss_sum_mbs, rss_max_mbs) lists."""
    times       = []
    pss_sum_mbs = []   # PSS sum: true physical-memory footprint of the job
    rss_max_mbs = []   # RSS max: worst-case single-rank footprint

    label = fn.__name__

    for rep in range(reps):
        # Drain residual memory from the previous run/rep before measuring.
        gc.collect()
        comm.Barrier()
        domain = make_domain(mesh_filename, grid_size)
        comm.Barrier()

        # Start memory monitor on every rank; only rank 0 prints the ticker.
        mon = MemoryMonitor(label=label, interval=ticker_interval,
                            print_ticker=(myid == 0))
        with mon:
            t0 = MPI.Wtime()
            pd = fn(domain)
            t1 = MPI.Wtime()

        elapsed  = t1 - t0
        # Capture a final sample in case the thread missed the peak.
        peak_rss = max(mon.peak_rss_mb, _rss_mb())
        peak_pss = max(mon.peak_pss_mb, _pss_mb())

        # Reduce timing and memory across all ranks.
        wall    = comm.reduce(elapsed,  op=MPI.MAX, root=0)
        pss_sum = comm.reduce(peak_pss, op=MPI.SUM, root=0)
        rss_max = comm.reduce(peak_rss, op=MPI.MAX, root=0)

        if myid == 0:
            times.append(wall)
            pss_sum_mbs.append(pss_sum)
            rss_max_mbs.append(rss_max)

        del pd

    return times, pss_sum_mbs, rss_max_mbs


# ── Timed dump + load ─────────────────────────────────────────────────────────

def time_dump_load(mesh_filename, grid_size, reps, ticker_interval):
    """Benchmark sequential_distribute_dump (rank-0 only) + load (all ranks).

    The partition directory must be on a filesystem visible to all ranks
    (e.g. Lustre scratch on Gadi). A temporary directory is created under
    the system's default tempdir; set TMPDIR to redirect if needed.

    Returns (times_dump, times_load, pss_sum_mbs, rss_max_mbs) where:
        times_dump   -- wall time for the serial dump on rank 0
        times_load   -- max wall time across ranks for the parallel load
        pss_sum_mbs  -- PSS sum across ranks during the load phase
        rss_max_mbs  -- peak RSS on the busiest rank during the load phase
    """
    times_dump  = []
    times_load  = []
    pss_sum_mbs = []
    rss_max_mbs = []

    for rep in range(reps):
        gc.collect()
        comm.Barrier()

        # All ranks need the same tmpdir path (shared filesystem).
        if myid == 0:
            tmpdir = tempfile.mkdtemp(prefix='anuga_bench_')
        else:
            tmpdir = None
        tmpdir = comm.bcast(tmpdir, root=0)

        domain = make_domain(mesh_filename, grid_size)
        domain_name = domain.get_name()
        comm.Barrier()

        # ── Dump phase: rank 0 partitions and writes files ────────────────
        if myid == 0:
            mon_dump = MemoryMonitor(label='dump', interval=ticker_interval,
                                     print_ticker=True)
            with mon_dump:
                t0 = MPI.Wtime()
                sequential_distribute_dump(domain, nproc, partition_dir=tmpdir)
                t1 = MPI.Wtime()
            dump_elapsed = t1 - t0
        else:
            dump_elapsed = 0.0

        del domain
        comm.Barrier()   # ensure all files are written before load

        # ── Load phase: every rank reads its own partition files ──────────
        mon_load = MemoryMonitor(label='load', interval=ticker_interval,
                                 print_ticker=(myid == 0))
        with mon_load:
            t0 = MPI.Wtime()
            pd = sequential_distribute_load(filename=domain_name,
                                            partition_dir=tmpdir)
            t1 = MPI.Wtime()

        load_elapsed = t1 - t0
        peak_rss = max(mon_load.peak_rss_mb, _rss_mb())
        peak_pss = max(mon_load.peak_pss_mb, _pss_mb())

        dump_wall = comm.bcast(dump_elapsed, root=0)
        load_wall = comm.reduce(load_elapsed, op=MPI.MAX, root=0)
        pss_sum   = comm.reduce(peak_pss,     op=MPI.SUM, root=0)
        rss_max   = comm.reduce(peak_rss,     op=MPI.MAX, root=0)

        if myid == 0:
            times_dump.append(dump_wall)
            times_load.append(load_wall)
            pss_sum_mbs.append(pss_sum)
            rss_max_mbs.append(rss_max)

        del pd

        if myid == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
        comm.Barrier()

    return times_dump, times_load, pss_sum_mbs, rss_max_mbs


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_time(times):
    if len(times) == 1:
        return f'{times[0]:.3f}s'
    med = statistics.median(times)
    return f'{med:.3f}s  (min {min(times):.3f}  max {max(times):.3f})'


def fmt_mem(mbs):
    """Format a list of MiB values, converting to GiB if large."""
    med = statistics.median(mbs)
    if med >= 1024:
        return f'{med/1024:.2f} GiB'
    return f'{med:.0f} MiB'


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--reps',     type=int,   default=3)
    p.add_argument('--mesh',     type=str,   default=None)
    p.add_argument('--size',     type=int,   default=None,
                   help='Grid size M for synthetic M×M mesh (~4*M*M triangles)')
    p.add_argument('--interval', type=float, default=1.0,
                   help='Memory ticker sample interval in seconds (default: 1.0)')
    return p.parse_args()


def main():
    args = parse_args()

    grid_size = args.size
    if grid_size is not None:
        mesh_filename = None
        mesh_label = f'synthetic {grid_size}×{grid_size}'
    elif args.mesh is not None:
        mesh_filename = args.mesh
        mesh_label = os.path.basename(mesh_filename)
    else:
        mod_path = get_pathname_from_package('anuga.parallel')
        mesh_filename = os.path.join(mod_path, 'data', 'merimbula_10785_1.tsh')
        mesh_label = os.path.basename(mesh_filename)

    if myid == 0:
        domain_info = make_domain(mesh_filename, grid_size)
        ntri   = domain_info.number_of_triangles
        nnodes = domain_info.number_of_nodes
        del domain_info
        print(f'\n{"="*62}')
        print(f'  distribute() benchmark')
        print(f'  Mesh:       {mesh_label}')
        print(f'  Triangles:  {ntri:,}   Nodes: {nnodes:,}')
        print(f'  MPI ranks:  {nproc}   Repetitions: {args.reps}')
        print(f'{"="*62}\n')

    comm.Barrier()

    times_std, pss_std, rss_std = time_distribute(
        distribute, mesh_filename, grid_size, args.reps, args.interval)

    gc.collect()
    comm.Barrier()

    times_collab, pss_collab, rss_collab = time_distribute(
        distribute_collaborative, mesh_filename, grid_size, args.reps, args.interval)

    gc.collect()
    comm.Barrier()

    times_dump, times_load, pss_dl, rss_dl = time_dump_load(
        mesh_filename, grid_size, args.reps, args.interval)

    if myid == 0:
        # Total dump+load time per rep.
        times_dl_total = [d + l for d, l in zip(times_dump, times_load)]

        W = 32
        C = 16
        sep = '─' * (W + 2 + 3 * (C + 2))
        hdr = (f'  {"":>{W}}  {"distribute()":>{C}}  '
               f'{"collaborative()":>{C}}  {"dump()+load()":>{C}}')
        print(f'\n{sep}')
        print(hdr)
        print(sep)
        print(f'  {"Wall time (max across ranks)":<{W}}  '
              f'{fmt_time(times_std):>{C}}  '
              f'{fmt_time(times_collab):>{C}}  '
              f'{fmt_time(times_dl_total):>{C}}')
        print(f'  {"Peak PSS — sum (physical total)":<{W}}  '
              f'{fmt_mem(pss_std):>{C}}  '
              f'{fmt_mem(pss_collab):>{C}}  '
              f'{fmt_mem(pss_dl):>{C}}')
        print(f'  {"Peak RSS — max single rank":<{W}}  '
              f'{fmt_mem(rss_std):>{C}}  '
              f'{fmt_mem(rss_collab):>{C}}  '
              f'{fmt_mem(rss_dl):>{C}}')
        print(sep)
        print(f'  Note: PSS sums shared pages proportionally — '
              f'reflects true physical memory.')
        if len(times_dump) > 0:
            med_dump = statistics.median(times_dump)
            med_load = statistics.median(times_load)
            print(f'  dump()+load() breakdown: '
                  f'dump (serial) {med_dump:.3f}s  +  '
                  f'load (parallel) {med_load:.3f}s')

        candidates = [
            ('distribute()',    statistics.median(times_std)),
            ('collaborative()', statistics.median(times_collab)),
            ('dump()+load()',   statistics.median(times_dl_total)),
        ]
        best_name, best_time = min(candidates, key=lambda x: x[1])
        print(f'\n  Fastest: {best_name}  ({best_time:.3f}s)')

        for label, t in candidates:
            if label != best_name and t > 0:
                print(f'    vs {label}: {t / best_time:.2f}x slower')

        print()
        print('  Physical memory (PSS sum):')
        for label, pss in [('distribute()', pss_std),
                            ('collaborative()', pss_collab),
                            ('dump()+load() load-phase', pss_dl)]:
            print(f'    {label:<30}  {fmt_mem(pss)}')
        print()


if __name__ == '__main__':
    main()
