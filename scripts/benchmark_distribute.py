"""Benchmark distribute() vs distribute_collaborative().

Runs both functions on a mesh and reports wall-clock time (max across
all ranks) for each.  Run with:

    mpiexec -np <N> python scripts/benchmark_distribute.py [--reps <R>] [options]

Options
-------
--reps R      Number of timed repetitions per function (default: 3).
--mesh PATH   Path to a .tsh mesh file (uses merimbula by default).
--size M      Use a synthetic M×M rectangular mesh (overrides --mesh).
              Each cell is split into 4 triangles, giving 4*M*M triangles.
"""

import argparse
import os
import statistics

from mpi4py import MPI

import anuga
from anuga.parallel.parallel_api import distribute, distribute_collaborative
from anuga import create_domain_from_file
from anuga.utilities.system_tools import get_pathname_from_package

comm  = MPI.COMM_WORLD
myid  = comm.Get_rank()
nproc = comm.Get_size()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--mesh', type=str, default=None)
    p.add_argument('--size', type=int, default=None,
                   help='Grid size M for synthetic M×M mesh (~4*M*M triangles)')
    return p.parse_args()


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


def time_distribute(fn, mesh_filename, grid_size, reps):
    """Return list of wall-clock times (seconds) for `reps` calls to fn."""
    times = []
    for _ in range(reps):
        comm.Barrier()
        domain = make_domain(mesh_filename, grid_size)
        comm.Barrier()
        t0 = MPI.Wtime()
        pd = fn(domain)
        t1 = MPI.Wtime()
        elapsed = t1 - t0
        # Reduce to max across ranks (slowest rank determines wall time)
        wall = comm.reduce(elapsed, op=MPI.MAX, root=0)
        if myid == 0:
            times.append(wall)
        del pd
    return times


def fmt(times):
    if len(times) == 1:
        return f'{times[0]:.3f}s'
    med = statistics.median(times)
    mn  = min(times)
    mx  = max(times)
    return f'{med:.3f}s  (min {mn:.3f}  max {mx:.3f})'


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
        print(f'{"="*62}')

    comm.Barrier()

    times_std    = time_distribute(distribute,              mesh_filename, grid_size, args.reps)
    times_collab = time_distribute(distribute_collaborative, mesh_filename, grid_size, args.reps)

    if myid == 0:
        print(f'\n  distribute()               {fmt(times_std)}')
        print(f'  distribute_collaborative()  {fmt(times_collab)}')

        med_std    = statistics.median(times_std)
        med_collab = statistics.median(times_collab)
        if med_collab > 0:
            speedup = med_std / med_collab
            print(f'\n  Speedup: {speedup:.2f}x  (distribute / collaborative)')
        print()


if __name__ == '__main__':
    main()
