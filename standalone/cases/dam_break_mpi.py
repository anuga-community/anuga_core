"""Multi-rank validation for the standalone libanuga_sw (ctypes + MPI) path.

In a single mpirun job it partitions a dam break across ranks and runs the SAME
evolve two ways, then compares per-rank owned (full) cells:

  A. ABI+halo correctness (tight): production parallel mode-2 (Cython gpu-ext)
     vs standalone ctypes path. Both do the MPI dt-allreduce and ghost exchange
     in C, so a correct halo/comm plumbing => bit-identical per rank.

Requires a GPU+MPI build of libanuga_sw:
    cmake -S standalone -B standalone/build_gpu_mpi -DCMAKE_C_COMPILER=nvc \
          -DCMAKE_BUILD_TYPE=Release -DANUGA_OFFLOAD=ON -DANUGA_MPI=ON
Run:
    source standalone/env_dgx.sh
    export ANUGA_SW_LIB=$PWD/standalone/build_gpu_mpi/libanuga_sw.so
    mpirun -n 2 python standalone/cases/dam_break_mpi.py
"""

import os
import sys
import numpy as np
from mpi4py import MPI

import anuga
from anuga import Reflective_boundary

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from anuga_sw import AnugaSW  # noqa: E402

N = 40
LEN = 100.0
FINALTIME = 2.0

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nproc = comm.Get_size()


def make_full():
    d = anuga.rectangular_cross_domain(N, N, len1=LEN, len2=LEN)
    d.set_flow_algorithm("DE0")
    d.set_low_froude(0)
    d.set_name("db_mpi")
    d.set_datadir("/tmp")
    d.store = False
    d.set_quantity("elevation", 0.0)
    d.set_quantity("friction", 0.0)
    d.set_quantity("stage", lambda x, y: np.where(x < LEN / 2.0, 4.0, 1.0))
    Br = Reflective_boundary(d)
    d.set_boundary({"left": Br, "right": Br, "top": Br, "bottom": Br})
    return d


def distributed_domain():
    full = make_full() if myid == 0 else None
    d = anuga.distribute(full)
    # Re-bind boundary objects on the sub-domain (distribute keeps the tags but
    # GPU mode needs concrete boundary objects). 'ghost' is an internal tag.
    Br = Reflective_boundary(d)
    bm = {tag: (None if tag == "ghost" else Br) for tag in d.boundary_map.keys()}
    d.set_boundary(bm)
    return d


def owned(domain):
    """Centroid values of owned (full) cells only."""
    m = domain.tri_full_flag == 1
    q = domain.quantities
    return (q["stage"].centroid_values[m].copy(),
            q["xmomentum"].centroid_values[m].copy(),
            q["ymomentum"].centroid_values[m].copy())


def run_reference():
    """Production parallel mode-2 (Cython gpu-ext) evolve to FINALTIME."""
    d = distributed_domain()
    d.set_multiprocessor_mode(2)
    for _ in d.evolve(yieldstep=FINALTIME, finaltime=FINALTIME):
        pass
    from anuga.shallow_water.sw_domain_gpu_ext import sync_from_device
    sync_from_device(d.gpu_interface.gpu_dom)
    return owned(d)


def run_ctypes():
    """Standalone ctypes + MPI evolve to FINALTIME, same step sequence."""
    d = distributed_domain()
    d.distribute_to_vertices_and_edges()
    d.update_boundary()
    sw = AnugaSW(d, comm=comm)
    sw.map_to_device()
    sw.sync_to_device()
    method = d.get_timestepping_method()
    step = {"euler": sw.evolve_one_euler_step,
            "rk2": sw.evolve_one_rk2_step,
            "rk3": sw.evolve_one_rk3_step}[method]
    t = 0.0
    while t < FINALTIME - 1e-12:
        dt = step(min(d.evolve_max_timestep, FINALTIME - t), apply_forcing=1)
        t += dt
        # Mirror _evolve_one_euler_step_c: post-step ghost exchange. The euler C
        # kernel (unlike rk2/rk3/ader) does NOT exchange ghosts internally.
        if nproc > 1:
            sw.exchange_ghosts()
    sw.sync_all_from_device()
    sw.close()
    return owned(d), t, method


def main():
    if myid == 0:
        print(f"== MPI dam-break validation, {nproc} ranks ==")

    ref = run_reference()
    test, t_end, method = run_ctypes()

    worst = 0.0
    for name, a, b in zip(("stage", "xmom", "ymom"), ref, test):
        linf = float(np.max(np.abs(a - b))) if a.size else 0.0
        worst = max(worst, linf)
    worst_global = comm.allreduce(worst, op=MPI.MAX)
    ncells = comm.allreduce(ref[0].size, op=MPI.SUM)

    if myid == 0:
        print(f"  timestepping='{method}'  reached t={t_end:.6f}s  total owned cells={ncells}")
        print(f"  ctypes+MPI vs production parallel mode-2: worst Linf = {worst_global:.3e}")
        ok = worst_global < 1e-9
        print(f"  -> {'PASS' if ok else 'FAIL'} (<1e-9)")
        if not ok:
            comm.Abort(1)
    anuga.finalize()


if __name__ == "__main__":
    main()
