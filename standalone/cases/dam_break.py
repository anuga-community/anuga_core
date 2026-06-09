"""Dam-break validation for the standalone libanuga_sw (ctypes) path.

Runs two checks:

  A. ABI correctness (tight): drive an identical manual RK2 loop through
     (1) the production Cython gpu-ext and (2) the standalone ctypes path.
     Both call the same C gpu_evolve_one_rk2_step, so correct pointer
     marshaling => bit-identical results (rtol ~1e-11).

  B. Physical sanity (loose): compare the ctypes result against ANUGA's own
     mode-1 (CPU OpenMP) full evolve to the same final time.

Run (env must be active):
    source /home/jorge/install/miniconda3/etc/profile.d/conda.sh
    conda activate anuga_env_3.13
    python standalone/cases/dam_break.py
"""

import os
import sys
import numpy as np

import anuga
from anuga import Reflective_boundary

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from anuga_sw import AnugaSW  # noqa: E402

N = 30          # mesh resolution per side
N_STEPS = 40    # number of RK2 steps for the lock-step comparison
FINALTIME = 2.0 # seconds, for the full-evolve physical comparison
LEN = 100.0


def make_domain(name):
    """Build a dam-break domain: wet block on the left, dry-ish on the right."""
    d = anuga.rectangular_cross_domain(N, N, len1=LEN, len2=LEN)
    d.set_flow_algorithm("DE0")
    d.set_low_froude(0)
    d.set_name(name)
    d.set_datadir("/tmp")
    d.store = False

    d.set_quantity("elevation", 0.0)
    d.set_quantity("friction", 0.0)

    def dam(x, y):
        return np.where(x < LEN / 2.0, 4.0, 1.0)

    d.set_quantity("stage", dam)

    Br = Reflective_boundary(d)
    d.set_boundary({"left": Br, "right": Br, "top": Br, "bottom": Br})
    return d


def centroids(domain):
    q = domain.quantities
    return (q["stage"].centroid_values.copy(),
            q["xmomentum"].centroid_values.copy(),
            q["ymomentum"].centroid_values.copy())


def report(tag, ref, test):
    worst = 0.0
    for name, a, b in zip(("stage", "xmom", "ymom"), ref, test):
        linf = np.max(np.abs(a - b))
        denom = np.max(np.abs(a)) or 1.0
        l2 = np.sqrt(np.mean((a - b) ** 2))
        print(f"  [{tag}] {name:5s}  Linf={linf:.3e}  L2={l2:.3e}  rel_Linf={linf/denom:.3e}")
        worst = max(worst, linf)
    return worst


def run_ctypes_path(steps):
    """Drive a fresh domain through the standalone ctypes library (fixed steps)."""
    d = make_domain("db_ctypes")
    d.distribute_to_vertices_and_edges()
    d.update_boundary()
    sw = AnugaSW(d)
    print(f"  ctypes lib: offload={sw.built_with_offload} mpi={sw.built_with_mpi}")
    sw.map_to_device()
    sw.sync_to_device()
    t = 0.0
    for _ in range(steps):
        dt = sw.evolve_one_rk2_step(d.evolve_max_timestep, apply_forcing=1)
        t += dt
    sw.sync_all_from_device()
    sw.close()
    return centroids(d), t


def run_ctypes_evolve(finaltime):
    """Drive the ctypes library to finaltime, mirroring ANUGA's _evolve_base /
    _evolve_one_*_step_c: use the domain's own timestepping method and the same
    max_timestep capping (yieldstep == finaltime)."""
    d = make_domain("db_ctypes_ev")
    d.distribute_to_vertices_and_edges()
    d.update_boundary()
    sw = AnugaSW(d)
    sw.map_to_device()
    sw.sync_to_device()

    method = d.get_timestepping_method()
    step = {
        "euler": sw.evolve_one_euler_step,
        "rk2": sw.evolve_one_rk2_step,
        "rk3": sw.evolve_one_rk3_step,
    }[method]

    t = 0.0
    while t < finaltime - 1e-12:
        max_dt = min(d.evolve_max_timestep, finaltime - t)
        dt = step(max_dt, apply_forcing=1)
        t += dt
    sw.sync_all_from_device()
    sw.close()
    return centroids(d), t, method


def run_cython_gpu_path(steps):
    """Drive a fresh domain through the production Cython gpu-ext, same loop."""
    d = make_domain("db_cyext")
    d.distribute_to_vertices_and_edges()
    d.update_boundary()
    d.set_multiprocessor_mode(2)  # creates gpu_interface (CPU_ONLY build => runs on CPU)
    from anuga.shallow_water.sw_domain_gpu_ext import (
        sync_to_device, sync_all_from_device, evolve_one_rk2_step_gpu,
    )
    gpu_dom = d.gpu_interface.gpu_dom
    sync_to_device(gpu_dom)
    t = 0.0
    for _ in range(steps):
        dt = evolve_one_rk2_step_gpu(gpu_dom, d.evolve_max_timestep, 1)
        t += dt
    sync_all_from_device(gpu_dom)
    return centroids(d), t


def run_anuga_mode1(finaltime):
    """ANUGA's own mode-1 (CPU OpenMP) evolve to finaltime."""
    d = make_domain("db_mode1")
    d.set_multiprocessor_mode(1)
    for _ in d.evolve(yieldstep=finaltime, finaltime=finaltime):
        pass
    return centroids(d)


def main():
    print("== Dam-break validation (standalone libanuga_sw) ==")

    print(f"\nA. ABI correctness: ctypes vs Cython gpu-ext, {N_STEPS} lock-step RK2 steps")
    ct, t_ct = run_ctypes_path(N_STEPS)
    cy, t_cy = run_cython_gpu_path(N_STEPS)
    print(f"  final time: ctypes={t_ct:.6f}s  cython={t_cy:.6f}s  (dt-sum diff={abs(t_ct-t_cy):.2e})")
    worst_a = report("ABI", cy, ct)

    print(f"\nB. Full-evolve vs ANUGA mode-1 (CPU OpenMP), evolve to t={FINALTIME}s")
    cev, t_ev, method = run_ctypes_evolve(FINALTIME)
    m1 = run_anuga_mode1(FINALTIME)
    print(f"  ctypes evolve reached t={t_ev:.6f}s using '{method}' timestepping")
    worst_b = report("PHYS", m1, cev)

    print("\n== Result ==")
    ok_a = worst_a < 1e-9
    ok_b = worst_b < 1e-8   # matches ANUGA's own mode-1 vs mode-2 cross-check tolerance
    print(f"  A (ABI marshaling)        worst Linf={worst_a:.3e}  -> {'PASS' if ok_a else 'FAIL'} (<1e-9)")
    print(f"  B (full evolve vs mode-1) worst Linf={worst_b:.3e}  -> {'PASS' if ok_b else 'FAIL'} (<1e-8)")
    if not (ok_a and ok_b):
        sys.exit(1)
    print("  ALL PASS")


if __name__ == "__main__":
    main()
