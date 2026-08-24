#!/usr/bin/env python3
"""Run the miniapp's benchmark case through the full ANUGA stack and write a
snapshot the standalone binary can be checked against.

The point is to prove the miniapp sets up the *same problem* ANUGA does: the
mesh, geometry, initial conditions and boundary structures are rebuilt here by
ANUGA itself, and the timestep is the same C entry point
(``evolve_one_rk2_step_gpu``) the miniapp calls.  A clean match means any later
difference is a kernel change, not a harness artefact.

    python tools/anuga_reference.py --nx 100 --ny 100 --steps 30 --out ref.bin
    ./bin/bench_gpu --nx 100 --ny 100 --steps 25 --warmup 5 --check ref.bin

Note the miniapp's total step count is ``warmup + steps``; this script's
``--steps`` is that total.
"""

import argparse
import struct
import sys

import numpy as np

SNAP_MAGIC = b"ANUGASNP"
SNAP_VERSION = 1
# char[8] + int32 + int32 + int64*5 + double*2   (naturally aligned, no padding)
SNAP_FMT = "<8sii5qdd"

CASES = {"dam": 0, "dambumps": 1, "lake": 2}

# Must stay identical to bed_value()/stage_value() in standalone/src/setup.c.
_CX = np.array([0.30, 0.55, 0.70, 0.45, 0.85])
_CY = np.array([0.35, 0.65, 0.25, 0.85, 0.55])
_AMP = np.array([6.0, 4.0, 5.0, 3.0, 7.0])
_RAD = np.array([0.08, 0.06, 0.05, 0.07, 0.05])


def bed_value(x, y, args):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if args.case == "dam":
        return np.zeros_like(x)

    u = x / args.lenx
    v = y / args.leny
    z = 2.0 * u
    for i in range(5):
        du = u - _CX[i]
        dv = v - _CY[i]
        z = z + _AMP[i] * np.exp(-(du * du + dv * dv) / (2.0 * _RAD[i] * _RAD[i]))
    return z


def stage_value(x, y, z, args):
    x = np.asarray(x, dtype=float)
    if args.case == "dam":
        return np.where(x < 0.5 * args.lenx, args.dam, args.water)
    if args.case == "dambumps":
        return np.maximum(z, np.where(x < 0.5 * args.lenx, args.dam, args.water))
    return np.maximum(z, args.water)


def build_domain(args):
    import anuga
    from anuga.abstract_2d_finite_volumes.mesh_factory import rectangular_cross

    points, vertices, boundary = rectangular_cross(
        args.nx, args.ny, len1=args.lenx, len2=args.leny
    )
    domain = anuga.Domain(points, vertices, boundary)
    domain.set_flow_algorithm("DE1")          # rk2, CFL 1.0, beta_* = 1.0
    domain.set_timestepping_method(2)
    domain.set_cfl(args.cfl)
    domain.set_name("bench_reference")
    domain.set_quantities_to_be_stored(None)  # no sww output

    # Elevation: evaluated at the vertices, then averaged down to edges and
    # centroids -- the same order of operations as bench_domain_build().
    vc = domain.get_vertex_coordinates()      # (3N, 2)
    zv = bed_value(vc[:, 0], vc[:, 1], args).reshape(-1, 3)
    elev = domain.quantities["elevation"]
    elev.set_values(zv, location="vertices")

    # Conserved quantities start from centroid values only.
    cc = domain.centroid_coordinates
    zc = elev.centroid_values
    w = stage_value(cc[:, 0], cc[:, 1], zc, args)
    domain.quantities["stage"].set_values(w, location="centroids")
    domain.quantities["xmomentum"].set_values(0.0, location="centroids")
    domain.quantities["ymomentum"].set_values(0.0, location="centroids")
    domain.quantities["height"].set_values(np.maximum(w - zc, 0.0), location="centroids")
    domain.set_quantity("friction", args.manning)

    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({"left": Br, "right": Br, "top": Br, "bottom": Br})
    return domain


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--ny", type=int, default=200)
    ap.add_argument("--steps", type=int, default=105,
                    help="total RK2 steps (miniapp warmup + steps)")
    ap.add_argument("--case", choices=sorted(CASES), default="dam")
    ap.add_argument("--lenx", type=float, default=1000.0)
    ap.add_argument("--leny", type=float, default=1000.0)
    ap.add_argument("--manning", type=float, default=0.03)
    ap.add_argument("--water", type=float, default=5.0)
    ap.add_argument("--dam", type=float, default=10.0)
    ap.add_argument("--cfl", type=float, default=1.0)
    ap.add_argument("--mode", type=int, default=2, choices=(1, 2),
                    help="1 = legacy CPU path, 2 = unified GPU/OpenMP path")
    ap.add_argument("--out", default="reference.bin")
    args = ap.parse_args(argv)

    domain = build_domain(args)
    domain.set_multiprocessor_mode(args.mode)

    t = 0.0
    dt = 0.0
    if args.mode == 2:
        from anuga.shallow_water.sw_domain_gpu_ext import (
            evolve_one_rk2_step_gpu, sync_from_device)

        gpu_dom = domain.gpu_interface.gpu_dom
        domain.gpu_interface.ensure_boundaries_initialized()
        max_dt = domain.evolve_max_timestep
        for _ in range(args.steps):
            dt = evolve_one_rk2_step_gpu(gpu_dom, max_dt, 1)
            t += dt
        sync_from_device(gpu_dom)
    else:
        for _ in range(args.steps):
            domain.evolve_one_rk2_step(domain.evolve_max_timestep,
                                       domain.evolve_max_timestep)
            dt = domain.timestep
            t += dt

    q = domain.quantities
    fields = [
        q["stage"].centroid_values,
        q["xmomentum"].centroid_values,
        q["ymomentum"].centroid_values,
        q["height"].centroid_values,
        q["elevation"].centroid_values,
    ]
    n = len(fields[0])

    header = struct.pack(SNAP_FMT, SNAP_MAGIC, SNAP_VERSION, CASES[args.case],
                         n, domain.boundary_length, args.nx, args.ny,
                         args.steps, t, dt)
    with open(args.out, "wb") as fp:
        fp.write(header)
        for f in fields:
            fp.write(np.ascontiguousarray(f, dtype=np.float64).tobytes())

    print(f"ANUGA reference: {n} triangles, {args.steps} rk2 steps, "
          f"t = {t:.9g}, last dt = {dt:.6g}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
