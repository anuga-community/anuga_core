"""Single-GPU validation of the meshpy/mesh -> C geometry builder.

Builds a dam break, hands the raw mesh (nodes + triangles) and per-cell initial
state to anuga_build_from_mesh (which computes ALL geometry in C and sets every
boundary edge reflective), runs a single-GPU evolve, and compares cell-by-cell
to ANUGA's own mode-1 evolve on the same mesh and ordering.

Two mesh sources:
  --anuga  : use ANUGA's rectangular_cross mesh (isolates the C geometry port)
  --meshpy : triangulate the same rectangle with meshpy (the target workflow)

    source standalone/env_dgx.sh
    export ANUGA_SW_LIB=$PWD/standalone/build_gpu/libanuga_sw.so
    python standalone/cases/dam_break_mesh.py --anuga
    python standalone/cases/dam_break_mesh.py --meshpy
"""
import ctypes
import os
import sys
import numpy as np

import anuga

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from anuga_sw import _find_library  # noqa: E402

LEN = 100.0
N = 40
FINALTIME = 2.0
c_void_p, c_int, c_long, c_double = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_double


class MeshParams(ctypes.Structure):
    _fields_ = [(k, c_double) for k in
                ("g", "CFL", "H0", "epsilon", "minimum_allowed_height",
                 "maximum_allowed_speed", "evolve_max_timestep", "finaltime",
                 "beta_w", "beta_w_dry", "beta_uh", "beta_uh_dry", "beta_vh", "beta_vh_dry")] + \
               [(k, c_int) for k in
                ("timestepping_method", "extrapolate_velocity_second_order",
                 "low_froude", "optimise_dry_cells")]


def dam(x, y):
    return np.where(x < LEN / 2.0, 4.0, 1.0)


def meshpy_rectangle():
    """Triangulate the LEN x LEN square with meshpy (Triangle)."""
    from meshpy.triangle import MeshInfo, build
    mi = MeshInfo()
    mi.set_points([(0, 0), (LEN, 0), (LEN, LEN), (0, LEN)])
    mi.set_facets([(0, 1), (1, 2), (2, 3), (3, 0)])
    # target area ~ matches N x N x 2 cross triangles
    max_area = (LEN * LEN) / (N * N * 4.0)
    mesh = build(mi, max_volume=max_area, min_angle=30)
    nodes = np.array(mesh.points, dtype=np.float64)
    tris = np.array(mesh.elements, dtype=np.int64)
    return nodes, tris


def build_c(lib, nodes, triangles, stage_cv, elev_cv, fric_cv, params):
    lib.anuga_build_from_mesh.restype = c_void_p
    lib.anuga_build_from_mesh.argtypes = [
        ctypes.POINTER(c_double), c_long, ctypes.POINTER(c_long), c_long,
        ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_double),
        ctypes.POINTER(MeshParams)]
    nodes = np.ascontiguousarray(nodes, np.float64)
    triangles = np.ascontiguousarray(triangles, np.int64)
    stage_cv = np.ascontiguousarray(stage_cv, np.float64)
    elev_cv = np.ascontiguousarray(elev_cv, np.float64)
    fric_cv = np.ascontiguousarray(fric_cv, np.float64)
    dp = lambda a: a.ctypes.data_as(ctypes.POINTER(c_double))
    lp = lambda a: a.ctypes.data_as(ctypes.POINTER(c_long))
    ld = lib.anuga_build_from_mesh(dp(nodes), nodes.shape[0], lp(triangles), triangles.shape[0],
                                   dp(stage_cv), dp(elev_cv), dp(fric_cv), ctypes.byref(params))
    # keep arrays alive
    build_c._keep = (nodes, triangles, stage_cv, elev_cv, fric_cv)
    return ld


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--anuga"
    lib = ctypes.CDLL(_find_library())
    for nm, (a, r) in {
        "anuga_dump_domain": ([c_void_p], c_void_p),
        "anuga_dump_num_elements": ([c_void_p], c_long),
        "anuga_dump_get_owned_stage": ([c_void_p, ctypes.POINTER(c_double)], c_long),
        "anuga_dump_finaltime": ([c_void_p], c_double),
        "anuga_dump_timestepping_method": ([c_void_p], c_int),
        "anuga_dump_free": ([c_void_p], None),
        "anuga_domain_map_to_device": ([c_void_p], c_int),
        "anuga_sync_to_device": ([c_void_p], None),
        "anuga_sync_from_device": ([c_void_p], None),
        "anuga_evolve_one_euler_step": ([c_void_p, c_double, c_int], c_double),
        "anuga_evolve_one_rk2_step": ([c_void_p, c_double, c_int], c_double),
    }.items():
        getattr(lib, nm).argtypes, getattr(lib, nm).restype = a, r

    if mode == "--meshpy":
        nodes, tris = meshpy_rectangle()
        d = anuga.Domain(nodes, tris)
        d.set_flow_algorithm("DE0"); d.set_low_froude(0)
        d.set_name("db_mp"); d.set_datadir("/tmp"); d.store = False
        d.set_quantity("elevation", 0.0); d.set_quantity("friction", 0.0)
        d.set_quantity("stage", dam)
        Br = anuga.Reflective_boundary(d)
        d.set_boundary({t: Br for t in d.get_boundary_tags()})
        print(f"meshpy mesh: {nodes.shape[0]} nodes, {tris.shape[0]} triangles")
    else:
        d = anuga.rectangular_cross_domain(N, N, len1=LEN, len2=LEN)
        d.set_flow_algorithm("DE0"); d.set_low_froude(0)
        d.set_name("db_mesh"); d.set_datadir("/tmp"); d.store = False
        d.set_quantity("elevation", 0.0); d.set_quantity("friction", 0.0)
        d.set_quantity("stage", dam)
        Br = anuga.Reflective_boundary(d)
        d.set_boundary({t: Br for t in d.get_boundary_tags()})
        nodes = np.array(d.nodes, np.float64)
        tris = np.array(d.triangles, np.int64)
        print(f"anuga mesh: {nodes.shape[0]} nodes, {tris.shape[0]} triangles")

    stage0 = d.quantities["stage"].centroid_values.copy()
    elev0 = d.quantities["elevation"].centroid_values.copy()
    fric0 = d.quantities["friction"].centroid_values.copy()

    p = MeshParams(
        g=d.g, CFL=d.CFL, H0=d.H0, epsilon=d.epsilon,
        minimum_allowed_height=d.minimum_allowed_height,
        maximum_allowed_speed=d.maximum_allowed_speed,
        evolve_max_timestep=d.evolve_max_timestep, finaltime=FINALTIME,
        beta_w=d.beta_w, beta_w_dry=d.beta_w_dry, beta_uh=d.beta_uh,
        beta_uh_dry=d.beta_uh_dry, beta_vh=d.beta_vh, beta_vh_dry=d.beta_vh_dry,
        timestepping_method={"euler": 0, "rk2": 1}.get(d.get_timestepping_method(), 0),
        extrapolate_velocity_second_order=d.extrapolate_velocity_second_order,
        low_froude=d.low_froude, optimise_dry_cells=getattr(d, "optimise_dry_cells", 0))

    ld = build_c(lib, nodes, tris, stage0, elev0, fric0, p)
    if not ld:
        print("anuga_build_from_mesh returned NULL"); sys.exit(1)
    dom = lib.anuga_dump_domain(ld)
    lib.anuga_domain_map_to_device(dom)
    lib.anuga_sync_to_device(dom)
    method = lib.anuga_dump_timestepping_method(ld)
    step = lib.anuga_evolve_one_rk2_step if method == 1 else lib.anuga_evolve_one_euler_step
    t = 0.0
    while t < FINALTIME - 1e-12:
        t += step(dom, FINALTIME - t, 1)
    lib.anuga_sync_from_device(dom)
    nC = lib.anuga_dump_num_elements(ld)
    buf = (c_double * nC)()
    m = lib.anuga_dump_get_owned_stage(ld, buf)
    c_stage = np.array(buf[:m])
    lib.anuga_dump_free(ld)

    # ANUGA reference on the same mesh
    d.set_multiprocessor_mode(1)
    for _ in d.evolve(yieldstep=FINALTIME, finaltime=FINALTIME):
        pass
    ref = d.quantities["stage"].centroid_values

    if mode == "--meshpy":
        # different triangulation order vs nothing to compare elementwise across libs;
        # compare permutation-invariant + integral
        linf = float(np.max(np.abs(np.sort(c_stage) - np.sort(ref))))
        print(f"  sorted-Linf vs ANUGA = {linf:.3e}, C sum={c_stage.sum():.8g}, ref sum={ref.sum():.8g}")
        ok = linf < 1e-8
    else:
        linf = float(np.max(np.abs(c_stage - ref)))
        print(f"  elementwise Linf vs ANUGA = {linf:.3e}, C sum={c_stage.sum():.8g}, ref sum={ref.sum():.8g}")
        ok = linf < 1e-8
    print(f"  -> {'PASS' if ok else 'FAIL'} (<1e-8)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
