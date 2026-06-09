"""Single-GPU dam break with NO ANUGA build required.

Imports only numpy + ctypes (+ meshpy for the unstructured option). The mesh is
generated in Python; the C library builds all geometry and runs on one GPU. This
is the fully self-contained path - it never imports anuga, so nothing here needs
the (meson) ANUGA build.

    export ANUGA_SW_LIB=$PWD/standalone/build_gpu/libanuga_sw.so

    # structured M x N box (exact cell count) -- numpy only, no meshpy:
    python standalone/python/run_meshpy.py 1024 1024            # 1024x1024 cells
    python standalone/python/run_meshpy.py 1024 1024 100 2.0    # len=100, t=2s

    # unstructured (meshpy/Triangle), controlled by max triangle area:
    python standalone/python/run_meshpy.py --meshpy 0.5 100 2.0

The DE0 numerical parameters below are baked-in constants (ANUGA's DE0 defaults),
so no anuga import is needed to configure the solver.
"""
import ctypes
import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from anuga_sw import _find_library  # numpy-only locator; does NOT import anuga

c_void_p, c_int, c_long, c_double = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_double

# ANUGA DE0 defaults (baked in so we don't import anuga to read them)
DE0 = dict(g=9.8, CFL=0.9, H0=1e-5, epsilon=1e-12,
           minimum_allowed_height=1e-12, maximum_allowed_speed=0.0,
           evolve_max_timestep=1000.0,
           beta_w=0.5, beta_w_dry=0.0, beta_uh=0.5, beta_uh_dry=0.0,
           beta_vh=0.5, beta_vh_dry=0.0,
           timestepping_method=0, extrapolate_velocity_second_order=1,
           low_froude=0, optimise_dry_cells=0)


class MeshParams(ctypes.Structure):
    _fields_ = [(k, c_double) for k in
                ("g", "CFL", "H0", "epsilon", "minimum_allowed_height",
                 "maximum_allowed_speed", "evolve_max_timestep", "finaltime",
                 "beta_w", "beta_w_dry", "beta_uh", "beta_uh_dry", "beta_vh", "beta_vh_dry")] + \
               [(k, c_int) for k in
                ("timestepping_method", "extrapolate_velocity_second_order",
                 "low_froude", "optimise_dry_cells")]


def structured_grid(m, n, len1, len2):
    """Structured m x n rectangular mesh: (m+1)(n+1) nodes, 2*m*n CCW triangles."""
    xs = np.linspace(0.0, len1, m + 1)
    ys = np.linspace(0.0, len2, n + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")          # node (i,j) -> i*(n+1)+j
    nodes = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)
    i, j = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    i, j = i.ravel(), j.ravel()
    v00 = i * (n + 1) + j
    v10 = (i + 1) * (n + 1) + j
    v11 = (i + 1) * (n + 1) + (j + 1)
    v01 = i * (n + 1) + (j + 1)
    t1 = np.column_stack([v00, v10, v11])                # CCW
    t2 = np.column_stack([v00, v11, v01])
    tris = np.empty((2 * m * n, 3), dtype=np.int64)
    tris[0::2] = t1
    tris[1::2] = t2
    return nodes, tris


def meshpy_rectangle(length, max_area):
    from meshpy.triangle import MeshInfo, build
    mi = MeshInfo()
    mi.set_points([(0, 0), (length, 0), (length, length), (0, length)])
    mi.set_facets([(0, 1), (1, 2), (2, 3), (3, 0)])
    mesh = build(mi, max_volume=max_area, min_angle=30)
    return (np.array(mesh.points, dtype=np.float64),
            np.array(mesh.elements, dtype=np.int64))


def main():
    args = sys.argv[1:]
    if args and args[0] == "--meshpy":
        max_area = float(args[1]) if len(args) > 1 else 1.0
        length = float(args[2]) if len(args) > 2 else 100.0
        finaltime = float(args[3]) if len(args) > 3 else 2.0
        nodes, tris = meshpy_rectangle(length, max_area)
    else:
        m = int(args[0]) if len(args) > 0 else 100
        n = int(args[1]) if len(args) > 1 else m
        length = float(args[2]) if len(args) > 2 else 100.0
        finaltime = float(args[3]) if len(args) > 3 else 2.0
        nodes, tris = structured_grid(m, n, length, length)
    # centroids -> initial conditions (dam break, flat bed, no friction)
    cx = nodes[tris, 0].mean(axis=1)
    stage = np.where(cx < length / 2.0, 4.0, 1.0).astype(np.float64)
    elevation = np.zeros(tris.shape[0], np.float64)
    friction = np.zeros(tris.shape[0], np.float64)
    print(f"mesh: {nodes.shape[0]} nodes, {tris.shape[0]} triangles, "
          f"length={length}, finaltime={finaltime}")

    lib = ctypes.CDLL(_find_library())
    lib.anuga_build_from_mesh.restype = c_void_p
    lib.anuga_build_from_mesh.argtypes = [
        ctypes.POINTER(c_double), c_long, ctypes.POINTER(c_long), c_long,
        ctypes.POINTER(c_double), ctypes.POINTER(c_double), ctypes.POINTER(c_double),
        ctypes.POINTER(MeshParams)]
    for nm, (a, r) in {
        "anuga_dump_domain": ([c_void_p], c_void_p),
        "anuga_dump_num_elements": ([c_void_p], c_long),
        "anuga_dump_get_owned_stage": ([c_void_p, ctypes.POINTER(c_double)], c_long),
        "anuga_dump_free": ([c_void_p], None),
        "anuga_built_with_offload": ([], c_int),
        "anuga_gpu_available": ([], c_int),
        "anuga_domain_map_to_device": ([c_void_p], c_int),
        "anuga_sync_to_device": ([c_void_p], None),
        "anuga_sync_from_device": ([c_void_p], None),
        "anuga_evolve_one_euler_step": ([c_void_p, c_double, c_int], c_double),
    }.items():
        getattr(lib, nm).argtypes, getattr(lib, nm).restype = a, r

    p = MeshParams(finaltime=finaltime, **DE0)
    dp = lambda x: x.ctypes.data_as(ctypes.POINTER(c_double))
    ld = lib.anuga_build_from_mesh(dp(nodes), nodes.shape[0],
                                   tris.ctypes.data_as(ctypes.POINTER(c_long)), tris.shape[0],
                                   dp(stage), dp(elevation), dp(friction), ctypes.byref(p))
    if not ld:
        print("anuga_build_from_mesh failed"); sys.exit(1)

    print(f"offload={bool(lib.anuga_built_with_offload())} "
          f"gpu_available={bool(lib.anuga_gpu_available())}")
    dom = lib.anuga_dump_domain(ld)
    lib.anuga_domain_map_to_device(dom)
    lib.anuga_sync_to_device(dom)
    t = 0.0
    while t < finaltime - 1e-12:
        t += lib.anuga_evolve_one_euler_step(dom, finaltime - t, 1)
    lib.anuga_sync_from_device(dom)

    n = lib.anuga_dump_num_elements(ld)
    buf = (c_double * n)()
    m = lib.anuga_dump_get_owned_stage(ld, buf)
    s = np.array(buf[:m])
    lib.anuga_dump_free(ld)
    print(f"t={t:.6f}  cells={m}  stage: min={s.min():.6f} max={s.max():.6f} "
          f"mean={s.mean():.6f} sum={s.sum():.6f}")


if __name__ == "__main__":
    main()
