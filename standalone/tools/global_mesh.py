"""Write a global .agm mesh file for the MPI-aware standalone runtime.

Generates a mesh (structured numpy grid, or meshpy/Triangle), partitions the
dual graph with pymetis, sets dam-break ICs, and writes ONE .agm file. The C
mini-app reads it under mpirun and self-distributes (owned + ghost layers) -- so
no anuga and no mpi4py are needed.

    python standalone/tools/global_mesh.py NPARTS M N [length] [finaltime] [outdir]
    python standalone/tools/global_mesh.py --meshpy NPARTS MAXAREA [length] [finaltime] [outdir]

The NPARTS here is only the partition count baked into the file's partition
vector; you still launch with `mpirun -n NPARTS`.
"""
import os
import struct
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "python"))
from run_meshpy import structured_grid, meshpy_rectangle, DE0  # mesh gens + DE0 params

MAGIC = b"ANUGAGM1"
VERSION = 1


def dual_graph(tris, n_nodes):
    """Adjacency list (cell-cell via shared edge) for pymetis."""
    nt = tris.shape[0]
    a = np.empty(3 * nt, np.int64)
    b = np.empty(3 * nt, np.int64)
    cell = np.repeat(np.arange(nt, dtype=np.int64), 3)
    for i in range(3):
        a[i::3] = tris[:, (i + 1) % 3]
        b[i::3] = tris[:, (i + 2) % 3]
    key = np.minimum(a, b) * np.int64(n_nodes) + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    ks, cs = key[order], cell[order]
    same = ks[:-1] == ks[1:]
    c1 = cs[:-1][same]
    c2 = cs[1:][same]
    adj = [[] for _ in range(nt)]
    for x, y in zip(c1.tolist(), c2.tolist()):
        adj[x].append(y)
        adj[y].append(x)
    return adj


def partition(tris, n_nodes, nparts):
    if nparts == 1:
        return np.zeros(tris.shape[0], np.int32)
    import pymetis
    _, parts = pymetis.part_graph(nparts, adjacency=dual_graph(tris, n_nodes))
    return np.asarray(parts, np.int32)


def write_agm(path, nodes, tris, parts, stage, elev, fric, finaltime):
    n_nodes, n_tris = nodes.shape[0], tris.shape[0]
    Hi = [VERSION, n_nodes, n_tris,
          DE0["timestepping_method"], DE0["extrapolate_velocity_second_order"],
          DE0["low_froude"], DE0["optimise_dry_cells"]]
    Hd = [DE0["g"], DE0["CFL"], DE0["H0"], DE0["epsilon"],
          DE0["minimum_allowed_height"], DE0["maximum_allowed_speed"],
          DE0["evolve_max_timestep"], float(finaltime),
          DE0["beta_w"], DE0["beta_w_dry"], DE0["beta_uh"], DE0["beta_uh_dry"],
          DE0["beta_vh"], DE0["beta_vh_dry"]]
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(f"<{len(Hi)}q", *Hi))
        f.write(struct.pack(f"<{len(Hd)}d", *Hd))
        np.ascontiguousarray(nodes, "<f8").tofile(f)
        np.ascontiguousarray(tris, "<i8").tofile(f)
        np.ascontiguousarray(parts, "<i4").tofile(f)
        np.ascontiguousarray(stage, "<f8").tofile(f)
        np.ascontiguousarray(elev, "<f8").tofile(f)
        np.ascontiguousarray(fric, "<f8").tofile(f)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--meshpy":
        nparts = int(args[1]); max_area = float(args[2])
        length = float(args[3]) if len(args) > 3 else 100.0
        finaltime = float(args[4]) if len(args) > 4 else 2.0
        outdir = args[5] if len(args) > 5 else "/tmp/anuga_global"
        nodes, tris = meshpy_rectangle(length, max_area)
    else:
        nparts = int(args[0]); m = int(args[1]); n = int(args[2])
        length = float(args[3]) if len(args) > 3 else 100.0
        finaltime = float(args[4]) if len(args) > 4 else 2.0
        outdir = args[5] if len(args) > 5 else "/tmp/anuga_global"
        nodes, tris = structured_grid(m, n, length, length)

    cx = nodes[tris, 0].mean(axis=1)
    stage = np.where(cx < length / 2.0, 4.0, 1.0).astype(np.float64)
    elev = np.zeros(tris.shape[0], np.float64)
    fric = np.zeros(tris.shape[0], np.float64)
    parts = partition(tris, nodes.shape[0], nparts)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "dam_break.agm")
    write_agm(path, nodes, tris, parts, stage, elev, fric, finaltime)
    counts = np.bincount(parts, minlength=nparts)
    print(f"wrote {path}: {nodes.shape[0]} nodes, {tris.shape[0]} triangles, "
          f"{nparts} parts {counts.tolist()}")


if __name__ == "__main__":
    main()
