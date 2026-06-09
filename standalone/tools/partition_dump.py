"""Offline partitioner + .adm writer for the standalone libanuga_sw runtime.

Runs SERIALLY (no MPI) with full ANUGA, partitions a domain into N pieces and
writes one portable .adm file per rank. The target then runs libanuga_sw under
the system MPI (C owns MPI_COMM_WORLD) reading these files - no mpi4py/conda
needed at runtime. Format: see standalone/include/anuga_dump.h.

Usage (env active):
    python standalone/tools/partition_dump.py            # writes dam break, np=1,2,4
or import and call dump_case(...).
"""

import os
import struct
import tempfile
import numpy as np

import anuga
from anuga.parallel.sequential_distribute import (
    sequential_distribute_dump, sequential_distribute_load_pickle_file,
)

MAGIC = b"ANUGADM1"
VERSION = 1
_METHOD = {"euler": 0, "rk2": 1, "rk3": 2, "ader2": 3}


def _i64(a):
    return np.ascontiguousarray(a).astype("<i8").ravel()


def _f64(a):
    return np.ascontiguousarray(a).astype("<f8").ravel()


def _i32(a):
    return np.ascontiguousarray(a).astype("<i4").ravel()


def _boundary_edges(domain, tag, btype, spec):
    """Return (boundary_indices, vol_ids, edge_ids) int32 arrays for one tag."""
    seg = domain.tag_boundary_cells.get(tag, None)
    if seg is None or len(seg) == 0:
        return None
    ids = np.array(seg, dtype=np.intc)
    return (_i32(ids),
            _i32(domain.boundary_cells[ids]),
            _i32(domain.boundary_edges[ids]))


def _collect_boundaries(domain, spec):
    """Group boundary edges by type using spec {tag: 'reflective' |
    ('dirichlet',(s,xm,ym)) | 'transmissive'}. Tags not in spec (e.g. 'ghost')
    are skipped."""
    groups = {"reflective": [], "dirichlet": [], "transmissive": []}
    dvals = (0.0, 0.0, 0.0)
    use_centroid = 1 if getattr(domain, "centroid_transmissive_bc", False) else 0
    for tag in domain.tag_boundary_cells.keys():
        bc = spec.get(tag)
        if bc is None:
            continue
        kind = bc[0] if isinstance(bc, tuple) else bc
        arrs = _boundary_edges(domain, tag, kind, spec)
        if arrs is None:
            continue
        groups[kind].append(arrs)
        if kind == "dirichlet":
            dvals = tuple(float(v) for v in bc[1])

    def cat(lst):
        if not lst:
            z = np.zeros(0, np.intc)
            return z, z, z
        return (np.concatenate([a[0] for a in lst]),
                np.concatenate([a[1] for a in lst]),
                np.concatenate([a[2] for a in lst]))

    return cat(groups["reflective"]), cat(groups["dirichlet"]), cat(groups["transmissive"]), dvals, use_centroid


def _write_adm(path, d, spec, finaltime, yieldstep, rank, nprocs):
    d.distribute_to_vertices_and_edges()
    d.update_boundary()

    n = int(d.number_of_elements)
    nb = int(d.boundary_length)
    q = d.quantities

    # halo
    my = rank
    fs = getattr(d, "full_send_dict", {}) or {}
    gr = getattr(d, "ghost_recv_dict", {}) or {}
    neigh = [p for p in fs if p != my]
    nbr = _i32(neigh) if neigh else np.zeros(0, np.intc)
    sc = _i32([len(fs[p][0]) for p in neigh]) if neigh else np.zeros(0, np.intc)
    rc = _i32([len(gr[p][0]) for p in neigh]) if neigh else np.zeros(0, np.intc)
    fsend = _i32(np.concatenate([np.asarray(fs[p][0]) for p in neigh])) if neigh else np.zeros(0, np.intc)
    frecv = _i32(np.concatenate([np.asarray(gr[p][0]) for p in neigh])) if neigh else np.zeros(0, np.intc)

    refl, diri, tran, dvals, use_centroid = _collect_boundaries(d, spec)

    Hi = [n, nb, rank, nprocs,
          int(getattr(d, "optimise_dry_cells", 0)),
          int(d.extrapolate_velocity_second_order),
          int(d.low_froude),
          int(getattr(d, "timestep_fluxcalls", 1)),
          _METHOD[d.get_timestepping_method()],
          len(neigh), int(fsend.size), int(frecv.size),
          int(refl[0].size), int(diri[0].size), int(tran[0].size),
          int(use_centroid)]
    fft = getattr(d, "fixed_flux_timestep", None)
    Hd = [d.epsilon, d.H0, d.g, d.evolve_max_timestep,
          float(getattr(d, "evolve_min_timestep", 0.0)),
          d.minimum_allowed_height, d.maximum_allowed_speed,
          d.beta_w, d.beta_w_dry, d.beta_uh, d.beta_uh_dry, d.beta_vh, d.beta_vh_dry,
          d.CFL, float(fft) if fft else -1.0,
          float(finaltime), float(yieldstep),
          dvals[0], dvals[1], dvals[2]]

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<q", VERSION))
        f.write(struct.pack("<16q", *Hi))
        f.write(struct.pack("<20d", *Hd))
        # int64 mesh
        for a in (d.neighbours, d.neighbour_edges, d.surrogate_neighbours,
                  d.number_of_boundaries, d.tri_full_flag):
            _i64(a).tofile(f)
        # double geometry
        for a in (d.normals, d.edgelengths, d.radii, d.areas,
                  d.centroid_coordinates, d.edge_coordinates):
            _f64(a).tofile(f)
        # double quantities (centroid, edge) x5 + friction centroid
        for name in ("stage", "xmomentum", "ymomentum", "elevation", "height"):
            _f64(q[name].centroid_values).tofile(f)
            _f64(q[name].edge_values).tofile(f)
        _f64(q["friction"].centroid_values).tofile(f)
        # int32 halo
        for a in (nbr, sc, rc, fsend, frecv):
            _i32(a).tofile(f)
        # int32 boundaries
        for grp in (refl, diri, tran):
            for a in grp:
                _i32(a).tofile(f)


def dump_case(make_full, spec, nprocs, outdir, name, finaltime, yieldstep):
    os.makedirs(outdir, exist_ok=True)
    full = make_full()
    full.set_name(name)
    if nprocs == 1:
        path = os.path.join(outdir, f"{name}_P1_0.adm")
        _write_adm(path, full, spec, finaltime, yieldstep, 0, 1)
        print(f"wrote {path}")
        return
    pdir = tempfile.mkdtemp()
    sequential_distribute_dump(full, nprocs, partition_dir=pdir)
    for p in range(nprocs):
        pkl = os.path.join(pdir, f"{name}_P{nprocs}_{p}.pickle")
        sub = sequential_distribute_load_pickle_file(pkl, nprocs)
        path = os.path.join(outdir, f"{name}_P{nprocs}_{p}.adm")
        _write_adm(path, sub, spec, finaltime, yieldstep, p, nprocs)
        print(f"wrote {path}")


# --------------------------------------------------------------------------
def _dam_break(N=40, LEN=100.0):
    d = anuga.rectangular_cross_domain(N, N, len1=LEN, len2=LEN)
    d.set_flow_algorithm("DE0")
    d.set_low_froude(0)
    d.set_datadir("/tmp")
    d.store = False
    d.set_quantity("elevation", 0.0)
    d.set_quantity("friction", 0.0)
    d.set_quantity("stage", lambda x, y: np.where(x < LEN / 2.0, 4.0, 1.0))
    Br = anuga.Reflective_boundary(d)
    d.set_boundary({"left": Br, "right": Br, "top": Br, "bottom": Br})
    return d


if __name__ == "__main__":
    import sys
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/anuga_adm"
    spec = {t: "reflective" for t in ("left", "right", "top", "bottom", "exterior")}
    for np_ in (1, 2, 4):
        dump_case(lambda: _dam_break(), spec, np_, outdir, "dam_break",
                  finaltime=2.0, yieldstep=2.0)
    print("done ->", outdir)
