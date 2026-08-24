#!/usr/bin/env python3
"""Generate a synthetic river-basin mesh with ANUGA's own mesher and dump it
for the standalone benchmark.

Why this exists: the miniapp's rectangular-cross generator is layout-identical
to a structured grid, and measured locality bounds show up to 2.4x between
grid ordering and random ordering.  This script produces what production
actually runs: a Triangle-generated unstructured mesh with variable
resolution (fine along a meandering channel, coarse on the floodplain) in the
element order ANUGA really emits -- plus terrain and a flood-release initial
condition, all frozen into one binary file the C benchmark loads with no
Python anywhere near the timed loop.

    python tools/make_basin_mesh.py --target 2000000 --out build/basin_2m.msh

Scenario: a 20 km x 10 km valley sloping toward the outlet, a sinusoidal
channel carved down its middle, a full reservoir in the headwater released at
t=0 over a baseflow-wetted channel and dry floodplain.
"""

import argparse
import struct
import sys

import numpy as np

MAGIC = b"ANUGAMSH"
VERSION = 1
# magic, version, pad, n_nodes, n_tris, n_bdry
HDR = "<8sii3q"

LEN_X = 20000.0   # m, downstream
LEN_Y = 10000.0

# Channel centreline: y_c(x) = LEN_Y/2 + A sin(2 pi x / LAMBDA)
MEANDER_A = 1500.0
MEANDER_L = 8000.0
CHANNEL_W = 220.0      # carved width
CHANNEL_D = 6.0        # carved depth
VALLEY_SLOPE = 0.004   # downstream fall: 80 m over 20 km
SIDE_SLOPE = 0.008     # valley cross-slope

RESERVOIR_X = 1800.0   # headwater pool centre
RESERVOIR_R = 1200.0
RESERVOIR_STAGE_ABOVE = 8.0   # m above local bed: the flood volume
BASEFLOW_DEPTH = 1.0          # channel wet at baseflow; floodplain dry


def y_channel(x):
    return LEN_Y / 2.0 + MEANDER_A * np.sin(2.0 * np.pi * x / MEANDER_L)


def bed(x, y):
    dy = np.abs(y - y_channel(x))
    z = VALLEY_SLOPE * (LEN_X - x) + SIDE_SLOPE * dy
    # smooth parabolic channel carve
    carve = CHANNEL_D * np.maximum(0.0, 1.0 - (dy / CHANNEL_W) ** 2)
    return z - carve


def stage0(x, y):
    z = bed(x, y)
    # baseflow: water only where the carve holds it
    channel_stage = bed(x, y_channel(x)) + BASEFLOW_DEPTH
    s = np.maximum(z, np.where(np.abs(y - y_channel(x)) < CHANNEL_W,
                               channel_stage, z))
    # the reservoir: a full pool about to let go
    r = np.hypot(x - RESERVOIR_X, y - y_channel(RESERVOIR_X))
    pool = bed(RESERVOIR_X, y_channel(RESERVOIR_X)) + RESERVOIR_STAGE_ABOVE
    return np.where(r < RESERVOIR_R, np.maximum(s, pool), s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=2_000_000,
                    help="approximate triangle count (default 2M)")
    ap.add_argument("--channel-refine", type=float, default=8.0,
                    help="floodplain/channel triangle-area ratio (default 8)")
    ap.add_argument("--still", type=float, default=None, metavar="LEVEL",
                    help="ignore the flood IC; still water at constant stage "
                         "LEVEL (well-balance testing on the real mesh)")
    ap.add_argument("--out", default="build/basin.msh")
    args = ap.parse_args(argv)

    import anuga

    # Split the area budget: channel band gets triangles channel-refine x
    # smaller than the floodplain.
    band_frac = 2.0 * CHANNEL_W * 3.0 / LEN_Y   # refined band ~3 channel widths
    area_total = LEN_X * LEN_Y
    coarse = area_total / args.target * (1.0 - band_frac + band_frac * args.channel_refine)
    fine = coarse / args.channel_refine

    bounding = [[0.0, 0.0], [LEN_X, 0.0], [LEN_X, LEN_Y], [0.0, LEN_Y]]
    tags = {"bottom": [0], "right": [1], "top": [2], "left": [3]}

    # Interior refinement polygon: a band following the meander
    xs = np.linspace(0.0, LEN_X, 200)
    half = 3.0 * CHANNEL_W
    upper = [[float(x), float(min(y_channel(x) + half, LEN_Y - 1.0))] for x in xs]
    lower = [[float(x), float(max(y_channel(x) - half, 1.0))] for x in xs[::-1]]
    channel_poly = upper + lower

    print(f"meshing: coarse area {coarse:.1f} m^2, channel area {fine:.1f} m^2",
          flush=True)
    domain = anuga.create_domain_from_regions(
        bounding, boundary_tags=tags,
        maximum_triangle_area=coarse,
        interior_regions=[(channel_poly, fine)],
        verbose=False)

    nodes = np.ascontiguousarray(domain.get_nodes(absolute=True), dtype=np.float64)
    tris = np.ascontiguousarray(domain.triangles, dtype=np.int64)
    n_nodes, n_tris = len(nodes), len(tris)

    bdry = sorted(domain.boundary.keys())     # [(vol, edge)] in ANUGA's order
    bt = np.array([v for v, e in bdry], dtype=np.int64)
    be = np.array([e for v, e in bdry], dtype=np.int64)

    zb = bed(nodes[:, 0], nodes[:, 1]).astype(np.float64)
    if args.still is not None:
        # A truly at-rest IC is a CONSTANT stage plane -- max(bed, L) per node
        # tilts the free surface inside shoreline cells and that water then
        # legitimately flows.  The solver clamps height to zero where bed > L.
        st = np.full_like(zb, args.still)
    else:
        st = stage0(nodes[:, 0], nodes[:, 1]).astype(np.float64)

    with open(args.out, "wb") as fp:
        fp.write(struct.pack(HDR, MAGIC, VERSION, 0, n_nodes, n_tris, len(bt)))
        fp.write(nodes.tobytes())
        fp.write(tris.tobytes())
        fp.write(np.stack([bt, be], axis=1).astype(np.int64).tobytes())
        fp.write(zb.tobytes())
        fp.write(st.tobytes())

    wet = np.count_nonzero(st > zb + 1e-9)
    print(f"wrote {args.out}: {n_tris} triangles, {n_nodes} nodes, "
          f"{len(bt)} boundary edges, {wet}/{n_nodes} wet nodes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
