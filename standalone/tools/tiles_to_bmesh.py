#!/usr/bin/env python
"""
Convert ANUGA NetCDF .msh files -- one tile each from cdac_script/mesh_tile.py,
or the merged mesh from merge_tiles.py -- into the miniapp's ANUGAMSH binary
(nodes, triangles, boundary pairs, bed, stage per node) with a synthetic,
analytic bed and initial stage evaluated at the node coordinates.

Tiles:   python tools/tiles_to_bmesh.py --tiles mesh_tiles_10000sqm --out build/tiles10k
         -> build/tiles10k/tile_NNNNN.bmesh + build/tiles10k/index.txt
Merged:  python tools/tiles_to_bmesh.py --merged mesh_file/10000sqm.msh --out build/10k.bmesh

Both round vertex coordinates to 6 decimals exactly as merge_tiles.py does,
so a tiled run and the merged run see bit-identical triangle geometry, and
bed/stage are pure functions of the (rounded) coordinates, so they agree
too.  Canonical triangle id of (tile t, local k) = offset(t) + k, where the
offsets accumulate over tiles in id order -- the order merge_tiles.py
concatenates them in.

Synthetic case ("delta"): a plane rising northward with gentle hills, dry
everywhere except a 10 km lake released dam-break style near the middle of
the southern half.  Wet fraction starts at ~1% and grows -- the load-balance
demonstration case for the tiled partition.
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

DECIMALS = 6


def bed(x, y):
    return 4.0e-4 * y + 30.0 * np.sin(x / 9000.0) * np.sin(y / 7000.0)


def stage0(x, y, cx=90000.0, cy=40000.0, r=10000.0, depth=15.0):
    zb = bed(x, y)
    st = zb.copy()
    inside = (x - cx) ** 2 + (y - cy) ** 2 < r * r
    st[inside] = np.maximum(zb[inside], bed(np.array([cx]), np.array([cy]))[0] + depth)
    return st


def write_bmesh(path, nodes, tris, nbrs):
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    tris = np.ascontiguousarray(tris, dtype=np.int64)
    bt, be = np.nonzero(np.asarray(nbrs, dtype=np.int64) < 0)
    zb = bed(nodes[:, 0], nodes[:, 1]).astype(np.float64)
    st = stage0(nodes[:, 0], nodes[:, 1]).astype(np.float64)
    with open(path, 'wb') as fp:
        fp.write(b'ANUGAMSH')
        fp.write(struct.pack('<ii', 1, 0))
        fp.write(struct.pack('<qqq', len(nodes), len(tris), len(bt)))
        fp.write(nodes.tobytes())
        fp.write(tris.tobytes())
        fp.write(np.column_stack([bt, be]).astype(np.int64).tobytes())
        fp.write(zb.tobytes())
        fp.write(st.tobytes())
    return len(nodes), len(tris)


def load(path):
    from anuga.load_mesh.loadASCII import import_mesh_file
    m = import_mesh_file(path)
    v = np.round(np.asarray(m['vertices'], float), DECIMALS)
    return v, np.asarray(m['triangles'], np.int64), np.asarray(m['triangle_neighbors'], np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles', help='directory of tile_NNNNN.msh')
    ap.add_argument('--merged', help='one merged .msh')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.merged:
        v, t, n = load(a.merged)
        nn, nt = write_bmesh(a.out, v, t, n)
        print(f'wrote {a.out}: {nn} nodes, {nt} triangles')
        return
    files = sorted(glob.glob(os.path.join(a.tiles, 'tile_*.msh')))
    if not files:
        sys.exit(f'no tile_*.msh in {a.tiles}')
    os.makedirs(a.out, exist_ok=True)
    rows = []
    off = 0
    for f in files:
        tid = int(os.path.basename(f)[5:10])
        v, t, n = load(f)
        out = os.path.join(a.out, f'tile_{tid:05d}.bmesh')
        nn, nt = write_bmesh(out, v, t, n)
        lo, hi = v.min(axis=0), v.max(axis=0)
        rows.append(f'{tid} {nt} {nn} {off} {lo[0]:.6f} {lo[1]:.6f} {hi[0]:.6f} {hi[1]:.6f} {os.path.basename(out)}')
        off += nt
    with open(os.path.join(a.out, 'index.txt'), 'w') as fp:
        fp.write(f'{len(rows)} {off}\n')
        fp.write('\n'.join(rows) + '\n')
    print(f'wrote {len(rows)} tiles, {off} triangles -> {a.out}/index.txt')


if __name__ == '__main__':
    main()
