#!/usr/bin/env python
"""
Merge the per-tile .msh files from mesh_tile.py into one conforming mesh.

* vertices are de-duplicated on exact (rounded) coordinates -- tiles share the
  same geo_reference origin and identical lattice points along cut lines
* 'internal' segments are dropped, only 'exterior' segments survive
* triangle_neighbors are stitched across tile edges (ANUGA reads them from
  the file and does NOT recompute), by matching the -1 edges of one tile
  against those of its neighbours on (global vertex pair) keys

Usage:
    python merge_tiles.py mesh_tiles/ mesh_file/300sqm.msh
"""

import glob
import os
import sys
import time

import numpy as np


def main(indir, out, decimals=6):
    from anuga.load_mesh.loadASCII import import_mesh_file, export_mesh_file

    files = sorted(glob.glob(os.path.join(indir, 'tile_*.msh')))
    if not files:
        sys.exit(f'no tile_*.msh in {indir}')

    V, T, N, S, Stag = [], [], [], [], []
    geo = None
    voff = toff = 0
    t0 = time.time()
    for f in files:
        m = import_mesh_file(f)
        if geo is None:
            geo = m['geo_reference']
        v = np.asarray(m['vertices'], float)
        t = np.asarray(m['triangles'], np.int64)
        n = np.asarray(m['triangle_neighbors'], np.int64)
        s = np.asarray(m['segments'], np.int64)
        tags = np.asarray(m['segment_tags'])
        keep = tags == 'exterior'
        V.append(v)
        T.append(t + voff)
        N.append(np.where(n >= 0, n + toff, -1))
        S.append(s[keep] + voff)
        Stag.extend(tags[keep].tolist())
        voff += len(v)
        toff += len(t)
    print(f'loaded {len(files)} tiles: {voff} vertices, {toff} triangles '
          f'in {time.time()-t0:.1f} s')

    V = np.vstack(V); T = np.vstack(T); N = np.vstack(N); S = np.vstack(S)

    # --- de-duplicate vertices ---------------------------------------------
    key = np.round(V, decimals)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    T = inv[T]; S = inv[S]
    print(f'unique vertices: {len(uniq)} (merged {voff-len(uniq)} duplicates)')

    # --- stitch neighbours across tile edges ---------------------------------
    # edge j of triangle i is opposite vertex j: (v[(j+1)%3], v[(j+2)%3])
    ti, ej = np.where(N < 0)
    if len(ti) == 0:
        sys.exit('no boundary (-1) neighbours found: tiles were meshed without the n flag?')
    a = T[ti, (ej + 1) % 3]; b = T[ti, (ej + 2) % 3]
    ek = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
    order = np.lexsort((ek[:, 1], ek[:, 0]))
    ek, ti, ej = ek[order], ti[order], ej[order]
    same = np.all(ek[1:] == ek[:-1], axis=1)
    p = np.where(same)[0]           # pairs (p, p+1) share an edge
    N[ti[p], ej[p]] = ti[p + 1]
    N[ti[p + 1], ej[p + 1]] = ti[p]
    print(f'stitched {len(p)} internal edges; {len(ti)-2*len(p)} boundary edges remain '
          f'({len(S)} exterior segments)')

    mesh = {
        'vertices': uniq, 'vertex_attributes': None, 'vertex_attribute_titles': [],
        'triangles': T, 'triangle_tags': [''] * len(T), 'triangle_neighbors': N,
        'segments': S, 'segment_tags': Stag,
        'points': [], 'point_attributes': [], 'outline_segments': [],
        'outline_segment_tags': [], 'holes': [], 'regions': [],
        'region_tags': [], 'region_max_areas': [], 'geo_reference': geo,
    }
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    t0 = time.time()
    export_mesh_file(out, mesh)
    print(f'wrote {out} ({os.path.getsize(out)/1e9:.2f} GB) in {time.time()-t0:.1f} s')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
