#!/usr/bin/env python
"""
Triangulate ONE tile produced by split_delta.py and write it as a .msh file.

Calls anuga.mesh_engine.generate_mesh (Triangle) directly, bypassing the
pmesh Python object layer (which costs ~1 KB per triangle), and uses the -Y
flag so Triangle never inserts vertices on the tile boundary -> tiles
conform exactly along their shared edges.

All tiles share the same geo_reference origin (from index.json), so vertex
coordinates in every .msh are in the same local frame and can be merged
by de-duplicating boundary vertices on exact coordinate equality.

Usage:
    python mesh_tile.py tiles/tile_00042.json --area 1 --outdir mesh_tiles
    python mesh_tile.py tiles/tile_00042.json --area 1 --zone 45   # set UTM zone
"""

import argparse
import json
import os
import time

import numpy as np

TAG_ID = {'exterior': 1, 'internal': 2}
ID_TAG = {v: k for k, v in TAG_ID.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tile_json')
    ap.add_argument('--area', type=float, default=1.0, help='max triangle area m^2')
    ap.add_argument('--min-angle', type=float, default=28.0)
    ap.add_argument('--outdir', default='mesh_tiles')
    ap.add_argument('--zone', type=int, default=-1, help='UTM zone for geo_reference')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    from anuga.mesh_engine.mesh_engine import generate_mesh
    from anuga.load_mesh.loadASCII import export_mesh_file
    from anuga.coordinate_transforms.geo_reference import Geo_reference

    with open(args.tile_json) as f:
        rec = json.load(f)
    x0, y0 = rec['origin']
    pts = np.asarray(rec['points'], dtype=float) - [x0, y0]  # local frame
    n = len(pts)
    segs = np.column_stack([np.arange(n), (np.arange(n) + 1) % n]).astype(np.int32)
    segatts = [TAG_ID[t] for t in rec['segment_tags']]  # list: mesh_engine does 'segatts == []'

    area = float(rec.get('max_area', args.area))   # multi-scale: per-tile target
    mode = f"{'' if args.verbose else 'Q'}pznq{args.min_angle}Ya{area:.20f}"
    est = 2 * rec['area_m2'] / area
    print(f"tile {rec['id']}: {n} boundary pts, {rec['area_m2']/1e6:.2f} km^2, "
          f"~{est:.1e} triangles, mode {mode}")

    t0 = time.time()
    g = generate_mesh(points=pts, segments=segs, segatts=segatts, mode=mode)
    ntri = g['generatedtrianglelist'].shape[0]
    print(f'  triangulated {ntri} triangles in {time.time()-t0:.1f} s')

    geo = Geo_reference(zone=args.zone, xllcorner=x0, yllcorner=y0)
    mesh = {
        'vertices': g['generatedpointlist'],
        'vertex_attributes': None,
        'vertex_attribute_titles': [],
        'triangles': g['generatedtrianglelist'],
        'triangle_tags': [''] * ntri,
        'triangle_neighbors': g['generatedtriangleneighborlist'],
        'segments': g['generatedsegmentlist'],
        'segment_tags': [ID_TAG.get(int(m), '') for m in g['generatedsegmentmarkerlist']],
        'points': pts,
        'point_attributes': [],
        'outline_segments': segs,
        'outline_segment_tags': rec['segment_tags'],
        'holes': [],
        'regions': [],
        'region_tags': [],
        'region_max_areas': [],
        'geo_reference': geo,
    }
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"tile_{rec['id']:05d}.msh")
    t0 = time.time()
    export_mesh_file(out, mesh)
    print(f'  wrote {out} ({os.path.getsize(out)/1e9:.2f} GB) in {time.time()-t0:.1f} s')


if __name__ == '__main__':
    main()
