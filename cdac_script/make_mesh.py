#!/usr/bin/env python
"""
Generate a uniform ANUGA mesh (via anuga.pmesh / meshpy-triangle) from the
Mahanadi Delta bounding polygon in Delta_11372_sqkm.csv.

Usage:
    python make_mesh.py                          # 1 sqm  -> mesh_file/1sqm.msh
    python make_mesh.py --area 100               # 100 sqm -> mesh_file/100sqm.msh
    python make_mesh.py --area 1 --format tsh    # ASCII .tsh instead of .msh

Output name follows the `${resolution}sqm.msh` convention used by
partition_mahanadi.py / run_model_3_partitioned.py (see README.md).

NOTE: the polygon encloses ~11 372 km^2. At 1 sqm that is ~1.1e10 triangles
(~2.3e10 with the 28-degree angle constraint).  Triangle is serial and holds
the whole mesh in RAM, so expect O(TB) memory — this script prints the
estimate and refuses unless --force is passed.
"""

import argparse
import os
import sys
import time

import numpy as np

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Delta_11372_sqkm.csv')


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--area', type=float, default=1.0,
                    help='maximum triangle area in m^2 (default 1)')
    ap.add_argument('--csv', default=CSV, help='bounding polygon CSV (x,y UTM)')
    ap.add_argument('--outdir', default='mesh_file')
    ap.add_argument('--format', choices=['msh', 'tsh'], default='msh')
    ap.add_argument('--min-angle', type=float, default=28.0,
                    help='minimum triangle angle (default 28)')
    ap.add_argument('--force', action='store_true',
                    help='proceed even if the triangle-count estimate is enormous')
    args = ap.parse_args()

    # --- load polygon (plain x,y rows; closing point repeats the first) -----
    poly = np.loadtxt(args.csv, delimiter=',')
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    poly = poly.tolist()
    n = len(poly)

    area_m2 = polygon_area(np.asarray(poly))
    # triangle typically produces ~2x the "area/max_area" lower bound
    est_tri = 2.0 * area_m2 / args.area
    print(f'Polygon: {n} vertices, {area_m2/1e6:.1f} km^2')
    print(f'Max triangle area: {args.area} m^2  ->  ~{est_tri:.2e} triangles (estimate)')
    print(f'Rough RAM for .tsh/.msh mesh dict: ~{est_tri*150/1e9:.0f} GB')
    if est_tri > 5e8 and not args.force:
        print('Estimate exceeds 5e8 triangles; re-run with --force if you really mean it.')
        sys.exit(1)

    # --- build ----------------------------------------------------------------
    import anuga
    from anuga.pmesh.mesh_interface import create_mesh_from_regions

    label = int(args.area) if float(args.area).is_integer() else args.area
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f'{label}sqm.{args.format}')

    # single 'exterior' tag on every segment — matches run_prod.py which uses
    # dict {'exterior': Reflective_boundary}
    boundary_tags = {'exterior': list(range(n))}

    print(f'Triangulating -> {out}')
    t0 = time.time()
    create_mesh_from_regions(poly,
                             boundary_tags=boundary_tags,
                             maximum_triangle_area=args.area,
                             minimum_triangle_angle=args.min_angle,
                             filename=out,
                             use_cache=False,
                             verbose=True)
    print(f'Done in {time.time()-t0:.1f} s: {out} ({os.path.getsize(out)/1e9:.2f} GB)')


if __name__ == '__main__':
    main()
