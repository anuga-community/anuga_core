#!/usr/bin/env python
"""
Split the Delta bounding polygon into rectangular tiles whose shared edges
are discretised IDENTICALLY on both sides, so each tile can be triangulated
independently (in parallel) and the results still line up vertex-for-vertex.

How it works
------------
1. The exterior polygon is densified to a fixed spacing `s` (so Triangle never
   needs to split an exterior segment).
2. The polygon is intersected with a regular grid of `tile` x `tile` metre
   cells (shapely).
3. Every tile edge that lies on a grid line is rebuilt from *lattice* points
   (multiples of `s` along that line). The neighbouring tile reproduces the
   same lattice points, so the shared edge is bit-identical on both sides.
4. Each tile is written as tiles/tile_NNNN.json with its points and a tag per
   segment: 'exterior' (original outer boundary) or 'internal' (cut line).

mesh_tile.py then triangulates a tile with Triangle's -Y flag (no Steiner
points on segments), which is what guarantees conformity.

Usage:
    python split_delta.py --tile 3000 --spacing 1.2 --outdir tiles
"""

import argparse
import json
import os

import numpy as np
from shapely.geometry import Polygon, box, MultiPolygon

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Delta_11372_sqkm.csv')


def densify(pts, s):
    """Insert points along each edge so no edge is longer than s."""
    out = []
    n = len(pts)
    for i in range(n):
        a, b = np.asarray(pts[i]), np.asarray(pts[(i + 1) % n])
        L = np.linalg.norm(b - a)
        k = max(1, int(np.ceil(L / s)))
        for j in range(k):
            out.append(a + (b - a) * j / k)
    return np.asarray(out)


def on_grid_line(a, b, x0, y0, tile, tol=1e-6):
    """Return ('x', X) / ('y', Y) if segment a-b lies on a vertical/horizontal
    grid line, else None."""
    if abs(a[0] - b[0]) < tol and abs((a[0] - x0) / tile - round((a[0] - x0) / tile)) * tile < tol:
        return 'x', a[0]
    if abs(a[1] - b[1]) < tol and abs((a[1] - y0) / tile - round((a[1] - y0) / tile)) * tile < tol:
        return 'y', a[1]
    return None


MARGIN = 0.3   # keep inserted points >= MARGIN * s away from segment endpoints


def lattice_between(lo, hi, origin, s):
    """Lattice points origin + k*s strictly between lo and hi (ascending),
    excluding those within MARGIN*s of either end: a point almost on top of
    a segment endpoint (the corner where a cut line meets the outline) forces
    a needle triangle whose inradius sets the global time step.  Both tiles
    sharing the line see the same endpoints, so the exclusion is symmetric
    and the edge still conforms."""
    k0 = int(np.floor((lo - origin) / s)) + 1
    k1 = int(np.ceil((hi - origin) / s)) - 1
    return [origin + k * s for k in range(k0, k1 + 1)
            if origin + k * s - lo >= MARGIN * s and hi - (origin + k * s) >= MARGIN * s]


def build_tile(poly, x0, y0, tile, s, line_spacing=None):
    """Rebuild a clipped tile polygon with conforming points + segment tags.

    line_spacing(axis, coord) -> spacing to use on that internal cut line
    (multi-scale meshes: the finer of the two tiles sharing the line, so both
    sides produce the same lattice points).  Exterior segments are densified
    to this tile's own spacing s -- they are never shared."""
    ring = list(poly.exterior.coords)[:-1]  # drop closing point
    # Drop outline vertices closer than MARGIN*s to the previous kept one
    # (shapely puts the outline/grid-line intersection millimetres from an
    # existing outline vertex -> needle triangles).  Points on a grid line are
    # shared with the neighbour tile and are never dropped.
    def on_grid(p):
        return (abs((p[0] - x0) / tile - round((p[0] - x0) / tile)) * tile < 1e-6 or
                abs((p[1] - y0) / tile - round((p[1] - y0) / tile)) * tile < 1e-6)
    kept = []
    for q in ring:
        if kept and not on_grid(q) and np.linalg.norm(np.asarray(q) - np.asarray(kept[-1])) < MARGIN * s:
            continue
        kept.append(q)
    if len(kept) > 3 and not on_grid(kept[-1]) and np.linalg.norm(np.asarray(kept[-1]) - np.asarray(kept[0])) < MARGIN * s:
        kept.pop()
    ring = kept
    pts, tags = [], []
    n = len(ring)
    for i in range(n):
        a, b = np.asarray(ring[i]), np.asarray(ring[(i + 1) % n])
        pts.append(a.tolist())
        g = on_grid_line(a, b, x0, y0, tile)
        if g is None:
            tags.append('exterior')
            L = np.linalg.norm(b - a)
            k = int(np.ceil(L / s))           # k-1 evenly spaced points: none
            for j in range(1, k):             # closer than L/k >= s/2 to an end
                pts.append((a + (b - a) * j / k).tolist())
                tags.append('exterior')
            continue
        # internal cut: insert lattice points between a and b
        axis = 1 if g[0] == 'x' else 0
        origin = y0 if g[0] == 'x' else x0
        lo, hi = sorted([a[axis], b[axis]])
        s_line = line_spacing(g[0], g[1]) if line_spacing else s
        lat = lattice_between(lo, hi, origin, s_line)
        if a[axis] > b[axis]:
            lat = lat[::-1]
        tags.append('internal')
        for v in lat:
            p = a.copy()
            p[axis] = v
            pts.append(p.tolist())
            tags.append('internal')
    return pts, tags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default=CSV)
    ap.add_argument('--tile', type=float, default=3000.0,
                    help='tile edge length in metres (default 3000 -> ~9 km^2, '
                         '~2e7 triangles per tile at 1 sqm)')
    ap.add_argument('--spacing', type=float, default=1.2,
                    help='boundary point spacing in metres. Use ~0.8*sqrt(area*2.3) '
                         'i.e. slightly below the natural edge length (1.52 m at 1 sqm)')
    ap.add_argument('--outdir', default='tiles')
    ap.add_argument('--min-area', type=float, default=1.0,
                    help='discard tile slivers smaller than this (m^2)')
    ap.add_argument('--refine', default='',
                    help='comma-separated tile ids (from a previous split with the same '
                         '--tile/--spacing) to cut into 4 half-size sub-tiles; sub-tile id = '
                         '10000 + 4*parent + quadrant, unsplit tiles keep their ids')
    ap.add_argument('--refine-file', help='file with tile ids to refine, one per line')
    ap.add_argument('--areas', help='multi-scale: file of "tile_id max_area_m2" (ids from a previous '
                                    'split on the same grid); other tiles get --area-default')
    ap.add_argument('--area-default', type=float, help='max triangle area (m^2) that --spacing belongs to')
    args = ap.parse_args()
    areas = {}
    if args.areas:
        if args.area_default is None:
            raise SystemExit('--areas needs --area-default (the area --spacing corresponds to)')
        for ln in open(args.areas):
            if ln.strip() and not ln.startswith('#'):
                t, a = ln.split()[:2]
                areas[int(t)] = float(a)
        # spacings must nest: area ratios of 4^k give spacing ratios of 2^k,
        # so the finer lattice contains the coarser one on a shared line
        for t, a in areas.items():
            r = np.log(a / args.area_default) / np.log(4.0)
            if abs(r - round(r)) > 1e-6:
                raise SystemExit(f'tile {t}: area {a} is not area_default * 4^k')
    if areas and (args.refine or args.refine_file):
        raise SystemExit('--areas and --refine cannot be combined (yet)')

    def spacing_for(area):
        if area is None or args.area_default is None:
            return args.spacing
        return args.spacing * 2.0 ** round(np.log(area / args.area_default) / np.log(4.0))
    refine = set(int(t) for t in args.refine.split(',') if t.strip())
    if args.refine_file:
        refine |= set(int(l.split()[0]) for l in open(args.refine_file) if l.strip() and not l.startswith('#'))
    if refine:
        # A sub-tile corner sits in the middle of a neighbouring full tile's
        # edge; it conforms only if that midpoint is a lattice point.
        half = args.tile / 2.0
        if abs(half / args.spacing - round(half / args.spacing)) > 1e-9 * round(half / args.spacing):
            raise SystemExit(f'--refine needs tile/2 = {half} to be a multiple of '
                             f'--spacing {args.spacing} (pass it exactly, e.g. {half/round(half/args.spacing)!r})')

    raw = np.loadtxt(args.csv, delimiter=',')
    if np.allclose(raw[0], raw[-1]):
        raw = raw[:-1]
    # Drop outline vertices closer than MARGIN * (finest spacing) to the
    # previous kept one: the raw outline has vertices centimetres apart, and
    # each such pair becomes a needle triangle (9 mm inradius seen).
    s_finest = min([args.spacing] + [spacing_for(a) for a in areas.values()]) if areas else args.spacing
    keep = [raw[0]]
    for q in raw[1:]:
        if np.linalg.norm(q - keep[-1]) >= MARGIN * s_finest:
            keep.append(q)
    if np.linalg.norm(keep[0] - keep[-1]) < MARGIN * s_finest:
        keep.pop()
    print(f'outline: {len(raw)} -> {len(keep)} vertices after dropping gaps < {MARGIN * s_finest:.2f} m')
    raw = np.asarray(keep)

    # common origin for ALL tiles: lower-left corner of the polygon, snapped
    x0, y0 = np.floor(raw.min(axis=0))
    s_coarsest = max([args.spacing] + [spacing_for(a) for a in areas.values()])
    dense = densify(raw, s_coarsest)
    full = Polygon(dense)
    if not full.is_valid:
        full = full.buffer(0)
    print(f'Polygon: {len(raw)} -> {len(dense)} boundary points, '
          f'{full.area/1e6:.1f} km^2, origin ({x0:.0f}, {y0:.0f})')

    xmax, ymax = raw.max(axis=0)
    nx = int(np.ceil((xmax - x0) / args.tile))
    ny = int(np.ceil((ymax - y0) / args.tile))
    os.makedirs(args.outdir, exist_ok=True)

    index = []
    tid = 0
    nsub = 0

    cell_s = {}      # (i, j) -> spacing of that tile (multi-scale)
    cell_area = {}   # (i, j) -> max triangle area

    def emit(part, i, j, this_id, size, parent=None):
        s_tile = cell_s.get((i, j), args.spacing)

        def line_spacing(axis, coord):
            # neighbour across this cut line: the cell on the other side
            if axis == 'x':
                gi = int(round((coord - x0) / size)); ni, nj = (gi if gi > i else i - 1), j
            else:
                gj = int(round((coord - y0) / size)); ni, nj = i, (gj if gj > j else j - 1)
            return min(s_tile, cell_s.get((ni, nj), s_tile))
        pts, tags = build_tile(part, x0, y0, size, s_tile, line_spacing if cell_s else None)
        rec = {'id': this_id, 'ij': [i, j], 'area_m2': part.area,
               'origin': [float(x0), float(y0)],
               'points': pts, 'segment_tags': tags}
        if (i, j) in cell_area:
            rec['max_area'] = cell_area[(i, j)]
            rec['spacing'] = s_tile
        if parent is not None:
            rec['parent'] = parent
        fn = os.path.join(args.outdir, f'tile_{this_id:05d}.json')
        with open(fn, 'w') as f:
            json.dump(rec, f)
        index.append({'id': this_id, 'file': fn, 'area_m2': part.area,
                      'n_points': len(pts)})

    # pass 1: which cells exist and what id they get (same order as before)
    cells = []
    for j in range(ny):
        for i in range(nx):
            cell = box(x0 + i * args.tile, y0 + j * args.tile,
                       x0 + (i + 1) * args.tile, y0 + (j + 1) * args.tile)
            inter = full.intersection(cell)
            if inter.is_empty:
                continue
            parts = list(inter.geoms) if isinstance(inter, MultiPolygon) else [inter]
            for part in parts:
                if part.area < args.min_area:
                    continue
                cells.append((i, j, part))
    if areas:
        for k, (i, j, part) in enumerate(cells):
            a = areas.get(k, args.area_default)
            cell_area[(i, j)] = a
            cell_s[(i, j)] = spacing_for(a)
    # pass 2: build
    for (i, j, part) in cells:
            if True:
                if part.interiors:
                    raise RuntimeError(f'tile ({i},{j}) has holes; not handled')
                if tid in refine:
                    # quadtree split: 4 half-size cells, on_grid_line sees the
                    # half-grid so the new cut lines get lattice points too
                    half = args.tile / 2.0
                    for q, (di, dj) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
                        sub = box(x0 + (2 * i + di) * half, y0 + (2 * j + dj) * half,
                                  x0 + (2 * i + di + 1) * half, y0 + (2 * j + dj + 1) * half)
                        si = part.intersection(sub)
                        if si.is_empty:
                            continue
                        sparts = list(si.geoms) if isinstance(si, MultiPolygon) else [si]
                        for k, sp in enumerate(sparts):
                            if sp.area < args.min_area:
                                continue
                            if sp.interiors:
                                raise RuntimeError(f'sub-tile of {tid} has holes; not handled')
                            if k > 0:
                                raise RuntimeError(f'sub-tile of {tid} splits into pieces; not handled')
                            emit(sp, 2 * i + di, 2 * j + dj, 10000 + 4 * tid + q, half, parent=tid)
                            nsub += 1
                else:
                    emit(part, i, j, tid, args.tile)
                tid += 1

    with open(os.path.join(args.outdir, 'index.json'), 'w') as f:
        json.dump({'origin': [float(x0), float(y0)], 'tile': args.tile,
                   'spacing': args.spacing, 'tiles': index}, f, indent=1)
    tot = sum(t['area_m2'] for t in index)
    print(f'{len(index)} tiles written to {args.outdir}/ (grid {nx}x{ny}'
          f'{f", {len(refine)} refined into {nsub} sub-tiles" if refine else ""}), '
          f'covering {tot/1e6:.1f} km^2')


if __name__ == '__main__':
    main()
