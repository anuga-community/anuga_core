#!/usr/bin/env python
"""
Quality gate for a tile set: per-tile triangle inradius tail and minimum
angle, flagging tiles whose smallest triangles would set the time step.
ANUGA's dt is min(inradius / speed) over WET cells, so the numbers to watch
are the smallest inradii, not the mean.

    python tile_quality.py mesh_tiles_ms --min-inradius-frac 0.5 [--min-angle 25]

Reads .msh tiles (NetCDF, via anuga) or a bmesh dir (index.txt).  Exit 1
if any tile fails: min inradius < frac * that tile's median, or min angle
below --min-angle.
"""
import argparse, glob, os, struct, sys
from multiprocessing import Pool
import numpy as np


def load_bmesh(p):
    with open(p, 'rb') as f:
        f.read(8); ver, _ = struct.unpack('<ii', f.read(8)); nn, nt, nb = struct.unpack('<qqq', f.read(24))
        nodes = np.frombuffer(f.read(16 * nn), dtype=np.float64).reshape(nn, 2)
        tris = np.frombuffer(f.read((4 if ver == 2 else 8) * 3 * nt), dtype=np.int32 if ver == 2 else np.int64).reshape(nt, 3)
    return nodes, tris


def load_msh(p):
    from anuga.load_mesh.loadASCII import import_mesh_file
    m = import_mesh_file(p)
    return np.asarray(m['vertices'], float), np.asarray(m['triangles'], np.int64)


def quality(nodes, tris):
    P = nodes[tris]
    a = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    e = np.stack([np.linalg.norm(P[:, 1] - P[:, 0], axis=1), np.linalg.norm(P[:, 2] - P[:, 1], axis=1), np.linalg.norm(P[:, 0] - P[:, 2], axis=1)], 1)
    inr = 2 * a / e.sum(1)
    # smallest angle per triangle (law of cosines on the shortest edge)
    e_s = np.sort(e, axis=1)
    cosA = (e_s[:, 1] ** 2 + e_s[:, 2] ** 2 - e_s[:, 0] ** 2) / (2 * e_s[:, 1] * e_s[:, 2])
    ang = np.degrees(np.arccos(np.clip(cosA, -1, 1)))
    return a, inr, ang, P.mean(1)


_SCAN_ARGS = [0.5, 25.0]        # [min_inradius_frac, min_angle], set by main()


def _scan_one(item):
    """One tile -> its quality row, or None if it is too small to judge."""
    tid, f, loader = item
    nodes, tris = loader(f)
    if len(tris) < 10:
        return None
    area, inr, ang, c = quality(nodes, tris)
    med = float(np.median(inr)); k = int(np.argmin(inr))
    frac, min_ang = _SCAN_ARGS
    bad = bool(inr[k] < frac * med or ang.min() < min_ang)
    return (tid, len(tris), med, float(np.percentile(inr, 1)), float(inr[k]),
            float(ang.min()), bad, float(c[k][0]), float(c[k][1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tiles')
    ap.add_argument('--min-inradius-frac', type=float, default=0.5)
    ap.add_argument('--min-angle', type=float, default=25.0)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--jobs', type=int, default=1,
                    help='worker processes (the 1 m^2 set is 17.4G triangles; use $PBS_NCPUS)')
    a = ap.parse_args()
    if os.path.isfile(os.path.join(a.tiles, 'index.txt')):
        rows = [l.split() for l in open(os.path.join(a.tiles, 'index.txt')).read().splitlines()[1:] if l.strip()]
        files = [(int(r[0]), os.path.join(a.tiles, r[8]), load_bmesh) for r in rows]
    else:
        files = [(int(os.path.basename(f)[5:10]), f, load_msh) for f in sorted(glob.glob(os.path.join(a.tiles, 'tile_*.msh')))]
    fails, tot, worst = 0, 0, (1e9, None, None)
    print(f'{"tile":>6} {"tris":>10} {"med inr":>8} {"p1":>7} {"min":>7} {"min ang":>8}  flag')
    _SCAN_ARGS[:] = [a.min_inradius_frac, a.min_angle]
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            results = pool.imap(_scan_one, files, chunksize=1)
            rows = [r for r in results if r is not None]
    else:
        rows = [r for r in map(_scan_one, files) if r is not None]
    for tid, ntri, med, p1, mn, mnang, bad, cx, cy in rows:
        tot += ntri
        fails += bad
        if mn / med < worst[0]:
            worst = (mn / med, tid, (cx, cy))
        if bad or a.verbose:
            print(f'{tid:>6} {ntri:>10} {med:8.2f} {p1:7.2f} {mn:7.3f} {mnang:8.1f}  '
                  f'{"FAIL" if bad else "ok"}  smallest at ({cx:.0f}, {cy:.0f})')
    print(f'{len(files)} tiles, {tot} triangles: {fails} tile(s) fail; worst min/median inradius {worst[0]:.2f} '
          f'in tile {worst[1]} at ({worst[2][0]:.0f}, {worst[2][1]:.0f})')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
