#!/usr/bin/env python
"""
Cut a sub-index from a tile index: the tiles whose bounding box intersects
a window.  Offsets are recomputed contiguously, so canonical ids are
consistent between every run that uses the same sub-index (np=1 golden vs
np=N).  Written next to the tiles (rows reference file basenames).

    python tools/tile_subset.py /g/data/bm55/jlv900/tiles1sqm/index.txt \
        --box 84000 34000 96000 46000 --out /g/data/bm55/jlv900/tiles1sqm/index_lake16.txt
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('index')
    ap.add_argument('--box', nargs=4, type=float, required=True, metavar=('X0', 'Y0', 'X1', 'Y1'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    x0, y0, x1, y1 = a.box
    with open(a.index) as f:
        f.readline()
        rows = [ln.split() for ln in f if ln.strip()]
    keep, off, tot = [], 0, 0
    for r in rows:
        bx0, by0, bx1, by1 = map(float, r[4:8])
        if bx0 < x1 and x0 < bx1 and by0 < y1 and y0 < by1:
            keep.append(f'{r[0]} {r[1]} {r[2]} {off} {r[4]} {r[5]} {r[6]} {r[7]} {r[8]}')
            off += int(r[1]); tot += 1
    if os.path.dirname(os.path.abspath(a.out)) != os.path.dirname(os.path.abspath(a.index)):
        raise SystemExit('--out must be in the same directory as the index (rows reference basenames)')
    with open(a.out, 'w') as f:
        f.write(f'{tot} {off}\n' + '\n'.join(keep) + '\n')
    print(f'{tot} tiles, {off} triangles -> {a.out}')


if __name__ == '__main__':
    main()
