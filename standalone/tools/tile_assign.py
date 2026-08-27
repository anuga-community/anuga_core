#!/usr/bin/env python
"""
Turn per-tile activity (bench --tile-stats, from a coarse or previous run)
into a tile -> rank assignment for bench --assign, balanced by EXPECTED WORK
rather than triangle count:

    weight(tile) = ntris * (active_fraction + floor)

`floor` charges the per-cell work the active set never skips (the rebuild
passes, prepare/update over listed cells' rings) and keeps fully dry tiles
from being free.  Without a stats file every tile gets active_fraction = 1
(pure triangle-count balance).

Methods:
  contig  contiguous split in Morton order of tile centres (compact ranks,
          balance limited by the heaviest tile).  DEFAULT: on 4 H200s it
          matched lpt's kernel balance (9%) with 4x fewer ghosts and was
          1.6x faster on wall time.
  lpt     greedy longest-processing-time bin packing: same kernel balance,
          but scattered tiles quadruple the halo and the exchange cost eats
          the gain.  Kept for reference.

Usage:
    python tools/tile_assign.py build/tiles10k/index.txt --stats stats.txt \
        --nprocs 8 --out assign8.txt [--method lpt] [--floor 0.05]
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('index')
    ap.add_argument('--stats', help='bench --tile-stats output')
    ap.add_argument('--nprocs', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--method', choices=['lpt', 'contig'], default='contig')
    ap.add_argument('--floor', type=float, default=0.035)
    ap.add_argument('--split-threshold', type=float, default=0.5, help='flag tiles heavier than this fraction of a rank share')
    ap.add_argument('--split-out', help='write the tile ids above half a share (for split_delta.py --refine-file)')
    a = ap.parse_args()

    with open(a.index) as f:
        ntiles, nglobal = map(int, f.readline().split())
        rows = [ln.split() for ln in f if ln.strip()]
    ids = [int(r[0]) for r in rows]
    ntris = {int(r[0]): int(r[1]) for r in rows}
    # bbox centre per tile -> Morton order for the contiguous split, so a
    # rank is a compact 2-D patch (sub-tiles from a refinement carry ids
    # >= 10000 and would otherwise all sit at the end of the id order).
    cx = {int(r[0]): (float(r[4]) + float(r[6])) / 2 for r in rows}
    cy = {int(r[0]): (float(r[5]) + float(r[7])) / 2 for r in rows}
    x0, y0 = min(cx.values()), min(cy.values())
    span = max(max(cx.values()) - x0, max(cy.values()) - y0) or 1.0

    def morton(t):
        ix = int((cx[t] - x0) / span * 65535); iy = int((cy[t] - y0) / span * 65535)
        code = 0
        for b in range(16):
            code |= ((ix >> b) & 1) << (2 * b) | ((iy >> b) & 1) << (2 * b + 1)
        return code
    order = sorted(ids, key=morton)
    frac = {t: 1.0 for t in ids}
    if a.stats:
        meas = {}
        with open(a.stats) as f:
            for ln in f:
                if ln.startswith('#') or not ln.strip():
                    continue
                t, _, fr, _ = ln.split()
                meas[int(t)] = float(fr)
        for t in ids:
            # sub-tiles (id = 10000 + 4*parent + q, split_delta.py --refine)
            # inherit the parent's fraction when only the parent was measured
            frac[t] = meas.get(t, meas.get((t - 10000) // 4 if t >= 10000 else -1, 1.0))
    w = {t: ntris[t] * (frac[t] + a.floor) for t in ids}
    total = sum(w.values())
    assign = {}
    if a.method == 'lpt':
        load = [0.0] * a.nprocs
        for t in sorted(ids, key=lambda t: -w[t]):
            r = min(range(a.nprocs), key=lambda r: load[r])
            assign[t] = r
            load[r] += w[t]
    else:
        acc, r = 0.0, 0
        for t in order:
            if r < a.nprocs - 1 and acc + w[t] / 2 > (r + 1) * total / a.nprocs:
                r += 1
            assign[t] = r
            acc += w[t]
        load = [0.0] * a.nprocs
        for t in ids:
            load[assign[t]] += w[t]
    with open(a.out, 'w') as f:
        f.write('\n'.join(str(assign[t]) for t in ids) + '\n')
    mean = total / a.nprocs
    tri_load = [0] * a.nprocs
    for t in ids:
        tri_load[assign[t]] += ntris[t]
    print(f'{ntiles} tiles -> {a.nprocs} ranks ({a.method}, floor {a.floor}): '
          f'max/mean weight {max(load)/mean:.3f}, min/mean {min(load)/mean:.3f}; '
          f'triangles per rank {min(tri_load)} .. {max(tri_load)}')
    # Granularity bound: no assignment of whole tiles can bring the slowest
    # rank below the heaviest tile.  Tiles heavier than half a rank's share
    # are the ones to split (cdac_script/split_delta.py at half the tile
    # size keeps the outer lattice, so neighbours still conform).
    heavy = sorted((t for t in ids if w[t] > a.split_threshold * mean), key=lambda t: -w[t])
    print(f'heaviest tile = {max(w.values())/mean:.2f} x mean rank share '
          f'(lower bound on max/mean); {len(heavy)} tile(s) above {a.split_threshold} share'
          + (f': {heavy[:20]}{"..." if len(heavy) > 20 else ""}' if heavy else ''))
    if a.split_out:
        with open(a.split_out, 'w') as f:
            f.write('\n'.join(str(t) for t in heavy) + ('\n' if heavy else ''))
        print(f'wrote {a.split_out} ({len(heavy)} tiles to refine)')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
