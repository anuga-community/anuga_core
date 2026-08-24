#!/usr/bin/env python3
"""Convert model rainfall data to the benchmark's ANUGARN1 raster timeseries.

Accepts a LONG CSV with header and columns  time,x,y,rate  on a regular grid
of (x, y) points.  Units are flagged, defaults: seconds and mm/hr.

    python tools/rain_csv_to_grid.py rain.csv --out build/rain.grid \
        [--time-units s|min|hr] [--rate-units mmhr|ms]

If your model's CSV has a different shape (wide/one-column-per-gauge,
NetCDF-exported, etc.), send one header line and one data line and this
converter grows a reader for it.
"""
import argparse, struct, sys
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--time-units", choices=("s", "min", "hr"), default="s")
    ap.add_argument("--rate-units", choices=("mmhr", "ms"), default="mmhr")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    tmul = {"s": 1.0, "min": 60.0, "hr": 3600.0}[args.time_units]
    rmul = 1.0 if args.rate_units == "mmhr" else 3.6e6   # m/s -> mm/hr

    raw = np.loadtxt(args.csv, delimiter=",", skiprows=1)
    times = np.unique(raw[:, 0]) * tmul
    xs = np.unique(raw[:, 1])
    ys = np.unique(raw[:, 2])
    nx, ny, nt = len(xs), len(ys), len(times)

    ti = np.searchsorted(times, raw[:, 0] * tmul)
    gi = np.searchsorted(xs, raw[:, 1])
    gj = np.searchsorted(ys, raw[:, 2])
    grid = np.zeros((nt, ny, nx), dtype=np.float32)
    grid[ti, gj, gi] = raw[:, 3] * rmul

    dx = xs[1] - xs[0] if nx > 1 else 1.0
    dy = ys[1] - ys[0] if ny > 1 else 1.0
    with open(args.out, "wb") as fp:
        fp.write(struct.pack("<8s2q4dq", b"ANUGARN1", nx, ny,
                             xs[0] - dx / 2, ys[0] - dy / 2, dx, dy, nt))
        fp.write(times.astype(np.float64).tobytes())
        fp.write(grid.tobytes())
    print(f"wrote {args.out}: {nx}x{ny} grid, {nt} intervals, "
          f"mean {grid.mean():.2f} mm/hr, raining-fraction "
          f"{np.count_nonzero(grid) / grid.size:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
