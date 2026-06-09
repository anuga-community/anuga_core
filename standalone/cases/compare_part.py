"""Validate the MPI self-distributed run against a single-rank run, by global id.

Reassembles per-cell stage[global_id] from the .result files written by
anuga_miniapp_part and compares two runs (e.g. np=1 vs np=4). A correct
distribution => identical owned-cell values. numpy only; no anuga.

    python standalone/cases/compare_part.py "<ref glob>" "<test glob>"
    python standalone/cases/compare_part.py "/tmp/agm1/dam_break.agm.P1_*.result" \
                                            "/tmp/agm4/dam_break.agm.P4_*.result"
"""
import glob
import struct
import sys
import numpy as np


def assemble(pattern):
    gid_all, stage_all = [], []
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    for p in files:
        with open(p, "rb") as f:
            (count,) = struct.unpack("<q", f.read(8))
            gid_all.append(np.fromfile(f, "<i8", count))
            stage_all.append(np.fromfile(f, "<f8", count))
    gids = np.concatenate(gid_all)
    stage = np.concatenate(stage_all)
    n = int(gids.max()) + 1
    out = np.full(n, np.nan)
    out[gids] = stage
    return out, len(files)


def main():
    ref_glob = sys.argv[1]
    test_glob = sys.argv[2]
    ref, nref = assemble(ref_glob)
    test, ntest = assemble(test_glob)
    if ref.shape != test.shape:
        print(f"FAIL: size mismatch {ref.shape} vs {test.shape}")
        sys.exit(1)
    if np.isnan(ref).any() or np.isnan(test).any():
        print(f"FAIL: missing cells (ref {np.isnan(ref).sum()}, test {np.isnan(test).sum()})")
        sys.exit(1)
    linf = float(np.max(np.abs(ref - test)))
    print(f"ref ({nref} files) sum={ref.sum():.8g}  test ({ntest} files) sum={test.sum():.8g}")
    print(f"elementwise Linf (by global id) = {linf:.3e}")
    ok = linf < 1e-9
    print(f"  -> {'PASS' if ok else 'FAIL'} (<1e-9)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
