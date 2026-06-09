"""Validate the pure-C / dump-based mini-app results against ANUGA.

Reads the <case>_P<N>_<rank>.result files written by anuga_miniapp and compares
the owned-cell stage values (permutation-invariant: sorted, since global ids are
not dumped) against:
  - each other across decompositions (np=1 vs 2 vs 4), and
  - a fresh ANUGA serial reference run.

A bit-identical parallel/serial solver => sorted values match to ~round-off.

    source standalone/env_dgx.sh
    python standalone/cases/compare_dump.py /tmp/anuga_adm
"""
import os
import sys
import struct
import numpy as np

import anuga

LEN = 100.0
N = 40
FINALTIME = 2.0


def read_result(path):
    with open(path, "rb") as f:
        (m,) = struct.unpack("<q", f.read(8))
        return np.fromfile(f, dtype="<f8", count=m)


def gather(dumpdir, name, nprocs):
    vals = [read_result(os.path.join(dumpdir, f"{name}_P{nprocs}_{p}.result"))
            for p in range(nprocs)]
    return np.concatenate(vals)


def anuga_reference():
    d = anuga.rectangular_cross_domain(N, N, len1=LEN, len2=LEN)
    d.set_flow_algorithm("DE0")
    d.set_low_froude(0)
    d.set_name("db_ref")
    d.set_datadir("/tmp")
    d.store = False
    d.set_quantity("elevation", 0.0)
    d.set_quantity("friction", 0.0)
    d.set_quantity("stage", lambda x, y: np.where(x < LEN / 2.0, 4.0, 1.0))
    Br = anuga.Reflective_boundary(d)
    d.set_boundary({"left": Br, "right": Br, "top": Br, "bottom": Br})
    d.set_multiprocessor_mode(1)
    for _ in d.evolve(yieldstep=FINALTIME, finaltime=FINALTIME):
        pass
    return d.quantities["stage"].centroid_values.copy()


def main():
    dumpdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/anuga_adm"
    name = "dam_break"

    ref = np.sort(anuga_reference())
    print(f"ANUGA serial reference: {ref.size} cells, sum={ref.sum():.10g}")

    worst = 0.0
    for nprocs in (1, 2, 4):
        try:
            g = gather(dumpdir, name, nprocs)
        except FileNotFoundError as e:
            print(f"  np={nprocs}: missing results ({e}); skipped")
            continue
        gs = np.sort(g)
        if gs.size != ref.size:
            print(f"  np={nprocs}: FAIL size {gs.size} != {ref.size}")
            worst = max(worst, 1e9)
            continue
        linf = float(np.max(np.abs(gs - ref)))
        worst = max(worst, linf)
        print(f"  np={nprocs}: {g.size} owned cells, sum={g.sum():.10g}, "
              f"sorted-Linf vs ANUGA = {linf:.3e}")

    ok = worst < 1e-8
    print(f"\n  -> {'PASS' if ok else 'FAIL'} (sorted-Linf < 1e-8, worst={worst:.3e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
