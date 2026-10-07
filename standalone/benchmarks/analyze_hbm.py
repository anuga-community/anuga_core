#!/usr/bin/env python3
"""HBM traffic per kernel from rocprofv3 TCC_EA counters, summed over profiled ranks.

    python3 benchmarks/analyze_hbm.py results/frontier_hbm/n1024/weak --tris 40001536

Each subdirectory r<rank>/ holds one profiled rank's run_counter_collection.csv and
run_kernel_trace.csv (see results/frontier_hbm/*/prof_wrap.sh). MI200 L2<->HBM bytes:
read = 32*RDREQ_32B + 64*(RDREQ - RDREQ_32B), write = 64*WRREQ_64B + 32*(WRREQ - WRREQ_64B).
Durations come from the kernel trace of the same (profiled, serialized) run.
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

PEAK_GBS = 1600.0          # MI250X, per GCD


def short(name):
    m = re.search(r"core_(\w+?)_l\d+$", name)
    return m.group(1) if m else None


def rank_totals(d):
    cnt = defaultdict(lambda: defaultdict(float))
    calls = defaultdict(int)
    for r in csv.DictReader(open(os.path.join(d, "run_counter_collection.csv"))):
        k = short(r["Kernel_Name"])
        if k is None:
            continue
        cnt[k][r["Counter_Name"]] += float(r["Counter_Value"])
        if r["Counter_Name"] == "TCC_EA_RDREQ_sum":
            calls[k] += 1
    dur = defaultdict(float)
    for r in csv.DictReader(open(os.path.join(d, "run_kernel_trace.csv"))):
        k = short(r["Kernel_Name"])
        if k is not None:
            dur[k] += (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) * 1e-9
    out = {}
    for k, c in cnt.items():
        rd = 32 * c["TCC_EA_RDREQ_32B_sum"] + 64 * (c["TCC_EA_RDREQ_sum"] - c["TCC_EA_RDREQ_32B_sum"])
        wr = 64 * c["TCC_EA_WRREQ_64B_sum"] + 32 * (c["TCC_EA_WRREQ_sum"] - c["TCC_EA_WRREQ_64B_sum"])
        out[k] = (calls[k], rd, wr, dur[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tris", type=float, required=True, help="owned triangles per rank")
    ap.add_argument("--ms", type=float, help="unprofiled ms/step, for the per-GCD rate at speed")
    ap.add_argument("--ranks", type=int, help="total ranks in the run, for the aggregate")
    a = ap.parse_args()
    ranks = sorted(glob.glob(os.path.join(a.dir, "r*")))
    per = []
    for d in ranks:                    # a rank cut off mid-write (job timeout) is skipped
        try:
            p = rank_totals(d)
        except (OSError, ValueError, KeyError, csv.Error):
            continue
        if p:
            per.append(p)
    kernels = sorted({k for p in per for k in p}, key=lambda k: -sum(p[k][3] for p in per if k in p))
    steps = max(p[k][0] for p in per for k in p)          # calls of the per-step kernels
    print("%s: %d profiled ranks (%d incomplete, skipped), %d steps each"
          % (a.dir, len(per), len(ranks) - len(per), steps))
    print("%-28s %9s %10s %11s %8s" % ("kernel", "ms/step", "read B/tri", "write B/tri", "GB/s"))
    tot_b = tot_t = 0.0
    for k in kernels:
        rows = [p[k] for p in per if k in p]
        n = sum(r[0] for r in rows)
        if n < steps * len(rows):
            continue                                       # not a per-step kernel
        rd = sum(r[1] for r in rows) / n
        wr = sum(r[2] for r in rows) / n
        t = sum(r[3] for r in rows) / n
        tot_b += rd + wr; tot_t += t
        print("%-28s %9.3f %10.1f %11.1f %8.0f" % (k, t * 1e3, rd / a.tris, wr / a.tris, (rd + wr) / t / 1e9))
    btri = tot_b / a.tris
    spread = [sum(p[k][1] + p[k][2] for k in p) / sum(p[k][0] for k in p if k == kernels[0]) / a.tris
              for p in per]
    print("step: %.1f GB/rank, %.0f B/tri (rank spread %.0f..%.0f), %.0f GB/s in kernels (%.0f%% of peak)"
          % (tot_b / 1e9, btri, min(spread), max(spread), tot_b / tot_t / 1e9, 100 * tot_b / tot_t / 1e9 / PEAK_GBS))
    if a.ms:
        g = tot_b / (a.ms * 1e-3) / 1e9
        print("at %.2f ms/step (unprofiled): %.0f GB/s per GCD" % (a.ms, g), end="")
        if a.ranks:
            print(", %.2f PB/s aggregate over %d GCDs (%.0f%% of %.1f PB/s peak)"
                  % (g * a.ranks / 1e6, a.ranks, 100 * g / PEAK_GBS, PEAK_GBS * a.ranks / 1e6), end="")
        print()


if __name__ == "__main__":
    main()
