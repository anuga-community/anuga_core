#!/usr/bin/env python3
"""Turn benchmarks/results/scaling_<machine>.csv into figures + a table.

    python3 benchmarks/analyze_scaling.py benchmarks/results/scaling_*.csv --out figs/

Weak:   efficiency = ms(base nodes) / ms(nodes)      (ideal 1.0)
Strong: speedup    = ms(base nodes) / ms(nodes), efficiency = speedup / (nodes/base)
Bars show where the step time goes: kernels, halo exchange, dt allreduce+wait.
"""
import argparse
import csv
import os
import sys

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows += [r for r in csv.DictReader(f) if r["status"] == "ok"]
    return rows


def series(rows, machine, study):
    rs = [r for r in rows if r["machine"] == machine and r["study"] == study]
    if study == "strong" and rs:      # re-runs at a different global size must not mix
        from collections import Counter
        T = Counter(r["triangles"] for r in rs).most_common(1)[0][0]
        rs = [r for r in rs if r["triangles"] == T]
    rs = sorted(rs, key=lambda r: int(r["nodes"]))
    # last measurement wins if a size was re-run
    d = {int(r["nodes"]): r for r in rs}
    n = np.array(sorted(d))
    f = lambda k: np.array([float(d[i][k]) for i in n])
    # kernel_ms (sum of per-phase maxima) can exceed the step; prefer the per-rank
    # compute max when the run recorded it.
    has = all(d[i].get("kernel_rank_max_ms", "") not in ("", "nan") for i in n)
    ker = f("kernel_rank_max_ms") if has else f("kernel_ms")
    return n, f("ms_per_step"), ker, f("halo_ms"), f("dt_ms"), f("triangles"), f("ranks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="scaling_figs")
    a = ap.parse_args()
    rows = load(a.csv)
    os.makedirs(a.out, exist_ok=True)
    machines = sorted({r["machine"] for r in rows})
    md = ["# Scaling results", ""]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for study in ("weak", "strong"):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        have = False
        for m in machines:
            if not any(r["machine"] == m and r["study"] == study for r in rows):
                continue
            have = True
            n, ms, ker, halo, dt, tris, ranks = series(rows, m, study)
            if study == "weak":
                eff = ms[0] / ms
                ax[0].semilogx(n, eff, "o-", label=m)
                ax[1].semilogx(n, tris / ms * 1e3 / 1e9, "o-", label=m)   # Gcell-steps/s
                md += ["## weak -- %s" % m, "",
                       "| nodes | ranks | triangles | ms/step | efficiency | halo ms | dt ms |", "|---|---|---|---|---|---|---|"]
                for i in range(len(n)):
                    md.append("| %d | %d | %.3g | %.3f | %.3f | %.3f | %.3f |"
                              % (n[i], ranks[i], tris[i], ms[i], eff[i], halo[i], dt[i]))
            else:
                sp = ms[0] / ms
                ideal = n / n[0]
                ax[0].loglog(n, sp, "o-", label=m)
                ax[0].loglog(n, ideal, "k:", lw=0.8)
                ax[1].semilogx(n, sp / ideal, "o-", label=m)
                md += ["## strong -- %s (%.3g triangles)" % (m, tris[0]), "",
                       "| nodes | ranks | tris/rank | ms/step | speedup | efficiency | halo ms | dt ms |", "|---|---|---|---|---|---|---|---|"]
                for i in range(len(n)):
                    md.append("| %d | %d | %.3g | %.3f | %.1f | %.3f | %.3f | %.3f |"
                              % (n[i], ranks[i], tris[i] / ranks[i], ms[i], sp[i], sp[i] / ideal[i], halo[i], dt[i]))
            md.append("")
        if not have:
            plt.close(fig)
            continue
        if study == "weak":
            ax[0].set_ylabel("weak-scaling efficiency"); ax[0].set_ylim(0, 1.1)
            ax[1].set_ylabel("Gcell-steps / s (total)")
        else:
            ax[0].set_ylabel("speedup vs smallest run"); ax[1].set_ylabel("parallel efficiency")
            ax[1].set_ylim(0, 1.1)
        for x in ax:
            x.set_xlabel("nodes"); x.grid(alpha=0.3); x.legend()
        fig.suptitle("%s scaling" % study)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "%s_scaling.png" % study), dpi=150)
        plt.close(fig)

        # breakdown bars, one panel per machine
        fig, axs = plt.subplots(1, len(machines), figsize=(5 * len(machines), 3.8), squeeze=False)
        for ax_, m in zip(axs[0], machines):
            if not any(r["machine"] == m and r["study"] == study for r in rows):
                continue
            n, ms, ker, halo, dt, *_ = series(rows, m, study)
            x = np.arange(len(n))
            ax_.bar(x, ker, label="kernels")
            ax_.bar(x, halo, bottom=ker, label="halo")
            ax_.bar(x, dt, bottom=ker + halo, label="dt allreduce + wait")
            ax_.set_xticks(x); ax_.set_xticklabels(n, rotation=45)
            ax_.set_xlabel("nodes"); ax_.set_ylabel("ms / step"); ax_.set_title(m)
            ax_.legend()
        fig.suptitle("%s: where the step goes" % study)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "%s_breakdown.png" % study), dpi=150)
        plt.close(fig)

    with open(os.path.join(a.out, "scaling.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("figures in", a.out)


if __name__ == "__main__":
    sys.exit(main())
