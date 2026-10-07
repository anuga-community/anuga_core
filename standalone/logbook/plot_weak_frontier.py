import csv, glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "results")
rows = []
for f in sorted(glob.glob(R + "/frontier_n*/scaling_frontier.csv") +
                glob.glob(R + "/frontier_full/n*/weak/scaling_frontier.csv")):
    for r in csv.DictReader(open(f)):
        if r["study"] == "weak" and r["status"] == "ok":
            rows.append((int(r["nodes"]), float(r["ms_per_step"]), float(r["halo_ms"]),
                         float(r["dt_ms"]), int(r["triangles"])))
rows.sort()
base = min(r for r in rows if r[0] == 1)[1]
nodes = sorted({r[0] for r in rows})
mean = {n: sum(r[1] for r in rows if r[0] == n) / sum(1 for r in rows if r[0] == n) for n in nodes}
tris = {r[0]: r[4] for r in rows}

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.6, 1]},
                           facecolor=SURF)
for ax in (a, b):
    ax.set_facecolor(SURF); ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(nodes))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % v))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("Frontier nodes (8 MI250X GCDs each)")

# (a) efficiency
a.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
a.text(nodes[-1], 1.004, "ideal", color=INK2, ha="right", va="bottom", fontsize=9)
a.plot(nodes, [base / mean[n] for n in nodes], color=BLUE, lw=2, zorder=2)
a.scatter([r[0] for r in rows], [base / r[1] for r in rows], s=46, color=BLUE,
          edgecolor=SURF, linewidth=2, zorder=3)
a.set_ylim(0.80, 1.03); a.set_ylabel("weak-scaling efficiency  (t₁ / tₙ)")
last = nodes[-1]
a.annotate("%.1f%%\n%d GCDs, %.0fG triangles\n%.1f ms/step" % (100 * base / mean[last], 8 * last,
           tris[last] / 1e9, mean[last]), (last, base / mean[last]), xytext=(-8, -58),
           textcoords="offset points", ha="right", color=INK, fontsize=9.5,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.annotate("1 node: %.1f ms/step\n40M triangles / GCD" % base, (1, 1.0), xytext=(4, -150),
           textcoords="offset points", color=INK, fontsize=9.5)
a.set_title("ANUGA weak scaling on Frontier (preliminary)", loc="left", color=INK,
            fontsize=12.5, fontweight="bold")

# (b) overheads
hal = [sum(r[2] for r in rows if r[0] == n) / sum(1 for r in rows if r[0] == n) for n in nodes]
dts = [sum(r[3] for r in rows if r[0] == n) / sum(1 for r in rows if r[0] == n) for n in nodes]
b.plot(nodes, dts, color=ORANGE, lw=2, marker="o", ms=7, mec=SURF, mew=2, label="dt allreduce (incl. wait)")
b.plot(nodes, hal, color=AQUA, lw=2, marker="o", ms=7, mec=SURF, mew=2, label="halo exchange")
b.set_ylim(0, 18); b.set_ylabel("ms per step (max over ranks)")
b.text(nodes[-1], dts[-1] + 0.9, "dt", color=INK, ha="right", fontsize=9.5)
b.text(nodes[-1], hal[-1] + 0.9, "halo", color=INK, ha="right", fontsize=9.5)
b.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
b.set_title("Communication cost per step", loc="left", color=INK, fontsize=12.5, fontweight="bold")

fig.text(0.01, 0.01, "Dam break, ader2/scatter, 1-D slab, 40M triangles per GCD, 100 steps x 3 repeats. "
         "Points at the same node count are separate jobs (run-to-run noise ~2%). "
         "Kernels are ~99% of each step.", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = sys.argv[1]
fig.savefig(out, dpi=160, facecolor=SURF)
print(out, nodes)
