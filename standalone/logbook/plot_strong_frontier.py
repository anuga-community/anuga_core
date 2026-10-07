import csv, glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# Strong scaling on Frontier: reads every
# benchmarks/results/frontier_strong/n*/{large,small}/scaling_frontier.csv and the
# full-machine series in benchmarks/results/frontier_full/n*/strong/.
# Usage: python3 plot_strong_frontier.py out.png
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "results")
series = {}                                   # tris -> {nodes: [ms, ...]}
for f in sorted(glob.glob(R + "/frontier_strong/n*/*/scaling_frontier.csv") +
                glob.glob(R + "/frontier_full/n*/strong/scaling_frontier.csv")):
    for r in csv.DictReader(open(f)):
        if r["study"] == "strong" and r["status"] == "ok":
            series.setdefault(int(r["triangles"]), {}).setdefault(int(r["nodes"]), []).append(
                float(r["ms_per_step"]))
# Drop the 2.56G mesh past 256 nodes (<1M tris/GCD); the fall-off is discussed in the text.
DROP = {2560098304: {512, 1024}}
mean = {t: {n: sum(v) / len(v) for n, v in d.items() if n not in DROP.get(t, ())}
        for t, d in series.items()}
order = sorted(mean, reverse=True)            # largest mesh first = series 1


def thin(ticks, r):
    """Node-count labels, dropping any within a factor r of the next kept one (right to left)."""
    keep, last = set(), None
    for t in sorted(ticks, reverse=True):
        if last is None or last / t >= r:
            keep.add(t); last = t
    return FuncFormatter(lambda v, _: "%d" % v if round(v) in keep else "")

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1f9e6e"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = dict(zip(order, (GREEN, BLUE, ORANGE)))
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]},
                           facecolor=SURF)
allnodes = sorted({n for t in mean for n in mean[t]})
for ax in (a, b):
    ax.set_facecolor(SURF); ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(allnodes))
    ax.xaxis.set_major_formatter(thin(allnodes, 1.3 if ax is a else 3.5))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("Frontier nodes (8 MI250X GCDs each)")


def label(t):
    return "%.3gT triangles" % (t / 1e12) if t >= 1e12 else "%.3gG triangles" % (t / 1e9)


# (a) speedup relative to each mesh's smallest node count; ideal = N / N0
a.set_yscale("log", base=2)
smax = max(max(mean[t]) / min(mean[t]) for t in order)
ticks = [2 ** k for k in range(0, len(bin(int(smax))) - 2)]
a.yaxis.set_major_locator(FixedLocator(ticks))
a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d×" % v))
a.yaxis.set_minor_locator(NullLocator())
for t in order:
    ns = sorted(mean[t]); n0 = ns[0]; t0 = mean[t][n0]
    a.plot(ns, [n / n0 for n in ns], color=COL[t], lw=1.2, ls=(0, (4, 3)), alpha=0.8)
    sp = [t0 / mean[t][n] for n in ns]
    a.plot(ns, sp, color=COL[t], lw=2, marker="o", ms=7, mec=SURF, mew=2, zorder=3,
           label="%s  (ref. %d nodes)" % (label(t), n0))
    right = ns[-1] < allnodes[-1] / 2          # room to the right of the last point?
    a.annotate("%.1f× (ideal %.3g×)" % (sp[-1], ns[-1] / n0), (ns[-1], sp[-1]),
               xytext=(10, 0) if right else (-10, -4), textcoords="offset points",
               ha="left" if right else "right", va="center" if right else "top",
               color=INK, fontsize=9.5)
a.plot([], [], color=INK2, lw=1.2, ls=(0, (4, 3)), label="ideal  (N / N₀)")
a.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
a.set_title("ANUGA strong scaling on Frontier", loc="left", color=INK,
            fontsize=12.5, fontweight="bold")

# (b) parallel efficiency = speedup / ideal
b.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)))
for t in order:
    ns = sorted(mean[t]); n0 = ns[0]; t0 = mean[t][n0]
    eff = [t0 * n0 / (mean[t][n] * n) for n in ns]
    b.plot(ns, eff, color=COL[t], lw=2, marker="o", ms=7, mec=SURF, mew=2, zorder=3)
    b.annotate("%.1f%%\n%.2gM tris/GCD" % (100 * eff[-1], t / (8 * ns[-1]) / 1e6), (ns[-1], eff[-1]),
               xytext=(0, 9) if t == order[-1] else (10, -10), textcoords="offset points",
               ha="center" if t == order[-1] else "right", va="bottom" if t == order[-1] else "top",
               color=INK, fontsize=9.5)
b.set_ylim(0.9, 1.05); b.set_ylabel("efficiency  (t₀N₀ / tN N)")
b.set_title("Parallel efficiency", loc="left", color=INK, fontsize=12.5, fontweight="bold")

fig.text(0.01, 0.01, "Dam break, ader2/scatter, 1-D slab, ny = 2048, 100 steps x 3 repeats (best). "
         "Speedup and efficiency relative to each mesh's smallest node count.", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = sys.argv[1]
fig.savefig(out, dpi=160, facecolor=SURF)
print(out, {label(t): sorted(mean[t]) for t in order})
