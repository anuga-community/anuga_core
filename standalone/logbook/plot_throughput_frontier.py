import csv, glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# Aggregate throughput on Frontier, every weak and strong run on one log-log axis.
# Ideal = 1-node weak throughput x nodes.
# Usage: python3 plot_throughput_frontier.py out.png
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "results")
files = (glob.glob(R + "/frontier_n*/scaling_frontier.csv") +
         glob.glob(R + "/frontier_strong/n*/*/scaling_frontier.csv") +
         glob.glob(R + "/frontier_full/n*/*/scaling_frontier.csv"))
series = {}                                   # (study, tris/GCD-or-mesh) -> {nodes: [cell-steps/s]}
for f in sorted(files):
    for r in csv.DictReader(open(f)):
        if r["status"] != "ok":
            continue
        key = ("weak", int(r["tris_per_rank"])) if r["study"] == "weak" else ("strong", int(r["triangles"]))
        series.setdefault(key, {}).setdefault(int(r["nodes"]), []).append(
            float(r["mcellsteps_per_s"]) * 1e6)
# Drop the 2.56G mesh past 256 nodes (<1M tris/GCD); the fall-off is discussed in the text.
DROP = {("strong", 2560098304): {512, 1024}}
best = {k: {n: max(v) for n, v in d.items() if n not in DROP.get(k, ())}
        for k, d in series.items()}


def thin(ticks, r):
    """Node-count labels, dropping any within a factor r of the next kept one (right to left)."""
    keep, last = set(), None
    for t in sorted(ticks, reverse=True):
        if last is None or last / t >= r:
            keep.add(t); last = t
    return FuncFormatter(lambda v, _: "%d" % v if round(v) in keep else "")

BLUE, ORANGE, GREEN, INKC = "#2a78d6", "#eb6834", "#1f9e6e", "#0b0b0b"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURF)
ax.set_facecolor(SURF)
allnodes = sorted({n for d in best.values() for n in d})
ax.set_xscale("log", base=2); ax.set_yscale("log", base=10)
ax.xaxis.set_major_locator(FixedLocator(allnodes))
ax.xaxis.set_major_formatter(thin(allnodes, 1.3))
ax.xaxis.set_minor_locator(NullLocator())
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)


def fmt(v):
    for u, s in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M")):
        if v >= u:
            return "%.3g%s" % (v / u, s)
    return "%.3g" % v


ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
weak = [k for k in best if k[0] == "weak"]
strong = sorted((k for k in best if k[0] == "strong"), key=lambda k: -k[1])
w = best[weak[0]]
base = w[min(w)] / min(w)                     # per-node throughput at the smallest weak run
ax.plot([allnodes[0], allnodes[-1]], [base * allnodes[0], base * allnodes[-1]],
        color=INK2, lw=1, ls=(0, (4, 3)), label="ideal (1-node weak × N)")
ns = sorted(w)
ax.plot(ns, [w[n] for n in ns], color=INKC, lw=2, marker="s", ms=6, mec=SURF, mew=1.5, zorder=3,
        label="weak, %.0fM tris/GCD" % (weak[0][1] / 1e6))
for k, c in zip(strong, (GREEN, BLUE, ORANGE)):
    d = best[k]; ns = sorted(d)
    lab = ("%.3gT" % (k[1] / 1e12)) if k[1] >= 1e12 else ("%.3gG" % (k[1] / 1e9))
    ax.plot(ns, [d[n] for n in ns], color=c, lw=2, marker="o", ms=7, mec=SURF, mew=2, zorder=4,
            label="strong, %s triangles" % lab)
top = max((d[n], n) for d in best.values() for n in d)
ax.annotate("%s triangle-steps/s\n%d nodes, %d GCDs" % (fmt(top[0]), top[1], 8 * top[1]), (top[1], top[0]),
            xytext=(-16, 2), textcoords="offset points", ha="right", va="bottom", color=INK, fontsize=9.5)
ax.set_xlabel("Frontier nodes (8 MI250X GCDs each)")
ax.set_ylabel("triangle-steps per second")
ax.set_title("ANUGA aggregate throughput on Frontier", loc="left", color=INK,
             fontsize=12.5, fontweight="bold")
ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
fig.text(0.01, 0.01, "Dam break, ader2/scatter, 1-D slab, ny = 2048, 100 steps x 3 repeats (best). "
         "Strong series sit below ideal only by their efficiency loss.", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = sys.argv[1]
fig.savefig(out, dpi=160, facecolor=SURF)
print(out, fmt(top[0]), top[1])
