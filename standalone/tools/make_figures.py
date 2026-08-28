#!/usr/bin/env python
"""Slide figures from the measured results (build/*/SUMMARY.txt etc.)."""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'build', 'figures'); os.makedirs(OUT, exist_ok=True)
TD = '/g/data/bm55/jlv900/tiles1sqm'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'       # categorical slots 1-3
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#d9d8d3', '#fcfcfb'
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.dpi': 200})


def style(ax, ygrid=True):
    ax.grid(axis='y' if ygrid else 'x', color=GRID, lw=1); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def rank_lines(path):
    d = {}
    for ln in open(path):
        m = re.match(r'rank (\d+): (\d+) owned, (\d+) ghost, (\d+) active cells/step, ([\d.]+) ms/step kernels', ln)
        if m: d[int(m[1])] = (int(m[2]), int(m[3]), int(m[4]), float(m[5]))
    return d


def wall(path):
    for ln in open(path):
        m = re.search(r'([\d.]+) ms/step,', ln)
        if m: return float(m[1])


# ---------------------------------------------------------------- 1. slab scaling
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
np_ = [1, 2, 4]
runs = {'dam 64M (RK2, cell)': [1528, 2844, 5367], 'dam 144M (RK2, cell)': [1482, 2812, 5458],
        'dam 144M scatter+active': [1167, 2193, 4310]}
for (name, v), c in zip(runs.items(), [C1, C2, C3]):
    ax.plot(np_, v, '-o', color=c, lw=2, ms=8, mec=SURF, mew=2, label=name)
ax.plot(np_, [1482 * n for n in np_], ':', color=GRID, lw=2, label='linear')
ax.set_xticks(np_); ax.set_xlabel('H200 GPUs'); ax.set_ylabel('Mcell-steps / s')
ax.set_title('Strong scaling, slab MPI (bit-exact vs serial)', loc='left', fontsize=12)
ax.legend(frameon=False, fontsize=9); style(ax)
ax.annotate('92%', (4, 5458), xytext=(6, -14), textcoords='offset points', fontsize=10, color=INK2)
ax = axes[1]
ax.bar(['1', '2', '4'], [24.0, 25.4, 25.7], color=C1, width=0.5)
for i, v in enumerate([24.0, 25.4, 25.7]): ax.text(i, v + 0.4, f'{v:.1f}', ha='center', fontsize=10, color=INK2)
ax.set_ylim(0, 32); ax.set_xlabel('H200 GPUs (36M triangles each)'); ax.set_ylabel('ms / step')
ax.set_title('Weak scaling: 93% at 4 GPUs', loc='left', fontsize=12); style(ax)
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_slab_scaling.png'); plt.close(fig)

# ------------------------------------------- 2. the imbalance that motivates everything
fig, ax = plt.subplots(figsize=(7.5, 4))
labels = ['dam\ncell', 'dam\nscatter', 'dam\nscatter+active', 'river\ncell', 'river\nscatter', 'river\nscatter+active']
eff = [92, 91, 92, 89, 87, 38]
cols = [C1] * 5 + [C2]
ax.bar(labels, eff, color=cols, width=0.55)
for i, v in enumerate(eff): ax.text(i, v + 1.5, f'{v}%', ha='center', fontsize=10, color=INK2)
ax.set_ylim(0, 105); ax.set_ylabel('parallel efficiency at 4 GPUs')
ax.set_title('144M triangles, x-slab partition: the active set breaks the geometric partition', loc='left', fontsize=11, pad=26)
ax.text(0.0, 1.02, 'river + active set: rank 0 owns the whole reservoir, 53% of its cells active vs 18% average',
        transform=ax.transAxes, fontsize=9.5, color=INK2)
style(ax); fig.tight_layout(); fig.savefig(f'{OUT}/fig2_slab_active_imbalance.png'); plt.close(fig)

# ------------------------------------------------ 3. per-rank kernel time, 1 sqm block
R2 = f'{ROOT}/build/rehearsal2'
post_tri, post_con = rank_lines(f'{R2}/bal_tri.txt'), rank_lines(f'{R2}/bal_contig.txt')
pre_tri = {i: v for i, v in enumerate([37.02, 44.14, 29.23, 22.67, 24.18, 6.85, 1.88, 2.69])}
pre_con = {i: v for i, v in enumerate([22.16, 25.53, 23.18, 14.66, 24.56, 14.66, 23.14, 19.37])}
panels = [('triangle-balanced\n26 tiles', pre_tri, 44.8), ('wetness-weighted\n26 tiles', pre_con, 27.3),
          ('wetness-weighted,\n13 wet tiles split -> 63 tiles', {k: v[3] for k, v in post_con.items()}, wall(f'{R2}/bal_contig.txt'))]
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, (name, d, w), c in zip(axes, panels, [C2, C1, C3]):
    ranks = sorted(d); vals = [d[r] for r in ranks]
    ax.bar([str(r) for r in ranks], vals, color=c, width=0.6)
    ax.axhline(w, color=INK2, lw=1, ls='--'); ax.text(-0.4, w + 1, f'wall {w:.1f} ms/step', ha='left', fontsize=9, color=INK2)
    ax.set_title(f'{name}\nimbalance {100 * (max(vals) / min(vals) - 1):.0f}%', fontsize=11); ax.set_xlabel('GPU rank (2 nodes)'); style(ax)
    ax.set_ylim(0, 52)
axes[0].set_ylabel('per-rank kernel time (ms / step)')
fig.suptitle('Real 1 m$^2$ tiles, 301M triangles, 8 H200s: coarse stats -> weight -> split -> assign', x=0.01, ha='left', fontsize=12)
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_1sqm_balance_ranks.png'); plt.close(fig)

# ------------------------------------------------------- 4. wall time summary bars
fig, ax = plt.subplots(figsize=(8, 3.8))
names = ['np=4, one node\nfull mesh', 'np=8, two nodes\nfull mesh', 'np=8 active set\ntriangle-balanced', 'np=8 active set\nweighted', 'np=8 active set\nweighted + split']
vals = [71.2, 35.1, 41.9, 27.3, 24.6]
cols = [C1, C1, C2, C3, C3]
ax.barh(names[::-1], vals[::-1], color=cols[::-1], height=0.55)
for i, v in enumerate(vals[::-1]): ax.text(v + 0.8, i, f'{v:.1f}', va='center', fontsize=10, color=INK2)
ax.set_xlabel('ms / step (lower is better)'); ax.set_xlim(0, 80)
ax.set_title('Same 301M-triangle 1 m$^2$ block; every run bit-exact vs the np=4 golden', loc='left', fontsize=11)
style(ax, ygrid=False); ax.grid(axis='x', color=GRID); fig.tight_layout(); fig.savefig(f'{OUT}/fig4_1sqm_walltime.png'); plt.close(fig)

# ------------------------------------------------------------------ 5. tile maps
def read_index(p):
    rows = [l.split() for l in open(p).read().splitlines()[1:] if l.strip()]
    return {int(r[0]): (int(r[1]), *map(float, r[4:8])) for r in rows}
full = read_index(f'{TD}/index.txt'); blk = read_index(f'{TD}/index_lake48.txt')
stats = {}
for l in open(f'{R2}/stats_tri.txt'):
    if not l.startswith('#'): t, n, f, r = l.split(); stats[int(t)] = float(f)
assign = [int(x) for x in open(f'{R2}/assign_contig.txt').read().split()]
ids_blk = [int(l.split()[0]) for l in open(f'{TD}/index_lake48.txt').read().splitlines()[1:] if l.strip()]
poly = np.loadtxt('/scratch/bm55/jlv900/dev/mesh_exp/cdac_script/Delta_11372_sqkm.csv', delimiter=',') - [291883.0, 2145325.0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.35, 1]})
ax = axes[0]
for t, (n, x0, y0, x1, y1) in full.items():
    ax.add_patch(Rectangle((x0 / 1e3, y0 / 1e3), (x1 - x0) / 1e3, (y1 - y0) / 1e3, fc='#e9e8e3', ec=SURF, lw=0.4))
ax.plot(poly[:, 0] / 1e3, poly[:, 1] / 1e3, color=INK2, lw=0.8)
bx = [min(v[1] for v in blk.values()) / 1e3, max(v[3] for v in blk.values()) / 1e3]
by = [min(v[2] for v in blk.values()) / 1e3, max(v[4] for v in blk.values()) / 1e3]
ax.add_patch(Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0], fc='none', ec=C2, lw=2))
ax.set_aspect('equal'); ax.set_xlabel('km'); ax.set_ylabel('km')
ax.set_title(f'11,372 km$^2$ delta: {len(full)} conforming tiles, 17.45 G triangles at 1 m$^2$', loc='left', fontsize=11)
ax.text(bx[1] + 2, by[1], 'rehearsal block\n(synthetic lake)', color=C2, fontsize=9, va='top')
style(ax, ygrid=False); ax.grid(False)
ax = axes[1]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue', ['#e9eef7', C1])
for t in ids_blk:
    n, x0, y0, x1, y1 = blk[t]
    ax.add_patch(Rectangle((x0 / 1e3, y0 / 1e3), (x1 - x0) / 1e3, (y1 - y0) / 1e3, fc=cmap(stats.get(t, 0.0)), ec=SURF, lw=1))
    if (x1 - x0) > 900 and (y1 - y0) > 900:
        ax.text((x0 + x1) / 2e3, (y0 + y1) / 2e3, str(assign[ids_blk.index(t)]), ha='center', va='center', fontsize=7 if t >= 10000 else 9, color=INK)
ax.set_xlim(bx[0] - 0.5, bx[1] + 0.5); ax.set_ylim(by[0] - 0.5, by[1] + 0.5); ax.set_aspect('equal'); ax.set_xlabel('km')
ax.set_title('Block: fill = fraction of steps active, number = GPU rank', loc='left', fontsize=11)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label='active fraction')
style(ax, ygrid=False); ax.grid(False)
fig.tight_layout(); fig.savefig(f'{OUT}/fig5_tile_map.png'); plt.close(fig)

# ------------------------------------------------------------ 6. 300 m^2 balance
fig, ax = plt.subplots(figsize=(7.5, 3.6))
names = ['triangle-balanced', 'weighted, contig', 'weighted, lpt']
kern = [(1.05, 1.58), (1.14, 1.25), (1.17, 1.28)]; wl = [1.80, 1.71, 2.72]; gh = ['10-24k', '8-23k', '67-98k']
y = np.arange(3)
ax.barh(y + 0.18, [k[1] for k in kern], height=0.32, color=C1, label='slowest rank, kernel time')
ax.barh(y - 0.18, wl, height=0.32, color=C2, label='wall time')
for i in range(3): ax.text(wl[i] + 0.04, y[i] - 0.18, f'{wl[i]:.2f} ms  ({gh[i]} ghosts/rank)', va='center', fontsize=9, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_xlim(0, 3.6); ax.set_xlabel('ms / step')
ax.legend(frameon=False, fontsize=9, loc='lower right')
ax.set_title('58M synthetic delta, 4 H200s: balance the work, keep ranks contiguous', loc='left', fontsize=11)
style(ax, ygrid=False); ax.grid(axis='x', color=GRID); fig.tight_layout(); fig.savefig(f'{OUT}/fig6_300sqm_contig_vs_lpt.png'); plt.close(fig)

# ---------------------------------------------------------------- 7. pipeline
fig, ax = plt.subplots(figsize=(12, 4.6)); ax.axis('off')
def box(x, y, top, sub, col, w=0.21, h=0.30):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01', fc='#e9eef7' if col == C1 else '#fdeee6', ec=col, lw=1.5, transform=ax.transAxes))
    ax.text(x + w / 2, y + h * 0.62, top, ha='center', va='center', fontsize=10.5, transform=ax.transAxes, fontweight='bold')
    ax.text(x + w / 2, y + h * 0.22, sub, ha='center', va='center', fontsize=8.2, color=INK2, transform=ax.transAxes)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes, arrowstyle='-|>', mutation_scale=14, color=INK2, lw=1.2))
yc, yf = 0.56, 0.06
ax.text(0.0, yc + 0.34, 'coarse set: 300 m$^2$, 158 tiles, 58M triangles  --  only its wetness is kept', fontsize=10, color=C2, transform=ax.transAxes)
box(0.00, yc, 'tile + mesh', 'split_delta, mesh_tile\n15 min / 18 cores', C2)
box(0.25, yc, 'per-tile init', 'bed, stage, friction\nsampled per tile', C2)
box(0.50, yc, 'coarse run', '--active-set --tile-stats\n4 GPUs, minutes', C2)
box(0.75, yc, 'weights', 'active fraction\nper tile', C2)
ax.text(0.0, yf + 0.34, 'fine set: 1 m$^2$, 1441 tiles, 17.45 G triangles  --  same 3 km grid, same origin', fontsize=10, color=C1, transform=ax.transAxes)
box(0.00, yf, 'tile + mesh', 'split_delta, mesh_tile\n26 min / 104 cores', C1)
box(0.25, yf, 'per-tile init', 'tiles_to_bmesh --compact\n18 min / 104 cores, 391 GB', C1)
box(0.50, yf, 'assign + split', 'tile_assign (contig, Morton)\nrefine_tiles: 13 -> 50, minutes', C1)
box(0.75, yf, 'launch', '--tiles --assign\nranks read own tiles + strips', C1)
for x in (0.212, 0.462, 0.712):
    arrow(x, yc + 0.15, x + 0.036, yc + 0.15); arrow(x, yf + 0.15, x + 0.036, yf + 0.15)
arrow(0.855, yc - 0.005, 0.66, yf + 0.31)
ax.text(0.99, 0.47, 'weights cross over:\ncoarse tile -> the fine tiles it covers', fontsize=8.5, color=INK2, transform=ax.transAxes, ha='right')
ax.text(0.0, 1.04, 'No rank ever sees the global mesh: tiles are the partition', fontsize=12, va='bottom', transform=ax.transAxes)
fig.savefig(f'{OUT}/fig7_pipeline.png', bbox_inches='tight'); plt.close(fig)

print('\n'.join(sorted(glob.glob(f'{OUT}/*.png'))))
