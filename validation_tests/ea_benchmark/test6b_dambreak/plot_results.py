"""Figures for the EA Test 6B report: set-up, levels and speeds, checks.

Exits non-zero if a check fails, so the validation report marks the case.
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import anuga  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import ea_common as ea  # noqa: E402
import checks  # noqa: E402

args = anuga.get_args()

# Set-up: the whole channel, and the reservoir, dam and building close up.
with open('test6b_geometry.json') as f:
    geo = json.load(f)
points = ea.read_points(ea.data_path('Test6', 'Test6Boutput.csv'))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.8), gridspec_kw={'height_ratios': [1, 3.2]})
for ax in (ax1, ax2):
    ax.add_patch(Polygon(geo['bounding_polygon'], closed=True, fc='#dbe9f6', ec='k', lw=1))
    ax.add_patch(Polygon(geo['building'], closed=True, fc='0.4', ec='k', lw=1))
    ax.set_aspect('equal')
ax1.set_xlim(-170, 1850)
ax1.set_ylim(-45, 45)
ax1.set_title('The closed channel, 1981 m x 73 m (dam at x = -16 to 0 m)', fontsize=10)
ax2.set_xlim(-160, 200)
ax2.set_ylim(-42, 42)
ax2.fill_between([-151.5, -16], -37.5, 35.5, color='C0', alpha=0.25, label='8 m deep at t = 0')
for n, (x, y) in points.items():
    ax2.plot(x, y, 'o', color='C3', ms=5)
    ax2.text(x + 3, y + 3, n, color='C3')
ax2.text(75, -30, '16 m x 8 m building', ha='center', fontsize=8)
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.legend(fontsize=8, loc='lower left')
ax2.set_title('Reservoir, dam with its 20 m opening, building and gauges', fontsize=10)
fig.tight_layout()
fig.savefig('setup.png', dpi=120)
plt.close(fig)

g = ea.load_gauges('test6b_gauges.csv')
ea.plot_gauges(g, checks.NAMES, 'stage', 'stage_plot.png', 'EA Test 6B: water level, first 10 min',
               'Level (m)', ncol=3, tmax=600.0,
               label='ANUGA (%s)' % args.alg)
ea.plot_gauges(g, checks.NAMES, 'speed', 'speed_plot.png', 'EA Test 6B: speed, first 10 min',
               'Speed (m/s)', ncol=3, tmax=600.0,
               label='ANUGA (%s)' % args.alg)

c = checks.evaluate(args.alg)
c.figure('checks.png', 'EA Test 6B checks (%s)' % args.alg)
c.report('    ')
sys.exit(1 if c.failed() else 0)
