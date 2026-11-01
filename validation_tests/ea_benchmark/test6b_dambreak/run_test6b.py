"""
EA 2D benchmark Test 6B: dam break past a building, at field scale.

The IMPACT laboratory case (Soares-Frazao and Zech 2002) scaled up by 20:
a channel with a dam that has a 20 m opening, 8 m of water behind it and
0.4 m in front, and a 16 m x 8 m building set at an angle downstream. All
boundaries are vertical walls. Manning n = 0.05, 2 m resolution, run to
30 min.

The geometry is taken from Test6BDEM.asc, where walls stand 10 m above a
flat floor (with a shallow 7 m fillet along each side wall, kept here):
the channel spans the DEM's open cells, the dam's two blocks are cut out
of the bounding polygon and the building is a hole. The DEM's partly
covered cells put the dam at -16 < x < 0 with its opening at -10 < y < 10.

Writes test6b_gauges.csv (level, depth and speed at G1-G6 every 1 s),
test6b_volume.txt (volumes and the largest speed in the domain) and
test6b.sww every minute.

For a mesh convergence study set EA_6B_MAX_AREA (default 4 m^2); the
outputs are then named test6b_a<area>_*. EA_6B_BUILDING picks how the
building is modelled: 'hole' (default, a reflective hole in the mesh),
'block' (the fitted rectangle as a breakline with the floor raised 10 m
inside it) or 'dem' (the DEM's own cells, sampled at the centroids, as the
grid-based packages saw it); the outputs gain a _<building> suffix. Finer meshes are best run in
'unified' compute mode (ANUGA_DEFAULT_COMPUTE_MODE=unified) on a GPU build.
"""

import os
import sys

import numpy as np
import anuga

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import ea_common as ea  # noqa: E402

args = anuga.get_args()
alg = args.alg
verbose = args.verbose

MAX_AREA = float(os.environ.get('EA_6B_MAX_AREA', 4.0))   # 4 = the 2 m grid
BUILDING = os.environ.get('EA_6B_BUILDING', 'hole')
NAME = 'test6b' if MAX_AREA == 4.0 else 'test6b_a%g' % MAX_AREA
if BUILDING != 'hole':
    NAME += '_' + BUILDING
FINAL_TIME = 1800.0
YIELD_STEP = 1.0
OUTPUT_STEP = 60.0
H_UP, H_DOWN = 8.0, 0.4
WALL = 5.0                 # DEM cells above this are walls

x_dem, y_dem, z_dem = ea.read_asc(ea.data_path('Test6', 'Test6BDEM.asc'))
points = ea.read_points(ea.data_path('Test6', 'Test6Boutput.csv'))

# Channel extent: the DEM's open rows and columns, to cell edges.
open_cells = z_dem < WALL
cs = x_dem[1] - x_dem[0]
rows = np.where(open_cells.any(axis=1))[0]
cols = np.where(open_cells.any(axis=0))[0]
x0, x1 = x_dem[cols[0]] - cs / 2, x_dem[cols[-1]] + cs / 2
y0, y1 = y_dem[rows[0]] - cs / 2, y_dem[rows[-1]] + cs / 2

# Dam blocks at -16 < x < 0, leaving the opening -10 < y < 10.
dam_x0, dam_x1, gap = -16.0, 0.0, 10.0
bounding_polygon = [[x0, y0], [dam_x0, y0], [dam_x0, -gap], [dam_x1, -gap],
                    [dam_x1, y0], [x1, y0], [x1, y1], [dam_x1, y1],
                    [dam_x1, gap], [dam_x0, gap], [dam_x0, y1], [x0, y1]]


def building_polygon():
    """The 16 m x 8 m building as a rectangle fitted to its DEM cells."""
    X, Y = np.meshgrid(x_dem, y_dem)
    in_building = (X > 10.0) & (X < 200.0) & (Y > y0) & (Y < y1)
    # Skip the low wall fillets: over 190 m of wall they would swamp the
    # covariance and turn the fit towards the channel axis.
    w = np.where(in_building & (z_dem > 0.2), np.clip(z_dem / 10.0, 0.0, 1.0), 0.0)
    xc, yc = np.sum(w * X) / w.sum(), np.sum(w * Y) / w.sum()
    dx, dy = (X - xc)[w > 0], (Y - yc)[w > 0]
    cov = np.cov(np.vstack((dx, dy)), aweights=w[w > 0])
    evals, evecs = np.linalg.eigh(cov)
    major, minor = evecs[:, 1], evecs[:, 0]
    a, b = 8.0, 4.0                                    # half sides
    corners = [(xc, yc) + s * a * major + r * b * minor
               for s, r in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    return [list(c) for c in corners]


building = building_polygon()

domain = anuga.create_domain_from_regions(
    bounding_polygon,
    boundary_tags={'wall': list(range(len(bounding_polygon)))},
    maximum_triangle_area=MAX_AREA,
    interior_holes=[building] if BUILDING == 'hole' else None,
    breaklines=[building + building[:1]] if BUILDING == 'block' else None,
    use_cache=False, verbose=False)

domain.set_name(NAME)
domain.set_flow_algorithm(alg)

# Floor only: drop the walls and the partly covered cells beside them.
z_floor = np.where(z_dem > 0.2, 0.0, z_dem)
domain.set_quantity('elevation',
                    ea.absolute(domain, ea.grid_function(x_dem, y_dem, z_floor)),
                    location='centroids')
if BUILDING != 'hole':
    xc, yc = domain.centroid_coordinates.T + [[domain.geo_reference.xllcorner],
                                               [domain.geo_reference.yllcorner]]
    if BUILDING == 'block':
        inside = anuga.inside_polygon(np.column_stack((xc, yc)), building)
    else:
        # The DEM cell under each centroid, as a grid model would see it.
        i = np.clip(np.round((yc - y_dem[0]) / cs).astype(int), 0, len(y_dem) - 1)
        j = np.clip(np.round((xc - x_dem[0]) / cs).astype(int), 0, len(x_dem) - 1)
        inside = np.where((xc > 10.0) & (xc < 200.0) & (z_dem[i, j] > 5.0))[0]
    elev = domain.quantities['elevation'].centroid_values
    elev[inside] = 10.0
    print('building: %d centroids raised to 10 m' % len(inside))
domain.set_quantity('friction', 0.05, location='centroids')


def initial_stage(x, y):
    return np.where(x < 0.0, H_UP, H_DOWN)


domain.set_quantity('stage', ea.absolute(domain, initial_stage),
                    location='centroids')
if BUILDING != 'hole':   # dry on top of the building
    stage = domain.quantities['stage'].centroid_values
    stage[:] = np.maximum(stage, domain.quantities['elevation'].centroid_values)

Br = anuga.Reflective_boundary(domain)
domain.set_boundary({'wall': Br, 'interior': Br} if BUILDING == 'hole' else {'wall': Br})

gauges = ea.Gauges(domain, points)
volume_0 = ea.domain_volume(domain)
max_speed = 0.0
for t in domain.evolve(yieldstep=YIELD_STEP, outputstep=OUTPUT_STEP,
                       finaltime=FINAL_TIME):
    gauges.sample()
    h = (domain.quantities['stage'].centroid_values
         - domain.quantities['elevation'].centroid_values)
    wet = h > 0.05
    if wet.any():
        u = np.hypot(domain.quantities['xmomentum'].centroid_values[wet],
                     domain.quantities['ymomentum'].centroid_values[wet]) / h[wet]
        max_speed = max(max_speed, float(u.max()))
    if verbose and abs(t % 60.0) < 1.0e-6:
        print(domain.timestepping_statistics())

gauges.save(NAME + '_gauges.csv')
with open(NAME + '_volume.txt', 'w') as f:
    f.write('initial %.6f\n' % volume_0)
    f.write('final %.6f\n' % ea.domain_volume(domain))
    f.write('max_speed %.6f\n' % max_speed)
    f.write('triangles %d\n' % domain.number_of_triangles)
print('compute mode %s' % domain.compute_mode)
