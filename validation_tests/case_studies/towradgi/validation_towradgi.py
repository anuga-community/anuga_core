#!/usr/bin/env python
"""
Towradgi Validation Script - Standard Rate_operator Setup

This script uses the NORMAL Rate_operator setup (one per rainfall polygon)
without any GPU optimizations. Used for validating that the standard code
path works correctly with MPI and GPU modes.

Based on run_towradgi.py but with simplified settings for testing.

Usage:
    # Run GPU mode and validate against baseline:
    python validation_towradgi.py --validate --mode 2

    # Run and dump results to JSON:
    python validation_towradgi.py --output results.json

    # Custom parameters (ignored when --validate is used):
    python validation_towradgi.py -ft 300 -ys 30 --scale 1.0 --output quick.json

    # CPU mode baseline generation:
    python validation_towradgi.py --mode 0 --output new_baseline.json

    # Run with profiling:
    python validation_towradgi.py --mode 2 --profile

Custom args: --mode, --output, --validate, --baseline, --no-rain, --scale, --profile
ANUGA args: -ft (finaltime), -ys (yieldstep), -alg (algorithm), etc.
"""

# ------------------------------------------------------------------------------
# IMPORT NECESSARY MODULES
# ------------------------------------------------------------------------------
from project import *
from anuga import distribute, myid, numprocs, finalize, barrier
from anuga import Rate_operator
from anuga import Domain
from anuga import create_mesh_from_regions
from anuga import read_polygon
from anuga import Polygon_function
from anuga import file_function

import anuga.utilities.spatialInputUtil as su

from os.path import join
import glob
import os
import sys
import numpy
import time
import json
import argparse
import anuga

# ------------------------------------------------------------------------------
# COMMAND LINE ARGUMENTS (parsed before ANUGA sees sys.argv)
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Towradgi Validation Script', add_help=False)
parser.add_argument('--mode', type=int, default=2, choices=[0, 1, 2],
                    help='Multiprocessor mode: 0=serial, 1=OpenMP CPU, 2=OpenMP GPU (default: 2)')
parser.add_argument('--output', type=str, default=None,
                    help='Output JSON file for results (default: None)')
parser.add_argument('--validate', action='store_true',
                    help='Validate against baseline.json')
parser.add_argument('--baseline', type=str, default='baseline.json',
                    help='Baseline JSON file for validation (default: baseline.json)')
parser.add_argument('--no-rain', action='store_true',
                    help='Disable rainfall for debugging')
parser.add_argument('--scale', type=float, default=0.1,
                    help='Mesh scale factor (default: 0.1). Ignored with --validate.')
parser.add_argument('--profile', action='store_true',
                    help='Enable cProfile profiling and print top functions')

# Parse our custom args, leave the rest for ANUGA
cli_args, remaining = parser.parse_known_args()

# Replace sys.argv with remaining args so ANUGA's parser doesn't see our custom args
sys.argv = [sys.argv[0]] + remaining

# ------------------------------------------------------------------------------
# CONFIGURATION - Edit these for testing
# ------------------------------------------------------------------------------

# Baseline validation parameters (must match baseline.json)
BASELINE_SCALE = 0.1
BASELINE_YIELDSTEP = 60.0
BASELINE_FINALTIME = 600.0

# Get ANUGA's parsed args (for -ft, -ys, -alg etc)
anuga_args = anuga.get_args()

# When validating, enforce baseline parameters
if cli_args.validate:
    if myid == 0:
        print(f'NOTE: --validate mode using baseline parameters:')
        print(f'  scale={BASELINE_SCALE}, yieldstep={BASELINE_YIELDSTEP}, finaltime={BASELINE_FINALTIME}')

    yieldstep = BASELINE_YIELDSTEP
    outputstep = BASELINE_YIELDSTEP
    finaltime = BASELINE_FINALTIME
    scale = BASELINE_SCALE
else:
    # Allow custom values when not validating
    # Use ANUGA's -ft and -ys if provided, otherwise defaults
    finaltime = getattr(anuga_args, 'finaltime', BASELINE_FINALTIME)
    yieldstep = getattr(anuga_args, 'yieldstep', BASELINE_YIELDSTEP)
    outputstep = yieldstep
    scale = cli_args.scale

# Multiprocessor mode: 0 = serial, 1 = OpenMP CPU, 2 = OpenMP GPU
multiprocessor_mode = cli_args.mode

# Features
useCulverts = False     # Disable culverts for simpler validation
useCheckpointing = False
useRainfall = not cli_args.no_rain

verbose = True

# Results collection for JSON output
results = {
    "description": "Towradgi validation run",
    "parameters": {
        "finaltime": finaltime,
        "yieldstep": yieldstep,
        "scale": scale,
        "multiprocessor_mode": multiprocessor_mode,
        "numprocs": numprocs,
        "useRainfall": useRainfall
    },
    "yieldsteps": []
}

# ------------------------------------------------------------------------------
# Helper function
# ------------------------------------------------------------------------------
def read_polygon_list(poly_list):
    result = []
    for i in range(len(poly_list)):
        result.append((read_polygon(poly_list[i][0]), poly_list[i][1]))
    return result

# ------------------------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------------------------

if myid == 0:
    print('='*70)
    print('Towradgi Validation Script - Standard Rate_operator Setup')
    print('='*70)
    print(f'  multiprocessor_mode = {multiprocessor_mode}')
    print(f'  numprocs = {numprocs}')
    print(f'  finaltime = {finaltime}')
    print(f'  scale = {scale}')
    print('='*70)

if myid == 0 and not os.path.isdir('DEM_bridges'):
    msg = """
################################################################################
#
# Could not find data directories
#
# You can download these directories using the data_download.py script.
# This will download over 86 MB of data!
#
################################################################################
"""
    raise Exception(msg)

alg = anuga_args.alg

# ------------------------------------------------------------------------------
# Setup parameters
# ------------------------------------------------------------------------------
maximum_triangle_area = 1000
base_friction = 0.04
checkpoint_time = max(600/scale, 60)
checkpoint_dir = 'CHECKPOINTS'

basename = join('DEM_bridges', 'towradgi')
domain_name = 'Towradgi_validation'
meshname = join('DEM_bridges', 'towradgi.tsh')
func = file_function(join('Forcing', 'Tide', 'Pioneer.tms'), quantities='rainfall')

# ------------------------------------------------------------------------------
# Create domain
# ------------------------------------------------------------------------------

# Catchment regions
CatchmentList = [
    [join('Model', 'Bdy', 'Catchment.csv'), scale*100.0],
    [join('Model', 'Bdy', 'FineCatchment.csv'), scale*36.0],
    [join('Model', 'Bdy', 'CreekBanks.csv'), 8.0]
]

# Manning's roughness - full list from run_towradgi.py
ManningList = [
   [ join('Model', 'Mannings', '1.csv'),0.04], #park
   [ join('Model', 'Mannings', '2.csv'),0.15],
   [ join('Model', 'Mannings', '3.csv'),0.15],
   [ join('Model', 'Mannings', '4.csv'),0.04],
   [ join('Model', 'Mannings', '5.csv'),0.15],
   [ join('Model', 'Mannings', '6.csv'),0.15],
   [ join('Model', 'Mannings', '7.csv'),0.15],
   [ join('Model', 'Mannings', '8.csv'),0.15],
   [ join('Model', 'Mannings', '9.csv'),0.04], #park
   [ join('Model', 'Mannings', '10.csv'), 0.15],
   [ join('Model', 'Mannings', '11.csv'), 0.15],
   [ join('Model', 'Mannings', '12.csv'), 0.15],
   [ join('Model', 'Mannings', '13.csv'), 0.04],
   [ join('Model', 'Mannings', '14.csv'), 0.15],
   [ join('Model', 'Mannings', '15.csv'), 0.15],
   [ join('Model', 'Mannings', '16.csv'), 0.15],
   [ join('Model', 'Mannings', '17.csv'), 0.15],
   [ join('Model', 'Mannings', '18.csv'), 0.045],
   [ join('Model', 'Mannings', '18a.csv'), 0.15],
   [ join('Model', 'Mannings', '18b.csv'), 0.15],
   [ join('Model', 'Mannings', '18c.csv'), 0.15],
   [ join('Model', 'Mannings', '18d.csv'), 0.15],
   [ join('Model', 'Mannings', '18e.csv'), 0.08], #cokeworks site
   [ join('Model', 'Mannings', '19.csv'), 0.15],
   [ join('Model', 'Mannings', '20.csv'), 0.15],
   [ join('Model', 'Mannings', '21.csv'), 0.15],
   [ join('Model', 'Mannings', '22.csv'), 0.15],
   [ join('Model', 'Mannings', '23.csv'), 0.15],
   [ join('Model', 'Mannings', '24.csv'), 0.05],
   [ join('Model', 'Mannings', '25.csv'), 0.15],
   [ join('Model', 'Mannings', '26.csv'), 0.15],
   [ join('Model', 'Mannings', '27.csv'), 0.15],
   [ join('Model', 'Mannings', '28.csv'), 0.15],
   [ join('Model', 'Mannings', '29.csv'), 0.15],
   [ join('Model', 'Mannings', '30.csv'), 0.15],
   [ join('Model', 'Mannings', '31.csv'), 0.15],
   [ join('Model', 'Mannings', '32.csv'), 0.15],
   [ join('Model', 'Mannings', '33.csv'), 0.15],
   [ join('Model', 'Mannings', '34.csv'), 0.15],
   [ join('Model', 'Mannings', '35.csv'), 0.15],
   [ join('Model', 'Mannings', '36.csv'), 0.05],
   [ join('Model', 'Mannings', '37.csv'), 0.15],
   [ join('Model', 'Mannings', '38.csv'), 0.15],
   [ join('Model', 'Mannings', '39.csv'), 0.15],
   [ join('Model', 'Mannings', '40.csv'), 0.15],
   [ join('Model', 'Mannings', '41.csv'), 0.15],
   [ join('Model', 'Mannings', '42.csv'), 0.15],
   [ join('Model', 'Mannings', '43.csv'), 0.15],
   [ join('Model', 'Mannings', '44.csv'), 0.15],
   [ join('Model', 'Mannings', '45.csv'), 0.15],
   [ join('Model', 'Mannings', '46.csv'), 0.15],
   [ join('Model', 'Mannings', '47.csv'), 0.15],
   [ join('Model', 'Mannings', '48.csv'), 0.15],
   [ join('Model', 'Mannings', '49.csv'), 0.15],
   [ join('Model', 'Mannings', '50.csv'), 0.15],
   [ join('Model', 'Mannings', '51.csv'), 0.15],
   [ join('Model', 'Mannings', '52.csv'), 0.15],
   [ join('Model', 'Mannings', '53.csv'), 0.15],
   [ join('Model', 'Mannings', '54.csv'), 0.15],
   [ join('Model', 'Mannings', '55.csv'), 0.15],
   [ join('Model', 'Mannings', '56.csv'), 0.15],
   [ join('Model', 'Mannings', '57.csv'), 0.15],
   [ join('Model', 'Mannings', '58.csv'), 0.15],
   [ join('Model', 'Mannings', '59.csv'), 0.08],
   [ join('Model', 'Mannings', '60.csv'), 0.15],
   [ join('Model', 'Mannings', '61.csv'), 0.08],
   [ join('Model', 'Mannings', '62.csv'), 0.15],
   [ join('Model', 'Mannings', '63.csv'), 0.08],
   [ join('Model', 'Mannings', '64.csv'), 0.15],
   [ join('Model', 'Mannings', '65.csv'), 0.15],
   [ join('Model', 'Mannings', '66.csv'), 0.15],
   [ join('Model', 'Mannings', '67.csv'), 0.15],
   [ join('Model', 'Mannings', '68.csv'), 0.15],
   [ join('Model', 'Mannings', '69.csv'), 0.15],
   [ join('Model', 'Mannings', '70.csv'), 0.15],
   [ join('Model', 'Mannings', '71.csv'), 0.05],
   [ join('Model', 'Mannings', '72.csv'), 0.15],
   [ join('Model', 'Mannings', '73.csv'), 0.15],
   [ join('Model', 'Mannings', '74.csv'), 0.15],
   [ join('Model', 'Mannings', '75.csv'), 0.15],
   [ join('Model', 'Mannings', '76.csv'), 0.15],
   [ join('Model', 'Mannings', '77.csv'), 0.07],
   [ join('Model', 'Mannings', '78.csv'), 0.15],
   [ join('Model', 'Mannings', '79.csv'), 0.15],
   [ join('Model', 'Mannings', '80.csv'), 0.15],
   [ join('Model', 'Mannings', '81.csv'), 0.15],
   [ join('Model', 'Mannings', '82.csv'), 0.15],
   [ join('Model', 'Mannings', '83.csv'), 0.15],
   [ join('Model', 'Mannings', '84.csv'), 0.15],
   [ join('Model', 'Mannings', '85.csv'), 0.15],
   [ join('Model', 'Mannings', '86.csv'), 0.15],
   [ join('Model', 'Mannings', 'Escarpement.csv'), 0.15],
   [ join('Model', 'Mannings', 'Railway.csv'), 0.04],
   [ join('Model', 'Creeks', 'creeks1.csv'), channel_manning],
   [ join('Model', 'Creeks', 'creeks2.csv'), channel_manning],
   [ join('Model', 'Creeks', 'creeks3.csv'), channel_manning],
   [ join('Model', 'Creeks', 'creeks4.csv'), channel_manning],
   [ join('Model', 'Creeks', 'creeks5.csv'), channel_manning],
   [ join('Model', 'Creeks', 'creeks6.csv'), channel_manning],
   # Buildings with high friction
   [ join('Model', 'Buildings', 'Building1.csv'),  10.0],
   [ join('Model', 'Buildings', 'Building4.csv'),  10.0],
   [ join('Model', 'Buildings', 'Building5.csv'),  10.0],
   [ join('Model', 'Buildings', 'Building6.csv'),  10.0],
   [ join('Model', 'Buildings', 'Building7.csv'),  10.0],
   [ join('Model', 'Buildings', 'Building8.csv'),  10.0],
   [ join('Model', 'Buildings', 'Building9.csv'),  10.0],
   [ join('Model', 'Buildings', 'Building10.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building11.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building12.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building13.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building14.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building15.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building16.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building17.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building18.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building19.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building20.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building21.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building22.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building23.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building24.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building25.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building26.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building27.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building28.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building29.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building30.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building31.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building32.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building33.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building34.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building35.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building36.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building37.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building38.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building39.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building40.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building41.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building42.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building43.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building44.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building45.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building46.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building47.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building48.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building49.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building50.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building51.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building52.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building53.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building54.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building55.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building56.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building57.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building62.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building63.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building64.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building65.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building66.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building67.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building68.csv'), 10.0],
   [ join('Model', 'Buildings', 'Building69.csv'), 10.0]
]

W = 303517
N = 6195670
E = 308570
S = 6193140

model_output_dir = 'MODEL_OUTPUTS'
try:
    os.mkdir(model_output_dir)
except:
    pass

# Make a list of the csv files in BREAKLINES
riverWall_csv_files = glob.glob('Model/Riverwalls/*.csv')
(riverWalls, riverWall_parameters) = su.readListOfRiverWalls(riverWall_csv_files)

if myid == 0:
    # --------------------------------------------------------------------------
    # Create mesh and domain on rank 0
    # --------------------------------------------------------------------------
    bounding_polygon = [[W, S], [E, S], [E, N], [W, N]]
    interior_regions = read_polygon_list(CatchmentList)

    create_mesh_from_regions(bounding_polygon,
                             boundary_tags={'south': [0], 'east': [1], 'north': [2], 'west': [3]},
                             maximum_triangle_area=maximum_triangle_area,
                             interior_regions=interior_regions,
                             breaklines=riverWalls.values(),
                             filename=meshname,
                             use_cache=False,
                             verbose=False)

    domain = Domain(meshname, use_cache=False, verbose=False)
    domain.set_flow_algorithm(alg)

    if not domain.get_using_discontinuous_elevation():
        raise Exception('This model requires discontinuous elevation solver')

    domain.set_datadir(model_output_dir)
    domain.set_name(domain_name)

    print(domain.statistics())

    # Apply friction
    print('FITTING polygon_function for friction')
    friction_list = read_polygon_list(ManningList)
    domain.set_quantity('friction', Polygon_function(
        friction_list, default=base_friction, geo_reference=domain.geo_reference))

    # Initial water level
    domain.set_quantity('stage', 0)

    # Elevation
    print('Loading elevation data...')
    try:
        elev_xyz = numpy.load(basename+'.npy')
    except:
        elev_xyz = numpy.genfromtxt(fname=basename+'.csv', delimiter=',')
        numpy.save(basename+'.npy', elev_xyz)

    from anuga.utilities.quantity_setting_functions import make_nearestNeighbour_quantity_function
    elev_fun_wrapper = make_nearestNeighbour_quantity_function(elev_xyz, domain)
    domain.set_quantity('elevation', elev_fun_wrapper, location='centroids')

else:
    domain = None

barrier()

# ------------------------------------------------------------------------------
# Distribute domain
# ------------------------------------------------------------------------------
if myid == 0 and verbose:
    print('DISTRIBUTING DOMAIN')

domain = distribute(domain, verbose=verbose)

barrier()

domain.quantities_to_be_stored = {'elevation': 2,
                                  'friction': 1,
                                  'stage': 2,
                                  'xmomentum': 2,
                                  'ymomentum': 2}

if myid == 0:
    print('CREATING RIVERWALLS')

domain.create_riverwalls(riverWalls)

barrier()

# ------------------------------------------------------------------------------
# APPLY RAINFALL - Standard method (one Rate_operator per polygon)
# ------------------------------------------------------------------------------
rain_operators = []

if useRainfall:
    if myid == 0:
        print('CREATING RAINFALL OPERATORS (standard method - one per polygon)')

    Rainfall_Gauge_directory = join('Forcing', 'Rainfall', 'Gauge')

    for filename in sorted(os.listdir(Rainfall_Gauge_directory)):
        Gaugefile = join(Rainfall_Gauge_directory, filename)
        Rainfile = join('Forcing', 'Rainfall', 'Hort', filename[0:-4]+'.tms')

        polygon = anuga.read_polygon(Gaugefile)
        rainfall = anuga.file_function(Rainfile, quantities='rate')

        op = Rate_operator(domain, rate=rainfall, factor=1.0e-3,
                           polygon=polygon, default_rate=0.0)
        rain_operators.append(op)

    if myid == 0:
        print(f'  Created {len(rain_operators)} Rate_operators')
else:
    if myid == 0:
        print('RAINFALL DISABLED for debugging')

def update_rainfall_quantity(t):
    pass  # No-op for standard method

barrier()

# ------------------------------------------------------------------------------
# BOUNDARY CONDITIONS
# ------------------------------------------------------------------------------
print(f'Available boundary tags on process {myid}: {domain.get_boundary_tags()}')

Bd = anuga.Dirichlet_boundary([0, 0, 0])
Bw = anuga.Time_boundary(domain=domain, function=lambda t: [func(t)[0], 0.0, 0.0])

domain.set_boundary({'west': Bd, 'south': Bd, 'north': Bd, 'east': Bw})

# ------------------------------------------------------------------------------
# Set multiprocessor mode (AFTER boundaries are set!)
# ------------------------------------------------------------------------------
if myid == 0:
    print(f'Setting multiprocessor_mode = {multiprocessor_mode}')

if multiprocessor_mode > 0:
    domain.set_multiprocessor_mode(multiprocessor_mode)
    if multiprocessor_mode == 2:
        domain.use_c_rk2_loop = True
else:
    if myid == 0:
        print('  (CPU serial mode - no GPU acceleration)')

if myid == 0:
    print('Starting evolution...')

# ------------------------------------------------------------------------------
# EVOLVE
# ------------------------------------------------------------------------------
barrier()
t0 = time.time()
step_count = 0
prev_step_count = 0

# Setup profiler if requested
if cli_args.profile:
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()

for t in domain.evolve(yieldstep=yieldstep, outputstep=outputstep, finaltime=finaltime):
    if myid == 0:
        domain.write_time()

    # Get step count for this yieldstep
    current_step_count = domain.get_step_count() if hasattr(domain, 'get_step_count') else 0
    steps_this_yield = current_step_count - prev_step_count if t > 0 else 0
    prev_step_count = current_step_count

    # Use built-in methods that handle MPI safely
    volume = domain.compute_total_volume()

    # Get delta_t range
    delta_t = domain.get_timestep() if hasattr(domain, 'get_timestep') else None

    # Sum up total Q from all rain operators on this rank
    local_total_Q = 0.0
    for op in rain_operators:
        if hasattr(op, 'local_influx'):
            q = op.local_influx
            # Handle both scalar and array cases
            if hasattr(q, '__len__'):
                local_total_Q += float(q.sum()) if len(q) > 0 else 0.0
            else:
                local_total_Q += float(q)

    # Use MPI allreduce (same pattern as get_water_volume)
    if numprocs > 1:
        from mpi4py import MPI
        global_total_Q = MPI.COMM_WORLD.allreduce(local_total_Q, op=MPI.SUM)
    else:
        global_total_Q = local_total_Q

    # Collect results (only on rank 0)
    if myid == 0:
        results["yieldsteps"].append({
            "time": float(t),
            "volume": round(volume, 2),
            "total_Q": round(global_total_Q, 4),
            "steps": steps_this_yield,
            "delta_t": round(delta_t, 6) if delta_t else None
        })

    # Print stats (only from rank 0)
    if myid == 0:
        print(f'  Volume: {volume:.2f} m^3, Total Rain Q: {global_total_Q:.4f} m^3')

# Stop profiler and print results
if cli_args.profile:
    profiler.disable()
    barrier()

    if myid == 0:
        print("\n" + "="*80)
        print("PROFILING RESULTS - Top 40 by cumulative time")
        print("="*80)
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(40)

        # Save to file for detailed analysis
        profile_file = 'profile.prof'
        profiler.dump_stats(profile_file)
        print(f"\nProfile saved to {profile_file} - view with: python -m pstats {profile_file}")

barrier()

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
if myid == 0:
    print('\n' + '='*70)
    print('FINAL SUMMARY')
    print('='*70)

for p in range(numprocs):
    if myid == p:
        print(f'Processor {myid}:')
        print(f'  Total time: {time.time()-t0:.2f} seconds')
        print(f'  Communication time: {domain.communication_time:.2f} seconds')
        print(f'  Reduction time: {domain.communication_reduce_time:.2f} seconds')
        print(f'  Broadcast time: {domain.communication_broadcast_time:.2f} seconds')
    barrier()

# ------------------------------------------------------------------------------
# JSON Output and Validation (rank 0 only)
# ------------------------------------------------------------------------------
if myid == 0:
    results["timing"] = {
        "total_seconds": round(time.time() - t0, 2),
        "communication_seconds": round(domain.communication_time, 2)
    }

    # Write JSON output if requested
    if cli_args.output:
        with open(cli_args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults written to: {cli_args.output}')

    # Validate against baseline if requested
    if cli_args.validate:
        print('\n' + '='*70)
        print('VALIDATION')
        print('='*70)

        baseline_file = cli_args.baseline
        if not os.path.exists(baseline_file):
            print(f'ERROR: Baseline file not found: {baseline_file}')
            validation_passed = False
        else:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)

            # Get tolerances from baseline or use defaults
            tol = baseline.get("tolerances", {})
            vol_rel_tol = tol.get("volume_rel", 0.001)      # 0.1% relative
            q_rel_tol = tol.get("total_Q_rel", 0.01)        # 1% relative
            steps_abs_tol = tol.get("steps_abs", 5)         # 5 steps absolute

            validation_passed = True
            validation_errors = []

            # Compare each yieldstep
            baseline_steps = {ys["time"]: ys for ys in baseline["yieldsteps"]}

            for result_ys in results["yieldsteps"]:
                t = result_ys["time"]
                if t not in baseline_steps:
                    continue

                baseline_ys = baseline_steps[t]

                # Volume check (relative)
                vol_result = result_ys["volume"]
                vol_baseline = baseline_ys["volume"]
                vol_rel_err = abs(vol_result - vol_baseline) / max(vol_baseline, 1.0)
                if vol_rel_err > vol_rel_tol:
                    validation_passed = False
                    validation_errors.append(
                        f't={t}: Volume {vol_result:.2f} vs baseline {vol_baseline:.2f} '
                        f'(rel err {vol_rel_err*100:.3f}% > {vol_rel_tol*100:.1f}%)')

                # Total Q check (relative, but handle zeros)
                q_result = result_ys["total_Q"]
                q_baseline = baseline_ys["total_Q"]
                if q_baseline > 0.0001:  # Only check if baseline has significant Q
                    q_rel_err = abs(q_result - q_baseline) / q_baseline
                    if q_rel_err > q_rel_tol:
                        validation_passed = False
                        validation_errors.append(
                            f't={t}: Total Q {q_result:.4f} vs baseline {q_baseline:.4f} '
                            f'(rel err {q_rel_err*100:.2f}% > {q_rel_tol*100:.1f}%)')

                # Steps check (absolute) - only if baseline has steps
                steps_result = result_ys.get("steps", 0)
                steps_baseline = baseline_ys.get("steps", 0)
                if steps_baseline > 0 and steps_result > 0:
                    steps_diff = abs(steps_result - steps_baseline)
                    if steps_diff > steps_abs_tol:
                        # This is a warning, not a failure (different partition = different steps)
                        print(f'  WARNING t={t}: Steps {steps_result} vs baseline {steps_baseline} '
                              f'(diff {steps_diff} > {steps_abs_tol})')

            # Print validation summary
            if validation_passed:
                print('VALIDATION PASSED')
                print(f'  All {len(results["yieldsteps"])} yieldsteps within tolerance')
                print(f'  Volume tolerance: {vol_rel_tol*100:.1f}% relative')
                print(f'  Total Q tolerance: {q_rel_tol*100:.1f}% relative')
            else:
                print('VALIDATION FAILED')
                for err in validation_errors:
                    print(f'  {err}')

        print('='*70)

# Merge output files
domain.sww_merge(delete_old=True)

finalize()
