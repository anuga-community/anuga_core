#!/usr/bin/env python3
"""
Modified run_model_3.py to use pre-partitioned mesh

CHANGES FROM ORIGINAL:
1. Removed domain creation on rank 0 (lines 159-221)
2. Removed distribute() call (line 233)
3. Added sequential_distribute_load() - all ranks load in parallel
4. Everything else stays the same

Author: Girishchandra Y. (modified for partitioned mesh)
"""
# ------------------------------------------------------------------------------
# Import necessary modules
# ------------------------------------------------------------------------------
import os
import time
import sys
import numpy as np
import itertools
import datetime
import re
from discharge_data import setup_discharge_3
from tide_data import setup_tide_3
from rainfall_loader import RainfallLoader
import anuga
from anuga import Inlet_operator
from anuga import file_function
from anuga import Rate_operator
from anuga import Quantity
from anuga import myid, numprocs, finalize, barrier
from anuga.utilities import log as log
from anuga.abstract_2d_finite_volumes import quantity

t0 = time.time()
# --------------------------------------------------------------------------
# Setup parameters
# --------------------------------------------------------------------------
if len(sys.argv) > 1:
    input = str(sys.argv[1]).split("-")
    Current_Date = datetime.datetime(int(input[2]), int(input[1]), int(input[0])).replace(minute=0, hour=0, second=0, microsecond=0)
    Previous_Date = datetime.datetime(int(input[2]), int(input[1]), int(input[0])).replace(minute=0, hour=0, second=0, microsecond=0) - datetime.timedelta(days=1)
    Previous_hotstart_Date = datetime.datetime(int(input[2]), int(input[1]), int(input[0])).replace(minute=0, hour=0, second=0, microsecond=0) - datetime.timedelta(days=2)
else:
    Current_Date = datetime.datetime.today().replace(minute=0, hour=0, second=0, microsecond=0)
    Previous_Date = datetime.datetime.today().replace(minute=0, hour=0, second=0, microsecond=0) - datetime.timedelta(days=1)
    Previous_hotstart_Date = datetime.datetime.today().replace(minute=0, hour=0, second=0, microsecond=0) - datetime.timedelta(days=2)

bypass = True
TIMEFORMATCU = "%d_%m_%Y"

# CONFIGURATION: Set mesh resolution here (100, 300, or 900)
MESH_SIZE = 900  # <-- CHANGE THIS for different mesh sizes

name_stem = f'delta11372sqkm_uniform_mesh_{MESH_SIZE}sqm_Chilka{MESH_SIZE}sqm'
partition_name = f'mahanadi_delta_{MESH_SIZE}sqm'  # Must match partition_mahanadi.py output
partition_dir = './partitions'

poly = anuga.read_polygon('polygons/Chilka_Modified_Poly.csv')
poly_bounding = anuga.read_polygon('polygons/Delta_11372_sqkm.csv')
chilka_stage = 0.15
cache = True
verbose = True
intermediate_result_extraction = True
domain_name = 'mahanadi_delta_' + Current_Date.strftime(TIMEFORMATCU)
output_dir = 'output/' + Current_Date.strftime(TIMEFORMATCU)
log.log_filename = output_dir + '/log_' + domain_name
checkpoint_dir = 'checkpoints'
yieldstep = 5
finaltime = 20.0
min_allowed_height = 0.008
max_allowed_speed = 1.0
checkpoint_time = 60 * 60
flow_algorithm = 'DE1'
useCheckpointing = False

# File paths (some used later for hotstart)
friction_filename = 'friction_data/LULC_Apr2020_Norm.asc'
friction_location = "centroids"
#elevation_filename = f'elevation_data/Elev_{name_stem}_LIDAR1m_ALOS30m.csv'
elevation_filename = 'elevation_data/Elev_delta11372sqkm_uniform_mesh_300sqm_Chilka_300sqm_LIDAR1m_ALOS30m.csv'
elevation_location = "vertices"
previous_hotstart_filename = 'output/' + Previous_Date.strftime(TIMEFORMATCU) + '/hotstart_' + Previous_hotstart_Date.strftime(TIMEFORMATCU) + '.sww'
previous_output_filename = 'output/' + Previous_Date.strftime(TIMEFORMATCU) + '/mahanadi_delta_' + Previous_Date.strftime(TIMEFORMATCU) + '.sww'
tobe_hotstart_filename = 'output/' + Current_Date.strftime(TIMEFORMATCU) + '/hotstart_' + Previous_Date.strftime(TIMEFORMATCU) + '.sww'
sww_filename = output_dir + '/' + domain_name + '.sww'
tobe_currentday_filename = 'output/' + Current_Date.strftime(TIMEFORMATCU) + '/current_day_' + Current_Date.strftime(TIMEFORMATCU) + '_' + Current_Date.strftime(TIMEFORMATCU) + '.sww'
tide_filename = 'tide_data/output/paradip_tide_' + Current_Date.strftime(TIMEFORMATCU) + '.tms'

number_of_inlets = 2
coastal_tag_start = 289
coastal_tag_end = 303
line_naraj_barrage = [[374437.043804152, 2263860.85985414], [374665.046154188, 2264351.06490672]]
discharge_filename_naraj_barrage = 'discharge_data/output/naraj_barrage_discharge_' + Current_Date.strftime(TIMEFORMATCU) + '.tms'
line_mahanadi_barrage = [[387363.196978616, 2263420.51264919], [388385.893737998, 2265788.45575452]]
discharge_filename_mahanadi_barrage = 'discharge_data/output/mahanadi_barrage_discharge_' + Current_Date.strftime(TIMEFORMATCU) + '.tms'

imd_rainfall_factor_pt_bhub = 0.00000001157407
imd_rainfall_factor_rgdata_rain25 = 0.00000001157407
imd_rainfall_factor_wrf = 0.00000009259259
imd_rainfall_factor_gfs = 0.00000009259259
gpm_rainfall_factor = 0.00000000925925926
gfs_rainfall_factor = 0.00000009259259259
gauge_filename = 'gauage_locations_data/gauges.csv'
data_path = '/scratch/samir/15-sep/mahanadi-delta_1'
output_frequency = 432000.0
intermediate_outputs = np.arange((yieldstep * output_frequency), finaltime, (yieldstep * output_frequency))
if 86400.0 in intermediate_outputs:
    np.delete(intermediate_outputs, np.where(86400.0))
daily_time = np.arange(86400.0, finaltime, 86400.0)

t1 = time.time()
if myid == 0:
    print('Loading of data: Time', t1 - t0, flush=True)

# --------------------------------------------------------------------------
# Setup procedures
# --------------------------------------------------------------------------
def check_for_dir(release_dir):
    if os.path.isdir(release_dir):
        pass
    else:
        print(("Creating directory " + str(release_dir)))
        os.mkdir(release_dir)
    pass

# --------------------------------------------------------------------------
# LOAD PRE-PARTITIONED DOMAIN
# This replaces the old rank 0 domain creation + distribute() pattern
# --------------------------------------------------------------------------
if myid == 0:
    print('Starting domain loading from partitions', flush=True)
    print(f'Partition name: {partition_name}', flush=True)
    print(f'Partition dir: {partition_dir}', flush=True)

# Setup discharge and tide on rank 0
if myid == 0:
    tsub0 = time.time()
    if len(sys.argv) > 1:
        setup_discharge_3.setup_discharge_old(sys.argv[1])
        setup_tide_3.setup_tide_old(sys.argv[1])
    else:
        setup_discharge_3.setup_discharge()
        setup_tide_3.setup_tide()
    check_for_dir(output_dir)
    tsub1 = time.time()
    print('Setup discharge/tide: Time', tsub1 - tsub0, flush=True)

barrier()

# ALL RANKS load their partition in parallel
t_load_start = time.time()
domain = anuga.sequential_distribute_load(partition_name, partition_dir=partition_dir, verbose=verbose)
t_load = time.time() - t_load_start

if myid == 0:
    print(f'Loaded partitioned domain in {t_load:.2f}s', flush=True)

# Set domain parameters (needed after loading)
domain.set_name(domain_name)
domain.set_datadir(output_dir)
domain.set_multiprocessor_mode(2)

# Handle hotstart stage if previous output exists
# NOTE: This modifies stage AFTER loading partition
if myid == 0:
    tsub0 = time.time()

if os.path.exists(previous_hotstart_filename):
    if myid == 0:
        print(f'Loading hotstart from {previous_hotstart_filename}')
    pc = anuga.plot_utils.get_centroids(previous_hotstart_filename, timeSlices='last')
    domain.set_quantity('stage', numeric=pc.stage[0], location='centroids')
elif os.path.exists(previous_output_filename):
    if myid == 0:
        print(f'Loading previous output from {previous_output_filename}')
    pc = anuga.plot_utils.get_centroids(previous_output_filename)
    domain.set_quantity('stage', numeric=pc.stage[8], location='centroids')
else:
    # Stage already set in partition (elevation + Chilka polygon)
    if myid == 0:
        print('Using initial stage from partition')

if myid == 0:
    tsub1 = time.time()
    print('Stage setting: Time', tsub1 - tsub0, flush=True)

barrier()

t1 = time.time()
if myid == 0:
    print('Domain loading complete: Time', t1 - t0)
    domain.print_statistics()

barrier()

# ------------------------------------------------------------------------------
# Setup boundary conditions
# This must happen *after* domain has been loaded
# ------------------------------------------------------------------------------
Br = anuga.Reflective_boundary(domain)
Bw = anuga.Reflective_boundary(domain)
if os.path.exists(tide_filename):
    wave_function = anuga.file_function(tide_filename, quantities='stage', verbose=verbose)
    Bw = anuga.Time_boundary(domain=domain, function=lambda t: [float(wave_function(t)), 0.0, 0.0])
else:
    if myid == 0 and verbose:
        print("The paradip tide file %s does not exist !!" % tide_filename)

dict1 = {str(n): Bw for n in range(coastal_tag_start, coastal_tag_end)}
dict2 = {'exterior': Br}
dict3 = dict(itertools.chain(list(dict2.items()), list(dict1.items())))
domain.set_boundary(dict3)

barrier()

# Input Discharge with Time Series Naraj Barrage
if os.path.exists(discharge_filename_naraj_barrage):
    Q0 = file_function(discharge_filename_naraj_barrage, quantities='flow')
    inlet0 = Inlet_operator(domain, line_naraj_barrage, Q0, logging=True, description='Naraj Barrage Discharge', verbose=False)
else:
    if myid == 0 and verbose:
        print("The Naraj Barrage Discharge file %s does not exist !!" % discharge_filename_naraj_barrage)

# Input Discharge with Time Series Mahanadi Barrage
if os.path.exists(discharge_filename_mahanadi_barrage):
    Q1 = file_function(discharge_filename_mahanadi_barrage, quantities='flow')
    inlet1 = Inlet_operator(domain, line_mahanadi_barrage, Q1, logging=True, description='Mahanadi Barrage Discharge', verbose=False)
else:
    if myid == 0 and verbose:
        print("The Mahanadi Barrage Discharge file %s does not exist !!" % discharge_filename_mahanadi_barrage)

# Initialize Rainfall parameter
Q = Quantity(domain, name='Rain', register=True)
rain_opertor = anuga.Rate_operator(domain, rate=Q, default_rate=0.0)

# Turn on checkpointing
if useCheckpointing:
    domain.set_checkpointing(checkpoint_time=checkpoint_time, checkpoint_dir=checkpoint_dir)

# ------------------------------------------------------------------------------
# Pre-load rainfall data (avoids slow CSV reading during evolve loop)
# ------------------------------------------------------------------------------
if myid == 0:
    print('Pre-loading rainfall data...', flush=True)

rainfall_loader = RainfallLoader(
    date_str=Current_Date.strftime(TIMEFORMATCU),
    rainfall_factor=gpm_rainfall_factor,  # Will auto-detect source type
    base_dir="rainfall_data",
    verbose=(myid == 0)
)

barrier()

# ------------------------------------------------------------------------------
# Evolution (unchanged from original)
# ------------------------------------------------------------------------------
if myid == 0 and verbose:
    print('EVOLVE')

barrier()
import time

t0 = time.time()
rain_set_zero = True

last_rainfall_timestep = -1  # Track which rainfall timestep we're using
yieldstep_count = 0
STATS_INTERVAL = 360  # Compute volume/stats every N yieldsteps (360 * 5s = 30 min sim time)

for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
    yieldstep_count += 1

    # Check if we need to update rainfall (only when crossing to new rainfall timestep)
    if not rain_set_zero and len(np.where(daily_time == t)[0]) == 1:
        rain_set_zero = True

    if rain_set_zero and rainfall_loader.has_data():
        # Get pre-loaded rainfall data for current time
        rain_data = rainfall_loader.get_rainfall_at_time(t)
        if rain_data is not None:
            # Find which timestep this data is from
            available_ts = rainfall_loader.get_timesteps()
            current_ts = max([ts for ts in available_ts if ts <= t], default=-1)

            # Only update if we've moved to a new rainfall timestep
            if current_ts != last_rainfall_timestep:
                if myid == 0:
                    print(f"Setting rainfall from {rainfall_loader.source_type} t={current_ts}")

                # Get triangle vertices for mapping vertex data to triangles
                triangles = domain.triangles
                # rain_data is per-vertex, we need per-triangle vertex values
                rain_at_vertices = rain_data[triangles]  # Shape: (num_triangles, 3)

                domain.set_quantity('Rain',
                                    numeric=rain_at_vertices,
                                    location='vertices')
                rain_opertor.set_rate(rate=Q)
                last_rainfall_timestep = current_ts
                rain_set_zero = False
        else:
            if myid == 0:
                print("No rainfall data for t=%s, setting to zero" % t)
            domain.set_quantity('Rain', 0.0)
            rain_opertor.set_rate(rate=Q)
    elif rain_set_zero:
        # No pre-loaded data, set to zero
        if myid == 0:
            print("No rainfall data available, setting to zero")
        domain.set_quantity('Rain', 0.0)
        rain_opertor.set_rate(rate=Q)
        rain_set_zero = False

    # Only compute expensive stats every STATS_INTERVAL yieldsteps
    # This reduces MPI reductions significantly
    if yieldstep_count % STATS_INTERVAL == 0 or t == 0:
        if myid == 0: domain.write_time()
        volume = domain.compute_total_volume()
        stats = domain.timestepping_statistics()
        rainstats = rain_opertor.timestepping_statistics()
        maxInundation = Q.get_maximum_value()
        indices = domain.get_wet_elements()
        element_count = len(indices)

        # Write to files
        with open("rain_data.txt", "a+") as f:
            try:
                rain_arr = re.findall(r"\d+\.\d+", rainstats)
                f.write(str(float(rain_arr[0])) + "\n")
            except:
                f.write("0.0\n")

        with open("wet_elements.txt", "a+") as f:
            f.write(str(element_count) + "\n")

        with open("max_inandation.txt", "a+") as f:
            f.write(str(maxInundation) + "\n")

        if myid == 0:
            print("Total Volume =: %s Time Stepping Statistics =: %s Rain Operator Statistics=: %s" % (
                str(volume), str(stats), str(rainstats)))

barrier()

for p in range(numprocs):
    if myid == p:
        print('Processor %g ' % myid)
        print('That took %.2f seconds' % (time.time() - t0))
        print('Communication time %.2f seconds' % domain.communication_time)
        print('Reduction Communication time %.2f seconds' % domain.communication_reduce_time)
        print('Broadcast time %.2f seconds' % domain.communication_broadcast_time)
    else:
        pass
    barrier()

# Merge the individual sww files
#domain.sww_merge(verbose=verbose, delete_old=True)
finalize()
