.. _toml_scenario:

=====================================
Running Scenarios with anuga_run_toml
=====================================

.. currentmodule:: anuga

The ``anuga_run_toml`` script provides a ready-made runner for ANUGA flood and
tsunami scenarios.  All simulation inputs are described in a single
`TOML <https://toml.io>`_ configuration file — no Python coding required for
standard setups.  The same script also accepts legacy Excel (``.xlsx``) files
from the older ``cairns_excel`` interface.

.. contents:: Contents
   :local:
   :depth: 2


Quick Start
-----------

**Serial run:**

.. code-block:: bash

   anuga_run_toml  path/to/scenario.toml

**Parallel run (MPI):**

.. code-block:: bash

   mpirun -np 6  anuga_run_toml  path/to/scenario.toml

All relative paths inside the TOML file are resolved relative to the directory
that contains the TOML file, so the script can be invoked from any working
directory.


Working example — Cairns tsunami scenario
------------------------------------------

A complete, ready-to-run example is provided in
``examples/cairns_toml_excel/`` of the repository.  It models a synthetic
tsunami entering Cairns Harbour (Queensland, Australia) and exercises most
TOML features: a shapefile boundary, a DEM elevation raster, Flather open-
ocean boundaries, and stubs for bridges, pumping stations, culverts, and weirs.

.. code-block:: bash

   cd examples/cairns_toml_excel
   anuga_run_toml cairns_example.toml          # serial
   mpirun -np 4 anuga_run_toml cairns_example.toml   # parallel


Minimal annotated TOML
-----------------------

The snippet below is the smallest valid TOML that will run a real simulation.
Copy it, adjust the file paths, and add extra sections as needed.

.. code-block:: toml

   # ── Project ───────────────────────────────────────────────────────────────
   [project]
   scenario                  = "my_run"      # name for the SWW file & output dir
   output_base_directory     = "OUTPUT/"
   yieldstep                 = 120.0         # seconds between evolve yields
   finaltime                 = 21600.0       # total simulation duration [s]
   projection_information    = -55           # UTM zone (negative = southern hemisphere)
   flow_algorithm            = "DE0"         # "DE0" (fast) or "DE1" (accurate)
   output_tif_cellsize       = 50.0          # cell size [m] for output GeoTiff rasters

   # ── Mesh ──────────────────────────────────────────────────────────────────
   [mesh]
   bounding_polygon          = "mesh/boundary.shp"   # shapefile with boundary tags
   default_res               = 1000000.0             # default triangle area [m²]
   # [[mesh.interior_regions]]
   # polygon    = "mesh/fine_zone.csv"
   # resolution = 10000.0   # finer triangles inside this polygon [m²]

   # ── Boundary conditions ───────────────────────────────────────────────────
   [boundary_conditions]
   boundary_tags_attribute_name = "Boundary"   # shapefile attribute holding tag names

   [[boundary_conditions.boundaries]]
   tag        = "Ocean"
   type       = "Flather_Stage"                # weakly-reflecting open boundary
   file       = "boundarycond/tide.csv"        # CSV: columns time_s, stage_m
   start_time = 0.0

   [[boundary_conditions.boundaries]]
   tag  = "Land"
   type = "Reflective"                         # solid wall — no file needed

   # ── Initial conditions ────────────────────────────────────────────────────
   [initial_conditions]

   [[initial_conditions.elevation]]
   polygon  = "Extent"                         # full extent of the raster
   value    = "initialcond/dem.asc"            # GDAL raster (asc, tif, …)
   clip_min = -inf
   clip_max = inf

   [[initial_conditions.elevation]]
   polygon  = "All"                            # catch-all fallback
   value    = 0.0
   clip_min = -inf
   clip_max = inf

   [[initial_conditions.friction]]
   polygon  = "All"
   value    = 0.03                             # Manning's n
   clip_min = 0.01
   clip_max = 0.5

   [[initial_conditions.stage]]
   polygon  = "All"
   value    = 0.0                              # start domain dry
   clip_min = -inf
   clip_max = inf

The full reference for every key is in the `Configuration File Reference`_
section below.


Scenario Directory Layout
--------------------------

A typical scenario directory looks like this::

   my_scenario/
   ├── scenario.toml          ← main configuration file
   ├── user_functions.py      ← optional custom callbacks (see below)
   ├── mesh/
   │   ├── boundary.shp       ← domain boundary shapefile (or .csv)
   │   └── ...
   ├── boundarycond/
   │   ├── tide.csv
   │   └── ...
   └── initialcond/
       └── dem.asc

On completion, outputs are written to a timestamped subdirectory under
``output_base_directory`` (e.g. ``OUTPUT/RUN_20260315_120000_scenario/``).
A copy of all code files — including the TOML config and the runner script
itself — is archived there for reproducibility.


Configuration File Reference
-----------------------------

The TOML file is divided into sections (TOML *tables*).  Every key shown
without a comment is **required**; keys shown with a ``# default:`` comment
are optional.

.. _toml-project:

[project]
~~~~~~~~~

Top-level simulation settings.

.. code-block:: toml

   [project]

   # Name used for the output SWW file and the output directory prefix
   scenario = "my_scenario"

   # Base directory where per-run output subdirectories are created
   output_base_directory = "OUTPUT/"

   # Time between successive yields of the evolve loop [seconds].
   # Controls how often timestepping statistics are printed.
   yieldstep = 120.0

   # Total simulation duration [seconds]
   finaltime = 21600.0

   # SWW output interval [seconds].
   # Must be an integer multiple of yieldstep.
   # Omit to write SWW output at every yieldstep (the default).
   # outputstep = 600.0

   # Coordinate reference system.
   #   Integer: UTM zone (negative = southern hemisphere, e.g. -55)
   #   String:  full proj4 string for non-UTM projections
   projection_information = -55

   # Numerical flow algorithm.
   #   "DE0" — first-order (faster)
   #   "DE1" — second-order (more accurate)
   flow_algorithm = "DE0"

   # Enable local extrapolation and flux updating (improves stability
   # near wet/dry fronts)
   # default: false
   use_local_extrapolation_and_flux_updating = false

   # Grid cell size [m] for output GeoTiff rasters
   output_tif_cellsize = 50.0

   # Polygon CSV to clip output GeoTiffs.
   # Leave empty to use the full domain extent.
   output_tif_bounding_polygon = ""

   # How often (every N timesteps) to update tracked maximum quantities
   # default: 1
   max_quantity_update_frequency = 1

   # Time from which maximum-quantity tracking begins [seconds]
   # Must be < finaltime.  default: 0.0
   max_quantity_collection_starttime = 0.0

   # Store each triangle's three vertex values separately.
   # Produces larger files; needed for some visualisation workflows.
   # default: false
   store_vertices_uniquely = false

   # Re-write elevation at every yieldstep.
   # Only needed if elevation changes during the run (e.g. erosion operators).
   # default: false
   store_elevation_every_timestep = false

   # Directory for text-format copies of spatial inputs (used for QC)
   # default: "SPATIAL_TEXT"
   spatial_text_output_dir = "SPATIAL_TEXT"

   # --- Per-yieldstep diagnostics ---
   # default: false for all
   report_mass_conservation_statistics      = false
   report_peak_velocity_statistics          = false   # requires user_functions.py
   report_smallest_edge_timestep_statistics = false
   report_operator_statistics               = false   # requires user_functions.py

   # --- Solver threading ---
   # Number of OpenMP threads.  Omit to read OMP_NUM_THREADS env var (default 1).
   # omp_num_threads = 4

   # Multiprocessor mode: 1 = OpenMP CPU (default), 2 = OpenMP GPU offload (experimental, branch sp26)
   multiprocessor_mode = 1

.. _toml-mesh:

[mesh]
~~~~~~

Mesh geometry and resolution.

.. code-block:: toml

   [mesh]

   # Set true to reuse a previously built partitioned mesh (skips mesh generation)
   # default: false
   use_existing_mesh_pickle = false

   # Domain boundary file — two formats are supported:
   #
   #   Shapefile (.shp): each line feature must carry a tag attribute whose name
   #     is given by boundary_conditions.boundary_tags_attribute_name.
   #
   #   CSV (.csv): a plain x,y polygon.  You must also provide
   #     [[mesh.boundary_tags]] entries (see below).
   bounding_polygon = "mesh/boundary.shp"

   # Default maximum triangle area [m²] across the whole domain
   default_res = 1000000.0

   # Breakline files — polylines that force triangle edges.
   # Glob patterns are accepted (e.g. "mesh/breaklines_*.csv").
   # default: []
   breakline_files = []

   # Riverwall CSV files — like breaklines but with an elevation column.
   # default: []
   riverwall_csv_files = []

   # Snapping distance [m] for breakline / bounding-polygon intersections.
   # Set to "ignore" to skip intersection processing.
   # default: 0.1
   breakline_intersection_threshold = 0.1

   # Point-based resolution file — CSV with columns x, y, resolution.
   # Mutually exclusive with [[mesh.interior_regions]].
   region_areas_file = ""

   # Interpretation of the resolution column in region_areas_file.
   #   "area"   — maximum triangle area [m²]
   #   "length" — approximate maximum edge length [m]
   # default: "area"
   region_areas_type = "area"

**Interior regions** (polygon-based resolution refinement):

.. code-block:: toml

   [[mesh.interior_regions]]
   polygon    = "mesh/fine_region.csv"
   resolution = 10000.0   # maximum triangle area [m²]

Interior regions and ``region_areas_file`` / breaklines are mutually exclusive.

**Explicit boundary tags** (required when ``bounding_polygon`` is a CSV):

.. code-block:: toml

   [[mesh.boundary_tags]]
   tag   = "ocean"
   edges = [0, 1]

   [[mesh.boundary_tags]]
   tag   = "land"
   edges = [2, 3]

Edge indices are 0-based and refer to the segments between consecutive polygon
vertices.

.. _toml-boundary:

[boundary_conditions]
~~~~~~~~~~~~~~~~~~~~~

One ``[[boundary_conditions.boundaries]]`` entry is required for every tag
that appears in the bounding polygon.

.. code-block:: toml

   [boundary_conditions]

   # Shapefile attribute name that carries the boundary tag.
   # Must match the attribute used in the boundary shapefile.
   boundary_tags_attribute_name = "Boundary"

   # Reflective (solid wall) boundary — no file needed
   [[boundary_conditions.boundaries]]
   tag  = "Land"
   type = "Reflective"

   # Stage boundary — transmissive with prescribed stage timeseries
   [[boundary_conditions.boundaries]]
   tag        = "Ocean"
   type       = "Stage"
   file       = "boundarycond/tide.csv"   # columns: time_s, stage_m
   start_time = 0.0                       # subtract from CSV times [s]

   # Flather radiation boundary with prescribed stage timeseries
   [[boundary_conditions.boundaries]]
   tag        = "Open"
   type       = "Flather_Stage"
   file       = "boundarycond/tide.csv"
   start_time = 0.0

.. _toml-initial:

[initial_conditions]
~~~~~~~~~~~~~~~~~~~~~

Each quantity is built from an ordered list of ``[[initial_conditions.<qty>]]``
entries.  Earlier entries take priority over later ones.

Supported quantities: ``elevation``, ``friction``, ``stage``, ``xmomentum``,
``ymomentum``.

.. code-block:: toml

   [initial_conditions]

   # Optional: spatially average elevation on a grid at this spacing [m]
   # elevation_spatial_average = 100.0

   [[initial_conditions.elevation]]
   polygon  = "Extent"                    # apply over full raster extent
   value    = "initialcond/dem.asc"       # GDAL-compatible raster
   clip_min = -inf
   clip_max = inf

   [[initial_conditions.elevation]]
   polygon  = "All"                       # catch-all fallback
   value    = 0.0
   clip_min = -inf
   clip_max = inf

   [[initial_conditions.friction]]
   polygon  = "mesh/mannings_zone.csv"    # apply inside polygon
   value    = 0.025
   clip_min = 0.01
   clip_max = 0.5

   [[initial_conditions.friction]]
   polygon  = "All"
   value    = 0.03
   clip_min = 0.01
   clip_max = 0.5

**Polygon options:**

===============  ==============================================================
``"Extent"``     Apply over the full spatial extent of the raster ``value``
``"All"``        Apply everywhere (use as the final catch-all entry)
``"None"``       Skip this entry
path to file     ``.shp`` or ``.csv`` polygon
===============  ==============================================================

**Value options:**

====================  =========================================================
constant number       applied uniformly across the polygon
path to raster        GDAL-compatible raster (``.asc``, ``.tif``, …)
path to CSV/TXT       file with columns ``x``, ``y``, ``z``
====================  =========================================================

.. _toml-additions:

[initial_condition_additions]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same structure as ``[initial_conditions]``, but values are *added on top of*
whatever was set there.  Useful for local offsets such as a pre-dredged
channel, a localised initial flood pool, or a small friction adjustment.

.. code-block:: toml

   [initial_condition_additions]

   [[initial_condition_additions.elevation]]
   polygon  = "mesh/dredge_channel.csv"
   value    = -2.0
   clip_min = -inf
   clip_max = inf

.. _toml-inlets:

[[inlets]]
~~~~~~~~~~

Line sources / sinks of discharge across a cross-section.

.. code-block:: toml

   [[inlets]]
   name            = "river_inflow"
   line_file       = "mesh/river_line.csv"       # cross-section polyline
   timeseries_file = "boundarycond/river_flow.csv"  # columns: time_s, discharge_m3s
   start_time      = 0.0

Positive discharge adds water; negative removes it.  Multiple ``[[inlets]]``
entries are supported.

.. _toml-rainfall:

[[rainfall]]
~~~~~~~~~~~~

Spatially uniform rainfall rate over a region.

.. code-block:: toml

   [[rainfall]]
   timeseries_file    = "boundarycond/rain.csv"  # columns: time_s, rate_mm_per_hr
   start_time         = 0.0
   interpolation_type = "linear"                 # "linear" or "nearest"
   polygon            = "All"                    # path to CSV polygon, or "All"
   multiplier         = 1.0                      # scale factor (default 1.0)

Multiple ``[[rainfall]]`` entries are supported.

.. _toml-bridges:

[[bridges]]
~~~~~~~~~~~

HEC-RAS rating-curve internal boundary operator.  Set ``enabled = false`` to
disable a bridge without removing its definition.

.. code-block:: toml

   [[bridges]]
   enabled                      = true
   label                        = "bridge_1"
   deck_file                    = "mesh/bridge1_deck.csv"
   deck_elevation               = 4.5         # [m] bed elevation forced inside deck polygon
   exchange_line_0              = "mesh/bridge1_exchange_up.csv"
   exchange_line_1              = "mesh/bridge1_exchange_down.csv"
   enquiry_gap                  = 2.0         # [m] from exchange lines to enquiry points
   internal_boundary_curve_file = "mesh/bridge1_rating_curve.csv"
   vertical_datum_offset        = 0.0         # [m] added to curve elevations
   smoothing_timescale          = 20.0        # [s] exponential smoothing constant

.. _toml-pumping:

[[pumping_stations]]
~~~~~~~~~~~~~~~~~~~~~

Pump operator transferring water between a wet-well basin and a discharge
point.  Set ``enabled = false`` to disable without removing the definition.

.. code-block:: toml

   [[pumping_stations]]
   enabled               = true
   label                 = "pump_1"
   pump_capacity         = 2.0     # [m³/s] maximum flow rate
   pump_rate_of_increase = 0.5     # [m³/s per second]
   pump_rate_of_decrease = 0.5     # [m³/s per second]
   hw_to_start_pumping   = 0.3     # headwater depth [m] to switch on
   hw_to_stop_pumping    = 0.1     # headwater depth [m] to switch off
   basin_polygon_file    = "mesh/pump1_basin.csv"
   basin_elevation       = -1.0    # [m] bed elevation inside basin polygon
   exchange_line_0       = "mesh/pump1_wet_well_line.csv"
   exchange_line_1       = "mesh/pump1_discharge_line.csv"
   smoothing_timescale   = 30.0    # [s]


.. _toml-culverts:

[[culverts]]
~~~~~~~~~~~~

Box or pipe culverts using the Boyd (1987) head-discharge algorithm.
Set ``enabled = false`` to disable without removing the definition.

``type`` selects the cross-section shape:

* ``"boyd_box"``  — rectangular barrel (``width`` × ``height``).
* ``"boyd_pipe"`` — circular barrel (``diameter``).

**Geometry** — choose exactly one of:

* ``exchange_line_0`` / ``exchange_line_1``: paths to CSV polyline files
  that define the upstream and downstream exchange zones.  The exchange
  zone widths govern how much of the mesh perimeter couples to the culvert.
  This is the preferred approach for field models.

* ``end_point_0`` / ``end_point_1``: ``[x, y]`` coordinate pairs for the
  two barrel ends.  ANUGA derives short perpendicular exchange lines
  automatically.

**Boyd box example:**

.. code-block:: toml

   [[culverts]]
   enabled              = true
   type                 = "boyd_box"
   label                = "road_culvert_1"
   width                = 0.9              # [m] internal barrel width
   height               = 0.6             # [m] internal barrel height (omit = square)
   exchange_line_0      = "mesh/culvert1_exchange_up.csv"
   exchange_line_1      = "mesh/culvert1_exchange_down.csv"
   enquiry_gap          = 0.2             # [m] from exchange line to enquiry point
   losses               = 0.5             # head-loss coefficient (0.5 = sharp-edge inlet)
   barrels              = 1.0             # number of parallel identical barrels
   blockage             = 0.0             # fractional blockage [0,1]; 0 = fully open
   z1                   = 0.0             # batter slope at end 0 (rise/run)
   z2                   = 0.0             # batter slope at end 1
   apron                = 0.1             # [m] flat apron at each end
   manning              = 0.013           # barrel Manning's n
   smoothing_timescale  = 0.0             # [s] exponential smoothing constant
   use_momentum_jet     = true
   use_velocity_head    = true
   # invert_elevations  = [1.0, 0.8]      # [m] upstream, downstream; sampled from DEM if omitted

**Boyd pipe example:**

.. code-block:: toml

   [[culverts]]
   enabled              = true
   type                 = "boyd_pipe"
   label                = "drain_pipe_1"
   diameter             = 0.6              # [m] internal barrel diameter
   exchange_line_0      = "mesh/pipe1_exchange_up.csv"
   exchange_line_1      = "mesh/pipe1_exchange_down.csv"
   enquiry_gap          = 0.2
   losses               = 0.5
   barrels              = 2.0              # two parallel pipes
   manning              = 0.013
   smoothing_timescale  = 0.0

**Using end points instead of exchange lines:**

.. code-block:: toml

   [[culverts]]
   type        = "boyd_box"
   label       = "simple_culvert"
   width       = 0.9
   end_point_0 = [355420.0, 8132050.0]    # [x, y] upstream barrel end [m]
   end_point_1 = [355435.0, 8132050.0]    # [x, y] downstream barrel end [m]
   losses      = 0.5

Multiple ``[[culverts]]`` entries are supported.


.. _toml-weirs:

[[weirs]]
~~~~~~~~~

Weir / orifice structures with a trapezoidal cross-section, using combined
weir and orifice flow formulae (``Weir_orifice_trapezoid_operator``).
Set ``enabled = false`` to disable without removing the definition.

Geometry is specified with the same ``exchange_line_0`` / ``exchange_line_1``
(file paths) or ``end_point_0`` / ``end_point_1`` (coordinates) choice as for
``[[culverts]]``.

.. code-block:: toml

   [[weirs]]
   enabled              = true
   label                = "outlet_weir"
   width                = 3.0             # [m] bottom width of trapezoidal section
   height               = 1.2             # [m] section height (omit = width)
   exchange_line_0      = "mesh/weir1_exchange_up.csv"
   exchange_line_1      = "mesh/weir1_exchange_down.csv"
   enquiry_gap          = 0.0             # [m]; 0 is typical for weirs
   losses               = 0.5             # head-loss coefficient
   barrels              = 1.0
   blockage             = 0.0
   z1                   = 0.0
   z2                   = 0.0
   apron                = 0.1
   manning              = 0.013
   smoothing_timescale  = 0.0
   use_momentum_jet     = true
   use_velocity_head    = true
   # invert_elevations  = [1.0, 0.8]      # [m]; sampled from DEM if omitted

Multiple ``[[weirs]]`` entries are supported.


Custom Callbacks: user_functions.py
-------------------------------------

Place a file called ``user_functions.py`` alongside the TOML file to add
custom per-timestep actions.  The runner imports it automatically; if the file
is absent it is silently skipped.

Two hooks are called when the corresponding TOML flags are ``true``:

``print_velocity_statistics(domain, max_quantities)``
   Called when ``report_peak_velocity_statistics = true``.
   Receives the live domain and the
   :class:`~anuga.operators.collect_max_quantities_operator.Collect_max_quantities_operator`
   instance.

``print_operator_inputs(domain)``
   Called when ``report_operator_statistics = true``.
   Receives the live domain; useful for printing rainfall rates, inlet
   discharges, etc.

A minimal example:

.. code-block:: python

   # user_functions.py
   from anuga.parallel import myid, barrier

   def print_velocity_statistics(domain, max_quantities):
       if myid == 0:
           import numpy as np
           xx = domain.quantities['xmomentum'].centroid_values
           yy = domain.quantities['ymomentum'].centroid_values
           dd = domain.quantities['stage'].centroid_values \
               - domain.quantities['elevation'].centroid_values
           dd = np.maximum(dd, 1.0e-3)
           speed = np.sqrt(xx**2 + yy**2) / dd
           print(f'  Peak speed: {speed.max():.3f} m/s')

   def print_operator_inputs(domain):
       if myid == 0:
           for op in domain.fractional_step_operators:
               if hasattr(op, 'applied_Q'):
                   print(f'  {op.label}: Q = {op.applied_Q:.3f} m³/s')


Output Directory Structure
---------------------------

Each run creates a timestamped directory under ``output_base_directory``::

   OUTPUT/
   └── RUN_20260315_120000_scenario/
       ├── scenario.sww           ← NetCDF output (stage, momentum, elevation)
       ├── Simulation_logfile.log ← full stdout log
       ├── Max_quantities_*.csv   ← peak stage, depth, speed per triangle
       ├── *.tif                  ← GeoTiff rasters of peak quantities
       ├── code/                  ← archived copy of all input files
       │   ├── scenario.toml
       │   ├── anuga_run_toml     ← copy of the runner script
       │   └── user_functions.py
       └── SPATIAL_TEXT/          ← text copies of spatial inputs (for QC)


Excel Compatibility
--------------------

``anuga_run_toml`` also accepts legacy Excel files::

   anuga_run_toml  path/to/ANUGA_setup.xlsx

The Excel format is described in the ``cairns_toml_excel`` example directory.
Attributes that exist only in the TOML interface
(``multiprocessor_mode``, ``omp_num_threads``, ``outputstep``,
``report_operator_statistics``) are set to sensible defaults when reading
Excel files.
