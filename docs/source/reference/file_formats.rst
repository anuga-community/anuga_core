.. currentmodule:: anuga

File Formats
============

ANUGA reads and writes a number of file formats.  The table below gives a
summary; the sections that follow describe each format in detail.

.. list-table::
   :header-rows: 1
   :widths: 12 12 76

   * - Extension
     - Encoding
     - Description
   * - ``.sww``
     - NetCDF
     - Primary simulation output — mesh geometry plus time-varying quantities
   * - ``.sts``
     - NetCDF
     - Boundary-condition time series at a set of points (no mesh)
   * - ``.tms``
     - NetCDF
     - Scalar time series independent of position
   * - ``.tsh``
     - ASCII
     - Triangular mesh with boundary tags and region attributes
   * - ``.msh``
     - NetCDF
     - Triangular mesh (binary equivalent of TSH)
   * - ``.dem``
     - NetCDF
     - Regular Digital Elevation Model (intermediate ANUGA format)
   * - ``.asc``
     - ASCII
     - ArcView / ArcInfo ASCII grid (regular DEM)
   * - ``.prj``
     - ASCII
     - ArcView projection metadata (accompanies ``.asc``)
   * - ``.ers``
     - ASCII
     - ERMapper header for regular DEM grids
   * - ``.pts``
     - NetCDF
     - Arbitrary point cloud with associated attributes
   * - ``.csv`` / ``.txt``
     - ASCII
     - Comma-separated point data with named attribute columns


SWW — simulation output
------------------------

SWW files are the primary output of ANUGA simulations.  They store the
triangular mesh geometry and the time evolution of all requested quantities
in `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ format.

**Dimensions**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Dimension
     - Meaning
   * - ``number_of_volumes``
     - Number of triangles in the mesh
   * - ``number_of_vertices``
     - Number of nodes (vertices) in the mesh
   * - ``number_of_triangle_vertices``
     - Always 3 (vertices per triangle)
   * - ``number_of_timesteps``
     - Number of stored time steps

**Variables**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Shape
     - Description
   * - ``x``, ``y``
     - ``(number_of_vertices,)``
     - Node coordinates relative to ``xllcorner``, ``yllcorner``
   * - ``volumes``
     - ``(number_of_volumes, 3)``
     - Triangle vertex indices into ``x``/``y``
   * - ``time``
     - ``(number_of_timesteps,)``
     - Simulation time in seconds from ``starttime``
   * - ``elevation``
     - ``(number_of_vertices,)`` or ``(number_of_timesteps, number_of_volumes)``
     - Bed elevation at vertices or (if time-varying) at centroids
   * - ``stage``
     - ``(number_of_timesteps, number_of_volumes)``
     - Water surface elevation at triangle centroids
   * - ``xmomentum``, ``ymomentum``
     - ``(number_of_timesteps, number_of_volumes)``
     - Depth-averaged momentum at centroids (m²/s)
   * - ``elevation_c``, ``stage_c``, ``xmomentum_c``, ``ymomentum_c``
     - ``(number_of_timesteps, number_of_volumes)``
     - Centroid values (present when ``set_store_centroids()`` is called)

**Global attributes** include ``xllcorner``, ``yllcorner`` (UTM origin),
``zone``, ``hemisphere``, ``projection``, ``datum``, ``starttime``,
``timezone``, ``anuga_version``, and ``revision_number``.

SWW files can be inspected with:

- ``anuga_viewer`` — interactive 3-D visualisation (see :ref:`use_anuga_viewer`)
- ``anuga.SWW_plotter`` — Python plotting interface (see :ref:`use_sww_plotter`)
- QGIS with the Mesh layer type (see :ref:`use_qgis`)
- ``ncdump -h file.sww`` — dump header in CDL text format

SWW files are also used as **input** to :func:`File_boundary` and
:func:`file_function` to drive boundary conditions from a previous simulation.

**Controlling SWW output**

.. code-block:: python

   # Choose which quantities to store and at what order
   domain.set_quantities_to_be_stored({
       'elevation':  1,   # store once (static)
       'stage':      2,   # store every output step
       'xmomentum':  2,
       'ymomentum':  2,
   })

   # Store centroid values in addition to vertex values
   domain.set_store_centroids(True)

   # Suppress all SWW output
   domain.set_quantities_to_be_stored(None)

   # Merge per-rank parallel SWW files into one (all timesteps in RAM)
   domain.sww_merge(delete_old=True)

   # Memory-bounded merge: process 100 timesteps at a time
   domain.sww_merge(delete_old=True, chunk_size=100)

**Post-processing conversions**

.. code-block:: python

   import anuga

   # Export a quantity from an SWW file to a gridded ASC/ERS raster
   anuga.sww2dem('simulation.sww', basename_out='depth',
                 quantity='depth', cellsize=10, format='asc')

   # Read all centroid data into NumPy arrays
   data = anuga.sww2array('simulation.sww')


STS — boundary time series
---------------------------

STS files store time-varying data at a set of scattered points without
mesh connectivity.  They are used as input to :func:`create_sts_boundary`
to drive open-ocean boundary conditions, for example from tsunami source
models.

**Variables**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variable
     - Description
   * - ``x``, ``y``
     - Point coordinates
   * - ``elevation``
     - Bed elevation at each point
   * - ``time``
     - Time array
   * - ``stage``, ``xmomentum``, ``ymomentum``
     - Time-varying quantities at each point
   * - ``permutation``
     - Original point ordering (used when an ordering file is supplied to
       ``urs2sts()``)

STS files can be created from URS tsunami source output:

.. code-block:: python

   anuga.urs2sts('source_prefix', basename_out='boundary')

And used as a boundary condition:

.. code-block:: python

   from anuga import create_sts_boundary
   Bsts = create_sts_boundary('boundary.sts')
   domain.set_boundary({'ocean': Bsts})


TMS — scalar time series
-------------------------

TMS files store a single scalar quantity as a function of time only,
independent of spatial location.  They are used internally by ANUGA for
recording gauge output and time-varying forcing terms.  The format is
NetCDF with ``time`` and one or more data variables.


Mesh file formats — TSH and MSH
---------------------------------

Mesh files describe the triangular mesh and its boundary structure.
They can be created by :func:`create_mesh_from_regions` or by the
built-in mesh generator.

**TSH** (ASCII) is the human-readable format.  It encodes:

- Mesh outline vertices and enclosing line segments
- Triangle vertices and their indices
- Boundary segment tags (used by ``set_boundary()``)
- Holes (regions excluded from the mesh)
- Named interior regions with associated attributes (e.g. friction)
- An optional geo-reference (UTM origin and zone)

**MSH** is a NetCDF binary encoding of the same information.

Both formats are accepted by :func:`create_domain_from_file`:

.. code-block:: python

   domain = anuga.create_domain_from_file('mymesh.tsh')
   domain = anuga.create_domain_from_file('mymesh.msh')

To create a TSH or MSH file from polygon regions:

.. code-block:: python

   anuga.create_mesh_from_regions(
       bounding_polygon,
       boundary_tags={'left': [0], 'right': [1], 'top': [2], 'bottom': [3]},
       maximum_triangle_area=100.0,
       filename='mymesh.msh',
   )


DEM — Digital Elevation Model formats
---------------------------------------

ANUGA uses several DEM formats in a conversion pipeline:

.. code-block:: text

   ASC / ERS  →  DEM (NetCDF)  →  PTS (NetCDF)
                                      ↓
                              fit onto mesh → TSH with elevation

**ASC** (ArcView ASCII grid) header format:

.. code-block:: text

   ncols         753
   nrows         766
   xllcorner     314036.587
   yllcorner     6224951.296
   cellsize      100
   NODATA_value  -9999
   <elevation data rows>

An accompanying **PRJ** file provides projection metadata.

**ERS** is the ERMapper header format used for the same purpose.

**DEM** is ANUGA's internal NetCDF representation of regular grid elevation
data, produced by ``asc2dem``.

Convert ASC to DEM, then DEM to a point cloud:

.. code-block:: python

   import anuga

   # Convert ArcView ASCII grid to ANUGA's NetCDF DEM format
   anuga.asc2dem('elevation.asc', use_cache=False, verbose=True)

   # Convert DEM to a point cloud (PTS) for fitting onto a mesh
   anuga.dem2pts('elevation.dem', use_cache=False, verbose=True)


Point data — CSV, TXT and PTS
-------------------------------

**CSV / TXT** files store point data with named columns.  The first two
columns must be ``x`` and ``y``.  Additional columns are arbitrary named
attributes.

Example:

.. code-block:: text

   x, y, elevation, friction
   0.6, 0.7, 4.9, 0.3
   1.9, 2.8, 5.0, 0.3
   2.7, 2.4, 5.2, 0.3

**PTS** files are the NetCDF binary equivalent: an ``(N, 2)`` float array
for coordinates and an ``(N,)`` float array per attribute.  They are
generated by ``dem2pts`` and consumed by the least-squares fitting
routines that assign elevation to a mesh.

Reading CSV data into a domain:

.. code-block:: python

   domain.set_quantity('elevation',
                       filename='elevation_points.csv',
                       use_cache=False,
                       verbose=True)

.. seealso::

   `ANUGA User Manual — Chapter 15: ANUGA File Formats
   <https://github.com/anuga-community/anuga_user_manual>`_
   describes all file formats in greater detail, including the full NetCDF
   variable listings for SWW and STS files, the TSH ASCII format
   specification, and a guide to the file conversion utilities.
