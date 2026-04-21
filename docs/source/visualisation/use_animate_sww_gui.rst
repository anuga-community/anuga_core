.. _animate_sww_gui:

SWW Animation GUI (``anuga_animate_sww_gui``)
=============================================

``anuga_animate_sww_gui`` is an interactive desktop application for
visualising and exploring the output of ANUGA simulations stored in SWW
files.  It generates PNG frames from a chosen quantity, plays them as an
animation, and supports interactive timeseries extraction at any mesh
point.

.. contents:: Contents
   :local:
   :depth: 2


Starting the GUI
----------------

From the command line::

    anuga_animate_sww_gui

Pass an SWW file and/or an initial quantity to open directly::

    anuga_animate_sww_gui --sww results/towradgi.sww --qty depth

Available quantities for ``--qty`` are:
``depth``, ``stage``, ``speed``, ``speed_depth``,
``max_depth``, ``max_speed``, ``max_speed_depth``.


Quick-start workflow
--------------------

1. **Open an SWW file** — click *Browse…* next to *SWW file*, or type
   the path directly and press Enter.  The info bar shows the number of
   triangles and timesteps.

2. **Configure generation settings** — choose a *Quantity*, set
   *vmin* / *vmax* (or tick *Auto from data*), adjust *DPI* and other
   options as needed.

3. **Click Generate Frames** — PNG files are written to the *Output dir*
   (default ``_plot``).  A progress bar tracks completion.  Click
   *Cancel* to abort early.

4. **Animate** — use the *Play* button and frame controls to step through
   or continuously play the animation.


Generation settings
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Setting
     - Description
   * - **Quantity**
     - The variable to plot.  Animated quantities (``depth``,
       ``stage``, ``speed``, ``speed_depth``) produce one PNG per
       selected timestep.  Maximum quantities (``max_depth``,
       ``max_speed``, ``max_speed_depth``) produce a single frame
       showing the spatial maximum over the entire simulation.
   * - **vmin / vmax**
     - Colormap range.  Tick *Auto from data* to set automatically
       from the full data range.
   * - **DPI**
     - Resolution of generated PNG frames.  Higher values give sharper
       images but take longer to generate and more disk space.
   * - **min depth**
     - Triangles with depth below this threshold are treated as dry and
       rendered in grey using elevation shading.
   * - **Every N frames**
     - Stride: generate one PNG every *N* SWW timesteps.  Use a larger
       value for a quick preview of a long simulation.
   * - **Colormap / Reverse**
     - Any matplotlib colormap name.  Tick *Reverse* to invert it.
   * - **Basemap / provider / Alpha**
     - Overlay an online tile basemap (OpenStreetMap, Esri Satellite,
       etc.) behind the mesh.  Requires the SWW file to carry an EPSG
       code and an active internet connection.  *Alpha* controls the
       transparency of the mesh colour overlay.


Playback controls
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Control
     - Description
   * - **Play / Pause**
     - Start or stop the animation.
   * - **|< < > >|**
     - Jump to first frame, step back one frame, step forward, jump to
       last frame.
   * - **FPS**
     - Playback speed in frames per second.
   * - **Frame slider**
     - Drag to scrub through frames.


Pick timeseries
---------------

Click **Pick timeseries** to enter interactive pick mode:

* The cursor changes to a crosshair and an instruction banner appears
  at the bottom of the image.
* **Click** anywhere on the image to select the nearest triangle
  centroid.  The timeseries panel opens below the animation showing the
  selected quantity vs. time.  A red star marks the picked location on
  every animation frame.
* **Click again** to pick a different location — the timeseries and
  marker update immediately without leaving pick mode.
* **Escape** or click **Cancel pick** to exit pick mode.

The timeseries panel
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Control
     - Description
   * - **Quantity dropdown**
     - Switch between ``depth``, ``stage``, ``speed``,
       ``speed_depth`` for the picked triangle without re-picking.
   * - **Red dashed line**
     - Tracks the current animation frame time.  Moves as you step
       through frames.
   * - **Export CSV**
     - Opens a save-file dialog and writes the displayed time series
       to a CSV file.  The file includes a header comment with the
       triangle index and centroid coordinates (easting, northing),
       followed by ``time_s`` and the quantity column.
   * - **Close**
     - Hide the timeseries panel.  Pick state is reset.


Saving frames and animations
----------------------------

Save Frame
~~~~~~~~~~

The **Save Frame** button (playback row, right side) saves the currently
displayed frame — including any pick-marker overlay — to a file.
Supported formats: **PNG**, **PDF**, **SVG**.  PNG is the default.

Save Animation
~~~~~~~~~~~~~~

The **Save Animation…** button (next to *Generate Frames*) assembles all
loaded frames into a single animation file.  The playback **FPS** setting
is used as the frame rate.

Two output formats are supported:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Format
     - Notes
   * - **MP4**
     - Requires ``ffmpeg`` to be installed and on ``PATH``.  Produces
       compact, high-quality video using the H.264 codec
       (``libx264``, ``yuv420p`` pixel format).  Offered as the default
       when ``ffmpeg`` is detected.  Install via conda
       (``conda install ffmpeg``), apt, brew, or from
       https://ffmpeg.org.
   * - **GIF**
     - Requires `Pillow <https://pillow.readthedocs.io>`_
       (``pip install Pillow``).  Works everywhere with no codec
       required.  File size can be large for long or high-DPI
       animations.

If ``ffmpeg`` is not found, only GIF is offered.  If Pillow is missing, an
error dialog is shown when GIF is selected.


Maximum-envelope quantities
----------------------------

Selecting ``max_depth``, ``max_speed``, or ``max_speed_depth`` as the
*Quantity* generates a **single static frame** showing the maximum
value at each triangle across all timesteps.  This is useful for:

* Mapping **inundation extent** (maximum depth > 0).
* Identifying **peak hazard zones** (maximum speed or speed×depth).
* Producing figures for reports without needing to review every
  timestep.

The *Every N frames* stride setting is ignored for maximum quantities.
All other settings (colormap, vmin/vmax, basemap, DPI) apply as normal.
The *Pick timeseries* tool remains available — clicking a triangle
while viewing a maximum frame still shows the full time series for that
location.


Output files
------------

Frames are saved as::

    <output_dir>/<sww_prefix>_<quantity>_<frame_number>.png

For example::

    _plot/towradgi_depth_0000000000.png
    _plot/towradgi_depth_0000000001.png
    ...
    _plot/towradgi_max_depth_0000000000.png

Exported CSV files follow the naming convention::

    <sww_prefix>_<quantity>_tri<triangle_index>.csv


See also
--------

* :ref:`use_sww_plotter` — programmatic access to the same plotting
  functions used by this GUI.
* :ref:`use_domain_plotter` — live plotting during a simulation run.
