.. _use_domain_plotter:

.. currentmodule:: anuga

Domain Plotter
==============

``Domain_plotter`` provides matplotlib-based plotting of domain quantities
**during a simulation**, making it suitable for interactive use in a Jupyter
notebook or for saving a sequence of frame images alongside the evolve loop.

For post-processing a completed simulation from a saved SWW file, use
:ref:`use_sww_plotter` instead.

.. list-table:: Domain_plotter vs SWW_plotter
   :header-rows: 1
   :widths: 40 30 30

   * - Feature
     - ``Domain_plotter``
     - ``SWW_plotter``
   * - Data source
     - Live domain centroid values
     - SWW file (NetCDF)
   * - When to use
     - During or just after evolve loop
     - Post-processing
   * - Seeks to frame
     - Current time only
     - Any stored time step
   * - Typical context
     - Jupyter notebook
     - Script or notebook


Setup
-----

Create a ``Domain_plotter`` from any ``Domain`` object, or call
``domain.set_plotter()`` which creates one and attaches its methods
directly to the domain for convenience.

**Direct construction**

.. code-block:: python

   import anuga

   domain = anuga.rectangular_cross_domain(40, 40, len1=10.0, len2=10.0)
   # ... set quantities, boundaries ...

   dplotter = anuga.Domain_plotter(domain, plot_dir='_plot', min_depth=0.01)

**Via ``set_plotter``** (attaches methods to the domain object)

.. code-block:: python

   domain.set_plotter(plot_dir='_plot', min_depth=0.01)
   # Now domain.plot_depth_frame(), domain.save_depth_frame(), etc. are available

.. list-table:: Constructor arguments
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``domain``
     - —
     - The ``Domain`` instance to attach to
   * - ``plot_dir``
     - ``'_plot'``
     - Directory for saved frame images.  Created automatically if it does
       not exist.  Pass ``None`` to save frames in the current directory.
   * - ``min_depth``
     - ``0.01``
     - Water depth threshold (m) below which cells are treated as dry and
       shown using the elevation colour map instead of the quantity colour map.
   * - ``absolute``
     - ``False``
     - If ``True``, add the geo_reference offset so that coordinates are
       absolute UTM values rather than relative to the domain origin.


Data attributes
---------------

After construction the following NumPy arrays are available.  All
centroid-based arrays update in place as the simulation progresses
(they are views into the domain's quantity arrays, not copies).

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Attribute
     - Description
   * - ``triang``
     - ``matplotlib.tri.Triangulation`` object for the mesh
   * - ``x``, ``y``
     - Node coordinates (1-D arrays)
   * - ``xc``, ``yc``
     - Centroid coordinates (1-D arrays)
   * - ``elev``
     - Elevation at centroids (m)
   * - ``stage``
     - Water surface elevation at centroids (m)
   * - ``depth``
     - Water depth at centroids: ``stage - elev`` (m)
   * - ``xmom``, ``ymom``
     - x- and y-momentum at centroids (m²/s)
   * - ``xvel``, ``yvel``
     - x- and y-velocity at centroids (m/s), zero where dry
   * - ``speed``
     - Flow speed at centroids (m/s)
   * - ``friction``
     - Manning friction coefficient at centroids


Plotting methods
----------------

Each quantity has three methods: ``plot_*`` (display), ``save_*`` (write
PNG to ``plot_dir``), and ``make_*_animation`` (assemble saved PNGs into
a ``FuncAnimation``).

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``plot_mesh(figsize, dpi)``
     - Plot the triangular mesh
   * - ``plot_depth_frame(figsize, dpi, vmin, vmax)``
     - Plot water depth at the current simulation time.
       Dry cells shown in greyscale elevation.
   * - ``save_depth_frame(figsize, dpi, vmin, vmax)``
     - Save depth frame as PNG named ``<domain_name>_depth_<time>.png``
   * - ``make_depth_animation()``
     - Assemble all saved depth PNGs into a ``FuncAnimation``
   * - ``plot_stage_frame(figsize, dpi, vmin, vmax)``
     - Plot water surface stage
   * - ``save_stage_frame(figsize, dpi, vmin, vmax)``
     - Save stage frame as PNG
   * - ``make_stage_animation()``
     - Assemble all saved stage PNGs into a ``FuncAnimation``
   * - ``plot_speed_frame(figsize, dpi, vmin, vmax)``
     - Plot flow speed
   * - ``save_speed_frame(figsize, dpi, vmin, vmax)``
     - Save speed frame as PNG
   * - ``make_speed_animation()``
     - Assemble all saved speed PNGs into a ``FuncAnimation``

All ``plot_*`` methods return ``(fig, ax)`` so the caller can add titles,
annotations, or further customisation before displaying.


Example: interactive notebook
------------------------------

The typical pattern in a Jupyter notebook is to call ``set_plotter`` once
before the evolve loop, then call ``save_*_frame`` at each yield step to
accumulate frames, and finally assemble them into an animation.

.. code-block:: python

   import anuga
   import matplotlib.pyplot as plt

   domain = anuga.rectangular_cross_domain(40, 40, len1=10.0, len2=10.0)

   domain.set_quantity('elevation', lambda x, y: -1.0 - 0.1 * x)
   domain.set_quantity('friction', 0.03)
   domain.set_quantity('stage', 0.0)

   Br = anuga.Reflective_boundary(domain)
   Bd = anuga.Dirichlet_boundary([0.5, 0.0, 0.0])
   domain.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})

   # Set up the plotter once before the evolve loop.
   domain.set_plotter(plot_dir='_plot', min_depth=0.01)

   # Determine colour scale limits from the initial state.
   vmax = domain.dplotter.depth.max() + 0.5

   for t in domain.evolve(yieldstep=10.0, finaltime=100.0):
       domain.print_timestepping_statistics()
       domain.save_depth_frame(vmin=0.0, vmax=vmax)

   # Assemble saved frames into an animation (displays inline in a notebook).
   anim = domain.make_depth_animation()
   anim   # display in Jupyter


Example: quick single-frame plot
----------------------------------

To inspect the state at a single point during or after the evolve loop:

.. code-block:: python

   domain.set_plotter()

   # Run a few steps
   for t in domain.evolve(yieldstep=10.0, finaltime=30.0):
       pass

   fig, ax = domain.plot_depth_frame(vmin=0.0, vmax=2.0)
   ax.set_title('Depth at t = 30 s')
   plt.show()


Example: plot the mesh
-----------------------

.. code-block:: python

   domain.set_plotter()
   fig, ax = domain.plot_mesh()
   plt.show()


.. note::

   ``Domain_plotter`` holds **live references** to the domain's centroid
   arrays.  Plots produced during the evolve loop always reflect the current
   simulation time — there is no need to re-create the plotter at each step.

.. note::

   ``make_*_animation`` assembles images that were previously saved with
   ``save_*_frame``.  Call ``save_*_frame`` at every yield step you want
   to appear in the animation before calling ``make_*_animation``.
