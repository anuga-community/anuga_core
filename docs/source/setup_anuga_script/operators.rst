.. currentmodule:: anuga

Operators
=========

Operators are objects that modify domain quantities each timestep.  They are
attached to a :class:`Domain` and called automatically during the evolve loop.
All operators accept a *region* (polygon, centre+radius, or explicit triangle
indices) so that their effect can be restricted to a spatial subset of the
mesh.

.. contents:: Contents
   :local:
   :depth: 1


Summary table
-------------

.. list-table::
   :header-rows: 1
   :widths: 32 35 33

   * - Operator
     - What it does
     - Typical use case
   * - :class:`Rate_operator`
     - Adds or removes water at a specified volumetric rate per unit area
       (m s⁻¹). Rate can be a constant, a function of ``(t)``,
       ``(x, y)``, or ``(x, y, t)``, or a quantity/array.
     - Rainfall, evaporation, infiltration, or a prescribed source/sink
       over a catchment polygon.
   * - :class:`Inlet_operator`
     - Injects a prescribed discharge (m³ s⁻¹) into a region, with
       optional velocity direction.  More physically meaningful than
       ``Rate_operator`` when the total flow rate (not areal rate) is known.
     - Pump outlets, stormwater inlets, river inflows with a known
       hydrograph.
   * - :class:`Set_quantity_operator`
     - Overrides any named quantity to a given value (constant, function of
       time, or array) each timestep over a region.
     - Pinning stage or friction to a prescribed value in a localised area.
   * - :class:`Set_stage_operator`
     - Sets water stage over a region each timestep, enforcing
       stage ≥ elevation.  Equivalent to ``Set_quantity_operator`` for
       ``'stage'`` with the wet-floor guard.
     - Maintaining a constant or time-varying water level in a reservoir or
       tidal basin.
   * - :class:`Set_elevation_operator`
     - Updates bed elevation over a region each timestep while maintaining
       flow continuity (adjusts stage so depth is preserved).
     - Moving boundaries, sediment deposition/erosion, or a gate that
       changes bed level during the simulation.
   * - :class:`Boyd_box_operator`
     - Models flow through a rectangular box culvert between two exchange
       regions using the Boyd head-discharge algorithm.  Accounts for inlet
       losses, Manning friction, and free/submerged flow states.
     - Road culverts, drainage pipes with rectangular cross-section.
   * - :class:`Boyd_pipe_operator`
     - Same as :class:`Boyd_box_operator` but for circular pipes
       (diameter instead of width × height).
     - Round drainage pipes and stormwater culverts.
   * - :class:`Weir_orifice_trapezoid_operator`
     - Models flow over or through a trapezoidal weir/orifice structure,
       switching between weir and orifice flow regimes automatically.
     - Spillways, broad-crested weirs, and trapezoidal channel structures.
   * - :class:`Internal_boundary_operator`
     - General user-defined structure: the discharge Q is computed by a
       Python function ``f(hw, tw)`` of headwater and tailwater stage.
       Positive Q flows from headwater to tailwater.
     - Non-standard hydraulic structures where no standard Boyd formula
       applies — e.g. tide gates, sluice gates with custom rating curves.
   * - :class:`Kinematic_viscosity_operator`
     - Applies a parabolic smoothing step that diffuses the depth-averaged
       velocity field according to ``du/dt = div(h ∇u)``.  Acts as a
       numerical stability aid for very fine meshes or sharp fronts.
     - Stabilising simulations with strong velocity gradients or sharp wet/
       dry fronts where spurious oscillations appear.


Rate Operators
--------------

:class:`Rate_operator` adds or removes water over a region each timestep.
The rate is in m s⁻¹ (volume per unit area per second); positive values add
water, negative values remove it.

.. code-block:: python

    import anuga

    # Uniform rainfall of 10 mm/hr over a polygon
    rainfall_rate = 10e-3 / 3600.0   # m/s

    rain_polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    rain = anuga.Rate_operator(domain, rate=rainfall_rate, polygon=rain_polygon)

Time-varying rate:

.. code-block:: python

    def storm(t):
        """20 mm/hr for the first hour, then dry."""
        if t < 3600:
            return 20e-3 / 3600.0
        return 0.0

    rain = anuga.Rate_operator(domain, rate=storm, polygon=rain_polygon)

.. autosummary::
   :toctree: generated

   Rate_operator


Inlet and Outlet Operators
---------------------------

:class:`Inlet_operator` injects a total discharge Q (m³ s⁻¹) into a region.
It is more convenient than :class:`Rate_operator` when the flow rate is known
rather than the areal rate.

.. code-block:: python

    import anuga
    import math

    # Inflow region — a circle of radius 5 m centred at (50, 50)
    inflow_region = anuga.Region(domain, center=[50.0, 50.0], radius=5.0)

    # Sinusoidal hydrograph peaking at 2 m³/s
    inflow = anuga.Inlet_operator(
        domain,
        inflow_region,
        Q=lambda t: max(0.0, 2.0 * math.sin(math.pi * t / 3600.0)),
    )

.. autosummary::
   :toctree: generated

   Inlet_operator


Setting Operators
-----------------

These operators overwrite a quantity each timestep.  They are useful for
maintaining prescribed conditions inside the domain (e.g. a reservoir with a
fixed level, or a moving bed).

.. code-block:: python

    import anuga

    reservoir = [[200, 300], [200, 400], [300, 400], [300, 300]]

    # Hold stage at 10.5 m in the reservoir polygon
    stage_ctrl = anuga.Set_stage_operator(domain, stage=10.5, polygon=reservoir)

    # Raise the bed by 0.01 m/hr (e.g. sediment deposition)
    t0 = 0.0
    initial_elev = 5.0
    deposition = anuga.Set_elevation_operator(
        domain,
        elevation=lambda t: initial_elev + 0.01 / 3600.0 * t,
        polygon=reservoir,
    )

.. autosummary::
   :toctree: generated

   Set_quantity_operator
   Set_stage_operator
   Set_elevation_operator


Culvert and Bridge Operators
-----------------------------

Culvert operators model flow between two exchange regions through a
hydraulic structure.  They are specified by two ``end_points`` (or
``exchange_lines``) that define the upstream and downstream faces.

.. code-block:: python

    import anuga

    # Rectangular box culvert: 1.2 m wide × 0.9 m high
    culvert = anuga.Boyd_box_operator(
        domain,
        losses=1.5,
        width=1.2,
        height=0.9,
        end_points=[[10.0, 5.0], [15.0, 5.0]],
        manning=0.013,
        verbose=False,
    )

    # Circular pipe culvert: 0.6 m diameter
    pipe = anuga.Boyd_pipe_operator(
        domain,
        losses=1.5,
        diameter=0.6,
        end_points=[[30.0, 5.0], [35.0, 5.0]],
        manning=0.013,
    )

    # Custom structure with a user-defined rating curve
    def gate_flow(hw, tw):
        """Simple orifice equation — positive Q flows hw→tw."""
        delta_h = max(hw - tw, 0.0)
        return 2.5 * delta_h ** 0.5

    gate = anuga.Internal_boundary_operator(
        domain,
        gate_flow,
        end_points=[[50.0, 5.0], [55.0, 5.0]],
    )

.. autosummary::
   :toctree: generated

   Boyd_box_operator
   Boyd_pipe_operator
   Weir_orifice_trapezoid_operator
   Internal_boundary_operator


Other Operators
---------------

.. autosummary::
   :toctree: generated

   Kinematic_viscosity_operator


.. seealso::

   `ANUGA User Manual — Chapter 10: Operators and Structures, Erosion and Culverts
   <https://github.com/anuga-community/anuga_user_manual>`_
   covers operators in depth, including erosion operators, culvert sizing
   worked examples, and the Boyd hydraulic flow equations.
