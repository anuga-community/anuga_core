.. currentmodule:: anuga

.. _sediment:

Sediment transport
==================

This documents the user-facing interface: every parameter, its units, its
default, and how to choose between the alternative methods. It assumes you
know ANUGA, and it does not derive the physics. Labels like :spec:`E-1` and the
section numbers in the tables refer to the internal sediment specification,
which is not distributed with ANUGA; they are kept as stable identifiers for
each term rather than as links you can follow.

A sediment grain size **is** a tracer with settling parameters attached, so
:ref:`tracers` covers the transport, boundary and conservation machinery that
this page builds on.

Verification evidence for these terms is in
``anuga/shallow_water/tests/test_sediment_*.py``; runnable
examples are in ``examples/sediment/``.

--------------

The shortest useful program
---------------------------

.. code-block:: python

   import anuga

   domain = anuga.rectangular_cross_domain(40, 10, len1=100.0, len2=25.0)
   domain.set_flow_algorithm('DE0')
   domain.set_quantity('elevation', lambda x, y: -0.01 * x)
   domain.set_quantity('stage', lambda x, y: -0.01 * x + 0.5)   # 0.5 m deep
   domain.set_quantity('friction', 0.03)
   domain.set_boundary({'left':   anuga.Dirichlet_boundary([0.8, 1.0, 0.0]),
                        'right':  anuga.Dirichlet_boundary([-0.5, 0.0, 0.0]),
                        'top':    anuga.Reflective_boundary(domain),
                        'bottom': anuga.Reflective_boundary(domain)})

   domain.add_grain_size('sand', diameter=2.0e-4)   # <- the only new line

   for t in domain.evolve(yieldstep=1.0, finaltime=30.0):
       pass

   print('peak concentration : %.3f' % domain.get_tracer('sand').max())

After 30 s this scours 5-22 cm from the bed and carries a peak concentration of
about 0.145.

.. note::

   The stage follows the bed (``-0.01 * x + 0.5``) and the boundaries drive
   flow through the channel. A *flat* stage over a sloping bed is hydrostatic
   -- the water does not move, the bed shear stress never reaches the Shields
   threshold, and nothing erodes. Sediment needs flow; if a model reports zero
   concentration everywhere, check the hydrodynamics before the sediment
   settings.

``add_grain_size`` is the entry point. One call gives you a transported
concentration, erosion, deposition, the settling velocity, the bed exchange,
and the limiters, with defaults chosen for a sand bed. It creates the sediment
operator and registers the fractional step for you, so there is nothing else to
wire up.

Everything below is about changing those defaults.

Print what you configured
-------------------------

.. code-block:: python

   print(domain.sediment_summary())

This is the single most useful call in the interface. It reports the active
configuration -- every law selected, every scalar in force, and each grain
size's derived settling velocity -- as text. For the program above, every line
but the first is a default:

::

   sediment configuration
     grain sizes        : 1  --  sand (d=0.0002 m)
     erosion            : Shields / Smith-McLean, non-cohesive (sand, gravel)   [E-1]
     deposition         : D = d* c v_s   [D-1]
     near-bed d*        : constant, per grain size
     shear closure      : quadratic drag, tau_b = rho f_c |v|^2   [T-1]
     friction closure   : constant n, from the domain friction quantity
     bedload            : off
     bed evolution      : True  (spec 2.4 Phase 4, evolving)
     porosity lambda    : 0.3
     c_max      [L-2]   : 0.3
     c_pack     [L-4]   : 0.65
     rho_w              : 1000 kg/m3
     erodible base [L-5]: none (unlimited depth)
     angle of repose    : off (spec 7)
     erodible region    : whole domain
     ([E-1] and the like are cross-references to the term in the physics;
      see the Sediment physics appendix -- the description before each
      label is the whole story.)
     per grain size:
       sand       d=0.0002 m  v_s=2.6695e-02 m/s  R=1.65  tau_c*=0.04

Settling velocity in particular is *derived*, not set: if ``v_s`` is not what
you expected, the diameter or the fluid properties are not what you thought.
Print this at the top of every run.

Lines appear only when they apply -- configure the Rouse near-bed profile, for
instance, and an ``a/h floor`` line joins the list. Everything below this point
is about changing one of these defaults, and the summary is how you check that
the change took.

The interface at a glance
-------------------------

Choices are made by naming the **physics**, never by setting a flag:

.. list-table::
   :header-rows: 1
   :widths: 54 36 10

   * - call
     - chooses
     - spec
   * - ``initialize_sediment_operator(...)``
     - sediment transport on, and the domain-wide parameters
     - 2.2
   * - ``add_grain_size(name, diameter, ...)``
     - a grain size to carry, and its own properties
     - 2.2
   * - ``set_bed_material(material, ...)``
     - the erosion law
     - 4.1.1
   * - ``set_deposition(law, near_bed, ...)``
     - the deposition law and near-bed ratio
     - 4.4
   * - ``set_shear_closure(closure)``
     - how ``tau_b`` is formed
     - 3.2
   * - ``set_sediment_friction(mode, ...)``
     - the friction factor feeding ``tau_b``
     - 3.3
   * - ``set_bedload(formula, ...)``
     - bedload transport, or off
     - 5
   * - ``set_sediment_parameters(...)``
     - the scalar physical properties
     - 2.4, 6
   * - ``set_erodible_base(...)``
     - the depth below which nothing erodes
     - 4.5
   * - ``set_erodible_region(...)``
     - where erosion may act at all
     - 4.5
   * - ``set_angle_of_repose(...)``
     - relaxation of over-steep bed slopes
     - 7
   * - ``set_tracer_source(name, values)``
     - an external source
     - 2.6
   * - ``set_tracer_boundary(name, tag, value)``
     - inflow concentration, per boundary tag
     - 2.5

.. seealso::

   :ref:`sediment_physics`
      What each of those calls is choosing between -- the erosion, deposition,
      shear and bedload laws, and how to tell which one your problem wants.
      The defaults are a working sand-bed configuration, so you can leave them
      alone until you need to say otherwise.

   :ref:`sediment_labels`
      What the bracketed labels mean. :spec:`E-1`, :spec:`T-1` and the rest name a
      term in the physics; they appear in this page, in the source, and in the
      output of ``sediment_summary()``.

Order does not matter, with one exception noted under :ref:`grain_sizes`
below: call them before ``evolve()``, in whatever order reads best.

**Anything not in that table is internal.** The domain carries roughly fifteen
``sediment_*`` arrays (``sediment_qbx``, ``sediment_settling_velocity``,
``sediment_erosion_mode``, ...) that exist to be handed to the C kernel. Setting
them directly can leave the GPU mapping stale, and no validation runs. Use the
setters; they invalidate the device mapping for you.

--------------

Setting sediment up
-------------------

Sediment transport is configured through two calls, and the split between them
is the point: a parameter belongs to exactly one of the two, so there is never
a question of which call wins.

.. code-block:: python

   domain.initialize_sediment_operator(porosity=0.28, rho_w=1000.0)  # the run
   domain.add_grain_size('sand', diameter=2.0e-4)                    # a grain
   domain.add_grain_size('silt', diameter=2.0e-5, tau_c_star=0.11)   # another

``initialize_sediment_operator`` takes what describes the **run**;
``add_grain_size`` takes what describes **one grain size**. Neither accepts the
other's parameters -- passing ``diameter=`` to the first, or ``rho_w=`` to the
second, is a ``TypeError`` rather than a silently ignored argument.

The two may be called in either order, and ``initialize_sediment_operator`` is
optional: ``add_grain_size`` creates the operator with default domain-wide
parameters if none exists, which is why the program at the top of this page is
a single line.

.. _sediment_operator_init:

``initialize_sediment_operator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   domain.initialize_sediment_operator(
       porosity=None, c_max=None, c_pack=None, bed_evolution=None, rho_w=None,
       description=None, label=None, logging=False, verbose=False)

Switches sediment transport on and returns the operator. The physical
parameters are the same ones :ref:`set_sediment_parameters <scalar_parameters>`
takes, and are documented there; passing them here is a convenience, and
``set_sediment_parameters`` can still change them afterwards.

**One operator per domain.** Calling it again returns the same operator,
applying any parameters given the second time. That is not just tidiness: the
kernel makes a single pass over every registered grain size, so a second
operator in the fractional-step list would apply the bed exchange twice per
timestep.

.. code-block:: python

   op = domain.initialize_sediment_operator(porosity=0.28)
   op is domain.initialize_sediment_operator(rho_w=1025.0)   # True

The return value is the operator, which you need only for controlling operator
order -- see :ref:`operator_order` below.

.. _grain_sizes:

``add_grain_size``
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   domain.add_grain_size(
       name, diameter,
       rho_s=2650.0, tau_c_star=0.04, d_star=1.0, beta=None,
       initial_concentration=0.0, reference_height=None, **settling_kwargs)

.. list-table::
   :header-rows: 1
   :widths: 24 10 12 54

   * - parameter
     - units
     - default
     - meaning
   * - ``name``
     - --
     - required
     - label; also the tracer name
   * - ``diameter``
     - m
     - required
     - grain diameter :math:`d`; sets :math:`v_s` via :spec:`S-1`
   * - ``rho_s``
     - kg/m3
     - 2650
     - sediment density (quartz); enters as
       :math:`R = \rho_s/\rho_w - 1`
   * - ``tau_c_star``
     - --
     - 0.04
     - critical Shields stress :math:`\tau_c^{*}`, :spec:`E-1`
   * - ``d_star``
     - --
     - 1.0
     - near-bed ratio :math:`d^{*} = c_b/c`; 1.0 is well-mixed
   * - ``initial_concentration``
     - --
     - 0.0
     - volumetric :math:`c`, uniform
   * - ``beta``
     - --
     - domain's
     - edge reconstruction limiter, shared by all tracers
   * - ``reference_height``
     - m
     - ``None``
     - Rouse reference height :math:`a` in :spec:`S-4`; see
       :ref:`near_bed_d_star`

Multiple grain sizes are independent: each has its own concentration, settling
velocity and critical stress, and each exchanges with the same bed. Call it
once per grain size.

There is no ``rho_w`` here. Water density is a property of the fluid, and there
is one fluid, so it lives on ``initialize_sediment_operator`` and
``set_sediment_parameters``. Changing it afterwards recomputes :math:`R` and
:math:`v_s` for every grain size already registered.


What belongs to a grain size, and what to the run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only the grain's own properties are per grain size; every physics choice, and
the bed itself, is shared.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Per grain size
     - Set by
   * - ``diameter``
     - ``add_grain_size``
   * - settling velocity :math:`v_s`
     - computed from ``diameter``, ``rho_s`` and any ``settling_kwargs``
   * - ``tau_c_star``
     - ``add_grain_size``
   * - ``d_star``, ``reference_height``
     - ``add_grain_size``
   * - submerged specific gravity :math:`R`
     - computed from ``rho_s`` and the shared ``rho_w``

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Shared by every grain size
     - Set by
   * - ``porosity``, ``c_max``, ``c_pack``, ``rho_w``, ``bed_evolution``
     - ``initialize_sediment_operator`` or ``set_sediment_parameters``
   * - erosion law, ``tau_crit``, ``K_e``, ``K_partheniades``
     - :ref:`set_bed_material <sediment_physics>`
   * - deposition law, ``tau_d``, near-bed :math:`d^{*}` mode
     - ``set_deposition``
   * - shear closure; friction mode and its parameters
     - ``set_shear_closure``, ``set_sediment_friction``
   * - bedload mode, ``K``, ``m``, and bedload's own ``tau_c_star``
     - ``set_bedload``
   * - ``beta``
     - one ``beta_tracer`` for every tracer on the domain
   * - **the bed**
     - one ``elevation``, which all grain sizes erode and deposit onto

Because the shared settings live on the **domain**, not on the operator, order
does not matter: ``set_bed_material('cohesive')`` called after both grain sizes
are registered applies to both.

Grain sizes occupy tracer slots in call order, so grain size ``s`` is tracer
``s``. **The one ordering rule**: do not interleave ``add_tracer`` and
``add_grain_size`` on the same domain if you rely on that correspondence.


.. _operator_order:

Controlling operator order
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fractional-step operators run in the order they are created. If another
operator must run before the sediment one -- an external source that the bed
exchange then consumes, say -- create the sediment operator at the point you
want it in the sequence, and add the grain sizes afterwards:

.. code-block:: python

   My_source_operator(domain, ...)          # runs first
   domain.initialize_sediment_operator()    # then this

   for nm, d50 in grain_sizes:
       domain.add_grain_size(nm, diameter=d50)

``add_grain_size`` will not displace an operator that already exists, so the
order established here survives however many grain sizes follow. Ordinary
models do not need this: calling ``add_grain_size`` straight away, as
everywhere else on this page, puts the operator in a sensible place by itself.


Choosing a name
~~~~~~~~~~~~~~~

``name`` is a label you choose, not a value from a list. Nothing is inferred
from it: ``diameter`` and the settling parameters do all the work, so a grain
size called ``'boulder'`` with ``diameter=2e-5`` is silt, and behaves like
silt.

.. code-block:: python

   domain.add_grain_size('fine_sand', diameter=1.5e-4)
   domain.add_grain_size('mud',       diameter=2.0e-5)

   domain.get_sediment_names()        # ['fine_sand', 'mud']
   domain.get_tracer('fine_sand')     # its concentration, per cell

It is also the **tracer** name, which is what constrains it. It must be a
non-empty string, unique on the domain, and it may not be

* the name of a quantity -- ``stage``, ``elevation``, ``friction``,
  ``xmomentum``, ``ymomentum``, ``height``, ``x``, ``y``, ``xvelocity``,
  ``yvelocity``. Both a quantity and a tracer are written to the sww as
  ``<name>_c``, so a grain size called ``stage`` would overwrite the stage in
  the output.
* anything beginning ``max_``, which is reserved for the running maxima
  ``Collect_max_quantities_operator`` writes.

Both are refused rather than allowed to corrupt the output. The name also
becomes the sww variable ``<name>_c``, so pick something you will recognise
when you open the file six months later.

.. warning::

   Do not confuse this with ``set_sediment_friction(bed='sand')``. **That**
   ``'sand'`` is one of a fixed set -- ``'sand'``, ``'gravel'``,
   ``'boulder'`` -- selecting a roughness closure, and has nothing to do with
   what you called your grain size. ``name`` is the only sediment argument
   that is free text; every other choice below names a physics option from a
   fixed vocabulary.


If you leave a grain size out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``name`` and ``diameter`` describe one grain size, so they travel together:
both are required positional parameters, and omitting either is a
``TypeError`` at the call rather than a partial registration.

.. code-block:: python

   domain.add_grain_size('sand')
   # TypeError: add_grain_size() missing 1 required positional argument: 'diameter'

Setting sediment up with **no** grain size at all is legal, and is what
``initialize_sediment_operator`` on its own does -- see :ref:`operator_order`.
It is a configured run waiting for its sediment, not an error. But it is only
useful as a step on the way to ``add_grain_size``: an operator with no grain
sizes transports nothing, and evolving that way warns rather than completing
silently with the bed untouched.

.. note::

   **This includes bedload.** Moving the bed without carrying anything in
   suspension is a reasonable thing to want, and it does not remove the need
   for a grain size: the diameter and :math:`R` that set the Shields stress
   -- and so the transport vector :math:`\mathbf{q}_b` -- live on a grain
   size. With none registered, the bedload kernel returns immediately and the
   bed does not move.

   For bed evolution with no suspended sediment, register the grain size and
   choose the total-load formula, which turns the suspended exchange off:

   .. code-block:: python

      domain.add_grain_size('sand', diameter=2.0e-4)
      domain.set_bedload('engelund_hansen')    # [K-5], total load

   On the channel at the top of this page that scours about 14 cm of bed while
   the concentration stays at zero throughout.


.. _42-choosing-tau_c_star:

Initial and boundary concentrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   domain.set_tracer('sand', 0.001)          # uniform, or an array of centroids
   domain.set_tracer_boundary('sand', 'inflow', 0.02)   # entering across 'inflow'

Both are volumetric concentration ``c`` (dimensionless), not ``h*c``. The
conserved quantity is ``m = h*c``; the interface works in ``c`` throughout.

--------------

.. _scalar_parameters:

Scalar parameters
-----------------

.. code-block:: python

   domain.set_sediment_parameters(porosity=0.30, c_max=0.30, c_pack=0.65,
                                  bed_evolution=True, rho_w=1000.0)

All optional; only what you pass is changed. All are validated.

.. list-table::
   :header-rows: 1
   :widths: 37 10 15 37

   * - parameter
     - units
     - default
     - meaning
   * - ``porosity``
     - --
     - 0.30
     - bed porosity ``lambda``, :spec:`G-4`. Sediment volume leaving suspension is ``(1-lambda) dz``; the rest is pore space filled from the water column. LM15 use 0.28.
   * - ``c_max``
     - --
     - 0.30
     - :spec:`L-2`, ceiling on depth-averaged concentration (FG21; anugaSed use 0.20).
   * - ``c_pack``
     - --
     - 0.65
     - :spec:`L-4`, maximum packing bounding *near-bed* ``c_b = d* c``. Only bites when ``d* != 1``.
   * - ``bed_evolution``
     - --
     - ``True``
     - whether the bed moves
   * - ``rho_w``
     - kg/m3
     - 1000
     - fluid density used to form dimensional ``tau_b``

.. _91-bed_evolution-is-the-coupling-stage:

``bed_evolution`` is the coupling stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``False`` gives a **fixed bed**: sediment is entrained and deposited, and
concentration evolves, but elevation never changes. Choose it when

- comparing against analytic solutions, which assume constant depth;
- comparing against RDycore v1.0, which is configured this way;
- isolating a transport question from a morphology question.

``True`` (the default) evolves the bed through :spec:`G-4` and :spec:`G-5`. Choose it
for any real morphological problem. Both bed terms are applied in a single
fractional step.

--------------

The non-erodible base
---------------------

.. code-block:: python

   domain.set_erodible_base(depth=0.5)          # 0.5 m of erodible material
   domain.set_erodible_base(elevation=z_rock)   # or an absolute surface, (n,)
   domain.set_erodible_base()                   # remove it again

By default the bed is **bottomless**: erosion lowers it for as long as the flow
has the strength to. That is right for a deep alluvial channel and wrong
wherever the erodible layer is finite -- a reach floored by an outcrop, a lined
culvert, a dam apron, a soil layer of known depth over rock. :spec:`L-5` gives it a
floor.

The base is a **per-centroid field**, because bedrock is a surface. ``depth=``
measures down from the elevation set so far and is recorded as an elevation at
the moment of the call, so later changes to the elevation quantity do not drag
it around. ``elevation=`` gives the surface directly, in the domain's datum.
Scalars broadcast; arrays must be ``(n,)``. Give exactly one.

A base above the bed is rejected rather than silently accepted -- it would mean
negative erodible thickness, which is a mistake, not a configuration.

.. code-block:: python

   domain.erodible_thickness()   # (n,) metres remaining; 0 means bedrock

``sediment_summary()`` reports the range and how many cells have reached bedrock.

What it guarantees
~~~~~~~~~~~~~~~~~~

The limit is applied to the **source**, not by clamping elevation. Erosion is
scaled back to what the remaining thickness can supply, so the sediment that is
not eroded never enters the water column and the budget still closes to machine
precision. Clamping ``z`` afterwards would leave suspended sediment that came
from nowhere.

Where several classes compete for the last of the material they are scaled by
one shared proportional factor, not served in registration order: the bed
carries no per-class stratigraphy, so no class has a better claim, and the
answer must not depend on the order you registered the grain sizes.
Deposition is never scaled -- it is what replenishes the bed.

The two transport routes give **different strengths of guarantee**, and it is
worth knowing which you are relying on:

.. list-table::
   :header-rows: 1
   :widths: 33 33 33

   * - route
     - floor is
     - why
   * - suspended exchange :spec:`G-4`
     - **exact**
     - the limit is on the exchange term itself
   * - bedload :spec:`G-5`
     - within one step's flux
     - bedload is a divergence; see below

Bedload only redistributes, and stays exactly conservative with a base present,
because the limit is applied to the transport vector and to whole edges -- both
of which the two cells sharing an edge evaluate identically. The price is that
the floor is not exact: closing an edge for a cell that cannot pay also cancels
its neighbour's inflow, so the deficit migrates. Measured overshoot is
5.1e-6 m on a 1.0e-2 m layer. If you need bedload's floor exact, that is a
known limitation with a known fix (iterating the exhaustion flag to a fixed
point), not a mystery.

Restricting erosion to a region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base says how *deep* erosion may go; a region says *where* it may happen
at all.

.. code-block:: python

   domain.set_erodible_region(polygon=breach)                 # ONLY here erodes
   domain.set_erodible_region(polygon=apron, erodible=False)  # everywhere BUT here
   domain.set_erodible_region(center=[x, y], radius=25.0)     # a circle
   domain.set_erodible_region(indices=ids)                    # triangles directly
   domain.set_erodible_region(my_region)                      # a Region object
   domain.set_erodible_region()                               # remove it

The keyword arguments are the ones the region-based operators
(``Erosion_operator`` and friends) already take, resolved by the same ``Region``
class, so a polygon that selects a set of cells there selects the same set
here. A region that selects no cells is rejected rather than silently doing
nothing -- that is almost always a polygon in the wrong coordinates.

**Passing a Region is the general form.** ``Region`` understands more than the
keywords above -- ``line=``, ``poly=``, ``expand_polygon=`` -- so build one and hand
it over when you need those:

.. code-block:: python

   from anuga.abstract_2d_finite_volumes.region import Region

   domain.set_erodible_region(Region(domain, line=thalweg))
   domain.set_erodible_region(Region(domain, polygon=reach, expand_polygon=True))

Both do reach through: a line selects the cells it crosses (83 of 960 on a test
mesh), and ``expand_polygon`` genuinely changes the selection (504 cells against
480), because it intersects on vertices rather than centroids.

It must be a ``Region`` built on **this** domain -- one built elsewhere carries
triangle indices that mean nothing here, and is refused rather than silently
mis-selecting. A bare list of points passed positionally is refused too, with a
message pointing at ``polygon=``: taking it as a region would select every cell
and look like it worked.

**Locked means unscourable, not inert.** A locked cell is held at the
elevation it has when you call this, by giving it zero erodible thickness --
the restriction is :spec:`L-5` with the layer set to nothing, not a separate
mechanism. Sediment may still settle onto it, which is what a concrete apron
or a rock bar does in the field, and that new material is erodible again
because it now sits above the base. Under genuinely erosive flow such a cell
sits at exactly net zero: the limiter scales erosion back until it just
cancels deposition, so nothing piles up on a scoured apron.

The two compose, in either order, and neither discards the other:

.. code-block:: python

   domain.set_erodible_base(depth=0.4)        # 0.4 m of erodible material
   domain.set_erodible_region(polygon=reach)  # but only inside this reach

Where they disagree the stricter wins. ``sediment_summary()`` reports both, and
the thickness range it prints covers only the erodible cells -- locked ones
carry zero thickness and would otherwise drag the minimum to zero whatever the
layer is.

Cost
~~~~

None when unset. With no base configured the kernels take the path they took
before the feature existed, and produce bitwise identical results -- which is
asserted, not assumed, in ``test_sediment_erodible_base.py`` check E1.

--------------

Angle-of-repose relaxation
--------------------------

.. code-block:: python

   domain.set_angle_of_repose(35.0)     # degrees; FG21 use 35
   domain.set_angle_of_repose(None)     # off again (the default)

Where the centroid-to-centroid bed slope exceeds the critical angle, bed
material is moved downslope until it does not. Without it, scour will cut a
vertical face that in the field would collapse.

**It is off by default, and that is a considered default.** FG21, whose
formulation this is, are explicit that it is *a numerical heuristic, not
physics* -- real slope failures are advective. It exists to stop the rest of
the model breaking on over-steep slopes, and it has a side effect worth
knowing before you switch it on: it limits the steepness of canyon walls and
knickpoints, and so suppresses knickpoint retreat that may be real.

**Mass is conserved.** Material removed from an over-steep cell is deposited on
its neighbours, never discarded (measured drift 2.3e-13 m3 on 4.0e2 m3 of bed).
This is the sharpest difference from ANUGA's ``sanddune_erosion_operator``, which
lowers an over-steep cell and lets the material vanish.

It respects :spec:`L-5`: a cell cannot slump away material it is not allowed to
lose, so a locked cell or one already at its base stays put and its neighbours
relax around it.

The sweep count, which will surprise you
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is an explicit diffusion solve, and convergence from a badly over-steep
bed is slow. An over-steep cone (36.8 degrees) needed **793 sweeps** to reach a
30 degree limit from cold.

That is not what the per-timestep cap is sized for. In a running model the bed
is already near-relaxed and each step needs a handful of sweeps. The cap
(``max_sweeps``, default 50) exists for the pathological case, and **hitting it
is not fatal** -- progress carries over, so the bed keeps relaxing on
subsequent steps. It is reported so you know relaxation is lagging rather than
finished:

::

   Sediment_transport_operator: angle-of-repose relaxation hit its 50-sweep cap at
   t = 0.3 s; the bed may still exceed 30.0 degrees.

If you *start* from a bed steeper than the critical angle, expect that on the
first few steps. Either let it settle, or raise ``max_sweeps`` for that run.
``operator.repose_sweeps`` and ``operator.repose_cap_hits`` are available if you
want to watch it, and ``timestepping_statistics()`` prints the sweep count.

``relax`` defaults to 1.0. That is the fastest **stable** setting, not an
aggressive one -- the kernel already divides by the edge count for stability,
and 1.0 converged the cone in 793 sweeps against 2400+ at 0.3. Lower it only if
you see something pathological.

In parallel
~~~~~~~~~~~

The kernel sweeps internally so it stays on the device, and elevation is
exchanged once per timestep afterwards. Spec 7 asks for an exchange per sweep.
The consequence is that relaxation crosses a subdomain boundary one sweep per
*timestep* rather than one per sweep, so a slump spanning a boundary relaxes
more slowly there than it would in serial. Serial and single-subdomain results
are unaffected. See the specification, section 7.1.

--------------

External sources
----------------

.. code-block:: python

   domain.set_tracer_source('sand', values)   # array over centroids, or a scalar

Adds a source to the tracer equation directly (spec 2.6), in units of ``m``
(that is, ``h*c``) per second. This is what the manufactured-solution tests use
to impose an analytic forcing, and it is the hook for anything the bed
exchange does not describe -- a lateral inflow, a point discharge, a
prescribed release.

--------------

Running on the GPU
------------------

.. code-block:: python

   domain.set_compute_mode('unified')

``'legacy'`` is the CPU/OpenMP path; ``'unified'`` is the shared path that can
run on the device. **Both paths share the same kernel file, so the physics is
the same code** -- ``core_kernels.c`` -- and ``test_sediment_gpu.py`` holds
them to agreement.

Nothing about the sediment configuration changes between modes: set it up the
same way and switch the mode.

``'unified'`` selects the *unified* code path; whether that path actually
offloads to a device is a property of the **build**, not of this call. A build
without offload compiles the same kernels under ``CPU_ONLY_MODE`` and runs them
on the host. ``set_compute_mode('unified')`` therefore does not fail on a
machine with no GPU, but it also does not report one:
``domain.get_compute_mode()`` reads ``'unified'`` either way. To find out what
you are actually running on, ask the build:

.. code-block:: python

   import anuga
   anuga.gpu_offload_enabled()    # True if this build offloads

To confirm kernels are reaching the device on a run, set
``NVCOMPILER_ACC_NOTIFY=1`` in the environment. Polling ``nvidia-smi`` is
unreliable for this -- the sampling interval misses short kernel bursts.

Call ``set_compute_mode`` **after** the sediment setup. Each setter
invalidates the device mapping, so configuring sediment after selecting
``'unified'`` simply forces the mapping to be rebuilt.

--------------

.. _coming_from_erosion_operators:

Coming from the erosion operators
---------------------------------

ANUGA has carried a family of erosion operators for a long time --
``Polygonal_erosion_operator``, ``Circular_erosion_operator``,
``Bed_shear_erosion_operator``, ``Flat_slice_erosion_operator``,
``Flat_fill_slice_erosion_operator`` and ``Sanddune_erosion_operator``. They
still work and nothing is scheduled for removal, but they are the expensive
way to evolve a bed: each runs in Python on the host every timestep, and none
is GPU-safe, so under compute mode ``'unified'`` each one forces a
GPU-to-host sync on every RK step.

Measured on 115,200 triangles, the overhead an erosion operator adds above a
plain run is **12.9x that of the entire sediment transport module on the GPU**,
and 3.3x on the CPU -- while modelling less and conserving nothing.

Structural equivalents
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - erosion operator argument
     - sediment equivalent
   * - ``base=``
     - :meth:`set_erodible_base` (``elevation=``), :spec:`L-5`
   * - ``polygon=``, ``center=``/``radius=``
     - :meth:`set_erodible_region`
   * - Sanddune's repose relaxation
     - :meth:`set_angle_of_repose`
   * - eroded material simply disappears
     - ``set_deposition(law='threshold', tau_d=0.0)`` suppresses redeposition

``Bed_shear_erosion_operator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This one maps almost exactly. It forms ``1000 * 9.81 * d * EN_slope`` -- that
is :math:`\rho g h S` with :math:`S` the **energy** slope -- and then erodes
``de = tau_b / shear_factor * dt``. That is the cohesive Hanson & Simon law
:spec:`E-3`, :math:`E = K_e(\tau_b - \tau_c)`, with no threshold and
:math:`K_e = 1/\texttt{shear\_factor}`:

.. code-block:: python

   domain.add_grain_size('sand', diameter=2.0e-4)
   domain.set_shear_closure('energy_slope')            # tau_b = rho g h S   [T-7e]
   domain.set_bed_material('cohesive', tau_crit=1e-9,
                           K_e=0.5 / shear_factor)     # E = K_e tau_b
   domain.set_deposition(law='threshold', tau_d=0.0)   # no redeposition
   domain.set_erodible_base(elevation=base)
   domain.set_erodible_region(polygon=polygon)

Use ``'energy_slope'`` rather than ``'depth_slope'``: the old operator used the
free-surface slope, and choosing the bed slope instead is what costs the
agreement. On a sloping channel over 30 s, correlation of the bed-change field
against the original operator:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - closure
     - correlation with the old operator
   * - ``'depth_slope'`` -- bed slope, :spec:`T-7`
     - 0.815
   * - ``'energy_slope'`` -- free surface, :spec:`T-7e`
     - **0.964**
   * - ``'energy_slope'`` with ``K_e`` calibrated
     - **0.988**

``K_e`` needs calibrating: the two gradient reconstructions differ, so
``1/shear_factor`` is the right form but not the right constant. Halving it
matched the case above. Calibrate against a run you trust.

The others
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - operator
     - notes
   * - ``Polygonal``, ``Circular``
     - Erode at ``de = |momentum| * dt``. There is no sediment law of that
       form, and it is dimensionally inconsistent -- ``|momentum|`` is
       m\ :sup:`2`/s, so ``de`` is not a length. Treat it as a tuning knob,
       not a rate, and move to the Shields route :spec:`E-1` (the default)
       rather than trying to reproduce it.
   * - ``Flat_slice``, ``Flat_fill_slice``
     - Not erosion: they set elevation to a target value. Use
       ``set_quantity('elevation', ...)`` or ``Set_elevation_operator``.
       Sediment has no equivalent because these are not scour models.
   * - ``Sanddune``
     - Erosion plus repose. The repose half is
       :meth:`set_angle_of_repose`; the erosion half is another excess-shear
       law, as for ``Bed_shear`` above.

.. warning::

   Nothing prevents enabling an erosion operator **and** sediment transport on
   the same domain. Both write ``elevation`` and the bed changes simply add.
   One conserves mass and the other does not, so the sum is unlikely to mean
   anything -- pick one.

--------------

Choosing a configuration
------------------------

If you do not know where to start:

- **Sand bed, flood or dam break, morphology wanted.** Defaults, plus one
  grain size:
  ``domain.add_grain_size('sand', diameter=2e-4)``. Add
  ``set_bedload('wong_parker_eq24')`` if the grains are coarse enough to move
  along the bed.
- **Fine cohesive sediment, muddy estuary.** ``set_bed_material('cohesive')``
  with a ``tau_crit`` you trust, ``d*`` left at 1.0.
- **Deep, slow, stratified flow.** ``set_deposition(near_bed='rouse')``, and give
  each class a ``reference_height``.
- **Shallow flow over gravel.** ``set_sediment_friction('wilson', bed='gravel', grain_size=...)``.
- **Reproducing anugaSed.** ``set_bed_material('cohesive')`` and
  ``set_shear_closure('depth_slope')``; see ``examples/sediment/``.
- **Comparing against an analytic solution.**
  ``set_sediment_parameters(bed_evolution=False)`` and leave ``d*`` at 1.0.
- **A finite erodible layer over rock.** ``set_erodible_base(depth=...)``, and
  check ``erodible_thickness()`` afterwards to see where it bit.
- **Scour confined to one structure or reach.** ``set_erodible_region(polygon=...)``,
  or ``erodible=False`` to lock an apron while the rest of the domain erodes.
- **A dune or a steep bank that should collapse rather than stand vertical.**
  ``set_angle_of_repose(35.0)``, and read section 11.1 first.

Then print ``sediment_summary()`` and check it says what you meant.

--------------

What is not implemented
-----------------------

Vegetation drag (spec 8) is Phase 5 and absent. Neither validation rung of
spec 10 -- Rio Puerco, the crater breach -- has been attempted; the evidence
in ``anuga/shallow_water/tests/test_sediment_*.py`` is verification (the
equations are solved
correctly), which is a different claim from validation (they are the right
equations for the field case).
