.. currentmodule:: anuga

.. _sediment:

Sediment transport
==================

This documents the user-facing interface: every parameter, its units, its
default, and how to choose between the alternative methods. It assumes you
know ANUGA, and it does not derive the physics. Labels like ``[E-1]`` and the
section numbers in the tables refer to the internal sediment specification,
which is not distributed with ANUGA; they are kept as stable identifiers for
each term rather than as links you can follow.

A sediment class **is** a tracer with settling parameters attached, so
:ref:`tracers` covers the transport, boundary and conservation machinery that
this page builds on.

Verification evidence for these terms is in
``anuga/shallow_water/tests/test_sediment_*.py``; runnable
examples are in ``sandpit/sediment_examples/``.

--------------

The shortest useful program
---------------------------

.. code-block:: python

   import anuga

   domain = anuga.rectangular_cross_domain(40, 10, len1=100.0, len2=25.0)
   domain.set_flow_algorithm('DE0')
   domain.set_quantity('elevation', lambda x, y: -0.01 * x)
   domain.set_quantity('stage', 0.5)
   domain.set_quantity('friction', 0.03)
   domain.set_boundary({t: anuga.Reflective_boundary(domain)
                        for t in domain.get_boundary_tags()})

   domain.add_sediment_class('sand', diameter=2.0e-4)   # <- the only new line

   for t in domain.evolve(yieldstep=1.0, finaltime=30.0):
       pass

``add_sediment_class`` is the entry point. One call gives you a transported
concentration, erosion, deposition, the settling velocity, the bed exchange,
and the limiters, with defaults chosen for a sand bed. It registers the
fractional-step ``Sediment_operator`` for you, so there is nothing else to wire
up.

Everything below is about changing those defaults.

Print what you configured
-------------------------

.. code-block:: python

   print(domain.sediment_summary())

This is the single most useful call in the interface. It reports the active
configuration -- every law selected, every scalar in force, and each class's
derived settling velocity -- as text:

::

   sediment configuration
     classes            : 2  ['fine', 'coarse']
     erosion            : [E-1] Shields / Smith-McLean, non-cohesive (sand, gravel)
     deposition         : [D-1] D = d* c v_s
     near-bed d*        : [S-4] Rouse profile
     shear closure      : [T-1] quadratic drag, tau_b = rho f_c |v|^2
     friction closure   : wilson [T-8..10], bed=gravel, D=0.02 m
     bedload            : [K-1] power law, K=3.97 m=1.5 tau_c*=0.0495
     bed evolution      : True  (spec 2.4 Phase 4, evolving)
     porosity lambda    : 0.28
     c_max      [L-2]   : 0.35
     c_pack     [L-4]   : 0.65
     rho_w              : 1000 kg/m3
     a/h floor          : 0.01
     per class:
       fine       d=0.0001 m  v_s=8.0040e-03 m/s  R=1.65  tau_c*=0.04
       coarse     d=0.0005 m  v_s=9.4839e-02 m/s  R=1.65  tau_c*=0.04

Settling velocity in particular is *derived*, not set: if ``v_s`` is not what
you expected, the diameter or the fluid properties are not what you thought.
Print this at the top of every run.

The interface at a glance
-------------------------

Choices are made by naming the **physics**, never by setting a flag:

+---------------------------------------------+------------------------------+--------+
| call                                        | chooses                      | spec   |
+=============================================+==============================+========+
| ``add_sediment_class(name, diameter, ...)`` | a grain size to transport    | 2.2    |
+---------------------------------------------+------------------------------+--------+
| ``set_bed_material(material, ...)``         | the erosion law              | 4.1.1  |
+---------------------------------------------+------------------------------+--------+
| ``set_deposition(law, near_bed, ...)``      | the deposition law and       | 4.4    |
|                                             | near-bed ratio               |        |
+---------------------------------------------+------------------------------+--------+
| ``set_shear_closure(closure)``              | how ``tau_b`` is formed      | 3.2    |
+---------------------------------------------+------------------------------+--------+
| ``set_sediment_friction(mode, ...)``        | the friction factor feeding  | 3.3    |
|                                             | ``tau_b``                    |        |
+---------------------------------------------+------------------------------+--------+
| ``set_bedload(formula, ...)``               | bedload transport, or off    | 5      |
+---------------------------------------------+------------------------------+--------+
| ``set_sediment_parameters(...)``            | the scalar physical          | 2.4, 6 |
|                                             | properties                   |        |
+---------------------------------------------+------------------------------+--------+
| ``set_erodible_base(...)``                  | the depth below which        | 4.5    |
|                                             | nothing erodes               |        |
+---------------------------------------------+------------------------------+--------+
| ``set_erodible_region(...)``                | where erosion may act at all | 4.5    |
+---------------------------------------------+------------------------------+--------+
| ``set_angle_of_repose(...)``                | relaxation of over-steep bed | 7      |
|                                             | slopes                       |        |
+---------------------------------------------+------------------------------+--------+
| ``set_tracer_source(name, values)``         | an external source           | 2.6    |
+---------------------------------------------+------------------------------+--------+
| ``set_tracer_boundary(name, tag, value)``   | inflow concentration, per    | 2.5    |
|                                             | boundary tag                 |        |
+---------------------------------------------+------------------------------+--------+

.. seealso::

   :ref:`sediment_physics`
      What each of those calls is choosing between -- the erosion, deposition,
      shear and bedload laws, and how to tell which one your problem wants.
      The defaults are a working sand-bed configuration, so you can leave them
      alone until you need to say otherwise.

Order does not matter, with one exception noted under
``add_sediment_class`` below: call them before ``evolve()``, in whatever
order reads best.

**Anything not in that table is internal.** The domain carries roughly fifteen
``sediment_*`` arrays (``sediment_qbx``, ``sediment_settling_velocity``,
``sediment_erosion_mode``, ...) that exist to be handed to the C kernel. Setting
them directly can leave the GPU mapping stale, and no validation runs. Use the
setters; they invalidate the device mapping for you.

--------------

Sediment classes
----------------

.. _41-add_sediment_class:

``add_sediment_class``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   domain.add_sediment_class(name, diameter, d_star=1.0, beta=None,
                             initial_concentration=0.0, rho_s=2650.0,
                             rho_w=1000.0, tau_c_star=0.04,
                             reference_height=None, auto_operator=True,
                             **settling_kwargs)

+---------------------------+-------+----------+-------------------------+
| parameter                 | units | default  | meaning                 |
+===========================+=======+==========+=========================+
| ``name``                  | --    | required | label; also the tracer  |
|                           |       |          | name                    |
+---------------------------+-------+----------+-------------------------+
| ``diameter``              | m     | required | grain diameter ``d``;   |
|                           |       |          | sets ``v_s`` via        |
|                           |       |          | ``[S-1]``               |
+---------------------------+-------+----------+-------------------------+
| ``rho_s``                 | kg/m3 | 2650     | sediment density        |
|                           |       |          | (quartz)                |
+---------------------------+-------+----------+-------------------------+
| ``rho_w``                 | kg/m3 | 1000     | fluid density;          |
|                           |       |          | ``R = rho_s/rho_w - 1`` |
+---------------------------+-------+----------+-------------------------+
| ``tau_c_star``            | --    | 0.04     | critical Shields        |
|                           |       |          | stress, ``[E-1]``       |
+---------------------------+-------+----------+-------------------------+
| ``d_star``                | --    | 1.0      | near-bed ratio          |
|                           |       |          | ``c_b/c``; 1.0 is       |
|                           |       |          | well-mixed              |
+---------------------------+-------+----------+-------------------------+
| ``initial_concentration`` | --    | 0.0      | volumetric, uniform     |
+---------------------------+-------+----------+-------------------------+
| ``beta``                  | --    | domain's | limiter coefficient     |
|                           |       |          | ``[L-3]``               |
+---------------------------+-------+----------+-------------------------+
| ``reference_height``      | m     | ``None`` | Rouse reference height  |
|                           |       |          | ``a``; see the appendix |
+---------------------------+-------+----------+-------------------------+
| ``auto_operator``         | --    | ``True`` | register                |
|                           |       |          | ``Sediment_operator``   |
+---------------------------+-------+----------+-------------------------+

Multiple classes are independent: each has its own concentration, settling
velocity and critical stress, and each exchanges with the same bed. Call it
once per grain size.

Classes occupy tracer slots in call order, so class ``s`` is tracer ``s``. **The
one ordering rule**: do not interleave ``add_tracer`` and ``add_sediment_class``
on the same domain if you rely on that correspondence.

.. _42-choosing-tau_c_star:

Initial and boundary concentrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   domain.set_tracer('sand', 0.001)          # uniform, or an array of centroids
   domain.set_tracer_boundary('sand', 'inflow', 0.02)   # entering across 'inflow'

Both are volumetric concentration ``c`` (dimensionless), not ``h*c``. The
conserved quantity is ``m = h*c``; the interface works in ``c`` throughout.

--------------

Scalar parameters
-----------------

.. code-block:: python

   domain.set_sediment_parameters(porosity=0.30, c_max=0.30, c_pack=0.65,
                                  bed_evolution=True, rho_w=1000.0)

All optional; only what you pass is changed. All are validated.

+-------------------------+-------+----------+-------------------------+
| parameter               | units | default  | meaning                 |
+=========================+=======+==========+=========================+
| ``porosity``            | --    | 0.30     | bed porosity            |
|                         |       |          | ``lambda``, ``[G-4]``.  |
|                         |       |          | Sediment volume leaving |
|                         |       |          | suspension is           |
|                         |       |          | ``(1-lambda) dz``; the  |
|                         |       |          | rest is pore space      |
|                         |       |          | filled from the water   |
|                         |       |          | column. LM15 use 0.28.  |
+-------------------------+-------+----------+-------------------------+
| ``c_max``               | --    | 0.30     | ``[L-2]``, ceiling on   |
|                         |       |          | depth-averaged          |
|                         |       |          | concentration (FG21;    |
|                         |       |          | anugaSed use 0.20).     |
+-------------------------+-------+----------+-------------------------+
| ``c_pack``              | --    | 0.65     | ``[L-4]``, maximum      |
|                         |       |          | packing bounding        |
|                         |       |          | *near-bed*              |
|                         |       |          | ``c_b = d* c``. Only    |
|                         |       |          | bites when ``d* != 1``. |
+-------------------------+-------+----------+-------------------------+
| ``bed_evolution``       | --    | ``True`` | whether the bed moves   |
+-------------------------+-------+----------+-------------------------+
| ``rho_w``               | kg/m3 | 1000     | fluid density used to   |
|                         |       |          | form dimensional        |
|                         |       |          | ``tau_b``               |
+-------------------------+-------+----------+-------------------------+

.. _91-bed_evolution-is-the-coupling-stage:

``bed_evolution`` is the coupling stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``False`` gives a **fixed bed**: sediment is entrained and deposited, and
concentration evolves, but elevation never changes. Choose it when

- comparing against analytic solutions, which assume constant depth;
- comparing against RDycore v1.0, which is configured this way;
- isolating a transport question from a morphology question.

``True`` (the default) evolves the bed through ``[G-4]`` and ``[G-5]``. Choose it
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
culvert, a dam apron, a soil layer of known depth over rock. ``[L-5]`` gives it a
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
answer must not depend on the order you called ``add_sediment_class``.
Deposition is never scaled -- it is what replenishes the bed.

The two transport routes give **different strengths of guarantee**, and it is
worth knowing which you are relying on:

+----------------------+----------------------+----------------------+
| route                | floor is             | why                  |
+======================+======================+======================+
| suspended exchange   | **exact**            | the limit is on the  |
| ``[G-4]``            |                      | exchange term itself |
+----------------------+----------------------+----------------------+
| bedload ``[G-5]``    | within one step's    | bedload is a         |
|                      | flux                 | divergence; see      |
|                      |                      | below                |
+----------------------+----------------------+----------------------+

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

**Passing a ``Region`` is the general form.** ``Region`` understands more than the
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
the restriction is ``[L-5]`` with the layer set to nothing, not a separate
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

It respects ``[L-5]``: a cell cannot slump away material it is not allowed to
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

   Sediment_operator: angle-of-repose relaxation hit its 50-sweep cap at
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

   domain.set_multiprocessor_mode(2)

Mode 1 is the legacy CPU/OpenMP path; mode 2 is the unified path that runs on
the device. **Both paths share ``core_kernels.c``, so the physics is the same
code**, and ``test_sediment_gpu.py`` holds them to agreement.

Nothing about the sediment configuration changes between modes: set it up the
same way and switch the mode.

Mode 2 selects the *unified* code path; whether that path actually offloads to
a device is a property of the **build**, not of this call. A build without
offload compiles the same kernels under ``CPU_ONLY_MODE`` and runs them on the
host. ``set_multiprocessor_mode(2)`` therefore does not fail on a machine with
no GPU, but it also does not report one: ``domain.multiprocessor_mode`` will
read 2 either way. To find out what you are actually running on, ask the
build:

.. code-block:: python

   import anuga
   anuga.gpu_offload_enabled()    # True if this build offloads

To confirm kernels are reaching the device on a run, set
``NVCOMPILER_ACC_NOTIFY=1`` in the environment. Polling ``nvidia-smi`` is
unreliable for this -- the sampling interval misses short kernel bursts.

Call ``set_multiprocessor_mode`` **after** the sediment setup. Each setter
invalidates the device mapping, so configuring sediment after selecting mode 2
simply forces the mapping to be rebuilt.

--------------

Choosing a configuration
------------------------

If you do not know where to start:

- **Sand bed, flood or dam break, morphology wanted.** Defaults, plus a class:
  ``add_sediment_class('sand', diameter=2e-4)``. Add
  ``set_bedload('wong_parker_eq24')`` if the grains are coarse enough to move
  along the bed.
- **Fine cohesive sediment, muddy estuary.** ``set_bed_material('cohesive')``
  with a ``tau_crit`` you trust, ``d*`` left at 1.0.
- **Deep, slow, stratified flow.** ``set_deposition(near_bed='rouse')``, and give
  each class a ``reference_height``.
- **Shallow flow over gravel.** ``set_sediment_friction('wilson', bed='gravel', grain_size=...)``.
- **Reproducing anugaSed.** ``set_bed_material('cohesive')`` and
  ``set_shear_closure('depth_slope')``; see ``sandpit/sediment_examples/``.
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
