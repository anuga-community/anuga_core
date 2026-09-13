.. _sediment_physics:

.. currentmodule:: anuga

Sediment physics: choosing the laws
===================================

.. note::

   **You can skip this page to begin with.** ``add_grain_size`` picks a
   working set of laws for a sand bed, and :ref:`sediment` shows how to run a
   model with them. This appendix is for when you need to say *which* physics,
   rather than accept the defaults.

This page is a short review of the formulations ANUGA implements: for each
term in the governing equations, what the published alternatives are, what they
assume, and how to tell which one your problem wants. Sources are cited by
label -- [FG21]_, [RDy26]_ and so on -- and collected in `References`_ at the
end.

Where the sources disagree, they disagree about physics rather than notation,
and the page says so. Erosion is the clearest case: the cohesive and
non-cohesive routes are not competing fits to the same data but descriptions of
different bed material, so choosing between them is a statement about the bed.

Bracketed equation labels like :spec:`E-1` are internal identifiers for each
term. They appear in the source comments and in ``sediment_summary()`` so a
term can be traced from the code to the equation and on to the paper; they are
listed under `What the bracketed labels mean`_.

.. _sediment_notation:

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 16 56 28

   * - Symbol
     - Meaning
     - Units
   * - :math:`h`
     - water depth
     - m
   * - :math:`u, v`
     - depth-averaged velocity components
     - m s\ :sup:`-1`
   * - :math:`|\mathbf{v}|`
     - velocity magnitude, :math:`\sqrt{u^2+v^2}`
     - m s\ :sup:`-1`
   * - :math:`z`
     - bed elevation
     - m
   * - :math:`g`
     - gravitational acceleration (a domain parameter, not a constant)
     - m s\ :sup:`-2`
   * - :math:`\rho`, :math:`\rho_s`
     - water density; sediment particle density
     - kg m\ :sup:`-3`
   * - :math:`R`
     - submerged specific gravity, :math:`(\rho_s-\rho)/\rho`
     - --
   * - :math:`D`, :math:`d`
     - grain diameter
     - m
   * - :math:`\nu`
     - kinematic viscosity of water
     - m\ :sup:`2` s\ :sup:`-1`
   * - :math:`\kappa`
     - von Kármán constant, 0.41
     - --
   * - :math:`\lambda`
     - bed porosity
     - --
   * - :math:`n`
     - Manning coefficient
     - s m\ :sup:`-1/3`
   * - :math:`f`, :math:`f_c`
     - Darcy-Weisbach friction factor; friction coefficient
       :math:`f_c \equiv f/8`
     - --
   * - :math:`\tau_b`
     - bed shear stress
     - Pa
   * - :math:`\tau^{*}`
     - Shields (dimensionless) shear stress
     - --
   * - :math:`\tau_c^{*}`, :math:`\tau_c`
     - critical Shields stress for motion; its dimensional form
     - --, Pa
   * - :math:`\tau_x`
     - excess Shields stress, :math:`\tau^{*}-\tau_c^{*}`
     - --
   * - :math:`\tau_d`
     - critical stress for *deposition*
     - Pa
   * - :math:`u_*`
     - shear velocity, :math:`\sqrt{\tau_b/\rho}`
     - m s\ :sup:`-1`
   * - :math:`c`, :math:`c_s`
     - depth-averaged volumetric concentration (of grain size :math:`s`)
     - --
   * - :math:`c_b`
     - near-bed concentration
     - --
   * - :math:`m`, :math:`m_s`
     - conserved sediment variable, :math:`m_s \equiv h\,c_s`
     - m
   * - :math:`v_s`
     - settling velocity
     - m s\ :sup:`-1`
   * - :math:`Z`
     - Rouse number, :math:`v_s/(\kappa u_*)`
     - --
   * - :math:`d^{*}`
     - near-bed concentration ratio, :math:`c_b/c`
     - --
   * - :math:`a`, :math:`z_0`
     - reference height; roughness length
     - m
   * - :math:`h_\epsilon`
     - regularisation depth in the velocity recovery
     - m
   * - :math:`E`, :math:`D`
     - entrainment (erosion) flux; deposition flux
     - m s\ :sup:`-1`
   * - :math:`K_e`
     - erodibility coefficient
     - m\ :sup:`3` N\ :sup:`-1` s\ :sup:`-1`
   * - :math:`\mathbf{q}_b`, :math:`q_b^{*}`
     - bedload flux per unit width; its dimensionless form
     - m\ :sup:`2` s\ :sup:`-1`, --
   * - :math:`N_s`
     - number of grain sizes
     - --

**Grain-size percentiles.** :math:`D_{50}` is the median grain diameter: half
the bed material by mass is finer. :math:`D_{84}` is the diameter that 84% is
finer than, and so on for any :math:`D_{xx}`. The coarser percentiles are used
where the roughness is set by the largest grains present rather than the
typical one -- which is why the ``'wilson'`` closure below asks for
:math:`D_{50}` on a sand bed but :math:`D_{84}` on gravel or boulders.

.. warning::

   Two symbols are overloaded, by long convention in this literature, and both
   appear on this page.

   :math:`D` is **grain diameter** in the shear and bedload relations, and the
   **deposition flux** in the mass balance. :math:`m` is the **conserved
   variable** :math:`h\,c` in the transport equation, and the **exponent** in
   the bedload power law :math:`q_b^{*} = K\tau_x^{\,m}`.

   Which is meant is unambiguous from the equation, but they are worth
   flagging.

**Sign convention.** :math:`E` is positive from the bed into the water column,
:math:`D` is positive from the water column onto the bed, and the bed rises
under net deposition.


.. _sediment_governing_equations:

The equations being solved
--------------------------

Everything on this page is a choice of closure for one of the terms below. It is
worth reading the equations first: most of the parameters name a term here.

A sediment grain size is a tracer, so it starts from :ref:`the tracer transport
equation <tracer_transport_equation>` -- the conserved variable is mass per unit
area, :math:`m_s = h\,c_s`, for grain size :math:`s = 1 \dots N_s`. The state
vector the solver carries is

.. math::

   \mathbf{U} = \begin{bmatrix} h & uh & vh & m_1 & \dots & m_{N_s}\end{bmatrix}^{T}

**Suspended transport** [RDy26]_, which is [DL09]_ written per grain size. The
tracer equation with a source: what the bed gives up and what settles out of
the water column.

.. _spec-g-3:

.. math::

   \frac{\partial m_s}{\partial t}
   + \frac{\partial (u\,m_s)}{\partial x}
   + \frac{\partial (v\,m_s)}{\partial y}
   = E_s - D_s + S_{m_s}
   \qquad \text{[G-3]}

:math:`E_s` is entrainment from the bed and :math:`D_s` deposition onto it, both
per grain size. :math:`S_{m_s}` is an optional external supply -- hillslope yield,
a tributary load, rainfall washoff -- and is zero unless you set one.

**Bed evolution.** What leaves the water column arrives at the bed, and the bed
moves by the volume it gains, allowing for pore space:

.. _spec-g-4:

.. math::

   \frac{\partial z}{\partial t} = \frac{D - E}{1 - \lambda}
   \qquad \text{[G-4]}

with :math:`\lambda` the bed porosity, since a deposited volume
:math:`(1-\lambda)\,dz` of grains fills a bed volume :math:`dz`. This is the
Exner equation [Exn25]_, in the form used by [P14]_ and [FG21]_.

**Bedload.** When bedload is switched on it moves the bed too, by the divergence
of the bedload transport vector :math:`\mathbf{q}_b`:

.. _spec-g-5:

.. math::

   \frac{\partial z}{\partial t} = -\frac{1}{1 - \lambda}\,\nabla \cdot \mathbf{q}_b
   \qquad \text{[G-5]}

Both act on the same :math:`z`, and when both are active their contributions sum.

.. note::

   **What is not coupled.** The bed feeds back into the flow -- a moving :math:`z`
   changes the bed slope in the shallow water source term -- but the sediment does
   **not** feed back into momentum: the fluid density is held at :math:`\rho`
   regardless of :math:`c_s`, and the momentum equations are unchanged by the
   presence of sediment.

   Neither [FG21]_ nor [RDy26]_ does that coupling either. [LM15]_ is the
   contrasting approach: a fully coupled system in which the sediment is part
   of the hyperbolic problem rather than a tracer advected by an
   already-computed flux. It is the assumption most likely to fail for very energetic flows,
   where the mixture starts to behave like a debris flow.

   Whether the bed moves at all is itself a choice --
   ``set_sediment_parameters(bed_evolution=False)`` holds :math:`z` fixed and
   solves only the transport equation above.

The rest of this page is the closures: what :math:`E_s`, :math:`D_s`,
:math:`\mathbf{q}_b` and the bed shear stress they all depend on are taken to be.

--------------

Bed shear stress
----------------

Every erosion and deposition rate below depends on the bed shear stress, so this
is the term to get right first. [FG21]_, [RDy26]_ and [P14]_ all specify the
same quadratic drag law, in different notation:

.. _spec-t-1:

.. math::

   \tau_b = \rho\, f_c\, |\mathbf{v}|^2 \qquad \text{[T-1]}

where :math:`f_c` is the Darcy-Weisbach friction factor divided by eight,
:math:`f_c = f/8`. With a Manning closure :math:`f_c = g n^2 h^{-1/3}`.

From it follow the shear velocity, the dimensionless (Shields) stress, and the
excess stress that drives transport:

.. _spec-t-2:

.. math::

   u_* = \sqrt{\tau_b/\rho} = |\mathbf{v}|\sqrt{f_c}
   \qquad \text{[T-2]}

.. _spec-t-3:

.. math::

   \tau^{*} = \frac{\tau_b}{(\rho_s - \rho)\, g\, D} = \frac{u_*^2}{R\,g\,D}
   \qquad \text{[T-3]}

.. _spec-t-4:

.. math::

   \tau_x = \tau^{*} - \tau_c^{*} \qquad \text{[T-4]}

with :math:`R = \rho_s/\rho - 1` the submerged specific gravity and
:math:`\tau_c^{*}` the critical Shields stress for that grain size.

Velocity is recovered from momentum with ANUGA's depth-limiting form [MR12]_,
so that a vanishing depth does not produce a divergent velocity. [RDy26]_
adopts the same form:

.. _spec-t-5:

.. math::

   u = \frac{(uh)\,h}{h^2 + h_\epsilon^2}
   \qquad \text{[T-5]}

with :math:`h_\epsilon` a small regularisation depth: it leaves :math:`u`
unchanged where the water is deep and drives it smoothly to zero as the cell
dries, instead of dividing by a depth approaching zero.

The alternative depth-slope closure, :math:`\tau_b = \rho\,g\,h\,S`, is the one
[aSM16]_ used, and is kept for reproducing anugaSed's results.


Two independent choices feed ``tau_b``: how the stress is formed, and what
friction factor goes into it.

.. _71-set_shear_closure----how:

``set_shear_closure`` -- how
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   domain.set_shear_closure('quadratic_drag')   # default
   domain.set_shear_closure('depth_slope')

.. list-table::
   :header-rows: 1
   :widths: 26 50 24

   * - value
     - expression
     - label
   * - ``'quadratic_drag'``
     - :math:`\tau_b = \rho\, f_c\, |\mathbf{v}|^2`
     - :spec:`T-1`
   * - ``'depth_slope'``
     - :math:`\tau_b = \rho\, g\, h\, S`
     - :spec:`T-7`

``'quadratic_drag'`` is the default and the right choice for unsteady or
rapidly varying flow -- dam breaks, floods, anything with significant
inertia.

``'depth_slope'`` assumes locally uniform flow, where friction balances gravity.
It is what anugaSed uses, so choose it when reproducing their results
(divergence **D1** in the spec). It degrades where that balance does not hold.

The two are interchangeable by construction: the kernel returns
:math:`\tau_b/\rho`,
so everything downstream is unchanged by the choice.

.. _72-set_sediment_friction----what:

``set_sediment_friction`` -- what
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   domain.set_sediment_friction('constant')    # default
   domain.set_sediment_friction('wilson', bed='gravel', grain_size=0.02)
   domain.set_sediment_friction('larsen_lamb', k_s=0.05, r_d=2.0, r_br=2.0)

``'wilson'`` [Wil04]_ and ``'larsen_lamb'`` [LL16]_ are not callable with the
mode alone -- they
require a length scale and refuse without one, rather than inventing a
default:

- ``'wilson'`` needs ``grain_size > 0`` (:math:`D_{50}` for sand,
  :math:`D_{84}` for gravel or boulder -- see :ref:`sediment_notation`);
- ``'larsen_lamb'`` needs either ``k_s`` or ``sigma_br``. There is no universal
  ``sigma_br``: it is site-measured, and [LL16]_ report about 5 m at Moses
  Coulee.

.. list-table::
   :header-rows: 1
   :widths: 28 26 46

   * - mode
     - spec
     - when
   * - ``'constant'``
     - :spec:`T-6`
     - default: ``f_c`` from the domain's Manning ``n``. Ordinary flood and channel work.
   * - ``'wilson'``
     - ``[T-8..T-12]``
     - depth-dependent, from grain size. Shallow flow over coarse beds, where relative submergence matters.
   * - ``'larsen_lamb'``
     - ``[T-13..T-15]``
     - partitions total stress into grain and form drag. Bedforms or roughness elements, where only the grain part drives sediment.

``bed`` is ``'sand'`` or ``'gravel'``; ``grain_size`` (m) is the roughness length
scale; ``k_s`` (m) is the roughness height; ``r_d`` and ``r_br`` (default 2.0) are
Larsen-Lamb's drag partitioning ratios.

This affects **only** the sediment source term. The hydrodynamic friction
operator is untouched, so momentum still sees the domain's Manning ``n``
whatever you choose here.

--------------


Choosing ``tau_c_star``
~~~~~~~~~~~~~~~~~~~~~~~

0.04 is a reasonable default for sand. It is the threshold at which grains
begin to move, and erosion is zero below it, so it sets *when* the bed becomes
active, not how fast. If the bed does not erode when you expect it to, check
this against the Shields curve for your grain size before adjusting anything
else.



--------------

Erosion: naming the bed material
--------------------------------

The three routes are different physics, not different tunings of one law. Which
you want is a statement about the bed material.

**Non-cohesive** -- sand, gravel, boulders. A saturating near-bed reference
concentration after [SM77]_ and [Par98]_, as used by [FG21]_, recovered as a
flux by the settling velocity. The threshold is a Shields stress [Shi36]_:

.. _spec-e-1:

.. math::

   E^{*} = \frac{0.65\, \gamma_0\, S}{1 + \gamma_0\, S},
   \qquad S = \frac{\tau^{*}}{\tau_c^{*}} - 1
   \qquad \text{[E-1]}

.. _spec-e-2:

.. math::

   E = v_s\, E^{*} \qquad \text{[E-2]}

with :math:`\gamma_0 = 0.0024` empirical. The 0.65 is the maximum packing
fraction, so :math:`E^{*}` saturates rather than growing without bound.

**Cohesive** -- silt, clay, cohesive bank material; the regime of the Rio
Puerco field data [P13]_ that anugaSed was built for. An excess *dimensional*
stress law, calibrated by jet test [HS01]_ and specified for ANUGA by
[aSM16]_:

.. _spec-e-3:

.. math::

   E = K_e\,(\tau_b - \tau_c) \qquad \text{[E-3]}

.. _spec-e-5:

.. math::

   K_e = \frac{0.2 \times 10^{-6}}{\sqrt{\tau_c}}
   \quad [\mathrm{m^3\,N^{-1}\,s^{-1}}],
   \qquad \tau_c = \tau_c^{*}\,(\rho_s - \rho)\, g\, D_{50}
   \qquad \text{[E-5]}

**Partheniades** [Par65]_ -- the form [RDy26]_ uses, also on dimensional
stress, normalised by the threshold:

.. _spec-e-4:

.. math::

   E = K_e \, \frac{\tau_b - \tau_c}{\tau_c}
   \quad \text{for } \tau_b > \tau_c, \text{ else } 0
   \qquad \text{[E-4]}

Note that :spec:`E-1` is written in Shields stress while :spec:`E-3` and :spec:`E-4`
are in dimensional stress; that is the usual source of confusion between them.


.. code-block:: python

   domain.set_bed_material('noncohesive')   # default
   domain.set_bed_material('cohesive', tau_crit=0.088, K_e=6.742e-7)
   domain.set_bed_material('partheniades', tau_crit=0.088, K_e=...)

The argument is the **material**, not the formula, because spec 4.1.1 is
explicit that these describe different sediment rather than competing
descriptions of the same sediment. Picking the wrong one is a physics error.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - material
     - law
     - when
   * - ``noncohesive``
     - :spec:`E-1`/:spec:`E-2` Shields, Smith & McLean
     - sand, gravel, boulders
   * - ``cohesive``
     - :spec:`E-3` Hanson & Simon
     - clay, silt, consolidated mud
   * - ``partheniades``
     - :spec:`E-4` Partheniades
     - cohesive, where you have a site-calibrated ``K_e``

``tau_crit`` (Pa, default 0.088) and ``K_e`` (m3/N/s) apply to the cohesive
routes only; the non-cohesive route takes its threshold per class from
``tau_c_star`` instead. The default ``K_e = 6.742e-7`` is anugaSed's.

The choice is not a small correction. On the same channel over 30 s, the
non-cohesive route scours 3-6 cm while the cohesive route accretes about a
millimetre -- the sign of the bed change reverses. See
``examples/sediment/README.md``.

--------------

Deposition
----------

Deposition is the settling flux out of the water column. The default, after
[P14]_ and [FG21]_, is the near-bed concentration times the settling velocity:

.. _spec-d-1:

.. math::

   D = c_b\, v_s = d^{*}(Z)\, c\, v_s \qquad \text{[D-1]}

The alternative, from [RDy26]_, is a threshold form with a critical
*deposition* stress, which switches deposition off in flow strong enough to
keep grains suspended:

.. _spec-d-2:

.. math::

   D = v_s\, c \left(1 - \frac{\tau_b}{\tau_d}\right)
   \quad \text{for } \tau_b < \tau_d, \text{ else } 0
   \qquad \text{[D-2]}

Setting :math:`\tau_d = 0` disables deposition entirely, which is how the
passive-transport benchmarks are run.

The settling velocity itself is [FC04]_, smooth across the Stokes-to-turbulent
transition and branch-free. [Die82]_ is the more accurate polynomial fit for
natural irregular grains, at the cost of a branchy evaluation:

.. _spec-s-1:

.. math::

   v_s = \frac{R\,g\,d^2}{C_1 \nu + \sqrt{0.75\, C_2\, R\, g\, d^3}}
   \qquad \text{[S-1]}

with :math:`C_1 = 18`, :math:`C_2 = 0.4` for smooth spheres, and
:math:`1.0`, :math:`1.1` for natural irregular grains.


.. code-block:: python

   domain.set_deposition(law='d_star', tau_d=0.0, near_bed='constant',
                         reference_height_floor=0.01)

``law``
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 37 37

   * - value
     - expression
     - when
   * - ``'d_star'``
     - :math:`D = d^{*}\, c\, v_s`, :spec:`D-1`
     - default; always deposits
   * - ``'threshold'``
     - :spec:`D-2`, deposition only where :math:`\tau_b < \tau_d`
     - when you need deposition suppressed under strong flow

``tau_d`` (Pa) is the threshold for ``'threshold'`` and is ignored otherwise.

.. _near_bed_d_star:

.. _62-near_bed----the-d-ratio:

``near_bed`` -- the ``d*`` ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`d^{*}` is the ratio of near-bed to depth-averaged concentration, and it
is what makes :spec:`D-1` a near-bed law rather than a depth-averaged one. It is a
function of the Rouse number [Rou37]_

.. _spec-s-2:

.. math::

   Z = \frac{v_s}{\kappa\, u_*}, \qquad \kappa = 0.41
   \qquad \text{[S-2]}

At low :math:`Z` the grain is well mixed through the column and
:math:`d^{*} \to 1` (washload); at high :math:`Z` it concentrates near the bed
and :math:`d^{*} \gg 1`.

``'constant'`` takes :math:`d^{*} = 1` -- a uniform suspension. ``'rouse'``
evaluates the defining ratio given by [aSM16]_ after [DL09]_, obtained by
requiring that the sediment discharge be the depth integral of concentration
times velocity over a Rouse-Vanoni concentration profile and a logarithmic
velocity profile:

.. _spec-s-4:

.. math::

   d^{*} = \frac{\displaystyle\int_a^h \ln(z/z_0)\, dz}
                {\displaystyle\int_a^h
                   \left(\frac{h-z}{h-a}\cdot\frac{a}{z}\right)^{Z}
                   \ln(z/z_0)\, dz}
   \qquad \text{[S-4]}

with :math:`a` the reference height, in the sense of [vR84]_. ANUGA evaluates a fitted form of this
rather than the integral itself, for the reasons given below.


Deposition is driven by the concentration *at the bed*, but the transported
quantity is depth-averaged. ``d* = c_b/c`` bridges them.

.. list-table::
   :header-rows: 1
   :widths: 23 77

   * - value
     - meaning
   * - ``'constant'``
     - :math:`d^{*}` is whatever each grain size was given
       (:math:`d^{*}=1` is well-mixed). Default.
   * - ``'rouse'``
     - :math:`d^{*}` from the Rouse profile :spec:`S-4`, recomputed per cell
       per step

``'constant'`` with ``d* = 1`` is the well-mixed assumption: fine sediment,
vigorous mixing, shallow flow. It is also what the analytic decay solutions
assume, so use it when comparing against them.

``'rouse'`` is the physical choice when the profile is stratified -- coarser
grains, or deeper and slower flow, where near-bed concentration genuinely
exceeds the mean. It costs an evaluation of the fitted ``d*(Z, a/h)``
polynomial per cell per grain size per step (§9.5 of the spec; 28 terms, maximum
error 0.82% over ``Z`` in [0.01, 2.5], ``a/h`` in [1e-3, 0.15], clamped at the
edges rather than extrapolated).

``reference_height_floor`` (default 0.01) is the floor on ``a/h``. The Rouse
profile is singular as the reference height approaches the bed, so ``a/h`` is
never allowed below this. Lowering it admits more stratification and more
near-bed concentration; it is a numerical guard, not a physical parameter, and
0.01 sits comfortably inside the fit range.

Near-bed concentration is bounded by ``c_pack`` :spec:`L-4` regardless. That bound
exists because equilibrium Rouse ``d*`` at vanishing shear will otherwise
deposit the entire water column in under a second.

--------------

Bedload
-------

Bedload moves grains along the bed rather than through the water column.
Following [FG21]_, the transport rate is a power law in the excess Shields
stress, made dimensional by the grain size:

.. _spec-k-1:

.. math::

   q_b^{*} = K\, \tau_x^{\,m} \qquad \text{[K-1]}

.. _spec-k-2:

.. math::

   q_b = q_b^{*}\, \sqrt{\left(\tfrac{\rho_s}{\rho} - 1\right) g}\; D^{3/2}
   \qquad \text{[K-2]}

and it moves the bed by its divergence:

.. _spec-k-3:

.. math::

   \frac{\partial z}{\partial t}
     = -\frac{1}{1-\lambda}\, \nabla \cdot \mathbf{q}_b
   \qquad \text{[K-3]}

:math:`q_b` is a magnitude, so it is directed along the flow to give the
vector :math:`\mathbf{q}_b`, following [Par98]_.

.. list-table::
   :header-rows: 1
   :widths: 34 14 12 14 26

   * - Parameter set
     - :math:`K`
     - :math:`m`
     - :math:`\tau_c^{*}`
     - Notes
   * - [WP06]_ Eq 24, correcting [MPM48]_
     - 3.97
     - 1.5
     - 0.0495
     - the default; bedload only
   * - .. _spec-k-5:

       [EH67]_, as ``[K-5]``
     - :math:`0.05/f_c`
     - 2.5
     - 0
     - **total load**, no threshold

.. warning::

   Engelund & Hansen is a **total load** relation -- it already includes
   suspended transport. Selecting it therefore *replaces* the suspended source
   rather than supplementing it, and running both would double-count. Note also
   that its :math:`K` is friction-dependent, not a constant, and that it has no
   threshold: subtracting a :math:`\tau_c^{*}` from it would be a different
   model.


.. code-block:: python

   domain.set_bedload('wong_parker_eq24')   # K=3.97, m=1.5, tau_c*=0.0495
   domain.set_bedload('wong_parker_eq23')   # K=4.93, m=1.6, tau_c*=0.0470
   domain.set_bedload('engelund_hansen')
   domain.set_bedload('off')                # default

Bedload :spec:`K-1`-:spec:`K-4` transports sediment along the bed rather than in
suspension, and drives its own bed evolution term :spec:`G-5`. It is **off by
default**: it is a separate transport mode, not a refinement of suspension,
and enabling it changes what the model represents.

Enable it when the grain size is coarse enough that a significant fraction of
the load moves without going into suspension -- sand and gravel beds under
moderate flow. Leave it off for fine, fully suspended sediment.

The two Wong & Parker variants are their corrected Meyer-Peter-Muller fits;
Eq 24 is the default. ``K``, ``m`` and ``tau_c_star`` override the formula's
constants if you have a calibration.

Bedload only redistributes: it moves sediment between cells and conserves the
total exactly. The flux across each edge is centred, which is what makes it
antisymmetric and therefore conservative; see ``test_sediment_bedload.py``.


.. _sediment_labels:

What the bracketed labels mean
------------------------------

Labels like :spec:`E-1` name a **term in the physics**, not a reference. They
appear throughout this page, in the source comments, and in the output of
``domain.sediment_summary()``, so that a given term can be pointed at
unambiguously wherever it comes up. The list below is what each one names.

They originate in an internal specification that is not distributed with
ANUGA; the numbering is kept because it is already in the code and the
summaries, and renumbering would only break the correspondence.

**Settling and suspension**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - :spec:`S-1`
     - settling velocity :math:`v_s`, Ferguson & Church (2004)
   * - :spec:`S-2`
     - Rouse number :math:`Z = v_s/(\kappa\, u_*)`
   * - :spec:`S-4`
     - the Rouse near-bed concentration ratio :math:`d^{*}(Z,\, a/h)`

**Erosion**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - :spec:`E-1`, :spec:`E-2`
     - non-cohesive entrainment: Shields threshold with a Smith & McLean
       reference concentration
   * - :spec:`E-3`, :spec:`E-5`
     - cohesive erosion, Hanson & Simon excess-stress form
   * - :spec:`E-4`
     - Partheniades cohesive erosion

**Deposition**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - :spec:`D-1`
     - :math:`D = d^{*}\, c\, v_s`
   * - :spec:`D-2`
     - threshold deposition, :math:`D = v_s\, c\,(1 - \tau_b/\tau_d)`

**Bed shear and friction**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - :spec:`T-1`
     - quadratic drag, :math:`\tau_b = \rho\, f_c\, |\mathbf{v}|^2`
       -- the default closure
   * - :spec:`T-2`
     - shear velocity :math:`u_* = |\mathbf{v}|\sqrt{f_c}`
   * - :spec:`T-3`
     - dimensionless stress :math:`\tau^{*} = f_c |\mathbf{v}|^2/(R\, g\, d)`
   * - :spec:`T-4`
     - excess stress :math:`\tau_x = \tau^{*} - \tau_c^{*}`
   * - :spec:`T-5`
     - the depth-limiting velocity form ANUGA uses
   * - .. _spec-t-6:

       ``[T-6]``
     - constant Manning ``n``, taken from the domain's friction quantity
   * - .. _spec-t-7:

       ``[T-7]``
     - depth-slope closure, :math:`\tau_b = \rho\, g\, h\, S`
   * - .. _spec-t-8:
       .. _spec-t-10:

       ``[T-8]`` to ``[T-10]``
     - the ``'wilson'`` friction closure
   * - .. _spec-t-13:
       .. _spec-t-15:

       ``[T-13]`` to ``[T-15]``
     - the ``'larsen_lamb'`` friction closure

**Bedload**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - :spec:`K-1`, :spec:`K-2`
     - power law, :math:`q_b^{*} = K\, \tau_x^{\,m}`
   * - :spec:`K-3`
     - bed change from bedload,
       :math:`\partial z/\partial t = -\dfrac{1}{1-\lambda}\,\nabla\cdot\mathbf{q}_b`
   * - .. _spec-k-4:

       ``[K-4]``
     - the per-cell bedload transport vector :math:`\mathbf{q}_b`
   * - :spec:`K-5`
     - Engelund & Hansen total load, no threshold

**Coupling to the flow and the bed**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - :spec:`G-3`
     - the suspended source term,
       :math:`m_s \leftarrow m_s + \Delta t\,(E_s - D_s)`, including
       any external source
   * - :spec:`G-4`
     - Exner bed evolution from the suspended exchange
   * - :spec:`G-5`
     - the bedload divergence contribution to the bed

**Limiters**

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Label
     - Term
   * - .. _spec-l-1:

       ``[L-1]``
     - positivity
   * - .. _spec-l-2:

       ``[L-2]``
     - the concentration ceiling :math:`c_{\max}` (``c_max``)
   * - .. _spec-l-3:

       ``[L-3]``
     - a cap on the rate of bed change, ``|dz/dt| <= max_dz``. **Not
       implemented in ANUGA** -- listed so the gap in the numbering is not
       mistaken for an omission here. It is unrelated to the ``beta`` edge
       reconstruction limiter, which the tracers share.
   * - .. _spec-l-4:

       ``[L-4]``
     - the packing fraction :math:`c_\text{pack}` (``c_pack``) bounding
       near-bed concentration
   * - .. _spec-l-5:

       ``[L-5]``
     - the non-erodible base

.. _sediment_references:

References
----------

Cited by label throughout the page. The short labels are also the ones used in
the source comments and in the internal specification, so a term can be traced
from the code to the paper it comes from.

.. [DL09] Davy, P. and Lague, D. (2009). Fluvial erosion/transport equation of
   landscape evolution models revisited. *Journal of Geophysical Research:
   Earth Surface*, 114, F03007. doi:10.1029/2008JF001146

.. [Die82] Dietrich, W. E. (1982). Settling velocity of natural particles.
   *Water Resources Research*, 18(6), 1615-1626.

.. [EH67] Engelund, F. and Hansen, E. (1967). *A Monograph on Sediment
   Transport in Alluvial Streams.* Teknisk Forlag, Copenhagen.

.. [Exn25] Exner, F. M. (1925). Über die Wechselwirkung zwischen Wasser und
   Geschiebe in Flüssen. *Sitzungsberichte der Akademie der Wissenschaften*,
   Vienna.

.. [FC04] Ferguson, R. I. and Church, M. (2004). A simple universal equation
   for grain settling velocity. *Journal of Sedimentary Research*, 74(6),
   933-937.

.. [FG21] Fassett, C. I. and Goudge, T. A. (2021). Modeling the hydrodynamics,
   sediment transport, and valley incision of outlet-forming floods from
   Martian crater lakes. *Journal of Geophysical Research: Planets*, 126,
   e2021JE006979. doi:10.1029/2021JE006979

.. [HS01] Hanson, G. J. and Simon, A. (2001). Erodibility of cohesive
   streambeds in the loess area of the midwestern USA. *Hydrological
   Processes*, 15(1), 23-38.

.. [LL16] Larsen, I. J. and Lamb, M. P. (2016). Progressive incision of the
   Channeled Scablands by outburst floods. *Nature*, 538(7624), 229-232.
   doi:10.1038/nature19817. Uses ANUGA.

.. [LM15] Liu, X., Mohammadian, A., Kurganov, A. and Infante Sedano, J. A.
   (2015). Well-balanced central-upwind scheme for a fully coupled shallow
   water system modeling flows over erodible bed. *Journal of Computational
   Physics*, 300, 202-218. doi:10.1016/j.jcp.2015.07.043

.. [MPM48] Meyer-Peter, E. and Müller, R. (1948). Formulas for bed-load
   transport. *Proceedings of the 2nd Meeting of the IAHR*, Stockholm, 39-64.

.. [MR12] Mungkasi, S. and Roberts, S. G. (2012). Approximations of the
   Carrier-Greenspan periodic solution to the shallow water wave equations for
   flows on a sloping beach. *International Journal for Numerical Methods in
   Fluids*, 69(4), 763-780. Source of the velocity regularisation ANUGA uses.

.. [P13] Perignon, M. C., Tucker, G. E., Griffin, E. R. and Friedman, J. M.
   (2013). Effects of riparian vegetation on topographic change during a large
   flood event, Rio Puerco, New Mexico. *Journal of Geophysical Research:
   Earth Surface*, 118(3), 1193-1209. doi:10.1002/jgrf.20073

.. [P14] Perignon, M. C. (2014). *A Rolling Stone Gathers No Moss.* PhD thesis,
   University of Colorado Boulder.

.. [Par65] Partheniades, E. (1965). Erosion and deposition of cohesive soils.
   *Journal of the Hydraulics Division, ASCE*, 91(1), 105-139.

.. [Par98] Parker, G. (1998). *1D Sediment Transport Morphodynamics with
   Applications to Rivers and Turbidity Currents.* e-book.

.. [RDy26] Feng, D., Tan, Z., Xu, D., Johnson, J. and Bisht, G. (2026).
   *RDycore-sediment v1.0.* EGUsphere preprint 2026-4859.
   doi:10.5194/egusphere-2026-4859

.. [Rou37] Rouse, H. (1937). Modern conceptions of the mechanics of fluid
   turbulence. *Transactions of the ASCE*, 102, 463-505.

.. [SM77] Smith, J. D. and McLean, S. R. (1977). Spatially averaged flow over a
   wavy surface. *Journal of Geophysical Research*, 82(12), 1735-1746.

.. [Shi36] Shields, A. (1936). *Anwendung der Ähnlichkeitsmechanik und der
   Turbulenzforschung auf die Geschiebebewegung.* Mitteilungen der
   Preussischen Versuchsanstalt für Wasserbau und Schiffbau, Berlin.

.. [Wil04] Wilson, L., Ghatan, G. J., Head, J. W. and Mitchell, K. L. (2004).
   Mars outflow channels: a reappraisal of the estimation of water flow
   velocities from water depths, regional slopes, and channel floor
   properties. *Journal of Geophysical Research: Planets*, 109, E09003.
   doi:10.1029/2004JE002281

.. [WP06] Wong, M. and Parker, G. (2006). Reanalysis and correction of bed-load
   relation of Meyer-Peter and Müller using their own database. *Journal of
   Hydraulic Engineering*, 132(11), 1159-1168.

.. [aSM16] Perignon, M. C. (2016). *Using the Sediment Transport and Vegetation
   Operators in ANUGA.* anugaSed manual. The authoritative specification for
   the original anugaSed code.

.. [vR84] van Rijn, L. C. (1984). Sediment transport, part II: suspended load
   transport. *Journal of Hydraulic Engineering*, 110(11), 1613-1641.

.. note::

   Two things in the source that look like citations but are not. ``A22`` and
   ``A23`` are equation numbers in [RDy26]_'s appendix, and ``FP64`` in the GPU
   sources is double-precision floating point.
