.. _sediment_physics:

.. currentmodule:: anuga

Sediment physics: choosing the laws
===================================

.. note::

   **You can skip this page to begin with.** ``add_sediment_class`` picks a
   working set of laws for a sand bed, and :ref:`sediment` shows how to run a
   model with them. This appendix is for when you need to say *which* physics,
   rather than accept the defaults.

Each section below covers one choice: what the alternatives are, what they
assume, and how to tell which one your problem wants. Labels like ``[E-1]``
and the section numbers refer to the internal sediment specification, which is
not distributed with ANUGA; they are stable identifiers for each term rather
than links you can follow.

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

.. code-block:: python

   domain.set_bed_material('noncohesive')   # default
   domain.set_bed_material('cohesive', tau_crit=0.088, K_e=6.742e-7)
   domain.set_bed_material('partheniades', tau_crit=0.088, K_e=...)

The argument is the **material**, not the formula, because spec 4.1.1 is
explicit that these describe different sediment rather than competing
descriptions of the same sediment. Picking the wrong one is a physics error.

+------------------+-------------------------+-------------------------+
| material         | law                     | when                    |
+==================+=========================+=========================+
| ``noncohesive``  | ``[E-1]``/``[E-2]``     | sand, gravel, boulders  |
|                  | Shields, Smith & McLean |                         |
+------------------+-------------------------+-------------------------+
| ``cohesive``     | ``[E-3]`` Hanson &      | clay, silt,             |
|                  | Simon                   | consolidated mud        |
+------------------+-------------------------+-------------------------+
| ``partheniades`` | ``[E-4]`` Partheniades  | cohesive, where you     |
|                  |                         | have a site-calibrated  |
|                  |                         | ``K_e``                 |
+------------------+-------------------------+-------------------------+

``tau_crit`` (Pa, default 0.088) and ``K_e`` (m3/N/s) apply to the cohesive
routes only; the non-cohesive route takes its threshold per class from
``tau_c_star`` instead. The default ``K_e = 6.742e-7`` is anugaSed's.

The choice is not a small correction. On the same channel over 30 s, the
non-cohesive route scours 3-6 cm while the cohesive route accretes about a
millimetre -- the sign of the bed change reverses. See
``sandpit/sediment_examples/README.md``.

--------------

Deposition
----------

.. code-block:: python

   domain.set_deposition(law='d_star', tau_d=0.0, near_bed='constant',
                         reference_height_floor=0.01)

``law``
~~~~~~~

+-----------------+-------------------------+-------------------------+
| value           | expression              | when                    |
+=================+=========================+=========================+
| ``'d_star'``    | ``D = d* c v_s``,       | default; always         |
|                 | ``[D-1]``               | deposits                |
+-----------------+-------------------------+-------------------------+
| ``'threshold'`` | ``[D-2]``, deposition   | when you need           |
|                 | only where              | deposition suppressed   |
|                 | ``tau_b < tau_d``       | under strong flow       |
+-----------------+-------------------------+-------------------------+

``tau_d`` (Pa) is the threshold for ``'threshold'`` and is ignored otherwise.

.. _62-near_bed----the-d-ratio:

``near_bed`` -- the ``d*`` ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deposition is driven by the concentration *at the bed*, but the transported
quantity is depth-averaged. ``d* = c_b/c`` bridges them.

+----------------+-----------------------------------------------------+
| value          | meaning                                             |
+================+=====================================================+
| ``'constant'`` | ``d*`` is whatever each class was given (1.0 =      |
|                | well-mixed). Default.                               |
+----------------+-----------------------------------------------------+
| ``'rouse'``    | ``d*`` from the Rouse profile ``[S-4]``, recomputed |
|                | per cell per step                                   |
+----------------+-----------------------------------------------------+

``'constant'`` with ``d* = 1`` is the well-mixed assumption: fine sediment,
vigorous mixing, shallow flow. It is also what the analytic decay solutions
assume, so use it when comparing against them.

``'rouse'`` is the physical choice when the profile is stratified -- coarser
grains, or deeper and slower flow, where near-bed concentration genuinely
exceeds the mean. It costs an evaluation of the fitted ``d*(Z, a/h)``
polynomial per cell per class per step (§9.5 of the spec; 28 terms, maximum
error 0.82% over ``Z`` in [0.01, 2.5], ``a/h`` in [1e-3, 0.15], clamped at the
edges rather than extrapolated).

``reference_height_floor`` (default 0.01) is the floor on ``a/h``. The Rouse
profile is singular as the reference height approaches the bed, so ``a/h`` is
never allowed below this. Lowering it admits more stratification and more
near-bed concentration; it is a numerical guard, not a physical parameter, and
0.01 sits comfortably inside the fit range.

Near-bed concentration is bounded by ``c_pack`` ``[L-4]`` regardless. That bound
exists because equilibrium Rouse ``d*`` at vanishing shear will otherwise
deposit the entire water column in under a second.

--------------

Bed shear stress
----------------

Two independent choices feed ``tau_b``: how the stress is formed, and what
friction factor goes into it.

.. _71-set_shear_closure----how:

``set_shear_closure`` -- how
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   domain.set_shear_closure('quadratic_drag')   # default
   domain.set_shear_closure('depth_slope')

==================== ===================== =========
value                expression            spec
==================== ===================== =========
``'quadratic_drag'`` \`tau_b = rho f_c     v
``'depth_slope'``    ``tau_b = rho g h S`` ``[T-7]``
==================== ===================== =========

``'quadratic_drag'`` is the default and the right choice for unsteady or
rapidly varying flow -- dam breaks, floods, anything with significant
inertia.

``'depth_slope'`` assumes locally uniform flow, where friction balances gravity.
It is what anugaSed uses, so choose it when reproducing their results
(divergence **D1** in the spec). It degrades where that balance does not hold.

The two are interchangeable by construction: the kernel returns ``tau_b/rho``,
so everything downstream is unchanged by the choice.

.. _72-set_sediment_friction----what:

``set_sediment_friction`` -- what
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   domain.set_sediment_friction('constant')    # default
   domain.set_sediment_friction('wilson', bed='gravel', grain_size=0.02)
   domain.set_sediment_friction('larsen_lamb', k_s=0.05, r_d=2.0, r_br=2.0)

``'wilson'`` and ``'larsen_lamb'`` are not callable with the mode alone -- they
require a length scale and refuse without one, rather than inventing a
default:

- ``'wilson'`` needs ``grain_size > 0`` (D50 for sand, D84 for gravel or boulder);
- ``'larsen_lamb'`` needs either ``k_s`` or ``sigma_br``. There is no universal
  ``sigma_br``: it is site-measured, and LL16 report about 5 m at Moses Coulee.

+-------------------+------------------+-------------------------------+
| mode              | spec             | when                          |
+===================+==================+===============================+
| ``'constant'``    | ``[T-6]``        | default: ``f_c`` from the     |
|                   |                  | domain's Manning ``n``.       |
|                   |                  | Ordinary flood and channel    |
|                   |                  | work.                         |
+-------------------+------------------+-------------------------------+
| ``'wilson'``      | ``[T-8..T-12]``  | depth-dependent, from grain   |
|                   |                  | size. Shallow flow over       |
|                   |                  | coarse beds, where relative   |
|                   |                  | submergence matters.          |
+-------------------+------------------+-------------------------------+
| ``'larsen_lamb'`` | ``[T-13..T-15]`` | partitions total stress into  |
|                   |                  | grain and form drag. Bedforms |
|                   |                  | or roughness elements, where  |
|                   |                  | only the grain part drives    |
|                   |                  | sediment.                     |
+-------------------+------------------+-------------------------------+

``bed`` is ``'sand'`` or ``'gravel'``; ``grain_size`` (m) is the roughness length
scale; ``k_s`` (m) is the roughness height; ``r_d`` and ``r_br`` (default 2.0) are
Larsen-Lamb's drag partitioning ratios.

This affects **only** the sediment source term. The hydrodynamic friction
operator is untouched, so momentum still sees the domain's Manning ``n``
whatever you choose here.

--------------

Bedload
-------

.. code-block:: python

   domain.set_bedload('wong_parker_eq24')   # K=3.97, m=1.5, tau_c*=0.0495
   domain.set_bedload('wong_parker_eq23')   # K=4.93, m=1.6, tau_c*=0.0470
   domain.set_bedload('engelund_hansen')
   domain.set_bedload('off')                # default

Bedload ``[K-1]``-``[K-4]`` transports sediment along the bed rather than in
suspension, and drives its own bed evolution term ``[G-5]``. It is **off by
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