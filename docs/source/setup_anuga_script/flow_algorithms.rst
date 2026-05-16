
.. currentmodule:: anuga

Choosing a Flow Algorithm
=========================

ANUGA's flow algorithm controls how the shallow water equations are advanced in
time.  All available algorithms use the same 2nd-order discontinuous-elevation
spatial reconstruction (the "DE" family), but differ in their time-stepping
scheme and the aggressiveness of their slope limiters.

.. code-block:: python

    domain.set_flow_algorithm('DE1')   # set before evolve


Quick comparison
----------------

.. list-table::
   :header-rows: 1
   :widths: 14 12 12 12 14 36

   * - Algorithm
     - Time order
     - Flux calls / step
     - CFL
     - Slope limiter β
     - When to use
   * - ``DE0``
     - 1st
     - 1
     - 0.9
     - 0.5 (conservative)
     - Fast exploratory runs; very large or very dry domains
   * - ``DE1``
     - 2nd
     - 2
     - 1.0
     - 1.0 (full)
     - **General-purpose default — robust and accurate**
   * - ``DE_ader2``
     - 2nd
     - 1
     - 1.0
     - 1.0 (full)
     - Same accuracy as DE1 at the cost of DE0 — **recommended for production**
   * - ``DE2``
     - 3rd
     - 3
     - 1.0
     - 1.0 (full)
     - Rarely needed; reserved for highly transient flows

Recommendation
--------------

* **Start with DE1.** It is 2nd-order in both space and time, uses the full
  slope limiter, and is the most widely validated algorithm in ANUGA.  It is
  roughly half the speed of DE0 (two flux evaluations per step), but produces
  materially better results in flows with steep gradients, bores, or rapidly
  varying inundation fronts.

* **Use DE0 when speed matters more than temporal accuracy.** It is
  approximately twice as fast as DE1 per simulated second, but uses a more
  conservative slope limiter (β = 0.5) and first-order time stepping.  Results
  are still 2nd-order in space, so spatial gradients are well resolved; the
  1st-order time stepping introduces a mild diffusivity that is often acceptable
  for long-duration flood studies on large, mildly varying domains.

* **DE_ader2 is the future default.** It achieves the same 2nd-order temporal
  accuracy as DE1 but requires only one flux call per timestep (the same as
  DE0) by using an ADER Cauchy–Kovalewski predictor.  The first timestep
  bootstraps with a plain Euler step to establish the initial CFL dt; from the
  second step onwards it runs at full 2nd-order accuracy.  It is still
  being validated across the full range of ANUGA's test cases, so DE1 remains
  the safer choice for critical production work at present.

* **Avoid DE2 in routine use.** The SSP-RK3 scheme adds a third flux call per
  step (50% more expensive than DE1) for only marginal accuracy gains in the
  smooth-solution regime typical of shallow water problems.  Reserve it for
  specialised studies where 3rd-order temporal accuracy is demonstrably needed.


Algorithm details
-----------------

DE0 — Euler, CFL 0.9
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    domain.set_flow_algorithm('DE0')

* **Time stepping:** forward Euler — one extrapolation and one flux call per
  step.
* **CFL factor:** 0.9 (10% safety margin; slightly more conservative than the
  other algorithms).
* **Slope limiter:** β = 0.5, applied to both wet and transitional cells.  The
  reduced β damps spurious oscillations near wet/dry fronts and makes DE0 more
  robust on very heterogeneous terrain.
* **Minimum allowed height:** 1 × 10\ :sup:`−12` m (effectively zero); water
  can persist in very thin layers.
* **Cost:** ~1× (baseline).


DE1 — Runge–Kutta 2, CFL 1.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    domain.set_flow_algorithm('DE1')

* **Time stepping:** 2nd-order Runge–Kutta (Heun's predictor–corrector) —
  two flux calls per step.
* **CFL factor:** 1.0.
* **Slope limiter:** β = 1.0 (full reconstruction, no artificial limiting in
  smooth regions).
* **Minimum allowed height:** 1 × 10\ :sup:`−5` m.
* **Cost:** ~2× relative to DE0.


DE_ader2 — ADER-2 Cauchy–Kovalewski, CFL 1.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    domain.set_flow_algorithm('DE_ader2')

* **Time stepping:** 2nd-order ADER scheme using a Cauchy–Kovalewski (C-K)
  predictor.  After an initial Euler bootstrap step, each subsequent step
  requires only a single flux call: edge values are advanced to the midpoint
  time level Q\ :sup:`n+1/2` in-place using local SWE derivatives, and the
  flux is computed from those predicted edges.
* **CFL factor:** 1.0.
* **Slope limiter:** β = 1.0 (same as DE1).
* **Minimum allowed height:** 1 × 10\ :sup:`−5` m (same as DE1).
* **Cost:** ~1× (same as DE0 after the bootstrap step).
* **Note:** The predictor reuses the timestep from the previous step
  (``_ader2_prev_dt``), which is the global-minimum CFL dt across all MPI
  ranks.  This ensures consistency in parallel runs.


DE2 — SSP-RK3, CFL 1.0
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    domain.set_flow_algorithm('DE2')

* **Time stepping:** strong-stability-preserving 3rd-order Runge–Kutta — three
  flux calls per step.
* **CFL factor:** 1.0.
* **Slope limiter:** β = 1.0.
* **Minimum allowed height:** 1 × 10\ :sup:`−5` m.
* **Cost:** ~3× relative to DE0.


Algorithm and GPU compatibility
--------------------------------

All four algorithms run on both CPU and GPU (OpenMP target offloading).  See
:doc:`../parallel/use_gpu_offloading` for a full compatibility table and
instructions on enabling GPU acceleration.


Reference
---------

.. automethod:: Domain.set_flow_algorithm
.. automethod:: Domain.get_flow_algorithm
