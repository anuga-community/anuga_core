.. currentmodule:: anuga


Setting up the Boundaries
=========================

Boundaries are an important part of the ANUGA model. They are used to set the
boundary conditions for the model.

To set up the boundaries you first need to create boundary objects. These boundary objects are then
assigned to the edges of the domain using the `set_boundary` method of the :class:`Domain` class.

For example, to set up reflective boundaries on all sides of a rectangular domain,
you would do the following:

.. code-block:: python

    from anuga import Domain, Reflective_boundary

    # Create a rectangular domain
    domain = Domain(...)

    # Create a reflective boundary object
    Br = Reflective_boundary(domain)

    # Set the boundaries of the domain
    domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

Standard Boundary Types
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Class
     - Description
   * - :class:`Reflective_boundary`
     - Returns the same conserved quantities as the neighbouring interior
       triangle but with the normal momentum negated, so that no mass crosses
       the boundary (a "wall").  Suitable for closed edges such as coastlines
       and levees.
   * - :class:`Dirichlet_boundary`
     - Holds stage and momenta at fixed, constant values for the full
       simulation.  Useful for specifying a steady inflow or maintaining a
       constant water level on an open boundary.
   * - :class:`Time_boundary`
     - Like :class:`Dirichlet_boundary` but the conserved quantities are
       specified as a Python function of simulation time *t*.  Use this when
       you want a time-varying stage or discharge that you can express as a
       formula (e.g. a tide signal or a gate opening schedule).
   * - :class:`Transmissive_n_momentum_zero_t_momentum_set_stage_boundary`
     - Sets the stage from a user-supplied function of time, transmits the
       normal momentum from the adjacent interior cell, and zeros the
       tangential momentum.  Approximates a weakly-reflective open boundary
       where the outgoing signal can leave with minimal reflection.
   * - :class:`Flather_external_stage_zero_velocity_boundary`
     - Implements a Flather-type radiation condition (Blayo & Debreu 2005):
       the external stage is set by a function of time and the external
       velocity is zero, but the boundary flux is blended with the interior
       state using characteristic-like variables.  Useful as a weakly
       reflecting open-ocean boundary where the stage should be approximately
       specified but outgoing waves are allowed to leave.
   * - :class:`File_boundary`
     - Reads stage and momentum time series from an SWW file and interpolates
       them spatially to each boundary midpoint and linearly in time.  Used
       to nest a fine-resolution domain inside a coarser simulation.
   * - :class:`Field_boundary`
     - A thin wrapper around :class:`File_boundary` that additionally applies
       a ``mean_stage`` offset to the stage read from the SWW file.  Useful
       when you want to re-use one boundary SWW file across multiple tide
       scenarios without regenerating the file.


Usage examples
--------------

**Reflective boundary (closed wall)**

.. code-block:: python

    import anuga

    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

**Dirichlet boundary (fixed values)**

.. code-block:: python

    import anuga

    # stage = 0.5 m, xmomentum = 0, ymomentum = 0
    Bd = anuga.Dirichlet_boundary([0.5, 0.0, 0.0])
    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})

**Time boundary (time-varying stage)**

.. code-block:: python

    import anuga
    import math

    def tide(t):
        """Sinusoidal tide with 12-hour period and 1 m amplitude."""
        return [math.sin(2 * math.pi * t / 43200.0), 0.0, 0.0]

    Bt = anuga.Time_boundary(domain, function=tide)
    domain.set_boundary({'ocean': Bt, 'land': anuga.Reflective_boundary(domain)})

**File boundary (nesting from an SWW file)**

.. code-block:: python

    import anuga

    Bf = anuga.File_boundary('coarse_run.sww', domain)
    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({'ocean': Bf, 'land': Br})

**Field boundary (SWW file with tide offset)**

.. code-block:: python

    import anuga

    # Reuse an SWW file generated at mean sea level; add 0.8 m for high tide
    Bff = anuga.Field_boundary('boundary_msl.sww', domain, mean_stage=0.8)
    domain.set_boundary({'ocean': Bff, 'land': anuga.Reflective_boundary(domain)})

**Flather boundary (weakly reflecting open ocean)**

.. code-block:: python

    import anuga

    sea_level = 0.0

    def waveform(t):
        return sea_level + 0.5 / math.cosh(t - 25.0) ** 2

    Bfl = anuga.Flather_external_stage_zero_velocity_boundary(domain, waveform)
    domain.set_boundary({'ocean': Bfl, 'land': anuga.Reflective_boundary(domain)})


.. seealso::

   `ANUGA User Manual — Chapter 9: Boundary Conditions and set_boundary
   <https://github.com/anuga-community/anuga_user_manual>`_
   gives extended examples of each boundary type, discusses time-varying
   stage specifications in detail, and explains how to diagnose common
   boundary-tag errors.

Reference
---------

.. autoclass:: Reflective_boundary
.. autoclass:: Dirichlet_boundary
.. autoclass:: Time_boundary
.. autoclass:: Transmissive_n_momentum_zero_t_momentum_set_stage_boundary
.. autoclass:: Flather_external_stage_zero_velocity_boundary
.. autoclass:: File_boundary
.. autoclass:: Field_boundary
