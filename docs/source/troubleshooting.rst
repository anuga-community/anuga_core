.. currentmodule:: anuga

Troubleshooting
===============

This page covers the most common errors encountered when installing or running
ANUGA, along with their causes and fixes.

.. contents:: Contents
   :local:
   :depth: 1


MPI not available / parallel runs fall back to serial
------------------------------------------------------

**Symptom**

Running with ``mpiexec -np 4`` produces output as if only one process is
running, or you see::

    WARNING: mpi4py not available. Running in serial mode.

**Cause**

``mpi4py`` is not installed, or its underlying MPI library cannot be found
at runtime.

**Fix**

Install ``mpi4py`` into your conda environment::

    conda install -c conda-forge mpi4py

Verify the installation::

    python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"

When launched with ``mpiexec -np 4`` the above should print ``4``.

If you are on an HPC cluster, make sure you load the correct MPI module
before activating your conda environment, so that ``mpi4py`` links against
the system MPI rather than a bundled one::

    module load openmpi/4.1.5     # example — use your cluster's module name
    conda activate anuga_env
    mpiexec -np 4 python my_script.py

Check which MPI ``mpi4py`` has found::

    python -c "import mpi4py; print(mpi4py.get_config())"

**Parallel API when mpi4py is absent**

ANUGA's parallel functions (``distribute``, ``distribute_basic_mesh``, etc.)
fall back to a serial no-op when ``mpi4py`` is not available.  The same
script therefore runs serially without modification — useful for testing, but
not for production parallel runs.


OpenMP threading not activating
---------------------------------

**Symptom**

Setting ``OMP_NUM_THREADS=8`` or calling ``domain.set_omp_num_threads(8)``
has no effect — the simulation uses only one CPU core.

**Cause — build not compiled with OpenMP**

ANUGA's Cython extensions must be compiled with OpenMP support.  On Linux
this requires GCC with ``-fopenmp``; on macOS it requires a GCC or
clang+libomp toolchain.  A pre-built conda-forge package should already
include OpenMP, but a manual build may not.

**Check**::

    python -c "
    import anuga
    domain = anuga.rectangular_cross_domain(10, 10)
    domain.set_omp_num_threads(4, verbose=True)
    "

If the extension loaded correctly you will see::

    Setting omp_num_threads to 4

If you instead see an ``ImportError`` referencing ``sw_domain_openmp_ext``,
the OpenMP extension was not compiled.

**Fix — conda install (recommended)**::

    conda install -c conda-forge anuga

**Fix — rebuild from source with OpenMP**

On Linux::

    conda activate anuga_env
    pip install --no-build-isolation -v .

On macOS, install ``libomp`` first::

    brew install libomp
    pip install --no-build-isolation -v .

**Cause — OMP_NUM_THREADS not propagated to subprocesses**

When running under ``mpiexec``, some MPI launchers do not forward
environment variables.  Set the variable explicitly in the launch command::

    OMP_NUM_THREADS=4 mpiexec -np 2 python my_script.py

Or set it inside the script before creating the domain::

    import os
    os.environ['OMP_NUM_THREADS'] = '4'
    import anuga


Rank-0 memory exhaustion on large meshes
------------------------------------------

**Symptom**

A ``MemoryError`` or the system swap fills up when building or distributing
a large mesh, typically in ``distribute()`` or while calling
``create_domain_from_regions()`` / ``rectangular_cross_domain()``.

**Cause**

The standard ``distribute()`` workflow builds the full mesh **and** all
quantity arrays on rank 0 before any MPI communication takes place.  For a
mesh of N triangles and P ranks, rank 0 must hold O(N) data — the same
amount as a serial run — before the submeshes are sent out.

**Fix — use the mesh-first workflow**

:func:`distribute_basic_mesh` distributes topology only; quantities are set
per-rank after distribution, so rank 0 never holds the full quantity arrays:

.. code-block:: python

   from anuga.abstract_2d_finite_volumes.basic_mesh import \
       rectangular_cross_basic_mesh
   from anuga.parallel.parallel_api import distribute_basic_mesh

   if anuga.myid == 0:
       bm = rectangular_cross_basic_mesh(1000, 1000, len1=10.0, len2=10.0)
   else:
       bm = None

   domain = distribute_basic_mesh(bm)

   # Set quantities per-rank — rank 0 never allocates the full arrays
   domain.set_quantity('elevation', lambda x, y: -x / 10.0)
   domain.set_quantity('stage', 0.0)

For meshes where even rank-0 topology is a concern, use
:func:`distribute_basic_mesh_collaborative`, which broadcasts topology via
shared memory (one copy per node rather than one per rank).

See :ref:`use_distribute_basic_mesh` for full details.

**Fix — use ``in_place=True`` with distribute()**

If you must use ``distribute()``, pass ``in_place=True`` so that the full
domain object on rank 0 is freed as each submesh is sent::

    domain = distribute(domain, in_place=True)


SWW output errors
------------------

**``ValueError: cannot reshape array of size N into shape (1, M)``**

This error occurs when two MPI ranks write to the same SWW file.  It is
almost always caused by all ranks using the default domain name ``'domain'``
(i.e. ``domain.set_name()`` was never called), so every rank writes to
``domain.sww`` and they collide.

**Fix**

Call ``domain.set_name()`` before the evolve loop::

    domain.set_name('my_simulation')

For parallel runs ``Parallel_domain`` automatically appends
``_P<nproc>_<rank>`` to the name, producing separate files
``my_simulation_P4_0.sww``, ``my_simulation_P4_1.sww``, etc.  Merge them
afterwards with::

    domain.sww_merge(delete_old=True)

**SWW file grows unexpectedly large**

Each ``yieldstep`` adds a time slice to the SWW file.  Reduce file size by:

- Increasing ``yieldstep`` (write output less frequently)
- Using ``outputstep`` to decouple the yield interval from the output
  interval — see :ref:`evolve`
- Storing fewer quantities::

    domain.set_quantities_to_be_stored({
        'stage': 2,
        'elevation': 1,
    })

- Suppressing output entirely::

    domain.set_quantities_to_be_stored(None)


Boundary tag errors
--------------------

**``Exception: Tag "X" provided does not exist in the domain``**

The tag name passed to ``set_boundary()`` does not match any tag in the
mesh.  Check the exact tag names the mesh defines::

    print(domain.get_boundary_tags())

Common causes:

- Typo in the tag name (``'Left'`` vs ``'left'``)
- Using tags from a different mesh file
- For rectangular domains the built-in tags are exactly
  ``'left'``, ``'right'``, ``'top'``, ``'bottom'``

**``Exception: ERROR (domain.py): Tag "X" has not been bound to a boundary object``**

Every tag that exists in the mesh *must* appear in the ``set_boundary()``
call.  If a tag is present in the mesh but not in your boundary map, ANUGA
raises this error.

**Fix** — assign a boundary to every tag, using ``Reflective_boundary`` as a
default for any tag you do not want to treat specially::

    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({
        'left':   anuga.Dirichlet_boundary([0.5, 0.0, 0.0]),
        'right':  Br,
        'top':    Br,
        'bottom': Br,
    })


Mesh file errors
-----------------

**``IOError: File X could not be opened``**

ANUGA could not read the mesh file.  Common causes:

- The file does not exist at the given path.  Check with an absolute path::

    import os
    print(os.path.exists('my_mesh.tsh'))

- The file extension is not ``.tsh`` or ``.msh``.  Only these two extensions
  are recognised by ``import_mesh_file``/``create_domain_from_file``.

- The ``.tsh`` file is malformed (e.g. truncated, wrong encoding).  Open it
  in a text editor to check its structure.

**``IOError: Extension .xyz is unknown``**

``create_domain_from_file`` only accepts ``.tsh`` and ``.msh`` files.  To
load elevation from a raster, convert it first::

    anuga.asc2dem('elevation.asc')
    anuga.dem2pts('elevation.dem')
    domain.set_quantity('elevation', filename='elevation.pts')


Installation and import errors
--------------------------------

**``ModuleNotFoundError: No module named 'anuga'``**

The conda environment containing ANUGA is not active, or ANUGA has not been
installed.  Activate the environment and verify::

    conda activate anuga_env
    python -c "import anuga; print(anuga.__version__)"

**``ImportError`` referencing a Cython extension (e.g. ``_quantity_ext``)``**

The Cython/C extensions have not been compiled, or were compiled for a
different Python version.  Reinstall::

    pip install --no-build-isolation -v .

**Slow import / ``anuga`` takes many seconds to import**

This is usually caused by ``matplotlib`` being imported at module load time
on a system with a slow display backend.  Set a non-interactive backend
before importing::

    import matplotlib
    matplotlib.use('Agg')
    import anuga

Or set it in your environment::

    export MPLBACKEND=Agg


Performance issues
-------------------

**The simulation is slower than expected**

- Check ``OMP_NUM_THREADS``.  By default ANUGA uses one thread.  Set it to
  the number of physical cores on your machine::

    OMP_NUM_THREADS=8 python my_script.py

- Check the number of triangles per MPI rank.  Below roughly 1000–2000
  triangles per rank, MPI communication overhead outweighs the speedup.
  Use fewer ranks or a larger mesh.

- Check the flow algorithm.  ``DE1`` is generally faster than ``DE0`` for
  large meshes with many wet/dry transitions::

    domain.set_flow_algorithm('DE1')

- Use ``domain.print_timestepping_statistics()`` inside the evolve loop to
  see the timestep distribution — very small timesteps indicate a CFL
  constraint from a locally steep gradient or a very fine mesh region.


Getting further help
---------------------

If your issue is not covered here:

- Check the `GitHub issue tracker
  <https://github.com/anuga-community/anuga_core/issues>`_ for similar
  reports.
- Open a new issue with a minimal reproducible example, the full traceback,
  your ANUGA version (``python -c "import anuga; print(anuga.__version__)"``),
  and your platform (OS, Python version, MPI library).
