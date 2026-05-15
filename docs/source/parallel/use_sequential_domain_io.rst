.. _use_sequential_domain_io:

.. currentmodule:: anuga

Offline domain partitioning (``sequential_distribute_dump`` / ``sequential_distribute_load``)
==============================================================================================

Overview
--------

The ``sequential_distribute_dump`` / ``sequential_distribute_load`` pair
implements an *offline domain partitioning* workflow (also called
**sdpl** — Sequential Dump, Parallel Load):

1. **Preprocessing** — run once, on any machine with enough RAM:

   .. code-block:: bash

      python create_dump.py -np N

   Rank 0 builds a complete :class:`Domain` (mesh + all quantities),
   partitions it into *N* subdomains, and writes one set of files per rank:
   a pickle file plus NumPy ``.npy`` arrays for the mesh and each quantity.

2. **Simulation** — run as many times as needed:

   .. code-block:: bash

      mpiexec -np N python -u run_evolve.py

   Each rank reads its own files independently, reconstructs a
   :class:`~anuga.parallel.parallel_shallow_water.Parallel_domain` with all
   quantities already loaded, and proceeds directly to ``domain.evolve()``.

This differs from the ``smpl`` (:ref:`use_sequential_mesh_io`) workflow,
which stores only the mesh topology (no quantities).


When to use this approach
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Situation
     - Recommendation
   * - Full domain + quantities fits in rank-0 RAM at runtime
     - :func:`distribute` (simpler, no preprocessing step)
   * - Quantities are expensive to recompute per rank (large DEM fits)
     - **This workflow** — dump once with quantities, reload many times
   * - Mesh fits in rank-0 RAM but you want MPI startup to be near-instant
     - **This workflow** — partition files are already split per rank
   * - Quantities come from a function / DEM, no need to store them
     - :ref:`use_sequential_mesh_io` (mesh-only, smaller files)
   * - Mesh is too large for rank-0 RAM at runtime but a preprocessing
       node has sufficient memory
     - :ref:`use_sequential_mesh_io` (mesh-only workflow)


Memory comparison
^^^^^^^^^^^^^^^^^

For a mesh with N triangles and P quantities:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Workflow
     - Rank-0 peak RAM
     - Required at runtime
   * - ``distribute``
     - O(N) × (mesh + P qty)
     - MPI job running
   * - ``sequential_distribute_dump`` / ``load``
     - O(N) × (mesh + P qty) (preprocessing)
     - Only per-rank partition
   * - ``sequential_mesh_dump`` / ``load``
     - O(N) × mesh only (preprocessing)
     - Only per-rank partition


API
---

.. autofunction:: sequential_distribute_dump

.. autofunction:: sequential_distribute_load


File format
-----------

For a domain named ``flood`` partitioned into *N* ranks, the preprocessing
step creates the following files per rank *p* in ``partition_dir``:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - File
     - Contents
   * - ``flood_P<N>_<p>.pickle``
     - Python pickle: mesh topology, boundary conditions, domain settings,
       and per-quantity filenames.
   * - ``flood_P<N>_<p>.pickle.np1.npy``
     - Node (x, y) coordinates as a NumPy ``float64`` array, shape (nnodes, 2).
   * - ``flood_P<N>_<p>.pickle.np2.npy``
     - Triangle connectivity as a NumPy ``int32`` array, shape (ntris, 3).
   * - ``flood_P<N>_<p>.pickle.np4.<qty>.npy``
     - One file per quantity (e.g. ``elevation``, ``stage``, ``friction``):
       centroid values as a NumPy ``float64`` array, shape (ntris,).

The pickle file stores domain metadata (name, flow algorithm,
geo_reference, ``store`` flag, etc.) so that ``sequential_distribute_load``
can reconstruct the domain identically to how it was configured during
preprocessing.


Preprocessing example
---------------------

.. code-block:: python

   # create_partitions.py  — run once; python create_partitions.py -np 64
   import argparse
   import anuga
   from anuga import rectangular_cross_domain

   parser = argparse.ArgumentParser()
   parser.add_argument('-np', '--numprocs', type=int, default=8)
   args = parser.parse_args()

   domain = rectangular_cross_domain(500, 500, len1=10.0, len2=10.0)
   domain.set_name('flood')
   domain.set_quantity('elevation', lambda x, y: 0.1 * x)
   domain.set_quantity('stage',     expression='elevation + 0.5')
   domain.set_quantity('friction',  0.03)
   domain.set_flow_algorithm('DE0')

   anuga.sequential_distribute_dump(
       domain,
       numprocs=args.numprocs,
       partition_dir='Partitions',
       verbose=True,
   )
   # Writes: Partitions/flood_P<N>_<rank>.pickle + .npy arrays


Parallel load-and-evolve example
---------------------------------

.. code-block:: python

   # run_evolve.py  — run with:  mpiexec -np N python -u run_evolve.py
   import anuga
   from anuga import myid, numprocs, finalize, barrier, Reflective_boundary

   barrier()
   domain = anuga.sequential_distribute_load(filename='flood',
                                             partition_dir='Partitions',
                                             verbose=(myid == 0))
   barrier()

   # Boundary conditions only — quantities were loaded from the partition files
   Br = Reflective_boundary(domain)
   domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

   for t in domain.evolve(yieldstep=60.0, finaltime=3600.0):
       if myid == 0:
           domain.print_timestepping_statistics()

   domain.sww_merge(delete_old=True)
   finalize()

Note that ``set_quantity`` calls are *not* needed in the load script — the
quantity arrays were already stored in the partition files during
preprocessing.


Combined dump-and-load (single MPI script)
-------------------------------------------

It is also possible to dump and load within the same MPI job.  Rank 0
builds the domain and dumps it; then all ranks load their partition:

.. code-block:: python

   # run_combined.py  — mpiexec -np N python -u run_combined.py
   import anuga
   from anuga import myid, numprocs, finalize, barrier
   from anuga import Reflective_boundary, rectangular_cross_domain
   from anuga import sequential_distribute_dump, sequential_distribute_load

   partition_dir = 'Partitions'
   domain_name   = 'flood'

   if myid == 0:
       domain = rectangular_cross_domain(500, 500, len1=10.0, len2=10.0)
       domain.set_name(domain_name)
       domain.set_quantity('elevation', lambda x, y: 0.1 * x)
       domain.set_quantity('stage',     expression='elevation + 0.5')
       domain.set_flow_algorithm('DE0')
       sequential_distribute_dump(domain, numprocs=numprocs,
                                  partition_dir=partition_dir)

   barrier()

   domain = sequential_distribute_load(filename=domain_name,
                                       partition_dir=partition_dir)

   Br = Reflective_boundary(domain)
   domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

   for t in domain.evolve(yieldstep=60.0, finaltime=3600.0):
       if myid == 0:
           domain.print_timestepping_statistics()

   domain.sww_merge(delete_old=True)
   finalize()

This pattern is useful for one-off runs on clusters where the domain +
quantities fit comfortably in rank-0 RAM but you want the startup cost
(partitioning + I/O) to happen outside the evolve loop timing.


Example scripts
---------------

Ready-to-run examples are in ``examples/parallel/``:

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - Script
     - Description
   * - ``run_sdpl_rectangular_create_partition_dump.py``
     - Creates a rectangular-cross domain (with quantities), partitions it,
       and writes partition files.
       Command line: ``python run_sdpl_rectangular_create_partition_dump.py -np N -sn 100``
   * - ``run_sdpl_rectangular_load_evolve.py``
     - Loads partition files and runs the evolve loop.
       Command line: ``mpiexec -np N python -u run_sdpl_rectangular_load_evolve.py -sn 100``
   * - ``run_sequential_dump_parallel_load.py``
     - Combined script: dump on rank 0, load on all ranks, then evolve.
       Command line: ``mpiexec -np N python -u run_sequential_dump_parallel_load.py``
