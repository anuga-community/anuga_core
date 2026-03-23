.. currentmodule:: anuga


Parallelisation
===============

ANUGA uses two main approaches to parallelisation: MPI-based domain
decomposition (multiple processes, distributed memory) and OpenMP threading
(multiple threads, shared memory).  The two approaches can be combined.

An experimental OpenMP target-offloading GPU backend is under development in
the ``sp26`` branch — see :ref:`use_parallel_openmp` for details.

For long parallel runs, see :ref:`checkpointing` for how to save and restart
simulations from periodic checkpoints.

.. only:: html

.. toctree::
   :maxdepth: 1

   use_parallel_openmp
   use_parallel_mpi
   use_distribute_basic_mesh


Choosing an MPI distribution strategy
--------------------------------------

Four functions are available for distributing a domain across MPI ranks.
The right choice depends on mesh size and how you construct the domain.

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Function
     - Rank-0 peak memory
     - When to use
   * - :func:`distribute`
     - Full domain + quantities
     - Default choice.  Rank 0 builds the complete ``Domain`` (mesh +
       all quantities), then sends submeshes to other ranks.  Simple
       drop-in for serial scripts.
   * - :func:`distribute_collaborative`
     - Full domain + quantities (shared)
     - Same interface as ``distribute`` but rank 0 broadcasts topology
       via MPI shared memory (one copy per node instead of one per rank).
       Use when several ranks share a node and rank-0 memory is tight.
   * - :func:`distribute_basic_mesh`
     - Topology only
     - Rank 0 builds a lightweight :class:`Basic_mesh` (no quantity
       arrays), distributes topology, then every rank sets its own
       quantities.  Best when quantity arrays would exhaust rank-0 memory.
   * - :func:`distribute_basic_mesh_collaborative`
     - Topology only (shared)
     - Like ``distribute_basic_mesh`` but topology is broadcast via shared
       memory.  Use for the largest meshes where even per-rank copies of
       the topology are expensive.

**Decision guide**

.. code-block:: text

    Does rank 0 have enough RAM to hold the full Domain (mesh + quantities)?
    │
    ├─ Yes ──► Do multiple ranks share the same node?
    │           ├─ No  ──► distribute()
    │           └─ Yes ──► distribute_collaborative()
    │
    └─ No  ──► Does rank 0 have enough RAM for topology only (no quantities)?
                ├─ Yes ──► Do multiple ranks share the same node?
                │           ├─ No  ──► distribute_basic_mesh()
                │           └─ Yes ──► distribute_basic_mesh_collaborative()
                └─ No  ──► Reduce mesh resolution or use more nodes

As a rough guide, a mesh of N triangles with P quantities requires
approximately ``8 × N × P`` bytes for quantity arrays alone on rank 0
(double precision).  Topology (coordinates + connectivity) is
``~56 × N`` bytes.

.. seealso::

   `ANUGA User Manual — Chapter 12: Parallel Simulation
   <https://github.com/anuga-community/anuga_user_manual>`_
   covers the MPI domain decomposition in depth, including ghost cell
   communication, scalability benchmarks, and HPC cluster setup.

.. only:: html
