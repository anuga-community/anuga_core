.. currentmodule:: anuga


Parallelisation
===============

ANUGA uses two main approaches to parallelisation: MPI-based domain
decomposition (multiple processes, distributed memory) and OpenMP threading
(multiple threads, shared memory).  The two approaches can be combined.

An experimental OpenMP target-offloading GPU backend is included in the
``develop`` branch — see :ref:`use_gpu_offloading` for full details.

For long parallel runs, see :ref:`checkpointing` for how to save and restart
simulations from periodic checkpoints.

.. only:: html

.. toctree::
   :maxdepth: 1

   use_parallel_openmp
   use_gpu_offloading
   use_parallel_mpi
   use_distribute_basic_mesh



.. seealso::

   `ANUGA User Manual — Chapter 12: Parallel Simulation
   <https://github.com/anuga-community/anuga_user_manual>`_
   covers the MPI domain decomposition in depth, including ghost cell
   communication, scalability benchmarks, and HPC cluster setup.

.. only:: html
