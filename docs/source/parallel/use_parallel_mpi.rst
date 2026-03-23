.. _use_parallel_mpi:

.. currentmodule:: anuga

MPI Parallelisation
===================

ANUGA uses the `mpi4py` package for MPI-based parallelisation.  Three
distribution functions are available; they differ in how much memory and
work rank 0 must do before MPI communication begins.

.. list-table:: Distribution function comparison
   :header-rows: 1
   :widths: 28 24 24 24

   * - Feature
     - ``distribute``
     - ``distribute_collaborative``
     - ``distribute_basic_mesh`` / ``distribute_basic_mesh_collaborative``
   * - Mesh built on rank 0
     - Full ``Domain`` with quantities
     - Full ``Domain`` with quantities
     - ``Basic_mesh`` only (no quantities)
   * - Quantities set
     - On rank 0, then distributed
     - On rank 0, then distributed
     - Per-rank after distribution
   * - Topology broadcast
     - Point-to-point per rank
     - Shared memory per node
     - Point-to-point / shared memory
   * - Peak rank-0 memory
     - O(N) mesh + O(N) quantities
     - O(N) mesh + O(N) quantities
     - O(N) mesh only
   * - Best for
     - Small–medium meshes, simple scripts
     - Many ranks per node, large meshes
     - Very large meshes; quantities set from functions


``distribute``
--------------

The simplest workflow.  Rank 0 builds a full ``Domain`` (mesh *and*
quantities), then ``distribute`` partitions it and sends each rank its
submesh.  All ranks can set boundary conditions and operators after the
call.

.. code-block:: python

   import anuga

   if anuga.myid == 0:
       domain = anuga.rectangular_cross_domain(1000, 1000, len1=10.0, len2=10.0)
       domain.set_quantity('elevation', lambda x, y: x / 10.0)
       domain.set_quantity('stage', expression='elevation + 0.2')
   else:
       domain = None

   # Rank 0 partitions; all ranks receive their submesh
   domain = anuga.distribute(domain)

   Br = anuga.Reflective_boundary(domain)
   domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

   rain = anuga.Rate_operator(domain, rate=lambda t: 0.001, factor=1.0)

   for t in domain.evolve(yieldstep=1.0, finaltime=10.0):
       if anuga.myid == 0:
           domain.print_timestepping_statistics()

   domain.sww_merge()
   anuga.finalize()

The main limitation is that rank 0 must hold the complete mesh **and**
all quantity arrays before any MPI communication starts.  For very large
meshes this can exhaust available memory on rank 0.


``distribute_collaborative``
-----------------------------

A drop-in replacement for ``distribute`` that reduces peak memory use and
inter-rank data movement.  The API is identical — rank 0 still builds a
full ``Domain`` with quantities set — but the distribution is done
differently:

1. Rank 0 partitions the mesh (METIS / Morton / Hilbert).
2. The full mesh topology is broadcast using ``MPI.Win.Allocate_shared``:
   one physical copy per compute node, shared by all ranks on that node.
   Between nodes a single ``Bcast`` among node leaders transfers the data.
3. Each rank independently builds its own ghost layer using a Cython BFS
   kernel rather than receiving it from rank 0.
4. Quantities are distributed with ``MPI_Scatterv`` (each rank receives
   only its O(N/P) full-triangle rows) plus targeted ``MPI_Isend`` for
   ghost triangles.

The result is that topology is stored only once per node (not once per
rank), and quantity communication scales as O(N/P) rather than O(N).

.. code-block:: python

   import anuga
   from anuga.parallel.parallel_api import distribute_collaborative

   if anuga.myid == 0:
       domain = anuga.rectangular_cross_domain(1000, 1000, len1=10.0, len2=10.0)
       domain.set_quantity('elevation', lambda x, y: x / 10.0)
       domain.set_quantity('stage', expression='elevation + 0.2')
   else:
       domain = None

   # Shared-memory broadcast + per-rank BFS ghost layer
   domain = distribute_collaborative(domain)

   Br = anuga.Reflective_boundary(domain)
   domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

   for t in domain.evolve(yieldstep=1.0, finaltime=10.0):
       if anuga.myid == 0:
           domain.print_timestepping_statistics()

   domain.sww_merge()
   anuga.finalize()

Use ``distribute_collaborative`` when you have many MPI ranks sharing the
same node (e.g. 32–64 ranks on a large-memory node) or when the mesh is
large enough that topology duplication per rank becomes a concern.

.. note::

   ``distribute_collaborative`` still requires rank 0 to allocate the full
   quantity arrays before distribution.  If that alone exceeds available
   memory, use ``distribute_basic_mesh`` or
   ``distribute_basic_mesh_collaborative`` instead.


Mesh-first workflows
---------------------

For very large meshes where even allocating the full quantity arrays on
rank 0 is impractical, use the *mesh-first* functions described in
:ref:`use_distribute_basic_mesh`.  These functions accept a ``Basic_mesh``
(topology only, no quantities) on rank 0; every rank then sets its own
initial conditions independently after distribution.

.. code-block:: python

   import anuga
   from anuga.abstract_2d_finite_volumes.basic_mesh import rectangular_cross_basic_mesh
   from anuga.parallel.parallel_api import distribute_basic_mesh

   if anuga.myid == 0:
       bm = rectangular_cross_basic_mesh(1000, 1000, len1=10.0, len2=10.0)
   else:
       bm = None

   # Only mesh topology is distributed; quantities are never on rank 0
   domain = distribute_basic_mesh(bm)

   domain.set_quantity('elevation', lambda x, y: x / 10.0)
   domain.set_quantity('stage', expression='elevation + 0.2')

   Br = anuga.Reflective_boundary(domain)
   domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

   for t in domain.evolve(yieldstep=1.0, finaltime=10.0):
       if anuga.myid == 0:
           domain.print_timestepping_statistics()

   domain.sww_merge()
   anuga.finalize()

See :ref:`use_distribute_basic_mesh` for full details, including the
shared-memory variant ``distribute_basic_mesh_collaborative``.


Running parallel scripts
-------------------------

Use ``mpiexec`` (or ``mpirun``) to launch the script.  For example, to
run on 8 MPI processes::

   mpiexec -np 8 python -u run_model.py

The ``-u`` flag gives unbuffered output so that ``print`` statements from
all ranks appear in real time.

Example scripts are in ``examples/parallel/``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Script
     - Demonstrates
   * - ``run_parallel_rectangular.py``
     - ``distribute`` with a rectangular domain
   * - ``run_parallel_merimbula.py``
     - ``distribute`` with a real DEM and polygon mesh
   * - ``run_parallel_basic_mesh.py``
     - ``distribute_basic_mesh``
   * - ``run_parallel_basic_mesh_collaborative.py``
     - ``distribute_basic_mesh_collaborative``


Choosing the number of processes
----------------------------------

There is a practical limit to how many MPI ranks benefit a simulation.
As a rough guide, aim for at least 1000–2000 triangles per rank after
partitioning.  Below that threshold, the overhead of ghost-cell
communication and MPI synchronisation outweighs the computational
saving.  Some experimentation for your specific mesh and hardware is
always worthwhile.
