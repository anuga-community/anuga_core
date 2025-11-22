.. _use_parallel_mpi:

.. currentmodule:: anuga

MPI Parallelisation
===================

Check out the examples in `anuga_core/examples/parallel` such 
as `run_parallel_rectangular.py` for a full example of how to set up and run an MPI parallel simulation.
ANUGA uses the `mpi4py` package to provide MPI parallelisation.

First you create an ordinary sequential domain on process 0  
and then run the distribute function: `domain = domain.distribute()` 
then you can merge all the parallel sww files at the end with a `domain.sww_merge()` command. 

A typical MPI parallel script looks like this:

.. code-block:: python

    from anuga import rectangular_cross_domain
    from anuga import distribute
    from anuga import finalize
    from anuga import myid

    # Create a sequential domain on process 0
    if myid == 0:
        domain = rectangular_cross_domain(1000, 1000, 10, 10)
        # Set up domain (e.g., set quantities)
    else:
        domain = None

    # Distribute the domain to all processes
    domain = distribute(domain)

    # Set up boundary conditions, (must be after distribute)
    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

    # Set up operators
    rate_operator = anuga.Rate_operator(domain, rate=0.01)

    # Evolve the domain in parallel
    domain.evolve(yieldstep=1.0, finaltime=10.0):
      if myid == 0:
         domain.print_timestepping_statistics()

    # Merge sww files on process 0
    domain.sww_merge()

    # Finalize MPI
    finalize()

To execute an MPI parallel run you need to use the mpiexec command. 
Suppose you script is named `run_model.py` and you want to use 8 MPI processes, 
then you need to run the command:

.. code-block:: bash

   mpiexec -np 8 python -u run_model.py

The `-u` flag is to ensure unbuffered output from the python script so that you can see 
print statements in real time.

I suggest experimenting with the optimal number of processes for you machine and your model. 
There is a limit to the number of processes that can be realistically used to speed up a simulation.
The number of processes should be chosen so that the size of the partitioned submeshes 
are no smaller that 1000-2000. Some experimentation is sensible. 




