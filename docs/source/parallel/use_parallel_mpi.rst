.. _use_parallel_mpi:

MPI Parallelisation
===================

Check out the examples in `anuga_core/examples/parallel` such 
as :doc:`../../../../examples/parallel/run_parallel_rectangular.py`.

First you create an ordinary sequential domain on process 0  
and then run the distribute function: `domain = domain.distribute()` 
then you can merge all the parallel sww files at the end with a `domain.sww_merge()` command. 

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




