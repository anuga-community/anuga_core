.. _use_parallel_openmp:

.. currentmodule:: anuga

OpenMP Parallelisation
======================

By default ANUGA (from version 3.2) can use multiple threads to speed up
certain computationally intensive tasks (mostly within the `evolve` loop). This is done using OpenMP, 
a widely used API for shared-memory parallel programming in C, C++, and Fortran.

`ANUGA` (version 3.2 and later) from `conda-forge` is compiled with OpenMP support enabled on 
platforms that support it (windows, linux and macOS). This means that if you install ANUGA via conda,
you should be able to take advantage of OpenMP parallelisation without any additional configuration.

Controlling the Number of Threads
---------------------------------

To control the number of threads used by OpenMP, you can set the `OMP_NUM_THREADS` environment variable.
For example, in a Unix-like terminal, you can set it as follows before running your ANUGA script:

.. code-block:: bash

    export OMP_NUM_THREADS=4 # Replace 4 with the desired number of threads

    python your_anuga_script.py

The `export` can be added to your shell profile (e.g., `.bashrc`, `.zshrc`) 
to make it persistent across sessions.

You can also explicitly set the number of threads in your Python script
using the `set_omp_num_threads` method of the `Domain` class.

.. code-block:: python

    from anuga import Domain

    domain = Domain(...)
    domain.set_omp_num_threads(4)  # Replace 4 with the desired number of threads

Keep in mind that the optimal number of threads may depend on your specific hardware and 
the nature of your simulation.

Installing ANUGA with OpenMP Support
-------------------------------------

If you install ANUGA from source and want to enable OpenMP support, you need to have a C compiler 
that supports OpenMP (e.g., GCC). 
When installing ANUGA from source, ensure that the compiler flags for OpenMP are included. 
For example, with GCC, you would typically add the `-fopenmp` flag.


 See Also
--------

.. autosummary::
   :toctree:

   Domain.set_omp_num_threads   

