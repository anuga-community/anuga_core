.. _use_parallel_openmp:

.. currentmodule:: anuga

OpenMP Parallelisation
======================

From version 3.2, ANUGA can use multiple CPU threads to speed up the
computationally intensive parts of the evolve loop (flux computation, friction,
and momentum updates).  This is done using
`OpenMP <https://www.openmp.org/>`_, a widely-used API for shared-memory
parallel programming in C, C++, and Fortran.

By default, ANUGA uses **one thread**.  Setting ``OMP_NUM_THREADS`` or calling
``domain.set_omp_num_threads()`` enables multi-threading without any other
changes to your script.


Checking your ANUGA version
-----------------------------

.. code-block:: python

    import anuga
    print(anuga.__version__)   # should be 3.2 or later for OpenMP support


Controlling the number of threads
----------------------------------

**Environment variable (recommended)**

Set ``OMP_NUM_THREADS`` before launching Python:

.. code-block:: bash

    export OMP_NUM_THREADS=4
    python my_anuga_script.py

You can add this line to your shell profile (``~/.bashrc``, ``~/.zshrc``)
to make it permanent, or set it inline for a single run:

.. code-block:: bash

    OMP_NUM_THREADS=8 python my_anuga_script.py

**Inside the script**

Call ``domain.set_omp_num_threads()`` after creating the domain and before
the evolve loop:

.. code-block:: python

    import anuga

    domain = anuga.rectangular_cross_domain(200, 200)
    domain.set_omp_num_threads(4)   # use 4 threads

    for t in domain.evolve(yieldstep=1.0, duration=100.0):
        domain.print_timestepping_statistics()

The method reads ``OMP_NUM_THREADS`` from the environment if no argument is
passed, and defaults to 1 if neither is set.

.. note::

    The optimal thread count depends on your hardware.  A good starting point
    is the number of physical cores on the machine.  Hyper-threaded (logical)
    cores typically give little or no speedup for ANUGA's memory-bound loops.


If the build was compiled without OpenMP
-----------------------------------------

ANUGA's multi-threading is provided by the compiled extension
``sw_domain_openmp_ext``.  If this extension was not built with OpenMP
support (or was not built at all), calling ``domain.set_omp_num_threads()``
will raise an ``ImportError``::

    ImportError: cannot import name 'set_omp_num_threads'
                 from 'anuga.shallow_water.sw_domain_openmp_ext'

In this case the simulation still runs correctly — it simply uses a single
thread.

**How to check**

.. code-block:: python

    import anuga
    domain = anuga.rectangular_cross_domain(10, 10)
    domain.set_omp_num_threads(4, verbose=True)

If OpenMP is active you will see::

    Setting omp_num_threads to 4

If you see an ``ImportError``, the OpenMP extension is missing or was built
without OpenMP.


Installing ANUGA with OpenMP support
--------------------------------------

**conda-forge (recommended)**

Pre-built conda-forge packages are compiled with OpenMP enabled on Linux,
macOS, and Windows:

.. code-block:: bash

    conda install -c conda-forge anuga

**Building from source on Linux**

GCC includes OpenMP by default.  The standard install command is sufficient:

.. code-block:: bash

    conda activate anuga_env_3.12
    pip install --no-build-isolation -v .

**Building from source on macOS**

Apple's Clang does not include OpenMP.  Install ``libomp`` via Homebrew
first:

.. code-block:: bash

    brew install libomp
    conda activate anuga_env_3.12
    pip install --no-build-isolation -v .

**Building from source on Windows**

Use the conda-forge MinGW compilers, which include OpenMP:

.. code-block:: bash

    conda install -c conda-forge m2w64-gcc
    pip install --no-build-isolation -v .


OpenMP with MPI
----------------

OpenMP and MPI can be used together (hybrid parallelism).  Each MPI rank
runs its own OpenMP thread pool independently.  A typical hybrid launch
using 4 MPI ranks with 4 OpenMP threads each (16 cores total):

.. code-block:: bash

    OMP_NUM_THREADS=4 mpiexec -np 4 python my_parallel_script.py

.. note::

    Some MPI launchers do not forward environment variables to child
    processes.  If threads do not activate under ``mpiexec``, set
    ``OMP_NUM_THREADS`` explicitly inside the script before creating the
    domain::

        import os
        os.environ['OMP_NUM_THREADS'] = '4'
        import anuga


GPU acceleration (experimental)
--------------------------------

An experimental GPU backend is under active development in the ``sp26``
branch.  It uses **OpenMP target offloading** to run the flux and friction
kernels on a GPU without requiring Python-level changes to your script.

To try it, check out the ``sp26`` branch and build with a compiler that
supports OpenMP offloading (e.g. GCC 12+ with offload targets, or LLVM with
``libomptarget``):

.. code-block:: bash

   git checkout sp26
   pip install --no-build-isolation -v .

Then set ``multiprocessor_mode = 2`` in your TOML file or via:

.. code-block:: python

   domain.set_multiprocessor_mode(2)

.. note::

   The GPU backend in ``sp26`` is experimental and subject to change without
   notice.  It has been tested on NVIDIA GPUs with GCC offload support.
   For production runs, use the standard OpenMP CPU backend
   (``multiprocessor_mode = 1``).


Intel Cascade Lake / NCI Gadi tuning
--------------------------------------

The following notes apply to **Intel Xeon Scalable 2nd-generation (Cascade Lake)**
nodes, such as those on `NCI Gadi <https://opus.nci.org.au/display/Help/Gadi+User+Guide>`_.
These nodes have 28 physical cores per socket (56 with Hyper-Threading), 6-channel
DDR4-2933 (~45 GB/s per socket), and AVX-512 SIMD units.

Because all ANUGA kernels are **memory-bandwidth-bound** (arithmetic intensity
< 2 FLOP/byte), the goal is to maximise memory throughput and avoid wasting
bandwidth on thread synchronisation.

**Thread count and affinity (Intel OpenMP runtime)**

.. code-block:: bash

   # Single-socket run — maximise L3 cache sharing, no NUMA crossing
   export OMP_NUM_THREADS=28
   export KMP_AFFINITY=granularity=core,compact,1,0

   # Full-node run — spread threads across both sockets for 2× memory BW
   export OMP_NUM_THREADS=56
   export KMP_AFFINITY=granularity=core,balanced

``KMP_AFFINITY`` is an Intel OpenMP runtime variable.  The portable OpenMP 4.5
equivalent is::

   export OMP_PROC_BIND=close
   export OMP_PLACES=cores

**Spin-wait policy**

Between parallel regions ANUGA performs Python-level work (I/O, boundary
evaluation).  By default Intel OpenMP threads busy-wait, consuming CPU cycles
during this Python gap.  Setting ``KMP_BLOCKTIME=0`` makes threads yield
immediately; ``KMP_BLOCKTIME=1`` (1 ms) balances wake latency vs. idle cost:

.. code-block:: bash

   export KMP_BLOCKTIME=1     # 1 ms – good balance for most ANUGA timestep sizes

**Building with AVX-512 and IPO (Intel compiler)**

The ``environments/environment_<version>_intel.yml`` conda environment sets up
Intel oneAPI (``icx`` / ``icpx``) with MKL.  Configure with:

.. code-block:: bash

   conda activate anuga_env_3.12_intel
   pip install --no-build-isolation -v .

This build already enables ``-xCORE-AVX512 -qopt-zmm-usage=high -ipo`` and
``-fimf-use-svml=true`` (Intel SVML for vectorised ``cbrt``/``sqrt`` in the
Manning friction loop), delivering up to **2–4× speedup** over the default
SSE2 binary on Cascade Lake.

To additionally tune for the exact host microarchitecture (e.g. ``-xHost``
equivalent of ``-march=native``), add ``-Dmarch_native=true`` at configure
time:

.. code-block:: bash

   pip install --no-build-isolation -v . -- -Dmarch_native=true

.. warning::

   Binaries built with ``-Dmarch_native=true`` or ``-xHost`` may not run on
   CPUs older than the build node.

**MKL for the CG solver**

When MKL is detected at build time, the Conjugate Gradient solver used during
mesh fitting automatically uses MKL ``cblas_ddot``, ``cblas_daxpy``, and
``cblas_dscal`` instead of hand-rolled OpenMP loops.  These MKL routines are
cache-tiled and AVX-512 vectorised, typically giving **2–4× speedup** for the
fitting stage on Cascade Lake.

**Example Gadi PBS job script**

.. code-block:: bash

   #!/bin/bash
   #PBS -l ncpus=28
   #PBS -l mem=192GB
   #PBS -l walltime=02:00:00

   module load intel-compiler
   module load intel-mkl

   export OMP_NUM_THREADS=28
   export KMP_AFFINITY=granularity=core,compact,1,0
   export KMP_BLOCKTIME=1
   export MKL_NUM_THREADS=1       # MKL used only for CBLAS (sequential)

   conda activate anuga_env_3.12_intel
   python my_anuga_script.py


See Also
---------

.. autosummary::
   :toctree:

   Domain.set_omp_num_threads
