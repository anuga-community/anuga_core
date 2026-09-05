

.. currentmodule:: anuga

===================
ANUGA documentation
===================

ANUGA (pronounced "AHnooGAH") is open-source software for the simulation of the 
shallow water equation, 
in particular it can be used to model tsunamis and floods.

ANUGA is a Python 3 package (Python 3.10 – 3.14) with some C and Cython
extensions.

ANUGA was created in a collaboration by Geoscience Australia 
and Mathematical Sciences Institute at the Australian National University. 
It is now developed and maintained by a community of volunteers.

.. note::

   The main documentation (**Contents**) covers the standard workflow most users
   need: install ANUGA, try an example, write a script (or use the TOML scenario
   interface), run it, view the results, and parallelise if needed. The
   **Appendices** hold advanced topics — the developer and GPU installs, advanced
   parallel methods, compute modes and GPU offloading, and advanced scripting.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   background
   installation/index
   examples/index
   setup_anuga_script/index
   toml_scenario/index
   visualisation/index
   parallel/index
   reference/index
   glossary
   troubleshooting
   citing
   genindex

.. toctree::
   :maxdepth: 2
   :caption: Appendices:

   installation/install_anuga_developers
   Installing for GPU <appendices/install_gpu>
   appendices/advanced_script
   parallel/advanced
   Sediment physics: choosing the laws <appendices/sediment_physics>
   Compute modes: legacy vs unified <appendices/compute_modes>
   appendices/profiling_gpu
   mathematical_background
   appendices/contributing


   



