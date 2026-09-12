.. currentmodule:: anuga


Visualisation
=============

ANUGA provides several tools for visualising simulation results. During a
simulation run, the ``domain`` object can generate plots of the current state via
:ref:`Domain_plotter <use_domain_plotter>`. After the simulation, the ``.sww``
output is most quickly viewed — especially for large files — with the interactive
:ref:`ANUGA Viewer <use_anuga_viewer>`. Results can also be opened in the
``anuga_sww_gui`` GUI, explored programmatically via ``anuga.SWW_plotter``, or
exported to GIS such as QGIS.

.. only:: html

.. toctree::
   :maxdepth: 1

   use_anuga_viewer
   use_sww_gui
   use_domain_plotter
   use_sww_plotter
   use_qgis

.. only:: html




