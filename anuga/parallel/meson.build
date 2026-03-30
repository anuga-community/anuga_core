
inc_dir = include_directories('../utilities', incdir_numpy)

py3.extension_module('distribute_mesh_ext',
  sources: ['distribute_mesh_ext.pyx'],
  c_args: openmp_c_args,
  include_directories: inc_dir,
  dependencies: openmp_deps,
  subdir: 'anuga/parallel',
  install: true,
)

python_sources = [
  '__init__.py',
  'config.py',
  'distribute_mesh.py',
  'partitioning.py',
  'parallel_advection.py',
  'parallel_api.py',
  'parallel_boyd_box_operator.py',
  'parallel_boyd_pipe_operator.py',
  'parallel_generic_communications.py',
  'parallel_inlet_enquiry.py',
  'parallel_inlet_operator.py',
  'parallel_inlet.py',
  'parallel_internal_boundary_operator.py',
  'parallel_meshes.py',
  'parallel_operator_factory.py',
  'parallel_shallow_water.py',
  'parallel_structure_operator.py',
  'parallel_weir_orifice_trapezoid_operator.py',
  'print_stats.py',
  'sequential_distribute.py',
]

py3.install_sources(
  python_sources,
  subdir: 'anuga/parallel'
)

subdir('tests')
subdir('data')