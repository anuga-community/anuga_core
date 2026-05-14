
inc_dir = include_directories('../utilities', incdir_numpy)

py3.extension_module('distribute_mesh_ext',
  sources: ['distribute_mesh_ext.pyx'],
  c_args: openmp_c_args,
  include_directories: inc_dir,
  dependencies: openmp_deps,
  subdir: 'anuga/parallel',
  install: true,
)

# ParMETIS extension (optional — only built if parmetis + MPI + mpi4py available)
cc = meson.get_compiler('c')
_pm_prefix_lib = py_prefix / 'lib'
_pm_prefix_inc = py_prefix / 'include'

parmetis_dep = cc.find_library('parmetis', required: false, dirs: [_pm_prefix_lib])
metis_lib_dep = cc.find_library('metis', required: false, dirs: [_pm_prefix_lib])

if parmetis_dep.found() and metis_lib_dep.found() and mpi_dep.found() and have_mpi4py
  parmetis_inc_dir = include_directories('../utilities', incdir_numpy, mpi4py_incdir, _pm_prefix_inc)

  py3.extension_module('parmetis_ext',
    sources: ['parmetis_ext.pyx'],
    include_directories: parmetis_inc_dir,
    dependencies: [dep_py, dep_numpy, mpi_dep, parmetis_dep, metis_lib_dep],
    subdir: 'anuga/parallel',
    install: true,
  )
  message('ParMETIS extension: ENABLED')
else
  message('ParMETIS extension: DISABLED (parmetis=' + parmetis_dep.found().to_string() + ', metis=' + metis_lib_dep.found().to_string() + ', mpi=' + mpi_dep.found().to_string() + ', mpi4py=' + have_mpi4py.to_string() + ')')
endif

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