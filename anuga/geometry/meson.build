
inc_dir = include_directories('../utilities', incdir_numpy)

# Compile the Cython-generated C code and additional C code
py3.extension_module('polygon_ext',
  sources: ['polygon_ext.pyx'],
  c_args : openmp_c_args,
  include_directories: inc_dir,
  dependencies: openmp_deps,
  subdir: 'anuga/geometry',
  install: true,
)

python_sources = [
  'aabb.py',
  '__init__.py',
  'polygon_function.py',
  'polygon.py',
  'quad.py',
]

py3.install_sources(
  python_sources,
  subdir: 'anuga/geometry'
)

subdir('tests')
