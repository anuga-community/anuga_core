
openmp = dependency('openmp', required: false)
if openmp.found()
  if host_machine.system() == 'windows'
    # On Windows, the mingw compiler does not support OpenMP ATOMIC operations
    openmp_deps = dependencies + [openmp]
  else
    openmp_deps = dependencies + [openmp]
  endif

else
  openmp_deps = dependencies
endif

inc_dir = include_directories('../utilities', incdir_numpy)

# Compile the Cython-generated C code and additional C code
py3.extension_module('polygon_ext',
  sources: ['polygon_ext.pyx'],
  c_args : ['-O3', '-march=native', '-fopenmp', '-g'],
  include_directories: inc_dir,
  #dependencies: dependencies,
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
