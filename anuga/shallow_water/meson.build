
inc_dir = include_directories('../utilities', incdir_numpy)



# py3.extension_module('boundaries',
#   sources: ['boundaries.pyx'],
#   include_directories: inc_dir,
#   dependencies: dependencies,
#   subdir: 'anuga/shallow_water',
#   install: true,
# )


py3.extension_module('sw_domain_openmp_ext',
  sources: ['sw_domain_openmp_ext.pyx'],
  c_args : openmp_c_args,
  include_directories: inc_dir,
  dependencies: openmp_deps,
  subdir: 'anuga/shallow_water',
  install: true,
)



python_sources = [
'boundaries.py',
'checkpoint.py',
'forcing.py',
'friction.py',
'__init__.py',
'most2nc.py',
'shallow_water_domain.py',
'sww_interrogate.py',
'tsh2sww.py',
]


py3.install_sources(
  python_sources,
  subdir: 'anuga/shallow_water'
)

subdir('tests')
