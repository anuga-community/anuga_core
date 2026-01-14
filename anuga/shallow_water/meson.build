
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


# GPU extension with MPI support (optional)
gpu_enabled = get_option('gpu')

if gpu_enabled
  mpi_dep = dependency('mpi', language: 'c', required: true)

  # Get mpi4py include directory
  mpi4py_incdir = run_command(py3,
    ['-c', 'import mpi4py; print(mpi4py.get_include())'],
    check: true
  ).stdout().strip()

  gpu_inc_dir = include_directories('../utilities', incdir_numpy, mpi4py_incdir)

  # Compiler-specific GPU offloading flags
  if cc.get_id() == 'nvidia_hpc'
    # NVIDIA HPC SDK - best GPU support
    gpu_c_args = ['-O3', '-mp=gpu,multicore', '-Minfo=accel', '-g']
    if get_option('gpu_aware_mpi')
      gpu_c_args += ['-DGPU_AWARE_MPI']
    endif
  elif cc.get_id() == 'clang'
    # Clang with LLVM offloading
    gpu_c_args = ['-O3', '-fopenmp', '-fopenmp-targets=nvptx64-nvidia-cuda', '-g']
    if get_option('gpu_aware_mpi')
      gpu_c_args += ['-DGPU_AWARE_MPI']
    endif
  else
    # GCC with nvptx offloading
    gpu_c_args = ['-O3', '-fopenmp', '-foffload=nvptx-none', '-g']
    if get_option('gpu_aware_mpi')
      gpu_c_args += ['-DGPU_AWARE_MPI']
    endif
  endif

  gpu_deps = openmp_deps + [mpi_dep]

  py3.extension_module('sw_domain_gpu_ext',
    sources: ['sw_domain_gpu_ext.pyx'],
    c_args: gpu_c_args,
    include_directories: gpu_inc_dir,
    dependencies: gpu_deps,
    subdir: 'anuga/shallow_water',
    install: true,
  )

  message('GPU extension enabled')
  if get_option('gpu_aware_mpi')
    message('GPU-aware MPI enabled')
  endif
endif



python_sources = [
'boundaries.py',
'checkpoint.py',
'forcing.py',
'friction.py',
'__init__.py',
'most2nc.py',
'shallow_water_domain.py',
'sw_domain_gpu_omp.py',
'sww_interrogate.py',
'tsh2sww.py',
]


py3.install_sources(
  python_sources,
  subdir: 'anuga/shallow_water'
)

subdir('tests')
