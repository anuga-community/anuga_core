
inc_dir = include_directories('../utilities', incdir_numpy)

# Include directory with GPU subfolder for unified kernels
inc_dir_unified = include_directories('../utilities', incdir_numpy, 'gpu')

# Compiler flags for unified kernels (CPU-only mode)
unified_c_args = openmp_c_args + ['-DCPU_ONLY_MODE']

# py3.extension_module('boundaries',
#   sources: ['boundaries.pyx'],
#   include_directories: inc_dir,
#   dependencies: dependencies,
#   subdir: 'anuga/shallow_water',
#   install: true,
# )


py3.extension_module('sw_domain_openmp_ext',
  sources: [
    'sw_domain_openmp_ext.pyx',
    'gpu/core_kernels.c',
  ],
  c_args : unified_c_args,
  include_directories: inc_dir_unified,
  dependencies: openmp_deps,
  subdir: 'anuga/shallow_water',
  install: true,
)


# New OpenMP implementation (mode 2) with MPI support
# Can target GPU (gpu_offload=true) or CPU multicore (gpu_offload=false)
new_omp_enabled = get_option('new_omp')
gpu_offload = get_option('gpu_offload')

if new_omp_enabled
  mpi_dep = dependency('mpi', language: 'c', required: true)

  # Get mpi4py include directory
  mpi4py_incdir = run_command(py3,
    ['-c', 'import mpi4py; print(mpi4py.get_include())'],
    check: true
  ).stdout().strip()

  gpu_inc_dir = include_directories('../utilities', incdir_numpy, mpi4py_incdir, 'gpu')

  # GPU architecture option (only used when gpu_offload=true)
  # NVIDIA: cc70 (V100), cc80 (A100), cc90 (H100)
  # AMD: gfx908 (MI100), gfx90a (MI210/MI250), gfx942 (MI300X)
  gpu_arch = get_option('gpu_arch')

  # Detect if targeting AMD or NVIDIA based on architecture string
  is_amd_gpu = gpu_arch.startswith('gfx')

  # Compiler-specific flags
  # When gpu_offload=false, compile for CPU multicore (no GPU offloading)
  # This allows multiprocessor_mode=2 to run on CPUs with any OpenMP compiler
  if gpu_offload
    # GPU offloading enabled
    if cc.get_id() == 'nvidia_hpc'
      # NVIDIA HPC SDK - best GPU support for NVIDIA GPUs
      # -noacc disables OpenACC to avoid __acc_compiled symbol issues
      gpu_c_args = ['-O3', '-mp=gpu,multicore', '-Minfo=accel', '-g', '-gpu=' + gpu_arch]
    elif cc.get_id() == 'clang'
      if is_amd_gpu
        # Clang/AOMP for AMD GPUs
        gpu_c_args = ['-O3', '-fopenmp', '-fopenmp-targets=amdgcn-amd-amdhsa',
                      '-Xopenmp-target=amdgcn-amd-amdhsa', '-march=' + gpu_arch, '-g']
      else
        # Clang with LLVM offloading for NVIDIA
        # Convert cc70 -> sm_70 format for clang
        sm_arch = 'sm_' + gpu_arch.substring(2)
        gpu_c_args = ['-O3', '-fopenmp', '-fopenmp-targets=nvptx64-nvidia-cuda',
                      '-Xopenmp-target=nvptx64-nvidia-cuda', '-march=' + sm_arch, '-g']
      endif
    else
      # GCC with nvptx offloading (NVIDIA only, AMD not well supported)
      gpu_c_args = ['-O3', '-fopenmp', '-foffload=nvptx-none', '-g']
    endif
    message('GPU offloading enabled')
    message('GPU architecture: ' + gpu_arch)
  else
    # CPU multicore execution (no GPU offloading)
    # #pragma omp target teams loop becomes regular OpenMP parallel
    if cc.get_id() == 'nvidia_hpc'
      gpu_c_args = ['-O3', '-mp=multicore', '-Minfo=accel', '-g']
    elif cc.get_id() == 'intel' or cc.get_id() == 'intel-cl'
      gpu_c_args = ['-O3', '-qopenmp', '-g']
    elif cc.get_id() == 'intel-llvm'
      gpu_c_args = ['-O3', '-fiopenmp', '-g']
    else
      # GCC, Clang, and others
      gpu_c_args = ['-O3', '-fopenmp', '-g']
    endif
    # Define macro so code knows it's CPU-only mode
    gpu_c_args += ['-DCPU_ONLY_MODE']
    message('CPU multicore mode enabled (no GPU offloading)')
  endif

  if get_option('gpu_aware_mpi') and gpu_offload
    gpu_c_args += ['-DGPU_AWARE_MPI']
    message('GPU-aware MPI enabled')
  endif

  gpu_deps = openmp_deps + [mpi_dep]

  py3.extension_module('sw_domain_gpu_ext',
    sources: [
      'sw_domain_gpu_ext.pyx',
      'gpu/gpu_domain_core.c',
      'gpu/gpu_boundaries.c',
      'gpu/gpu_kernels.c',
      'gpu/gpu_halo.c',
      'gpu/gpu_rate_operator.c',
      'gpu/gpu_inlet_operator.c',
      'gpu/gpu_flop.c',
      'gpu/core_kernels.c',
    ],
    c_args: gpu_c_args,
    include_directories: gpu_inc_dir,
    dependencies: gpu_deps,
    subdir: 'anuga/shallow_water',
    install: true,
  )

  if gpu_offload
    message('Mode 2 extension enabled (GPU target)')
  else
    message('Mode 2 extension enabled (CPU multicore target)')
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
