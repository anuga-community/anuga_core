
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
cray_system = get_option('cray')

if gpu_enabled
  # Get mpi4py include directory
  mpi4py_incdir = run_command(py3,
    ['-c', 'import mpi4py; print(mpi4py.get_include())'],
    check: true
  ).stdout().strip()

  gpu_inc_dir = include_directories('../utilities', incdir_numpy, mpi4py_incdir, 'gpu')

  # GPU architecture option
  # NVIDIA: cc70 (V100), cc80 (A100), cc90 (H100)
  # AMD: gfx908 (MI100), gfx90a (MI210/MI250), gfx942 (MI300X)
  gpu_arch = get_option('gpu_arch')

  # Detect if targeting AMD or NVIDIA based on architecture string
  is_amd_gpu = gpu_arch.startswith('gfx')

  if cray_system
    # On Cray systems (Perlmutter, Frontier, etc.), the compiler wrappers (cc, CC)
    # handle MPI linking automatically. We just need GPU offloading flags.
    # The wrapper must be set via CC=cc environment variable or native file.
    message('Cray system detected - using compiler wrapper for MPI')

    # NVHPC flags for GPU offloading (Cray wrapper passes these through)
    gpu_c_args = ['-O3', '-mp=gpu,multicore', '-Minfo=accel', '-g', '-gpu=' + gpu_arch]
    if get_option('gpu_aware_mpi')
      gpu_c_args += ['-DGPU_AWARE_MPI']
    endif

    # On Cray, don't add explicit MPI dependency - wrapper handles it
    gpu_deps = openmp_deps

  else
    # Non-Cray systems - need explicit MPI dependency
    mpi_dep = dependency('mpi', language: 'c', required: true)

    # Compiler-specific GPU offloading flags
    if cc.get_id() == 'nvidia_hpc'
      # NVIDIA HPC SDK - best GPU support for NVIDIA GPUs
      gpu_c_args = ['-O3', '-mp=gpu,multicore', '-Minfo=accel', '-g', '-gpu=' + gpu_arch]
      if get_option('gpu_aware_mpi')
        gpu_c_args += ['-DGPU_AWARE_MPI']
      endif
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
      if get_option('gpu_aware_mpi')
        gpu_c_args += ['-DGPU_AWARE_MPI']
      endif
    else
      # GCC with nvptx offloading (NVIDIA only, AMD not well supported)
      gpu_c_args = ['-O3', '-fopenmp', '-foffload=nvptx-none', '-g']
      if get_option('gpu_aware_mpi')
        gpu_c_args += ['-DGPU_AWARE_MPI']
      endif
    endif

    gpu_deps = openmp_deps + [mpi_dep]
  endif

  py3.extension_module('sw_domain_gpu_ext',
    sources: [
      'sw_domain_gpu_ext.pyx',
      'gpu/gpu_domain_core.c',
      'gpu/gpu_boundaries.c',
      'gpu/gpu_kernels.c',
      'gpu/gpu_halo.c',
      'gpu/gpu_rate_operator.c',
      'gpu/gpu_flop.c',
    ],
    c_args: gpu_c_args,
    include_directories: gpu_inc_dir,
    dependencies: gpu_deps,
    subdir: 'anuga/shallow_water',
    install: true,
  )

  message('GPU extension enabled')
  message('GPU architecture: ' + gpu_arch)
  if cray_system
    message('Cray system mode: MPI handled by compiler wrapper')
  endif
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
