
inc_dir = include_directories('../utilities', incdir_numpy)

# Include directory with GPU subfolder for unified kernels
inc_dir_unified = include_directories('../utilities', incdir_numpy, 'gpu')

# Compiler flags for unified kernels (CPU-only mode)
unified_c_args = openmp_c_args + ['-DCPU_ONLY_MODE']

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


# Unified OpenMP implementation with MPI support
# Can target GPU (gpu_offload=true) or CPU multicore (gpu_offload=false)
gpu_offload = get_option('gpu_offload')

# MPI detection is done in the root meson.build (mpi_dep, have_mpi4py, mpi4py_incdir)
if mpi_dep.found() and have_mpi4py
  gpu_inc_dir = include_directories('../utilities', incdir_numpy, mpi4py_incdir, 'gpu')
else
  gpu_inc_dir = include_directories('../utilities', incdir_numpy, 'gpu')
endif

# GPU architecture option (only used when gpu_offload=true)
# NVIDIA: cc70 (V100), cc80 (A100), cc90 (H100)
# AMD: gfx908 (MI100), gfx90a (MI210/MI250), gfx942 (MI300X)
gpu_arch = get_option('gpu_arch')

# Detect if targeting AMD or NVIDIA based on architecture string
is_amd_gpu = gpu_arch.startswith('gfx')

# Compiler-specific flags
# When gpu_offload=false, compile for CPU multicore (no GPU offloading)
# This allows multiprocessor_mode=2 to run on CPUs with any OpenMP compiler
#
# Uses effective_compiler_id to handle Cray wrappers correctly
if gpu_offload
  # GPU offloading enabled
  if effective_compiler_id == 'nvidia_hpc'
    # NVIDIA HPC SDK - best GPU support for NVIDIA GPUs
    gpu_c_args = ['-O3', '-mp=gpu,multicore', '-Minfo=accel', '-g', '-gpu=' + gpu_arch]
  elif effective_compiler_id == 'clang'
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
  if effective_compiler_id == 'nvidia_hpc'
    gpu_c_args = ['-O3', '-mp=multicore', '-Minfo=accel', '-g']
  elif effective_compiler_id == 'intel' or effective_compiler_id == 'intel-cl'
    gpu_c_args = ['-O3', '-qopenmp', '-g']
  elif effective_compiler_id == 'intel-llvm' or effective_compiler_id == 'icx'
    gpu_c_args = ['-O3', '-fiopenmp', '-g']
  elif effective_compiler_id == 'clang'
    # Clang with libomp (LLVM, Homebrew macOS, or AOMP)
    gpu_c_args = ['-O3', '-fopenmp', '-g']
  else
    # GCC and others
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

# Runtime GPU-aware MPI detection via MPIX_Query_cuda_support / MPIX_Query_rocm_support.
# These are non-standard extensions present in Open MPI and MVAPICH2 GPU builds.
# We probe the MPI compiler for <mpi-ext.h> and the relevant symbol; if found,
# compile-time flags enable the runtime detection path in detect_gpu_aware_mpi().
if mpi_dep.found()
  if cc.has_header('mpi-ext.h', dependencies: [mpi_dep])
    if cc.has_function('MPIX_Query_cuda_support',
                       prefix: '#include <mpi.h>\n#include <mpi-ext.h>',
                       dependencies: [mpi_dep])
      gpu_c_args += ['-DHAVE_MPI_EXT_H', '-DHAVE_MPIX_CUDA_SUPPORT']
      message('MPIX_Query_cuda_support found - GPU-aware MPI runtime detection enabled (CUDA)')
    elif cc.has_function('MPIX_Query_rocm_support',
                         prefix: '#include <mpi.h>\n#include <mpi-ext.h>',
                         dependencies: [mpi_dep])
      gpu_c_args += ['-DHAVE_MPI_EXT_H', '-DHAVE_MPIX_ROCM_SUPPORT']
      message('MPIX_Query_rocm_support found - GPU-aware MPI runtime detection enabled (ROCm)')
    else
      message('mpi-ext.h found but no MPIX_Query_*_support - runtime detection disabled')
    endif
  else
    message('mpi-ext.h not found - GPU-aware MPI runtime detection disabled')
  endif
endif

# NVTX profiling hooks (-Duse_nvtx=true)
# Probes for nvToolsExt in standard CUDA toolkit locations.
# Tries NVTX v3 (standalone header-only install) first, then v1 (CUDA toolkit).
# When found: adds -DNVTX_ENABLED and links nvToolsExt.
nvtx_dep = []
if get_option('use_nvtx')
  # Check for NVTX v3 header (preferred - header-only, no link needed)
  if cc.has_header('nvtx3/nvToolsExt.h')
    gpu_c_args += ['-DNVTX_ENABLED', '-DHAVE_NVTX3']
    message('NVTX v3 headers found - NVTX profiling enabled (header-only)')
  else
    # Fall back to NVTX v1 from CUDA toolkit (requires linking nvToolsExt)
    nvtx_dep_probe = dependency('nvToolsExt', required: false)
    if not nvtx_dep_probe.found()
      # Try manual search in common CUDA toolkit locations
      cuda_home = run_command('sh', ['-c', 'echo ${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}'],
                              check: false).stdout().strip()
      nvtx_lib = cc.find_library('nvToolsExt',
                                  dirs: [cuda_home / 'lib64', cuda_home / 'lib',
                                         '/usr/local/cuda/lib64', '/usr/local/cuda/lib'],
                                  required: false)
      if nvtx_lib.found() and cc.has_header('nvToolsExt.h',
                                             args: ['-I' + cuda_home / 'include',
                                                    '-I/usr/local/cuda/include'])
        nvtx_dep = [nvtx_lib]
        gpu_c_args += ['-DNVTX_ENABLED',
                       '-I' + cuda_home / 'include',
                       '-I/usr/local/cuda/include']
        message('NVTX v1 found in CUDA toolkit - NVTX profiling enabled')
      else
        warning('use_nvtx=true but nvToolsExt not found - NVTX profiling disabled')
      endif
    else
      nvtx_dep = [nvtx_dep_probe]
      gpu_c_args += ['-DNVTX_ENABLED']
      message('NVTX found via pkg-config - NVTX profiling enabled')
    endif
  endif
endif

# Only build sw_domain_gpu_ext if MPI and mpi4py are available
if mpi_dep.found() and have_mpi4py
  gpu_deps = openmp_deps + [mpi_dep] + nvtx_dep

  py3.extension_module('sw_domain_gpu_ext',
    sources: [
      'sw_domain_gpu_ext.pyx',
      'gpu/gpu_domain_core.c',
      'gpu/gpu_boundaries.c',
      'gpu/gpu_kernels.c',
      'gpu/gpu_halo.c',
      'gpu/gpu_rate_operator.c',
      'gpu/gpu_inlet_operator.c',
      'gpu/gpu_culvert_operator.c',
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
    message('Unified extension built with GPU target')
  else
    message('Unified extension built with CPU multicore target')
  endif
else
  warning('MPI or mpi4py not found - sw_domain_gpu_ext will NOT be built')
  warning('multiprocessor_mode=2 will not be available')
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
