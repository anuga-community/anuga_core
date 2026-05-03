
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

# MPI is optional - if not found, sw_domain_gpu_ext won't be built
mpi_dep = dependency('mpi', language: 'c', required: false)

# Check if mpi4py is available
mpi4py_check = run_command(py3, ['-c', 'import mpi4py; print(mpi4py.get_include())'], check: false)
have_mpi4py = mpi4py_check.returncode() == 0

# Fallback: if standard MPI detection failed but mpi4py is available,
# construct MPI dependency from mpicc -show output so that compile flags
# (especially -I paths) match exactly what mpi4py was built against.
# Falls back to sys.prefix/{include,lib} only if mpicc cannot be executed.
if not mpi_dep.found() and have_mpi4py
  mpi_prefix_result = run_command(py3, ['-c', '''
import sys, os, subprocess, shutil
try:
    import mpi4py
    prefix = sys.prefix
    is_windows = sys.platform == "win32"
    # conda on Windows puts headers/libs under Library/; Unix uses prefix directly
    if is_windows:
        inc = os.path.join(prefix, "Library", "include")
        lib = os.path.join(prefix, "Library", "lib")
    else:
        inc = os.path.join(prefix, "include")
        lib = os.path.join(prefix, "lib")
    cfg = mpi4py.get_config()
    # mpi4py.get_config() may supply explicit dirs; prefer those if present
    cfg_incs = cfg.get("include_dirs", [])
    cfg_libs = cfg.get("library_dirs", [])
    if cfg_incs:
        inc = cfg_incs[0]
    if cfg_libs:
        lib = cfg_libs[0]
    if not os.path.isfile(os.path.join(inc, "mpi.h")):
        sys.exit(1)
    if is_windows:
        # No mpicc on Windows; build flags from known conda paths.
        # Library names from mpi4py config, defaulting to msmpi (conda-forge).
        libnames = cfg.get("libraries", ["msmpi"]) or ["msmpi"]
        cflags = ["-I" + inc]
        lflags = ["-L" + lib] + ["-l" + n for n in libnames]
    else:
        # Unix: run mpicc -show to get the exact flags mpi4py was built with
        mpicc = cfg.get("mpicc", "")
        if mpicc and not os.path.isabs(mpicc):
            mpicc = shutil.which(mpicc) or ""
        if not mpicc or not os.path.isfile(mpicc):
            mpicc = os.path.join(prefix, "bin", "mpicc")
        try:
            out = subprocess.check_output([mpicc, "-show"], text=True).strip()
            tokens = out.split()
            cflags = [t for t in tokens[1:] if t.startswith(("-I", "-D"))]
            lflags = [t for t in tokens[1:] if t.startswith(("-L", "-l", "-Wl,", "-rpath"))]
            if not cflags:
                cflags = ["-I" + inc]
        except Exception:
            cflags = ["-I" + inc]
            lflags = ["-L" + lib, "-lmpi"]
    print("|".join(cflags) + "##" + "|".join(lflags))
except Exception:
    sys.exit(1)
'''], check: false)

  if mpi_prefix_result.returncode() == 0
    mpi_output = mpi_prefix_result.stdout().strip()
    mpi_parts = mpi_output.split('##')
    if mpi_parts.length() == 2
      mpi_compile_args = mpi_parts[0] != '' ? mpi_parts[0].split('|') : []
      mpi_link_args = mpi_parts[1] != '' ? mpi_parts[1].split('|') : ['-lmpi']
      mpi_dep = declare_dependency(
        compile_args: mpi_compile_args,
        link_args: mpi_link_args,
      )
      message('MPI found via mpicc -show (fallback): ' + mpi_parts[0])
    endif
  endif
endif

if mpi_dep.found() and have_mpi4py
  mpi4py_incdir = mpi4py_check.stdout().strip()
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
  elif effective_compiler_id == 'intel-llvm'
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
  # GCC/Clang warn about variables used only in #pragma omp target map() clauses
  # in CPU_ONLY_MODE because the target region is a no-op; NVIDIA HPC SDK does not.
  if effective_compiler_id == 'gcc'
    gpu_c_args += ['-Wno-unused-variable', '-Wno-unused-but-set-variable', '-Wno-unused-function']
  elif effective_compiler_id == 'clang'
    gpu_c_args += ['-Wno-unused-variable', '-Wno-unused-function']
  endif
  message('CPU multicore mode enabled (no GPU offloading)')
endif

if get_option('gpu_aware_mpi') and gpu_offload
  gpu_c_args += ['-DGPU_AWARE_MPI']
  message('GPU-aware MPI enabled')
endif

# Runtime GPU-aware MPI detection via MPIX_Query_cuda_support / MPIX_Query_rocm_support.
# These are non-standard extensions present in Open MPI and MVAPICH2 GPU builds.
# Only probe when actually building with GPU offloading: on CPU-only builds (CI on
# macOS/Windows) Open MPI ships MPIX_Query_cuda_support() as a zero-returning stub
# even without CUDA, causing a false-positive detection and a misleading message.
if mpi_dep.found() and gpu_offload
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

  # On macOS, SIP strips DYLD_LIBRARY_PATH from child processes so libmpi.dylib
  # is not found at dlopen time even though it is present in the conda env.
  # Embed two rpaths:
  #   @executable_path/../lib  — resolves to $CONDA_PREFIX/lib relative to the
  #                              Python binary; portable across any conda env path
  #   sys.prefix/lib           — absolute fallback for non-standard layouts
  gpu_link_args = []
  if host_machine.system() == 'darwin'
    _py_prefix = run_command(py3, ['-c', 'import sys; print(sys.prefix)'], check: false)
    if _py_prefix.returncode() == 0
      _mpi_lib_dir = _py_prefix.stdout().strip() + '/lib'
      gpu_link_args = [
        '-Wl,-rpath,@executable_path/../lib',
        '-Wl,-rpath,' + _mpi_lib_dir,
      ]
      message('macOS: embedding rpath for libmpi.dylib: ' + _mpi_lib_dir)
    endif
  endif

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
    link_args: gpu_link_args,
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
  # Expected on Windows (MSMPI has no mpicc/pkg-config) and minimal Linux installs.
  # GPU tests will be skipped via ImportError on those platforms.
  message('MPI or mpi4py not found - sw_domain_gpu_ext will NOT be built (multiprocessor_mode=2 unavailable)')
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
