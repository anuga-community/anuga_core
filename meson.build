# Meson build file for ANUGA core.
# version read from pyproject.toml
# This ensures that the version is consistent across the project files
# and avoids the need for manual updates in multiple places.
project('anuga', 'c', 'cpp', 'cython', version: run_command('python', ['-c', 'import toml; print(toml.load("pyproject.toml")["project"]["version"])']).stdout().strip()  , default_options: ['cpp_std=c++17'])

add_project_arguments('-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION', language: 'cpp')
add_project_arguments('-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION', language: 'c')

py3 = import('python').find_installation(pure: false)

dep_py = py3.dependency()
py3_version = py3.language_version()

#========================================================
# Setup numpy dependencies
#========================================================
incdir_numpy = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

if py3_version.version_compare('>3.8')
  # Recommended for Python > 3.8
  dep_numpy = dependency('numpy')
else
  # Fallback for Python 3.8
  inc_numpy = include_directories(incdir_numpy)
  dep_numpy = declare_dependency(include_directories: inc_numpy)
endif



# FIXME SR: Do we use this now?
#message('NumPy include directory: ' + inc_numpy)

dependencies = [dep_py, dep_numpy]

#========================================================
# Deal with OpenMP
#========================================================

# cc = meson.get_compiler('c')
# if openmp_dep.found()
#   if cc.get_id() in ['intel', 'intel-cl', 'icx']
#     openmp_flag = '-qopenmp'
#   else
#     openmp_flag = '-fopenmp'
#   endif
# endif

openmp = dependency('openmp', required: false)
if openmp.found()
  # On Windows, the mingw compiler does not support OpenMP ATOMIC operations
  # so using gcc_win-64 gxx_win-64 provided by conda-forge
  # On linux, OpenMP is supported by gcc and g++
  # On macOS, OpenMP is not supported by the Apple clang compiler so using 
  # clang and llvm-openmp provided by conda-forge
 
  #if host_machine.system() == 'windows'
  
  openmp_c_args = ['-O3', '-march=native', '-fopenmp', '-g']
  # openmp_c_args = ['-O3', '-march=native','-funroll-loops', '-fvectorize', '-Rpass=loop-vectorize', '-Rpass=loop-unroll', '-g'],
  # openmp_c_args = ['-O3', '-march=native', openmp_flag, '-g'],

  openmp_deps = dependencies + [openmp]

else
  openmp_deps = dependencies
endif



#========================================================
# Install the Python scripts
#========================================================
conf = configuration_data()
conf.set('PYTHON', py3.path())


message('BinDir: ' + get_option('bindir'))
configure_file(
    input: 'scripts/anuga_pmesh_gui.py',
    output: 'anuga_pmesh_gui',
    configuration: conf,
    install_dir: get_option('bindir'),
    install_mode: 'rwxr-xr-x'
)

configure_file(
    input: 'scripts/anuga_sww_merge.py',
    output: 'anuga_sww_merge',
    configuration: conf,
    install_dir: get_option('bindir'),
    install_mode: 'rwxr-xr-x'
)

configure_file(
    input: 'scripts/anuga_benchmark_omp.py',
    output: 'anuga_benchmark_omp',
    configuration: conf,
    install_dir: get_option('bindir'),
    install_mode: 'rwxr-xr-x'
)

# Add subdirectories which contains python sources
subdir('anuga')





