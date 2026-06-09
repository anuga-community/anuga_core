# Standalone CMake build for the ANUGA shallow-water solver.
#
# Produces libanuga_sw, built from the *unmodified* production kernels in
# anuga/shallow_water/gpu/*.c plus the thin C ABI shim (src/anuga_sw_capi.c).
# No Cython, no meson, no NetCDF. Python (via ctypes) is a runtime-only driver.
#
# Typical use:
#   cmake -S standalone -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# Options:
#   -DANUGA_OFFLOAD=ON   enable OpenMP target offload (GPU). Default OFF => CPU_ONLY_MODE.
#   -DANUGA_MPI=ON       link real MPI (multi-rank halo exchange). Default OFF => single-process stubs.
#
cmake_minimum_required(VERSION 3.18)
project(anuga_sw C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ANUGA_OFFLOAD "Enable OpenMP target offload (GPU)" OFF)
option(ANUGA_MPI     "Link real MPI for multi-rank halo exchange" OFF)
# GPU architecture (mirrors meson gpu_arch). NVIDIA: cc70(V100) cc80(A100) cc90(H100).
# AMD: gfx90a, gfx942. A CUDA version can be appended for nvhpc, e.g. "cc70,cuda11.4".
set(ANUGA_GPU_ARCH "cc70" CACHE STRING "GPU arch for offload (nvhpc -gpu= / clang -march=)")

# ---------------------------------------------------------------- source layout
# The production kernels live in the main tree, one level up from standalone/.
get_filename_component(ANUGA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(GPU_DIR "${ANUGA_ROOT}/anuga/shallow_water/gpu")

# Full kernel set (closed dependency graph; covers hydro + boundaries + operators).
# The CUDA file (core_kernels_cuda.cu) is intentionally excluded.
set(ANUGA_KERNEL_SOURCES
  ${GPU_DIR}/core_kernels.c
  ${GPU_DIR}/gpu_domain_core.c
  ${GPU_DIR}/gpu_kernels.c
  ${GPU_DIR}/gpu_boundaries.c
  ${GPU_DIR}/gpu_halo.c
  ${GPU_DIR}/gpu_flop.c
  ${GPU_DIR}/gpu_rate_operator.c
  ${GPU_DIR}/gpu_inlet_operator.c
  ${GPU_DIR}/gpu_culvert_operator.c
  ${GPU_DIR}/gpu_max_quantities_operator.c
)

add_library(anuga_sw SHARED
  ${ANUGA_KERNEL_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/anuga_sw_capi.c
)

target_include_directories(anuga_sw PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ANUGA_ROOT}/anuga/shallow_water
  ${GPU_DIR}
  ${ANUGA_ROOT}/anuga/utilities
)

# ---------------------------------------------------------------- OpenMP
find_package(OpenMP REQUIRED COMPONENTS C)
target_link_libraries(anuga_sw PUBLIC OpenMP::OpenMP_C)

# ---------------------------------------------------------------- offload (GPU)
# Per-compiler offload flags, mirroring anuga/shallow_water/meson.build. Override
# the whole set with -DANUGA_OFFLOAD_FLAGS=... if your toolchain differs.
if(ANUGA_OFFLOAD)
  message(STATUS "anuga_sw: OpenMP target offload ENABLED (arch ${ANUGA_GPU_ARCH})")
  set(ANUGA_OFFLOAD_FLAGS "" CACHE STRING "Override OpenMP offload compile/link flags")
  if(NOT ANUGA_OFFLOAD_FLAGS)
    if(CMAKE_C_COMPILER_ID STREQUAL "NVHPC")
      set(ANUGA_OFFLOAD_FLAGS -mp=gpu,multicore -gpu=${ANUGA_GPU_ARCH} -Minfo=accel)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "Clang")
      if(ANUGA_GPU_ARCH MATCHES "^gfx")
        set(ANUGA_OFFLOAD_FLAGS -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa
            -Xopenmp-target=amdgcn-amd-amdhsa -march=${ANUGA_GPU_ARCH})
      else()
        string(SUBSTRING "${ANUGA_GPU_ARCH}" 2 -1 _sm)  # cc70 -> 70
        set(ANUGA_OFFLOAD_FLAGS -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
            -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_${_sm})
      endif()
    else()  # GCC nvptx
      set(ANUGA_OFFLOAD_FLAGS -fopenmp -foffload=nvptx-none)
    endif()
  endif()
  message(STATUS "anuga_sw: offload flags = ${ANUGA_OFFLOAD_FLAGS}")
  target_compile_options(anuga_sw PRIVATE ${ANUGA_OFFLOAD_FLAGS})
  target_link_options(anuga_sw PRIVATE ${ANUGA_OFFLOAD_FLAGS})
else()
  message(STATUS "anuga_sw: CPU_ONLY_MODE (no offload)")
  target_compile_definitions(anuga_sw PRIVATE CPU_ONLY_MODE)
endif()

# ---------------------------------------------------------------- MPI
# CRITICAL: link the SAME MPI that mpi4py was built against, because the
# MPI_Comm handle is passed from Python (comm.py2f) and is only valid within
# that library. On systems with several MPIs (e.g. conda Open MPI + nvhpc HPC-X)
# default MPI_C_COMPILER to conda's mpicc so find_package(MPI) extracts conda's
# flags. Override with -DMPI_C_COMPILER=/path/to/mpicc if needed.
if(ANUGA_MPI)
  message(STATUS "anuga_sw: MPI ENABLED")
  if(NOT MPI_C_COMPILER AND DEFINED ENV{CONDA_PREFIX})
    if(EXISTS "$ENV{CONDA_PREFIX}/bin/mpicc")
      set(MPI_C_COMPILER "$ENV{CONDA_PREFIX}/bin/mpicc" CACHE FILEPATH "")
      message(STATUS "anuga_sw: using conda mpicc: ${MPI_C_COMPILER}")
    endif()
  endif()
  find_package(MPI REQUIRED COMPONENTS C)
  target_link_libraries(anuga_sw PUBLIC MPI::MPI_C)
  target_compile_definitions(anuga_sw PRIVATE HAVE_MPI)
else()
  message(STATUS "anuga_sw: single-process (MPI stubs)")
endif()

target_link_libraries(anuga_sw PUBLIC m)

# Export the ABI header location for downstream users.
set_target_properties(anuga_sw PROPERTIES
  PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/anuga_sw.h"
)

install(TARGETS anuga_sw
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)
