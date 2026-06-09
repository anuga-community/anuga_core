# libanuga_sw — standalone build

A CMake build of ANUGA's shallow-water GPU/OpenMP kernels with no Cython, meson,
or NetCDF. Produces `libanuga_sw.so` (+ `anuga_miniapp`).

## Requirements
- CMake ≥ 3.18 and a C compiler
- OpenMP (always); for GPU: an OpenMP-offload compiler (nvhpc `nvc`, or clang)
- Optional MPI for multi-rank

## Build

CPU only (any machine, gcc):
```bash
cmake -S standalone -B standalone/build -DCMAKE_BUILD_TYPE=Release
cmake --build standalone/build -j
```

GPU offload (NVIDIA HPC SDK, e.g. V100):
```bash
cmake -S standalone -B standalone/build_gpu \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_BUILD_TYPE=Release \
  -DANUGA_OFFLOAD=ON -DANUGA_GPU_ARCH=cc70
cmake --build standalone/build_gpu -j
```

GPU + MPI (links the MPI found on PATH; no conda needed):
```bash
cmake -S standalone -B standalone/build_mpi \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_BUILD_TYPE=Release \
  -DANUGA_OFFLOAD=ON -DANUGA_GPU_ARCH=cc70 -DANUGA_MPI=ON
cmake --build standalone/build_mpi -j
```

## Options
| Option | Default | Meaning |
|---|---|---|
| `ANUGA_OFFLOAD` | OFF | GPU offload (OFF → CPU `CPU_ONLY_MODE`) |
| `ANUGA_MPI` | OFF | link MPI (OFF → single-process) |
| `ANUGA_GPU_ARCH` | `cc70` | GPU arch: `cc70` V100, `cc80` A100, `cc90` H100, `gfx90a`/`gfx942` AMD |

Offload flags are auto-derived per compiler (nvhpc / clang-nvptx / clang-amd /
gcc-nvptx); override with `-DANUGA_OFFLOAD_FLAGS="..."` if needed.

## Run

Point the Python drivers at the library you built:
```bash
export ANUGA_SW_LIB=$PWD/standalone/build_gpu/libanuga_sw.so
```

### Single GPU (meshpy → C), validated against ANUGA
```bash
source standalone/env_dgx.sh                 # conda + nvc on PATH
python standalone/cases/dam_break_mesh.py --meshpy   # meshpy mesh → C geometry → GPU
python standalone/cases/dam_break.py                 # ctypes vs ANUGA, machine precision
```
Confirm it really runs on the device (aborts if it can't offload):
```bash
OMP_TARGET_OFFLOAD=MANDATORY python standalone/cases/dam_break.py
```

### Multi-GPU, system MPI, no conda (pure-C mini-app)
```bash
# 1. write per-rank partition files ONCE (needs conda + ANUGA)
source standalone/env_dgx.sh
python standalone/tools/partition_dump.py /tmp/anuga_adm     # np = 1,2,4

# 2. run on the GPUs with the system MPI — no conda
NVROOT=/home/jorge/install/nvhpc/26.3/Linux_x86_64/26.3
export PATH=$NVROOT/compilers/bin:$NVROOT/comm_libs/mpi/bin:/usr/local/bin:/usr/bin:/bin
unset CONDA_PREFIX
export LD_LIBRARY_PATH=$PWD/standalone/build_mpi:$NVROOT/compilers/lib:$NVROOT/comm_libs/mpi/lib:$NVROOT/comm_libs/12.9/hpcx/hpcx-2.25.1/ompi/lib
export OMPI_MCA_coll_hcoll_enable=0

mpirun -n 4 ./standalone/build_mpi/anuga_miniapp /tmp/anuga_adm dam_break
```
`mpirun -n 1` works too; a bare `./anuga_miniapp` does not (HPC-X needs the launcher).
Validate the results against ANUGA: `python standalone/cases/compare_dump.py /tmp/anuga_adm`.

