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
