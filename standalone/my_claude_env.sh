# Gadi modules for the MPI + GPU-offload miniapp build (source me).
# openmpi/5.0.5 is CUDA-aware: with OpenMP target, device buffers are packed
# on the GPU and staged through host for MPI (see gpu_halo.c GPU_AWARE_MPI).
source /opt/Modules/v4.3.0/init/bash 2>/dev/null || true
module load nvidia-hpc-sdk/25.9
module load cuda/12.9.0     # nvlink pairing for nvhpc 25.9
module load openmpi/5.0.5
