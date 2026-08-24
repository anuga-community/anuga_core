# Source this before building/running the GPU targets:  source env.sh
#
# Keep the cuda module matched to the CUDA that nvhpc bundles (25.9 -> 12.9);
# a mismatched nvlink refuses objects it thinks are "newer than toolkit".
#
# nvc (NVIDIA HPC SDK) is the only compiler that builds the `omp target teams
# loop` kernels; the cuda module brings nsys/ncu along for profiling.
if ! type module >/dev/null 2>&1; then
    source /opt/Modules/v4.3.0/init/bash 2>/dev/null || \
    source /etc/profile.d/modules.sh    2>/dev/null
fi
module load nvidia-hpc-sdk/25.9
module load cuda/12.9.0
