#!/bin/bash
#SBATCH -J stream
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -A CSC693
#SBATCH -o tools/stream/stream.%j.log
cd /lustre/orion/csc693/scratch/jg4/anuga_core/standalone/tools/stream
module load PrgEnv-amd amd/10.0.0 rocm/10.0.0
S1="srun -N 1 -n 1 -c 7 --threads-per-core=1 --gpus-per-task=1 --gpu-bind=closest"
S8="srun -N 1 -n 8 -c 7 --threads-per-core=1 --gpus-per-task=1 --gpu-bind=closest"
echo "=== 1 GCD, OpenMP";  $S1 ./stream_omp 28
echo "=== 1 GCD, OpenMP, 4x larger"; $S1 ./stream_omp 30
echo "=== 1 GCD, HIP";     $S1 ./stream_hip 28
echo "=== 8 GCDs concurrently, OpenMP"; $S8 bash -c './stream_omp 28 | sed "s/^/[$SLURM_PROCID] /"'
