#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# SLURM job script for ANUGA GPU run on CDAC Param Prabha
#
# TOPOLOGY (edit these 3 lines for different configs):
#   --nodes          : number of GPU nodes to allocate
#   --ntasks-per-node: GPU processes per node (must equal --gres=gpu:N)
#   --gres=gpu:N     : GPUs per node to reserve
#
# COMMON CONFIGS:
#   2 nodes × 2 GPUs = 4 ranks  (testing, small runs)
#   4 nodes × 2 GPUs = 8 ranks  (production 300sqm)
#   8 nodes × 2 GPUs = 16 ranks (large 100sqm mesh)
#
# IMPORTANT: --ntasks-per-node MUST equal --gres=gpu:N
#            Each MPI rank owns exactly one GPU.
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH --output=anuga-gpu_sc26.%J.out
#SBATCH --error=anuga-gpu_sc26.%J.err
#SBATCH --exclusive
#SBATCH --gres=gpu:2
##SBATCH --reservation=anuga_gpu

# ── Print job info ────────────────────────────────────────────────────────────
echo "SLURM_JOBID          = $SLURM_JOBID"
echo "SLURM_JOB_NODELIST   = $SLURM_JOB_NODELIST"
echo "SLURM_NNODES         = $SLURM_NNODES"
echo "SLURM_NTASKS         = $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE= $SLURM_NTASKS_PER_NODE"
echo "SLURMTMPDIR          = $SLURMTMPDIR"

# ── MPI / UCX transport settings ─────────────────────────────────────────────
# Use shared-memory + self transport within a node; ob1 pml for inter-node.
# UCX_MEMTYPE_CACHE=n avoids a known UCX bug with CUDA memory type detection.
export UCX_TLS=sm,self
export OMPI_MCA_pml=ob1
export UCX_MEMTYPE_CACHE=n
export UCX_UD_RX_QUEUE_LEN=16000

# ── OpenMP / GPU settings ────────────────────────────────────────────────────
# mandatory: abort if GPU offload fails (catches silent CPU fallback)
export OMP_TARGET_OFFLOAD=mandatory

# One CPU thread per MPI rank — all compute is on GPU, extra threads
# only add NUMA contention on the host.
export OMP_NUM_THREADS=1

# ── Stack / core limits ───────────────────────────────────────────────────────
ulimit -s unlimited
ulimit -c unlimited
ulimit -l unlimited

# ── Activate environment ──────────────────────────────────────────────────────
source /home/appadmin/Harsha/sc26-perf-optimizations/env.sh

# ── Simulation date (pass as argument or default to today) ───────────────────
SIM_DATE=${1:-"15-09-2021"}

# ── Run ───────────────────────────────────────────────────────────────────────
# --bind-to core        : pin each rank to a distinct core set
# --map-by ppr:2:node   : 2 ranks per node (matches ntasks-per-node)
# opal_cuda_support 1   : enable CUDA-aware MPI in OpenMPI
(time mpirun \
    --mca opal_cuda_support 1 \
    -np $SLURM_NTASKS \
    --bind-to core \
    --map-by ppr:${SLURM_NTASKS_PER_NODE}:node \
    python run_model_active_cell.py ${SIM_DATE}) \
    2>&1 | tee test_GPU_sc26_${SLURM_JOBID}.log
