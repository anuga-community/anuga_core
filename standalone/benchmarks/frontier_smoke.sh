#!/bin/bash
# Frontier first-contact checks, run inside an interactive allocation
# (salloc -A CSC693 -N 1 -t 1:00:00), from standalone/:
#   bash benchmarks/frontier_smoke.sh            # all three steps
#   bash benchmarks/frontier_smoke.sh bind|ab|mpi
#   TRIS=40000000 bash benchmarks/frontier_smoke.sh ab   # production size (slow)
# Needs bin/bench_amdgpumpi and bin/bench_amdgpumpi_teamsloop (built on a login
# node with the same modules; see the amdgpumpi notes in the Makefile).
set -u
cd "$(dirname "$0")/.."
module load PrgEnv-amd amd/10.0.0 rocm/10.0.0
# libfabric via pkg-config: a shell rc that *sets* LD_LIBRARY_PATH (rather
# than prepending) drops the libfabric module's path while the module stays loaded.
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$(pkg-config --variable=libdir libfabric):$LD_LIBRARY_PATH
export OMP_NUM_THREADS=7 MPICH_GPU_SUPPORT_ENABLED=0
SRUN="srun -c 7 --threads-per-core=1 --gpus-per-task=1 --gpu-bind=closest"
ARGS="--case dam --scheme ader2 --flux scatter --warmup 10 --repeat 3 --phases"
what=${1:-all}
TRIS=${TRIS:-1000000}                 # triangles per GCD (default ~1M: quick)
NY=2048; NX=$(( (TRIS + 4*NY - 1) / (4*NY) ))

if [[ $what == all || $what == bind ]]; then
    echo "=== 1. binding: expect 8 ranks, one distinct GCD each, 7 cores each"
    $SRUN -N 1 -n 8 --ntasks-per-node=8 bash -c \
        'echo "rank $SLURM_PROCID $(hostname) ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES cpus=$(taskset -cp $$ | cut -d: -f2)"' \
        | sort -n -k2
fi

if [[ $what == all || $what == ab ]]; then
    echo "=== 2. one GCD, $((4*NX*NY)) tris: teams loop vs teams distribute parallel for"
    for b in bench_amdgpumpi_teamsloop bench_amdgpumpi; do
        echo "--- $b"
        $SRUN -N 1 -n 1 bin/$b --nx $NX --ny $NY --lenx $NX --leny $NY --steps 100 $ARGS \
            | grep -E 'devices|ms/step|kernels|halo|drift'
    done
fi

if [[ $what == all || $what == mpi ]]; then
    echo "=== 3. weak point, 1 node x 8 GCDs x $TRIS tris via the harness"
    python3 benchmarks/scaling.py run --machine frontier --study weak --nodes 1 \
        --bin bin/bench_amdgpumpi --tris-per-rank $TRIS --out benchmarks/results/smoke
    cat benchmarks/results/smoke/scaling_frontier.csv
fi
