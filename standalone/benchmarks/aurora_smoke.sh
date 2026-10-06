#!/bin/bash
# Aurora first-contact checks, run inside an interactive allocation
# (qsub -I -l select=1 -l walltime=1:00:00 -l filesystems=home:flare -A mom6_anuga_gpu -q debug),
# from standalone/:
#   bash benchmarks/aurora_smoke.sh            # all three steps
#   bash benchmarks/aurora_smoke.sh bind|ab|mpi
#   TRIS=40000000 bash benchmarks/aurora_smoke.sh mpi    # production size
# Needs the binaries built on a login node with the default PE:
#   make intelgpumpi MPICC=mpicc INTEL_AOT=pvc EXTRA_CFLAGS=-DANUGA_OMP_TEAMS_DPF
#        -> bin/bench_intelgpumpi_dpf   (AOT, teams distribute parallel for)
#   make intelgpumpi MPICC=mpicc INTEL_AOT=pvc
#        -> bin/bench_intelgpumpi_teamsloop   (AOT, teams loop)
#   make intelgpumpi MPICC=mpicc EXTRA_CFLAGS=-DANUGA_OMP_TEAMS_DPF
#        -> bin/bench_intelgpumpi_jit   (JIT spir64, DPF: startup-cost check)
# bin/bench_intelgpumpi is the one the harness runs.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 ZE_FLAT_DEVICE_HIERARCHY=FLAT MPIR_CVAR_ENABLE_GPU=0
BIND="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
ARGS="--case dam --scheme ader2 --flux scatter --warmup 10 --repeat 3 --phases"
what=${1:-all}
TRIS=${TRIS:-1000000}                 # triangles per tile (default ~1M: quick)
NY=2048; NX=$(( (TRIS + 4*NY - 1) / (4*NY) ))

if [[ $what == all || $what == bind ]]; then
    echo "=== 1. binding: expect 12 ranks, one distinct ZE_AFFINITY_MASK each, 8 disjoint cores each"
    mpiexec -n 12 -ppn 12 $BIND gpu_tile_compact.sh bash -c \
        'echo "rank $PALS_RANKID $(hostname) ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK cpus=$(taskset -cp $$ | cut -d: -f2)"' \
        | sort -n -k2
fi

if [[ $what == all || $what == ab ]]; then
    echo "=== 2. one tile, $((4*NX*NY)) tris: teams loop vs teams distribute parallel for (+ JIT build)"
    for b in bench_intelgpumpi_teamsloop bench_intelgpumpi_dpf bench_intelgpumpi_jit; do
        echo "--- $b"
        /usr/bin/time -f "process wall %e s (includes startup/JIT)" \
            mpiexec -n 1 -ppn 1 --cpu-bind=list:1-8 gpu_tile_compact.sh \
            bin/$b --nx $NX --ny $NY --lenx $NX --leny $NY --steps 100 $ARGS 2>&1 \
            | grep -E 'devices|ms/step|kernels|compute_fluxes|extrapolate|halo|drift|process wall|rror'
    done
fi

if [[ $what == all || $what == mpi ]]; then
    echo "=== 3. weak point, 1 node x 12 tiles x $TRIS tris via the harness"
    python3 benchmarks/scaling.py run --machine aurora --study weak --nodes 1 \
        --bin bin/bench_intelgpumpi --tris-per-rank $TRIS --timeout 600 \
        --out benchmarks/results/smoke_aurora
    cat benchmarks/results/smoke_aurora/scaling_aurora.csv
fi
