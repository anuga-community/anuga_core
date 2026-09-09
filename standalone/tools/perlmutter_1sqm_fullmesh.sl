#!/bin/bash
#SBATCH -J anuga_1sqm_full
#SBATCH -A __REPO__                 # <-- NERSC repo, e.g. m1234_g  (REQUIRED)
#SBATCH -C gpu&hbm80g               # A100-80 nodes; see the sizing note below
#SBATCH -q regular
#SBATCH -N 30
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -t 01:30:00
#SBATCH -o %x-%j.out
#
# THE WHOLE DELTA AT 1 m^2 ON PERLMUTTER: 1404 tiles, 17.445 G triangles.
#
# WHY 30 NODES (A100-80).  The 512 B/triangle ceiling model is verified on
# this exact card: 166.4M triangles fit and 169M OOM, predicted 166.3M.  With
# the active-set candidate arrays the cost is 528 B/tri, so the ceiling is
# 161.4M; at a 10% margin call it 145M per GPU.
#
#     17.445 G / 145M  =  120 GPUs  =  30 nodes.
#
# On the standard 40 GB nodes (-C gpu, no hbm80g) halve the capacity: ~60
# nodes.  With the two memory levers described in h200_1sqm_fullmesh.pbs --
# not allocating the RK2 backup arrays under ADER2 (-24 B/tri) and GEOM=fp32
# (-76 B/tri) -- 428 B/tri gives ~199M per GPU and 25 nodes / 50 nodes.
#
# WHY THIS IS NOT SLOWER THAN THE H200 PLAN.  ADER2 + scatter plateaus at
# ~1500 Mcell-steps/s on the A100-80 against 2837 on an H200, so per GPU the
# A100 is 1.89x slower -- but 120 of them aggregate 180 Tcell-steps/s against
# 60 H200s' 170.  Perlmutter is the better machine for this run simply
# because the node count is not capped at 15, which is what stopped the full
# mesh fitting at all.
#
# BEFORE RUNNING: the tile set (391 GB, 1404 files + index.txt) has to be on
# $SCRATCH or CFS -- Globus from gadi /g/data/bm55/jlv900/tiles1sqm.  Set TD
# below to wherever it lands.  index.txt holds BASENAMES, so the directory
# can move freely as long as index.txt travels with the tiles.
set -x
ROOT=${ROOT:-$SCRATCH/mesh_exp/standalone}
TD=${TD:-$SCRATCH/tiles1sqm}
OUT=$ROOT/build/fullmesh_pm
IDX=$TD/index.txt
NP=$((SLURM_NNODES * 4))
mkdir -p "$OUT"; cd "$ROOT"

# Cray: cc drives nvc under PrgEnv-nvidia, and OMPI_CC is ignored, which is
# why the Makefile takes MPICC rather than hard-coding mpicc.
module load PrgEnv-nvidia
module load cudatoolkit
module load craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1     # the halo passes device pointers
export OMP_NUM_THREADS=8

make --no-print-directory CONFIG=gpumpi MPICC=cc GPU_ARCH=cc80 \
     OBJDIR=$OUT/obj BIN=$OUT/bench_gpumpi build 2>&1 | tail -2
BENCH=$OUT/bench_gpumpi
CSV="--csv $OUT/runs.csv"
SRUN="srun -N $SLURM_NNODES -n $NP --gpus-per-node=4 --cpu-bind=cores"

# ---- 1. does it fit, and is it the same physics
# Short and unbalanced: the memory figure and the reference state line.  A
# device-memory failure here is the headline, so it runs before anything else
# spends time.
$SRUN "$BENCH" --tiles $IDX --steps 20 --warmup 0 $CSV > "$OUT/fit20.txt" 2>&1
grep -E "memory|ms/step,|state|volume" "$OUT/fit20.txt" | tee "$OUT/FIT.txt"

# ---- 2. wetness, then the assignment that balances it
$SRUN "$BENCH" --tiles $IDX --steps 60 --warmup 10 --flux scatter --active-set \
    --scheme ader2 --active-every 32 --tile-stats "$OUT/stats.txt" \
    --phases --phases-sync $CSV > "$OUT/bal_tri.txt" 2>&1
python3 tools/tile_assign.py $IDX --stats "$OUT/stats.txt" --nprocs $NP \
    --floor 0.058 --out "$OUT/assign.txt" --split-out "$OUT/split.txt" \
    --split-threshold 0.25 | tee "$OUT/assign.log"

# ---- 3. the production configuration, balanced.  This is the number.
$SRUN "$BENCH" --tiles $IDX --assign "$OUT/assign.txt" --steps 100 --warmup 10 \
    --flux scatter --active-set --scheme ader2 --active-every 32 \
    --phases --phases-sync $CSV > "$OUT/prod.txt" 2>&1

# ---- 4. what each lever is worth on THIS domain and THIS machine
for V in "--scheme rk2 --active-every 32:rk2_k32" \
         "--scheme ader2 --active-every 1:ader2_k1"; do
  FLAGS=${V%%:*}; NAME=${V##*:}
  $SRUN "$BENCH" --tiles $IDX --assign "$OUT/assign.txt" --steps 100 --warmup 10 \
      --flux scatter --active-set $FLAGS --phases --phases-sync $CSV \
      > "$OUT/$NAME.txt" 2>&1
done

{
  echo "=== FULL 1 sqm delta on Perlmutter: $(head -1 $IDX), $SLURM_NNODES nodes, np=$NP ==="
  echo "--- does it fit"; cat "$OUT/FIT.txt"
  for t in prod rk2_k32 ader2_k1 bal_tri; do
    f="$OUT/$t.txt"; echo; echo "--- $t"
    grep -E "memory|ms/step,|sim rate|last dt|mpi_|active_sets|sum  |candidate |active    :|per-rank kernel|volume|state" "$f"
  done
  echo; echo "--- assignment"; cat "$OUT/assign.log"
} | tee "$OUT/SUMMARY.txt"
touch "$OUT/DONE"
echo "== done"
