#!/bin/bash
# CPU regression gate for the MPI slab decomposition.  Runs the serial
# binary as golden and checks np=2 and np=4 against it:
#   cell fluxes / --phases            : atol 0, rtol 0 (bit-exact)
#   scatter / scatter + --active-set  : atol 1e-6 vs the serial *scatter*
#     run.  Scatter accumulates the three edge contributions per cell with
#     atomics, so it is only ever exact to summation order: a 1-thread vs
#     4-thread serial scatter run differs by the same ~1e-9 (dam, ader2) the
#     MPI runs show.  Cell-based fluxes have no such freedom, hence atol 0.
# Usage: tools/mpi_verify.sh [BIN] [MPIRUN]     (defaults: bin/bench_cpumpi)
set -u
BIN=${1:-bin/bench_cpumpi}
MPIRUN=${2:-"mpirun --bind-to none"}
TMP=${TMPDIR:-/tmp}/mpi_verify.$$
mkdir -p "$TMP"; trap 'rm -rf "$TMP"' EXIT
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
fail=0; n=0
for dims in "120 100" "100 100"; do set -- $dims; ARGS="--nx $1 --ny $2 --steps 40"
for c in dam river; do for sc in rk2 ader2; do
  "$BIN" $ARGS --case $c --scheme $sc --save "$TMP/gc.bin" > /dev/null 2>&1
  "$BIN" $ARGS --case $c --scheme $sc --flux scatter --save "$TMP/gs.bin" > /dev/null 2>&1
  for np in 2 4; do
    for mode in "--flux cell" "--phases"; do
      r=$($MPIRUN -np $np "$BIN" $ARGS --case $c --scheme $sc $mode \
            --check "$TMP/gc.bin" --atol 0 --rtol 0 2>&1 | grep tolerance | awk '{print $NF}')
      printf '%-8s %-6s %-6s np=%d %-28s %s\n' "$1x$2" $c $sc $np "$mode" "$r"; n=$((n+1))
      [ "$r" = OK ] || fail=$((fail+1))
    done
    for mode in "--flux scatter" "--flux scatter --active-set"; do
      r=$($MPIRUN -np $np "$BIN" $ARGS --case $c --scheme $sc $mode \
            --check "$TMP/gs.bin" --atol 1e-6 --rtol 0 2>&1 | grep tolerance | awk '{print $NF}')
      printf '%-8s %-6s %-6s np=%d %-28s %s\n' "$1x$2" $c $sc $np "$mode" "$r"; n=$((n+1))
      [ "$r" = OK ] || fail=$((fail+1))
    done
  done
done; done; done
echo "mpi_verify: $((n-fail))/$n OK"
[ $fail -eq 0 ]
