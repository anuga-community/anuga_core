#!/bin/bash
# CPU regression gate for the MPI slab decomposition.  Runs the serial
# binary as golden and checks np=2 and np=4 against it:
#   cell fluxes / --phases            : atol 0, rtol 0 (bit-exact)
#   tiled mesh (--tiles, np=1/2/4/7)  : same, vs the merged-mesh serial run;
#                                       scatter gated with --ftol 1e-7
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
# ---- tiled distributed mesh (if the coarse Delta tiles have been built) -----
# tools/tiles_to_bmesh.py --tiles cdac_script/mesh_tiles_10000sqm --out build/tiles10k
# tools/tiles_to_bmesh.py --merged cdac_script/mesh_file/10000sqm.msh --out build/delta10k.bmesh
TILES=build/tiles10k/index.txt; MERGED=build/delta10k.bmesh
if [ -f "$TILES" ] && [ -f "$MERGED" ]; then
  ARGS="--steps 40"
  for sc in rk2 ader2; do
    "$BIN" --mesh "$MERGED" $ARGS --scheme $sc --save "$TMP/mc.bin" > /dev/null 2>&1
    "$BIN" --mesh "$MERGED" $ARGS --scheme $sc --flux scatter --save "$TMP/ms.bin" > /dev/null 2>&1
    for np in 1 2 4 7; do
      for mode in "--flux cell" "--phases"; do
        r=$($MPIRUN -np $np "$BIN" --tiles "$TILES" $ARGS --scheme $sc $mode \
              --check "$TMP/mc.bin" --atol 0 --rtol 0 2>&1 | grep tolerance | awk '{print $NF}')
        printf '%-8s %-6s %-6s np=%d %-28s %s\n' tiles10k delta $sc $np "$mode" "$r"; n=$((n+1))
        [ "$r" = OK ] || fail=$((fail+1))
      done
      for mode in "--flux scatter" "--flux scatter --active-set"; do
        r=$($MPIRUN -np $np "$BIN" --tiles "$TILES" $ARGS --scheme $sc $mode \
              --check "$TMP/ms.bin" --atol 0 --ftol 1e-7 2>&1 | grep tolerance | awk '{print $NF}')
        printf '%-8s %-6s %-6s np=%d %-28s %s\n' tiles10k delta $sc $np "$mode" "$r"; n=$((n+1))
        [ "$r" = OK ] || fail=$((fail+1))
      done
    done
  done
else
  echo "(tiled-mesh gates skipped: $TILES / $MERGED not built)"
fi
echo "mpi_verify: $((n-fail))/$n OK"
[ $fail -eq 0 ]
