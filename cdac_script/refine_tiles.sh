#!/bin/bash
# Retile: split the listed tiles into 4 half-size sub-tiles, mesh ONLY the
# new sub-tiles, reuse every other tile's .msh, merge, and (optionally)
# convert for the miniapp.  The base set must have been made with the same
# --tile/--spacing, and tile/2 must be a multiple of spacing.
#
#   ./refine_tiles.sh <area_m2> <tile_m> <spacing_m> <refine-file> [nprocs]
#   e.g. ./refine_tiles.sh 10000 10000 86.2069 split.txt 8
#
# Writes tiles_<L>sqm_r1/, mesh_tiles_<L>sqm_r1/ (symlinks + new tiles),
# mesh_file/<L>sqm_r1.msh.
set -euo pipefail
cd "$(dirname "$0")"
AREA=$1; TILE=$2; SPACING=$3; REFINE=$4; NPROCS=${5:-$(nproc)}
LABEL=$(python -c "a=$AREA; print(int(a) if float(a).is_integer() else a)")
BASE_MT=mesh_tiles_${LABEL}sqm
TILES=tiles_${LABEL}sqm_r1; MT=mesh_tiles_${LABEL}sqm_r1; OUT=mesh_file/${LABEL}sqm_r1.msh
export OMP_NUM_THREADS=1

[ -s "$REFINE" ] || { echo "nothing to refine ($REFINE is empty)"; exit 1; }
echo "== 1/4 split with refinement ($(wc -l < "$REFINE") tiles)"
rm -rf "$TILES" "$MT"; mkdir -p "$MT"
python split_delta.py --tile "$TILE" --spacing "$SPACING" --outdir "$TILES" --refine-file "$REFINE"

echo "== 2/4 mesh new sub-tiles, link the rest"
NEW=0
for j in "$TILES"/tile_*.json; do
  id=$(basename "$j" .json); id=${id#tile_}
  if [ -f "$BASE_MT/tile_$id.msh" ] && [ "$id" -lt 10000 ]; then
    ln -s "../$BASE_MT/tile_$id.msh" "$MT/tile_$id.msh"
  else
    echo "$j"; NEW=$((NEW+1))
  fi
done | grep json | xargs -P "$NPROCS" -I{} sh -c \
  'python mesh_tile.py "$1" --area '"$AREA"' --outdir '"$MT"' > '"$MT"'/$(basename "$1" .json).log 2>&1 || echo "FAILED $1"' _ {}
echo "   $(ls "$MT"/*.log | wc -l) new tiles meshed, $(ls "$MT"/*.msh | wc -l) total"

NTRI=$(python -c "import json; print(int(2*sum(t['area_m2'] for t in json.load(open('$TILES/index.json'))['tiles'])/$AREA))")
if [ "$NTRI" -gt 500000000 ]; then
  echo "== 3/4 merge SKIPPED (~$NTRI triangles); tiles are in $MT/"
else
  echo "== 3/4 merge -> $OUT"
  python merge_tiles.py "$MT" "$OUT"
fi
echo "== 4/4 done; convert with: python ../standalone/tools/tiles_to_bmesh.py --tiles $MT --out ... / --merged $OUT --out ..."
