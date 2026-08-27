#!/bin/bash
# Split the Delta polygon into tiles, triangulate them in parallel, merge.
#
#   ./make_tiled_mesh.sh <area_m2> [nprocs] [tile_m] [spacing_m]
#
#   ./make_tiled_mesh.sh 300            # 10 km tiles, 20 m spacing -> mesh_file/300sqm.msh
#   ./make_tiled_mesh.sh 1              # 3 km tiles, 1.2 m spacing, NO merge (too big)
#   ./make_tiled_mesh.sh 1 104 3000 1.2 # explicit
#
# Defaults: tile size scales so a tile holds ~2e7 triangles; spacing is
# ~0.8x the natural edge length so Triangle (-Y) never needs to split a
# boundary segment. Merge is skipped when the estimate exceeds 5e8 triangles.
set -euo pipefail
cd "$(dirname "$0")"

AREA=${1:?usage: $0 <area_m2> [nprocs] [tile_m] [spacing_m]}
NPROCS=${2:-$(nproc)}
EDGE=$(python -c "import math; print(math.sqrt(4*$AREA/math.sqrt(3)))")
TILE=${3:-$(python -c "import math; print(int(min(10000, max(3000, round(math.sqrt(1.5e6*$AREA)/1000)*1000)))")}
SPACING=${4:-$(python -c "print(round(0.8*$EDGE, 3))")}
LABEL=$(python -c "a=$AREA; print(int(a) if float(a).is_integer() else a)")

TILES=tiles_${LABEL}sqm
MT=mesh_tiles_${LABEL}sqm
OUT=mesh_file/${LABEL}sqm.msh
export OMP_NUM_THREADS=1

echo "area=$AREA m^2  tile=$TILE m  spacing=$SPACING m  nprocs=$NPROCS"
echo "== 1/3 split"
python split_delta.py --tile "$TILE" --spacing "$SPACING" --outdir "$TILES"

echo "== 2/3 triangulate ($(ls "$TILES"/tile_*.json | wc -l) tiles, $NPROCS at a time)"
mkdir -p "$MT"
ls "$TILES"/tile_*.json | xargs -P "$NPROCS" -I{} sh -c \
  'python mesh_tile.py "$1" --area '"$AREA"' --outdir '"$MT"' > '"$MT"'/$(basename "$1" .json).log 2>&1 || echo "FAILED $1"' _ {}
NFAIL=$(grep -L "wrote" "$MT"/*.log | wc -l)
NTRI=$(grep -h "triangulated" "$MT"/*.log | awk '{s+=$2} END {print s}')
echo "   $NTRI triangles total, $NFAIL failed tiles"
[ "$NFAIL" -eq 0 ] || { echo "see failing logs in $MT/"; exit 1; }

if [ "$NTRI" -gt 500000000 ]; then
  echo "== 3/3 merge SKIPPED: $NTRI triangles is too large for a single .msh; tiles are in $MT/"
  exit 0
fi
echo "== 3/3 merge -> $OUT"
python merge_tiles.py "$MT" "$OUT"
