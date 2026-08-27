# PICKUP — session handoff (updated 2026-08-27)

Written because the interactive node hosting the Claude session was about to
expire.  State of everything in flight, how to check it, what's next.

## 1. MPI-across-GPUs (this worktree, branch `experiment/opt`)

**Done and committed (`9b8e87c4`):** C-only MPI layer for the standalone
miniapp — no mpi4py anywhere.

- `standalone/src/bench_mpi.h` — thin wrapper (init/rank/size/reductions/
  gatherv; identity stubs without `-DHAVE_MPI`).  ALL driver MPI goes
  through it.  This is the seed of the mpi4py-free layer for full ANUGA.
- Slab decomposition along x (cell id = i*n+j → contiguous per-rank id
  ranges), 1 ghost column per interior side, per-substep centroid exchange
  via the gpu layer's own `gpu_exchange_ghosts` + one end-of-step call.
- **Bit-exact**: 18 checks at `--atol 0 --rtol 0` — np=2,4 vs serial golden
  on dam/dambumps/river × rk2/ader2/euler (CPU build).
- Build: `make cpumpi` / `make gpumpi` (mpicc drives gcc/nvc via OMPI_CC).
  `source standalone/my_claude_env.sh` first (nvhpc 25.9 + cuda 12.9.0 +
  openmpi 5.0.5; non-login shells need
  `source /opt/Modules/v4.3.0/init/bash` for `module` to exist).
- v1 restrictions (enforced in bench.c): generated mesh, `--order row`,
  cell fluxes, no --active-set/--phases/--cuda-extrap under MPI.

**First GPU attempt (job 177503337) found a REAL BUG in production
gpu_halo.c**: the GPU_AWARE_MPI pack/unpack kernels dereference the flat
index arrays on device but gpu_halo_init never mapped them -> the first
exchange aborted (cuStreamSynchronize).  The path had never run on a GPU
before.  Fixed + committed; CPU bit-exactness re-verified.

**DONE: job 177517465 (4xH200 scaling)** -- gates OK, strong 88%/92% at
np=4 (64M/144M), weak 93%.  Banked in standalone/README.md (`26c6d7fc`).

**DONE (`271236f2`, 2026-08-27): scatter fluxes, --phases and --active-set
under MPI.**  Ghost-aware owned_edges in setup.c; stepped loops do the dt
allreduce + mid-step exchange; slab generator roundoff bug fixed (non-square
meshes were off by 1e-12 even in cell mode).  `tools/mpi_verify.sh` = 64
CPU checks: cell/--phases bit-exact, scatter/active-set at atol 1e-6
(atomic summation order -- same ~1e-9 as serial 1- vs 4-thread scatter).

**DONE: job 177579044 (4xH200, scatter/phases/active under MPI)** -- all
16 GPU gates OK; strong scaling 87-92% for everything except river+active
(38%): the x-slab gives rank 0 the whole reservoir, so it holds ~53% active
cells vs the 18% average.  Measured proof that active-set load must be
partitioned by wetness, not triangle count.  Banked in standalone/README.md.

**DONE (`dfc3aa60`, 2026-08-27): tiled distributed mesh (`--tiles`).**
The launcher for the 1 sqm delta: ranks load their tiles + bbox-touching
neighbour tiles, stitch on exact vertices, ghosts = vertex-ring of owned,
halo lists derived symmetrically.  Bit-exact vs merged mesh (np 1/2/4/7,
cell/phases); scatter to --ftol 1e-7.  --tile-stats + tools/tile_assign.py
= the wetness-weighted assignment; the heaviest-tile bound says which
tiles to re-split.  Coarse synthetic delta data: cdac_script/
{tiles,mesh_tiles}_{10000,300}sqm + mesh_file/{10000,300}sqm.msh (built
here, NOT the user's provenance-restricted 300 sqm data; bed/stage are
analytic in tools/tiles_to_bmesh.py), converted to standalone/build/
tiles10k, delta10k.bmesh, tiles300, delta300.bmesh.  mpi_verify.sh: 96/96.

**DONE (`c04a3bc8`): ghost rims no longer seed the active set (exact,
96/96); quadtree retile (split_delta.py --refine-file + refine_tiles.sh);
coarse loop: 131% imbalance -> 13% weighted -> tiles 30/31 split, refined
mesh bit-exact vs merged at np=4/7.  Two real 1 sqm tiles (10, 11; 13.8M
each) np=2 bit-exact vs stitched np=1 (build/pair1sqm).

**DONE: GPU jobs 177612520 / 177625536 / 177626077** (build/mpitiles,
build/tilesbal/tiles300{,_r1}): all tiled gates OK at 58M on 4xH200;
wetness-weighted contig assignment 50% -> 9% kernel imbalance; lpt same
balance but 4x ghosts -> 1.6x slower wall (contiguity mandatory; contig is
now the tile_assign default, Morton order of tile centres).  H200 cost
model floor 0.073.  Refined set (tiles 30/31 split): gates OK; balance
residual is model error not granularity at 1% wet.

**DONE: job 177630729** -- refined set, Morton contig: 10% kernel
imbalance, wall 1.66 ms/step (best run); predicted per-rank actives matched
measured exactly.  All banked in standalone/README.md.  No jobs in flight.

**DONE: all 1404 1 sqm tiles converted** (job 177634970, bm55): 17.45G
triangles, 391 GB compact v2 on /g/data/bm55/jlv900/tiles1sqm (index.txt).
scratch/bm55 quota is nearly full (706/1024 GiB) -- keep big outputs on gdata.

**IN FLIGHT: job `177636235`** -- `tools/h200_1sqm_rehearsal.pbs`: 16 real
1 sqm tiles (~220M tris) around the synthetic lake on 4xH200: np=1 golden,
np=2/4 atol-0 gates, active-set timing tri-balanced vs weighted contig.
Results `standalone/build/rehearsal/` (SUMMARY.txt, runs.csv, DONE).

**Next steps:**
1. Read rehearsal results; then scale the subset up (more tiles / 8 GPUs)
   and add per-rank output (max depth, stage at yieldsteps).  Elevation: replace the analytic
   bed in tools/tiles_to_bmesh.py by raster sampling once the user provides
   a path on their side (their data cannot be shared with Claude).
2. Memory: neighbour tiles are read whole then discarded -- fine at 300
   sqm, ~20 GB transient/rank at 1 sqm; extract boundary strips at
   conversion time if that bites.
3. Same C-only layer under full ANUGA (drop mpi4py); remaining miniapp MPI
   restrictions: --order row generated mesh only, no --flux edge/--cuda-extrap.

## 2. This worktree vs the main repo

- `/scratch/bm55/jlv900/dev/mesh_exp` = worktree for `experiment/opt`
  (miniapp + all optimization work).  Also holds the user's real
  catchment: `cdac_script/Delta_11372_sqkm.csv` (11,372 km², commit
  "Delta").
- `/scratch/bm55/jlv900/dev/anuga_core` = main checkout, currently on
  `feature/gpu-active-set`.

## 3. Branches in the main repo (all suite-green, ready to push/PR)

| branch | contents |
|---|---|
| `feature/gpu-step-optimizations` | kernel fusion + ADER2 + scatter + active-set C machinery extracted off develop for upstream (2 commits, 10 files, default behaviour bit-identical) |
| `feature/parallel-sww-merge` | `sww_merge_parallel(..., workers=N)`: ~2.7× measured, byte-identical output, progress bar, per-phase diagnostics; unit tests |
| `feature/gpu-active-set` | `domain.set_use_active_set()` production wiring + rain warnings + tests (stacked on gpu-step-optimizations) |

User pushes themselves (Claude's node can't reach github ssh): `! git push -u origin <branch>`.

## 4. Collaborator deliverables (sww merge)

- `tmp_artifacts/merge_sww_parallel.py` (main repo) — their merge script
  upgraded: --workers auto-detect, --chunk_size, --delete_old opt-in,
  mpi4py-segfault fix, trailing-slash fix.  Needs branch
  `feature/parallel-sww-merge`'s `anuga/utilities/sww_merge.py` (pure
  Python drop-in) to actually parallelise; falls back to serial otherwise.
- Their 3 h merge diagnosis: old-version RAM blowout / login-node env; on
  the new code 50 GB ≈ 5 min (measured 52 GB on an SR node).
- 52 GB synthetic test set: `tmp_artifacts/sww_synth/` (12 × 4.33 GB,
  `synth_run_P12_*.sww`) + `gen12.py` to regenerate.
- If they still see 20–30 min: have them run once with the new
  sww_merge.py (verbose) and send the "merge diagnostics" block — it
  names the bottleneck (read/scatter/write/RSS).

## 5. Gotchas worth re-reading before touching anything

- `claude/KNOWN_ISSUES.md` (main repo): PyMem_Malloc struct-init trap,
  ninja clean-rebuild after sw_domain.h/gpu_domain.h changes,
  OMP_TEAMS_THREAD_LIMIT corruption.
- GPU builds: never benchmark without a physics gate (lake ~1e-13).
- The editable install's build dir: `anuga_core/build/cp313`
  (`ninja -t clean && ninja` after header changes).
- conda env: `anuga_env_3.13`; GPU tests in-process need
  `ANUGA_GPU_TESTS_ISOLATED=1` on a gpu_offload build.

## 6. Artifacts

- Cross-vendor scaling chart: https://claude.ai/code/artifact/d56522e3-f2f0-4232-b0ff-896111b33ac1
- V100 ceiling report: https://claude.ai/code/artifact/d396add3-d6a3-4fc6-a0cd-daa67cbeb8fb
