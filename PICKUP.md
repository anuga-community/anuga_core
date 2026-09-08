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

**DONE: job 177636235 (1 sqm rehearsal, 9 tiles / 89.5M)**: np=2/4
bit-exact vs np=1 on real tiles; 488 B/tri device, 543 B/tri host, 0.45
us/tri build.  Block was 100% wet -> balance not measurable; 97% imbalance
= granularity (9 tiles / 4 ranks).  Banked in README.

**DONE: job 177636649 (2 nodes, 8 H200, 26 tiles / 301M real 1 sqm)**:
np=8 bit-exact vs np=4 across nodes; full mesh 89% efficient 4->8 GPUs;
active-set weighted contig 27.3 ms vs 44.8 tri-balanced (1.64x); residual
74% = wet-tile granularity; 13 tiles flagged for split
(build/rehearsal2/split.txt).  No jobs in flight.

**DONE: 13 flagged 1 sqm tiles refined** -> 50 sub-tiles, 1441 tiles /
17.445G tris; gdata tile set + index.txt updated in place (parents removed);
lake block now 63 tiles, predicted contig balance 1.05 (was 1.22).

**DONE: job 177650566 (refined block, 63 tiles, 8 H200 / 2 nodes)**:
np=8 bit-exact vs np=4; weighted contig 24.6 ms/step vs 41.9 tri-balanced
(1.71x), kernel imbalance 13.9% (74% before the split).  The whole
coarse->assign->split->assign loop is proven on real 1 sqm tiles.  Banked
in README.  No jobs in flight.

**DONE (2026-08-28): multi-scale tiling** (`split_delta.py --areas`,
nested 4^k areas, finer-of-two lattice on cut lines): delta at 1024/256/64
-> 20.35M tris, conforming, tiled bit-exact vs merged.  **dt is set by
outline/cut-line corner triangles**: needle fix + MARGIN 0.5 -> dt 0.032 ->
0.0445 s (1.38x).  Small-triangle PoC (README table): margin 0.5 = the lever (1.38x dt), -q32
useless, per-tile vertex dropping breaks conformity (rejected), global
outline simplify(s/4) is safe and removes notches; tile_quality.py is the
gate.  Figures fig9 (transition), fig10 (portability), fig11 (corner).  Deck: posters_and_slides/anuga_1sqm_tiles (18+ frames).
Ganges framing: ~2M km^2 = ~3T tris uniform 1 m^2; multi-scale ~185G.

**DONE: job 177741904 (2026-08-30, 15 nodes / 60 H200, 12.39 G tris)** --
the largest run so far, 41x the previous one.  Western 1035 tiles
(x < 126 km), 28 min, 2532 SU, exit 0, `standalone/build/scale15/`.
Volume drift 6.9e-14 / 6.4e-15; state line character-identical across
full-mesh and both active-set runs.  488 B/tri device confirmed exactly at
two rank sizes.  Full mesh 173.5 ms/step (71.4 Gcell-steps/s); active-set
weighted contig **25.1 ms/step, 6.9x** the full mesh; 1.24% wet.
Three findings, all banked in README:

1. **48 triangles set dt for the whole simulation, and the fix is now
   measured.**  dt = 6.55e-4 s = 38 wall-days per simulated day.
   tile_quality.py over the 47 wet tiles (inradius -- note `quality()`
   returns `(area, inradius, angle, centroid)`, use [1]): median inradius
   0.338 m, governing triangle a **0.870 mm needle in tile 1171**, a
   25-triangle coastal fragment, 0.0026 of median.
   **Controlled A/B done (2026-08-30):** split_delta re-run over the same
   polygon at the shipped settings (MARGIN 0.3, from git 74cd2175) and at
   today's defaults, identical sub-tiles meshed at --area 1.0.  Control
   reproduces the shipped bmesh to 4 sig figs (11096: 9.973 vs 9.9727 mm).
   Result on the 8 worst wet tiles: 1171 0.870 -> 152.473 mm (175x),
   10966 4.767 -> 193.974, 11096 9.973 -> 195.744, 212 76.089 -> 136.786.
   All clear the 0.5 gate; triangle count unchanged to 0.07% -- **the fix
   is free**.  Governing inradius 0.870 -> 136.786 mm = **dt 157x ->
   ~0.10 s, i.e. 38 wall-days per simulated day becomes ~5.8 hours**.
   Artifacts: cdac_script/tiles_{1sqm_fix,ctrl_r1,fix_r1,fix_r2}/,
   tmp_local_artifacts/measure.py.  This is the top priority.
2. **The active-set rebuild is 85.8% of kernel time** (12.14 of 14.15 ms);
   a rank with 248M dry triangles and 0 active cells still costs 14.4
   ms/step.  Balance work is finished (the floor bounds it at ~12 ms);
   the lever is rebuilding every k steps against a ring-k dilated set
   (CFL makes ring-k exact), worth ~3-4x.  Kernels are 58% of wall -- the
   other 11 ms is halo + per-substep dt allreduce over 60 ranks.
3. **Cost model form validated, constant recalibrated.**  Fit on 60 ranks:
   0.0582 ns/cell floor + 1.0058 ns/active cell, reproducing the measured
   spread to 0.3%.  `tile_assign.py --floor 0.073` is 26% high at 1 m^2;
   `--floor 0.058` predicts max kernel 19.23 -> 16.90 ms (1.14x, free).

**Full mesh does not fit 15 nodes**: 17.45 G at 488 B/tri = 132 GiB/GPU vs
109.89 proven.  At ~120 GiB/GPU, 15 nodes hold 15.8 G (91% of the delta);
the full mesh needs 17 nodes.  Multi-scale tiling is the way in, not nodes.

**DONE (2026-08-30): old 1 sqm mesh artifacts cleared, regeneration job
written.**  `cdac_script/mesh_tiles_1sqm{,_r1}` (~520 GB of .msh) deleted --
scratch 733 -> 498 GiB, so the new .msh set fits.  The gdata bmesh set
`/g/data/bm55/jlv900/tiles1sqm` (391 GB) is **still there**: Claude's
permission classifier blocks a recursive delete of that size, so the user
must run `rm -rf /g/data/bm55/jlv900/tiles1sqm` themselves.  The old index
files are preserved in `standalone/build/scale15/old_indexes/` (they define
the scale15 subsets and the old per-tile triangle counts).

`cdac_script/regen_1sqm.pbs` (normalsr, 104 cores, 6 h) does the whole
regeneration from the already-written split `tiles_1sqm_fix/` (1404 tiles):
mesh at 1 m^2 -> quality gate -> compact bmesh on gdata -> assemble index.
It **aborts if the gdata output dir is non-empty**, so it cannot mix a new
set into the old one.  `tile_quality.py --jobs N` is now parallel (verified
byte-identical to serial); the serial scan would have taken hours over
17.4 G triangles.

**DONE (2026-09-08): the active-set rebuild is amortized -- 2.6x
(`94125578`, `--active-every K`).**  Finding 2 above is closed.  Rebuilding
every K steps only pays if the extra rings are DILATED ON A LIST
(`core_active_dilate_rings`, breadth-first over the frontier); done with a
sweep per ring it caps at 1.56x, which is why the 3-4x estimate needed
rework.  The candidate = ring-2 set + 2K+2 rings is a provable superset for
K steps, the per-step rebuild runs over it and yields the SAME SETS.  Two
ordering effects, both measured, neither visible to a correctness test: the
BFS list order cost 22% on the step kernels (fixed by re-emitting the
candidate in index order), and deriving edges from the cell list cost 2.2x
on scatter (fixed by narrowing the edge ARRAY instead).  V100, 58.2M real
delta tris, 1.00% active, 200 steps: rebuild 12.50 -> 1.57 ms (8x), step
16.58 -> 6.36 ms (**2.61x**) at K=32; K=16-32 flat optimum.  Sparse-regime
only (at 19.8% active there is nothing to win).  `--active-verify` compares
every restricted rebuild against a full scan: zero mismatches on CPU, GPU
and MPI np=2/4.  Primitives are in the shared `gpu/core_kernels.c`;
`feature/gpu-active-set` still rebuilds every step and can adopt them.

**DONE: job 178439403 (regen_1sqm, 2026-09-08) -- the 1 sqm set is
regenerated with the corner fix.**  1404 tiles, **17.445 G triangles**, 391
GB on `/g/data/bm55/jlv900/tiles1sqm`, 0 failed conversions, `DONE` written.
Smoke-tested on the V100 (4-tile subset, 14.95M tris: loads, stitches, steps).

*The A/B is confirmed exactly and the headline is 71x, not 157x.*  Tile 1171
(the old governor, a 25-triangle coastal fragment, 76% active) went
**0.870 mm -> 152 mm**, against the A/B's predicted 152.473.  But the
whole-set gate -- which the 8-tile A/B could not do -- finds smaller needles
in OTHER wet tiles:

| tile | old-run active | new min inradius |
|---|---|---|
| 1167 |  1.09% | **0.062 m**  <- the new governor |
| 665  |  0.25% |   0.064 m |
| 1248 |  (not in the 15-node subset) | 0.065 m |
| 1383 |  (not in the subset) | 0.077 m |
| 243  | **100%** |   0.094 m |
| 1171 | 76%    |   0.152 m |

So dt goes 6.55e-4 -> ~0.047 s, **71x** (62 mm / 0.870 mm), or 108x if
tile 1167's 1% wet part misses its own worst triangle and tile 243 governs.
**38 wall-days per simulated day becomes ~13 hours**; with the ring-k
amortization on top (~1.7x end-to-end at 60 ranks, since 42% of that wall is
halo + allreduce that does not shrink) **~7.6 hours**.

35 tiles fail the gate, median inradius 0.34 m throughout: another targeted
corner-fix pass on the wet ones is worth up to a further ~3.6x on dt (0.094
-> 0.34 m), and is the obvious next lever if more is wanted.

**Two things this set does NOT carry:**
1. The 13-tile load-balance refinement (the old 1441-tile index) is NOT in
   it -- this is the 1404-tile original split re-meshed.  The
   coarse->assign->split->assign loop has to be re-run on the new set before
   the hero run (it was worth 1.71x, imbalance 74% -> 13.9%).
2. `cdac_script/mesh_tiles_1sqm_fix` holds **521 GB of .msh** intermediates;
   scratch is at 730/1024 GiB.  Removable now that index.txt checks out.

**Superseded: job 178439403 (regen_1sqm, normalsr 104c, 6 h, started
2026-09-08).**  The old 391 GB gdata set was deleted by the user, so the
guard passed.  Meshes `cdac_script/tiles_1sqm_fix/` (1404 tiles) at 1 m^2 ->
quality gate -> compact bmesh on `/g/data/bm55/jlv900/tiles1sqm` -> index.
Writes `DONE` on success.  Check `QUALITY.txt` for the governing inradius:
the A/B predicts 0.870 -> ~137 mm, i.e. dt 157x.

**DONE (2026-09-08): ADER2 composes with the amortized rebuild** -- rk2 K=1
1.083 -> rk2 K=32 2.835 -> **ader2 K=32 4.486** simulated s per wall s on the
58M delta tiles (V100), gated at K=8/32.  The 15-node run used the default
rk2, so this is free and unexploited there.

**DONE (2026-09-08): the MPI cost is now measured, not inferred.**  bench.c
times `mpi_halo` and `mpi_dt_allreduce`; `--phases-sync` bills the wait for
the slowest rank to `mpi_wait` separately, because a collective absorbs
imbalance and would otherwise be blamed for it.  The breakdown closes to
100% under MPI.  First reading (2 ranks / 1 V100): **halo 24.6%, allreduce
0.0%** -- the halo dominates, so do NOT go amortize the allreduce.

**DONE: job 178473399 -- the MPI split at rank scale.**  8 H200 / 2 nodes,
301M tris, 77%-wet lake block; all gates OK (np8 bit-exact vs np4, K=8/32 OK,
0 verify mismatches).  ms/step rk2 31.26 -> **ader2 17.62 (1.77x)**.
- halo: 4.15 ms (13%) under rk2 -> **0.34 ms (1.9%) under ader2**, 12x, since
  rk2 also exchanges mid-step and that exchange absorbs imbalance.
- **dt allreduce 0.008-0.024 ms (0.0-0.1%): the amortize-the-allreduce idea
  is dead.**  Do not spend time on it.
- **mpi_wait (imbalance) is now the largest MPI item**: 16% rk2, 25% ader2.
  The partitioner, not the network -- wetness balancing gave 13.6M..124M
  triangles/rank and the rebuild's per-cell floor bills the triangle-heavy
  rank for its dry cells.
- The block is 77% wet so the candidate hits 99.67% and --active-every gains
  nothing: the control confirming it is a sparse-regime lever.  It should
  also SHRINK the imbalance on the sparse hero domain, since the floor term
  is what drives it.
Next: re-run this matrix on a SPARSE subset (the western block, 1.24% wet)
with a floor recalibrated for the amortized rebuild, then the hero run on
ader2 + --active-every 32.

**Superseded: job 178473399 (`tools/h200_mpi_phases.pbs`, 2 nodes / 8 H200,
90 min).**  The same question at rank scale: regenerated 1 sqm lake block
(26 tiles, 301M tris), {rk2, ader2} x {K=1, K=32}, wetness-balanced,
--phases --phases-sync, with np=4-vs-np=8 and K-vs-K=1 gates and
--active-verify.  Output `standalone/build/mpiphases/SUMMARY.txt`, `DONE`
marker.  It answers: what is the halo actually worth at scale, and does
ADER2 halve it (one flux call per step = half the exchanges AND half the
allreduces).

**Next steps:**
1. ~~Regenerate the 1 sqm tiles~~ (job 178439403 in flight, above).  When it
   lands: confirm the gate, then re-run the 15-node rehearsal with
   `--active-every 32` to get both the dt fix and the 2.6x together.  Then
   smooth the area map to one level per neighbour (16:1 jumps make an
   elongated first layer).
1b. **Old step 1, kept for reference: regenerate the 1 sqm tiles with the corner fix (MARGIN 0.5 +
   simplify s/4) -- measured at 157x on dt, not 1.38x.**  The 1.38x was
   the coarse multi-scale PoC; at 1 m^2 the pre-fix needles are 0.0026 of
   the median inradius and 48 of them set dt for 12.4 G triangles.  Do not
   wait for the real resolution map -- a uniform 1 m^2 regeneration with
   the current defaults already buys the two orders of magnitude, and the
   split for it is already done (`cdac_script/tiles_1sqm_fix/`, 1404
   tiles).  Remaining: mesh the 1404 tiles (26 min on 104 cores), convert
   to bmesh (18 min), gate with tile_quality.py over the whole set to
   confirm no untested tile becomes the new limiter.  Needs ~400 GB free
   on gdata (currently 622/1024 GiB used) -- the old set must go first.
   Then smooth the area map to one level per neighbour (16:1 jumps make an
   elongated first layer).
2. Per-rank output (max depth, stage at
   yieldsteps) and the full-basin coarse->assign pass for all 1441 tiles.  Elevation: replace the analytic
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
