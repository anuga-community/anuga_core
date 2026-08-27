# ANUGA shallow-water miniapp

A standalone benchmark and correctness harness for the shallow-water
OpenMP/GPU-offload kernels. It compiles the **production kernel sources**
(`anuga/shallow_water/gpu/*.c`) with a small C driver — no meson, no Cython, no
Python, no MPI — so the optimise → profile → verify loop takes seconds instead
of a full package rebuild.

The timestep it runs is `gpu_evolve_one_rk2_step()`: the same C entry point the
Python mode-2 (`unified`) path calls, with DE1 defaults (RK2, CFL 1.0,
`beta_* = 1.0`).

---

## Quick start

```bash
source env.sh                 # nvidia-hpc-sdk + cuda modules (gadi)

make cpu                      # gcc, host OpenMP          -> bin/bench_cpu
make gpu                      # nvc, OpenMP target offload -> bin/bench_gpu

./bin/bench_gpu --nx 500 --ny 500 --steps 50 --phases
```

```
ANUGA shallow-water miniapp -- OpenMP target offload
  mesh      : 500 x 500 cross -> 1000000 triangles, 4000 boundary edges
  case      : dam, 1000 x 1000 m, manning 0.03
  scheme    : rk2, CFL 1, DE1 limiter betas
  devices   : 1 visible, using 0

  timed     : 50 steps (+5 warmup) in 0.2319 s
              4.6382 ms/step, 215.601 Mcell-steps/s
  flops     : 66.850 GFLOP over the timed loop, 288.26 GFLOP/s

  per-kernel breakdown (per step, averaged over 50 steps)
    backup              0.0689 ms     1.5%
    protect             0.1687 ms     3.6%
    extrapolate         1.6866 ms    36.4%
    boundary            0.0299 ms     0.6%
    compute_fluxes      1.9954 ms    43.0%
    manning             0.2221 ms     4.8%
    update              0.3300 ms     7.1%
    saxpy               0.1361 ms     2.9%

  volume    : 7500000 -> 7500000 m^3 (drift -1.738e-15 relative)
```

## Build targets

| target          | compiler | what it builds                                       |
|-----------------|----------|------------------------------------------------------|
| `make cpu`      | gcc      | `-DCPU_ONLY_MODE`, host `omp parallel for`           |
| `make gpu`      | nvc      | `-mp=gpu -gpu=<arch>`, `omp target teams loop`        |
| `make ompcpu`   | nvc      | `-mp=multicore`, host-only (A/B against the gpu build)|
| `make clanggpu` | clang    | LLVM nvptx offload                                    |
| `make amdgpu`   | amdclang | `-fopenmp --offload-arch=<gfx>` (autodetected), CDNA  |
| `make intelgpu` | icx      | `-fiopenmp -fopenmp-targets=spir64`, PVC              |
| `make cpumpi`   | mpicc/gcc| the cpu build with `-DHAVE_MPI` (slab decomposition)  |
| `make gpumpi`   | mpicc/nvc| the gpu build with `-DHAVE_MPI`, one GPU per rank     |

`GPU_ARCH` is autodetected from `nvidia-smi` (`cc70` on a V100, `cc90` on an
H100); override with `make gpu GPU_ARCH=cc80`. Each config has its own object
directory, so switching back and forth does not force a rebuild.

Other knobs: `make cpu CC=clang`, `make gpu EXTRA_CFLAGS=-DNVTX_ENABLED`
(NVTX ranges around every kernel, needs the nvtx3 headers on the include path).

## The optimise → verify loop

Three levels of checking, cheapest first.

**1. Built-in invariants** — printed on every run, no reference needed:

- `volume` drift: the scheme is conservative, so relative drift should stay at
  round-off (`~1e-16`). A broken flux kernel shows up here immediately.
- `--case lake`: still water over a bumpy bed. The scheme is well-balanced, so
  `max |momentum|` must stay at round-off (`~1e-14`). This is the sharpest
  cheap test there is — it fails on any error in the geometry, the limiter, or
  the bed-slope terms.
- NaN count and stage range.

**2. Golden-file regression** — freeze a known-good result, then diff after
each change:

```bash
./bin/bench_gpu --nx 200 --ny 200 --steps 50 --save golden.bin
# ... optimise a kernel, rebuild ...
./bin/bench_gpu --nx 200 --ny 200 --steps 50 --check golden.bin
```

`--check` prints a per-field table (max abs diff, diff relative to the field's
scale, RMS) and exits non-zero on failure. Tolerances: `--atol` (default
`1e-10`), `--rtol` (default `1e-8`).

`make verify` does the cross-compiler version of this — builds both, runs the
CPU build to a golden file, and checks the GPU build against it:

```bash
make verify NX=200 NY=200 STEPS=50 CASE=dambumps
```

**3. Against real ANUGA** — proves the miniapp is solving the same problem,
not just a self-consistent one. `tools/anuga_reference.py` rebuilds the case
through the full Python stack (`rectangular_cross` → `anuga.Domain` →
`set_multiprocessor_mode(2)`) and drives the same C entry point:

```bash
conda activate anuga_env_3.13
python tools/anuga_reference.py --nx 100 --ny 100 --steps 35 --out ref.bin
./bin/bench_gpu --nx 100 --ny 100 --steps 30 --warmup 5 --check ref.bin
```

The script's `--steps` is the miniapp's `warmup + steps`. On the unchanged
kernels this matches **bit for bit** on the flat-bed case and to `~1e-14` on
the bumpy cases (numpy vs libm `exp()` in the bed function). Re-run it after
any change whose correctness you want to establish against ANUGA proper rather
than against your own earlier build.

## Profiling

Wall-clock, no tools required:

```bash
./bin/bench_gpu --nx 500 --ny 500 --steps 50 --phases
```

Target regions are synchronous, so the per-phase timers are honest. FLOP counts
come from ANUGA's own counters (`gpu_flop.c`), so the GFLOP/s figure is directly
comparable to what the Python path reports.

Timeline and kernel counters:

```bash
nsys profile -o bench --stats=true ./bin/bench_gpu --nx 500 --ny 500 --steps 20
ncu --set full -k regex:compute_fluxes ./bin/bench_gpu --nx 300 --ny 300 --steps 3
```

Build with `EXTRA_CFLAGS=-DNVTX_ENABLED` to get named NVTX ranges in the nsys
timeline instead of raw kernel names.

Keep `--warmup` at 5 or more: the first steps pay for JIT and first-touch page
migration, and would otherwise dominate a short run.

## Scaling sweep

`tools/scaling_sweep.sh` runs increasing mesh sizes until one fails, recording
throughput and the real device footprint (sampled from `nvidia-smi`, because
ANUGA's `gpu_query_device_memory()` only reports numbers under `-DUSE_CUDA`,
which no build sets):

```bash
tools/scaling_sweep.sh --steps 20 --csv build/scaling.csv \
    100 200 300 500 700 1000 1600 2400 3200 4000 4200
```

It prints a live table, writes a CSV, and keeps every run's full output in the
matching `.log`. Measured on one Tesla V100-PCIE-32GB (nvc 25.9, `cc70`),
original kernels vs the optimisation rounds described below:

| triangles | Mcell-steps/s orig | optimised | gain | device |
|-----------|--------------------|-----------|------|--------|
| 40 K      | 74                 | 95        | +28% | 0.46 GiB |
| 1 M       | 213                | 258       | +21% | 0.93 GiB |
| 4 M       | 236                | 277       | +17% | 2.36 GiB |
| 16 M      | 236                | 278       | +18% | 8.02 GiB |
| 41 M      | 227                | 270       | +19% | 19.71 GiB |
| 64 M      | 220                | 262       | +19% | 30.53 GiB |
| 65.6 M    | 209                | 251       | +20% | 31.28 GiB |
| 67.2 M    | —                  | —         | —    | **out of memory** |

The device footprint is 512 bytes per triangle, flat across three decades, so
the ceiling for any card is `(VRAM - 0.3 GiB) / 512 B` — unchanged by the
optimisations, which remove launches and loads, not arrays. Throughput
saturates at about 1M triangles; below that the step is launch-latency bound
and not worth optimising against.

The same sweep for the fastest configuration, **ADER2 + scatter fluxes**
(`tools/scaling_sweep.sh ... -- --scheme ader2 --flux scatter`), after the
second optimisation round (fused C-K predictor, compacted owned-edge list,
no auxiliary arrays):

| triangles | ms/step | Mcell-steps/s | device |
|-----------|---------|---------------|--------|
| 40 K      | 0.18    | 218           | 0.15 GiB |
| 1 M       | 1.67    | 598           | 0.94 GiB |
| 4 M       | 6.16    | 649           | 2.37 GiB |
| 16 M      | 24.39   | **656**       | 8.08 GiB |
| 41 M      | 63.12   | 649           | 19.86 GiB |
| 51.8 M    | 87.27   | 594           | 25.00 GiB |
| 65.6 M    | 111.75  | 587           | 31.52 GiB |

Same shape as ever: saturation at ~1M triangles, mild fade toward the wall —
and the ceiling is back to the full **65.6M** (the owned-edge list costs
12 B/triangle where the retired speeds array cost 24). At the maximum mesh
size this is 2.8x the original baseline; at 16M it is 2.8x as well
(67.8 -> 24.4 ms of RK2-equivalent work per dt).

## Kernel optimisations

Changes to `anuga/shallow_water/gpu/`, every one bit-exact against the
pre-change goldens on all three cases and green on ANUGA's full suite. One RK2
step went from **21 kernel launches to 10** and from 67.8 to 57.6 ms at 16M
triangles (−15%):

- `core_forcing_and_update()` — Manning friction + conserved-quantity update +
  RK2 average, one launch instead of three. All cell-local; the semi-implicit
  and centroid intermediates stay in registers (25.3 → 16.6 ms at 41M). Falls
  back to separate kernels under sloped Manning.
- `core_prepare_step()` — RK2 backup + protect + the extrapolation's centroid
  pass, one launch instead of four (protect's follow-up height refresh was
  provably redundant before an extrapolate and is simply gone). The
  extrapolation is split into `core_extrapolate_centroid_pass()` /
  `core_extrapolate_edge_pass()`; the combined entry point survives unchanged
  for every other caller.
- `extrapolate` also lost its momentum-restore third pass (the velocity now
  rides in `x/y_centroid_work`, so nothing needs restoring).
- `D->reconstruct_edge_bed` (opt-in, default 0) — the flux kernel reconstructs
  edge bed values as `stage - height` instead of gathering `bed_ev`:
  bit-identical whenever fluxes follow an extrapolate (every evolve step),
  ~6 fewer scattered loads per cell, worth ~20% of the flux kernel. Off in
  ANUGA because tests call `compute_fluxes` directly with independently set
  edge values (`test_flux` fails if you force it on globally); the miniapp
  opts in.

**`compute_fluxes` cannot be fused with `extrapolate` or with `update`.** It is
a stencil kernel — it reads `height_cv[neighbour]`, `bed_cv[neighbour]`,
`stage_cv[neighbour]` and the neighbours' edge values — so writing any centroid
or edge value from inside it races against another team still reading that
value, and `omp target teams loop` has no device-wide barrier to order them.

### Mesh, and how structured it really is (`--order`)

The generator replicates ANUGA's `rectangular_cross` node-for-node: an
nx x ny grid of cells, each split into 4 triangles around a centre node.
The kernels treat it as FULLY unstructured (all access goes through the
general connectivity arrays), so the code paths are representative of any
triangulation -- but the grid-ordered numbering makes the gathers
unrealistically cache-friendly, so the throughput numbers are the
optimistic bound. `--order` quantifies the spread (ADER2 + scatter, 16M
triangles, V100; all three bit-exact against the same goldens via
canonical-order snapshots):

| ordering | Mcell-steps/s | |
|---|---|---|
| `row` (ANUGA rectangular_cross) | 652 | optimistic bound |
| `morton` (Z-order curve) | 658 | (a slight WIN on the optimized kernels; the -12% measured earlier was against the old cell-based RK2) |
| `random` (fixed-seed shuffle) | 275 | pessimistic bound: **2.4x slower** |

A real flood mesh (mesher-ordered, variable resolution) sits between the
bounds -- which makes element renumbering (Morton/RCM on centroids, a
one-time preprocessing permutation that these experiments prove is
result-identical) likely the single biggest performance lever for
production ANUGA on real meshes. ANUGA already has reordering machinery
in `neighbour_mesh.py`.

### A realistic basin (`--mesh`, tools/make_basin_mesh.py)

`tools/make_basin_mesh.py` builds what production actually runs: a synthetic
river basin (20 x 10 km sloped valley, meandering channel carved down the
middle, 8x finer triangles along the channel than the floodplain) meshed by
ANUGA's own Triangle wrapper -- so the element order is exactly what ANUGA
emits -- with terrain and a flood IC (full headwater reservoir released over
a baseflow-wetted channel) frozen into one binary file:

```bash
python tools/make_basin_mesh.py --target 2000000 --out build/basin_2m.msh
./bin/bench_gpu --mesh build/basin_2m.msh --steps 100 --scheme ader2 --flux scatter
```

The benchmark stays Python-free at run time. `--still LEVEL` emits a flat
constant-stage variant for the lake-at-rest gate on the real mesh (note:
the at-rest IC must be a constant plane; a per-node `max(bed, L)` tilts the
free surface inside shoreline cells and that water legitimately flows --
all flux variants agreed bit-for-bit on that motion, which is how the gate
proved the solvers innocent). Gates on the basin: volume drift 2e-16,
lake-at-rest 5e-13, CPU vs GPU 1.6e-13.

**The ordering result that matters** (2.43M-triangle basin, V100,
ADER2 + scatter):

| ordering | Mcell-steps/s | |
|---|---|---|
| as-meshed (ANUGA's Triangle output) | 567 | what production gets today |
| + Morton reorder (`--order morton`) | **634** | **+12%, free, result-identical** |
| random | 170 | pessimistic bound |

ANUGA's mesher order is decent (87% of the structured-grid figure) but not
optimal: a one-time centroid-Morton renumbering at domain construction is
worth ~12% on real meshes and nothing on structured ones -- the strongest
argument yet for wiring the reorder into production mode 2.

### Active-set stepping (`--active-set`) and what rain does to it

A dry cell with an all-dry neighbourhood provably cannot change, so the step
kernels can skip it exactly: each step rebuilds active cells (wet, 2-ring
halo -- the MPI ghost-layer rule, since RK2's two flux calls advance water up
to two rings per rebuild) and active edges (either side in ring 1).  The
first step runs full; classification uses stage - bed (operators such as
rain modify stage directly, and a height-based test misses them) with a
1e-12 m wetness threshold (a strict > 0 suffers roundoff creep: ~1e-15 m
films from update-sum cancellation would activate the mesh ring by ring;
films below 1e-12 m flux at ~1e-21 relative, beneath double precision, so
skipping them stays bit-exact).  All measured on the 2.43M basin, V100,
ADER2 + scatter, every case bit-exact against full-run goldens:

| scenario | active | speedup |
|---|---|---|
| dam-break flood, no rain | 11.4% | **2.75x** |
| basin-wide rain (`--rain 20`) | 100% | 0.8x (rebuild overhead only) |
| localized storm (`--rain-band`) | 36% | 1.57x |

Distributed rain-on-grid defeats the optimisation (everything wets and,
with no infiltration, never un-wets); localized forcing keeps a real win;
models with losses/infiltration (cells re-drying between pulses) sit in
between.  The worst case is bounded: ~10-20%% rebuild overhead, still
bit-exact.  For event-driven floods (dam break, levee breach, surge -- and
the 11,000 km^2 spec basin, which starts 99.5%% dry) this is the largest
single lever measured in this project.

### Timestepping schemes (`--scheme`)

`rk2 | ader2 | euler | rk3`, each selecting its ANUGA preset (DE1 / DE_ader2 /
DE0 / DE2). The honest cross-scheme metric is the printed **sim rate**
(simulated seconds per wall second), since ms/step ignores dt. Measured at 16M
triangles: **ADER2 delivers 1.64x the sim rate of RK2** — same CFL timestep
(dt 0.00734 vs 0.00732), same formal order, one flux call per step instead of
two, with the C-K predictor costing ~7 ms against the ~35 ms flux+extrapolate
round it replaces.

### ADER2 second round

Three further changes took ADER2 + scatter from 32.2 to 24.4 ms/step at 16M:

- **C-K predictor fused into the extrapolation edge pass**
  (`core_extrapolate_edge_pass(D, predictor_dt)`): the predictor is strictly
  cell-local and reuses the dxv/dyv edge offsets the limiter already holds in
  registers, so the shift to Q^{n+1/2} costs 2.7 ms fused vs 6.85 standalone
  -- and since it never reads boundary values, the step's first boundary
  evaluation (whose outputs the second always overwrote) is gone. Bit-exact
  against the unfused sequence; RK2 passes predictor_dt = 0 and is untouched.
- **Scatter mode needs no auxiliary arrays**: selected by
  `reconstruct_edge_bed = 2` plus a driver-built compacted owned-edge list
  (`owned_edges`, ~1.5 slots/cell), which also gives the kernel one thread
  per physical edge instead of one per slot with half idle -- flux kernel
  11.2 -> 7.8 ms. The serial benchmark also passes tri_full_flag = NULL,
  skipping the ghost-ownership gathers.
- The fused reconstruction kernel (~48% of the step) has now resisted three
  structural attacks, all measured: ncu shows it register-limited at 24%
  occupancy (~128 regs/thread), but global `-gpu=maxregcount` capping trades
  spills for occupancy at net zero; **splitting it by quantity** (stage+height
  kernel, then velocity+momentum+predictor kernel -- bit-exact, and it even
  fixes the dry-zero visibility race) costs +33% because the re-read geometry
  and edge values outweigh the doubled occupancy; and the interleaved-gather
  pack measured -40% in its branchy form, with the branch-free variant
  predicted marginal since row-major locality already keeps the gathers
  L2-warm (the Morton result). At 55-60% DRAM utilisation with its access
  pattern the kernel is near its structural floor for portable OpenMP.
  **Do NOT reach for `OMP_TEAMS_THREAD_LIMIT`**: it produces up to 1.6x
  "speedups" with silently corrupt physics (see `claude/KNOWN_ISSUES.md`).

### Flux kernel structure (`--flux`)

The cell-based production kernel solves every interior edge's Riemann problem
twice — once per side. The central-upwind flux is antisymmetric under the
side swap and its shared scalars (pressure_flux, wave speed, z_half) are
swap-invariant, so one owner-side solve serves both cells. Two opt-in
restructurings (kernels select purely on the dead work-array pointers, so
ANUGA's default path is untouched):

- `--flux scatter` — **the winner**: single solve per edge, both sides'
  area-scaled contributions accumulated straight into the explicit updates
  with `omp atomic` (portable OpenMP; each entry sees at most 3 adds).
  RK2 57.6 → 51.1 ms/step at 16M (−11%); **ADER2 + scatter: 32.2 ms/step,
  497 Mcell-steps/s** — 2.1x the sim rate of the original baseline.
- `--flux edge` — the same single-solve idea via materialized per-edge slot
  records and a gather kernel: **measured 15% SLOWER** than cell-based.
  The 144 B/cell of slot records cost more to move than the duplicate
  Riemann solves saved. Kept as the deterministic-order variant and as
  documentation of why scatter is shaped the way it is.

Neither is bit-exact against cell-based (the neighbour side receives the
negated owner flux instead of its own evaluation — roundoff-level
difference). Validation: mass conservation and lake-at-rest hold at machine
precision; friction-free field comparisons agree with cell-based at ~4e-14
over 16 steps (CPU vs GPU likewise); riverwalls and sloped Manning force the
cell-based path automatically.

**Trajectory-divergence caveat** (applies to comparing ANY two roundoff-
different runs of this scheme, not just these variants): the semi-implicit
update guard `num * Q > 0` is discontinuous where a momentum component
crosses zero. In problems whose exact solution has a zero momentum component
(the dam cases: ymom ≡ 0, so the field is pure roundoff), two trajectories
seeded 1e-14 apart straddle zero-crossings, flip the guard, and diverge to
~1e-6 within a few steps once friction populates the semi-implicit terms.
Compare such runs with loose tolerances (`--rtol 1e-4`) or with
`--no-friction`, where agreement returns to ~1e-14.

### The CUDA control experiment (`--cuda-extrap N`)

To find out whether OpenMP itself was costing anything, the fused
reconstruction kernel was transcribed line-for-line into CUDA
(`src/cuda_extrapolate.cu`, built as a pure-nvcc shared library and
dlopen()ed -- every way of *linking* CUDA objects into the OpenMP-target
binary broke nvomp's offload registration or ICEd nvc; the two runtimes meet
only through the shared CUDA primary context, with device pointers resolved
via `omp_get_mapped_ptr`). Result, V100 @16M, same arithmetic, goldens green:

| variant | ms |
|---|---|
| OpenMP (`nvc -mp=gpu`) | **11.6** |
| CUDA, best threads/block, default regs | 12.3 |
| CUDA, `-maxrregcount=96` | 12.3 |
| CUDA, `-maxrregcount=64` | 21.7 (spills) |

Hand-written CUDA with explicit launch control **loses ~5%** to nvc's OpenMP
codegen. There is no portability tax on this kernel; it is at its
algorithmic floor, and further speed means changing the math, not the
programming model.

### H200 (gpuhopper) results

**Memory wall, measured**: 293.8M triangles runs (nx 8570, 139.6 of 143 GiB);
295.8M fails -- the 512 B/triangle model predicted 294M, correct to 0.7% on
its second card (the V100 was its first). The OOM allocation that fails is
the owned-edge list, the last array mapped.

**The large-mesh "fade" is an alignment artifact, not a capacity effect.**
nx = 6400 and nx = 8000 (the two swept sizes with ny divisible by 64) hold
~2830-2843 Mcell-steps/s -- full peak -- at 164M and 256M triangles, while
every non-aligned neighbour size sits ~10% lower (reproduced 3/3 repeats;
nx 6400 = 2828/2831/2814 vs 6300 = 2582/2567/2558). With ny % 64 == 0 the
regular cross-mesh's i-neighbour stride lands page-aligned and the card
sustains peak throughput to within 15% of the memory wall. Real unstructured
meshes have no such regular stride, so quote the aligned numbers as the
kernel ceiling and the unaligned ones as the pessimistic bound.


`tools/h100_campaign.pbs` runs the whole campaign on the batch queue
(build for the node's arch, correctness gates, both sweeps, kernel balance).
On an H200 (143 GB, cc90): **ADER2 + scatter peaks at 2837 Mcell-steps/s**
(4.3x the V100), RK2 cell at 1535 (5.5x -- pure bandwidth scaling); all
gates hold at 1e-14; 169M triangles ran in 80.5 of 143 GiB (the 512 B/cell
model predicts a ~294M ceiling, unswept). The kernel balance FLIPS on
Hopper: flux 44% vs reconstruction 39% (per-cell, reconstruction scaled
5.3x with bandwidth while the atomic scatter scaled only 3.1x) -- so the
next kernel worth attacking depends on the target architecture.

### Multi-GPU with MPI (`make gpumpi`, 4x H200, 2026-08-26)

A C-only MPI layer (`src/bench_mpi.h`, no mpi4py anywhere) slab-decomposes
the generated mesh along x -- cell id `i*n+j` gives each rank a contiguous
id range -- with one ghost column per interior side, exchanged through the
production gpu layer's own `gpu_exchange_ghosts` each substep. The
decomposition is **bit-exact**: np=2 and np=4 match the np=1 run at
`--atol 0 --rtol 0` on CPU (18 checks, dam/dambumps/river x rk2/ader2/euler)
and on the H200s (`tools/h200_mpi_scaling.pbs`, job 177517465, results in
`build/mpiscale/`). Bringing the path up on real GPUs found a production bug
in `gpu_halo.c`: the GPU-aware pack/unpack kernels read flat index arrays
that `gpu_halo_init` never mapped to the device (fixed here).

RK2 cell fluxes, dam case, 100 timed steps, ms/step and Mcell-steps/s:

| mesh            | np=1         | np=2         | np=4         | eff. @4 |
|-----------------|--------------|--------------|--------------|---------|
| 64M (nx 4000)   | 41.89 / 1528 | 22.50 / 2844 | 11.92 / 5367 | 88%     |
| 144M (nx 6000)  | 97.15 / 1482 | 51.21 / 2812 | 26.38 / 5458 | 92%     |

Weak scaling at 36M triangles per GPU: 24.0 / 25.4 / 25.7 ms/step on
1 / 2 / 4 GPUs (5.6 Gcell-steps/s aggregate, 93% efficiency). Strong
efficiency rises with problem size (88% -> 92%), i.e. the residual is the
fixed per-step exchange cost, and the per-rank work simply has to be large
enough to hide it -- the "linear given enough work" claim, now measured.

**Scatter fluxes, `--phases` and the active set under MPI (2026-08-27).**
The scatter kernel's owned-edge list is built ghost-aware in `setup.c`: a
slot is taken only from an owned cell, and an interface edge is taken from
the owned side regardless of index order, so each rank scatters every edge
touching one of its cells exactly once (the ghost side's accumulated update
is garbage and is overwritten by the halo exchange). The miniapp's own
stepped loops (`rk2_step_timed` / `ader2_step_timed`) gained the dt
allreduce and the mid-step exchange the library step does. The 2-ring active
set needs no change: one ghost column is four triangle hops wide, so every
cell within two hops of an owned cell is local and the rings come out
identical to the serial build.

`tools/mpi_verify.sh` is the CPU regression gate (64 checks, np=2/4 vs the
serial run, 120x100 and 100x100 x dam/river x rk2/ader2): cell fluxes and
`--phases` are **bit-exact** (`--atol 0 --rtol 0`); scatter and
scatter+active-set are checked at `--atol 1e-6` because scatter accumulates
the three edge contributions per cell with atomics and is therefore only
ever exact to summation order -- a *serial* 1-thread vs 4-thread scatter run
differs by the same ~1e-9 (dam, ader2) the MPI runs show, and typical MPI
residuals are 1e-14. Building the slab this way also exposed and fixed a
generator roundoff bug: node x-coordinates were `x0_local + i_local*dx`,
which differs from the full mesh's `i*dx` whenever `dx` is inexact (any
non-square case, e.g. 1000/120) -- now the slab uses the global column index.

Remaining MPI restrictions (enforced in `bench.c`): generated mesh in
`--order row`, no `--flux edge`, no `--cuda-extrap`.

**4x H200 results** (`tools/h200_mpi_active.pbs`, job 177579044, results in
`build/mpiactive/`). All 16 GPU gates OK -- cell and `--phases` at atol 0,
scatter and scatter+active-set at atol 1e-6 with measured max relative
differences of 7e-16 (dam) and 1.5e-15 (river). Strong scaling at 144M
triangles (6000x6000, RK2, 100 timed steps), ms/step and Mcell-steps/s:

| case / flux          | np=1          | np=2          | np=4          | eff. @4 |
|----------------------|---------------|---------------|---------------|---------|
| dam, cell            | 97.2 / 1482   | 51.1 / 2818   | 26.4 / 5447   | 92%     |
| dam, scatter         | 102.5 / 1406  | 54.8 / 2628   | 28.1 / 5119   | 91%     |
| dam, scatter+active  | 123.4 / 1167  | 65.7 / 2193   | 33.4 / 4310   | 92%     |
| river, cell          | 85.4 / 1687   | 45.9 / 3137   | 24.0 / 5990   | 89%     |
| river, scatter       | 88.7 / 1624   | 48.0 / 3001   | 25.4 / 5667   | 87%     |
| river, scatter+active| 28.6 / 5039   | 22.6 / 6381   | 18.9 / 7601   | **38%** |

The dam rows scale like the cell path (the active set is 100% there, so
it only adds the rebuild cost). The river row is the interesting one: the
active set makes the serial run 3x faster (18% of cells active), but the
x-slab partition hands rank 0 the whole reservoir (x < 0.15 L lies in its
columns [0, 1500)), so rank 0 carries ~53% of its cells active against the
18% average and its kernel phases alone sum to 18.6 of the 18.9 ms step.
**A geometric partition balances triangles; the active set makes the load
proportional to wet cells.** Partitioning a mostly-dry basin therefore has
to be weighted by (expected) wetness -- e.g. from a coarse run -- which is
the plan for the tiled 1 m^2 delta mesh. The report line now prints the
per-rank min..max active fraction so this imbalance is visible directly.

### Tiled distributed mesh (`--tiles`): no rank ever sees the global mesh

The 1 m^2 challenge mesh of the 11,372 km^2 delta is 17.45 G triangles in
1404 conforming 3 km tiles (`cdac_script/split_delta.py` + `mesh_tile.py`,
26 min on 104 cores, 521 GB of `.msh`).  A merged mesh would be ~1 TB of
connectivity, a Python `Domain` ~17 TB, and pymetis on it is out of the
question -- so the tiles themselves are the partition:

- `tools/tiles_to_bmesh.py` converts each tile (and, for validation, the
  merged mesh) to the miniapp's binary format with a synthetic analytic bed
  and initial stage evaluated at the node coordinates -- the per-tile
  initialisation step, embarrassingly parallel, never global.  Vertices are
  rounded to 6 decimals exactly as `merge_tiles.py` does, so tiles and the
  merged mesh carry bit-identical geometry.
- `src/tiles.c` (`--tiles index.txt [--assign FILE]`): each rank reads its
  own tiles plus every tile whose bounding box touches them, stitches on
  exact vertex coordinates, keeps as ghosts the foreign triangles that
  share a vertex with an owned one (a superset of the 2-edge-hop ring RK2
  needs), and builds the `gpu_halo_init` lists from that.  Both ranks of a
  pair derive the same sets from the same tile data in the same (tile,
  local index) order, so the lists agree without communication (a count
  handshake asserts it).  Canonical ids are tile offset + local index, the
  order `merge_tiles.py` concatenates in, so snapshots check against the
  merged run.
- Gates (`tools/mpi_verify.sh`: 10,000 m^2 coarse delta, 1.77M
  triangles, 158 tiles): cell fluxes and `--phases` are **bit-exact**
  against the merged-mesh serial run at np = 1, 2, 4 and 7 (odd splits),
  rk2 and ader2.  Scatter and scatter+active-set match to `--ftol 1e-7`
  (max diff relative to the field's largest value): the interface edges
  change which side accumulates first, and the seed grows 0 -> 0 -> 7e-15
  -> 4e-14 -> 7e-13 over steps 1/2/5/10/20 -- a missed or doubled flux
  would show at step 1 at ~1e-3.

**Load balance by wetness, and its limit.**  `--tile-stats FILE` counts,
on the device, how many rebuilds each cell spent in the active set and
writes the mean active fraction per owned tile; `tools/tile_assign.py`
turns that into a `--assign` map balancing `ntris * (fraction + floor)`
(the floor is the never-skipped per-cell work: rebuild passes vs. full
step, ~0.035 on this CPU, ~0.07 on an H200).  On the coarse delta with
the lake released in two tiles, the triangle-balanced assignment has a
114% spread in per-rank kernel time (rank 0: 21k of 26k active cells);
the weighted one balances the predicted work exactly, but no assignment
of whole tiles can beat the heaviest tile -- tile 30 alone is 40% of the
active cells against a 25% share.  `tile_assign.py` prints that bound and
the tiles above half a share: those are the ones to re-split (half-size
sub-tiles keep the outer lattice, so neighbours still conform).  That is
the coarse-run -> weight -> retile -> assign loop for the 1 m^2 mesh.

**The loop, closed on the coarse delta (4 ranks, CPU).**  Two changes
made it work: (1) `core_build_active_sets` no longer seeds ring-1 from the
neighbour-less edges of *ghost* cells -- those are the artificial rim of
the halo strip, and seeding from them kept every rank's whole perimeter
active (103k vs 33k active cells/step under a scattered assignment; the
change is exact, 96/96 gates); (2) `cdac_script/split_delta.py --refine-file`
cuts listed tiles into 4 half-size sub-tiles (ids 10000 + 4*parent + q)
whose new cut lines carry the same global lattice, and
`cdac_script/refine_tiles.sh` re-meshes only those and re-merges (the
half-tile must be an exact multiple of the spacing -- pass 5000/58 as
86.20689655172414, not 86.2069, or the corner misses the lattice by 0.2 mm).

| assignment (4 ranks)                    | active cells/step, per rank | kernel ms/step spread | wall ms/step |
|-----------------------------------------|-----------------------------|-----------------------|--------------|
| triangle-balanced (default)             | 21.5k / 4.7k / 3.4k / 2.7k  | 4.43 .. 10.22 (131%)  | 10.34        |
| wetness-weighted (lpt, floor 0.035)     | 12.0k / 9.6k / 6.3k / 5.0k  | 6.16 .. 6.95 (13%)    | 7.71         |
| + tiles 30, 31 split (164 tiles), re-weighted | 9.1k / 8.6k / 8.2k / 8.1k | 7.22 .. 8.35 (16%) | 9.15      |

The refined tiled mesh is bit-exact against its own merged mesh at np=4
and np=7 (cell fluxes, atol 0) -- the split conforms.  After the split the
heaviest tile is 0.14 of a rank share and the active cells are within 6%
across ranks; the remaining kernel-time spread on this shared CPU node is
noise and base-cost variation, not granularity.  (Wall time did not drop
further here because 4 x 4 threads share 20 loaded cores; the GPU run is
the real measurement.)

GPU validation + balance at 300 m^2 (58M triangles): `tools/h200_mpi_tiles.pbs`.

### Cross-vendor portability: AMD MI250X and Intel PVC (2026-08-25)

First contact with non-NVIDIA hardware, same source, no code changes:
`make amdgpu` (amdclang, CDNA2) and `make intelgpu` (icx, spir64).  Both
targets pass the full correctness ladder: lake-at-rest at roundoff, the
CPU-reference golden, scatter-vs-cell at roundoff tolerance, and the
active-set run **bit-exact** against full stepping (the `omp atomic
capture` paths).  Stronger still: the dam-break physics report (final t,
dt, stage range, max |momentum|) agrees with the x86 gcc reference to
every printed digit on both vendors.

ADER2 + scatter, 4M triangles, row order, one device:

| device | Mcell-steps/s | balance |
|---|---|---|
| MI250X (1 GCD of 8) | 412 | extrapolate-heavy: 62%% reconstruction |
| V100 | ~650 | reconstruction ~48%% |
| A100-80 | ~1500 | |
| PVC (1 of 6) | 1542 | flux-heavy 46%% -- Hopper-like |
| H200 | 2837 | flux 44%% / reconstruction 39%% |

Full scaling sweeps (19 sizes, 40k -> 134.56M triangles, 61.2 GiB
mapped, build/mi250x_scaling.csv + build/pvc_scaling.csv): the MI250X
GCD is the flattest device measured -- ~385 Mc/s held from 2M to 134.6M
with no large-mesh fade at all (peak 419 at 2M).  PVC peaks at 1552
(10M, A100-class) and fades ~30%% to ~1050-1080 above 90M -- the same
large-mesh shape as the V100.  Neither hit its memory ceiling at
nx=5800; the 488 B/tri footprint puts 64-GiB walls near nx~5900.
Cross-vendor determinism is total: AMD and Intel print identical
max-momentum digits at every one of the 19 sizes, volume drift <=1e-16,
zero NaNs.  The MI250X's 62%% reconstruction share makes that
register-heavy kernel the one target for AMD tuning.  Both MI250X GCDs
and PVC stacks enumerate as separate OpenMP devices, so a node is
~8x / ~6x the single-device figure under MPI: one MI250X node (8 GCDs
x ~385 = 3.1 Gc/s sustained) edges out one H200.  Five-device chart:
tmp_artifacts/anuga_v100_scaling.html's cross-vendor successor lives at
the "One Source, Five GPUs" artifact.

### Cross-architecture results (V100 / A100-80 / H200)

The 512 B/triangle ceiling model is verified on three architectures, each to
sub-1%: V100 65.6M measured (65.6M predicted), A100-80 166.4M ok / 169M OOM
(166.3M predicted), H200 293.8M ok / 295.8M OOM (294M predicted).  The
168.2M spec mesh OOMs on the A100-80 exactly in the predicted 2M gap.
Physics gates hold identically everywhere (lake-at-rest 1e-13..1e-14).
ADER2 + scatter plateaus: V100 656, A100-80 ~1500, H200 2837 Mcell-steps/s
-- 80-90%% of pure bandwidth ratios.  The Morton reorder gain on real basins
replicates: +12%% on both A100 and H200 at 24M triangles.

### The 11,000 km^2 spec run (168.2M triangles <= 100 m^2, one H200)

Measured hours of wall time per simulated day, early-event (60 steps,
0.62%% wet):

| configuration | 1 GPU | /4 GPUs (linear) |
|---|---|---|
| as-meshed, full stepping | 33.9 h | 8.5 h |
| + Morton | 24.8 h | 6.2 h |
| + Morton + active-set | **3.7 h** | **0.92 h** |

The active configuration runs at 20.9 Gcell-steps/s effective on one card.
**Measured over half a simulated day** (tools/h200_longrun.pbs, 4 GPUs x
~105 min, one config each): Morton + active-set sustained **2.98 h of wall
time per simulated day on ONE H200** across 12.25 simulated hours of flood
development -- ~45 min on 4 GPUs at linear scaling.  The evolution curves:
active fraction grows only 0.66% -> ~3% by t = 12 h (the flood channels,
it does not sheet), dt IMPROVES from 0.053 to 0.077 s as the release
smooths, and Morton is worth 47% under active-set (8.07 vs 5.50 sim-s/s
against row order -- scattered active-cell iteration amplifies locality).
Full stepping measured 23.4 h/sim-day on the same event: active-set is
7.9x sustained.  Under the spec-scale synthetic storm (22% raining, no
losses) activity climbs monotonically to ~39% and the rate settles near
2.3 -> ~10.5 h/sim-day single-GPU; real rain with infiltration sits
between the two curves.

### Measured dead ends (kept out, documented so nobody re-tries them blind)

- **FP32 geometry** (`make gpu GEOM=fp32`, via the `anuga_geom_t` typedef in
  sw_domain.h -- the fp64 default stays bit-exact and ANUGA never defines the
  flag): **+2%**. Mass conservation survives perfectly, but lake-at-rest
  degrades from 1e-14 to ~1e-5 momentum on the real basin (float epsilon
  leaking through the well-balance cancellation), and the payoff is tiny
  because the geometry gathers are already L2-resident -- halving bytes that
  rarely touch DRAM buys nothing. FP32 *state* is not even an experiment:
  stage - bed has ~mm significance against ~100 m magnitudes and does not
  survive single precision. Verdict: no case for mixed precision here, and
  therefore no case for a C++/template migration to support it -- the whole
  question was answerable with one C typedef and an afternoon.

- **Morton element ordering** (`--order morton`, still available): −12%. The
  cross mesh in row-major order already has 2 of 3 neighbours inside the same
  cell and the ±j neighbour adjacent; Z-ordering fixes the ±i stride but
  breaks the j-adjacency, a net loss. Bit-exact (snapshots are canonical-order
  either way), so the flag remains as a locality experiment for other meshes.
- **Interleaved gather packs** (32-byte `{stage, xmom, ymom, height}` records
  for the flux and limiter gathers): −40%. The stride-4 streaming accesses and
  the dual-source loads wrecked nvc's kernel scheduling — with the pack
  branches merely *compiled in* but disabled, the flux kernel was 1.7× slower
  than the branch-free version. Reverted entirely; lesson: keep these kernels
  branch-free above all.
- **`-gpu=fastmath`**, **`-gpu=loadcache:L1`**, **`OMP_NUM_TEAMS` /
  `OMP_THREAD_LIMIT` sweeps**: all within noise or worse. The kernels are
  memory-bound and nvc's default launch heuristics are already right.

## Options

```
mesh / problem
  --nx N --ny N        cells; the mesh is a rectangular cross -> 4*nx*ny triangles
  --lenx L --leny L    domain size in metres (default 1000 x 1000)
  --case NAME          dam | dambumps | lake
                         dam       flat bed, wet dam break -- every cell wet,
                                   no dry-cell branches, maximum work per step
                         dambumps  bumpy bed dam break -- exercises wet/dry
                         lake      water at rest over bumps -- well-balancedness
  --manning V          Manning n (default 0.03)
  --water V --dam V    downstream / upstream stage
  --no-friction        skip the Manning forcing term

run
  --steps N            timed RK2 steps (default 100)
  --warmup N           untimed RK2 steps first (default 5)
  --repeat N           repeat the timed loop, report the best
  --cfl V              CFL number (default 1.0)
  --phases             per-kernel timing breakdown
  --verbose            let the kernels print their own setup messages

correctness
  --save FILE          write the final centroid state
  --check FILE         compare the final centroid state against FILE
  --atol V --rtol V    tolerances for --check
```

## Layout

```
Makefile                  builds ANUGA's gpu/*.c + src/*.c into one binary
env.sh                    module loads for the nvc/GPU build
src/mesh.c                rectangular_cross generator, node-for-node identical
                          to anuga.abstract_2d_finite_volumes.mesh_factory
src/setup.c               geometry, connectivity, quantities, boundaries --
                          mirrors General_mesh._compute_geometry and
                          Neighbour_mesh, so the kernels see exactly the arrays
                          ANUGA would hand them
src/bench.c               CLI, timed loop, per-phase timers, diagnostics
src/snapshot.c            binary state snapshots (save / diff)
src/gcc_offload_shim.c    one gcc link quirk, explained in the file
tools/anuga_reference.py  same case through the full ANUGA stack
```

## Notes and limits

- Without `-DHAVE_MPI` the build is serial: MPI is stubbed out via
  `gpu_mpi_stubs.h` (`nprocs == 1`, no halo exchange). The `*mpi` targets add
  a slab decomposition with the v1 restrictions listed under "Multi-GPU with
  MPI".
- Reflective boundaries only. The other boundary evaluators are still called
  each step (with zero edges) so the timing matches the production step.
- No riverwalls, no operators (rate/inlet/culvert). Those sources are compiled
  and linked, but the benchmark never activates them.
- `--phases` re-implements the RK2 step in `bench.c` to place timers around
  each kernel. It mirrors `gpu_evolve_one_rk2_step()` in `gpu_kernels.c` — if
  you change that function, change `rk2_step_timed()` to match. The two produce
  identical state today; `--check` will catch it if they diverge.
- `--repeat` does not reset the state between runs; later repeats start from a
  more-evolved field. Fine for timing, not for `--save`.
