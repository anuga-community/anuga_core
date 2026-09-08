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

The rebuild itself then becomes the cost -- 61-86% of kernel time once the
mesh is big and mostly dry, because all four passes sweep it.  See
`--active-every` below, which amortizes them against a dilated candidate set
and takes the rebuild down 6.7x.

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

**Multi-scale tiles (2026-08-28).**  Resolution can vary per tile on the
same grid: `split_delta.py --areas FILE --area-default A` gives each tile a
max triangle area from a map (tile id -> m^2; areas must be `A * 4^k` so
lattice spacings nest as `s * 2^k`), and a shared cut line carries the
*finer* of its two tiles' lattices -- both sides compute the same points,
so the tiles still conform and Triangle grades from the fine edge into the
coarse interior.  Exterior outline segments are densified per tile at that
tile's own spacing (they are never shared).  Delta test with 1024 m^2
background, 256 on the fringe and 64 m^2 on the lake tiles (from the coarse
run's wetness): 20.35M triangles against ~355M for uniform 64 m^2; the
merge stitches every internal edge (boundary edges = exterior segments
exactly) and the tiled run is bit-exact vs the merged mesh at np=4 and 7.
`build/figures/fig9_multiscale_transition.png` shows a 16:1 transition.
For a basin the mix is what matters: at the delta's density every 1% of
area at 1 m^2 is ~30 G triangles.

**The time step is set by corners, and that is fixable.**  Sizing the
triangles of that mesh: the class medians are 2.7 / 5.4 / 10.8 m inradius,
but the smallest were 0.47 / 0.29 / 0.009 m -- needles from (a) raw outline
vertices centimetres from the outline/grid-line intersection point and
(b) lattice or densification points landing within a metre or two of a
segment endpoint, where Triangle's 28-degree minimum angle forces tiny
triangles.  ANUGA's dt is the minimum of inradius / speed, so a corner on
the wet lake edge with a 1 m inradius and a 30 m/s front set dt = 0.032 s
against ~0.09 s for a median triangle.  `split_delta.py` now drops outline
vertices closer than `MARGIN * s` to the previous kept one (never a point
on a grid line) and keeps inserted points at least `MARGIN * s` from
segment endpoints (symmetric on both sides of a cut line, so conformity
holds).  Sweep on the same mesh: MARGIN 0.3 -> dt 0.0322 s, 0.5 -> 0.0445,
0.7 -> 0.0445 (the corner is no longer the limiter).  **1.38x on dt for
free**; the default is 0.5.  Minimum inradius after the fix: 0.78 / 0.53 /
0.05 m by class.

**Proof of concept on the small-triangle question (2026-08-28).**  Five
variants of the same multi-scale delta mesh, each re-split, re-meshed,
merged (conformity = boundary edges equal exterior segments), gated with
`cdac_script/tile_quality.py` (per tile: min inradius vs median, min angle)
and timed for dt:

| variant                                   | stitched | tiles flagged | worst min/median | dt (s) |
|-------------------------------------------|----------|---------------|------------------|--------|
| endpoint margin 0.3 s                     | yes      | --            | 0.001 (9 mm)     | 0.0322 |
| margin 0.5 s                              | yes      | 84 / 158      | 0.00 (notch)     | 0.0445 |
| margin 0.5 s + Triangle `-q32`            | yes      | 79            | 0.00             | 0.0431 |
| margin 0.5 s + drop outline vertices at acute corners | **no** (4 edges) | 72 | 0.20   | 0.0398 |
| margin 0.5 s + global outline `simplify(s/4)` | yes   | 60            | 0.11             | 0.0433 |

Conclusions.  (1) The endpoint margin is the lever: it removed the corner
wedges that set dt on the wet edge, 1.38x.  (2) Triangle's minimum-angle
flag does not move dt (+6% triangles for nothing).  (3) Any rule that
drops outline vertices per tile can change the domain -- where the outline
hugs a cut line within s, dropping the hugging vertex turned an exterior
sliver into a cut-line segment the neighbour does not own: 4 unstitched
edges.  Rejected.  (4) Simplifying the GLOBAL outline before intersection
(tolerance s/4) is the symmetric way to remove notches -- both tiles see
the same polygon -- and it takes the worst needle from 9 mm to 0.11 of the
median; it does not change dt because the remaining flagged tiles are dry.
The 60 still flagged are dry coastline tiles with corner triangles at
0.1-0.5 of their median; if one of those ever wets, `tile_quality.py`
names it and the fix is local (a coarser tile there, or a refined split
moving the cut line).  Default now: margin 0.5, simplify s/4, q28.

**At the real tile size.**  Two adjacent 1 m^2 delta tiles (ids 10 and 11,
13.8M triangles each, 440 MB `.msh`) through the same path: np=2 with one
tile per rank (5001 ghost cells -- the halo is 0.04% of a tile) is
bit-exact against the np=1 stitched pair; 8.8 s to build a rank's domain,
129 s and 2.4 GB to convert the pair (so all 1404 tiles convert in ~15 min
on 104 cores, tile-parallel like the meshing).  The synthetic lake is far
from these tiles, so this checks the mechanics at scale; flow across
interfaces is covered by the coarse dam-break gates above.

**All 1404 one-square-metre tiles converted** (`tools/sr_convert_tiles.pbs`,
job 177634970, normalsr 104 cores): 17,445,343,465 triangles, 391 GB in the
compact v2 format (int32 indices, float32 bed/stage, float64 coordinates)
on `/g/data/bm55/jlv900/tiles1sqm/`, 18 min wall, 439 GB peak node RAM
(the NetCDF import is what costs memory).  The v1 format would have been
~780 GB, and scratch/bm55 had 300 GB left -- hence v2 and gdata.
`tools/tile_subset.py` cuts a sub-index (e.g. the 16-tile lake block for
`tools/h200_1sqm_rehearsal.pbs`).

**First real 1 m^2 tiles on GPUs** (`tools/h200_1sqm_rehearsal.pbs`, job
177636235, `build/rehearsal/`): 9 tiles / 89.5M triangles at the lake.
np=2 and np=4 tiled runs are bit-exact against the np=1 run on one H200.
Costs at the real tile size: 488 B/triangle device, 543 B/triangle host,
build 0.45 us/triangle (40 s serial, 10.7 s per rank at np=4), 6 s map.
np=4 full mesh 25 ms/step = 3.6 Gcell-steps/s, bounded by the rank holding
two tiles (27.6M) while another holds one (13.8M): with 9 tiles over 4
ranks the 97% kernel-time imbalance is pure granularity -- the reason the
full run needs ~14 tiles per rank (or refinement) and not 2.  The block is
entirely inside the synthetic lake (100% active), so no wetness balance
was measurable here; the 2-node 48-tile run covers shoreline and dry land.

**Two nodes, 8 H200s, 26 real tiles / 301M triangles** spanning lake,
shoreline and dry land (`tools/h200_1sqm_2node.pbs`, job 177636649,
`build/rehearsal2/`) -- the first run across a node boundary with the
GPU-aware halo exchange:

| run (8 GPUs unless noted)               | ms/step | note |
|-----------------------------------------|---------|------|
| np=4, one node, full mesh (golden)      | 76.0    | 37 GiB/GPU |
| np=8, two nodes, full mesh              | 42.6    | **bit-exact vs np=4**; 89% efficiency across the node boundary, 7.1 Gcell-steps/s |
| np=8 active-set, triangle-balanced      | 44.8    | kernel 1.9 .. 44.1 ms (2251%): rank 1 has 41M active cells, rank 6 has 10k -- no gain at all over the full mesh |
| np=8 active-set, wetness-weighted contig| **27.3**| kernel 14.7 .. 25.5 ms (74%); 1.64x the triangle-balanced run |

The remaining 74% is granularity on real tiles: the wet tiles are 13.8M
triangles each (heaviest = 0.70 of a rank share) and 26 tiles over 8
ranks leaves one rank with one wet tile and another with two.
`tile_assign.py` flagged 13 tiles for the half-size split.  At full scale
(1404 tiles, ~14 per rank) the same weighting lands within ~10% without
splitting, as the 300 m^2 runs showed.

**After the split** (`cdac_script/refine_tiles.sh` on the 13 flagged tiles
-> 50 sub-tiles, 1441 tiles in the set; converted in place into the gdata
tile dir; job 177650566, same block, now 63 tiles / 301.3M triangles):

| run (8 GPUs, refined block)             | ms/step | note |
|-----------------------------------------|---------|------|
| np=4 full mesh (golden)                 | 71.2    | |
| np=8 full mesh                          | 35.1    | bit-exact vs np=4 |
| np=8 active-set, triangle-balanced      | 41.9    | kernel 2.7 .. 41.4 ms (1422%) |
| np=8 active-set, weighted contig        | **24.6**| kernel 19.2 .. 21.9 ms (**13.9%**, was 74% before the split); 1.71x triangle-balanced |

The complete loop -- coarse stats, weighted contiguous assignment, split
the tiles the tool flags, re-mesh only those, re-assign -- works on the
real 1 m^2 tiles with exactness held at every step (np=8 across two nodes
bit-exact against np=4 before and after the split).  The ranks holding
100M dry triangles (rank 7) and 19M wet ones (rank 1) finish within 8% of
each other, which is the cost model doing its job.

Extrapolation for the full 17.45G mesh from these numbers: 8.5 TB device
(62 H200s at 137 GB usable, minimum), 9.5 TB host (fits 25 gpuhopper nodes
at ~95-135 GB per rank), ~90 s per-rank build.

### Scale rehearsal: 60 H200s, 12.39 G triangles (job 177741904)

`tools/h200_1sqm_15node.pbs`, 15 gpuhopper nodes (the queue cap), 60 GPUs,
28 min wall, 2532 SU, exit 0.  The western 1035 tiles of the delta,
`x < 126 km`, **12,392,554,994 triangles** -- 71% of the full 1 m^2 mesh
and 41x the largest previous run.  Synthetic bed/stage (no real elevation
yet).  Results in `build/scale15/`.

*Exactness.*  No smaller-np golden fits in memory, so the gates are the
solver's own invariants: volume conserved to `-6.9e-14` relative (cell) and
`6.4e-15` (scatter) over 12.4 G cells, and the `state` line is
character-identical across all four runs -- full-mesh scatter, active-set
triangle-balanced and active-set weighted contig all report
`stage in [-16.9361, 78.1486], max |momentum| 2.511298e+02`, the same
max-momentum digits as the 8-GPU rehearsal.  The active set changes what is
computed, not what comes out.

*Costs at scale.*  488 B/triangle device, confirmed to three digits at two
different rank sizes (205.1M owned -> 93.24 GiB, 241.7M -> 109.89 GiB).
Host 7.34 TB for the job = 489 GB per node.  Build 104-130 s per rank
(0.5-0.6 us/triangle, a little above the 0.45 measured at 90M), 16-20 s
map-to-device.

| run (60 GPUs)                      | ms/step | note |
|------------------------------------|---------|------|
| full mesh, cell fluxes             | 185.9   | 66.7 Gcell-steps/s |
| full mesh, scatter                 | 173.5   | 71.4 Gcell-steps/s (1.19 G per GPU) |
| active-set, triangle-balanced      | 154.0   | kernel 11.6 .. 151.7 ms; rank 59 holds 135.7M of the 153M active cells |
| active-set, weighted contig        | **25.1**| kernel 13.5 .. 19.2 ms (42% max/min); **6.9x the full mesh** |

Global wetness is 1.24% (153.2M active of 12.39 G) and is identical under
both assignments, as it must be.  Note the printed `active : N% of cells
on average` line is a *mean of per-rank ratios*, not the global ratio: it
reads 1.22% when the wet cells sit on 200M-cell ranks and 11.13% when the
assignment concentrates them on 16M-cell ranks.  Same physics, different
denominator -- read the per-rank counts, not that percentage.

*The cost model is exact, its constant was not.*  Least squares on the 60
measured per-rank kernel times gives

    kernel ms = 0.0582 ns/cell x owned  +  1.0058 ns/cell x active

which reproduces the measured spread to 0.3% (predicted 13.47 .. 19.23 ms
against measured 13.51 .. 19.23).  The model form `tile_assign.py` uses is
right; the floor calibrated at 300 m^2 (`0.073`) is 26% high at this scale.
Re-scoring the same tiles with `--floor 0.058` predicts max kernel
19.23 -> 16.90 ms, **1.14x, free**.  Use `--floor 0.058` at 1 m^2.

*The floor now dominates.*  Per-kernel breakdown of the weighted run:

    active_sets     12.1409 ms   85.8%
    compute_fluxes   0.9384 ms    6.6%
    extrapolate+ck   0.6301 ms    4.5%
    forcing+update   0.2055 ms    1.5%
    prepare          0.2008 ms    1.4%
    boundary         0.0311 ms    0.2%

At 1.24% wetness the active-set *rebuild* -- an O(owned) scan run every
step (99 rebuilds in 100 steps) -- is 86% of the kernel time, and the
useful flux work is 6.6%.  A rank with 248M dry triangles and **zero**
active cells still costs 14.4 ms/step.  Perfect balance cannot go below
~12 ms/step of pure floor, against 2.6 ms/step of real work.  Balance
tuning is finished; the floor is the target.  The obvious lever: rebuild
every k steps against a ring-k dilated set (information moves one cell per
step under CFL, so ring-k preserves exactness) -- worth ~3-4x, far more
than anything left in the assignment.  Kernels are also only 58% of wall
(14.1 of 25.1 ms): the remaining 11 ms is the halo exchange and the
per-substep dt allreduce over 60 ranks, which becomes the next bottleneck
once the floor is cut.

*The time step is the blocker, and it is a mesh defect.*  The run achieved
**dt = 6.55e-4 s** (sim rate 0.026 simulated s per wall s = 38 wall-days
per simulated day).  Scanning the 47 wet tiles with `cdac_script/
tile_quality.py` (inradius, the quantity dt actually depends on): median
inradius 0.338 m, and **48 triangles out of 136.7 M wet ones** sit below
half that median.  The governing triangle is a **0.870 mm needle** in tile
1171 -- 0.0026 of the median, and tile 1171 is a 25-triangle coastal
fragment.  Twenty-five triangles in an 11,372 km^2 domain were setting the
time step for 12.4 G of them.  (Caution when reading `tile_quality.quality`:
it returns `(area, inradius, angle, centroid)` -- the *second* element is
the one that matters here.)

*Measured, not argued: what the corner fix does to those tiles.*  The 1 m^2
tiles predate `MARGIN 0.5` and the global `simplify(s/4)`.  Re-running
`split_delta.py` twice over the same polygon -- once at the shipped
settings (`MARGIN 0.3`, no simplify, taken from git at `74cd2175`) and once
at today's defaults -- then meshing the identical sub-tiles at `--area 1.0`
gives a controlled A/B.  The control reproduces the shipped mesh to four
significant figures (tile 11096: 9.973 mm regenerated vs 9.9727 mm in the
bmesh on gdata), so the comparison is sound:

| wet tile | min inradius, MARGIN 0.3 | min inradius, current | gain |
|----------|--------------------------|-----------------------|------|
| 1171  (governs dt) |   0.870 mm | **152.473 mm** | 175x |
| 10966 |   4.767 mm | 193.974 mm | 40.7x |
| 11096 |   9.973 mm | 195.744 mm | 19.6x |
| 10846 |  26.539 mm | 195.830 mm |  7.4x |
| 10844 |  39.229 mm | 195.145 mm |  5.0x |
| 10964 |  41.154 mm | 154.436 mm |  3.8x |
| 10847 |  66.743 mm | 194.032 mm |  2.9x |
| 212   |  76.089 mm | 136.786 mm |  1.8x |

Every one clears the `tile_quality.py` gate afterwards (min/median 0.41 to
0.58 against a 0.5 threshold), and the minimum angle goes to Triangle's
`-q28` target on the large tiles -- there is no sliver left, not a smaller
one.  Triangle counts are unchanged to within 0.07% (tile 212:
916,531 -> 917,146; tile 11096: 3,264,777 -> 3,264,480), so **the fix is
free**: same mesh size, same cost per step.

Taking the worst of the eight as the new limiter, the governing wet
inradius goes 0.870 mm -> 136.786 mm: **dt 6.55e-4 s -> ~0.10 s, 157x**,
purely geometric and independent of the flow.  That turns 38 wall-days per
simulated day into **~5.8 hours**.  (The other 39 wet tiles were all above
76 mm before the fix and were not re-measured; one of them could become the
new limiter, which is what `tile_quality.py` over the regenerated set is
for.)

*Does the full mesh fit 15 nodes?*  No.  At the measured 488 B/triangle,
17.45 G needs 290.8M per GPU = 132 GiB, above the 109.89 GiB proven here
and above what the card holds with runtime overhead.  Taking ~120 GiB as
the safe ceiling (264M triangles per GPU): **15 nodes hold 15.8 G, 91% of
the delta; the full mesh needs 17 nodes (67 GPUs).**  Multi-scale tiling,
which cut the test delta 355M -> 20M, is the way to make it fit rather than
more nodes.


**4x H200 at 300 m^2** (`tools/h200_mpi_tiles.pbs`, job 177612520, 58.2M
triangles, 158 tiles, results in `build/mpitiles/`): all 12 tiled gates
OK -- cell and `--phases` at atol 0 for np = 1, 2, 4 against the merged
mesh on one GPU, scatter and scatter+active-set at max relative 1e-15.
Triangle-balanced active-set run: 1.79 ms/step for 58M triangles
(the merged mesh on one GPU: 3.74), rank 0 holding 539k of 584k active
cells against a 51% kernel-time spread.  That run calibrates the H200 cost
model: 0.073 ns per local cell of never-skipped work vs 1.0 ns per active
cell, i.e. `--floor 0.073`.

**Weighted assignment on 4 H200s** (`tools/h200_tiles_balance.pbs`, job
177625536, same 58M mesh, 200 timed steps, results in
`build/tilesbal/tiles300/`; tiled np=4 and np=7 bit-exact vs the merged
mesh first):

| assignment (4 GPUs)             | active cells/step per rank       | kernel ms/step spread | wall ms/step | ghosts per rank |
|---------------------------------|----------------------------------|-----------------------|--------------|-----------------|
| triangle-balanced (default)     | 539k / 19k / 14k / 11k           | 1.05 .. 1.58 (50%)    | 1.80         | 10k .. 24k      |
| wetness-weighted, contig        | 345k / 206k / 23k / 11k          | 1.14 .. 1.25 (9%)     | **1.71**     | 8k .. 23k       |
| wetness-weighted, lpt           | 343k / 208k / 18k / 16k          | 1.17 .. 1.28 (9%)     | 2.72         | 67k .. 98k      |
| lpt re-weighted from own stats  | same                             | 1.17 .. 1.27 (9%)     | 2.78         | 67k .. 98k      |

Two lessons.  The cost model holds on the GPU: with the measured floor the
weighted assignments land within 9% on kernel time, from 50%.  And
**contiguity is not optional**: lpt reaches the same kernel balance but
scatters each rank's tiles over the basin, so the halo grows 4x and the
per-step exchange (host-staged pack/MPI/unpack, ~13 ns per ghost cell per
step here) costs more than the imbalance it removed -- wall time 2.72 vs
1.71 ms.  `tile_assign.py --method contig` is therefore the default choice;
its residual comes from the heaviest tile (0.30 of a share, tile 30), which
is what the refinement step addresses.  The remaining gap between kernel
max (1.25) and wall (1.71) is the per-step exchange + dt allreduce, ~0.45
ms at 8-23k ghosts -- proportionally large only because a 99%-dry 58M mesh
steps in 1.2 ms.

**Refined set** (tiles 30 and 31 split, 164 tiles; jobs 177626077 and
177630729, `build/tilesbal/tiles300_r1/`): bit-exact vs its merged mesh at
np=4/7.  The first contig run went to 14% because sub-tiles carry ids
>= 10000 and an id-ordered split bunched all six wet ones on one rank;
`tile_assign.py` now orders tiles by the Morton code of their centres.
With that: contig 1.16 .. 1.28 ms (10%), **wall 1.66 ms/step -- the best
of every run**, and its per-rank active counts 374k / 179k / 15k / 17k are
exactly what the model predicted offline (373k / 178k / 15k / 17k).  lpt
re-weighted from its own stats reaches 2% kernel balance yet 2.46 ms wall
-- the halo again.  Two conclusions for the 1 m^2 run: the weight model is
predictive enough to assign from a coarse run, and at ~1% wet the
never-skipped base work dominates, so the optimum deliberately gives the
wet ranks fewer triangles rather than equal active counts; splitting the
heaviest tile matters only once a single tile's *weight* approaches a
rank's share (it was 0.30 here, 0.09 after the split).

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

### Amortized active-set rebuild: `--active-every K` (2026-09-08)

The 60-GPU rehearsal left the active-set fast path with one dominant cost:
**the rebuild was 85.8% of kernel time** (12.14 of 14.15 ms), and a rank
holding 248M dry triangles and zero active cells still paid 14.4 ms/step.
The reason is that all four classification passes sweep the whole mesh --
wet flags, ring-1, ring-2 with compaction, and the owned-edge filter, about
200 B/cell in total, which at 206M cells is 41 GB per rebuild and lands
exactly on the measured time.

Rebuilding every K steps does **not** by itself fix that.  Wetness advances at
most 2 rings per RK2 step, so a set that stays valid for K steps has to be
dilated by 2K more rings, and a dilation done the way the passes above are
done is another full sweep each: `24 + 128K` bytes per cell per K steps, i.e.
128 B/cell/step against 200 as K grows.  A 1.56x ceiling, not the 3-4x the
handoff estimated.

What makes it pay is dilating **on the list**:

* every K steps, one full rebuild, then `core_active_dilate_rings()` grows its
  cell list by 2K+2 rings breadth-first, touching only the frontier -- O(list),
  never O(mesh).  That is the **candidate**: a provable superset of everything
  that can change for the next K steps (2K for wetness, +1 because the update
  reaches a ring beyond the fluxes, +1 so the neighbour flags the restricted
  passes read are still correct).
* every step, `core_build_active_sets_on()` runs the ordinary classification
  over the candidate instead of the mesh.  Cells outside it keep their flags
  from the last full build, which stay correct precisely because those cells
  are provably unchanged -- so a neighbour lookup reaching outside the list
  reads a correct (dry) flag.

The per-step lists come out as the **same sets** the full scan produces, so
the existing exactness argument carries over untouched; `--active-verify`
checks that directly by running both every step and comparing sorted lists.

**Two ordering effects, both measured, both load-bearing.**  The dilation
appends breadth-first, so its list is spatially scattered; iterating that
directly cost **22% on the step kernels** (2.35 -> 2.87 ms), swamping the
saving.  One coalesced sweep over the flags (`core_active_compact_flag`,
rebuild steps only) restores the ordering the full scan gives for free.  And
deriving the edge list from the cell list emits it in three strided blocks --
**2.2x on the scatter kernel** (0.57 -> 1.26 ms).  Fixed by narrowing the
*edge array* instead: the rebuild filters `owned_edges` down to the
candidate's own slots once, still ascending, and the per-step pass filters
that.  Neither effect is visible in a correctness test; both dominate.

V100, 58.2M real delta triangles (`build/tiles300`), 1.00% active, 40 steps:

| K | ms/step | rebuild | candidate | vs K=1 |
|---:|--------:|--------:|----------:|-------:|
|  1 |  19.90  | 12.25 (61.6%) |    --   | 1.00x |
|  2 |  16.52  |  8.48 (51.3%) |  1.33%  | 1.20x |
|  4 |  12.72  |  4.49 (35.3%) |  1.57%  | 1.56x |
|  8 |  10.83  |  2.57 (23.7%) |  2.03%  | 1.84x |
| 16 |  10.25  |  1.93 (18.8%) |  2.94%  | 1.94x |
| 32 |  10.17  |  1.84 (18.1%) |  4.72%  | 1.96x |
| 64 |  10.29  |  1.93 (18.7%) |  8.16%  | 1.93x |

Over 200 steps, where the amortization is properly sampled (13 and 7 full
rebuilds rather than 3 and 2), the same set gives:

| K | ms/step | rebuild | candidate | vs K=1 |
|---:|--------:|--------:|----------:|-------:|
|  1 |  16.58  | 12.50 (75.4%) |    --   | 1.00x |
| 16 |   6.50  |  1.77 (27.2%) |  2.94%  | **2.55x** |
| 32 |   6.36  |  1.57 (24.6%) |  4.73%  | **2.61x** |

**K = 16-32 is the flat optimum: the rebuild drops 8x and the step 2.6x.**
Past that the candidate itself grows faster than the amortization saves.  The optimum tracks the wet fraction, so it is a knob, not a constant:
at 19.8% active (`--case river`) the candidate reaches 32% of the mesh by
K=8 and nothing is gained -- this pays in the sparse regime the flood runs
actually live in, and the 1.24% of the hero run is sparser still than the
1.00% measured here.

Gates: `--check` against a K=1 golden passes on dam/dambumps/lake/river and
on the tiled delta at every K tried (diffs ~3e-14, scatter atomic order);
`--active-verify` reports zero set mismatches on the CPU build, the GPU build
and under MPI at np=2/4; the MPI tiled gate passes against the serial golden
at K=1/4/8.  Ghosts are seeded into the candidate and dilated with it, so a
front crossing a rank boundary is inside the superset before it arrives.

**It composes with ADER2**, and the two together are the whole picture on
this set (58.2M real delta triangles, V100, 200 steps, sim rate = simulated
seconds per wall second, the only honest cross-scheme metric since ms/step
ignores dt):

| scheme | K | sim rate | vs baseline |
|---|---:|---:|---:|
| rk2   |  1 | 1.083 | 1.00x |
| rk2   | 32 | 2.835 | 2.62x |
| ader2 | 32 | 4.486 | **4.14x** |

ADER2 gates clean with the amortized rebuild (`--check` against an ADER2
K=1 golden, K=8 and K=32, OK).  At rank scale it should be worth more than
the 1.58x it adds here: one flux call per step instead of two also halves the
halo exchanges and the per-substep dt allreduces, which are 42% of the wall
at 60 ranks and untouched by any kernel work.  The 15-node rehearsal ran the
default rk2, so this is unexploited there.

Costs two extra `anuga_int` arrays (16 B/tri, ~3% on top of 488) plus one
owned-edge array.  Incompatible with rain, which wets cells no superset can
predict -- and widespread rain activates the whole mesh anyway, so there is
nothing to win; the driver refuses the combination.

The primitives are in `gpu/core_kernels.c` (shared with production); only the
miniapp drives them so far, so `feature/gpu-active-set` still rebuilds every
step.

### Where the step actually goes: MPI phases (`--phases-sync`, 2026-09-08)

Amortizing the rebuild moved the bottleneck.  At 60 ranks the step was 25.1
ms with 14.15 ms of kernels; with the rebuild 8x cheaper the kernels are
~3.5 ms, so **~11 ms -- three quarters of the step -- is neither kernel nor
measured**.  It had only ever been inferred as wall-minus-kernels.

`--phases` now times the two MPI calls that gap is made of, `mpi_halo` (the
mid-step and end-of-step ghost exchanges) and `mpi_dt_allreduce` (the global
CFL min, once per substep), and the breakdown closes to 100% under MPI
instead of leaving a hole.

`--phases-sync` adds a barrier immediately before the dt allreduce and bills
its time to `mpi_wait(imbal)`.  This matters: **a collective absorbs the wait
for the slowest rank**, so without the barrier an imbalanced run reports its
imbalance as MPI cost and sends tuning after the wrong thing -- and the
15-node run had 1206% kernel imbalance on the triangle-balanced assignment.

First reading, 2 ranks on one V100, 1.77M triangles (small, so the halo is a
large share -- indicative of the ordering, not the magnitude):

    mpi_halo           1.1719 ms   24.6%
    mpi_wait(imbal)    0.0619 ms    1.3%
    mpi_dt_allreduce   0.0021 ms    0.0%

**The halo dominates and the allreduce is free** -- the opposite of the guess
that motivated the instrumentation, and worth knowing before anyone
amortizes an allreduce that costs 2 microseconds.  `tools/h200_mpi_phases.pbs`
asks it where it matters: 8 H200s across 2 nodes on the regenerated 1 sqm
lake block (26 tiles, 301M triangles), {rk2, ader2} x {rebuild every step,
every 32}, wetness-balanced.

### Device-side initialisation: `--device-init` (2026-09-03)

For serial generated meshes in row order, `--device-init` builds the domain
ON the device instead of building on the host and shipping it: every big
array is mapped `alloc` and filled by target kernels that are the closed-form
twin of the host build (same expressions, shared case functions in
`src/setup_cases.h`, connectivity from the closed form of the cross pattern,
boundary numbering and the scatter owned-edge list via deterministic block
prefix scans).  No production source changed: the premap makes
`gpu_domain_map_arrays`'s `map(to:)` clauses refcount no-ops, and setting
`gpu_initialized` skips it (and its free-memory check, which would
double-count) entirely.

Correctness, V100 (nvc 25.9) against pre-change host-init goldens, 30 steps:
**dam and river are bit-exact**; dambumps and lake differ by exactly 1 ulp in
elevation -- device `exp()` vs glibc `exp()` -- and pass at the default
`--check` tolerances.  On the CPU build (gcc, same libm both paths) all cases
are bit-exact.  Lake-at-rest holds at 2.1e-14, scatter==cell at ftol 1e-6,
and active-set-vs-full shows diffs identical to the host-init pair to every
printed digit (the difference is scatter atomic order, not the init; on this
V100 scatter is not run-to-run bitwise even unmodified, ~5e-13 after 30
steps).

Cost, V100, dambumps + scatter (`build/di_measure.csv`):

|  triangles | host build+map | host peak | device init | host peak |
|-----------:|---------------:|----------:|------------:|----------:|
|         4M |        2.83 s  | 2.06 GiB  |  0.174 s    | 0.41 GiB  |
|        16M |       11.36 s  | 7.83 GiB  |  0.250 s    | 1.21 GiB  |
|        64M |     (~45 s est)| (~33 GiB) |  0.545 s    |    --     |

The 64M run (29.1 GiB mapped, near the V100 wall) initialises in half a
second and steps at full speed (607 Mc/s).  Remaining host bytes are the
`bench_mesh` node/triangle arrays (~32 B/tri, removable by generating mesh
metadata only) plus one deliberate `update from(bed_centroid_values)`
(8 B/tri) so `--save`/`--check` snapshots see the elevation.

Not yet covered (host path unchanged): `--order morton|random`, `--mesh`,
`--tiles`, MPI.  See `DEVICE_INIT_PLAN.md` for the phases and the follow-ups.
