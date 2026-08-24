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

### Timestepping schemes (`--scheme`)

`rk2 | ader2 | euler | rk3`, each selecting its ANUGA preset (DE1 / DE_ader2 /
DE0 / DE2). The honest cross-scheme metric is the printed **sim rate**
(simulated seconds per wall second), since ms/step ignores dt. Measured at 16M
triangles: **ADER2 delivers 1.64x the sim rate of RK2** — same CFL timestep
(dt 0.00734 vs 0.00732), same formal order, one flux call per step instead of
two, with the C-K predictor costing ~7 ms against the ~35 ms flux+extrapolate
round it replaces.

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

### Measured dead ends (kept out, documented so nobody re-tries them blind)

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

- Serial only: MPI is stubbed out via `gpu_mpi_stubs.h` (`nprocs == 1`, no halo
  exchange). Multi-GPU decomposition is out of scope here.
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
