# Session Guide

How to orient a new Claude session for ANUGA development work.

---

## Quick orientation

```bash
git branch          # see all branches
git log --oneline -10   # recent commits
git status          # current state
```

Key files to read first:
- `CLAUDE.md` — build system, test commands, architecture overview
- `claude/PROGRESS.md` — what has been done and what remains
- `claude/DECISIONS.md` — why things are the way they are
- `claude/KNOWN_ISSUES.md` — surprises and gotchas

---

## Release roadmap

| Milestone | Branch | Status |
|-----------|--------|--------|
| **v3.3.2** | `develop` → `main` | **SHIPPED 2026-04-05** — tagged, PyPI + conda-forge published; propagated to GA remote |
| **v3.3.8** | cherry-picks → `main` | **SHIPPED 2026-07-11** — latest 3.3.x patch; tagged + PyPI (see session 48) |
| **v4.0.0** | `feat/sc26` → `develop` → `main` | In progress — feat/sc26 merged into develop |

**v3.3.2:** Shipped. Includes EPSG/CRS support, utm→pyproj replacement, sww_merge fixes,
sww2vtu converter, pyproj DeprecationWarning fixes, ruff linting, riverwall throughflow,
NPY002 fixes, GDAL removal, regression snapshot tests.

**v3.3.8:** Shipped (2026-07-11). Patch release on the 3.3.x line: parallel
structure-operator logging (accumulated_flow column, unique operator numbering,
sequential-matching filenames) and the weir critical-depth gravity fix
(hardcoded 9.81 → `domain.g`) — both cherry-picked to `main` via PR #182 — plus
the SWW large-t store crash (#149), osx-64 wheels (#142) and checkout 6→7 (#146).
**Release procedure:** the publish workflow uploads to PyPI on a *published
GitHub Release*, **not** on a bare tag push — so create the annotated tag
(`git tag -a 3.3.8 -m "ANUGA 3.3.8"`, bare version, no `v`), push it, then
`gh release create 3.3.8 --verify-tag`. conda-forge follows via its feedstock bot.

**v4.0.0:** `feat/sc26` has been merged into `develop` (2026-04-01). `develop` is now
the active working branch. feat/sc26 contains GPU/OpenMP-offloading work
(`multiprocessor_mode=2`) forming the basis of a **Supercomputing 2026 (SC26)** paper.

## Active branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable — v3.3.1 release |
| `develop` | Active development for v4.0.0 — contains GPU work + ADER-2 |
| `develop_sc26` | Working branch for GPU/SC26 incremental improvements |
| `develop_gpu` / `develop_cupy` | Earlier GPU experiments (CuPy-based) |
| `experiment/claude_culvert_refactor` | Culvert structure refactoring experiment |

Target PR branch is `develop` for all new work going into v4.0.0.

`develop_ader` merged into `develop` 2026-04-29.

---

## Common tasks

### Run tests
```bash
pytest --pyargs anuga                    # full suite (~163s)
pytest --pyargs anuga --run-fast         # skip slow tests (~41s)
pytest --pyargs anuga -m slow            # only slow tests
pytest anuga/shallow_water/tests/test_shallow_water_domain.py  # single file

# Per-test process isolation (required on a GPU build; works on any build).
# -cm legacy|unified sets the default compute mode for every child.
anuga_run_isolated_tests --pyargs anuga.shallow_water -cm unified   # 408 pass, 2 skip
anuga_run_isolated_tests                                            # the GPU file
```
See `CLAUDE.md` → "Testing a GPU-offload (nvc) build" for the full GPU recipe.

### Build
```bash
conda activate anuga_env_3.14
pip install --no-build-isolation -e .

# CPU multicore (distribution default — fast on CPU). gcc, host omp parallel for:
CC=gcc pip install --no-build-isolation -e . -Csetup-args=-Dgpu_offload=false

# GPU offload (fast on GPU). nvc from the NVIDIA HPC SDK:
CC=nvc pip install --no-build-isolation -e . \
    -Csetup-args=-Dgpu_offload=true -Csetup-args=-Dgpu_arch=cc120
```
**CPU and GPU are separate builds** — one nvc binary can't be fast on both (NVHPC's OpenMP
host fallback is single-threaded; see KNOWN_ISSUES). For CPU performance use the gcc
`gpu_offload=false` build; the GPU build is for GPU runs (and correctness A/B on CPU only).

### Compute model (v4.0) — two orthogonal knobs
```python
# Per-domain: which compute path (legacy mode 1 vs unified C kernels mode 2)
domain.set_compute_mode('legacy')    # = set_multiprocessor_mode(1): openmp_ext + Python ops
domain.set_compute_mode('unified')   # = set_multiprocessor_mode(2): unified gpu_ext C kernels

# Process-global (call before the first evolve()):
anuga.set_gpu_offload(True/False)    # offload unified kernels to GPU (GPU build only)
anuga.set_omp_num_threads(16)        # OpenMP thread count for the whole process
anuga.gpu_offload_enabled()          # resolved offload state
domain.compute_capabilities()        # {gpu_offload, num_gpu_devices, mpi, modes}
```
`cpu` = unified + offload off; `gpu` = unified + offload on (compositions, not modes).
CLI: `-mpm 1|2`, `-nt N`, `-go`/`-ngo` (gpu offload on/off), `-ro metis_rcm` (reorder).
Rationale + the nvc finding: `claude/DECISIONS.md` and `claude/KNOWN_ISSUES.md`.

### Check code quality
```bash
pyflakes anuga/path/to/module.py
autopep8 anuga/path/to/module.py
```

### Transmissive riverwalls (`Cd_through`) — issue #32

Riverwalls can leak *through* the wall body (below the crest), not just overtop —
via a per-riverwall `Cd_through` discharge coefficient in the hydraulic-parameter
dict (alongside `Qfactor, s1, s2, h1, h2`):

```python
riverWall_Par = {'fence': {'Cd_through': 0.5}}   # 0.0 (default) = impermeable
domain.riverwallData.create_riverwalls(riverWall, riverWall_Par)
```

- Physics: submerged orifice `Q = Cd_through · h_eff · √(2g·|Δstage|) · sign(Δstage)`,
  `h_eff` = upstream (driving-side) submerged depth below the crest (so it flows
  even when the downstream side is dry). Momentum contribution is zero (conservative).
- Applied **on top of** the weir/overtopping discharge, so a transmissive wall does
  seepage *and* overtopping. `Cd_through=0` reproduces the old impermeable behaviour.
- Lives in the **shared** flux kernel `anuga/shallow_water/gpu/core_kernels.c`
  (`gpu_adjust_edgeflux_with_throughflow`, hydraulic-properties **column 5**), which
  the legacy `_openmp_compute_fluxes_central` (mode 1) *and* the unified path (mode 2)
  both call — so it works in **both compute modes with bit-identical results**. Do
  NOT look for it in a `sw_domain_openmp_ext.c` (that's a build artifact and doesn't
  exist). Added in commit `6ebb4453`; documented on issue #32.

---

## Benchmark timings — Towradgi small (MSI laptop, RTX 5070, AMD Ryzen 9, 2026-06-11)

Case: `run_small_towradgi.py -ft 200 -ys 50`, ~256k triangles, DE1 algorithm.

### Baseline and reference

| Mode | Config | Time (s) | Speedup |
|------|--------|----------|---------|
| Serial | 1 rank / 1 thread | 96.27 | 1× |
| OpenMP (no reorder) | `OMP_NUM_THREADS=16`, mode=1 | 22.73 | 4.2× |
| MPI | `mpiexec -np 16`, mode=1 | 12.13 | 7.9× |
| MPI + RCM reorder | `mpiexec -np 16 -ro rcm` | 11.08 | **8.7×** |
| GPU | mode=2 (RTX 5070, cc120) | 6.25 | **15.4×** |
| GPU + Hilbert reorder | mode=2, `-ro hilbert` | 5.62 | **17.1×** |
| GPU + metis_rcm reorder | mode=2, `-ro metis_rcm` | 5.79 | 16.6× |

### Reordering comparison (all `OMP_NUM_THREADS=16`, mode=1)

| Reorder method | CLI | Time (s) | vs no-reorder |
|----------------|-----|----------|---------------|
| None | — | 22.73 | — |
| Metis-16 | `-ro metis` | 19.74 | −13% |
| Metis-Hilbert-16 | `-ro metis_hilbert` | 18.73 | −18% |
| Hilbert | `-ro hilbert` | 18.53 | −18% |
| Morton | `-ro morton` | 18.36 | −19% |
| RCM (global) | `-ro rcm` | 18.24 | −20% |
| **Metis-RCM-16** | **`-ro metis_rcm`** | **17.43** | **−23%** |
| Metis-RCM-24 | `-ro metis_rcm -rn 24` | 18.28 | −20% |
| Metis-RCM-32 | `-ro metis_rcm -rn 32` | 18.50 | −19% |

**Best reorder by mode: OpenMP → `metis_rcm` (17.43 s, 5.5×); MPI → `rcm` (11.63 s); GPU → `hilbert` (5.62 s, 17.1×).**

### Multi-GPU strong scaling — gadi (NVIDIA V100, GPU-aware MPI), 2026-06-17

First multi-GPU run via the GPU-aware-MPI build (`-Dgpu_aware_mpi=true`, nvc,
cc70). Times are the **evolve-loop** wall time for `run_small_towradgi.py -mpm 2
-ro hilbert -ft 3600` (18× the laptop table's `-ft 200`). The absolute seconds
are therefore NOT comparable to the 5.62 s reference above — only the V100
scaling below is meaningful.

| Config | Evolve (s) | Speedup vs 1×V100 | Parallel eff. |
|--------|-----------:|-------------------|---------------|
| RTX 5070 (1 GPU, local) | 200 | — | — |
| V100 ×1 | 151 | 1.00× | — |
| V100 ×2 | 105 | 1.44× | 72% |
| V100 ×4 | 83 | 1.82× | 45% |

- A single **V100 (151 s) beats the RTX 5070 (200 s)** — this solver is
  memory-bandwidth-bound and V100 HBM2 (~900 GB/s) > 5070 GDDR7.
- Strong scaling tails off (72% → 45%): "small" Towradgi (~257k tris) gives only
  ~64k cells/rank at 4 GPUs, so halo-exchange/compute ratio rises and the GPUs
  underutilise. A larger mesh should scale substantially better.

### CPU multicore via the unified gpu_ext C kernels (mode=2, gpu_offload=false)

The operators (rainfall/culverts/inlet) run as **serial Python** in mode=1 — profiling showed
they cost ~8.5 s of a 21 s run (~40%), and do NOT scale with OMP threads. This is the main
reason MPI beats OpenMP: MPI splits operators across ranks.

The `shallow_water/gpu/` C kernels (`gpu_rate_operator.c`, `gpu_culvert_operator.c`,
`gpu_inlet_operator.c`) implement these operators with `OMP_PARALLEL_LOOP_*` macros
(`gpu_omp_macros.h`) that compile to `#pragma omp parallel for` when built with
`gpu_offload=false` (`-DCPU_ONLY_MODE`). They are dispatched only when `multiprocessor_mode=2`.
So a `gpu_offload=false` build + `-mpm 2` runs the *entire* step — solver AND operators —
as multicore CPU OpenMP.

Build: `CC=gcc pip install --no-build-isolation -e . -Csetup-args=-Dgpu_offload=false`

| Config | Time (s) | Notes |
|--------|----------|-------|
| mode=1 (Python ops) + metis_rcm | 18.64 | previous best OpenMP |
| mode=2 (C ops) + no reorder | 17.03 | C operators alone |
| **mode=2 (C ops) + metis_rcm** | **12.27** | **both — nearly matches MPI** |
| MPI-16 + rcm (reference) | 11.08 | — |

The two optimisations are **super-additive**: C operators alone save ~1.6 s, reorder alone
~4 s, but together ~6.4 s — because in mode=2 the gpu_ext *solver* kernels are also strongly
cache-sensitive (17.03→12.27 s with reorder) while the operators become parallel C.
This closes the OpenMP→MPI gap to within ~11%.

**Re-confirmed 2026-06-13** on the RTX 5070 laptop with a fresh gcc `gpu_offload=false`
build (this session's compute-model work):

| `-mpm 2` config | no reorder | `-ro metis_rcm` | reorder win |
|-----------------|-----------:|----------------:|------------:|
| `-nt 1`  (serial)   | 95.12 s | 46.22 s | **2.06×** |
| `-nt 16` (16 cores) | 19.18 s | 11.27 s | **1.7×**  |

11.27 s now *matches* MPI-16+rcm (11.08 s) with no MPI setup. Reorder helps more
serially (cache locality dominates) than at 16 threads (closer to memory-bandwidth
bound on this single-NUMA box). Thread scaling 1→16: ~5.0× (no reorder), ~4.1×
(reorder). Validates a stock single-node `pip install` + `-mpm 2 -ro metis_rcm` as
the CPU path the migration targets.

**Note:** `gpu_offload=false` overwrites the GPU build. Rebuild with
`-Dgpu_offload=true -Dgpu_arch=cc120` (and `CC=nvc`) to restore GPU mode.

**Migration plan:** making `mode=2 + gpu_offload=false` the standard distribution
default is tracked in `claude/PLAN_default_mode2_cpu.md`. Step 1 (deferred interface
build) is in review as PR #144; step 2 (audit operator fall-back) is next.

Optimal reorder differs by execution model: CPU sequential traversal benefits from RCM
graph-bandwidth minimisation; GPU warp-parallel execution benefits from Hilbert's tight
spatial clustering for coalesced memory access. metis_rcm (5.79s) is slightly worse than
hilbert (5.62s) for GPU but better than no reorder (6.25s).

### MPI partition scheme comparison + reorder

| Config | Time (s) | vs default |
|--------|----------|------------|
| MPI-16, default (`-ps metis`) | 12.13 | — |
| MPI-16, `-ps morton` | 13.29 | +10% |
| MPI-16, `-ps hilbert` | 13.68 | +13% |
| MPI-16, `-ps rcm` | 15.59 | +29% |
| **MPI-16, `-ps metis -ro rcm`** | **11.08** | **−9%** |

**Metis must remain as the partition scheme** — it explicitly minimises inter-rank edge
cuts, which determines communication cost. RCM/Hilbert/Morton as partition schemes hurt
because they optimise spatial/topological locality but ignore communication.

The winning combination is orthogonal: Metis to partition (minimise communication) +
RCM to reorder within each rank (minimise cache misses). Benefits stack: 11.08 s vs
12.13 s baseline.

### OMP binding flags and numactl — do not use

Thread pinning (`OMP_PROC_BIND`, `OMP_PLACES`) slows culvert/polygon setup, which uses
scipy/numpy internally and benefits from free core migration. `numactl --interleave=all`
caused only 2 active threads → 57 s (worse than serial); likely a GCC OpenMP
affinity-detection quirk on single-NUMA systems.

Hardware: **single NUMA node**, 32 CPUs, 30 GB — no NUMA effects to exploit.

### Why MPI-16 still beats OpenMP-16+metis_rcm (~40% gap)

With a single NUMA node the gap is NOT memory topology. Remaining causes:
- **False sharing**: OpenMP thread chunk boundaries share 64-byte cache lines in the flux
  arrays. MPI processes have separate address spaces so this cannot happen.
- **Single-threaded operators**: culverts, rainfall, boundary conditions run serially.
- **Reduction overhead**: global stage max, wet count, water volume require barriers.

The false sharing fix requires padding flux array chunks to cache-line boundaries in the
C kernel — beyond Python-side mesh reordering. **17.43 s with metis_rcm is the ceiling
for Python-side OpenMP optimisation on this hardware.**

Key findings:
- RCM beats Hilbert/Morton because it minimises graph bandwidth (neighbour-index distance),
  directly targeting the flux kernel's `domain.neighbours[i]` access pattern.
- Metis-RCM beats global RCM because Metis produces compact sub-graphs; local RCM wavefronts
  stay tight rather than snaking across the whole mesh.
- Sweet spot for OpenMP is `n_procs = OMP_NUM_THREADS`: more partitions introduce seams
  within each thread's working set and hurt more than they help.
- For MPI, reordering applies after distribute() to each rank's local ~16k-triangle mesh;
  benefit is smaller but positive (5.5%).

---

## Weak-scaling benchmark — rectangular, constant timestep (MSI laptop, RTX 5070, AMD Ryzen 9 32C/30GB, 2026-06-26)

Case: `examples/parallel/run_parallel_rectangular_weak_scaling.py` — the extent grows with
the grid (`length = cell_size * sqrtN`, `cell_size=0.002`) so the triangle size, and hence
the explicit CFL timestep, stays ~constant as the triangle count grows (true weak scaling).
Each `-sn` doubling is ×4 triangles. DE0, `finaltime=0.015` (~276–318 steps), no reorder.
`dt` drifts only `4.72→5.44e-5` across the range (the fixed central bump is a smaller
fraction of larger domains). `-mp 2` offloads to the GPU; `-sn 1900` (14.44M, ~6.7 GB) is
the largest that fits the 8 GB RTX 5070 — `-sn 2000` (16M) OOMs the GPU.

16 cores where applicable: MPI = `mpiexec -np 16` × 1 thread/rank; OpenMP = serial `-mp 1`,
`OMP_NUM_THREADS=16`; GPU = `-mp 2`, np=1.

### Evolve time (s)

| triangles | MPI-16 | OpenMP-16 | GPU |
|-----------|-------:|----------:|----:|
| 0.90M (sn 475)  | 5.01  | 5.75  | 2.25 |
| 3.61M (sn 950)  | 22.03 | 22.56 | 8.03 |
| 14.44M (sn 1900)| 92.52 | 96.84 | 31.21 |

### Cost per (million-triangles · step) — flat = ideal weak scaling

| triangles | MPI-16 | OpenMP-16 | GPU |
|-----------|-------:|----------:|----:|
| 0.90M  | 17.5 ms | 20.1 ms | 7.86 ms |
| 3.61M  | 21.5 ms | 22.0 ms | 7.83 ms |
| 14.44M | 23.2 ms | 24.3 ms | 7.84 ms |
| **weak-scaling eff.** | 100→75% | 100→83% | 100→**~100%** |

### GPU evolve speedup

| triangles | vs MPI-16 | vs OpenMP-16 |
|-----------|----------:|-------------:|
| 0.90M  | 2.23× | 2.55× |
| 3.61M  | 2.74× | 2.81× |
| 14.44M | **2.96×** | **3.10×** |

Findings:
- **GPU weak-scales near-perfectly** (~7.84 ms/(Mtri·step), <0.5% spread over 16× size),
  2.2–3.1× faster than 16 CPU cores, lead widening with size.
- **MPI-16 evolve is only ~2–13% faster than OpenMP-16** here — this is a *pure-solver* case;
  the larger MPI>OpenMP gap on Towradgi came from serial operators (culverts/rainfall) that
  this rectangle has none of. Both are memory-bandwidth bound and degrade with size.
- **MPI setup overhead is large and grows**: `distribute` 1.3 → 5.8 → **29.9 s** plus rank-0
  `creation` 0.4 → 1.9 → 9.6 s. End-to-end at 14.44M: MPI 132.1 s vs OpenMP 104.4 s vs GPU
  38.7 s — for these short (~276-step) runs MPI's faster evolve does NOT repay its ~30 s
  distribute; on production-length runs the one-time setup amortizes and MPI's evolve edge wins.

---

## Recent session summaries (sessions 21–48)

**Session 48 (2026-07-11/12):** Culvert/weir mode-1 vs mode-2 reporting parity,
two real physics bugs, the **3.3.8 patch release**, then **partition-save
performance** (parallel dump, single-file layout) and docs. A user reported the parallel
(MPI) culvert/trap logs differed from sequential. Root-caused and fixed on
`develop` (branch `fix/culvert-parallel-mode2-reporting`, PRs pushed direct to
both develops, four commits `d55e1984`→`467255f0`):
- **Parallel structure logging** — parallel `.log` files were missing the
  `accumulated_flow` column (computed on master, never written) and mis-numbered
  operators (counter advanced only on the master proc, so a culvert and trap with
  different masters collided). Fixed: write the column + header, advance the
  counter on every rank in lockstep (participants in the ctor, non-participants
  in the factory `return None` branch), and drop the `_P<rank>` filename suffix so
  names match sequential. Verified np=4 and np=16 bit-identical to serial.
- **Mode-2 (GPU) stats write-back** — in mode 2 the batched `GPUCulvertManager`
  runs the Boyd/weir physics in C and skips the Python `__call__`, so the logs
  were **all zeros**. Added per-culvert report fields to `culvert_state`
  (C `gpu_culvert_operator.c`), a `gpu_culverts_get_report` getter, and a
  write-back in `apply_all` mirroring mode-1's accumulation.
- **boyd_pipe critical-depth bug** (real): the GPU C translation **divided** by
  `(bf·d)^2.5` where the Python reference **multiplies** → flow_area/velocity ~8%
  off (discharge matched only because it was inlet-controlled; outlet-controlled
  flow would corrupt discharge too). Fixed both dcrit pairs; added a boyd-pipe
  cross-mode regression test (the existing one only used boyd_box).
- **weir critical-depth gravity** (real, issue #181, closed): the Python
  `weir_orifice_trapezoid_function` hardcoded **9.81** in the trapezoid
  critical-depth Newton solve, vs the C using `domain.g` (9.8) → ~0.034% velocity
  offset in partly-full flow. Fixed to derive `g` from `domain.g` on both sides
  (Python grows a `g` arg, discharge routines pass `self.domain.g`); note this
  shifts mode-1 weir results ~0.034%. Guard test added.
- **3.3.8 release.** Cherry-picked the two **legacy** (non-GPU) fixes — parallel
  logging + weir gravity — to `main` via **PR #182** (GPU-only pipe/mode-2 fixes
  omitted; that machinery isn't on 3.3.x). Also in 3.3.8 since 3.3.7: SWW
  large-t store crash (#149), osx-64 wheels (#142), checkout 6→7 (#146). Tagged
  **annotated `3.3.8`** on the merge commit (bare version, no `v`, per the
  convention), published the GitHub Release → PyPI (21 files, trusted OIDC);
  conda-forge follows via feedstock bot.
- **Gotcha (memory):** MPI parallel + mode-2 needs `#ranks == #GPUs` or it hangs/
  garbles — validate GPU mode-2 in serial on a 1-GPU box.

Then, **partition-save performance** (a 173M-triangle / 18,400-partition run
measured ~1,000 s load + ~4,000 s partition + **~37,000 s to save the
partitions** — ~88% of the job; the dump routines wrote files in a serial loop):
- **Four stacking speedups (PR #183).**
  - **Opt-in parallel dump** (`num_workers`) for `sequential_distribute_dump`
    (domain) and `sequential_mesh_dump` (mesh). `extract_submesh` only *reads*
    the shared submesh, so a **fork** `ProcessPoolExecutor` shares the
    partitioned mesh **copy-on-write** (never pickled), and `gc.freeze()` keeps
    the big inherited structure out of the workers' GC scans (avoids both the
    rescan cost and COW page-dirtying). Default 1 = today's serial, memory-frugal
    path (it releases each rank as it writes; the parallel path keeps the whole
    submesh live — the memory-for-speed trade-off).
  - **Single-file domain dump** (`single_file=True`, new default): inline
    points/triangles/quantities into the one per-partition pickle instead of
    separate `.npy` files → **8 files/partition → 1** (~147k → 18.4k files at
    18,400 ranks), the dominant cost on metadata-bound Lustre/GPFS. The loader
    reads **both** layouts (filename-string vs inline array), so old dumps still
    load; `single_file=False` restores the legacy layout.
  - **Batched serial gc** — was collecting every rank (18,400 full graph
    traversals); now every 64 (big arrays are freed by refcounting anyway).
  - **Vectorized mesh `boundary_tag` write** — `_write_mesh_partition` wrote the
    NetCDF string variable **one row per boundary edge** (one HDF5 write each).
    cProfile showed this loop dominating; assembling the char array and writing
    it once took the mesh write **9.45 → 2.57 ms/file (~3.7×)**, byte-identical.
- **Partition file format — pickle vs NetCDF (decision).** Benchmarked on real
  partitions. **Domain → keep pickle**: writes ~6–8× faster than NetCDF at ~equal
  size, and it carries arbitrary Python state (boundary map, geo_reference) that
  NetCDF cannot hold without a blob. **Mesh → keep NetCDF**: its content is 100%
  arrays, it is **1.6× smaller** than pickle *and* portable/inspectable
  (`ncdump -h`), safe to load, and write-once/read-many. Prototyped a **hybrid**
  (bulk arrays as NetCDF variables + a small pickled config blob, one file) — it
  round-trips exactly but writes ~6× slower, so **not adopted**. Two formats for
  two genuinely different jobs; the mesh's apparent 34× gap was mostly the
  boundary_tag bug above, not NetCDF itself.
- **Docs (PRs #184, #185).** Documented `num_workers` / `single_file` and added
  "Performance for large partition counts" sections. Then **harmonized** the
  domain/mesh offline-partitioning pages: `sequential_distribute_dump` had a
  **stub docstring** (no `Parameters` block) and `sequential_distribute_load` had
  **none**, so the domain page's autodoc API rendered sparse next to the mesh
  page — both now have full numpydoc + `See Also`, and each dump states its
  on-disk format (domain → pickle, mesh → NetCDF). Also reordered the
  **Appendices** (builds → advanced usage → GPU internals → theory → Contributing)
  and shortened two long nav labels.
  **Gotcha:** #183's final commit landed on the branch *after* the merge captured
  it, so it silently missed `develop` — recovered via #184. Check the merge picked
  up the head you expect.
- **`pymetis` → core dependency** (was a `parallel` extra). `partition_mesh` needs
  it, and the *sequential* offline-partitioning preprocessing uses that without
  mpi4py — so gating it behind `parallel` meant a plain install could not
  partition a mesh.
- **GPU install docs.** Stated the real requirement is **a compiler with working
  OpenMP offloading** — the kernels are standard OpenMP `target`, not CUDA, so
  nvc is simply the best option *at the moment* (GCC's nvptx backend ICEs on the
  ANUGA kernels; LLVM/Clang offload, AMD AOMP, Intel icx untested but feasible).
- **`tools/install_anuga_nvc.sh` preflight (user report).** A user ran the script
  in a conda env **not** created from `environments/environment_<PY>.yml` and it
  died with `BackendUnavailable: Cannot import 'mesonpy'` — *after* nvc was found,
  so it read like a GPU/compiler failure when it was just a missing build dep.
  Cause: the build uses `pip install --no-build-isolation`, so pip does **not**
  create a temp build env — the **meson-python** backend and the rest of
  pyproject's `build-system.requires` must already be in the env. Added a
  **preflight** that checks `mesonpy`/`cython`/`pybind11`/`numpy` are importable
  and `meson`/`ninja` are on PATH, and on failure names the missing packages and
  gives the fix (`conda install -c conda-forge …`, or recreate from the matching
  `environments/environment_<ver>.yml`). It also prints the env's **actual**
  Python version — the script prefers an *already-activated* env and ignores `$PY`,
  so the banner could say `PY=3.14` while building into a 3.12 env (confusing in
  the user's log).
  **Triage tip:** `Cannot import 'mesonpy'` from this script is never an nvc
  problem — it is a missing build backend in the target env.
- **Test isolation: `Test_File_Conversion` (issue #186, closed).**
  `test_ferret2sww3` passed in isolation but ERRORed in a full run. `setUp` and
  several tests in the class create files with **bare relative names**
  (`most_small_*.nc`, `test_*.nc`, `test.sww`, `*.tms`, `*.asc`/`.dem`) in the
  **CWD**, and clean them up *inline at the end of each test body* — so a test
  that fails or is interrupted leaves stale files behind and the **next** run dies
  overwriting them with `NetCDFFile(..., 'w')` (triggered by a stale `tmp/` left by
  a previously failed `install_anuga.py`). Fixed by giving each test its **own
  temp working directory** (`setUp`: `mkdtemp` + `chdir`; `tearDown`: `chdir` back
  + `rmtree`), so cleanup is **unconditional** even on failure. Chose the temp-CWD
  approach over patching ferret3 alone because it fixes *every* CWD-writer in the
  class at once and stops the suite polluting the directory it runs from; safe
  because the class reads no fixture files by relative path (it creates everything
  it reads). Verified it still passes with **corrupt + read-only** leftovers
  planted in the CWD (the exact failure mode), leaving them untouched.
  **Pattern for similar failures:** "passes alone, fails in a full run" + files
  written to the CWD ⇒ isolate the whole TestCase in a temp CWD rather than
  chasing individual `os.remove` calls. (An earlier fix, `55d91479`, had used
  per-file `mkstemp` for `test_sww_extent` but never touched the ferret tests.)

**Session 47 (2026-07-07/08):** Documentation overhaul — restructure, API
cross-linking, meta-pages, and a warning-free Read the Docs build.
- **Issue #32 "Make riverwalls transmissive."** Verified the `Cd_through`
  submerged-orifice throughflow is already active in **both** compute modes:
  legacy mode-1's `_openmp_compute_fluxes_central` delegates to the shared
  `core_compute_fluxes_central` in `core_kernels.c` (no mode gating), so the
  earlier "not in legacy" reading was wrong. Confirmed empirically (mode-1 and
  mode-2 give bit-identical throughflow). Documented on the issue and here.
- **Docs overhaul (PR #157).** Split into a standard-user **Contents** vs
  advanced **Appendices** structure; added a landing quick-start, a
  **Conventions & units** primer, and an evolve **Stability/blow-ups** section;
  moved Parallelisation into Contents and TOML/ANUGA-Viewer/QGIS into the
  standard sections; led the appendices with the developer + new **GPU install**
  pages. Converted the narrative "Reference" blocks to compact autosummary tables
  linking into the **API Reference** (expanded API consolidated there), and made
  every class page's method summary link to per-method signatures
  (`autodoc_default_options={'members':True}` + `sphinx.ext.napoleon`). Reframed
  the **ANUGA Viewer** as the recommended fast viewer for large `.sww` (dropped
  "legacy"). Content-review fixes (typos, heading levels, missing pip path,
  smoke-test, quantity units). New **Citing**, **Contributing**, **Glossary**
  meta-pages (surfacing `CITATION.cff`, `CONTRIBUTING.rst`, Apache-2.0).
- **Docstring + tooling fixes.** Reformatted malformed-RST docstrings surfaced by
  autodoc `members` (`Quantity`/`Domain` ×6 in #158, `internal_boundary_operator`
  ×2 in #160). `install_anuga_nvc.sh` now builds into an already-activated conda
  env if present (#158).
- **Warning-free build (PRs #159, #161).** Fixed the local warnings —
  `html_static_path` (`_static` dir) and the `Geo_reference.epsg` duplicate
  (napoleon renders a class-docstring *Attribute* **and** the real property →
  `napoleon_use_ivar = True`). Then **watched Read the Docs** (it builds
  `develop`): it had surfaced 62 warnings the local build hid — 56 `ipython3`
  Pygments-lexer (RTD lacks IPython → add `ipython` to `docs/requirements.txt`)
  plus the operator docstrings. RTD `develop` now builds **clean** (only the
  harmless MPI-less `Could not import mpi4py`). Added a Contributing "Building
  the documentation" note on reproducing RTD locally (#162). **Lesson:** a
  locally-installed IPython and `-D nbsphinx_execute=never` hid warnings RTD
  shows — build docs from a clean `docs/requirements.txt` env.
- **Branch policy (PR #163 + memory).** Recorded in `ROADMAP.md`: **do not merge
  `develop` → `main` until the team cuts v4.0.0**; all work/syncs stay on
  `develop`.
- **PDF structure + Conventions polish.** Noted the Unix-epoch time origin
  (seconds from 00:00:00 UTC 1 Jan 1970; `set_starttime` via `datetime`/
  `zoneinfo`) in the Conventions page (#166). Fixed the PDF putting every
  section under a single "Quick start" chapter — the `.. toctree::` sat *inside*
  the Quick-start section on the landing page, so moved Quick start to its own
  `quickstart.rst` (first in the toctree); verified against the live RTD PDF that
  Quick start and the other sections are now sibling chapters (#167).
- **Versioning diagnosis + wheel-repair cherry-pick (2026-07-08).**
  `git describe` on `develop` showed `1.3.1-4093-g…`, but the package version is
  correct — `anuga.__version__ = 3.3.6.dev767+g…` (`_git_version.py` uses
  `git describe --tags`). Two causes: the `3.x` tags are **lightweight**, so a
  plain `git describe` skips them and falls back to the old **annotated** `1.3.1`
  (`git describe --tags` → `3.3.6-…`); and `3.3.7` is a release tagged on `main`
  **not reachable from `develop`**. That 3.3.7 release had exactly one commit
  `develop` lacked — `e4700a7b` *"Repair wheels on all platforms so they are
  self-contained"* (delocate/delvewheel/auditwheel so mac/win/linux wheels are
  self-contained) — **cherry-picked into `develop`** (`014451da`; pushed **direct
  to `anuga-community/develop`** as an authorised one-off, no PR). Recorded a
  **tagging convention** in `ROADMAP.md` (PR #168): future release tags must be
  **annotated**, bare version, no `v` prefix (`git tag -a 3.3.8 -m "ANUGA 3.3.8"`)
  so plain `git describe` is correct.
- **nvc GPU rebuild + GPU sanity check.** Rebuilt via `tools/install_anuga_nvc.sh`
  (active-env path from #158) → `3.3.6.dev772+g08d4c10b`, 65/65 isolated GPU
  tests, and a dam-break sanity sim (GPU offload engaged, mass conserved to ~1e-13,
  GPU vs CPU-legacy ~4e-7).
- **GPU `Time_boundary` mode-1/mode-2 divergence — root-caused and fixed
  (PR #171, issue #170).** A rising-tide flood diverged ~4e-3 between mode-1 and
  mode-2. Isolated it: **single-substep** algorithms (DE0, DE_ader2) agree to
  machine precision, **multi-substep** (DE1/DE2) diverge — and only with
  **Python-evaluated boundaries** (Time/File/wave/Flather); reflective/steady are
  exact. Cause: the single-call **C RK loop** (`_evolve_one_rk*_step_c`) sets
  time-varying boundaries on the device **once per step**, reusing that value for
  every RK substep, whereas mode-1 calls `update_boundary()` **per substep**
  (an O(dt) boundary-forcing error). Fix (**option B**): route domains with any
  Python-evaluated boundary to the Python-orchestrated GPU loop (refreshes per
  substep → bit-matches mode-1; benchmarked GPU cost ≤~4%, within noise). Added
  `Test_GPU_TimeBoundarySubstep`; 69/69 GPU tests green. Filed **issue #170** for
  the proper per-substep C-loop fix (option A).
- **Fractional-step operator/structure timing (PRs #174, #175, #177; issue #176).**
  Checked operators/structures for the yieldstep-vs-inner-step concern — they are
  fine (applied **every inner timestep** in both modes: 13/13 and 46/46 evals ≫
  yieldsteps). But found a related **operator-evaluation-time** bug: fractional-step
  operators (applied by the evolve loop *before* it advances time to t+dt) should
  see the pre-step time **t**, and DE0/DE_ader2 do — but **DE1 (rk2)** and **DE2
  (rk3)** left `relative_time` advanced, so operators evaluated forcing at t+dt
  ("one step too far"), ~4e-4 for a time-varying rate. For **DE1** only mode-1 was
  affected (diverged from mode-2), fixed by restoring the pre-step time in the
  mode-1 rk2 body (**PR #174**). For **DE2** *both* modes advanced (mode-1 body +
  mode-2 C **and** GPU loops), so it stayed self-consistent and the cross-mode
  check missed it — fixed all three rk3 paths (**PR #177**, closes #176). Now all
  four algorithms evaluate operators at t in both modes. Guards: `test_operator_timing.py`
  (CPU, mode-1 — runs on any build) + `Test_GPU_OperatorTimeAlignment` (cross-mode);
  the GPU-only guard was split out to a CPU test in **PR #175**. No prior test used
  a time-varying operator (coverage gap). CPU suite 2610 pass, GPU 73/73.
- All merged to `anuga-community/develop` (PRs **#157–#177**, admin-merged by
  number, except the one-off direct cherry-pick `014451da`); RTD `develop` HTML
  **and PDF** confirmed clean.

**Session 46 (2026-07-06):** Issue #33 (memory) documentation + measurement,
plus follow-ups.
- **Issue #33 "Reduce Memory usage".** Documented on the issue everything already
  implemented for v4.0.0: the **Quantity per-type allocation** (QM1–QM7 — each
  quantity allocates only the arrays its `qty_type` needs instead of the blanket
  9; lazy `vertex_values`; centroid-primary elevation; lazy gradients/`phi`;
  shared gradient workspace `22559a5b`; ~54–58% off quantity memory) and the
  **domain C work-array reduction** (DM1: 9 dead arrays removed + deferred to
  first evolve; DM2: riverwall arrays lazy; ~740 MB at 2.25M tris), plus the
  exported `memory_stats`/`quantity_memory_stats`/`domain_memory_stats`/
  `domain_struct_stats` instrumentation. Then **re-ran the issue's exact
  benchmark** (`run_parallel_rectangular.py`, `mpiexec -np 2`, proc-0 Max RSS):
  **710→511 MB (−28%), 2.5 GB→1.37 GB (−45%), 5.1 GB→2.74 GB (−46%)** — RSS
  roughly **halved at 2.25M triangles**, the saving growing with N. Recorded the
  numbers in `PROGRESS_ARCHIVE.md` (PR #155) and posted them to the issue. The
  `print_domain_memory_stats` breakdown shows `river wall 0.00 MB` (DM2 lazy) and
  the trimmed work arrays (DM1). Remaining lever: rank-0's peak building the full
  domain before `distribute`.
- **`tools/install_ubuntu.sh`.** Combined the ~90%-identical `install_ubuntu_2X_04.sh`
  scripts into one version-aware `install_ubuntu.sh` (22.04/24.04/26.04 `case`,
  auto-derived python version); fixed the earlier cleanup's broken 20.04 dispatch
  and README mislabel (PR #154).
- **PR triage.** Reviewed **#148** (multi-compiler flag tuning, GCC 15/NVHPC/ICX)
  — recommended gating the FP-semantics flags (`-ffinite-math-only`,
  `-fassociative-math`, `-Mfprelaxed`) behind an opt-in `-Dfast_math` option so
  the default `pip install` keeps strict IEEE, while keeping the safe flags +
  `pow→cbrt` rewrite as default; asked @samcom12 to make that change.
- **Synced `develop` → `anuga-community`** (PRs #153/#154/#155); upstream CI green.

**Session 45 (2026-07-02 – 07-05):** Geodata CI breakage (libjxl), NumPy-2.5
warnings, PR triage, and a `tools/` cleanup.
- **Geodata CI failure.** A red CI on PR #150 (SeanWong's timestep clamp) turned
  out to be unrelated: conda-forge omits `libjxl` on Linux/macOS for py3.11–3.14,
  so `libgdal-core 3.12.3` can't load (`libjxl.so.0.11: cannot open shared object
  file`) and the whole geodata stack (fiona/rasterio/shapely) is down. Fixes,
  merged to `develop`: (1) skip-guard the one un-guarded geodata test
  (`test_rain_with_polygon_csv`) — **PR #151**; (2) a `spatialInputUtil`
  diagnostic that surfaces the real import error instead of the blind
  `except ImportError` (this is what revealed `libjxl`); (3) pin `libjxl <0.12`
  in `environment_3.10..3.14.yml` to force the `0.11` soname — **PR #152**.
  Restored ~20 geodata tests (upstream CI: 2667 passed, **0** `requires rasterio`
  skips). Confirmed transient-vs-persistent by re-running CI and by a fresh
  `conda env create` locally (imports clean — the same versions work; CI's solve
  was the outlier).
- **NumPy 2.5 `arr.shape=` deprecations.** Surfaced once the geodata tests
  un-skipped. Converted ANUGA's own 14 sites to `arr.reshape(...)`; suppressed
  the remaining one (rasterio ≤1.5.0 doing it internally in `raster.read()`) via
  a `filterwarnings` ignore, removable after a rasterio update.
- **PR triage.** Reviewed **#147** (BFS locality partitioning — recommended
  *not* merging: functionally equivalent to existing RCM but 3–4× slower, per the
  author's own benchmarks). Reviewed **#150** (timestep clamp at yield/final
  boundaries — a good dedup + real negative-timestep fix; LGTM with minor notes;
  merged).
- **`tools/` cleanup.** Renamed `install_gpu_anuga.sh` → `install_anuga_nvc.sh`
  (it's the nvc GPU build); removed 12 obsolete scripts (Travis/AppVeyor, Python
  3.8 era, Ubuntu 20.04, Travis-era conda, dead `old_div` helper); added
  `tools/README.md` naming the canonical path (`install_miniforge*` +
  `pip install -e .`) and `environments/*.yml`/`CLAUDE.md` as authoritative.
  `tools/` went 22 → 11 files (commits `dac3d8a5`, `2cf16e91`, `8bce344a`,
  `a02fe190`).

**Session 44 (2026-06-29 – 07-01):** SWW writer crash fix, laptop-guide
reconciliation, and GPU install-script fixes.
- **SWW crash on `main`.** `Write_sww.store_quantities()` raised `IndexError:
  index 0 is out of bounds for axis 0 with size 0` deep into a long run
  (t≈6960 s). The checkpoint/overwrite path locates the existing time slot with
  an *absolute* `1e-14` tolerance — below the float64 ULP (~9e-13) for any t
  beyond a few tens of seconds — so `numpy.where` returns empty and `check[0][0]`
  throws, killing the run at the write. Scaled the tolerance to the time
  magnitude (`1e-9*max(1,|t|)`) and append-with-warning when nothing matches;
  reproduced at t≈6960 and verified (commit `d04fa3ef`). Cherry-picked to `main`
  as **PR #149**.
- **Laptop guide.** Reconciled `cdac_script/ANUGA_Laptop_Guide.docx`
  benchmarking claims against this file and fixed the docx: the "dual-CCD NUMA
  limits OpenMP" claim contradicted the measured **single-NUMA-node** finding
  (the OpenMP-vs-MPI gap is false sharing in the flux arrays + serial operators);
  corrected "~3x"→"~4–5x" on 16 cores; added GPU/MPI speedup figures; repaired a
  garbled `<12 GB VRAM` bullet (commit `4fc2f134`).
- **GPU install script (`tools/install_anuga_nvc.sh`).** (1) Run
  `scripts/anuga_run_isolated_tests.py` instead of `pytest test_DE_gpu_omp.py`,
  which auto-skips on a GPU build so the test step ran nothing (commit
  `99bbc29f`). (2) `rm -rf build/cp*` before the nvc build — meson-python reads
  `CC` only on the *first* configure of a build dir, so a leftover
  gcc-configured `build/cp314` kept gcc and failed the `gpu_offload=true` guard
  (commit `ec079cd7`). Verified end to end: fresh nvc build succeeds, isolated
  runner reports **65/65 GPU tests pass**. Documented the compiler-switch gotcha
  in `KNOWN_ISSUES.md` (commit `89624513`).
- Gitignored the generated `validation_tests/case_studies/towradgi/MODEL_OUTPUTS/`
  artifacts (commit `6cfb4e62`).

**Session 43 (2026-06-26):** GPU test skip visibility + weak-scaling benchmark.
`test_DE_gpu_omp.py`'s module-level skip (on a GPU-offload build, to dodge the
NVHPC mode-2 abort) now also emits a `UserWarning` with the reason, so a plain
`pytest` shows *why* it skipped in the warnings summary without `-rs` (commit
`e37c0e76`). New `examples/parallel/run_parallel_rectangular_weak_scaling.py`
(based on `run_parallel_rectangular.py`): grows the extent with the grid
(`length = cell_size*sqrtN`) **and** normalises the sloped bed by the extent, so
triangle size, water depth, and hence the explicit CFL timestep all stay constant
as the triangle count grows — true weak scaling (commit `490cc00a`). Ran a
constant-`dt` sweep (sn 475/950/1900 → 0.9M/3.6M/14.4M triangles) comparing
**MPI-16 vs OpenMP-16 vs GPU** (full table above, "Weak-scaling benchmark"): GPU
weak-scales near-perfectly (~7.84 ms/(Mtri·step), 2.2–3.1× faster than 16 CPU
cores); MPI≈OpenMP for this pure-solver case (the Towradgi MPI>OpenMP gap was
serial operators, absent here); MPI's growing `distribute` overhead (→30 s at
14.4M) isn't repaid on these short runs.

**Session 42 (2026-06-25):** Test robustness, PR #144 conflict resolution, and an
MPI build gotcha. Made the run_toml end-to-end test directory-independent: locate
the checkout via `cwd` (not just `__file__`, which lives in site-packages for an
installed pkg), then self-contained — prefer the installed `anuga_run_toml`
console command + inline-generated inputs, so it runs from any directory and
skips only when no runner exists (commits `316620e5`, `e9400c6d`). Resolved the
conflicts on **PR #144** ("Defer GPU/offload interface build to first evolve"):
develop's compute-mode refactor (`set_compute_mode`/`_ensure_gpu_interface`/
`_boundaries_ready`, commit `289bd5c7`) had independently *superseded* the PR's
mechanism, so both conflicted files were resolved to develop's version; the PR's
only remaining net contribution is the operator-audit (graceful GPU-init fallback
in `rate_operators.py`/`inlet_operator.py`); merge pushed to the PR branch
(`e73adbfa`). Diagnosed why `gpu_has_mpi()` was False on the gcc CPU build (the
four `test_parallel_sw_flow_gpu_*` skipping): `sw_domain_gpu_ext` was built
against the single-process MPI stubs because a **reused meson build dir** never
re-ran MPI detection; fixed by rebuilding into a fresh `-Cbuild-dir` (mpi4py
fallback then finds `mpicc`/`mpi.h`), documented in `KNOWN_ISSUES.md` (commit
`a07216dd`). Those four tests then pass (8/8).

**Session 41 (2026-06-23):** `anuga_run_toml` TOML scenario runner — examples,
georeferencing, and an end-to-end test. Added `examples/run_toml/` with
self-contained `simple/` (dam break) and `complex/` (floodplain) scenarios + per-
scenario READMEs (commits `0c3c9546`, `d2f85ad4`). Added `"EPSG:<code>"`
projection support to the TOML parser and made the runner resolve the scenario
CRS to an EPSG code and call `domain.set_epsg()`, so zone/hemisphere/EPSG
propagate into the SWW (commits `ca0eedd4`, `43010ab9`, `c4ef89f5`; cairns uses
`EPSG:32755`, simple `-56`→zone 56/EPSG:32756). Consolidated the real-DEM Cairns
scenario as a third TOML example `examples/run_toml/cairns/` (TOML-only; the
~9 MB DEM moved to the shared `examples/data/cairns/`; `cairns_toml_excel/` kept
as the legacy Excel front-end) and added the runner smoke test
`anuga/scenario/tests/test_run_toml_end_to_end.py` (commit `036cb484`). Fixes: a
parallel race opening the run log before its dir exists (`acc826f5`), `Make_Geotif`
misreading a MaskedArray as a filename (`7fe595b0`); plus `-ro/-rn` reorder args
on `run_parallel_rectangular` (`39f4ccc6`) and GUI font scaling (`c0b599b3`,
`f0c7418d`).

**Session 40 (2026-06-17):** Mode-2 ('unified') triage on the **GPU build** + the
isolated runner became a first-class installed tool. Running
`anuga_run_isolated_tests --pyargs anuga.shallow_water` under the unified default on
a GPU-offload build surfaced 11 failures — all GPU-offload artifacts, not solver
regressions (Session 39's all-green was the CPU build, where device memory == host
memory). Two groups: (1) **9 white-box tests** call `compute_forcing_terms()` /
`compute_fluxes()` and assert on the host `semi_implicit_update` / `explicit_update`
arrays, which mode-2 GPU computes on-device and never syncs back (`test_forcing.py`,
`test_friction.py`, `test_physics_sw.py` Manning cases, `test_data_manager.py::
test_sww_extrema`); (2) **2 numerical tests** compare against legacy-recorded
references and diverge at ~1e-6 from mode-2's reduction/eval order
(`test_regression_snapshots.py::test_dam_break_DE1_stage_snapshot`,
`test_sww_interrogate.py::test_get_maximum_inundation_de0`). Fix: pin each to legacy
with `domain.set_compute_mode('legacy')` (snapshot helpers pinned so the whole file
is deterministic); no-op for the legacy default. The shallow_water set is now
**408 pass / 2 skip** under `-cm unified` on the GPU build (commit `0c50947d`).
Then **moved** the harness `anuga/shallow_water/tests/run_isolated_tests.py` →
`scripts/anuga_run_isolated_tests.py`, installed it to bindir via `meson.build`
(`configure_file`, matching the other `anuga_*` scripts), and made it install-safe
(importlib-resolved default target; cwd-seeded rootdir; `_abs_nodeid` passes through
absolute/`--pyargs` ids) — commit `37eccc6d`. Added **`-cm`/`--compute-mode
{legacy,unified}`** to set `ANUGA_DEFAULT_COMPUTE_MODE` for every child (omit to
inherit; banner prints the resolved mode) — commit `34401cde`. Docs: `KNOWN_ISSUES.md`
(green-run note + new command/flag), `CLAUDE.md` (testing section), new
`CONVENTIONS.md` → "Compute mode in tests" (commit `f57d0532` + this session's docs).

**Session 39 (2026-06-15):** Mode-2 ('unified') unit-suite triage — drove the fast suite
to **zero failures** under `ANUGA_DEFAULT_COMPUTE_MODE=unified` (2657 passed; 2658 in
legacy). Four genuine code fixes (commit `6bec9f8b`): (1) `recorded_min/max_timestep`
now records the CFL step before the yield cap (new `GD.recorded_flux_timestep` exposed by
the C evolve kernels, read by the Python step wrappers); (2) `gpu_manning_friction`
dispatches sloped-vs-flat on `domain.use_sloped_mannings` (new GD flag); (3) Domain
pickling restored — `__getstate__`/`__setstate__` drop the non-picklable cdef GPUDomain
and rebuild lazily; (4) `set_boundary()` invalidates+rebuilds the mode-2 device interface
so mid-run boundary changes (Reflective→Dirichlet) reach the device — fixed a progressive
runup-inundation divergence. Test/doc adaptations (commit `ff9083b4`): pinned legacy on the
deprecated forcing-function tests (Rainfall/Inflow are mode-1-only, skipped in mode 2),
skipped `test_default_is_legacy` under the env override, and pinned
`run_parallel_riverwall.py` to legacy. **Known gap:** mode-2 riverwall flux diverges from
legacy (~0.095 m on the riverwall case) — root-cause/fix tracked as FUTURE_WORK P1.9.

**Session 38 (2026-06-11):** Mesh reordering suite for OpenMP/MPI/GPU cache locality.
Added `rcm_partition()` (scipy `reverse_cuthill_mckee` on triangle adjacency graph),
`_rcm_within_partition()` helper, and `metis_rcm`/`metis_hilbert` hybrid methods to
`anuga/parallel/partitioning.py`. `reorder_domain()` gained `n_procs` parameter
(defaults to `OMP_NUM_THREADS`); for metis variants this controls partition count.
Added `-rn`/`--reorder_nprocs` CLI arg to standard parser so partition count can be
set independently of `OMP_NUM_THREADS`. Added `metis_rcm`, `metis_hilbert`, `rcm` to
`-ro` CLI choices. `hilbert_order_from_points()` refactored to use new
`hilbert_codes_from_points()` helper (raw codes without argsort, needed for
per-partition Hilbert sorting). Best OpenMP result: **`-ro metis_rcm` at 17.43 s (5.5×)**
vs 22.73 s baseline — 23% improvement. Sweet spot is `n_procs=OMP_NUM_THREADS=16`.
MPI + RCM: 11.63 s vs 12.31 s baseline (5.5% improvement). GPU + Hilbert: 5.62 s vs
6.25 s baseline (10% improvement, 17.1×). GPU + metis_rcm: 5.79 s (better than no
reorder but worse than hilbert — GPU warps prefer Hilbert's spatial clustering for
coalesced access over RCM's sequential bandwidth minimisation). Hardware is single NUMA
node so NUMA effects are absent; remaining OpenMP vs MPI gap is false sharing in flux
arrays. OMP binding flags and numactl both harmful. Best reorder per mode:
OpenMP→metis_rcm, MPI→rcm, GPU→hilbert. Full benchmark table in SESSION_GUIDE.
Commits `b5ed2b6d`, `93670929`.

**Session 37 (2026-06-11):** CLI improvements to `run_small_towradgi.py` and standard
arg parser. Added `--multiprocessor_mode`/`-mpm` (choices 1/2, default 1) to standard
parser; `run_small_towradgi.py` reads it from `args`. Added `-os`/`--outputstep` to
standard parser (SUPPRESS default, scripts default to `yieldstep`). Wired `-ys`/`-ft`
overrides in `run_small_towradgi.py` via `getattr(args, key, default)` — removed
`verbose = False` hardcode that was silently swallowing `--verbose`. Fixed
`Rate_operator` crash on empty `local_rates[fid]` array (NumPy 2.x raises `ValueError`
on `.max()` of zero-size array; MPI ranks with no polygon triangles hit this). Towradgi
benchmark results recorded (see table above). Commits `a51eec3d`–`7e6b8f40`.

**Session 36 (2026-06-11):** GCC CPU build fix after `gcc-15-offload-nvptx` install:
`-foffload=disable` added to GCC `openmp_c_args` and link args in `meson.build` so
`sw_domain_openmp_ext` (CPU-only) doesn't trigger the nvptx mkoffload pass. `--threads`
argument added to `scripts/anuga_benchmark_omp.py` (comma-separated OMP_NUM_THREADS
override, e.g. `--threads 1,2,4,8,16`). Parallel GPU test hang fix: added
`gpu_get_num_devices()` to `gpu_domain_core.c`/`gpu_domain.h`, exposed as
`get_num_gpu_devices()` in `sw_domain_gpu_ext.pyx`; parallel GPU tests now skip per-rank
when `real_gpu_available() and get_num_gpu_devices() < N` — CPU_ONLY_MODE builds are
unaffected and continue to run. Towradgi benchmark data downloaded (~86 MB). Full test
suite: 2852 passed. Commits `a96c3ca6`, `c5eb80eb`.

**Session 35 (2026-06-10):** GPU build on RTX 5070 laptop (Blackwell GB206M, 8 GB,
cc120). GCC 15 nvptx backend ICEs on `core_kernels.c` (`ompdevlow` GIMPLE pass
segfault in `core_extrapolate_second_order_edge`) — unfixable at source level.
Solution: NVIDIA HPC SDK 26.3 (`nvc`) installed via apt; `meson` auto-detects it
as `nvidia_hpc`; build command: `CC=nvc pip install --no-build-isolation -e .
-Csetup-args=-Dgpu_offload=true -Csetup-args=-Dgpu_arch=cc120`. All 56 GPU tests
pass. New `tools/install_anuga_nvc.sh`: auto-detects nvc under
`/opt/nvidia/hpc_sdk/Linux_x86_64/`, configurable via `PY`/`GPU_ARCH`/`NVHPC_ROOT`.
Fixed: `pytest-regressions` missing from all 5 intel conda environment YMLs (already
present in non-intel variants; pip-installed in existing env). `KNOWN_ISSUES.md`
updated with full GPU build recipe. Full test suite: 2852 passed. Commits
`6a10f5d0`, `f47967d2`.

**Session 21 (2026-04-15):** Domain work-array memory reduction — ~740 MB saved
at N=2.25M. DM1: 9 dead C work arrays removed from `_ensure_work_arrays()`;
only 3 live arrays remain. DM2: `edge_flux_type`/`edge_river_wall_counter` lazy
for non-riverwall simulations; `sw_domain.h` NULL guard added. DM3:
`domain_memory_stats`/`print_domain_memory_stats`/`domain_struct_stats`/`print_domain_struct_stats`
added to `system_tools.py` and exported from `anuga`.

**Session 22 (2026-04-21):** `anuga_animate_sww_gui` major feature release.
Parallel frame generation (ProcessPoolExecutor, fork on Linux, up to 4 workers)
via new `_animate_worker.py`. Zoom region (Set Zoom / Reset Zoom). `elev`
quantity: static or time-varying (erosion); terrain colormap; timeseries panel.
Fix View Mesh multiple windows, Cancel button, app-close hang. Sphinx docs with
automated screenshot capture. Commit `ebc68c37`.

**Session 23 (2026-04-24):** `anuga_sww_gui` GUI overhaul (renamed from
`anuga_animate_sww_gui`). Baked overlays (elevation contours + mesh at correct
z-order). Multi-point timeseries picking (tab10 palette, legend, CSV export,
Clear button). Save Frame time-selection dialog. 3-tab ttk.Notebook UI. Basemap
checkbox for mesh viewer and Save Mesh dialog. Sphinx RST, help, screenshots
updated. Commit `49c5b7d8`.

**Session 24 (2026-04-24):** P2.5 `Rate_operator` factories — `rainfall()` and
`inflow()` classmethods, input validation (TypeError/ValueError), 13 new tests.
P1.4/P1.8: `gauge.py` print hygiene (all `log.info()`), `file_function.py`
FIXMEs resolved. P3.6 erosion delta-bed view (`elev_delta` quantity, RdBu_r
colormap, symmetric auto-limits, 6 new tests). All P1 FUTURE_WORK items done.

**Session 25 (2026-04-25):** P2.3 `create_riverwalls` refactor — extracted 3
helpers, orchestrator ~50 lines. P2.2 `Generic_Domain.__init__` refactor —
extracted 4 helpers, `__init__` ~25 lines. Split `test_shallow_water_domain.py`
into 5 files; cleanup −101 lines. Fix 383 pytest warnings (animate.py,
rate_operators.py, pyproject.toml). P2.8/P2.9 scenario validation and TOML
docs done. claude/ rationalisation.

**Session 26 (2026-04-26):** P3.3 `fit_interpolate` L-curve alpha auto-selection.
`Fit.select_alpha()`: 20 log-spaced candidates (1e-6 … 100), numerically stable
RSS via normal equations, max-curvature corner detection, fallback to DEFAULT_ALPHA.
`dok_to_csr` added to `fitsmooth_ext.pyx` (non-destructive DOK→CSR). `alpha='auto'`
wired in `Fit.fit()`. Removed dead `_RawCSR`/`_SumRawCSR`. 13 new tests covering
row_ptr extension, multi-attribute, degenerate/interior paths. fit.py 85→92%.
P2.7 gauge modernisation fully done (session continuation). Commit `12864187`.

**Session 27 (2026-04-27):** `Kinematic_viscosity_operator` MPI-parallel (Option B
distributed CG). Phase 1: removed Apple OpenMP guards from 4 C files. Phase 2:
`parabolic_solve` serial path routed through C CG (`cg_solve_c_precon`) with Jacobi
preconditioner. Phase 3: full distributed CG — `_exchange_ghost_vector` (MPI tag 198
non-blocking), `_distributed_dot` (Allreduce), `_parabolic_matvec_distributed` (ghost
exchange before SpMV), `_parabolic_solve_distributed` (n_full-length CG loop). `parallel_safe()`
returns True. New `run_parallel_kv_operator.py` + `test_parallel_kv_operator.py`
(serial-vs-parallel xvel, max diff 8.6×10⁻⁶). New `run_parallel_kv_unit_tests.py`
+ `test_parallel_kv_unit_tests.py` (4 in-process MPI assertions for each primitive).
Bug fix: `test_select_alpha_degenerate_falls_back_to_default` platform-dependent on
Windows py3.10/3.11/3.13 — now uses `return_curve=True` to branch on actual kappa.
Commits `61418742`, `5498f98d`. All CI passed.

**Session 32 (2026-05-03):** CI fixes. macOS `sw_domain_gpu_ext` import failure
resolved: `_omp_target_is_present` missing from conda-forge libomp — fixed by adding
`static inline` stubs + macro redirects for all OpenMP 4.5 target-alloc API functions
(`omp_target_alloc/free/memcpy`, `omp_target_is_present`, `omp_get_initial_device`) in
`gpu_omp_macros.h` under `CPU_ONLY_MODE`. Also: MPI ABI fix for Linux conda envs (use
`mpicc -show` compile flags, not hardcoded sys.prefix), Windows MPI detection (Library/
include path, msmpi), macOS rpath for libmpi.dylib (`@executable_path/../lib` +
`DYLD_LIBRARY_PATH` in CI), macOS simd-reduction warnings (dropped `simd` from reduction
macros), CI Node.js 24 upgrades (setup-miniconda v4, setup-python v6), skip-reason
diagnostics in test_DE_gpu_omp.py. macOS GPU tests now run (not skipped).
Commits `94c4d74f`–`7cc69fac`.

**Session 31 (2026-04-29):** ADER-2 GPU wiring + optimisation. `gpu_ader_ck_predictor` /
`gpu_evolve_one_ader2_step` added to `gpu_kernels.c` + declared in `gpu_domain.h` (fix for
missing header causing Windows/3.12 CI failure). `evolve_one_ader2_step` dispatches to
`_evolve_one_ader2_step_c` / `_evolve_one_ader2_step_gpu` in GPU mode. `DE_ader2` flow
algorithm added to `set_flow_algorithm()` (DE1 settings + ader2 timestepping). Fused
predict-extrapolate C-K loop (P3.8): merged extrapolation and C-K predictor into a single
kernel pass — eliminates the second extrapolation entirely → **1.75× faster than DE1**.
`FLOPS_ADER_PREDICTOR=105` constant. Fix: `create_sts_boundary`
in `sts.py` now calls `fid.close()` in a `try/finally` block (Windows WinError 32).
`develop_ader` merged into `develop`. Commits `e9d15803`–`3b00dc79`.

**Session 30 (2026-04-29):** ADER-2 timestepping via Cauchy-Kovalewski predictor.
`core_ader_ck_predictor()` in `core_kernels.c`: recovers cell slopes from the 2×2
edge system (edges 0 and 1), evaluates well-balanced SWE time derivatives locally
(dz/dx = dw/dx − dh/dx from reconstruction), advances centroids by dt in-place.
`evolve_one_ader2_step()` in `shallow_water_domain.py`: backup → CFL step →
C-K predictor(dt/2) → midpoint flux → restore Q^n → update. `boundary_flux_integral_operator`
updated for 'ader2'. 10 tests: well-balance (flat/sloped), mass conservation, dam-break
consistency, non-negative depths. All 2656 fast-suite tests pass. Commit `825f1e5f`.

**Session 29 (2026-04-28):** Investigated numpy `_NoValue` reload issue triggered
by `--cov=anuga.submodule` targeted coverage runs. Root cause: coverage.py's
`sys_modules_saved()` context in `inorout.py` calls `importlib.util.find_spec()`
on the subpackage, auto-importing the parent chain including numpy, then purging
all new imports from sys.modules. The second real numpy import fires the reload
guard and corrupts the C-level `_NoValue` singleton. Fix: removed `numpy._pytesttester.PytestTester`
from `anuga/__init__.py` (replaced with plain `def test()`) and all 23 subpackage
`__init__.py` files (was unused boilerplate). Targeted submodule `--cov=` runs are
unfixable from conftest.py (pytest-cov starts before conftest.py loads);
use `--cov=anuga` always. Documented in KNOWN_ISSUES.md. Commit `af71f10b`.

**Session 28 (2026-04-27):** P2.6 fast-suite coverage continued. `anuga/file/`:
`test_netcdf_nc.py` (10 tests, netcdf.py 34%→100%), `test_sts.py` (11 tests,
sts.py 47%→89%). `anuga/structures/`: 9 new tests in `test_inlet_operator.py`
(inlet_operator.py 45%→64%); 16 new tests in `test_structure_operator.py`
(structure_operator.py 65%→96% — enquiry getters, setters, skew 4-point, error
paths, print/timestepping stats, non-constant elevation warning). Overall fast
suite: 58.13% → 58.68%.

---

## File locations for common operations

| Task | Files |
|------|-------|
| Add public API export | `anuga/__init__.py` (import + `__all__`) |
| Add slow test marker | `@pytest.mark.slow` decorator or module-level `pytestmark` |
| Configure pytest options | `conftest.py` (repo root), `pyproject.toml` `[tool.pytest.ini_options]` |
| Per-test process isolation runner | `scripts/anuga_run_isolated_tests.py` (installed `anuga_run_isolated_tests`; `-cm legacy\|unified`) |
| Default compute mode | `ANUGA_DEFAULT_COMPUTE_MODE` env var; `domain.set_compute_mode('legacy'\|'unified')` |
| Memory reporting | `anuga/utilities/system_tools.py::memory_stats()` |
| Timestepping output | `anuga/abstract_2d_finite_volumes/generic_domain.py::timestepping_statistics()` |
| Triangle quiet/verbose | `anuga/pmesh/mesh.py::_generateMesh_impl()` |
| TOML scenario config | `anuga/scenario/`, `scripts/anuga_run_toml.py`, `examples/run_toml/` (simple/complex/cairns; shared DEM in `examples/data/cairns/`); legacy Excel front-end in `examples/cairns_toml_excel/` |
| Single-process benchmark | `benchmarks/run_benchmarks.py` + `benchmarks/compare_benchmarks.py` |
| MPI distribution benchmark | `benchmarks/distribute_benchmarks.py` + `benchmarks/run_benchmark_grid.py` |

---

## Key reference documents

| Document | URL |
|----------|-----|
| Hydrata refactor plan | https://github.com/Hydrata/anuga_core/blob/anuga-4.0-refactor-plan/REFACTOR_PLAN.md |
| anuga-community GitHub | https://github.com/anuga-community/anuga_core |
| Hydrata fork | https://github.com/Hydrata/anuga_core |

## Next priorities

See `claude/FUTURE_WORK.md` for the full prioritised list (P1–P3).

**SC26 (needs GPU hardware):** G4.1 Gordon Bell metrics, G4.2 physical benchmark validation, G4.3 multi-node strong scaling (scripts in `benchmarks/` and `scripts/hpc/` are ready).

**Best standalone value:** P2.6 fast-suite coverage, P2.7 gauge module modernisation, P2.4 culvert compute_rates deduplication.
