# Plan — make `multiprocessor_mode=2` + `gpu_offload=false` the standard distribution

Created: 2026-06-12. Owner: Stephen Roberts. Status: **in progress** (step 1 in review as PR #144).

## Goal

Ship ANUGA so that a standard `pip install` builds the unified `sw_domain_gpu_ext`
kernels for **CPU multicore** (`gpu_offload=false`) and runs the solver **and operators**
through them by default (`multiprocessor_mode=2`). This routes the rainfall / culvert /
inlet operators through C OpenMP kernels instead of serial Python, closing nearly the
entire OpenMP→MPI gap for single-node users with no MPI setup.

## Why (evidence)

Towradgi small (~256k tri, OMP_NUM_THREADS=16), gpu_offload=false build:

| Config | Time | Note |
|--------|------|------|
| mode=1 (Python ops) + metis_rcm | 18.64 s | current default path |
| mode=2 (C ops) + no reorder | 17.03 s | C operators alone |
| **mode=2 (C ops) + metis_rcm** | **12.27 s** | both — within ~11% of MPI-16 (11.08 s) |

Profiling showed operators cost ~40% of a mode-1 OpenMP run as serial Python that does
not scale with threads — the structural reason MPI beats OpenMP. The C kernels already
exist in `anuga/shallow_water/gpu/` (`gpu_rate_operator.c`, `gpu_culvert_operator.c`,
`gpu_inlet_operator.c`) and compile to `#pragma omp parallel for` under
`-DCPU_ONLY_MODE` (see `gpu_omp_macros.h`). Validated numerically: `test_DE_gpu_omp.py`
56/56 mode-1-vs-mode-2 equivalence tests pass on a `gpu_offload=false` build.

## Naming note

`multiprocessor_mode=2` is labelled "GPU" but really means "use the unified gpu_ext
kernels" — which target a GPU when `gpu_offload=true` and CPU multicore when
`gpu_offload=false`. The terminology should change (see step 4).

---

## Steps

### Step 1 — Deferred interface build  ✅ done, in review (PR #144)
Branch `feat/defer-gpu-interface-build`. Decouples *choosing* mode 2 from *building*
the device interface: mode recorded immediately, interface built eagerly if boundaries
are ready, else lazily at first `evolve()`. Removes the "boundaries before mode" ordering
constraint so mode 2 can be selected at `__init__`. Awaiting Jorge's review.

### Step 2 — Audit fall-back for kernel-less operators  ⬜ NEXT (correctness gate)
The equivalence tests only cover operators that HAVE C kernels (rate, inlet, culvert,
weir). Before any default switch, confirm every other operator behaves correctly in
mode 2 — **graceful fall-through to Python, never a silent no-op**.
- Operators to check: `Kinematic_viscosity_operator`, `Bed_shear_erosion_operator` and
  other erosion/sediment operators, `Sanddune_erosion_operator`, any `Rate_operator`
  subclasses with `rate_spatial`/`rate_xarray` paths, generic/user operators.
- For each: does its `__call__` run correctly when `multiprocessor_mode==2`? Trace the
  dispatch — `rate_operators.py` already falls through to Python for spatial/xarray
  rates (good pattern); confirm the others either have a kernel or fall through.
- Deliverable: a short table operator → {has C kernel | falls back to Python | BROKEN}.
  Any BROKEN must be fixed (add fallback) before step 5.

### Step 3 — Build/packaging: default `gpu_offload=false`  ⬜
- Confirm `sw_domain_gpu_ext` builds on a minimal **no-MPI** conda env (meson says it's
  always built with MPI stubs — verify on a clean env, since a mode-2 default hard-fails
  on import if the extension is missing).
- Decide the default value of the `gpu_offload` meson option for distribution builds
  (currently defaults to ? — check `meson.options`/`meson_options.txt`). For wheels/
  conda-forge it should be `false` (CPU multicore). GPU users opt in with
  `-Dgpu_offload=true -Dgpu_arch=...`.
- Verify CI builds and the conda recipes pass with the CPU-multicore extension.

### Step 4 — Rename the mode concept  ⬜ (do before flipping default, to avoid churn)
Replace the misleading "GPU" label. Proposed: a `backend` selector with values
`python` (= old mode 1), `openmp_c` (= mode 2 on CPU), `gpu` (= mode 2 with offload).
- Keep `set_multiprocessor_mode()` / `multiprocessor_mode` as deprecated aliases mapping
  1→python, 2→openmp_c|gpu (resolved by build). Emit `DeprecationWarning`.
- Update `-mpm` CLI help and docs.
- This is optional-but-recommended; can be deferred if it risks scope creep, but the
  default-switch reads badly if "GPU" is the CPU default.

### Step 5 — Flip the default  ⬜ (the actual switch)
- In `Domain.__init__` (shallow_water), default to mode 2 **with an auto-fallback**:
  if `sw_domain_gpu_ext` failed to import OR the build lacks the needed kernels, fall
  back to mode 1 and warn once. Never hard-fail for basic users.
- Gate on import success:
  ```python
  try:
      from anuga.shallow_water import sw_domain_gpu_ext   # noqa: F401
      _default_mode = MULTIPROCESSOR_GPU
  except Exception:
      _default_mode = MULTIPROCESSOR_OPENMP
  ```
- Run the FULL suite (`pytest --pyargs anuga`, not just --run-fast) with the new default
  on a `gpu_offload=false` build. Many unit tests build domains without driving a full
  evolve / setting boundaries — verify the deferred build (step 1) means they don't
  trip the mode-2 setup. Fix any that assume mode 1 internals.
- Re-run validation_tests/ (analytical + experimental) under the new default.

### Step 6 — Docs & comms  ⬜
- README / install docs: explain CPU-multicore is the default, GPU is opt-in.
- Note `OMP_NUM_THREADS` controls parallelism; recommend `-ro metis_rcm` (or wire a
  sensible default reorder — see open question).
- Migration note for users who relied on mode-1-specific behaviour.

---

## Open questions / risks
- **Determinism**: mode 2 reductions (OpenMP `reduction(+:)`) may differ from mode 1 at
  the ULP level → bit-for-bit reproducibility across thread counts is not guaranteed.
  Confirm regression-snapshot tests tolerate this (they passed at 56/56, but the full
  snapshot suite under a forced default should be checked in step 5).
- **Should reorder be automatic?** metis_rcm gives a big chunk of the mode-2 win. Consider
  a default `reorder='metis_rcm'` at domain build (or first evolve) rather than requiring
  `-ro`. Separate decision; don't couple to this plan unless cheap.
- **Windows**: mode-2 CPU build needs an OpenMP-capable compiler (mingw) — same constraint
  as the existing `sw_domain_openmp_ext`. Verify the gpu_ext builds under the Windows CI.
- **mode 1 retirement**: if mode 2/CPU is strictly better and fully covering, `sw_domain_
  openmp_ext` (mode 1) eventually becomes redundant. Out of scope here; revisit later.

## Key references
- PR #144 (step 1): https://github.com/anuga-community/anuga_core/pull/144
- Benchmark + validation detail: `claude/SESSION_GUIDE.md` → "CPU multicore via the unified
  gpu_ext C kernels" section.
- C kernel macros: `anuga/shallow_water/gpu/gpu_omp_macros.h`
- Mode dispatch: `anuga/shallow_water/shallow_water_domain.py` (`set_multiprocessor_mode`,
  `set_gpu_interface`, `_boundaries_ready`, evolve lazy hook); `anuga/config.py`
  (`MULTIPROCESSOR_OPENMP=1`, `MULTIPROCESSOR_GPU=2`).
- Equivalence tests: `anuga/shallow_water/tests/test_DE_gpu_omp.py`.
