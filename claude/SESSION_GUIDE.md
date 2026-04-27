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
| **v4.0.0** | `feat/sc26` → `develop` → `main` | In progress — feat/sc26 merged into develop |

**v3.3.2:** Shipped. Includes EPSG/CRS support, utm→pyproj replacement, sww_merge fixes,
sww2vtu converter, pyproj DeprecationWarning fixes, ruff linting, riverwall throughflow,
NPY002 fixes, GDAL removal, regression snapshot tests.

**v4.0.0:** `feat/sc26` has been merged into `develop` (2026-04-01). `develop` is now
the active working branch. feat/sc26 contains GPU/OpenMP-offloading work
(`multiprocessor_mode=2`) forming the basis of a **Supercomputing 2026 (SC26)** paper.

## Active branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable — v3.3.1 release |
| `develop` | Active development for v4.0.0 — contains feat/sc26 GPU work |
| `develop_sc26` | Working branch for GPU/SC26 incremental improvements |
| `develop_gpu` / `develop_cupy` | Earlier GPU experiments (CuPy-based) |
| `experiment/claude_culvert_refactor` | Culvert structure refactoring experiment |

Target PR branch is `develop` for all new work going into v4.0.0.

---

## Common tasks

### Run tests
```bash
pytest --pyargs anuga                    # full suite (~163s)
pytest --pyargs anuga --run-fast         # skip slow tests (~41s)
pytest --pyargs anuga -m slow            # only slow tests
pytest anuga/shallow_water/tests/test_shallow_water_domain.py  # single file
```

### Build
```bash
conda activate anuga_env_3.14
pip install --no-build-isolation -e .
```

### Check code quality
```bash
pyflakes anuga/path/to/module.py
autopep8 anuga/path/to/module.py
```

---

## Recent session summaries (sessions 21–26)

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
| Memory reporting | `anuga/utilities/system_tools.py::memory_stats()` |
| Timestepping output | `anuga/abstract_2d_finite_volumes/generic_domain.py::timestepping_statistics()` |
| Triangle quiet/verbose | `anuga/pmesh/mesh.py::_generateMesh_impl()` |
| TOML scenario config | `anuga/utilities/model_tools.py`, `examples/cairns_toml_excel/` |
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
