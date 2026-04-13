# ANUGA Code & Documentation Improvement Progress

Last updated: 2026-04-13 (session 19 cont)
Branch: `develop` (all feature branches merged)

---

## Overview

| Area | Total actions | Done | Remaining |
|------|--------------|------|-----------|
| Code improvements (original list) | 60 | 50 | 10 |
| Documentation improvements | 20 | 20 | 0 |
| Additional enhancements | 27 | 27 | 0 |
| Hydrata Phase 0 — Test infrastructure | 5 | 5 | 0 |
| Hydrata Phase 1 — Dependencies | 4 | 4 | 0 |
| Hydrata Phase 2 — Linting | 3 | 3 | 0 |
| Hydrata Phase 3 — Deduplication | 4 | 4 | 0 |
| Hydrata Phase 4 — Coverage | 3 | 3 | 0 |
| GPU Phase 1 — Correctness & tests | 7 | 7 | 0 |
| GPU Phase 2 — Performance validation | 4 | 4 | 0 |
| GPU Phase 3 — Feature parity | 4 | 4 | 0 |
| GPU Phase 4 — SC26 paper | 3 | 0 | 3 |
| Riverwall throughflow | 6 | 6 | 0 |
| Quantity memory reduction | 7 | 7 | 0 |
| Benchmark suite | 2 | 2 | 0 |
| Bug fixes | 7 | 7 | 0 |
| **Total** | **169** | **161** | **8** |

---

## Code Improvement Actions

Source: `docs/code_improvement_actions.md`
Generated: 2026-03-23

### Priority 1 — Quick wins (bug risk, no behaviour change)

#### 1.1 Fix mutable default arguments (~43 functions)

- [x] `anuga/caching/caching.py:145` *(2026-03-24)*
- [x] `anuga/file/sww.py:535` *(2026-03-24)*
- [x] `anuga/parallel/parallel_boyd_box_operator.py:22` *(2026-03-24)*
- [x] `anuga/abstract_2d_finite_volumes/ermapper_grids.py:8,88,203` *(2026-03-24)*
- [x] Full repo audit — also fixed parallel_structure_operator, parallel_boyd_pipe_operator, parallel_weir_orifice_trapezoid_operator, parallel_internal_boundary_operator, parallel_operator_factory, riverwall, util.py *(2026-03-24)*

#### 1.2 Replace bare `except:` with specific exception types

- [x] `anuga/utilities/system_tools.py` — already OK *(2026-03-24)*
- [x] `anuga/shallow_water/boundaries.py` — already OK *(2026-03-24)*
- [x] `anuga/caching/caching.py` — already OK *(2026-03-24)*
- [x] `anuga/abstract_2d_finite_volumes/tests/test_quantity.py` — already OK *(2026-03-24)*
- [x] `anuga/abstract_2d_finite_volumes/tests/test_generic_domain.py` — already OK *(2026-03-24)*
- [x] `anuga/file_conversion/dem2pts.py` — already OK *(2026-03-24)*

#### 1.3 Convert file operations to use `with` statements

- [x] `anuga/file/csv_file.py:47,196,206,216,224` *(2026-03-24)*
- [x] `anuga/file/ungenerate.py:16` *(2026-03-24)*
- [ ] `anuga/file/urs.py:29` — skipped: file handle stored as `self.mux_file` for iterator lifecycle
- [x] `anuga/utilities/system_tools.py:29` *(2026-03-24)*
- [x] Audit `anuga/file/` for remaining bare `open()` calls — all test files fixed (test_mux.py, test_csv.py, test_ungenerate.py); `urs.py:29` intentionally kept (iterator lifecycle) *(2026-04-03)*

#### 1.4 Fix invalid escape sequences in docstrings

- [x] `anuga/utilities/norms.py:15` *(2026-03-24)*
- [x] `anuga/utilities/system_tools.py:133` — no issue found *(2026-03-24)*
- [x] `python -W error::DeprecationWarning -c "import anuga"` — clean *(2026-03-24)*

#### 1.5 Delete large commented-out dead code

- [x] `anuga/file_conversion/dem2pts.py:164–281` — 118-line pre-vectorisation loop deleted *(2026-03-24)*
- [x] `anuga/abstract_2d_finite_volumes/neighbour_mesh.py:615–668` — 53-line disabled block deleted *(2026-03-24)*
- [x] Grep for large legacy comment blocks in `shallow_water/` and `operators/` — deleted `##` debug block in `boundaries.py` and two debug dump blocks in `test_sw_domain_openmp.py` *(2026-04-03)*

### Priority 2 — Correctness and stability ✅ Complete

- [x] 2.1 Fix silent error suppression in `set_quantity.py` — documented expected ValueError *(2026-03-24)*
- [x] 2.2 Log xarray import failures in `rate_operators.py` — `log.debug(...)` *(2026-03-24)*
- [x] 2.3 Address FIXME items — `boyd_box_operator.py`, `fit.py`, `polygon.py`, `rate_operators.py` *(2026-03-24)*

### Priority 3 — Test coverage

#### 3.1 Add tests for untested operator classes ✅ Complete

- [x] `Bed_shear_erosion_operator`, `Circular_erosion_operator`, `Flat_slice_erosion_operator`, `Flat_fill_slice_erosion_operator` *(2026-03-24)*
- [x] `Collect_max_quantities_operator`, `Collect_max_stage_operator` — `test_collect_operators.py` *(2026-03-24)*
- [x] `Elliptic_operator` — `test_elliptic_operator.py` *(2026-03-24)*
- [x] `Circular_rate_operator`, `Circular_set_quantity_operator`, `Circular_set_stage_operator` *(2026-03-24)*

#### 3.2 Add tests for untested structure classes

- [x] `Structure_operator` base class — `test_structure_operator.py` *(2026-03-24)*
- [x] `Internal_boundary_operator` — `test_internal_boundary_operator.py` *(2026-03-24)*
- [x] `RiverWall` — `Test_riverwall_notebook` class (5 tests): `create_domain_from_regions` with breaklines, wall name registration, crest heights, edge alignment, impermeable sub-crest, overtopping. *(Done 2026-04-13, commit a62e9c96)*

#### 3.3 Add tests for untested scenario module

- [ ] `anuga/scenario/` — deferred; depends on compiled `spatialInputUtil` and real test data

### Priority 4 — API and code quality

- [ ] 4.1 Reduce parameter counts (`gauge.py`, `generic_domain.py`, `boyd_box_operator.py`) — deferred
- [x] 4.2 Standardise naming in `pmesh/mesh.py` — 39 methods renamed; camelCase kept as deprecated wrappers *(2026-03-24)*
- [x] 4.3 Deprecate camelCase `get_CFL`/`set_CFL` in `generic_domain.py` *(2026-03-24)*
- [x] 4.4 Add `__all__` to `anuga/__init__.py` and sub-package `__init__.py` files *(2026-03-24)*

### Priority 5 — Performance

- [x] 5.1 Vectorise loops — `fit.py:598`, `csv_file.py:136`, `util.py:786` *(2026-03-24)*
- [x] 5.2 `polygon.intersection()` — not a confirmed hotspot; deferred *(2026-03-24)*
- [ ] `util.py:301` (`csv2timeseries_graphs`) — dominated by matplotlib; defer

### Priority 6 — Documentation improvements ✅ Complete

- [x] 6.1 `boyd_box_operator.py`, `boyd_pipe_operator.py`, `weir_orifice_trapezoid_operator.py` — full NumPy-style docstrings *(2026-03-24)*
- [x] 6.2 `rate_operators.py`, `erosion_operators.py` — Returns sections added *(2026-03-24)*

---

## Documentation Improvement Actions

Source: `docs/doc_improvement_actions.md`
Generated: 2026-03-23
**All 20 items complete** ✅

| # | Item | Done |
|---|------|------|
| 1 | Fill out `visualisation/use_domain_plotter.rst` | 2026-03-23 |
| 2 | Fix `reference/index.rst` navigation | 2026-03-23 |
| 3 | Fix `anuga_user_manual/version.txt` stale SVN variables | 2026-03-23 |
| 4 | Add `setup_anuga_script/checkpointing.rst` | 2026-03-23 |
| 5 | Add `reference/file_formats.rst` | 2026-03-23 |
| 6 | Add `troubleshooting.rst` | 2026-03-23 |
| 7 | Expand `setup_anuga_script/boundaries.rst` | 2026-03-23 |
| 8 | Add comparison table to `setup_anuga_script/operators.rst` | 2026-03-23 |
| 9 | Add descriptions to `examples/index.rst` notebooks | 2026-03-23 |
| 10 | Add MPI section to `install_anuga_developers.rst` | 2026-03-23 |
| 11 | Clarify OpenMP support in `use_parallel_openmp.rst` | 2026-03-23 |
| 12 | Soften QGIS version in `use_qgis.rst` | 2026-03-23 |
| 13 | Add parallel decision guide to `parallel/index.rst` | 2026-03-23 |
| 14 | Add annotated TOML example to `toml_scenario/index.rst` | 2026-03-23 |
| 15 | Add GPU/`multiprocessor_mode=2` note in parallel docs | 2026-03-23 |
| 16 | Standardise quantity names in `initial_conditions.rst` | 2026-03-23 |
| 17 | Reconcile Python version statements across install docs | 2026-03-23 |
| 18 | Port mathematical background into Sphinx | 2026-03-23 |
| 19 | Add cross-references from RST pages to user manual | 2026-03-23 |
| 20 | Add `reference/validation.rst` | 2026-03-23 |

---

## Additional Enhancements (beyond original action lists)

These were completed during sessions as natural extensions or user requests:

| Item | Files | Done |
|------|-------|------|
| Suppress triangle library verbose output in pytest | `anuga/pmesh/mesh.py` | 2026-03-26 |
| Suppress General_mesh logging in test | `anuga/abstract_2d_finite_volumes/tests/test_pmesh_to_mesh.py` | 2026-03-26 |
| Replace `print_timestepping_statistics()` calls in tests with `pass` | `anuga/shallow_water/tests/test_sw_domain_openmp.py` | 2026-03-26 |
| Add `memory_stats()` and `print_memory_stats()` | `anuga/utilities/system_tools.py` | 2026-03-26 |
| Add memory usage to `timestepping_statistics()` output | `anuga/abstract_2d_finite_volumes/generic_domain.py` | 2026-03-26 |
| Export `memory_stats`, `print_memory_stats` from `anuga` | `anuga/__init__.py` | 2026-03-26 |
| Export `distribute_basic_mesh`, `distribute_basic_mesh_collaborative` from `anuga` | `anuga/__init__.py` | 2026-03-26 |
| Add `basic_mesh_from_mesh_file()` factory function | `anuga/abstract_2d_finite_volumes/basic_mesh.py` | 2026-03-26 |
| Export `basic_mesh_from_mesh_file` from `anuga` | `anuga/__init__.py` | 2026-03-26 |
| Fast/slow test infrastructure (`--run-fast` flag, `@pytest.mark.slow`) | `conftest.py`, `pyproject.toml` | 2026-03-26 |
| Mark 10 slow tests across 5 test files | Various test files | 2026-03-26 |
| Document `--run-fast` in developer install docs | `docs/source/installation/install_anuga_developers.rst` | 2026-03-26 |
| Update `CLAUDE.md` with `--run-fast` and slow marker info | `CLAUDE.md` | 2026-03-26 |
| Declare missing runtime deps in `pyproject.toml`; add `[parallel]`, `[data]`, `[dev]` extras; fix classifiers | `pyproject.toml` | 2026-03-26 |
| Add EPSG/CRS support to `Geo_reference` — `epsg` property, `is_located()`, non-UTM support via pyproj, `write/read_NetCDF`, fix pre-existing zone/hemisphere bug in `read_NetCDF` | `anuga/coordinate_transforms/geo_reference.py` | 2026-03-26 |
| 23 new tests for EPSG/CRS behaviour (UTM auto-compute, inference, non-UTM national grids, false e/n, NetCDF round-trip, `is_located`) | `anuga/coordinate_transforms/tests/test_geo_reference.py` | 2026-03-26 |
| New CRS documentation page; `Geo_reference` API reference; cross-references from `domain.rst` and `reference/index.rst` | `docs/source/setup_anuga_script/coordinate_reference.rst`, `docs/source/reference/anuga.Geo_reference.rst` | 2026-03-26 |
| Create `claude/` session-continuity directory (PROGRESS, DECISIONS, CONVENTIONS, KNOWN_ISSUES, SESSION_GUIDE, ROADMAP, README) | `claude/` | 2026-03-26 |
| Incorporate Hydrata REFACTOR_PLAN.md into claude/ docs | `claude/PROGRESS.md`, `DECISIONS.md`, `KNOWN_ISSUES.md` | 2026-03-26 |
| Fix `sww_merge` not propagating `hemisphere` and `epsg` to merged SWW — replace field-by-field attribute copy with `Geo_reference(NetCDFObject=fid)` + `write_NetCDF()` in all three merge functions | `anuga/utilities/sww_merge.py` | 2026-03-28 |
| Fix `sww_merge` not propagating `timezone` to merged SWW — read from first input file, pass to `store_header()` | `anuga/utilities/sww_merge.py` | 2026-03-28 |
| Add `sww2vtu` converter — SWW → VTU + PVD for ParaView, no VTK dependency, binary base64 encoding, derived depth and speed quantities, `--z-scale` and `--absolute-coords` options | `anuga/file_conversion/sww2vtu.py` | 2026-03-28 |
| GPU verbose flag (`int verbose` in C struct, controlled from Cython) — suppresses C printf output during pytest, shown with `-s` | `gpu_domain.h`, `gpu_domain_core.c`, `gpu_boundaries.c`, `sw_domain_gpu_ext.pyx` | 2026-04-01 |
| Fix pyproj DeprecationWarning for 1-element arrays (NumPy ≥ 2.0) — use `.item()` for scalar path in `epsg_to_ll`, `ll_to_epsg`, `tif2point_values` | `redfearn.py`, `tif2point_values.py` | 2026-04-01 |
| Fix ReadTheDocs shallow-clone version showing `0.0.0+unknown` — add `git fetch --unshallow --tags` pre-install step | `.readthedocs.yaml` | 2026-04-02 |
| Vectorise `get_flow_through_cross_section` — NumPy segment scan replaces Python loop; C pre-filter skips non-intersecting triangles | `anuga/shallow_water/shallow_water_domain.py` | 2026-04-03 |
| Add ruff linting config (`[tool.ruff]`, `target-version="py310"`, `line-length=120`, E/F/W/B/I/UP rules) and fix all genuine violations | `pyproject.toml`, various `.py` files | 2026-04-03 |
| L1-L4 logging refactor: `TeeStream` (tees print/sys.stdout to both terminal and file), lazy log file (no file created until `set_logfile()` called), `set_logfile()` public API, `anuga.set_logfile()` export; `prepare_data.py` updated to use `TeeStream`; `anuga_run_toml.py` and `run_model.py` updated | `anuga/utilities/log.py`, `anuga/scenario/prepare_data.py`, `scripts/anuga_run_toml.py`, `examples/cairns_toml_excel/run_model.py`, `anuga/__init__.py` | 2026-04-05 |
| `log.verbose()` shortcut + `verbose_to_screen` flag; `setup_mesh.py` quieted — mesh construction messages go to file only via `file_only()` context manager | `anuga/utilities/log.py`, `anuga/scenario/setup_mesh.py` | 2026-04-05 |
| `log.file_only()` context manager — temporarily routes stdout to file only, suppressing terminal output for verbose third-party calls | `anuga/utilities/log.py` | 2026-04-05 |
| Add logging documentation page with examples for `set_logfile`, `file_only`, log levels | `docs/source/setup_anuga_script/logging.rst`, `docs/source/setup_anuga_script/index.rst` | 2026-04-05 |
| Fix pyproj DeprecationWarning in `tif2point_values.py` for single-point queries — use `.item()` to dispatch scalar path | `anuga/file_conversion/tif2point_values.py` | 2026-04-05 |
| Archive CuPy/CUDA files out of `anuga/shallow_water/` into `archive/cupy_cuda/` | `archive/cupy_cuda/` | 2026-04-05 |
| Fix `test_sww2csv_multiple_files` stale-file pollution — chdir into `tempfile.mkdtemp()` so glob cannot see SWW files from previous interrupted runs; `self.sww = None` before restoring CWD | `anuga/abstract_2d_finite_volumes/tests/test_gauge.py` | 2026-04-05 |
| CI: add `pytest-regressions` to all 13 conda environment YMLs | `environments/environment_*.yml` | 2026-04-05 |
| CI: drop Python 3.8 (numpy>=2.0.0 requires ≥3.9); fix `list \| np.ndarray` PEP-604 annotation for Python 3.9 compat | `.github/workflows/conda-setup.yml`, `pyproject.toml`, `anuga/utilities/animate.py` | 2026-04-05 |
| Fix NPY002 test recalibration — three hardcoded expected values in `test_geospatial_data.py` recomputed for `default_rng()` sequences | `anuga/geospatial_data/tests/test_geospatial_data.py` | 2026-04-05 |
| Propagate v3.3.0, v3.3.1, v3.3.2 tags/releases to GeoscienceAustralia remote | `ga` remote | 2026-04-05 |
| L5: 715 `log.critical()` → `log.info()` across 70+ production files; ~35 genuine warning conditions → `log.warning()`; `log.py` itself unchanged; fix `test_sww2dem_verbose_True` to use `mock.patch` instead of fragile FileHandler approach | 70+ `anuga/**/*.py`, `anuga/file_conversion/tests/test_sww2dem.py` | 2026-04-06 |
| Drop Python 3.9: `X \| Y` union type syntax requires Python>=3.10; update `requires-python`, CI matrix, remove `Union` import from `animate.py` | `pyproject.toml`, `.github/workflows/conda-setup.yml`, `anuga/utilities/animate.py` | 2026-04-06 |

---

## Hydrata Refactor Plan (anuga-community integration)

Source: [Hydrata/anuga_core REFACTOR_PLAN.md](https://github.com/Hydrata/anuga_core/blob/anuga-4.0-refactor-plan/REFACTOR_PLAN.md)
Date: 2026-02-28. Five phases covering test infrastructure, dependencies, linting, deduplication, coverage.

Cross-reference with what we have already completed is noted below.

### Phase 0 — Test Infrastructure ("Refactor Without Fear")

- [x] **0.2 Add test markers** — `@pytest.mark.slow`, `--run-fast` flag, auto-mark parallel tests *(Done 2026-03-26, anuga-community)*
- [x] **0.1 Fix test isolation** — Replaced all `set_datadir('.')` calls and `tempfile.mktemp()` uses with `mkdtemp()`; fixed all tests writing to CWD; all `sww2dem` output paths now use full temp paths; cleaned up orphaned CWD artifacts. *(Done 2026-04-03)*
- [x] **0.3 Golden-master snapshots** — 6 `pytest-regressions` snapshot tests: dam break DE0/DE1, friction, Thacker bowl, extrapolation edge values, timestep sequence. Baselines committed; meson.build registered. *(Done 2026-04-04)*
- [x] **0.4 Coverage baseline** — Configured `.coveragerc` with `branch=true, fail_under=55`; `pytest-cov` in `[dev]` extras. *(Done 2026-04-03)*
- [x] **0.5 CI test matrix** — `conda-setup.yml` updated: PRs run `--run-fast`, pushes to main/develop run full suite; coverage step on Linux+Python 3.12. *(Done 2026-04-03)*

### Phase 1 — Dependency Consolidation

Current state: `pyproject.toml` declares only `numpy>=2.0.0` despite the codebase importing
scipy, netCDF4, matplotlib, meshpy, dill, pymetis, pyproj, affine.

- [x] **1.1 Declare runtime deps** — Added `dill>=0.3.7`, `matplotlib>=3.7`, `netCDF4>=1.6`, `scipy>=1.11`, `meshpy>=2022.1` to core deps. Added `[parallel]`, `[data]`, `[dev]` optional extras. `cartopy` (phantom) excluded. *(Done 2026-03-26)*
- [x] **1.2 Remove dead deps** — GDAL fully removed: Python imports already gone; `gdal_available` → `spatial_available`; all stale "gdal-compatible" docstring references updated; `gdalwarp`/`gdal_rasterize`/`gdal_calc.py` CLI calls in `scenario/raster_outputs.py` replaced with rasterio+fiona+numpy. NPY002: all 17 legacy `np.random.*` calls modernised to `np.random.default_rng()`. *(Done 2026-04-04)*
- [x] **1.3 Delete `setup.py`** — already absent; no action needed. *(Confirmed 2026-03-26)*
- [x] **1.4 Fix classifiers** — removed Python 3.9 classifier (conflicts with `requires-python = ">=3.10"`). *(Done 2026-03-26)*

### Phase 2 — Linting & Code Quality

Current state: no linter, formatter, type checker, or pre-commit hooks. 4,189 functions with zero type annotations.

- [x] **2.1 Add ruff configuration** — `pyproject.toml` `[tool.ruff]` section added; `ruff check --fix` run to fix all genuine violations. *(Done 2026-04-03)*
- [x] **2.2 Pre-commit hooks** — `.pre-commit-config.yaml` with `ruff check --fix` and `ruff-format` (astral-sh/ruff-pre-commit v0.11.2). Format only files being modified — no bulk format. *(2026-04-03)*
- [x] **2.3 CI enforcement** — `.github/workflows/lint.yml` — `ruff check anuga` runs on PRs and pushes to main/develop. *(Done 2026-04-03)*

### Phase 3 — Code Deduplication (~7,700 redundant lines)

- [x] **3.1 Unify quantity kernels** — consolidated to single `quantity_openmp_ext.pyx`; `quantity_ext2.pyx` and separate non-OpenMP variant removed. *(done, commit 5c191dc7)*
- [x] **3.2 Consolidate parallel operator wrappers** — Extracted `_gather_enquiry_stage_and_energy()`, `_broadcast_flow_direction()`, `_gather_inflow_outflow_depths()` helpers into `Parallel_Structure_operator`; rewrote `discharge_routine` in `Parallel_Boyd_box_operator`, `Parallel_Boyd_pipe_operator`, `Parallel_Weir_orifice_trapezoid_operator` (each ~150→50 lines, −125 lines net). `Parallel_Internal_boundary_operator` and `Parallel_Inlet_operator` left unchanged (genuinely different logic). *(2026-04-12)*
- [x] **3.3 Merge duplicate culvert classes** — `Culvert_operator_Parallel` removed; GPU path via `gpu_culvert_manager.py`; merged via PR #118. *(done)*
- [x] **3.4 Clean up `system_tools.py`** — removed `six` dependency (`string_types` → `str` in gauge.py and file_function.py); deleted dead code: `store_svn_revision_info` (SVN legacy), `get_web_file` (six.moves.urllib), `tar_file`, `untar_file`, `get_file_hexdigest`, `make_digest_file`, `MemoryUpdate`; fixed `string_to_char`; trimmed imports. 335 lines removed. Structural split rejected: 62 import sites + wildcard import make it risky. *(Done 2026-04-13, commit f083ad29)*

### Phase 4 — Expanded Test Coverage

Current state: **63.88% coverage** (fail_under=63, full suite passing), 2,231+ tests (fast run), ~67 s fast / ~3 min full. All 37 validation scenarios have automated scripts.

- [ ] **4.1 Modernise test patterns** — Convert key test classes from `unittest.TestCase` to plain pytest functions selectively. Add pytest fixtures for domain creation.
- [x] **4.2 Integrate validation tests** — Added 33 `validate_*.py` scripts covering all remaining scenarios: analytical comparison (transcritical, MacDonald, depth expansion, parabolic/paraboloid basin), run-only short/slow (lake-at-rest, river-at-rest, runup, rundown, trapezoidal channel, deep wave, landslide tsunami), behaviour-only (bridge/weir HEC-RAS, lid-driven cavity), and case studies (merewether, towradgi, patong). Patong skips unless data downloaded; added to `dirs_to_skip` in runner. *(2026-04-10)*
- [x] **4.3 Coverage targets** — Extended `.coveragerc` omit rules to exclude ~3000 lines of dead/untestable code (visualiser, pmesh UI, validation typesetting, benchmark scripts, scenario scaffolding, duplicate `change_friction_operator.py`); deleted `change_friction_operator.py`; added `test_mannings_operator.py` (8 tests) and `test_sww2vtu.py`; registered `sww2vtu.py` in meson.build (was causing `ModuleNotFoundError`); switched CI coverage from `--run-fast` to full suite; removed `continue-on-error: true`; enforced `fail_under=52`. Full-suite baseline ~53%. Lifting to 65% deferred as long-term (needs systematic test writing across untested modules). *(2026-04-10)*
- [x] **4.4 Push coverage to 63%** — Systematic new-test-class pass across previously-untested branches. Added `Test_*_extra` / `TestCase_extra` classes in: `test_polygon.py` (26 tests, geometry 76%→95%), `test_lat_long_UTM_conversion.py` (6 tests), `test_numerical_tools.py` (13 tests), `test_xml_tools.py` (9 tests), `test_alpha_shape.py` (7 tests), `test_sparse.py` (10 tests), `test_geo_reference.py` (12 tests), `test_function_utils.py` (8 tests), `test_cg_solve.py` (3 tests), `test_set_quantity.py` (4 tests). Fixed `log.debug` call-signature bug in `operators/set_quantity.py`. Full suite (with slow tests): **63.88%**. Raised `fail_under` to 63. *(2026-04-13, commits 133b26b1, eab9cea6)*

---

## Quantity Memory Reduction

Full plan: `claude/QUANTITY_MEMORY_PLAN.md`

Target achieved: ~54% memory reduction for typical 10-quantity 1M-triangle domain (800 MB → ~368 MB).
Key insight: DE solver computes gradients on C stack — Python-level gradient arrays are NEVER read by the solver.
This enabled lazy gradients for ALL types, not just elevation.

- [x] **QM1** Introduce `qty_type` concept (`evolved`, `edge_diagnostic`, `centroid_only`, `coordinate`) controlling which arrays are allocated at construction *(2026-04-09)*
- [x] **QM2** Lazy `vertex_values` property on all quantity types — allocate on first access, transparent to callers *(2026-04-09)*
- [x] **QM3** Strip `explicit_update`, `semi_implicit_update`, `centroid_backup_values`, `phi` from `elevation` *(2026-04-09)*
- [x] **QM4** Strip all arrays except `centroid_values` from `friction` *(2026-04-09)*
- [x] **QM5** Reduce `height`, `xvelocity`, `yvelocity` to centroid + edge only (no update arrays) *(2026-04-09)*
- [x] **QM6** Make `x_gradient`, `y_gradient`, `phi` lazy for ALL types including `evolved` — saves 24N per evolved quantity + 16N for elevation. Removed `static_with_gradients` type. Elevation now uses `edge_diagnostic`. Erosion operators trigger lazy allocation transparently. *(2026-04-10)*
- [x] **QM7** Shared gradient workspace on domain — added `_grad_workspace_x/y` + `_phi_workspace` to `Generic_Domain`; modified `extrapolate_second_order_and_limit_by_edge/vertex` in `quantity_openmp_ext.pyx` to accept optional workspace arrays; `Quantity` methods always pass domain workspace so per-quantity `_x_gradient/_y_gradient/_phi` stay `None`. Also fixed pre-existing `quantity.object` AttributeError bug in both functions. 4 new tests. For the SW domain the benefit is preventative (SW C kernel never used Python-level gradients); for advection/generic domains it prevents gradient allocation on evolved quantities. *(2026-04-13, commit 22559a5b)*

---

## Benchmark Suite

- [x] **B1 Single-process benchmark** — `benchmarks/run_benchmarks.py`: dam-break scenarios (small/medium/large), modes 0/1/2, peak RSS + cells/s, JSON output. `benchmarks/compare_benchmarks.py` shows ±% deltas across commits. *(2026-04-07)*
- [x] **B2 MPI distribution benchmark** — `benchmarks/distribute_benchmarks.py`: all four distribution methods (`distribute()`, `collaborative()`, `distribute_basic_mesh()`, `dump+load`) with PSS/RSS memory, ghost-triangle stats, evolve check. `benchmarks/run_benchmark_grid.py` sweeps np × scheme grid. Merged from `scripts/benchmark_distribute.py` + `scripts/benchmark_distribute_mesh.py`. *(2026-04-07)*

---

## Bug Fixes

- [x] **BF1 Basic_mesh.reorder() stale neighbours** — `distribute_basic_mesh()` produced ~59% more ghost triangles than `distribute()` for the same mesh/scheme (measured: 9,002 vs 5,655 on 1M-tri metis/4-rank). Root cause: lazy `_neighbours` not computed before reorder; `_build_neighbours()` then reconstructed from the pre-reorder `_triangle_neighbours` cache. Fix: call `self.neighbours` at start of `reorder()`. Regression test added to `test_neighbour_mesh_reorder.py`. *(2026-04-07)*
- [x] **BF2 GPU test tolerances** — `test_culvert_cpu_gpu_match`, `test_weir_trapezoid_cpu_gpu_match`, `test_weir_trapezoid_nonrect_section` used `atol=1e-12` designed for CPU_ONLY_MODE (bit-for-bit identical kernels). On real GPU hardware, mode=1 (CPU) and mode=2 (GPU) use different FP arithmetic; after ~50 timesteps divergence can reach ~0.01 m. Fixed to `atol=0.02`. *(2026-04-11)*
- [x] **BF3 Mannings operator RuntimeWarning** — `numpy.where` evaluates both branches before masking; `power(height, 7/3)` on negative depths in dry cells raised `invalid value` warning. Fixed with `safe_h = maximum(height, 1e-15)`. *(2026-04-11)*
- [x] **BF4 Rate_operator empty-check for numpy array** — `elif self.indices == []:` broadcasts instead of testing emptiness when `self.indices` is a numpy array → `ValueError` in `_init_gpu`. Fixed to `hasattr(..., '__len__') and len(...) == 0`. *(2026-04-11)*
- [x] **BF5 GPU_AWARE_MPI segfault intra-node** — `omp_target_alloc` device pointers passed directly to `MPI_Isend`; UCX `uct_mm` (shared-memory) transport selected for intra-node MPI cannot access GPU device memory → SIGSEGV. Added `host_send_buffer`/`host_recv_buffer` staging in `gpu_halo.c`; MPI always uses host buffers, `omp_target_memcpy` handles D2H/H2D. *(2026-04-11)*
- [x] **BF6 Rate_operator parallel false CPU-only** — In MPI runs, many rainfall polygon operators have empty local indices (polygon on another rank). `_init_gpu` returned early without setting `_gpu_initialized=True`, so all such operators were counted as CPU-only and triggered `sync_from_device`/`sync_to_device` every timestep (~70 s overhead on towradgi). Fix: mark empty-indices operators as `_gpu_initialized=True` (they're no-ops; `__call__` already returns at line 242). *(2026-04-11)*
- [x] **BF7 Double `get_triangle_containing_point` call in parallel inlet enquiry** — `Parallel_Inlet_enquiry.compute_enquiry_index` discarded the result `k` from the first `get_triangle_containing_point` call, then called it again (O(N) search repeated for every inlet at startup). Fixed to reuse `k` from the first call. *(2026-04-12)*
- [x] **BF8 Threshold-triggered spatial index for `get_triangle_containing_point`** — O(N) brute-force loop replaced with `MeshQuadtree` after 5 calls. Counter and index cached on the `Mesh` object; first 5 calls use brute force (fast to start), 6th call builds the C quad-tree and all subsequent queries use `search_fast()`. Measurable speedup on culvert setup (many inlets × O(N) per inlet → near-constant time). *(2026-04-12)*

---

## Riverwall Throughflow

Full plan: `claude/RIVERWALL_THROUGHFLOW_PLAN.md`

Flow through the wall body (below the crest) driven by stage difference and submerged depth.
Uses submerged orifice formula: `Q = Cd_through * h_eff * sqrt(2g * |Δstage|)`.
Additive to existing Villemonte overtopping flow. Single new parameter, default 0 (impermeable).

- [x] **RW1** Add `Cd_through` to `hydraulic_variable_names` and `default_riverwallPar` in `riverwall.py` *(2026-04-04)*
- [x] **RW2** Add `gpu_adjust_edgeflux_with_throughflow()` to `gpu_device_helpers.h` (inside `#pragma omp declare target`) *(2026-04-04)*
- [x] **RW3** Call new function in `core_kernels.c` after existing weir call (read column 5 with guard for old files) *(2026-04-04)*
- [x] **RW4** No separate CPU path needed — `core_kernels.c` is shared via `sw_domain_openmp.c` include *(2026-04-04)*
- [x] **RW5** Tests: parameter stored, default zero, dry downstream, more Cd→more flow, additive to overtopping, backward compat *(2026-04-04)*
- [x] **RW6** Update docstring and user docs: `create_riverwalls` docstring + Jupyter notebook `Cd_through` demo section *(2026-04-04)*

---

## GPU / OpenMP Offloading (v4.0.0 / SC26)

Full plan: `claude/GPU_DEVELOPMENT_PLAN.md`

### Phase 1 — Correctness and test coverage (weeks 1–4)

- [x] **G1.1 File_boundary GPU support** — `File_boundary` / `Field_boundary` (spatially varying, time-dependent, per-edge values from SWW interpolation). Struct + Python push pattern; `gpu_file_boundary_init/set_values/evaluate`; `init_file_boundary` in setup; `set_file_boundary_values_from_domain` + `evaluate_file_boundary_gpu` called each sub-step in both Python and C RK loops. 3 tests (mode=1 vs mode=2, type recognised, per-edge push). *(2026-04-09)*
- [x] **G1.2 Device memory check** — `gpu_check_device_memory()` before first `omp target enter data`; prints estimated memory, queries CUDA/HIP when available, `map_to_gpu` raises `RuntimeError` on OOM. 5 tests. *(2026-04-09)*
- [x] **G1.3 Slot limit assertions** — Original hard limits (`MAX_RATE_OPERATORS=64`, `MAX_INLET_OPERATORS=32`, `MAX_CULVERTS=64`) replaced by dynamic heap growth in G3.3; tests updated to verify arrays grow beyond initial capacity instead of raising `RuntimeError`. *(2026-04-07; superseded by G3.3 2026-04-10)*
- [x] **G1.4 End-to-end regression test** — 10 s tidal + 10 s dam break; mode=1 vs mode=2; `atol=1e-12`; in CPU_ONLY_MODE differences are machine-epsilon. *(2026-04-07)*
- [x] **G1.4 Multi-rank halo exchange test** — `test_parallel_sw_flow_gpu.py`: 2-rank MPI GPU test; `tri_full_flag` fix (ghost cells excluded from timestep min). *(2026-04-09)*
- [x] **G1.4 Culvert test in GPU mode** — `Test_GPU_Culvert`: mode=1 vs mode=2, volume conservation, flow direction. *(2026-04-07)*
- [x] **G1.5 SSP-RK3 GPU support** — `gpu_evolve_one_rk3_step` (3-stage Shu-Osher C loop); `gpu_saxpy3_conserved_quantities`; Python-orchestrated `_evolve_one_rk3_step_gpu` + C loop `_evolve_one_rk3_step_c`; `evolve_one_rk3_step` dispatches to GPU; `use_c_rk2_loop` → `use_c_rk_loop` (deprecated property kept). 3 tests. *(2026-04-09)*

### Phase 2 — Performance validation (weeks 5–10)

- [x] **G2.1 Benchmark suite** — `benchmarks/run_gpu_benchmarks.py` with 100 K / 2 M / 20 M triangle cases; prints GFLOP/s via existing `gpu_flop.c` / `flop_counters_*` infrastructure; speedup summary table; JSON output compatible with `compare_benchmarks.py`; runs in CPU_ONLY_MODE for CI. (2026-04-10)
- [x] **G2.2 GPU-aware MPI validation** — `detect_gpu_aware_mpi()` now has full runtime detection via `MPIX_Query_cuda_support()` / `MPIX_Query_rocm_support()` (Open MPI / MVAPICH2 GPU builds); meson probe for `mpi-ext.h` and symbols sets `-DHAVE_MPIX_CUDA_SUPPORT` / `-DHAVE_MPIX_ROCM_SUPPORT`; `gpu_is_available()` C function + `gpu_available()` and `is_gpu_aware_mpi()` Python bindings added to `sw_domain_gpu_ext`; meson already had `gpu_aware_mpi` option and `GPU_AWARE_MPI` compile flag. (2026-04-10)
- [x] **G2.3 NVTX/OMPT profiling hooks** — `gpu_nvtx.h` with `NVTX_PUSH(name)`/`NVTX_POP()` macros (no-ops without `-DNVTX_ENABLED`); instrumented 10 kernel functions: `gpu_extrapolate_second_order`, `gpu_compute_fluxes`, `gpu_update_conserved_quantities`, `gpu_backup_conserved_quantities`, `gpu_saxpy_conserved_quantities`, `gpu_saxpy3_conserved_quantities`, `gpu_protect`, `gpu_manning_friction`, `gpu_exchange_ghosts`, `gpu_culverts_apply_all`, plus `gpu_evolve_one_rk2/rk3_step` outer markers; meson `use_nvtx` option probes NVTX v3 (header-only) then v1 (CUDA toolkit); `meson_options.txt` updated. (2026-04-10)
- [x] **G2.4 Weak scaling experiment** — `benchmarks/run_weak_scaling.py` (MPI Python driver: mesh scales as `√N` per axis so elements/rank stays constant, timing by MPI barriers, JSON output, `--analyse` mode prints efficiency table); `scripts/hpc/weak_scaling.slurm` (SLURM job template, one GPU per task); `scripts/hpc/submit_weak_scaling.sh` (sweep 1→64 ranks, dry-run mode, optional analysis job on completion). Code complete; needs real GPU cluster to produce SC26 efficiency numbers. (2026-04-10)

### Phase 3 — Feature parity (weeks 11–20)

- [x] **G3.1 Gate/weir operators on GPU** — `Weir_orifice_trapezoid_operator` added to GPUCulvertManager; `CULVERT_TYPE_WEIR_TRAPEZOID=2`; `weir_orifice_trapezoid_discharge()` C function; z1/z2 side slopes in `culvert_params`; 3 tests (cpu/gpu match, volume conservation, non-rect section). *(2026-04-10)*
- [x] **G3.2 Riverwall GPU support** — physics in `core_kernels.c`, arrays mapped in `gpu_domain_core.c`, Cython pointers wired. All 4 `Test_GPU_Riverwall` tests pass (init, cpu/gpu match, flux kernel, weir discharge). Completed as part of G1/RW work. *(2026-04-10)*
- [x] **G3.3 Dynamic operator slot limits** — `rate_operators.ops`, `inlet_operators.ops`, and `culvert_operators.{params,indices,state}` changed from static arrays to heap-allocated pointers with `capacity` field; `grow_rate_ops()`/`grow_inlet_ops()` helpers double capacity on overflow; `enquiry_ids[MAX_CULVERTS]` stack array in `gpu_culvert_gather_enquiry` replaced with `malloc`; `gpu_domain_init` zero-inits all new pointer fields; `gpu_domain_finalize` calls rate/inlet finalize_all; slot-overflow tests updated to test dynamic growth. *(2026-04-10)*
- [x] **G3.4 GPU documentation** — `docs/source/parallel/use_gpu_offloading.rst`: architecture overview, slot limits, operator support table, HPC SLURM example, NVTX profiling guide, FAQ. *(2026-04-10)*

### Phase 4 — SC26 paper preparation (months 4–6)

- [ ] **G4.1 Gordon Bell metrics** — per-kernel timing (not just totals), roofline model comparison, peak theoretical FLOP/s.
- [ ] **G4.2 Physical benchmark validation** — Thacker paraboloid, dam break (Ritter), tide gauge comparison in GPU mode (use existing `validation_tests/` scripts).
- [ ] **G4.3 Multi-node strong scaling** — 20 M triangles, 1→64 GPUs; demonstrate ~50× runtime reduction.

---

## Remaining Work (priority order)

### Short term — SC26 prerequisites (needs GPU hardware)
1. **G2.1** GPU benchmark suite (actual GPU runs — `benchmarks/run_gpu_benchmarks.py` is ready)
3. **G2.4** Weak scaling experiment (1→64 GPUs on HPC cluster)
4. **G4.1** Gordon Bell metrics — per-kernel timing, roofline model comparison
5. **G4.2** Physical benchmark validation — Thacker paraboloid, dam break (Ritter) in GPU mode
6. **G4.3** Multi-node strong scaling — 20 M triangles, 1→64 GPUs

### Medium effort (1–3 days each)
*(None remaining)*

### Long-term / opportunistic
- **H4.1** Modernise test patterns — convert `unittest.TestCase` to plain pytest functions and shared fixtures incrementally, when files are touched for other reasons. Not worth a dedicated pass.
- **Coverage to 65%** — Currently enforced at 52% (`fail_under=52`, full suite). Reaching 65% needs ~3000 additional covered lines; requires systematic new tests for `file_conversion/`, `structures/`, `shallow_water/boundaries.py`, and scenario module. Not worth a dedicated pass; add tests opportunistically.
