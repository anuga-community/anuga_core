# ANUGA Code & Documentation Improvement Progress

Last updated: 2026-03-28 (session 5)
Branch: `develop_excel` (most recent work)

---

## Overview

| Area | Total actions | Done | Remaining |
|------|--------------|------|-----------|
| Code improvements (original list) | 60 | 49 | 11 |
| Documentation improvements | 20 | 20 | 0 |
| Additional enhancements | 19 | 19 | 0 |
| Hydrata Phase 0 — Test infrastructure | 5 | 1 | 4 |
| Hydrata Phase 1 — Dependencies | 4 | 2 | 2 |
| Hydrata Phase 2 — Linting | 3 | 0 | 3 |
| Hydrata Phase 3 — Deduplication | 4 | 0 | 4 |
| Hydrata Phase 4 — Coverage | 3 | 0 | 3 |
| **Total** | **118** | **91** | **27** |

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
- [ ] Audit `anuga/file/` and `anuga/utilities/` for remaining bare `open()` calls

#### 1.4 Fix invalid escape sequences in docstrings

- [x] `anuga/utilities/norms.py:15` *(2026-03-24)*
- [x] `anuga/utilities/system_tools.py:133` — no issue found *(2026-03-24)*
- [x] `python -W error::DeprecationWarning -c "import anuga"` — clean *(2026-03-24)*

#### 1.5 Delete large commented-out dead code

- [x] `anuga/file_conversion/dem2pts.py:164–281` — 118-line pre-vectorisation loop deleted *(2026-03-24)*
- [x] `anuga/abstract_2d_finite_volumes/neighbour_mesh.py:615–668` — 53-line disabled block deleted *(2026-03-24)*
- [ ] Grep for `#.*for i in range` and similar large legacy comment blocks in `shallow_water/` and `operators/`

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
- [ ] `RiverWall` — deferred; requires full mesh with breaklines

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

---

## Hydrata Refactor Plan (anuga-community integration)

Source: [Hydrata/anuga_core REFACTOR_PLAN.md](https://github.com/Hydrata/anuga_core/blob/anuga-4.0-refactor-plan/REFACTOR_PLAN.md)
Date: 2026-02-28. Five phases covering test infrastructure, dependencies, linting, deduplication, coverage.

Cross-reference with what we have already completed is noted below.

### Phase 0 — Test Infrastructure ("Refactor Without Fear")

- [x] **0.2 Add test markers** — `@pytest.mark.slow`, `--run-fast` flag, auto-mark parallel tests *(Done 2026-03-26, anuga-community)*
- [ ] **0.1 Fix test isolation** — Replace 47 `set_datadir('.')` calls and 198 `tempfile.mktemp()` uses with `tmp_path` fixtures or `mkdtemp()`. Fix 7+ tests that write `domain.sww` to CWD.
- [ ] **0.3 Golden-master snapshots** — Install `pytest-regressions`; create 8–12 numerical snapshots for `evolve`, `distribute`, `extrapolate`, `compute_fluxes`.
- [ ] **0.4 Coverage baseline** — Configure `.coveragerc` with `branch=true, fail_under=55`; install `diff-cover` for PR enforcement at 80% on changed lines.
- [ ] **0.5 CI test matrix** — GitHub Actions workflow for Python 3.10/3.12/3.13 with fast/slow test separation.

### Phase 1 — Dependency Consolidation

Current state: `pyproject.toml` declares only `numpy>=2.0.0` despite the codebase importing
scipy, netCDF4, matplotlib, meshpy, dill, pymetis, pyproj, affine.

- [x] **1.1 Declare runtime deps** — Added `dill>=0.3.7`, `matplotlib>=3.7`, `netCDF4>=1.6`, `scipy>=1.11`, `meshpy>=2022.1` to core deps. Added `[parallel]`, `[data]`, `[dev]` optional extras. `cartopy` (phantom) excluded. *(Done 2026-03-26)*
- [ ] **1.2 Remove dead deps** — `cartopy` not in pyproject.toml (was never there); `openpyxl` moved to `[data]` optional. Complete GDAL removal (partially done on `remove-gdal` branch).
- [ ] **1.3 Delete `setup.py`** — already absent (`NOT FOUND`); no action needed.
- [x] **1.4 Fix classifiers** — removed Python 3.9 classifier (conflicts with `requires-python = ">=3.10"`). *(Done 2026-03-26)*

### Phase 2 — Linting & Code Quality

Current state: no linter, formatter, type checker, or pre-commit hooks. 4,189 functions with zero type annotations.

- [ ] **2.1 Add ruff configuration** — `pyproject.toml` `[tool.ruff]` section: `target-version="py310"`, `line-length=120`, select E, F, W, B, I, UP rules. Run `ruff check --fix` once for auto-fixable issues.
- [ ] **2.2 Pre-commit hooks** — `.pre-commit-config.yaml` with `ruff check --fix` and `ruff-format`. **Never bulk-format** — format only files being modified to avoid massive merge conflicts.
- [ ] **2.3 CI enforcement** — `ruff check` in GitHub Actions on PRs.

### Phase 3 — Code Deduplication (~7,700 redundant lines)

- [ ] **3.1 Unify quantity kernels** — `quantity_ext.pyx`, `quantity_ext_openmp.pyx`, `quantity_ext2.pyx` share ~90% code. Create single source with compile-time OpenMP toggle.
- [ ] **3.2 Consolidate parallel operator wrappers** — 5 wrapper files thin-wrap `structures/` counterparts. Move MPI awareness into base classes with `self.parallel` flag.
- [ ] **3.3 Merge duplicate culvert classes** — `Culvert_operator` vs `Culvert_operator_Parallel` have near-identical logic. Extract shared base class.
- [ ] **3.4 Split `system_tools.py`** — 750-line file; split into `file_utils.py`, `env_utils.py`, `version_utils.py`. Deduplicate `numerical_tools.py` vs scipy wrappers.

### Phase 4 — Expanded Test Coverage

Current state: ~55% coverage, 1,319 tests (all `unittest.TestCase`), ~38 min wall time. Only 5 of ~37 validation scenarios have automated scripts.

- [ ] **4.1 Modernise test patterns** — Convert key test classes from `unittest.TestCase` to plain pytest functions selectively. Add pytest fixtures for domain creation.
- [ ] **4.2 Integrate validation tests** — Automate the remaining 32 validation scenarios (currently only 5 of 37 have scripts in `validation_tests/`).
- [ ] **4.3 Coverage targets** — Lift from ~55% to 65%; enforce `fail_under=65` in CI.

---

## Remaining Work (priority order)

### Quick wins (< 1 day each)
1. **1.3** Audit `anuga/file/` for remaining bare `open()` calls
2. **1.5** Grep for large legacy comment blocks in `shallow_water/` and `operators/`
3. **H1.2** Complete GDAL removal (continue `remove-gdal` branch work)
4. **H2.1** Add ruff configuration to `pyproject.toml`

### Medium effort (1–3 days each)
6. **H0.1** Fix test isolation — `set_datadir('.')` and `tempfile.mktemp()` sweep
7. **H0.4** Configure coverage baseline (`.coveragerc`, `diff-cover`)
8. **H0.5** GitHub Actions CI matrix
9. **3.2** `RiverWall` tests — requires mesh with breaklines
10. **H2.2** Pre-commit hooks

### Large effort (1+ weeks each)
11. **4.1** Reduce parameter counts via dataclasses — `gauge.py`, `generic_domain.py`
12. **H3.1** Unify quantity kernels (Cython refactor — high risk)
13. **H3.2** Consolidate parallel operator wrappers
14. **H4.2** Automate 32 remaining validation scenarios
15. **3.3** `anuga/scenario/` tests
