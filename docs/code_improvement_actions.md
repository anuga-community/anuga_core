# Code Improvement Actions

Generated from a systematic review of `anuga_core`.
Date: 2026-03-23

Work through each section in order.  Tick the checkbox when done and add a
*(Done YYYY-MM-DD)* note so progress is visible across sessions.

---

## Priority 1 — Quick wins (bug risk, no behaviour change)

### 1.1  Fix mutable default arguments (~43 functions)

`def f(x=[])` and `def f(x={})` share state between calls — a subtle but
real bug source.  Replace with `None` and create the object inside the body.

Key files (check whole repo with `grep -rn "def .*=\[\]" anuga/`):

- [x] `anuga/caching/caching.py:145` — `cache(..., kwargs={})` *(Done 2026-03-24)*
- [x] `anuga/file/sww.py:535` — `Write_sww.__init__(..., static_c_quantities=[])` *(Done 2026-03-24)*
- [x] `anuga/parallel/parallel_boyd_box_operator.py:22` — multiple list defaults *(Done 2026-03-24)*
- [x] `anuga/abstract_2d_finite_volumes/ermapper_grids.py:8,88,203` — dict/list defaults *(Done 2026-03-24)*
- [x] Audit full repo and fix remaining instances — also fixed parallel_structure_operator, parallel_boyd_pipe_operator, parallel_weir_orifice_trapezoid_operator, parallel_internal_boundary_operator, parallel_operator_factory, riverwall, util.py *(Done 2026-03-24)*

### 1.2  Replace bare `except:` with specific exception types

Bare `except:` catches `KeyboardInterrupt` and `SystemExit`, masking
serious failures.

- [x] `anuga/utilities/system_tools.py` — no bare except found, already using typed handlers *(Done 2026-03-24)*
- [x] `anuga/shallow_water/boundaries.py` — no bare except found *(Done 2026-03-24)*
- [x] `anuga/caching/caching.py` — no bare except found *(Done 2026-03-24)*
- [x] `anuga/abstract_2d_finite_volumes/tests/test_quantity.py` — no bare except found *(Done 2026-03-24)*
- [x] `anuga/abstract_2d_finite_volumes/tests/test_generic_domain.py` — no bare except found *(Done 2026-03-24)*
- [x] `anuga/file_conversion/dem2pts.py` — no bare except found *(Done 2026-03-24)*

### 1.3  Convert file operations to use `with` statements

Prevents file-handle leaks when exceptions occur.

- [x] `anuga/file/csv_file.py:47,196,206,216,224` *(Done 2026-03-24)*
- [x] `anuga/file/ungenerate.py:16` *(Done 2026-03-24)*
- [ ] `anuga/file/urs.py:29` — skipped: file handle stored as `self.mux_file` for iterator lifecycle, `with` not applicable
- [x] `anuga/utilities/system_tools.py:29` *(Done 2026-03-24)*
- [ ] Audit `anuga/file/` and `anuga/utilities/` for remaining bare `open()` calls

### 1.4  Fix invalid escape sequences in docstrings

These generate `DeprecationWarning` in Python 3.12 and will become
`SyntaxError` in a future version.  Prefix affected docstrings with `r`.

- [x] `anuga/utilities/norms.py:15` *(Done 2026-03-24)*
- [x] `anuga/utilities/system_tools.py:133` — no invalid escape found at that line *(Done 2026-03-24)*
- [x] Run `python -W error::DeprecationWarning -c "import anuga"` — no warnings remain *(Done 2026-03-24)*

### 1.5  Delete large commented-out dead code

No behaviour change — purely cleanup.

- [x] `anuga/file_conversion/dem2pts.py:164–281` — 118-line pre-vectorisation loop deleted *(Done 2026-03-24)*
- [x] `anuga/abstract_2d_finite_volumes/neighbour_mesh.py:615–668` — 53-line disabled validation block deleted *(Done 2026-03-24)*
- [ ] Grep for `#.*for i in range` and similar large legacy comment blocks
      across `shallow_water/` and `operators/`

---

## Priority 2 — Correctness and stability

### 2.1  Fix silent error suppression in quantity update

`anuga/operators/set_quantity.py:116–121` catches `ValueError` and does
nothing — if the update fails every timestep the simulation silently
produces wrong results.

- [x] Replace bare `except ValueError: pass` with at minimum a logged warning *(Done 2026-03-24)*
- [x] Investigate whether the `ValueError` case is actually expected; if so,
      document why with a comment — yes, expected for time-series outside range; documented *(Done 2026-03-24)*

### 2.2  Log xarray import failures properly

`anuga/operators/rate_operators.py:112–116` — `ImportError` from xarray is
silently swallowed.  A broken partial install is indistinguishable from
xarray not being present.

- [x] Add `import logging; log = logging.getLogger(__name__)` at top of file
      if not already present — `log` was already imported via `anuga.utilities.log` *(Done 2026-03-24)*
- [x] Replace silent `except ImportError: pass` with
      `except ImportError: log.debug("xarray not available: %s", e)` *(Done 2026-03-24)*

### 2.3  Address FIXME items left in production code

- [x] `anuga/structures/boyd_box_operator.py:119` — `max_velocity = 10.0` made a keyword argument (also in parallel version) *(Done 2026-03-24)*
- [x] `anuga/fit_interpolate/fit.py:119–121` — early-exit not possible (C solver requires D matrix); replaced misleading `if True:` with explanatory comment *(Done 2026-03-24)*
- [x] `anuga/geometry/polygon.py:133` — removed FIXME; documented why Python version is acceptable *(Done 2026-03-24)*
- [x] `anuga/operators/rate_operators.py` — documented factor file/array limitation clearly *(Done 2026-03-24)*

---

## Priority 3 — Test coverage

### 3.1  Add tests for untested operator classes

None of the following have any test coverage:

- [x] `Bed_shear_erosion_operator` — `anuga/operators/erosion_operators.py` *(Done 2026-03-24)*
- [x] `Circular_erosion_operator` — `anuga/operators/erosion_operators.py` *(Done 2026-03-24)*
- [x] `Flat_slice_erosion_operator` — `anuga/operators/erosion_operators.py` *(Done 2026-03-24)*
- [x] `Flat_fill_slice_erosion_operator` — `anuga/operators/erosion_operators.py` *(Done 2026-03-24)*
- [x] `Collect_max_quantities_operator` — `test_collect_operators.py` (new) *(Done 2026-03-24)*
- [x] `Collect_max_stage_operator` — `test_collect_operators.py` (new) *(Done 2026-03-24)*
- [x] `Elliptic_operator` — `test_elliptic_operator.py` (new) *(Done 2026-03-24)*
- [x] `Circular_rate_operator` — added to `test_erosion_operators.py` *(Done 2026-03-24)*
- [x] `Circular_set_quantity_operator` — added to `test_erosion_operators.py` *(Done 2026-03-24)*
- [x] `Circular_set_stage_operator` — added to `test_erosion_operators.py` *(Done 2026-03-24)*

Target test files: `anuga/operators/tests/test_erosion_operators.py`,
`test_collect_operators.py`, `test_elliptic_operator.py`.

### 3.2  Add tests for untested structure classes

- [x] `Structure_operator` (base class) — `test_structure_operator.py` (new): construction, dimensions, repr, statistics, discharge_routine raises *(Done 2026-03-24)*
- [x] `Internal_boundary_operator` — `test_internal_boundary_operator.py` (new): construction, type, width, call, statistics *(Done 2026-03-24)*
- [ ] `RiverWall` — `anuga/structures/riverwall.py` — deferred; requires full mesh with breaklines
- [x] `Inlet_enquiry` — covered in `test_structure_operator.py` *(Done 2026-03-24)*

Target test file: `anuga/structures/tests/test_internal_boundary_operator.py`,
`anuga/structures/tests/test_riverwall.py`.

### 3.3  Add tests for untested scenario module

- [ ] Scenario module (`anuga/scenario/`) — deferred. `prepare_data.py`,
      `setup_boundary_conditions.py`, `setup_rainfall.py`, `setup_inlets.py`
      all depend on `spatialInputUtil` (compiled C extension) and real
      shapefile/Excel data; meaningful unit tests require significant mocking
      infrastructure or test data that is not in the repo.

---

## Priority 4 — API and code quality

### 4.1  Reduce parameter counts on worst-offending functions

Functions with 15+ parameters are unusable without constantly consulting the
docs.  Introduce config dataclasses or use `**kwargs` consolidation.

- [ ] `anuga/abstract_2d_finite_volumes/gauge.py:616` — `_generate_figures()`
      19 parameters → group plot-style options into a `PlotConfig` dataclass
- [ ] `anuga/abstract_2d_finite_volumes/gauge.py:263` — `sww2timeseries()`
      16 parameters → same approach
- [ ] `anuga/abstract_2d_finite_volumes/generic_domain.py:63` —
      `Generic_Domain.__init__()` 21 parameters → split mesh/numerics/output
      options into separate config objects or keyword-only arguments
- [ ] `anuga/structures/boyd_box_operator.py:25` — `Boyd_box_operator.__init__()`
      21 parameters → group hydraulic and geometry parameters

### 4.2  Standardise naming conventions in `pmesh/`

`anuga/pmesh/mesh.py` has 49 camelCase methods alongside 31 snake_case ones.
The module is not in the primary public API but is used internally.

- [x] Create snake_case aliases for all camelCase public methods — 39 methods renamed; camelCase kept as deprecated wrappers *(Done 2026-03-24)*
- [x] Add deprecation warnings to the camelCase names *(Done 2026-03-24)*
- [x] Update all internal callers to use the snake_case names — mesh.py self-calls, visualmesh.py, test_mesh.py, test_ungenerate.py, benchmark_least_squares.py, timing.py *(Done 2026-03-24)*
- [ ] `anuga/pmesh/visualmesh.py` — camelCase methods inside vAbstract/vVertex/vRegion classes not yet treated (GUI-only code, deferred)

### 4.3  Deprecate legacy camelCase methods in shallow_water_domain

`anuga/shallow_water/shallow_water_domain.py` has 7 legacy camelCase methods
that are part of the public-facing API.

- [x] Identify the methods — located in `generic_domain.py` (not shallow_water_domain.py): `get_CFL` and `set_CFL` *(Done 2026-03-24)*
- [x] Add `warnings.warn(..., DeprecationWarning, stacklevel=2)` wrappers to `get_CFL` and `set_CFL` *(Done 2026-03-24)*
- [x] `set_cfl` promoted from alias to real method; `set_CFL` is now a deprecated wrapper *(Done 2026-03-24)*

### 4.4  Add `__all__` to major public modules

Only 2 files in the codebase currently define `__all__`.  Adding it to the
main public modules makes the API contract explicit and improves IDE support.

- [x] `anuga/__init__.py` — added 154-name `__all__` covering all public exports *(Done 2026-03-24)*
- [x] `anuga/shallow_water/__init__.py` — added `__all__ = ['test']` *(Done 2026-03-24)*
- [x] `anuga/operators/__init__.py` — added `__all__ = ['test']` *(Done 2026-03-24)*
- [x] `anuga/structures/__init__.py` — added `__all__ = ['test']` *(Done 2026-03-24)*
- [x] `anuga/file/__init__.py` — added `__all__ = ['test']` *(Done 2026-03-24)*

---

## Priority 5 — Performance

### 5.1  Vectorise Python loops over numpy arrays

- [x] `anuga/fit_interpolate/fit.py:598` — replaced per-row list extend loop
      with `num.hstack([num.array(old), f]).tolist()` *(Done 2026-03-24)*

- [x] `anuga/file/csv_file.py:136` — replaced inner float-conversion loop with
      `ret[:, index] = num.array(X[col_title], dtype=float)` *(Done 2026-03-24)*

- [x] `anuga/abstract_2d_finite_volumes/util.py:786` — `get_runup_data_for_locations_from_file`:
      file opened/closed on every iteration replaced with a single `with open(...)`
      wrapping the loop; also fixed `== None` → `is None` *(Done 2026-03-24)*

- [ ] `anuga/abstract_2d_finite_volumes/util.py:301` (`csv2timeseries_graphs`) and
      `util.py:279` (`make_plots_from_csv_file`) — inner loops are dominated by
      matplotlib and file I/O; list comprehensions for string→float conversion
      are negligible. Defer unless profiling shows these in the top-N.

### 5.2  Consider Cython implementation of `polygon.intersection()`

`anuga/geometry/polygon.py:133` — profiling has not confirmed `intersection()` as
a bottleneck in practice; the hot path for inside-polygon tests already uses
the C extension `polygon_ext.pyx`. Explanatory comment added in Priority 2 work.

- [x] Assessed — not a confirmed hotspot; deferred unless mesh generation profiling
      shows this function in the top-N *(Done 2026-03-24)*

---

## Priority 6 — Documentation improvements

### 6.1  Expand incomplete docstrings in structures

- [ ] `anuga/structures/boyd_box_operator.py:50–77` — document all 21
      constructor parameters with types, units, and valid ranges
- [ ] `anuga/structures/boyd_pipe_operator.py` — same treatment
- [ ] `anuga/structures/weir_orifice_trapezoid_operator.py` — same treatment

### 6.2  Add "Returns" sections to operator docstrings

Many operator `__call__` and `update` methods have no documented return value.

- [ ] `anuga/operators/rate_operators.py` — add Returns section
- [ ] `anuga/operators/erosion_operators.py` — add Returns section

---

## Tracking

| Priority | Total actions | Done |
|----------|--------------|------|
| 1 — Quick wins | 14 | 12 |
| 2 — Correctness | 6 | 6 |
| 3 — Test coverage | 17 | 15 |
| 4 — API quality | 13 | 11 |
| 5 — Performance | 5 | 4 |
| 6 — Documentation | 5 | 0 |
| **Total** | **60** | **44** |
