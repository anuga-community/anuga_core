# Key Technical Decisions

Decisions from two sources: **anuga-community** sessions (2026-03-23 to 2026-03-26)
and the **Hydrata** refactor plan (2026-02-28, [REFACTOR_PLAN.md](https://github.com/Hydrata/anuga_core/blob/anuga-4.0-refactor-plan/REFACTOR_PLAN.md)).

Decisions made during Claude sessions that are not obvious from reading the code.
Each entry includes what was decided, why, and when.

---

## EPSG / CRS in Geo_reference (2026-03-26)

### `epsg` is a property, auto-computed for WGS84 UTM

**Decision:** `epsg` is a computed property — for WGS84 UTM it returns `32600+zone`
(north) or `32700+zone` (south) without storing anything. Explicit storage (`_epsg`)
only kicks in when the user supplies a non-auto-computable EPSG (e.g. EPSG:28992).

**Why:** Avoids redundant storage for the overwhelmingly common UTM case, while still
allowing any EPSG to be stored explicitly.

### Non-UTM EPSG: zone stays DEFAULT_ZONE (-1), use `is_located()` to distinguish from wavetank

**Decision:** For national grid EPSG codes (28992, 27700, etc.), `zone` remains -1.
The new `is_located()` method returns `True` when `zone != -1 OR _epsg is not None`,
cleanly separating "real CRS, no UTM zone" from "wavetank / no CRS".

**Why:** `zone = -1` previously meant both "no UTM zone" and "wavetank" — adding a
non-UTM EPSG conflated a located simulation with an unlocated one.

### Use pyproj to populate datum, projection, false_easting, false_northing for non-UTM EPSG

**Decision:** `_populate_from_pyproj()` is called for non-UTM EPSG codes. Uses
`to_cf()` first (works for BNG but not RD New), falls back to `to_dict()` (works for
RD New). The PROJ-string conversion warning from pyproj is suppressed via
`warnings.catch_warnings()`.

**Why:** Without pyproj, `datum='wgs84'` and `projection='UTM'` would be written to
SWW files for RD New / BNG — plainly wrong metadata.
`pyproj` is in the `[data]` optional extras, so it may not always be present; the
code degrades gracefully to leaving the defaults.

### Fixed pre-existing bug in `read_NetCDF`: `self.zone == 'southern'`

**Decision:** Lines 234 and 250 of the original code compared `self.zone` (an `int`)
to the strings `'southern'` and `'northern'` — always `False`. Corrected to
`self.hemisphere == 'southern'` / `'northern'`.

**Why:** The warnings about non-standard false easting/northing were silently never
firing. Fixed as part of the EPSG work.

### `pyproject.toml` optional extras structure (2026-03-26)

**Decision:**
- Core deps: `numpy`, `dill`, `matplotlib`, `netCDF4`, `scipy`, `meshpy`
- `[parallel]`: `mpi4py`, `pymetis`
- `[data]`: `pyproj`, `affine`, `xarray`, `pandas`, `openpyxl`, `tomli` (Py<3.11)
- `[dev]`: `pytest`, `psutil`
- `cartopy` excluded entirely — never actually imported anywhere in the codebase.

**Why:** `pyproject.toml` previously only declared `numpy`, meaning `pip install anuga`
produced a broken package on a clean venv. Confirmed hard deps by tracing top-level
imports triggered by `import anuga`.

---

## Testing

### `--run-fast` skips slow, not the other way around (2026-03-26)

**Decision:** Default `pytest` runs ALL tests (including parallel MPI tests). `--run-fast`
skips tests marked `@pytest.mark.slow` and tests under `anuga/parallel/tests/`.

**Why:** User wanted parallel tests included in the standard CI run. The flag was originally
`--run-slow` (to opt-in to slow tests) but then inverted so the common default is the complete
run and `--run-fast` is the quick-feedback mode.

**Result:** Default ≈163s (~1600 tests); `--run-fast` ≈41s (~1500 tests, 101 skipped).

---

### `pytest.mark.slow` criteria (2026-03-26)

Tests are marked slow if they:
- Live under `anuga/parallel/tests/` — auto-marked by `conftest.py` path check
- Take more than ≈5 seconds individually — manually decorated with `@pytest.mark.slow`

Current slow tests (manually marked):
- `anuga/operators/tests/test_kinematic_viscosity_operator.py` — all 4 tests (module-level mark)
- `anuga/tsunami_source/tests/test_tsunami_okada.py` — all tests (module-level mark)
- `test_culvert_class.py`: `test_that_culvert_runs_rating`, `test_momentum_jet`
- `test_new_culvert_class.py`: `test_that_culvert_runs_rating`, `test_that_culvert_flows_conserves_volume`, `test_momentum_jet`
- `test_shallow_water_domain.py`: `test_inflow_using_flowline` (13s)
- `test_model_tools.py`: `test_create_culvert_bridge_operator`
- `test_csv2sts.py`: `test_run_via_commandline`

---

## Docstring Style

### Use NumPy docstring style, not Sphinx `:param:` style (2026-03-26)

**Decision:** All new docstrings in ANUGA use NumPy style.

**Example:**
```python
Parameters
----------
domain : Domain
    The ANUGA domain to operate on.
rate : float or callable
    Flow rate in m³/s. Negative values extract water.
```

**Not:**
```python
:param domain: The ANUGA domain to operate on.
:type domain: Domain
```

**Why:** User preference; NumPy style is more readable for scientific code and consistent
with NumPy/SciPy ecosystem conventions.

---

## Verbose/Quiet Output

### Triangle library quiet flag (2026-03-26)

**Decision:** `_generateMesh_impl` in `anuga/pmesh/mesh.py` now honours the `verbose`
parameter to choose between 'Q' (quiet) and 'V' (verbose) triangle flags when neither is
already in `self.mode`.

**Why:** `str.find()` returns 0 for a first-position match, making `not self.mode.find('Q')`
True when 'Q' is at position 0. The old conditional `if 'Q' not in self.mode` → add 'V'
was running even with `verbose=False`, causing triangle library output to appear in `pytest -s`.

**Fix location:** `anuga/pmesh/mesh.py` in `_generateMesh_impl`.

---

### Suppress logging in tests that intentionally call verbose=True (2026-03-26)

**Decision:** `test_verbose_does_not_raise` in `test_pmesh_to_mesh.py` wraps the call with
`logging.disable(logging.CRITICAL)` / `logging.disable(logging.NOTSET)` to prevent expected
verbose output from polluting test output.

**Why:** The test's purpose is to confirm no exception is raised — not to check the log output.

---

## Memory Reporting

### Memory display format (2026-03-26)

**Decision:** Memory stats show as `mem=NNNmb` below 1024 MB, and `mem=N.NNgb` above.
- Below 1024 MB: `f'mem={rss_mb:.0f}MB'`
- At or above 1024 MB: `f'mem={rss_mb / 1024:.2f}GB'`

Uses `psutil` if available, falls back to `_VmB('VmRSS:')` on Linux.
Added to `timestepping_statistics()` output in `generic_domain.py`.

---

## API Design

### `basic_mesh_from_mesh_file()` as a function, not a class method (2026-03-26)

**Decision:** Implemented as a standalone function in `basic_mesh.py`, not as `Basic_mesh.from_file()`.

**Why:** User explicitly requested a function, not a class method. Functions are easier to
import and use without knowing the class.

**Returns:** A `Basic_mesh` instance with extra instance attributes:
- `vertex_attributes` — ndarray or None
- `vertex_attribute_titles` — list of strings
- `triangle_tags` — list

---

### `is None` vs `== None` (2026-03-26)

**Decision:** Always use `is None` / `is not None` in this codebase.

**Why:** `== None` can raise `ValueError` for numpy arrays (ambiguous truth value).
`is None` tests identity and always works correctly.

---

## Deprecation Strategy

### camelCase → snake_case in `pmesh/mesh.py` (2026-03-24)

**Decision:** 39 public methods renamed to snake_case. camelCase names kept as wrappers
that emit `DeprecationWarning`. All internal callers updated to snake_case immediately.

**Why:** The module had 49 camelCase alongside 31 snake_case methods — inconsistent and
confusing. Keeping camelCase as deprecated wrappers avoids breaking external user code.

---

### `get_CFL` / `set_CFL` in `generic_domain.py` (2026-03-24)

**Decision:** `set_cfl` promoted to the real method. `set_CFL` / `get_CFL` are deprecated
wrappers with `DeprecationWarning`.

---

## Hydrata Refactor Plan Decisions

From [REFACTOR_PLAN.md](https://github.com/Hydrata/anuga_core/blob/anuga-4.0-refactor-plan/REFACTOR_PLAN.md) (2026-02-28). These are adopted as guidance for future work.

### Quantity kernels: single source with compile-time OpenMP toggle

**Decision:** Unify `quantity_ext.pyx`, `quantity_ext_openmp.pyx`, `quantity_ext2.pyx` into
a single source file. OpenMP pragmas become no-ops when compiled without `-fopenmp`.

**Why:** The three files share ~90% code, creating a maintenance burden and risk of divergence.

### Parallel operators: MPI awareness in base class, not wrapper files

**Decision:** Eliminate the 5 thin parallel wrapper files. Move `self.parallel` flag into base
operator classes so serial/parallel behaviour is a runtime switch, not a separate class.

**Why:** The wrappers add complexity without meaningful behaviour difference.

### Linting: incremental ruff enforcement, never bulk-format

**Decision:** Use `ruff` (not pyflakes/pep8) for linting. Format only files being modified —
never run a bulk formatter across the whole repo.

**Why:** Bulk formatting creates massive diffs that conflict with upstream PRs and obscure
meaningful changes in git history.

### Testing: pytest over unittest for new tests, convert selectively

**Decision:** New tests written as plain pytest functions. Convert existing `unittest.TestCase`
classes selectively when touching a file, not as a bulk migration.

**Why:** pytest fixtures (especially `tmp_path`) solve the test isolation problem more cleanly
than the current `set_datadir('.')` / `tempfile.mktemp()` pattern.

### Coverage: 65% target enforced in CI, 80% on changed lines via diff-cover

**Decision:** Configure `.coveragerc` with `branch=true, fail_under=65`. Install `diff-cover`
to enforce 80% coverage on lines changed in each PR.

**Why:** A blanket 65% target is achievable; the 80% diff-cover rule ensures new code is
always well-tested without requiring a wholesale coverage lift immediately.

### Fork hygiene: rebase monthly, squash-merge phases, never force-push main

**Decision (Hydrata-specific, relevant for upstream sync):**
- Rebase feature branches against `anuga-community/main` at least monthly
- Squash-merge completed phases rather than accumulating merge commits
- Never bulk-format to keep diffs clean for upstream PRs

**Why:** Hydrata's fork must stay mergeable with upstream. Coordination on
`pyproject.toml` dependency changes is especially important.
