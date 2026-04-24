# Known Issues and Gotchas

Things discovered during development sessions that are surprising, non-obvious,
or require caution when working in specific areas.

---

## Build

### `--no-build-isolation` is mandatory

`pip install -e .` without `--no-build-isolation` fails because meson-python
cannot find the Cython/numpy already installed in the conda environment.
Always use:
```bash
pip install --no-build-isolation -e .
```

### Generated C files appear as untracked in `git status`

`sw_domain_openmp_ext.c` and other generated `.c` files are listed in
`.gitignore` but still show up as untracked. This is expected — they are
build artifacts.

---

## Testing

### `str.find()` returns 0 for first-position match (2026-03-26)

In `anuga/pmesh/mesh.py::_generateMesh_impl`, the old check `not self.mode.find('Q')`
was buggy: `str.find()` returns 0 when 'Q' is at position 0, and `not 0` is `True`,
so the check was treating a 'Q' at position 0 as "not found". Fixed by using
`'Q' not in self.mode`.

### Triangle library prints to stdout during pytest

The triangle C library writes to stdout when in verbose mode ('V' flag). Since
pytest `-s` does not suppress stdout, these appear as noise during test runs.
Fixed by ensuring `_generateMesh_impl` adds 'Q' (quiet) when `verbose=False`.

### `test_verbose_does_not_raise` triggers logging output

`anuga/abstract_2d_finite_volumes/tests/test_pmesh_to_mesh.py::test_verbose_does_not_raise`
intentionally calls with `verbose=True`. This triggers `General_mesh:` log output.
Fixed by wrapping with `logging.disable(logging.CRITICAL)` / `logging.disable(logging.NOTSET)`.

### Parallel tests run as subprocesses

Tests in `anuga/parallel/tests/` spawn `mpiexec` subprocesses. They cannot be
parallelised with `pytest-xdist` and must run serially. They are marked slow
and skipped by `--run-fast`.

---

## Numerical

### `== None` vs `is None` with numpy arrays

Using `== None` on a numpy array raises `ValueError: The truth value of an array
is ambiguous`. Always use `is None` / `is not None` throughout the codebase.

### `epsilon = 1.0e-6` wet/dry threshold

`anuga/config.py` defines `epsilon` as the wet/dry threshold. Many conditional
checks use `depth > epsilon` rather than `depth > 0`. Be aware of this when
writing new flux/operator code.

### `minimum_allowed_height = 1.0e-05`

Cells below this height are treated as dry. Negative depths are clipped.

---

## API

### `numpy` imported as `num` (not `np`)

This is a project-wide convention — do not change it to `np` in existing files.

### `anuga/__init__.py` is the single public API surface

All public names must be both imported and listed in `__all__` in `anuga/__init__.py`.
The file is ~1000 lines; search carefully before adding to avoid duplicates.

### camelCase methods in `pmesh/mesh.py` are deprecated

As of 2026-03-24, camelCase public methods have snake_case equivalents.
The camelCase versions emit `DeprecationWarning`. Prefer snake_case in new code.

### `get_CFL` / `set_CFL` are deprecated in `generic_domain.py`

Use `get_cfl()` / `set_cfl()` instead.

---

## Memory and Performance

### `psutil` is optional

`anuga/utilities/system_tools.py::memory_stats()` tries `psutil` first and falls
back to parsing `/proc/self/status` via `_VmB('VmRSS:')`. If neither works it
returns `'mem=?'`. The `psutil` package is not in the conda environment files
by default.

### Kinematic viscosity operator is slow

`test_kinematic_viscosity_operator.py` runs 4 tests that take 2–5 seconds each.
These are marked `@pytest.mark.slow` at module level.

---

## Structures

### `RiverWall` tests require full mesh with breaklines

`anuga/structures/riverwall.py` — tests were deferred because `RiverWall`
requires a domain with a mesh that has breaklines (specific mesh construction).
Simple rectangular domains don't suffice.

---

## SWW GUI / animate.py

### `replace_all=True` in Edit tool can change more than intended

When reverting a colormap from `terrain` → `Greys_r` with `replace_all=True`, the `_elev_frame` and `save_elev_frame` default arguments (which must stay `terrain`) were also reverted — requiring a second manual fix. Always check every occurrence of the target string in the file before using `replace_all`.

### Worker must accept all params even when a save method doesn't use them

`worker_frame` in `_animate_worker.py` calls `save_fn(frame=..., show_elev=..., elev_levels=..., show_mesh=...)` for every quantity. If a `save_*` method (e.g. `save_elev_frame`) doesn't declare those params, it raises `TypeError`. All `save_*` methods must accept `show_elev`, `elev_levels`, `show_mesh` even if they ignore the values.

### Double overlays when baked + canvas overlay both active

If Show Elev or Show Mesh is ticked during generation (baked into PNGs) and the canvas overlay is also active, contours/mesh appear twice. The canvas overlay methods check `self._last_gen_show_elev` / `self._last_gen_show_mesh` and return early when already baked. This guard must be maintained if either system is extended.

### Live mesh viewer redraw requires `ax.cla()` + full re-draw

When toggling the Basemap checkbox in `_show_mesh`, a simple `ax.set_visible()` or artist removal is not sufficient — the basemap tiles are added by `contextily` as Axes-level patches. The only reliable approach is `ax.cla()` (clear axis), re-draw the triplot, conditionally call `_add_basemap`, call `mesh_fig.tight_layout()`, then `mesh_canvas.draw()`.

---

## Scenario Module

### `anuga/scenario/` depends on `spatialInputUtil`

The scenario module (`prepare_data.py`, `setup_boundary_conditions.py`, etc.)
imports `spatialInputUtil`, a compiled C extension not included in the main repo.
Meaningful unit tests require this extension plus real shapefile/Excel test data.
Tests for this module are deferred.

---

## Hydrata Current-State Assessment (2026-02-28)

These are known issues identified in the Hydrata fork analysis that also apply to anuga-community.

### `pyproject.toml` declares only `numpy` as a dependency

Despite the codebase importing scipy, netCDF4, matplotlib, meshpy, dill, pymetis, pyproj,
and affine, `pyproject.toml` only lists `numpy>=2.0.0`. This means `pip install anuga`
on a clean venv will produce a package that fails at runtime.

**Fix:** Add the missing dependencies to `[project].dependencies`.

### Phantom dependencies: `cartopy` and `openpyxl`

These appear in code paths but are never actually imported at runtime. Their presence in
any install documentation is misleading.

### GDAL remnants on `remove-gdal` branch

GDAL was partially removed but remnants remain. The `remove-gdal` branch has the work
in progress. Merge not yet complete in anuga-community.

### `setup.py` still present alongside meson-python

Both `setup.py` and `pyproject.toml` (meson-python) exist. The `setup.py` is a
legacy artifact and should be removed once meson-only builds are confirmed in CI.

### Test isolation problems

- **47 `set_datadir('.')` calls** — many tests write files relative to CWD rather than
  a temp directory. Running tests from a non-repo directory can fail or pollute the tree.
- **198 `tempfile.mktemp()` uses** — `mktemp()` is a security risk (TOCTOU) and deprecated.
  Should be replaced with `tmp_path` fixture (pytest) or `tempfile.mkdtemp()`.
- **7+ tests write `domain.sww` to CWD** — parallel test runs step on each other.

### Code duplication (~7,700 redundant lines)

- Three quantity kernels share ~90% code: `quantity_ext.pyx`, `quantity_ext_openmp.pyx`, `quantity_ext2.pyx`
- Five parallel operator wrappers are near-identical to their `structures/` counterparts
- `Culvert_operator` and `Culvert_operator_Parallel` have near-identical logic
- `system_tools.py` is 750 lines with overlap against `numerical_tools.py`

### No linting or type annotations

Zero pre-commit hooks, no ruff/flake8 config, 4,189 functions with no type annotations.
Current approach is manual `pyflakes` / `autopep8` before commits.
