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

## What was improved (session summaries)

**Session 1 (2026-03-23):** All 20 documentation improvements — new RST pages,
expanded existing pages, fixed broken content, added MPI/GPU docs.

**Session 2 (2026-03-24):** 49/60 code improvements — mutable defaults, bare except,
with statements, dead code, correctness fixes, 17 new tests, API quality (naming,
deprecations, `__all__`), performance vectorisation, docstrings.

**Session 3 (2026-03-26):** Noise reduction in pytest output (triangle library,
logging), memory reporting, fast/slow test infrastructure, new API functions
(`memory_stats`, `basic_mesh_from_mesh_file`, `distribute_basic_mesh`),
`claude/` session-continuity directory created, Hydrata REFACTOR_PLAN incorporated,
release ROADMAP documented, `pyproject.toml` dependencies fixed.

**Session 4 (2026-03-26):** EPSG/CRS support added to `Geo_reference` — `epsg`
property (auto-computed for WGS84 UTM), `is_located()`, non-UTM national grids
(RD New, BNG, etc.) with pyproj-populated metadata (datum, projection, false
easting/northing), fixed pre-existing `read_NetCDF` bug, 23 new tests. New
CRS documentation page in Sphinx; `Geo_reference` added to API reference.

**Session 5 (2026-03-28):** Fixed `sww_merge` not propagating `hemisphere`,
`epsg`, and `timezone` from individual SWW files to the merged output — all
three merge functions (`_sww_merge`, `_sww_merge_parallel_smooth`,
`_sww_merge_parallel_non_smooth`) now use `Geo_reference(NetCDFObject=fid)` +
`write_NetCDF()` instead of field-by-field copying, and pass `timezone` to
`store_header()`. Added `sww2vtu` converter
(`anuga/file_conversion/sww2vtu.py`) for ParaView — writes VTU + PVD directly
(no VTK dependency), includes derived `depth` and `speed` quantities,
`--z-scale` and `--absolute-coords` options.

**Session 6 (2026-04-01):** GPU verbose flag (`int verbose` in C struct) to
suppress C printf output during pytest; fix pyproj DeprecationWarning for
1-element NumPy ≥ 2.0 arrays in `redfearn.py` and `tif2point_values.py`.
v3.3.1 tagged and shipped (PyPI + conda-forge). `feat/sc26` merged into
`develop`; `develop` is now the active v4.0.0 branch.

**Session 7 (2026-04-02):** Fixed ReadTheDocs shallow-clone showing
`0.0.0+unknown` — added `git fetch --unshallow --tags` pre-install step in
`.readthedocs.yaml`.

**Session 8 (2026-04-03):** Vectorised `get_flow_through_cross_section` —
NumPy segment scan replaces Python loop; C pre-filter skips non-intersecting
triangles (merged from `develop_3.x.x`). Added ruff linting config to
`pyproject.toml` and fixed all genuine violations. Pre-commit hooks (ruff),
CI lint workflow (`.github/workflows/lint.yml`), CI fast/slow test split
(`conda-setup.yml`), coverage config (`.coveragerc`). Full test isolation
sweep: `tempfile.mktemp` → `mkstemp`, `set_datadir('.')` → `mkdtemp()`, all
`sww2dem` output paths use full temp dir paths, orphaned CWD files cleaned up.
1 pre-existing failure (`test_sww2csv_multiple_files`) confirmed, 1537 pass.

**Session 10 (2026-04-06):** L1-L4 logging refactor: `TeeStream` (tees print/stdout to terminal+file), lazy log file (no file until `set_logfile()` called), `log.verbose()`, `log.file_only()` context manager, `setup_mesh.py` quieted. Logging docs page added. Fix pyproj DeprecationWarning in `tif2point_values.py`. Archive CuPy/CUDA files to `archive/cupy_cuda/`. Fix `test_sww2csv_multiple_files` stale-file pollution (chdir to tmpdir). CI: add `pytest-regressions` to all 13 env YMLs; drop Python 3.8. Recalibrate 3 NPY002 expected values in `test_geospatial_data.py`. Propagate v3.3.0/3.3.1/3.3.2 to GA remote. All cherry-picked from `main` → `develop` (had been on wrong branch). L5: 715 `log.critical()` → `log.info()`/`log.warning()` across 70+ production files; fix `test_sww2dem_verbose_True` with `mock.patch`. Drop Python 3.9 (`X | Y` syntax requires ≥3.10). 116/156 tracked items done.

**Session 9 (2026-04-04):** Riverwall throughflow (`Cd_through`) — submerged
orifice formula in `core_kernels.c`/`gpu_device_helpers.h`; column 5 of
`hydraulic_variable_names`; 6 new tests; backward-compatible guard; Jupyter
notebook demo section added. Fixed 4 parallel test failures (MPI-safe tempdir
broadcast in `test_parallel_dist_settings`; sww path fixes in
`test_sequential_dist_sw_flow`; run-scripts reverted to CWD). NPY002: 17
legacy `np.random.*` calls → `np.random.default_rng()` in 6 files. H1.2
GDAL fully removed: `gdal_available` → `spatial_available`; `gdalwarp`,
`gdal_rasterize`, `gdal_calc.py` CLI calls in `scenario/raster_outputs.py`
replaced with rasterio/fiona/numpy. H0.2 `test_sww2csv_multiple_files`
already passing (fixed by session 8 isolation work). H0.3 golden-master
snapshots: 6 `pytest-regressions` tests (dam break DE0/DE1, friction, Thacker
bowl, extrapolation, timestep sequence) committed with baselines. 108/148
tracked items done.

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

---

## Key reference documents

| Document | URL |
|----------|-----|
| Hydrata refactor plan | https://github.com/Hydrata/anuga_core/blob/anuga-4.0-refactor-plan/REFACTOR_PLAN.md |
| anuga-community GitHub | https://github.com/anuga-community/anuga_core |
| Hydrata fork | https://github.com/Hydrata/anuga_core |

## Suggested next priorities

See `claude/PROGRESS.md` — "Remaining Work" section for full list. Summary:

### Best standalone value (no GPU hardware needed)
1. **QM1–QM6** Quantity memory reduction — pure Python, ~2 days, ~58% saving
   for 1M-triangle domain (832 → 352 MB). Directly benefits SC26 (larger
   problems fit in GPU memory). Plan in `claude/QUANTITY_MEMORY_PLAN.md`.
   *(Waiting on Jorge's feedback before starting)*

### Medium effort (1–3 days each)
2. **G1.4** End-to-end GPU regression test (CPU_ONLY_MODE, no hardware needed)
3. **G1.3** Slot limit hard errors in GPU operator managers
4. **H3.2** Consolidate parallel operator wrappers
5. **H3.3** Merge `Culvert_operator` / `Culvert_operator_Parallel`
6. **H3.1** Unify Cython quantity kernels (high risk)

### SC26 (needs Jorge's GPU hardware)
7. G1.1 File_boundary GPU support, G1.2 device memory check
8. G2.1 benchmark suite, G2.4 weak scaling
9. G4 Gordon Bell metrics + paper benchmarks
