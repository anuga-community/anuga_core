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
three merge functions use `Geo_reference(NetCDFObject=fid)` + `write_NetCDF()`
instead of field-by-field copying, and pass `timezone` to `store_header()`.
Added `sww2vtu` converter (`anuga/file_conversion/sww2vtu.py`) for ParaView —
writes VTU + PVD directly (no VTK dependency), derived `depth` and `speed`,
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
triangles. Added ruff linting config and fixed all genuine violations.
Pre-commit hooks (ruff), CI lint workflow, CI fast/slow test split, coverage
config. Full test isolation sweep: `tempfile.mktemp` → `mkstemp`,
`set_datadir('.')` → `mkdtemp()`, all `sww2dem` output paths use full temp
dir paths, orphaned CWD files cleaned up. 1537 pass.

**Session 9 (2026-04-04):** Riverwall throughflow (`Cd_through`) — submerged
orifice formula in `core_kernels.c`/`gpu_device_helpers.h`; column 5 of
`hydraulic_variable_names`; 6 new tests; backward-compatible guard; Jupyter
notebook demo section added. Fixed 4 parallel test failures. NPY002: 17
legacy `np.random.*` calls → `np.random.default_rng()` in 6 files. H1.2
GDAL fully removed: `gdal_available` → `spatial_available`; CLI calls in
`scenario/raster_outputs.py` replaced with rasterio/fiona/numpy. H0.3
golden-master snapshots: 6 `pytest-regressions` tests committed with baselines.
108/148 tracked items done.

**Session 10 (2026-04-06):** L1-L4 logging refactor: `TeeStream` (tees
print/stdout to terminal+file), lazy log file, `log.verbose()`, `log.file_only()`
context manager, `setup_mesh.py` quieted. Logging docs page added. Archive
CuPy/CUDA files to `archive/cupy_cuda/`. Fix `test_sww2csv_multiple_files`
stale-file pollution. CI: add `pytest-regressions` to all 13 env YMLs; drop
Python 3.8. Propagate v3.3.0/3.3.1/3.3.2 to GA remote. L5: 715
`log.critical()` → `log.info()`/`log.warning()` across 70+ production files.
Drop Python 3.9 (`X | Y` syntax requires ≥3.10). 116/156 tracked items done.

**Session 11 (2026-04-07):** G1.3 slot limit hard errors — `Rate_operator`,
`Inlet_operator`, `Parallel_inlet_operator` raise `RuntimeError` on GPU slot
overflow (2 tests). Benchmark suite: `benchmarks/run_benchmarks.py`
(single-process, all modes, JSON output) + `benchmarks/compare_benchmarks.py`
(±% delta table). MPI distribution benchmark: unified 4-method comparison
(`distribute()`, `collaborative()`, `distribute_basic_mesh()`, `dump+load`).
Bug fix BF1: `Basic_mesh.reorder()` stale neighbours caused `distribute_basic_mesh()`
to produce ~59% more ghost triangles. Regression test added. 122/159 done.

**Session 12 (2026-04-09):** GPU Phase 1 completion. G1.4 fix: `tri_full_flag`
not set in GPU domain — ghost cells were included in timestep minimum, causing
artificially small timesteps in multi-rank runs; fixed in `sw_domain_gpu_ext.pyx`
and test added (`test_parallel_sw_flow_gpu.py`). G1.2: GPU device memory check
(`gpu_check_device_memory()`) before first `omp target enter data`; raises
`RuntimeError` on OOM; 5 tests. G1.1: `File_boundary` / `Field_boundary` GPU
support — struct + Python push pattern, `gpu_file_boundary_init/set_values/evaluate`,
called each sub-step in both Python and C RK loops; 3 tests. G1.5: SSP-RK3 GPU
support — `gpu_evolve_one_rk3_step` (3-stage Shu-Osher C loop),
`gpu_saxpy3_conserved_quantities`; `use_c_rk2_loop` renamed to `use_c_rk_loop`
(deprecated alias kept); 3 tests.

**Session 13 (2026-04-09):** Quantity memory reduction QM1-QM5 and TOML scenario
extensions. QM1-QM5: `qty_type` concept (`evolved`, `edge_diagnostic`,
`centroid_only`, `coordinate`) controls which arrays are allocated; lazy
`vertex_values` property on all types; `explicit_update`, `semi_implicit_update`,
`centroid_backup_values`, `phi` stripped from `elevation`; `friction` reduced to
centroid only; `height`/`xvelocity`/`yvelocity` reduced to centroid+edge; ~58%
memory saving on typical domains; slow integration test added. Fixed Quantity
memory-layout docs example. TOML scenario runner: added `[[culverts]]` and
`[[weirs]]` support (`Boyd_box_operator`, `Boyd_pipe_operator`,
`Weir_orifice_trapezoid_operator`); corresponding docs section added.

**Session 14 (2026-04-10):** GPU Phase 2 and Phase 3, QM6, and H3.1. Removed dead
`'original'`/`'tsunami'` distribute path code and updated parallel docs. QM6: made
`x_gradient`, `y_gradient`, `phi` lazy for ALL quantity types including `evolved`;
removed `static_with_gradients` type; elevation now uses `edge_diagnostic`. H3.1:
consolidated quantity C extension to single `quantity_openmp_ext.pyx`; removed
dead `quantity_ext2.pyx`. G3.1: `Weir_orifice_trapezoid_operator` added to
`GPUCulvertManager`; 3 tests. G3.4: GPU offloading documentation page added
(`docs/source/parallel/use_gpu_offloading.rst`). G2.1-G2.4: GPU benchmark suite
(`run_gpu_benchmarks.py`), GPU-aware MPI runtime detection (`MPIX_Query_cuda/rocm_support`),
NVTX profiling hooks (`gpu_nvtx.h`, 10 kernel markers), weak-scaling experiment
scripts (`benchmarks/run_weak_scaling.py`, `scripts/hpc/weak_scaling.slurm`).
G3.3: dynamic heap growth for rate/inlet/culvert slot arrays (replaces static
`MAX_*` limits). Ruff UP009: removed unnecessary UTF-8 coding declarations.
145/163 tracked items done.

**Session 15 (2026-04-10):** H4.2 validation test scripts — 33 `validate_*.py`
scripts covering all remaining scenarios (analytical, behaviour-only, case
studies). H4.3 coverage enforcement — extended `.coveragerc` omit rules,
deleted dead `change_friction_operator.py`, added `test_mannings_operator.py`
and `test_sww2vtu.py`, switched CI coverage to full suite, enforced
`fail_under=52`. Commit `dcf57756`.

**Session 16 (2026-04-10):** H4.2 validation test scripts (cont.) and H4.3
coverage enforcement. Fixed `run_anuga_script.py` silently returning 0 on
subprocess failure; fixed `scipy.genfromtxt` → `numpy.genfromtxt` in
`landslide_tsunami/runup.py`; improved validate error reporting (`%.4f` →
`%.2e`, denominator guard). H4.3: extended `.coveragerc` omit rules (~3000
lines excluded), deleted dead `change_friction_operator.py`, added
`test_mannings_operator.py` (8 tests) and `test_sww2vtu.py`, registered
`sww2vtu.py` in meson.build, switched CI coverage to full suite, removed
`continue-on-error: true`, enforced `fail_under=52`. Commit `dcf57756`.

**Session 17 (2026-04-11):** 4 GPU bug fixes on real hardware. BF2: relaxed
culvert/weir CPU↔GPU tolerances to `atol=0.02`. BF3: Mannings `RuntimeWarning`
in dry cells fixed with `safe_h = maximum(h, 1e-15)`. BF4: `Rate_operator`
empty-array check fixed. BF5: GPU_AWARE_MPI segfault — added
`host_send_buffer`/`host_recv_buffer` staging in `gpu_halo.c`. BF6:
empty-indices rate operators now marked `_gpu_initialized=True`, eliminating
~70 s per-timestep overhead. 4-GPU towradgi run: 140 s (n=1) → 70 s (n=4).

**Session 18 (2026-04-12):** BF7: Fixed double `get_triangle_containing_point`
call in `Parallel_Inlet_enquiry.compute_enquiry_index`. BF8:
`MeshQuadtree`-based spatial index built on 6th call, reused thereafter;
measurable speedup in culvert setup. H3.2: Extracted 3 MPI helper methods into
`Parallel_Structure_operator`; rewrote `discharge_routine` in 3 culvert wrappers
(−125 lines net). Full test suite: 0 failures.

**Session 19 (2026-04-13):** H3.4: Cleaned up `system_tools.py` — removed
`six` dependency, deleted dead code (`store_svn_revision_info`, `get_web_file`,
`tar_file`, `untar_file`, `get_file_hexdigest`, `make_digest_file`,
`MemoryUpdate`). 335 lines removed. 158/169 tracked items done.

**Session 20 (2026-04-13):** H4.4 coverage push — systematic `Test_*_extra`
classes across 10 test files: `test_polygon.py` (26 tests, geometry 76%→95%),
`test_lat_long_UTM_conversion.py` (6), `test_numerical_tools.py` (13),
`test_xml_tools.py` (9), `test_alpha_shape.py` (7), `test_sparse.py` (10),
`test_geo_reference.py` (12), `test_function_utils.py` (8), `test_cg_solve.py`
(3), `test_set_quantity.py` (4). Full suite at **63.88%**; `fail_under` raised
to 63. Commit `133b26b1`.

**Session 21 (2026-04-15):** Domain work-array memory reduction — ~740 MB saved
at N=2.25M. DM1: 9 dead C work arrays removed from `_ensure_work_arrays()`;
only 3 live arrays remain. DM2: `edge_flux_type`/`edge_river_wall_counter` lazy
for non-riverwall simulations; `sw_domain.h` NULL guard added. DM3:
`domain_memory_stats`/`print_domain_memory_stats`/`domain_struct_stats`/`print_domain_struct_stats`
added to `system_tools.py` and exported from `anuga`. All 439 tests pass.

**Session 22 (2026-04-21):** `anuga_animate_sww_gui` major feature release.
Parallel frame generation (ProcessPoolExecutor, fork on Linux, up to 4 workers)
via new `_animate_worker.py`. Zoom region: rubber-band rectangle re-generates
frames at full resolution for selected area (Set Zoom / Reset Zoom). `elev`
quantity: static (1-D) or time-varying (2-D for erosion); terrain colormap
default; shown in timeseries panel. Add terrain/gist_earth/gray to colormap
dropdown. Fix View Mesh multiple windows, Cancel button, app-close hang. Full
Sphinx docs with automated screenshot capture (`docs/capture_gui_screenshots.py`).
Commit `ebc68c37`.

**Session 23 (2026-04-24):** `anuga_sww_gui` GUI overhaul (renamed from
`anuga_animate_sww_gui`). Baked overlays: Show Elev (elevation contours via
`_draw_elev_contours`/`_nice_contour_levels`) and Show Mesh baked into PNG
frames at correct z-order during parallel generation; canvas overlays skip
double-render when already baked. Multi-point timeseries: `_ts_triangles`
list, tab10 colour palette, legend, multi-column CSV export, Clear picks
button. Save Frame / Export Frame: time-selection dialog (current frame or
comma-separated list of simulation times). UI reorganisation: 9 flat rows →
3-tab ttk.Notebook (Plot / Generate / Output) + always-visible bars. Basemap
checkbox for mesh viewer (live re-render on toggle) and Save Mesh dialog.
Sphinx RST, in-app help, and screenshots all updated. Commit `49c5b7d8`.

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

## Suggested next priorities

See `claude/PROGRESS.md` — "Remaining Work" section for full list. Summary:

### SC26 (needs GPU hardware)
1. **G4.1** Gordon Bell metrics — per-kernel timing, roofline model
2. **G4.2** Physical benchmark validation — Thacker, dam break (Ritter) in GPU mode
3. **G4.3** Multi-node strong scaling — 20 M triangles, 1→64 GPUs (scripts ready)

### Best standalone value (no GPU hardware needed)
4. **Coverage** — fast suite **~55%**, full suite **~58%** (2026-04-22, branch+statement with omit list). `fail_under = 57` in `.coveragerc`; CI runs the full suite so 58% > 57 passes with a 1-point margin. Earlier "67–70%" figures were artefacts of CI running from `sandpit/` without finding `.coveragerc`. Add tests opportunistically when touching files.
5. **anuga_sww_gui** — further ideas: erosion delta-bed visualisation (show change in elevation vs frame 0), side-by-side dual-quantity view, export zoomed animation.
