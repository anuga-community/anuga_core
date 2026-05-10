# Future Work Recommendations

Generated: 2026-04-24 (session 23). Last updated: 2026-04-25 (session 25).
Based on codebase investigation cross-referenced against 25 sessions of completed work.

Items marked ~~strikethrough~~ have been invalidated (see notes).

---

## Priority 1 — High value, low effort (1–3 days each)

~~**P1.1 Delete or absorb `boyd_box_operator_Amended3.py`**~~ — Done (session 25).

~~**P1.2 Add tests for the `rain` module**~~ — Done (session 25).

~~**P1.3 Add tests for `simulation/` and `validation_utilities/`**~~ — Done. `test_simulation.py` and `test_validation_utilities.py` exist.

~~**P1.4 Fix `gauge.py` verbose/print hygiene**~~ — Done (P2.7 session 24). No bare `print` calls remain; all logging via `log.info()`/`log.warning()`.

~~**P1.5 Add deprecation warnings to legacy forcing classes**~~ — Done (session 25). `DeprecationWarning` added to `Inflow`, `Rainfall`, `Wind_stress`, `Barometric_pressure` in `shallow_water/forcing.py`; `filterwarnings` in `pyproject.toml` suppresses them in the test suite.

~~**P1.6 Remove local-timestepping dead infrastructure**~~ — Done (session 23). Removed
`max_flux_update_frequency`, `flux_update_frequency`, `update_next_flux`,
`update_extrapolation`, `edge_timestep`, and `allow_timestep_increase` from Python domain,
C header, Cython wrapper, scenario system, and tests. Deleted
`test_local_extrapolation_and_flux_updating.py`. See P3.1 for the future implementation plan.

~~**P1.7 Write tests for `anuga/utilities/animate.py`**~~ — Done. `test_animate.py` exists under `anuga/utilities/tests/`.

~~**P1.8 Clean up `file_function.py` FIXMEs**~~ — Done (session 24). FIXMEs already resolved; deleted dead commented-out blocks, replaced raw `fid.xllcorner`/`fid.yllcorner`/`fid.zone` reads with `Geo_reference(NetCDFObject=fid)`, cleaned up redundant `.csv` branch and trailing NOTE comment.

---

## Priority 2 — Medium effort (1–2 weeks each)

~~**P2.1 Type hints on the public API**~~ — Done (session 33). Annotated ~130 public methods
across four files using `from __future__ import annotations` (PEP 563, Python 3.10+ compatible).
`base_operator.Operator` (11 methods, full coverage), `quantity.Quantity` (~28 methods),
`structure_operator.Structure_operator` (~28 methods including all `get_enquiry_*` getters),
`shallow_water_domain.Domain` (~60 methods including `__init__`, `evolve`, `set_flow_algorithm`,
`get_wet_elements`, `timestepping_statistics`, all inundation/volume queries). Uses
`numpy.typing.ArrayLike`, `collections.abc.Callable`, `TYPE_CHECKING` guards for heavy imports.
Commits `6c16986e`, `8f39e645`.

~~**P2.2 Refactor `Generic_Domain.__init__` (367 lines)**~~ — Done (session 25). Extracted
`_init_mesh()`, `_init_quantities()`, `_init_parallel()`, `_init_timestepping()`.
`__init__` is now ~25 lines. 743 domain/shallow-water tests pass.

~~**P2.3 Refactor `create_riverwalls` (300 lines)**~~ — Done (session 25). Extracted
`_validate_riverwall_inputs()`, `_match_edges_to_segments()`, `_build_hydraulic_properties()`
from the 300-line monolith. `create_riverwalls` is now a ~50-line orchestrator. All 43
riverwall tests pass.

~~**P2.4 Delete the `anuga/culvert_flows/` package**~~ — Done (session 34). Deleted the
entire package (5 412 lines removed, commit `b151fa66`): `culvert_class.py`,
`culvert_routines.py`, `culvert_polygons.py`, all tests and test data. Updated
`run_open_slot_wide_bridge.py` to drop legacy imports and show a `Boyd_box_operator`
equivalent. Removed dead culvert_flows references from `test_failure.py` and five
parallel test files. `subdir('culvert_flows')` removed from `anuga/meson.build`.
All 2 633 fast tests pass.

~~**P2.5 Improve `Rate_operator` usability**~~ — Done (session 24). Added `Rate_operator.rainfall(domain, rate_mm_hr)` and `Rate_operator.inflow(domain, rate_m3s)` factory classmethods; input validation (bad rate type → TypeError, region+polygon conflict → ValueError); updated `__init__` docstring pointing to factories. 13 new tests.

**P2.6 Raise fast-suite coverage threshold**
Fast suite at 54.66% against `fail_under=57` set for the full suite. Either set separate
thresholds in `.coveragerc` for fast vs full, or add targeted tests in `anuga/file/`,
`anuga/fit_interpolate/`, and `anuga/structures/` to lift the fast-suite baseline. Session
20 added ~90 tests as a model.

~~**P2.7 Modernise `sww2timeseries` / gauge module**~~ — Done (sessions 27–28).
- `gauge_get_from_file` rewritten with `csv.DictReader` (case-insensitive, whitespace-tolerant)
- `open().close()` file-existence checks replaced with `os.path.isfile()`
- `_generate_figures` marked `# pragma: no cover` (matplotlib/LaTeX display dependency)
- `plot_polygons` in `geometry/polygon.py` fixed: replaced `matplotlib.use('Agg')` with
  `plt.switch_backend('Agg')` (safe post-import); added defensive try-except around import
  block and plot body — resolves the matplotlib 3.10 / numpy 2.x `_NoValueType` crash
- `test_gauge.py` at 41 tests covering gauge.py at 99% (only lines 177-178, read-permission
  error path, remain uncovered)

Speculative future work: add EPSG/`Geo_reference` coordinate support to
`gauge_get_from_file` (accept optional EPSG code, convert to domain projection).

~~**P2.8 Scenario system input validation**~~ — Done (session 25). Schema validation added to TOML inputs; detailed error messages naming bad fields and expected types; range checks for physical parameters.

~~**P2.9 Document the scenario/TOML system**~~ — Done (session 25). Sphinx reference page added listing all supported TOML keys with types, defaults, and examples.

---

## Priority 3 — Larger initiatives (weeks to months)

**P3.1 Implement local timestepping (GPU-compatible redesign)**
The 3.1.9 implementation (tag `anuga_core_3.1.9`, `swDE1_domain.c`) used per-edge power-of-2
update frequencies computed by a 3-pass algorithm (lines 482–641). The skip logic in
`_compute_fluxes_central` checked `update_next_flux[ki] != 1` before computing each edge flux.
**Why this design is not GPU-compatible**: uses `already_computed_flux[k,i]` as a per-edge
mutex (race condition under parallel), accumulates a static `local_timestep` across skipped
steps, and processes edge pairs sequentially. The current GPU kernel
(`core_kernels.c:_compute_fluxes_central`) uses `#pragma omp target teams distribute parallel
for` with reductions — incompatible with per-edge skip flags.

**GPU-compatible redesign**: per-triangle activity mask (grouped sub-cycling). Slow triangles
sit out for multiple steps, but the flux loop remains fully data-parallel. Requires: (1) CFL
criterion mapping triangle velocity+size → activity level, (2) grouped sub-cycle scheduler,
(3) conservation validation against analytical solutions. Estimated 2–5× speedup on domains
with large dry/slow areas. The 3.1.9 source in `/home/steve/anuga_core_3.1.9` is the
algorithmic reference.

**P3.2 Higher-order spatial reconstruction**
Current extrapolation is linear (second-order smooth, first-order near gradients). Limited
third-order reconstruction (MUSCL-Hancock or ADER) would improve accuracy for long-distance
tsunami propagation. The consolidated `quantity_openmp_ext.pyx` (session 14, H3.1) is the
right place. Requires careful monotonicity limiting.

~~**P3.8 ADER-2 / MUSCL-Hancock fused predict-extrapolate kernel (single extrapolation)**~~ — Done (session 31). Fused C-K predictor + extrapolation into a single kernel pass, eliminating the second extrapolation. **1.75× faster than DE1.**

**P3.3 Improve `fit_interpolate` accuracy and performance** *(partial — sessions 25–26)*
Session 25–26: `Fit.select_alpha()` added with L-curve criterion (20 log-spaced candidates
1e-6 … 100, scipy sparse solves, max-curvature corner detection, fallback to DEFAULT_ALPHA).
`dok_to_csr` added to `fitsmooth_ext.pyx` for non-destructive DOK→CSR conversion.
`alpha='auto'` wired in `Fit.fit()`.  fit.py coverage 78→92%.
Remaining: (2) validation suite against known surfaces, (3) profile and vectorise inner loops
in `interpolate.py` (1200 lines, multiple "DESIGN ISSUES" comments, 82% coverage).

**P3.4 Parallel load-balancing monitoring**
Static METIS decomposition doesn't adapt as the wet front advances in inundation simulations.
Add runtime imbalance reporting (which rank is the bottleneck, imbalance ratio) and explore
dynamic repartitioning. The weak-scaling scripts from session 14 provide the benchmarking
framework.

**P3.5 GPU memory ceiling for large domains**
Current GPU offloading caps at ~2.25M triangles on typical 16 GB VRAM. The quantity memory
reduction work (session 13, ~58% saving) helps but is not sufficient for continental-scale
runs. Options: CUDA Unified Memory (`cudaMallocManaged`) or selective quantity transfer (only
GPU what's needed per sub-step).

~~**P3.6 `anuga_sww_gui` erosion delta-bed view**~~ — Done (session 24). Added `elev_delta` quantity: `elev_delta` property on `SWW_plotter`, `_elev_delta_frame`/`save_elev_delta_frame` methods (RdBu_r colormap, symmetric auto-limits), `_elev_delta_frame_count`, worker entry in `_animate_worker.py`, full GUI wiring in `anuga_sww_gui.py`. 6 new tests.

**P3.7 Streaming SWW reads for very long simulations**
`SWW_plotter` currently reads the full time dimension into memory on load. For very long
simulations (thousands of timesteps at high resolution) this can become the memory bottleneck.
A lazy/chunked reader using NetCDF4 variable slicing would allow the GUI and animate.py to
work without loading all data upfront.

---

## Speculative / Long-term

**S1 ML-fitted friction coefficients** — Replace fixed Manning's n with spatially varying
values trained on observed water levels. Potentially high accuracy gain for urban flood
applications. Requires calibration data and adjoint or ensemble methods.

**S2 Adaptive mesh refinement** — Dynamically refine triangles near the wet/dry front or
structures during simulation. Significant algorithmic complexity (remeshing, quantity
projection, parallel redistribution) but would reduce element count for long-range runs.

**S3 Real-time web visualisation** — Replace the desktop GUI with a WebGL viewer that can
stream frames from a running simulation. Lower barrier for classroom and stakeholder use.
Existing frame-generation pipeline as backend.

**S4 Operator composition / scenario DSL** — Higher-level description language for scenarios
(e.g., "rainstorm at 50 mm/hr for 2 hours, then tidal forcing") that auto-composes
`Rate_operator`, `File_boundary`, etc. Would reduce boilerplate for common operational setups.

---

## Invalidated suggestions

~~**HDF5/Zarr output format**~~ — ANUGA uses NetCDF4 (HDF5-backed), which has no 2 GB
per-variable size limit. The NetCDF3 classic restriction does not apply. (Invalidated
2026-04-24.)

---

## Summary

| Priority | Total | Remaining | Effort | Biggest payoff |
|----------|-------|-----------|--------|----------------|
| P1 — Quick wins | 8 | 0 ✅ | 1–3 days | All done |
| P2 — Medium | 9 | 1 | 1–2 weeks | Test coverage |
| P3 — Initiatives | 8 | 6 | 1–3 months | Performance, scalability, accuracy |
| Speculative | 4 | 4 | 6+ months | Strategic differentiation |

**Top 3 near-term recommendations:**
1. **P2.6** — Raise fast-suite coverage threshold (currently ~58–59%; next targets in `fit_interpolate/` and `structures/`)
2. **P3.3** — `fit_interpolate` accuracy: validation suite against known surfaces, profile `interpolate.py` inner loops
3. **P3.1** — Local timestepping GPU-compatible redesign (per-triangle activity mask sub-cycling)
