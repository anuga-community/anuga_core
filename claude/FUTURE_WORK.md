# Future Work Recommendations

Generated: 2026-04-24 (session 23)
Based on codebase investigation cross-referenced against 23 sessions of completed work.

Items marked ~~strikethrough~~ have been invalidated (see notes).

---

## Priority 1 — High value, low effort (1–3 days each)

**P1.1 Delete or absorb `boyd_box_operator_Amended3.py`**
`anuga/structures/boyd_box_operator_Amended3.py` (247 lines) sits alongside the canonical
515-line `boyd_box_operator.py`. Not imported or exported anywhere. Either merge any unique
improvements into the canonical version or delete it.

**P1.2 Add tests for the `rain` module**
`anuga/rain/` has no test directory. `calibrated_radar_rain.py` and
`raster_time_slice_data.py` are used in real scenarios but have zero automated coverage.
Target: initialisation, temporal interpolation, boundary conditions.

~~**P1.3 Add tests for `simulation/` and `validation_utilities/`**~~ — Done. `test_simulation.py` and `test_validation_utilities.py` exist.

~~**P1.4 Fix `gauge.py` verbose/print hygiene**~~ — Done (P2.7 session 24). No bare `print` calls remain; all logging via `log.info()`/`log.warning()`.

**P1.5 Add deprecation warnings to legacy forcing classes**
`anuga/__init__.py` comments say "These are old, should use operators" for `Inflow`,
`Rainfall`, `Wind_stress` in `shallow_water/forcing.py`. Add explicit `DeprecationWarning`
pointing to `Rate_operator`. Matches the camelCase deprecation pattern already established.

~~**P1.6 Remove local-timestepping dead infrastructure**~~ — Done (session 23). Removed
`max_flux_update_frequency`, `flux_update_frequency`, `update_next_flux`,
`update_extrapolation`, `edge_timestep`, and `allow_timestep_increase` from Python domain,
C header, Cython wrapper, scenario system, and tests. Deleted
`test_local_extrapolation_and_flux_updating.py`. See P3.1 for the future implementation plan.

~~**P1.7 Write tests for `anuga/utilities/animate.py`**~~ — Done. `test_animate.py` exists under `anuga/utilities/tests/`.

~~**P1.8 Clean up `file_function.py` FIXMEs**~~ — Done (session 24). FIXMEs already resolved; deleted dead commented-out blocks, replaced raw `fid.xllcorner`/`fid.yllcorner`/`fid.zone` reads with `Geo_reference(NetCDFObject=fid)`, cleaned up redundant `.csv` branch and trailing NOTE comment.

---

## Priority 2 — Medium effort (1–2 weeks each)

**P2.1 Type hints on the public API**
`anuga/abstract_2d_finite_volumes/quantity.py` — 81 public methods, 0 type hints. Same for
`shallow_water_domain.py` (~50 methods), `operators/base_operator.py`,
`structures/structure_operator.py`. The 171-entry `__all__` is the natural scope boundary.
Start with return types and key parameters. Enables IDE autocomplete and catches signature
mismatches.

~~**P2.2 Refactor `Generic_Domain.__init__` (367 lines)**~~ — Done (session 25). Extracted
`_init_mesh()`, `_init_quantities()`, `_init_parallel()`, `_init_timestepping()`.
`__init__` is now ~25 lines. 743 domain/shallow-water tests pass.

~~**P2.3 Refactor `create_riverwalls` (300 lines)**~~ — Done (session 25). Extracted
`_validate_riverwall_inputs()`, `_match_edges_to_segments()`, `_build_hydraulic_properties()`
from the 300-line monolith. `create_riverwalls` is now a ~50-line orchestrator. All 43
riverwall tests pass.

**P2.4 Consolidate `culvert_class.py` / `new_culvert_class.py` compute_rates duplication**
Both have ~188-line `compute_rates` methods with nearly identical hydraulic logic. Extract
shared orifice/weir/pipe calculations into `anuga/culvert_flows/hydraulic_utils.py`. The
`new_culvert_class` naming signals an intended migration never completed.

~~**P2.5 Improve `Rate_operator` usability**~~ — Done (session 24). Added `Rate_operator.rainfall(domain, rate_mm_hr)` and `Rate_operator.inflow(domain, rate_m3s)` factory classmethods; input validation (bad rate type → TypeError, region+polygon conflict → ValueError); updated `__init__` docstring pointing to factories. 13 new tests.

**P2.6 Raise fast-suite coverage threshold**
Fast suite at 54.66% against `fail_under=57` set for the full suite. Either set separate
thresholds in `.coveragerc` for fast vs full, or add targeted tests in `anuga/file/`,
`anuga/fit_interpolate/`, and `anuga/structures/` to lift the fast-suite baseline. Session
20 added ~90 tests as a model.

**P2.7 Modernise `sww2timeseries` / gauge module**
`anuga/abstract_2d_finite_volumes/gauge.py` — `sww2timeseries` (270 lines) and
`_sww2timeseries` (140 lines) predate the EPSG work and logging refactor. Update to use
`Geo_reference` EPSG support, replace print statements with `log.info()`, add a test file.
Primary post-processing tool for users.

**P2.8 Scenario system input validation**
`anuga/scenario/parse_input_data.py` and `parse_input_data_toml.py` (combined 800+ lines)
have minimal error checking. Add: (1) schema validation for TOML inputs using `tomllib`
(already a dependency), (2) detailed error messages naming the bad field and expected type,
(3) range checks for physical parameters (Manning's n > 0, etc.).

**P2.9 Document the scenario/TOML system**
The TOML keys used by the scenario system are not documented in Sphinx. Add a reference page
listing every supported key with types, defaults, and examples. See `claude/PROGRESS.md`
entries for the TOML culvert/weir support added in session 13.

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

**P3.3 Improve `fit_interpolate` accuracy and performance**
`anuga/fit_interpolate/interpolate.py` (1200 lines, multiple "DESIGN ISSUES" comments) —
least-squares smoother is sensitive to alpha (smoothing parameter) with no auto-selection
guidance; quadtree search has known degenerate cases. Work: (1) alpha auto-selection via
L-curve or GCV, (2) validation suite against known surfaces, (3) profile and vectorise inner
loops.

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

| Priority | Items | Effort | Biggest payoff |
|----------|-------|--------|----------------|
| P1 — Quick wins | 8 | 1–3 days | Coverage lift, dead code, logging consistency |
| P2 — Medium | 9 | 1–2 weeks | Usability, type safety, test coverage |
| P3 — Initiatives | 7 | 1–3 months | Performance, scalability, accuracy |
| Speculative | 4 | 6+ months | Strategic differentiation |

**Top 3 near-term recommendations:**
1. **P1.6** — Resolve the local-timestepping dead code (implement or remove)
2. **P1.7** — Add animate.py tests (regression protection for the GUI's most complex code)
3. **P2.8** — Scenario system input validation (biggest source of user friction in operational use)
