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

**P1.3 Add tests for `simulation/` and `validation_utilities/`**
Neither module has a test file. `anuga/simulation/` handles checkpoint save/restore;
`anuga/validation_utilities/` wraps all the `validate_*.py` scripts. Both exercised only
via integration runs.

**P1.4 Fix `gauge.py` verbose/print hygiene**
`anuga/abstract_2d_finite_volumes/gauge.py` — `sww2csv_gauges` and `sww2timeseries` print
unconditionally despite logging refactor (sessions 10/11). Convert to `log.info()` and
honour a `verbose` flag.

**P1.5 Add deprecation warnings to legacy forcing classes**
`anuga/__init__.py` comments say "These are old, should use operators" for `Inflow`,
`Rainfall`, `Wind_stress` in `shallow_water/forcing.py`. Add explicit `DeprecationWarning`
pointing to `Rate_operator`. Matches the camelCase deprecation pattern already established.

**P1.6 Implement `local_timestepping` or remove the dead infrastructure**
`generic_domain.py:2627` — `compute_flux_update_frequency` is a `pass` stub called every
timestep when `max_flux_update_frequency != 1`. Four supporting arrays (`edge_timestep`,
`update_next_flux`, `update_extrapolation`, `flux_update_frequency`) are allocated
unconditionally. Either implement it (2–5× speedup on dry-area-heavy domains) or remove the
call and allocations to recover memory. Lazy-allocation pattern from session 21 (DM work)
is the model.

**P1.7 Write tests for `anuga/utilities/animate.py`**
`SWW_plotter`, `_animated_frame`, `_draw_elev_contours` etc. have no test file under
`anuga/utilities/tests/`. Create a small SWW fixture, call `save_depth_frame`, verify output
file exists and has plausible pixel statistics.

**P1.8 Clean up `file_function.py` FIXMEs**
`anuga/abstract_2d_finite_volumes/file_function.py` has 9 FIXME comments about coordinate
origin reconciliation and caching. The `Geo_reference` EPSG work (session 4) resolved the
coordinate system side — now `FIXME: Check that model origin is the same as file's origin`
and `FIXME: Use geo_reference to read and write xllcorner` are directly addressable.

---

## Priority 2 — Medium effort (1–2 weeks each)

**P2.1 Type hints on the public API**
`anuga/abstract_2d_finite_volumes/quantity.py` — 81 public methods, 0 type hints. Same for
`shallow_water_domain.py` (~50 methods), `operators/base_operator.py`,
`structures/structure_operator.py`. The 171-entry `__all__` is the natural scope boundary.
Start with return types and key parameters. Enables IDE autocomplete and catches signature
mismatches.

**P2.2 Refactor `Generic_Domain.__init__` (367 lines)**
`generic_domain.py:63` — constructor handles mesh setup, quantity initialisation,
timestepping config, algorithm selection, parallel setup, and NVTX hooks in 367 lines.
Extract into `_init_quantities()`, `_init_timestepping()`, `_init_algorithms()`,
`_init_parallel()`.

**P2.3 Refactor `create_riverwalls` (300 lines)**
`riverwall.py:158` — monolithic function with nested polygon loops. The throughflow work
(session 9) added more complexity. Extract `_validate_riverwall_inputs()`,
`_create_riverwall_segments()`, `_assign_boundary_tags()`, `_setup_weir_operators()`.

**P2.4 Consolidate `culvert_class.py` / `new_culvert_class.py` compute_rates duplication**
Both have ~188-line `compute_rates` methods with nearly identical hydraulic logic. Extract
shared orifice/weir/pipe calculations into `anuga/culvert_flows/hydraulic_utils.py`. The
`new_culvert_class` naming signals an intended migration never completed.

**P2.5 Improve `Rate_operator` usability**
`anuga/operators/rate_operators.py` — 16 constructor parameters, inconsistent defaults, no
input validation, no factory methods. Add `Rate_operator.rainfall(domain, rate)` and
`Rate_operator.inflow(domain, rate)` convenience constructors. Add validation for type
compatibility between `rate` and `factor`.

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

**P3.1 Implement local timestepping**
`compute_flux_update_frequency` has been a `pass` stub for years. Local timestepping (skipping
flux recalculation in slow-moving regions) can give 2–5× speedup on domains with large dry/slow
areas. Infrastructure is in place (four arrays allocated). Requires criterion function
(velocity + triangle size → update frequency) and validation against analytical solutions for
conservation. See P1.6 — implement rather than remove.

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

**P3.6 `anuga_sww_gui` erosion delta-bed view**
For erosion/deposition simulations, a "delta elevation" display mode (elevation at frame T
minus elevation at frame 0) would immediately reveal net deposition/erosion zones. The
`_elev_frame` infrastructure is in place; this is a new `qty='elev_delta'` path in
`SWW_plotter` and a new entry in the Quantity dropdown.

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
