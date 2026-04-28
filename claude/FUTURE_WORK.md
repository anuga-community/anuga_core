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

**P2.4 Delete the `anuga/culvert_flows/` package**
Session 26 cleanup: deleted `new_culvert_class.py` (was a re-export shim) and
`test_new_culvert_class.py` (duplicate of `test_culvert_class.py`); added package-level
`DeprecationWarning` in `__init__.py`. Remaining work to complete removal:
- Update `examples/structures/run_open_slot_wide_bridge.py` to use `Boyd_box_operator`
  instead of `Culvert_flow` and `boyd_generalised_culvert_model`
- Verify no external user scripts depend on `culvert_routines.boyd_generalised_culvert_model`
- Delete `culvert_class.py`, `culvert_routines.py`, `culvert_polygons.py` and their tests
- Target: v5.0

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

**P3.8 ADER-2 / MUSCL-Hancock fused predict-extrapolate kernel (single extrapolation)**
Current `DE_ader2` uses two extrapolation passes per step:
1. Extrapolate Q^n → edges (predictor reads these to recover limited slopes)
2. Predictor overwrites centroids Q^n → Q^{n+1/2}
3. Re-extrapolate Q^{n+1/2} → edges (flux reads these)

The two passes are forced because the predictor *consumes* Q^n edges as input but
the flux call needs Q^{n+1/2} edges as output — different states, different passes.

**Two equivalent approaches to eliminate the second extrapolation:**

**Option A — MUSCL-Hancock predictor (simpler):**
The Hancock half-step uses the Q^n edge values that already exist after extrapolation
to estimate the local flux divergence, then shifts all edge values by the same amount:
```
div_F ≈ (1/A_k) * Σ_i  F(Q_edge^n[k,i]) · n_i * |e_i|
Q_edge^{n+1/2}[k,i] = Q_edge^n[k,i]  -  (dt/2) * div_F
```
This is purely local (no neighbour access), requires no re-limiting (slopes are
preserved, only the centroid level shifts), and writes Q^{n+1/2} edge values in-place.
The full scheme per step:
```
extrapolate Q^n → edges                # 1 extrapolation (same as now)
update_boundary
fused_hancock_predict(prev_dt)         # in-place edge update, purely local
compute_fluxes from Q^{n+1/2} edges   # 1 flux call
compute_forcing_terms
saxpy(0,1) + update_conserved_quantities
```
**Well-balance caveat:** the flux divergence F = [uh, u²h+gh²/2, uvh] omits the bed
slope source term g*h*∂z/∂x. On a sloped still-water bed the Hancock predictor
produces nonzero div_F even at rest, disturbing the well-balance property. Fix: add
the discrete bed-slope correction `g*h_c * Σ_i z_edge[k,i] * n_i * |e_i| / A_k`
to the divergence estimate, analogous to the well-balanced pressure decomposition
already used in `core_compute_fluxes_central`.

**Option B — C-K fused kernel (more accurate, more complex):**
Recover slopes from neighbour centroid values (Green-Gauss), apply the existing
limiter, evaluate the full SWE C-K time derivative (inherently well-balanced via
w_x = h_x + z_x decomposition), and write Q^{n+1/2} directly into edge_values:
```
for each cell k:
    slopes from neighbours[k,0..2] centroid values via Green-Gauss
    apply beta limiter (replicate hfactor logic from core_extrapolate_second_order_edge)
    CK time derivatives (same as current core_ader_ck_predictor)
    edge_values[k,i] = (Q_c^n + slope*offset) + (dt/2)*dQ/dt   for i=0,1,2
```
More complex to implement (needs neighbour access, full limiter replication) but
naturally well-balanced and matches the existing C-K predictor's accuracy.

**Recommendation:** start with Option A (Hancock) — it reuses the existing
extrapolation output with minimal new code. Add the well-balance correction from
the outset. Option B is worth pursuing if Option A shows accuracy issues on
strongly sloped beds.

**Common to both options:**
- Bootstrap (first step) still runs plain extrapolation + Euler to establish
  `_ader2_prev_dt`. All subsequent steps use 1 extrapolation + 1 flux call.
- The fused kernel must NOT update centroid values — Q^n centroids are preserved
  intact for the `saxpy(0,1)` restore before `update_conserved_quantities`.
- Expected speedup over current `DE_ader2`: ~1.15–1.25× (one fewer extrapolation,
  ~220 FLOP/cell). Over `DE1` (RK2): ~1.35–1.45×.

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
| P2 — Medium | 9 | 3 | 1–2 weeks | Usability, type safety, test coverage |
| P3 — Initiatives | 8 | 7 | 1–3 months | Performance, scalability, accuracy |
| Speculative | 4 | 4 | 6+ months | Strategic differentiation |

**Top 3 near-term recommendations:**
1. **P2.1** — Type hints on public API (`quantity.py`, `shallow_water_domain.py`, `base_operator.py`)
2. **P2.4** — Complete `culvert_flows/` removal: update `run_open_slot_wide_bridge.py` example, delete remaining files, target v5.0
3. **P2.6** — Continue raising fast-suite coverage threshold (currently 58%; next targets in `fit_interpolate/` and `structures/`)
