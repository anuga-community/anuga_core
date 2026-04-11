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
| **v3.3.1** | `develop` → `main` | **SHIPPED 2026-04-01** — tagged, PyPI + conda-forge published |
| **v4.0.0** | `feat/sc26` → `develop` → `main` | In progress — feat/sc26 merged into develop |

**v3.3.1:** Shipped. Includes EPSG/CRS support, utm→pyproj replacement, sww_merge fixes,
sww2vtu converter, pyproj DeprecationWarning fixes.

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
conda activate anuga_env_3.12
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

### Quick wins (< 1 day)
1. Audit `anuga/file/` for remaining bare `open()` calls — 1.3
2. Grep for large legacy comment blocks in `shallow_water/` — 1.5
3. Complete GDAL removal (continue `remove-gdal` branch work) — H1.2
4. Add `ruff` configuration to `pyproject.toml` — H2.1
5. Export `Geo_reference` / `is_located` / `epsg` from `anuga/__init__.py` if not already there

### Medium effort (1–3 days)
6. Fix test isolation — sweep `set_datadir('.')` and `tempfile.mktemp()` — H0.1
7. Configure coverage baseline (`.coveragerc`, `diff-cover`) — H0.4
8. GitHub Actions CI matrix — H0.5
9. Pre-commit hooks with ruff — H2.2

### Large effort (1+ weeks)
10. Unify quantity kernels (Cython refactor, high risk) — H3.1
11. Consolidate parallel operator wrappers — H3.2
12. Automate 32 remaining validation scenarios — H4.2
13. Reduce parameter counts in `gauge.py`/`generic_domain.py` — 4.1
