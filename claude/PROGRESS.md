# ANUGA Code & Documentation Improvement Progress

Last updated: 2026-05-01 (session 28)
Branch: `develop` (all feature branches merged)

Full history of completed work: `claude/PROGRESS_ARCHIVE.md`
Future work recommendations: `claude/FUTURE_WORK.md`

---

## Overview

| Area | Total | Done | Remaining |
|------|-------|------|-----------|
| Code improvements (original list) | 60 | 53 | 7 |
| Documentation improvements | 20 | 20 | 0 |
| Additional enhancements | 57 | 57 | 0 |
| Hydrata Phases 0–4 | 21 | 21 | 0 |
| GPU Phase 1–3 | 15 | 15 | 0 |
| GPU Phase 4 — SC26 paper | 3 | 0 | 3 |
| Riverwall throughflow | 6 | 6 | 0 |
| Quantity memory reduction | 7 | 7 | 0 |
| Domain memory reduction | 3 | 3 | 0 |
| Benchmark suite | 2 | 2 | 0 |
| Bug fixes | 7 | 7 | 0 |
| Kinematic viscosity parallelisation | 4 | 4 | 0 |
| **Total** | **205** | **197** | **8** |

---

## Remaining / Deferred Items

### Code improvements

- [ ] **3.3** `anuga/scenario/` tests — deferred; depends on compiled `spatialInputUtil` and real test data
- [ ] **4.1** Reduce parameter counts (`gauge.py`, `generic_domain.py`, `boyd_box_operator.py`) — deferred
- [ ] **5.2** `polygon.intersection()` — not a confirmed hotspot; deferred
- [ ] **5.x** `util.py:301` (`csv2timeseries_graphs`) — dominated by matplotlib; deferred
- [ ] **1.3** `anuga/file/urs.py:29` — intentionally skipped: file handle stored as `self.mux_file` for iterator lifecycle

### GPU Phase 4 — SC26 paper preparation (months 4–6)

- [ ] **G4.1** Gordon Bell metrics — per-kernel timing (not just totals), roofline model comparison, peak theoretical FLOP/s
- [ ] **G4.2** Physical benchmark validation — Thacker paraboloid, dam break (Ritter), tide gauge comparison in GPU mode
- [ ] **G4.3** Multi-node strong scaling — 20 M triangles, 1→64 GPUs; demonstrate ~50× runtime reduction

---

## Remaining Work (priority order)

### Short term — SC26 prerequisites (needs GPU hardware)
1. **G2.1** GPU benchmark suite (actual GPU runs — `benchmarks/run_gpu_benchmarks.py` is ready)
2. **G2.4** Weak scaling experiment (1→64 GPUs on HPC cluster — scripts in `scripts/hpc/`)
3. **G4.1** Gordon Bell metrics — per-kernel timing, roofline model comparison
4. **G4.2** Physical benchmark validation — Thacker paraboloid, dam break (Ritter) in GPU mode
5. **G4.3** Multi-node strong scaling — 20 M triangles, 1→64 GPUs

### Medium effort — see `claude/FUTURE_WORK.md`

Top P2 recommendations: P2.6 (fast-suite coverage), P2.7 (gauge module modernisation), P2.4 (culvert compute_rates deduplication).

### Long-term / opportunistic
- **H4.1** Modernise test patterns — convert `unittest.TestCase` to plain pytest functions incrementally when files are touched. Not worth a dedicated pass.
- **Coverage** — Full suite ~58%, fast suite ~55%. `fail_under=57` in `.coveragerc`. Add tests opportunistically when touching files.
- **Local-timestepping** — `compute_flux_update_frequency` is a `pass` stub. When implementing, allocate work arrays on demand in `set_local_time_stepping()`. See P3.1 in `FUTURE_WORK.md` for GPU-compatible redesign notes.
