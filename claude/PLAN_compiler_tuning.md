# Plan — multi-compiler support (GCC / Intel ICX / NVHPC) with per-kernel performance parity

Created: 2026-06-23. Status: **Phase 0, Phase 1, Phase 2, and Phase 3 fully done.**
See `claude/KNOWN_ISSUES.md` for full per-phase findings.

## Goal

Confirm ANUGA builds and runs correctly with **GCC**, **Intel ICX**, and **NVHPC (nvc)**,
then tune each hot kernel per compiler so that no compiler is measurably slower than the
others on the same hardware (CPU multicore path) — i.e. compiler choice should not be a
performance regression vector. GPU-offload (NVHPC `gpu_offload=true`) is a separate
axis, evaluated against itself across driver/SDK versions, not against the CPU builds.

## Current state (verified 2026-06-23)

- `meson.build` already detects compiler via `effective_compiler_id` (handles Cray
  wrappers too) and branches flags for `gcc`, `intel`/`intel-cl`, `intel-llvm`/`icx`,
  `nvidia_hpc`, `clang` — both for mode-1 OpenMP (`anuga/shallow_water/meson.build:166-193`)
  and the unified mode-2 kernels (same file, GPU-offload and CPU-multicore branches,
  `:166-236`). Intel MKL auto-detected for Intel compilers.
- Conda envs exist for Intel per Python version: `environments/environment_3.{8..14}_intel.yml`.
- GCC is the documented/tested default (CI, all benchmark baselines in `benchmarks/`).
- NVHPC GPU-offload build is documented and verified working (`claude/KNOWN_ISSUES.md`)
  — 56/56 `test_DE_gpu_omp.py` pass on an RTX 5070. Two known NVHPC-specific issues are
  already root-caused there (host OpenMP fallback is single-threaded; present-table
  test-isolation abort, worked around by `anuga_run_isolated_tests`).
- **Not yet verified**: an actual ICX build + full test pass on this checkout. Flags
  exist in meson.build but no session log records a successful ICX build/test run.
- **Not yet verified**: NVHPC **CPU-multicore** (`gpu_offload=false`) build benchmarked
  against GCC — only the GPU-offload path has been validated.
- **No per-kernel benchmarking exists at all.** `benchmarks/run_benchmarks.py` measures
  whole-evolve throughput (cells/s) only. `claude/C_EXTENSION_AUDIT_TODOS.md` (P3) already
  flags this gap ("Per-kernel microbenchmarks ... so kernel-level regressions are
  attributable") — this plan's Phase 1 implements that TODO as a prerequisite.

## Toolchains available on this machine (verified 2026-06-23)

| Compiler | Location | Notes |
|----------|----------|-------|
| GCC | `module load gcc/11.2.0` (also system `/usr/bin/gcc`) | baseline |
| Intel ICX | `~/intel/oneapi/compiler/2025.1/bin/icx` (oneAPI, sourced manually) | 2025.1.0; old `hpc/compiler/intel/2018,2019` modules are classic icc, not relevant |
| NVHPC | `source ~/source/samir/spack/share/spack/setup-env.sh && spack load nvhpc /4jo2stp` → nvhpc@26.3 | managed via spack (spack hash `4jo2stp`); `run_compiler_matrix.sh` uses `SPACK_SETUP`/`NVHPC_SPACK_HASH` variables for this — **no GPU on this host**; only CPU-multicore (`gpu_offload=false`) path tested here |
| Conda | `module load miniconda` or `~/miniconda3`/`~/miniforge3` | use to create per-compiler env from `environments/environment_3.14*.yml` |

---

## Phase 0 — Baseline verification (do first, cheap) — ✅ DONE 2026-06-23

1. ✅ No GPU on this host (`nvidia-smi` not found) — NVHPC tested CPU-multicore
   (`gpu_offload=false`) only here.
2. ✅ GCC 11.2.0 (`module load gcc/11.2.0`), conda env `anuga_phase0_gcc`
   (from `environments/environment_3.13.yml`), default build dir `build/cp313`:
   2667 passed, 214 skipped, 0 failed, 56.11 s (`--run-fast`).
3. ✅ Intel ICX 2025.1.0, conda env `anuga_phase0_icx`
   (from `environments/environment_3.13_intel.yml`), build dir `build/cp313-icx`
   (explicit `-Cbuild-dir=` needed — same Python ABI tag as the GCC env would otherwise
   collide on the default `build/cp313`): 2667 passed, 214 skipped, 0 failed, 45.39 s.
   One cosmetic finding (`-fopenmp`/`-qopenmp` warning from meson's built-in
   `dependency('openmp')` — does not affect correctness) — see KNOWN_ISSUES.
4. ✅ NVHPC 24.11 (`module load hpc_sdk/nvhpc/24.11`), reused the `anuga_phase0_gcc`
   conda env (same deps, different `CC`/build dir — **note: this overwrote that env's
   editable-install pointer from the GCC build to the NVHPC build**, harmless since GCC's
   numbers were already captured first), build dir `build/cp313-nvc-cpu`:
   2667 passed, 214 skipped, 0 failed, 42.20 s.
5. ✅ Findings written to `claude/KNOWN_ISSUES.md` → "Multi-compiler build verification".

**Conclusion: no correctness divergence across GCC/ICX/NVHPC-CPU on this checkout
(`905cb1f2`).** The fast-suite wall-time numbers above are smoke timings on a shared node,
not controlled per-kernel measurements — Phase 1 builds the real benchmark harness before
any tuning claims are made from timing.

Three persistent conda envs now exist on this host for reuse in later phases:
`anuga_phase0_gcc` (currently configured for the NVHPC build — see note above, rebuild
with `CC=gcc` if a clean GCC env is needed again) and `anuga_phase0_icx`.

## Phase 1 — Per-kernel benchmark harness (prerequisite for Phase 3) — ✅ DONE 2026-06-23

1. ✅ `benchmarks/run_kernel_benchmarks.py` — times `compute_fluxes`,
   `protect_against_infinitesimal_and_negative_heights`,
   `distribute_to_vertices_and_edges`, `extrapolate_second_order_edge_sw` (edge-only,
   `distribute_to_vertices=False`), `manning_friction_semi_implicit` individually. Each
   domain is primed with a short evolve (`finaltime=5.0`) to reach a non-trivial wet/dry
   mixed state, then the kernel is called `--reps` times directly (no time advance) so
   every repetition's input is identical — isolates kernel cost from the timestep
   controller. `--modes 1,2 --size {small,medium,large} --kernels ... --reps --warmup`.
   `gravity`/`gravity_wb` were **not** added — not reachable from the current DE0 step
   loop in a way that's worth a separate harness entry; revisit if a future kernel audit
   needs them specifically.
2. ✅ `benchmarks/compare_kernel_benchmarks.py` — separate from `compare_benchmarks.py`
   (deliberately not unified: the result schemas differ enough — `mean_us`/`cells_per_s`
   keyed by kernel+mode vs `wall_time_s`/`peak_rss_mb` keyed by scenario name — that one
   script branching on both would be messier than two small ones). Output JSON saved as
   `benchmarks/results/kernels_<branch>_<commit>_<timestamp>.json` (or `--output` override).
3. ✅ Ran the baseline across all three Phase 0 compilers — see
   `claude/KNOWN_ISSUES.md` → "Phase 1 — per-kernel benchmark harness" for the full table.
   **Headline finding: ICX is 2-3x slower than GCC on `distribute`,
   `extrapolate_edge_only`, and `manning_friction_flat` specifically** (all three live in
   the "older OpenMP loops" `sw_domain_openmp.c`/`sw_domain_openmp_ext.pyx` code already
   flagged in `C_EXTENSION_AUDIT_TODOS.md` P3), while being faster than GCC on
   `compute_fluxes`/`protect`. NVHPC is faster than GCC on *every* kernel with no such
   regression — confirms this is ICX-specific + kernel-specific, not a generic
   "non-GCC is slower" story. This is Phase 3's first concrete target.

## Phase 2 — Build matrix automation — ✅ DONE 2026-06-28

1. ✅ Script written: `benchmarks/run_compiler_matrix.sh` — builds GCC / ICX / NVHPC
   (CPU-multicore) into separate build dirs, runs Phase 1's per-kernel harness + fast
   test suite per compiler, tags output files by compiler. Flags: `--compilers gcc,icx,nvhpc`,
   `--skip-build`, `--skip-tests`. Sources `~/intel/oneapi/setvars.sh` for ICX; uses
   spack (`SPACK_SETUP` / `NVHPC_SPACK_HASH=4jo2stp` → nvhpc@26.3) for NVHPC.
2. ✅ Run end-to-end at `67c6c6c0`: all three compilers 2667/214/0. Script also detects
   nvhpc version changes and auto-reconfigures the build dir (triggered on first run:
   24.11-0 → 26.3-0).

## Phase 3 — Tuning loop (the actual "no performance deviation" work)

### First target: eliminate ICX regressions — ✅ DONE 2026-06-27

Threshold agreed: within 10% of GCC baseline per kernel.

ICX 2-3x regression on `distribute`, `extrapolate_edge_only`, `manning_friction` was
root-caused to `#pragma omp parallel for simd` generating poor SIMD code on stride-3
scatter/gather loops (not the `shared(D)` struct-dereference hypothesis from Phase 1).

**Five changes made — see `claude/KNOWN_ISSUES.md` → "Phase 3" for full details:**
1. `gpu_omp_macros.h`: `OMP_PARALLEL_LOOP` → plain `parallel for` for ICX via `__INTEL_LLVM_COMPILER`
2. `gpu_device_helpers.h`: `static const double GPU_TINY` → `#define GPU_TINY`
3. `core_kernels.c`: three friction loops → `OMP_PARALLEL_LOOP_SIMD` + `h*h*cbrt(h)`
4. `meson.build`: ICX `openmp_c_args` += `-xHost`
5. `anuga/shallow_water/meson.build`: ICX `gpu_c_args` += `-xHost`

**Result (medium mesh, OMP_NUM_THREADS=4):** ICX now faster than GCC on every kernel:

| Kernel | GCC | ICX (after) | Delta |
|--------|-----|-------------|-------|
| compute_fluxes | 3147 us | 1679 us | -46.6% |
| distribute | 3648 us | 1587 us | -56.5% (was +201%) |
| extrapolate_edge_only | 3377 us | 1223 us | -63.8% (was +181%) |
| manning_friction_flat | 46.86 us | 23.23 us | -50.4% (was +100%) |
| protect | 27.63 us | 13.86 us | -49.8% |

GCC and ICX: 2667 passed, 214 skipped, 0 failed.

### Second target: NVHPC-CPU re-benchmark after cbrt change — ✅ DONE 2026-06-27 (re-benchmarked with nvc 26.3 on 2026-06-28)

NVHPC improved on every kernel vs the Phase 1 baseline — no regressions (result saved as
`benchmarks/results/kernels_nvhpc_cpu_phase3.json`):

| Kernel | mode | Phase 1 | Phase 3 | Delta |
|--------|------|------:|------:|------:|
| compute_fluxes | 1 | 2206 us | 1884 us | -14.6% |
| distribute | 1 | 2472 us | 1835 us | -25.8% |
| extrapolate_edge_only | 1 | 2431 us | 1522 us | -37.4% |
| manning_friction_flat | 1 | 42.0 us | 28.45 us | -32.3% |
| protect | 1 | 26.8 us | 20.43 us | -23.7% |

The cbrt + OMP_PARALLEL_LOOP_SIMD changes benefited NVHPC's vectoriser just as they did
for GCC/ICX. NVHPC is faster than GCC on every kernel (40–57% on hot kernels).
Tests: 2667/214/0.

### Third target: GCC `-march=native` — ✅ DONE 2026-06-28

GCC was the only compiler without a native-ISA flag (ICX has `-xHost`, NVHPC auto-targets
native). Added `-march=native` to GCC's `openmp_c_args` (root `meson.build`) and
`gpu_c_args` (`anuga/shallow_water/meson.build`).

Results (medium mesh, OMP_NUM_THREADS=4, `develop @ 67c6c6c0`):

| Kernel | GCC no flag | GCC `-march=native` | Δ |
|--------|------------|---------------------|---|
| compute_fluxes | 3141 us | 2952 us | −6% |
| distribute | 3891 us | 3596 us | −8% |
| extrapolate_edge_only | 3369 us | 3278 us | −3% |
| manning_friction_flat | 46 us | 47 us | flat |
| protect (m1) | 28 us | 22 us | **−19%** |

Tests: 2667/214/0. Modest gains across the board; `protect` is the clearest win at −19%.
The large ICX–GCC gap (44–63% on hot kernels) persists — GCC 11's auto-vectoriser does not
generate SIMD for stride-3 loops or vectorise `cbrt` via libmvec. Further GCC tuning
would require explicit intrinsics or loop restructuring. See `claude/KNOWN_ISSUES.md` for
the full table and analysis.

### Notes on structural gaps

- NVHPC host fallback (`-mp=gpu,multicore` with no GPU) is single-threaded by design —
  documented as a known NVHPC limitation in `claude/KNOWN_ISSUES.md`. No fix pursued.
- The `-fopenmp`/`-fiopenmp` cosmetic ICX warning from meson's `dependency('openmp')`
  (found in Phase 0) is still present and still harmless — not yet fixed.
- GCC 11 libmvec lacks vectorised `cbrt` (or doesn't engage it for this loop shape);
  `manning_friction_flat` shows no improvement from `-march=native`. A newer GCC (12+)
  may improve this — not yet tested.

## Phase 4 — CI / process integration

1. Add (or document, if CI runner resources don't allow it) a compile-check + fast-suite
   job per compiler. Full per-kernel benchmarking is noisy on shared CI runners — keep
   that manual/on this HPC host, not in CI.
2. Update `CLAUDE.md` "Build System" section with a compiler support matrix
   (compiler → status → flags → caveats), replacing scattered mentions.
3. Mark the P3 "Per-kernel microbenchmarks" TODO in `C_EXTENSION_AUDIT_TODOS.md` done
   once Phase 1 lands.

---

## Open questions for the user (ask before Phase 3 work starts)

- What performance-parity threshold counts as "no deviation"? (proposed 10%, see Phase 3.1)
- Is GPU-offload performance in scope for this plan, or strictly the CPU-multicore path?
  (NVHPC GPU-offload already has its own validated baseline — see KNOWN_ISSUES.)
- Priority order: is Intel ICX or NVHPC-CPU the more urgent gap to close first?
