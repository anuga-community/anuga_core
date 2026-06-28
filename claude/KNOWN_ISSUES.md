# Known Issues and Gotchas

Things discovered during development sessions that are surprising, non-obvious,
or require caution when working in specific areas.

---

## Build

### Multi-compiler build verification (2026-06-23, Phase 0 of `claude/PLAN_compiler_tuning.md`)

Verified on a shared HPC node (no GPU present — `nvidia-smi` not found; NVHPC tested
CPU-multicore only here): `git rev-parse HEAD` = `905cb1f2` (`develop`). All three builds
used `-Dgpu_offload=false`, isolated build dirs (`-Cbuild-dir=build/cp313-<tag>`, required
because same-Python-version envs share the default `build/cp<abi>` dir and would clobber
each other), `OMP_NUM_THREADS=4`, `pytest --pyargs anuga --run-fast`:

| Compiler | Version | Build | Fast suite | Wall |
|----------|---------|-------|-------------|------|
| GCC | 11.2.0 (`module load gcc/11.2.0`) | clean | 2667 passed, 214 skipped, 0 failed | 56.11 s |
| Intel ICX | 2025.1.0 (`~/intel/oneapi`) | clean | 2667 passed, 214 skipped, 0 failed | 45.39 s |
| NVHPC (nvc) | 24.11 (`module load hpc_sdk/nvhpc/24.11`) | clean | 2667 passed, 214 skipped, 0 failed | 42.20 s |

Identical pass/skip counts across all three — no correctness divergence found. Wall times
are fast-suite smoke timings on a shared/possibly-noisy node, **not** the controlled
per-kernel benchmarks Phase 1 of the plan will add; do not read anything into the ranking
yet.

**Finding — cosmetic ICX warning, not a failure:** every icx compile of an OpenMP-bearing
file emits `icx: warning: use of '-qopenmp' recommended over '-fopenmp'
[-Wrecommended-option]`, even though both `meson.build` (`openmp_c_args`) and
`anuga/shallow_water/meson.build` (`gpu_c_args`) correctly branch `effective_compiler_id
in ['intel-llvm', 'icx']` to `-fiopenmp` (confirmed via `-fiopenmp` never appearing and
`-fopenmp` appearing on *every* OpenMP file in the build log). The extra `-fopenmp` comes
from meson's own `dependency('openmp', required: false)` object, whose compile_args are
appended unconditionally via `openmp_deps`/`gpu_deps` regardless of which manual
`-fiopenmp`/`-qopenmp` flag was also added — meson's built-in OpenMP detection treats icx
as clang-family and defaults to `-fopenmp`. Harmless (icx accepts the flag, build and
tests pass) but worth fixing during Phase 3 tuning: either suppress the warning
specifically for intel-llvm, or stop relying on `dependency('openmp')`'s own flags for
that branch and pass `required: false` purely for its `-l`/link bits while keeping
compile flags fully manual.

Conclusion: GCC, ICX, and NVHPC-CPU all build and pass cleanly on this checkout. Phase 0
is done.

### Phase 1 — per-kernel benchmark harness, and a real ICX performance finding (2026-06-23)

Added `benchmarks/run_kernel_benchmarks.py` (times `compute_fluxes`,
`protect_against_infinitesimal_and_negative_heights`, `distribute_to_vertices_and_edges`,
`extrapolate_second_order_edge_sw`, `manning_friction_semi_implicit` individually on a
primed, fixed domain state — no evolve loop in the timed region) and
`benchmarks/compare_kernel_benchmarks.py` (delta table keyed by kernel+mode). Implements
the `C_EXTENSION_AUDIT_TODOS.md` P3 "Per-kernel microbenchmarks" TODO.

Baseline run (medium mesh, 90 000 tris, `OMP_NUM_THREADS=4`, 50 reps, `develop`@`905cb1f2`,
results saved as `benchmarks/results/kernels_{gcc,icx,nvhpc_cpu}.json`):

| Kernel | mode | GCC 11.2.0 | ICX 2025.1.0 | NVHPC 24.11 (CPU) |
|---|---|---:|---:|---:|
| compute_fluxes | 1 | 3081 us | 2692 us (**-13%**) | 2206 us (**-28%**) |
| compute_fluxes | 2 | 3210 us | 2718 us (**-15%**) | 2464 us (**-23%**) |
| protect | 1 | 29.6 us | 21.7 us (**-27%**) | 26.8 us (-10%) |
| protect | 2 | 88.4 us | 33.0 us (**-63%**) | 35.1 us (**-60%**) |
| distribute | 1 | 3760 us | **11321 us (+201%)** | 2472 us (-34%) |
| distribute | 2 | 3861 us | **11068 us (+187%)** | 2160 us (-44%) |
| extrapolate_edge_only | 1 | 3819 us | **10719 us (+181%)** | 2431 us (-36%) |
| manning_friction_flat | 1 | 58.2 us | **116.5 us (+100%)** | 42.0 us (-28%) |

**Finding: ICX is 2–3x *slower* than GCC on `distribute`, `extrapolate_edge_only`, and
`manning_friction_flat` specifically**, while being faster than GCC on `compute_fluxes`
and `protect`. NVHPC is faster than GCC on every kernel with no such regression — so
this is not "non-GCC compilers are slower," it is specific to ICX and to this subset of
kernels. All three regressing functions live in `sw_domain_openmp_ext.pyx` /
`sw_domain_openmp.c` and are exactly the "older OpenMP loops" already flagged in
`C_EXTENSION_AUDIT_TODOS.md` P3 ("False sharing / shared(D) in older OpenMP loops ...
dereference the domain struct inside the loop" — `extrapolate`/`protect`/friction are in
that older code, `compute_fluxes` and the unified-kernel `protect`/`distribute` mode-2
paths go through the newer `core_kernels.c` pattern that hoists `restrict` pointers).
**Working hypothesis for Phase 3:** ICX's vectorizer/inliner handles the
`shared(D)`-with-struct-dereference-in-loop pattern far worse than gcc's and nvc's; the
fix is likely the same one already planned for that TODO (hoist hot pointers to locals)
rather than a new ICX-specific flag — re-run this benchmark after that refactor to confirm
it closes the gap on all three compilers, not just ICX.

Not yet investigated (per original Phase 1 hypothesis): `-qopt-report=2` vectorization
report for icx on these three files. The root cause turned out to be the combined
`parallel for simd` directive on stride-3 scatter / branch-heavy loops, not struct
dereferencing as originally hypothesised — fixed in Phase 3 without needing the report.

### Phase 3 — ICX regression fixed; ICX now faster than GCC on all hot kernels (2026-06-27)

**Root cause:** ICX generates inefficient SIMD code for `#pragma omp parallel for simd`
(`OMP_PARALLEL_LOOP` macro) on stride-3 scatter loops (`distribute`), complex
gather+branch loops (`extrapolate`), and math-heavy loops (`manning_friction`). The
combined directive forces a SIMD trip count that ICX can't vectorise well on those loop
shapes, causing the 2–3x regression found in Phase 1.

**Five code changes** to eliminate all ICX regressions (all files already existed — no
new files):

1. **`anuga/shallow_water/gpu/gpu_omp_macros.h`** — ICX now gets plain
   `#pragma omp parallel for` (no simd) from `OMP_PARALLEL_LOOP` via
   `#ifdef __INTEL_LLVM_COMPILER`. ICX's own auto-vectoriser then applies per-loop and
   produces faster code on the scatter/gather patterns. GCC/NVHPC are unchanged.

2. **`anuga/shallow_water/gpu/gpu_device_helpers.h`** — Changed `GPU_TINY` from
   `static const double GPU_TINY = 1.0e-100` to `#define GPU_TINY 1.0e-100`. GCC's
   OpenMP SIMD vectoriser inside `#pragma omp declare target` generated an external
   symbol reference (`U GPU_TINY`) for the constant, causing `ImportError: undefined
   symbol: GPU_TINY` at import after the `core_kernels.c` changes below activated more
   SIMD loops. Using `#define` inlines the literal everywhere.

3. **`anuga/shallow_water/gpu/core_kernels.c`** — Three Manning friction functions
   (`core_manning_friction_flat_semi_implicit`, `core_manning_friction_sloped_semi_implicit`,
   `core_manning_friction_sloped_semi_implicit_edge_based`) changed from `OMP_PARALLEL_LOOP`
   to `OMP_PARALLEL_LOOP_SIMD` (always-simd, defined unconditionally) and from
   `pow(h, 7.0/3.0)` to `h * h * cbrt(h)`. The `simd` hint forces ICX into its SVML
   cbrt path; `cbrt` is vectorisable while `pow` with a runtime-generic exponent is not.
   `seven_thirds` constants removed. **Side-effect: GCC manning improved too**
   (58 us → 47 us) since GCC also vectorises cbrt better.

4. **`meson.build` (root)** — Added `-xHost` to ICX `openmp_c_args`
   (`['-O3', '-fiopenmp', '-xHost', '-g']`). Without `-xHost`, ICX targets a generic ISA
   and falls back to scalar `cbrt()` even with the `simd` hint — AVX-512/AVX-2 SVML is
   only engaged when the target ISA is explicitly native.

5. **`anuga/shallow_water/meson.build`** — Added `-xHost` to ICX `gpu_c_args`
   (`['-O3', '-fiopenmp', '-xHost', '-g']`). Mode-2 (`sw_domain_gpu_ext`) also compiles
   `core_kernels.c` with its own flags; without matching `-xHost`, mode-1 and mode-2 get
   different ISA for `cbrt`, breaking the `atol=1e-12` mode-agreement tests in
   `test_DE_gpu_omp.py::Test_GPU_NonGPUBoundaryFallback` (3 failures). Matching `-xHost`
   in both modes keeps results bit-identical under AVX-512.

**Final benchmark results** (medium mesh ~90k tris, `OMP_NUM_THREADS=4`, `develop` @
post-phase-3, `kernels_icx_phase3_done.json` vs `kernels_gcc_final.json`):

| Kernel | mode | GCC 11.2.0 | ICX 2025.1.0 | Delta |
|--------|------|--------:|--------:|------:|
| compute_fluxes | 1 | 3147 us | 1679 us | **ICX -46.6%** |
| distribute | 1 | 3648 us | 1587 us | **ICX -56.5%** (was +201%) |
| extrapolate_edge_only | 1 | 3377 us | 1223 us | **ICX -63.8%** (was +181%) |
| manning_friction_flat | 1 | 46.86 us | 23.23 us | **ICX -50.4%** (was +100%) |
| protect | 1 | 27.63 us | 13.86 us | **ICX -49.8%** (was -27%) |

ICX is now faster than GCC on every benchmarked kernel. Test suites: GCC 2667/214/0,
ICX 2667/214/0 — no correctness divergence.

**Two failure patterns hit during Phase 3 (both fixed):**
- `ImportError: undefined symbol: GPU_TINY` — see change 2 above.
- 3 mode-agreement test failures after adding `-xHost` to mode-1 only — see change 5 above.

**NVHPC-CPU re-benchmark** (`benchmarks/results/kernels_nvhpc_cpu_phase3.json`,
`OMP_NUM_THREADS=4`, `develop @ e0672897`): no regressions — every kernel improved vs the
Phase 1 baseline. The `cbrt` + `OMP_PARALLEL_LOOP_SIMD` changes also benefited NVHPC's
vectoriser:

| Kernel | mode | Phase 1 | Phase 3 | Delta |
|--------|------|------:|------:|------:|
| compute_fluxes | 1 | 2206 us | 1884 us | -14.6% |
| distribute | 1 | 2472 us | 1835 us | -25.8% |
| extrapolate_edge_only | 1 | 2431 us | 1522 us | -37.4% |
| manning_friction_flat | 1 | 42.0 us | 28.45 us | -32.3% |
| protect | 1 | 26.8 us | 20.43 us | -23.7% |

NVHPC is now 40–57% faster than GCC on hot kernels. Tests: 2667/214/0.

### GCC `-march=native` tuning — modest gains, large gap to ICX remains (2026-06-28)

After Phase 3 closed the ICX regression, GCC was the only compiler without a native-ISA
flag. ICX uses `-xHost`; NVHPC auto-targets native. GCC was compiled with only `-O3
-fopenmp` — no AVX-512, no libmvec. Adding `-march=native` to both meson.build files
(root `openmp_c_args` for mode-1, `shallow_water/meson.build` `gpu_c_args` for mode-2)
gave modest but real improvements on `develop @ 67c6c6c0` (medium mesh, OMP_NUM_THREADS=4):

| Kernel | GCC no `-march` | GCC `-march=native` | Δ |
|--------|----------------|---------------------|---|
| compute_fluxes (m1) | 3141 us | 2952 us | −6% |
| compute_fluxes (m2) | 3111 us | 2942 us | −5% |
| distribute (m1) | 3891 us | 3596 us | −8% |
| distribute (m2) | 3616 us | 3599 us | −0.5% (noise) |
| extrapolate_edge_only | 3369 us | 3278 us | −3% |
| manning_friction_flat | 46 us | 47 us | +1% (noise) |
| protect (m1) | 28 us | 22 us | **−19%** |
| protect (m2) | 72 us | 71 us | −1% (noise) |

Tests: 2667/214/0 — no regressions.

**Remaining gap to ICX after `-march=native`:**

| Kernel | GCC+native | ICX | Gap |
|--------|-----------|-----|-----|
| compute_fluxes | 2952 us | 1645 us | −44% |
| distribute | 3596 us | 1551 us | −57% |
| extrapolate_edge_only | 3278 us | 1226 us | −63% |
| manning_friction_flat | 47 us | 19 us | −59% |
| protect (m1) | 22 us | 15 us | −34% |

**Why the gap persists:** GCC 11's auto-vectoriser does not generate SIMD code for the
stride-3 scatter/gather loops (`distribute`, `extrapolate`) or transcendental-heavy friction
loops (`manning_friction_flat`) even with AVX-512 enabled. The `cbrt` case is
particularly illustrative: ICX with `-xHost` drops from 47 µs to 19 µs by using vectorised
SVML `cbrt`; GCC 11 with `-march=native` stays at 47 µs — GCC 11's libmvec either lacks a
vectorised `cbrt` or doesn't engage it for this loop shape. Further GCC tuning would require
explicit SIMD intrinsics or loop restructuring, not compiler flags.

### GCC 15 / NVHPC 26.3 / ICX cross-compiler flag tuning — parity achieved on hot kernels (2026-06-28)

Following the `-march=native` result (GCC 11, modest gains only), a second pass explored
GCC 15.1.0 (spack `di5tvfm`) and additional flags for all three compilers.

**GCC 15 baseline vs GCC 15 with tuning flags:**

GCC 15 with only `-march=native` was essentially identical to GCC 11 on hot kernels and
had a +28% regression on `protect` mode-1. Adding the full math+vectorisation flag set
(`-fno-math-errno -fno-trapping-math -ffinite-math-only -fassociative-math
-fvect-cost-model=unlimited -funroll-loops`) halved compute_fluxes/distribute/extrapolate
but left `protect` mode-1 still regressed. Switching the mode-1 cost model to
`-fvect-cost-model=cheap` partially recovered protect (−2% vs previous, still +28% vs
GCC 11) while keeping most of the hot-kernel gains.

**Flags that were tried and reverted:**

- **ICX `-Ofast`** (adds `-no-prec-div` / `fp-model=fast=2`): `manning_friction_flat`
  regressed +13% — the faster-but-less-accurate cbrt path is slower in this loop shape.
  Reverted to `-O3 -fiopenmp -xHost`.
- **NVHPC `-Mvect=simd` in mode-1**: `protect` mode-1 regressed +15% (force-vectorising
  the short branch-heavy loop hurts). Kept `-Mvect=simd` only for mode-2 (no regression
  there); mode-1 uses `-Mfprelaxed -Munroll` only.

**Final flags committed (`797ad4c8`, `develop @ 2d6e7c07`):**

| Compiler | Mode-1 (`openmp_c_args`) | Mode-2 (`gpu_c_args`) |
|----------|--------------------------|-----------------------|
| GCC ≥12 | `-O3 -fopenmp -foffload=disable -march=native -fno-math-errno -fno-trapping-math -ffinite-math-only -fassociative-math -fvect-cost-model=cheap -funroll-loops` | same but `fvect-cost-model=unlimited` |
| NVHPC | `-O3 -mp=gpu,multicore -Mfprelaxed -Munroll` | `-O3 -mp=multicore -Mfprelaxed -Mvect=simd -Munroll` |
| ICX | `-O3 -fiopenmp -xHost` (unchanged) | same |

**Final benchmark results (medium mesh, OMP_NUM_THREADS=4, GCC 11 as reference):**

| Kernel | m | GCC 11 | GCC 15 tuned | NVHPC 26.3 | ICX 2025.1 | spread (3 tuned) |
|--------|---|--------|-------------|------------|-----------|-----------------|
| compute_fluxes | 1 | 2952 µs | 1563 µs | 1627 µs | 1658 µs | +6% |
| compute_fluxes | 2 | 2942 µs | 1544 µs | 1590 µs | 1656 µs | +7% |
| distribute | 1 | 3596 µs | 1486 µs | 1482 µs | 1556 µs | +5% |
| distribute | 2 | 3599 µs | 1460 µs | 1459 µs | 1566 µs | +7% |
| extrapolate_edge_only | 1 | 3278 µs | 1159 µs | 1166 µs | 1228 µs | +6% |
| manning_friction_flat | 1 | 47 µs | 27 µs | 29 µs | **19 µs** | +51% |
| protect | 1 | 22 µs | 28 µs | 24 µs | **15 µs** | +84% |
| protect | 2 | 71 µs | 38 µs | 31 µs | **23 µs** | +66% |

Tests: all three compilers 2667/214/0.

**Hot kernels (compute_fluxes, distribute, extrapolate): all three tuned compilers within
5–7% of each other — the cross-compiler parity goal is met for these kernels.**

**Remaining gaps on `manning` and `protect`:** ICX is structurally faster here because:
- `manning_friction_flat`: ICX uses vectorised SVML `cbrt` via `-xHost`; GCC 15 libmvec
  cbrt path is slower; NVHPC's vectorised cbrt similarly limited.
- `protect`: ICX's branch predictor and SIMD handling of the wet/dry threshold loop
  outperform GCC 15 and NVHPC. The loop is short and branch-heavy; wider SIMD hurts
  both (GCC 15 regression with `unlimited`, NVHPC with `-Mvect=simd`).
  Closing these gaps requires loop restructuring or explicit SIMD intrinsics, not flags.

**GCC 11 vs GCC 15 tuned overall:** GCC 15 with tuning is 45–62% faster than GCC 11 on
hot kernels, and its `protect` mode-1 regression (+28% vs GCC 11) is a GCC 15 codegen
issue unrelated to the tuning flags — present both with `unlimited` and `cheap` cost model.
GCC 11 remains preferable for `protect` mode-1 if that kernel is critical.

### Building with GPU offloading (NVIDIA HPC SDK / nvc)

ANUGA's GPU extension (`sw_domain_gpu_ext`, `multiprocessor_mode=2`) requires
`nvc` from the NVIDIA HPC SDK — GCC 15's nvptx backend ICEs on `core_kernels.c`
(ompdevlow GIMPLE pass segfault in `core_extrapolate_second_order_edge`).

**One-time setup (Ubuntu, requires sudo):**
```bash
# Add NVIDIA HPC SDK apt repo
curl -fsSL https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' \
  | sudo tee /etc/apt/sources.list.d/nvhpc.list
sudo apt-get update -y && sudo apt-get install -y nvhpc   # ~5 GB
```

**GPU build (RTX 5070 = Blackwell cc120; adjust gpu_arch for other GPUs):**
```bash
NVC=/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin/nvc
conda run -n anuga_env_3.14 bash -c "CC=$NVC pip install --no-build-isolation -v -e . \
  -Csetup-args=-Dgpu_offload=true \
  -Csetup-args=-Dgpu_arch=cc120"
```

Meson auto-detects nvc as `nvidia_hpc`; the build uses `-mp=gpu,multicore -gpu=cc120`.
The build dir must be clean if switching from a prior GCC build (`rm -rf build/cp314`).

**Verify GPU works:**
```bash
conda run -n anuga_env_3.14 pytest anuga/shallow_water/tests/test_DE_gpu_omp.py -v
```
All 56 tests pass on the RTX 5070.

**Switching back to CPU-only build:**
```bash
rm -rf build/cp314
conda run -n anuga_env_3.14 pip install --no-build-isolation -e .
```

### `--no-build-isolation` is recommended

`pip install --no-build-isolation -e .` is the recommended build approach.
It is not strictly required in all environments, but is preferred because
it ensures meson-python uses the Cython/numpy already installed in the conda
environment rather than fetching isolated build dependencies.

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

### GPU build: `test_DE_gpu_omp.py` aborts mid-file (NVHPC target present-table) (2026-06-15)

On a GPU build (`-Dgpu_offload=true`, nvc), running the whole
`anuga/shallow_water/tests/test_DE_gpu_omp.py` file in one process **aborts
silently** (exit 1, no traceback, not a SIGSEGV, GPU idle/no-OOM) partway
through — around the 9th–11th test (`Test_GPU_InletOperator::test_inlet_operator_basic`).
The NVHPC OpenMP-target runtime calls `exit()`.

**Not caused by the mode-2 session changes** — reproduces identically on the
pre-change commit (`d96ae357`) rebuilt with nvc.

**It is NOT a simple cumulative-resource leak.** Diagnostics:
- Each test class *alone*, and `test_inlet_operator_basic` alone, pass.
- Creating 16–20 GPU domains in a loop — both dropped each iteration *and* kept
  simultaneously live — works fine.
- The crash only appears with the file's specific mix of low-level kernel tests,
  `set_multiprocessor_mode(1)`↔`(2)` switching, `sync_to/from_device`, and
  operator setup, accumulated across ≥9 tests.
- Forcing finalization between tests (`gc.collect()`, or nulling
  `domain.gpu_interface`) makes it **worse**: introduces assertion *failures*
  before the abort.

**Root cause (diagnosed, not yet fixed):** device arrays are bound with
`#pragma omp target enter data map(to: host_ptr...)` keyed on the *host* pointer,
and released with `map(delete:)` (`gpu_domain_unmap_arrays` /
`gpu_domain_finalize`). The OpenMP present table is reference-counted and
host-pointer-keyed, so repeated map/unmap of arrays whose host addresses numpy
recycles across domains corrupts the table (stale entry reused, or a live
entry deleted) — hence both the "leak then abort" and the "eager-unmap then
assertion failure" signatures. Two reference cycles
(`domain ↔ gpu_interface`, and `gpu_dom → python_domain → domain`) also defer
`GPUDomain.__dealloc__`/`gpu_domain_finalize` to the cyclic GC, so finalization
timing is non-deterministic — but breaking either cycle does not fix the
underlying present-table issue (the other cycle still pins the domain, and eager
finalize corrupts).

It also crashes the *whole* `pytest --pyargs anuga.shallow_water` run (in either
`ANUGA_DEFAULT_COMPUTE_MODE`) because `test_DE_gpu_omp.py` collects early and
these GPU tests set mode 2 explicitly regardless of the env default — the abort
at ~3% kills the run before the rest of the suite executes.

**On a GPU build, `ANUGA_DEFAULT_COMPUTE_MODE=unified` over the full suite is not
viable even with the GPU file excluded:** every default domain then offloads to
the GPU, so the whole suite churns hundreds of mode-2 GPU domains in one process
and aborts early (~20%). This is the same root cause. Validate the unified
default over the full suite on the **gcc CPU build** (`-Dgpu_offload=false`),
where it is the documented 2657-passed run; on a GPU build, drive the bulk suite
with `ANUGA_DEFAULT_COMPUTE_MODE=legacy` and cover GPU paths via the per-class
runner below.

**Impact:** production use (a single, or a few sequential, mode-2 GPU domains)
is unaffected — single-domain evolve and each test class pass. Only running many
GPU-domain tests in *one* process trips it.

**Auto-skip:** on a GPU-offload build, `test_DE_gpu_omp.py` skips itself at
collection (module-level `pytest.skip`, gated on `anuga.gpu_offload_supported()`
and `not ANUGA_GPU_TESTS_ISOLATED`), so a normal `pytest --pyargs anuga` no longer
crashes — the file is reported as skipped with a message pointing here. On a CPU
build the guard is inert and the file runs in-process as usual.

**Workaround — run the GPU tests in isolated processes:**
```bash
# one fresh process per CLASS (fast):
bash anuga/shallow_water/tests/run_gpu_tests_isolated.sh
# one fresh process per TEST FUNCTION, with a per-test timeout (most robust;
# turns a genuine hang into a reported TIMEOUT). Works on any pytest target.
# Installed as `anuga_run_isolated_tests` (scripts/, via meson); in a source
# checkout run scripts/anuga_run_isolated_tests.py directly:
anuga_run_isolated_tests [TARGET] [--timeout S] [-k EXPR]
```
Both set `ANUGA_GPU_TESTS_ISOLATED=1` to bypass the auto-skip; all tests pass
this way (verified 65/65 per-function on the nvc build). Then run the rest of the
suite normally (it does not trip the issue) under the **legacy** default:
```bash
ANUGA_DEFAULT_COMPUTE_MODE=legacy \
  pytest anuga/shallow_water/tests/ --ignore=anuga/shallow_water/tests/test_DE_gpu_omp.py
```
**Do NOT use `pytest --forked`** for these tests: CUDA contexts are fork-unsafe,
so forking from a GPU-initialised parent poisons every child (it turns the abort
into ~53 spurious failures). Isolation must be *fresh* processes (separate
`python -m pytest` invocations), not `os.fork()`.

A real fix (FUTURE_WORK P1.10) needs either per-test fresh-process isolation
baked into the GPU test file, strict 1:1 map/unmap reference-count discipline per
domain, or device-pointer allocation (`omp_target_alloc` + `is_device_ptr`)
instead of host-pointer-keyed `map(to:)`. (`omp target enter/exit data` cleanup
is reference-counted and host-pointer-keyed; forcing finalization between tests
removes the abort but still yields ~7 aliasing failures, so clean teardown alone
is not sufficient.)

### GPU build: `anuga.shallow_water` is green under `unified` via the isolated runner (2026-06-17)

The per-function isolated runner now passes the **entire** `anuga.shallow_water`
set under the unified default on a GPU-offload build:

```bash
anuga_run_isolated_tests --pyargs anuga.shallow_water -cm unified
# 410 collected -> pass=408 skip=2 (2 skips are pre-existing legacy-default guards)
# (-cm/--compute-mode sets ANUGA_DEFAULT_COMPUTE_MODE for every child; omit to
#  inherit the environment.)
```

This works because each test runs in its own fresh process (no mode-2 domain
accumulation -> no NVHPC abort), **and** because 11 tests that probed mode-1-only
host state are now pinned to `legacy` (`domain.set_compute_mode('legacy')`):

- 9 white-box tests call `compute_forcing_terms()` / `compute_fluxes()` and assert
  on the host `semi_implicit_update` / `explicit_update` arrays, which mode-2 GPU
  computes on-device and never syncs back (so the host arrays read stale zeros) —
  in `test_forcing.py`, `test_friction.py`, `test_physics_sw.py` (Manning friction
  cases) and `test_data_manager.py::test_sww_extrema` (extrema monitoring).
- 2 numerical tests compare against legacy-recorded references and diverge at the
  ~1e-6 level under mode-2's different reduction/eval order
  (`test_regression_snapshots.py::test_dam_break_DE1_stage_snapshot` and
  `test_sww_interrogate.py::test_get_maximum_inundation_de0`). The two
  regression-snapshot domain helpers are pinned so that whole file stays
  deterministic under any `ANUGA_DEFAULT_COMPUTE_MODE`.

These are test-harness artifacts, not solver bugs; the pins are no-ops for the
distribution-default legacy path. Mode-2 numerical fidelity remains covered by the
mode1-vs-mode2 comparison tests in `test_DE_gpu_omp.py`. Note this complements —
does not replace — the guidance above: the *full* `pytest --pyargs anuga.shallow_water`
(non-isolated) under `unified` on a GPU build still aborts; use the isolated runner.
Commit `0c50947d`.

### Targeted `--cov=anuga.submodule` runs corrupt numpy's `_NoValue` sentinel

Running `pytest --cov=anuga.structures.structure_operator` (or any sub-package
path) causes test failures with:
```
TypeError: float() argument must be a string or a real number, not '_NoValueType'
```

**Root cause:** coverage.py calls `importlib.util.find_spec('anuga.structures.structure_operator')`
inside a `sys_modules_saved()` context (in `inorout.py`). This auto-imports parent
packages (including `anuga/__init__.py` → `shallow_water_domain.py` → numpy),
then purges all newly-imported modules from `sys.modules`. The subsequent real
import re-executes `numpy/__init__.py`. Since numpy's C extension (`_multiarray_umath`)
was already initialized, the reload guard fires and a new `_NoValue` singleton is
created — but C extensions hold references to the old one, breaking identity checks.

**Workaround:** Always use `--cov=anuga` (not a sub-path). For per-module numbers:
```bash
pytest --run-fast --cov=anuga anuga/structures/tests/ -q 2>&1 | grep structure_operator
```

**Not fixable** from conftest.py: pytest-cov creates `CovPlugin` (which starts
coverage) inside `pytest_load_initial_conftests(tryfirst=True)` — before conftest.py
is even loaded.

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

### RESOLVED (2026-06-15): "riverwall flux divergence" was really a DE0 boundary bug

**Symptom (now fixed):** a riverwall simulation under `multiprocessor_mode=2`
diverged from legacy — on `run_parallel_riverwall.py` (sequential), stage drifted
from 0 at t=0 to ~0.095 m by t≈100 s.

**Misdiagnosis → real cause.** It was *not* the riverwall flux. The riverwall
kernel (`core_compute_fluxes_central` elevation override + Villemonte weir) is
correct: with a GPU-supported boundary (e.g. `Dirichlet`), mode-1 vs mode-2
riverwall results are **bit-identical (0.0)**. The actual bug was the **boundary**:
`run_parallel_riverwall.py` uses `Transmissive_momentum_set_stage_boundary`
(*not* in `GPU_BOUNDARY_TYPES`), and it was **euler-specific** —
`evolve_one_euler_step()` dispatched straight to `_evolve_one_euler_step_c`, which
handles only GPU boundary types in C and **skips `update_boundary()` entirely**,
so the Transmissive boundary was silently never evaluated (stale edge values →
drift). rk2/rk3/ader2 already fell back to a Python-orchestrated `_gpu` loop for
non-GPU boundaries; **euler had no such fallback**. Confirmed by DE1/DE2/DE_ader2
matching (0.0) while DE0 diverged.

**Fix:** added `_evolve_one_euler_step_gpu()` (host evaluation of non-GPU
boundaries via `evaluate_segment` + `sync_boundary_values`, mirroring rk2/rk3),
and `_evolve_one_euler_step_c()` now delegates to it when
`not self._gpu_all_on_gpu`. DE0 + `Transmissive_momentum_set_stage` + riverwall is
now bit-identical to legacy; `run_parallel_riverwall.py` is **un-pinned** (passes
under `ANUGA_DEFAULT_COMPUTE_MODE=unified` again). Regression test:
`test_DE_gpu_omp.py::Test_GPU_NonGPUBoundaryFallback` (DE0/DE1/DE2/DE_ader2 with a
Transmissive_momentum_set_stage boundary).

**General lesson:** in mode 2, a boundary type not in `GPU_BOUNDARY_TYPES` is only
correct if the active step path falls back to host evaluation. All four DE
algorithms now do. If you add a new evolve path, replicate the
`if not self._gpu_all_on_gpu: return self._evolve_one_*_step_gpu(...)` fallback.

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

### GPU build forced to CPU (`set_gpu_offload(False)` / `-ngo`) is slow — nvc limitation

A `gpu_offload=true` (nvc `-mp=gpu,multicore`) build forced onto the host runs the
`omp target teams distribute` regions through nvc's host fallback, which **does not
scale with threads** (it gets *slower* with more threads). Microbenchmark (40M-element
memory-bound loop, 60 iters, RTX 5070 box, HPC SDK 26.3):

| config | 1t | 8t | 16t |
|--------|----|----|-----|
| nvc `-mp=gpu,multicore` + `OMP_TARGET_OFFLOAD=disabled` | 0.91s | 4.48s | 3.00s (pathological) |
| nvc `-mp=multicore` (multicore-only build) | 0.92s | 0.73s | 0.64s |
| gcc `-fopenmp` (`#pragma omp parallel for`) | 0.92s | 0.76s | 0.61s |
| nvc GPU offload | — | — | 0.12s |

Neither `OMP_TARGET_OFFLOAD=disabled` nor `CUDA_VISIBLE_DEVICES=` engages the good
multicore variant of the dual build — the host always gets the GPU variant's serial-ish
fallback. towradgi small (256k tri, -ft 200 -ys 50) confirms it: GPU 6.35s, `-ngo`
60–100s (1.7× scaling 1→16 threads), vs a gcc `gpu_offload=false` build at ~17s.

**Implication:** a GPU build is not a substitute for a CPU build. `set_gpu_offload(False)` /
`-ngo` is for **correctness A/B** (verify GPU and CPU give identical results — they are
bit-identical) only, NOT timing. For CPU-multicore performance, build with
`-Dgpu_offload=false` (gcc → host-optimised `omp parallel for` via the `CPU_ONLY_MODE`
macros). `set_gpu_offload(False)` warns about this on a GPU build.

**This is a confirmed, documented NVHPC limitation, not an ANUGA bug** (investigated
2026-06-13). The NVHPC Reference Guide defines `-mp=gpu` as "compiled for GPU execution
*as well as host fallback to the CPU*" — and that host fallback runs **single-threaded**.
A peer-reviewed compiler comparison (IPDPSW 2023, Iowa State) measured NVHPC host fallback
at "OMP 1" (1 CPU thread) and found "the GPU code version on CPU in host fallback mode
performs worse than the CPU version with 1 thread". NVHPC is also documented to handle
nested/inner parallel regions poorly (NVIDIA recommends the `loop` directive over
`teams distribute parallel for` for this reason). The fast multicore variant only exists
when `-mp=multicore` is the *sole* mode; `-mp=gpu,multicore` does not let the runtime pick
it on the host (verified across `OMP_TARGET_OFFLOAD=disabled`, `CUDA_VISIBLE_DEVICES=`,
argument order, and `ACC_DEVICE_TYPE` — the last hangs). So a single nvc binary cannot be
fast on both GPU and CPU; the two-build split (gcc CPU / nvc GPU) is required.

References:
- NVHPC Compilers Reference Guide 26.3 — https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-ref-guide/index.html
- "OpenMP Offload Features and Strategies for High Performance across Architectures and
  Compilers", IPDPSW 2023 — https://swapp.cs.iastate.edu/files/inline-files/OpenMP_Offload_Features_and_Strategies_for_High_Performance_across_Architectures_and_Compilers-ipdpsw-may-2023.pdf
- OMP_TARGET_OFFLOAD, OpenMP 5.0 spec — https://www.openmp.org/spec-html/5.0/openmpse65.html
