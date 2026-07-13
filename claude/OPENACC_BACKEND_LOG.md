# OpenACC GPU backend — work log

**Status:** implemented, uncommitted on branch `develop`. Builds/validates as a third
GPU backend alongside OpenMP-target. **Perf resolved, faster than OpenMP-target** via two
changes: (1) `default(present)` removed per-launch present-or-copyin analysis (fluxes-central
2.0s → 1.62s, ~19%); (2) a **single async queue** (`async(1)` + targeted `acc wait(1)`)
removed the residual per-launch synchronous-launch stall (faster again, user-confirmed).
**⚠ Correctness of the async change is UNVERIFIED** until towradgi `.sww` is diffed against
the OpenMP-target build — see §6.

**Goal:** let the shallow-water GPU kernels compile to **OpenACC** (`nvc -acc=gpu`) as
well as the existing **OpenMP-target** (`nvc -mp=gpu`), to A/B whether OpenACC gives a
speedup on the GPU. OpenMP stays the CPU-multicore backend (it produces better CPU code).
All backend selection funnels through one header so kernels are never edited per-backend.

---

## 1. What changed (files)

| File | Change |
|------|--------|
| `anuga/shallow_water/gpu/gpu_omp_macros.h` | **Core of the work.** Now a clean 3-mode header (see §2). Fixed a latent bug, added new macros, added the whole OpenACC branch + runtime-API shims + `default(present)` experiment. |
| `anuga/shallow_water/gpu/*.c` (8 files) | 103 raw `#pragma omp target …` directives routed through the macros (see §3). Files: `core_kernels.c`, `gpu_boundaries.c`, `gpu_culvert_operator.c`, `gpu_domain_core.c`, `gpu_halo.c`, `gpu_inlet_operator.c`, `gpu_max_quantities_operator.c`, `gpu_rate_operator.c`. |
| `meson_options.txt` | New `gpu_backend` combo option (`openmp` \| `openacc`, default `openmp`). |
| `anuga/shallow_water/meson.build` | Reads `gpu_backend`; guards; nvc flags for OpenACC; link-args for OpenACC (see §4). |
| `.gitignore` | Ignores `tmp_artifacts/` (local scratch/compile-probe dir used this session). |

---

## 2. The macro header — 3 modes

`gpu_omp_macros.h` selects one of three modes at build time. Kernels only ever use the
`OMP_*` macros (the `OMP_` prefix is kept in all modes purely for call-site stability):

| Mode | Selected by | Compute loop | Data region |
|------|-------------|--------------|-------------|
| `CPU_ONLY_MODE` | `-Dgpu_offload=false` | `omp parallel for simd` | no-op (host memory) |
| `ACC_OFFLOAD_MODE` | `-Dgpu_backend=openacc` | `acc parallel loop` | `acc enter/exit data`, `acc update` |
| *(default)* | `-Dgpu_offload=true` | `omp target teams loop` | `omp target enter/exit data`, `omp target update` |

**Bug fixed along the way:** the pre-existing `OMP_TARGET_ENTER_DATA_MAP_*` macros were
**broken** — they used `_Pragma("…" #__VA_ARGS__ "…")`, but `_Pragma` requires a *single*
parenthesised string literal (adjacent-string concatenation is rejected: `error: _Pragma
takes a parenthesized string literal`). That's why all 105 directives had been written raw
and none used the macros. All macros now use the working `DO_PRAGMA(x) => _Pragma(#x)`
idiom.

**New macros added** (the raw directives that didn't fit the existing set):
- `OMP_PARALLEL_LOOP_IS_DEVICE_PTR(ptr)` — `is_device_ptr` / `deviceptr` loop (gpu_halo.c ×2)
- `OMP_PARALLEL_LOOP_MAP_TO(...)` — loop with a `map(to:)` / `copyin` (gpu_culvert_operator.c ×1)
- `OMP_PARALLEL_LOOP_REDUCTION_MIN_PLUS(a, b)` — double reduction (core_kernels.c ×1)

**OpenACC is "pure":** in ACC mode the compute loops, the data management, AND the
runtime API all use OpenACC — no OpenMP-target device memory is involved, so there is no
OpenMP↔OpenACC present-table interop to reason about. The OpenMP target runtime API calls
in the `.c` files are redirected (via `#define`, same pattern as the CPU stubs) to
OpenACC equivalents:

| OpenMP call | OpenACC redirect |
|-------------|------------------|
| `omp_target_alloc(size, dev)` | `acc_malloc(size)` |
| `omp_target_free(ptr, dev)` | `acc_free(ptr)` |
| `omp_target_memcpy(...)` | direction-detecting shim → `acc_memcpy_to_device` / `_from_device` / `_device` (host sentinel = `-1`) |
| `omp_target_is_present(ptr, dev)` | `acc_is_present(ptr, 1)` (1-byte probe of base addr) |
| `omp_get_initial_device()` | `-1` sentinel (host) |
| `omp_get_default_device()` | `acc_get_device_num(acc_device_default)` |
| `omp_get_num_devices()` | `acc_get_num_devices(acc_device_default)` |
| `omp_set_default_device(n)` | `acc_set_device_num(n, acc_device_default)` |

---

## 3. The directive conversion (103 sites)

Done with a reviewed one-off script (`tmp_artifacts/convert_directives.py`), then diffed.
It reported **all** `#pragma omp target` directives recognised (the 2 remaining "matches"
in the codebase are comments). Two classes of intentional, semantics-preserving change:

1. **4 combined `map(to:) map(alloc:)` directives** split into two calls
   (`OMP_TARGET_ENTER_DATA_MAP_TO(...)` + `OMP_TARGET_ENTER_DATA_MAP_ALLOC(...)`), which is
   equivalent. Sites: `gpu_inlet_operator.c`, `gpu_culvert_operator.c`, `gpu_domain_core.c` (×2).
2. **2 bare `omp target teams distribute parallel for`** normalized to `OMP_PARALLEL_LOOP`
   (= `omp target teams loop`), matching the 39 existing sites; functionally identical on nvc.

Also simplified a now-redundant `#ifdef CPU_ONLY_MODE / #else / #endif` guard in
`core_kernels.c` (the macro handles all 3 modes itself).

**Behavior-preservation proof:** preprocessed all 8 files in OpenMP-target mode *before*
and *after* the conversion and diffed the emitted `#pragma omp target` lines. The only
deltas were exactly the 4 splits (+4 directives) and the 2 `distribute parallel for →
teams loop` relabelings. Everything else byte-identical.

---

## 4. Build

Both GPU backends use **nvc** and `gpu_offload=true`; they differ only in `gpu_backend`.

```bash
module load nvidia-hpc-sdk          # nvc must be on PATH

# OpenMP-target GPU (the original backend)
CC=$(which nvc) pip install --no-build-isolation -e . \
  -Csetup-args=-Dgpu_offload=true \
  -Csetup-args=-Dgpu_backend=openmp \
  -Csetup-args=-Dgpu_arch=cc80        # cc70=V100, cc80=A100, cc90=H100

# OpenACC GPU
CC=$(which nvc) pip install --no-build-isolation -e . \
  -Csetup-args=-Dgpu_offload=true \
  -Csetup-args=-Dgpu_backend=openacc \
  -Csetup-args=-Dgpu_arch=cc80
```

- OpenACC nvc flags: `-O3 -acc=gpu -mp=multicore -Minfo=accel -g -gpu=<arch> -DACC_OFFLOAD_MODE`,
  plus `-acc=gpu -gpu=<arch>` in link args (so the OpenACC device image + runtime link in).
- **Switching `openmp ↔ openacc` is a reconfigure, not a from-scratch build** (compiler is
  nvc both ways): `meson configure build/<cpXX-dir> -Dgpu_backend=openacc`, then next import
  / `meson compile` rebuilds only the gpu extension. **Full `rm -rf build/` is only needed
  when you change the compiler (gcc↔nvc).**
- Guards: `gpu_backend=openacc` errors if `gpu_offload=false`, or on a non-nvc compiler.

### Two builds side by side (for validation)
Same package name `anuga` ⇒ two backends need two environments. Use a worktree so build
dirs don't collide:
```bash
git worktree add ../anuga-openacc develop
# env 1 (this tree): build openmp; env 2 (worktree, cloned conda env): build openacc
```
Then `conda activate` whichever to A/B the same script. (Or Option B: one env, reconfigure
in place, test one at a time.)

---

## 5. Verification done (without a GPU run)

- **CPU mode:** all 8 files compile clean with gcc.
- **Pragma expansion:** correct in all 3 modes (checked via `gcc -E`).
- **OpenACC shims:** compile against `openacc.h` (`gcc -fopenacc`).
- **Behavior preservation:** the before/after pragma diff in §3.
- **`meson introspect`** shows `gpu_backend` parses as an option.
- Caveat: `acc_memcpy_device` isn't in *gcc's* `openacc.h` (warns under gcc) — it IS
  standard OpenACC 2.6 and in NVHPC's header, and its device→device branch is never hit by
  current call sites (both memcpy calls are H2D/D2H). Fine on the real nvc build.

---

## 6. Performance finding (the open question)

- On the **towradgi** case, **OpenACC is ~10% slower** than OpenMP-target. Surprising.
- Both backends pick **identical kernel geometry** (`gang vector 128` / `teams … 128`).
  ⇒ device-side compute per kernel is ~equal; the gap is **host-side per-launch overhead**,
  not the kernels. This is the "OpenACC runtime overhead" hypothesis.

### Experiment `default(present)` — DONE, SUCCESS ✅
Added to all 8 ACC compute-loop macros (`acc parallel loop default(present) …`). Rationale:
without it, `acc parallel loop` emits present-or-copyin analysis at *every* launch for each
of the ~20 arrays the flux kernel touches — a per-launch host cost OpenMP-target doesn't pay
the same way.

**Result (towradgi):** all kernels faster; **fluxes-central 2.0s → 1.62s (~19%)**. The
~10% OpenACC deficit is gone and OpenACC is now ahead of OpenMP-target. Confirms the gap
was per-launch present-analysis overhead (device geometry was already identical). No
"not present" errors ⇒ the `copyin`/`create` translation covers every array a kernel
touches (no hidden per-launch copies).

To revert (if ever needed): drop `default(present)` from those macros only.

### The 2-minute diagnostic to run next
In nsys (there's a `validation_tests/case_studies/towradgi/report1.nsys-rep` already):
compare **total kernel time vs total wall time** per build.
- kernel time ≈ equal, wall time longer on OpenACC → pure between-kernel host overhead.
- **recurring small HtoD/DtoH memcpys per timestep in the OpenACC trace but not OpenMP** →
  an array isn't staying resident (silent per-launch copy → translation bug).

### Experiment: single async queue — DONE, faster ✅ (correctness validation PENDING)
`default(present)` closed the gap but a residual per-launch **synchronous-launch** stall
remained (each `acc parallel` does a host `cuStreamSynchronize`; OpenMP-target doesn't).
Fix: enqueue ALL device work on **one in-order async queue (`async(1)`)** so the host never
blocks between kernels; drain with `acc wait(1)` only where the host consumes device
results. No real overlap — a single in-order queue preserves all data deps for free.

Implementation (all in `gpu_omp_macros.h` ACC branch + 1 line in `gpu_halo.c`):
- compute loops → `acc parallel loop default(present) async(1)`
- reduction loops → `acc wait(1)` then a **synchronous** reduction (scalar valid for host immediately)
- `update self`/`update device`/`exit data` → `acc wait(1)` then synchronous op
- `enter data` → stays synchronous (setup, runs before its kernels)
- new `OMP_TARGET_WAIT` macro (no-op in OpenMP-target/CPU) → one explicit drain before the
  GPU-aware-MPI D2H `omp_target_memcpy` in `gpu_halo.c` (the only host read not already macro'd)

**Result:** builds and is **faster again** (user-confirmed). All wait points enumerated from
an audit: 9 reduction sites + 17 `UPDATE_FROM` sites + 2 halo memcpy sites = the complete set
of host-reads-of-device-data.

**⚠ PENDING — correctness validation.** An async bug is SILENT (missed wait = data race, not
a crash). MUST diff towradgi `.sww` output: OpenACC-async vs OpenMP-target build → must match
to roundoff. Until that passes, treat the async change as unverified. If a mismatch appears,
it's a missing `acc wait(1)` before some host read; the audit set above is where to look.

### Further levers (later, only if needed), ranked
1. Make device-only intermediate reductions `async(1)` (skip their drain) — some of the 9
   reduction sites feed the host (need the drain) but any that only feed later kernels don't.
2. Explicit `gang vector` + `vector_length(...)` tuning (only if `-Minfo` geometry regresses).

All future tuning stays localized to the ACC branch of `gpu_omp_macros.h` — add new macro
variants (e.g. `OMP_PARALLEL_LOOP_GANG_VECTOR`) so both backends stay in sync; don't edit
the kernels.

---

## 6b. Startup banner reports the active backend

The GPU init banner used to hard-code "OpenMP target offloading". Now it reports the
compiled backend. Wiring (mirrors `gpu_available()`):
- `gpu/gpu_domain_core.c`: `gpu_backend_name()` → `"OpenACC"` / `"OpenMP target"` / `"OpenMP multicore"` via `#ifdef`
- `gpu/gpu_domain.h`: declaration
- `sw_domain_gpu_ext.pyx`: `cdef extern` + `get_gpu_backend_name()` wrapper (decodes to str)
- `shallow_water_domain.py` (~line 5739): banner prints `… using {backend} offloading`
  (falls back to "OpenMP target" if the symbol isn't importable). Takes effect on rebuild.

## 7. Key references

- Macro header: `anuga/shallow_water/gpu/gpu_omp_macros.h` (all 3 modes + shims + experiment)
- Build logic: `anuga/shallow_water/meson.build` (search `gpu_backend`), `meson_options.txt`
- Conversion script (scratch, gitignored): `tmp_artifacts/convert_directives.py`
- Session memory: `openacc-gpu-backend.md` in the Claude project memory dir
- Nothing is committed — all changes are in the `develop` working tree.
