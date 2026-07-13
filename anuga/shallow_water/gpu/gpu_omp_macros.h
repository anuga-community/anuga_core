// Parallel-execution pragma macros for GPU vs CPU execution
//
// The same kernel source compiles for three back ends, selected at build time
// by the meson `gpu_offload` / `gpu_backend` options (see shallow_water/meson.build):
//
//   * CPU_ONLY_MODE      -Dgpu_offload=false        regular OpenMP multicore
//   * ACC_OFFLOAD_MODE   -Dgpu_backend=openacc      OpenACC GPU offloading (nvc -acc=gpu)
//   * (default)          -Dgpu_offload=true         OpenMP-target GPU offloading
//
// Every compute loop, data-region and data-transfer directive in gpu/*.c goes
// through one of the OMP_* macros below so that switching back end is a pure
// build-flag change with no edits to the kernels.  The OMP_ prefix is kept for
// all three modes (including OpenACC) purely for call-site stability.

#ifndef GPU_OMP_MACROS_H
#define GPU_OMP_MACROS_H

// Helper macro to stringify pragma arguments (allows macro expansion before stringification).
// NOTE: _Pragma takes a SINGLE parenthesised string literal - adjacent string-literal
// concatenation ("a" "b") is NOT accepted by the operator, so every pragma must be built
// through DO_PRAGMA rather than by concatenating pieces inside _Pragma().
#define DO_PRAGMA(x) _Pragma(#x)

#ifdef CPU_ONLY_MODE

// ============================================================================
// CPU MULTICORE MODE - Regular OpenMP, no device offloading
// ============================================================================

// Parallel loops with SIMD vectorization
#define OMP_PARALLEL_LOOP _Pragma("omp parallel for simd")
#define OMP_PARALLEL_LOOP_SIMD _Pragma("omp parallel for simd")

// Reductions - use DO_PRAGMA to allow variable name expansion
// Note: simd is omitted here; reduction loops cannot be SIMD-vectorized and
// clang warns (-Wpass-failed=transform-warning) when simd is requested.
#define OMP_PARALLEL_LOOP_REDUCTION_PLUS(var) DO_PRAGMA(omp parallel for reduction(+:var))
#define OMP_PARALLEL_LOOP_REDUCTION_MIN(var) DO_PRAGMA(omp parallel for reduction(min:var))
#define OMP_PARALLEL_LOOP_REDUCTION_MAX(var) DO_PRAGMA(omp parallel for reduction(max:var))
#define OMP_PARALLEL_LOOP_REDUCTION_MIN_PLUS(minvar, plusvar) \
    DO_PRAGMA(omp parallel for reduction(min:minvar) reduction(+:plusvar))

// Loop over data reachable through a device pointer / mapped array: on the host
// these are plain host pointers, so an ordinary parallel loop is correct.
#define OMP_PARALLEL_LOOP_IS_DEVICE_PTR(ptr) _Pragma("omp parallel for simd")
#define OMP_PARALLEL_LOOP_MAP_TO(...) _Pragma("omp parallel for simd")

// Data mapping - no-op on CPU (data already in host memory)
#define OMP_TARGET_ENTER_DATA_MAP_TO(...)
#define OMP_TARGET_ENTER_DATA_MAP_ALLOC(...)
#define OMP_TARGET_EXIT_DATA_MAP_DELETE(...)

// Data transfer - no-op on CPU
#define OMP_TARGET_UPDATE_TO(...)
#define OMP_TARGET_UPDATE_FROM(...)

// Device pointer clause - no-op on CPU
#define OMP_IS_DEVICE_PTR(ptr)

// Queue drain - no-op on CPU (no device, kernels are synchronous)
#define OMP_TARGET_WAIT

// ============================================================================
// CPU MULTICORE MODE stubs for OpenMP target-offloading API
//
// The standard libomp on macOS (and minimal Linux builds) does NOT ship the
// OpenMP 4.5 target-allocation API (omp_target_alloc, omp_target_free,
// omp_target_memcpy, omp_target_is_present).  In CPU_ONLY_MODE there is no
// GPU device, so we provide static inline stubs and redirect every call site
// via macros.  The redirect must happen here (after <omp.h> is already
// included by the .c files) so the macros suppress external symbol references
// without conflicting with omp.h's extern declarations.
// ============================================================================

#include <stdlib.h>
#include <string.h>

static inline void *_anuga_omp_target_alloc(size_t size, int dev) {
    (void)dev; return malloc(size);
}
static inline void _anuga_omp_target_free(void *ptr, int dev) {
    (void)dev; free(ptr);
}
static inline int _anuga_omp_target_memcpy(
        void *dst, const void *src, size_t length,
        size_t dst_offset, size_t src_offset,
        int dst_dev, int src_dev) {
    (void)dst_dev; (void)src_dev;
    memcpy((char *)dst + dst_offset, (const char *)src + src_offset, length);
    return 0;
}
static inline int _anuga_omp_target_is_present(const void *ptr, int dev) {
    (void)ptr; (void)dev; return 1;
}
static inline int _anuga_omp_get_initial_device(void) { return 0; }

#define omp_target_alloc        _anuga_omp_target_alloc
#define omp_target_free         _anuga_omp_target_free
#define omp_target_memcpy       _anuga_omp_target_memcpy
#define omp_target_is_present   _anuga_omp_target_is_present
#define omp_get_initial_device  _anuga_omp_get_initial_device

#elif defined(ACC_OFFLOAD_MODE)

// ============================================================================
// GPU MODE - OpenACC offloading (nvc -acc=gpu)
//
// Pure OpenACC: BOTH the compute loops AND the device data management use
// OpenACC directives/runtime, so no OpenMP-target device memory is involved
// and there is no OpenMP<->OpenACC present-table interop to reason about.
// OpenMP is still used for the CPU multicore back end (CPU_ONLY_MODE above).
// ============================================================================

// ---------------------------------------------------------------------------
// Single global async queue (number 1).
//
// ALL device work - compute loops AND data transfers - is enqueued on ONE queue
// so it executes back-to-back, in order, WITHOUT the host blocking after every
// kernel. That per-launch host synchronisation (a cuStreamSynchronize per
// `acc parallel`) is the OpenACC-specific overhead that OpenMP-target does not
// pay; on the towradgi case it left OpenACC behind even after default(present).
// We do NOT overlap anything - a single in-order queue preserves every data
// dependency for free - we only stop paying the per-kernel stall.
//
// The queue is drained with `acc wait(1)` ONLY where the host actually consumes
// device results. Those points are baked into the macros below:
//   - reduction loops  (host reads the reduced scalar)         -> wait, then SYNC reduction
//   - update self/device, exit data (host produces/consumes)   -> wait, then sync op
// plus one place the macros can't cover - the GPU-aware-MPI D2H copy in
// gpu_halo.c - which calls OMP_TARGET_WAIT explicitly.
//
// CORRECTNESS: every host-side read of device data must be preceded by a drain.
// The three categories above are the complete set for this codebase (all host
// reads go through a reduction, an update-from, or the halo memcpy). A MISSED
// wait is a silent data race, not a crash - so after any change here, validate
// .sww output against the OpenMP-target build (must match to roundoff).
//
// default(present) (on every compute loop) asserts all referenced aggregates are
// already resident (persistently mapped via `acc enter data`); without it nvc
// emits a present-or-copyin analysis at every launch. It also turns any missing
// mapping into a hard "not present" error rather than a silent per-launch copy.
// ---------------------------------------------------------------------------

// Parallel loops on device - enqueued async on the shared queue.
#define OMP_PARALLEL_LOOP _Pragma("acc parallel loop default(present) async(1)")
#define OMP_PARALLEL_LOOP_SIMD _Pragma("acc parallel loop default(present) async(1)")

// Reductions: drain the queue first (so the inputs are final), then run the
// reduction SYNCHRONOUSLY so the reduced scalar is valid on the host immediately
// after the loop - no separate wait needed at the call site.
#define OMP_PARALLEL_LOOP_REDUCTION_PLUS(var) \
    DO_PRAGMA(acc wait(1)) DO_PRAGMA(acc parallel loop default(present) reduction(+:var))
#define OMP_PARALLEL_LOOP_REDUCTION_MIN(var) \
    DO_PRAGMA(acc wait(1)) DO_PRAGMA(acc parallel loop default(present) reduction(min:var))
#define OMP_PARALLEL_LOOP_REDUCTION_MAX(var) \
    DO_PRAGMA(acc wait(1)) DO_PRAGMA(acc parallel loop default(present) reduction(max:var))
#define OMP_PARALLEL_LOOP_REDUCTION_MIN_PLUS(minvar, plusvar) \
    DO_PRAGMA(acc wait(1)) DO_PRAGMA(acc parallel loop default(present) reduction(min:minvar) reduction(+:plusvar))

// Device-pointer / map(to) loops also ride the async queue.
#define OMP_PARALLEL_LOOP_IS_DEVICE_PTR(ptr) DO_PRAGMA(acc parallel loop default(present) deviceptr(ptr) async(1))
#define OMP_PARALLEL_LOOP_MAP_TO(...) DO_PRAGMA(acc parallel loop default(present) copyin(__VA_ARGS__) async(1))

// Data mapping to device. enter data is setup - it runs before the kernels that
// use the memory, so it stays synchronous. exit data frees memory the async
// kernels may still be using, so drain the queue first.
#define OMP_TARGET_ENTER_DATA_MAP_TO(...) DO_PRAGMA(acc enter data copyin(__VA_ARGS__))
#define OMP_TARGET_ENTER_DATA_MAP_ALLOC(...) DO_PRAGMA(acc enter data create(__VA_ARGS__))
#define OMP_TARGET_EXIT_DATA_MAP_DELETE(...) \
    DO_PRAGMA(acc wait(1)) DO_PRAGMA(acc exit data delete(__VA_ARGS__))

// Data transfer. The host produces (update device) or consumes (update self) this
// data, so drain the async queue first, then transfer synchronously.
#define OMP_TARGET_UPDATE_TO(...) DO_PRAGMA(acc wait(1)) DO_PRAGMA(acc update device(__VA_ARGS__))
#define OMP_TARGET_UPDATE_FROM(...) DO_PRAGMA(acc wait(1)) DO_PRAGMA(acc update self(__VA_ARGS__))

// Explicit queue drain for host-side sync points the macros above can't cover
// (currently only the GPU-aware-MPI halo D2H memcpy in gpu_halo.c).
#define OMP_TARGET_WAIT _Pragma("acc wait(1)")

// Device pointer clause
#define OMP_IS_DEVICE_PTR(ptr) deviceptr(ptr)

// ============================================================================
// OpenACC shims for the OpenMP target-offloading runtime API
//
// The kernels call the OpenMP 4.5 device API (omp_target_alloc, ...) directly.
// In OpenACC mode we redirect each to its OpenACC runtime equivalent so the
// same call sites work unchanged.  As in CPU_ONLY_MODE, the redirect happens
// here (after <omp.h>) so the macros mask the omp.h extern declarations.
//
//   * omp_target_is_present(ptr, dev) -> acc_is_present(ptr, bytes): OpenACC needs a
//     byte extent; we probe 1 byte, which tests whether the base address is mapped.
//   * omp_target_memcpy direction is encoded in the device-number arguments; we use a
//     host sentinel (-1) and dispatch to acc_memcpy_to_device / _from_device / _device.
// ============================================================================

#include <openacc.h>
#include <stdlib.h>
#include <string.h>

#define ANUGA_ACC_HOST_DEVICE (-1)

static inline void *_anuga_acc_target_alloc(size_t size, int dev) {
    (void)dev; return acc_malloc(size);
}
static inline void _anuga_acc_target_free(void *ptr, int dev) {
    (void)dev; acc_free(ptr);
}
static inline int _anuga_acc_target_is_present(const void *ptr, int dev) {
    (void)dev; return acc_is_present((void *)ptr, 1);
}
static inline int _anuga_acc_get_initial_device(void) { return ANUGA_ACC_HOST_DEVICE; }
static inline int _anuga_acc_get_default_device(void) {
    return acc_get_device_num(acc_device_default);
}
static inline int _anuga_acc_get_num_devices(void) {
    return acc_get_num_devices(acc_device_default);
}
static inline int _anuga_acc_target_memcpy(
        void *dst, const void *src, size_t length,
        size_t dst_offset, size_t src_offset,
        int dst_dev, int src_dev) {
    char *d = (char *)dst + dst_offset;
    const char *s = (const char *)src + src_offset;
    int dst_host = (dst_dev == ANUGA_ACC_HOST_DEVICE);
    int src_host = (src_dev == ANUGA_ACC_HOST_DEVICE);
    if (src_host && !dst_host) {
        acc_memcpy_to_device(d, (void *)s, length);
    } else if (dst_host && !src_host) {
        acc_memcpy_from_device(d, (void *)s, length);
    } else if (!dst_host && !src_host) {
        acc_memcpy_device(d, (void *)s, length);
    } else {
        memcpy(d, s, length);
    }
    return 0;
}

#define omp_target_alloc        _anuga_acc_target_alloc
#define omp_target_free         _anuga_acc_target_free
#define omp_target_memcpy       _anuga_acc_target_memcpy
#define omp_target_is_present   _anuga_acc_target_is_present
#define omp_get_initial_device  _anuga_acc_get_initial_device
#define omp_get_default_device  _anuga_acc_get_default_device
#define omp_get_num_devices     _anuga_acc_get_num_devices
#define omp_set_default_device(n) acc_set_device_num((n), acc_device_default)

#else

// ============================================================================
// GPU MODE - OpenMP target offloading
// ============================================================================

// Parallel loops on device
#define OMP_PARALLEL_LOOP _Pragma("omp target teams loop")
#define OMP_PARALLEL_LOOP_SIMD _Pragma("omp target teams loop")

// Reductions on device - use DO_PRAGMA to allow variable name expansion
// Note: Using distribute parallel for for better reduction support
#define OMP_PARALLEL_LOOP_REDUCTION_PLUS(var) DO_PRAGMA(omp target teams distribute parallel for reduction(+:var))
#define OMP_PARALLEL_LOOP_REDUCTION_MIN(var) DO_PRAGMA(omp target teams loop reduction(min:var))
#define OMP_PARALLEL_LOOP_REDUCTION_MAX(var) DO_PRAGMA(omp target teams distribute parallel for reduction(max:var))
#define OMP_PARALLEL_LOOP_REDUCTION_MIN_PLUS(minvar, plusvar) \
    DO_PRAGMA(omp target teams distribute parallel for reduction(min:minvar) reduction(+:plusvar))

// Loop over a device pointer (omp_target_alloc'd) - keep distribute parallel for form
#define OMP_PARALLEL_LOOP_IS_DEVICE_PTR(ptr) \
    DO_PRAGMA(omp target teams distribute parallel for is_device_ptr(ptr))
// Loop that also maps a small array in for the duration of the region
#define OMP_PARALLEL_LOOP_MAP_TO(...) \
    DO_PRAGMA(omp target teams distribute parallel for map(to: __VA_ARGS__))

// Data mapping to device
#define OMP_TARGET_ENTER_DATA_MAP_TO(...) DO_PRAGMA(omp target enter data map(to: __VA_ARGS__))
#define OMP_TARGET_ENTER_DATA_MAP_ALLOC(...) DO_PRAGMA(omp target enter data map(alloc: __VA_ARGS__))
#define OMP_TARGET_EXIT_DATA_MAP_DELETE(...) DO_PRAGMA(omp target exit data map(delete: __VA_ARGS__))

// Data transfer
#define OMP_TARGET_UPDATE_TO(...) DO_PRAGMA(omp target update to(__VA_ARGS__))
#define OMP_TARGET_UPDATE_FROM(...) DO_PRAGMA(omp target update from(__VA_ARGS__))

// Device pointer clause
#define OMP_IS_DEVICE_PTR(ptr) is_device_ptr(ptr)

// Queue drain - no-op for OpenMP target (kernels are synchronous, no async queue)
#define OMP_TARGET_WAIT

#endif // CPU_ONLY_MODE / ACC_OFFLOAD_MODE / OpenMP-target

#endif // GPU_OMP_MACROS_H
