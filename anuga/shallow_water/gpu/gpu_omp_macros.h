// OpenMP pragma macros for GPU vs CPU execution
//
// When CPU_ONLY_MODE is defined (via -Dgpu_offload=false), use regular OpenMP
// Otherwise, use OpenMP target for GPU offloading
//
// This allows the same code to run on both GPU and CPU multicore

#ifndef GPU_OMP_MACROS_H
#define GPU_OMP_MACROS_H

// Helper macro to stringify pragma arguments (allows macro expansion before stringification)
#define DO_PRAGMA(x) _Pragma(#x)

#ifdef CPU_ONLY_MODE

// ============================================================================
// CPU MULTICORE MODE - Regular OpenMP, no device offloading
// ============================================================================

// Parallel loops with SIMD vectorization
#define OMP_PARALLEL_LOOP _Pragma("omp parallel for simd")
#define OMP_PARALLEL_LOOP_SIMD _Pragma("omp parallel for simd")

// Reductions - use DO_PRAGMA to allow variable name expansion
#define OMP_PARALLEL_LOOP_REDUCTION_PLUS(var) DO_PRAGMA(omp parallel for simd reduction(+:var))
#define OMP_PARALLEL_LOOP_REDUCTION_MIN(var) DO_PRAGMA(omp parallel for simd reduction(min:var))
#define OMP_PARALLEL_LOOP_REDUCTION_MAX(var) DO_PRAGMA(omp parallel for simd reduction(max:var))

// Data mapping - no-op on CPU (data already in host memory)
#define OMP_TARGET_ENTER_DATA_MAP_TO(...)
#define OMP_TARGET_ENTER_DATA_MAP_ALLOC(...)
#define OMP_TARGET_EXIT_DATA_MAP_DELETE(...)

// Data transfer - no-op on CPU
#define OMP_TARGET_UPDATE_TO(...)
#define OMP_TARGET_UPDATE_FROM(...)

// Device pointer clause - no-op on CPU
#define OMP_IS_DEVICE_PTR(ptr)

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

// Data mapping to device
// Note: These need to be used with care - the actual map clause arguments
// must be specified in the calling code since macros can't handle variable arguments well
// For now, these are placeholders - actual mapping is done inline
#define OMP_TARGET_ENTER_DATA_MAP_TO(...) _Pragma("omp target enter data map(to:" #__VA_ARGS__ ")")
#define OMP_TARGET_ENTER_DATA_MAP_ALLOC(...) _Pragma("omp target enter data map(alloc:" #__VA_ARGS__ ")")
#define OMP_TARGET_EXIT_DATA_MAP_DELETE(...) _Pragma("omp target exit data map(delete:" #__VA_ARGS__ ")")

// Data transfer
#define OMP_TARGET_UPDATE_TO(...) _Pragma("omp target update to(" #__VA_ARGS__ ")")
#define OMP_TARGET_UPDATE_FROM(...) _Pragma("omp target update from(" #__VA_ARGS__ ")")

// Device pointer clause
#define OMP_IS_DEVICE_PTR(ptr) is_device_ptr(ptr)

#endif // CPU_ONLY_MODE

#endif // GPU_OMP_MACROS_H
