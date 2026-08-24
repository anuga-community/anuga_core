// gcc quirk shim.
//
// gpu_device_helpers.h declares `static const double GPU_TINY` inside a
// `#pragma omp declare target` region.  gcc records that in the object's
// .gnu.offload_vars table under the unmangled name `GPU_TINY`, even when
// offloading is disabled.  ANUGA's own build links the kernels into a shared
// module, where an unresolved entry is tolerated; linking a plain executable
// it is not.  Host code always reads the TU-local static, so this definition
// exists purely to satisfy that table entry.
#if defined(__GNUC__) && !defined(__clang__) && !defined(__NVCOMPILER)
const double GPU_TINY = 1.0e-100;
#endif
