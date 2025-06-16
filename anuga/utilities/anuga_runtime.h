#ifndef ANUGA_RUNTIME_H
#define ANUGA_RUNTIME_H

// useful definitions for ANUGA runtime

// there must be a better way to do this... TODO JLGV
#if defined(__APPLE__)
// clang doesn't have openmp
#else
#include "omp.h"
#endif


#endif