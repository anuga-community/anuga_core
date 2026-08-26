// Thin C-only MPI wrapper for the standalone benchmark -- the seed of an
// mpi4py-free MPI layer for ANUGA.  Compiled with -DHAVE_MPI it wraps real
// MPI; without it every call is a single-process identity, so the same
// driver source builds and runs serially.  Keep every operation the driver
// needs behind this interface: init/finalize, rank/size, reductions,
// gathers -- never call MPI directly from bench code.
#ifndef BENCH_MPI_H
#define BENCH_MPI_H

#include <stdint.h>
#include <string.h>

#ifdef HAVE_MPI
#include <mpi.h>

static inline void bmpi_init(int *argc, char ***argv) { MPI_Init(argc, argv); }
static inline void bmpi_finalize(void)                { MPI_Finalize(); }
static inline int  bmpi_rank(void) { int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }
static inline int  bmpi_size(void) { int s; MPI_Comm_size(MPI_COMM_WORLD, &s); return s; }
static inline void bmpi_barrier(void) { MPI_Barrier(MPI_COMM_WORLD); }
static inline double bmpi_sum_d(double v) { double o; MPI_Allreduce(&v, &o, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); return o; }
static inline double bmpi_max_d(double v) { double o; MPI_Allreduce(&v, &o, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); return o; }
static inline double bmpi_min_d(double v) { double o; MPI_Allreduce(&v, &o, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); return o; }
static inline int    bmpi_max_i(int v)    { int o;    MPI_Allreduce(&v, &o, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); return o; }
static inline void bmpi_bcast_i(int *v)   { MPI_Bcast(v, 1, MPI_INT, 0, MPI_COMM_WORLD); }

// Gather variable-length per-rank blocks to rank 0.  recvbuf/rcounts/displs
// are only used on rank 0 (displs computed here from gathered counts).
// Returns the global element count (valid on rank 0).
static inline int64_t bmpi_gatherv_d(const double *send, int count,
                                     double *recv, int *rcounts, int *displs) {
    MPI_Gather(&count, 1, MPI_INT, rcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int64_t total = 0;
    if (bmpi_rank() == 0) {
        for (int r = 0; r < bmpi_size(); r++) { displs[r] = (int)total; total += rcounts[r]; }
    }
    MPI_Gatherv((void *)send, count, MPI_DOUBLE, recv, rcounts, displs,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return total;
}
static inline int64_t bmpi_gatherv_i64(const int64_t *send, int count,
                                       int64_t *recv, int *rcounts, int *displs) {
    MPI_Gather(&count, 1, MPI_INT, rcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int64_t total = 0;
    if (bmpi_rank() == 0) {
        for (int r = 0; r < bmpi_size(); r++) { displs[r] = (int)total; total += rcounts[r]; }
    }
    MPI_Gatherv((void *)send, count, MPI_INT64_T, recv, rcounts, displs,
                MPI_INT64_T, 0, MPI_COMM_WORLD);
    return total;
}

#else  // single-process identities

static inline void bmpi_init(int *argc, char ***argv) { (void)argc; (void)argv; }
static inline void bmpi_finalize(void) {}
static inline int  bmpi_rank(void) { return 0; }
static inline int  bmpi_size(void) { return 1; }
static inline void bmpi_barrier(void) {}
static inline double bmpi_sum_d(double v) { return v; }
static inline double bmpi_max_d(double v) { return v; }
static inline double bmpi_min_d(double v) { return v; }
static inline int    bmpi_max_i(int v)    { return v; }
static inline void bmpi_bcast_i(int *v)   { (void)v; }
static inline int64_t bmpi_gatherv_d(const double *send, int count,
                                     double *recv, int *rcounts, int *displs) {
    rcounts[0] = count; displs[0] = 0;
    memcpy(recv, send, (size_t)count * sizeof(double));
    return count;
}
static inline int64_t bmpi_gatherv_i64(const int64_t *send, int count,
                                       int64_t *recv, int *rcounts, int *displs) {
    rcounts[0] = count; displs[0] = 0;
    memcpy(recv, send, (size_t)count * sizeof(int64_t));
    return count;
}

#endif // HAVE_MPI
#endif // BENCH_MPI_H
