// Binary state snapshots: save a run's final centroid state, or diff a run
// against a previously saved one.  The same format is written and read by
// tools/anuga_reference.py, so a standalone run can be compared against a
// Python ANUGA run of the same case.

#ifndef ANUGA_BENCH_SNAPSHOT_H
#define ANUGA_BENCH_SNAPSHOT_H

#include <stdint.h>
#include "gpu_domain.h"

#define SNAP_MAGIC   "ANUGASNP"
#define SNAP_VERSION 1

typedef struct {
    char    magic[8];
    int32_t version;
    int32_t which_case;
    int64_t n;             // number of triangles
    int64_t nb;            // boundary length
    int64_t nx, ny;        // mesh resolution
    int64_t total_steps;   // warmup + timed steps actually taken
    double  t;             // simulated time reached
    double  last_dt;
} snap_header;

// Field order in the payload, 5 * n doubles.
enum { SNAP_STAGE = 0, SNAP_XMOM, SNAP_YMOM, SNAP_HEIGHT, SNAP_BED, SNAP_NFIELDS };

// orig_id: optional triangle permutation (orig_id[k] = canonical index of the
// run's triangle k, from bench_mesh.orig_id).  Snapshots are always written in
// CANONICAL order, so runs with different element orderings diff bit-for-bit.
// Pass NULL when the run already uses canonical (row-major) order.
int snapshot_save(const char *path, const struct gpu_domain *GD,
                  const int64_t *orig_id,
                  int64_t nx, int64_t ny, int which_case,
                  int64_t total_steps, double t, double last_dt);

// Returns 0 if every field is within tolerance, 1 otherwise (or on error).
// Always prints a per-field diff table.
int snapshot_check(const char *path, const struct gpu_domain *GD,
                   const int64_t *orig_id,
                   double rtol, double atol);

#endif // ANUGA_BENCH_SNAPSHOT_H
