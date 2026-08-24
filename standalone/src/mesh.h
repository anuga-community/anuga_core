// Mesh generation for the standalone ANUGA shallow-water benchmark.
//
// Reproduces anuga.abstract_2d_finite_volumes.mesh_factory.rectangular_cross
// exactly (same node ordering, same triangle ordering, same boundary tags) so
// that a run here can be compared element-for-element with a Python ANUGA run
// on the same mesh.

#ifndef ANUGA_BENCH_MESH_H
#define ANUGA_BENCH_MESH_H

#include <stdint.h>

// Boundary tags, matching the strings rectangular_cross() emits.
enum { BTAG_LEFT = 0, BTAG_BOTTOM = 1, BTAG_RIGHT = 2, BTAG_TOP = 3 };

typedef struct {
    int64_t  num_nodes;
    int64_t  num_triangles;
    double  *nodes;          // [2*num_nodes]     x, y
    int64_t *triangles;      // [3*num_triangles] node indices

    int64_t  num_boundary;   // number of tagged boundary edges
    int64_t *boundary_tri;   // [num_boundary]
    int64_t *boundary_edge;  // [num_boundary]
    int     *boundary_tag;   // [num_boundary]

    // After bench_mesh_reorder_morton: orig_id[k] = the triangle's index in
    // the canonical (row-major) ordering.  NULL when unpermuted.  Snapshots
    // are always written in canonical order via this map, so runs with
    // different orderings can be diffed bit-for-bit.
    int64_t *orig_id;
} bench_mesh;

// m x n cells, each split into 4 triangles about a centre node.
// Domain is [x0, x0+len1] x [y0, y0+len2].
void bench_mesh_rectangular_cross(int64_t m, int64_t n,
                                  double len1, double len2,
                                  double x0, double y0,
                                  bench_mesh *M);

// Renumber the triangles along a Morton (Z-order) curve over the (i, j) cell
// grid, keeping each cell's 4 triangles together.  Purely a permutation: the
// same mesh, the same physics, but neighbouring triangles get nearby indices
// in BOTH grid directions, instead of the row-major ordering's ~4*n stride to
// the +/-i neighbours -- which at large n is a guaranteed cache miss on every
// neighbour gather in the flux and extrapolation kernels.
void bench_mesh_reorder_morton(bench_mesh *M, int64_t m, int64_t n);

// Renumber the CELLS randomly (deterministic LCG shuffle, fixed seed), still
// keeping each cell's 4 triangles together.  This is the pessimistic locality
// bound: a real unstructured flood mesh (mesher-ordered, variable resolution)
// sits between row-major (the optimistic bound) and this.
void bench_mesh_reorder_random(bench_mesh *M, int64_t m, int64_t n);

// Load a mesh dumped by tools/make_basin_mesh.py ("ANUGAMSH" v1): nodes,
// triangles, boundary (all tags treated as reflective), and per-NODE bed and
// initial-stage values (malloc'd here; caller frees).  orig_id starts NULL.
void bench_mesh_load(const char *path, bench_mesh *M,
                     double **bed_node, double **stage_node);

// Triangle-granularity reorderings for loaded (non-grid) meshes: Morton on
// quantized centroids, or a fixed-seed random shuffle.  Both maintain
// orig_id so snapshots stay canonical.
void bench_mesh_reorder_tris_morton(bench_mesh *M);
void bench_mesh_reorder_tris_random(bench_mesh *M);

void bench_mesh_free(bench_mesh *M);

#endif // ANUGA_BENCH_MESH_H
