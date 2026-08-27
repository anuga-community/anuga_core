// Tiled distributed mesh: each rank loads the .bmesh tiles assigned to it
// (tools/tiles_to_bmesh.py), stitches them on exact vertex coordinates,
// and takes its ghost layer from the neighbouring tiles it also reads --
// so no rank ever sees the global mesh and no global partitioner runs.
//
// Ghost rule: every triangle of a foreign tile that shares a VERTEX with an
// owned triangle.  That is a superset of the 2-edge-hop ring the RK2 step
// needs (a triangle two edge-hops away always shares a vertex with the
// origin), so owned cells see correct centroids at distance <= 2 and
// correct edge values at distance 1 after each exchange.  Both sides of a
// rank pair compute the same set from the same tile data, in the same
// (tile id, local index) order, which is what makes the send/recv lists
// agree without any communication.
#pragma once
#include <stdint.h>
#include "mesh.h"

typedef struct {
    int64_t  n_own;           // owned triangles = local ids [0, n_own)
    int64_t  n_global;        // triangles over all tiles
    int64_t *tri_full_flag;   // [num_triangles] 1 owned, 0 ghost
    // halo lists in gpu_halo_init's flat form
    int  nnb;
    int *nb_ranks, *scnt, *rcnt, *fsend, *frecv;
    // owned tiles, for per-tile statistics: tile ids and local id ranges
    int      ntiles_own;
    int     *own_tile_id;
    int64_t *own_tile_start;  // [ntiles_own + 1]
    int      ntiles;          // in the index
} bench_tiles;

// assign_path: text file with one rank per tile in index order, or NULL for
// a contiguous split balanced by triangle count.
void bench_mesh_load_tiles(const char *index_path, const char *assign_path,
                           int rank, int nprocs,
                           bench_mesh *M, double **bed_node, double **stage_node,
                           bench_tiles *T);
void bench_tiles_free(bench_tiles *T);
