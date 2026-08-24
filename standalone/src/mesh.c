#include "mesh.h"

#include <stdio.h>
#include <stdlib.h>

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "bench: out of memory (%zu bytes)\n", bytes);
        exit(1);
    }
    return p;
}

void bench_mesh_rectangular_cross(int64_t m, int64_t n,
                                  double len1, double len2,
                                  double x0, double y0,
                                  bench_mesh *M) {
    const double delta1 = len1 / (double)m;
    const double delta2 = len2 / (double)n;

    // (m+1)*(n+1) grid nodes, then one centre node per cell -- appended in the
    // same (i, j) loop order the Python factory uses.
    const int64_t num_grid   = (m + 1) * (n + 1);
    const int64_t num_nodes  = num_grid + m * n;
    const int64_t num_tris   = 4 * m * n;
    const int64_t num_bdry   = 2 * (m + n);

    M->num_nodes     = num_nodes;
    M->num_triangles = num_tris;
    M->nodes         = (double  *)xmalloc((size_t)2 * num_nodes * sizeof(double));
    M->triangles     = (int64_t *)xmalloc((size_t)3 * num_tris  * sizeof(int64_t));
    M->num_boundary  = num_bdry;
    M->boundary_tri  = (int64_t *)xmalloc((size_t)num_bdry * sizeof(int64_t));
    M->boundary_edge = (int64_t *)xmalloc((size_t)num_bdry * sizeof(int64_t));
    M->boundary_tag  = (int     *)xmalloc((size_t)num_bdry * sizeof(int));
    M->orig_id       = NULL;

    // Grid nodes: vertices[i][j] = i*(n+1) + j
    for (int64_t i = 0; i <= m; i++) {
        for (int64_t j = 0; j <= n; j++) {
            int64_t v = i * (n + 1) + j;
            M->nodes[2 * v + 0] = delta1 * (double)i + x0;
            M->nodes[2 * v + 1] = delta2 * (double)j + y0;
        }
    }

    int64_t nb = 0;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            const int64_t v1 = i * (n + 1) + (j + 1);        // (i,   j+1)
            const int64_t v2 = i * (n + 1) + j;              // (i,   j)
            const int64_t v3 = (i + 1) * (n + 1) + (j + 1);  // (i+1, j+1)
            const int64_t v4 = (i + 1) * (n + 1) + j;        // (i+1, j)
            const int64_t v5 = num_grid + i * n + j;         // cell centre

            M->nodes[2 * v5 + 0] = 0.25 * (M->nodes[2 * v1 + 0] + M->nodes[2 * v2 + 0] +
                                           M->nodes[2 * v3 + 0] + M->nodes[2 * v4 + 0]);
            M->nodes[2 * v5 + 1] = 0.25 * (M->nodes[2 * v1 + 1] + M->nodes[2 * v2 + 1] +
                                           M->nodes[2 * v3 + 1] + M->nodes[2 * v4 + 1]);

            const int64_t base = 4 * (i * n + j);

            // left, bottom, right, top -- edge 1 of each is the outer edge.
            M->triangles[3 * (base + 0) + 0] = v2;
            M->triangles[3 * (base + 0) + 1] = v5;
            M->triangles[3 * (base + 0) + 2] = v1;

            M->triangles[3 * (base + 1) + 0] = v4;
            M->triangles[3 * (base + 1) + 1] = v5;
            M->triangles[3 * (base + 1) + 2] = v2;

            M->triangles[3 * (base + 2) + 0] = v3;
            M->triangles[3 * (base + 2) + 1] = v5;
            M->triangles[3 * (base + 2) + 2] = v4;

            M->triangles[3 * (base + 3) + 0] = v1;
            M->triangles[3 * (base + 3) + 1] = v5;
            M->triangles[3 * (base + 3) + 2] = v3;

            if (i == 0) {
                M->boundary_tri[nb] = base + 0;
                M->boundary_edge[nb] = 1;
                M->boundary_tag[nb++] = BTAG_LEFT;
            }
            if (j == 0) {
                M->boundary_tri[nb] = base + 1;
                M->boundary_edge[nb] = 1;
                M->boundary_tag[nb++] = BTAG_BOTTOM;
            }
            if (i == m - 1) {
                M->boundary_tri[nb] = base + 2;
                M->boundary_edge[nb] = 1;
                M->boundary_tag[nb++] = BTAG_RIGHT;
            }
            if (j == n - 1) {
                M->boundary_tri[nb] = base + 3;
                M->boundary_edge[nb] = 1;
                M->boundary_tag[nb++] = BTAG_TOP;
            }
        }
    }

    if (nb != num_bdry) {
        fprintf(stderr, "bench: internal error, %lld boundary edges (expected %lld)\n",
                (long long)nb, (long long)num_bdry);
        exit(1);
    }
}

// Interleave the low 32 bits of i and j -> 64-bit Morton key.
static uint64_t morton2(uint64_t i, uint64_t j) {
    uint64_t out = 0;
    for (int b = 0; b < 32; b++) {
        out |= ((i >> b) & 1ull) << (2 * b);
        out |= ((j >> b) & 1ull) << (2 * b + 1);
    }
    return out;
}

typedef struct { uint64_t key; int64_t cell; } morton_entry;

static int morton_cmp(const void *a, const void *b) {
    uint64_t ka = ((const morton_entry *)a)->key;
    uint64_t kb = ((const morton_entry *)b)->key;
    return (ka > kb) - (ka < kb);
}

void bench_mesh_reorder_morton(bench_mesh *M, int64_t m, int64_t n) {
    const int64_t ncells = m * n;
    const int64_t ntris  = M->num_triangles;

    morton_entry *order = (morton_entry *)xmalloc((size_t)ncells * sizeof(morton_entry));
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            const int64_t c = i * n + j;             // canonical cell index
            order[c].key  = morton2((uint64_t)i, (uint64_t)j);
            order[c].cell = c;
        }
    qsort(order, (size_t)ncells, sizeof(morton_entry), morton_cmp);

    // new_of_old[old triangle id] -> new triangle id
    int64_t *new_of_old = (int64_t *)xmalloc((size_t)ntris * sizeof(int64_t));
    int64_t *orig_id    = (int64_t *)xmalloc((size_t)ntris * sizeof(int64_t));
    for (int64_t c = 0; c < ncells; c++)
        for (int t = 0; t < 4; t++) {
            const int64_t old_id = 4 * order[c].cell + t;
            const int64_t new_id = 4 * c + t;
            new_of_old[old_id] = new_id;
            orig_id[new_id]    = old_id;
        }
    free(order);

    int64_t *tris = (int64_t *)xmalloc((size_t)3 * ntris * sizeof(int64_t));
    for (int64_t k = 0; k < ntris; k++)
        for (int v = 0; v < 3; v++)
            tris[3 * k + v] = M->triangles[3 * orig_id[k] + v];
    free(M->triangles);
    M->triangles = tris;

    for (int64_t b = 0; b < M->num_boundary; b++)
        M->boundary_tri[b] = new_of_old[M->boundary_tri[b]];

    free(new_of_old);
    free(M->orig_id);
    M->orig_id = orig_id;
}

// Apply an arbitrary cell permutation: perm[new_cell] = old_cell.
static void apply_cell_permutation(bench_mesh *M, const int64_t *perm, int64_t ncells) {
    const int64_t ntris = M->num_triangles;

    int64_t *new_of_old = (int64_t *)xmalloc((size_t)ntris * sizeof(int64_t));
    int64_t *orig_id    = (int64_t *)xmalloc((size_t)ntris * sizeof(int64_t));
    for (int64_t c = 0; c < ncells; c++)
        for (int t = 0; t < 4; t++) {
            const int64_t old_id = 4 * perm[c] + t;
            const int64_t new_id = 4 * c + t;
            new_of_old[old_id] = new_id;
            orig_id[new_id]    = old_id;
        }

    int64_t *tris = (int64_t *)xmalloc((size_t)3 * ntris * sizeof(int64_t));
    for (int64_t k = 0; k < ntris; k++)
        for (int v = 0; v < 3; v++)
            tris[3 * k + v] = M->triangles[3 * orig_id[k] + v];
    free(M->triangles);
    M->triangles = tris;

    for (int64_t b = 0; b < M->num_boundary; b++)
        M->boundary_tri[b] = new_of_old[M->boundary_tri[b]];

    free(new_of_old);
    free(M->orig_id);
    M->orig_id = orig_id;
}

void bench_mesh_reorder_random(bench_mesh *M, int64_t m, int64_t n) {
    const int64_t ncells = m * n;
    int64_t *perm = (int64_t *)xmalloc((size_t)ncells * sizeof(int64_t));
    for (int64_t c = 0; c < ncells; c++) perm[c] = c;

    // Deterministic 64-bit LCG Fisher-Yates (fixed seed: runs reproduce)
    unsigned long long state = 0x9E3779B97F4A7C15ull;
    for (int64_t c = ncells - 1; c > 0; c--) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const int64_t j = (int64_t)((state >> 17) % (unsigned long long)(c + 1));
        int64_t tmp = perm[c]; perm[c] = perm[j]; perm[j] = tmp;
    }
    apply_cell_permutation(M, perm, ncells);
    free(perm);
}

void bench_mesh_free(bench_mesh *M) {
    free(M->nodes);
    free(M->triangles);
    free(M->boundary_tri);
    free(M->boundary_edge);
    free(M->boundary_tag);
    free(M->orig_id);
    M->orig_id = NULL;
    M->nodes = NULL;
    M->triangles = NULL;
    M->boundary_tri = NULL;
    M->boundary_edge = NULL;
    M->boundary_tag = NULL;
}
