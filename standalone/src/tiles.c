#include "tiles.h"
#include "bench_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "bench: out of memory (%zu bytes)\n", bytes); exit(1); }
    return p;
}

// ---- index ---------------------------------------------------------------
typedef struct {
    int64_t id, ntris, nnodes, off;
    double  xmin, ymin, xmax, ymax;
    char    file[256];
} tile_rec;

// ---- vertex hash: exact (x, y) bit patterns -> global vertex id ----------
typedef struct { uint64_t *kx, *ky; int64_t *val; uint64_t mask; } vhash;

static uint64_t bits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static uint64_t hmix(uint64_t x, uint64_t y) {
    uint64_t h = x * 0x9E3779B97F4A7C15ull ^ (y + 0x7F4A7C15ull) * 0xC2B2AE3D27D4EB4Full;
    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32;
    return h;
}
static void vhash_init(vhash *H, int64_t n) {
    uint64_t cap = 16; while (cap < (uint64_t)(2 * n)) cap <<= 1;
    H->kx = (uint64_t *)xmalloc(cap * 8); H->ky = (uint64_t *)xmalloc(cap * 8);
    H->val = (int64_t *)xmalloc(cap * 8);
    for (uint64_t i = 0; i < cap; i++) H->val[i] = -1;
    H->mask = cap - 1;
}
// returns the id; inserts with *next if absent (and increments it)
static int64_t vhash_get_or_add(vhash *H, double x, double y, int64_t *next) {
    const uint64_t bx = bits(x), by = bits(y);
    uint64_t i = hmix(bx, by) & H->mask;
    for (;;) {
        if (H->val[i] < 0) { H->kx[i] = bx; H->ky[i] = by; H->val[i] = (*next)++; return H->val[i]; }
        if (H->kx[i] == bx && H->ky[i] == by) return H->val[i];
        i = (i + 1) & H->mask;
    }
}
static void vhash_free(vhash *H) { free(H->kx); free(H->ky); free(H->val); }

// ---- one loaded tile ----------------------------------------------------
typedef struct {
    int t;                    // index row
    bench_mesh m;
    double *bed, *stage;
    int64_t *gv;              // [m.num_nodes] global vertex id
} loaded_tile;

static int bbox_touch(const tile_rec *a, const tile_rec *b, double tol) {
    return a->xmin <= b->xmax + tol && b->xmin <= a->xmax + tol &&
           a->ymin <= b->ymax + tol && b->ymin <= a->ymax + tol;
}

void bench_mesh_load_tiles(const char *index_path, const char *assign_path,
                           int rank, int nprocs,
                           bench_mesh *M, double **bed_node, double **stage_node,
                           bench_tiles *T) {
    memset(T, 0, sizeof(*T));
    // ---- index -----------------------------------------------------------
    FILE *fp = fopen(index_path, "r");
    if (!fp) { perror(index_path); exit(1); }
    int ntiles = 0; long long nglobal = 0;
    if (fscanf(fp, "%d %lld", &ntiles, &nglobal) != 2) {
        fprintf(stderr, "bench: bad tile index %s\n", index_path); exit(1);
    }
    tile_rec *R = (tile_rec *)xmalloc((size_t)ntiles * sizeof(tile_rec));
    for (int t = 0; t < ntiles; t++) {
        long long id, nt, nn, off;
        if (fscanf(fp, "%lld %lld %lld %lld %lf %lf %lf %lf %255s", &id, &nt, &nn, &off,
                   &R[t].xmin, &R[t].ymin, &R[t].xmax, &R[t].ymax, R[t].file) != 9) {
            fprintf(stderr, "bench: bad tile index row %d in %s\n", t, index_path); exit(1);
        }
        R[t].id = id; R[t].ntris = nt; R[t].nnodes = nn; R[t].off = off;
    }
    fclose(fp);
    T->ntiles = ntiles;
    T->n_global = nglobal;
    char dir[1024]; strncpy(dir, index_path, sizeof dir - 1); dir[sizeof dir - 1] = 0;
    { char *s = strrchr(dir, '/'); if (s) s[1] = 0; else dir[0] = 0; }

    // ---- assignment -------------------------------------------------------
    int *assign = (int *)xmalloc((size_t)ntiles * sizeof(int));
    if (assign_path) {
        FILE *fa = fopen(assign_path, "r");
        if (!fa) { perror(assign_path); exit(1); }
        for (int t = 0; t < ntiles; t++) {
            if (fscanf(fa, "%d", &assign[t]) != 1 || assign[t] < 0 || assign[t] >= nprocs) {
                fprintf(stderr, "bench: bad rank for tile %d in %s\n", t, assign_path); exit(1);
            }
        }
        fclose(fa);
    } else {
        // contiguous in tile order, balanced by triangle count
        for (int t = 0; t < ntiles; t++) {
            const long long mid = R[t].off + R[t].ntris / 2;
            assign[t] = (int)((mid * nprocs) / (nglobal > 0 ? nglobal : 1));
            if (assign[t] >= nprocs) assign[t] = nprocs - 1;
        }
    }

    // ---- which tiles to read: mine + bbox-touching foreign ---------------
    const double tol = 1.0e-3;
    int *need = (int *)calloc((size_t)ntiles, sizeof(int));   // 1 own, 2 neighbour
    int n_own_tiles = 0;
    for (int t = 0; t < ntiles; t++) if (assign[t] == rank) { need[t] = 1; n_own_tiles++; }
    for (int t = 0; t < ntiles; t++) {
        if (need[t] == 1) continue;
        for (int u = 0; u < ntiles; u++)
            if (need[u] == 1 && bbox_touch(&R[t], &R[u], tol)) { need[t] = 2; break; }
    }
    int nload = 0; for (int t = 0; t < ntiles; t++) if (need[t]) nload++;
    loaded_tile *L = (loaded_tile *)xmalloc((size_t)(nload ? nload : 1) * sizeof(loaded_tile));
    int64_t total_nodes = 0;
    {
        int j = 0;
        for (int t = 0; t < ntiles; t++) {
            if (!need[t]) continue;
            char path[1300]; snprintf(path, sizeof path, "%s%s", dir, R[t].file);
            L[j].t = t;
            bench_mesh_load(path, &L[j].m, &L[j].bed, &L[j].stage);
            if (L[j].m.num_triangles != R[t].ntris) {
                fprintf(stderr, "bench: tile %s has %lld triangles, index says %lld\n", path,
                        (long long)L[j].m.num_triangles, (long long)R[t].ntris); exit(1);
            }
            total_nodes += L[j].m.num_nodes;
            j++;
        }
    }

    // ---- stitch: global vertex ids on exact coordinates ------------------
    vhash H; vhash_init(&H, total_nodes);
    int64_t nglob_v = 0;
    for (int j = 0; j < nload; j++) {
        const int64_t nn = L[j].m.num_nodes;
        L[j].gv = (int64_t *)xmalloc((size_t)nn * sizeof(int64_t));
        for (int64_t v = 0; v < nn; v++)
            L[j].gv[v] = vhash_get_or_add(&H, L[j].m.nodes[2 * v], L[j].m.nodes[2 * v + 1], &nglob_v);
    }
    vhash_free(&H);

    // ---- owned triangles first (tile order, local order) -----------------
    int64_t n_own = 0;
    for (int j = 0; j < nload; j++) if (need[L[j].t] == 1) n_own += L[j].m.num_triangles;
    unsigned char *touched = (unsigned char *)calloc((size_t)nglob_v, 1);
    T->ntiles_own = n_own_tiles;
    T->own_tile_id = (int *)xmalloc((size_t)(n_own_tiles ? n_own_tiles : 1) * sizeof(int));
    T->own_tile_start = (int64_t *)xmalloc((size_t)(n_own_tiles + 1) * sizeof(int64_t));
    // (tile j, k) pairs for the final local order: owned then ghosts
    int64_t cap = n_own;
    int *tri_tile = (int *)xmalloc((size_t)(cap ? cap : 1) * sizeof(int));
    int64_t *tri_k = (int64_t *)xmalloc((size_t)(cap ? cap : 1) * sizeof(int64_t));
    int64_t n_local = 0;
    {
        int ot = 0;
        for (int j = 0; j < nload; j++) {
            if (need[L[j].t] != 1) continue;
            T->own_tile_id[ot] = (int)R[L[j].t].id;
            T->own_tile_start[ot] = n_local;
            ot++;
            for (int64_t k = 0; k < L[j].m.num_triangles; k++) {
                tri_tile[n_local] = j; tri_k[n_local] = k; n_local++;
                for (int c = 0; c < 3; c++) touched[L[j].gv[L[j].m.triangles[3 * k + c]]] = 1;
            }
        }
        T->own_tile_start[ot] = n_local;
    }
    // ---- ghosts: foreign triangles sharing a vertex with an owned one ----
    for (int j = 0; j < nload; j++) {
        if (need[L[j].t] != 2) continue;
        for (int64_t k = 0; k < L[j].m.num_triangles; k++) {
            int hit = 0;
            for (int c = 0; c < 3 && !hit; c++) hit = touched[L[j].gv[L[j].m.triangles[3 * k + c]]];
            if (!hit) continue;
            if (n_local == cap) {
                cap = cap * 2 + 1024;
                tri_tile = (int *)realloc(tri_tile, (size_t)cap * sizeof(int));
                tri_k = (int64_t *)realloc(tri_k, (size_t)cap * sizeof(int64_t));
            }
            tri_tile[n_local] = j; tri_k[n_local] = k; n_local++;
        }
    }
    const int64_t n_ghost = n_local - n_own;

    // ---- halo lists ---------------------------------------------------------
    // recv from rank r: ghosts of r's tiles, in local (= tile, k) order.
    // send to rank r: owned triangles sharing a vertex with any loaded tile
    // of r, in local order.  r computes the mirror sets identically.
    int *rank_seen = (int *)calloc((size_t)nprocs, sizeof(int));
    for (int64_t q = n_own; q < n_local; q++) rank_seen[assign[L[tri_tile[q]].t]] = 1;
    int nnb = 0; for (int r = 0; r < nprocs; r++) if (rank_seen[r]) nnb++;
    T->nnb = nnb;
    T->nb_ranks = (int *)xmalloc((size_t)(nnb ? nnb : 1) * sizeof(int));
    T->scnt = (int *)calloc((size_t)(nnb ? nnb : 1), sizeof(int));
    T->rcnt = (int *)calloc((size_t)(nnb ? nnb : 1), sizeof(int));
    { int i = 0; for (int r = 0; r < nprocs; r++) if (rank_seen[r]) T->nb_ranks[i++] = r; }
    int64_t *stamp = (int64_t *)xmalloc((size_t)(nglob_v ? nglob_v : 1) * sizeof(int64_t));
    for (int64_t v = 0; v < nglob_v; v++) stamp[v] = -1;
    int *fsend = NULL, *frecv = NULL; int64_t ns = 0, nr = 0, scap = 0, rcap = 0;
    for (int i = 0; i < nnb; i++) {
        const int r = T->nb_ranks[i];
        // recv: ghosts owned by r
        for (int64_t q = n_own; q < n_local; q++) {
            if (assign[L[tri_tile[q]].t] != r) continue;
            if (nr == rcap) { rcap = rcap * 2 + 1024; frecv = (int *)realloc(frecv, (size_t)rcap * sizeof(int)); }
            frecv[nr++] = (int)q; T->rcnt[i]++;
        }
        // send: owned triangles touching r's loaded tiles
        for (int j = 0; j < nload; j++) {
            if (need[L[j].t] != 2 || assign[L[j].t] != r) continue;
            for (int64_t v = 0; v < L[j].m.num_nodes; v++) stamp[L[j].gv[v]] = r;
        }
        for (int64_t q = 0; q < n_own; q++) {
            const loaded_tile *lt = &L[tri_tile[q]];
            int hit = 0;
            for (int c = 0; c < 3 && !hit; c++) hit = (stamp[lt->gv[lt->m.triangles[3 * tri_k[q] + c]]] == r);
            if (!hit) continue;
            if (ns == scap) { scap = scap * 2 + 1024; fsend = (int *)realloc(fsend, (size_t)scap * sizeof(int)); }
            fsend[ns++] = (int)q; T->scnt[i]++;
        }
    }
    free(stamp); free(rank_seen); free(touched);
    T->fsend = fsend ? fsend : (int *)xmalloc(4);
    T->frecv = frecv ? frecv : (int *)xmalloc(4);
    // The mirror rank must have derived the same counts; a mismatch would
    // otherwise hang or silently corrupt the exchange.
    for (int i = 0; i < nnb; i++) {
        const int their_send = bmpi_sendrecv_i(T->scnt[i], T->nb_ranks[i], T->nb_ranks[i]);
        if (their_send != T->rcnt[i]) {
            fprintf(stderr, "bench: halo count mismatch rank %d <-> %d: I expect %d, they send %d\n",
                    rank, T->nb_ranks[i], T->rcnt[i], their_send);
            exit(1);
        }
    }

    // ---- assemble the local bench_mesh -------------------------------------
    int64_t *vmap = (int64_t *)xmalloc((size_t)(nglob_v ? nglob_v : 1) * sizeof(int64_t));
    for (int64_t v = 0; v < nglob_v; v++) vmap[v] = -1;
    int64_t nn_local = 0;
    M->triangles = (int64_t *)xmalloc((size_t)3 * n_local * sizeof(int64_t));
    M->orig_id   = (int64_t *)xmalloc((size_t)n_local * sizeof(int64_t));
    T->tri_full_flag = (int64_t *)xmalloc((size_t)n_local * sizeof(int64_t));
    for (int64_t q = 0; q < n_local; q++) {
        const loaded_tile *lt = &L[tri_tile[q]];
        for (int c = 0; c < 3; c++) {
            const int64_t g = lt->gv[lt->m.triangles[3 * tri_k[q] + c]];
            if (vmap[g] < 0) vmap[g] = nn_local++;
            M->triangles[3 * q + c] = vmap[g];
        }
        M->orig_id[q] = R[lt->t].off + tri_k[q];
        T->tri_full_flag[q] = (q < n_own) ? 1 : 0;
    }
    M->num_nodes = nn_local;
    M->num_triangles = n_local;
    M->nodes = (double *)xmalloc((size_t)2 * nn_local * sizeof(double));
    *bed_node   = (double *)xmalloc((size_t)nn_local * sizeof(double));
    *stage_node = (double *)xmalloc((size_t)nn_local * sizeof(double));
    for (int j = 0; j < nload; j++) {
        for (int64_t v = 0; v < L[j].m.num_nodes; v++) {
            const int64_t l = vmap[L[j].gv[v]];
            if (l < 0) continue;
            M->nodes[2 * l] = L[j].m.nodes[2 * v]; M->nodes[2 * l + 1] = L[j].m.nodes[2 * v + 1];
            (*bed_node)[l] = L[j].bed[v]; (*stage_node)[l] = L[j].stage[v];
        }
    }
    // Boundary edges are enumerated by setup.c from the stitched connectivity
    // (every neighbour-less edge, reflective): the true outline for owned
    // cells, and the cut rim of the ghost strip, whose fluxes are discarded.
    M->num_boundary = 0; M->boundary_tri = NULL; M->boundary_edge = NULL; M->boundary_tag = NULL;
    T->n_own = n_own;

    for (int j = 0; j < nload; j++) {
        bench_mesh_free(&L[j].m); free(L[j].bed); free(L[j].stage); free(L[j].gv);
    }
    free(L); free(vmap); free(tri_tile); free(tri_k); free(need); free(assign); free(R);

    if (rank == 0 || n_ghost < 0)
        fprintf(stderr, "tiles: rank %d owns %d of %d tiles, %lld triangles + %lld ghosts, %d neighbour rank(s)\n",
                rank, n_own_tiles, ntiles, (long long)n_own, (long long)n_ghost, nnb);
}

void bench_tiles_free(bench_tiles *T) {
    free(T->tri_full_flag); free(T->nb_ranks); free(T->scnt); free(T->rcnt);
    free(T->fsend); free(T->frecv); free(T->own_tile_id); free(T->own_tile_start);
    memset(T, 0, sizeof(*T));
}
