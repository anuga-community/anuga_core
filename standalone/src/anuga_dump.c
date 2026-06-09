/* anuga_dump.c - reader/loader for the .adm partition-dump format (anuga_dump.h). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "anuga_dump.h"

struct AnugaLoadedDomain {
    AnugaDomain *dom;
    void **allocs;
    int nallocs, cap;
    int64_t n, nb, rank, nprocs;
    double finaltime, yieldstep;
    int method;
    int64_t *tri_full_flag;   /* aliased */
    double  *stage_centroid;  /* aliased */
    int64_t *global_ids;      /* aliased; local cell -> global id (partition build) */
};

static void track(AnugaLoadedDomain *ld, void *p) {
    if (ld->nallocs == ld->cap) {
        ld->cap = ld->cap ? ld->cap * 2 : 64;
        ld->allocs = (void **)realloc(ld->allocs, ld->cap * sizeof(void *));
    }
    ld->allocs[ld->nallocs++] = p;
}

static void *xcalloc(AnugaLoadedDomain *ld, size_t count, size_t sz) {
    void *p = calloc(count ? count : 1, sz);
    track(ld, p);
    return p;
}

/* read `count` elements of size `sz` into a fresh tracked buffer */
static void *rdbuf(AnugaLoadedDomain *ld, FILE *f, long count, size_t sz, int *err) {
    void *p = xcalloc(ld, (size_t)count, sz);
    if (count > 0 && fread(p, sz, (size_t)count, f) != (size_t)count) *err = 1;
    return p;
}

AnugaLoadedDomain *anuga_dump_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "anuga_dump_load: cannot open %s\n", path); return NULL; }

    AnugaLoadedDomain *ld = (AnugaLoadedDomain *)calloc(1, sizeof(*ld));
    int err = 0;

    char magic[8];
    int64_t version;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, ANUGA_DUMP_MAGIC, 8) != 0) {
        fprintf(stderr, "anuga_dump_load: bad magic in %s\n", path);
        fclose(f); free(ld); return NULL;
    }
    int64_t H[16];
    fread(&version, sizeof(int64_t), 1, f);
    if (fread(H, sizeof(int64_t), 16, f) != 16) err = 1;
    int64_t n   = H[0],  nb = H[1];
    ld->n = n; ld->nb = nb; ld->rank = H[2]; ld->nprocs = H[3];
    int64_t optimise_dry_cells = H[4], extrap2 = H[5], low_froude = H[6],
            timestep_fluxcalls = H[7];
    ld->method = (int)H[8];
    int64_t num_neighbors = H[9], total_send = H[10], total_recv = H[11];
    int64_t n_reflective = H[12], n_dirichlet = H[13], n_transmissive = H[14],
            transmissive_use_centroid = H[15];

    double D[20];
    if (fread(D, sizeof(double), 20, f) != 20) err = 1;
    ld->finaltime = D[15]; ld->yieldstep = D[16];

    /* ---- arrays (order matches anuga_dump.h / partition_dump.py) ---- */
    int64_t *neighbours           = rdbuf(ld, f, 3 * n, sizeof(int64_t), &err);
    int64_t *neighbour_edges      = rdbuf(ld, f, 3 * n, sizeof(int64_t), &err);
    int64_t *surrogate_neighbours = rdbuf(ld, f, 3 * n, sizeof(int64_t), &err);
    int64_t *number_of_boundaries = rdbuf(ld, f, n,     sizeof(int64_t), &err);
    int64_t *tri_full_flag        = rdbuf(ld, f, n,     sizeof(int64_t), &err);

    double *normals              = rdbuf(ld, f, 6 * n, sizeof(double), &err);
    double *edgelengths          = rdbuf(ld, f, 3 * n, sizeof(double), &err);
    double *radii                = rdbuf(ld, f, n,     sizeof(double), &err);
    double *areas                = rdbuf(ld, f, n,     sizeof(double), &err);
    double *centroid_coordinates = rdbuf(ld, f, 2 * n, sizeof(double), &err);
    double *edge_coordinates     = rdbuf(ld, f, 6 * n, sizeof(double), &err);

    double *stage_centroid  = rdbuf(ld, f, n,     sizeof(double), &err);
    double *stage_edge      = rdbuf(ld, f, 3 * n, sizeof(double), &err);
    double *xmom_centroid   = rdbuf(ld, f, n,     sizeof(double), &err);
    double *xmom_edge       = rdbuf(ld, f, 3 * n, sizeof(double), &err);
    double *ymom_centroid   = rdbuf(ld, f, n,     sizeof(double), &err);
    double *ymom_edge       = rdbuf(ld, f, 3 * n, sizeof(double), &err);
    double *bed_centroid    = rdbuf(ld, f, n,     sizeof(double), &err);
    double *bed_edge        = rdbuf(ld, f, 3 * n, sizeof(double), &err);
    double *height_centroid = rdbuf(ld, f, n,     sizeof(double), &err);
    double *height_edge     = rdbuf(ld, f, 3 * n, sizeof(double), &err);
    double *friction_centroid = rdbuf(ld, f, n,   sizeof(double), &err);

    int32_t *neighbor_ranks = rdbuf(ld, f, num_neighbors, sizeof(int32_t), &err);
    int32_t *send_counts    = rdbuf(ld, f, num_neighbors, sizeof(int32_t), &err);
    int32_t *recv_counts    = rdbuf(ld, f, num_neighbors, sizeof(int32_t), &err);
    int32_t *flat_send      = rdbuf(ld, f, total_send,    sizeof(int32_t), &err);
    int32_t *flat_recv      = rdbuf(ld, f, total_recv,    sizeof(int32_t), &err);

    int32_t *refl_bidx = rdbuf(ld, f, n_reflective, sizeof(int32_t), &err);
    int32_t *refl_vol  = rdbuf(ld, f, n_reflective, sizeof(int32_t), &err);
    int32_t *refl_edge = rdbuf(ld, f, n_reflective, sizeof(int32_t), &err);
    int32_t *dir_bidx  = rdbuf(ld, f, n_dirichlet,  sizeof(int32_t), &err);
    int32_t *dir_vol   = rdbuf(ld, f, n_dirichlet,  sizeof(int32_t), &err);
    int32_t *dir_edge  = rdbuf(ld, f, n_dirichlet,  sizeof(int32_t), &err);
    int32_t *tr_bidx   = rdbuf(ld, f, n_transmissive, sizeof(int32_t), &err);
    int32_t *tr_vol    = rdbuf(ld, f, n_transmissive, sizeof(int32_t), &err);
    int32_t *tr_edge   = rdbuf(ld, f, n_transmissive, sizeof(int32_t), &err);
    fclose(f);

    if (err) { fprintf(stderr, "anuga_dump_load: short read in %s\n", path);
               anuga_dump_free(ld); return NULL; }

    /* ---- scratch arrays the solver writes each step (allocated zeroed) ---- */
    AnugaDomainDesc d;
    memset(&d, 0, sizeof(d));
    d.number_of_elements = n;
    d.boundary_length    = nb;
    d.number_of_riverwall_edges = 0;
    d.optimise_dry_cells = optimise_dry_cells;
    d.extrapolate_velocity_second_order = extrap2;
    d.low_froude = low_froude;
    d.timestep_fluxcalls = timestep_fluxcalls;
    d.epsilon = D[0]; d.H0 = D[1]; d.g = D[2];
    d.evolve_max_timestep = D[3]; d.evolve_min_timestep = D[4];
    d.minimum_allowed_height = D[5]; d.maximum_allowed_speed = D[6];
    d.beta_w = D[7]; d.beta_w_dry = D[8]; d.beta_uh = D[9]; d.beta_uh_dry = D[10];
    d.beta_vh = D[11]; d.beta_vh_dry = D[12];
    d.CFL = D[13]; d.fixed_flux_timestep = D[14];

    d.neighbours = neighbours; d.neighbour_edges = neighbour_edges;
    d.surrogate_neighbours = surrogate_neighbours;
    d.number_of_boundaries = number_of_boundaries; d.tri_full_flag = tri_full_flag;
    d.normals = normals; d.edgelengths = edgelengths; d.radii = radii; d.areas = areas;
    d.centroid_coordinates = centroid_coordinates; d.edge_coordinates = edge_coordinates;
    d.max_speed       = xcalloc(ld, n, sizeof(double));
    d.x_centroid_work = xcalloc(ld, n, sizeof(double));
    d.y_centroid_work = xcalloc(ld, n, sizeof(double));

    d.stage_centroid_values = stage_centroid; d.stage_edge_values = stage_edge;
    d.stage_boundary_values      = xcalloc(ld, nb, sizeof(double));
    d.stage_explicit_update      = xcalloc(ld, n, sizeof(double));
    d.stage_semi_implicit_update = xcalloc(ld, n, sizeof(double));
    d.stage_backup_values        = xcalloc(ld, n, sizeof(double));

    d.xmom_centroid_values = xmom_centroid; d.xmom_edge_values = xmom_edge;
    d.xmom_boundary_values      = xcalloc(ld, nb, sizeof(double));
    d.xmom_explicit_update      = xcalloc(ld, n, sizeof(double));
    d.xmom_semi_implicit_update = xcalloc(ld, n, sizeof(double));
    d.xmom_backup_values        = xcalloc(ld, n, sizeof(double));

    d.ymom_centroid_values = ymom_centroid; d.ymom_edge_values = ymom_edge;
    d.ymom_boundary_values      = xcalloc(ld, nb, sizeof(double));
    d.ymom_explicit_update      = xcalloc(ld, n, sizeof(double));
    d.ymom_semi_implicit_update = xcalloc(ld, n, sizeof(double));
    d.ymom_backup_values        = xcalloc(ld, n, sizeof(double));

    d.bed_centroid_values = bed_centroid; d.bed_edge_values = bed_edge;
    d.bed_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.height_centroid_values = height_centroid; d.height_edge_values = height_edge;
    d.height_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.friction_centroid_values = friction_centroid;

    ld->dom = anuga_domain_create(&d);
    if (!ld->dom) { anuga_dump_free(ld); return NULL; }
    ld->tri_full_flag  = tri_full_flag;
    ld->stage_centroid = stage_centroid;

    if (num_neighbors > 0)
        anuga_init_halo(ld->dom, (int)num_neighbors, neighbor_ranks,
                        send_counts, recv_counts, flat_send, flat_recv);
    if (n_reflective > 0)
        anuga_init_reflective(ld->dom, (int)n_reflective, refl_bidx, refl_vol, refl_edge);
    if (n_dirichlet > 0)
        anuga_init_dirichlet(ld->dom, (int)n_dirichlet, dir_bidx, dir_vol, dir_edge,
                             D[17], D[18], D[19]);
    if (n_transmissive > 0)
        anuga_init_transmissive(ld->dom, (int)n_transmissive, tr_bidx, tr_vol, tr_edge,
                                (int)transmissive_use_centroid);

    return ld;
}

/* ===================== build from a raw triangular mesh (single rank) ===== */
typedef struct { long lo, hi; int tri, edge; } edge_rec;

static int edge_cmp(const void *A, const void *B) {
    const edge_rec *a = A, *b = B;
    if (a->lo != b->lo) return a->lo < b->lo ? -1 : 1;
    if (a->hi != b->hi) return a->hi < b->hi ? -1 : 1;
    return 0;
}

AnugaLoadedDomain *anuga_build_from_mesh(const double *nodes, long n_nodes,
        const long *triangles, long n_tris,
        const double *stage_cv, const double *elevation_cv, const double *friction_cv,
        const AnugaMeshParams *p) {
    (void)n_nodes;
    long n = n_tris;
    AnugaLoadedDomain *ld = (AnugaLoadedDomain *)calloc(1, sizeof(*ld));
    ld->n = n;

    int64_t *neighbours      = xcalloc(ld, 3 * n, sizeof(int64_t));
    int64_t *neighbour_edges = xcalloc(ld, 3 * n, sizeof(int64_t));
    int64_t *surrogate       = xcalloc(ld, 3 * n, sizeof(int64_t));
    int64_t *num_boundaries  = xcalloc(ld, n,     sizeof(int64_t));
    double  *normals     = xcalloc(ld, 6 * n, sizeof(double));
    double  *edgelengths = xcalloc(ld, 3 * n, sizeof(double));
    double  *radii       = xcalloc(ld, n,     sizeof(double));
    double  *areas       = xcalloc(ld, n,     sizeof(double));
    double  *centroid    = xcalloc(ld, 2 * n, sizeof(double));
    double  *edge_coords = xcalloc(ld, 6 * n, sizeof(double));

    for (long k = 0; k < n; k++) {
        long i0 = triangles[3 * k + 0], i1 = triangles[3 * k + 1], i2 = triangles[3 * k + 2];
        double x0 = nodes[2 * i0], y0 = nodes[2 * i0 + 1];
        double x1 = nodes[2 * i1], y1 = nodes[2 * i1 + 1];
        double x2 = nodes[2 * i2], y2 = nodes[2 * i2 + 1];
        areas[k] = -((x1 * y0 - x0 * y1) + (x2 * y1 - x1 * y2) + (x0 * y2 - x2 * y0)) / 2.0;
        /* edge 0 = (V1,V2), edge 1 = (V2,V0), edge 2 = (V0,V1); normal = (yn,-xn) */
        double xn0 = x2 - x1, yn0 = y2 - y1, l0 = sqrt(xn0 * xn0 + yn0 * yn0);
        double xn1 = x0 - x2, yn1 = y0 - y2, l1 = sqrt(xn1 * xn1 + yn1 * yn1);
        double xn2 = x1 - x0, yn2 = y1 - y0, l2 = sqrt(xn2 * xn2 + yn2 * yn2);
        normals[6 * k + 0] =  yn0 / l0; normals[6 * k + 1] = -xn0 / l0;
        normals[6 * k + 2] =  yn1 / l1; normals[6 * k + 3] = -xn1 / l1;
        normals[6 * k + 4] =  yn2 / l2; normals[6 * k + 5] = -xn2 / l2;
        edgelengths[3 * k + 0] = l0; edgelengths[3 * k + 1] = l1; edgelengths[3 * k + 2] = l2;
        double cx = (x0 + x1 + x2) / 3.0, cy = (y0 + y1 + y2) / 3.0;
        centroid[2 * k] = cx; centroid[2 * k + 1] = cy;
        double xm0 = (x1 + x2) / 2, ym0 = (y1 + y2) / 2;
        double xm1 = (x2 + x0) / 2, ym1 = (y2 + y0) / 2;
        double xm2 = (x0 + x1) / 2, ym2 = (y0 + y1) / 2;
        edge_coords[6 * k + 0] = xm0; edge_coords[6 * k + 1] = ym0;
        edge_coords[6 * k + 2] = xm1; edge_coords[6 * k + 3] = ym1;
        edge_coords[6 * k + 4] = xm2; edge_coords[6 * k + 5] = ym2;
        double d0 = hypot(cx - xm0, cy - ym0), d1 = hypot(cx - xm1, cy - ym1),
               d2 = hypot(cx - xm2, cy - ym2);
        double r = d0 < d1 ? d0 : d1; radii[k] = r < d2 ? r : d2;
        for (int i = 0; i < 3; i++) { neighbours[3 * k + i] = -1; neighbour_edges[3 * k + i] = -1; }
    }

    /* neighbour matching: edge i connects vertices (i+1)%3, (i+2)%3 */
    edge_rec *recs = (edge_rec *)malloc((size_t)(3 * n) * sizeof(edge_rec));
    for (long k = 0; k < n; k++)
        for (int i = 0; i < 3; i++) {
            long a = triangles[3 * k + (i + 1) % 3], b = triangles[3 * k + (i + 2) % 3];
            recs[3 * k + i].lo = a < b ? a : b;
            recs[3 * k + i].hi = a < b ? b : a;
            recs[3 * k + i].tri = (int)k; recs[3 * k + i].edge = i;
        }
    qsort(recs, (size_t)(3 * n), sizeof(edge_rec), edge_cmp);
    for (long e = 0; e < 3 * n;) {
        if (e + 1 < 3 * n && recs[e].lo == recs[e + 1].lo && recs[e].hi == recs[e + 1].hi) {
            int ka = recs[e].tri, ia = recs[e].edge, kb = recs[e + 1].tri, ib = recs[e + 1].edge;
            neighbours[3 * ka + ia] = kb; neighbour_edges[3 * ka + ia] = ib;
            neighbours[3 * kb + ib] = ka; neighbour_edges[3 * kb + ib] = ia;
            e += 2;
        } else {
            e += 1;   /* boundary edge */
        }
    }
    free(recs);

    long nb = 0;
    for (long k = 0; k < n; k++) {
        int c = 0;
        for (int i = 0; i < 3; i++) if (neighbours[3 * k + i] < 0) c++;
        num_boundaries[k] = c; nb += c;
    }
    int32_t *bvol  = xcalloc(ld, nb, sizeof(int32_t));
    int32_t *bedge = xcalloc(ld, nb, sizeof(int32_t));
    int32_t *bidx  = xcalloc(ld, nb, sizeof(int32_t));
    long m = 0;
    for (long k = 0; k < n; k++)
        for (int i = 0; i < 3; i++)
            if (neighbours[3 * k + i] < 0) {
                neighbours[3 * k + i] = -(m + 1);   /* encode boundary index */
                bvol[m] = (int32_t)k; bedge[m] = (int32_t)i; bidx[m] = (int32_t)m; m++;
            }
    for (long k = 0; k < n; k++)
        for (int i = 0; i < 3; i++)
            surrogate[3 * k + i] = neighbours[3 * k + i] >= 0 ? neighbours[3 * k + i] : k;

    double *stageC = xcalloc(ld, n, sizeof(double));
    double *xmomC  = xcalloc(ld, n, sizeof(double));
    double *ymomC  = xcalloc(ld, n, sizeof(double));
    double *bedC   = xcalloc(ld, n, sizeof(double));
    double *heightC = xcalloc(ld, n, sizeof(double));
    double *fricC  = xcalloc(ld, n, sizeof(double));
    int64_t *tff   = xcalloc(ld, n, sizeof(int64_t));
    for (long k = 0; k < n; k++) {
        stageC[k] = stage_cv[k]; bedC[k] = elevation_cv[k];
        heightC[k] = stageC[k] - bedC[k];
        fricC[k] = friction_cv ? friction_cv[k] : 0.0;
        tff[k] = 1;
    }

    AnugaDomainDesc d;
    memset(&d, 0, sizeof(d));
    d.number_of_elements = n; d.boundary_length = nb;
    d.optimise_dry_cells = p->optimise_dry_cells;
    d.extrapolate_velocity_second_order = p->extrapolate_velocity_second_order;
    d.low_froude = p->low_froude; d.timestep_fluxcalls = 1;
    d.epsilon = p->epsilon; d.H0 = p->H0; d.g = p->g;
    d.evolve_max_timestep = p->evolve_max_timestep; d.evolve_min_timestep = 0.0;
    d.minimum_allowed_height = p->minimum_allowed_height;
    d.maximum_allowed_speed = p->maximum_allowed_speed;
    d.beta_w = p->beta_w; d.beta_w_dry = p->beta_w_dry;
    d.beta_uh = p->beta_uh; d.beta_uh_dry = p->beta_uh_dry;
    d.beta_vh = p->beta_vh; d.beta_vh_dry = p->beta_vh_dry;
    d.CFL = p->CFL; d.fixed_flux_timestep = -1.0;
    d.neighbours = neighbours; d.neighbour_edges = neighbour_edges;
    d.surrogate_neighbours = surrogate; d.number_of_boundaries = num_boundaries;
    d.tri_full_flag = tff;
    d.normals = normals; d.edgelengths = edgelengths; d.radii = radii; d.areas = areas;
    d.centroid_coordinates = centroid; d.edge_coordinates = edge_coords;
    d.max_speed = xcalloc(ld, n, sizeof(double));
    d.x_centroid_work = xcalloc(ld, n, sizeof(double));
    d.y_centroid_work = xcalloc(ld, n, sizeof(double));
    d.stage_centroid_values = stageC; d.stage_edge_values = xcalloc(ld, 3 * n, sizeof(double));
    d.stage_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.stage_explicit_update = xcalloc(ld, n, sizeof(double));
    d.stage_semi_implicit_update = xcalloc(ld, n, sizeof(double));
    d.stage_backup_values = xcalloc(ld, n, sizeof(double));
    d.xmom_centroid_values = xmomC; d.xmom_edge_values = xcalloc(ld, 3 * n, sizeof(double));
    d.xmom_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.xmom_explicit_update = xcalloc(ld, n, sizeof(double));
    d.xmom_semi_implicit_update = xcalloc(ld, n, sizeof(double));
    d.xmom_backup_values = xcalloc(ld, n, sizeof(double));
    d.ymom_centroid_values = ymomC; d.ymom_edge_values = xcalloc(ld, 3 * n, sizeof(double));
    d.ymom_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.ymom_explicit_update = xcalloc(ld, n, sizeof(double));
    d.ymom_semi_implicit_update = xcalloc(ld, n, sizeof(double));
    d.ymom_backup_values = xcalloc(ld, n, sizeof(double));
    d.bed_centroid_values = bedC; d.bed_edge_values = xcalloc(ld, 3 * n, sizeof(double));
    d.bed_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.height_centroid_values = heightC; d.height_edge_values = xcalloc(ld, 3 * n, sizeof(double));
    d.height_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.friction_centroid_values = fricC;

    ld->dom = anuga_domain_create(&d);
    if (!ld->dom) { anuga_dump_free(ld); return NULL; }
    ld->tri_full_flag = tff; ld->stage_centroid = stageC;
    ld->finaltime = p->finaltime; ld->method = p->timestepping_method;
    ld->nprocs = 1; ld->rank = 0; ld->nb = nb;

    if (nb > 0) anuga_init_reflective(ld->dom, (int)nb, bidx, bvol, bedge);
    return ld;
}

/* ===================== MPI-aware partitioned build ======================= */
/* per-rank growable list of local indices */
typedef struct { int32_t *v; long n, cap; } ilist;
static void ilist_push(ilist *L, int32_t x) {
    if (L->n == L->cap) { L->cap = L->cap ? L->cap * 2 : 16;
        L->v = (int32_t *)realloc(L->v, L->cap * sizeof(int32_t)); }
    L->v[L->n++] = x;
}

AnugaLoadedDomain *anuga_build_from_partition(const double *nodes, long n_nodes,
        const long *tris, long nt, const int *partition,
        const double *stage_g, const double *elev_g, const double *fric_g,
        int myrank, int nprocs, const AnugaMeshParams *p) {

    /* 1. global neighbour structure (edge matching) */
    int64_t *gneigh = (int64_t *)malloc((size_t)(3 * nt) * sizeof(int64_t));
    int64_t *gnedge = (int64_t *)malloc((size_t)(3 * nt) * sizeof(int64_t));
    for (long i = 0; i < 3 * nt; i++) { gneigh[i] = -1; gnedge[i] = -1; }
    edge_rec *recs = (edge_rec *)malloc((size_t)(3 * nt) * sizeof(edge_rec));
    for (long k = 0; k < nt; k++)
        for (int i = 0; i < 3; i++) {
            long a = tris[3 * k + (i + 1) % 3], b = tris[3 * k + (i + 2) % 3];
            recs[3 * k + i].lo = a < b ? a : b; recs[3 * k + i].hi = a < b ? b : a;
            recs[3 * k + i].tri = (int)k; recs[3 * k + i].edge = i;
        }
    qsort(recs, (size_t)(3 * nt), sizeof(edge_rec), edge_cmp);
    for (long e = 0; e < 3 * nt;) {
        if (e + 1 < 3 * nt && recs[e].lo == recs[e + 1].lo && recs[e].hi == recs[e + 1].hi) {
            int ka = recs[e].tri, ia = recs[e].edge, kb = recs[e + 1].tri, ib = recs[e + 1].edge;
            gneigh[3 * ka + ia] = kb; gnedge[3 * ka + ia] = ib;
            gneigh[3 * kb + ib] = ka; gnedge[3 * kb + ib] = ia;
            e += 2;
        } else e += 1;
    }
    free(recs);

    /* 2. owned + 2-layer ghosts via BFS (dist 0 owned, 1, 2) */
    int *dist = (int *)malloc((size_t)nt * sizeof(int));
    for (long g = 0; g < nt; g++) dist[g] = (partition[g] == myrank) ? 0 : -1;
    long *frontier = (long *)malloc((size_t)nt * sizeof(long)); long fn = 0;
    for (long g = 0; g < nt; g++) if (dist[g] == 0) frontier[fn++] = g;
    for (int d = 0; d < 2; d++) {
        long *next = (long *)malloc((size_t)nt * sizeof(long)); long nn2 = 0;
        for (long fi = 0; fi < fn; fi++) {
            long g = frontier[fi];
            for (int i = 0; i < 3; i++) {
                long gn = gneigh[3 * g + i];
                if (gn >= 0 && dist[gn] < 0) { dist[gn] = d + 1; next[nn2++] = gn; }
            }
        }
        free(frontier); frontier = next; fn = nn2;
    }
    free(frontier);

    /* 3. local ordering: owned (gid order) then ghosts (gid order) */
    long nO = 0, nG = 0;
    for (long g = 0; g < nt; g++) { if (dist[g] == 0) nO++; else if (dist[g] > 0) nG++; }
    long nL = nO + nG;
    int64_t *g2l = (int64_t *)malloc((size_t)nt * sizeof(int64_t));
    for (long g = 0; g < nt; g++) g2l[g] = -1;
    AnugaLoadedDomain *ld = (AnugaLoadedDomain *)calloc(1, sizeof(*ld));
    int64_t *global_ids = xcalloc(ld, nL, sizeof(int64_t));
    long li = 0;
    for (long g = 0; g < nt; g++) if (dist[g] == 0) { g2l[g] = li; global_ids[li] = g; li++; }
    for (long g = 0; g < nt; g++) if (dist[g] > 0)  { g2l[g] = li; global_ids[li] = g; li++; }
    ld->n = nL; ld->global_ids = global_ids;

    /* 4. local node renumbering + local triangles */
    int64_t *node_g2l = (int64_t *)malloc((size_t)n_nodes * sizeof(int64_t));
    for (long v = 0; v < n_nodes; v++) node_g2l[v] = -1;
    long nln = 0;
    for (long L = 0; L < nL; L++) { long g = global_ids[L];
        for (int i = 0; i < 3; i++) { long v = tris[3 * g + i]; if (node_g2l[v] < 0) node_g2l[v] = nln++; } }
    double *lnodes = (double *)malloc((size_t)nln * 2 * sizeof(double));
    for (long v = 0; v < n_nodes; v++) { long lv = node_g2l[v];
        if (lv >= 0) { lnodes[2 * lv] = nodes[2 * v]; lnodes[2 * lv + 1] = nodes[2 * v + 1]; } }
    int64_t *ltris = (int64_t *)malloc((size_t)nL * 3 * sizeof(int64_t));
    for (long L = 0; L < nL; L++) { long g = global_ids[L];
        for (int i = 0; i < 3; i++) ltris[3 * L + i] = node_g2l[tris[3 * g + i]]; }

    /* 5. geometry on the local mesh (same formulas as anuga_build_from_mesh) */
    double *normals = xcalloc(ld, 6 * nL, sizeof(double));
    double *edgelengths = xcalloc(ld, 3 * nL, sizeof(double));
    double *radii = xcalloc(ld, nL, sizeof(double));
    double *areas = xcalloc(ld, nL, sizeof(double));
    double *centroid = xcalloc(ld, 2 * nL, sizeof(double));
    double *edge_coords = xcalloc(ld, 6 * nL, sizeof(double));
    for (long k = 0; k < nL; k++) {
        long i0 = ltris[3 * k], i1 = ltris[3 * k + 1], i2 = ltris[3 * k + 2];
        double x0 = lnodes[2 * i0], y0 = lnodes[2 * i0 + 1];
        double x1 = lnodes[2 * i1], y1 = lnodes[2 * i1 + 1];
        double x2 = lnodes[2 * i2], y2 = lnodes[2 * i2 + 1];
        areas[k] = -((x1 * y0 - x0 * y1) + (x2 * y1 - x1 * y2) + (x0 * y2 - x2 * y0)) / 2.0;
        double xn0 = x2 - x1, yn0 = y2 - y1, l0 = sqrt(xn0 * xn0 + yn0 * yn0);
        double xn1 = x0 - x2, yn1 = y0 - y2, l1 = sqrt(xn1 * xn1 + yn1 * yn1);
        double xn2 = x1 - x0, yn2 = y1 - y0, l2 = sqrt(xn2 * xn2 + yn2 * yn2);
        normals[6 * k + 0] = yn0 / l0; normals[6 * k + 1] = -xn0 / l0;
        normals[6 * k + 2] = yn1 / l1; normals[6 * k + 3] = -xn1 / l1;
        normals[6 * k + 4] = yn2 / l2; normals[6 * k + 5] = -xn2 / l2;
        edgelengths[3 * k + 0] = l0; edgelengths[3 * k + 1] = l1; edgelengths[3 * k + 2] = l2;
        double cx = (x0 + x1 + x2) / 3.0, cy = (y0 + y1 + y2) / 3.0;
        centroid[2 * k] = cx; centroid[2 * k + 1] = cy;
        double xm0 = (x1 + x2) / 2, ym0 = (y1 + y2) / 2, xm1 = (x2 + x0) / 2, ym1 = (y2 + y0) / 2,
               xm2 = (x0 + x1) / 2, ym2 = (y0 + y1) / 2;
        edge_coords[6 * k + 0] = xm0; edge_coords[6 * k + 1] = ym0;
        edge_coords[6 * k + 2] = xm1; edge_coords[6 * k + 3] = ym1;
        edge_coords[6 * k + 4] = xm2; edge_coords[6 * k + 5] = ym2;
        double d0 = hypot(cx - xm0, cy - ym0), d1 = hypot(cx - xm1, cy - ym1), d2 = hypot(cx - xm2, cy - ym2);
        double r = d0 < d1 ? d0 : d1; radii[k] = r < d2 ? r : d2;
    }

    /* 6. neighbours from the global graph (mapped to local; -1 -> boundary) */
    int64_t *neigh = xcalloc(ld, 3 * nL, sizeof(int64_t));
    int64_t *nedge = xcalloc(ld, 3 * nL, sizeof(int64_t));
    int64_t *surrogate = xcalloc(ld, 3 * nL, sizeof(int64_t));
    int64_t *numb = xcalloc(ld, nL, sizeof(int64_t));
    for (long L = 0; L < nL; L++) { long g = global_ids[L];
        for (int i = 0; i < 3; i++) {
            long gn = gneigh[3 * g + i];
            if (gn >= 0 && g2l[gn] >= 0) { neigh[3 * L + i] = g2l[gn]; nedge[3 * L + i] = gnedge[3 * g + i]; }
            else { neigh[3 * L + i] = -1; nedge[3 * L + i] = -1; }
        }
    }
    long nb = 0;
    for (long L = 0; L < nL; L++) { int c = 0; for (int i = 0; i < 3; i++) if (neigh[3 * L + i] < 0) c++;
        numb[L] = c; nb += c; }
    int32_t *bvol = xcalloc(ld, nb, sizeof(int32_t));
    int32_t *bedge = xcalloc(ld, nb, sizeof(int32_t));
    int32_t *bidx = xcalloc(ld, nb, sizeof(int32_t));
    long m = 0;
    for (long L = 0; L < nL; L++) for (int i = 0; i < 3; i++) if (neigh[3 * L + i] < 0) {
        neigh[3 * L + i] = -(m + 1); bvol[m] = (int32_t)L; bedge[m] = (int32_t)i; bidx[m] = (int32_t)m; m++; }
    for (long L = 0; L < nL; L++) for (int i = 0; i < 3; i++)
        surrogate[3 * L + i] = neigh[3 * L + i] >= 0 ? neigh[3 * L + i] : L;

    /* 7. quantity centroids from the global IC */
    double *stageC = xcalloc(ld, nL, sizeof(double)), *xmomC = xcalloc(ld, nL, sizeof(double)),
           *ymomC = xcalloc(ld, nL, sizeof(double)), *bedC = xcalloc(ld, nL, sizeof(double)),
           *heightC = xcalloc(ld, nL, sizeof(double)), *fricC = xcalloc(ld, nL, sizeof(double));
    int64_t *tff = xcalloc(ld, nL, sizeof(int64_t));
    for (long L = 0; L < nL; L++) { long g = global_ids[L];
        stageC[L] = stage_g[g]; bedC[L] = elev_g[g]; heightC[L] = stageC[L] - bedC[L];
        fricC[L] = fric_g ? fric_g[g] : 0.0; tff[L] = (dist[g] == 0) ? 1 : 0; }

    /* 8. halo: recv = ghosts grouped by owner; send = owned within 2 hops of a
       foreign cell. Both ordered by global id so they match across ranks. */
    ilist *recv = (ilist *)calloc(nprocs, sizeof(ilist));
    ilist *send = (ilist *)calloc(nprocs, sizeof(ilist));
    for (long L = nO; L < nL; L++) {            /* ghosts already in gid order */
        int owner = partition[global_ids[L]];
        ilist_push(&recv[owner], (int32_t)L);
    }
    int *rank_stamp = (int *)calloc(nprocs, sizeof(int));
    long *q = (long *)malloc((size_t)nt * sizeof(long));
    int *visit = (int *)calloc(nt, sizeof(int)); int gen = 0;
    for (long L = 0; L < nO; L++) {             /* owned in gid order */
        long g = global_ids[L]; gen++;
        long head = 0, tail = 0; q[tail++] = g; visit[g] = gen; int dd0 = 0;
        long level_end = tail;
        int depth = 0;
        while (head < tail) {
            long c = q[head++];
            if (partition[c] != myrank && rank_stamp[partition[c]] != gen) {
                rank_stamp[partition[c]] = gen; ilist_push(&send[partition[c]], (int32_t)L);
            }
            if (head > level_end) { depth++; level_end = tail; }   /* advance BFS level */
            if (depth < 2) {
                for (int i = 0; i < 3; i++) { long gn = gneigh[3 * c + i];
                    if (gn >= 0 && visit[gn] != gen) { visit[gn] = gen; q[tail++] = gn; } }
            }
        }
        (void)dd0;
    }
    free(rank_stamp); free(q); free(visit);

    /* flatten halo in ascending neighbour-rank order */
    int num_neighbors = 0;
    for (int r = 0; r < nprocs; r++) if (send[r].n || recv[r].n) num_neighbors++;
    int *nbr = (int *)malloc(num_neighbors * sizeof(int));
    int *sc = (int *)malloc(num_neighbors * sizeof(int));
    int *rc = (int *)malloc(num_neighbors * sizeof(int));
    long tot_s = 0, tot_r = 0;
    for (int r = 0; r < nprocs; r++) { tot_s += send[r].n; tot_r += recv[r].n; }
    int *fsend = (int *)malloc((tot_s ? tot_s : 1) * sizeof(int));
    int *frecv = (int *)malloc((tot_r ? tot_r : 1) * sizeof(int));
    int ni = 0; long so = 0, ro = 0;
    for (int r = 0; r < nprocs; r++) {
        if (!(send[r].n || recv[r].n)) continue;
        nbr[ni] = r; sc[ni] = (int)send[r].n; rc[ni] = (int)recv[r].n; ni++;
        for (long i = 0; i < send[r].n; i++) fsend[so++] = send[r].v[i];
        for (long i = 0; i < recv[r].n; i++) frecv[ro++] = recv[r].v[i];
    }

    /* 9. descriptor + create + halo + reflective */
    AnugaDomainDesc d; memset(&d, 0, sizeof(d));
    d.number_of_elements = nL; d.boundary_length = nb;
    d.optimise_dry_cells = p->optimise_dry_cells;
    d.extrapolate_velocity_second_order = p->extrapolate_velocity_second_order;
    d.low_froude = p->low_froude; d.timestep_fluxcalls = 1;
    d.epsilon = p->epsilon; d.H0 = p->H0; d.g = p->g;
    d.evolve_max_timestep = p->evolve_max_timestep; d.evolve_min_timestep = 0.0;
    d.minimum_allowed_height = p->minimum_allowed_height;
    d.maximum_allowed_speed = p->maximum_allowed_speed;
    d.beta_w = p->beta_w; d.beta_w_dry = p->beta_w_dry; d.beta_uh = p->beta_uh;
    d.beta_uh_dry = p->beta_uh_dry; d.beta_vh = p->beta_vh; d.beta_vh_dry = p->beta_vh_dry;
    d.CFL = p->CFL; d.fixed_flux_timestep = -1.0;
    d.neighbours = neigh; d.neighbour_edges = nedge; d.surrogate_neighbours = surrogate;
    d.number_of_boundaries = numb; d.tri_full_flag = tff;
    d.normals = normals; d.edgelengths = edgelengths; d.radii = radii; d.areas = areas;
    d.centroid_coordinates = centroid; d.edge_coordinates = edge_coords;
    d.max_speed = xcalloc(ld, nL, sizeof(double));
    d.x_centroid_work = xcalloc(ld, nL, sizeof(double));
    d.y_centroid_work = xcalloc(ld, nL, sizeof(double));
    d.stage_centroid_values = stageC; d.stage_edge_values = xcalloc(ld, 3 * nL, sizeof(double));
    d.stage_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.stage_explicit_update = xcalloc(ld, nL, sizeof(double));
    d.stage_semi_implicit_update = xcalloc(ld, nL, sizeof(double));
    d.stage_backup_values = xcalloc(ld, nL, sizeof(double));
    d.xmom_centroid_values = xmomC; d.xmom_edge_values = xcalloc(ld, 3 * nL, sizeof(double));
    d.xmom_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.xmom_explicit_update = xcalloc(ld, nL, sizeof(double));
    d.xmom_semi_implicit_update = xcalloc(ld, nL, sizeof(double));
    d.xmom_backup_values = xcalloc(ld, nL, sizeof(double));
    d.ymom_centroid_values = ymomC; d.ymom_edge_values = xcalloc(ld, 3 * nL, sizeof(double));
    d.ymom_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.ymom_explicit_update = xcalloc(ld, nL, sizeof(double));
    d.ymom_semi_implicit_update = xcalloc(ld, nL, sizeof(double));
    d.ymom_backup_values = xcalloc(ld, nL, sizeof(double));
    d.bed_centroid_values = bedC; d.bed_edge_values = xcalloc(ld, 3 * nL, sizeof(double));
    d.bed_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.height_centroid_values = heightC; d.height_edge_values = xcalloc(ld, 3 * nL, sizeof(double));
    d.height_boundary_values = xcalloc(ld, nb, sizeof(double));
    d.friction_centroid_values = fricC;

    ld->dom = anuga_domain_create(&d);
    ld->tri_full_flag = tff; ld->stage_centroid = stageC;
    ld->finaltime = p->finaltime; ld->method = p->timestepping_method;
    ld->rank = myrank; ld->nprocs = nprocs; ld->nb = nb;

    if (num_neighbors > 0)
        anuga_init_halo(ld->dom, num_neighbors, nbr, sc, rc, fsend, frecv);
    if (nb > 0) anuga_init_reflective(ld->dom, (int)nb, bidx, bvol, bedge);

    for (int r = 0; r < nprocs; r++) { free(send[r].v); free(recv[r].v); }
    free(send); free(recv); free(nbr); free(sc); free(rc); free(fsend); free(frecv);
    free(gneigh); free(gnedge); free(dist); free(g2l); free(node_g2l); free(lnodes); free(ltris);
    return ld;
}

/* read a .agm global mesh file and build this rank's sub-domain */
AnugaLoadedDomain *anuga_global_load(const char *path, int myrank, int nprocs) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "anuga_global_load: cannot open %s\n", path); return NULL; }
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, ANUGA_GLOBAL_MAGIC, 8) != 0) {
        fprintf(stderr, "anuga_global_load: bad magic\n"); fclose(f); return NULL; }
    int64_t H[7]; double D[14];
    if (fread(H, sizeof(int64_t), 7, f) != 7 || fread(D, sizeof(double), 14, f) != 14) {
        fclose(f); return NULL; }
    long n_nodes = (long)H[1], nt = (long)H[2];
    double  *nodes = (double *)malloc((size_t)n_nodes * 2 * sizeof(double));
    int64_t *tris  = (int64_t *)malloc((size_t)nt * 3 * sizeof(int64_t));
    int32_t *parts = (int32_t *)malloc((size_t)nt * sizeof(int32_t));
    double  *stage = (double *)malloc((size_t)nt * sizeof(double));
    double  *elev  = (double *)malloc((size_t)nt * sizeof(double));
    double  *fric  = (double *)malloc((size_t)nt * sizeof(double));
    int ok = (fread(nodes, sizeof(double), n_nodes * 2, f) == (size_t)(n_nodes * 2))
          && (fread(tris, sizeof(int64_t), nt * 3, f) == (size_t)(nt * 3))
          && (fread(parts, sizeof(int32_t), nt, f) == (size_t)nt)
          && (fread(stage, sizeof(double), nt, f) == (size_t)nt)
          && (fread(elev, sizeof(double), nt, f) == (size_t)nt)
          && (fread(fric, sizeof(double), nt, f) == (size_t)nt);
    fclose(f);
    AnugaLoadedDomain *ld = NULL;
    if (ok) {
        AnugaMeshParams p;
        p.g = D[0]; p.CFL = D[1]; p.H0 = D[2]; p.epsilon = D[3];
        p.minimum_allowed_height = D[4]; p.maximum_allowed_speed = D[5];
        p.evolve_max_timestep = D[6]; p.finaltime = D[7];
        p.beta_w = D[8]; p.beta_w_dry = D[9]; p.beta_uh = D[10]; p.beta_uh_dry = D[11];
        p.beta_vh = D[12]; p.beta_vh_dry = D[13];
        p.timestepping_method = (int)H[3]; p.extrapolate_velocity_second_order = (int)H[4];
        p.low_froude = (int)H[5]; p.optimise_dry_cells = (int)H[6];
        ld = anuga_build_from_partition(nodes, n_nodes, (const long *)tris, nt,
                                        (const int *)parts, stage, elev, fric,
                                        myrank, nprocs, &p);
    } else {
        fprintf(stderr, "anuga_global_load: short read\n");
    }
    free(nodes); free(tris); free(parts); free(stage); free(elev); free(fric);
    return ld;
}

long anuga_dump_get_owned_global_ids(AnugaLoadedDomain *ld, long *out) {
    if (!ld || !ld->global_ids) return 0;
    long mm = 0;
    for (int64_t i = 0; i < ld->n; i++)
        if (ld->tri_full_flag[i] == 1) out[mm++] = (long)ld->global_ids[i];
    return mm;
}

AnugaDomain *anuga_dump_domain(AnugaLoadedDomain *ld)        { return ld ? ld->dom : NULL; }
double anuga_dump_finaltime(AnugaLoadedDomain *ld)           { return ld ? ld->finaltime : 0.0; }
double anuga_dump_yieldstep(AnugaLoadedDomain *ld)           { return ld ? ld->yieldstep : 0.0; }
int    anuga_dump_timestepping_method(AnugaLoadedDomain *ld) { return ld ? ld->method : 0; }
int    anuga_dump_rank(AnugaLoadedDomain *ld)                { return ld ? (int)ld->rank : 0; }
int    anuga_dump_nprocs(AnugaLoadedDomain *ld)              { return ld ? (int)ld->nprocs : 1; }
long   anuga_dump_num_elements(AnugaLoadedDomain *ld)        { return ld ? (long)ld->n : 0; }

long anuga_dump_get_owned_stage(AnugaLoadedDomain *ld, double *out) {
    if (!ld) return 0;
    long m = 0;
    for (int64_t i = 0; i < ld->n; i++)
        if (ld->tri_full_flag[i] == 1) out[m++] = ld->stage_centroid[i];
    return m;
}

void anuga_dump_free(AnugaLoadedDomain *ld) {
    if (!ld) return;
    if (ld->dom) anuga_domain_destroy(ld->dom);   /* destroy before freeing aliased arrays */
    for (int i = 0; i < ld->nallocs; i++) free(ld->allocs[i]);
    free(ld->allocs);
    free(ld);
}
