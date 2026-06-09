/* anuga_dump.c - reader/loader for the .adm partition-dump format (anuga_dump.h). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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
