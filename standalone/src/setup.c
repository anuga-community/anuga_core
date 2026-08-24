#include "setup.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Tracked allocation
// ---------------------------------------------------------------------------

static void *breg(bench_domain *B, size_t count, size_t elem) {
    void *p = calloc(count ? count : 1, elem);
    if (!p) {
        fprintf(stderr, "bench: out of memory (%zu x %zu bytes)\n", count, elem);
        exit(1);
    }
    if (B->nallocs == B->allocs_cap) {
        B->allocs_cap = B->allocs_cap ? 2 * B->allocs_cap : 64;
        B->allocs = (void **)realloc(B->allocs, (size_t)B->allocs_cap * sizeof(void *));
        if (!B->allocs) { fprintf(stderr, "bench: out of memory\n"); exit(1); }
    }
    B->allocs[B->nallocs++] = p;
    return p;
}

#define DALLOC(B, n) ((double  *)breg((B), (size_t)(n), sizeof(double)))
#define IALLOC(B, n) ((anuga_int*)breg((B), (size_t)(n), sizeof(anuga_int)))

// ---------------------------------------------------------------------------
// Directed-edge hash map, used to build the neighbour structure.
//
// Mirrors Neighbour_mesh.build_neighbour_structure: edge 0 of a triangle is
// (v1,v2), edge 1 is (v2,v0), edge 2 is (v0,v1); a triangle's neighbour across
// edge i is whoever registered the reversed pair.
// ---------------------------------------------------------------------------

typedef struct {
    int64_t *key;   // a*num_nodes + b, or -1 when empty
    int64_t *val;   // 4*triangle + edge
    int64_t  cap;   // power of two
} edgemap;

static uint64_t mix64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static void edgemap_init(edgemap *H, int64_t nentries) {
    int64_t cap = 16;
    while (cap < 2 * nentries) cap <<= 1;
    H->cap = cap;
    H->key = (int64_t *)malloc((size_t)cap * sizeof(int64_t));
    H->val = (int64_t *)malloc((size_t)cap * sizeof(int64_t));
    if (!H->key || !H->val) { fprintf(stderr, "bench: out of memory\n"); exit(1); }
    for (int64_t i = 0; i < cap; i++) H->key[i] = -1;
}

static void edgemap_free(edgemap *H) { free(H->key); free(H->val); }

static void edgemap_put(edgemap *H, int64_t key, int64_t val) {
    int64_t mask = H->cap - 1;
    int64_t i = (int64_t)(mix64((uint64_t)key) & (uint64_t)mask);
    while (H->key[i] != -1) {
        if (H->key[i] == key) {
            fprintf(stderr, "bench: duplicate mesh edge (key %lld) -- bad triangulation\n",
                    (long long)key);
            exit(1);
        }
        i = (i + 1) & mask;
    }
    H->key[i] = key;
    H->val[i] = val;
}

// Returns 4*triangle+edge, or -1 if absent.
static int64_t edgemap_get(const edgemap *H, int64_t key) {
    int64_t mask = H->cap - 1;
    int64_t i = (int64_t)(mix64((uint64_t)key) & (uint64_t)mask);
    while (H->key[i] != -1) {
        if (H->key[i] == key) return H->val[i];
        i = (i + 1) & mask;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Bed / initial stage profiles
// ---------------------------------------------------------------------------

static double bed_value(const bench_params *P, double x, double y) {
    if (P->which_case == BENCH_CASE_DAM) return 0.0;

    // Five Gaussian humps on a gentle downstream slope.  Deterministic, smooth,
    // and tall enough that parts of the domain go dry.
    static const double cx[5]  = {0.30, 0.55, 0.70, 0.45, 0.85};
    static const double cy[5]  = {0.35, 0.65, 0.25, 0.85, 0.55};
    static const double amp[5] = {6.0,  4.0,  5.0,  3.0,  7.0};
    static const double rad[5] = {0.08, 0.06, 0.05, 0.07, 0.05};

    const double u = x / P->length_x;
    const double v = y / P->length_y;

    double z = 2.0 * u;   // slope
    for (int i = 0; i < 5; i++) {
        const double du = u - cx[i];
        const double dv = v - cy[i];
        z += amp[i] * exp(-(du * du + dv * dv) / (2.0 * rad[i] * rad[i]));
    }
    return z;
}

static double stage_value(const bench_params *P, double x, double y, double z) {
    switch (P->which_case) {
        case BENCH_CASE_DAM:
            return (x < 0.5 * P->length_x) ? P->dam_height : P->water_level;
        case BENCH_CASE_DAMBUMPS:
            return fmax(z, (x < 0.5 * P->length_x) ? P->dam_height : P->water_level);
        case BENCH_CASE_LAKE:
        default:
            return fmax(z, P->water_level);
    }
}

// ---------------------------------------------------------------------------

void bench_params_defaults(bench_params *P) {
    memset(P, 0, sizeof(*P));

    // ANUGA DE1 preset (_set_DE1_defaults + config.py), timestepping = rk2.
    P->cfl                    = 1.0;
    P->g                      = 9.8;
    P->epsilon                = 1.0e-12;
    P->minimum_allowed_height = 1.0e-5;
    P->H0                     = 1.0e-5;   // Domain.set_minimum_allowed_height sets H0 too
    P->maximum_allowed_speed  = 0.0;
    P->evolve_max_timestep    = 1.0e3;    // config.max_timestep
    P->beta_w                 = 1.0;
    P->beta_w_dry             = 0.0;
    P->beta_uh                = 1.0;
    P->beta_uh_dry            = 0.0;
    P->beta_vh                = 1.0;
    P->beta_vh_dry            = 0.0;
    P->low_froude             = 0;
    P->extrapolate_velocity_second_order = 1;
    P->use_sloped_mannings    = 0;

    P->scheme      = BENCH_SCHEME_RK2;
    P->flux_mode   = 0;
    P->which_case  = BENCH_CASE_DAM;
    P->length_x    = 1000.0;
    P->length_y    = 1000.0;
    P->manning     = 0.03;
    P->water_level = 5.0;
    P->dam_height  = 10.0;
}

void bench_params_apply_scheme(bench_params *P) {
    // Mirrors _set_DE1_defaults / _set_DE_ader2_defaults / _set_DE0_defaults /
    // _set_DE2_defaults in shallow_water_domain.py: the per-scheme limiter
    // betas and CFL, so a scheme comparison here matches what an ANUGA user
    // actually gets when switching flow algorithms.
    double beta;
    switch (P->scheme) {
        case BENCH_SCHEME_ADER2: beta = 0.5; P->cfl = 1.0; break;   // DE_ader2
        case BENCH_SCHEME_EULER: beta = 0.5; P->cfl = 0.9; break;   // DE0
        case BENCH_SCHEME_RK3:   beta = 1.0; P->cfl = 1.0; break;   // DE2
        case BENCH_SCHEME_RK2:
        default:                 beta = 1.0; P->cfl = 1.0; break;   // DE1
    }
    P->beta_w  = beta;  P->beta_w_dry  = 0.0;
    P->beta_uh = beta;  P->beta_uh_dry = 0.0;
    P->beta_vh = beta;  P->beta_vh_dry = 0.0;
}

void bench_domain_build(bench_domain *B, const bench_mesh *M, const bench_params *P) {
    memset(B, 0, sizeof(*B));
    B->mesh = *M;

    struct domain *D = &B->GD.D;
    const int64_t n = M->num_triangles;

    // --- scalars ----------------------------------------------------------
    D->number_of_elements   = n;
    D->epsilon              = P->epsilon;
    D->H0                   = P->H0;
    D->g                    = P->g;
    D->minimum_allowed_height = P->minimum_allowed_height;
    D->maximum_allowed_speed  = P->maximum_allowed_speed;
    D->evolve_max_timestep  = P->evolve_max_timestep;
    D->optimise_dry_cells   = 0;
    D->low_froude           = P->low_froude;
    D->extrapolate_velocity_second_order = P->extrapolate_velocity_second_order;
    D->timestep_fluxcalls   = (P->scheme == BENCH_SCHEME_RK2) ? 2
                            : (P->scheme == BENCH_SCHEME_RK3) ? 3 : 1;
    D->beta_w      = P->beta_w;
    D->beta_w_dry  = P->beta_w_dry;
    D->beta_uh     = P->beta_uh;
    D->beta_uh_dry = P->beta_uh_dry;
    D->beta_vh     = P->beta_vh;
    D->beta_vh_dry = P->beta_vh_dry;

    // No riverwalls in the benchmark: the kernels take the NULL path.
    D->number_of_riverwall_edges          = 0;
    D->ncol_riverwall_hydraulic_properties = 0;
    D->nrow_riverwall_hydraulic_properties = 0;
    D->edge_flux_type                     = NULL;
    D->edge_river_wall_counter            = NULL;
    D->riverwall_elevation                = NULL;
    D->riverwall_rowIndex                 = NULL;
    D->riverwall_hydraulic_properties     = NULL;
    D->already_computed_flux              = NULL;
    D->edge_flux_work                     = NULL;   // confirmed dead in ANUGA
    D->neigh_work                         = NULL;
    D->pressuregrad_work                  = NULL;

    // The benchmark's step always runs the extrapolation before the flux call,
    // so the flux kernel may reconstruct edge bed values as stage - height
    // (bit-identical to bed_ev, ~6 fewer scattered loads per cell).  ANUGA
    // leaves this 0 because tests call compute_fluxes directly.
    D->reconstruct_edge_bed               = 1;


    // --- mesh geometry ----------------------------------------------------
    D->vertex_coordinates   = DALLOC(B, 6 * n);
    D->edge_coordinates     = DALLOC(B, 6 * n);
    D->centroid_coordinates = DALLOC(B, 2 * n);
    D->normals              = DALLOC(B, 6 * n);
    D->edgelengths          = DALLOC(B, 3 * n);
    D->areas                = DALLOC(B, n);
    D->radii                = DALLOC(B, n);

    for (int64_t k = 0; k < n; k++) {
        const int64_t i0 = M->triangles[3 * k + 0];
        const int64_t i1 = M->triangles[3 * k + 1];
        const int64_t i2 = M->triangles[3 * k + 2];

        const double x0 = M->nodes[2 * i0], y0 = M->nodes[2 * i0 + 1];
        const double x1 = M->nodes[2 * i1], y1 = M->nodes[2 * i1 + 1];
        const double x2 = M->nodes[2 * i2], y2 = M->nodes[2 * i2 + 1];

        D->vertex_coordinates[6 * k + 0] = x0;
        D->vertex_coordinates[6 * k + 1] = y0;
        D->vertex_coordinates[6 * k + 2] = x1;
        D->vertex_coordinates[6 * k + 3] = y1;
        D->vertex_coordinates[6 * k + 4] = x2;
        D->vertex_coordinates[6 * k + 5] = y2;

        D->areas[k] = -((x1 * y0 - x0 * y1) + (x2 * y1 - x1 * y2) + (x0 * y2 - x2 * y0)) / 2.0;
        if (!(D->areas[k] > 0.0)) {
            fprintf(stderr, "bench: degenerate triangle %lld (area %g)\n",
                    (long long)k, D->areas[k]);
            exit(1);
        }

        double xn0 = x2 - x1, yn0 = y2 - y1;
        double xn1 = x0 - x2, yn1 = y0 - y2;
        double xn2 = x1 - x0, yn2 = y1 - y0;
        const double l0 = sqrt(xn0 * xn0 + yn0 * yn0);
        const double l1 = sqrt(xn1 * xn1 + yn1 * yn1);
        const double l2 = sqrt(xn2 * xn2 + yn2 * yn2);
        xn0 /= l0; yn0 /= l0;
        xn1 /= l1; yn1 /= l1;
        xn2 /= l2; yn2 /= l2;

        D->normals[6 * k + 0] =  yn0;  D->normals[6 * k + 1] = -xn0;
        D->normals[6 * k + 2] =  yn1;  D->normals[6 * k + 3] = -xn1;
        D->normals[6 * k + 4] =  yn2;  D->normals[6 * k + 5] = -xn2;

        D->edgelengths[3 * k + 0] = l0;
        D->edgelengths[3 * k + 1] = l1;
        D->edgelengths[3 * k + 2] = l2;

        const double cxk = (x0 + x1 + x2) / 3.0;
        const double cyk = (y0 + y1 + y2) / 3.0;
        D->centroid_coordinates[2 * k + 0] = cxk;
        D->centroid_coordinates[2 * k + 1] = cyk;

        // Edge midpoints: edge i is opposite vertex i.
        const double xm0 = 0.5 * (x1 + x2), ym0 = 0.5 * (y1 + y2);
        const double xm1 = 0.5 * (x2 + x0), ym1 = 0.5 * (y2 + y0);
        const double xm2 = 0.5 * (x0 + x1), ym2 = 0.5 * (y0 + y1);
        D->edge_coordinates[6 * k + 0] = xm0;  D->edge_coordinates[6 * k + 1] = ym0;
        D->edge_coordinates[6 * k + 2] = xm1;  D->edge_coordinates[6 * k + 3] = ym1;
        D->edge_coordinates[6 * k + 4] = xm2;  D->edge_coordinates[6 * k + 5] = ym2;

        // radius = distance from the centroid to the nearest edge midpoint
        const double d0 = hypot(cxk - xm0, cyk - ym0);
        const double d1 = hypot(cxk - xm1, cyk - ym1);
        const double d2 = hypot(cxk - xm2, cyk - ym2);
        D->radii[k] = fmin(fmin(d0, d1), d2);
    }

    // --- connectivity -----------------------------------------------------
    D->neighbours           = IALLOC(B, 3 * n);
    D->neighbour_edges      = IALLOC(B, 3 * n);
    D->surrogate_neighbours = IALLOC(B, 3 * n);
    D->number_of_boundaries = IALLOC(B, n);
    // Serial benchmark: no ghost cells, so leave tri_full_flag NULL.  The
    // kernels then skip the per-edge ownership gathers in the dt guard and
    // the boundary-flux integral entirely (the integral is a parallel-run
    // diagnostic nothing here consumes).
    D->tri_full_flag        = NULL;

    edgemap H;
    edgemap_init(&H, 3 * n);
    const int64_t nn = M->num_nodes;
    for (int64_t k = 0; k < n; k++) {
        const int64_t a = M->triangles[3 * k + 0];
        const int64_t b = M->triangles[3 * k + 1];
        const int64_t c = M->triangles[3 * k + 2];
        edgemap_put(&H, a * nn + b, 4 * k + 2);
        edgemap_put(&H, b * nn + c, 4 * k + 0);
        edgemap_put(&H, c * nn + a, 4 * k + 1);
    }
    for (int64_t k = 0; k < n; k++) {
        const int64_t a = M->triangles[3 * k + 0];
        const int64_t b = M->triangles[3 * k + 1];
        const int64_t c = M->triangles[3 * k + 2];
        const int64_t rev[3] = { c * nn + b, a * nn + c, b * nn + a };

        D->number_of_boundaries[k] = 3;
        for (int i = 0; i < 3; i++) {
            const int64_t hit = edgemap_get(&H, rev[i]);
            if (hit >= 0) {
                D->neighbours[3 * k + i]      = hit / 4;
                D->neighbour_edges[3 * k + i] = hit % 4;
                D->number_of_boundaries[k]--;
            } else {
                D->neighbours[3 * k + i]      = -1;
                D->neighbour_edges[3 * k + i] = -1;
            }
        }
    }
    edgemap_free(&H);

    // Boundary enumeration: ANUGA sorts the (volume, edge) keys and numbers
    // them from 0, writing neighbours[k,i] = -(index+1).
    int64_t nb = 0;
    for (int64_t k = 0; k < n; k++)
        for (int i = 0; i < 3; i++)
            if (D->neighbours[3 * k + i] < 0) nb++;

    D->boundary_length = nb;

    anuga_int *bcells = IALLOC(B, nb);
    anuga_int *bedges = IALLOC(B, nb);
    {
        int64_t j = 0;
        for (int64_t k = 0; k < n; k++) {
            for (int i = 0; i < 3; i++) {
                if (D->neighbours[3 * k + i] < 0) {
                    D->neighbours[3 * k + i] = -(j + 1);
                    bcells[j] = k;
                    bedges[j] = i;
                    j++;
                }
            }
        }
    }

    for (int64_t k = 0; k < n; k++)
        for (int i = 0; i < 3; i++)
            D->surrogate_neighbours[3 * k + i] =
                (D->neighbours[3 * k + i] < 0) ? k : D->neighbours[3 * k + i];

    // Experimental flux paths (needs the connectivity built just above):
    //   slot    -- edge-based pair; per-slot records [F0,F1,F2,pf,zh,s] x 3n
    //              (selected by the edge_flux_work pointer being set)
    //   scatter -- single-solve atomic scatter; NO auxiliary work arrays
    //              (selected by reconstruct_edge_bed = 2 + owned_edges)
    if (P->flux_mode == 1) {
        D->edge_flux_work = DALLOC(B, 6 * 3 * n);
    } else if (P->flux_mode == 2) {
        D->reconstruct_edge_bed = 2;
        // Compacted owned-slot list: every boundary slot + the larger-index
        // side of each interior edge.  One scatter thread per physical edge.
        anuga_int *owned = IALLOC(B, 3 * n);
        anuga_int ne = 0;
        for (int64_t p2 = 0; p2 < 3 * n; p2++) {
            const anuga_int nbr2 = D->neighbours[p2];
            if (nbr2 < 0 || nbr2 > p2 / 3) owned[ne++] = p2;
        }
        D->owned_edges = owned;
        D->num_owned_edges = ne;
    }

    // --- quantities -------------------------------------------------------
    D->stage_centroid_values    = DALLOC(B, n);
    D->xmom_centroid_values     = DALLOC(B, n);
    D->ymom_centroid_values     = DALLOC(B, n);
    D->bed_centroid_values      = DALLOC(B, n);
    D->height_centroid_values   = DALLOC(B, n);
    D->friction_centroid_values = DALLOC(B, n);

    D->stage_edge_values   = DALLOC(B, 3 * n);
    D->xmom_edge_values    = DALLOC(B, 3 * n);
    D->ymom_edge_values    = DALLOC(B, 3 * n);
    D->bed_edge_values     = DALLOC(B, 3 * n);
    D->height_edge_values  = DALLOC(B, 3 * n);
    D->xvelocity_edge_values = DALLOC(B, 3 * n);
    D->yvelocity_edge_values = DALLOC(B, 3 * n);

    D->stage_vertex_values  = DALLOC(B, 3 * n);
    D->xmom_vertex_values   = DALLOC(B, 3 * n);
    D->ymom_vertex_values   = DALLOC(B, 3 * n);
    D->bed_vertex_values    = DALLOC(B, 3 * n);
    D->height_vertex_values = DALLOC(B, 3 * n);

    D->stage_boundary_values  = DALLOC(B, nb);
    D->xmom_boundary_values   = DALLOC(B, nb);
    D->ymom_boundary_values   = DALLOC(B, nb);
    D->bed_boundary_values    = DALLOC(B, nb);
    D->height_boundary_values = DALLOC(B, nb);
    D->xvelocity_boundary_values = DALLOC(B, nb);
    D->yvelocity_boundary_values = DALLOC(B, nb);

    D->stage_explicit_update = DALLOC(B, n);
    D->xmom_explicit_update  = DALLOC(B, n);
    D->ymom_explicit_update  = DALLOC(B, n);
    D->stage_semi_implicit_update = DALLOC(B, n);
    D->xmom_semi_implicit_update  = DALLOC(B, n);
    D->ymom_semi_implicit_update  = DALLOC(B, n);

    D->stage_backup_values = DALLOC(B, n);
    D->xmom_backup_values  = DALLOC(B, n);
    D->ymom_backup_values  = DALLOC(B, n);

    D->max_speed        = DALLOC(B, n);
    D->x_centroid_work  = DALLOC(B, n);
    D->y_centroid_work  = DALLOC(B, n);
    D->boundary_flux_sum = DALLOC(B, 3);   // one slot per RK substep

    // Elevation is set per vertex and averaged down, exactly as
    // Quantity.set_values(f, location='vertices') + interpolate() does.
    for (int64_t k = 0; k < n; k++) {
        double zv[3];
        for (int i = 0; i < 3; i++)
            zv[i] = bed_value(P, D->vertex_coordinates[6 * k + 2 * i],
                                 D->vertex_coordinates[6 * k + 2 * i + 1]);

        for (int i = 0; i < 3; i++) {
            D->bed_vertex_values[3 * k + i] = zv[i];
            D->bed_edge_values[3 * k + i]   = 0.5 * (zv[(i + 1) % 3] + zv[(i + 2) % 3]);
        }
        D->bed_centroid_values[k] = (zv[0] + zv[1] + zv[2]) / 3.0;
    }

    // Conserved quantities start from centroid values; the first extrapolate
    // of every step rebuilds the edge values, so only the centroids matter.
    for (int64_t k = 0; k < n; k++) {
        const double cxk = D->centroid_coordinates[2 * k + 0];
        const double cyk = D->centroid_coordinates[2 * k + 1];
        const double zc  = D->bed_centroid_values[k];
        const double w   = stage_value(P, cxk, cyk, zc);

        D->stage_centroid_values[k]  = w;
        D->xmom_centroid_values[k]   = 0.0;
        D->ymom_centroid_values[k]   = 0.0;
        D->height_centroid_values[k] = fmax(w - zc, 0.0);
        D->friction_centroid_values[k] = P->manning;

        for (int i = 0; i < 3; i++) {
            D->stage_edge_values[3 * k + i]    = w;
            D->stage_vertex_values[3 * k + i]  = w;
            D->height_edge_values[3 * k + i]   = fmax(w - D->bed_edge_values[3 * k + i], 0.0);
            D->height_vertex_values[3 * k + i] = fmax(w - D->bed_vertex_values[3 * k + i], 0.0);
        }
    }

    // --- reflective boundary description (all four sides) -----------------
    // Stored on the bench_domain so bench_domain_to_device can hand it to
    // gpu_reflective_init, which takes its own copy.
    B->GD.reflective.num_edges = 0;   // filled in below via the scratch arrays
    {
        int *bidx = (int *)breg(B, (size_t)nb, sizeof(int));
        int *vids = (int *)breg(B, (size_t)nb, sizeof(int));
        int *eids = (int *)breg(B, (size_t)nb, sizeof(int));
        for (int64_t j = 0; j < nb; j++) {
            bidx[j] = (int)j;
            vids[j] = (int)bcells[j];
            eids[j] = (int)bedges[j];
        }
        // Stash them in the (unused-until-init) reflective slots.
        B->GD.reflective.boundary_indices = bidx;
        B->GD.reflective.vol_ids          = vids;
        B->GD.reflective.edge_ids         = eids;
        B->GD.reflective.num_edges        = (int)nb;
    }
}

void bench_domain_to_device(bench_domain *B, const bench_params *P, int verbose) {
    // Keep the scratch boundary description; gpu_domain_init clears the slots.
    int  ne   = B->GD.reflective.num_edges;
    int *bidx = B->GD.reflective.boundary_indices;
    int *vids = B->GD.reflective.vol_ids;
    int *eids = B->GD.reflective.edge_ids;

    gpu_domain_init(&B->GD, MPI_COMM_WORLD, 0, 1);

    B->GD.verbose             = verbose;
    B->GD.CFL                 = P->cfl;
    B->GD.evolve_max_timestep = P->evolve_max_timestep;
    B->GD.fixed_flux_timestep = -1.0;
    B->GD.use_sloped_mannings = P->use_sloped_mannings;

    if (gpu_reflective_init(&B->GD, ne, bidx, vids, eids) != 0) {
        fprintf(stderr, "bench: gpu_reflective_init failed\n");
        exit(1);
    }

    if (!gpu_domain_map_arrays(&B->GD)) {
        fprintf(stderr, "bench: gpu_domain_map_arrays failed\n");
        exit(1);
    }

    // Experimental flux-path work arrays (device-produced/consumed: alloc only)
    if (B->GD.D.edge_flux_work != NULL) {
        double *slots = B->GD.D.edge_flux_work;
        const anuga_int nslots = 6 * 3 * B->GD.D.number_of_elements;
        #pragma omp target enter data map(alloc: slots[0:nslots])
    }
    // Owned-edge list for the scatter kernel (host-built, read-only on device)
    if (B->GD.D.owned_edges != NULL) {
        anuga_int *owned = B->GD.D.owned_edges;
        const anuga_int ne = B->GD.D.num_owned_edges;
        #pragma omp target enter data map(to: owned[0:ne])
    }
}

void bench_domain_free(bench_domain *B) {
    gpu_domain_finalize(&B->GD);
    for (int i = 0; i < B->nallocs; i++) free(B->allocs[i]);
    free(B->allocs);
    B->allocs = NULL;
    B->nallocs = B->allocs_cap = 0;
}
