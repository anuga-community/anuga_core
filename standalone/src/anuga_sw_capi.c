/* anuga_sw_capi.c - implementation of the stable C ABI (anuga_sw.h).
 *
 * This is the standalone replacement for the Cython bridge
 * (anuga/shallow_water/sw_domain_gpu_ext.pyx). Instead of reading a Python
 * domain object, it receives an AnugaDomainDesc of raw pointers and copies them
 * into the production `struct gpu_domain`, then forwards every operation to the
 * unmodified kernels in the anuga/shallow_water/gpu/ directory.
 *
 * The field copy below mirrors get_domain_pointers() in the .pyx exactly -
 * including which fields are deliberately left NULL (vertex values,
 * boundary_flux_sum, edge_flux_work, ...). Those are only touched by code paths
 * the GPU/edge evolve never executes, or are NULL-guarded in the kernels.
 */
#include <stdlib.h>
#include <string.h>

#include "gpu_domain.h"   /* production struct gpu_domain + all gpu_* prototypes */
#include "anuga_sw.h"

struct AnugaDomain {
    struct gpu_domain GD;
};

/* ------------------------------------------------------------------ build info */
int anuga_built_with_offload(void) {
#ifdef CPU_ONLY_MODE
    return 0;
#else
    return 1;
#endif
}

int anuga_built_with_mpi(void) {
#ifdef HAVE_MPI
    return 1;
#else
    return 0;
#endif
}

int anuga_gpu_available(void) {
    return gpu_is_available();
}

/* ------------------------------------------------------------------ C-owned MPI */
void anuga_mpi_init(void) {
#ifdef HAVE_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) MPI_Init(NULL, NULL);
#endif
}

int anuga_mpi_initialized(void) {
#ifdef HAVE_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    return inited;
#else
    return 0;
#endif
}

void anuga_mpi_finalize(void) {
#ifdef HAVE_MPI
    int fin = 0;
    MPI_Finalized(&fin);
    if (!fin) MPI_Finalize();
#endif
}

int anuga_comm_world_rank(void) {
#ifdef HAVE_MPI
    int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r;
#else
    return 0;
#endif
}

int anuga_comm_world_size(void) {
#ifdef HAVE_MPI
    int s = 1; MPI_Comm_size(MPI_COMM_WORLD, &s); return s;
#else
    return 1;
#endif
}

void anuga_domain_use_comm_world(AnugaDomain *dom) {
    if (!dom) return;
#ifdef HAVE_MPI
    dom->GD.comm = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &dom->GD.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &dom->GD.nprocs);
#else
    dom->GD.rank = 0;
    dom->GD.nprocs = 1;
#endif
}

/* ------------------------------------------------------------------ create */
static void fill_domain_from_desc(struct gpu_domain *GD, const AnugaDomainDesc *s) {
    struct domain *D = &GD->D;

    /* scalar sizes / parameters */
    D->number_of_elements              = s->number_of_elements;
    D->boundary_length                 = s->boundary_length;
    D->number_of_riverwall_edges       = s->number_of_riverwall_edges;
    D->optimise_dry_cells              = s->optimise_dry_cells;
    D->extrapolate_velocity_second_order = s->extrapolate_velocity_second_order;
    D->low_froude                      = s->low_froude;
    D->timestep_fluxcalls              = s->timestep_fluxcalls;
    D->ncol_riverwall_hydraulic_properties = s->ncol_riverwall_hydraulic_properties;
    D->nrow_riverwall_hydraulic_properties = s->nrow_riverwall_hydraulic_properties;

    D->epsilon                = s->epsilon;
    D->H0                     = s->H0;
    D->g                      = s->g;
    D->evolve_max_timestep    = s->evolve_max_timestep;
    D->evolve_min_timestep    = s->evolve_min_timestep;
    D->minimum_allowed_height = s->minimum_allowed_height;
    D->maximum_allowed_speed  = s->maximum_allowed_speed;
    D->beta_w                 = s->beta_w;
    D->beta_w_dry             = s->beta_w_dry;
    D->beta_uh                = s->beta_uh;
    D->beta_uh_dry            = s->beta_uh_dry;
    D->beta_vh                = s->beta_vh;
    D->beta_vh_dry            = s->beta_vh_dry;

    /* extras living on gpu_domain */
    GD->CFL                = s->CFL;
    GD->evolve_max_timestep = s->evolve_max_timestep;
    GD->fixed_flux_timestep = s->fixed_flux_timestep;

    /* mesh connectivity / geometry */
    D->neighbours           = s->neighbours;
    D->neighbour_edges      = s->neighbour_edges;
    D->surrogate_neighbours = s->surrogate_neighbours;
    D->number_of_boundaries = s->number_of_boundaries;
    D->tri_full_flag        = s->tri_full_flag;
    D->normals              = s->normals;
    D->edgelengths          = s->edgelengths;
    D->radii                = s->radii;
    D->areas                = s->areas;
    D->max_speed            = s->max_speed;
    D->centroid_coordinates = s->centroid_coordinates;
    D->edge_coordinates     = s->edge_coordinates;
    D->x_centroid_work      = s->x_centroid_work;
    D->y_centroid_work      = s->y_centroid_work;

    /* stage */
    D->stage_centroid_values      = s->stage_centroid_values;
    D->stage_edge_values          = s->stage_edge_values;
    D->stage_boundary_values      = s->stage_boundary_values;
    D->stage_explicit_update      = s->stage_explicit_update;
    D->stage_semi_implicit_update = s->stage_semi_implicit_update;
    D->stage_backup_values        = s->stage_backup_values;

    /* xmomentum */
    D->xmom_centroid_values      = s->xmom_centroid_values;
    D->xmom_edge_values          = s->xmom_edge_values;
    D->xmom_boundary_values      = s->xmom_boundary_values;
    D->xmom_explicit_update      = s->xmom_explicit_update;
    D->xmom_semi_implicit_update = s->xmom_semi_implicit_update;
    D->xmom_backup_values        = s->xmom_backup_values;

    /* ymomentum */
    D->ymom_centroid_values      = s->ymom_centroid_values;
    D->ymom_edge_values          = s->ymom_edge_values;
    D->ymom_boundary_values      = s->ymom_boundary_values;
    D->ymom_explicit_update      = s->ymom_explicit_update;
    D->ymom_semi_implicit_update = s->ymom_semi_implicit_update;
    D->ymom_backup_values        = s->ymom_backup_values;

    /* elevation (bed) */
    D->bed_centroid_values = s->bed_centroid_values;
    D->bed_edge_values     = s->bed_edge_values;
    D->bed_boundary_values = s->bed_boundary_values;

    /* height */
    D->height_centroid_values = s->height_centroid_values;
    D->height_edge_values     = s->height_edge_values;
    D->height_boundary_values = s->height_boundary_values;

    /* friction */
    D->friction_centroid_values = s->friction_centroid_values;

    /* riverwall (nullable) */
    D->edge_flux_type                = s->edge_flux_type;
    D->edge_river_wall_counter       = s->edge_river_wall_counter;
    D->riverwall_elevation           = s->riverwall_elevation;
    D->riverwall_rowIndex            = s->riverwall_rowIndex;
    D->riverwall_hydraulic_properties = s->riverwall_hydraulic_properties;
}

AnugaDomain *anuga_domain_create(const AnugaDomainDesc *desc) {
    if (!desc) return NULL;
    AnugaDomain *dom = (AnugaDomain *)calloc(1, sizeof(AnugaDomain));
    if (!dom) return NULL;

    /* Serial init: comm = MPI_COMM_WORLD (a harmless stub value in non-MPI
     * builds), rank 0, 1 process. Phase 2 overrides via set_mpi_comm_fint. */
    gpu_domain_init(&dom->GD, MPI_COMM_WORLD, 0, 1);
    dom->GD.verbose = 0;

    fill_domain_from_desc(&dom->GD, desc);
    return dom;
}

void anuga_domain_set_mpi_comm_fint(AnugaDomain *dom, int comm_fint) {
    if (!dom) return;
#ifdef HAVE_MPI
    if (comm_fint >= 0) {
        dom->GD.comm = MPI_Comm_f2c((MPI_Fint)comm_fint);
        MPI_Comm_rank(dom->GD.comm, &dom->GD.rank);
        MPI_Comm_size(dom->GD.comm, &dom->GD.nprocs);
    }
#else
    (void)comm_fint;  /* single-process build: nothing to do */
#endif
}

void anuga_init_halo(AnugaDomain *dom, int num_neighbors,
                     const int *neighbor_ranks,
                     const int *send_counts, const int *recv_counts,
                     const int *flat_send_indices, const int *flat_recv_indices) {
    if (!dom || num_neighbors <= 0) return;
    gpu_halo_init(&dom->GD, num_neighbors,
                  (int *)neighbor_ranks, (int *)send_counts, (int *)recv_counts,
                  (int *)flat_send_indices, (int *)flat_recv_indices);
}

void anuga_exchange_ghosts(AnugaDomain *dom) {
    if (dom) gpu_exchange_ghosts(&dom->GD);
}

int anuga_rank(AnugaDomain *dom)   { return dom ? dom->GD.rank : 0; }
int anuga_nprocs(AnugaDomain *dom) { return dom ? dom->GD.nprocs : 1; }

int anuga_domain_map_to_device(AnugaDomain *dom) {
    if (!dom) return 0;
    return gpu_domain_map_arrays(&dom->GD);
}

void anuga_domain_destroy(AnugaDomain *dom) {
    if (!dom) return;
    gpu_domain_unmap_arrays(&dom->GD);
    gpu_domain_finalize(&dom->GD);
    free(dom);
}

/* ------------------------------------------------------------------ sync */
void anuga_sync_to_device(AnugaDomain *dom)       { gpu_domain_sync_to_device(&dom->GD); }
void anuga_sync_from_device(AnugaDomain *dom)     { gpu_domain_sync_from_device(&dom->GD); }
void anuga_sync_all_from_device(AnugaDomain *dom) { gpu_domain_sync_all_from_device(&dom->GD); }

/* ------------------------------------------------------------------ boundaries */
void anuga_init_reflective(AnugaDomain *dom, int num_edges,
                           const int *boundary_indices,
                           const int *vol_ids, const int *edge_ids) {
    gpu_reflective_init(&dom->GD, num_edges,
                        (int *)boundary_indices, (int *)vol_ids, (int *)edge_ids);
}

void anuga_init_dirichlet(AnugaDomain *dom, int num_edges,
                          const int *boundary_indices,
                          const int *vol_ids, const int *edge_ids,
                          double stage_value, double xmom_value, double ymom_value) {
    gpu_dirichlet_init(&dom->GD, num_edges,
                       (int *)boundary_indices, (int *)vol_ids, (int *)edge_ids,
                       stage_value, xmom_value, ymom_value);
}

void anuga_init_transmissive(AnugaDomain *dom, int num_edges,
                             const int *boundary_indices,
                             const int *vol_ids, const int *edge_ids,
                             int use_centroid) {
    gpu_transmissive_init(&dom->GD, num_edges,
                          (int *)boundary_indices, (int *)vol_ids, (int *)edge_ids,
                          use_centroid);
}

void anuga_evaluate_boundaries(AnugaDomain *dom) {
    /* Each is a no-op when its num_edges == 0. */
    gpu_evaluate_reflective_boundary(&dom->GD);
    gpu_evaluate_dirichlet_boundary(&dom->GD);
    gpu_evaluate_transmissive_boundary(&dom->GD);
}

/* ------------------------------------------------------------------ evolve */
double anuga_evolve_one_euler_step(AnugaDomain *dom, double max_timestep, int apply_forcing) {
    return gpu_evolve_one_euler_step(&dom->GD, max_timestep, apply_forcing);
}
double anuga_evolve_one_rk2_step(AnugaDomain *dom, double max_timestep, int apply_forcing) {
    return gpu_evolve_one_rk2_step(&dom->GD, max_timestep, apply_forcing);
}
double anuga_evolve_one_rk3_step(AnugaDomain *dom, double max_timestep, int apply_forcing) {
    return gpu_evolve_one_rk3_step(&dom->GD, max_timestep, apply_forcing);
}
double anuga_evolve_one_ader2_step(AnugaDomain *dom, double max_timestep,
                                   int apply_forcing, double prev_dt) {
    return gpu_evolve_one_ader2_step(&dom->GD, max_timestep, apply_forcing, prev_dt);
}

/* ------------------------------------------------------------------ kernels */
void   anuga_extrapolate_second_order(AnugaDomain *dom) { gpu_extrapolate_second_order(&dom->GD); }
double anuga_compute_fluxes(AnugaDomain *dom)           { return gpu_compute_fluxes(&dom->GD); }
void   anuga_update_conserved_quantities(AnugaDomain *dom, double timestep) {
    gpu_update_conserved_quantities(&dom->GD, timestep);
}
double anuga_protect(AnugaDomain *dom)          { return gpu_protect(&dom->GD); }
void   anuga_manning_friction(AnugaDomain *dom) { gpu_manning_friction(&dom->GD); }
void   anuga_backup_conserved_quantities(AnugaDomain *dom) {
    gpu_backup_conserved_quantities(&dom->GD);
}
void   anuga_saxpy_conserved_quantities(AnugaDomain *dom, double a, double b) {
    gpu_saxpy_conserved_quantities(&dom->GD, a, b);
}

/* ------------------------------------------------------------------ diagnostics */
double anuga_compute_water_volume(AnugaDomain *dom) { return gpu_compute_water_volume(&dom->GD); }
