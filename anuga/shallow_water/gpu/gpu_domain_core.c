// GPU-accelerated shallow water solver
// Split from sw_domain_gpu.c for maintainability

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "gpu_domain.h"
#include "gpu_omp_macros.h"

// Domain initialization, memory management, and sync functions

// Forward declarations for functions in other files
extern void gpu_flop_counters_init(struct gpu_domain *GD);
extern void gpu_halo_finalize(struct gpu_domain *GD);
extern void gpu_reflective_finalize(struct gpu_domain *GD);
extern void gpu_dirichlet_finalize(struct gpu_domain *GD);
extern void gpu_transmissive_finalize(struct gpu_domain *GD);
extern void gpu_transmissive_n_zero_t_finalize(struct gpu_domain *GD);
extern void gpu_boundary_edge_sync_finalize(struct gpu_domain *GD);

// ============================================================================
// Utility Functions
// ============================================================================

// Detect if MPI library supports GPU-aware communication
// This is a runtime check - compile with -DGPU_AWARE_MPI to force enable
int detect_gpu_aware_mpi(void) {
#ifdef GPU_AWARE_MPI
    return 1;
#else
    // Could add runtime detection here (e.g., check MPIX_Query_cuda_support)
    // For now, default to disabled - user must compile with -DGPU_AWARE_MPI
    return 0;
#endif
}

void print_gpu_domain_info(struct gpu_domain *GD) {
    if (!GD->verbose) return;
    printf("\n--- GPU Domain (rank %d/%d) ---\n", GD->rank, GD->nprocs);
    printf("  Elements: %" PRId64 "  |  Device: %d  |  GPU-aware MPI: %s",
           GD->D.number_of_elements, GD->device_id,
           GD->gpu_aware_mpi ? "yes" : "no");
    if (GD->halo.num_neighbors > 0) {
        printf("  |  Halo procs: %d (send %d / recv %d)",
               GD->halo.num_neighbors,
               GD->halo.total_send_size, GD->halo.total_recv_size);
    }
    printf("\n");
    fflush(stdout);
}


// ============================================================================
// Initialization and Cleanup
// ============================================================================

int gpu_domain_init(struct gpu_domain *GD, MPI_Comm comm, int rank, int nprocs) {
    // Store MPI info
    GD->comm = comm;
    GD->rank = rank;
    GD->nprocs = nprocs;

    // Initialize GPU state
    GD->gpu_initialized = 0;
    GD->backup_arrays_mapped = 0;

    // Select GPU device (round-robin if more ranks than GPUs)
    int num_devices = omp_get_num_devices();
    if (num_devices > 0) {
        GD->device_id = rank % num_devices;
        omp_set_default_device(GD->device_id);
    } else {
        GD->device_id = -1;
    }

    // Detect GPU-aware MPI
    GD->gpu_aware_mpi = detect_gpu_aware_mpi();

    // Initialize halo to empty
    GD->halo.num_neighbors = 0;
    GD->halo.neighbor_ranks = NULL;
    GD->halo.send_counts = NULL;
    GD->halo.recv_counts = NULL;
    GD->halo.total_send_size = 0;
    GD->halo.total_recv_size = 0;
    GD->halo.flat_send_indices = NULL;
    GD->halo.flat_recv_indices = NULL;
    GD->halo.send_offsets = NULL;
    GD->halo.recv_offsets = NULL;
    GD->halo.send_buffer = NULL;
    GD->halo.recv_buffer = NULL;
    GD->halo.requests = NULL;

    // Initialize reflective boundary to empty
    GD->reflective.num_edges = 0;
    GD->reflective.boundary_indices = NULL;
    GD->reflective.vol_ids = NULL;
    GD->reflective.edge_ids = NULL;
    GD->reflective.mapped = 0;

    // Initialize dirichlet boundary to empty
    GD->dirichlet.num_edges = 0;
    GD->dirichlet.boundary_indices = NULL;
    GD->dirichlet.vol_ids = NULL;
    GD->dirichlet.edge_ids = NULL;
    GD->dirichlet.stage_value = 0.0;
    GD->dirichlet.xmom_value = 0.0;
    GD->dirichlet.ymom_value = 0.0;
    GD->dirichlet.mapped = 0;

    // Initialize transmissive boundary to empty
    GD->transmissive.num_edges = 0;
    GD->transmissive.boundary_indices = NULL;
    GD->transmissive.vol_ids = NULL;
    GD->transmissive.edge_ids = NULL;
    GD->transmissive.use_centroid = 0;
    GD->transmissive.mapped = 0;

    // Initialize transmissive_n_zero_t boundary to empty
    GD->transmissive_n_zero_t.num_edges = 0;
    GD->transmissive_n_zero_t.boundary_indices = NULL;
    GD->transmissive_n_zero_t.vol_ids = NULL;
    GD->transmissive_n_zero_t.edge_ids = NULL;
    GD->transmissive_n_zero_t.stage_value = 0.0;
    GD->transmissive_n_zero_t.mapped = 0;

    // Initialize time_boundary to empty
    GD->time_bdry.num_edges = 0;
    GD->time_bdry.boundary_indices = NULL;
    GD->time_bdry.vol_ids = NULL;
    GD->time_bdry.edge_ids = NULL;
    GD->time_bdry.stage_value = 0.0;
    GD->time_bdry.xmom_value = 0.0;
    GD->time_bdry.ymom_value = 0.0;
    GD->time_bdry.mapped = 0;

    // Initialize boundary edge sync to empty
    GD->edge_sync.num_boundary_cells = 0;
    GD->edge_sync.cell_ids = NULL;
    GD->edge_sync.buf_size = 0;
    GD->edge_sync.stage_buf = NULL;
    GD->edge_sync.xmom_buf = NULL;
    GD->edge_sync.ymom_buf = NULL;
    GD->edge_sync.bed_buf = NULL;
    GD->edge_sync.height_buf = NULL;
    GD->edge_sync.initialized = 0;

    // Default simulation parameters
    GD->CFL = 1.0;
    GD->evolve_max_timestep = 1.0e10;
    GD->fixed_flux_timestep = -1.0;  // Disabled by default

    // Initialize culvert operators to empty
    GD->culvert_ops.num_culverts = 0;
    GD->culvert_ops.initialized = 0;
    GD->culvert_ops.mapped = 0;
    GD->culvert_ops.total_inlet_triangles = 0;
    GD->culvert_ops.scratch_stage = NULL;
    GD->culvert_ops.scratch_xmom = NULL;
    GD->culvert_ops.scratch_ymom = NULL;
    GD->culvert_ops.scratch_elev = NULL;
    GD->culvert_ops.scratch_inlet_indices = NULL;
    GD->culvert_ops.scratch_inlet_areas = NULL;
    GD->culvert_ops.scratch_inlet_stage = NULL;
    GD->culvert_ops.scratch_inlet_xmom = NULL;
    GD->culvert_ops.scratch_inlet_ymom = NULL;
    GD->culvert_ops.scratch_inlet_elev = NULL;

    // Initialize FLOP counters (Gordon Bell profiling)
    gpu_flop_counters_init(GD);

    return 0;
}

void gpu_domain_finalize(struct gpu_domain *GD) {
    // Unmap GPU arrays if mapped
    if (GD->gpu_initialized) {
        gpu_domain_unmap_arrays(GD);
    }

    // Free halo structures
    gpu_halo_finalize(GD);

    // Free boundary structures
    gpu_reflective_finalize(GD);
    gpu_dirichlet_finalize(GD);
    gpu_transmissive_finalize(GD);
    gpu_transmissive_n_zero_t_finalize(GD);

    // Free boundary edge sync structures
    gpu_boundary_edge_sync_finalize(GD);

    // Free culvert operator structures
    gpu_culverts_finalize_all(GD);

    GD->gpu_initialized = 0;
}


// ============================================================================
// GPU Memory Management
// ============================================================================

void gpu_domain_map_arrays(struct gpu_domain *GD) {
    if (GD->gpu_initialized) return;
    // Note: Don't return early if device_id < 0. With OMP_TARGET_OFFLOAD=disabled,
    // the OpenMP target directives run on CPU, so we still need to set up the
    // data structures and flags for the boundary/kernel functions to work.

    anuga_int n = GD->D.number_of_elements;
    anuga_int nb = GD->D.boundary_length;
    struct halo_exchange *H = &GD->halo;
    struct reflective_boundary *R = &GD->reflective;

    // Extract pointers to local variables (OpenMP target can't handle struct->member syntax)
    double *stage_cv = GD->D.stage_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *bed_cv = GD->D.bed_centroid_values;
    double *height_cv = GD->D.height_centroid_values;
    double *stage_ev = GD->D.stage_edge_values;
    double *xmom_ev = GD->D.xmom_edge_values;
    double *ymom_ev = GD->D.ymom_edge_values;
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;
    double *stage_eu = GD->D.stage_explicit_update;
    double *xmom_eu = GD->D.xmom_explicit_update;
    double *ymom_eu = GD->D.ymom_explicit_update;
    double *stage_siu = GD->D.stage_semi_implicit_update;
    double *xmom_siu = GD->D.xmom_semi_implicit_update;
    double *ymom_siu = GD->D.ymom_semi_implicit_update;
    anuga_int *neighbours = GD->D.neighbours;
    anuga_int *neighbour_edges = GD->D.neighbour_edges;
    double *normals = GD->D.normals;
    double *edgelengths = GD->D.edgelengths;
    double *areas = GD->D.areas;
    double *radii = GD->D.radii;
    double *max_speed = GD->D.max_speed;
    double *centroid_coords = GD->D.centroid_coordinates;
    double *edge_coords = GD->D.edge_coordinates;

    // Additional arrays for extrapolation
    anuga_int *surrogate_neighbours = GD->D.surrogate_neighbours;
    anuga_int *number_of_boundaries = GD->D.number_of_boundaries;
    double *x_centroid_work = GD->D.x_centroid_work;
    double *y_centroid_work = GD->D.y_centroid_work;

    // Friction array
    double *friction_cv = GD->D.friction_centroid_values;

    // tri_full_flag for MPI ghost cell identification
    anuga_int *tri_full_flag = GD->D.tri_full_flag;

    // Map all domain arrays to GPU - persistent for entire simulation
    #pragma omp target enter data map(to: \
        stage_cv[0:n], xmom_cv[0:n], ymom_cv[0:n], \
        bed_cv[0:n], height_cv[0:n], friction_cv[0:n], \
        stage_ev[0:3*n], xmom_ev[0:3*n], ymom_ev[0:3*n], \
        bed_ev[0:3*n], height_ev[0:3*n], \
        stage_eu[0:n], xmom_eu[0:n], ymom_eu[0:n], \
        stage_siu[0:n], xmom_siu[0:n], ymom_siu[0:n], \
        neighbours[0:3*n], neighbour_edges[0:3*n], \
        surrogate_neighbours[0:3*n], number_of_boundaries[0:n], \
        x_centroid_work[0:n], y_centroid_work[0:n], \
        normals[0:6*n], edgelengths[0:3*n], \
        areas[0:n], radii[0:n], max_speed[0:n], \
        centroid_coords[0:2*n], edge_coords[0:6*n])

    // Map tri_full_flag only when present (parallel domains only)
    if (tri_full_flag != NULL) {
        #pragma omp target enter data map(to: tri_full_flag[0:n])
    }

    // Map boundary values if present (including bed and height for reflective boundary)
    if (nb > 0) {
        double *stage_bv = GD->D.stage_boundary_values;
        double *xmom_bv = GD->D.xmom_boundary_values;
        double *ymom_bv = GD->D.ymom_boundary_values;
        double *bed_bv = GD->D.bed_boundary_values;
        double *height_bv = GD->D.height_boundary_values;
        #pragma omp target enter data map(to: \
            stage_bv[0:nb], xmom_bv[0:nb], ymom_bv[0:nb], \
            bed_bv[0:nb], height_bv[0:nb])
    }

    // Map reflective boundary arrays if initialized (must be done BEFORE this function is called)
    if (R->num_edges > 0 && R->boundary_indices != NULL) {
        int ne = R->num_edges;
        int *b_idx = R->boundary_indices;
        int *v_ids = R->vol_ids;
        int *e_ids = R->edge_ids;

        #pragma omp target enter data map(to: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        R->mapped = 1;

        if (GD->rank == 0 && GD->verbose) {
            printf("  Reflective boundary: %d edges\n", ne);
            fflush(stdout);
        }
    }

    // Map dirichlet boundary arrays if initialized
    struct dirichlet_boundary *Dir = &GD->dirichlet;
    if (Dir->num_edges > 0 && Dir->boundary_indices != NULL) {
        int ne = Dir->num_edges;
        int *b_idx = Dir->boundary_indices;
        int *v_ids = Dir->vol_ids;
        int *e_ids = Dir->edge_ids;

        #pragma omp target enter data map(to: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        Dir->mapped = 1;

        if (GD->rank == 0 && GD->verbose) {
            printf("  Dirichlet boundary:  %d edges\n", ne);
            fflush(stdout);
        }
    }

    // Map transmissive boundary arrays if initialized
    struct transmissive_boundary *T = &GD->transmissive;
    if (T->num_edges > 0 && T->boundary_indices != NULL) {
        int ne = T->num_edges;
        int *b_idx = T->boundary_indices;
        int *v_ids = T->vol_ids;
        int *e_ids = T->edge_ids;

        #pragma omp target enter data map(to: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        T->mapped = 1;

        if (GD->rank == 0 && GD->verbose) {
            printf("  Transmissive boundary: %d edges\n", ne);
            fflush(stdout);
        }
    }

    // Map transmissive_n_zero_t boundary arrays if initialized
    struct transmissive_n_zero_t_boundary *Tnzt = &GD->transmissive_n_zero_t;
    if (Tnzt->num_edges > 0 && Tnzt->boundary_indices != NULL) {
        int ne = Tnzt->num_edges;
        int *b_idx = Tnzt->boundary_indices;
        int *v_ids = Tnzt->vol_ids;
        int *e_ids = Tnzt->edge_ids;

        #pragma omp target enter data map(to: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        Tnzt->mapped = 1;

        if (GD->rank == 0 && GD->verbose) {
            printf("  Transmissive_n_zero_t boundary: %d edges\n", ne);
            fflush(stdout);
        }
    }

    // Map time_boundary arrays if initialized
    struct time_boundary *TB = &GD->time_bdry;
    if (TB->num_edges > 0 && TB->boundary_indices != NULL) {
        int ne = TB->num_edges;
        int *b_idx = TB->boundary_indices;
        int *v_ids = TB->vol_ids;
        int *e_ids = TB->edge_ids;

        #pragma omp target enter data map(to: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        TB->mapped = 1;

        if (GD->rank == 0 && GD->verbose) {
            printf("  Time boundary: %d edges\n", ne);
            fflush(stdout);
        }
    }

    // Map riverwall arrays if present
    // edge_flux_type is always mapped (size 3*n); other arrays only if riverwalls exist
    anuga_int *edge_flux_type = GD->D.edge_flux_type;
    if (edge_flux_type != NULL) {
        #pragma omp target enter data map(to: edge_flux_type[0:3*n])
    }

    anuga_int n_rw_edges = GD->D.number_of_riverwall_edges;
    if (n_rw_edges > 0) {
        anuga_int *edge_river_wall_counter = GD->D.edge_river_wall_counter;
        double *riverwall_elevation = GD->D.riverwall_elevation;
        anuga_int *riverwall_rowIndex = GD->D.riverwall_rowIndex;
        double *riverwall_hydraulic_properties = GD->D.riverwall_hydraulic_properties;
        anuga_int ncol_hp = GD->D.ncol_riverwall_hydraulic_properties;
        anuga_int nrow_hp = GD->D.nrow_riverwall_hydraulic_properties;

        // edge_river_wall_counter has same size as edge_flux_type (3*n)
        if (edge_river_wall_counter != NULL) {
            #pragma omp target enter data map(to: edge_river_wall_counter[0:3*n])
        }
        // riverwall_elevation and riverwall_rowIndex have size n_rw_edges
        if (riverwall_elevation != NULL) {
            #pragma omp target enter data map(to: riverwall_elevation[0:n_rw_edges])
        }
        if (riverwall_rowIndex != NULL) {
            #pragma omp target enter data map(to: riverwall_rowIndex[0:n_rw_edges])
        }
        // riverwall_hydraulic_properties is (nrow_hp x ncol_hp)
        // nrow_hp = number of unique riverwall segments
        if (riverwall_hydraulic_properties != NULL && ncol_hp > 0 && nrow_hp > 0) {
            #pragma omp target enter data map(to: riverwall_hydraulic_properties[0:nrow_hp*ncol_hp])
        }

        if (GD->rank == 0 && GD->verbose) {
            printf("  Riverwall: %" PRId64 " edges, %" PRId64 " segments\n", n_rw_edges, nrow_hp);
            fflush(stdout);
        }
    }

    // Map backup arrays for RK2 if present
    if (GD->D.stage_backup_values != NULL) {
        double *stage_backup = GD->D.stage_backup_values;
        double *xmom_backup = GD->D.xmom_backup_values;
        double *ymom_backup = GD->D.ymom_backup_values;
        #pragma omp target enter data map(to: \
            stage_backup[0:n], xmom_backup[0:n], ymom_backup[0:n])
        GD->backup_arrays_mapped = 1;
    }

    // Map halo exchange arrays if we have neighbors
    if (H->num_neighbors > 0) {
        int send_size = H->total_send_size;
        int recv_size = H->total_recv_size;
        int *flat_send = H->flat_send_indices;
        int *flat_recv = H->flat_recv_indices;
        double *send_buf = H->send_buffer;
        double *recv_buf = H->recv_buffer;

        #pragma omp target enter data map(to: flat_send[0:send_size], flat_recv[0:recv_size]) \
            map(alloc: send_buf[0:3*send_size], recv_buf[0:3*recv_size])
    }

    GD->gpu_initialized = 1;

    if (GD->rank == 0 && GD->verbose) {
        printf("  Arrays mapped to device %d\n", GD->device_id);
        fflush(stdout);
    }
}

void gpu_remap_boundary_arrays(struct gpu_domain *GD) {
    // Remap boundary arrays that were initialized after the initial map_to_gpu call.
    // This is needed when set_boundary() is called AFTER set_multiprocessor_mode().

    // Map reflective boundary arrays if initialized but not yet mapped
    struct reflective_boundary *R = &GD->reflective;
    if (R->num_edges > 0 && R->boundary_indices != NULL && !R->mapped) {
        int ne = R->num_edges;
        int *b_idx = R->boundary_indices;
        int *v_ids = R->vol_ids;
        int *e_ids = R->edge_ids;

        #pragma omp target enter data map(to: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        R->mapped = 1;

        if (GD->rank == 0 && GD->verbose) {
            printf("Reflective boundary arrays mapped to GPU (late): %d edges\n", ne);
            fflush(stdout);
        }
    }

    // Map dirichlet boundary arrays if initialized but not yet mapped
    struct dirichlet_boundary *Dir = &GD->dirichlet;
    if (Dir->num_edges > 0 && Dir->boundary_indices != NULL && !Dir->mapped) {
        int ne = Dir->num_edges;
        int *b_idx = Dir->boundary_indices;
        int *v_ids = Dir->vol_ids;
        int *e_ids = Dir->edge_ids;

        #pragma omp target enter data map(to: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        Dir->mapped = 1;

        if (GD->rank == 0 && GD->verbose) {
            printf("Dirichlet boundary arrays mapped to GPU (late): %d edges\n", ne);
            fflush(stdout);
        }
    }

    // Map transmissive boundary arrays if initialized but not yet mapped
    struct transmissive_boundary *T = &GD->transmissive;
    if (T->num_edges > 0 && T->boundary_indices != NULL && !T->mapped) {
        int ne = T->num_edges;
        int *b_idx = T->boundary_indices;
        int *v_ids = T->vol_ids;
        int *e_ids = T->edge_ids;

        #pragma omp target enter data map(to: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        T->mapped = 1;

        if (GD->rank == 0 && GD->verbose) {
            printf("Transmissive boundary arrays mapped to GPU (late): %d edges\n", ne);
            fflush(stdout);
        }
    }

    // Map transmissive_n_zero_t boundary arrays if initialized but not yet mapped
    struct transmissive_n_zero_t_boundary *Tnzt = &GD->transmissive_n_zero_t;
    if (Tnzt->num_edges > 0 && Tnzt->boundary_indices != NULL && !Tnzt->mapped) {
        int ne = Tnzt->num_edges;
        int *b_idx = Tnzt->boundary_indices;
        int *v_ids = Tnzt->vol_ids;
        int *e_ids = Tnzt->edge_ids;

        #pragma omp target enter data map(to: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        Tnzt->mapped = 1;

        if (GD->rank == 0 && GD->verbose) {
            printf("Transmissive_n_zero_t boundary arrays mapped to GPU (late): %d edges\n", ne);
            fflush(stdout);
        }
    }

    // Map time_boundary arrays if initialized but not yet mapped
    struct time_boundary *TB = &GD->time_bdry;
    if (TB->num_edges > 0 && TB->boundary_indices != NULL && !TB->mapped) {
        int ne = TB->num_edges;
        int *b_idx = TB->boundary_indices;
        int *v_ids = TB->vol_ids;
        int *e_ids = TB->edge_ids;

        #pragma omp target enter data map(to: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        TB->mapped = 1;

        if (GD->rank == 0 && GD->verbose) {
            printf("Time_boundary arrays mapped to GPU (late): %d edges\n", ne);
            fflush(stdout);
        }
    }
}

void gpu_domain_unmap_arrays(struct gpu_domain *GD) {
    if (!GD->gpu_initialized) return;

    anuga_int n = GD->D.number_of_elements;
    anuga_int nb = GD->D.boundary_length;
    struct halo_exchange *H = &GD->halo;

    // Extract pointers to local variables
    double *stage_cv = GD->D.stage_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *bed_cv = GD->D.bed_centroid_values;
    double *height_cv = GD->D.height_centroid_values;
    double *stage_ev = GD->D.stage_edge_values;
    double *xmom_ev = GD->D.xmom_edge_values;
    double *ymom_ev = GD->D.ymom_edge_values;
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;
    double *stage_eu = GD->D.stage_explicit_update;
    double *xmom_eu = GD->D.xmom_explicit_update;
    double *ymom_eu = GD->D.ymom_explicit_update;
    double *stage_siu = GD->D.stage_semi_implicit_update;
    double *xmom_siu = GD->D.xmom_semi_implicit_update;
    double *ymom_siu = GD->D.ymom_semi_implicit_update;
    anuga_int *neighbours = GD->D.neighbours;
    anuga_int *neighbour_edges = GD->D.neighbour_edges;
    double *normals = GD->D.normals;
    double *edgelengths = GD->D.edgelengths;
    double *areas = GD->D.areas;
    double *radii = GD->D.radii;
    double *max_speed = GD->D.max_speed;
    double *centroid_coords = GD->D.centroid_coordinates;
    double *edge_coords = GD->D.edge_coordinates;

    // Additional arrays for extrapolation
    anuga_int *surrogate_neighbours = GD->D.surrogate_neighbours;
    anuga_int *number_of_boundaries = GD->D.number_of_boundaries;
    double *x_centroid_work = GD->D.x_centroid_work;
    double *y_centroid_work = GD->D.y_centroid_work;

    // Friction array
    double *friction_cv = GD->D.friction_centroid_values;

    // tri_full_flag for MPI ghost cell identification
    anuga_int *tri_full_flag = GD->D.tri_full_flag;

    // Unmap domain arrays
    #pragma omp target exit data map(delete: \
        stage_cv[0:n], xmom_cv[0:n], ymom_cv[0:n], \
        bed_cv[0:n], height_cv[0:n], friction_cv[0:n], \
        stage_ev[0:3*n], xmom_ev[0:3*n], ymom_ev[0:3*n], \
        bed_ev[0:3*n], height_ev[0:3*n], \
        stage_eu[0:n], xmom_eu[0:n], ymom_eu[0:n], \
        stage_siu[0:n], xmom_siu[0:n], ymom_siu[0:n], \
        neighbours[0:3*n], neighbour_edges[0:3*n], \
        surrogate_neighbours[0:3*n], number_of_boundaries[0:n], \
        x_centroid_work[0:n], y_centroid_work[0:n], \
        normals[0:6*n], edgelengths[0:3*n], \
        areas[0:n], radii[0:n], max_speed[0:n], \
        centroid_coords[0:2*n], edge_coords[0:6*n])

    // Unmap tri_full_flag only when it was mapped (parallel domains only)
    if (tri_full_flag != NULL) {
        #pragma omp target exit data map(delete: tri_full_flag[0:n])
    }

    if (nb > 0) {
        double *stage_bv = GD->D.stage_boundary_values;
        double *xmom_bv = GD->D.xmom_boundary_values;
        double *ymom_bv = GD->D.ymom_boundary_values;
        double *bed_bv = GD->D.bed_boundary_values;
        double *height_bv = GD->D.height_boundary_values;
        #pragma omp target exit data map(delete: \
            stage_bv[0:nb], xmom_bv[0:nb], ymom_bv[0:nb], \
            bed_bv[0:nb], height_bv[0:nb])
    }

    // Unmap reflective boundary arrays if mapped
    struct reflective_boundary *R = &GD->reflective;
    if (R->mapped && R->num_edges > 0) {
        int ne = R->num_edges;
        int *b_idx = R->boundary_indices;
        int *v_ids = R->vol_ids;
        int *e_ids = R->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        R->mapped = 0;
    }

    // Unmap dirichlet boundary arrays if mapped
    struct dirichlet_boundary *Dir = &GD->dirichlet;
    if (Dir->mapped && Dir->num_edges > 0) {
        int ne = Dir->num_edges;
        int *b_idx = Dir->boundary_indices;
        int *v_ids = Dir->vol_ids;
        int *e_ids = Dir->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        Dir->mapped = 0;
    }

    // Unmap transmissive boundary arrays if mapped
    struct transmissive_boundary *T = &GD->transmissive;
    if (T->mapped && T->num_edges > 0) {
        int ne = T->num_edges;
        int *b_idx = T->boundary_indices;
        int *v_ids = T->vol_ids;
        int *e_ids = T->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        T->mapped = 0;
    }

    // Unmap transmissive_n_zero_t boundary arrays if mapped
    struct transmissive_n_zero_t_boundary *Tnzt = &GD->transmissive_n_zero_t;
    if (Tnzt->mapped && Tnzt->num_edges > 0) {
        int ne = Tnzt->num_edges;
        int *b_idx = Tnzt->boundary_indices;
        int *v_ids = Tnzt->vol_ids;
        int *e_ids = Tnzt->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        Tnzt->mapped = 0;
    }

    // Unmap time_boundary arrays if mapped
    struct time_boundary *TB = &GD->time_bdry;
    if (TB->mapped && TB->num_edges > 0) {
        int ne = TB->num_edges;
        int *b_idx = TB->boundary_indices;
        int *v_ids = TB->vol_ids;
        int *e_ids = TB->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
        TB->mapped = 0;
    }

    // Unmap riverwall arrays
    anuga_int *edge_flux_type = GD->D.edge_flux_type;
    if (edge_flux_type != NULL) {
        #pragma omp target exit data map(delete: edge_flux_type[0:3*n])
    }

    anuga_int n_rw_edges = GD->D.number_of_riverwall_edges;
    if (n_rw_edges > 0) {
        anuga_int *edge_river_wall_counter = GD->D.edge_river_wall_counter;
        double *riverwall_elevation = GD->D.riverwall_elevation;
        anuga_int *riverwall_rowIndex = GD->D.riverwall_rowIndex;
        double *riverwall_hydraulic_properties = GD->D.riverwall_hydraulic_properties;
        anuga_int ncol_hp = GD->D.ncol_riverwall_hydraulic_properties;
        anuga_int nrow_hp = GD->D.nrow_riverwall_hydraulic_properties;

        if (edge_river_wall_counter != NULL) {
            #pragma omp target exit data map(delete: edge_river_wall_counter[0:3*n])
        }
        if (riverwall_elevation != NULL) {
            #pragma omp target exit data map(delete: riverwall_elevation[0:n_rw_edges])
        }
        if (riverwall_rowIndex != NULL) {
            #pragma omp target exit data map(delete: riverwall_rowIndex[0:n_rw_edges])
        }
        if (riverwall_hydraulic_properties != NULL && ncol_hp > 0 && nrow_hp > 0) {
            #pragma omp target exit data map(delete: riverwall_hydraulic_properties[0:nrow_hp*ncol_hp])
        }
    }

    if (GD->backup_arrays_mapped) {
        double *stage_backup = GD->D.stage_backup_values;
        double *xmom_backup = GD->D.xmom_backup_values;
        double *ymom_backup = GD->D.ymom_backup_values;
        #pragma omp target exit data map(delete: \
            stage_backup[0:n], xmom_backup[0:n], ymom_backup[0:n])
    }

    // Unmap halo arrays
    if (H->num_neighbors > 0) {
        int send_size = H->total_send_size;
        int recv_size = H->total_recv_size;
        int *flat_send = H->flat_send_indices;
        int *flat_recv = H->flat_recv_indices;
        double *send_buf = H->send_buffer;
        double *recv_buf = H->recv_buffer;

        #pragma omp target exit data map(delete: \
            flat_send[0:send_size], flat_recv[0:recv_size], \
            send_buf[0:3*send_size], recv_buf[0:3*recv_size])
    }

    GD->gpu_initialized = 0;
}

void gpu_domain_sync_to_device(struct gpu_domain *GD) {
    // Sync all centroid values to GPU (use at start of GPU computation)
    if (!GD->gpu_initialized) return;

    anuga_int n = GD->D.number_of_elements;
    double *stage_cv = GD->D.stage_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *height_cv = GD->D.height_centroid_values;

    #pragma omp target update to(stage_cv[0:n], xmom_cv[0:n], ymom_cv[0:n], height_cv[0:n])
}

void gpu_domain_sync_from_device(struct gpu_domain *GD) {
    // Sync centroid values from GPU (use at yieldstep for Python I/O)
    if (!GD->gpu_initialized) return;

    anuga_int n = GD->D.number_of_elements;
    double *stage_cv = GD->D.stage_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *height_cv = GD->D.height_centroid_values;

    #pragma omp target update from(stage_cv[0:n], xmom_cv[0:n], ymom_cv[0:n], height_cv[0:n])
}

void gpu_domain_sync_all_from_device(struct gpu_domain *GD) {
    // Sync ALL arrays from GPU (for debugging/testing intermediate values)
    if (!GD->gpu_initialized) return;

    anuga_int n = GD->D.number_of_elements;
    anuga_int nb = GD->D.boundary_length;

    // Centroid values
    double *stage_cv = GD->D.stage_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *height_cv = GD->D.height_centroid_values;

    // Edge values
    double *stage_ev = GD->D.stage_edge_values;
    double *xmom_ev = GD->D.xmom_edge_values;
    double *ymom_ev = GD->D.ymom_edge_values;
    double *height_ev = GD->D.height_edge_values;
    double *bed_ev = GD->D.bed_edge_values;

    // Explicit and semi-implicit updates
    double *stage_eu = GD->D.stage_explicit_update;
    double *xmom_eu = GD->D.xmom_explicit_update;
    double *ymom_eu = GD->D.ymom_explicit_update;
    double *stage_siu = GD->D.stage_semi_implicit_update;
    double *xmom_siu = GD->D.xmom_semi_implicit_update;
    double *ymom_siu = GD->D.ymom_semi_implicit_update;

    // Sync centroid values
    #pragma omp target update from(stage_cv[0:n], xmom_cv[0:n], ymom_cv[0:n], height_cv[0:n])

    // Sync edge values
    #pragma omp target update from(stage_ev[0:3*n], xmom_ev[0:3*n], ymom_ev[0:3*n], \
                                   height_ev[0:3*n], bed_ev[0:3*n])

    // Sync explicit and semi-implicit updates
    #pragma omp target update from(stage_eu[0:n], xmom_eu[0:n], ymom_eu[0:n], \
                                   stage_siu[0:n], xmom_siu[0:n], ymom_siu[0:n])

    // Sync boundary values if present
    if (nb > 0) {
        double *stage_bv = GD->D.stage_boundary_values;
        double *xmom_bv = GD->D.xmom_boundary_values;
        double *ymom_bv = GD->D.ymom_boundary_values;
        double *height_bv = GD->D.height_boundary_values;
        double *bed_bv = GD->D.bed_boundary_values;

        #pragma omp target update from(stage_bv[0:nb], xmom_bv[0:nb], ymom_bv[0:nb], \
                                       height_bv[0:nb], bed_bv[0:nb])
    }

    // Sync backup values if mapped
    if (GD->backup_arrays_mapped) {
        double *stage_backup = GD->D.stage_backup_values;
        double *xmom_backup = GD->D.xmom_backup_values;
        double *ymom_backup = GD->D.ymom_backup_values;
        #pragma omp target update from(stage_backup[0:n], xmom_backup[0:n], ymom_backup[0:n])
    }
}

void gpu_sync_boundary_values(struct gpu_domain *GD) {
    // Sync boundary values TO GPU (after CPU boundary evaluation)
    if (!GD->gpu_initialized) return;

    anuga_int nb = GD->D.boundary_length;
    if (nb == 0) return;

    double *stage_bv = GD->D.stage_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;
    double *bed_bv = GD->D.bed_boundary_values;
    double *height_bv = GD->D.height_boundary_values;

    #pragma omp target update to(stage_bv[0:nb], xmom_bv[0:nb], ymom_bv[0:nb], \
                                 bed_bv[0:nb], height_bv[0:nb])
}

void gpu_sync_edge_values_from_device(struct gpu_domain *GD) {
    // Sync ALL edge values FROM GPU - expensive, use sparse version if possible
    if (!GD->gpu_initialized) return;

    anuga_int n = GD->D.number_of_elements;

    double *stage_ev = GD->D.stage_edge_values;
    double *xmom_ev = GD->D.xmom_edge_values;
    double *ymom_ev = GD->D.ymom_edge_values;
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;

    #pragma omp target update from(stage_ev[0:3*n], xmom_ev[0:3*n], ymom_ev[0:3*n], \
                                   bed_ev[0:3*n], height_ev[0:3*n])
}

// Initialize boundary edge sync buffers - call once after boundaries are set
int gpu_boundary_edge_sync_init(struct gpu_domain *GD,
                                int num_boundary_cells,
                                int *boundary_cell_ids) {
    if (!GD->gpu_initialized) return -1;

    struct boundary_edge_sync *S = &GD->edge_sync;

    if (S->initialized) {
        // Already initialized - clean up first
        gpu_boundary_edge_sync_finalize(GD);
    }

    S->num_boundary_cells = num_boundary_cells;
    S->buf_size = num_boundary_cells * 3;  // 3 edges per cell

    if (num_boundary_cells == 0) {
        S->initialized = 1;
        return 0;  // Nothing to do
    }

    // Allocate cell IDs array (copy from Python's array)
    S->cell_ids = (int*)malloc(num_boundary_cells * sizeof(int));
    memcpy(S->cell_ids, boundary_cell_ids, num_boundary_cells * sizeof(int));

    // Allocate staging buffers
    S->stage_buf = (double*)malloc(S->buf_size * sizeof(double));
    S->xmom_buf = (double*)malloc(S->buf_size * sizeof(double));
    S->ymom_buf = (double*)malloc(S->buf_size * sizeof(double));
    S->bed_buf = (double*)malloc(S->buf_size * sizeof(double));
    S->height_buf = (double*)malloc(S->buf_size * sizeof(double));

    // Map all buffers to GPU once
    int nc = num_boundary_cells;
    int bs = S->buf_size;
    int *cell_ids_ptr = S->cell_ids;
    double *stage_buf = S->stage_buf;
    double *xmom_buf = S->xmom_buf;
    double *ymom_buf = S->ymom_buf;
    double *bed_buf = S->bed_buf;
    double *height_buf = S->height_buf;

    #pragma omp target enter data map(to: cell_ids_ptr[0:nc]) \
        map(alloc: stage_buf[0:bs], xmom_buf[0:bs], ymom_buf[0:bs], \
                   bed_buf[0:bs], height_buf[0:bs])

    S->initialized = 1;
    if (GD->verbose) {
        printf("Rank %d: Boundary edge sync initialized for %d cells (%d edge values)\n",
               GD->rank, num_boundary_cells, S->buf_size);
        fflush(stdout);
    }

    return 0;
}

// Finalize boundary edge sync buffers
void gpu_boundary_edge_sync_finalize(struct gpu_domain *GD) {
    struct boundary_edge_sync *S = &GD->edge_sync;

    if (!S->initialized) return;

    if (S->num_boundary_cells > 0) {
        // Unmap from GPU
        int nc = S->num_boundary_cells;
        int bs = S->buf_size;
        int *cell_ids_ptr = S->cell_ids;
        double *stage_buf = S->stage_buf;
        double *xmom_buf = S->xmom_buf;
        double *ymom_buf = S->ymom_buf;
        double *bed_buf = S->bed_buf;
        double *height_buf = S->height_buf;

        #pragma omp target exit data map(delete: cell_ids_ptr[0:nc], \
            stage_buf[0:bs], xmom_buf[0:bs], ymom_buf[0:bs], \
            bed_buf[0:bs], height_buf[0:bs])

        // Free host memory
        free(S->cell_ids);
        free(S->stage_buf);
        free(S->xmom_buf);
        free(S->ymom_buf);
        free(S->bed_buf);
        free(S->height_buf);
    }

    S->cell_ids = NULL;
    S->stage_buf = NULL;
    S->xmom_buf = NULL;
    S->ymom_buf = NULL;
    S->bed_buf = NULL;
    S->height_buf = NULL;
    S->num_boundary_cells = 0;
    S->buf_size = 0;
    S->initialized = 0;
}

// Sync boundary edge values from GPU - fast version using pre-allocated buffers
void gpu_boundary_edge_sync(struct gpu_domain *GD) {
    struct boundary_edge_sync *S = &GD->edge_sync;

    if (!S->initialized || S->num_boundary_cells == 0) return;

    int nc = S->num_boundary_cells;
    int bs = S->buf_size;
    int *cell_ids = S->cell_ids;
    double *stage_buf = S->stage_buf;
    double *xmom_buf = S->xmom_buf;
    double *ymom_buf = S->ymom_buf;
    double *bed_buf = S->bed_buf;
    double *height_buf = S->height_buf;

    double *stage_ev = GD->D.stage_edge_values;
    double *xmom_ev = GD->D.xmom_edge_values;
    double *ymom_ev = GD->D.ymom_edge_values;
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;

    // Gather on GPU into contiguous staging buffers
    OMP_PARALLEL_LOOP
    for (int i = 0; i < nc; i++) {
        int vid = cell_ids[i];
        for (int e = 0; e < 3; e++) {
            int buf_idx = i * 3 + e;
            int src_idx = vid * 3 + e;
            stage_buf[buf_idx] = stage_ev[src_idx];
            xmom_buf[buf_idx] = xmom_ev[src_idx];
            ymom_buf[buf_idx] = ymom_ev[src_idx];
            bed_buf[buf_idx] = bed_ev[src_idx];
            height_buf[buf_idx] = height_ev[src_idx];
        }
    }

    // Sync staging buffers from GPU to host
    #pragma omp target update from(stage_buf[0:bs], xmom_buf[0:bs], ymom_buf[0:bs], \
                                   bed_buf[0:bs], height_buf[0:bs])

    // Scatter on CPU to the actual edge value arrays (host copies)
    for (int i = 0; i < nc; i++) {
        int vid = S->cell_ids[i];  // Use host copy of cell_ids
        for (int e = 0; e < 3; e++) {
            int buf_idx = i * 3 + e;
            int dst_idx = vid * 3 + e;
            stage_ev[dst_idx] = stage_buf[buf_idx];
            xmom_ev[dst_idx] = xmom_buf[buf_idx];
            ymom_ev[dst_idx] = ymom_buf[buf_idx];
            bed_ev[dst_idx] = bed_buf[buf_idx];
            height_ev[dst_idx] = height_buf[buf_idx];
        }
    }
}

