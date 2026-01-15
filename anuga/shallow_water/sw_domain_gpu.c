// GPU-accelerated shallow water solver with MPI halo exchange
//
// Implements:
// - MPI communicator integration (passed from mpi4py)
// - GPU-aware MPI halo exchange (with fallback for non-GPU-aware MPI)
// - OpenMP target offloading for GPU kernels
//
// Based on miniapp_mpi.c patterns adapted for ANUGA's data structures

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "sw_domain_gpu.h"

// ============================================================================
// FLOP Counting Constants (Gordon Bell Performance Profiling)
// ============================================================================
//
// FLOP counts per kernel per element (triangle):
// Counted: +, -, *, /, sqrt, fmin, fmax, pow (comparisons not counted)
//
// | Kernel                      | FLOPs/element | Notes                              |
// |-----------------------------|---------------|-----------------------------------|
// | extrapolate_second_order    | 220           | Gradient limiting, 4 quantities   |
// | compute_fluxes              | 400           | 3 edges × flux function + pressure|
// | update_conserved_quantities | 21            | Explicit + semi-implicit          |
// | protect                     | 5             | Depth check, mass error           |
// | manning_friction            | 18            | sqrt, pow(7/3), semi-implicit     |
// | backup_conserved_quantities | 0             | Memory copy only                  |
// | saxpy_conserved_quantities  | 9             | 3 × (2 mul + 1 add)               |
// | rate_operator_apply         | 8             | Per affected cell                 |
// | ghost_exchange              | 0             | Memory operations only            |
//
// Total per RK2 step: ~673 FLOPs/element (excluding rate_operator)

#define FLOPS_EXTRAPOLATE        220
#define FLOPS_COMPUTE_FLUXES     400
#define FLOPS_UPDATE             21
#define FLOPS_PROTECT            5
#define FLOPS_MANNING            18
#define FLOPS_BACKUP             0
#define FLOPS_SAXPY              9
#define FLOPS_RATE_OPERATOR      8
#define FLOPS_GHOST_EXCHANGE     0

// ============================================================================
// Flux Computation Helper Functions (device code)
// These are marked with #pragma omp declare target for GPU execution
// ============================================================================

#pragma omp declare target

// Small constant for avoiding division by zero
static const double GPU_TINY = 1.0e-100;

// ============================================================================
// Extrapolation Helper Functions (device code)
// ============================================================================

// Find qmin and qmax from three differences
static inline void gpu_find_qmin_and_qmax_dq1_dq2(
    double dq0, double dq1, double dq2,
    double *qmin, double *qmax) {
    *qmax = fmax(fmax(dq0, fmax(dq0 + dq1, dq0 + dq2)), 0.0);
    *qmin = fmin(fmin(dq0, fmin(dq0 + dq1, dq0 + dq2)), 0.0);
}

// Find qmin and qmax from single difference
static inline void gpu_compute_qmin_qmax_from_dq1(double dq1, double *qmin, double *qmax) {
    if (dq1 >= 0.0) {
        *qmin = 0.0;
        *qmax = dq1;
    } else {
        *qmin = dq1;
        *qmax = 0.0;
    }
}

// Limit gradient to enforce monotonicity
static inline void gpu_limit_gradient(double *dqv, double qmin, double qmax, double beta_w) {
    double r = 1000.0;

    double dq_x = dqv[0];
    double dq_y = dqv[1];
    double dq_z = dqv[2];

    if (dq_x < -GPU_TINY) {
        r = fmin(r, qmin / dq_x);
    } else if (dq_x > GPU_TINY) {
        r = fmin(r, qmax / dq_x);
    }
    if (dq_y < -GPU_TINY) {
        r = fmin(r, qmin / dq_y);
    } else if (dq_y > GPU_TINY) {
        r = fmin(r, qmax / dq_y);
    }
    if (dq_z < -GPU_TINY) {
        r = fmin(r, qmin / dq_z);
    } else if (dq_z > GPU_TINY) {
        r = fmin(r, qmax / dq_z);
    }

    double phi = fmin(r * beta_w, 1.0);

    dqv[0] *= phi;
    dqv[1] *= phi;
    dqv[2] *= phi;
}

// Compute edge values with gradient limiting (for typical case with 3 neighbors)
static inline void gpu_calc_edge_values_with_gradient(
    double cv_k, double cv_k0, double cv_k1, double cv_k2,
    double dxv0, double dxv1, double dxv2,
    double dyv0, double dyv1, double dyv2,
    double dx1, double dx2, double dy1, double dy2,
    double inv_area2, double beta_tmp, double *edge_values) {

    double dqv[3];
    double dq0 = cv_k0 - cv_k;
    double dq1 = cv_k1 - cv_k0;
    double dq2 = cv_k2 - cv_k0;

    double a = (dy2 * dq1 - dy1 * dq2) * inv_area2;
    double b = (dx1 * dq2 - dx2 * dq1) * inv_area2;

    dqv[0] = a * dxv0 + b * dyv0;
    dqv[1] = a * dxv1 + b * dyv1;
    dqv[2] = a * dxv2 + b * dyv2;

    double qmin, qmax;
    gpu_find_qmin_and_qmax_dq1_dq2(dq0, dq1, dq2, &qmin, &qmax);
    gpu_limit_gradient(dqv, qmin, qmax, beta_tmp);

    edge_values[0] = cv_k + dqv[0];
    edge_values[1] = cv_k + dqv[1];
    edge_values[2] = cv_k + dqv[2];
}

// Set constant edge values (for zero beta or boundary cases)
static inline void gpu_set_constant_edge_values(double cv_k, double *edge_values) {
    edge_values[0] = cv_k;
    edge_values[1] = cv_k;
    edge_values[2] = cv_k;
}

// Compute dqv from gradient (for boundary case with single neighbor)
static inline void gpu_compute_dqv_from_gradient(
    double dq1, double dx2, double dy2,
    double dxv0, double dxv1, double dxv2,
    double dyv0, double dyv1, double dyv2,
    double *dqv) {
    double a = dq1 * dx2;
    double b = dq1 * dy2;

    dqv[0] = a * dxv0 + b * dyv0;
    dqv[1] = a * dxv1 + b * dyv1;
    dqv[2] = a * dxv2 + b * dyv2;
}

// ============================================================================
// Flux Computation Helper Functions (device code)
// ============================================================================

// Rotate momentum components to align with edge normal
static inline void gpu_rotate(double *q, double n1, double n2) {
    double q1 = q[1];
    double q2 = q[2];
    q[1] = n1 * q1 + n2 * q2;
    q[2] = -n2 * q1 + n1 * q2;
}

// Compute velocity terms with zero-depth handling
static inline void gpu_compute_velocity_terms(
    double h, double h_edge,
    double uh_raw, double vh_raw,
    double *u, double *uh, double *v, double *vh) {
    if (h_edge > 0.0) {
        double inv_h_edge = 1.0 / h_edge;
        *u = uh_raw * inv_h_edge;
        *uh = h * (*u);
        *v = vh_raw * inv_h_edge;
        *vh = h * inv_h_edge * vh_raw;
    } else {
        *u = 0.0;
        *uh = 0.0;
        *v = 0.0;
        *vh = 0.0;
    }
}

// Compute local Froude number for low-Froude corrections
static inline double gpu_compute_local_froude(
    anuga_int low_froude,
    double u_left, double u_right,
    double v_left, double v_right,
    double soundspeed_left, double soundspeed_right) {
    double numerator = u_right * u_right + u_left * u_left +
                       v_right * v_right + v_left * v_left;
    double denominator = soundspeed_left * soundspeed_left +
                         soundspeed_right * soundspeed_right + 1.0e-10;

    if (low_froude == 1) {
        return sqrt(fmax(0.001, fmin(1.0, numerator / denominator)));
    } else if (low_froude == 2) {
        double fr = sqrt(numerator / denominator);
        return sqrt(fmin(1.0, 0.01 + fmax(fr - 0.01, 0.0)));
    } else {
        return 1.0;
    }
}

// Maximum wave speed (positive direction)
static inline double gpu_compute_s_max(double u_left, double u_right,
                                       double c_left, double c_right) {
    double s = fmax(u_left + c_left, u_right + c_right);
    return (s < 0.0) ? 0.0 : s;
}

// Minimum wave speed (negative direction)
static inline double gpu_compute_s_min(double u_left, double u_right,
                                       double c_left, double c_right) {
    double s = fmin(u_left - c_left, u_right - c_right);
    return (s > 0.0) ? 0.0 : s;
}

// Central flux function - Kurganov-Noelle-Petrova scheme
static inline void gpu_flux_function_central(
    double *q_left, double *q_right,
    double h_left, double h_right,
    double hle, double hre,
    double n1, double n2,
    double epsilon, double ze, double g,
    double *edgeflux, double *max_speed, double *pressure_flux,
    anuga_int low_froude) {

    double uh_left, vh_left, u_left, v_left;
    double uh_right, vh_right, u_right, v_right;
    double soundspeed_left, soundspeed_right;
    double q_left_rotated[3], q_right_rotated[3];
    double flux_left[3], flux_right[3];

    // Copy and rotate to edge-aligned coordinates
    for (int i = 0; i < 3; i++) {
        q_left_rotated[i] = q_left[i];
        q_right_rotated[i] = q_right[i];
    }
    gpu_rotate(q_left_rotated, n1, n2);
    gpu_rotate(q_right_rotated, n1, n2);

    // Compute velocities
    uh_left = q_left_rotated[1];
    vh_left = q_left_rotated[2];
    gpu_compute_velocity_terms(h_left, hle, q_left_rotated[1], q_left_rotated[2],
                               &u_left, &uh_left, &v_left, &vh_left);

    uh_right = q_right_rotated[1];
    vh_right = q_right_rotated[2];
    gpu_compute_velocity_terms(h_right, hre, q_right_rotated[1], q_right_rotated[2],
                               &u_right, &uh_right, &v_right, &vh_right);

    // Wave speeds
    soundspeed_left = sqrt(g * h_left);
    soundspeed_right = sqrt(g * h_right);

    double local_fr = gpu_compute_local_froude(low_froude, u_left, u_right,
                                               v_left, v_right,
                                               soundspeed_left, soundspeed_right);

    double s_max = gpu_compute_s_max(u_left, u_right, soundspeed_left, soundspeed_right);
    double s_min = gpu_compute_s_min(u_left, u_right, soundspeed_left, soundspeed_right);

    // Physical fluxes
    flux_left[0] = u_left * h_left;
    flux_left[1] = u_left * uh_left;
    flux_left[2] = u_left * vh_left;

    flux_right[0] = u_right * h_right;
    flux_right[1] = u_right * uh_right;
    flux_right[2] = u_right * vh_right;

    // Central upwind flux
    double denom = s_max - s_min;
    double inverse_denominator = 1.0 / fmax(denom, 1.0e-100);
    double s_max_s_min = s_max * s_min;

    if (denom < epsilon) {
        // Both wave speeds very small
        edgeflux[0] = 0.0;
        edgeflux[1] = 0.0;
        edgeflux[2] = 0.0;
        *max_speed = 0.0;
        *pressure_flux = 0.5 * g * 0.5 * (h_left * h_left + h_right * h_right);
    } else {
        *max_speed = fmax(s_max, -s_min);

        double flux_0 = s_max * flux_left[0] - s_min * flux_right[0];
        flux_0 += s_max_s_min * (fmax(q_right_rotated[0], ze) - fmax(q_left_rotated[0], ze));
        edgeflux[0] = flux_0 * inverse_denominator;

        double flux_1 = s_max * flux_left[1] - s_min * flux_right[1];
        flux_1 += local_fr * s_max_s_min * (uh_right - uh_left);
        edgeflux[1] = flux_1 * inverse_denominator;

        double flux_2 = s_max * flux_left[2] - s_min * flux_right[2];
        flux_2 += local_fr * s_max_s_min * (vh_right - vh_left);
        edgeflux[2] = flux_2 * inverse_denominator;

        *pressure_flux = 0.5 * g * (s_max * h_left * h_left - s_min * h_right * h_right) * inverse_denominator;

        // Rotate back
        gpu_rotate(edgeflux, n1, -n2);
    }
}

#pragma omp end declare target

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
    printf("GPU Domain Info (rank %d/%d):\n", GD->rank, GD->nprocs);
    printf("  Elements: %" PRId64 "\n", GD->D.number_of_elements);
    printf("  GPU initialized: %d\n", GD->gpu_initialized);
    printf("  GPU-aware MPI: %d\n", GD->gpu_aware_mpi);
    printf("  Halo neighbors: %d\n", GD->halo.num_neighbors);
    if (GD->halo.num_neighbors > 0) {
        printf("  Total send: %d, Total recv: %d\n",
               GD->halo.total_send_size, GD->halo.total_recv_size);
    }
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
        if (rank == 0) {
            fprintf(stderr, "Warning: No GPU devices found, will use CPU fallback\n");
        }
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

    GD->gpu_initialized = 0;
}

// ============================================================================
// Halo Exchange Setup
// ============================================================================

int gpu_halo_init(struct gpu_domain *GD,
                  int num_neighbors,
                  int *neighbor_ranks,
                  int *send_counts,
                  int *recv_counts,
                  int *flat_send_indices,
                  int *flat_recv_indices) {
    struct halo_exchange *H = &GD->halo;

    H->num_neighbors = num_neighbors;

    if (num_neighbors == 0) {
        // No communication needed
        return 0;
    }

    // Allocate and copy neighbor info
    H->neighbor_ranks = (int *)malloc(num_neighbors * sizeof(int));
    H->send_counts = (int *)malloc(num_neighbors * sizeof(int));
    H->recv_counts = (int *)malloc(num_neighbors * sizeof(int));
    H->send_offsets = (int *)malloc((num_neighbors + 1) * sizeof(int));
    H->recv_offsets = (int *)malloc((num_neighbors + 1) * sizeof(int));

    memcpy(H->neighbor_ranks, neighbor_ranks, num_neighbors * sizeof(int));
    memcpy(H->send_counts, send_counts, num_neighbors * sizeof(int));
    memcpy(H->recv_counts, recv_counts, num_neighbors * sizeof(int));

    // Compute total sizes and offsets
    H->total_send_size = 0;
    H->total_recv_size = 0;
    H->send_offsets[0] = 0;
    H->recv_offsets[0] = 0;

    for (int ni = 0; ni < num_neighbors; ni++) {
        H->total_send_size += send_counts[ni];
        H->total_recv_size += recv_counts[ni];
        H->send_offsets[ni + 1] = H->total_send_size;
        H->recv_offsets[ni + 1] = H->total_recv_size;
    }

    // Allocate and copy flattened index arrays
    H->flat_send_indices = (int *)malloc(H->total_send_size * sizeof(int));
    H->flat_recv_indices = (int *)malloc(H->total_recv_size * sizeof(int));

    memcpy(H->flat_send_indices, flat_send_indices, H->total_send_size * sizeof(int));
    memcpy(H->flat_recv_indices, flat_recv_indices, H->total_recv_size * sizeof(int));

    // Allocate communication buffers
    // 3 quantities per element: stage, xmom, ymom centroid values
    H->send_buffer = (double *)malloc(3 * H->total_send_size * sizeof(double));
    H->recv_buffer = (double *)malloc(3 * H->total_recv_size * sizeof(double));

    // Allocate MPI request array
    H->requests = (MPI_Request *)malloc(2 * num_neighbors * sizeof(MPI_Request));

    if (GD->rank == 0) {
        printf("GPU halo exchange initialized:\n");
        printf("  Neighbors: %d\n", num_neighbors);
        printf("  Total send: %d elements\n", H->total_send_size);
        printf("  Total recv: %d elements\n", H->total_recv_size);
    }

    return 0;
}

void gpu_halo_finalize(struct gpu_domain *GD) {
    struct halo_exchange *H = &GD->halo;

    if (H->neighbor_ranks) free(H->neighbor_ranks);
    if (H->send_counts) free(H->send_counts);
    if (H->recv_counts) free(H->recv_counts);
    if (H->send_offsets) free(H->send_offsets);
    if (H->recv_offsets) free(H->recv_offsets);
    if (H->flat_send_indices) free(H->flat_send_indices);
    if (H->flat_recv_indices) free(H->flat_recv_indices);
    if (H->send_buffer) free(H->send_buffer);
    if (H->recv_buffer) free(H->recv_buffer);
    if (H->requests) free(H->requests);

    H->num_neighbors = 0;
    H->neighbor_ranks = NULL;
    H->send_counts = NULL;
    H->recv_counts = NULL;
    H->send_offsets = NULL;
    H->recv_offsets = NULL;
    H->flat_send_indices = NULL;
    H->flat_recv_indices = NULL;
    H->send_buffer = NULL;
    H->recv_buffer = NULL;
    H->requests = NULL;
}

// ============================================================================
// Reflective Boundary Setup
// ============================================================================

int gpu_reflective_init(struct gpu_domain *GD, int num_edges,
                        int *boundary_indices, int *vol_ids, int *edge_ids) {
    struct reflective_boundary *R = &GD->reflective;

    R->num_edges = num_edges;
    R->mapped = 0;  // Will be mapped later in gpu_domain_map_arrays

    if (num_edges == 0) {
        R->boundary_indices = NULL;
        R->vol_ids = NULL;
        R->edge_ids = NULL;
        return 0;
    }

    // Allocate and copy arrays (will be mapped to GPU in gpu_domain_map_arrays)
    R->boundary_indices = (int*)malloc(num_edges * sizeof(int));
    R->vol_ids = (int*)malloc(num_edges * sizeof(int));
    R->edge_ids = (int*)malloc(num_edges * sizeof(int));

    if (!R->boundary_indices || !R->vol_ids || !R->edge_ids) {
        fprintf(stderr, "Failed to allocate reflective boundary arrays\n");
        return -1;
    }

    memcpy(R->boundary_indices, boundary_indices, num_edges * sizeof(int));
    memcpy(R->vol_ids, vol_ids, num_edges * sizeof(int));
    memcpy(R->edge_ids, edge_ids, num_edges * sizeof(int));

    // NOTE: GPU mapping now happens in gpu_domain_map_arrays() to ensure
    // all arrays are mapped in the same data region. Call gpu_reflective_init
    // BEFORE gpu_domain_map_arrays().

    if (GD->rank == 0) {
        printf("Reflective boundary initialized: %d edges (GPU mapping deferred)\n", num_edges);
    }

    return 0;
}

void gpu_reflective_finalize(struct gpu_domain *GD) {
    struct reflective_boundary *R = &GD->reflective;

    if (R->mapped && R->num_edges > 0) {
        int *b_idx = R->boundary_indices;
        int *v_ids = R->vol_ids;
        int *e_ids = R->edge_ids;
        int ne = R->num_edges;

        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
    }

    if (R->boundary_indices) free(R->boundary_indices);
    if (R->vol_ids) free(R->vol_ids);
    if (R->edge_ids) free(R->edge_ids);

    R->num_edges = 0;
    R->boundary_indices = NULL;
    R->vol_ids = NULL;
    R->edge_ids = NULL;
    R->mapped = 0;
}

void gpu_evaluate_reflective_boundary(struct gpu_domain *GD) {
    // Evaluate reflective boundary entirely on GPU
    // No data transfer needed - reads edge values, writes boundary values

    struct reflective_boundary *R = &GD->reflective;
    if (R->num_edges == 0) return;

    // Only run GPU kernel if arrays are mapped
    if (!R->mapped) {
        // Arrays not on GPU - caller should use CPU fallback
        return;
    }

    int num_edges = R->num_edges;
    int *boundary_indices = R->boundary_indices;
    int *vol_ids = R->vol_ids;
    int *edge_ids = R->edge_ids;

    // Edge values (read)
    double *stage_ev = GD->D.stage_edge_values;
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;
    double *xmom_ev = GD->D.xmom_edge_values;
    double *ymom_ev = GD->D.ymom_edge_values;
    double *normals = GD->D.normals;

    // Boundary values (write)
    double *stage_bv = GD->D.stage_boundary_values;
    double *bed_bv = GD->D.bed_boundary_values;
    double *height_bv = GD->D.height_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;

    // All arrays already mapped via target enter data
    #pragma omp target teams distribute parallel for
    for (int k = 0; k < num_edges; k++) {
        int bid = boundary_indices[k];
        int vid = vol_ids[k];
        int eid = edge_ids[k];

        // Copy conserved quantities from edge to boundary
        stage_bv[bid] = stage_ev[3 * vid + eid];
        bed_bv[bid] = bed_ev[3 * vid + eid];
        height_bv[bid] = height_ev[3 * vid + eid];

        // Get normal vector for this edge
        double n1 = normals[vid * 6 + 2 * eid];
        double n2 = normals[vid * 6 + 2 * eid + 1];

        // Get interior momentum
        double q1 = xmom_ev[3 * vid + eid];
        double q2 = ymom_ev[3 * vid + eid];

        // Reflect momentum: negate normal component, keep tangential
        // r = q - 2*(q.n)*n  but we compute it via rotation
        double r1 = -q1 * n1 - q2 * n2;  // -(q dot n)
        double r2 = -q1 * n2 + q2 * n1;  // tangential component

        // Rotate back
        xmom_bv[bid] = n1 * r1 - n2 * r2;
        ymom_bv[bid] = n2 * r1 + n1 * r2;
    }
}

// ============================================================================
// Dirichlet Boundary Setup and Evaluation
// ============================================================================

int gpu_dirichlet_init(struct gpu_domain *GD, int num_edges,
                       int *boundary_indices, int *vol_ids, int *edge_ids,
                       double stage_value, double xmom_value, double ymom_value) {
    struct dirichlet_boundary *D = &GD->dirichlet;

    D->num_edges = num_edges;
    D->mapped = 0;
    D->stage_value = stage_value;
    D->xmom_value = xmom_value;
    D->ymom_value = ymom_value;

    if (num_edges == 0) {
        D->boundary_indices = NULL;
        D->vol_ids = NULL;
        D->edge_ids = NULL;
        return 0;
    }

    D->boundary_indices = (int*)malloc(num_edges * sizeof(int));
    D->vol_ids = (int*)malloc(num_edges * sizeof(int));
    D->edge_ids = (int*)malloc(num_edges * sizeof(int));

    if (!D->boundary_indices || !D->vol_ids || !D->edge_ids) {
        fprintf(stderr, "Failed to allocate Dirichlet boundary arrays\n");
        return -1;
    }

    memcpy(D->boundary_indices, boundary_indices, num_edges * sizeof(int));
    memcpy(D->vol_ids, vol_ids, num_edges * sizeof(int));
    memcpy(D->edge_ids, edge_ids, num_edges * sizeof(int));

    if (GD->rank == 0) {
        printf("Dirichlet boundary initialized: %d edges (GPU mapping deferred)\n", num_edges);
    }

    return 0;
}

void gpu_dirichlet_finalize(struct gpu_domain *GD) {
    struct dirichlet_boundary *D = &GD->dirichlet;

    if (D->mapped && D->num_edges > 0) {
        int ne = D->num_edges;
        int *b_idx = D->boundary_indices;
        int *v_ids = D->vol_ids;
        int *e_ids = D->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
    }

    if (D->boundary_indices) free(D->boundary_indices);
    if (D->vol_ids) free(D->vol_ids);
    if (D->edge_ids) free(D->edge_ids);

    D->num_edges = 0;
    D->boundary_indices = NULL;
    D->vol_ids = NULL;
    D->edge_ids = NULL;
    D->mapped = 0;
}

void gpu_evaluate_dirichlet_boundary(struct gpu_domain *GD) {
    struct dirichlet_boundary *D = &GD->dirichlet;
    if (D->num_edges == 0) return;
    if (!D->mapped) return;

    int num_edges = D->num_edges;
    int *boundary_indices = D->boundary_indices;
    int *vol_ids = D->vol_ids;
    int *edge_ids = D->edge_ids;
    double stage_val = D->stage_value;
    double xmom_val = D->xmom_value;
    double ymom_val = D->ymom_value;

    // Edge values (read for bed/height)
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;

    // Boundary values (write)
    double *stage_bv = GD->D.stage_boundary_values;
    double *bed_bv = GD->D.bed_boundary_values;
    double *height_bv = GD->D.height_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < num_edges; k++) {
        int bid = boundary_indices[k];
        int vid = vol_ids[k];
        int eid = edge_ids[k];

        // Set constant Dirichlet values
        stage_bv[bid] = stage_val;
        xmom_bv[bid] = xmom_val;
        ymom_bv[bid] = ymom_val;

        // Copy bed/height from interior edge
        bed_bv[bid] = bed_ev[3 * vid + eid];
        height_bv[bid] = height_ev[3 * vid + eid];
    }
}

// ============================================================================
// Transmissive Boundary Setup and Evaluation
// ============================================================================

int gpu_transmissive_init(struct gpu_domain *GD, int num_edges,
                          int *boundary_indices, int *vol_ids, int *edge_ids,
                          int use_centroid) {
    struct transmissive_boundary *T = &GD->transmissive;

    T->num_edges = num_edges;
    T->mapped = 0;
    T->use_centroid = use_centroid;

    if (num_edges == 0) {
        T->boundary_indices = NULL;
        T->vol_ids = NULL;
        T->edge_ids = NULL;
        return 0;
    }

    T->boundary_indices = (int*)malloc(num_edges * sizeof(int));
    T->vol_ids = (int*)malloc(num_edges * sizeof(int));
    T->edge_ids = (int*)malloc(num_edges * sizeof(int));

    if (!T->boundary_indices || !T->vol_ids || !T->edge_ids) {
        fprintf(stderr, "Failed to allocate Transmissive boundary arrays\n");
        return -1;
    }

    memcpy(T->boundary_indices, boundary_indices, num_edges * sizeof(int));
    memcpy(T->vol_ids, vol_ids, num_edges * sizeof(int));
    memcpy(T->edge_ids, edge_ids, num_edges * sizeof(int));

    if (GD->rank == 0) {
        printf("Transmissive boundary initialized: %d edges, use_centroid=%d (GPU mapping deferred)\n",
               num_edges, use_centroid);
    }

    return 0;
}

void gpu_transmissive_finalize(struct gpu_domain *GD) {
    struct transmissive_boundary *T = &GD->transmissive;

    if (T->mapped && T->num_edges > 0) {
        int ne = T->num_edges;
        int *b_idx = T->boundary_indices;
        int *v_ids = T->vol_ids;
        int *e_ids = T->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
    }

    if (T->boundary_indices) free(T->boundary_indices);
    if (T->vol_ids) free(T->vol_ids);
    if (T->edge_ids) free(T->edge_ids);

    T->num_edges = 0;
    T->boundary_indices = NULL;
    T->vol_ids = NULL;
    T->edge_ids = NULL;
    T->mapped = 0;
}

void gpu_evaluate_transmissive_boundary(struct gpu_domain *GD) {
    struct transmissive_boundary *T = &GD->transmissive;
    if (T->num_edges == 0) return;
    if (!T->mapped) return;

    int num_edges = T->num_edges;
    int *boundary_indices = T->boundary_indices;
    int *vol_ids = T->vol_ids;
    int *edge_ids = T->edge_ids;
    int use_centroid = T->use_centroid;

    // Centroid values (for use_centroid mode)
    double *stage_cv = GD->D.stage_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *bed_cv = GD->D.bed_centroid_values;
    double *height_cv = GD->D.height_centroid_values;

    // Edge values
    double *stage_ev = GD->D.stage_edge_values;
    double *xmom_ev = GD->D.xmom_edge_values;
    double *ymom_ev = GD->D.ymom_edge_values;
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;

    // Boundary values (write)
    double *stage_bv = GD->D.stage_boundary_values;
    double *bed_bv = GD->D.bed_boundary_values;
    double *height_bv = GD->D.height_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < num_edges; k++) {
        int bid = boundary_indices[k];
        int vid = vol_ids[k];
        int eid = edge_ids[k];

        if (use_centroid) {
            // Copy from centroid values
            stage_bv[bid] = stage_cv[vid];
            xmom_bv[bid] = xmom_cv[vid];
            ymom_bv[bid] = ymom_cv[vid];
            bed_bv[bid] = bed_cv[vid];
            height_bv[bid] = height_cv[vid];
        } else {
            // Copy from edge values
            stage_bv[bid] = stage_ev[3 * vid + eid];
            xmom_bv[bid] = xmom_ev[3 * vid + eid];
            ymom_bv[bid] = ymom_ev[3 * vid + eid];
            bed_bv[bid] = bed_ev[3 * vid + eid];
            height_bv[bid] = height_ev[3 * vid + eid];
        }
    }
}

// ============================================================================
// Transmissive_n_momentum_zero_t_momentum_set_stage Boundary
// ============================================================================

int gpu_transmissive_n_zero_t_init(struct gpu_domain *GD, int num_edges,
                                   int *boundary_indices, int *vol_ids, int *edge_ids) {
    struct transmissive_n_zero_t_boundary *B = &GD->transmissive_n_zero_t;

    B->num_edges = num_edges;
    B->mapped = 0;
    B->stage_value = 0.0;

    if (num_edges == 0) {
        B->boundary_indices = NULL;
        B->vol_ids = NULL;
        B->edge_ids = NULL;
        return 0;
    }

    B->boundary_indices = (int*)malloc(num_edges * sizeof(int));
    B->vol_ids = (int*)malloc(num_edges * sizeof(int));
    B->edge_ids = (int*)malloc(num_edges * sizeof(int));

    if (!B->boundary_indices || !B->vol_ids || !B->edge_ids) {
        fprintf(stderr, "Failed to allocate transmissive_n_zero_t boundary arrays\n");
        return -1;
    }

    memcpy(B->boundary_indices, boundary_indices, num_edges * sizeof(int));
    memcpy(B->vol_ids, vol_ids, num_edges * sizeof(int));
    memcpy(B->edge_ids, edge_ids, num_edges * sizeof(int));

    if (GD->rank == 0) {
        printf("Transmissive_n_zero_t boundary initialized: %d edges (GPU mapping deferred)\n", num_edges);
    }

    return 0;
}

void gpu_transmissive_n_zero_t_finalize(struct gpu_domain *GD) {
    struct transmissive_n_zero_t_boundary *B = &GD->transmissive_n_zero_t;

    if (B->mapped && B->num_edges > 0) {
        int ne = B->num_edges;
        int *b_idx = B->boundary_indices;
        int *v_ids = B->vol_ids;
        int *e_ids = B->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
    }

    if (B->boundary_indices) free(B->boundary_indices);
    if (B->vol_ids) free(B->vol_ids);
    if (B->edge_ids) free(B->edge_ids);

    B->num_edges = 0;
    B->boundary_indices = NULL;
    B->vol_ids = NULL;
    B->edge_ids = NULL;
    B->mapped = 0;
}

void gpu_transmissive_n_zero_t_set_stage(struct gpu_domain *GD, double stage_value) {
    // Update stage value - called from Python each timestep before evaluate
    GD->transmissive_n_zero_t.stage_value = stage_value;
}

void gpu_evaluate_transmissive_n_zero_t_boundary(struct gpu_domain *GD) {
    // Transmissive normal momentum, zero tangential momentum, set stage
    struct transmissive_n_zero_t_boundary *B = &GD->transmissive_n_zero_t;
    if (B->num_edges == 0) return;
    if (!B->mapped) return;

    int num_edges = B->num_edges;
    int *boundary_indices = B->boundary_indices;
    int *vol_ids = B->vol_ids;
    int *edge_ids = B->edge_ids;
    double stage_val = B->stage_value;

    // Edge values (read)
    double *xmom_ev = GD->D.xmom_edge_values;
    double *ymom_ev = GD->D.ymom_edge_values;
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;
    double *normals = GD->D.normals;

    // Boundary values (write)
    double *stage_bv = GD->D.stage_boundary_values;
    double *bed_bv = GD->D.bed_boundary_values;
    double *height_bv = GD->D.height_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < num_edges; k++) {
        int bid = boundary_indices[k];
        int vid = vol_ids[k];
        int eid = edge_ids[k];

        // Set stage from external value
        stage_bv[bid] = stage_val;

        // Copy bed/height from interior edge
        bed_bv[bid] = bed_ev[3 * vid + eid];
        height_bv[bid] = height_ev[3 * vid + eid];

        // Get normal vector for this edge
        double n1 = normals[vid * 6 + 2 * eid];
        double n2 = normals[vid * 6 + 2 * eid + 1];

        // Get interior momentum
        double q1 = xmom_ev[3 * vid + eid];
        double q2 = ymom_ev[3 * vid + eid];

        // Compute normal component of momentum (dot product with normal)
        double ndotq = n1 * q1 + n2 * q2;

        // Set boundary momentum to just the normal component (zero tangential)
        xmom_bv[bid] = ndotq * n1;
        ymom_bv[bid] = ndotq * n2;
    }
}

// ============================================================================
// Time_boundary - time-dependent Dirichlet values
// ============================================================================

int gpu_time_boundary_init(struct gpu_domain *GD, int num_edges,
                           int *boundary_indices, int *vol_ids, int *edge_ids) {
    struct time_boundary *B = &GD->time_bdry;

    B->num_edges = num_edges;
    B->mapped = 0;
    B->stage_value = 0.0;
    B->xmom_value = 0.0;
    B->ymom_value = 0.0;

    if (num_edges == 0) {
        B->boundary_indices = NULL;
        B->vol_ids = NULL;
        B->edge_ids = NULL;
        return 0;
    }

    B->boundary_indices = (int*)malloc(num_edges * sizeof(int));
    B->vol_ids = (int*)malloc(num_edges * sizeof(int));
    B->edge_ids = (int*)malloc(num_edges * sizeof(int));

    if (!B->boundary_indices || !B->vol_ids || !B->edge_ids) {
        fprintf(stderr, "Failed to allocate time_boundary arrays\n");
        return -1;
    }

    memcpy(B->boundary_indices, boundary_indices, num_edges * sizeof(int));
    memcpy(B->vol_ids, vol_ids, num_edges * sizeof(int));
    memcpy(B->edge_ids, edge_ids, num_edges * sizeof(int));

    if (GD->rank == 0) {
        printf("Time_boundary initialized: %d edges (GPU mapping deferred)\n", num_edges);
    }

    return 0;
}

void gpu_time_boundary_finalize(struct gpu_domain *GD) {
    struct time_boundary *B = &GD->time_bdry;

    if (B->mapped && B->num_edges > 0) {
        int ne = B->num_edges;
        int *b_idx = B->boundary_indices;
        int *v_ids = B->vol_ids;
        int *e_ids = B->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
    }

    if (B->boundary_indices) free(B->boundary_indices);
    if (B->vol_ids) free(B->vol_ids);
    if (B->edge_ids) free(B->edge_ids);

    B->num_edges = 0;
    B->boundary_indices = NULL;
    B->vol_ids = NULL;
    B->edge_ids = NULL;
    B->mapped = 0;
}

void gpu_time_boundary_set_values(struct gpu_domain *GD, double stage, double xmom, double ymom) {
    // Update values - called from Python each timestep before evaluate
    GD->time_bdry.stage_value = stage;
    GD->time_bdry.xmom_value = xmom;
    GD->time_bdry.ymom_value = ymom;
}

void gpu_evaluate_time_boundary(struct gpu_domain *GD) {
    // Time-dependent Dirichlet boundary - sets constant values (that vary with time)
    struct time_boundary *B = &GD->time_bdry;
    if (B->num_edges == 0) return;
    if (!B->mapped) return;

    int num_edges = B->num_edges;
    int *boundary_indices = B->boundary_indices;
    int *vol_ids = B->vol_ids;
    int *edge_ids = B->edge_ids;
    double stage_val = B->stage_value;
    double xmom_val = B->xmom_value;
    double ymom_val = B->ymom_value;

    // Edge values (read for bed/height)
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;

    // Boundary values (write)
    double *stage_bv = GD->D.stage_boundary_values;
    double *bed_bv = GD->D.bed_boundary_values;
    double *height_bv = GD->D.height_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < num_edges; k++) {
        int bid = boundary_indices[k];
        int vid = vol_ids[k];
        int eid = edge_ids[k];

        // Set stage and momentum from time-dependent values
        stage_bv[bid] = stage_val;
        xmom_bv[bid] = xmom_val;
        ymom_bv[bid] = ymom_val;

        // Copy bed/height from interior edge
        bed_bv[bid] = bed_ev[3 * vid + eid];
        height_bv[bid] = height_ev[3 * vid + eid];
    }
}

// ============================================================================
// Rate Operators (rain, extraction, etc.)
// ============================================================================

int gpu_rate_operator_init(struct gpu_domain *GD, int num_indices, int *indices,
                           double *areas, int *full_indices, int num_full) {
    struct rate_operators *RO = &GD->rate_ops;

    // Find a free slot
    int op_id = -1;
    for (int i = 0; i < MAX_RATE_OPERATORS; i++) {
        if (!RO->ops[i].active) {
            op_id = i;
            break;
        }
    }

    if (op_id < 0) {
        fprintf(stderr, "ERROR: No free rate operator slots (max %d)\n", MAX_RATE_OPERATORS);
        return -1;
    }

    struct rate_operator_info *op = &RO->ops[op_id];

    op->num_indices = num_indices;
    op->num_full = num_full;
    op->active = 1;
    op->mapped = 0;
    // Initialize rate array cache
    op->rate_array_cache = NULL;
    op->rate_array_size = 0;
    op->rate_array_mapped = 0;

    if (num_indices == 0) {
        op->indices = NULL;
        op->areas = NULL;
        op->full_indices = NULL;
        RO->num_operators++;
        return op_id;
    }

    // Allocate and copy arrays
    op->indices = (int*)malloc(num_indices * sizeof(int));
    op->areas = (double*)malloc(num_indices * sizeof(double));

    if (!op->indices || !op->areas) {
        fprintf(stderr, "Failed to allocate rate_operator arrays\n");
        if (op->indices) free(op->indices);
        if (op->areas) free(op->areas);
        op->active = 0;
        return -1;
    }

    memcpy(op->indices, indices, num_indices * sizeof(int));
    memcpy(op->areas, areas, num_indices * sizeof(double));

    if (num_full > 0 && full_indices != NULL) {
        op->full_indices = (int*)malloc(num_full * sizeof(int));
        if (op->full_indices) {
            memcpy(op->full_indices, full_indices, num_full * sizeof(int));
        }
    } else {
        op->full_indices = NULL;
    }

    // Map to GPU immediately if GPU is already initialized
    if (GD->gpu_initialized) {
        int ni = op->num_indices;
        int *idx = op->indices;
        double *ar = op->areas;
        #pragma omp target enter data map(to: idx[0:ni], ar[0:ni])
        op->mapped = 1;
    }

    RO->num_operators++;

    if (GD->rank == 0) {
        printf("Rate_operator %d initialized: %d indices, %d full (GPU mapped: %d)\n",
               op_id, num_indices, num_full, op->mapped);
    }

    return op_id;
}

void gpu_rate_operator_finalize(struct gpu_domain *GD, int op_id) {
    if (op_id < 0 || op_id >= MAX_RATE_OPERATORS) return;

    struct rate_operator_info *op = &GD->rate_ops.ops[op_id];
    if (!op->active) return;

    if (op->mapped && op->num_indices > 0) {
        int ni = op->num_indices;
        int *idx = op->indices;
        double *ar = op->areas;
        #pragma omp target exit data map(delete: idx[0:ni], ar[0:ni])
    }

    // Clean up rate array cache
    if (op->rate_array_mapped && op->rate_array_cache != NULL) {
        double *rac = op->rate_array_cache;
        int ras = op->rate_array_size;
        #pragma omp target exit data map(delete: rac[0:ras])
    }
    if (op->rate_array_cache) free(op->rate_array_cache);

    if (op->indices) free(op->indices);
    if (op->areas) free(op->areas);
    if (op->full_indices) free(op->full_indices);

    op->indices = NULL;
    op->areas = NULL;
    op->full_indices = NULL;
    op->rate_array_cache = NULL;
    op->num_indices = 0;
    op->num_full = 0;
    op->rate_array_size = 0;
    op->active = 0;
    op->mapped = 0;
    op->rate_array_mapped = 0;

    GD->rate_ops.num_operators--;
}

void gpu_rate_operators_finalize_all(struct gpu_domain *GD) {
    for (int i = 0; i < MAX_RATE_OPERATORS; i++) {
        if (GD->rate_ops.ops[i].active) {
            gpu_rate_operator_finalize(GD, i);
        }
    }
    GD->rate_ops.initialized = 0;
}

double gpu_rate_operator_apply(struct gpu_domain *GD, int op_id,
                               double rate, double factor, double timestep) {
    if (op_id < 0 || op_id >= MAX_RATE_OPERATORS) return 0.0;

    struct rate_operator_info *op = &GD->rate_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return 0.0;

    // Ensure mapped
    if (!op->mapped) {
        int ni = op->num_indices;
        int *idx = op->indices;
        double *ar = op->areas;
        #pragma omp target enter data map(to: idx[0:ni], ar[0:ni])
        op->mapped = 1;
    }

    int num_indices = op->num_indices;
    int *indices = op->indices;
    double *areas = op->areas;

    // Domain arrays
    double *stage_c = GD->D.stage_centroid_values;
    double *xmom_c = GD->D.xmom_centroid_values;
    double *ymom_c = GD->D.ymom_centroid_values;
    double *bed_c = GD->D.bed_centroid_values;

    double local_rate = factor * timestep * rate;
    double local_influx = 0.0;

    if (rate >= 0.0) {
        // Simple positive rate - just add to stage
        // Reduction for mass tracking
        #pragma omp target teams distribute parallel for reduction(+:local_influx)
        for (int k = 0; k < num_indices; k++) {
            int i = indices[k];
            stage_c[i] += local_rate;
            local_influx += local_rate * areas[k];
        }
    } else {
        // Negative rate (extraction) - need to limit and scale momentum
        #pragma omp target teams distribute parallel for reduction(+:local_influx)
        for (int k = 0; k < num_indices; k++) {
            int i = indices[k];

            // Current height
            double height = stage_c[i] - bed_c[i];

            // Can't remove more water than exists
            double actual_rate = (local_rate > -height) ? local_rate : -height;

            // Scaling factor for momentum (when extracting water)
            double scale_factor;
            if (actual_rate < 0.0) {
                scale_factor = (actual_rate + height) / (height + 1.0e-10);
            } else {
                scale_factor = 1.0;
            }

            // Apply updates
            stage_c[i] += actual_rate;
            xmom_c[i] *= scale_factor;
            ymom_c[i] *= scale_factor;

            local_influx += actual_rate * areas[k];
        }
    }

    // Count FLOPs: 8 FLOPs per affected cell
    if (GD->flops.enabled) {
        GD->flops.rate_operator_flops += (uint64_t)op->num_indices * FLOPS_RATE_OPERATOR;
        GD->flops.rate_operator_calls++;
    }

    return local_influx;
}

double gpu_rate_operator_apply_array(struct gpu_domain *GD, int op_id,
                                     double *rate_array, int rate_array_size,
                                     int use_indices_into_rate,
                                     int rate_changed,
                                     double factor, double timestep) {
    if (op_id < 0 || op_id >= MAX_RATE_OPERATORS) return 0.0;

    struct rate_operator_info *op = &GD->rate_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return 0.0;

    // Ensure operator arrays are mapped
    if (!op->mapped) {
        int ni = op->num_indices;
        int *idx = op->indices;
        double *ar = op->areas;
        #pragma omp target enter data map(to: idx[0:ni], ar[0:ni])
        op->mapped = 1;
    }

    int num_indices = op->num_indices;
    int *indices = op->indices;
    double *areas = op->areas;

    // Domain arrays
    double *stage_c = GD->D.stage_centroid_values;
    double *xmom_c = GD->D.xmom_centroid_values;
    double *ymom_c = GD->D.ymom_centroid_values;
    double *bed_c = GD->D.bed_centroid_values;

    double local_influx = 0.0;
    double ft = factor * timestep;

    // Use cached rate array on GPU (avoids H2D transfer every call)
    // Only reallocate if size changed
    if (op->rate_array_size != rate_array_size) {
        // Size changed - need to reallocate
        if (op->rate_array_mapped && op->rate_array_cache != NULL) {
            double *old_rac = op->rate_array_cache;
            int old_size = op->rate_array_size;
            #pragma omp target exit data map(delete: old_rac[0:old_size])
        }
        if (op->rate_array_cache) free(op->rate_array_cache);

        op->rate_array_cache = (double*)malloc(rate_array_size * sizeof(double));
        op->rate_array_size = rate_array_size;
        op->rate_array_mapped = 0;
        rate_changed = 1;  // Force update since we reallocated
    }

    // Only transfer data to GPU if rate has changed
    if (rate_changed || !op->rate_array_mapped) {
        // Copy data to cache
        memcpy(op->rate_array_cache, rate_array, rate_array_size * sizeof(double));

        // Map or update cache on GPU
        double *rac = op->rate_array_cache;
        int ras = rate_array_size;
        if (!op->rate_array_mapped) {
            #pragma omp target enter data map(to: rac[0:ras])
            op->rate_array_mapped = 1;
        } else {
            #pragma omp target update to(rac[0:ras])
        }
    }

    // Use the GPU-resident cache
    double *gpu_rate_array = op->rate_array_cache;

    if (use_indices_into_rate) {
        // gpu_rate_array is full domain size, index with indices[k]
        #pragma omp target teams distribute parallel for reduction(+:local_influx)
        for (int k = 0; k < num_indices; k++) {
            int i = indices[k];
            double rate = gpu_rate_array[i];
            double local_rate = ft * rate;

            if (rate >= 0.0) {
                stage_c[i] += local_rate;
                local_influx += local_rate * areas[k];
            } else {
                // Negative rate - limit and scale momentum
                double height = stage_c[i] - bed_c[i];
                double actual_rate = (local_rate > -height) ? local_rate : -height;
                double scale_factor = (actual_rate < 0.0) ?
                    (actual_rate + height) / (height + 1.0e-10) : 1.0;

                stage_c[i] += actual_rate;
                xmom_c[i] *= scale_factor;
                ymom_c[i] *= scale_factor;
                local_influx += actual_rate * areas[k];
            }
        }
    } else {
        // gpu_rate_array matches indices size, index with k
        #pragma omp target teams distribute parallel for reduction(+:local_influx)
        for (int k = 0; k < num_indices; k++) {
            int i = indices[k];
            double rate = gpu_rate_array[k];
            double local_rate = ft * rate;

            if (rate >= 0.0) {
                stage_c[i] += local_rate;
                local_influx += local_rate * areas[k];
            } else {
                // Negative rate - limit and scale momentum
                double height = stage_c[i] - bed_c[i];
                double actual_rate = (local_rate > -height) ? local_rate : -height;
                double scale_factor = (actual_rate < 0.0) ?
                    (actual_rate + height) / (height + 1.0e-10) : 1.0;

                stage_c[i] += actual_rate;
                xmom_c[i] *= scale_factor;
                ymom_c[i] *= scale_factor;
                local_influx += actual_rate * areas[k];
            }
        }
    }

    // Rate array cache stays mapped on GPU for next call

    // Count FLOPs: 8 FLOPs per affected cell
    if (GD->flops.enabled) {
        GD->flops.rate_operator_flops += (uint64_t)op->num_indices * FLOPS_RATE_OPERATOR;
        GD->flops.rate_operator_calls++;
    }

    return local_influx;
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

        if (GD->rank == 0) {
            printf("Reflective boundary arrays mapped to GPU: %d edges\n", ne);
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

        if (GD->rank == 0) {
            printf("Dirichlet boundary arrays mapped to GPU: %d edges\n", ne);
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

        if (GD->rank == 0) {
            printf("Transmissive boundary arrays mapped to GPU: %d edges\n", ne);
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

        if (GD->rank == 0) {
            printf("Transmissive_n_zero_t boundary arrays mapped to GPU: %d edges\n", ne);
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

        if (GD->rank == 0) {
            printf("Time_boundary arrays mapped to GPU: %d edges\n", ne);
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

    if (GD->rank == 0) {
        printf("GPU arrays mapped to device %d\n", GD->device_id);
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

        if (GD->rank == 0) {
            printf("Reflective boundary arrays mapped to GPU (late): %d edges\n", ne);
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

        if (GD->rank == 0) {
            printf("Dirichlet boundary arrays mapped to GPU (late): %d edges\n", ne);
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

        if (GD->rank == 0) {
            printf("Transmissive boundary arrays mapped to GPU (late): %d edges\n", ne);
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

        if (GD->rank == 0) {
            printf("Transmissive_n_zero_t boundary arrays mapped to GPU (late): %d edges\n", ne);
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

        if (GD->rank == 0) {
            printf("Time_boundary arrays mapped to GPU (late): %d edges\n", ne);
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
    printf("Rank %d: Boundary edge sync initialized for %d cells (%d edge values)\n",
           GD->rank, num_boundary_cells, S->buf_size);

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
    #pragma omp target teams distribute parallel for
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

// ============================================================================
// Ghost Exchange - Key MPI Function
// ============================================================================

// Exchange ghost cell data between MPI ranks
// Adapted from miniapp_mpi.c exchange_halo()
void gpu_exchange_ghosts(struct gpu_domain *GD) {
    struct halo_exchange *H = &GD->halo;

    if (H->num_neighbors == 0) return;

    int send_size = H->total_send_size;
    int recv_size = H->total_recv_size;

    double *stage = GD->D.stage_centroid_values;
    double *xmom = GD->D.xmom_centroid_values;
    double *ymom = GD->D.ymom_centroid_values;
    double *send_buf = H->send_buffer;
    double *recv_buf = H->recv_buffer;
    int *flat_send = H->flat_send_indices;
    int *flat_recv = H->flat_recv_indices;

    // Pack send buffer on GPU
    #pragma omp target teams distribute parallel for
    for (int idx = 0; idx < send_size; idx++) {
        int k = flat_send[idx];  // Local element index
        send_buf[3*idx + 0] = stage[k];
        send_buf[3*idx + 1] = xmom[k];
        send_buf[3*idx + 2] = ymom[k];
    }

#ifdef GPU_AWARE_MPI
    // GPU-aware MPI path: pass device pointers directly to MPI
    #pragma omp target data use_device_addr(send_buf, recv_buf)
    {
        int req_count = 0;
        int send_offset = 0, recv_offset = 0;

        // Post all receives first
        for (int ni = 0; ni < H->num_neighbors; ni++) {
            int partner = H->neighbor_ranks[ni];
            int count = H->recv_counts[ni];
            MPI_Irecv(&recv_buf[3*recv_offset], 3*count, MPI_DOUBLE,
                      partner, 0, GD->comm, &H->requests[req_count++]);
            recv_offset += count;
        }

        // Post all sends
        for (int ni = 0; ni < H->num_neighbors; ni++) {
            int partner = H->neighbor_ranks[ni];
            int count = H->send_counts[ni];
            MPI_Isend(&send_buf[3*send_offset], 3*count, MPI_DOUBLE,
                      partner, 0, GD->comm, &H->requests[req_count++]);
            send_offset += count;
        }

        // Wait for all communication to complete
        MPI_Waitall(req_count, H->requests, MPI_STATUSES_IGNORE);
    }

#else
    // Non-GPU-aware MPI path: transfer halo buffers through host
    // This is still efficient because halo is much smaller than full domain

    // Copy packed send buffer from GPU to host
    #pragma omp target update from(send_buf[0:3*send_size])

    // MPI communication on host
    int req_count = 0;
    int send_offset = 0, recv_offset = 0;

    // Post all receives first
    for (int ni = 0; ni < H->num_neighbors; ni++) {
        int partner = H->neighbor_ranks[ni];
        int count = H->recv_counts[ni];
        MPI_Irecv(&recv_buf[3*recv_offset], 3*count, MPI_DOUBLE,
                  partner, 0, GD->comm, &H->requests[req_count++]);
        recv_offset += count;
    }

    // Post all sends
    for (int ni = 0; ni < H->num_neighbors; ni++) {
        int partner = H->neighbor_ranks[ni];
        int count = H->send_counts[ni];
        MPI_Isend(&send_buf[3*send_offset], 3*count, MPI_DOUBLE,
                  partner, 0, GD->comm, &H->requests[req_count++]);
        send_offset += count;
    }

    // Wait for all communication to complete
    MPI_Waitall(req_count, H->requests, MPI_STATUSES_IGNORE);

    // Copy received halo data from host to GPU
    #pragma omp target update to(recv_buf[0:3*recv_size])
#endif

    // Unpack receive buffer on GPU
    #pragma omp target teams distribute parallel for
    for (int idx = 0; idx < recv_size; idx++) {
        int k = flat_recv[idx];  // Local ghost element index
        stage[k] = recv_buf[3*idx + 0];
        xmom[k] = recv_buf[3*idx + 1];
        ymom[k] = recv_buf[3*idx + 2];
    }
}

// ============================================================================
// GPU Kernel Stubs - To be implemented with ANUGA's numerical methods
// ============================================================================

void gpu_extrapolate_second_order(struct gpu_domain *GD) {
    // GPU implementation of second-order edge extrapolation
    // Based on _openmp_extrapolate_second_order_edge_sw in sw_domain_openmp.c

    anuga_int n = GD->D.number_of_elements;
    double minimum_allowed_height = GD->D.minimum_allowed_height;
    anuga_int extrapolate_velocity_second_order = GD->D.extrapolate_velocity_second_order;

    // Parameters for hfactor computation (wet-dry limiting)
    double a_tmp = 0.3;  // Highest depth ratio with hfactor=1
    double b_tmp = 0.1;  // Highest depth ratio with hfactor=0
    double c_tmp = 1.0 / (a_tmp - b_tmp);
    double d_tmp = 1.0 - (c_tmp * a_tmp);

    // Beta values for gradient limiting
    double beta_w = GD->D.beta_w;
    double beta_w_dry = GD->D.beta_w_dry;
    double beta_uh = GD->D.beta_uh;
    double beta_uh_dry = GD->D.beta_uh_dry;
    double beta_vh = GD->D.beta_vh;
    double beta_vh_dry = GD->D.beta_vh_dry;

    // Extract array pointers
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

    double *centroid_coords = GD->D.centroid_coordinates;
    double *edge_coords = GD->D.edge_coordinates;

    anuga_int *surrogate_neighbours = GD->D.surrogate_neighbours;
    anuga_int *number_of_boundaries = GD->D.number_of_boundaries;
    double *x_centroid_work = GD->D.x_centroid_work;
    double *y_centroid_work = GD->D.y_centroid_work;

    // Step 1: Update centroid values (compute height, optionally convert momentum to velocity)
    #pragma omp target teams distribute parallel for
    for (anuga_int k = 0; k < n; k++) {
        double stage = stage_cv[k];
        double bed = bed_cv[k];
        double xmom = xmom_cv[k];
        double ymom = ymom_cv[k];

        double dk = fmax(stage - bed, 0.0);
        height_cv[k] = dk;

        int is_dry = (dk <= minimum_allowed_height);
        int extrapolate = (extrapolate_velocity_second_order == 1) && (dk > minimum_allowed_height);

        // Zero momentum in dry cells
        double xmom_out = is_dry ? 0.0 : xmom;
        double ymom_out = is_dry ? 0.0 : ymom;

        double inv_dk = extrapolate ? (1.0 / dk) : 1.0;

        // Store original momentum in work arrays before converting to velocity
        x_centroid_work[k] = extrapolate ? xmom_out : 0.0;
        y_centroid_work[k] = extrapolate ? ymom_out : 0.0;

        // Convert to velocity for extrapolation (or zero if dry)
        xmom_cv[k] = xmom_out * inv_dk;
        ymom_cv[k] = ymom_out * inv_dk;
    }

    // Step 2: Main extrapolation loop
    #pragma omp target teams distribute parallel for
    for (anuga_int k = 0; k < n; k++) {
        anuga_int k2 = k * 2;
        anuga_int k3 = k * 3;
        anuga_int k6 = k * 6;

        // Get edge coordinates
        double xv0 = edge_coords[k6 + 0];
        double yv0 = edge_coords[k6 + 1];
        double xv1 = edge_coords[k6 + 2];
        double yv1 = edge_coords[k6 + 3];
        double xv2 = edge_coords[k6 + 4];
        double yv2 = edge_coords[k6 + 5];

        // Get centroid coordinates
        double x = centroid_coords[k2 + 0];
        double y = centroid_coords[k2 + 1];

        // Differences from centroid to edge midpoints
        double dxv0 = xv0 - x;
        double dxv1 = xv1 - x;
        double dxv2 = xv2 - x;
        double dyv0 = yv0 - y;
        double dyv1 = yv1 - y;
        double dyv2 = yv2 - y;

        // Get surrogate neighbour indices
        anuga_int k0 = surrogate_neighbours[k3 + 0];
        anuga_int k1 = surrogate_neighbours[k3 + 1];
        anuga_int sn2 = surrogate_neighbours[k3 + 2];

        // Get neighbour centroids
        double x0 = centroid_coords[2 * k0 + 0];
        double y0 = centroid_coords[2 * k0 + 1];
        double x1 = centroid_coords[2 * k1 + 0];
        double y1 = centroid_coords[2 * k1 + 1];
        double x2 = centroid_coords[2 * sn2 + 0];
        double y2 = centroid_coords[2 * sn2 + 1];

        // Differences between neighbour centroids
        double dx1 = x1 - x0;
        double dx2 = x2 - x0;
        double dy1 = y1 - y0;
        double dy2 = y2 - y0;

        double area2 = dy2 * dx1 - dy1 * dx2;

        // Check if all neighbours are dry
        int dry = ((height_cv[k0] < minimum_allowed_height) || (k0 == k)) &&
                  ((height_cv[k1] < minimum_allowed_height) || (k1 == k)) &&
                  ((height_cv[sn2] < minimum_allowed_height) || (sn2 == k));

        if (dry) {
            x_centroid_work[k] = 0.0;
            xmom_cv[k] = 0.0;
            y_centroid_work[k] = 0.0;
            ymom_cv[k] = 0.0;
        }

        int num_boundaries = number_of_boundaries[k];

        if (num_boundaries == 3) {
            // No neighbours - set edge values to centroid values
            double stage_c = stage_cv[k];
            double xmom_c = xmom_cv[k];
            double ymom_c = ymom_cv[k];
            double height_c = height_cv[k];
            double bed_c = bed_cv[k];

            for (int i = 0; i < 3; i++) {
                stage_ev[k3 + i] = stage_c;
                xmom_ev[k3 + i] = xmom_c;
                ymom_ev[k3 + i] = ymom_c;
                height_ev[k3 + i] = height_c;
                bed_ev[k3 + i] = bed_c;
            }

        } else if (num_boundaries <= 1) {
            // Typical case - full gradient reconstruction
            // Compute hfactor for wet-dry limiting
            double hc = height_cv[k];
            double h0 = height_cv[k0];
            double h1 = height_cv[k1];
            double h2 = height_cv[sn2];

            double hmin = fmin(fmin(h0, fmin(h1, h2)), hc);
            double hmax = fmax(fmax(h0, fmax(h1, h2)), hc);

            double tmp1 = c_tmp * fmax(hmin, 0.0) / fmax(hc, 1.0e-06) + d_tmp;
            double tmp2 = c_tmp * fmax(hc, 0.0) / fmax(hmax, 1.0e-06) + d_tmp;
            double hfactor = fmax(0.0, fmin(tmp1, fmin(tmp2, 1.0)));

            // Smooth shutoff near dry areas
            hfactor = fmin(1.2 * fmax(hmin - minimum_allowed_height, 0.0) /
                           (fmax(hmin, 0.0) + minimum_allowed_height), hfactor);

            double inv_area2 = 1.0 / area2;
            double edge_vals[3];

            // Stage
            double beta_stage = beta_w_dry + (beta_w - beta_w_dry) * hfactor;
            if (beta_stage > 0.0) {
                gpu_calc_edge_values_with_gradient(
                    stage_cv[k], stage_cv[k0], stage_cv[k1], stage_cv[sn2],
                    dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                    dx1, dx2, dy1, dy2, inv_area2, beta_stage, edge_vals);
            } else {
                gpu_set_constant_edge_values(stage_cv[k], edge_vals);
            }
            stage_ev[k3 + 0] = edge_vals[0];
            stage_ev[k3 + 1] = edge_vals[1];
            stage_ev[k3 + 2] = edge_vals[2];

            // Height (same beta as stage)
            if (beta_stage > 0.0) {
                gpu_calc_edge_values_with_gradient(
                    height_cv[k], height_cv[k0], height_cv[k1], height_cv[sn2],
                    dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                    dx1, dx2, dy1, dy2, inv_area2, beta_stage, edge_vals);
            } else {
                gpu_set_constant_edge_values(height_cv[k], edge_vals);
            }
            height_ev[k3 + 0] = edge_vals[0];
            height_ev[k3 + 1] = edge_vals[1];
            height_ev[k3 + 2] = edge_vals[2];

            // X-momentum
            double beta_xmom = beta_uh_dry + (beta_uh - beta_uh_dry) * hfactor;
            if (beta_xmom > 0.0) {
                gpu_calc_edge_values_with_gradient(
                    xmom_cv[k], xmom_cv[k0], xmom_cv[k1], xmom_cv[sn2],
                    dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                    dx1, dx2, dy1, dy2, inv_area2, beta_xmom, edge_vals);
            } else {
                gpu_set_constant_edge_values(xmom_cv[k], edge_vals);
            }
            xmom_ev[k3 + 0] = edge_vals[0];
            xmom_ev[k3 + 1] = edge_vals[1];
            xmom_ev[k3 + 2] = edge_vals[2];

            // Y-momentum
            double beta_ymom = beta_vh_dry + (beta_vh - beta_vh_dry) * hfactor;
            if (beta_ymom > 0.0) {
                gpu_calc_edge_values_with_gradient(
                    ymom_cv[k], ymom_cv[k0], ymom_cv[k1], ymom_cv[sn2],
                    dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                    dx1, dx2, dy1, dy2, inv_area2, beta_ymom, edge_vals);
            } else {
                gpu_set_constant_edge_values(ymom_cv[k], edge_vals);
            }
            ymom_ev[k3 + 0] = edge_vals[0];
            ymom_ev[k3 + 1] = edge_vals[1];
            ymom_ev[k3 + 2] = edge_vals[2];

        } else {
            // Number of boundaries == 2
            // One internal neighbour, gradient is in direction of neighbour's centroid
            // Find the only internal neighbour
            anuga_int kn = k;  // Will be set to internal neighbour
            for (int i = 0; i < 3; i++) {
                anuga_int sn = surrogate_neighbours[k3 + i];
                if (sn != k) {
                    kn = sn;
                    break;
                }
            }

            // Compute gradient projection between centroids
            double xn = centroid_coords[2 * kn + 0];
            double yn = centroid_coords[2 * kn + 1];
            double dx = xn - x;
            double dy = yn - y;
            double dist2 = dx * dx + dy * dy;

            double grad_dx2 = (dist2 > 0.0) ? dx / dist2 : 0.0;
            double grad_dy2 = (dist2 > 0.0) ? dy / dist2 : 0.0;

            double dqv[3], qmin, qmax, dq1;

            // Stage
            dq1 = stage_cv[kn] - stage_cv[k];
            gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                          dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
            gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
            gpu_limit_gradient(dqv, qmin, qmax, beta_w);
            stage_ev[k3 + 0] = stage_cv[k] + dqv[0];
            stage_ev[k3 + 1] = stage_cv[k] + dqv[1];
            stage_ev[k3 + 2] = stage_cv[k] + dqv[2];

            // Height
            dq1 = height_cv[kn] - height_cv[k];
            gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                          dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
            gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
            gpu_limit_gradient(dqv, qmin, qmax, beta_w);
            height_ev[k3 + 0] = height_cv[k] + dqv[0];
            height_ev[k3 + 1] = height_cv[k] + dqv[1];
            height_ev[k3 + 2] = height_cv[k] + dqv[2];

            // X-momentum
            dq1 = xmom_cv[kn] - xmom_cv[k];
            gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                          dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
            gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
            gpu_limit_gradient(dqv, qmin, qmax, beta_w);
            xmom_ev[k3 + 0] = xmom_cv[k] + dqv[0];
            xmom_ev[k3 + 1] = xmom_cv[k] + dqv[1];
            xmom_ev[k3 + 2] = xmom_cv[k] + dqv[2];

            // Y-momentum
            dq1 = ymom_cv[kn] - ymom_cv[k];
            gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                          dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
            gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
            gpu_limit_gradient(dqv, qmin, qmax, beta_w);
            ymom_ev[k3 + 0] = ymom_cv[k] + dqv[0];
            ymom_ev[k3 + 1] = ymom_cv[k] + dqv[1];
            ymom_ev[k3 + 2] = ymom_cv[k] + dqv[2];
        }

        // Convert velocity edge values back to momentum if needed
        if (extrapolate_velocity_second_order == 1) {
            for (int i = 0; i < 3; i++) {
                double dk = height_ev[k3 + i];
                xmom_ev[k3 + i] *= dk;
                ymom_ev[k3 + i] *= dk;
            }
        }

        // Compute bed edge values from stage - height
        for (int i = 0; i < 3; i++) {
            bed_ev[k3 + i] = stage_ev[k3 + i] - height_ev[k3 + i];
        }
    }

    // Step 3: Restore centroid momentum values if we converted to velocity
    if (extrapolate_velocity_second_order == 1) {
        #pragma omp target teams distribute parallel for
        for (anuga_int k = 0; k < n; k++) {
            xmom_cv[k] = x_centroid_work[k];
            ymom_cv[k] = y_centroid_work[k];
        }
    }

    // Count FLOPs: 150 FLOPs per element (gradient limiting, 5 quantities)
    if (GD->flops.enabled) {
        GD->flops.extrapolate_flops += (uint64_t)n * FLOPS_EXTRAPOLATE;
        GD->flops.extrapolate_calls++;
    }
}

double gpu_compute_fluxes(struct gpu_domain *GD) {
    // Compute fluxes across all edges using central upwind scheme
    // Returns the local minimum timestep (caller should do MPI_Allreduce)
    //
    // This is the GPU version of _openmp_compute_fluxes_central
    // Arrays are already mapped to GPU via gpu_domain_map_arrays()

    anuga_int n = GD->D.number_of_elements;
    double g = GD->D.g;
    double epsilon = GD->D.epsilon;
    anuga_int low_froude = GD->D.low_froude;

    // Extract array pointers for GPU kernel
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

    double *stage_bv = GD->D.stage_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;

    double *stage_eu = GD->D.stage_explicit_update;
    double *xmom_eu = GD->D.xmom_explicit_update;
    double *ymom_eu = GD->D.ymom_explicit_update;

    anuga_int *neighbours = GD->D.neighbours;
    anuga_int *neighbour_edges = GD->D.neighbour_edges;
    double *normals = GD->D.normals;
    double *edgelengths = GD->D.edgelengths;
    double *radii = GD->D.radii;
    double *areas = GD->D.areas;
    double *max_speed_array = GD->D.max_speed;

    double local_timestep = 1.0e+100;

    // Main flux computation loop
    #pragma omp target teams distribute parallel for \
        reduction(min: local_timestep)
    for (anuga_int k = 0; k < n; k++) {
        double edgeflux[3];
        double ql[3], qr[3];
        double speed_max_last = 0.0;

        // Zero the explicit updates for this element
        stage_eu[k] = 0.0;
        xmom_eu[k] = 0.0;
        ymom_eu[k] = 0.0;

        // Get centroid values for this element
        double hc = height_cv[k];
        double zc = bed_cv[k];

        // Loop over the 3 edges
        for (int i = 0; i < 3; i++) {
            int ki = 3 * k + i;
            int ki2 = 2 * ki;

            // Left state (this element's edge values)
            ql[0] = stage_ev[ki];
            ql[1] = xmom_ev[ki];
            ql[2] = ymom_ev[ki];
            double zl = bed_ev[ki];
            double hle = height_ev[ki];

            // Edge geometry
            double length = edgelengths[ki];
            double n1 = normals[ki2];
            double n2 = normals[ki2 + 1];

            // Get neighbour info
            anuga_int neighbour = neighbours[ki];
            int is_boundary = (neighbour < 0);

            double zr, hre, hc_n, zc_n;

            if (is_boundary) {
                // Boundary edge - get values from boundary arrays
                int m = -neighbour - 1;
                qr[0] = stage_bv[m];
                qr[1] = xmom_bv[m];
                qr[2] = ymom_bv[m];
                zr = zl;
                hre = fmax(qr[0] - zr, 0.0);
                hc_n = hc;
                zc_n = zc;
            } else {
                // Internal edge - get values from neighbour element
                int m = neighbour_edges[ki];
                int nm = neighbour * 3 + m;
                qr[0] = stage_ev[nm];
                qr[1] = xmom_ev[nm];
                qr[2] = ymom_ev[nm];
                zr = bed_ev[nm];
                hre = height_ev[nm];
                hc_n = height_cv[neighbour];
                zc_n = bed_cv[neighbour];
            }

            // Compute z_half (max bed elevation at edge)
            double z_half = fmax(zl, zr);

            // Compute effective heights at the edge
            double h_left = fmax(hle + zl - z_half, 0.0);
            double h_right = fmax(hre + zr - z_half, 0.0);

            double max_speed_local = 0.0;
            double pressure_flux = 0.0;

            if (h_left == 0.0 && h_right == 0.0) {
                // Both heights zero - no flux
                edgeflux[0] = 0.0;
                edgeflux[1] = 0.0;
                edgeflux[2] = 0.0;
            } else {
                // Compute flux using central scheme
                gpu_flux_function_central(ql, qr,
                                          h_left, h_right,
                                          hle, hre,
                                          n1, n2,
                                          epsilon, z_half, g,
                                          edgeflux, &max_speed_local, &pressure_flux,
                                          low_froude);
            }

            // Multiply flux by edge length (and negate for conservation)
            edgeflux[0] *= -length;
            edgeflux[1] *= -length;
            edgeflux[2] *= -length;

            // Update timestep based on max wave speed
            if (max_speed_local > epsilon) {
                double edge_timestep = radii[k] / fmax(max_speed_local, epsilon);
                local_timestep = fmin(local_timestep, edge_timestep);
                speed_max_last = fmax(speed_max_last, max_speed_local);
            }

            // Accumulate flux contributions
            stage_eu[k] += edgeflux[0];
            xmom_eu[k] += edgeflux[1];
            ymom_eu[k] += edgeflux[2];

            // Pressure gradient (gravity) terms
            double pressuregrad_work = length * (-g * 0.5 * (h_left * h_left - hle * hle
                                       - (hle + hc) * (zl - zc)) + pressure_flux);
            xmom_eu[k] -= normals[ki2] * pressuregrad_work;
            ymom_eu[k] -= normals[ki2 + 1] * pressuregrad_work;

        } // End edge loop

        // Store max speed for this element
        max_speed_array[k] = speed_max_last;

        // Normalize by area
        double inv_area = 1.0 / areas[k];
        stage_eu[k] *= inv_area;
        xmom_eu[k] *= inv_area;
        ymom_eu[k] *= inv_area;

    } // End element loop

    // Count FLOPs: 380 FLOPs per element (3 edges × flux function)
    if (GD->flops.enabled) {
        GD->flops.compute_fluxes_flops += (uint64_t)n * FLOPS_COMPUTE_FLUXES;
        GD->flops.compute_fluxes_calls++;
    }

    return local_timestep;
}

void gpu_update_conserved_quantities(struct gpu_domain *GD, double timestep) {
    // Update centroid values using explicit and semi-implicit updates
    // Q_new = (Q_old + timestep * explicit_update) / (1 - timestep * semi_implicit_update)
    // Arrays are already mapped to GPU

    anuga_int n = GD->D.number_of_elements;

    double *stage_cv = GD->D.stage_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *stage_eu = GD->D.stage_explicit_update;
    double *xmom_eu = GD->D.xmom_explicit_update;
    double *ymom_eu = GD->D.ymom_explicit_update;
    double *stage_siu = GD->D.stage_semi_implicit_update;
    double *xmom_siu = GD->D.xmom_semi_implicit_update;
    double *ymom_siu = GD->D.ymom_semi_implicit_update;

    #pragma omp target teams distribute parallel for
    for (anuga_int k = 0; k < n; k++) {
        // Normalize semi-implicit update by current value
        double stage_c = stage_cv[k];
        double xmom_c = xmom_cv[k];
        double ymom_c = ymom_cv[k];

        double stage_si = (stage_c == 0.0) ? 0.0 : stage_siu[k] / stage_c;
        double xmom_si = (xmom_c == 0.0) ? 0.0 : xmom_siu[k] / xmom_c;
        double ymom_si = (ymom_c == 0.0) ? 0.0 : ymom_siu[k] / ymom_c;

        // Apply explicit update
        stage_cv[k] += timestep * stage_eu[k];
        xmom_cv[k] += timestep * xmom_eu[k];
        ymom_cv[k] += timestep * ymom_eu[k];

        // Apply semi-implicit update (from friction etc.)
        double denom;

        denom = 1.0 - timestep * stage_si;
        if (denom > 0.0) stage_cv[k] /= denom;

        denom = 1.0 - timestep * xmom_si;
        if (denom > 0.0) xmom_cv[k] /= denom;

        denom = 1.0 - timestep * ymom_si;
        if (denom > 0.0) ymom_cv[k] /= denom;

        // Reset semi-implicit update for next timestep
        stage_siu[k] = 0.0;
        xmom_siu[k] = 0.0;
        ymom_siu[k] = 0.0;
    }

    // Count FLOPs: 21 FLOPs per element (explicit + semi-implicit update)
    if (GD->flops.enabled) {
        GD->flops.update_flops += (uint64_t)n * FLOPS_UPDATE;
        GD->flops.update_calls++;
    }
}

void gpu_backup_conserved_quantities(struct gpu_domain *GD) {
    // Backup centroid values for RK2 - simple array copy
    // Arrays are already mapped to GPU via gpu_domain_map_arrays()

    anuga_int n = GD->D.number_of_elements;

    double *stage_cv = GD->D.stage_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *stage_backup = GD->D.stage_backup_values;
    double *xmom_backup = GD->D.xmom_backup_values;
    double *ymom_backup = GD->D.ymom_backup_values;

    #pragma omp target teams distribute parallel for
    for (anuga_int k = 0; k < n; k++) {
        stage_backup[k] = stage_cv[k];
        xmom_backup[k] = xmom_cv[k];
        ymom_backup[k] = ymom_cv[k];
    }

    // Count FLOPs: 0 FLOPs per element (memory copy only)
    if (GD->flops.enabled) {
        GD->flops.backup_flops += (uint64_t)n * FLOPS_BACKUP;
        GD->flops.backup_calls++;
    }
}

void gpu_saxpy_conserved_quantities(struct gpu_domain *GD, double a, double b) {
    // RK2 combination: Q = a*Q_current + b*Q_backup
    // Typically called with a=0.5, b=0.5 for standard RK2
    // Arrays are already mapped to GPU

    anuga_int n = GD->D.number_of_elements;

    double *stage_cv = GD->D.stage_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *stage_backup = GD->D.stage_backup_values;
    double *xmom_backup = GD->D.xmom_backup_values;
    double *ymom_backup = GD->D.ymom_backup_values;
    double *height_cv = GD->D.height_centroid_values;
    double *bed_cv = GD->D.bed_centroid_values;

    #pragma omp target teams distribute parallel for
    for (anuga_int k = 0; k < n; k++) {
        double stage = a * stage_cv[k] + b * stage_backup[k];
        stage_cv[k] = stage;
        xmom_cv[k] = a * xmom_cv[k] + b * xmom_backup[k];
        ymom_cv[k] = a * ymom_cv[k] + b * ymom_backup[k];
        // Update height to match the new stage (needed for volume calculation)
        height_cv[k] = fmax(stage - bed_cv[k], 0.0);
    }

    // Count FLOPs: 9 FLOPs per element (3 quantities × (2 mul + 1 add) + height calc)
    if (GD->flops.enabled) {
        GD->flops.saxpy_flops += (uint64_t)n * FLOPS_SAXPY;
        GD->flops.saxpy_calls++;
    }
}

double gpu_protect(struct gpu_domain *GD) {
    // Protect against negative water depths
    // Sets stage = bed where water depth would be negative
    // Also zeros momentum where depth is very small
    // Returns mass_error: total mass added to prevent negative depths

    anuga_int n = GD->D.number_of_elements;
    double min_height = GD->D.minimum_allowed_height;
    double mass_error = 0.0;

    double *stage_cv = GD->D.stage_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *bed_cv = GD->D.bed_centroid_values;
    double *height_cv = GD->D.height_centroid_values;
    double *areas = GD->D.areas;

    #pragma omp target teams distribute parallel for reduction(+:mass_error)
    for (anuga_int k = 0; k < n; k++) {
        double h = stage_cv[k] - bed_cv[k];

        if (h < min_height) {
            // Very shallow - zero momentum to prevent instability
            xmom_cv[k] = 0.0;
            ymom_cv[k] = 0.0;
        }

        if (h < 0.0) {
            // Negative depth - track mass error and set stage to bed
            mass_error += (-h) * areas[k];
            stage_cv[k] = bed_cv[k];
            h = 0.0;
        }

        // Update height quantity
        height_cv[k] = h;
    }

    // Count FLOPs: 5 FLOPs per element (depth check, mass error)
    if (GD->flops.enabled) {
        GD->flops.protect_flops += (uint64_t)n * FLOPS_PROTECT;
        GD->flops.protect_calls++;
    }

    return mass_error;
}

void gpu_manning_friction(struct gpu_domain *GD) {
    // GPU implementation of Manning friction (flat, semi-implicit)
    // Based on _openmp_manning_friction_flat_semi_implicit in sw_domain_openmp.c
    //
    // Adds friction contribution to semi_implicit_update arrays.
    // The semi-implicit formulation provides better stability for friction terms.

    anuga_int n = GD->D.number_of_elements;
    double g = GD->D.g;
    double eps = GD->D.minimum_allowed_height;
    double seven_thirds = 7.0 / 3.0;

    // Small threshold for friction coefficient
    double eta_small = 1.0e-12;

    // Extract array pointers
    double *stage_cv = GD->D.stage_centroid_values;
    double *bed_cv = GD->D.bed_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *friction_cv = GD->D.friction_centroid_values;
    double *xmom_siu = GD->D.xmom_semi_implicit_update;
    double *ymom_siu = GD->D.ymom_semi_implicit_update;

    #pragma omp target teams distribute parallel for
    for (anuga_int k = 0; k < n; k++) {
        double S = 0.0;
        double uh = xmom_cv[k];
        double vh = ymom_cv[k];
        double eta = friction_cv[k];

        // Compute absolute momentum
        double abs_mom = sqrt(uh * uh + vh * vh);

        if (eta > eta_small) {
            double h = stage_cv[k] - bed_cv[k];
            if (h >= eps) {
                // Manning friction: S = -g * n^2 * |u| / h^(7/3)
                // Applied semi-implicitly via: du/dt = S * u
                S = -g * eta * eta * abs_mom;
                S /= pow(h, seven_thirds);
            }
        }

        // Add to semi-implicit update (will be applied in update_conserved_quantities)
        xmom_siu[k] += S * uh;
        ymom_siu[k] += S * vh;
    }

    // Count FLOPs: 15 FLOPs per element (sqrt, pow, semi-implicit)
    if (GD->flops.enabled) {
        GD->flops.manning_flops += (uint64_t)n * FLOPS_MANNING;
        GD->flops.manning_calls++;
    }
}

// ============================================================================
// Full RK2 Step - Orchestrates all GPU operations
// ============================================================================

double gpu_evolve_one_rk2_step(struct gpu_domain *GD, double max_timestep, int apply_forcing) {
    // Full RK2 step orchestrated entirely in C - eliminates Python round-trip overhead
    //
    // This function performs:
    // 1. Backup conserved quantities
    // 2. First Euler step (protect, extrapolate, boundaries, fluxes, forcing, update, ghost exchange)
    // 3. Second Euler step (same pattern)
    // 4. RK2 averaging (saxpy)
    //
    // Parameters:
    // - max_timestep: Maximum allowed timestep (respecting yieldstep/finaltime constraints)
    // - apply_forcing: Whether to apply forcing terms (Manning friction)
    //
    // Time-dependent boundary values (Time_boundary, Transmissive_n_zero_t) must be set
    // by Python BEFORE calling this function via set_time_boundary_values() and
    // set_transmissive_n_zero_t_stage().

    double local_timestep, global_timestep, timestep;

    // ========================================
    // Backup conserved quantities for RK2
    // ========================================
    gpu_backup_conserved_quantities(GD);

    // ========================================
    // First Euler step
    // ========================================

    // Protect against negative depths
    gpu_protect(GD);

    // Extrapolate to vertices and edges
    gpu_extrapolate_second_order(GD);

    // Evaluate all GPU-supported boundary conditions
    // (Time-dependent values must be set by Python before this call)
    gpu_evaluate_reflective_boundary(GD);
    gpu_evaluate_dirichlet_boundary(GD);
    gpu_evaluate_transmissive_boundary(GD);
    gpu_evaluate_transmissive_n_zero_t_boundary(GD);
    gpu_evaluate_time_boundary(GD);

    // Compute fluxes - returns local minimum timestep
    local_timestep = gpu_compute_fluxes(GD);

    // MPI reduce to get global minimum timestep
    if (GD->nprocs > 1) {
        MPI_Allreduce(&local_timestep, &global_timestep, 1, MPI_DOUBLE, MPI_MIN, GD->comm);
    } else {
        global_timestep = local_timestep;
    }

    // Apply CFL condition and respect max_timestep from Python
    timestep = GD->CFL * global_timestep;
    if (timestep > max_timestep) {
        timestep = max_timestep;
    }

    // Apply forcing terms (Manning friction on GPU)
    if (apply_forcing) {
        gpu_manning_friction(GD);
    }

    // Update conserved quantities with computed timestep
    gpu_update_conserved_quantities(GD, timestep);

    // Ghost exchange (MPI) - sync ghost cells between processes
    if (GD->nprocs > 1) {
        gpu_exchange_ghosts(GD);
    }

    // ========================================
    // Second Euler step
    // ========================================

    // Protect against negative depths
    gpu_protect(GD);

    // Extrapolate to vertices and edges
    gpu_extrapolate_second_order(GD);

    // Evaluate boundary conditions (same as first step)
    gpu_evaluate_reflective_boundary(GD);
    gpu_evaluate_dirichlet_boundary(GD);
    gpu_evaluate_transmissive_boundary(GD);
    gpu_evaluate_transmissive_n_zero_t_boundary(GD);
    gpu_evaluate_time_boundary(GD);

    // Compute fluxes (ignore timestep from second step - use first step's timestep)
    gpu_compute_fluxes(GD);

    // Apply forcing terms (Manning friction on GPU)
    if (apply_forcing) {
        gpu_manning_friction(GD);
    }

    // Update conserved quantities (same timestep as first step)
    gpu_update_conserved_quantities(GD, timestep);

    // ========================================
    // RK2 averaging: Q_final = 0.5 * Q_backup + 0.5 * Q_current
    // ========================================
    gpu_saxpy_conserved_quantities(GD, 0.5, 0.5);

    return timestep;
}

// ============================================================================
// FLOP Counter Functions (Gordon Bell Performance Profiling)
// ============================================================================

void gpu_flop_counters_init(struct gpu_domain *GD) {
    // Initialize all FLOP counters to zero
    memset(&GD->flops, 0, sizeof(struct flop_counters));
    GD->flops.enabled = 1;  // Enable by default
}

void gpu_flop_counters_reset(struct gpu_domain *GD) {
    // Reset counters but keep enabled state
    int enabled = GD->flops.enabled;
    memset(&GD->flops, 0, sizeof(struct flop_counters));
    GD->flops.enabled = enabled;
}

void gpu_flop_counters_enable(struct gpu_domain *GD, int enable) {
    GD->flops.enabled = enable;
}

void gpu_flop_counters_start_timer(struct gpu_domain *GD) {
    GD->flops.start_time = MPI_Wtime();
}

void gpu_flop_counters_stop_timer(struct gpu_domain *GD) {
    GD->flops.elapsed_time = MPI_Wtime() - GD->flops.start_time;
}

uint64_t gpu_flop_counters_get_total(struct gpu_domain *GD) {
    // Recalculate total from individual counters
    GD->flops.total_flops =
        GD->flops.extrapolate_flops +
        GD->flops.compute_fluxes_flops +
        GD->flops.update_flops +
        GD->flops.protect_flops +
        GD->flops.manning_flops +
        GD->flops.backup_flops +
        GD->flops.saxpy_flops +
        GD->flops.rate_operator_flops +
        GD->flops.ghost_exchange_flops;
    return GD->flops.total_flops;
}

double gpu_flop_counters_get_flops(struct gpu_domain *GD) {
    // Return FLOP/s (floating point operations per second)
    if (GD->flops.elapsed_time <= 0.0) {
        return 0.0;
    }
    return (double)gpu_flop_counters_get_total(GD) / GD->flops.elapsed_time;
}

void gpu_flop_counters_print(struct gpu_domain *GD) {
    uint64_t total = gpu_flop_counters_get_total(GD);
    double gflops = (double)total / 1.0e9;
    double elapsed = GD->flops.elapsed_time;
    double gflops_per_sec = (elapsed > 0.0) ? gflops / elapsed : 0.0;

    printf("\n");
    printf("============================================================\n");
    printf("FLOP Counter Summary (Gordon Bell Profiling)\n");
    printf("============================================================\n");
    printf("Kernel                       |      FLOPs |     Calls |  FLOPs/call\n");
    printf("-----------------------------|------------|-----------|------------\n");

    if (GD->flops.extrapolate_calls > 0) {
        printf("extrapolate_second_order     | %10lu | %9lu | %10lu\n",
               (unsigned long)GD->flops.extrapolate_flops,
               (unsigned long)GD->flops.extrapolate_calls,
               (unsigned long)(GD->flops.extrapolate_flops / GD->flops.extrapolate_calls));
    }
    if (GD->flops.compute_fluxes_calls > 0) {
        printf("compute_fluxes               | %10lu | %9lu | %10lu\n",
               (unsigned long)GD->flops.compute_fluxes_flops,
               (unsigned long)GD->flops.compute_fluxes_calls,
               (unsigned long)(GD->flops.compute_fluxes_flops / GD->flops.compute_fluxes_calls));
    }
    if (GD->flops.update_calls > 0) {
        printf("update_conserved_quantities  | %10lu | %9lu | %10lu\n",
               (unsigned long)GD->flops.update_flops,
               (unsigned long)GD->flops.update_calls,
               (unsigned long)(GD->flops.update_flops / GD->flops.update_calls));
    }
    if (GD->flops.protect_calls > 0) {
        printf("protect                      | %10lu | %9lu | %10lu\n",
               (unsigned long)GD->flops.protect_flops,
               (unsigned long)GD->flops.protect_calls,
               (unsigned long)(GD->flops.protect_flops / GD->flops.protect_calls));
    }
    if (GD->flops.manning_calls > 0) {
        printf("manning_friction             | %10lu | %9lu | %10lu\n",
               (unsigned long)GD->flops.manning_flops,
               (unsigned long)GD->flops.manning_calls,
               (unsigned long)(GD->flops.manning_flops / GD->flops.manning_calls));
    }
    if (GD->flops.saxpy_calls > 0) {
        printf("saxpy_conserved_quantities   | %10lu | %9lu | %10lu\n",
               (unsigned long)GD->flops.saxpy_flops,
               (unsigned long)GD->flops.saxpy_calls,
               (unsigned long)(GD->flops.saxpy_flops / GD->flops.saxpy_calls));
    }
    if (GD->flops.rate_operator_calls > 0) {
        printf("rate_operator_apply          | %10lu | %9lu | %10lu\n",
               (unsigned long)GD->flops.rate_operator_flops,
               (unsigned long)GD->flops.rate_operator_calls,
               (unsigned long)(GD->flops.rate_operator_flops / GD->flops.rate_operator_calls));
    }

    printf("-----------------------------|------------|-----------|------------\n");
    printf("TOTAL                        | %10lu |\n", (unsigned long)total);
    printf("============================================================\n");
    printf("Total GFLOPs:     %.3f\n", gflops);
    printf("Elapsed time:     %.3f s\n", elapsed);
    printf("Performance:      %.3f GFLOP/s\n", gflops_per_sec);
    printf("============================================================\n\n");
}

// Per-kernel FLOP getters
uint64_t gpu_flop_counters_get_extrapolate(struct gpu_domain *GD) {
    return GD->flops.extrapolate_flops;
}

uint64_t gpu_flop_counters_get_compute_fluxes(struct gpu_domain *GD) {
    return GD->flops.compute_fluxes_flops;
}

uint64_t gpu_flop_counters_get_update(struct gpu_domain *GD) {
    return GD->flops.update_flops;
}

uint64_t gpu_flop_counters_get_protect(struct gpu_domain *GD) {
    return GD->flops.protect_flops;
}

uint64_t gpu_flop_counters_get_manning(struct gpu_domain *GD) {
    return GD->flops.manning_flops;
}

uint64_t gpu_flop_counters_get_backup(struct gpu_domain *GD) {
    return GD->flops.backup_flops;
}

uint64_t gpu_flop_counters_get_saxpy(struct gpu_domain *GD) {
    return GD->flops.saxpy_flops;
}

uint64_t gpu_flop_counters_get_rate_operator(struct gpu_domain *GD) {
    return GD->flops.rate_operator_flops;
}

uint64_t gpu_flop_counters_get_ghost_exchange(struct gpu_domain *GD) {
    return GD->flops.ghost_exchange_flops;
}

// ============================================================================
// MPI Reduction for Multi-GPU FLOP Counters (Gordon Bell)
// ============================================================================

uint64_t gpu_flop_counters_get_global_total(struct gpu_domain *GD) {
    // Get local total first
    uint64_t local_total = gpu_flop_counters_get_total(GD);

    // MPI_Allreduce to sum across all ranks
    // MPI_UNSIGNED_LONG_LONG maps to uint64_t
    uint64_t global_total = 0;
    MPI_Allreduce(&local_total, &global_total, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, GD->comm);

    return global_total;
}

double gpu_flop_counters_get_global_flops(struct gpu_domain *GD) {
    // Get global total FLOPs
    uint64_t global_total = gpu_flop_counters_get_global_total(GD);

    // Use local elapsed time (should be same across ranks if synchronized)
    if (GD->flops.elapsed_time <= 0.0) {
        return 0.0;
    }
    return (double)global_total / GD->flops.elapsed_time;
}

void gpu_flop_counters_print_global(struct gpu_domain *GD) {
    // Gather per-kernel FLOPs from all ranks
    uint64_t local_flops[9];
    local_flops[0] = GD->flops.extrapolate_flops;
    local_flops[1] = GD->flops.compute_fluxes_flops;
    local_flops[2] = GD->flops.update_flops;
    local_flops[3] = GD->flops.protect_flops;
    local_flops[4] = GD->flops.manning_flops;
    local_flops[5] = GD->flops.backup_flops;
    local_flops[6] = GD->flops.saxpy_flops;
    local_flops[7] = GD->flops.rate_operator_flops;
    local_flops[8] = GD->flops.ghost_exchange_flops;

    uint64_t global_flops[9];
    MPI_Reduce(local_flops, global_flops, 9, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, GD->comm);

    // Get global total and max elapsed time
    uint64_t local_total = gpu_flop_counters_get_total(GD);
    uint64_t global_total = 0;
    MPI_Reduce(&local_total, &global_total, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, GD->comm);

    double local_elapsed = GD->flops.elapsed_time;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE,
               MPI_MAX, 0, GD->comm);

    // Only rank 0 prints
    if (GD->rank != 0) return;

    double gflops = (double)global_total / 1.0e9;
    double gflops_per_sec = (max_elapsed > 0.0) ? gflops / max_elapsed : 0.0;

    printf("\n");
    printf("============================================================\n");
    printf("GLOBAL FLOP Counter Summary (Gordon Bell - %d GPUs)\n", GD->nprocs);
    printf("============================================================\n");
    printf("Kernel                       |      GFLOPs | %% of total\n");
    printf("-----------------------------|-------------|------------\n");

    const char* names[] = {
        "extrapolate_second_order",
        "compute_fluxes",
        "update_conserved_quantities",
        "protect",
        "manning_friction",
        "backup_conserved_quantities",
        "saxpy_conserved_quantities",
        "rate_operator_apply",
        "ghost_exchange"
    };

    for (int i = 0; i < 9; i++) {
        if (global_flops[i] > 0) {
            double kernel_gflops = (double)global_flops[i] / 1.0e9;
            double pct = 100.0 * (double)global_flops[i] / (double)global_total;
            printf("%-28s | %11.3f | %9.1f%%\n", names[i], kernel_gflops, pct);
        }
    }

    printf("-----------------------------|-------------|------------\n");
    printf("TOTAL                        | %11.3f | 100.0%%\n", gflops);
    printf("============================================================\n");
    printf("Number of GPUs:   %d\n", GD->nprocs);
    printf("Total GFLOPs:     %.3f\n", gflops);
    printf("Elapsed time:     %.3f s\n", max_elapsed);
    printf("Performance:      %.3f GFLOP/s\n", gflops_per_sec);
    printf("Per-GPU average:  %.3f GFLOP/s\n", gflops_per_sec / GD->nprocs);
    printf("============================================================\n\n");
}
