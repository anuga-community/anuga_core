// GPU-accelerated domain struct with MPI halo exchange support
//
// This extends the base ANUGA domain struct with:
// - MPI communicator and process info
// - Flattened halo exchange structures for GPU efficiency
// - GPU state tracking
//
// Based on miniapp_mpi.c patterns

#ifndef SW_DOMAIN_GPU_H
#define SW_DOMAIN_GPU_H

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>
#include "anuga_typedefs.h"
#include "sw_domain.h"

// Halo exchange structure - flattened for GPU efficiency
// Mirrors the pattern from miniapp_mpi.c halo_info struct
struct halo_exchange {
    int num_neighbors;           // Number of MPI neighbors we communicate with
    int *neighbor_ranks;         // Ranks we communicate with [num_neighbors]
    int *send_counts;            // Elements to send to each neighbor [num_neighbors]
    int *recv_counts;            // Elements to receive from each neighbor [num_neighbors]

    // Total sizes for buffer allocation
    int total_send_size;         // Sum of send_counts
    int total_recv_size;         // Sum of recv_counts

    // Flattened index arrays for GPU (contiguous memory)
    // These replace the per-neighbor arrays from Python dicts
    int *flat_send_indices;      // Local element indices to pack for sending [total_send_size]
    int *flat_recv_indices;      // Local element indices where received data goes [total_recv_size]

    // Offset arrays for navigating flattened indices per neighbor
    int *send_offsets;           // Start offset in flat_send_indices for each neighbor [num_neighbors+1]
    int *recv_offsets;           // Start offset in flat_recv_indices for each neighbor [num_neighbors+1]

    // Communication buffers (allocated on both host and device)
    // Each element sends/receives 3 quantities: stage, xmom, ymom (centroid values)
    double *send_buffer;         // [3 * total_send_size]
    double *recv_buffer;         // [3 * total_recv_size]

    // MPI request arrays for non-blocking communication
    MPI_Request *requests;       // [2 * num_neighbors] for Isend/Irecv pairs
};

// Reflective boundary info - stored on GPU for efficient evaluation
struct reflective_boundary {
    int num_edges;               // Number of reflective boundary edges
    int *boundary_indices;       // Where to write in boundary_values arrays [num_edges]
    int *vol_ids;                // Interior cell IDs [num_edges]
    int *edge_ids;               // Which edge (0, 1, or 2) [num_edges]
    int mapped;                  // Whether arrays are mapped to GPU
};

// Dirichlet boundary info - constant values at boundary
struct dirichlet_boundary {
    int num_edges;               // Number of Dirichlet boundary edges
    int *boundary_indices;       // Where to write in boundary_values arrays [num_edges]
    int *vol_ids;                // Interior cell IDs [num_edges]
    int *edge_ids;               // Which edge (0, 1, or 2) [num_edges]
    double stage_value;          // Constant stage value
    double xmom_value;           // Constant xmom value
    double ymom_value;           // Constant ymom value
    int mapped;                  // Whether arrays are mapped to GPU
};

// Transmissive boundary info - copies interior values to boundary
struct transmissive_boundary {
    int num_edges;               // Number of transmissive boundary edges
    int *boundary_indices;       // Where to write in boundary_values arrays [num_edges]
    int *vol_ids;                // Interior cell IDs [num_edges]
    int *edge_ids;               // Which edge (0, 1, or 2) [num_edges]
    int use_centroid;            // 1 = use centroid values, 0 = use edge values
    int mapped;                  // Whether arrays are mapped to GPU
};

// Transmissive_n_momentum_zero_t_momentum_set_stage boundary
// Sets stage from external value, keeps normal momentum, zeros tangential
struct transmissive_n_zero_t_boundary {
    int num_edges;               // Number of boundary edges
    int *boundary_indices;       // Where to write in boundary_values arrays [num_edges]
    int *vol_ids;                // Interior cell IDs [num_edges]
    int *edge_ids;               // Which edge (0, 1, or 2) [num_edges]
    double stage_value;          // Current stage value (updated each timestep from Python)
    int mapped;                  // Whether arrays are mapped to GPU
};

// Time_boundary - time-dependent Dirichlet values (function called from Python each timestep)
struct time_boundary {
    int num_edges;               // Number of time boundary edges
    int *boundary_indices;       // Where to write in boundary_values arrays [num_edges]
    int *vol_ids;                // Interior cell IDs [num_edges]
    int *edge_ids;               // Which edge (0, 1, or 2) [num_edges]
    double stage_value;          // Current stage value (updated each timestep from Python)
    double xmom_value;           // Current xmom value (updated each timestep from Python)
    double ymom_value;           // Current ymom value (updated each timestep from Python)
    int mapped;                  // Whether arrays are mapped to GPU
};

// Boundary edge sync buffers - pre-allocated for efficient sparse sync
// Allocated once during setup, reused every timestep
struct boundary_edge_sync {
    int num_boundary_cells;      // Number of unique boundary-adjacent cells
    int *cell_ids;               // Cell IDs to sync [num_boundary_cells] - mapped to GPU

    // Staging buffers for gather/scatter (mapped to GPU once)
    // Size = num_boundary_cells * 3 (3 edges per cell)
    int buf_size;
    double *stage_buf;
    double *xmom_buf;
    double *ymom_buf;
    double *bed_buf;
    double *height_buf;

    int initialized;
};

// Rate operator info - for GPU-accelerated rate application (rain, etc.)
// Supports both positive rates (inflow) and negative rates (extraction)
#define MAX_RATE_OPERATORS 16

struct rate_operator_info {
    int num_indices;             // Number of triangles this operator applies to
    int *indices;                // Triangle indices [num_indices] - mapped to GPU
    double *areas;               // Triangle areas for mass tracking [num_indices]
    int *full_indices;           // Indices that are "full" (not ghost) for mass tracking
    int num_full;                // Number of full indices
    int active;                  // Whether this operator slot is in use
    int mapped;                  // Whether arrays are mapped to GPU
};

struct rate_operators {
    struct rate_operator_info ops[MAX_RATE_OPERATORS];
    int num_operators;           // Number of active operators
    int initialized;
};

// GPU domain struct - extends base domain with MPI/GPU state
struct gpu_domain {
    // Base domain struct (contains all ANUGA arrays)
    struct domain D;

    // MPI state (passed from Python via mpi4py)
    MPI_Comm comm;
    int rank;
    int nprocs;

    // GPU state
    int gpu_initialized;
    int device_id;
    int gpu_aware_mpi;           // Runtime flag: 1 if GPU-aware MPI available

    // Halo exchange info
    struct halo_exchange halo;

    // Boundary conditions
    struct reflective_boundary reflective;
    struct dirichlet_boundary dirichlet;
    struct transmissive_boundary transmissive;
    struct transmissive_n_zero_t_boundary transmissive_n_zero_t;
    struct time_boundary time_bdry;

    // Boundary edge sync (for sparse edge value sync)
    struct boundary_edge_sync edge_sync;

    // Rate operators (rain, extraction, etc.)
    struct rate_operators rate_ops;

    // Simulation parameters for GPU kernels
    double CFL;
    double evolve_max_timestep;

    // RK2 backup arrays (allocated on GPU)
    // These may already exist in base domain, but we track GPU copies here
    int backup_arrays_mapped;
};

// ============================================================================
// Function declarations
// ============================================================================

// Initialization and cleanup
int gpu_domain_init(struct gpu_domain *GD, MPI_Comm comm, int rank, int nprocs);
void gpu_domain_finalize(struct gpu_domain *GD);

// Halo exchange setup - called once after Python domain is partitioned
int gpu_halo_init(struct gpu_domain *GD,
                  int num_neighbors,
                  int *neighbor_ranks,
                  int *send_counts,
                  int *recv_counts,
                  int *flat_send_indices,
                  int *flat_recv_indices);
void gpu_halo_finalize(struct gpu_domain *GD);

// GPU memory management
void gpu_domain_map_arrays(struct gpu_domain *GD);
void gpu_domain_unmap_arrays(struct gpu_domain *GD);
void gpu_domain_sync_to_device(struct gpu_domain *GD);
void gpu_domain_sync_from_device(struct gpu_domain *GD);

// Sync boundary values TO GPU (after CPU boundary evaluation)
void gpu_sync_boundary_values(struct gpu_domain *GD);

// Sync edge values FROM GPU (before CPU boundary evaluation)
void gpu_sync_edge_values_from_device(struct gpu_domain *GD);

// Boundary edge sync - init once, sync every timestep
// Call init after boundaries are known (after set_boundary)
int gpu_boundary_edge_sync_init(struct gpu_domain *GD,
                                int num_boundary_cells,
                                int *boundary_cell_ids);
void gpu_boundary_edge_sync_finalize(struct gpu_domain *GD);
void gpu_boundary_edge_sync(struct gpu_domain *GD);  // Fast sync using pre-allocated buffers

// Reflective boundary - setup and evaluation on GPU
int gpu_reflective_init(struct gpu_domain *GD, int num_edges,
                        int *boundary_indices, int *vol_ids, int *edge_ids);
void gpu_reflective_finalize(struct gpu_domain *GD);
void gpu_evaluate_reflective_boundary(struct gpu_domain *GD);

// Dirichlet boundary - constant values at boundary
int gpu_dirichlet_init(struct gpu_domain *GD, int num_edges,
                       int *boundary_indices, int *vol_ids, int *edge_ids,
                       double stage_value, double xmom_value, double ymom_value);
void gpu_dirichlet_finalize(struct gpu_domain *GD);
void gpu_evaluate_dirichlet_boundary(struct gpu_domain *GD);

// Transmissive boundary - copies interior values to boundary
int gpu_transmissive_init(struct gpu_domain *GD, int num_edges,
                          int *boundary_indices, int *vol_ids, int *edge_ids,
                          int use_centroid);
void gpu_transmissive_finalize(struct gpu_domain *GD);
void gpu_evaluate_transmissive_boundary(struct gpu_domain *GD);

// Transmissive_n_momentum_zero_t_momentum_set_stage boundary
int gpu_transmissive_n_zero_t_init(struct gpu_domain *GD, int num_edges,
                                   int *boundary_indices, int *vol_ids, int *edge_ids);
void gpu_transmissive_n_zero_t_finalize(struct gpu_domain *GD);
void gpu_transmissive_n_zero_t_set_stage(struct gpu_domain *GD, double stage_value);
void gpu_evaluate_transmissive_n_zero_t_boundary(struct gpu_domain *GD);

// Time_boundary - time-dependent Dirichlet values
int gpu_time_boundary_init(struct gpu_domain *GD, int num_edges,
                           int *boundary_indices, int *vol_ids, int *edge_ids);
void gpu_time_boundary_finalize(struct gpu_domain *GD);
void gpu_time_boundary_set_values(struct gpu_domain *GD, double stage, double xmom, double ymom);
void gpu_evaluate_time_boundary(struct gpu_domain *GD);

// Rate operators - rain, extraction, etc.
// Returns operator ID (0 to MAX_RATE_OPERATORS-1) or -1 on error
int gpu_rate_operator_init(struct gpu_domain *GD, int num_indices, int *indices,
                           double *areas, int *full_indices, int num_full);
void gpu_rate_operator_finalize(struct gpu_domain *GD, int op_id);
void gpu_rate_operators_finalize_all(struct gpu_domain *GD);

// Apply rate operator on GPU - returns local_influx (mass added to full cells)
// rate: the rate value in m/s (scalar - computed from Python function if time-dependent)
// factor: conversion factor
// timestep: current timestep
// For negative rates, also scales momentum appropriately
double gpu_rate_operator_apply(struct gpu_domain *GD, int op_id,
                               double rate, double factor, double timestep);

// Apply rate operator with per-cell rate array (for quantity-type rates)
// rate_array: array of rate values, one per cell in indices
// rate_array_size: size of rate_array (must match num_indices or be full domain size)
// use_indices_into_rate: if 1, rate_array is full domain size, index with indices[k]
//                        if 0, rate_array matches indices size, index with k
double gpu_rate_operator_apply_array(struct gpu_domain *GD, int op_id,
                                     double *rate_array, int rate_array_size,
                                     int use_indices_into_rate,
                                     double factor, double timestep);

// Ghost exchange - the key MPI function
// Uses GPU-aware MPI if available, otherwise does D2H/H2D for small halo buffers
void gpu_exchange_ghosts(struct gpu_domain *GD);

// GPU kernels (stubs - will be implemented in sw_domain_gpu.c)
void gpu_extrapolate_second_order(struct gpu_domain *GD);
double gpu_compute_fluxes(struct gpu_domain *GD);
void gpu_update_conserved_quantities(struct gpu_domain *GD, double timestep);
void gpu_backup_conserved_quantities(struct gpu_domain *GD);
void gpu_saxpy_conserved_quantities(struct gpu_domain *GD, double a, double b);
double gpu_protect(struct gpu_domain *GD);
void gpu_manning_friction(struct gpu_domain *GD);

// Full RK2 step on GPU (calls all the above in sequence)
double gpu_evolve_one_rk2_step(struct gpu_domain *GD, double yieldstep, int apply_forcing);

// Utility functions
int detect_gpu_aware_mpi(void);
void print_gpu_domain_info(struct gpu_domain *GD);

#endif // SW_DOMAIN_GPU_H
