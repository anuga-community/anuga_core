// GPU-accelerated domain struct with MPI halo exchange support
//
// This extends the base ANUGA domain struct with:
// - MPI communicator and process info
// - Flattened halo exchange structures for GPU efficiency
// - GPU state tracking
//
// Based on miniapp_mpi.c patterns

#ifndef GPU_DOMAIN_H
#define GPU_DOMAIN_H

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

// File_boundary / Field_boundary - spatially varying time-dependent values
// The Python side evaluates F(t, point_id=i) for each edge each timestep and
// pushes the resulting per-edge arrays to the device via set_values.
struct file_boundary {
    int num_edges;               // Number of file boundary edges
    int *boundary_indices;       // Where to write in boundary_values arrays [num_edges]
    int *vol_ids;                // Interior cell IDs [num_edges]
    int *edge_ids;               // Which edge (0, 1, or 2) [num_edges]
    double *stage_values;        // Per-edge stage  (updated each timestep from Python) [num_edges]
    double *xmom_values;         // Per-edge xmom   [num_edges]
    double *ymom_values;         // Per-edge ymom   [num_edges]
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
#define MAX_RATE_OPERATORS 64

struct rate_operator_info {
    int num_indices;             // Number of triangles this operator applies to
    int *indices;                // Triangle indices [num_indices] - mapped to GPU
    double *areas;               // Triangle areas for mass tracking [num_indices]
    int *full_indices;           // Indices that are "full" (not ghost) for mass tracking
    int num_full;                // Number of full indices
    int active;                  // Whether this operator slot is in use
    int mapped;                  // Whether arrays are mapped to GPU
    // Rate array caching (avoids H2D transfer every call)
    double *rate_array_cache;    // GPU-resident copy of rate array
    int rate_array_size;         // Size of cached rate array
    int rate_array_mapped;       // Whether rate array is mapped to GPU
};

struct rate_operators {
    struct rate_operator_info ops[MAX_RATE_OPERATORS];
    int num_operators;           // Number of active operators
    int initialized;
};

// Inlet operator info - for GPU-accelerated inlet operations
// The inlet only touches 10-100 triangles, so we do small D2H/H2D
// of just those values (~6KB) instead of full domain sync (~235MB)
#define MAX_INLET_OPERATORS 32

struct inlet_operator_info {
    int num_indices;             // Number of inlet triangles
    int *indices;                // Triangle indices [num_indices] - mapped to GPU
    double *areas;               // Triangle areas [num_indices] - mapped to GPU
    double total_area;           // Precomputed sum of areas
    double *scratch_stages;      // Host scratch buffer [num_indices]
    double *scratch_bed;         // Host scratch buffer [num_indices]
    double *scratch_xmom;        // Host scratch buffer [num_indices]
    double *scratch_ymom;        // Host scratch buffer [num_indices]
    double *scratch_depths;      // Host scratch buffer [num_indices]
    int active;
    int mapped;
};

struct inlet_operators {
    struct inlet_operator_info ops[MAX_INLET_OPERATORS];
    int num_operators;
};

// Culvert operator types
#define MAX_CULVERTS 64
#define MAX_INLET_TRIANGLES 64
#define CULVERT_TYPE_BOX  0
#define CULVERT_TYPE_PIPE 1

// Static geometry parameters for one culvert
struct culvert_params {
    int type;                    // CULVERT_TYPE_BOX or CULVERT_TYPE_PIPE
    double g;                    // Gravity [m/s^2] (from domain)
    double width;                // Box width [m]
    double height;               // Box height [m]
    double diameter;             // Pipe diameter [m]
    double length;               // Culvert length [m]
    double manning;              // Manning's n for culvert
    double sum_loss;             // Sum of loss coefficients
    double blockage;             // Blockage fraction [0,1]
    double barrels;              // Number of barrels
    int use_velocity_head;
    int use_momentum_jet;
    int use_old_momentum_method;
    int always_use_Q_wetdry_adjustment;
    double max_velocity;
    double smoothing_timescale;
    double outward_vector_0[2];
    double outward_vector_1[2];
    double invert_elevation_0;
    double invert_elevation_1;
    int has_invert_elevation_0;
    int has_invert_elevation_1;
};

// Per-culvert indexing into domain arrays
struct culvert_indices {
    int enquiry_index_0;         // -1 if not on this rank
    int enquiry_index_1;         // -1 if not on this rank
    int inlet0_num;              // 0 if no local triangles
    int inlet0_indices[MAX_INLET_TRIANGLES];
    double inlet0_areas[MAX_INLET_TRIANGLES];
    double inlet0_total_area;    // LOCAL area (partial if cross-boundary)
    int inlet1_num;
    int inlet1_indices[MAX_INLET_TRIANGLES];
    double inlet1_areas[MAX_INLET_TRIANGLES];
    double inlet1_total_area;

    // MPI topology (for cross-boundary culverts)
    int master_proc;             // Rank that computes discharge
    int enquiry_proc[2];         // Rank owning each enquiry point
    int inlet_master_proc[2];    // Master rank for each inlet region
    int is_local;                // 1 = fully local (no MPI), 0 = cross-boundary
    int mpi_tag_base;            // Base MPI tag for this culvert's messages
};

// Per-culvert dynamic state (smoothing)
struct culvert_state {
    double smooth_delta_total_energy;
    double smooth_Q;
};

// Culvert manager (lives inside gpu_domain)
struct culvert_operators {
    int num_culverts;
    struct culvert_params params[MAX_CULVERTS];
    struct culvert_indices indices[MAX_CULVERTS];
    struct culvert_state state[MAX_CULVERTS];
    int initialized;

    // Scratch buffers for batched gather/scatter
    double *scratch_stage;
    double *scratch_xmom;
    double *scratch_ymom;
    double *scratch_elev;

    int total_inlet_triangles;
    int *scratch_inlet_indices;
    double *scratch_inlet_areas;
    double *scratch_inlet_stage;
    double *scratch_inlet_xmom;
    double *scratch_inlet_ymom;
    double *scratch_inlet_elev;

    int mapped;
};

// FLOP counter structure for performance profiling (Gordon Bell)
// Counts floating-point operations per kernel for FLOPS reporting
struct flop_counters {
    // Per-kernel FLOP counts (accumulated across all calls)
    uint64_t extrapolate_flops;      // gpu_extrapolate_second_order
    uint64_t compute_fluxes_flops;   // gpu_compute_fluxes
    uint64_t update_flops;           // gpu_update_conserved_quantities
    uint64_t protect_flops;          // gpu_protect
    uint64_t manning_flops;          // gpu_manning_friction
    uint64_t backup_flops;           // gpu_backup_conserved_quantities
    uint64_t saxpy_flops;            // gpu_saxpy_conserved_quantities
    uint64_t rate_operator_flops;    // gpu_rate_operator_apply*
    uint64_t ghost_exchange_flops;   // gpu_exchange_ghosts (pack/unpack)

    // Total FLOP count
    uint64_t total_flops;

    // Call counts for verification
    uint64_t extrapolate_calls;
    uint64_t compute_fluxes_calls;
    uint64_t update_calls;
    uint64_t protect_calls;
    uint64_t manning_calls;
    uint64_t backup_calls;
    uint64_t saxpy_calls;
    uint64_t rate_operator_calls;
    uint64_t ghost_exchange_calls;

    // Timing for FLOPS calculation (optional, can use external timer)
    double start_time;
    double elapsed_time;

    // Enable/disable flag
    int enabled;
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
    int verbose;                 // 0 = silent (default), 1 = print init/mapping messages

    // Halo exchange info
    struct halo_exchange halo;

    // Boundary conditions
    struct reflective_boundary reflective;
    struct dirichlet_boundary dirichlet;
    struct transmissive_boundary transmissive;
    struct transmissive_n_zero_t_boundary transmissive_n_zero_t;
    struct time_boundary time_bdry;
    struct file_boundary file_bdry;

    // Boundary edge sync (for sparse edge value sync)
    struct boundary_edge_sync edge_sync;

    // Rate operators (rain, extraction, etc.)
    struct rate_operators rate_ops;

    // Inlet operators (GPU-accelerated inlet flow)
    struct inlet_operators inlet_ops;

    // Culvert operators (Boyd box/pipe - batched GPU gather/scatter)
    struct culvert_operators culvert_ops;

    // Simulation parameters for GPU kernels
    double CFL;
    double evolve_max_timestep;
    double fixed_flux_timestep;  // <= 0 means disabled (use CFL-based timestep)

    // RK2 backup arrays (allocated on GPU)
    // These may already exist in base domain, but we track GPU copies here
    int backup_arrays_mapped;

    // FLOP counters for performance profiling (Gordon Bell)
    struct flop_counters flops;
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
size_t gpu_estimate_required_memory(anuga_int n, anuga_int nb);
int    gpu_query_device_memory(size_t *free_bytes, size_t *total_bytes);
int    gpu_check_device_memory(struct gpu_domain *GD);
int    gpu_domain_map_arrays(struct gpu_domain *GD);
void gpu_remap_boundary_arrays(struct gpu_domain *GD);
void gpu_domain_unmap_arrays(struct gpu_domain *GD);
void gpu_domain_sync_to_device(struct gpu_domain *GD);
void gpu_domain_sync_from_device(struct gpu_domain *GD);
void gpu_domain_sync_all_from_device(struct gpu_domain *GD);  // Debug: sync ALL arrays

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

// File_boundary / Field_boundary - spatially varying time-dependent values
int  gpu_file_boundary_init(struct gpu_domain *GD, int num_edges,
                             int *boundary_indices, int *vol_ids, int *edge_ids);
void gpu_file_boundary_finalize(struct gpu_domain *GD);
void gpu_file_boundary_set_values(struct gpu_domain *GD,
                                   double *stage, double *xmom, double *ymom);
void gpu_evaluate_file_boundary(struct gpu_domain *GD);

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
// rate_changed: if 1, transfer new data to GPU; if 0, reuse cached data on GPU
double gpu_rate_operator_apply_array(struct gpu_domain *GD, int op_id,
                                     double *rate_array, int rate_array_size,
                                     int use_indices_into_rate,
                                     int rate_changed,
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
double gpu_compute_water_volume(struct gpu_domain *GD);
void gpu_manning_friction(struct gpu_domain *GD);

// Full RK2 step on GPU (calls all the above in sequence)
// max_timestep: Maximum allowed timestep (respecting yieldstep/finaltime constraints)
double gpu_evolve_one_rk2_step(struct gpu_domain *GD, double max_timestep, int apply_forcing);

// Utility functions
int detect_gpu_aware_mpi(void);
void print_gpu_domain_info(struct gpu_domain *GD);

// FLOP counter functions (Gordon Bell performance profiling)
void gpu_flop_counters_init(struct gpu_domain *GD);
void gpu_flop_counters_reset(struct gpu_domain *GD);
void gpu_flop_counters_enable(struct gpu_domain *GD, int enable);
void gpu_flop_counters_start_timer(struct gpu_domain *GD);
void gpu_flop_counters_stop_timer(struct gpu_domain *GD);
uint64_t gpu_flop_counters_get_total(struct gpu_domain *GD);
double gpu_flop_counters_get_flops(struct gpu_domain *GD);  // FLOP/s
void gpu_flop_counters_print(struct gpu_domain *GD);

// Per-kernel FLOP getters
uint64_t gpu_flop_counters_get_extrapolate(struct gpu_domain *GD);
uint64_t gpu_flop_counters_get_compute_fluxes(struct gpu_domain *GD);
uint64_t gpu_flop_counters_get_update(struct gpu_domain *GD);
uint64_t gpu_flop_counters_get_protect(struct gpu_domain *GD);
uint64_t gpu_flop_counters_get_manning(struct gpu_domain *GD);
uint64_t gpu_flop_counters_get_backup(struct gpu_domain *GD);
uint64_t gpu_flop_counters_get_saxpy(struct gpu_domain *GD);
uint64_t gpu_flop_counters_get_rate_operator(struct gpu_domain *GD);
uint64_t gpu_flop_counters_get_ghost_exchange(struct gpu_domain *GD);

// MPI reduction for multi-GPU (Gordon Bell)
// Returns global totals across all ranks
uint64_t gpu_flop_counters_get_global_total(struct gpu_domain *GD);
double gpu_flop_counters_get_global_flops(struct gpu_domain *GD);
void gpu_flop_counters_print_global(struct gpu_domain *GD);

// Inlet operators - GPU-accelerated inlet flow
// Returns operator ID (0 to MAX_INLET_OPERATORS-1) or -1 on error
int gpu_inlet_operator_init(struct gpu_domain *GD, int num_indices, int *indices, double *areas);
void gpu_inlet_operator_finalize(struct gpu_domain *GD, int op_id);
void gpu_inlet_operators_finalize_all(struct gpu_domain *GD);

// Inlet operator queries (small GPU reductions returning scalars)
double gpu_inlet_get_volume(struct gpu_domain *GD, int op_id);
void gpu_inlet_get_velocities(struct gpu_domain *GD, int op_id, double *u_out, double *v_out);

// Inlet operator modifications (small GPU kernels on 10-100 triangles)
void gpu_inlet_set_depths(struct gpu_domain *GD, int op_id, double depth);
void gpu_inlet_set_xmoms(struct gpu_domain *GD, int op_id, double value);
void gpu_inlet_set_ymoms(struct gpu_domain *GD, int op_id, double value);
void gpu_inlet_set_xmoms_array(struct gpu_domain *GD, int op_id, double *values, int n);
void gpu_inlet_set_ymoms_array(struct gpu_domain *GD, int op_id, double *values, int n);
void gpu_inlet_set_stages_evenly(struct gpu_domain *GD, int op_id, double volume);

// Main entry point combining all 3 cases of inlet application
// Returns actual applied volume
double gpu_inlet_apply(struct gpu_domain *GD, int op_id, double volume,
                       double current_volume, double total_area,
                       double *vel_u, double *vel_v, int num_vel,
                       int has_velocity, double ext_vel_u, double ext_vel_v,
                       int zero_velocity);

// Culvert operators (Boyd box/pipe - batched GPU gather/scatter with MPI)
int gpu_culvert_init(struct gpu_domain *GD,
                     const struct culvert_params *params,
                     int enquiry_index_0, int enquiry_index_1,
                     int inlet0_num, int *inlet0_indices, double *inlet0_areas,
                     int inlet1_num, int *inlet1_indices, double *inlet1_areas,
                     int master_proc, int enquiry_proc_0, int enquiry_proc_1,
                     int inlet_master_proc_0, int inlet_master_proc_1,
                     int is_local, int mpi_tag_base);
void gpu_culvert_finalize(struct gpu_domain *GD, int culvert_id);
void gpu_culverts_finalize_all(struct gpu_domain *GD);
void gpu_culverts_map(struct gpu_domain *GD);
void gpu_culverts_apply_all(struct gpu_domain *GD, double timestep);

#endif // GPU_DOMAIN_H
