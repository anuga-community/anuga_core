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

// Update boundary values from Python (small transfer at start of yieldstep)
void gpu_sync_boundary_values(struct gpu_domain *GD);

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
