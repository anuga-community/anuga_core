// GPU-accelerated shallow water solver
// Split from sw_domain_gpu.c for maintainability

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "gpu_domain.h"

#include "gpu_device_helpers.h"

// Include pragma macros for GPU vs CPU execution
#include "gpu_omp_macros.h"

// Core kernels (shared with sw_domain_openmp_ext)
#include "core_kernels.h"

// GPU compute kernels: extrapolate, flux, protect, update, etc.

void gpu_extrapolate_second_order(struct gpu_domain *GD) {
    // Delegate to core kernel (shared with CPU implementation)
    core_extrapolate_second_order_edge(&GD->D);

    // Count FLOPs
    if (GD->flops.enabled) {
        GD->flops.extrapolate_flops += (uint64_t)GD->D.number_of_elements * FLOPS_EXTRAPOLATE;
        GD->flops.extrapolate_calls++;
    }
}

double gpu_compute_fluxes(struct gpu_domain *GD) {
    // Unified: calls core_compute_fluxes_central from core_kernels.c
    // GPU mode always uses substep_count=0 (RK2 handled at higher level)
    // timestep_fluxcalls=1 since boundary_flux_sum not used in GPU MPI mode

    double local_timestep = core_compute_fluxes_central(&GD->D, 0, 1);

    // Count FLOPs: 380 FLOPs per element (3 edges × flux function)
    if (GD->flops.enabled) {
        GD->flops.compute_fluxes_flops += (uint64_t)GD->D.number_of_elements * FLOPS_COMPUTE_FLUXES;
        GD->flops.compute_fluxes_calls++;
    }

    return local_timestep;
}

void gpu_update_conserved_quantities(struct gpu_domain *GD, double timestep) {
    // Delegate to core kernel
    core_update_conserved_quantities(&GD->D, timestep);

    // Count FLOPs: 21 FLOPs per element (explicit + semi-implicit update)
    if (GD->flops.enabled) {
        GD->flops.update_flops += (uint64_t)GD->D.number_of_elements * FLOPS_UPDATE;
        GD->flops.update_calls++;
    }
}

void gpu_backup_conserved_quantities(struct gpu_domain *GD) {
    // Delegate to core kernel
    core_backup_conserved_quantities(&GD->D);

    // Count FLOPs: 0 FLOPs per element (memory copy only)
    if (GD->flops.enabled) {
        GD->flops.backup_flops += (uint64_t)GD->D.number_of_elements * FLOPS_BACKUP;
        GD->flops.backup_calls++;
    }
}

void gpu_saxpy_conserved_quantities(struct gpu_domain *GD, double a, double b) {
    // Delegate to core kernel (c=0.0 for RK2)
    core_saxpy_conserved_quantities(&GD->D, a, b, 0.0);

    // Also update height to match the new stage (needed for volume calculation)
    anuga_int n = GD->D.number_of_elements;
    double * restrict stage_cv = GD->D.stage_centroid_values;
    double * restrict height_cv = GD->D.height_centroid_values;
    double * restrict bed_cv = GD->D.bed_centroid_values;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        height_cv[k] = fmax(stage_cv[k] - bed_cv[k], 0.0);
    }

    // Count FLOPs: 9 FLOPs per element (3 quantities × (2 mul + 1 add) + height calc)
    if (GD->flops.enabled) {
        GD->flops.saxpy_flops += (uint64_t)n * FLOPS_SAXPY;
        GD->flops.saxpy_calls++;
    }
}

double gpu_protect(struct gpu_domain *GD) {
    // Delegate to core kernel
    double mass_error = core_protect(&GD->D);

    // Also update height quantity (core_protect doesn't do this)
    anuga_int n = GD->D.number_of_elements;
    double * restrict stage_cv = GD->D.stage_centroid_values;
    double * restrict bed_cv = GD->D.bed_centroid_values;
    double * restrict height_cv = GD->D.height_centroid_values;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        height_cv[k] = fmax(stage_cv[k] - bed_cv[k], 0.0);
    }

    // Count FLOPs: 5 FLOPs per element (depth check, mass error)
    if (GD->flops.enabled) {
        GD->flops.protect_flops += (uint64_t)GD->D.number_of_elements * FLOPS_PROTECT;
        GD->flops.protect_calls++;
    }

    return mass_error;
}

double gpu_compute_water_volume(struct gpu_domain *GD) {
    // Compute total water volume on GPU
    // Returns local volume (caller should do MPI_Allreduce for global sum)
    //
    // Volume = sum((stage - elevation) * area) for all elements

    anuga_int n = GD->D.number_of_elements;
    double volume = 0.0;

    double * restrict stage_cv = GD->D.stage_centroid_values;
    double * restrict bed_cv = GD->D.bed_centroid_values;
    double * restrict areas = GD->D.areas;

    OMP_PARALLEL_LOOP_REDUCTION_PLUS(volume)
    for (anuga_int k = 0; k < n; k++) {
        double h = stage_cv[k] - bed_cv[k];
        if (h > 0.0) {
            volume += h * areas[k];
        }
    }

    return volume;
}

void gpu_manning_friction(struct gpu_domain *GD) {
    // Delegate to core kernel
    core_manning_friction_flat_semi_implicit(&GD->D);

    // Count FLOPs: 15 FLOPs per element (sqrt, pow, semi-implicit)
    if (GD->flops.enabled) {
        GD->flops.manning_flops += (uint64_t)GD->D.number_of_elements * FLOPS_MANNING;
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

    // Backup conserved quantities for RK2
    gpu_backup_conserved_quantities(GD);

    // ========================================
    // First Euler step
    // ========================================

    gpu_protect(GD);
    gpu_extrapolate_second_order(GD);

    // Evaluate all GPU-supported boundary conditions
    gpu_evaluate_reflective_boundary(GD);
    gpu_evaluate_dirichlet_boundary(GD);
    gpu_evaluate_transmissive_boundary(GD);
    gpu_evaluate_transmissive_n_zero_t_boundary(GD);
    gpu_evaluate_time_boundary(GD);

    // Compute fluxes - returns local minimum timestep
    local_timestep = gpu_compute_fluxes(GD);

    // Compute global timestep
    static int fixed_ts_printed = 0;
    if (GD->fixed_flux_timestep > 0.0) {
        // Fixed timestep - skip MPI allreduce entirely
        if (GD->rank == 0 && !fixed_ts_printed) {
            printf("Using a fixed timestep! (dt = %e)\n", GD->fixed_flux_timestep);
            fflush(stdout);
            fixed_ts_printed = 1;
        }
        timestep = GD->fixed_flux_timestep;
        if (timestep > max_timestep) {
            timestep = max_timestep;
        }
    } else {
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

    gpu_protect(GD);
    gpu_extrapolate_second_order(GD);

    // Evaluate boundary conditions (same as first step)
    gpu_evaluate_reflective_boundary(GD);
    gpu_evaluate_dirichlet_boundary(GD);
    gpu_evaluate_transmissive_boundary(GD);
    gpu_evaluate_transmissive_n_zero_t_boundary(GD);
    gpu_evaluate_time_boundary(GD);

    // Compute fluxes (ignore timestep from second step)
    gpu_compute_fluxes(GD);

    // Apply forcing terms (Manning friction on GPU)
    if (apply_forcing) {
        gpu_manning_friction(GD);
    }

    // Update conserved quantities (same timestep as first step)
    gpu_update_conserved_quantities(GD, timestep);

    // RK2 averaging: Q_final = 0.5 * Q_backup + 0.5 * Q_current
    gpu_saxpy_conserved_quantities(GD, 0.5, 0.5);

    return timestep;
}

