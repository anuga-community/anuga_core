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

// NVTX profiling hooks (no-ops unless -DNVTX_ENABLED)
#include "gpu_nvtx.h"

// GPU compute kernels: extrapolate, flux, protect, update, etc.

void gpu_extrapolate_second_order(struct gpu_domain *GD) {
    NVTX_PUSH("gpu_extrapolate_second_order");
    // Delegate to core kernel (shared with CPU implementation)
    core_extrapolate_second_order_edge(&GD->D);

    // Count FLOPs
    if (GD->flops.enabled) {
        GD->flops.extrapolate_flops += (uint64_t)GD->D.number_of_elements * FLOPS_EXTRAPOLATE;
        GD->flops.extrapolate_calls++;
    }
    NVTX_POP();
}

double gpu_compute_fluxes(struct gpu_domain *GD) {
    return gpu_compute_fluxes_substep(GD, 0, 1, 1, 1);
}

double gpu_compute_fluxes_substep(struct gpu_domain *GD,
                                  int substep_count,
                                  int timestep_fluxcalls,
                                  int compute_timestep,
                                  int compute_boundary_flux) {
    NVTX_PUSH("gpu_compute_fluxes");
    // Unified: calls core_compute_fluxes_central from core_kernels.c.
    // Later RK stages can skip timestep and boundary-flux work.

    double local_timestep = core_compute_fluxes_central_substep(
        &GD->D, substep_count, timestep_fluxcalls, compute_timestep, compute_boundary_flux);

    // Count FLOPs: 380 FLOPs per element (3 edges × flux function)
    if (GD->flops.enabled) {
        GD->flops.compute_fluxes_flops += (uint64_t)GD->D.number_of_elements * FLOPS_COMPUTE_FLUXES;
        GD->flops.compute_fluxes_calls++;
    }

    NVTX_POP();
    return local_timestep;
}

void gpu_update_conserved_quantities(struct gpu_domain *GD, double timestep) {
    NVTX_PUSH("gpu_update_conserved_quantities");
    // Delegate to core kernel
    core_update_conserved_quantities(&GD->D, timestep);

    // Count FLOPs: 21 FLOPs per element (explicit + semi-implicit update)
    if (GD->flops.enabled) {
        GD->flops.update_flops += (uint64_t)GD->D.number_of_elements * FLOPS_UPDATE;
        GD->flops.update_calls++;
    }
    NVTX_POP();
}

void gpu_backup_conserved_quantities(struct gpu_domain *GD) {
    NVTX_PUSH("gpu_backup_conserved_quantities");
    // Delegate to core kernel
    core_backup_conserved_quantities(&GD->D);

    // Count FLOPs: 0 FLOPs per element (memory copy only)
    if (GD->flops.enabled) {
        GD->flops.backup_flops += (uint64_t)GD->D.number_of_elements * FLOPS_BACKUP;
        GD->flops.backup_calls++;
    }
    NVTX_POP();
}

void gpu_saxpy_conserved_quantities(struct gpu_domain *GD, double a, double b) {
    NVTX_PUSH("gpu_saxpy_conserved_quantities");
    // Delegate to core kernel (c=0.0 means "skip division", used for RK2)
    core_saxpy_conserved_quantities(&GD->D, a, b, 0.0);

    // Count FLOPs: 9 FLOPs per element (3 quantities × (2 mul + 1 add) + height calc)
    if (GD->flops.enabled) {
        GD->flops.saxpy_flops += (uint64_t)GD->D.number_of_elements * FLOPS_SAXPY;
        GD->flops.saxpy_calls++;
    }
    NVTX_POP();
}

void gpu_saxpy3_conserved_quantities(struct gpu_domain *GD, double a, double b, double c) {
    NVTX_PUSH("gpu_saxpy3_conserved_quantities");
    // Divide-by-c variant used for the final RK3 combination:
    //   Q = (a*Q_current + b*Q_backup) / c
    // Calling core with c != 0 and c != 1 triggers the division pass.
    core_saxpy_conserved_quantities(&GD->D, a, b, c);

    if (GD->flops.enabled) {
        GD->flops.saxpy_flops += (uint64_t)GD->D.number_of_elements * FLOPS_SAXPY;
        GD->flops.saxpy_calls++;
    }
    NVTX_POP();
}

double gpu_protect(struct gpu_domain *GD) {
    NVTX_PUSH("gpu_protect");
    // Delegate to core kernel
    double mass_error = core_protect(&GD->D);

    // Count FLOPs: 5 FLOPs per element (depth check, mass error)
    if (GD->flops.enabled) {
        GD->flops.protect_flops += (uint64_t)GD->D.number_of_elements * FLOPS_PROTECT;
        GD->flops.protect_calls++;
    }

    NVTX_POP();
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
    NVTX_PUSH("gpu_manning_friction");
    // Delegate to core kernel
    core_manning_friction_flat_semi_implicit(&GD->D);

    // Count FLOPs: 15 FLOPs per element (sqrt, pow, semi-implicit)
    if (GD->flops.enabled) {
        GD->flops.manning_flops += (uint64_t)GD->D.number_of_elements * FLOPS_MANNING;
        GD->flops.manning_calls++;
    }
    NVTX_POP();
}

// ============================================================================
// Fused extrapolate + flux GPU wrapper
// ============================================================================

double gpu_extrapolate_and_compute_fluxes_substep(struct gpu_domain *GD,
                                                   int substep_count,
                                                   int timestep_fluxcalls,
                                                   int compute_timestep,
                                                   int compute_boundary_flux) {
    NVTX_PUSH("gpu_extrapolate_and_compute_fluxes");

    double local_timestep = core_extrapolate_and_compute_fluxes(
        &GD->D, substep_count, timestep_fluxcalls, compute_timestep, compute_boundary_flux);

    if (GD->flops.enabled) {
        uint64_t n = (uint64_t)GD->D.number_of_elements;
        // Charge both extrapolation and flux FLOPs in a single call
        GD->flops.extrapolate_flops   += n * FLOPS_EXTRAPOLATE;
        GD->flops.compute_fluxes_flops += n * FLOPS_COMPUTE_FLUXES;
        GD->flops.extrapolate_calls++;
        GD->flops.compute_fluxes_calls++;
    }

    NVTX_POP();
    return local_timestep;
}

// ============================================================================
// Active cell list management
// ============================================================================

void gpu_active_cells_init(struct gpu_domain *GD) {
    if (GD->active_list_mapped) return;
    anuga_int n = GD->D.number_of_elements;

    // Allocate host-side scratch and device-side active_ids/flags.
    // We reuse a single int[n] array on device both for the flag pass
    // (core_update_active_cell_list) and then overwrite with the compacted IDs.
    GD->D.active_cell_ids = (int *)malloc(n * sizeof(int));
    if (!GD->D.active_cell_ids) {
        fprintf(stderr, "[gpu_active_cells_init] malloc failed for active_cell_ids\n");
        return;
    }
    GD->D.n_active_cells = (int)n; // start as full domain

    // Allocate an int[n] device buffer for the flag pass
    GD->active_cell_flags = (int *)malloc(n * sizeof(int));
    if (!GD->active_cell_flags) {
        fprintf(stderr, "[gpu_active_cells_init] malloc failed for active_cell_flags\n");
        free(GD->D.active_cell_ids);
        GD->D.active_cell_ids = NULL;
        return;
    }

#ifndef CPU_ONLY_MODE
    {
        int *flags    = GD->active_cell_flags;
        int *act_ids  = GD->D.active_cell_ids;
        #pragma omp target enter data map(alloc: flags[0:n], act_ids[0:n])
        // OMP pointer attachment (OpenMP 5.0+ §2.21.7.1): when D is passed
        // firstprivate to a target region AFTER this map(alloc), the runtime
        // automatically replaces D->active_cell_ids / D->active_cell_flags with
        // their device counterparts. This is correct for NVC/NVHPC 25.x+.
        // For compilers without attachment support add:
        //   #pragma omp target update to(GD->D.active_cell_ids)
        // (PR review comment #2)
    }
#endif

    GD->active_list_mapped = 1;
}

void gpu_active_cells_finalize(struct gpu_domain *GD) {
    if (!GD->active_list_mapped) return;
    anuga_int n = GD->D.number_of_elements;

#ifndef CPU_ONLY_MODE
    {
        int *flags   = GD->active_cell_flags;
        int *act_ids = GD->D.active_cell_ids;
        #pragma omp target exit data map(delete: flags[0:n])
        #pragma omp target exit data map(delete: act_ids[0:n])
    }
#endif

    free(GD->active_cell_flags);
    free(GD->D.active_cell_ids);
    GD->active_cell_flags     = NULL;
    GD->D.active_cell_ids     = NULL;
    GD->D.n_active_cells      = 0;
    GD->active_list_mapped    = 0;
}

// Rebuild the active cell list each timestep.
// Returns the new n_active_cells (0 if disabled).
int gpu_active_cells_update(struct gpu_domain *GD) {
    if (!GD->use_active_cells) return 0;

    // Issue 9: guard against use_active_cells=1 but active_list_mapped=0.
    // This can happen when a gpu_domain struct is re-mapped (map_arrays resets
    // active_list_mapped to 0) while use_active_cells was already 1.  Without
    // this guard the optimisation silently becomes a no-op with no warning.
    if (!GD->active_list_mapped) {
        fprintf(stderr,
            "[gpu_active_cells_update] WARNING: use_active_cells=1 but "
            "active_list_mapped=0.  Calling gpu_active_cells_init now.\n");
        gpu_active_cells_init(GD);
        if (!GD->active_list_mapped) {
            // Allocation failed inside init — disable to avoid a crash.
            fprintf(stderr,
                "[gpu_active_cells_update] ERROR: gpu_active_cells_init failed; "
                "disabling active cell optimisation for safety.\n");
            GD->use_active_cells = 0;
            return 0;
        }
    }

    anuga_int n   = GD->D.number_of_elements;
    double    mah = GD->D.minimum_allowed_height;
    (void)mah;  // used indirectly through core_update_active_cell_list

    int *flags = GD->active_cell_flags;

    // Pass 1 on device: mark flags via shared core kernel (avoids duplication
    // with core_update_active_cell_list in core_kernels.c).
    // core_update_active_cell_list writes 1/0 into act_ids[] as a flag array,
    // then returns -1 to signal that the caller must compact.
    int *act_ids = GD->D.active_cell_ids;
    core_update_active_cell_list(&GD->D, flags);

    // Pass 2 on host: D2H flags, compact, H2D active_ids
#ifndef CPU_ONLY_MODE
    #pragma omp target update from(flags[0:n])
#endif

    int count = 0;
    for (anuga_int k = 0; k < n; k++) {
        if (flags[k]) act_ids[count++] = (int)k;
    }
    GD->D.n_active_cells = count;

#ifndef CPU_ONLY_MODE
    #pragma omp target update to(act_ids[0:count])
#endif

    return count;
}

// ============================================================================
// Full RK2 Step - Orchestrates all GPU operations
// ============================================================================

double gpu_evolve_one_rk2_step(struct gpu_domain *GD, double max_timestep, int apply_forcing,
                               int compute_boundary_flux) {
    NVTX_PUSH("gpu_evolve_one_rk2_step");

    double local_timestep, global_timestep;

    // Defensive initialisation: timestep is explicitly set through every branch
    // of the fixed/async/sync/serial MPI logic below, but starting with
    // max_timestep ensures that any future code insertion between the async
    // MPI_Iallreduce and its MPI_Wait cannot silently read an uninitialised
    // value (Issue 7).
    double timestep = max_timestep;

    // Active cell list reflects wet/dry state from END of previous timestep
    // (one-step lag). Newly wet cells may be skipped once; newly dried cells
    // are processed once extra. Both are conservative and bounded by CFL.
    // Rebuilding after protect/update would add a full-domain scan per substep
    // at higher cost than the occasional extra work. (PR review comment #4)
    gpu_active_cells_update(GD);

    // Backup conserved quantities for RK2
    gpu_backup_conserved_quantities(GD);

    // ========================================
    // First Euler step - FUSED extrapolate+flux
    // ========================================

    gpu_protect(GD);

    // Evaluate all GPU-supported boundary conditions
    gpu_evaluate_reflective_boundary(GD);
    gpu_evaluate_dirichlet_boundary(GD);
    gpu_evaluate_transmissive_boundary(GD);
    gpu_evaluate_transmissive_n_zero_t_boundary(GD);
    gpu_evaluate_time_boundary(GD);
    gpu_evaluate_file_boundary(GD);

    // Fused extrapolate + flux: single kernel launch, edge data L2-hot between passes
    local_timestep = gpu_extrapolate_and_compute_fluxes_substep(GD, 0, 2, 1, compute_boundary_flux);

    // -----------------------------------------------------------------------
    // NON-BLOCKING MPI TIMESTEP REDUCTION
    // Issue MPI_Iallreduce immediately after the first flux call so the
    // reduction network latency overlaps with Manning friction + update.
    // We complete the request just before it is needed for the timestep value.
    // -----------------------------------------------------------------------
    static int fixed_ts_printed = 0;
    MPI_Request ts_request = MPI_REQUEST_NULL;
    int use_async_mpi = (GD->nprocs > 1 && GD->fixed_flux_timestep <= 0.0 && apply_forcing);

    if (GD->fixed_flux_timestep > 0.0) {
        if (GD->rank == 0 && !fixed_ts_printed) {
            printf("Using a fixed timestep! (dt = %e)\n", GD->fixed_flux_timestep);
            fflush(stdout);
            fixed_ts_printed = 1;
        }
        timestep = fmin(GD->fixed_flux_timestep, max_timestep);
    } else if (use_async_mpi) {
        // Fire-and-forget: overlap reduction with forcing + update below
        MPI_Iallreduce(&local_timestep, &global_timestep, 1, MPI_DOUBLE,
                       MPI_MIN, GD->comm, &ts_request);
    } else if (GD->nprocs > 1) {
        MPI_Allreduce(&local_timestep, &global_timestep, 1, MPI_DOUBLE, MPI_MIN, GD->comm);
        timestep = fmin(GD->CFL * global_timestep, max_timestep);
    } else {
        timestep = fmin(GD->CFL * local_timestep, max_timestep);
    }

    // Apply forcing (Manning friction) while MPI reduction is in flight
    if (apply_forcing) {
        gpu_manning_friction(GD);
    }

    // Complete async reduction (if started) now that forcing is done
    if (use_async_mpi) {
        MPI_Wait(&ts_request, MPI_STATUS_IGNORE);
        timestep = fmin(GD->CFL * global_timestep, max_timestep);
    }

    // Update conserved quantities with computed timestep
    gpu_update_conserved_quantities(GD, timestep);

    // Ghost exchange (MPI) - sync ghost cells between processes
    if (GD->nprocs > 1) {
        gpu_exchange_ghosts(GD);
    }

    // ========================================
    // Second Euler step - FUSED extrapolate+flux
    // ========================================

    gpu_protect(GD);

    // Evaluate boundary conditions (same as first step)
    gpu_evaluate_reflective_boundary(GD);
    gpu_evaluate_dirichlet_boundary(GD);
    gpu_evaluate_transmissive_boundary(GD);
    gpu_evaluate_transmissive_n_zero_t_boundary(GD);
    gpu_evaluate_time_boundary(GD);
    gpu_evaluate_file_boundary(GD);

    // Fused: skip timestep reduction and ignore the returned local_timestep
    gpu_extrapolate_and_compute_fluxes_substep(GD, 1, 2, 0, compute_boundary_flux);

    // Apply forcing terms (Manning friction on GPU)
    if (apply_forcing) {
        gpu_manning_friction(GD);
    }

    // Update conserved quantities (same timestep as first step)
    gpu_update_conserved_quantities(GD, timestep);

    // RK2 averaging: Q_final = 0.5 * Q_backup + 0.5 * Q_current
    gpu_saxpy_conserved_quantities(GD, 0.5, 0.5);

    NVTX_POP();  // gpu_evolve_one_rk2_step
    return timestep;
}

// ============================================================================
// Full SSP-RK3 Step (Shu-Osher)
// ============================================================================

double gpu_evolve_one_rk3_step(struct gpu_domain *GD, double max_timestep, int apply_forcing,
                               int compute_boundary_flux) {
    NVTX_PUSH("gpu_evolve_one_rk3_step");

    double local_timestep, global_timestep;

    // Same defensive initialisation as the RK2 step (Issue 7).
    double timestep = max_timestep;

    // Update active cell list once per timestep
    gpu_active_cells_update(GD);

    // Backup Q^n
    gpu_backup_conserved_quantities(GD);

    // ========================================
    // Stage 1: Q^(1) = Q^n + h*L(Q^n)
    // ========================================

    gpu_protect(GD);

    gpu_evaluate_reflective_boundary(GD);
    gpu_evaluate_dirichlet_boundary(GD);
    gpu_evaluate_transmissive_boundary(GD);
    gpu_evaluate_transmissive_n_zero_t_boundary(GD);
    gpu_evaluate_time_boundary(GD);
    gpu_evaluate_file_boundary(GD);

    local_timestep = gpu_extrapolate_and_compute_fluxes_substep(GD, 0, 3, 1, compute_boundary_flux);

    // Non-blocking MPI reduction overlapped with forcing
    static int fixed_ts_printed_rk3 = 0;
    MPI_Request ts_request = MPI_REQUEST_NULL;
    int use_async_mpi = (GD->nprocs > 1 && GD->fixed_flux_timestep <= 0.0 && apply_forcing);

    if (GD->fixed_flux_timestep > 0.0) {
        if (GD->rank == 0 && !fixed_ts_printed_rk3) {
            printf("RK3: Using a fixed timestep! (dt = %e)\n", GD->fixed_flux_timestep);
            fflush(stdout);
            fixed_ts_printed_rk3 = 1;
        }
        timestep = fmin(GD->fixed_flux_timestep, max_timestep);
    } else if (use_async_mpi) {
        MPI_Iallreduce(&local_timestep, &global_timestep, 1, MPI_DOUBLE,
                       MPI_MIN, GD->comm, &ts_request);
    } else if (GD->nprocs > 1) {
        MPI_Allreduce(&local_timestep, &global_timestep, 1, MPI_DOUBLE, MPI_MIN, GD->comm);
        timestep = fmin(GD->CFL * global_timestep, max_timestep);
    } else {
        timestep = fmin(GD->CFL * local_timestep, max_timestep);
    }

    if (apply_forcing) gpu_manning_friction(GD);

    if (use_async_mpi) {
        MPI_Wait(&ts_request, MPI_STATUS_IGNORE);
        timestep = fmin(GD->CFL * global_timestep, max_timestep);
    }

    gpu_update_conserved_quantities(GD, timestep);
    if (GD->nprocs > 1) gpu_exchange_ghosts(GD);

    // ========================================
    // Stage 2: Q^(2) = Q^(1) + h*L(Q^(1))
    // ========================================

    gpu_protect(GD);
    gpu_evaluate_reflective_boundary(GD);
    gpu_evaluate_dirichlet_boundary(GD);
    gpu_evaluate_transmissive_boundary(GD);
    gpu_evaluate_transmissive_n_zero_t_boundary(GD);
    gpu_evaluate_time_boundary(GD);
    gpu_evaluate_file_boundary(GD);

    gpu_extrapolate_and_compute_fluxes_substep(GD, 1, 3, 0, compute_boundary_flux);
    if (apply_forcing) gpu_manning_friction(GD);
    gpu_update_conserved_quantities(GD, timestep);

    // Intermediate: Q = 0.25*Q^(2) + 0.75*Q^n
    gpu_saxpy_conserved_quantities(GD, 0.25, 0.75);
    if (GD->nprocs > 1) gpu_exchange_ghosts(GD);

    // ========================================
    // Stage 3: Q^(3) = Q_mid + h*L(Q_mid)
    // ========================================

    gpu_protect(GD);
    gpu_evaluate_reflective_boundary(GD);
    gpu_evaluate_dirichlet_boundary(GD);
    gpu_evaluate_transmissive_boundary(GD);
    gpu_evaluate_transmissive_n_zero_t_boundary(GD);
    gpu_evaluate_time_boundary(GD);
    gpu_evaluate_file_boundary(GD);

    gpu_extrapolate_and_compute_fluxes_substep(GD, 2, 3, 0, compute_boundary_flux);
    if (apply_forcing) gpu_manning_friction(GD);
    gpu_update_conserved_quantities(GD, timestep);

    // Final: Q^{n+1} = (2*Q^(3) + Q^n) / 3
    gpu_saxpy3_conserved_quantities(GD, 2.0, 1.0, 3.0);

    NVTX_POP();  // gpu_evolve_one_rk3_step
    return timestep;
}

