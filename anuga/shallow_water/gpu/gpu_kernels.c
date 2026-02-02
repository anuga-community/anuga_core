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

// ============================================================================
// GPU Kernel Stubs - To be implemented with ANUGA's numerical methods
// ============================================================================

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
    // Compute fluxes across all edges using central upwind scheme
    // Returns the local minimum timestep (caller should do MPI_Allreduce)
    //
    // This is the GPU version of _openmp_compute_fluxes_central
    // Arrays are already mapped to GPU via gpu_domain_map_arrays()

    anuga_int n = GD->D.number_of_elements;
    double g = GD->D.g;
    double epsilon = GD->D.epsilon;
    anuga_int low_froude = GD->D.low_froude;

    // Extract array pointers for GPU kernel (restrict enables better optimization)
    double * restrict stage_cv = GD->D.stage_centroid_values;
    double * restrict xmom_cv = GD->D.xmom_centroid_values;
    double * restrict ymom_cv = GD->D.ymom_centroid_values;
    double * restrict bed_cv = GD->D.bed_centroid_values;
    double * restrict height_cv = GD->D.height_centroid_values;

    double * restrict stage_ev = GD->D.stage_edge_values;
    double * restrict xmom_ev = GD->D.xmom_edge_values;
    double * restrict ymom_ev = GD->D.ymom_edge_values;
    double * restrict bed_ev = GD->D.bed_edge_values;
    double * restrict height_ev = GD->D.height_edge_values;

    double * restrict stage_bv = GD->D.stage_boundary_values;
    double * restrict xmom_bv = GD->D.xmom_boundary_values;
    double * restrict ymom_bv = GD->D.ymom_boundary_values;

    double * restrict stage_eu = GD->D.stage_explicit_update;
    double * restrict xmom_eu = GD->D.xmom_explicit_update;
    double * restrict ymom_eu = GD->D.ymom_explicit_update;

    anuga_int * restrict neighbours = GD->D.neighbours;
    anuga_int * restrict neighbour_edges = GD->D.neighbour_edges;
    double * restrict normals = GD->D.normals;
    double * restrict edgelengths = GD->D.edgelengths;
    double * restrict radii = GD->D.radii;
    double * restrict areas = GD->D.areas;
    double * restrict max_speed_array = GD->D.max_speed;

    // Main flux computation loop - no reduction, compute timestep from max_speed after
    OMP_PARALLEL_LOOP
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

            // Track max speed for this element
            speed_max_last = fmax(speed_max_last, max_speed_local);

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

    // Sync max_speed from device to host for timestep computation
#ifndef CPU_ONLY_MODE
    #pragma omp target update from(max_speed_array[0:n])
#endif

    // Compute minimum timestep from max_speed array on host (avoids GPU reduction issues)
    double local_timestep = 1.0e+100;
    for (anuga_int k = 0; k < n; k++) {
        if (max_speed_array[k] > epsilon) {
            double cell_timestep = radii[k] / max_speed_array[k];
            if (cell_timestep < local_timestep) {
                local_timestep = cell_timestep;
            }
        }
    }

    // Count FLOPs: 380 FLOPs per element (3 edges × flux function)
    if (GD->flops.enabled) {
        GD->flops.compute_fluxes_flops += (uint64_t)n * FLOPS_COMPUTE_FLUXES;
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

