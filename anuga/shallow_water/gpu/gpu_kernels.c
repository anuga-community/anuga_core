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

    // Extract array pointers (restrict enables better optimization)
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

    double * restrict centroid_coords = GD->D.centroid_coordinates;
    double * restrict edge_coords = GD->D.edge_coordinates;

    anuga_int * restrict surrogate_neighbours = GD->D.surrogate_neighbours;
    anuga_int * restrict number_of_boundaries = GD->D.number_of_boundaries;
    double * restrict x_centroid_work = GD->D.x_centroid_work;
    double * restrict y_centroid_work = GD->D.y_centroid_work;

    // Step 1: Update centroid values (compute height, optionally convert momentum to velocity)
    OMP_PARALLEL_LOOP
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
    OMP_PARALLEL_LOOP
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
        OMP_PARALLEL_LOOP
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

