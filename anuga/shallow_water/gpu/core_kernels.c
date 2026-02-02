// Core kernels for CPU/GPU execution
//
// These functions use OpenMP parallel loops that compile to:
// - CPU multicore: #pragma omp parallel for simd (when -DCPU_ONLY_MODE)
// - GPU offload: #pragma omp target teams loop (otherwise)
//
// Both sw_domain_openmp_ext and sw_domain_gpu_ext use these same kernels.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "sw_domain.h"
#include "core_kernels.h"
#include "gpu_omp_macros.h"
#include "gpu_device_helpers.h"

// ============================================================================
// Extrapolation: centroid values -> edge values (second-order reconstruction)
// ============================================================================

void core_extrapolate_second_order_edge(struct domain *D) {
    anuga_int n = D->number_of_elements;
    double minimum_allowed_height = D->minimum_allowed_height;
    anuga_int extrapolate_velocity_second_order = D->extrapolate_velocity_second_order;

    // Parameters for hfactor computation (wet-dry limiting)
    double a_tmp = 0.3;
    double b_tmp = 0.1;
    double c_tmp = 1.0 / (a_tmp - b_tmp);
    double d_tmp = 1.0 - (c_tmp * a_tmp);

    // Beta values for gradient limiting
    double beta_w = D->beta_w;
    double beta_w_dry = D->beta_w_dry;
    double beta_uh = D->beta_uh;
    double beta_uh_dry = D->beta_uh_dry;
    double beta_vh = D->beta_vh;
    double beta_vh_dry = D->beta_vh_dry;

    // Extract array pointers
    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict height_cv = D->height_centroid_values;

    double * restrict stage_ev = D->stage_edge_values;
    double * restrict xmom_ev = D->xmom_edge_values;
    double * restrict ymom_ev = D->ymom_edge_values;
    double * restrict bed_ev = D->bed_edge_values;
    double * restrict height_ev = D->height_edge_values;

    double * restrict centroid_coords = D->centroid_coordinates;
    double * restrict edge_coords = D->edge_coordinates;

    anuga_int * restrict surrogate_neighbours = D->surrogate_neighbours;
    anuga_int * restrict number_of_boundaries = D->number_of_boundaries;
    double * restrict x_centroid_work = D->x_centroid_work;
    double * restrict y_centroid_work = D->y_centroid_work;

    // Step 1: Update centroid values
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

        double xmom_out = is_dry ? 0.0 : xmom;
        double ymom_out = is_dry ? 0.0 : ymom;

        double inv_dk = extrapolate ? (1.0 / dk) : 1.0;

        x_centroid_work[k] = extrapolate ? xmom_out : 0.0;
        y_centroid_work[k] = extrapolate ? ymom_out : 0.0;

        xmom_cv[k] = xmom_out * inv_dk;
        ymom_cv[k] = ymom_out * inv_dk;
    }

    // Step 2: Main extrapolation loop
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        anuga_int k2 = k * 2;
        anuga_int k3 = k * 3;
        anuga_int k6 = k * 6;

        double xv0 = edge_coords[k6 + 0];
        double yv0 = edge_coords[k6 + 1];
        double xv1 = edge_coords[k6 + 2];
        double yv1 = edge_coords[k6 + 3];
        double xv2 = edge_coords[k6 + 4];
        double yv2 = edge_coords[k6 + 5];

        double x = centroid_coords[k2 + 0];
        double y = centroid_coords[k2 + 1];

        double dxv0 = xv0 - x;
        double dxv1 = xv1 - x;
        double dxv2 = xv2 - x;
        double dyv0 = yv0 - y;
        double dyv1 = yv1 - y;
        double dyv2 = yv2 - y;

        anuga_int k0 = surrogate_neighbours[k3 + 0];
        anuga_int k1 = surrogate_neighbours[k3 + 1];
        anuga_int sn2 = surrogate_neighbours[k3 + 2];

        double x0 = centroid_coords[2 * k0 + 0];
        double y0 = centroid_coords[2 * k0 + 1];
        double x1 = centroid_coords[2 * k1 + 0];
        double y1 = centroid_coords[2 * k1 + 1];
        double x2 = centroid_coords[2 * sn2 + 0];
        double y2 = centroid_coords[2 * sn2 + 1];

        double dx1 = x1 - x0;
        double dx2 = x2 - x0;
        double dy1 = y1 - y0;
        double dy2 = y2 - y0;

        double area2 = dy2 * dx1 - dy1 * dx2;

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
            double hc = height_cv[k];
            double h0 = height_cv[k0];
            double h1 = height_cv[k1];
            double h2 = height_cv[sn2];

            double hmin = fmin(fmin(h0, fmin(h1, h2)), hc);
            double hmax = fmax(fmax(h0, fmax(h1, h2)), hc);

            double tmp1 = c_tmp * fmax(hmin, 0.0) / fmax(hc, 1.0e-06) + d_tmp;
            double tmp2 = c_tmp * fmax(hc, 0.0) / fmax(hmax, 1.0e-06) + d_tmp;
            double hfactor = fmax(0.0, fmin(tmp1, fmin(tmp2, 1.0)));

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
            anuga_int kn = k;
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
    if (extrapolate_velocity_second_order) {
        OMP_PARALLEL_LOOP
        for (anuga_int k = 0; k < n; k++) {
            xmom_cv[k] = x_centroid_work[k];
            ymom_cv[k] = y_centroid_work[k];
        }
    }
}

// ============================================================================
// Distribute edge values to vertices
// ============================================================================

void core_distribute_edges_to_vertices(struct domain *D) {
    anuga_int n = D->number_of_elements;

    double * restrict stage_ev = D->stage_edge_values;
    double * restrict xmom_ev = D->xmom_edge_values;
    double * restrict ymom_ev = D->ymom_edge_values;
    double * restrict bed_ev = D->bed_edge_values;
    double * restrict height_ev = D->height_edge_values;

    double * restrict stage_vv = D->stage_vertex_values;
    double * restrict xmom_vv = D->xmom_vertex_values;
    double * restrict ymom_vv = D->ymom_vertex_values;
    double * restrict bed_vv = D->bed_vertex_values;
    double * restrict height_vv = D->height_vertex_values;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        anuga_int k3 = k * 3;

        // Reconstruct vertex values from edge values
        // vertex[i] = edge[i+1] + edge[i+2] - edge[i]
        stage_vv[k3 + 0] = stage_ev[k3 + 1] + stage_ev[k3 + 2] - stage_ev[k3 + 0];
        stage_vv[k3 + 1] = stage_ev[k3 + 2] + stage_ev[k3 + 0] - stage_ev[k3 + 1];
        stage_vv[k3 + 2] = stage_ev[k3 + 0] + stage_ev[k3 + 1] - stage_ev[k3 + 2];

        xmom_vv[k3 + 0] = xmom_ev[k3 + 1] + xmom_ev[k3 + 2] - xmom_ev[k3 + 0];
        xmom_vv[k3 + 1] = xmom_ev[k3 + 2] + xmom_ev[k3 + 0] - xmom_ev[k3 + 1];
        xmom_vv[k3 + 2] = xmom_ev[k3 + 0] + xmom_ev[k3 + 1] - xmom_ev[k3 + 2];

        ymom_vv[k3 + 0] = ymom_ev[k3 + 1] + ymom_ev[k3 + 2] - ymom_ev[k3 + 0];
        ymom_vv[k3 + 1] = ymom_ev[k3 + 2] + ymom_ev[k3 + 0] - ymom_ev[k3 + 1];
        ymom_vv[k3 + 2] = ymom_ev[k3 + 0] + ymom_ev[k3 + 1] - ymom_ev[k3 + 2];

        bed_vv[k3 + 0] = bed_ev[k3 + 1] + bed_ev[k3 + 2] - bed_ev[k3 + 0];
        bed_vv[k3 + 1] = bed_ev[k3 + 2] + bed_ev[k3 + 0] - bed_ev[k3 + 1];
        bed_vv[k3 + 2] = bed_ev[k3 + 0] + bed_ev[k3 + 1] - bed_ev[k3 + 2];

        height_vv[k3 + 0] = height_ev[k3 + 1] + height_ev[k3 + 2] - height_ev[k3 + 0];
        height_vv[k3 + 1] = height_ev[k3 + 2] + height_ev[k3 + 0] - height_ev[k3 + 1];
        height_vv[k3 + 2] = height_ev[k3 + 0] + height_ev[k3 + 1] - height_ev[k3 + 2];
    }
}

// ============================================================================
// Compute fluxes (Kurganov-Noelle-Petrova central upwind scheme)
// ============================================================================

double core_compute_fluxes_central(struct domain *D, double timestep) {
    anuga_int n = D->number_of_elements;
    double g = D->g;
    double epsilon = D->epsilon;
    anuga_int low_froude = D->low_froude;
    double evolve_max_timestep = D->evolve_max_timestep;

    double * restrict stage_ev = D->stage_edge_values;
    double * restrict xmom_ev = D->xmom_edge_values;
    double * restrict ymom_ev = D->ymom_edge_values;
    double * restrict bed_ev = D->bed_edge_values;
    double * restrict height_ev = D->height_edge_values;
    double * restrict height_cv = D->height_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;

    double * restrict stage_bv = D->stage_boundary_values;
    double * restrict xmom_bv = D->xmom_boundary_values;
    double * restrict ymom_bv = D->ymom_boundary_values;

    double * restrict stage_eu = D->stage_explicit_update;
    double * restrict xmom_eu = D->xmom_explicit_update;
    double * restrict ymom_eu = D->ymom_explicit_update;

    anuga_int * restrict neighbours = D->neighbours;
    anuga_int * restrict neighbour_edges = D->neighbour_edges;
    double * restrict normals = D->normals;
    double * restrict edgelengths = D->edgelengths;
    double * restrict areas = D->areas;
    double * restrict radii = D->radii;

    anuga_int * restrict edge_flux_type = D->edge_flux_type;
    anuga_int * restrict edge_river_wall_counter = D->edge_river_wall_counter;
    double * restrict riverwall_elevation = D->riverwall_elevation;

    double local_timestep = evolve_max_timestep;

    // Zero explicit updates
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        stage_eu[k] = 0.0;
        xmom_eu[k] = 0.0;
        ymom_eu[k] = 0.0;
    }

    // Main flux loop
    OMP_PARALLEL_LOOP_REDUCTION_MIN(local_timestep)
    for (anuga_int k = 0; k < n; k++) {
        double area = areas[k];
        double inv_area = 1.0 / area;
        double max_speed_local = 0.0;

        for (int i = 0; i < 3; i++) {
            anuga_int ki = k * 3 + i;
            anuga_int ki2 = ki * 2;

            double ql[3], qr[3];
            ql[0] = stage_ev[ki];
            ql[1] = xmom_ev[ki];
            ql[2] = ymom_ev[ki];
            double zl = bed_ev[ki];
            double hle = height_ev[ki];
            double length = edgelengths[ki];

            anuga_int nb = neighbours[ki];
            double normal_x = normals[ki2];
            double normal_y = normals[ki2 + 1];

            double hc = height_cv[k];
            double zc = bed_cv[k];
            double hc_n, zc_n;
            double zr, hre;

            if (nb < 0) {
                // Boundary
                int m = -nb - 1;
                qr[0] = stage_bv[m];
                qr[1] = xmom_bv[m];
                qr[2] = ymom_bv[m];
                zr = zl;
                hre = fmax(qr[0] - zr, 0.0);
                hc_n = hc;
                zc_n = zc;
            } else {
                hc_n = height_cv[nb];
                zc_n = bed_cv[nb];
                int m = neighbour_edges[ki];
                int nm = nb * 3 + m;
                qr[0] = stage_ev[nm];
                qr[1] = xmom_ev[nm];
                qr[2] = ymom_ev[nm];
                zr = bed_ev[nm];
                hre = height_ev[nm];
            }

            double z_half = fmax(zl, zr);

            // Check for riverwall
            if (edge_flux_type[ki] == 1) {
                int rw_idx = edge_river_wall_counter[ki] - 1;
                double zwall = riverwall_elevation[rw_idx];
                z_half = fmax(zwall, z_half);
            }

            double h_left = fmax(hle + zl - z_half, 0.0);
            double h_right = fmax(hre + zr - z_half, 0.0);

            double edgeflux[3];
            double max_speed, pressure_flux;

            gpu_flux_function_central(ql, qr, h_left, h_right, hle, hre,
                                      normal_x, normal_y, epsilon, z_half, g,
                                      edgeflux, &max_speed, &pressure_flux, low_froude);

            // Accumulate flux
            double flux_factor = length * inv_area;
            stage_eu[k] -= edgeflux[0] * flux_factor;
            xmom_eu[k] -= edgeflux[1] * flux_factor;
            ymom_eu[k] -= edgeflux[2] * flux_factor;

            // Pressure gradient
            xmom_eu[k] -= pressure_flux * normal_x * flux_factor;
            ymom_eu[k] -= pressure_flux * normal_y * flux_factor;

            max_speed_local = fmax(max_speed_local, max_speed);
        }

        // Timestep constraint
        if (max_speed_local > epsilon) {
            double cell_timestep = radii[k] / max_speed_local;
            local_timestep = fmin(local_timestep, cell_timestep);
        }
    }

    return local_timestep;
}

// ============================================================================
// Update conserved quantities
// ============================================================================

void core_update_conserved_quantities(struct domain *D, double timestep) {
    anuga_int n = D->number_of_elements;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;

    double * restrict stage_eu = D->stage_explicit_update;
    double * restrict xmom_eu = D->xmom_explicit_update;
    double * restrict ymom_eu = D->ymom_explicit_update;

    double * restrict stage_siu = D->stage_semi_implicit_update;
    double * restrict xmom_siu = D->xmom_semi_implicit_update;
    double * restrict ymom_siu = D->ymom_semi_implicit_update;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        // Get current centroid values
        double stage_c = stage_cv[k];
        double xmom_c = xmom_cv[k];
        double ymom_c = ymom_cv[k];

        // Normalize semi-implicit update by centroid value
        double stage_si = (stage_c == 0.0) ? 0.0 : stage_siu[k] / stage_c;
        double xmom_si = (xmom_c == 0.0) ? 0.0 : xmom_siu[k] / xmom_c;
        double ymom_si = (ymom_c == 0.0) ? 0.0 : ymom_siu[k] / ymom_c;

        // Apply explicit updates
        stage_cv[k] += timestep * stage_eu[k];
        xmom_cv[k] += timestep * xmom_eu[k];
        ymom_cv[k] += timestep * ymom_eu[k];

        // Apply semi-implicit updates
        double denom;

        denom = 1.0 - timestep * stage_si;
        if (denom > 0.0) stage_cv[k] /= denom;

        denom = 1.0 - timestep * xmom_si;
        if (denom > 0.0) xmom_cv[k] /= denom;

        denom = 1.0 - timestep * ymom_si;
        if (denom > 0.0) ymom_cv[k] /= denom;

        // Reset semi-implicit updates for next timestep
        stage_siu[k] = 0.0;
        xmom_siu[k] = 0.0;
        ymom_siu[k] = 0.0;
    }
}

// ============================================================================
// Backup conserved quantities for RK2
// ============================================================================

void core_backup_conserved_quantities(struct domain *D) {
    anuga_int n = D->number_of_elements;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;

    double * restrict stage_bk = D->stage_backup_values;
    double * restrict xmom_bk = D->xmom_backup_values;
    double * restrict ymom_bk = D->ymom_backup_values;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        stage_bk[k] = stage_cv[k];
        xmom_bk[k] = xmom_cv[k];
        ymom_bk[k] = ymom_cv[k];
    }
}

// ============================================================================
// SAXPY for RK2: Q = a*Q + b*Q_backup (+ c*something for RK3)
// ============================================================================

void core_saxpy_conserved_quantities(struct domain *D, double a, double b, double c) {
    anuga_int n = D->number_of_elements;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;

    double * restrict stage_bk = D->stage_backup_values;
    double * restrict xmom_bk = D->xmom_backup_values;
    double * restrict ymom_bk = D->ymom_backup_values;

    // Note: c parameter currently unused (for future RK3 support)
    (void)c;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        stage_cv[k] = a * stage_cv[k] + b * stage_bk[k];
        xmom_cv[k] = a * xmom_cv[k] + b * xmom_bk[k];
        ymom_cv[k] = a * ymom_cv[k] + b * ymom_bk[k];
    }
}

// ============================================================================
// Protect against negative depths
// ============================================================================

double core_protect(struct domain *D) {
    anuga_int n = D->number_of_elements;
    double minimum_allowed_height = D->minimum_allowed_height;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict areas = D->areas;

    double mass_error = 0.0;

    OMP_PARALLEL_LOOP_REDUCTION_PLUS(mass_error)
    for (anuga_int k = 0; k < n; k++) {
        double h = stage_cv[k] - bed_cv[k];

        if (h < minimum_allowed_height) {
            // Very shallow - zero momentum to prevent instability
            xmom_cv[k] = 0.0;
            ymom_cv[k] = 0.0;
        }

        if (h < 0.0) {
            // Negative depth - track mass error and set stage to bed
            mass_error += (-h) * areas[k];
            stage_cv[k] = bed_cv[k];
        }
    }

    return mass_error;
}

// ============================================================================
// Fix negative cells
// ============================================================================

int core_fix_negative_cells(struct domain *D) {
    anuga_int n = D->number_of_elements;
    double minimum_allowed_height = D->minimum_allowed_height;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;

    int num_fixed = 0;

    OMP_PARALLEL_LOOP_REDUCTION_PLUS(num_fixed)
    for (anuga_int k = 0; k < n; k++) {
        double h = stage_cv[k] - bed_cv[k];
        if (h < minimum_allowed_height) {
            xmom_cv[k] = 0.0;
            ymom_cv[k] = 0.0;
            if (h < 0.0) {
                stage_cv[k] = bed_cv[k];
                num_fixed++;
            }
        }
    }

    return num_fixed;
}

// ============================================================================
// Manning friction (flat, semi-implicit)
// ============================================================================

void core_manning_friction_flat_semi_implicit(struct domain *D) {
    anuga_int n = D->number_of_elements;
    double g = D->g;
    double minimum_allowed_height = D->minimum_allowed_height;
    double seven_thirds = 7.0 / 3.0;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict friction_cv = D->friction_centroid_values;

    double * restrict xmom_siu = D->xmom_semi_implicit_update;
    double * restrict ymom_siu = D->ymom_semi_implicit_update;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        double S = 0.0;
        double uh = xmom_cv[k];
        double vh = ymom_cv[k];
        double eta = friction_cv[k];
        double abs_mom = sqrt(uh * uh + vh * vh);

        if (eta > 1.0e-15) {  // ETA_SMALL
            double h = stage_cv[k] - bed_cv[k];
            if (h >= minimum_allowed_height) {
                S = -g * eta * eta * abs_mom;
                S /= pow(h, seven_thirds);
            }
        }
        xmom_siu[k] += S * uh;
        ymom_siu[k] += S * vh;
    }
}

// ============================================================================
// Manning friction (sloped, semi-implicit)
// ============================================================================

void core_manning_friction_sloped_semi_implicit(struct domain *D) {
    anuga_int n = D->number_of_elements;
    double g = D->g;
    double minimum_allowed_height = D->minimum_allowed_height;

    double * restrict height_cv = D->height_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict friction_cv = D->friction_centroid_values;
    double * restrict bed_vv = D->bed_vertex_values;
    double * restrict vertex_coords = D->vertex_coordinates;

    double * restrict xmom_siu = D->xmom_semi_implicit_update;
    double * restrict ymom_siu = D->ymom_semi_implicit_update;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        double h = height_cv[k];

        if (h > minimum_allowed_height) {
            anuga_int k3 = k * 3;
            anuga_int k6 = k * 6;

            // Compute bed slope
            double x0 = vertex_coords[k6 + 0];
            double y0 = vertex_coords[k6 + 1];
            double x1 = vertex_coords[k6 + 2];
            double y1 = vertex_coords[k6 + 3];
            double x2 = vertex_coords[k6 + 4];
            double y2 = vertex_coords[k6 + 5];

            double z0 = bed_vv[k3 + 0];
            double z1 = bed_vv[k3 + 1];
            double z2 = bed_vv[k3 + 2];

            double det = (y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0);
            double dzx = ((y2 - y0) * (z1 - z0) - (y1 - y0) * (z2 - z0)) / det;
            double dzy = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / det;

            double slope = sqrt(dzx * dzx + dzy * dzy + 1.0e-10);

            double eta = friction_cv[k];
            double xmom = xmom_cv[k];
            double ymom = ymom_cv[k];

            double S = -g * eta * eta * sqrt(xmom * xmom + ymom * ymom) * slope;
            S /= pow(h, 7.0 / 3.0);

            xmom_siu[k] += S;
            ymom_siu[k] += S;
        }
    }
}

// ============================================================================
// Gravity term
// ============================================================================

int core_gravity(struct domain *D) {
    anuga_int n = D->number_of_elements;
    double g = D->g;

    double * restrict height_cv = D->height_centroid_values;
    double * restrict stage_vv = D->stage_vertex_values;
    double * restrict bed_vv = D->bed_vertex_values;

    double * restrict xmom_eu = D->xmom_explicit_update;
    double * restrict ymom_eu = D->ymom_explicit_update;

    double * restrict vertex_coords = D->vertex_coordinates;
    double * restrict areas = D->areas;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        double h = height_cv[k];
        if (h <= 0.0) continue;

        anuga_int k3 = k * 3;
        anuga_int k6 = k * 6;

        double x0 = vertex_coords[k6 + 0];
        double y0 = vertex_coords[k6 + 1];
        double x1 = vertex_coords[k6 + 2];
        double y1 = vertex_coords[k6 + 3];
        double x2 = vertex_coords[k6 + 4];
        double y2 = vertex_coords[k6 + 5];

        double z0 = bed_vv[k3 + 0];
        double z1 = bed_vv[k3 + 1];
        double z2 = bed_vv[k3 + 2];

        double det = (y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0);
        double dzx = ((y2 - y0) * (z1 - z0) - (y1 - y0) * (z2 - z0)) / det;
        double dzy = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / det;

        xmom_eu[k] += -g * h * dzx;
        ymom_eu[k] += -g * h * dzy;
    }

    return 0;
}

// ============================================================================
// Gravity term (well-balanced)
// ============================================================================

int core_gravity_wb(struct domain *D) {
    // For now, same as regular gravity
    // Well-balanced formulation can be added later
    return core_gravity(D);
}
