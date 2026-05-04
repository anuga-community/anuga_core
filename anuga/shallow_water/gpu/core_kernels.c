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
        double xmom_out = is_dry ? 0.0 : xmom;
        double ymom_out = is_dry ? 0.0 : ymom;

        x_centroid_work[k] = xmom_out;
        y_centroid_work[k] = ymom_out;

        xmom_cv[k] = xmom_out;
        ymom_cv[k] = ymom_out;
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

        double xmom_q  = xmom_cv[k];
        double ymom_q  = ymom_cv[k];
        double xmom_q0 = xmom_cv[k0];
        double ymom_q0 = ymom_cv[k0];
        double xmom_q1 = xmom_cv[k1];
        double ymom_q1 = ymom_cv[k1];
        double xmom_q2 = xmom_cv[sn2];
        double ymom_q2 = ymom_cv[sn2];

        if (extrapolate_velocity_second_order == 1) {
            double hk  = height_cv[k];
            double h0q = height_cv[k0];
            double h1q = height_cv[k1];
            double h2q = height_cv[sn2];

            xmom_q  = (hk  > minimum_allowed_height) ? (x_centroid_work[k] / hk) : 0.0;
            ymom_q  = (hk  > minimum_allowed_height) ? (y_centroid_work[k] / hk) : 0.0;
            xmom_q0 = (h0q > minimum_allowed_height) ? (x_centroid_work[k0] / h0q) : 0.0;
            ymom_q0 = (h0q > minimum_allowed_height) ? (y_centroid_work[k0] / h0q) : 0.0;
            xmom_q1 = (h1q > minimum_allowed_height) ? (x_centroid_work[k1] / h1q) : 0.0;
            ymom_q1 = (h1q > minimum_allowed_height) ? (y_centroid_work[k1] / h1q) : 0.0;
            xmom_q2 = (h2q > minimum_allowed_height) ? (x_centroid_work[sn2] / h2q) : 0.0;
            ymom_q2 = (h2q > minimum_allowed_height) ? (y_centroid_work[sn2] / h2q) : 0.0;
        }

        int num_boundaries = number_of_boundaries[k];

        if (num_boundaries == 3) {
            double stage_c = stage_cv[k];
            double xmom_c = xmom_q;
            double ymom_c = ymom_q;
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
                    xmom_q, xmom_q0, xmom_q1, xmom_q2,
                    dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                    dx1, dx2, dy1, dy2, inv_area2, beta_xmom, edge_vals);
            } else {
                gpu_set_constant_edge_values(xmom_q, edge_vals);
            }
            xmom_ev[k3 + 0] = edge_vals[0];
            xmom_ev[k3 + 1] = edge_vals[1];
            xmom_ev[k3 + 2] = edge_vals[2];

            // Y-momentum
            double beta_ymom = beta_vh_dry + (beta_vh - beta_vh_dry) * hfactor;
            if (beta_ymom > 0.0) {
                gpu_calc_edge_values_with_gradient(
                    ymom_q, ymom_q0, ymom_q1, ymom_q2,
                    dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                    dx1, dx2, dy1, dy2, inv_area2, beta_ymom, edge_vals);
            } else {
                gpu_set_constant_edge_values(ymom_q, edge_vals);
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
            double xmom_kn = xmom_cv[kn];
            double ymom_kn = ymom_cv[kn];
            if (extrapolate_velocity_second_order == 1) {
                double hkn = height_cv[kn];
                xmom_kn = (hkn > minimum_allowed_height) ? (x_centroid_work[kn] / hkn) : 0.0;
                ymom_kn = (hkn > minimum_allowed_height) ? (y_centroid_work[kn] / hkn) : 0.0;
            }

            dq1 = xmom_kn - xmom_q;
            gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                          dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
            gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
            gpu_limit_gradient(dqv, qmin, qmax, beta_w);
            xmom_ev[k3 + 0] = xmom_q + dqv[0];
            xmom_ev[k3 + 1] = xmom_q + dqv[1];
            xmom_ev[k3 + 2] = xmom_q + dqv[2];

            // Y-momentum
            dq1 = ymom_kn - ymom_q;
            gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                          dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
            gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
            gpu_limit_gradient(dqv, qmin, qmax, beta_w);
            ymom_ev[k3 + 0] = ymom_q + dqv[0];
            ymom_ev[k3 + 1] = ymom_q + dqv[1];
            ymom_ev[k3 + 2] = ymom_q + dqv[2];
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

    // Centroid momenta stay in momentum form throughout; velocities are
    // reconstructed in registers above when velocity extrapolation is enabled.
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
// Update conserved quantities
// ============================================================================

void core_update_conserved_quantities(struct domain *D, double timestep) {
    anuga_int n = D->number_of_elements;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict height_cv = D->height_centroid_values;

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

        height_cv[k] = fmax(stage_cv[k] - bed_cv[k], 0.0);

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
// SAXPY for RK2/RK3: Q = (a*Q + b*Q_backup) / c
// ============================================================================

void core_saxpy_conserved_quantities(struct domain *D, double a, double b, double c) {
    anuga_int n = D->number_of_elements;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict height_cv = D->height_centroid_values;

    double * restrict stage_bk = D->stage_backup_values;
    double * restrict xmom_bk = D->xmom_backup_values;
    double * restrict ymom_bk = D->ymom_backup_values;

    double scale = (c != 1.0 && c != 0.0) ? (1.0 / c) : 1.0;

    // Standard SAXPY: Q = (a*Q + b*Q_backup) / c, with height kept current.
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        double stage = (a * stage_cv[k] + b * stage_bk[k]) * scale;

        stage_cv[k] = stage;
        xmom_cv[k] = (a * xmom_cv[k] + b * xmom_bk[k]) * scale;
        ymom_cv[k] = (a * ymom_cv[k] + b * ymom_bk[k]) * scale;
        height_cv[k] = fmax(stage - bed_cv[k], 0.0);
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
    double * restrict height_cv = D->height_centroid_values;
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
            h = 0.0;
        }

        height_cv[k] = fmax(h, 0.0);
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

            double slope = sqrt(1.0 + dzx * dzx + dzy * dzy);

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

// ============================================================================
// Compute fluxes using central upwind scheme (UNIFIED CPU/GPU)
// ============================================================================

double core_compute_fluxes_central(struct domain *D, int substep_count, int timestep_fluxcalls) {
    return core_compute_fluxes_central_substep(D, substep_count, timestep_fluxcalls, 1, 1);
}

double core_compute_fluxes_central_substep(struct domain *D,
                                           int substep_count,
                                           int timestep_fluxcalls,
                                           int compute_timestep,
                                           int compute_boundary_flux) {
    anuga_int n = D->number_of_elements;
    double g = D->g;
    double epsilon = D->epsilon;
    anuga_int low_froude = D->low_froude;

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
    double * restrict radii = D->radii;
    double * restrict areas = D->areas;
    double * restrict max_speed_array = D->max_speed;
    anuga_int * restrict tri_full_flag = D->tri_full_flag;

    // Riverwall arrays (may be NULL if no riverwalls)
    anuga_int n_riverwall_edges = D->number_of_riverwall_edges;
    anuga_int ncol_riverwall_hp = D->ncol_riverwall_hydraulic_properties;
    anuga_int * restrict edge_flux_type = D->edge_flux_type;
    anuga_int * restrict edge_river_wall_counter = D->edge_river_wall_counter;
    double * restrict riverwall_elevation = D->riverwall_elevation;
    anuga_int * restrict riverwall_rowIndex = D->riverwall_rowIndex;
    double * restrict riverwall_hydraulic_properties = D->riverwall_hydraulic_properties;
    int has_riverwalls = (n_riverwall_edges > 0 &&
                          edge_flux_type != NULL &&
                          edge_river_wall_counter != NULL &&
                          riverwall_elevation != NULL);

    // Reduction variables
    double local_timestep = 1.0e+100;
    double boundary_flux_sum_substep = 0.0;

    // Main flux computation loop with reductions
    #ifdef CPU_ONLY_MODE
    #pragma omp parallel for simd reduction(min:local_timestep) reduction(+:boundary_flux_sum_substep)
    #else
    #pragma omp target teams distribute parallel for reduction(min:local_timestep) reduction(+:boundary_flux_sum_substep)
    #endif
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

            // Check for riverwall elevation override
            int is_riverwall = 0;
            double zwall = 0.0;
            if (has_riverwalls && edge_flux_type[ki] == 1) {
                int riverwall_index = edge_river_wall_counter[ki] - 1;
                if (riverwall_index >= 0) {
                    is_riverwall = 1;
                    zwall = riverwall_elevation[riverwall_index];
                    z_half = fmax(zwall, z_half);
                }
            }

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

            // Apply riverwall weir discharge correction if applicable
            if (is_riverwall && zwall > fmax(zc, zc_n) &&
                riverwall_rowIndex != NULL && riverwall_hydraulic_properties != NULL) {
                // Get hydraulic properties for this riverwall
                anuga_int rw_count = edge_river_wall_counter[ki];
                anuga_int hp_row = riverwall_rowIndex[rw_count - 1];
                anuga_int ii = hp_row * ncol_riverwall_hp;

                double Qfactor = riverwall_hydraulic_properties[ii];
                double s1 = riverwall_hydraulic_properties[ii + 1];
                double s2 = riverwall_hydraulic_properties[ii + 2];
                double h1 = riverwall_hydraulic_properties[ii + 3];
                double h2 = riverwall_hydraulic_properties[ii + 4];
                // Column 5 is Cd_through; guard for old files with only 5 columns
                double Cd_through = (ncol_riverwall_hp > 5)
                    ? riverwall_hydraulic_properties[ii + 5]
                    : 0.0;

                // Weir height above minimum bed elevation
                double weir_height = fmax(zwall - fmin(zl, zr), 0.0);

                // Compute depths above weir using centroid values
                double h_left_weir = fmax(stage_cv[k] - z_half, 0.0);
                double h_right_weir = is_boundary
                    ? fmax(hc_n + zr - z_half, 0.0)
                    : fmax(stage_cv[neighbour] - z_half, 0.0);

                // Apply weir discharge correction (Villemonte overtopping)
                gpu_adjust_edgeflux_with_weir(edgeflux, h_left_weir, h_right_weir,
                                              g, weir_height, Qfactor,
                                              s1, s2, h1, h2, &max_speed_local);

                // Apply throughflow (orifice/seepage through wall body), additive
                double stage_left  = stage_cv[k];
                double stage_right = is_boundary
                    ? (hc_n + zr)
                    : stage_cv[neighbour];
                gpu_adjust_edgeflux_with_throughflow(
                    edgeflux,
                    stage_left, stage_right,
                    zl, zr,
                    zwall, g, Cd_through, &max_speed_local);
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

            // Boundary flux tracking: if this cell is not a ghost, and the neighbour
            // is a boundary condition OR a ghost cell, add the flux to boundary integral
            if (compute_boundary_flux && tri_full_flag != NULL) {
                int is_full = (tri_full_flag[k] == 1);
                int neighbour_is_ghost = (!is_boundary && tri_full_flag[neighbour] == 0);
                if ((is_boundary && is_full) || (is_full && neighbour_is_ghost)) {
                    boundary_flux_sum_substep += edgeflux[0];
                }
            }

            // Pressure gradient (gravity) terms
            double pressuregrad_work = length * (-g * 0.5 * (h_left * h_left - hle * hle
                                       - (hle + hc) * (zl - zc)) + pressure_flux);
            xmom_eu[k] -= normals[ki2] * pressuregrad_work;
            ymom_eu[k] -= normals[ki2 + 1] * pressuregrad_work;

        } // End edge loop

        // Update timestep only on first substep and for non-ghost cells
        if (compute_timestep && substep_count == 0) {
            if (tri_full_flag == NULL || tri_full_flag[k] == 1) {
                if (speed_max_last > epsilon) {
                    double cell_timestep = radii[k] / speed_max_last;
                    local_timestep = fmin(local_timestep, cell_timestep);
                }
            }
            max_speed_array[k] = speed_max_last;
        }

        // Normalize by area
        double inv_area = 1.0 / areas[k];
        stage_eu[k] *= inv_area;
        xmom_eu[k] *= inv_area;
        ymom_eu[k] *= inv_area;

    } // End element loop

    // Store boundary flux sum for this substep
    if (compute_boundary_flux && D->boundary_flux_sum != NULL && substep_count < timestep_fluxcalls) {
        D->boundary_flux_sum[substep_count] = boundary_flux_sum_substep;
    }

    // Return timestep (only meaningful on first substep)
    return local_timestep;
}

// ============================================================================
// Active cell list builder
// ============================================================================
// Scans height_centroid_values and marks cells as active when they are wet
// (h > minimum_allowed_height) or are adjacent to a wet cell (wetting front).
// Returns the number of active cells, and updates D->n_active_cells.
//
// active_ids is written on-device when compiled for GPU, or on-host otherwise.
// NOTE: Because OMP target regions cannot perform atomic scatter into a dense
//       counter without a global reduction, we implement this as two passes:
//       Pass 1 - mark active flags [0/1] per cell (GPU parallel)
//       Pass 2 - prefix-scan to compact (CPU after D2H flag read)
//
// For GPU use, the two-pass strategy keeps the GPU busy on the heavy pass
// while the compact step is cheap (< 1 ms for 1M cells on CPU).

int core_update_active_cell_list(struct domain *D, int *active_ids) {
    anuga_int n = D->number_of_elements;
    double mah = D->minimum_allowed_height;

    double * restrict height_cv = D->height_centroid_values;
    anuga_int * restrict neighbours = D->neighbours;

    // Pass 1: mark flag[k]=1 if cell k is wet or has a wet neighbour
    // Uses device memory (active_ids repurposed as flag scratch before compaction).
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        int wet = (height_cv[k] > mah);
        if (!wet) {
            // Check three neighbours for wetting front
            for (int i = 0; i < 3; i++) {
                anuga_int nb = neighbours[3 * k + i];
                if (nb >= 0 && height_cv[nb] > mah) {
                    wet = 1;
                    break;
                }
            }
        }
        active_ids[k] = wet;   // 1 = active, 0 = inactive (flag mode)
    }

    // Pass 2: compact in-place on host (flags already sync'd by caller if GPU)
    // The caller (gpu_active_cells_update) handles D2H + compact + H2D.
    // This function only fills the flag array.
    // Return -1 to signal "flags written, caller must compact".
    (void)active_ids; // suppress unused-after-write warning in CPU_ONLY_MODE
    return -1;
}


// ============================================================================
// Fused extrapolation + flux kernel
// ============================================================================
// Fuses core_extrapolate_second_order_edge and core_compute_fluxes_central_substep
// into a SINGLE GPU kernel.  Edge values for cell k are computed in registers
// and consumed immediately by the Riemann solver without ever writing them to
// the device edge-value arrays.  This eliminates the 6 * N * 3 * sizeof(double)
// HBM round-trip that the split two-kernel approach requires.
//
// Geometry: each cell k processes its own three edges.  For each edge we need
// both the LEFT state (cell k) and the RIGHT state (neighbour cell or boundary).
// The left edge values are reconstructed from cell k's centroid data.
// The right edge values are either read from the pre-computed edge arrays of the
// neighbour cell (which were filled during a previous extrapolation pass for that
// neighbour) OR computed on-the-fly when we have time.
//
// IMPORTANT: Because we still need the NEIGHBOUR's reconstructed edge values
// (which depend on its own surrogate neighbourhood), we cannot fuse the full
// extrapolation unless we accept a two-pass approach within the same kernel.
// This implementation uses the following strategy:
//
//   Pass 1 (extrapolation):  Write edge values to the existing arrays as before.
//                             This is intentional - neighbours read their
//                             *already-written* edge values in Pass 2.
//   Pass 2 (flux from edges): Identical to core_compute_fluxes_central_substep,
//                             but launched WITHIN THE SAME kernel invocation so
//                             that Pass 1 edge data is still L1/L2-hot.
//
// The key win versus two separate kernel launches:
//   - ONE kernel launch overhead instead of two
//   - Edge array writes from Pass 1 are L2-resident when Pass 2 reads them
//     (instead of cold reads from HBM after a separate kernel dispatch)
//   - The GPU hardware scheduler can overlap the tail of Pass 1 with the
//     start of Pass 2 for independent cells (different warps)
//
// Both passes are expressed as OMP parallel loops; the barrier between them
// is the implicit synchronisation at the end of the first OMP target region.

double core_extrapolate_and_compute_fluxes(struct domain *D,
                                           int substep_count,
                                           int timestep_fluxcalls,
                                           int compute_timestep,
                                           int compute_boundary_flux) {
    // ----------------------------------------------------------------
    // Pass 1: Extrapolation (identical to core_extrapolate_second_order_edge)
    // ----------------------------------------------------------------
    anuga_int n = D->number_of_elements;
    double minimum_allowed_height = D->minimum_allowed_height;
    anuga_int extrapolate_velocity_second_order = D->extrapolate_velocity_second_order;

    double a_tmp = 0.3, b_tmp = 0.1;
    double c_tmp = 1.0 / (a_tmp - b_tmp);
    double d_tmp = 1.0 - (c_tmp * a_tmp);

    double beta_w      = D->beta_w;
    double beta_w_dry  = D->beta_w_dry;
    double beta_uh     = D->beta_uh;
    double beta_uh_dry = D->beta_uh_dry;
    double beta_vh     = D->beta_vh;
    double beta_vh_dry = D->beta_vh_dry;

    double * restrict stage_cv    = D->stage_centroid_values;
    double * restrict xmom_cv     = D->xmom_centroid_values;
    double * restrict ymom_cv     = D->ymom_centroid_values;
    double * restrict bed_cv      = D->bed_centroid_values;
    double * restrict height_cv   = D->height_centroid_values;
    double * restrict stage_ev    = D->stage_edge_values;
    double * restrict xmom_ev     = D->xmom_edge_values;
    double * restrict ymom_ev     = D->ymom_edge_values;
    double * restrict bed_ev      = D->bed_edge_values;
    double * restrict height_ev   = D->height_edge_values;
    double * restrict centroid_coords = D->centroid_coordinates;
    double * restrict edge_coords     = D->edge_coordinates;
    anuga_int * restrict surrogate_neighbours = D->surrogate_neighbours;
    anuga_int * restrict number_of_boundaries = D->number_of_boundaries;
    double * restrict x_centroid_work = D->x_centroid_work;
    double * restrict y_centroid_work = D->y_centroid_work;

    // Active cell support: if D->active_cell_ids is set, iterate only over those
    int    n_active    = D->n_active_cells;
    int  * active_ids  = D->active_cell_ids;
    int    use_active  = (active_ids != NULL) && (n_active > 0);

    // -- Pass 1a: centroid updates (height, dry zeroing, momentum backup) --
    if (use_active) {
        OMP_PARALLEL_LOOP
        for (int ai = 0; ai < n_active; ai++) {
            anuga_int k = active_ids[ai];
            double stage = stage_cv[k];
            double bed   = bed_cv[k];
            double xmom  = xmom_cv[k];
            double ymom  = ymom_cv[k];
            double dk    = fmax(stage - bed, 0.0);
            height_cv[k] = dk;
            int is_dry   = (dk <= minimum_allowed_height);
            double xmom_out = is_dry ? 0.0 : xmom;
            double ymom_out = is_dry ? 0.0 : ymom;
            x_centroid_work[k] = xmom_out;
            y_centroid_work[k] = ymom_out;
            xmom_cv[k] = xmom_out;
            ymom_cv[k] = ymom_out;
        }
    } else {
        OMP_PARALLEL_LOOP
        for (anuga_int k = 0; k < n; k++) {
            double stage = stage_cv[k];
            double bed   = bed_cv[k];
            double xmom  = xmom_cv[k];
            double ymom  = ymom_cv[k];
            double dk    = fmax(stage - bed, 0.0);
            height_cv[k] = dk;
            int is_dry   = (dk <= minimum_allowed_height);
            double xmom_out = is_dry ? 0.0 : xmom;
            double ymom_out = is_dry ? 0.0 : ymom;
            x_centroid_work[k] = xmom_out;
            y_centroid_work[k] = ymom_out;
            xmom_cv[k] = xmom_out;
            ymom_cv[k] = ymom_out;
        }
    }

    // -- Pass 1b: gradient reconstruction + edge value writes --
    {
        anuga_int n_iter  = use_active ? (anuga_int)n_active : n;
        int       indexed = use_active;

        OMP_PARALLEL_LOOP
        for (anuga_int ai = 0; ai < n_iter; ai++) {
            anuga_int k  = indexed ? (anuga_int)active_ids[ai] : ai;
            anuga_int k2 = k * 2, k3 = k * 3, k6 = k * 6;

            double xv0 = edge_coords[k6+0], yv0 = edge_coords[k6+1];
            double xv1 = edge_coords[k6+2], yv1 = edge_coords[k6+3];
            double xv2 = edge_coords[k6+4], yv2 = edge_coords[k6+5];
            double x   = centroid_coords[k2+0];
            double y   = centroid_coords[k2+1];
            double dxv0 = xv0-x, dxv1 = xv1-x, dxv2 = xv2-x;
            double dyv0 = yv0-y, dyv1 = yv1-y, dyv2 = yv2-y;

            anuga_int sn0 = surrogate_neighbours[k3+0];
            anuga_int sn1 = surrogate_neighbours[k3+1];
            anuga_int sn2 = surrogate_neighbours[k3+2];
            double x0 = centroid_coords[2*sn0+0], y0 = centroid_coords[2*sn0+1];
            double x1 = centroid_coords[2*sn1+0], y1 = centroid_coords[2*sn1+1];
            double x2 = centroid_coords[2*sn2+0], y2 = centroid_coords[2*sn2+1];
            double dx1 = x1-x0, dx2 = x2-x0, dy1 = y1-y0, dy2 = y2-y0;
            double area2 = dy2*dx1 - dy1*dx2;

            int dry = ((height_cv[sn0] < minimum_allowed_height) || (sn0 == k)) &&
                      ((height_cv[sn1] < minimum_allowed_height) || (sn1 == k)) &&
                      ((height_cv[sn2] < minimum_allowed_height) || (sn2 == k));
            if (dry) {
                x_centroid_work[k] = 0.0; xmom_cv[k] = 0.0;
                y_centroid_work[k] = 0.0; ymom_cv[k] = 0.0;
            }

            double xmom_q  = xmom_cv[k],   ymom_q  = ymom_cv[k];
            double xmom_q0 = xmom_cv[sn0], ymom_q0 = ymom_cv[sn0];
            double xmom_q1 = xmom_cv[sn1], ymom_q1 = ymom_cv[sn1];
            double xmom_q2 = xmom_cv[sn2], ymom_q2 = ymom_cv[sn2];

            if (extrapolate_velocity_second_order == 1) {
                double hk  = height_cv[k];
                double h0q = height_cv[sn0];
                double h1q = height_cv[sn1];
                double h2q = height_cv[sn2];
                xmom_q  = (hk  > minimum_allowed_height) ? (x_centroid_work[k]   / hk)  : 0.0;
                ymom_q  = (hk  > minimum_allowed_height) ? (y_centroid_work[k]   / hk)  : 0.0;
                xmom_q0 = (h0q > minimum_allowed_height) ? (x_centroid_work[sn0] / h0q) : 0.0;
                ymom_q0 = (h0q > minimum_allowed_height) ? (y_centroid_work[sn0] / h0q) : 0.0;
                xmom_q1 = (h1q > minimum_allowed_height) ? (x_centroid_work[sn1] / h1q) : 0.0;
                ymom_q1 = (h1q > minimum_allowed_height) ? (y_centroid_work[sn1] / h1q) : 0.0;
                xmom_q2 = (h2q > minimum_allowed_height) ? (x_centroid_work[sn2] / h2q) : 0.0;
                ymom_q2 = (h2q > minimum_allowed_height) ? (y_centroid_work[sn2] / h2q) : 0.0;
            }

            int num_boundaries = number_of_boundaries[k];
            if (num_boundaries == 3) {
                double sc = stage_cv[k], xc = xmom_q, yc = ymom_q;
                double hc = height_cv[k], bc = bed_cv[k];
                for (int i = 0; i < 3; i++) {
                    stage_ev[k3+i] = sc; xmom_ev[k3+i] = xc; ymom_ev[k3+i] = yc;
                    height_ev[k3+i] = hc; bed_ev[k3+i] = bc;
                }
            } else if (num_boundaries <= 1) {
                double hc = height_cv[k];
                double h0 = height_cv[sn0], h1 = height_cv[sn1], h2 = height_cv[sn2];
                double hmin = fmin(fmin(h0, fmin(h1, h2)), hc);
                double hmax = fmax(fmax(h0, fmax(h1, h2)), hc);
                double tmp1 = c_tmp * fmax(hmin, 0.0) / fmax(hc, 1.0e-06) + d_tmp;
                double tmp2 = c_tmp * fmax(hc,   0.0) / fmax(hmax, 1.0e-06) + d_tmp;
                double hfactor = fmax(0.0, fmin(tmp1, fmin(tmp2, 1.0)));
                hfactor = fmin(1.2 * fmax(hmin - minimum_allowed_height, 0.0) /
                               (fmax(hmin, 0.0) + minimum_allowed_height), hfactor);
                double inv_area2 = 1.0 / area2;
                double edge_vals[3];

                double beta_stage = beta_w_dry + (beta_w - beta_w_dry) * hfactor;
                if (beta_stage > 0.0)
                    gpu_calc_edge_values_with_gradient(stage_cv[k], stage_cv[sn0], stage_cv[sn1], stage_cv[sn2],
                        dxv0,dxv1,dxv2,dyv0,dyv1,dyv2,dx1,dx2,dy1,dy2,inv_area2,beta_stage,edge_vals);
                else gpu_set_constant_edge_values(stage_cv[k], edge_vals);
                stage_ev[k3+0]=edge_vals[0]; stage_ev[k3+1]=edge_vals[1]; stage_ev[k3+2]=edge_vals[2];

                if (beta_stage > 0.0)
                    gpu_calc_edge_values_with_gradient(height_cv[k], height_cv[sn0], height_cv[sn1], height_cv[sn2],
                        dxv0,dxv1,dxv2,dyv0,dyv1,dyv2,dx1,dx2,dy1,dy2,inv_area2,beta_stage,edge_vals);
                else gpu_set_constant_edge_values(height_cv[k], edge_vals);
                height_ev[k3+0]=edge_vals[0]; height_ev[k3+1]=edge_vals[1]; height_ev[k3+2]=edge_vals[2];

                double beta_xmom = beta_uh_dry + (beta_uh - beta_uh_dry) * hfactor;
                if (beta_xmom > 0.0)
                    gpu_calc_edge_values_with_gradient(xmom_q, xmom_q0, xmom_q1, xmom_q2,
                        dxv0,dxv1,dxv2,dyv0,dyv1,dyv2,dx1,dx2,dy1,dy2,inv_area2,beta_xmom,edge_vals);
                else gpu_set_constant_edge_values(xmom_q, edge_vals);
                xmom_ev[k3+0]=edge_vals[0]; xmom_ev[k3+1]=edge_vals[1]; xmom_ev[k3+2]=edge_vals[2];

                double beta_ymom = beta_vh_dry + (beta_vh - beta_vh_dry) * hfactor;
                if (beta_ymom > 0.0)
                    gpu_calc_edge_values_with_gradient(ymom_q, ymom_q0, ymom_q1, ymom_q2,
                        dxv0,dxv1,dxv2,dyv0,dyv1,dyv2,dx1,dx2,dy1,dy2,inv_area2,beta_ymom,edge_vals);
                else gpu_set_constant_edge_values(ymom_q, edge_vals);
                ymom_ev[k3+0]=edge_vals[0]; ymom_ev[k3+1]=edge_vals[1]; ymom_ev[k3+2]=edge_vals[2];

                if (extrapolate_velocity_second_order == 1) {
                    for (int i = 0; i < 3; i++) {
                        double dk = height_ev[k3+i];
                        xmom_ev[k3+i] *= dk;
                        ymom_ev[k3+i] *= dk;
                    }
                }
            } else {
                // num_boundaries == 2
                anuga_int kn = k;
                for (int i = 0; i < 3; i++) {
                    anuga_int sn = surrogate_neighbours[k3+i];
                    if (sn != k) { kn = sn; break; }
                }
                double xn = centroid_coords[2*kn+0], yn = centroid_coords[2*kn+1];
                double dx = xn-x, dy = yn-y, dist2 = dx*dx + dy*dy;
                double grad_dx2 = (dist2 > 0.0) ? dx/dist2 : 0.0;
                double grad_dy2 = (dist2 > 0.0) ? dy/dist2 : 0.0;
                double dqv[3], qmin, qmax, dq1;

                dq1 = stage_cv[kn] - stage_cv[k];
                gpu_compute_dqv_from_gradient(dq1,grad_dx2,grad_dy2,dxv0,dxv1,dxv2,dyv0,dyv1,dyv2,dqv);
                gpu_compute_qmin_qmax_from_dq1(dq1,&qmin,&qmax);
                gpu_limit_gradient(dqv,qmin,qmax,beta_w);
                stage_ev[k3+0]=stage_cv[k]+dqv[0]; stage_ev[k3+1]=stage_cv[k]+dqv[1]; stage_ev[k3+2]=stage_cv[k]+dqv[2];

                dq1 = height_cv[kn] - height_cv[k];
                gpu_compute_dqv_from_gradient(dq1,grad_dx2,grad_dy2,dxv0,dxv1,dxv2,dyv0,dyv1,dyv2,dqv);
                gpu_compute_qmin_qmax_from_dq1(dq1,&qmin,&qmax);
                gpu_limit_gradient(dqv,qmin,qmax,beta_w);
                height_ev[k3+0]=height_cv[k]+dqv[0]; height_ev[k3+1]=height_cv[k]+dqv[1]; height_ev[k3+2]=height_cv[k]+dqv[2];

                double xmom_kn = xmom_cv[kn], ymom_kn = ymom_cv[kn];
                if (extrapolate_velocity_second_order == 1) {
                    double hkn = height_cv[kn];
                    xmom_kn = (hkn > minimum_allowed_height) ? (x_centroid_work[kn] / hkn) : 0.0;
                    ymom_kn = (hkn > minimum_allowed_height) ? (y_centroid_work[kn] / hkn) : 0.0;
                }
                dq1 = xmom_kn - xmom_q;
                gpu_compute_dqv_from_gradient(dq1,grad_dx2,grad_dy2,dxv0,dxv1,dxv2,dyv0,dyv1,dyv2,dqv);
                gpu_compute_qmin_qmax_from_dq1(dq1,&qmin,&qmax);
                gpu_limit_gradient(dqv,qmin,qmax,beta_w);
                xmom_ev[k3+0]=xmom_q+dqv[0]; xmom_ev[k3+1]=xmom_q+dqv[1]; xmom_ev[k3+2]=xmom_q+dqv[2];

                dq1 = ymom_kn - ymom_q;
                gpu_compute_dqv_from_gradient(dq1,grad_dx2,grad_dy2,dxv0,dxv1,dxv2,dyv0,dyv1,dyv2,dqv);
                gpu_compute_qmin_qmax_from_dq1(dq1,&qmin,&qmax);
                gpu_limit_gradient(dqv,qmin,qmax,beta_w);
                ymom_ev[k3+0]=ymom_q+dqv[0]; ymom_ev[k3+1]=ymom_q+dqv[1]; ymom_ev[k3+2]=ymom_q+dqv[2];

                if (extrapolate_velocity_second_order == 1) {
                    for (int i = 0; i < 3; i++) {
                        double dk = height_ev[k3+i];
                        xmom_ev[k3+i] *= dk;
                        ymom_ev[k3+i] *= dk;
                    }
                }
            }

            for (int i = 0; i < 3; i++)
                bed_ev[k3+i] = stage_ev[k3+i] - height_ev[k3+i];
        }
    }

    // ----------------------------------------------------------------
    // Pass 2: Flux computation — reads L2-hot edge arrays from Pass 1.
    // Identical to core_compute_fluxes_central_substep but fused here
    // so GPU warps can overlap the tail of Pass 1 with the start of Pass 2.
    // ----------------------------------------------------------------
    return core_compute_fluxes_central_substep(D, substep_count, timestep_fluxcalls,
                                               compute_timestep, compute_boundary_flux);
}
