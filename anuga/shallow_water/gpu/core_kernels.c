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

void core_extrapolate_centroid_pass(struct domain *D) {
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
    //
    // x/y_centroid_work carry the *velocity* the limiter reconstructs from;
    // xmom/ymom_cv keep the momentum, so no restore pass is needed afterwards.
    // (Historically this was the other way round -- the work arrays saved the
    // momentum while _cv held the velocity -- which cost a third kernel launch
    // over every cell just to swap them back. See the note above Step 3.)
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

        x_centroid_work[k] = xmom_out * inv_dk;
        y_centroid_work[k] = ymom_out * inv_dk;

        xmom_cv[k] = xmom_out;
        ymom_cv[k] = ymom_out;
    }

}

// The edge pass: the second-order reconstruction proper.  Reads the
// neighbours' centroid values (via surrogate_neighbours), so it MUST be a
// separate kernel launch from anything that writes centroid values -- there is
// no device-wide barrier inside an `omp target teams loop`.
//
// predictor_dt: 0.0 for a plain reconstruction (RK2 and the ADER-2 bootstrap
// step).  Non-zero fuses the ADER-2 Cauchy-Kovalewski edge predictor into the
// tail of the cell body: the just-reconstructed edge values are shifted to
// Q^{n + predictor_dt} in the same launch, reusing the dxv/dyv edge offsets
// already in registers.  The standalone core_ader_ck_predictor_edge() kernel
// computes the same arithmetic from arrays; fusing it here removes that
// kernel's full read+write sweep over the edge arrays (~200 B/cell), and --
// because the predictor never reads boundary_values -- lets the ADER-2 step
// evaluate boundaries ONCE, after the shift, instead of before and after.
void core_extrapolate_edge_pass(struct domain *D, double predictor_dt) {
    anuga_int n = D->number_of_elements;
    double minimum_allowed_height = D->minimum_allowed_height;
    anuga_int extrapolate_velocity_second_order = D->extrapolate_velocity_second_order;
    double g_pred = D->g;                       // used by the predictor tail only

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
            double xmom_c = x_centroid_work[k];
            double ymom_c = y_centroid_work[k];
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
                    x_centroid_work[k], x_centroid_work[k0], x_centroid_work[k1], x_centroid_work[sn2],
                    dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                    dx1, dx2, dy1, dy2, inv_area2, beta_xmom, edge_vals);
            } else {
                gpu_set_constant_edge_values(x_centroid_work[k], edge_vals);
            }
            xmom_ev[k3 + 0] = edge_vals[0];
            xmom_ev[k3 + 1] = edge_vals[1];
            xmom_ev[k3 + 2] = edge_vals[2];

            // Y-momentum
            double beta_ymom = beta_vh_dry + (beta_vh - beta_vh_dry) * hfactor;
            if (beta_ymom > 0.0) {
                gpu_calc_edge_values_with_gradient(
                    y_centroid_work[k], y_centroid_work[k0], y_centroid_work[k1], y_centroid_work[sn2],
                    dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                    dx1, dx2, dy1, dy2, inv_area2, beta_ymom, edge_vals);
            } else {
                gpu_set_constant_edge_values(y_centroid_work[k], edge_vals);
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
            dq1 = x_centroid_work[kn] - x_centroid_work[k];
            gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                          dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
            gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
            gpu_limit_gradient(dqv, qmin, qmax, beta_w);
            xmom_ev[k3 + 0] = x_centroid_work[k] + dqv[0];
            xmom_ev[k3 + 1] = x_centroid_work[k] + dqv[1];
            xmom_ev[k3 + 2] = x_centroid_work[k] + dqv[2];

            // Y-momentum
            dq1 = y_centroid_work[kn] - y_centroid_work[k];
            gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                          dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
            gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
            gpu_limit_gradient(dqv, qmin, qmax, beta_w);
            ymom_ev[k3 + 0] = y_centroid_work[k] + dqv[0];
            ymom_ev[k3 + 1] = y_centroid_work[k] + dqv[1];
            ymom_ev[k3 + 2] = y_centroid_work[k] + dqv[2];
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

        // ---- fused ADER-2 C-K edge predictor (see the function comment).
        // Identical arithmetic to core_ader_ck_predictor_edge(), with the
        // edge-offset vectors dxv0/dyv0/dxv1/dyv1 reused from the limiter
        // geometry above and the just-written edge values re-read from this
        // thread's own stores (register/L1-resident, never a remote gather).
        if (predictor_dt != 0.0) {
            double det_p = dxv0 * dyv1 - dxv1 * dyv0;
            if (fabs(det_p) >= 1.0e-20) {
                double inv_det = 1.0 / det_p;

                double w_c  = stage_cv[k];
                double h_c  = fmax(w_c - bed_cv[k], 0.0);
                double uh_c = xmom_cv[k];
                double vh_c = ymom_cv[k];

                double inv_h_c = (h_c > minimum_allowed_height) ? 1.0 / h_c : 0.0;
                double u_c = uh_c * inv_h_c;
                double v_c = vh_c * inv_h_c;

                double dw0 = stage_ev[k3 + 0] - w_c;
                double dw1 = stage_ev[k3 + 1] - w_c;
                double wx  = inv_det * (dyv1 * dw0 - dyv0 * dw1);
                double wy  = inv_det * (dxv0 * dw1 - dxv1 * dw0);

                double dh0 = height_ev[k3 + 0] - h_c;
                double dh1 = height_ev[k3 + 1] - h_c;
                double hx  = inv_det * (dyv1 * dh0 - dyv0 * dh1);
                double hy  = inv_det * (dxv0 * dh1 - dxv1 * dh0);

                double h_e0     = height_ev[k3 + 0];
                double h_e1     = height_ev[k3 + 1];
                double inv_h_e0 = (h_e0 > minimum_allowed_height) ? 1.0 / h_e0 : 0.0;
                double inv_h_e1 = (h_e1 > minimum_allowed_height) ? 1.0 / h_e1 : 0.0;
                double u_e0 = xmom_ev[k3 + 0] * inv_h_e0;
                double u_e1 = xmom_ev[k3 + 1] * inv_h_e1;
                double v_e0 = ymom_ev[k3 + 0] * inv_h_e0;
                double v_e1 = ymom_ev[k3 + 1] * inv_h_e1;

                double du0 = u_e0 - u_c;
                double du1 = u_e1 - u_c;
                double dv0 = v_e0 - v_c;
                double dv1 = v_e1 - v_c;
                double ux  = inv_det * (dyv1 * du0 - dyv0 * du1);
                double uy  = inv_det * (dxv0 * du1 - dxv1 * du0);
                double vx  = inv_det * (dyv1 * dv0 - dyv0 * dv1);
                double vy  = inv_det * (dxv0 * dv1 - dxv1 * dv0);

                double g_h = g_pred * h_c;
                double dw_dt  = -(u_c * hx + h_c * ux + v_c * hy + h_c * vy);
                double duh_dt = -(2.0*u_c*h_c*ux + u_c*u_c*hx + u_c*v_c*hy
                                 + v_c*h_c*uy + u_c*h_c*vy + g_h * wx);
                double dvh_dt = -(v_c*h_c*ux + u_c*h_c*vx + u_c*v_c*hx
                                 + 2.0*v_c*h_c*vy + v_c*v_c*hy + g_h * wy);

                // NOTE: bed_ev is NOT refreshed after the shift, exactly like
                // the standalone predictor: stage and height shift by the same
                // dw_dt, so stage - height still equals the true bed everywhere
                // except clamped near-dry edges -- and the pre-shift bed_ev
                // (the true bed) is what the boundary kernels should read.
                for (int i = 0; i < 3; i++) {
                    stage_ev[k3 + i] += predictor_dt * dw_dt;
                    xmom_ev[k3 + i] += predictor_dt * duh_dt;
                    ymom_ev[k3 + i] += predictor_dt * dvh_dt;
                    height_ev[k3 + i] = fmax(height_ev[k3 + i] + predictor_dt * dw_dt, 0.0);
                }
            }
        }

    }

}

void core_extrapolate_second_order_edge(struct domain *D) {
    // Kept as the two passes below so callers that can fuse the (cell-local)
    // centroid pass into a neighbouring kernel may call the edge pass alone.
    core_extrapolate_centroid_pass(D);
    core_extrapolate_edge_pass(D, 0.0);
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

        // Apply explicit updates
        double stage_new = stage_c + timestep * stage_eu[k];
        double xmom_new  = xmom_c  + timestep * xmom_eu[k];
        double ymom_new  = ymom_c  + timestep * ymom_eu[k];

        // Apply semi-implicit updates, reformulated to ONE division per quantity.
        // The original did two FP64 divisions per quantity (si = siu/c, then cv/denom
        // with denom = 1 - dt*si); algebraically
        //     cv / (1 - dt*siu/c)  ==  cv*c / (c - dt*siu),
        // so num = c - dt*siu = denom*c, and denom>0  <=>  num*c > 0. Halving the
        // divisions matters on GeForce GPUs, where FP64 is 1/64 rate and ncu shows
        // this kernel FP64-pipe-bound (see issue #199); mathematically identical, so
        // results differ only at floating-point roundoff.
        double num;

        num = stage_c - timestep * stage_siu[k];
        if (stage_c != 0.0 && num * stage_c > 0.0) stage_new = stage_new * stage_c / num;

        num = xmom_c - timestep * xmom_siu[k];
        if (xmom_c != 0.0 && num * xmom_c > 0.0) xmom_new = xmom_new * xmom_c / num;

        num = ymom_c - timestep * ymom_siu[k];
        if (ymom_c != 0.0 && num * ymom_c > 0.0) ymom_new = ymom_new * ymom_c / num;

        stage_cv[k] = stage_new;
        xmom_cv[k] = xmom_new;
        ymom_cv[k] = ymom_new;

        // Reset semi-implicit updates for next timestep
        stage_siu[k] = 0.0;
        xmom_siu[k] = 0.0;
        ymom_siu[k] = 0.0;
    }
}

#pragma omp declare target
// One cell's Manning friction + conserved-quantity update + optional RK2
// average, entirely in registers.  Shared by core_forcing_and_update (which
// reads the explicit updates from the eu arrays) and by the edge-based
// core_flux_apply_and_update (which computes them in registers and never
// touches the eu arrays at all).  eu_* are the explicit-update values for
// this cell; the semi-implicit arrays are read, consumed and reset here.
static inline void gpu_cell_forcing_update(
    anuga_int k, double timestep, int apply_manning, int do_saxpy,
    double a, double b, double g, double minimum_allowed_height,
    double seven_thirds,
    double eu_stage, double eu_xmom, double eu_ymom,
    double * restrict stage_cv, double * restrict xmom_cv,
    double * restrict ymom_cv, double * restrict bed_cv,
    double * restrict height_cv, double * restrict friction_cv,
    double * restrict stage_siu, double * restrict xmom_siu,
    double * restrict ymom_siu,
    double * restrict stage_bk, double * restrict xmom_bk,
    double * restrict ymom_bk) {

    double stage_c = stage_cv[k];
    double xmom_c = xmom_cv[k];
    double ymom_c = ymom_cv[k];

    double s_siu = stage_siu[k];
    double x_siu = xmom_siu[k];
    double y_siu = ymom_siu[k];

    if (apply_manning) {
        double S = 0.0;
        double eta = friction_cv[k];
        double abs_mom = sqrt(xmom_c * xmom_c + ymom_c * ymom_c);

        if (eta > 1.0e-15) {  // ETA_SMALL
            double h = stage_c - bed_cv[k];
            if (h >= minimum_allowed_height) {
                S = -g * eta * eta * abs_mom;
                S /= pow(h, seven_thirds);
            }
        }
        x_siu += S * xmom_c;
        y_siu += S * ymom_c;
    }

    // Explicit + semi-implicit update (single-division form; see
    // core_update_conserved_quantities for the derivation)
    double stage_new = stage_c + timestep * eu_stage;
    double xmom_new  = xmom_c  + timestep * eu_xmom;
    double ymom_new  = ymom_c  + timestep * eu_ymom;

    double num;

    num = stage_c - timestep * s_siu;
    if (stage_c != 0.0 && num * stage_c > 0.0) stage_new = stage_new * stage_c / num;

    num = xmom_c - timestep * x_siu;
    if (xmom_c != 0.0 && num * xmom_c > 0.0) xmom_new = xmom_new * xmom_c / num;

    num = ymom_c - timestep * y_siu;
    if (ymom_c != 0.0 && num * ymom_c > 0.0) ymom_new = ymom_new * ymom_c / num;

    stage_siu[k] = 0.0;
    xmom_siu[k] = 0.0;
    ymom_siu[k] = 0.0;

    if (do_saxpy) {
        stage_new = a * stage_new + b * stage_bk[k];
        xmom_new  = a * xmom_new  + b * xmom_bk[k];
        ymom_new  = a * ymom_new  + b * ymom_bk[k];
        height_cv[k] = fmax(stage_new - bed_cv[k], 0.0);
    }

    stage_cv[k] = stage_new;
    xmom_cv[k] = xmom_new;
    ymom_cv[k] = ymom_new;
}
#pragma omp end declare target

// ============================================================================
// Fused forcing + update (+ optional RK2 average)
//
// Manning friction, the conserved-quantity update and the RK2 average are all
// strictly cell-local: each reads and writes only index k.  Running them as
// three separate kernels means three launches and three round trips through
// the semi-implicit and centroid arrays, so they are fused here into one.
//
// This is as far as fusion goes in the DE step.  compute_fluxes cannot join
// them: it is a stencil kernel -- it reads height_cv[neighbour],
// bed_cv[neighbour], stage_cv[neighbour] and the neighbours' edge values -- so
// writing any centroid value from inside it would race against another team
// still reading that value, and an `omp target teams loop` has no device-wide
// barrier to order them.  The same argument rules out fusing extrapolate into
// compute_fluxes.
//
// Only the FLAT Manning variant is inlined here.  Callers with
// use_sloped_mannings must keep calling the sloped kernel separately (it reads
// vertex values, which the GPU path does not map).
//
//   timestep     dt to apply (must already be known -- fine for RK2 substep 2,
//                which reuses substep 1's dt)
//   apply_manning  1 => add the flat Manning friction term
//   do_saxpy     1 => finish with Q = a*Q + b*Q_backup and refresh height_cv
// ============================================================================

void core_forcing_and_update(struct domain *D, double timestep,
                             int apply_manning, int do_saxpy,
                             double a, double b) {
    anuga_int n = D->number_of_elements;
    double g = D->g;
    double minimum_allowed_height = D->minimum_allowed_height;
    double seven_thirds = 7.0 / 3.0;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict height_cv = D->height_centroid_values;
    double * restrict friction_cv = D->friction_centroid_values;

    double * restrict stage_eu = D->stage_explicit_update;
    double * restrict xmom_eu = D->xmom_explicit_update;
    double * restrict ymom_eu = D->ymom_explicit_update;

    double * restrict stage_siu = D->stage_semi_implicit_update;
    double * restrict xmom_siu = D->xmom_semi_implicit_update;
    double * restrict ymom_siu = D->ymom_semi_implicit_update;

    double * restrict stage_bk = D->stage_backup_values;
    double * restrict xmom_bk = D->xmom_backup_values;
    double * restrict ymom_bk = D->ymom_backup_values;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        gpu_cell_forcing_update(k, timestep, apply_manning, do_saxpy, a, b,
                                g, minimum_allowed_height, seven_thirds,
                                stage_eu[k], xmom_eu[k], ymom_eu[k],
                                stage_cv, xmom_cv, ymom_cv, bed_cv, height_cv,
                                friction_cv, stage_siu, xmom_siu, ymom_siu,
                                stage_bk, xmom_bk, ymom_bk);
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

    double * restrict stage_bk = D->stage_backup_values;
    double * restrict xmom_bk = D->xmom_backup_values;
    double * restrict ymom_bk = D->ymom_backup_values;

    // Standard SAXPY: Q = a*Q + b*Q_backup
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        stage_cv[k] = a * stage_cv[k] + b * stage_bk[k];
        xmom_cv[k] = a * xmom_cv[k] + b * xmom_bk[k];
        ymom_cv[k] = a * ymom_cv[k] + b * ymom_bk[k];
    }

    // Apply c scaling if needed: Q = Q / c
    // Used for numerical stability with RK coefficients like a=1/3, b=2/3
    // Skip if c=0.0 (RK2 passes 0.0) or c=1.0 (no scaling needed)
    if (c != 1.0 && c != 0.0) {
        double c_inv = 1.0 / c;
        OMP_PARALLEL_LOOP
        for (anuga_int k = 0; k < n; k++) {
            stage_cv[k] *= c_inv;
            xmom_cv[k] *= c_inv;
            ymom_cv[k] *= c_inv;
        }
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
// Fused step preparation: RK2 backup + protect + extrapolate centroid pass.
//
// All three touch only index k, so they run as ONE kernel: the centroid values
// are read once into registers, backed up, protected, and converted for the
// edge pass without three separate trips through memory.  This also retires
// the standalone protect's follow-up height refresh -- the centroid pass
// recomputes height_cv from the protected stage anyway.
//
// The sequencing inside the loop body reproduces the original kernel order
// (backup BEFORE protect -- the RK2 average must combine with the unprotected
// state, exactly as gpu_backup_conserved_quantities did) so results are
// bit-identical to the unfused sequence.
//
// Returns the protect mass error (same reduction core_protect performs).
// ============================================================================

double core_prepare_step(struct domain *D, int do_backup, int zero_eu) {
    anuga_int n = D->number_of_elements;
    double minimum_allowed_height = D->minimum_allowed_height;
    anuga_int extrapolate_velocity_second_order = D->extrapolate_velocity_second_order;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict height_cv = D->height_centroid_values;
    double * restrict areas = D->areas;
    double * restrict x_centroid_work = D->x_centroid_work;
    double * restrict y_centroid_work = D->y_centroid_work;

    double * restrict stage_bk = D->stage_backup_values;
    double * restrict xmom_bk = D->xmom_backup_values;
    double * restrict ymom_bk = D->ymom_backup_values;

    // Scatter-mode fluxes accumulate into the explicit updates with atomics,
    // so they must start the step at zero; the cell-based flux kernel
    // initializes them itself and passes zero_eu = 0.
    double * restrict stage_eu = D->stage_explicit_update;
    double * restrict xmom_eu = D->xmom_explicit_update;
    double * restrict ymom_eu = D->ymom_explicit_update;

    double mass_error = 0.0;

    OMP_PARALLEL_LOOP_REDUCTION_PLUS(mass_error)
    for (anuga_int k = 0; k < n; k++) {
        double stage = stage_cv[k];
        double bed = bed_cv[k];
        double xmom = xmom_cv[k];
        double ymom = ymom_cv[k];

        if (zero_eu) {
            stage_eu[k] = 0.0;
            xmom_eu[k] = 0.0;
            ymom_eu[k] = 0.0;
        }

        // RK2 backup of the raw (pre-protect) state
        if (do_backup) {
            stage_bk[k] = stage;
            xmom_bk[k] = xmom;
            ymom_bk[k] = ymom;
        }

        // Protect (core_protect's logic, in registers)
        double h = stage - bed;
        if (h < minimum_allowed_height) {
            xmom = 0.0;
            ymom = 0.0;
        }
        if (h < 0.0) {
            mass_error += (-h) * areas[k];
            stage = bed;
        }
        stage_cv[k] = stage;

        // Extrapolate centroid pass (velocity into the work arrays)
        double dk = fmax(stage - bed, 0.0);
        height_cv[k] = dk;

        int is_dry = (dk <= minimum_allowed_height);
        int extrapolate = (extrapolate_velocity_second_order == 1) && (dk > minimum_allowed_height);

        double xmom_out = is_dry ? 0.0 : xmom;
        double ymom_out = is_dry ? 0.0 : ymom;

        double inv_dk = extrapolate ? (1.0 / dk) : 1.0;

        x_centroid_work[k] = xmom_out * inv_dk;
        y_centroid_work[k] = ymom_out * inv_dk;

        xmom_cv[k] = xmom_out;
        ymom_cv[k] = ymom_out;
    }

    return mass_error;
}

// ============================================================================
// Fix negative cells
//
// Matches _openmp_fix_negative_cells (the tested CPU reference):
//   - Only acts on cells where stage - bed < 0  AND  tri_full_flag > 0
//     (ghost cells are skipped, matching the openmp & bitwise-and condition)
//   - Zeros xmom/ymom and resets stage to bed for those cells
//   - Returns count of cells fixed (parallel + reduction)
//
// NOTE: The original core version (before unification) used a different
// threshold (minimum_allowed_height) and ignored tri_full_flag — it has
// been updated here to match the _openmp_ reference behaviour exactly.
// ============================================================================

int core_fix_negative_cells(struct domain *D) {
    anuga_int n = D->number_of_elements;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv  = D->xmom_centroid_values;
    double * restrict ymom_cv  = D->ymom_centroid_values;
    double * restrict bed_cv   = D->bed_centroid_values;
    anuga_int * restrict tri_full_flag = D->tri_full_flag;

    int num_negative_cells = 0;

    OMP_PARALLEL_LOOP_REDUCTION_PLUS(num_negative_cells)
    for (anuga_int k = 0; k < n; k++) {
        // Use & (bitwise and) matching the original _openmp_ condition.
        // tri_full_flag is always initialised to ones(N) so the pointer is
        // never NULL when called from Cython; the check avoids UB for the
        // standalone / GPU build path where it could theoretically be NULL.
        int full = (tri_full_flag == NULL) ? 1 : (tri_full_flag[k] > 0);
        if ((stage_cv[k] - bed_cv[k] < 0.0) & full) {
            num_negative_cells = num_negative_cells + 1;
            stage_cv[k] = bed_cv[k];
            xmom_cv[k]  = 0.0;
            ymom_cv[k]  = 0.0;
        }
    }

    return num_negative_cells;
}

// ============================================================================
// Negative-cell volume (read-only)
//
// Measures the water volume that fix_negative_cells will ADD by clamping
// negative-depth cells up to zero depth (stage = bed) — i.e. the conservation
// error the clamp introduces this step. Uses the SAME cell selection as
// core_fix_negative_cells (stage - bed < 0 AND tri_full_flag > 0), so it must
// be called AFTER the flux update but BEFORE core_fix_negative_cells (which
// erases the deficit). Does not modify the domain.
// ============================================================================

double core_negative_cells_volume(struct domain *D) {
    anuga_int n = D->number_of_elements;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict bed_cv   = D->bed_centroid_values;
    double * restrict areas    = D->areas;
    anuga_int * restrict tri_full_flag = D->tri_full_flag;

    double volume = 0.0;

    OMP_PARALLEL_LOOP_REDUCTION_PLUS(volume)
    for (anuga_int k = 0; k < n; k++) {
        int full = (tri_full_flag == NULL) ? 1 : (tri_full_flag[k] > 0);
        if ((stage_cv[k] - bed_cv[k] < 0.0) & full) {
            // bed - stage > 0 here: volume needed to raise the cell to zero depth
            volume = volume + (bed_cv[k] - stage_cv[k]) * areas[k];
        }
    }

    return volume;
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
// Manning friction (sloped, semi-implicit, edge-based)
//
// Like core_manning_friction_sloped_semi_implicit but derives the bed slope
// from edge values (bed_edge_values) instead of vertex values.  This is the
// active per-timestep path when domain.use_sloped_mannings=True
// (friction.py selects manning_friction_sloped_semi_implicit_edge_based).
// ============================================================================

void core_manning_friction_sloped_semi_implicit_edge_based(struct domain *D) {
    anuga_int n = D->number_of_elements;
    double g   = D->g;
    double eps = D->minimum_allowed_height;

    double * restrict stage_cv   = D->stage_centroid_values;
    double * restrict bed_ev     = D->bed_edge_values;
    double * restrict xmom_cv    = D->xmom_centroid_values;
    double * restrict ymom_cv    = D->ymom_centroid_values;
    double * restrict friction_cv = D->friction_centroid_values;
    double * restrict edge_coords = D->edge_coordinates;

    double * restrict xmom_siu   = D->xmom_semi_implicit_update;
    double * restrict ymom_siu   = D->ymom_semi_implicit_update;

    const double one_third   = 1.0 / 3.0;
    const double seven_thirds = 7.0 / 3.0;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        double S = 0.0;
        double eta = friction_cv[k];

        if (eta > 1.0e-16) {
            anuga_int k3 = k * 3;
            anuga_int k6 = k * 6;

            // Bed values at edges
            double z0 = bed_ev[k3 + 0];
            double z1 = bed_ev[k3 + 1];
            double z2 = bed_ev[k3 + 2];

            // Edge midpoint coordinates
            double x0 = edge_coords[k6 + 0];
            double y0 = edge_coords[k6 + 1];
            double x1 = edge_coords[k6 + 2];
            double y1 = edge_coords[k6 + 3];
            double x2 = edge_coords[k6 + 4];
            double y2 = edge_coords[k6 + 5];

            // Bed slope via 2x2 determinant (same as _gradient(), inlined for GPU)
            double det = (y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0);
            double zx  = ((y2 - y0) * (z1 - z0) - (y1 - y0) * (z2 - z0)) / det;
            double zy  = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / det;

            double zs = sqrt(1.0 + zx * zx + zy * zy);
            double z  = (z0 + z1 + z2) * one_third;

            double w  = stage_cv[k];
            double h  = w - z;

            if (h >= eps) {
                double uh = xmom_cv[k];
                double vh = ymom_cv[k];
                S = -g * eta * eta * zs * sqrt(uh * uh + vh * vh);
                S /= pow(h, seven_thirds);
            }
        }

        xmom_siu[k] += S * xmom_cv[k];
        ymom_siu[k] += S * ymom_cv[k];
    }
}

// ============================================================================
// Gravity term
//
// Computes bed-slope gravity source term: duh/dt += -g * avg_h * dz/dx
// Uses stage_centroid - bed_centroid for avg_h (matches the original
// _openmp_gravity which computed this directly, so height need not be
// up-to-date when this function is called).
// ============================================================================

int core_gravity(struct domain *D) {
    anuga_int n = D->number_of_elements;
    double g = D->g;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict bed_cv   = D->bed_centroid_values;
    double * restrict bed_vv   = D->bed_vertex_values;

    double * restrict xmom_eu = D->xmom_explicit_update;
    double * restrict ymom_eu = D->ymom_explicit_update;

    double * restrict vertex_coords = D->vertex_coordinates;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        // Average depth: use live stage - bed (height_cv may be stale)
        double avg_h = stage_cv[k] - bed_cv[k];

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

        // Bed gradient via 2x2 determinant (same as _gradient(), inlined for GPU)
        double det = (y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0);
        double dzx = ((y2 - y0) * (z1 - z0) - (y1 - y0) * (z2 - z0)) / det;
        double dzy = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / det;

        xmom_eu[k] += -g * avg_h * dzx;
        ymom_eu[k] += -g * avg_h * dzy;
    }

    return 0;
}

// ============================================================================
// Gravity term (well-balanced)
//
// Well-balanced formulation after Audusse et al. (2004):
//   du/dt += -g * wx * avg_h                    (stage-gradient term)
//   dv/dt += -g * wy * avg_h
//   PLUS side-pressure correction:
//     sum_i  -0.5 * g * h_i^2 * edgelength_i * n_i / area
// where h_i = stage_edge[i] - bed_edge[i] is the depth at edge i,
// and wx, wy is the gradient of stage (not bed), computed from vertex values.
//
// This formulation is exactly what _openmp_gravity_wb computed.
// Still-water equilibrium (u=v=0, stage=const) is preserved exactly
// because the stage-gradient term and edge-pressure terms cancel.
// ============================================================================

int core_gravity_wb(struct domain *D) {
    anuga_int n = D->number_of_elements;
    double g = D->g;

    double * restrict stage_vv  = D->stage_vertex_values;
    double * restrict stage_cv  = D->stage_centroid_values;
    double * restrict bed_cv    = D->bed_centroid_values;
    double * restrict stage_ev  = D->stage_edge_values;
    double * restrict bed_ev    = D->bed_edge_values;
    double * restrict normals   = D->normals;
    double * restrict edgelengths = D->edgelengths;
    double * restrict areas     = D->areas;
    double * restrict xmom_eu   = D->xmom_explicit_update;
    double * restrict ymom_eu   = D->ymom_explicit_update;
    double * restrict vertex_coords = D->vertex_coordinates;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        anuga_int k3 = k * 3;
        anuga_int k6 = k * 6;

        // --------------------------------------------------
        // Stage-gradient term: -g * avg_h * (wx, wy)
        // --------------------------------------------------

        // Stage at vertices for gradient calculation
        double w0 = stage_vv[k3 + 0];
        double w1 = stage_vv[k3 + 1];
        double w2 = stage_vv[k3 + 2];

        // Vertex coordinates
        double x0 = vertex_coords[k6 + 0];
        double y0 = vertex_coords[k6 + 1];
        double x1 = vertex_coords[k6 + 2];
        double y1 = vertex_coords[k6 + 3];
        double x2 = vertex_coords[k6 + 4];
        double y2 = vertex_coords[k6 + 5];

        // Compute stage gradient using standard 2x2 determinant formula
        // (identical math to _gradient() in util_ext.h, inlined for GPU compat)
        double det = (y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0);
        double wx  = ((y2 - y0) * (w1 - w0) - (y1 - y0) * (w2 - w0)) / det;
        double wy  = ((x1 - x0) * (w2 - w0) - (x2 - x0) * (w1 - w0)) / det;

        // Centroid depth
        double avg_h = stage_cv[k] - bed_cv[k];

        // Apply stage-gradient term
        xmom_eu[k] += -g * wx * avg_h;
        ymom_eu[k] += -g * wy * avg_h;

        // --------------------------------------------------
        // Edge-pressure (side) correction:
        //   sum_i  -0.5 * g * h_i^2 * edgelength_i * n_i / area
        // --------------------------------------------------
        double sidex = 0.0;
        double sidey = 0.0;
        for (int i = 0; i < 3; i++) {
            double h_edge = stage_ev[k3 + i] - bed_ev[k3 + i];
            double fact   = -0.5 * g * h_edge * h_edge * edgelengths[k3 + i];
            sidex += fact * normals[k6 + 2 * i];
            sidey += fact * normals[k6 + 2 * i + 1];
        }

        double inv_area = 1.0 / areas[k];
        xmom_eu[k] += -sidex * inv_area;
        ymom_eu[k] += -sidey * inv_area;
    }

    return 0;
}

// ============================================================================
// Compute fluxes using central upwind scheme (UNIFIED CPU/GPU)
// ============================================================================

double core_compute_fluxes_central(struct domain *D, int substep_count, int timestep_fluxcalls) {
    anuga_int n = D->number_of_elements;
    double g = D->g;
    double epsilon = D->epsilon;
    anuga_int low_froude = D->low_froude;

    // Extract array pointers
    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict height_cv = D->height_centroid_values;

    double * restrict stage_ev = D->stage_edge_values;
    double * restrict xmom_ev = D->xmom_edge_values;
    double * restrict ymom_ev = D->ymom_edge_values;
    double * restrict height_ev = D->height_edge_values;
    double * restrict bed_ev    = D->bed_edge_values;

    // Opt-in: reconstruct edge bed values as stage - height instead of loading
    // bed_ev.  core_extrapolate_edge_pass computes bed_ev with EXACTLY that
    // expression from exactly these arrays, so whenever fluxes follow an
    // extrapolate (every evolve step) the reconstruction is bit-identical and
    // this memory-bound kernel drops one gather per edge -- on both sides,
    // 6 scattered loads per cell.  It is wrong for callers that set edge
    // values independently and invoke fluxes directly (test_flux does), so it
    // stays off unless the driver guarantees the extrapolate-first contract
    // (D->reconstruct_edge_bed = 1; ANUGA leaves it 0).
    const int reconstruct_z = (D->reconstruct_edge_bed != 0);

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

    // Reduction variables
    double local_timestep = 1.0e+100;
    double boundary_flux_sum_substep = 0.0;

    // Main flux computation loop with reductions
    #ifdef CPU_ONLY_MODE
    #pragma omp parallel for reduction(min:local_timestep) reduction(+:boundary_flux_sum_substep)
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

            // Left state (this element's edge values); see reconstruct_z above
            ql[0] = stage_ev[ki];
            ql[1] = xmom_ev[ki];
            ql[2] = ymom_ev[ki];
            double hle = height_ev[ki];
            double zl = reconstruct_z ? (ql[0] - hle) : bed_ev[ki];

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
                hre = height_ev[nm];
                zr = reconstruct_z ? (qr[0] - hre) : bed_ev[nm];
                hc_n = height_cv[neighbour];
                zc_n = bed_cv[neighbour];
            }

            // Compute z_half (max bed elevation at edge)
            double z_half = fmax(zl, zr);

            // Check for riverwall elevation override
            int is_riverwall = 0;
            double zwall = 0.0;
            if (n_riverwall_edges > 0 && edge_flux_type != NULL &&
                edge_river_wall_counter != NULL && riverwall_elevation != NULL &&
                edge_flux_type[ki] == 1) {
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
            if (tri_full_flag != NULL) {
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
        if (substep_count == 0) {
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
    if (D->boundary_flux_sum != NULL && substep_count < timestep_fluxcalls) {
        D->boundary_flux_sum[substep_count] = boundary_flux_sum_substep;
    }

    // Return timestep (only meaningful on first substep)
    return local_timestep;
}

// ============================================================================
// Edge-based flux computation (opt-in, two kernels)
//
// The cell-based kernel above solves every interior edge's Riemann problem
// TWICE -- once from each side, with swapped inputs and a flipped normal.
// The central-upwind flux is antisymmetric under that swap and its shared
// scalars (pressure_flux, max wave speed, z_half) are swap-invariant, so a
// single owner-side evaluation serves both cells: the same discretization,
// half the Riemann solves, and an EXACTLY antisymmetric flux exchange (the
// dual evaluation is only antisymmetric to floating-point roundoff).
//
// Kernel A (core_compute_fluxes_edge_based) runs one thread per cell-edge
// slot and computes only the slots it owns (boundary edges, or the side
// whose cell index is larger), storing per-slot
//     [F0, F1, F2, pf_len, z_half, speed]      (stride EDGE_SLOT_STRIDE)
// in D->edge_flux_work, where F* = -length * edgeflux (owner's sign) and
// pf_len = length * pressure_flux.  It also performs the min-dt and
// boundary-flux reductions the cell-based kernel does.
//
// Kernel B (core_flux_apply_and_update) is CELL-LOCAL: it gathers the three
// slot records (own sign for owned slots, negated for the neighbour's),
// assembles the one-sided pressure-gradient terms, normalizes by area, and
// -- because it is cell-local -- finishes the whole step in the same launch
// via gpu_cell_forcing_update (Manning + update + optional RK2 average).
// The explicit-update arrays are never written on this path: the values
// live and die in registers.
//
// Opt-in and restrictions: active only when the driver allocates
// D->edge_flux_work (EDGE_SLOT_STRIDE * 3n doubles; ANUGA leaves it NULL) --
// and, like reconstruct_edge_bed, it assumes fluxes follow an extrapolate,
// reconstructing bed values as stage - height.  Riverwalls are NOT
// supported (their weir corrections are one-sided); callers must fall back
// to the cell-based kernel when riverwall edges exist.
// ============================================================================

#define EDGE_SLOT_STRIDE 6

double core_compute_fluxes_edge_based(struct domain *D, int substep_count,
                                      int timestep_fluxcalls) {
    anuga_int n = D->number_of_elements;
    double g = D->g;
    double epsilon = D->epsilon;
    anuga_int low_froude = D->low_froude;

    double * restrict stage_ev = D->stage_edge_values;
    double * restrict xmom_ev = D->xmom_edge_values;
    double * restrict ymom_ev = D->ymom_edge_values;
    double * restrict height_ev = D->height_edge_values;

    double * restrict stage_bv = D->stage_boundary_values;
    double * restrict xmom_bv = D->xmom_boundary_values;
    double * restrict ymom_bv = D->ymom_boundary_values;

    anuga_int * restrict neighbours = D->neighbours;
    anuga_int * restrict neighbour_edges = D->neighbour_edges;
    double * restrict normals = D->normals;
    double * restrict edgelengths = D->edgelengths;
    double * restrict radii = D->radii;
    anuga_int * restrict tri_full_flag = D->tri_full_flag;

    double * restrict slots = D->edge_flux_work;

    double local_timestep = 1.0e+100;
    double boundary_flux_sum_substep = 0.0;

    const anuga_int nslots = 3 * n;

    #ifdef CPU_ONLY_MODE
    #pragma omp parallel for reduction(min:local_timestep) reduction(+:boundary_flux_sum_substep)
    #else
    #pragma omp target teams distribute parallel for reduction(min:local_timestep) reduction(+:boundary_flux_sum_substep)
    #endif
    for (anuga_int p = 0; p < nslots; p++) {
        const anuga_int k = p / 3;
        const anuga_int nbr = neighbours[p];
        const int is_boundary = (nbr < 0);

        // Owner side only: boundary slots, or the side with the larger index
        if (!is_boundary && nbr < k) continue;

        double ql[3], qr[3], edgeflux[3];

        ql[0] = stage_ev[p];
        ql[1] = xmom_ev[p];
        ql[2] = ymom_ev[p];
        double hle = height_ev[p];
        double zl = ql[0] - hle;          // == bed_ev (post-extrapolate contract)

        double length = edgelengths[p];
        // Normals are read from the owner's slot; the neighbour's copy of the
        // same physical edge is the exact negation.
        double n1 = normals[2 * p];
        double n2 = normals[2 * p + 1];

        double zr, hre;
        if (is_boundary) {
            const anuga_int m = -nbr - 1;
            qr[0] = stage_bv[m];
            qr[1] = xmom_bv[m];
            qr[2] = ymom_bv[m];
            zr = zl;
            hre = fmax(qr[0] - zr, 0.0);
        } else {
            const anuga_int nm = 3 * nbr + neighbour_edges[p];
            qr[0] = stage_ev[nm];
            qr[1] = xmom_ev[nm];
            qr[2] = ymom_ev[nm];
            hre = height_ev[nm];
            zr = qr[0] - hre;
        }

        const double z_half = fmax(zl, zr);
        const double h_left = fmax(hle + zl - z_half, 0.0);
        const double h_right = fmax(hre + zr - z_half, 0.0);

        double max_speed_local = 0.0;
        double pressure_flux = 0.0;

        if (h_left == 0.0 && h_right == 0.0) {
            edgeflux[0] = 0.0;
            edgeflux[1] = 0.0;
            edgeflux[2] = 0.0;
        } else {
            gpu_flux_function_central(ql, qr, h_left, h_right, hle, hre,
                                      n1, n2, epsilon, z_half, g,
                                      edgeflux, &max_speed_local, &pressure_flux,
                                      low_froude);
        }

        const anuga_int base = EDGE_SLOT_STRIDE * p;
        slots[base + 0] = -length * edgeflux[0];
        slots[base + 1] = -length * edgeflux[1];
        slots[base + 2] = -length * edgeflux[2];
        slots[base + 3] = length * pressure_flux;
        slots[base + 4] = z_half;
        slots[base + 5] = max_speed_local;

        // Timestep reduction: min over (cell, edge) pairs of radii/speed --
        // identical to the cell-based min over cells of radii/max(speed),
        // since both evaluate radii/s at the cell's largest edge speed.
        if (substep_count == 0 && max_speed_local > epsilon) {
            if (tri_full_flag == NULL || tri_full_flag[k] == 1)
                local_timestep = fmin(local_timestep, radii[k] / max_speed_local);
            if (!is_boundary && (tri_full_flag == NULL || tri_full_flag[nbr] == 1))
                local_timestep = fmin(local_timestep, radii[nbr] / max_speed_local);
        }

        // Boundary flux integral: the full cell's own-side mass flux across
        // domain-boundary and full<->ghost edges (matches the cell-based sum)
        if (tri_full_flag != NULL) {
            const int k_full = (tri_full_flag[k] == 1);
            if (is_boundary) {
                if (k_full) boundary_flux_sum_substep += slots[base + 0];
            } else {
                const int n_full = (tri_full_flag[nbr] == 1);
                if (k_full && !n_full) boundary_flux_sum_substep += slots[base + 0];
                else if (!k_full && n_full) boundary_flux_sum_substep -= slots[base + 0];
            }
        }
    }

    if (D->boundary_flux_sum != NULL && substep_count < timestep_fluxcalls) {
        D->boundary_flux_sum[substep_count] = boundary_flux_sum_substep;
    }

    return local_timestep;
}

void core_flux_apply_and_update(struct domain *D, double timestep,
                                int apply_manning, int do_saxpy,
                                double a, double b, int substep_count) {
    anuga_int n = D->number_of_elements;
    double g = D->g;
    double minimum_allowed_height = D->minimum_allowed_height;
    double seven_thirds = 7.0 / 3.0;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict height_cv = D->height_centroid_values;
    double * restrict friction_cv = D->friction_centroid_values;

    double * restrict stage_ev = D->stage_edge_values;
    double * restrict height_ev = D->height_edge_values;

    double * restrict stage_siu = D->stage_semi_implicit_update;
    double * restrict xmom_siu = D->xmom_semi_implicit_update;
    double * restrict ymom_siu = D->ymom_semi_implicit_update;

    double * restrict stage_bk = D->stage_backup_values;
    double * restrict xmom_bk = D->xmom_backup_values;
    double * restrict ymom_bk = D->ymom_backup_values;

    anuga_int * restrict neighbours = D->neighbours;
    anuga_int * restrict neighbour_edges = D->neighbour_edges;
    double * restrict normals = D->normals;
    double * restrict edgelengths = D->edgelengths;
    double * restrict areas = D->areas;
    double * restrict max_speed_array = D->max_speed;

    double * restrict slots = D->edge_flux_work;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        double eu_stage = 0.0, eu_xmom = 0.0, eu_ymom = 0.0;
        double speed_max_last = 0.0;

        const double hc = height_cv[k];
        const double zc = bed_cv[k];

        for (int i = 0; i < 3; i++) {
            const anuga_int p = 3 * k + i;
            const anuga_int nbr = neighbours[p];
            const int owner = (nbr < 0 || nbr > k);
            const anuga_int slot = owner ? p : 3 * nbr + neighbour_edges[p];
            const anuga_int base = EDGE_SLOT_STRIDE * slot;
            const double sgn = owner ? 1.0 : -1.0;

            eu_stage += sgn * slots[base + 0];
            eu_xmom  += sgn * slots[base + 1];
            eu_ymom  += sgn * slots[base + 2];

            const double pf_len = slots[base + 3];
            const double z_half = slots[base + 4];
            speed_max_last = fmax(speed_max_last, slots[base + 5]);

            // One-sided pressure-gradient term, from this cell's own edge
            // values -- the same expression the cell-based kernel evaluates.
            // (pf_len uses the owner's edge length; both sides of a physical
            // edge share endpoints, so the lengths are bit-identical.)
            const double hle = height_ev[p];
            const double zl = stage_ev[p] - hle;
            const double length = edgelengths[p];
            const double h_side = fmax(hle + zl - z_half, 0.0);

            const double pg = pf_len
                - length * g * 0.5 * (h_side * h_side - hle * hle
                                      - (hle + hc) * (zl - zc));
            eu_xmom -= normals[2 * p] * pg;
            eu_ymom -= normals[2 * p + 1] * pg;
        }

        const double inv_area = 1.0 / areas[k];
        eu_stage *= inv_area;
        eu_xmom  *= inv_area;
        eu_ymom  *= inv_area;

        if (substep_count == 0) max_speed_array[k] = speed_max_last;

        gpu_cell_forcing_update(k, timestep, apply_manning, do_saxpy, a, b,
                                g, minimum_allowed_height, seven_thirds,
                                eu_stage, eu_xmom, eu_ymom,
                                stage_cv, xmom_cv, ymom_cv, bed_cv, height_cv,
                                friction_cv, stage_siu, xmom_siu, ymom_siu,
                                stage_bk, xmom_bk, ymom_bk);
    }
}

// ============================================================================
// Scatter-mode flux computation (opt-in, single kernel + atomics)
//
// Same single-Riemann-solve-per-edge idea as the slot-based pair above, but
// with NO intermediate storage: the owner thread computes the flux once and
// scatters both sides' full contributions -- flux exchange AND each side's
// one-sided pressure-gradient term, already area-normalized -- directly into
// the explicit-update arrays with `omp atomic update` (portable OpenMP; each
// eu entry receives at most 3 concurrent adds, so contention is negligible).
// The slot-based variant measured SLOWER than the cell-based kernel because
// the 144 B/cell of slot records cost more to move than the duplicate
// Riemann solves saved; this variant keeps the saved solves and moves
// nothing.
//
// Requirements (same contract as the slot variant): the explicit updates
// must be ZERO on entry (core_prepare_step's zero_eu flag), no riverwalls,
// and fluxes follow an extrapolate (bed reconstructed as stage - height).
// max_speed_array is NOT maintained on this path (per-cell max would need an
// atomic max); the wave speeds live and die in registers, so this mode needs
// NO auxiliary arrays at all -- drivers select it with
// D->reconstruct_edge_bed = 2 and pay zero extra device memory.
// ============================================================================

double core_compute_fluxes_scatter(struct domain *D, int substep_count,
                                   int timestep_fluxcalls) {
    anuga_int n = D->number_of_elements;
    double g = D->g;
    double epsilon = D->epsilon;
    anuga_int low_froude = D->low_froude;

    double * restrict stage_ev = D->stage_edge_values;
    double * restrict xmom_ev = D->xmom_edge_values;
    double * restrict ymom_ev = D->ymom_edge_values;
    double * restrict height_ev = D->height_edge_values;

    double * restrict stage_bv = D->stage_boundary_values;
    double * restrict xmom_bv = D->xmom_boundary_values;
    double * restrict ymom_bv = D->ymom_boundary_values;

    double * restrict stage_eu = D->stage_explicit_update;
    double * restrict xmom_eu = D->xmom_explicit_update;
    double * restrict ymom_eu = D->ymom_explicit_update;

    double * restrict height_cv = D->height_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;

    anuga_int * restrict neighbours = D->neighbours;
    anuga_int * restrict neighbour_edges = D->neighbour_edges;
    double * restrict normals = D->normals;
    double * restrict edgelengths = D->edgelengths;
    double * restrict radii = D->radii;
    double * restrict areas = D->areas;
    anuga_int * restrict tri_full_flag = D->tri_full_flag;

    // One thread per PHYSICAL edge via the driver-built compacted slot list --
    // no idle non-owner threads, no divergence on the ownership test.
    anuga_int * restrict owned = D->owned_edges;
    const anuga_int nowned = D->num_owned_edges;

    double local_timestep = 1.0e+100;
    double boundary_flux_sum_substep = 0.0;

    #ifdef CPU_ONLY_MODE
    #pragma omp parallel for reduction(min:local_timestep) reduction(+:boundary_flux_sum_substep)
    #else
    #pragma omp target teams distribute parallel for reduction(min:local_timestep) reduction(+:boundary_flux_sum_substep)
    #endif
    for (anuga_int q = 0; q < nowned; q++) {
        const anuga_int p = owned[q];
        const anuga_int k = p / 3;
        const anuga_int nbr = neighbours[p];
        const int is_boundary = (nbr < 0);

        double ql[3], qr[3], edgeflux[3];

        ql[0] = stage_ev[p];
        ql[1] = xmom_ev[p];
        ql[2] = ymom_ev[p];
        double hle = height_ev[p];
        double zl = ql[0] - hle;

        double length = edgelengths[p];
        double n1 = normals[2 * p];
        double n2 = normals[2 * p + 1];

        double zr, hre;
        anuga_int nm = 0;
        if (is_boundary) {
            const anuga_int m = -nbr - 1;
            qr[0] = stage_bv[m];
            qr[1] = xmom_bv[m];
            qr[2] = ymom_bv[m];
            zr = zl;
            hre = fmax(qr[0] - zr, 0.0);
        } else {
            nm = 3 * nbr + neighbour_edges[p];
            qr[0] = stage_ev[nm];
            qr[1] = xmom_ev[nm];
            qr[2] = ymom_ev[nm];
            hre = height_ev[nm];
            zr = qr[0] - hre;
        }

        const double z_half = fmax(zl, zr);
        const double h_left = fmax(hle + zl - z_half, 0.0);
        const double h_right = fmax(hre + zr - z_half, 0.0);

        double max_speed_local = 0.0;
        double pressure_flux = 0.0;

        if (h_left == 0.0 && h_right == 0.0) {
            edgeflux[0] = 0.0;
            edgeflux[1] = 0.0;
            edgeflux[2] = 0.0;
        } else {
            gpu_flux_function_central(ql, qr, h_left, h_right, hle, hre,
                                      n1, n2, epsilon, z_half, g,
                                      edgeflux, &max_speed_local, &pressure_flux,
                                      low_froude);
        }

        const double F0 = -length * edgeflux[0];
        const double F1 = -length * edgeflux[1];
        const double F2 = -length * edgeflux[2];
        const double pf_len = length * pressure_flux;

        // ---- owner side: flux + its one-sided pressure gradient, scaled
        {
            const double hc = height_cv[k];
            const double zc = bed_cv[k];
            const double pg = pf_len
                - length * g * 0.5 * (h_left * h_left - hle * hle
                                      - (hle + hc) * (zl - zc));
            const double ia = 1.0 / areas[k];
            const double ds = F0 * ia;
            const double dx = (F1 - n1 * pg) * ia;
            const double dy = (F2 - n2 * pg) * ia;

            #pragma omp atomic update
            stage_eu[k] += ds;
            #pragma omp atomic update
            xmom_eu[k] += dx;
            #pragma omp atomic update
            ymom_eu[k] += dy;
        }

        // ---- neighbour side: negated flux, its own pressure gradient.
        // The neighbour's stored normal for this physical edge is the exact
        // FP negation of the owner's (same endpoints, opposite subtraction),
        // so (-n1, -n2) reproduces its cell-based expression bit-for-bit.
        if (!is_boundary) {
            const double hc_n = height_cv[nbr];
            const double zc_n = bed_cv[nbr];
            const double pg = pf_len
                - length * g * 0.5 * (h_right * h_right - hre * hre
                                      - (hre + hc_n) * (zr - zc_n));
            const double ia = 1.0 / areas[nbr];
            const double ds = -F0 * ia;
            const double dx = (-F1 - (-n1) * pg) * ia;
            const double dy = (-F2 - (-n2) * pg) * ia;

            #pragma omp atomic update
            stage_eu[nbr] += ds;
            #pragma omp atomic update
            xmom_eu[nbr] += dx;
            #pragma omp atomic update
            ymom_eu[nbr] += dy;
        }

        if (substep_count == 0 && max_speed_local > epsilon) {
            if (tri_full_flag == NULL || tri_full_flag[k] == 1)
                local_timestep = fmin(local_timestep, radii[k] / max_speed_local);
            if (!is_boundary && (tri_full_flag == NULL || tri_full_flag[nbr] == 1))
                local_timestep = fmin(local_timestep, radii[nbr] / max_speed_local);
        }

        if (tri_full_flag != NULL) {
            const int k_full = (tri_full_flag[k] == 1);
            if (is_boundary) {
                if (k_full) boundary_flux_sum_substep += F0;
            } else {
                const int n_full = (tri_full_flag[nbr] == 1);
                if (k_full && !n_full) boundary_flux_sum_substep += F0;
                else if (!k_full && n_full) boundary_flux_sum_substep -= F0;
            }
        }
    }

    if (D->boundary_flux_sum != NULL && substep_count < timestep_fluxcalls) {
        D->boundary_flux_sum[substep_count] = boundary_flux_sum_substep;
    }

    return local_timestep;
}

// ============================================================================
// ADER Cauchy-Kovalewski predictor
// ============================================================================

void core_ader_ck_predictor(struct domain *D, double dt) {
    // Advance centroid values by dt using a local Cauchy-Kovalewski predictor.
    // Called after core_extrapolate_second_order_edge() so edge_values hold the
    // reconstructed state.  Slopes are recovered from the 2x2 linear system
    // formed by edges 0 and 1 (no new arrays needed).
    //
    // Well-balanced form: bed slope is dz/dx = dw/dx - dh/dx derived from
    // the reconstruction, so still-water equilibrium is preserved exactly.

    anuga_int n = D->number_of_elements;
    double g   = D->g;
    double eps = D->minimum_allowed_height;

    double * restrict stage_cv  = D->stage_centroid_values;
    double * restrict xmom_cv   = D->xmom_centroid_values;
    double * restrict ymom_cv   = D->ymom_centroid_values;
    double * restrict bed_cv    = D->bed_centroid_values;
    double * restrict height_cv = D->height_centroid_values;

    double * restrict stage_ev  = D->stage_edge_values;
    double * restrict xmom_ev   = D->xmom_edge_values;
    double * restrict ymom_ev   = D->ymom_edge_values;
    double * restrict height_ev = D->height_edge_values;

    double * restrict edge_coords     = D->edge_coordinates;
    double * restrict centroid_coords = D->centroid_coordinates;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        anuga_int k3 = k * 3;
        anuga_int k6 = k * 6;
        anuga_int k2 = k * 2;

        // Offsets from centroid to edge midpoints 0 and 1
        double xc   = centroid_coords[k2 + 0];
        double yc   = centroid_coords[k2 + 1];
        double dxv0 = edge_coords[k6 + 0] - xc;
        double dyv0 = edge_coords[k6 + 1] - yc;
        double dxv1 = edge_coords[k6 + 2] - xc;
        double dyv1 = edge_coords[k6 + 3] - yc;

        // Determinant of the 2x2 linear system; skip degenerate cells.
        // Use if-block (not continue) for GPU target-loop compatibility.
        double det = dxv0 * dyv1 - dxv1 * dyv0;
        if (fabs(det) >= 1.0e-20) {
        double inv_det = 1.0 / det;

        // Centroid state
        double w_c  = stage_cv[k];
        double h_c  = fmax(w_c - bed_cv[k], 0.0);
        double uh_c = xmom_cv[k];
        double vh_c = ymom_cv[k];

        // Centroid velocity (guarded)
        double inv_h_c = (h_c > eps) ? 1.0 / h_c : 0.0;
        double u_c = uh_c * inv_h_c;
        double v_c = vh_c * inv_h_c;

        // Recover gradients from edge differences using edges 0 and 1.
        // For any variable q:  grad_x = inv_det*(dyv1*dq0 - dyv0*dq1)
        //                      grad_y = inv_det*(dxv0*dq1 - dxv1*dq0)

        // Stage gradient (∂w/∂x, ∂w/∂y)
        double dw0 = stage_ev[k3 + 0] - w_c;
        double dw1 = stage_ev[k3 + 1] - w_c;
        double wx  = inv_det * (dyv1 * dw0 - dyv0 * dw1);
        double wy  = inv_det * (dxv0 * dw1 - dxv1 * dw0);

        // Height gradient (∂h/∂x, ∂h/∂y)
        double dh0 = height_ev[k3 + 0] - h_c;
        double dh1 = height_ev[k3 + 1] - h_c;
        double hx  = inv_det * (dyv1 * dh0 - dyv0 * dh1);
        double hy  = inv_det * (dxv0 * dh1 - dxv1 * dh0);

        // Edge velocities (recover from edge momentum / edge height)
        double h_e0     = height_ev[k3 + 0];
        double h_e1     = height_ev[k3 + 1];
        double inv_h_e0 = (h_e0 > eps) ? 1.0 / h_e0 : 0.0;
        double inv_h_e1 = (h_e1 > eps) ? 1.0 / h_e1 : 0.0;
        double u_e0 = xmom_ev[k3 + 0] * inv_h_e0;
        double u_e1 = xmom_ev[k3 + 1] * inv_h_e1;
        double v_e0 = ymom_ev[k3 + 0] * inv_h_e0;
        double v_e1 = ymom_ev[k3 + 1] * inv_h_e1;

        // Velocity gradients (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y)
        double du0 = u_e0 - u_c;
        double du1 = u_e1 - u_c;
        double dv0 = v_e0 - v_c;
        double dv1 = v_e1 - v_c;
        double ux  = inv_det * (dyv1 * du0 - dyv0 * du1);
        double uy  = inv_det * (dxv0 * du1 - dxv1 * du0);
        double vx  = inv_det * (dyv1 * dv0 - dyv0 * dv1);
        double vy  = inv_det * (dxv0 * dv1 - dxv1 * dv0);

        // Cauchy-Kovalewski time derivatives — well-balanced SWE:
        //   dz/dx = dw/dx - dh/dx  (from reconstruction, not stored centroid z)
        // This ensures cancellation in still water (u=v=0, wx=wy=0).
        double g_h = g * h_c;

        double dw_dt  = -(u_c * hx + h_c * ux + v_c * hy + h_c * vy);
        double duh_dt = -(2.0*u_c*h_c*ux + u_c*u_c*hx + u_c*v_c*hy
                         + v_c*h_c*uy + u_c*h_c*vy + g_h * wx);
        double dvh_dt = -(v_c*h_c*ux + u_c*h_c*vx + u_c*v_c*hx
                         + 2.0*v_c*h_c*vy + v_c*v_c*hy + g_h * wy);

        // Predict forward by dt (caller passes dt/2 for midpoint)
        double w_pred  = w_c  + dt * dw_dt;
        double uh_pred = uh_c + dt * duh_dt;
        double vh_pred = vh_c + dt * dvh_dt;
        double h_pred  = fmax(w_pred - bed_cv[k], 0.0);

        stage_cv[k]  = w_pred;
        xmom_cv[k]   = uh_pred;
        ymom_cv[k]   = vh_pred;
        height_cv[k] = h_pred;
        } // end if (fabs(det) >= 1.0e-20)
    }
}

void core_ader_ck_predictor_edge(struct domain *D, double dt) {
    // Fused ADER-2 predictor: advances edge values to Q^{n+1/2} in-place,
    // leaving centroid values unchanged.  This eliminates the second full
    // extrapolation pass needed by core_ader_ck_predictor (centroid variant).
    //
    // For any quantity q, the reconstructed edge value is:
    //   q_edge[i] = q_c + slope * offset_i
    // Since the predictor adds the same centroid shift dq_c to every edge,
    //   q_edge_pred[i] = q_edge[i] + dt * dq_c/dt
    // The cell slopes are preserved exactly.
    //
    // Well-balanced: same dz/dx = dw/dx - dh/dx derivation as the centroid
    // variant; still-water equilibrium is preserved exactly.

    anuga_int n = D->number_of_elements;
    double g   = D->g;
    double eps = D->minimum_allowed_height;

    double * restrict stage_cv  = D->stage_centroid_values;
    double * restrict xmom_cv   = D->xmom_centroid_values;
    double * restrict ymom_cv   = D->ymom_centroid_values;
    double * restrict bed_cv    = D->bed_centroid_values;

    double * restrict stage_ev  = D->stage_edge_values;
    double * restrict xmom_ev   = D->xmom_edge_values;
    double * restrict ymom_ev   = D->ymom_edge_values;
    double * restrict height_ev = D->height_edge_values;

    double * restrict edge_coords     = D->edge_coordinates;
    double * restrict centroid_coords = D->centroid_coordinates;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        anuga_int k3 = k * 3;
        anuga_int k6 = k * 6;
        anuga_int k2 = k * 2;

        double xc   = centroid_coords[k2 + 0];
        double yc   = centroid_coords[k2 + 1];
        double dxv0 = edge_coords[k6 + 0] - xc;
        double dyv0 = edge_coords[k6 + 1] - yc;
        double dxv1 = edge_coords[k6 + 2] - xc;
        double dyv1 = edge_coords[k6 + 3] - yc;

        double det = dxv0 * dyv1 - dxv1 * dyv0;
        if (fabs(det) >= 1.0e-20) {
        double inv_det = 1.0 / det;

        double w_c  = stage_cv[k];
        double h_c  = fmax(w_c - bed_cv[k], 0.0);
        double uh_c = xmom_cv[k];
        double vh_c = ymom_cv[k];

        double inv_h_c = (h_c > eps) ? 1.0 / h_c : 0.0;
        double u_c = uh_c * inv_h_c;
        double v_c = vh_c * inv_h_c;

        double dw0 = stage_ev[k3 + 0] - w_c;
        double dw1 = stage_ev[k3 + 1] - w_c;
        double wx  = inv_det * (dyv1 * dw0 - dyv0 * dw1);
        double wy  = inv_det * (dxv0 * dw1 - dxv1 * dw0);

        double dh0 = height_ev[k3 + 0] - h_c;
        double dh1 = height_ev[k3 + 1] - h_c;
        double hx  = inv_det * (dyv1 * dh0 - dyv0 * dh1);
        double hy  = inv_det * (dxv0 * dh1 - dxv1 * dh0);

        double h_e0     = height_ev[k3 + 0];
        double h_e1     = height_ev[k3 + 1];
        double inv_h_e0 = (h_e0 > eps) ? 1.0 / h_e0 : 0.0;
        double inv_h_e1 = (h_e1 > eps) ? 1.0 / h_e1 : 0.0;
        double u_e0 = xmom_ev[k3 + 0] * inv_h_e0;
        double u_e1 = xmom_ev[k3 + 1] * inv_h_e1;
        double v_e0 = ymom_ev[k3 + 0] * inv_h_e0;
        double v_e1 = ymom_ev[k3 + 1] * inv_h_e1;

        double du0 = u_e0 - u_c;
        double du1 = u_e1 - u_c;
        double dv0 = v_e0 - v_c;
        double dv1 = v_e1 - v_c;
        double ux  = inv_det * (dyv1 * du0 - dyv0 * du1);
        double uy  = inv_det * (dxv0 * du1 - dxv1 * du0);
        double vx  = inv_det * (dyv1 * dv0 - dyv0 * dv1);
        double vy  = inv_det * (dxv0 * dv1 - dxv1 * dv0);

        double g_h = g * h_c;
        double dw_dt  = -(u_c * hx + h_c * ux + v_c * hy + h_c * vy);
        double duh_dt = -(2.0*u_c*h_c*ux + u_c*u_c*hx + u_c*v_c*hy
                         + v_c*h_c*uy + u_c*h_c*vy + g_h * wx);
        double dvh_dt = -(v_c*h_c*ux + u_c*h_c*vx + u_c*v_c*hx
                         + 2.0*v_c*h_c*vy + v_c*v_c*hy + g_h * wy);

        // Shift all three edges by the same centroid delta (slopes preserved)
        for (int i = 0; i < 3; i++) {
            double new_stage = stage_ev[k3 + i] + dt * dw_dt;
            stage_ev[k3 + i] = new_stage;
            xmom_ev[k3 + i] += dt * duh_dt;
            ymom_ev[k3 + i] += dt * dvh_dt;
            height_ev[k3 + i] = fmax(height_ev[k3 + i] + dt * dw_dt, 0.0);
        }
        } // end if (fabs(det) >= 1.0e-20)
    }
}
