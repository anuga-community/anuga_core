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
    const anuga_int n_tracers_x = D->number_of_tracers;

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

    // Generic passive tracers. n_tracers == 0 in every ordinary run; keep only
    // the loop-invariant count live in the common path.
    const anuga_int n_tracers = D->number_of_tracers;
    const double beta_tracer = D->beta_tracer;
    // See the note above core_compute_fluxes_central: on a GPU build these must
    // be loaded at function scope, because D is not mapped inside the 'omp
    // target' regions below; on a CPU build they stay inside the guard.
#ifndef CPU_ONLY_MODE
    double * restrict t_cons = D->tracer_conserved_values;
    double * restrict t_cv   = D->tracer_centroid_values;
    double * restrict t_ev   = D->tracer_edge_values;
#endif
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

        // Derive tracer concentration c = m/h from the conserved m, exactly as
        // height is derived from stage above. Dry cells carry no concentration.
        if (n_tracers_x > 0) {
#ifdef CPU_ONLY_MODE
            double * restrict t_cons = D->tracer_conserved_values;
            double * restrict t_cv   = D->tracer_centroid_values;
#endif
            double inv_h = is_dry ? 0.0 : (1.0 / dk);
            for (anuga_int s = 0; s < n_tracers_x; s++) {
                t_cv[s * n + k] = t_cons[s * n + k] * inv_h;
            }
        }
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

            if (n_tracers > 0) {
#ifdef CPU_ONLY_MODE
                double * restrict t_cv = D->tracer_centroid_values;
                double * restrict t_ev = D->tracer_edge_values;
#endif
                for (anuga_int sidx = 0; sidx < n_tracers; sidx++) {
                    double tc = t_cv[sidx * n + k];
                    t_ev[sidx * 3 * n + k3 + 0] = tc;
                    t_ev[sidx * 3 * n + k3 + 1] = tc;
                    t_ev[sidx * 3 * n + k3 + 2] = tc;
                }
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

            // Tracers. Reconstruct c (the intensive variable) rather than the
            // conserved h*c: the limiter then bounds each edge value by the
            // cell-and-neighbour range of c, which is what preserves positivity
            // and prevents spurious extrema where h varies sharply.
            if (n_tracers > 0) {
#ifdef CPU_ONLY_MODE
                double * restrict t_cv = D->tracer_centroid_values;
                double * restrict t_ev = D->tracer_edge_values;
#endif
                double beta_t = beta_tracer * hfactor;
                for (anuga_int sidx = 0; sidx < n_tracers; sidx++) {
                    anuga_int off = sidx * n;
                    if (beta_t > 0.0) {
                        gpu_calc_edge_values_with_gradient(
                            t_cv[off + k], t_cv[off + k0], t_cv[off + k1], t_cv[off + sn2],
                            dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                            dx1, dx2, dy1, dy2, inv_area2, beta_t, edge_vals);
                    } else {
                        gpu_set_constant_edge_values(t_cv[off + k], edge_vals);
                    }
                    t_ev[sidx * 3 * n + k3 + 0] = edge_vals[0];
                    t_ev[sidx * 3 * n + k3 + 1] = edge_vals[1];
                    t_ev[sidx * 3 * n + k3 + 2] = edge_vals[2];
                }
            }

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

            // Tracers, 1D gradient toward the one internal neighbour
            if (n_tracers > 0) {
#ifdef CPU_ONLY_MODE
                double * restrict t_cv = D->tracer_centroid_values;
                double * restrict t_ev = D->tracer_edge_values;
#endif
                for (anuga_int sidx = 0; sidx < n_tracers; sidx++) {
                    anuga_int off = sidx * n;
                    double tk = t_cv[off + k];
                    if (beta_tracer > 0.0) {
                        dq1 = t_cv[off + kn] - tk;
                        gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                                      dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
                        gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
                        gpu_limit_gradient(dqv, qmin, qmax, beta_tracer);
                    } else {
                        dqv[0] = 0.0; dqv[1] = 0.0; dqv[2] = 0.0;
                    }
                    t_ev[sidx * 3 * n + k3 + 0] = tk + dqv[0];
                    t_ev[sidx * 3 * n + k3 + 1] = tk + dqv[1];
                    t_ev[sidx * 3 * n + k3 + 2] = tk + dqv[2];
                }
            }
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
// Near-bed concentration ratio d*(Z)   -- spec 4.3, open item S1a
// ============================================================================
//
// [S-4] is a 1-D quadrature per cell, far too expensive to run inside a kernel
// every step, so the spec calls for a fitted form. This is that fit.
//
// CORRECTION TO [S-4] -- THE TYPO IS IN DL09 AS PUBLISHED, not in PHYSICS_SPEC,
// which transcribed them faithfully. DL09 Eq 19 gives the Rouse profile factor
// as ((z-a)/(h-a) . a/z)^Z, which is ZERO at the reference height z = a.
//
// Their own paper disproves it: immediately above Eq 19 they write the flux as
// q_S = c_S(a) * integral( [...]^Z u(z) dz ). Factoring c_S(a) out REQUIRES the
// bracket to be 1 at z = a; the printed factor is 0 there, giving c_s(a) = 0.
//
// The Rouse-Vanoni profile [(h-z)/z . a/(h-a)]^Z rearranges exactly to
// ((h-z)/(h-a) . a/z)^Z, which is 1 at z = a and 0 at z = h. One glyph: (z-a)
// is a slip for (h-z). At Z = 2, a/h = 0.005 the printed form gives d* = 41227
// against 356 corrected -- not a value that appears on their Figure 4, so that
// figure was evidently computed with the correct profile.
//
// The fit below is to the CORRECTED integral. See PHYSICS_SPEC 4.3, Draft 5.
//
// FITTED FORM. Near the bed the Rouse profile behaves like z^(-Z), so d*
// diverges roughly as (a/h)^(-Z). Factoring that out first,
//
//     ln d* = -Z ln(a/h) + P(Z, ln(a/h))
//
// leaves a mild remainder a low-order polynomial captures well: 28 terms for
// 0.82 percent max / 0.05 percent mean error. A direct polynomial in
// (ln Z, ln(a/h)) needs far more terms for far worse accuracy. The structure is
// what buys it.
//
// RANGE. Fitted for Z in [0.01, 2.5] and a/h in [1e-3, 0.15]. Beyond Z ~ 2.5
// transport is essentially bedload and this ratio is not the right model. The
// a/h range reaches down to 1e-3 deliberately: the shipped anugaSed operator
// implies a/h ~ 9.3e-4 (spec 12, D4b), so its regime is reachable when the
// a >= floor*h floor is relaxed. (9.3e-4 itself clamps to 1e-3, ~7 percent in
// a/h and so ~7 percent in d* at Z = 1.)
//
// OUT-OF-RANGE INPUTS ARE CLAMPED, NOT EXTRAPOLATED -- the spec asks for that
// explicitly, because a polynomial taken outside its fitted range goes wrong
// quietly. anugaSed's own 8th-degree fit is extrapolated freely and reaches
// p(6) = 282088 (spec 12, D4c); this one cannot.
#define ANUGA_ROUSE_Z_LO   0.01
#define ANUGA_ROUSE_Z_HI   2.5
#define ANUGA_ROUSE_AH_LO  1e-3
#define ANUGA_ROUSE_AH_HI  0.15

static inline double core_rouse_d_star(double Z, double a_h) {
    /* P(Z, L) = sum_i Z^i (c_i0 + c_i1 L + c_i2 L^2 + c_i3 L^3), L = ln(a/h) */
    const double C[7][4] = {
    {+1.097192252266e-03, +9.816426876103e-04, +2.816550608693e-04, +2.216981593577e-05},
    {+8.152552738643e-01, +2.984288438662e-01, +4.717126357513e-02, +2.488390592718e-03},
    {-3.858022145865e-02, +7.497943739332e-01, +1.494530687071e-01, +1.016829763599e-02},
    {-1.416163484237e-01, -6.145585548869e-01, -2.181478641118e-01, -1.989085090511e-02},
    {+2.441798567588e-02, +2.477861105262e-01, +1.172562489413e-01, +1.380260943049e-02},
    {+1.714535144604e-02, -4.453006825886e-02, -2.922351673152e-02, -4.300807416335e-03},
    {-4.557991043496e-03, +2.511784955218e-03, +2.810236330103e-03, +5.048472261972e-04},
    };

    if (Z < ANUGA_ROUSE_Z_LO) Z = ANUGA_ROUSE_Z_LO;
    else if (Z > ANUGA_ROUSE_Z_HI) Z = ANUGA_ROUSE_Z_HI;
    if (a_h < ANUGA_ROUSE_AH_LO) a_h = ANUGA_ROUSE_AH_LO;
    else if (a_h > ANUGA_ROUSE_AH_HI) a_h = ANUGA_ROUSE_AH_HI;

    const double L = log(a_h);
    const double L2 = L * L;

    /* Horner in Z over coefficients that are cubics in L. */
    const double L3 = L2 * L;
    double P = 0.0;
    for (int i = 6; i >= 0; i--) {
        P = P * Z + (C[i][0] + C[i][1] * L + C[i][2] * L2 + C[i][3] * L3);
    }

    const double d = exp(-Z * L + P);
    /* DL09: d* is always > 1. Guard the fit against dipping below it. */
    return (d < 1.0) ? 1.0 : d;
}

/* tau_b / rho for a cell, under either shear closure (spec 3.1 / 3.4).
 *
 *   [T-1]  tau_b/rho = f_c |v|^2          quadratic drag, no equilibrium
 *                                         assumption
 *   [T-7]  tau_b/rho = g h S              depth-slope, aSM16 Eqs 6-7
 *
 * Returning tau_b/rho rather than tau_b keeps the two interchangeable
 * everywhere downstream: the Shields stress is tau* = (tau_b/rho)/(R g d), in
 * which the density cancels, and the dimensional stress the cohesive route
 * needs is simply rho_w times this.
 *
 * S is the bed gradient magnitude from the divergence theorem over the cell's
 * own edges, so this stays cell-local and offloads. */
static inline double core_tau_b_over_rho(anuga_int closure, double f_c,
                                         double vel2, double grav, double h,
                                         const double * restrict bed_ev,
                                         const double * restrict normals,
                                         const double * restrict edgelengths,
                                         double area, anuga_int k) {
    if (closure != 1) {
        return f_c * vel2;                       /* [T-1] */
    }
    /* [T-7]: grad z = (1/A) sum_e z_e n_e L_e */
    double gx = 0.0, gy = 0.0;
    for (anuga_int i = 0; i < 3; i++) {
        const double ze = bed_ev[3 * k + i];
        const double L = edgelengths[3 * k + i];
        gx += ze * normals[6 * k + 2 * i] * L;
        gy += ze * normals[6 * k + 2 * i + 1] * L;
    }
    if (area > 0.0) {
        gx /= area;
        gy /= area;
    }
    const double S = sqrt(gx * gx + gy * gy);
    return grav * h * S;
}

// ============================================================================
// Bedload transport and its bed evolution   [G-5]/[K-3], spec 6
// ============================================================================
//
//   [K-1]  q_b* = K tau_x^m                 tau_x = tau* - tau_c*  [T-4]
//   [K-5]  q_b* = 0.05 tau*^2.5 / f_c       Engelund-Hansen, no threshold
//   [K-2]  q_b  = q_b* sqrt(R g) d^1.5
//   [K-4]  q_b is parallel to the bed shear stress, hence to (u, v)
//   [K-3]  dz/dt = -(1/(1-lambda)) div q_b
//
// Unlike the suspended exchange, this is a DIVERGENCE: it moves sediment along
// the bed rather than between bed and water column, so in a closed domain it
// redistributes bed material and conserves total bed volume exactly. That is
// the property to test it with.
//
// Two passes, because the divergence at cell k needs its neighbours' q_b:
// pass 1 fills the per-cell transport vector, pass 2 takes the divergence.
// Both are ordinary cell loops, so both offload.
//
// Edge flux is CENTRED; see the note at the flux itself for why upwinding was
// tried and rejected. Boundary edges carry zero bedload flux, which is what
// makes the closed-domain conservation test exact.
void core_apply_bedload(struct domain *D, double timestep) {
    const anuga_int mode = D->sediment_bedload_mode;
    const anuga_int n_classes = D->n_sediment_classes;
    if (mode == 0 || n_classes <= 0 || timestep <= 0.0) {
        return;
    }
    if (!D->sediment_bed_evolution) {
        return;   /* fixed bed: bedload would have nowhere to go */
    }

    const anuga_int n = D->number_of_elements;
    const double grav = D->g;
    const double h_eps = D->epsilon;
    const double minimum_allowed_height = D->minimum_allowed_height;
    const double one_minus_lambda = 1.0 - D->sediment_porosity;
    const double K = D->sediment_bedload_K;
    const double mexp = D->sediment_bedload_m;
    const double tau_c_b = D->sediment_bedload_tau_c_star;
    const anuga_int fric_mode = D->sediment_friction_mode;
    const double n_ll = D->sediment_manning_ll;
    const anuga_int wbed = D->sediment_wilson_bed;
    const double wD = D->sediment_wilson_D;
    const anuga_int shear_closure = D->sediment_shear_closure;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict bed_ev = D->bed_edge_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict friction_cv = D->friction_centroid_values;
    double * restrict diam = D->sediment_diameter;
    double * restrict sedR = D->sediment_R;
    double * restrict qbx = D->sediment_qbx;
    double * restrict qby = D->sediment_qby;
    anuga_int * restrict neighbours = D->neighbours;
    double * restrict normals = D->normals;
    double * restrict edgelengths = D->edgelengths;
    double * restrict areas = D->areas;
    /* [L-5]. Both the flag and the arrays are required; see the note in
     * core_apply_sediment_source. */
    double * restrict z_base = D->sediment_z_base;
    anuga_int * restrict exhausted = D->sediment_bed_exhausted;
    const anuga_int has_z_base = (D->sediment_has_z_base
                                  && z_base != NULL && exhausted != NULL);

    if (one_minus_lambda <= 0.0) {
        return;
    }

    /* ---- pass 1: the transport vector, summed over classes ---- */
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        qbx[k] = 0.0;
        qby[k] = 0.0;

        const double h = fmax(stage_cv[k] - bed_cv[k], 0.0);
        if (h <= minimum_allowed_height) {
            continue;
        }

        const double denom = h * h + h_eps * h_eps;
        const double u = (denom > 0.0) ? (xmom_cv[k] * h / denom) : 0.0;
        const double v = (denom > 0.0) ? (ymom_cv[k] * h / denom) : 0.0;
        const double vel2 = u * u + v * v;
        if (vel2 <= 0.0) {
            continue;
        }
        const double speed = sqrt(vel2);

        double f_c;
        if (fric_mode == 2) {
            double rel = h / wD;
            if (!(rel > 1.0)) rel = 1.0;
            double X;
            if (wbed == 0)      X = 8.46 * pow(rel, 0.1005);
            else if (wbed == 1) X = 5.75 * log10(rel) + 3.514;
            else                X = 5.62 * log10(rel) + 4.0;
            f_c = 1.0 / (X * X);
        } else {
            const double nman = (fric_mode == 1) ? n_ll : friction_cv[k];
            f_c = grav * nman * nman / cbrt(h);
        }

        /* Same closure as the suspended source, [T-1] or [T-7]. */
        const double tbr = core_tau_b_over_rho(shear_closure, f_c, vel2, grav,
                                               h, bed_ev, normals,
                                               edgelengths, areas[k], k);

        double q_b_total = 0.0;
        for (anuga_int s = 0; s < n_classes; s++) {
            const double Rgd = sedR[s] * grav * diam[s];
            if (!(Rgd > 0.0)) continue;
            const double tau_star = tbr / Rgd;

            double q_star;
            if (mode == 2) {
                /* [K-5] Engelund-Hansen, total load, NO threshold. Subtracting
                 * tau_c* here would silently make it a different model. */
                q_star = (f_c > 0.0) ? 0.05 * pow(tau_star, 2.5) / f_c : 0.0;
            } else {
                const double tau_x = tau_star - tau_c_b;
                q_star = (tau_x > 0.0) ? K * pow(tau_x, mexp) : 0.0;
            }
            if (q_star <= 0.0) continue;

            /* [K-2] */
            q_b_total += q_star * sqrt(sedR[s] * grav) * pow(diam[s], 1.5);
        }

        if (q_b_total > 0.0) {
            /* [L-5]. A cell cannot export bed material it does not have.
             * The limit is applied to the TRANSPORT VECTOR, not to the
             * divergence: both cells sharing an edge then form their flux
             * from the same limited q, so the flux stays antisymmetric and
             * the scheme stays exactly conservative. Clamping the divergence
             * instead would let one side remove what the other never
             * received, which creates bed material. */
            if (has_z_base) {
                const double avail = bed_cv[k] - z_base[k];
                const double thickness = (avail > 0.0) ? avail : 0.0;
                /* The cell's own contribution to its outflow: its half of
                 * every edge's centred flux, counting only the outgoing
                 * ones. */
                double own_out = 0.0;
                const double ex = q_b_total * u / speed;
                const double ey = q_b_total * v / speed;
                for (anuga_int i = 0; i < 3; i++) {
                    const anuga_int ki = 3 * k + i;
                    if (neighbours[ki] < 0) continue;
                    const double qn = 0.5 * (ex * normals[6 * k + 2 * i]
                                           + ey * normals[6 * k + 2 * i + 1]);
                    if (qn > 0.0) own_out += qn * edgelengths[ki];
                }
                if (own_out > 0.0) {
                    const double cap = thickness * one_minus_lambda
                                     * areas[k] / timestep;
                    if (own_out > cap) {
                        q_b_total *= cap / own_out;
                    }
                }
            }

            /* [K-4] parallel to (u, v) */
            qbx[k] = q_b_total * u / speed;
            qby[k] = q_b_total * v / speed;
        }
    }

    /* ---- [L-5] pass 1.5: which cells cannot afford what is about to be
     * taken from them ----------------------------------------------------
     *
     * Flagging cells that have ALREADY reached the base is not enough. A cell
     * with a millimetre left can be asked for two in a single step and is
     * only found empty afterwards, which is how the first version of this
     * overshot its base by 6.1e-6 m on a 1e-2 m layer. So the test is
     * predictive: form the divergence this step WILL produce and flag the
     * cell if it cannot pay for it.
     *
     * Separate sweep, not folded into pass 2, because pass 2 writes bed
     * elevation while its neighbours are still reading it -- see the note in
     * sw_domain.h. This one only reads, so every cell sees the same state.
     */
    if (has_z_base) {
        OMP_PARALLEL_LOOP
        for (anuga_int k = 0; k < n; k++) {
            const double avail = bed_cv[k] - z_base[k];
            if (avail <= 0.0) {
                exhausted[k] = 1;      /* nothing left to give */
                continue;
            }
            double outflux = 0.0;
            for (anuga_int i = 0; i < 3; i++) {
                const anuga_int ki = 3 * k + i;
                const anuga_int nb = neighbours[ki];
                if (nb < 0) continue;
                const double qn = 0.5 *
                    ((qbx[k] + qbx[nb]) * normals[6 * k + 2 * i]
                   + (qby[k] + qby[nb]) * normals[6 * k + 2 * i + 1]);
                outflux += qn * edgelengths[ki];
            }
            /* dz this step, if nothing were blocked. */
            const double drop = (timestep * outflux / areas[k])
                              / one_minus_lambda;
            exhausted[k] = (drop > avail) ? 1 : 0;
        }
    }

    /* ---- pass 2: divergence, and the bed update ---- */
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        const double qx_k = qbx[k];
        const double qy_k = qby[k];
        double outflux = 0.0;

        for (anuga_int i = 0; i < 3; i++) {
            const anuga_int ki = 3 * k + i;
            const anuga_int nb = neighbours[ki];
            if (nb < 0) {
                continue;            /* boundary: no bedload across it */
            }
            const double nx = normals[6 * k + 2 * i];
            const double ny = normals[6 * k + 2 * i + 1];

            /* CENTRED edge flux: q_edge = (q_k + q_nb)/2.
             *
             * Exactly conservative -- cell nb forms the same average against
             * the opposite normal and so removes precisely what k gains -- and,
             * unlike an upwind donor choice, CONTINUOUS.
             *
             * Upwinding was tried first and rejected twice over. Deciding the
             * donor from each cell's own q.n is not antisymmetric and creates
             * bed material (measured 1.05e-5 of bed volume in 60 s). Fixing
             * that by deciding from the averaged vector is conservative but
             * DISCONTINUOUS: where q_k.n = -q_nb.n, which is exactly a
             * converging-flow edge, the average passes through zero while the
             * two candidate fluxes differ by a finite amount, so the donor
             * flips on roundoff. That put mode 1 and mode 2 1.3e-4 apart.
             *
             * The centred flux is also the physically right answer at such an
             * edge: bedload converging from both sides should deposit there,
             * not be attributed to one arbitrary donor.
             *
             * If oscillations ever appear in an advection-dominated case, the
             * upgrade is a Rusanov-type flux -- centred plus a dissipation
             * term in (z_nb - z_k) -- not a bare donor switch. */
            double qn = 0.5 * ((qx_k + qbx[nb]) * nx
                             + (qy_k + qby[nb]) * ny);

            /* [L-5]. A cell that cannot pay for this step's removal (pass
             * 1.5) may gain material but must not lose any, so close every
             * edge that would take material OUT of it. qn > 0 is outflow from
             * k, qn < 0 is outflow from nb.
             *
             * This is SYMMETRIC, which is the whole point: cell nb reaches
             * this edge with the opposite normal, hence -qn, and the same
             * two tests in the same order, so both sides close the same edge
             * and neither can remove what the other did not give up. That is
             * what keeps bedload exactly conservative with a base present.
             * It works on a snapshot taken in pass 1 rather than on live
             * elevation, because this loop writes elevation as it goes. */
            if (has_z_base) {
                if (qn > 0.0 && exhausted[k])  qn = 0.0;
                if (qn < 0.0 && exhausted[nb]) qn = 0.0;
            }
            outflux += qn * edgelengths[ki];
        }

        /* [K-3]: dz/dt = -(1/(1-lambda)) div q_b, div q_b = outflux/area */
        const double dz = -(timestep * outflux / areas[k]) / one_minus_lambda;
        if (dz != 0.0) {
            bed_cv[k] += dz;
            const anuga_int k3 = 3 * k;
            bed_ev[k3 + 0] += dz;
            bed_ev[k3 + 1] += dz;
            bed_ev[k3 + 2] += dz;
        }
    }
}

// ============================================================================
// Angle-of-repose relaxation  (spec 7)
// ============================================================================

// Diffuse bed material downslope wherever the centroid-to-centroid slope
// exceeds the critical angle, until it does not.
//
// FG21 §2.2.4, who are explicit that this is a NUMERICAL HEURISTIC and not
// physics: real bed slope failures are advective. It exists to stop the rest of
// the model breaking on over-steep slopes -- and it has a side effect worth
// remembering, that it limits the steepness of canyon walls and knickpoints and
// so suppresses knickpoint retreat that may be real.
//
// THE ONLY NON-CELL-LOCAL SEDIMENT KERNEL. A cell's update depends on its
// neighbours' elevation, which forces two things:
//
//   * Jacobi, not Gauss-Seidel. Each sweep reads elevation and writes
//     increments to a separate array, so no cell sees a neighbour that has
//     already moved this sweep. Reading live elevation would make the answer
//     depend on which thread got there first, and mode 1 and mode 2 would
//     diverge.
//   * a hard sweep cap, reported to the caller. Relaxation is iterative and a
//     pathological bed could otherwise spin.
//
// MASS IS CONSERVED, which is what makes this different from the older
// sanddune operator: material removed from an over-steep cell is DEPOSITED ON
// ITS NEIGHBOUR, never discarded. The mechanism is the same one bedload uses --
// the transfer is computed per EDGE from data both cells share, so both compute
// the identical volume and the pair balances exactly:
//
//     V = relax (|dz| - tan(theta) d) / (1/A_k + 1/A_nb)
//
// which is the volume that brings the pair exactly to the threshold slope when
// relax = 1: moving V lowers the donor by V/A_donor and raises the receiver by
// V/A_receiver, closing the excess by V(1/A_k + 1/A_nb).
//
// Interacts with [L-5]: material that cannot be scoured cannot slump either, so
// a transfer is capped by the DONOR's erodible thickness. The cap is a third of
// it per edge, because a cell has three edges and may be the donor on all of
// them; a full-thickness cap on each could lower it to three times its
// available depth in one sweep. The remainder is simply moved on later sweeps.
//
// Returns the number of sweeps used. Equal to max_sweeps means the cap was hit
// and the bed may still be over-steep -- the caller reports that rather than
// letting it pass silently.
/* Relative tolerance on the threshold slope; see the note at its use. */
#define REPOSE_TOL 1.0e-3

anuga_int core_apply_repose(struct domain *D) {
    const double tan_c = D->sediment_repose_tan;
    if (!(tan_c > 0.0)) {
        return 0;                      /* disabled, the default */
    }
    if (!D->sediment_bed_evolution) {
        return 0;                      /* a fixed bed cannot slump */
    }

    const anuga_int n = D->number_of_elements;
    const anuga_int max_sweeps = D->sediment_repose_max_sweeps;
    const double relax = D->sediment_repose_relax;

    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict bed_ev = D->bed_edge_values;
    double * restrict dz = D->sediment_repose_dz;
    double * restrict areas = D->areas;
    double * restrict cc = D->centroid_coordinates;
    anuga_int * restrict neighbours = D->neighbours;
    double * restrict z_base = D->sediment_z_base;
    const anuga_int has_z_base = (D->sediment_has_z_base && z_base != NULL);

    if (dz == NULL || max_sweeps <= 0) {
        return 0;
    }

    anuga_int sweeps = 0;
    for (anuga_int it = 0; it < max_sweeps; it++) {
        anuga_int n_steep = 0;

        OMP_PARALLEL_LOOP_REDUCTION_PLUS(n_steep)
        for (anuga_int k = 0; k < n; k++) {
            double acc = 0.0;
            const double z_k = bed_cv[k];
            const double A_k = areas[k];
            const double xk = cc[2 * k];
            const double yk = cc[2 * k + 1];

            for (anuga_int i = 0; i < 3; i++) {
                const anuga_int nb = neighbours[3 * k + i];
                if (nb < 0 || nb == k) {
                    continue;          /* boundary, or a ghost self-reference */
                }
                const double dx = xk - cc[2 * nb];
                const double dy = yk - cc[2 * nb + 1];
                const double d = sqrt(dx * dx + dy * dy);
                if (!(d > 0.0)) {
                    continue;
                }

                const double z_nb = bed_cv[nb];
                const double diff = z_k - z_nb;
                const double adiff = fabs(diff);
                const double thresh = tan_c * d;

                /* Convergence is ASYMPTOTIC: each sweep removes a fraction of
                 * the excess, so a strict `> thresh` test is never satisfied
                 * and the loop runs to its cap every timestep, reporting a
                 * failure that has not happened. Measured on an over-steep
                 * cone: 36.84 -> 30.09 degrees against a 30 degree limit in
                 * 400 sweeps, still "not converged".
                 *
                 * So converged means within REPOSE_TOL of the threshold
                 * slope, which bounds the final angle: at 30 degrees a
                 * tolerance of 1e-3 in tan leaves at most 30.03 degrees.
                 * Deliberately not exposed -- it is the kernel's own
                 * convergence criterion, not a physical parameter, and the
                 * physical knob (the angle) is already there. */
                if (!(adiff > thresh * (1.0 + REPOSE_TOL))) {
                    continue;
                }
                n_steep++;

                const double A_nb = areas[nb];
                const double inv_sum = 1.0 / A_k + 1.0 / A_nb;

                /* The /3 is a STABILITY limit, not a fudge. relax = 1 with no
                 * divisor brings a single over-steep PAIR exactly to the
                 * threshold in one sweep -- but a cell has three edges and can
                 * be the donor on all of them at once, so its total change is
                 * up to three times what any one edge intended. That is an
                 * explicit diffusion step past its stability limit: it
                 * overshoots, creates fresh over-steep edges on the far side,
                 * and oscillates instead of converging. Observed with relax =
                 * 0.5 and no divisor: an over-steep cone stalled at 30.09
                 * degrees against a 30 degree limit and burned all 400 sweeps
                 * every timestep, so the cap was reported hit on a problem
                 * that was simply never going to converge.
                 *
                 * Dividing by the edge count bounds a cell's total movement by
                 * relax times its worst excess, which is stable for any
                 * relax <= 1, and keeps relax meaning what the docstring says
                 * it means. Both cells divide by the same 3, so the transfer
                 * stays symmetric and B1 conservation is untouched. */
                double V = relax * (adiff - thresh) / inv_sum / 3.0;

                /* [L-5]: the donor cannot give up what it may not lose. Both
                 * cells identify the same donor -- the higher one -- and
                 * compute the same cap, so the transfer stays symmetric. */
                if (has_z_base) {
                    const anuga_int donor = (diff > 0.0) ? k : nb;
                    const double avail = (bed_cv[donor] - z_base[donor])
                                       * areas[donor] / 3.0;
                    if (V > avail) {
                        V = (avail > 0.0) ? avail : 0.0;
                    }
                }

                /* The higher cell gives, the lower receives. */
                acc += (diff > 0.0) ? (-V / A_k) : (V / A_k);
            }
            dz[k] = acc;
        }

        sweeps = it + 1;
        if (n_steep == 0) {
            /* Nothing was over-steep, so nothing was written; stop before
             * applying a sweep of zeros. */
            sweeps = it;
            break;
        }

        OMP_PARALLEL_LOOP
        for (anuga_int k = 0; k < n; k++) {
            const double d = dz[k];
            if (d != 0.0) {
                bed_cv[k] += d;
                const anuga_int k3 = 3 * k;
                bed_ev[k3 + 0] += d;
                bed_ev[k3 + 1] += d;
                bed_ev[k3 + 2] += d;
            }
        }
    }

    return sweeps;
}


// ============================================================================
// Suspended sediment source terms  (Phase 3)
// ============================================================================

// Apply the sediment exchange term E_s - D_s of [G-3], and the resulting bed
// change [G-4], for every registered sediment class.
//
// THIS IS A FRACTIONAL STEP. It is called ONCE PER TIMESTEP with the full dt,
// after the hydrodynamic step, not inside the RK substep loop. So it updates
// the STATE directly (m and z) rather than contributing to a tendency:
//
//     m_s  <-  m_s + dt (E_s - D_s)                       [G-3] source part
//     z    <-  z   + dt (D - E)/(1 - lambda)              [G-4]
//
// The two use the SAME limited source, so the sediment volume leaving
// suspension equals (1 - lambda) dz exactly and the budget closes by
// construction, whatever the timestepping method.
//
// Stage is deliberately NOT adjusted here: holding w while z rises makes the
// depth h = w - z fall by exactly dz, which is the quiescent-water behaviour
// LM15 Example 2 requires (their free surface stays flat while the bed
// aggrades). The pore space in the new bed is filled from the water column.
//
// Phase 3 is the FIXED-BED stage of spec 2.4: the bed does not evolve, there is
// no bed -> flow feedback and no sediment -> momentum feedback. Deposited mass
// simply leaves suspension.
//
// Deposition is [D-1]:      D_s = d*(Z) . c_s . v_s
// with d* the near-bed to depth-averaged concentration ratio (1.0 = well
// mixed). Erosion E_s is Phase 3b and is zero here.
//
// TWO LIMITERS, both from spec 4.5, and both applied to the SOURCE TERM rather
// than by clamping the state -- clamping m would break the exact conservation
// the transport scheme provides:
//
//   [L-1] positivity, mandatory:  F_s^net >= -m_s / dt
//         deposition can never remove more sediment than is present.
//   [L-2] concentration ceiling:  c_s <= c_max
//         applied as a cap on how much a cell may GAIN this step.
//
// Called after the flux kernel has filled tracer_explicit_update and before the
// time integration consumes it, so the source lands in the same dm/dt the
// integrator already applies.
void core_apply_sediment_source(struct domain *D, double timestep) {
    const anuga_int n_classes = D->n_sediment_classes;
    if (n_classes <= 0 || timestep <= 0.0) {
        return;
    }

    const anuga_int n = D->number_of_elements;
    const double c_max = D->sediment_c_max;
    const double minimum_allowed_height = D->minimum_allowed_height;

    const double gamma0 = D->sediment_gamma0;
    const anuga_int erosion_mode = D->sediment_erosion_mode;
    const double tau_crit = D->sediment_tau_crit;
    const double K_e = D->sediment_K_e;
    const double rho_w = D->sediment_rho_w;
    const double K_p = D->sediment_K_partheniades;
    const anuga_int dep_mode = D->sediment_deposition_mode;
    const double tau_d = D->sediment_tau_d;
    const anuga_int shear_closure = D->sediment_shear_closure;
    const double h_eps = D->epsilon;
    const double grav = D->g;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict friction_cv = D->friction_centroid_values;
    double * restrict v_s = D->sediment_settling_velocity;
    double * restrict d_star = D->sediment_d_star;
    double * restrict diam = D->sediment_diameter;
    double * restrict sedR = D->sediment_R;
    double * restrict tau_c_star = D->sediment_tau_c_star;
    double * restrict a_ref = D->sediment_reference_height;
    double * restrict bed_ev_r = D->bed_edge_values;
    double * restrict normals_r = D->normals;
    double * restrict edgelengths_r = D->edgelengths;
    double * restrict areas_r = D->areas;
    const anuga_int d_star_mode = D->sediment_d_star_mode;
    const double a_h_floor = D->sediment_a_h_floor;
    const double c_pack = D->sediment_c_pack;
    const anuga_int fric_mode = D->sediment_friction_mode;
    const double n_ll = D->sediment_manning_ll;
    const anuga_int wbed = D->sediment_wilson_bed;
    const double wD = D->sediment_wilson_D;

    // Hoisted for the same reason as in the update/backup/saxpy kernels: on a
    // GPU build the loop below is an 'omp target' region and D is NOT mapped to
    // the device, so a D->member load inside it reads a host address and the
    // work silently does not happen. See HANDOVER.md 2.4.
    double * restrict t_cons = D->tracer_conserved_values;
    double * restrict ext_src = D->tracer_external_source;
    double * restrict bed_cv_w = D->bed_centroid_values;
    double * restrict bed_ev_w = D->bed_edge_values;
    const double one_minus_lambda = 1.0 - D->sediment_porosity;
    const anuga_int bed_evolves = D->sediment_bed_evolution;
    /* [L-5]. Hoisted for the same device reason as the tracer pointers.
     *
     * The flag alone does not license the dereference: it and the pointer are
     * set by separate lines of the Cython binding, and when one of them was
     * missed the flag read as uninitialised garbage, tested true, and this
     * kernel dereferenced a NULL base. Require both. */
    double * restrict z_base = D->sediment_z_base;
    const anuga_int has_z_base = (D->sediment_has_z_base && z_base != NULL);
    double * restrict src_lim = D->sediment_source_limited;

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        const double h = fmax(stage_cv[k] - bed_cv[k], 0.0);

        // Dry cells carry no sediment: spec 9.4 sets c_s = 0 below the wet/dry
        // threshold. Leave the advective tendency untouched and add nothing.
        if (h <= minimum_allowed_height) {
            continue;
        }

        const double inv_h = 1.0 / h;
        const double m_max = c_max * h;
        double dz_cell = 0.0;
        double total_E = 0.0;   /* [L-5]: erosive demand on the bed */
        double total_D = 0.0;   /* [L-5]: what deposition returns to it */

        // Velocity by the ANUGA depth-limiting form [T-5], the same
        // regularisation RDy26 A22-A23 adopt. Never a bare (uh)/h.
        const double denom = h * h + h_eps * h_eps;
        const double u = (denom > 0.0) ? (xmom_cv[k] * h / denom) : 0.0;
        const double v = (denom > 0.0) ? (ymom_cv[k] * h / denom) : 0.0;
        const double vel2 = u * u + v * v;

        // Friction closure, spec 3.3. In EVERY mode f_c varies per cell per
        // timestep -- through h in [T-6] for the Manning-based modes, and
        // through the relative submergence h/D for wilson. Recomputed here
        // rather than cached, which spec 3.3 calls the coupling most easily
        // missed.
        double f_c;
        if (fric_mode == 2) {
            // wilson [T-8]..[T-10]. W04's equations give X = (8/f_W04)^1/2
            // with f_W04 the Darcy-Weisbach f; ours is f/8, so f_c = 1/X^2.
            // See the note in sw_domain.h: taking W04's f_c literally as ours
            // would make tau_b 8x too large.
            //
            // The relations assume the clasts are submerged. Below h = D the
            // logarithms go to zero or negative and the power law leaves its
            // calibration, so relative submergence is floored at 1.
            double rel = h / wD;
            if (!(rel > 1.0)) rel = 1.0;
            double X;
            if (wbed == 0) {
                X = 8.46 * pow(rel, 0.1005);           /* sand,    [T-8]  */
            } else if (wbed == 1) {
                X = 5.75 * log10(rel) + 3.514;         /* gravel,  [T-9]  */
            } else {
                X = 5.62 * log10(rel) + 4.0;           /* boulder, [T-10] */
            }
            f_c = 1.0 / (X * X);
        } else {
            // [T-6] f_c = g n^2 h^(-1/3) == RDy26 A5, with n either the
            // per-cell user field (constant mode) or the uniform Manning-
            // Strickler value of [T-14] (larsen_lamb).
            const double nman = (fric_mode == 1) ? n_ll : friction_cv[k];
            f_c = grav * nman * nman / cbrt(h);
        }

        /* tau_b/rho under the selected closure, [T-1] or [T-7]. */
        const double tbr = core_tau_b_over_rho(shear_closure, f_c, vel2, grav,
                                               h, bed_ev_r, normals_r,
                                               edgelengths_r, areas_r[k], k);

        for (anuga_int s = 0; s < n_classes; s++) {
            const anuga_int idx = s * n + k;

            const double m = t_cons[idx];
            const double c = m * inv_h;

            // Deposition is computed from a NON-NEGATIVE concentration.
            //
            // m can go slightly negative through the ADVECTIVE tendency (the
            // transport scheme guarantees positivity only under CFL, and the
            // source is added to a tendency it does not control). Fed through
            // unguarded, deposition = d* c v_s flips sign and starts ADDING
            // sediment -- and [L-1] below compounds it, because -m/dt is a
            // POSITIVE lower bound when m < 0, which forces the source
            // positive. Together they created 957% of the initial mass in a
            // deposition-only run. Both paths are guarded here.
            const double m_pos = (m > 0.0) ? m : 0.0;
            const double c_pos = m_pos * inv_h;

            // [D-1] deposition. d* is either the constant (P14's d* = 1
            // limiting case) or the Rouse ratio evaluated per cell.
            double ds;
            if (d_star_mode == 0) {
                ds = d_star[s];
            } else {
                // [T-2] u* = |v| sqrt(f_c);  [S-2] Z = v_s / (kappa u*)
                const double ustar = sqrt(f_c * vel2);
                const double Z = (ustar > 0.0)
                               ? v_s[s] / (0.41 * ustar)
                               : ANUGA_ROUSE_Z_HI;   /* no shear: fully settled */
                // a/h with the van Rijn-style floor a >= floor*h. The floor is
                // standard practice and stays on by default, but it is exposed:
                // it is the single largest divergence from anugaSed, which uses
                // no floor and so an ~10x smaller a at h = 1 m, giving roughly
                // 8x more deposition (spec 12, D4b). Set it to 0 to reach that
                // regime; the fit now covers a/h down to 1e-3.
                double a_h = a_ref[s] * inv_h;
                if (a_h < a_h_floor) a_h = a_h_floor;
                ds = core_rouse_d_star(Z, a_h);
            }
            // [L-4] NEAR-BED CONCENTRATION IS BOUNDED BY PACKING.
            //
            // [D-1] is D = c_b v_s with c_b = d* c, and nothing in the spec
            // bounds c_b. It needs bounding. d* comes from the EQUILIBRIUM
            // Rouse profile, which is not valid as shear vanishes: at rest
            // u* -> 0, so Z -> infinity and d* -> its clamp (~250 at
            // a/h = 0.01), making the deposition rate enormous. A lake at rest
            // then deposits its entire suspended load in under a second,
            // instead of over the physical h/v_s.
            //
            // c_b is a concentration and cannot exceed maximum packing, the
            // same 0.65 that bounds E* in [E-1]. Capping c_b there keeps the
            // still-water limit sane while leaving the well-mixed and
            // moderate-Z regimes untouched, where d* c is far below packing.
            //
            // Added in PHYSICS_SPEC Draft 5 as [L-4]; it has no counterpart in
            // P14, FG21, RDy26, DL09 or aSM16, being required by combining an
            // equilibrium profile with a transient solver.
            double deposition;
            if (dep_mode == 1) {
                /* [D-2] RDy26's threshold form. tau_d = 0 disables deposition
                 * entirely -- the hook their passive benchmarks rely on. */
                const double tau_b_d = rho_w * tbr;
                deposition = (tau_d > 0.0 && tau_b_d < tau_d)
                           ? v_s[s] * c_pos * (1.0 - tau_b_d / tau_d)
                           : 0.0;
            } else {
                double c_bed = ds * c_pos;
                if (c_bed > c_pack) c_bed = c_pack;
                deposition = c_bed * v_s[s];
            }

            // [E-1]/[E-2] entrainment, non-cohesive (Shields) route.
            //
            //   tau* = f_c |v|^2 / (R g d)   [T-3] -- rho cancels
            //   S    = tau*/tau_c* - 1
            //   E*   = 0.65 gamma0 S / (1 + gamma0 S)   saturating
            //
            // Below threshold (S <= 0) there is no entrainment at all; this is
            // a genuine threshold, not a smooth roll-off.
            double erosion = 0.0;
            if (erosion_mode == 2) {
                /* [E-4] Partheniades. K_p is a MASS flux, so divide by the
                 * class density to get the volume flux the rest of the source
                 * term works in. rho_s = (R + 1) rho_w. */
                const double tau_b = rho_w * tbr;
                if (tau_crit > 0.0 && tau_b > tau_crit) {
                    const double rho_s = (sedR[s] + 1.0) * rho_w;
                    if (rho_s > 0.0) {
                        erosion = (K_p * (tau_b - tau_crit) / tau_crit) / rho_s;
                    }
                }
            } else if (erosion_mode == 1) {
                /* [E-3] cohesive, Hanson & Simon. DIMENSIONAL excess shear:
                 * tau_b = rho f_c |v|^2 [T-1], and E = K_e (tau_b - tau_c),
                 * zero below threshold. Note this is per class only through
                 * the loop -- tau_c and K_e are bed properties, not grain
                 * properties, which is precisely the cohesive premise. */
                const double tau_b = rho_w * tbr;
                const double excess = tau_b - tau_crit;
                if (excess > 0.0) {
                    erosion = K_e * excess;
                }
            } else {
                /* [E-1]/[E-2] non-cohesive, Shields route. */
                const double Rgd = sedR[s] * grav * diam[s];
                if (Rgd > 0.0 && tau_c_star[s] > 0.0) {
                    const double tau_star = tbr / Rgd;
                    const double S = tau_star / tau_c_star[s] - 1.0;
                    if (S > 0.0) {
                        const double gS = gamma0 * S;
                        erosion = v_s[s] * (0.65 * gS / (1.0 + gS));
                    }
                }
            }

            // Net exchange of [G-3]. Deposition removes, erosion adds.
            double source = erosion - deposition;

            // [L-1] positivity. The most this term may remove over the step is
            // exactly the sediment PRESENT, so the state can reach zero but
            // never go below it. Applied to the SOURCE, not to m.
            //
            // m_pos, not m: with m < 0 the bound -m/dt is POSITIVE and would
            // force the source to inject sediment. The limiter must only ever
            // restrain removal, never mandate addition.
            const double min_source = -m_pos / timestep;
            if (source < min_source) {
                source = min_source;
            }

            // [L-2] ceiling. Only ever restrains a GAIN, so it cannot fight
            // [L-1] above: the two act on opposite signs of the source.
            if (source > 0.0) {
                const double max_source = (m_max - m) / timestep;
                if (source > max_source) {
                    source = (max_source > 0.0) ? max_source : 0.0;
                }
            }

            // Held, not applied. [L-5] below limits the classes TOGETHER
            // against the cell's erodible thickness, so no class may be
            // applied until every class's demand on the bed is known.
            //
            // The external supply is deliberately NOT folded in here: it is
            // not a bed exchange, so it must not be scaled by a bed-material
            // limiter, and it is added in the apply loop instead.
            src_lim[idx] = source;
            if (source > 0.0) total_E += source;
            else              total_D += source;
        }

        // ---- [L-5] non-erodible base -----------------------------------
        //
        // The bed may be lowered to sediment_z_base and no further. Erosion
        // is a bed-material budget, so the limit belongs on the SOURCE, like
        // [L-1] and [L-2], and not on z: clamping z after the fact would
        // leave sediment in the water column that no longer came from
        // anywhere, which is exactly how [L-1]'s sign bug created 957% of
        // the initial mass.
        //
        // Only the EROSIVE part is scaled. Deposition is not restrained by a
        // shortage of bed material -- it is what supplies it -- and scaling
        // it down would suppress the very process that reopens the cell.
        //
        // The scale is shared and proportional, so the answer does not depend
        // on the order the classes were registered. There is no bed
        // stratigraphy in this model: the bed is not tracked per class, so
        // no class has a better claim on the last millimetre than another,
        // and proportional is the only choice that does not invent one.
        double scale = 1.0;
        if (has_z_base && bed_evolves && one_minus_lambda > 0.0
                && total_E > 0.0) {
            const double avail = bed_cv[k] - z_base[k];
            const double thickness = (avail > 0.0) ? avail : 0.0;
            // The largest net removal from the bed this step, as a source.
            const double S_max = thickness * one_minus_lambda / timestep;
            if (total_E + total_D > S_max) {
                scale = (S_max - total_D) / total_E;
                if (scale < 0.0) scale = 0.0;
                if (scale > 1.0) scale = 1.0;
            }
        }

        // ---- apply ------------------------------------------------------
        for (anuga_int s = 0; s < n_classes; s++) {
            const anuga_int idx = s * n + k;
            double source = src_lim[idx];
            if (source > 0.0) {
                source *= scale;
            }

            // [G-4]. source = E - D, so dz = -source dt/(1-lambda). Taken
            // from the bed exchange ALONE, before the external supply is
            // added: sediment introduced from outside the model does not
            // come out of the bed, so it must not move it.
            if (bed_evolves && one_minus_lambda > 0.0) {
                dz_cell += -(timestep * source) / one_minus_lambda;
            }

            // [G-3] S_ms: external supply, added AFTER the limiters. They
            // bound bed exchange by what bed and water column can supply;
            // an external source is neither, and clipping it would also make
            // a manufactured solution impossible to impose exactly.
            if (ext_src != NULL) {
                source += ext_src[idx];
            }

            // Fractional step: update the state directly with the full dt.
            t_cons[idx] += timestep * source;
        }
        // Raise the bed by dz. The DE algorithms use DISCONTINUOUS elevation,
        // so edge values are not re-derived from the centroid and must be
        // shifted too; shifting all three by the same dz preserves the
        // within-cell bed slope, which is what keeps a flat bed flat. Vertex
        // values need no action: extrapolation recomputes them from the edges
        // (bed_vv = bed_ev1 + bed_ev2 - bed_ev0).
        //
        // Stage is left alone, so h = w - z falls by exactly dz -- the
        // quiescent-water behaviour of LM15 Example 2.
        if (dz_cell != 0.0) {
            bed_cv_w[k] += dz_cell;
            const anuga_int k3 = 3 * k;
            bed_ev_w[k3 + 0] += dz_cell;
            bed_ev_w[k3 + 1] += dz_cell;
            bed_ev_w[k3 + 2] += dz_cell;
        }
    }
}

// ============================================================================
// Update conserved quantities
// ============================================================================

void core_update_conserved_quantities(struct domain *D, double timestep) {
    anuga_int n = D->number_of_elements;
    const anuga_int n_tracers = D->number_of_tracers;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;

    double * restrict stage_eu = D->stage_explicit_update;
    double * restrict xmom_eu = D->xmom_explicit_update;
    double * restrict ymom_eu = D->ymom_explicit_update;

    double * restrict stage_siu = D->stage_semi_implicit_update;
    double * restrict xmom_siu = D->xmom_semi_implicit_update;
    double * restrict ymom_siu = D->ymom_semi_implicit_update;

    // Tracer pointers are hoisted to FUNCTION SCOPE here, not loaded inside the
    // n_tracers > 0 guard as in the flux kernel. On a GPU build OMP_PARALLEL_LOOP
    // is 'omp target teams loop', and D itself is NOT mapped to the device, so a
    // D->member load inside the loop reads a host address on the device: the
    // tracer update silently does nothing (m never changes, while the flux
    // kernel still fills explicit_update). Hoisting lets the pointer values be
    // captured as firstprivate scalars and address-translated. The flux kernel's
    // in-guard loading is a CPU hot-loop optimisation (HANDOVER 2.4, +2.26%% at
    // Ns=0) and does not apply to these much cheaper elementwise loops.
#ifndef CPU_ONLY_MODE
    double * restrict t_cons = D->tracer_conserved_values;
    double * restrict t_eu   = D->tracer_explicit_update;
#endif

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

        // Tracers: integrate the conserved m = h*c. No semi-implicit term and
        // deliberately NO clamping -- clamping would break exact conservation.
        // Positivity is instead a property of the upwind flux under CFL, and is
        // asserted by the tests rather than enforced here.
        if (n_tracers > 0) {
#ifdef CPU_ONLY_MODE
            double * restrict t_cons = D->tracer_conserved_values;
            double * restrict t_eu   = D->tracer_explicit_update;
#endif
            for (anuga_int s = 0; s < n_tracers; s++) {
                t_cons[s * n + k] += timestep * t_eu[s * n + k];
            }
        }
    }
}

// ============================================================================
// Backup conserved quantities for RK2
// ============================================================================

void core_backup_conserved_quantities(struct domain *D) {
    anuga_int n = D->number_of_elements;
    const anuga_int n_tracers = D->number_of_tracers;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;

    double * restrict stage_bk = D->stage_backup_values;
    double * restrict xmom_bk = D->xmom_backup_values;
    double * restrict ymom_bk = D->ymom_backup_values;

    // Tracer pointers are hoisted to FUNCTION SCOPE here, not loaded inside the
    // n_tracers > 0 guard as in the flux kernel. On a GPU build OMP_PARALLEL_LOOP
    // is 'omp target teams loop', and D itself is NOT mapped to the device, so a
    // D->member load inside the loop reads a host address on the device: the
    // tracer update silently does nothing (m never changes, while the flux
    // kernel still fills explicit_update). Hoisting lets the pointer values be
    // captured as firstprivate scalars and address-translated. The flux kernel's
    // in-guard loading is a CPU hot-loop optimisation (HANDOVER 2.4, +2.26%% at
    // Ns=0) and does not apply to these much cheaper elementwise loops.
#ifndef CPU_ONLY_MODE
    double * restrict t_cons = D->tracer_conserved_values;
    double * restrict t_bk   = D->tracer_backup_values;
#endif

    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        stage_bk[k] = stage_cv[k];
        xmom_bk[k] = xmom_cv[k];
        ymom_bk[k] = ymom_cv[k];

        if (n_tracers > 0) {
#ifdef CPU_ONLY_MODE
            double * restrict t_cons = D->tracer_conserved_values;
            double * restrict t_bk   = D->tracer_backup_values;
#endif
            for (anuga_int s = 0; s < n_tracers; s++) {
                t_bk[s * n + k] = t_cons[s * n + k];
            }
        }
    }
}

// ============================================================================
// SAXPY for RK2/RK3: Q = (a*Q + b*Q_backup) / c
// ============================================================================

void core_saxpy_conserved_quantities(struct domain *D, double a, double b, double c) {
    anuga_int n = D->number_of_elements;
    const anuga_int n_tracers = D->number_of_tracers;

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;

    double * restrict stage_bk = D->stage_backup_values;
    double * restrict xmom_bk = D->xmom_backup_values;
    double * restrict ymom_bk = D->ymom_backup_values;

    // Tracer pointers are hoisted to FUNCTION SCOPE here, not loaded inside the
    // n_tracers > 0 guard as in the flux kernel. On a GPU build OMP_PARALLEL_LOOP
    // is 'omp target teams loop', and D itself is NOT mapped to the device, so a
    // D->member load inside the loop reads a host address on the device: the
    // tracer update silently does nothing (m never changes, while the flux
    // kernel still fills explicit_update). Hoisting lets the pointer values be
    // captured as firstprivate scalars and address-translated. The flux kernel's
    // in-guard loading is a CPU hot-loop optimisation (HANDOVER 2.4, +2.26%% at
    // Ns=0) and does not apply to these much cheaper elementwise loops.
#ifndef CPU_ONLY_MODE
    double * restrict t_cons = D->tracer_conserved_values;
    double * restrict t_bk   = D->tracer_backup_values;
#endif

    // Standard SAXPY: Q = a*Q + b*Q_backup
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        stage_cv[k] = a * stage_cv[k] + b * stage_bk[k];
        xmom_cv[k] = a * xmom_cv[k] + b * xmom_bk[k];
        ymom_cv[k] = a * ymom_cv[k] + b * ymom_bk[k];

        // SAXPY must act on the CONSERVED m, not on c: h differs between RK
        // stages, so averaging c would not average the transported mass.
        if (n_tracers > 0) {
#ifdef CPU_ONLY_MODE
            double * restrict t_cons = D->tracer_conserved_values;
            double * restrict t_bk   = D->tracer_backup_values;
#endif
            for (anuga_int s = 0; s < n_tracers; s++) {
                t_cons[s * n + k] = a * t_cons[s * n + k] + b * t_bk[s * n + k];
            }
        }
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
            if (n_tracers > 0) {
#ifdef CPU_ONLY_MODE
                double * restrict t_cons = D->tracer_conserved_values;
#endif
                for (anuga_int s = 0; s < n_tracers; s++) t_cons[s * n + k] *= c_inv;
            }
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

    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict bed_cv = D->bed_centroid_values;
    double * restrict xmom_cv = D->xmom_centroid_values;
    double * restrict ymom_cv = D->ymom_centroid_values;
    double * restrict friction_cv = D->friction_centroid_values;

    double * restrict xmom_siu = D->xmom_semi_implicit_update;
    double * restrict ymom_siu = D->ymom_semi_implicit_update;

    // Use SIMD explicitly: cbrt/sqrt need the simd hint for vectorization on
    // all compilers (ICX auto-vectorizes these math calls better with simd).
    OMP_PARALLEL_LOOP_SIMD
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
                // h^(7/3) = h^2 * cbrt(h): avoids slow general pow() across all compilers
                S /= (h * h * cbrt(h));
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

    OMP_PARALLEL_LOOP_SIMD
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
            S /= (h * h * cbrt(h));

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

    const double one_third = 1.0 / 3.0;

    OMP_PARALLEL_LOOP_SIMD
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
                S /= (h * h * cbrt(h));
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

    // Generic passive tracers.  n_tracers == 0 in every ordinary run;
    // all tracer work below is guarded on this loop-invariant integer.
    const anuga_int n_tracers = D->number_of_tracers;

// Tracer base pointers: WHERE they are loaded is build-dependent, and both
// choices are load-bearing.
//
//   CPU build  -- load them INSIDE the n_tracers > 0 guard. Hoisting them to
//                 function scope keeps them live across the hot loop and cost
//                 +2.26% on the CPU path at Ns=0 (HANDOVER.md 2.4). Only the
//                 benchmark caught that; every correctness test passed.
//   GPU build  -- the loops below are 'omp target' regions and D is NOT mapped
//                 to the device, so a D->member load inside the region reads a
//                 host address on the device. The tracer work then silently
//                 does nothing: no crash, explicit_update stays zero on the
//                 device, and m never moves. They must be loaded at function
//                 scope so the pointer VALUES are captured as firstprivate
//                 scalars and address-translated via the present table.
//
// So: hoisted declarations under #ifndef CPU_ONLY_MODE, in-guard declarations
// under #ifdef CPU_ONLY_MODE. The loop bodies are identical either way.
#ifndef CPU_ONLY_MODE
    double * restrict t_eu = D->tracer_explicit_update;
    double * restrict t_ev = D->tracer_edge_values;
    double * restrict t_bv = D->tracer_boundary_values;
    double * restrict t_bf = D->tracer_boundary_flux;
#endif

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
        if (n_tracers > 0) {
#ifdef CPU_ONLY_MODE
            double * restrict t_eu = D->tracer_explicit_update;
#endif
            for (anuga_int s = 0; s < n_tracers; s++) t_eu[s * n + k] = 0.0;
        }

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

            // --- Passive tracer advection -------------------------------
            // edgeflux[0] is the water mass flux through this edge, already
            // multiplied by -length.  Sign convention after that negation:
            //     edgeflux[0] < 0  ->  OUTflow from k, donor is k
            //     edgeflux[0] > 0  ->  INflow  to   k, donor is the neighbour
            // Using the same edgeflux[0] for both cells sharing the edge makes
            // tracer mass conservation structural, independent of cell sizes.
            if (n_tracers > 0) {
#ifdef CPU_ONLY_MODE
                double * restrict t_ev = D->tracer_edge_values;
                double * restrict t_bv = D->tracer_boundary_values;
                double * restrict t_eu = D->tracer_explicit_update;
                double * restrict t_bf = D->tracer_boundary_flux;
#endif
                const anuga_int t_bl = D->boundary_length;
                const double wflux = edgeflux[0];
                const int    inflow = (wflux > 0.0);
                /* Conservation accounting: record what crosses a DOMAIN boundary edge,
                 * on the same terms the water balance uses -- a real boundary, and this
                 * cell owned rather than a ghost. A ghost cell's copy of the edge
                 * belongs to its owner, so counting it here would double it. */
                const int count_bdry = (t_bf != NULL) && is_boundary
                    && (tri_full_flag == NULL || tri_full_flag[k] == 1);
                for (anuga_int s = 0; s < n_tracers; s++) {
                    double c_up;
                    if (inflow) {
                        c_up = is_boundary
                             ? t_bv[s * t_bl + (-neighbour - 1)]
                             : t_ev[s * 3 * n + neighbour * 3 + neighbour_edges[ki]];
                    } else {
                        c_up = t_ev[s * 3 * n + ki];
                    }
                    t_eu[s * n + k] += wflux * c_up;
                    if (count_bdry) {
                        /* Same sign as edgeflux[0]: positive is inflow. */
                        t_bf[s * n + k] += wflux * c_up;
                    }
                }
            }
            // -------------------------------------------------------------

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
        if (n_tracers > 0) {
#ifdef CPU_ONLY_MODE
            double * restrict t_eu = D->tracer_explicit_update;
#endif
            for (anuga_int s = 0; s < n_tracers; s++) t_eu[s * n + k] *= inv_area;
        }

    } // End element loop

    // Store boundary flux sum for this substep
    if (D->boundary_flux_sum != NULL && substep_count < timestep_fluxcalls) {
        D->boundary_flux_sum[substep_count] = boundary_flux_sum_substep;
    }

    /* Same, per tracer. The main loop accumulated per cell so it needed no
     * reduction; total it here with ONE SCALAR REDUCTION PER TRACER, which is
     * a pattern every target supports, instead of the runtime-length array
     * reduction the main loop would have required. n_tracers is small, and the
     * whole block is skipped when there are none. The scratch is zeroed in the
     * same pass, ready for the next substep. */
    if (n_tracers > 0 && D->tracer_boundary_flux != NULL
            && D->tracer_boundary_flux_sum != NULL
            && substep_count < timestep_fluxcalls) {
#ifdef CPU_ONLY_MODE
        /* The GPU build has this hoisted to function scope (see the note on
         * tracer base pointers above); the CPU build loads it in-guard inside
         * the element loop, so it needs its own here. */
        double * restrict t_bf = D->tracer_boundary_flux;
#endif
        for (anuga_int s = 0; s < n_tracers; s++) {
            double tsum = 0.0;
            const anuga_int off = s * n;
#ifdef CPU_ONLY_MODE
            #pragma omp parallel for reduction(+:tsum)
#else
            #pragma omp target teams distribute parallel for reduction(+:tsum)
#endif
            for (anuga_int k = 0; k < n; k++) {
                tsum += t_bf[off + k];
                t_bf[off + k] = 0.0;
            }
            D->tracer_boundary_flux_sum[substep_count * n_tracers + s] = tsum;
        }
    }

    // Return timestep (only meaningful on first substep)
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
