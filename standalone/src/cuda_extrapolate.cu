// CUDA port of the fused reconstruction kernel (edge pass + C-K predictor).
//
// PURPOSE: an instrument, not a port.  The OpenMP version of this kernel has
// resisted register capping, quantity-splitting and gather packing (all
// measured losses -- see the README); this file exists to answer ONE
// question: does hand-written CUDA with explicit launch control beat nvc's
// `omp target teams loop` code for the same arithmetic?  It lives in the
// miniapp only and must never migrate into the shared ANUGA kernels.
//
// The arithmetic is a LITERAL transcription of core_extrapolate_edge_pass()
// (keep in sync manually -- the harness's golden checks catch divergence).
// This file is built as a PURE nvcc shared library and dlopen()ed by the
// bench: every attempt to link CUDA objects into the OpenMP-target binary
// broke nvomp's offload registration (or ICEd nvc), so the two runtimes only
// meet through the CUDA primary context, which they share.  The caller
// resolves device pointers with omp_get_mapped_ptr() and passes them in;
// target regions are synchronous, so the cudaDeviceSynchronize() here is all
// the ordering the step needs.
//
// The `static -> static __device__` define below turns the shared C helpers
// in gpu_device_helpers.h into device functions without editing the header.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "cuda_extrap.h"

#define restrict __restrict__
#define static static __device__
#include "gpu_device_helpers.h"
#undef static
#undef restrict


__global__ void extrapolate_ck_kernel(struct extrap_args a) {
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= a.n) return;

    const double minimum_allowed_height = a.minimum_allowed_height;

    // hfactor parameters (match the C kernel exactly)
    const double a_tmp = 0.3;
    const double b_tmp = 0.1;
    const double c_tmp = 1.0 / (a_tmp - b_tmp);
    const double d_tmp = 1.0 - (c_tmp * a_tmp);
    (void)b_tmp;

    const long long k2 = k * 2;
    const long long k3 = k * 3;
    const long long k6 = k * 6;

    double xv0 = a.edge_coords[k6 + 0];
    double yv0 = a.edge_coords[k6 + 1];
    double xv1 = a.edge_coords[k6 + 2];
    double yv1 = a.edge_coords[k6 + 3];
    double xv2 = a.edge_coords[k6 + 4];
    double yv2 = a.edge_coords[k6 + 5];

    double x = a.centroid_coords[k2 + 0];
    double y = a.centroid_coords[k2 + 1];

    double dxv0 = xv0 - x;
    double dxv1 = xv1 - x;
    double dxv2 = xv2 - x;
    double dyv0 = yv0 - y;
    double dyv1 = yv1 - y;
    double dyv2 = yv2 - y;

    long long k0 = a.surrogate_neighbours[k3 + 0];
    long long k1 = a.surrogate_neighbours[k3 + 1];
    long long sn2 = a.surrogate_neighbours[k3 + 2];

    double x0 = a.centroid_coords[2 * k0 + 0];
    double y0 = a.centroid_coords[2 * k0 + 1];
    double x1 = a.centroid_coords[2 * k1 + 0];
    double y1 = a.centroid_coords[2 * k1 + 1];
    double x2 = a.centroid_coords[2 * sn2 + 0];
    double y2 = a.centroid_coords[2 * sn2 + 1];

    double dx1 = x1 - x0;
    double dx2 = x2 - x0;
    double dy1 = y1 - y0;
    double dy2 = y2 - y0;

    double area2 = dy2 * dx1 - dy1 * dx2;

    int dry = ((a.height_cv[k0] < minimum_allowed_height) || (k0 == k)) &&
              ((a.height_cv[k1] < minimum_allowed_height) || (k1 == k)) &&
              ((a.height_cv[sn2] < minimum_allowed_height) || (sn2 == k));

    if (dry) {
        a.x_centroid_work[k] = 0.0;
        a.xmom_cv[k] = 0.0;
        a.y_centroid_work[k] = 0.0;
        a.ymom_cv[k] = 0.0;
    }

    long long num_boundaries = a.number_of_boundaries[k];

    if (num_boundaries == 3) {
        double stage_c = a.stage_cv[k];
        double xmom_c = a.x_centroid_work[k];
        double ymom_c = a.y_centroid_work[k];
        double height_c = a.height_cv[k];
        double bed_c = a.bed_cv[k];

        for (int i = 0; i < 3; i++) {
            a.stage_ev[k3 + i] = stage_c;
            a.xmom_ev[k3 + i] = xmom_c;
            a.ymom_ev[k3 + i] = ymom_c;
            a.height_ev[k3 + i] = height_c;
            a.bed_ev[k3 + i] = bed_c;
        }

    } else if (num_boundaries <= 1) {
        double hc = a.height_cv[k];
        double h0 = a.height_cv[k0];
        double h1 = a.height_cv[k1];
        double h2 = a.height_cv[sn2];

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
        double beta_stage = a.beta_w_dry + (a.beta_w - a.beta_w_dry) * hfactor;
        if (beta_stage > 0.0) {
            gpu_calc_edge_values_with_gradient(
                a.stage_cv[k], a.stage_cv[k0], a.stage_cv[k1], a.stage_cv[sn2],
                dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                dx1, dx2, dy1, dy2, inv_area2, beta_stage, edge_vals);
        } else {
            gpu_set_constant_edge_values(a.stage_cv[k], edge_vals);
        }
        a.stage_ev[k3 + 0] = edge_vals[0];
        a.stage_ev[k3 + 1] = edge_vals[1];
        a.stage_ev[k3 + 2] = edge_vals[2];

        // Height (same beta as stage)
        if (beta_stage > 0.0) {
            gpu_calc_edge_values_with_gradient(
                a.height_cv[k], a.height_cv[k0], a.height_cv[k1], a.height_cv[sn2],
                dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                dx1, dx2, dy1, dy2, inv_area2, beta_stage, edge_vals);
        } else {
            gpu_set_constant_edge_values(a.height_cv[k], edge_vals);
        }
        a.height_ev[k3 + 0] = edge_vals[0];
        a.height_ev[k3 + 1] = edge_vals[1];
        a.height_ev[k3 + 2] = edge_vals[2];

        // X-momentum (velocity while extrapolating velocity)
        double beta_xmom = a.beta_uh_dry + (a.beta_uh - a.beta_uh_dry) * hfactor;
        if (beta_xmom > 0.0) {
            gpu_calc_edge_values_with_gradient(
                a.x_centroid_work[k], a.x_centroid_work[k0], a.x_centroid_work[k1], a.x_centroid_work[sn2],
                dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                dx1, dx2, dy1, dy2, inv_area2, beta_xmom, edge_vals);
        } else {
            gpu_set_constant_edge_values(a.x_centroid_work[k], edge_vals);
        }
        a.xmom_ev[k3 + 0] = edge_vals[0];
        a.xmom_ev[k3 + 1] = edge_vals[1];
        a.xmom_ev[k3 + 2] = edge_vals[2];

        // Y-momentum
        double beta_ymom = a.beta_vh_dry + (a.beta_vh - a.beta_vh_dry) * hfactor;
        if (beta_ymom > 0.0) {
            gpu_calc_edge_values_with_gradient(
                a.y_centroid_work[k], a.y_centroid_work[k0], a.y_centroid_work[k1], a.y_centroid_work[sn2],
                dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                dx1, dx2, dy1, dy2, inv_area2, beta_ymom, edge_vals);
        } else {
            gpu_set_constant_edge_values(a.y_centroid_work[k], edge_vals);
        }
        a.ymom_ev[k3 + 0] = edge_vals[0];
        a.ymom_ev[k3 + 1] = edge_vals[1];
        a.ymom_ev[k3 + 2] = edge_vals[2];

    } else {
        // num_boundaries == 2: gradient toward the single internal neighbour
        long long kn = k;
        for (int i = 0; i < 3; i++) {
            long long sn = a.surrogate_neighbours[k3 + i];
            if (sn != k) {
                kn = sn;
                break;
            }
        }

        double xn = a.centroid_coords[2 * kn + 0];
        double yn = a.centroid_coords[2 * kn + 1];
        double dx = xn - x;
        double dy = yn - y;
        double dist2 = dx * dx + dy * dy;

        double grad_dx2 = (dist2 > 0.0) ? dx / dist2 : 0.0;
        double grad_dy2 = (dist2 > 0.0) ? dy / dist2 : 0.0;

        double dqv[3], qmin, qmax, dq1;

        // Stage
        dq1 = a.stage_cv[kn] - a.stage_cv[k];
        gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                      dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
        gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
        gpu_limit_gradient(dqv, qmin, qmax, a.beta_w);
        a.stage_ev[k3 + 0] = a.stage_cv[k] + dqv[0];
        a.stage_ev[k3 + 1] = a.stage_cv[k] + dqv[1];
        a.stage_ev[k3 + 2] = a.stage_cv[k] + dqv[2];

        // Height
        dq1 = a.height_cv[kn] - a.height_cv[k];
        gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                      dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
        gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
        gpu_limit_gradient(dqv, qmin, qmax, a.beta_w);
        a.height_ev[k3 + 0] = a.height_cv[k] + dqv[0];
        a.height_ev[k3 + 1] = a.height_cv[k] + dqv[1];
        a.height_ev[k3 + 2] = a.height_cv[k] + dqv[2];

        // X-momentum
        dq1 = a.x_centroid_work[kn] - a.x_centroid_work[k];
        gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                      dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
        gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
        gpu_limit_gradient(dqv, qmin, qmax, a.beta_w);
        a.xmom_ev[k3 + 0] = a.x_centroid_work[k] + dqv[0];
        a.xmom_ev[k3 + 1] = a.x_centroid_work[k] + dqv[1];
        a.xmom_ev[k3 + 2] = a.x_centroid_work[k] + dqv[2];

        // Y-momentum
        dq1 = a.y_centroid_work[kn] - a.y_centroid_work[k];
        gpu_compute_dqv_from_gradient(dq1, grad_dx2, grad_dy2,
                                      dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dqv);
        gpu_compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);
        gpu_limit_gradient(dqv, qmin, qmax, a.beta_w);
        a.ymom_ev[k3 + 0] = a.y_centroid_work[k] + dqv[0];
        a.ymom_ev[k3 + 1] = a.y_centroid_work[k] + dqv[1];
        a.ymom_ev[k3 + 2] = a.y_centroid_work[k] + dqv[2];
    }

    // Convert velocity edge values back to momentum if needed
    if (a.extrapolate_velocity_second_order == 1) {
        for (int i = 0; i < 3; i++) {
            double dk = a.height_ev[k3 + i];
            a.xmom_ev[k3 + i] *= dk;
            a.ymom_ev[k3 + i] *= dk;
        }
    }

    // Compute bed edge values from stage - height
    for (int i = 0; i < 3; i++) {
        a.bed_ev[k3 + i] = a.stage_ev[k3 + i] - a.height_ev[k3 + i];
    }

    // ---- fused ADER-2 C-K edge predictor (transcribed from the C kernel)
    if (a.predictor_dt != 0.0) {
        double det_p = dxv0 * dyv1 - dxv1 * dyv0;
        if (fabs(det_p) >= 1.0e-20) {
            double inv_det = 1.0 / det_p;

            double w_c  = a.stage_cv[k];
            double h_c  = fmax(w_c - a.bed_cv[k], 0.0);
            double uh_c = a.xmom_cv[k];
            double vh_c = a.ymom_cv[k];

            double inv_h_c = (h_c > minimum_allowed_height) ? 1.0 / h_c : 0.0;
            double u_c = uh_c * inv_h_c;
            double v_c = vh_c * inv_h_c;

            double dw0 = a.stage_ev[k3 + 0] - w_c;
            double dw1 = a.stage_ev[k3 + 1] - w_c;
            double wx  = inv_det * (dyv1 * dw0 - dyv0 * dw1);
            double wy  = inv_det * (dxv0 * dw1 - dxv1 * dw0);

            double dh0 = a.height_ev[k3 + 0] - h_c;
            double dh1 = a.height_ev[k3 + 1] - h_c;
            double hx  = inv_det * (dyv1 * dh0 - dyv0 * dh1);
            double hy  = inv_det * (dxv0 * dh1 - dxv1 * dh0);

            double h_e0     = a.height_ev[k3 + 0];
            double h_e1     = a.height_ev[k3 + 1];
            double inv_h_e0 = (h_e0 > minimum_allowed_height) ? 1.0 / h_e0 : 0.0;
            double inv_h_e1 = (h_e1 > minimum_allowed_height) ? 1.0 / h_e1 : 0.0;
            double u_e0 = a.xmom_ev[k3 + 0] * inv_h_e0;
            double u_e1 = a.xmom_ev[k3 + 1] * inv_h_e1;
            double v_e0 = a.ymom_ev[k3 + 0] * inv_h_e0;
            double v_e1 = a.ymom_ev[k3 + 1] * inv_h_e1;

            double du0 = u_e0 - u_c;
            double du1 = u_e1 - u_c;
            double dv0 = v_e0 - v_c;
            double dv1 = v_e1 - v_c;
            double ux  = inv_det * (dyv1 * du0 - dyv0 * du1);
            double uy  = inv_det * (dxv0 * du1 - dxv1 * du0);
            double vx  = inv_det * (dyv1 * dv0 - dyv0 * dv1);
            double vy  = inv_det * (dxv0 * dv1 - dxv1 * dv0);

            double g_h = a.g * h_c;
            double dw_dt  = -(u_c * hx + h_c * ux + v_c * hy + h_c * vy);
            double duh_dt = -(2.0*u_c*h_c*ux + u_c*u_c*hx + u_c*v_c*hy
                             + v_c*h_c*uy + u_c*h_c*vy + g_h * wx);
            double dvh_dt = -(v_c*h_c*ux + u_c*h_c*vx + u_c*v_c*hx
                             + 2.0*v_c*h_c*vy + v_c*v_c*hy + g_h * wy);

            for (int i = 0; i < 3; i++) {
                a.stage_ev[k3 + i] += a.predictor_dt * dw_dt;
                a.xmom_ev[k3 + i] += a.predictor_dt * duh_dt;
                a.ymom_ev[k3 + i] += a.predictor_dt * dvh_dt;
                a.height_ev[k3 + i] = fmax(a.height_ev[k3 + i] + a.predictor_dt * dw_dt, 0.0);
            }
        }
    }
}

extern "C" int cuda_extrapolate_launch(struct extrap_args a, int tpb) {
    const int blocks = (int)((a.n + tpb - 1) / tpb);
    extrapolate_ck_kernel<<<blocks, tpb>>>(a);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_extrapolate: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
