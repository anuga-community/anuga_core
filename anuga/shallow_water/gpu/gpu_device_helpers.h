// GPU device helper functions for OpenMP target offloading
//
// These are inline functions marked with #pragma omp declare target
// for execution on the GPU. They are shared across kernel files.

#ifndef GPU_DEVICE_HELPERS_H
#define GPU_DEVICE_HELPERS_H

#include <math.h>
#include "gpu_domain.h"

// ============================================================================
// FLOP Counting Constants (Gordon Bell Performance Profiling)
// ============================================================================
//
// FLOP counts per kernel per element (triangle):
// Counted: +, -, *, /, sqrt, fmin, fmax, pow (comparisons not counted)

#define FLOPS_EXTRAPOLATE        220
#define FLOPS_COMPUTE_FLUXES     400
#define FLOPS_UPDATE             21
#define FLOPS_PROTECT            5
#define FLOPS_MANNING            18
#define FLOPS_BACKUP             0
#define FLOPS_SAXPY              9
#define FLOPS_RATE_OPERATOR      8
#define FLOPS_GHOST_EXCHANGE     0

// ============================================================================
// Device Helper Functions
// ============================================================================

#pragma omp declare target

// Small constant for avoiding division by zero
static const double GPU_TINY = 1.0e-100;

// ============================================================================
// Extrapolation Helper Functions (device code)
// ============================================================================

// Find qmin and qmax from three differences
static inline void gpu_find_qmin_and_qmax_dq1_dq2(
    double dq0, double dq1, double dq2,
    double *qmin, double *qmax) {
    *qmax = fmax(fmax(dq0, fmax(dq0 + dq1, dq0 + dq2)), 0.0);
    *qmin = fmin(fmin(dq0, fmin(dq0 + dq1, dq0 + dq2)), 0.0);
}

// Find qmin and qmax from single difference
static inline void gpu_compute_qmin_qmax_from_dq1(double dq1, double *qmin, double *qmax) {
    if (dq1 >= 0.0) {
        *qmin = 0.0;
        *qmax = dq1;
    } else {
        *qmin = dq1;
        *qmax = 0.0;
    }
}

// Limit gradient to enforce monotonicity
static inline void gpu_limit_gradient(double *dqv, double qmin, double qmax, double beta_w) {
    double r = 1000.0;

    double dq_x = dqv[0];
    double dq_y = dqv[1];
    double dq_z = dqv[2];

    if (dq_x < -GPU_TINY) {
        r = fmin(r, qmin / dq_x);
    } else if (dq_x > GPU_TINY) {
        r = fmin(r, qmax / dq_x);
    }
    if (dq_y < -GPU_TINY) {
        r = fmin(r, qmin / dq_y);
    } else if (dq_y > GPU_TINY) {
        r = fmin(r, qmax / dq_y);
    }
    if (dq_z < -GPU_TINY) {
        r = fmin(r, qmin / dq_z);
    } else if (dq_z > GPU_TINY) {
        r = fmin(r, qmax / dq_z);
    }

    double phi = fmin(r * beta_w, 1.0);

    dqv[0] *= phi;
    dqv[1] *= phi;
    dqv[2] *= phi;
}

// Compute edge values with gradient limiting (for typical case with 3 neighbors)
static inline void gpu_calc_edge_values_with_gradient(
    double cv_k, double cv_k0, double cv_k1, double cv_k2,
    double dxv0, double dxv1, double dxv2,
    double dyv0, double dyv1, double dyv2,
    double dx1, double dx2, double dy1, double dy2,
    double inv_area2, double beta_tmp, double *edge_values) {

    double dqv[3];
    double dq0 = cv_k0 - cv_k;
    double dq1 = cv_k1 - cv_k0;
    double dq2 = cv_k2 - cv_k0;

    double a = (dy2 * dq1 - dy1 * dq2) * inv_area2;
    double b = (dx1 * dq2 - dx2 * dq1) * inv_area2;

    dqv[0] = a * dxv0 + b * dyv0;
    dqv[1] = a * dxv1 + b * dyv1;
    dqv[2] = a * dxv2 + b * dyv2;

    double qmin, qmax;
    gpu_find_qmin_and_qmax_dq1_dq2(dq0, dq1, dq2, &qmin, &qmax);
    gpu_limit_gradient(dqv, qmin, qmax, beta_tmp);

    edge_values[0] = cv_k + dqv[0];
    edge_values[1] = cv_k + dqv[1];
    edge_values[2] = cv_k + dqv[2];
}

// Set constant edge values (for zero beta or boundary cases)
static inline void gpu_set_constant_edge_values(double cv_k, double *edge_values) {
    edge_values[0] = cv_k;
    edge_values[1] = cv_k;
    edge_values[2] = cv_k;
}

// Compute dqv from gradient (for boundary case with single neighbor)
static inline void gpu_compute_dqv_from_gradient(
    double dq1, double dx2, double dy2,
    double dxv0, double dxv1, double dxv2,
    double dyv0, double dyv1, double dyv2,
    double *dqv) {
    double a = dq1 * dx2;
    double b = dq1 * dy2;

    dqv[0] = a * dxv0 + b * dyv0;
    dqv[1] = a * dxv1 + b * dyv1;
    dqv[2] = a * dxv2 + b * dyv2;
}

// ============================================================================
// Flux Computation Helper Functions (device code)
// ============================================================================

// Rotate momentum components to align with edge normal
static inline void gpu_rotate(double *q, double n1, double n2) {
    double q1 = q[1];
    double q2 = q[2];
    q[1] = n1 * q1 + n2 * q2;
    q[2] = -n2 * q1 + n1 * q2;
}

// Compute velocity terms with zero-depth handling
static inline void gpu_compute_velocity_terms(
    double h, double h_edge,
    double uh_raw, double vh_raw,
    double *u, double *uh, double *v, double *vh) {
    if (h_edge > 0.0) {
        double inv_h_edge = 1.0 / h_edge;
        *u = uh_raw * inv_h_edge;
        *uh = h * (*u);
        *v = vh_raw * inv_h_edge;
        *vh = h * inv_h_edge * vh_raw;
    } else {
        *u = 0.0;
        *uh = 0.0;
        *v = 0.0;
        *vh = 0.0;
    }
}

// Compute local Froude number for low-Froude corrections
static inline double gpu_compute_local_froude(
    anuga_int low_froude,
    double u_left, double u_right,
    double v_left, double v_right,
    double soundspeed_left, double soundspeed_right) {
    double numerator = u_right * u_right + u_left * u_left +
                       v_right * v_right + v_left * v_left;
    double denominator = soundspeed_left * soundspeed_left +
                         soundspeed_right * soundspeed_right + 1.0e-10;

    if (low_froude == 1) {
        return sqrt(fmax(0.001, fmin(1.0, numerator / denominator)));
    } else if (low_froude == 2) {
        double fr = sqrt(numerator / denominator);
        return sqrt(fmin(1.0, 0.01 + fmax(fr - 0.01, 0.0)));
    } else {
        return 1.0;
    }
}

// Maximum wave speed (positive direction)
static inline double gpu_compute_s_max(double u_left, double u_right,
                                       double c_left, double c_right) {
    double s = fmax(u_left + c_left, u_right + c_right);
    return (s < 0.0) ? 0.0 : s;
}

// Minimum wave speed (negative direction)
static inline double gpu_compute_s_min(double u_left, double u_right,
                                       double c_left, double c_right) {
    double s = fmin(u_left - c_left, u_right - c_right);
    return (s > 0.0) ? 0.0 : s;
}

// Central flux function - Kurganov-Noelle-Petrova scheme
static inline void gpu_flux_function_central(
    double *q_left, double *q_right,
    double h_left, double h_right,
    double hle, double hre,
    double n1, double n2,
    double epsilon, double ze, double g,
    double *edgeflux, double *max_speed, double *pressure_flux,
    anuga_int low_froude) {

    double uh_left, vh_left, u_left, v_left;
    double uh_right, vh_right, u_right, v_right;
    double soundspeed_left, soundspeed_right;
    double q_left_rotated[3], q_right_rotated[3];
    double flux_left[3], flux_right[3];

    // Copy and rotate to edge-aligned coordinates
    for (int i = 0; i < 3; i++) {
        q_left_rotated[i] = q_left[i];
        q_right_rotated[i] = q_right[i];
    }
    gpu_rotate(q_left_rotated, n1, n2);
    gpu_rotate(q_right_rotated, n1, n2);

    // Compute velocities
    uh_left = q_left_rotated[1];
    vh_left = q_left_rotated[2];
    gpu_compute_velocity_terms(h_left, hle, q_left_rotated[1], q_left_rotated[2],
                               &u_left, &uh_left, &v_left, &vh_left);

    uh_right = q_right_rotated[1];
    vh_right = q_right_rotated[2];
    gpu_compute_velocity_terms(h_right, hre, q_right_rotated[1], q_right_rotated[2],
                               &u_right, &uh_right, &v_right, &vh_right);

    // Wave speeds
    soundspeed_left = sqrt(g * h_left);
    soundspeed_right = sqrt(g * h_right);

    double local_fr = gpu_compute_local_froude(low_froude, u_left, u_right,
                                               v_left, v_right,
                                               soundspeed_left, soundspeed_right);

    double s_max = gpu_compute_s_max(u_left, u_right, soundspeed_left, soundspeed_right);
    double s_min = gpu_compute_s_min(u_left, u_right, soundspeed_left, soundspeed_right);

    // Physical fluxes
    flux_left[0] = u_left * h_left;
    flux_left[1] = u_left * uh_left;
    flux_left[2] = u_left * vh_left;

    flux_right[0] = u_right * h_right;
    flux_right[1] = u_right * uh_right;
    flux_right[2] = u_right * vh_right;

    // Central upwind flux
    double denom = s_max - s_min;
    double inverse_denominator = 1.0 / fmax(denom, 1.0e-100);
    double s_max_s_min = s_max * s_min;

    if (denom < epsilon) {
        // Both wave speeds very small
        edgeflux[0] = 0.0;
        edgeflux[1] = 0.0;
        edgeflux[2] = 0.0;
        *max_speed = 0.0;
        *pressure_flux = 0.5 * g * 0.5 * (h_left * h_left + h_right * h_right);
    } else {
        *max_speed = fmax(s_max, -s_min);

        double flux_0 = s_max * flux_left[0] - s_min * flux_right[0];
        flux_0 += s_max_s_min * (fmax(q_right_rotated[0], ze) - fmax(q_left_rotated[0], ze));
        edgeflux[0] = flux_0 * inverse_denominator;

        double flux_1 = s_max * flux_left[1] - s_min * flux_right[1];
        flux_1 += local_fr * s_max_s_min * (uh_right - uh_left);
        edgeflux[1] = flux_1 * inverse_denominator;

        double flux_2 = s_max * flux_left[2] - s_min * flux_right[2];
        flux_2 += local_fr * s_max_s_min * (vh_right - vh_left);
        edgeflux[2] = flux_2 * inverse_denominator;

        *pressure_flux = 0.5 * g * (s_max * h_left * h_left - s_min * h_right * h_right) * inverse_denominator;

        // Rotate back
        gpu_rotate(edgeflux, n1, -n2);
    }
}

#pragma omp end declare target

#endif // GPU_DEVICE_HELPERS_H
