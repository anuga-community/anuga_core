// GPU-accelerated culvert (Boyd box/pipe) operator
//
// Strategy: Batch all culverts into ONE gather → CPU compute → ONE scatter
// per timestep, reducing GPU↔CPU sync from 2×N_culverts to exactly 2.
//
// The Boyd discharge physics runs on CPU (200-300 FLOPs of branchy serial code).
// Only the data movement (gather/scatter ~2KB) touches the GPU.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "gpu_domain.h"
#include "gpu_culvert_operator.h"
#include "gpu_omp_macros.h"
#include "gpu_nvtx.h"

#define VELOCITY_PROTECTION 1.0e-6

// ============================================================================
// Pure computation: Boyd BOX discharge
// Direct translation from boyd_box_operator.py:boyd_box_function()
// ============================================================================

void boyd_box_discharge(const struct culvert_params *p,
                        double driving_energy,
                        double delta_total_energy,
                        double outlet_enquiry_depth,
                        double *Q_out, double *barrel_velocity_out,
                        double *outlet_culvert_depth_out, double *flow_area_out) {

    double width = p->width;
    double depth = p->height;
    double blockage = p->blockage;
    double barrels = p->barrels;
    double length = p->length;
    double sum_loss = p->sum_loss;
    double manning = p->manning;

    double bf = 1.0 - blockage;

    if (blockage >= 1.0) {
        *Q_out = 0.0;
        *barrel_velocity_out = 0.0;
        *outlet_culvert_depth_out = 0.0;
        *flow_area_out = 0.00001;
        return;
    }

    // Inlet control: unsubmerged vs submerged
    double Q_inlet_unsubmerged = 0.544 * sqrt(p->g) * bf * width * barrels * pow(driving_energy, 1.50);
    double Q_inlet_submerged = 0.702 * sqrt(p->g) * bf * width * barrels * pow(depth, 0.89) * pow(driving_energy, 0.61);

    double Q, dcrit, flow_area, perimeter;

    if (Q_inlet_unsubmerged < Q_inlet_submerged) {
        Q = Q_inlet_unsubmerged;
    } else {
        Q = Q_inlet_submerged;
    }

    dcrit = pow(Q * Q / p->g / pow(bf * width * barrels, 2.0), 0.333333);

    if (dcrit > depth) {
        dcrit = depth;
        flow_area = bf * width * dcrit * barrels;
        perimeter = 2.0 * (bf * width * barrels + dcrit);
    } else {
        flow_area = bf * width * barrels * dcrit;
        perimeter = 2.0 * dcrit + bf * width * barrels;
    }

    double outlet_culvert_depth = dcrit;

    // Recompute dcrit (matches Python exactly)
    dcrit = pow(Q * Q / p->g / pow(bf * width * barrels, 2.0), 0.333333);
    outlet_culvert_depth = dcrit;

    if (outlet_culvert_depth > depth) {
        outlet_culvert_depth = depth;
        flow_area = bf * width * barrels * depth;
        perimeter = 2.0 * (bf * width * barrels + depth);
    } else {
        flow_area = bf * width * barrels * outlet_culvert_depth;
        perimeter = bf * width * barrels + 2.0 * outlet_culvert_depth;
    }

    double hyd_rad = flow_area / perimeter;
    double culvert_velocity = sqrt(delta_total_energy / ((sum_loss / 2.0 / p->g) +
                                    (manning * manning * length) / pow(hyd_rad, 1.33333)));
    double Q_outlet_tailwater = flow_area * culvert_velocity;

    // Outlet control check
    if (delta_total_energy < driving_energy) {
        if (outlet_enquiry_depth > depth) {
            // Outlet submerged
            outlet_culvert_depth = depth;
            flow_area = bf * width * barrels * depth;
            perimeter = 2.0 * (bf * width * barrels + depth);
        } else {
            dcrit = pow(Q * Q / p->g / pow(bf * width * barrels, 2.0), 0.333333);
            outlet_culvert_depth = dcrit;
            if (outlet_culvert_depth > depth) {
                outlet_culvert_depth = depth;
                flow_area = bf * width * barrels * depth;
                perimeter = 2.0 * (bf * width * barrels + depth);
            } else {
                flow_area = bf * width * barrels * outlet_culvert_depth;
                perimeter = bf * width * barrels + 2.0 * outlet_culvert_depth;
            }
        }

        hyd_rad = flow_area / perimeter;
        culvert_velocity = sqrt(delta_total_energy / ((sum_loss / 2.0 / p->g) +
                                (manning * manning * length) / pow(hyd_rad, 1.33333)));
        Q_outlet_tailwater = flow_area * culvert_velocity;

        if (Q_outlet_tailwater < Q)
            Q = Q_outlet_tailwater;
    }

    // Barrel velocity with protection
    double barrel_velocity = Q / (flow_area + VELOCITY_PROTECTION / flow_area);

    *Q_out = Q;
    *barrel_velocity_out = barrel_velocity;
    *outlet_culvert_depth_out = outlet_culvert_depth;
    *flow_area_out = flow_area;
}

// ============================================================================
// Pure computation: Boyd PIPE discharge
// Direct translation from boyd_pipe_operator.py:boyd_pipe_function()
// ============================================================================

void boyd_pipe_discharge(const struct culvert_params *p,
                         double inflow_depth,
                         double driving_energy,
                         double delta_total_energy,
                         double outlet_enquiry_depth,
                         double *Q_out, double *barrel_velocity_out,
                         double *outlet_culvert_depth_out, double *flow_area_out) {

    double diameter = p->diameter;
    double blockage = p->blockage;
    double barrels = p->barrels;
    double length = p->length;
    double sum_loss = p->sum_loss;
    double manning = p->manning;

    if (blockage >= 1.0) {
        *Q_out = 0.0;
        *barrel_velocity_out = 0.0;
        *outlet_culvert_depth_out = 0.0;
        *flow_area_out = 0.00001;
        return;
    }

    double bf;
    if (blockage > 0.9) {
        bf = 3.333 - 3.333 * blockage;
    } else {
        bf = 1.0 - 0.4012316798 * blockage - 0.3768350138 * (blockage * blockage);
    }

    // Inlet control
    double Q_inlet_unsubmerged = barrels * (0.421 * sqrt(p->g) * pow(bf * diameter, 0.87) * pow(driving_energy, 1.63));
    double Q_inlet_submerged = barrels * (0.530 * sqrt(p->g) * pow(bf * diameter, 1.87) * pow(driving_energy, 0.63));

    double Q = (Q_inlet_unsubmerged < Q_inlet_submerged) ? Q_inlet_unsubmerged : Q_inlet_submerged;

    // Critical depth estimation (two formulas)
    double dcrit1 = (bf * diameter) / 1.26 * pow(Q / sqrt(p->g) / pow(bf * diameter, 2.5), 1.0 / 3.75);
    double dcrit2 = (bf * diameter) / 0.95 * pow(Q / sqrt(p->g) / pow(bf * diameter, 2.5), 1.0 / 1.95);

    double outlet_culvert_depth;
    if (dcrit1 / (bf * diameter) > 0.85) {
        outlet_culvert_depth = dcrit2;
    } else {
        outlet_culvert_depth = dcrit1;
    }

    double flow_area, perimeter, flow_width;
    double alpha;
    double bd = bf * diameter;

    if (outlet_culvert_depth >= bd) {
        // Pipe full
        outlet_culvert_depth = bd;
        flow_area = barrels * (bd / 2.0) * (bd / 2.0) * M_PI;
        perimeter = barrels * bd * M_PI;
        flow_width = barrels * bd;
    } else {
        // Partial fill
        alpha = acos(1.0 - 2.0 * outlet_culvert_depth / bd) * 2.0;
        flow_area = barrels * bd * bd / 8.0 * (alpha - sin(alpha));
        flow_width = barrels * bd * sin(alpha / 2.0);
        perimeter = barrels * (alpha * bd / 2.0);
    }

    // Outlet control
    if (delta_total_energy < driving_energy) {
        if (outlet_enquiry_depth > bd) {
            // Outlet submerged - pipe full
            outlet_culvert_depth = bd;
            flow_area = barrels * (bd / 2.0) * (bd / 2.0) * M_PI;
            perimeter = barrels * bd * M_PI;
            flow_width = barrels * bd;
        } else {
            // Partial flow - recalculate critical depth
            dcrit1 = bd / 1.26 * pow(Q / sqrt(p->g) / pow(bd, 2.5), 1.0 / 3.75);
            dcrit2 = bd / 0.95 * pow(Q / sqrt(p->g) / pow(bd, 2.5), 1.0 / 1.95);

            if (dcrit1 / bd > 0.85)
                outlet_culvert_depth = dcrit2;
            else
                outlet_culvert_depth = dcrit1;

            if (outlet_culvert_depth > bd) {
                outlet_culvert_depth = bd;
                flow_area = barrels * (bd / 2.0) * (bd / 2.0) * M_PI;
                perimeter = barrels * bd * M_PI;
                flow_width = barrels * bd;
            } else {
                alpha = acos(1.0 - 2.0 * outlet_culvert_depth / bd) * 2.0;
                flow_area = barrels * bd * bd / 8.0 * (alpha - sin(alpha));
                flow_width = barrels * bd * sin(alpha / 2.0);
                perimeter = barrels * alpha * bd / 2.0;
            }
        }
    }

    double hyd_rad = flow_area / perimeter;
    double culvert_velocity = sqrt(delta_total_energy / ((sum_loss / 2.0 / p->g) +
                                    (manning * manning * length) / pow(hyd_rad, 1.33333)));
    double Q_outlet_tailwater = flow_area * culvert_velocity;

    if (Q_outlet_tailwater < Q)
        Q = Q_outlet_tailwater;

    double barrel_velocity = Q / (flow_area + VELOCITY_PROTECTION / flow_area);

    *Q_out = Q;
    *barrel_velocity_out = barrel_velocity;
    *outlet_culvert_depth_out = outlet_culvert_depth;
    *flow_area_out = flow_area;
}

// ============================================================================
// Pure computation: Weir-orifice TRAPEZOID discharge
// Direct translation from weir_orifice_trapezoid_operator.py:
//   weir_orifice_trapezoid_function()
// Cross-section: bottom width `width`, side slopes z1 (left) and z2 (right).
// ============================================================================

// Newton iteration to find critical depth for a trapezoidal section given Q.
// Returns dcrit (clamped to [1e-5, depth] on convergence failure).
static double trapezoid_critical_depth(double Q, double bf_barrels_w,
                                       double z12, double sqrt_z1, double sqrt_z2,
                                       double depth, double g) {
    double dcrit = 1.0e-5;
    for (int ic = 0; ic < 100; ic++) {
        double Tc = bf_barrels_w + z12 * dcrit;
        double Ac = 0.5 * dcrit * (bf_barrels_w + Tc);
        if (Tc < 1.0e-12 || Ac < 1.0e-12) break;
        double fc  = pow(Ac, 1.5) / sqrt(Tc) - Q / sqrt(g);
        double ffc = -0.5 * pow(Ac, 1.5) * z12 / pow(Tc, 1.5)
                     + 1.5 * sqrt(Ac) * sqrt(Tc);
        if (fabs(ffc) < 1.0e-30) break;
        double dyc = -fc / ffc;
        dcrit += dyc;
        if (dcrit < 1.0e-5) dcrit = 1.0e-5;
        if (fabs(dyc) < 1.0e-5) break;
    }
    if (dcrit > depth) dcrit = depth;
    return dcrit;
}


void weir_orifice_trapezoid_discharge(const struct culvert_params *p,
                                      double driving_energy,
                                      double delta_total_energy,
                                      double outlet_enquiry_depth,
                                      double *Q_out, double *barrel_velocity_out,
                                      double *outlet_culvert_depth_out,
                                      double *flow_area_out) {
    double width   = p->width;
    double depth   = p->height;
    double blockage = p->blockage;
    double barrels  = p->barrels;
    double z1 = p->z1;
    double z2 = p->z2;
    double z12 = z1 + z2;
    double length  = p->length;
    double sum_loss = p->sum_loss;
    double manning  = p->manning;
    double g = p->g;

    if (blockage >= 1.0) {
        *Q_out = 0.0;
        *barrel_velocity_out = 0.0;
        *outlet_culvert_depth_out = 0.0;
        *flow_area_out = 1.0e-5;
        return;
    }

    double bf = 1.0 - blockage;
    // bf * barrels * width — used throughout
    double bfw = bf * barrels * width;

    // Pre-compute slant lengths for perimeter
    double sqrt_z1 = sqrt(z1 * z1 + 1.0);
    double sqrt_z2 = sqrt(z2 * z2 + 1.0);

    // Inlet control estimates
    // Weir flow (unsubmerged): Q = 1.7 * bfw_eff * driving_energy^1.5
    //   where bfw_eff = average of bottom and top widths = (2*width + depth*(z1+z2))/2
    double top_w  = 2.0 * width + depth * z12;
    double Q_inlet_unsubmerged = 1.7 * bf * barrels * (top_w / 2.0)
                                 * pow(driving_energy, 1.5);
    // Orifice flow (submerged): Q = 0.8 * bfw_eff_area * sqrt(g) * sqrt(driving_energy)
    double full_area = 0.5 * depth * (bfw + bfw + z12 * depth);
    double Q_inlet_submerged = 0.8 * bf * barrels * sqrt(g) * full_area
                               * sqrt(driving_energy);

    double Q;
    if (Q_inlet_unsubmerged < Q_inlet_submerged) {
        Q = Q_inlet_unsubmerged;
    } else {
        Q = Q_inlet_submerged;
    }

    // Critical depth for inlet-control Q
    double dcrit = trapezoid_critical_depth(Q, bfw, z12, sqrt_z1, sqrt_z2, depth, g);

    double flow_area, perimeter;
    if (dcrit >= depth) {
        dcrit = depth;
        flow_area = bfw * depth + 0.5 * z12 * depth * depth;
        perimeter = 2.0 * bfw + z12 * depth + sqrt_z1 * depth + sqrt_z2 * depth;
    } else {
        flow_area = bfw * dcrit + 0.5 * z12 * dcrit * dcrit;
        perimeter = bfw + sqrt_z1 * dcrit + sqrt_z2 * dcrit;
    }

    double outlet_culvert_depth = dcrit;

    // Re-solve critical depth (same as Python — redundant for rect but kept for fidelity)
    dcrit = trapezoid_critical_depth(Q, bfw, z12, sqrt_z1, sqrt_z2, depth, g);
    outlet_culvert_depth = dcrit;
    if (outlet_culvert_depth >= depth) {
        outlet_culvert_depth = depth;
        flow_area = bfw * depth + 0.5 * z12 * depth * depth;
        perimeter = 2.0 * bfw + z12 * depth + sqrt_z1 * depth + sqrt_z2 * depth;
    } else {
        flow_area = bfw * outlet_culvert_depth + 0.5 * z12 * outlet_culvert_depth * outlet_culvert_depth;
        perimeter = bfw + sqrt_z1 * outlet_culvert_depth + sqrt_z2 * outlet_culvert_depth;
    }

    // Outlet-control velocity and Q
    double hyd_rad = flow_area / fmax(perimeter, 1.0e-12);
    double culvert_velocity = sqrt(delta_total_energy
                                   / ((sum_loss / (2.0 * g))
                                      + (manning * manning * length)
                                        / pow(hyd_rad, 1.33333)));
    double Q_outlet_tailwater = flow_area * culvert_velocity;

    if (delta_total_energy < driving_energy) {
        // Outlet control
        if (outlet_enquiry_depth > depth) {
            // Outlet submerged — use full section
            outlet_culvert_depth = depth;
            flow_area = bfw * depth + 0.5 * z12 * depth * depth;
            perimeter = bfw + sqrt_z1 * depth + sqrt_z2 * depth;
        } else {
            Q = fmin(Q, Q_outlet_tailwater);
            dcrit = trapezoid_critical_depth(Q, bfw, z12, sqrt_z1, sqrt_z2, depth, g);
            outlet_culvert_depth = dcrit;
            if (outlet_culvert_depth >= depth) {
                outlet_culvert_depth = depth;
                flow_area = bfw * depth + 0.5 * z12 * depth * depth;
                perimeter = bfw + sqrt_z1 * depth + sqrt_z2 * depth;
            } else {
                flow_area = bfw * outlet_culvert_depth
                            + 0.5 * z12 * outlet_culvert_depth * outlet_culvert_depth;
                perimeter = bfw + sqrt_z1 * outlet_culvert_depth
                            + sqrt_z2 * outlet_culvert_depth;
            }
        }

        hyd_rad = flow_area / fmax(perimeter, 1.0e-12);
        culvert_velocity = sqrt(delta_total_energy
                                / ((sum_loss / (2.0 * g))
                                   + (manning * manning * length)
                                     / pow(hyd_rad, 1.33333)));
        Q_outlet_tailwater = flow_area * culvert_velocity;
        Q = fmin(Q, Q_outlet_tailwater);
    }

    double barrel_velocity = Q / (flow_area + VELOCITY_PROTECTION / fmax(flow_area, 1.0e-12));

    *Q_out = Q;
    *barrel_velocity_out = barrel_velocity;
    *outlet_culvert_depth_out = outlet_culvert_depth;
    *flow_area_out = flow_area;
}

// ============================================================================
// Energy smoothing (from boyd_box_operator.py:total_energy())
// ============================================================================

void culvert_smooth_energy(double *smooth_delta_total_energy,
                           double delta_total_energy,
                           double timestep,
                           double smoothing_timescale,
                           double *ts_out) {
    double ts;
    if (timestep > 0.0) {
        double denom = timestep;
        if (smoothing_timescale > denom) denom = smoothing_timescale;
        if (1.0e-06 > denom) denom = 1.0e-06;
        ts = timestep / denom;
    } else {
        ts = 1.0;
    }

    *smooth_delta_total_energy = *smooth_delta_total_energy +
        ts * (delta_total_energy - *smooth_delta_total_energy);
    *ts_out = ts;
}

// ============================================================================
// Discharge smoothing (from boyd_box_operator.py:smooth_discharge())
// ============================================================================

void culvert_smooth_discharge(double smooth_delta_total_energy,
                              double *smooth_Q,
                              double Q_in,
                              double flow_area,
                              double ts,
                              double *Q_out, double *velocity_out) {
    double Qsign = (smooth_delta_total_energy >= 0.0) ? 1.0 : -1.0;

    *smooth_Q = *smooth_Q + ts * (Q_in * Qsign - *smooth_Q);

    double Q;
    if ((*smooth_Q >= 0.0) != (Qsign >= 0.0)) {
        // Flow direction mismatch - set Q to zero
        Q = 0.0;
    } else {
        double abs_smooth_Q = fabs(*smooth_Q);
        Q = (abs_smooth_Q < Q_in) ? abs_smooth_Q : Q_in;
    }

    double barrel_velocity;
    if (flow_area == 0.0) {
        barrel_velocity = 0.0;
    } else {
        barrel_velocity = Q / flow_area;
    }

    *Q_out = Q;
    *velocity_out = barrel_velocity;
}

// ============================================================================
// Helper: compute enquiry-derived values from raw gathered data
// Mirrors inlet_enquiry.py get_enquiry_* methods
// ============================================================================

static void compute_enquiry_values(const struct inlet_data *data,
                                   const struct culvert_params *p,
                                   int inlet_idx,
                                   double *depth, double *velocity_head,
                                   double *total_energy, double *specific_energy) {
    double invert_elev;
    if (inlet_idx == 0 && p->has_invert_elevation_0)
        invert_elev = p->invert_elevation_0;
    else if (inlet_idx == 1 && p->has_invert_elevation_1)
        invert_elev = p->invert_elevation_1;
    else
        invert_elev = data->enquiry_elevation;

    double d = data->enquiry_stage - invert_elev;
    if (d < 0.0) d = 0.0;
    *depth = d;

    // Velocity head: 0.5 * speed² / g
    double water_depth = data->enquiry_stage - data->enquiry_elevation;
    double denom = water_depth * water_depth + VELOCITY_PROTECTION;
    double u = water_depth * data->enquiry_xmom / denom;
    double v = water_depth * data->enquiry_ymom / denom;
    double speed_sq = u * u + v * v;
    *velocity_head = 0.5 * speed_sq / p->g;

    *total_energy = *velocity_head + data->enquiry_stage;
    *specific_energy = *velocity_head + d;
}

// ============================================================================
// GPU Culvert Manager: Init / Finalize
// ============================================================================

int gpu_culvert_init(struct gpu_domain *GD,
                     const struct culvert_params *params,
                     int enquiry_index_0, int enquiry_index_1,
                     int inlet0_num, int *inlet0_indices, double *inlet0_areas,
                     int inlet1_num, int *inlet1_indices, double *inlet1_areas,
                     int master_proc, int enquiry_proc_0, int enquiry_proc_1,
                     int inlet_master_proc_0, int inlet_master_proc_1,
                     int is_local, int mpi_tag_base) {

    struct culvert_operators *CO = &GD->culvert_ops;

    if (inlet0_num > MAX_INLET_TRIANGLES || inlet1_num > MAX_INLET_TRIANGLES) {
        fprintf(stderr, "ERROR: Inlet has %d/%d triangles (max %d)\n",
                inlet0_num, inlet1_num, MAX_INLET_TRIANGLES);
        return -1;
    }

    // Grow params/indices/state arrays if full
    if (CO->num_culverts >= CO->capacity) {
        int new_cap = CO->capacity == 0 ? MAX_CULVERTS : CO->capacity * 2;
        struct culvert_params *np = (struct culvert_params*)
            realloc(CO->params, new_cap * sizeof(struct culvert_params));
        struct culvert_indices *ni = (struct culvert_indices*)
            realloc(CO->indices, new_cap * sizeof(struct culvert_indices));
        struct culvert_state *ns = (struct culvert_state*)
            realloc(CO->state, new_cap * sizeof(struct culvert_state));
        if (!np || !ni || !ns) {
            fprintf(stderr, "ERROR: Failed to grow culvert_operators to %d slots\n", new_cap);
            if (np) free(np); else if (CO->params) free(CO->params);
            if (ni) free(ni); else if (CO->indices) free(CO->indices);
            if (ns) free(ns); else if (CO->state) free(CO->state);
            CO->params = NULL; CO->indices = NULL; CO->state = NULL;
            return -1;
        }
        // Zero-init new entries so state is clean
        memset(np + CO->capacity, 0,
               (new_cap - CO->capacity) * sizeof(struct culvert_params));
        memset(ni + CO->capacity, 0,
               (new_cap - CO->capacity) * sizeof(struct culvert_indices));
        memset(ns + CO->capacity, 0,
               (new_cap - CO->capacity) * sizeof(struct culvert_state));
        CO->params = np;
        CO->indices = ni;
        CO->state = ns;
        CO->capacity = new_cap;
    }

    int id = CO->num_culverts;

    // Copy params
    CO->params[id] = *params;

    // Copy indices
    struct culvert_indices *ci = &CO->indices[id];
    ci->enquiry_index_0 = enquiry_index_0;
    ci->enquiry_index_1 = enquiry_index_1;

    ci->inlet0_num = inlet0_num;
    if (inlet0_num > 0) {
        memcpy(ci->inlet0_indices, inlet0_indices, inlet0_num * sizeof(int));
        memcpy(ci->inlet0_areas, inlet0_areas, inlet0_num * sizeof(double));
    }
    ci->inlet0_total_area = 0.0;
    for (int k = 0; k < inlet0_num; k++) ci->inlet0_total_area += inlet0_areas[k];

    ci->inlet1_num = inlet1_num;
    if (inlet1_num > 0) {
        memcpy(ci->inlet1_indices, inlet1_indices, inlet1_num * sizeof(int));
        memcpy(ci->inlet1_areas, inlet1_areas, inlet1_num * sizeof(double));
    }
    ci->inlet1_total_area = 0.0;
    for (int k = 0; k < inlet1_num; k++) ci->inlet1_total_area += inlet1_areas[k];

    // MPI topology
    ci->master_proc = master_proc;
    ci->enquiry_proc[0] = enquiry_proc_0;
    ci->enquiry_proc[1] = enquiry_proc_1;
    ci->inlet_master_proc[0] = inlet_master_proc_0;
    ci->inlet_master_proc[1] = inlet_master_proc_1;
    ci->is_local = is_local;
    ci->mpi_tag_base = mpi_tag_base;

    // Initialize state
    CO->state[id].smooth_delta_total_energy = 0.0;
    CO->state[id].smooth_Q = 0.0;

    CO->num_culverts++;
    return id;
}

void gpu_culvert_finalize(struct gpu_domain *GD, int culvert_id) {
    // Individual culvert cleanup if needed (static arrays, nothing to free)
    (void)GD;
    (void)culvert_id;
}

void gpu_culverts_finalize_all(struct gpu_domain *GD) {
    struct culvert_operators *CO = &GD->culvert_ops;

    if (CO->mapped) {
        int ne = 2 * CO->num_culverts;
        int nt = CO->total_inlet_triangles;

        double *ss = CO->scratch_stage;
        double *sx = CO->scratch_xmom;
        double *sy = CO->scratch_ymom;
        double *se = CO->scratch_elev;

        if (ne > 0) {
            #pragma omp target exit data map(delete: ss[0:ne], sx[0:ne], sy[0:ne], se[0:ne])
        }

        if (nt > 0) {
            int *si = CO->scratch_inlet_indices;
            double *sa = CO->scratch_inlet_areas;
            double *sis = CO->scratch_inlet_stage;
            double *six = CO->scratch_inlet_xmom;
            double *siy = CO->scratch_inlet_ymom;
            double *sie = CO->scratch_inlet_elev;
            #pragma omp target exit data map(delete: si[0:nt], sa[0:nt], \
                sis[0:nt], six[0:nt], siy[0:nt], sie[0:nt])
        }
        CO->mapped = 0;
    }

    if (CO->scratch_stage) { free(CO->scratch_stage); CO->scratch_stage = NULL; }
    if (CO->scratch_xmom) { free(CO->scratch_xmom); CO->scratch_xmom = NULL; }
    if (CO->scratch_ymom) { free(CO->scratch_ymom); CO->scratch_ymom = NULL; }
    if (CO->scratch_elev) { free(CO->scratch_elev); CO->scratch_elev = NULL; }
    if (CO->scratch_inlet_indices) { free(CO->scratch_inlet_indices); CO->scratch_inlet_indices = NULL; }
    if (CO->scratch_inlet_areas) { free(CO->scratch_inlet_areas); CO->scratch_inlet_areas = NULL; }
    if (CO->scratch_inlet_stage) { free(CO->scratch_inlet_stage); CO->scratch_inlet_stage = NULL; }
    if (CO->scratch_inlet_xmom) { free(CO->scratch_inlet_xmom); CO->scratch_inlet_xmom = NULL; }
    if (CO->scratch_inlet_ymom) { free(CO->scratch_inlet_ymom); CO->scratch_inlet_ymom = NULL; }
    if (CO->scratch_inlet_elev) { free(CO->scratch_inlet_elev); CO->scratch_inlet_elev = NULL; }

    if (CO->params)  { free(CO->params);  CO->params  = NULL; }
    if (CO->indices) { free(CO->indices); CO->indices = NULL; }
    if (CO->state)   { free(CO->state);   CO->state   = NULL; }
    CO->num_culverts = 0;
    CO->capacity = 0;
    CO->initialized = 0;
}

// ============================================================================
// GPU Mapping: allocate and map scratch buffers
// Call AFTER all culverts are registered AND GPU domain is initialized
// ============================================================================

void gpu_culverts_map(struct gpu_domain *GD) {
    struct culvert_operators *CO = &GD->culvert_ops;

    if (CO->num_culverts == 0) return;
    if (CO->mapped) return;

    omp_set_default_device(GD->device_id);

    int nc = CO->num_culverts;
    int ne = 2 * nc;  // 2 enquiry points per culvert

    // Allocate enquiry scratch buffers
    CO->scratch_stage = (double*)calloc(ne, sizeof(double));
    CO->scratch_xmom = (double*)calloc(ne, sizeof(double));
    CO->scratch_ymom = (double*)calloc(ne, sizeof(double));
    CO->scratch_elev = (double*)calloc(ne, sizeof(double));

    // Count total inlet triangles and build flattened index array
    CO->total_inlet_triangles = 0;
    for (int c = 0; c < nc; c++) {
        CO->total_inlet_triangles += CO->indices[c].inlet0_num + CO->indices[c].inlet1_num;
    }

    int nt = CO->total_inlet_triangles;
    CO->scratch_inlet_indices = (int*)calloc(nt, sizeof(int));
    CO->scratch_inlet_areas = (double*)calloc(nt, sizeof(double));
    CO->scratch_inlet_stage = (double*)calloc(nt, sizeof(double));
    CO->scratch_inlet_xmom = (double*)calloc(nt, sizeof(double));
    CO->scratch_inlet_ymom = (double*)calloc(nt, sizeof(double));
    CO->scratch_inlet_elev = (double*)calloc(nt, sizeof(double));

    // Flatten inlet indices and areas
    int offset = 0;
    for (int c = 0; c < nc; c++) {
        struct culvert_indices *ci = &CO->indices[c];
        for (int k = 0; k < ci->inlet0_num; k++) {
            CO->scratch_inlet_indices[offset] = ci->inlet0_indices[k];
            CO->scratch_inlet_areas[offset] = ci->inlet0_areas[k];
            offset++;
        }
        for (int k = 0; k < ci->inlet1_num; k++) {
            CO->scratch_inlet_indices[offset] = ci->inlet1_indices[k];
            CO->scratch_inlet_areas[offset] = ci->inlet1_areas[k];
            offset++;
        }
    }

    // Map to GPU
    double *ss = CO->scratch_stage;
    double *sx = CO->scratch_xmom;
    double *sy = CO->scratch_ymom;
    double *se = CO->scratch_elev;
    #pragma omp target enter data map(alloc: ss[0:ne], sx[0:ne], sy[0:ne], se[0:ne])

    if (nt > 0) {
        int *si = CO->scratch_inlet_indices;
        double *sa = CO->scratch_inlet_areas;
        double *sis = CO->scratch_inlet_stage;
        double *six = CO->scratch_inlet_xmom;
        double *siy = CO->scratch_inlet_ymom;
        double *sie = CO->scratch_inlet_elev;
        #pragma omp target enter data map(to: si[0:nt], sa[0:nt]) \
            map(alloc: sis[0:nt], six[0:nt], siy[0:nt], sie[0:nt])
    }

    CO->mapped = 1;
    CO->initialized = 1;
}

// ============================================================================
// Batched Gather: read enquiry + inlet data from GPU in TWO transfers
// ============================================================================

static void gpu_culvert_gather_enquiry(struct gpu_domain *GD,
                                       struct inlet_data *data0,
                                       struct inlet_data *data1) {
    struct culvert_operators *CO = &GD->culvert_ops;
    int nc = CO->num_culverts;
    int ne = 2 * nc;

    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict xmom_c = GD->D.xmom_centroid_values;
    double * restrict ymom_c = GD->D.ymom_centroid_values;
    double * restrict bed_c = GD->D.bed_centroid_values;

    double *ss = CO->scratch_stage;
    double *sx = CO->scratch_xmom;
    double *sy = CO->scratch_ymom;
    double *se = CO->scratch_elev;

    // Build index list on host and upload to scratch_stage temporarily
    // (We reuse the scratch buffers for the index transfer, then overwrite with values)
    // Since ne ≤ 128, we just use the flattened inlet indices approach:
    // Store enquiry indices at the start of the inlet index array during map.
    // Instead, we do individual reads -- ne is tiny (≤128), the overhead is negligible
    // compared to kernel launch latency.

    // For small ne, gather on host via scratch_inlet_indices trick:
    // Actually simplest: use a pre-built enquiry index array stored in scratch.
    // We'll store it during map. For now, use a host-side loop with target update.

    // Host-side: read enquiry indices, build gather list
    // Use index 0 as placeholder for remote enquiry points (-1)
    int *enquiry_ids = (int*)malloc(ne * sizeof(int));
    if (!enquiry_ids) {
        fprintf(stderr, "ERROR: Failed to allocate enquiry_ids (%d ints)\n", ne);
        return;
    }
    for (int c = 0; c < nc; c++) {
        int ei0 = CO->indices[c].enquiry_index_0;
        int ei1 = CO->indices[c].enquiry_index_1;
        enquiry_ids[2 * c] = (ei0 >= 0) ? ei0 : 0;
        enquiry_ids[2 * c + 1] = (ei1 >= 0) ? ei1 : 0;
    }

    // Upload enquiry IDs to scratch_stage (reinterpreted as int, ne is tiny)
    // Then do a GPU gather kernel. But ints-in-doubles is fragile.
    // Simpler: use target map for the small stack array.
    int *eid = enquiry_ids;
    #pragma omp target teams loop map(to: eid[0:ne])
    for (int k = 0; k < ne; k++) {
        int i = eid[k];
        ss[k] = stage_c[i];
        sx[k] = xmom_c[i];
        sy[k] = ymom_c[i];
        se[k] = bed_c[i];
    }

    // Single D2H transfer (~1KB for 20 culverts)
    #pragma omp target update from(ss[0:ne], sx[0:ne], sy[0:ne], se[0:ne])
    free(enquiry_ids);

    // Unpack into per-culvert inlet_data structs
    for (int c = 0; c < nc; c++) {
        data0[c].enquiry_stage = ss[2 * c];
        data0[c].enquiry_xmom = sx[2 * c];
        data0[c].enquiry_ymom = sy[2 * c];
        data0[c].enquiry_elevation = se[2 * c];

        data1[c].enquiry_stage = ss[2 * c + 1];
        data1[c].enquiry_xmom = sx[2 * c + 1];
        data1[c].enquiry_ymom = sy[2 * c + 1];
        data1[c].enquiry_elevation = se[2 * c + 1];
    }
}

static void gpu_culvert_gather_inlets(struct gpu_domain *GD,
                                      struct inlet_data *data0,
                                      struct inlet_data *data1) {
    struct culvert_operators *CO = &GD->culvert_ops;
    int nc = CO->num_culverts;
    int nt = CO->total_inlet_triangles;

    if (nt == 0) return;

    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict xmom_c = GD->D.xmom_centroid_values;
    double * restrict ymom_c = GD->D.ymom_centroid_values;
    double * restrict bed_c = GD->D.bed_centroid_values;

    int *si = CO->scratch_inlet_indices;
    double *sis = CO->scratch_inlet_stage;
    double *six = CO->scratch_inlet_xmom;
    double *siy = CO->scratch_inlet_ymom;
    double *sie = CO->scratch_inlet_elev;

    // GPU gather: read all inlet triangle values
    OMP_PARALLEL_LOOP
    for (int k = 0; k < nt; k++) {
        int i = si[k];
        sis[k] = stage_c[i];
        six[k] = xmom_c[i];
        siy[k] = ymom_c[i];
        sie[k] = bed_c[i];
    }

    // Single D2H transfer (~2KB for 20 culverts × ~6 triangles each)
    #pragma omp target update from(sis[0:nt], six[0:nt], siy[0:nt], sie[0:nt])

    // CPU: compute area-weighted averages per inlet
    double *sa = CO->scratch_inlet_areas;
    int offset = 0;
    for (int c = 0; c < nc; c++) {
        struct culvert_indices *ci = &CO->indices[c];

        // Inlet 0 averages
        double sum_stage = 0, sum_depth = 0, sum_xmom = 0, sum_ymom = 0;
        for (int k = 0; k < ci->inlet0_num; k++) {
            int idx = offset + k;
            double area = sa[idx];
            double depth = sis[idx] - sie[idx];
            if (depth < 0.0) depth = 0.0;
            sum_stage += sis[idx] * area;
            sum_depth += depth * area;
            sum_xmom += six[idx] * area;
            sum_ymom += siy[idx] * area;
        }
        data0[c].avg_stage = sum_stage / ci->inlet0_total_area;
        data0[c].avg_depth = sum_depth / ci->inlet0_total_area;
        data0[c].avg_xmom = sum_xmom / ci->inlet0_total_area;
        data0[c].avg_ymom = sum_ymom / ci->inlet0_total_area;
        data0[c].total_area = ci->inlet0_total_area;
        offset += ci->inlet0_num;

        // Inlet 1 averages
        sum_stage = sum_depth = sum_xmom = sum_ymom = 0;
        for (int k = 0; k < ci->inlet1_num; k++) {
            int idx = offset + k;
            double area = sa[idx];
            double depth = sis[idx] - sie[idx];
            if (depth < 0.0) depth = 0.0;
            sum_stage += sis[idx] * area;
            sum_depth += depth * area;
            sum_xmom += six[idx] * area;
            sum_ymom += siy[idx] * area;
        }
        data1[c].avg_stage = sum_stage / ci->inlet1_total_area;
        data1[c].avg_depth = sum_depth / ci->inlet1_total_area;
        data1[c].avg_xmom = sum_xmom / ci->inlet1_total_area;
        data1[c].avg_ymom = sum_ymom / ci->inlet1_total_area;
        data1[c].total_area = ci->inlet1_total_area;
        offset += ci->inlet1_num;
    }
}

// ============================================================================
// Batched Scatter: write updated depths/momenta back to GPU
// ============================================================================

static void gpu_culvert_scatter(struct gpu_domain *GD,
                                struct culvert_transfer *transfers) {
    struct culvert_operators *CO = &GD->culvert_ops;
    int nc = CO->num_culverts;
    int nt = CO->total_inlet_triangles;

    if (nt == 0) return;

    // Build scatter values on host (using the same flattened layout as gather)
    double *sis = CO->scratch_inlet_stage;
    double *six = CO->scratch_inlet_xmom;
    double *siy = CO->scratch_inlet_ymom;
    double *sie = CO->scratch_inlet_elev;  // bed elevations (already gathered)

    int offset = 0;
    for (int c = 0; c < nc; c++) {
        struct culvert_indices *ci = &CO->indices[c];
        struct culvert_transfer *t = &transfers[c];

        // Determine which flat offset corresponds to inflow vs outflow
        int inflow_offset, outflow_offset, inflow_num, outflow_num;
        if (t->inflow_idx == 0) {
            inflow_offset = offset;
            inflow_num = ci->inlet0_num;
            outflow_offset = offset + ci->inlet0_num;
            outflow_num = ci->inlet1_num;
        } else {
            inflow_offset = offset + ci->inlet0_num;
            inflow_num = ci->inlet1_num;
            outflow_offset = offset;
            outflow_num = ci->inlet0_num;
        }

        // Set inflow region: stage = bed + new_depth, uniform xmom/ymom
        for (int k = 0; k < inflow_num; k++) {
            int idx = inflow_offset + k;
            sis[idx] = sie[idx] + t->new_inflow_depth;  // stage = bed + depth
            six[idx] = t->new_inflow_xmom;
            siy[idx] = t->new_inflow_ymom;
        }

        // Set outflow region
        for (int k = 0; k < outflow_num; k++) {
            int idx = outflow_offset + k;
            sis[idx] = sie[idx] + t->new_outflow_depth;
            six[idx] = t->new_outflow_xmom;
            siy[idx] = t->new_outflow_ymom;
        }

        offset += ci->inlet0_num + ci->inlet1_num;
    }

    // Single H2D transfer
    #pragma omp target update to(sis[0:nt], six[0:nt], siy[0:nt])

    // GPU scatter: write from scratch to domain arrays
    int *si = CO->scratch_inlet_indices;
    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict xmom_c = GD->D.xmom_centroid_values;
    double * restrict ymom_c = GD->D.ymom_centroid_values;

    OMP_PARALLEL_LOOP
    for (int k = 0; k < nt; k++) {
        int i = si[k];
        stage_c[i] = sis[k];
        xmom_c[i] = six[k];
        ymom_c[i] = siy[k];
    }
}

// ============================================================================
// MPI exchange helpers for cross-boundary culverts
// Uses non-blocking MPI to avoid deadlocks when multiple culverts cross
// different rank boundaries.
//
// Message protocol per cross-boundary culvert:
//   enquiry_proc[i] → master: 4 doubles (stage, xmom, ymom, elev)  tag_base+i
//   inlet_master[i] → master: 5 doubles (sum_s, sum_d, sum_xm, sum_ym, area)  tag_base+2+i
//   master → inlet_master[i]: 3 doubles (new_depth, new_xmom, new_ymom)  tag_base+4+i
// ============================================================================

// Maximum MPI requests: 6 per culvert (2 enquiry + 2 inlet + 2 result)
#define MAX_MPI_REQS (MAX_CULVERTS * 6)

// MPI message buffers for cross-boundary exchange
struct culvert_mpi_bufs {
    double enquiry_send[MAX_CULVERTS][2][4];    // [culvert][inlet][stage,xmom,ymom,elev]
    double enquiry_recv[MAX_CULVERTS][2][4];
    double inlet_send[MAX_CULVERTS][2][5];      // [culvert][inlet][sum_s,sum_d,sum_xm,sum_ym,area]
    double inlet_recv[MAX_CULVERTS][2][5];
    double result_send[MAX_CULVERTS][2][3];     // [culvert][inlet][new_depth,new_xmom,new_ymom]
    double result_recv[MAX_CULVERTS][2][3];
};

// Exchange enquiry data: non-blocking sends/recvs, then waitall
static void mpi_exchange_enquiry(struct gpu_domain *GD,
                                  struct inlet_data *data0,
                                  struct inlet_data *data1,
                                  struct culvert_mpi_bufs *bufs) {
    struct culvert_operators *CO = &GD->culvert_ops;
    int nc = CO->num_culverts;
    int myrank = GD->rank;
    MPI_Comm comm = GD->comm;
    MPI_Request requests[MAX_MPI_REQS];
    int nreq = 0;

    for (int c = 0; c < nc; c++) {
        struct culvert_indices *ci = &CO->indices[c];
        if (ci->is_local) continue;

        struct inlet_data *data[2] = {&data0[c], &data1[c]};

        for (int inlet = 0; inlet < 2; inlet++) {
            int tag = ci->mpi_tag_base + inlet;

            if (myrank == ci->enquiry_proc[inlet] && myrank != ci->master_proc) {
                // I have this enquiry point, send to master
                bufs->enquiry_send[c][inlet][0] = data[inlet]->enquiry_stage;
                bufs->enquiry_send[c][inlet][1] = data[inlet]->enquiry_xmom;
                bufs->enquiry_send[c][inlet][2] = data[inlet]->enquiry_ymom;
                bufs->enquiry_send[c][inlet][3] = data[inlet]->enquiry_elevation;
                MPI_Isend(bufs->enquiry_send[c][inlet], 4, MPI_DOUBLE,
                          ci->master_proc, tag, comm, &requests[nreq++]);
            }

            if (myrank == ci->master_proc && ci->enquiry_proc[inlet] != myrank) {
                // Master receives from enquiry proc
                MPI_Irecv(bufs->enquiry_recv[c][inlet], 4, MPI_DOUBLE,
                          ci->enquiry_proc[inlet], tag, comm, &requests[nreq++]);
            }
        }
    }

    if (nreq > 0)
        MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);

    // Unpack received enquiry data into inlet_data structs
    for (int c = 0; c < nc; c++) {
        struct culvert_indices *ci = &CO->indices[c];
        if (ci->is_local) continue;
        if (myrank != ci->master_proc) continue;

        struct inlet_data *data[2] = {&data0[c], &data1[c]};

        for (int inlet = 0; inlet < 2; inlet++) {
            if (ci->enquiry_proc[inlet] != myrank) {
                data[inlet]->enquiry_stage     = bufs->enquiry_recv[c][inlet][0];
                data[inlet]->enquiry_xmom      = bufs->enquiry_recv[c][inlet][1];
                data[inlet]->enquiry_ymom      = bufs->enquiry_recv[c][inlet][2];
                data[inlet]->enquiry_elevation  = bufs->enquiry_recv[c][inlet][3];
            }
        }
    }
}

// Exchange inlet averages: each inlet_master sends local sums to structure master
static void mpi_exchange_inlet_averages(struct gpu_domain *GD,
                                         struct inlet_data *data0,
                                         struct inlet_data *data1,
                                         struct culvert_mpi_bufs *bufs) {
    struct culvert_operators *CO = &GD->culvert_ops;
    int nc = CO->num_culverts;
    int myrank = GD->rank;
    MPI_Comm comm = GD->comm;
    MPI_Request requests[MAX_MPI_REQS];
    int nreq = 0;

    for (int c = 0; c < nc; c++) {
        struct culvert_indices *ci = &CO->indices[c];
        if (ci->is_local) continue;

        struct inlet_data *data[2] = {&data0[c], &data1[c]};

        for (int inlet = 0; inlet < 2; inlet++) {
            int tag = ci->mpi_tag_base + 2 + inlet;

            if (myrank == ci->inlet_master_proc[inlet] && myrank != ci->master_proc) {
                // Send local area-weighted sums to structure master
                double area = data[inlet]->total_area;
                bufs->inlet_send[c][inlet][0] = data[inlet]->avg_stage * area;
                bufs->inlet_send[c][inlet][1] = data[inlet]->avg_depth * area;
                bufs->inlet_send[c][inlet][2] = data[inlet]->avg_xmom * area;
                bufs->inlet_send[c][inlet][3] = data[inlet]->avg_ymom * area;
                bufs->inlet_send[c][inlet][4] = area;
                MPI_Isend(bufs->inlet_send[c][inlet], 5, MPI_DOUBLE,
                          ci->master_proc, tag, comm, &requests[nreq++]);
            }

            if (myrank == ci->master_proc && ci->inlet_master_proc[inlet] != myrank) {
                MPI_Irecv(bufs->inlet_recv[c][inlet], 5, MPI_DOUBLE,
                          ci->inlet_master_proc[inlet], tag, comm, &requests[nreq++]);
            }
        }
    }

    if (nreq > 0)
        MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);

    // Master combines local + remote inlet averages
    for (int c = 0; c < nc; c++) {
        struct culvert_indices *ci = &CO->indices[c];
        if (ci->is_local) continue;
        if (myrank != ci->master_proc) continue;

        struct inlet_data *data[2] = {&data0[c], &data1[c]};

        for (int inlet = 0; inlet < 2; inlet++) {
            if (ci->inlet_master_proc[inlet] != myrank) {
                // Replace local (placeholder) with remote data
                double remote_area = bufs->inlet_recv[c][inlet][4];
                if (remote_area > 0.0) {
                    data[inlet]->avg_stage = bufs->inlet_recv[c][inlet][0] / remote_area;
                    data[inlet]->avg_depth = bufs->inlet_recv[c][inlet][1] / remote_area;
                    data[inlet]->avg_xmom  = bufs->inlet_recv[c][inlet][2] / remote_area;
                    data[inlet]->avg_ymom  = bufs->inlet_recv[c][inlet][3] / remote_area;
                    data[inlet]->total_area = remote_area;
                }
            }
            // If master has local inlet data too (inlet_master == master), it already
            // has the correct averages from the GPU gather phase.
        }
    }
}

// Send computed results from master to remote inlet procs for scatter
static void mpi_exchange_results(struct gpu_domain *GD,
                                  struct culvert_transfer *transfers,
                                  struct culvert_mpi_bufs *bufs) {
    struct culvert_operators *CO = &GD->culvert_ops;
    int nc = CO->num_culverts;
    int myrank = GD->rank;
    MPI_Comm comm = GD->comm;
    MPI_Request requests[MAX_MPI_REQS];
    int nreq = 0;

    for (int c = 0; c < nc; c++) {
        struct culvert_indices *ci = &CO->indices[c];
        if (ci->is_local) continue;

        struct culvert_transfer *t = &transfers[c];

        for (int inlet = 0; inlet < 2; inlet++) {
            int tag = ci->mpi_tag_base + 4 + inlet;

            if (myrank == ci->master_proc && ci->inlet_master_proc[inlet] != myrank) {
                // Determine if this inlet is inflow or outflow
                if (inlet == t->inflow_idx) {
                    bufs->result_send[c][inlet][0] = t->new_inflow_depth;
                    bufs->result_send[c][inlet][1] = t->new_inflow_xmom;
                    bufs->result_send[c][inlet][2] = t->new_inflow_ymom;
                } else {
                    bufs->result_send[c][inlet][0] = t->new_outflow_depth;
                    bufs->result_send[c][inlet][1] = t->new_outflow_xmom;
                    bufs->result_send[c][inlet][2] = t->new_outflow_ymom;
                }
                MPI_Isend(bufs->result_send[c][inlet], 3, MPI_DOUBLE,
                          ci->inlet_master_proc[inlet], tag, comm, &requests[nreq++]);
            }

            if (myrank == ci->inlet_master_proc[inlet] && ci->master_proc != myrank) {
                MPI_Irecv(bufs->result_recv[c][inlet], 3, MPI_DOUBLE,
                          ci->master_proc, tag, comm, &requests[nreq++]);
            }
        }
    }

    if (nreq > 0)
        MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);

    // Non-master inlet procs build transfer structs from received data
    for (int c = 0; c < nc; c++) {
        struct culvert_indices *ci = &CO->indices[c];
        if (ci->is_local) continue;
        if (myrank == ci->master_proc) continue;

        struct culvert_transfer *t = &transfers[c];

        for (int inlet = 0; inlet < 2; inlet++) {
            if (myrank == ci->inlet_master_proc[inlet]) {
                // This rank owns inlet triangles — use received scatter values
                // We don't know the inflow direction, but we know which inlet we own.
                // Set both inflow/outflow fields; scatter uses inflow_idx to pick.
                // Since we have triangles only for this inlet, we just need the values
                // for this inlet's role (inflow or outflow).
                double new_depth = bufs->result_recv[c][inlet][0];
                double new_xmom  = bufs->result_recv[c][inlet][1];
                double new_ymom  = bufs->result_recv[c][inlet][2];

                // Store the values. The scatter function uses inflow_idx to determine
                // which inlet gets inflow vs outflow values. Since the master computed
                // the correct values for each inlet role, we store them in the right slot.
                if (inlet == 0) {
                    t->new_inflow_depth = new_depth;
                    t->new_inflow_xmom  = new_xmom;
                    t->new_inflow_ymom  = new_ymom;
                    t->inflow_idx = 0;  // Treat inlet 0 as "inflow" for scatter
                } else {
                    t->new_outflow_depth = new_depth;
                    t->new_outflow_xmom  = new_xmom;
                    t->new_outflow_ymom  = new_ymom;
                    t->inflow_idx = 0;  // inlet 1 is "outflow" with inflow_idx=0
                }
            }
        }
    }
}

// ============================================================================
// Per-culvert GPU scatter for cross-boundary culverts on non-master ranks
// These ranks only have triangles for one or both inlets (not all data).
// Uses small stack arrays with target map instead of the batched scratch.
// ============================================================================

static void scatter_single_inlet(struct gpu_domain *GD,
                                  int *tri_indices, int ntri,
                                  double new_depth, double new_xmom, double new_ymom) {
    if (ntri == 0) return;

    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict xmom_c = GD->D.xmom_centroid_values;
    double * restrict ymom_c = GD->D.ymom_centroid_values;
    double * restrict bed_c = GD->D.bed_centroid_values;

    // Stack-allocate small arrays for GPU scatter
    double stages[MAX_INLET_TRIANGLES];
    double xmoms[MAX_INLET_TRIANGLES];
    double ymoms[MAX_INLET_TRIANGLES];

    // Build new stage values: stage = bed + depth
    // First, read bed elevations from GPU
    int *idx = tri_indices;
    double beds[MAX_INLET_TRIANGLES];
    double *b = beds;
    #pragma omp target teams loop map(to: idx[0:ntri]) map(from: b[0:ntri])
    for (int k = 0; k < ntri; k++) {
        b[k] = bed_c[idx[k]];
    }

    for (int k = 0; k < ntri; k++) {
        stages[k] = beds[k] + new_depth;
        xmoms[k] = new_xmom;
        ymoms[k] = new_ymom;
    }

    // Write to GPU
    double *s = stages, *x = xmoms, *y = ymoms;
    #pragma omp target teams loop map(to: idx[0:ntri], s[0:ntri], x[0:ntri], y[0:ntri])
    for (int k = 0; k < ntri; k++) {
        int i = idx[k];
        stage_c[i] = s[k];
        xmom_c[i] = x[k];
        ymom_c[i] = y[k];
    }
}

// ============================================================================
// Main entry point: execute ALL culverts in one batched cycle
// Local culverts use batched gather/scatter (existing fast path).
// Cross-boundary culverts use MPI between gather and scatter phases.
// ============================================================================

void gpu_culverts_apply_all(struct gpu_domain *GD, double timestep) {
    NVTX_PUSH("gpu_culverts_apply_all");
    struct culvert_operators *CO = &GD->culvert_ops;
    int nc = CO->num_culverts;

    if (nc == 0 || !CO->initialized) {
        NVTX_POP();
        return;
    }

    omp_set_default_device(GD->device_id);
    int myrank = GD->rank;

    // Check if any parallel culverts exist
    int has_parallel = 0;
    for (int c = 0; c < nc; c++) {
        if (!CO->indices[c].is_local) { has_parallel = 1; break; }
    }

    // Stack-allocate per-culvert working data
    struct inlet_data data0[MAX_CULVERTS];
    struct inlet_data data1[MAX_CULVERTS];
    struct culvert_result results[MAX_CULVERTS];
    struct culvert_transfer transfers[MAX_CULVERTS];

    // ----------------------------------------------------------------
    // PHASE 1: Batched GPU gather (2 target update from's)
    // Gathers LOCAL data for ALL culverts (local + parallel).
    // Remote enquiry points get placeholder values (overwritten by MPI).
    // ----------------------------------------------------------------
    gpu_culvert_gather_enquiry(GD, data0, data1);
    gpu_culvert_gather_inlets(GD, data0, data1);

    // ----------------------------------------------------------------
    // PHASE 1b: MPI exchange for cross-boundary culverts
    // ----------------------------------------------------------------
    // Static allocation of MPI buffers (MAX_CULVERTS * ~200 bytes = ~12KB)
    static struct culvert_mpi_bufs mpi_bufs;

    if (has_parallel) {
        mpi_exchange_enquiry(GD, data0, data1, &mpi_bufs);
        mpi_exchange_inlet_averages(GD, data0, data1, &mpi_bufs);
    }

    // ----------------------------------------------------------------
    // PHASE 2: CPU computation loop (all culverts, ~200 FLOPs each)
    // Only master_proc computes for cross-boundary culverts.
    // ----------------------------------------------------------------
    for (int c = 0; c < nc; c++) {
        struct culvert_indices *ci = &CO->indices[c];
        struct culvert_params *p = &CO->params[c];
        struct culvert_state *st = &CO->state[c];
        struct culvert_result *r = &results[c];

        // Non-master ranks skip computation for cross-boundary culverts
        if (!ci->is_local && myrank != ci->master_proc) {
            r->Q = 0.0;
            r->barrel_velocity = 0.0;
            r->outlet_culvert_depth = 0.0;
            r->flow_area = 0.00001;
            r->inflow_idx = 0;
            continue;
        }

        // Check culvert is open
        double dim = (p->type == CULVERT_TYPE_BOX || p->type == CULVERT_TYPE_WEIR_TRAPEZOID)
                     ? p->height : p->diameter;
        if (dim <= 0.0) {
            r->Q = 0.0;
            r->barrel_velocity = 0.0;
            r->outlet_culvert_depth = 0.0;
            r->flow_area = 0.00001;
            r->inflow_idx = 0;
            continue;
        }

        // Compute delta_total_energy to determine flow direction
        double delta_total_energy;
        if (p->use_velocity_head) {
            double depth0, vh0, te0, se0;
            double depth1, vh1, te1, se1;
            compute_enquiry_values(&data0[c], p, 0, &depth0, &vh0, &te0, &se0);
            compute_enquiry_values(&data1[c], p, 1, &depth1, &vh1, &te1, &se1);
            delta_total_energy = te0 - te1;
        } else {
            delta_total_energy = data0[c].enquiry_stage - data1[c].enquiry_stage;
        }

        // Smooth delta_total_energy
        double ts;
        culvert_smooth_energy(&st->smooth_delta_total_energy,
                              delta_total_energy, timestep,
                              p->smoothing_timescale, &ts);

        // Determine inflow/outflow
        struct inlet_data *inflow_data, *outflow_data;
        if (st->smooth_delta_total_energy >= 0.0) {
            r->inflow_idx = 0;
            inflow_data = &data0[c];
            outflow_data = &data1[c];
            delta_total_energy = st->smooth_delta_total_energy;
        } else {
            r->inflow_idx = 1;
            inflow_data = &data1[c];
            outflow_data = &data0[c];
            delta_total_energy = -st->smooth_delta_total_energy;
        }

        // Only calculate if there's water at inflow
        double inflow_depth, inflow_vh, inflow_te, inflow_se;
        compute_enquiry_values(inflow_data, p, r->inflow_idx,
                               &inflow_depth, &inflow_vh, &inflow_te, &inflow_se);

        double outflow_depth, outflow_vh, outflow_te, outflow_se;
        compute_enquiry_values(outflow_data, p, 1 - r->inflow_idx,
                               &outflow_depth, &outflow_vh, &outflow_te, &outflow_se);

        if (inflow_depth > 0.01) {
            double driving_energy;
            if (p->use_velocity_head)
                driving_energy = inflow_se;
            else
                driving_energy = inflow_depth;

            double Q, bv, ocd, fa;

            if (p->type == CULVERT_TYPE_BOX) {
                boyd_box_discharge(p, driving_energy, delta_total_energy,
                                   outflow_depth, &Q, &bv, &ocd, &fa);
            } else if (p->type == CULVERT_TYPE_WEIR_TRAPEZOID) {
                weir_orifice_trapezoid_discharge(p, driving_energy,
                                                 delta_total_energy,
                                                 outflow_depth,
                                                 &Q, &bv, &ocd, &fa);
            } else {
                boyd_pipe_discharge(p, inflow_depth, driving_energy,
                                    delta_total_energy, outflow_depth,
                                    &Q, &bv, &ocd, &fa);
            }

            // Apply discharge smoothing
            culvert_smooth_discharge(st->smooth_delta_total_energy,
                                     &st->smooth_Q, Q, fa, ts, &Q, &bv);

            r->Q = Q;
            r->barrel_velocity = bv;
            r->outlet_culvert_depth = ocd;
            r->flow_area = fa;

            // Clamp velocity
            if (r->barrel_velocity > p->max_velocity) {
                r->barrel_velocity = p->max_velocity;
                r->Q = r->flow_area * r->barrel_velocity;
            }
        } else {
            r->Q = 0.0;
            r->barrel_velocity = 0.0;
            r->outlet_culvert_depth = 0.0;
            r->flow_area = 0.00001;
        }
    }

    // ----------------------------------------------------------------
    // PHASE 3: CPU water transfer (semi-implicit update per culvert)
    // Only master computes for cross-boundary culverts.
    // ----------------------------------------------------------------
    for (int c = 0; c < nc; c++) {
        struct culvert_indices *ci = &CO->indices[c];
        struct culvert_params *p = &CO->params[c];
        struct culvert_result *r = &results[c];
        struct culvert_transfer *t = &transfers[c];

        // Non-master ranks: will receive transfer data via MPI
        if (!ci->is_local && myrank != ci->master_proc) {
            memset(t, 0, sizeof(*t));
            continue;
        }

        t->inflow_idx = r->inflow_idx;

        struct inlet_data *inflow_data = (r->inflow_idx == 0) ? &data0[c] : &data1[c];
        struct inlet_data *outflow_data = (r->inflow_idx == 0) ? &data1[c] : &data0[c];
        double inflow_area = inflow_data->total_area;
        double outflow_area = outflow_data->total_area;

        double old_inflow_depth = inflow_data->avg_depth;
        double old_inflow_xmom = inflow_data->avg_xmom;
        double old_inflow_ymom = inflow_data->avg_ymom;

        // Semi-implicit factor
        double dt_Q_on_d;
        if (old_inflow_depth > 0.0)
            dt_Q_on_d = timestep * r->Q / old_inflow_depth;
        else
            dt_Q_on_d = 0.0;

        double factor = 1.0 / (1.0 + dt_Q_on_d / inflow_area);

        // New inflow values (with wet-dry adjustment if always_use_Q_wetdry_adjustment)
        double new_inflow_depth, timestep_star;
        if (p->always_use_Q_wetdry_adjustment) {
            new_inflow_depth = old_inflow_depth * factor;
            if (old_inflow_depth > 0.0)
                timestep_star = timestep * new_inflow_depth / old_inflow_depth;
            else
                timestep_star = 0.0;
        } else {
            new_inflow_depth = old_inflow_depth - timestep * r->Q / inflow_area;
            timestep_star = timestep;
        }

        double new_inflow_xmom, new_inflow_ymom;
        if (p->use_old_momentum_method) {
            new_inflow_xmom = old_inflow_xmom * factor;
            new_inflow_ymom = old_inflow_ymom * factor;
        } else {
            double factor2;
            if (old_inflow_depth > 0.0) {
                if (p->always_use_Q_wetdry_adjustment)
                    factor2 = 1.0 / (1.0 + dt_Q_on_d * new_inflow_depth / (old_inflow_depth * inflow_area));
                else
                    factor2 = 1.0 / (1.0 + timestep * r->Q / (old_inflow_depth * inflow_area));
            } else {
                factor2 = 0.0;
            }
            new_inflow_xmom = old_inflow_xmom * factor2;
            new_inflow_ymom = old_inflow_ymom * factor2;
        }

        t->new_inflow_depth = new_inflow_depth;
        t->new_inflow_xmom = new_inflow_xmom;
        t->new_inflow_ymom = new_inflow_ymom;

        // Outflow
        double outflow_extra_depth = r->Q * timestep_star / outflow_area;
        double new_outflow_depth = outflow_data->avg_depth + outflow_extra_depth;

        // Outflow direction vector
        double *outflow_vec = (r->inflow_idx == 0) ? p->outward_vector_1 : p->outward_vector_0;
        double dir0 = -outflow_vec[0];
        double dir1 = -outflow_vec[1];

        double new_outflow_xmom, new_outflow_ymom;
        if (p->use_momentum_jet) {
            new_outflow_xmom = r->barrel_velocity * new_outflow_depth * dir0;
            new_outflow_ymom = r->barrel_velocity * new_outflow_depth * dir1;
        } else {
            new_outflow_xmom = 0.0;
            new_outflow_ymom = 0.0;
        }

        t->new_outflow_depth = new_outflow_depth;
        t->new_outflow_xmom = new_outflow_xmom;
        t->new_outflow_ymom = new_outflow_ymom;
    }

    // ----------------------------------------------------------------
    // PHASE 3b: MPI exchange results for cross-boundary culverts
    // Master sends scatter values to remote inlet procs.
    // ----------------------------------------------------------------
    if (has_parallel) {
        mpi_exchange_results(GD, transfers, &mpi_bufs);

        // Non-master ranks scatter to their local GPU inlets
        for (int c = 0; c < nc; c++) {
            struct culvert_indices *ci = &CO->indices[c];
            if (ci->is_local) continue;
            if (myrank == ci->master_proc) continue;

            for (int inlet = 0; inlet < 2; inlet++) {
                if (myrank != ci->inlet_master_proc[inlet]) continue;

                int ntri = (inlet == 0) ? ci->inlet0_num : ci->inlet1_num;
                int *tri_idx = (inlet == 0) ? ci->inlet0_indices : ci->inlet1_indices;
                double new_depth = mpi_bufs.result_recv[c][inlet][0];
                double new_xmom  = mpi_bufs.result_recv[c][inlet][1];
                double new_ymom  = mpi_bufs.result_recv[c][inlet][2];

                scatter_single_inlet(GD, tri_idx, ntri, new_depth, new_xmom, new_ymom);
            }
        }
    }

    // ----------------------------------------------------------------
    // PHASE 4: Batched GPU scatter for local culverts (and master's
    // local inlets for parallel culverts)
    // ----------------------------------------------------------------
    gpu_culvert_scatter(GD, transfers);
    NVTX_POP();
}
