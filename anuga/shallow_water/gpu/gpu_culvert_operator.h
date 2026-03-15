// GPU-accelerated culvert (Boyd box/pipe) operator
// Batches all culverts into a single gather -> CPU compute -> scatter cycle
//
// Public struct definitions (culvert_params, culvert_indices, etc.) live in
// gpu_domain.h. This header adds internal structs used only within the
// culvert implementation, plus pure-computation function prototypes.

#ifndef GPU_CULVERT_OPERATOR_H
#define GPU_CULVERT_OPERATOR_H

// Forward declare - full definition in gpu_domain.h
struct gpu_domain;
struct culvert_params;

// ============================================================================
// Internal structs (used by gpu_culvert_operator.c only)
// ============================================================================

// Gathered data for one inlet (filled by GPU gather)
struct inlet_data {
    // Enquiry point values
    double enquiry_stage;
    double enquiry_xmom;
    double enquiry_ymom;
    double enquiry_elevation;

    // Area-weighted averages over inlet region
    double avg_stage;
    double avg_depth;
    double avg_xmom;
    double avg_ymom;
    double total_area;
};

// Result from discharge calculation for one culvert
struct culvert_result {
    double Q;                    // Discharge [m^3/s]
    double barrel_velocity;      // Velocity through culvert [m/s]
    double outlet_culvert_depth; // Depth at outlet [m]
    double flow_area;            // Flow cross-section area [m^2]
    int inflow_idx;              // Which inlet (0 or 1) is the inflow
};

// Transfer specification for scattering results back to GPU
struct culvert_transfer {
    // Inflow region updates
    double new_inflow_depth;
    double new_inflow_xmom;
    double new_inflow_ymom;

    // Outflow region updates
    double new_outflow_depth;
    double new_outflow_xmom;
    double new_outflow_ymom;

    int inflow_idx;              // Which inlet (0 or 1) is inflow
};

// ============================================================================
// Pure computation functions (CPU-side, no GPU dependencies)
// ============================================================================

void boyd_box_discharge(const struct culvert_params *p,
                        double driving_energy,
                        double delta_total_energy,
                        double outlet_enquiry_depth,
                        double *Q, double *barrel_velocity,
                        double *outlet_culvert_depth, double *flow_area);

void boyd_pipe_discharge(const struct culvert_params *p,
                         double inflow_depth,
                         double driving_energy,
                         double delta_total_energy,
                         double outlet_enquiry_depth,
                         double *Q, double *barrel_velocity,
                         double *outlet_culvert_depth, double *flow_area);

void culvert_smooth_energy(double *smooth_delta_total_energy,
                           double delta_total_energy,
                           double timestep,
                           double smoothing_timescale,
                           double *ts_out);

void culvert_smooth_discharge(double smooth_delta_total_energy,
                              double *smooth_Q,
                              double Q_in,
                              double flow_area,
                              double ts,
                              double *Q_out, double *velocity_out);

#endif // GPU_CULVERT_OPERATOR_H
