// GPU-accelerated inlet operator
// Transfers only inlet triangle data (~6KB for 10-100 triangles)
// instead of full domain sync (~235MB for 4.9M elements)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "gpu_domain.h"
#include "gpu_device_helpers.h"
#include "gpu_omp_macros.h"

// velocity_protection constant (matches anuga.config)
#define VELOCITY_PROTECTION 1.0e-6

// ============================================================================
// Initialization and Cleanup
// ============================================================================

int gpu_inlet_operator_init(struct gpu_domain *GD, int num_indices,
                             int *indices, double *areas) {
    struct inlet_operators *IO = &GD->inlet_ops;

    // Find a free slot
    int op_id = -1;
    for (int i = 0; i < MAX_INLET_OPERATORS; i++) {
        if (!IO->ops[i].active) {
            op_id = i;
            break;
        }
    }

    if (op_id < 0) {
        fprintf(stderr, "ERROR: No free inlet operator slots (max %d)\n",
                MAX_INLET_OPERATORS);
        return -1;
    }

    struct inlet_operator_info *op = &IO->ops[op_id];
    op->num_indices = num_indices;
    op->active = 1;
    op->mapped = 0;

    if (num_indices == 0) {
        op->indices = NULL;
        op->areas = NULL;
        op->total_area = 0.0;
        op->scratch_stages = NULL;
        op->scratch_bed = NULL;
        op->scratch_xmom = NULL;
        op->scratch_ymom = NULL;
        op->scratch_depths = NULL;
        IO->num_operators++;
        return op_id;
    }

    // Allocate and copy index/area arrays
    op->indices = (int*)malloc(num_indices * sizeof(int));
    op->areas = (double*)malloc(num_indices * sizeof(double));

    if (!op->indices || !op->areas) {
        fprintf(stderr, "Failed to allocate inlet_operator arrays\n");
        if (op->indices) free(op->indices);
        if (op->areas) free(op->areas);
        op->active = 0;
        return -1;
    }

    memcpy(op->indices, indices, num_indices * sizeof(int));
    memcpy(op->areas, areas, num_indices * sizeof(double));

    // Precompute total area
    op->total_area = 0.0;
    for (int k = 0; k < num_indices; k++) {
        op->total_area += areas[k];
    }

    // Allocate host scratch buffers for small D2H/H2D transfers
    op->scratch_stages = (double*)malloc(num_indices * sizeof(double));
    op->scratch_bed = (double*)malloc(num_indices * sizeof(double));
    op->scratch_xmom = (double*)malloc(num_indices * sizeof(double));
    op->scratch_ymom = (double*)malloc(num_indices * sizeof(double));
    op->scratch_depths = (double*)malloc(num_indices * sizeof(double));

    // Map to GPU if already initialized
    if (GD->gpu_initialized) {
        // Ensure we map to the correct device for this domain
        omp_set_default_device(GD->device_id);

        int ni = op->num_indices;
        int *idx = op->indices;
        double *ar = op->areas;
        double *ss = op->scratch_stages;
        double *sb = op->scratch_bed;
        double *sx = op->scratch_xmom;
        double *sy = op->scratch_ymom;
        double *sd = op->scratch_depths;
        #pragma omp target enter data map(to: idx[0:ni], ar[0:ni]) \
            map(alloc: ss[0:ni], sb[0:ni], sx[0:ni], sy[0:ni], sd[0:ni])
        op->mapped = 1;

        // Verify mapping succeeded (skip check in host fallback mode)
        if (GD->device_id >= 0) {
            int chk_idx = omp_target_is_present(idx, GD->device_id);
            int chk_ar = omp_target_is_present(ar, GD->device_id);
            if (!chk_idx || !chk_ar) {
                fprintf(stderr, "[Rank %d] ERROR: Inlet_operator %d mapping FAILED after enter data! "
                        "indices_present=%d areas_present=%d idx=%p ar=%p ni=%d\n",
                        GD->rank, op_id, chk_idx, chk_ar, (void*)idx, (void*)ar, ni);
                fflush(stderr);
            }
        }
    }

    IO->num_operators++;

    // Bounds check indices against domain size
    int64_t n_elements = GD->D.number_of_elements;
    int bad = 0;
    for (int k = 0; k < num_indices; k++) {
        if (op->indices[k] < 0 || op->indices[k] >= n_elements) {
            fprintf(stderr, "[Rank %d] ERROR: inlet_operator %d index[%d]=%d out of range [0,%ld)\n",
                    GD->rank, op_id, k, op->indices[k], (long)n_elements);
            bad = 1;
        }
    }
    if (bad) {
        fprintf(stderr, "[Rank %d] ERROR: inlet_operator %d has out-of-range indices!\n",
                GD->rank, op_id);
    }

    //printf("[Rank %d] Inlet_operator %d initialized: %d indices (GPU mapped: %d) "
    //       "indices=%p areas=%p scratch_s=%p scratch_b=%p scratch_x=%p scratch_y=%p scratch_d=%p\n",
    //       GD->rank, op_id, num_indices, op->mapped,
    //       (void*)op->indices, (void*)op->areas,
    //       (void*)op->scratch_stages, (void*)op->scratch_bed,
    //       (void*)op->scratch_xmom, (void*)op->scratch_ymom,
    //       (void*)op->scratch_depths);
    //fflush(stdout);

    return op_id;
}

void gpu_inlet_operator_finalize(struct gpu_domain *GD, int op_id) {
    if (op_id < 0 || op_id >= MAX_INLET_OPERATORS) return;

    struct inlet_operator_info *op = &GD->inlet_ops.ops[op_id];
    if (!op->active) return;

    if (op->mapped && op->num_indices > 0) {
        int ni = op->num_indices;
        int *idx = op->indices;
        double *ar = op->areas;
        double *ss = op->scratch_stages;
        double *sb = op->scratch_bed;
        double *sx = op->scratch_xmom;
        double *sy = op->scratch_ymom;
        double *sd = op->scratch_depths;
        #pragma omp target exit data map(delete: idx[0:ni], ar[0:ni], \
            ss[0:ni], sb[0:ni], sx[0:ni], sy[0:ni], sd[0:ni])
    }

    if (op->indices) free(op->indices);
    if (op->areas) free(op->areas);
    if (op->scratch_stages) free(op->scratch_stages);
    if (op->scratch_bed) free(op->scratch_bed);
    if (op->scratch_xmom) free(op->scratch_xmom);
    if (op->scratch_ymom) free(op->scratch_ymom);
    if (op->scratch_depths) free(op->scratch_depths);

    memset(op, 0, sizeof(struct inlet_operator_info));
    GD->inlet_ops.num_operators--;
}

void gpu_inlet_operators_finalize_all(struct gpu_domain *GD) {
    for (int i = 0; i < MAX_INLET_OPERATORS; i++) {
        if (GD->inlet_ops.ops[i].active) {
            gpu_inlet_operator_finalize(GD, i);
        }
    }
}

// ============================================================================
// GPU Reductions (small - 10-100 elements)
// ============================================================================

double gpu_inlet_get_volume(struct gpu_domain *GD, int op_id) {
    if (op_id < 0 || op_id >= MAX_INLET_OPERATORS) return 0.0;
    struct inlet_operator_info *op = &GD->inlet_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return 0.0;

    omp_set_default_device(GD->device_id);

    int num = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict areas = op->areas;
    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict bed_c = GD->D.bed_centroid_values;

    // Debug: check if pointers are present on device (skip in host fallback mode)
    if (GD->device_id >= 0) {
        int present_idx = omp_target_is_present(indices, omp_get_default_device());
        int present_ar = omp_target_is_present(areas, omp_get_default_device());
        int present_sc = omp_target_is_present(stage_c, omp_get_default_device());
        int present_bc = omp_target_is_present(bed_c, omp_get_default_device());
        if (!present_idx || !present_ar || !present_sc || !present_bc) {
            fprintf(stderr, "[Rank %d] gpu_inlet_get_volume op=%d: MISSING device mapping! "
                    "indices=%d areas=%d stage_c=%d bed_c=%d\n",
                    GD->rank, op_id, present_idx, present_ar, present_sc, present_bc);
            fflush(stderr);
            return 0.0;
        }
    }

    double volume = 0.0;
    OMP_PARALLEL_LOOP_REDUCTION_PLUS(volume)
    for (int k = 0; k < num; k++) {
        int i = indices[k];
        double depth = stage_c[i] - bed_c[i];
        if (depth < 0.0) depth = 0.0;
        volume += depth * areas[k];
    }

    return volume;
}

void gpu_inlet_get_velocities(struct gpu_domain *GD, int op_id,
                               double *u_out, double *v_out) {
    *u_out = 0.0;
    *v_out = 0.0;
    if (op_id < 0 || op_id >= MAX_INLET_OPERATORS) return;
    struct inlet_operator_info *op = &GD->inlet_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return;

    omp_set_default_device(GD->device_id);

    // Debug: verify all pointers are mapped (skip in host fallback mode)
    int n = op->num_indices;
    if (GD->device_id >= 0) {
        int present_idx = omp_target_is_present(op->indices, GD->device_id);
        int present_sc = omp_target_is_present(GD->D.stage_centroid_values, GD->device_id);
        int present_bc = omp_target_is_present(GD->D.bed_centroid_values, GD->device_id);
        int present_xc = omp_target_is_present(GD->D.xmom_centroid_values, GD->device_id);
        int present_yc = omp_target_is_present(GD->D.ymom_centroid_values, GD->device_id);
        int present_sd = omp_target_is_present(op->scratch_depths, GD->device_id);
        int present_sx = omp_target_is_present(op->scratch_xmom, GD->device_id);
        int present_sy = omp_target_is_present(op->scratch_ymom, GD->device_id);
        if (!present_idx || !present_sc || !present_bc || !present_xc || !present_yc ||
            !present_sd || !present_sx || !present_sy) {
            fprintf(stderr, "[Rank %d] gpu_inlet_get_velocities op=%d: MISSING mapping! "
                    "idx=%d sc=%d bc=%d xc=%d yc=%d sd=%d sx=%d sy=%d\n",
                    GD->rank, op_id, present_idx, present_sc, present_bc,
                    present_xc, present_yc, present_sd, present_sx, present_sy);
            fflush(stderr);
            return;
        }
    }

    // Small D2H: gather depths, xmom, ymom for inlet triangles
    int * restrict indices = op->indices;
    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict bed_c = GD->D.bed_centroid_values;
    double * restrict xmom_c = GD->D.xmom_centroid_values;
    double * restrict ymom_c = GD->D.ymom_centroid_values;

    double *s_depths = op->scratch_depths;
    double *s_xmom = op->scratch_xmom;
    double *s_ymom = op->scratch_ymom;

    // GPU gather: read from domain arrays into scratch buffers on device
    OMP_PARALLEL_LOOP
    for (int k = 0; k < n; k++) {
        int i = indices[k];
        s_depths[k] = stage_c[i] - bed_c[i];
        s_xmom[k] = xmom_c[i];
        s_ymom[k] = ymom_c[i];
    }

    // Small D2H: transfer scratch buffers to host
    #pragma omp target update from(s_depths[0:n], s_xmom[0:n], s_ymom[0:n])

    // CPU computation on small arrays (matching inlet.py get_velocities)
    for (int k = 0; k < n; k++) {
        double d = s_depths[k];
        double denom = d * d + VELOCITY_PROTECTION;
        s_xmom[k] = s_xmom[k] * d / denom;  // u
        s_ymom[k] = s_ymom[k] * d / denom;  // v
    }

    // Return first element (caller accesses scratch buffers for full arrays)
    *u_out = s_xmom[0];
    *v_out = s_ymom[0];
}

// ============================================================================
// GPU Kernels (small - 10-100 elements)
// ============================================================================

void gpu_inlet_set_depths(struct gpu_domain *GD, int op_id, double depth) {
    if (op_id < 0 || op_id >= MAX_INLET_OPERATORS) return;
    struct inlet_operator_info *op = &GD->inlet_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return;

    int n = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict bed_c = GD->D.bed_centroid_values;

    OMP_PARALLEL_LOOP
    for (int k = 0; k < n; k++) {
        int i = indices[k];
        stage_c[i] = bed_c[i] + depth;
    }
}

void gpu_inlet_set_xmoms(struct gpu_domain *GD, int op_id, double value) {
    if (op_id < 0 || op_id >= MAX_INLET_OPERATORS) return;
    struct inlet_operator_info *op = &GD->inlet_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return;

    int n = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict xmom_c = GD->D.xmom_centroid_values;

    OMP_PARALLEL_LOOP
    for (int k = 0; k < n; k++) {
        xmom_c[indices[k]] = value;
    }
}

void gpu_inlet_set_ymoms(struct gpu_domain *GD, int op_id, double value) {
    if (op_id < 0 || op_id >= MAX_INLET_OPERATORS) return;
    struct inlet_operator_info *op = &GD->inlet_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return;

    int n = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict ymom_c = GD->D.ymom_centroid_values;

    OMP_PARALLEL_LOOP
    for (int k = 0; k < n; k++) {
        ymom_c[indices[k]] = value;
    }
}

void gpu_inlet_set_xmoms_array(struct gpu_domain *GD, int op_id,
                                double *values, int n_vals) {
    if (op_id < 0 || op_id >= MAX_INLET_OPERATORS) return;
    struct inlet_operator_info *op = &GD->inlet_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return;

    int n = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict xmom_c = GD->D.xmom_centroid_values;

    // Small H2D: copy values to scratch, update on GPU
    double *scratch = op->scratch_xmom;
    memcpy(scratch, values, n * sizeof(double));
    #pragma omp target update to(scratch[0:n])

    OMP_PARALLEL_LOOP
    for (int k = 0; k < n; k++) {
        xmom_c[indices[k]] = scratch[k];
    }
}

void gpu_inlet_set_ymoms_array(struct gpu_domain *GD, int op_id,
                                double *values, int n_vals) {
    if (op_id < 0 || op_id >= MAX_INLET_OPERATORS) return;
    struct inlet_operator_info *op = &GD->inlet_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return;

    int n = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict ymom_c = GD->D.ymom_centroid_values;

    double *scratch = op->scratch_ymom;
    memcpy(scratch, values, n * sizeof(double));
    #pragma omp target update to(scratch[0:n])

    OMP_PARALLEL_LOOP
    for (int k = 0; k < n; k++) {
        ymom_c[indices[k]] = scratch[k];
    }
}

// ============================================================================
// set_stages_evenly - the key algorithm (from inlet.py:192-225)
// Small D2H gather → CPU sort → small H2D scatter
// ============================================================================

// Comparison function for sorting by stage value
static int compare_stage_indices(const void *a, const void *b) {
    const double *da = (const double*)a;
    const double *db = (const double*)b;
    if (*da < *db) return -1;
    if (*da > *db) return 1;
    return 0;
}

void gpu_inlet_set_stages_evenly(struct gpu_domain *GD, int op_id, double volume) {
    if (op_id < 0 || op_id >= MAX_INLET_OPERATORS) return;
    struct inlet_operator_info *op = &GD->inlet_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return;

    omp_set_default_device(GD->device_id);

    int n = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict bed_c = GD->D.bed_centroid_values;

    double *stages = op->scratch_stages;
    double *bed = op->scratch_bed;

    // Small D2H: gather stages and bed for inlet triangles on GPU
    OMP_PARALLEL_LOOP
    for (int k = 0; k < n; k++) {
        int i = indices[k];
        stages[k] = stage_c[i];
        bed[k] = bed_c[i];
    }
    // Transfer to host
    #pragma omp target update from(stages[0:n], bed[0:n])

    // Areas: use the host copy (op->areas host allocation is still valid)
    double *areas_local = op->areas;

    // CPU sort: argsort by stage (10-100 elements, trivial)
    int *stages_order = (int*)malloc(n * sizeof(int));
    for (int k = 0; k < n; k++) stages_order[k] = k;

    // Simple insertion sort (n is small, 10-100)
    for (int i = 1; i < n; i++) {
        int key = stages_order[i];
        double key_val = stages[key];
        int j = i - 1;
        while (j >= 0 && stages[stages_order[j]] > key_val) {
            stages_order[j + 1] = stages_order[j];
            j--;
        }
        stages_order[j + 1] = key;
    }

    // Accumulate areas of cells ordered by stage
    double *summed_areas = (double*)malloc(n * sizeof(double));
    summed_areas[0] = areas_local[stages_order[0]];
    for (int k = 1; k < n; k++) {
        summed_areas[k] = summed_areas[k-1] + areas_local[stages_order[k]];
    }

    // Accumulate the volume needed to fill cells
    double *summed_volume = (double*)malloc(n * sizeof(double));
    summed_volume[0] = 0.0;
    for (int k = 1; k < n; k++) {
        summed_volume[k] = summed_volume[k-1] +
            summed_areas[k-1] * (stages[stages_order[k]] - stages[stages_order[k-1]]);
    }

    // Find last index where summed_volume <= volume
    int index = 0;
    for (int k = 0; k < n; k++) {
        if (summed_volume[k] <= volume) {
            index = k;
        }
    }

    // Calculate new stage
    double depth = (volume - summed_volume[index]) / summed_areas[index];
    double new_stage = stages[stages_order[index]] + depth;

    // Set stages for cells up to and including index
    for (int k = 0; k <= index; k++) {
        stages[stages_order[k]] = new_stage;
    }

    // Small H2D: update stages scratch buffer on GPU, then scatter to domain array
    #pragma omp target update to(stages[0:n])

    OMP_PARALLEL_LOOP
    for (int k = 0; k < n; k++) {
        int i = indices[k];
        stage_c[i] = stages[k];
    }

    free(stages_order);
    free(summed_areas);
    free(summed_volume);
}

// ============================================================================
// Main entry point: gpu_inlet_apply
// Combines all 3 cases from Inlet_operator.__call__()
// ============================================================================

double gpu_inlet_apply(struct gpu_domain *GD, int op_id, double volume,
                       double current_volume, double total_area,
                       double *vel_u, double *vel_v, int num_vel,
                       int has_velocity, double ext_vel_u, double ext_vel_v,
                       int zero_velocity) {
    if (op_id < 0 || op_id >= MAX_INLET_OPERATORS) return 0.0;
    struct inlet_operator_info *op = &GD->inlet_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return 0.0;

    omp_set_default_device(GD->device_id);

    int n = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict bed_c = GD->D.bed_centroid_values;
    double * restrict xmom_c = GD->D.xmom_centroid_values;
    double * restrict ymom_c = GD->D.ymom_centroid_values;

    double actual_volume = volume;

    if (volume >= 0.0) {
        // Case 1: Positive volume - set stages evenly + set momentum
        gpu_inlet_set_stages_evenly(GD, op_id, volume);

        // Get depths from GPU for momentum calculation
        double *s_depths = op->scratch_depths;
        OMP_PARALLEL_LOOP
        for (int k = 0; k < n; k++) {
            int i = indices[k];
            s_depths[k] = stage_c[i] - bed_c[i];
        }
        #pragma omp target update from(s_depths[0:n])

        if (zero_velocity) {
            gpu_inlet_set_xmoms(GD, op_id, 0.0);
            gpu_inlet_set_ymoms(GD, op_id, 0.0);
        } else {
            // Compute new xmom/ymom = depth * velocity
            double *s_xmom = op->scratch_xmom;
            double *s_ymom = op->scratch_ymom;

            if (has_velocity) {
                for (int k = 0; k < n; k++) {
                    s_xmom[k] = s_depths[k] * ext_vel_u;
                    s_ymom[k] = s_depths[k] * ext_vel_v;
                }
            } else {
                // Use existing velocities (passed in vel_u, vel_v arrays)
                for (int k = 0; k < n; k++) {
                    s_xmom[k] = s_depths[k] * vel_u[k];
                    s_ymom[k] = s_depths[k] * vel_v[k];
                }
            }

            gpu_inlet_set_xmoms_array(GD, op_id, s_xmom, n);
            gpu_inlet_set_ymoms_array(GD, op_id, s_ymom, n);
        }

    } else if (current_volume + volume >= 0.0) {
        // Case 2: Negative but sustainable - set uniform depth
        double depth = (current_volume + volume) / total_area;
        gpu_inlet_set_depths(GD, op_id, depth);

        // Get depths from GPU for momentum
        double *s_depths = op->scratch_depths;
        OMP_PARALLEL_LOOP
        for (int k = 0; k < n; k++) {
            int i = indices[k];
            s_depths[k] = stage_c[i] - bed_c[i];
        }
        #pragma omp target update from(s_depths[0:n])

        if (zero_velocity) {
            gpu_inlet_set_xmoms(GD, op_id, 0.0);
            gpu_inlet_set_ymoms(GD, op_id, 0.0);
        } else {
            double *s_xmom = op->scratch_xmom;
            double *s_ymom = op->scratch_ymom;

            if (has_velocity) {
                for (int k = 0; k < n; k++) {
                    s_xmom[k] = s_depths[k] * ext_vel_u;
                    s_ymom[k] = s_depths[k] * ext_vel_v;
                }
            } else {
                for (int k = 0; k < n; k++) {
                    s_xmom[k] = s_depths[k] * vel_u[k];
                    s_ymom[k] = s_depths[k] * vel_v[k];
                }
            }

            gpu_inlet_set_xmoms_array(GD, op_id, s_xmom, n);
            gpu_inlet_set_ymoms_array(GD, op_id, s_ymom, n);
        }

    } else {
        // Case 3: Extracting too much water - drain completely
        gpu_inlet_set_depths(GD, op_id, 0.0);
        gpu_inlet_set_xmoms(GD, op_id, 0.0);
        gpu_inlet_set_ymoms(GD, op_id, 0.0);
        actual_volume = -current_volume;
    }

    return actual_volume;
}
