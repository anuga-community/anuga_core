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

// Rate operators (rain, extraction, etc.)

// ============================================================================
// Rate Operators (rain, extraction, etc.)
// ============================================================================

int gpu_rate_operator_init(struct gpu_domain *GD, int num_indices, int *indices,
                           double *areas, int *full_indices, int num_full) {
    struct rate_operators *RO = &GD->rate_ops;

    // Find a free slot
    int op_id = -1;
    for (int i = 0; i < MAX_RATE_OPERATORS; i++) {
        if (!RO->ops[i].active) {
            op_id = i;
            break;
        }
    }

    if (op_id < 0) {
        fprintf(stderr, "ERROR: No free rate operator slots (max %d)\n", MAX_RATE_OPERATORS);
        return -1;
    }

    struct rate_operator_info *op = &RO->ops[op_id];

    op->num_indices = num_indices;
    op->num_full = num_full;
    op->active = 1;
    op->mapped = 0;
    // Initialize rate array cache
    op->rate_array_cache = NULL;
    op->rate_array_size = 0;
    op->rate_array_mapped = 0;

    if (num_indices == 0) {
        op->indices = NULL;
        op->areas = NULL;
        op->full_indices = NULL;
        op->is_full = NULL;
        RO->num_operators++;
        return op_id;
    }

    // Allocate and copy arrays
    op->indices = (int*)malloc(num_indices * sizeof(int));
    op->areas = (double*)malloc(num_indices * sizeof(double));
    op->is_full = (int*)malloc(num_indices * sizeof(int));

    if (!op->indices || !op->areas || !op->is_full) {
        fprintf(stderr, "Failed to allocate rate_operator arrays\n");
        if (op->indices) free(op->indices);
        if (op->areas) free(op->areas);
        if (op->is_full) free(op->is_full);
        op->active = 0;
        return -1;
    }

    memcpy(op->indices, indices, num_indices * sizeof(int));
    memcpy(op->areas, areas, num_indices * sizeof(double));

    // Initialize is_full to 0 (ghost), then mark full cells as 1
    memset(op->is_full, 0, num_indices * sizeof(int));
    if (num_full > 0 && full_indices != NULL) {
        op->full_indices = (int*)malloc(num_full * sizeof(int));
        if (op->full_indices) {
            memcpy(op->full_indices, full_indices, num_full * sizeof(int));
        }
        // Mark full cells in is_full array
        // full_indices[j] gives the position k in indices[] that is a full cell
        for (int j = 0; j < num_full; j++) {
            int k = full_indices[j];
            if (k >= 0 && k < num_indices) {
                op->is_full[k] = 1;
            }
        }
    } else {
        op->full_indices = NULL;
        // If no full_indices provided, assume all cells are full
        for (int k = 0; k < num_indices; k++) {
            op->is_full[k] = 1;
        }
    }

    // Map to GPU immediately if GPU is already initialized
    if (GD->gpu_initialized) {
        int ni = op->num_indices;
        int *idx = op->indices;
        double *ar = op->areas;
        int *isf = op->is_full;
        #pragma omp target enter data map(to: idx[0:ni], ar[0:ni], isf[0:ni])
        op->mapped = 1;
    }

    RO->num_operators++;

    if (GD->rank == 0) {
        printf("Rate_operator %d initialized: %d indices, %d full (GPU mapped: %d)\n",
               op_id, num_indices, num_full, op->mapped);
    }

    return op_id;
}

void gpu_rate_operator_finalize(struct gpu_domain *GD, int op_id) {
    if (op_id < 0 || op_id >= MAX_RATE_OPERATORS) return;

    struct rate_operator_info *op = &GD->rate_ops.ops[op_id];
    if (!op->active) return;

    if (op->mapped && op->num_indices > 0) {
        int ni = op->num_indices;
        int *idx = op->indices;
        double *ar = op->areas;
        int *isf = op->is_full;
        #pragma omp target exit data map(delete: idx[0:ni], ar[0:ni], isf[0:ni])
    }

    // Clean up rate array cache
    if (op->rate_array_mapped && op->rate_array_cache != NULL) {
        double *rac = op->rate_array_cache;
        int ras = op->rate_array_size;
        #pragma omp target exit data map(delete: rac[0:ras])
    }
    if (op->rate_array_cache) free(op->rate_array_cache);

    if (op->indices) free(op->indices);
    if (op->areas) free(op->areas);
    if (op->full_indices) free(op->full_indices);
    if (op->is_full) free(op->is_full);

    op->indices = NULL;
    op->areas = NULL;
    op->full_indices = NULL;
    op->is_full = NULL;
    op->rate_array_cache = NULL;
    op->num_indices = 0;
    op->num_full = 0;
    op->rate_array_size = 0;
    op->active = 0;
    op->mapped = 0;
    op->rate_array_mapped = 0;

    GD->rate_ops.num_operators--;
}

void gpu_rate_operators_finalize_all(struct gpu_domain *GD) {
    for (int i = 0; i < MAX_RATE_OPERATORS; i++) {
        if (GD->rate_ops.ops[i].active) {
            gpu_rate_operator_finalize(GD, i);
        }
    }
    GD->rate_ops.initialized = 0;
}

double gpu_rate_operator_apply(struct gpu_domain *GD, int op_id,
                               double rate, double factor, double timestep) {
    if (op_id < 0 || op_id >= MAX_RATE_OPERATORS) return 0.0;

    struct rate_operator_info *op = &GD->rate_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return 0.0;

    // Ensure mapped
    if (!op->mapped) {
        int ni = op->num_indices;
        int *idx = op->indices;
        double *ar = op->areas;
        int *isf = op->is_full;
        #pragma omp target enter data map(to: idx[0:ni], ar[0:ni], isf[0:ni])
        op->mapped = 1;
    }

    int num_indices = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict areas = op->areas;
    int * restrict is_full = op->is_full;

    // Domain arrays (restrict enables better optimization)
    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict xmom_c = GD->D.xmom_centroid_values;
    double * restrict ymom_c = GD->D.ymom_centroid_values;
    double * restrict bed_c = GD->D.bed_centroid_values;

    double local_rate = factor * timestep * rate;
    double local_influx = 0.0;

    if (rate >= 0.0) {
        // Simple positive rate - just add to stage
        // Apply to ALL cells, but only count influx for full (non-ghost) cells
        #pragma omp target teams distribute parallel for reduction(+:local_influx)
        for (int k = 0; k < num_indices; k++) {
            int i = indices[k];
            stage_c[i] += local_rate;
            if (is_full[k]) {
                local_influx += local_rate * areas[k];
            }
        }
    } else {
        // Negative rate (extraction) - need to limit and scale momentum
        // Apply to ALL cells, but only count influx for full (non-ghost) cells
        #pragma omp target teams distribute parallel for reduction(+:local_influx)
        for (int k = 0; k < num_indices; k++) {
            int i = indices[k];

            // Current height
            double height = stage_c[i] - bed_c[i];

            // Can't remove more water than exists
            double actual_rate = (local_rate > -height) ? local_rate : -height;

            // Scaling factor for momentum (when extracting water)
            double scale_factor;
            if (actual_rate < 0.0) {
                scale_factor = (actual_rate + height) / (height + 1.0e-10);
            } else {
                scale_factor = 1.0;
            }

            // Apply updates
            stage_c[i] += actual_rate;
            xmom_c[i] *= scale_factor;
            ymom_c[i] *= scale_factor;

            if (is_full[k]) {
                local_influx += actual_rate * areas[k];
            }
        }
    }

    // Count FLOPs: 8 FLOPs per affected cell
    if (GD->flops.enabled) {
        GD->flops.rate_operator_flops += (uint64_t)op->num_indices * FLOPS_RATE_OPERATOR;
        GD->flops.rate_operator_calls++;
    }

    return local_influx;
}

double gpu_rate_operator_apply_array(struct gpu_domain *GD, int op_id,
                                     double *rate_array, int rate_array_size,
                                     int use_indices_into_rate,
                                     int rate_changed,
                                     double factor, double timestep) {
    if (op_id < 0 || op_id >= MAX_RATE_OPERATORS) return 0.0;

    struct rate_operator_info *op = &GD->rate_ops.ops[op_id];
    if (!op->active || op->num_indices == 0) return 0.0;

    // Ensure operator arrays are mapped
    if (!op->mapped) {
        int ni = op->num_indices;
        int *idx = op->indices;
        double *ar = op->areas;
        int *isf = op->is_full;
        #pragma omp target enter data map(to: idx[0:ni], ar[0:ni], isf[0:ni])
        op->mapped = 1;
    }

    int num_indices = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict areas = op->areas;
    int * restrict is_full = op->is_full;

    // Domain arrays (restrict enables better optimization)
    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict xmom_c = GD->D.xmom_centroid_values;
    double * restrict ymom_c = GD->D.ymom_centroid_values;
    double * restrict bed_c = GD->D.bed_centroid_values;

    double local_influx = 0.0;
    double ft = factor * timestep;

    // Use cached rate array on GPU (avoids H2D transfer every call)
    // Only reallocate if size changed
    if (op->rate_array_size != rate_array_size) {
        // Size changed - need to reallocate
        if (op->rate_array_mapped && op->rate_array_cache != NULL) {
            double *old_rac = op->rate_array_cache;
            int old_size = op->rate_array_size;
            #pragma omp target exit data map(delete: old_rac[0:old_size])
        }
        if (op->rate_array_cache) free(op->rate_array_cache);

        op->rate_array_cache = (double*)malloc(rate_array_size * sizeof(double));
        op->rate_array_size = rate_array_size;
        op->rate_array_mapped = 0;
        rate_changed = 1;  // Force update since we reallocated
    }

    // Only transfer data to GPU if rate has changed
    if (rate_changed || !op->rate_array_mapped) {
        // Copy data to cache
        memcpy(op->rate_array_cache, rate_array, rate_array_size * sizeof(double));

        // Map or update cache on GPU
        double *rac = op->rate_array_cache;
        int ras = rate_array_size;
        if (!op->rate_array_mapped) {
            #pragma omp target enter data map(to: rac[0:ras])
            op->rate_array_mapped = 1;
        } else {
            #pragma omp target update to(rac[0:ras])
        }
    }

    // Use the GPU-resident cache
    double *gpu_rate_array = op->rate_array_cache;

    if (use_indices_into_rate) {
        // gpu_rate_array is full domain size, index with indices[k]
        // Apply rate to ALL cells (including ghosts for correct exchange)
        // but only sum local_influx for FULL cells (non-ghost) to avoid double-counting
        #pragma omp target teams distribute parallel for reduction(+:local_influx)
        for (int k = 0; k < num_indices; k++) {
            int i = indices[k];
            double rate = gpu_rate_array[i];
            double local_rate = ft * rate;

            if (rate >= 0.0) {
                stage_c[i] += local_rate;
                // Only count influx for full (non-ghost) cells
                if (is_full[k]) {
                    local_influx += local_rate * areas[k];
                }
            } else {
                // Negative rate - limit and scale momentum
                double height = stage_c[i] - bed_c[i];
                double actual_rate = (local_rate > -height) ? local_rate : -height;
                double scale_factor = (actual_rate < 0.0) ?
                    (actual_rate + height) / (height + 1.0e-10) : 1.0;

                stage_c[i] += actual_rate;
                xmom_c[i] *= scale_factor;
                ymom_c[i] *= scale_factor;
                // Only count influx for full (non-ghost) cells
                if (is_full[k]) {
                    local_influx += actual_rate * areas[k];
                }
            }
        }
    } else {
        // gpu_rate_array matches indices size, index with k
        // Apply rate to ALL cells (including ghosts for correct exchange)
        // but only sum local_influx for FULL cells (non-ghost) to avoid double-counting
        #pragma omp target teams distribute parallel for reduction(+:local_influx)
        for (int k = 0; k < num_indices; k++) {
            int i = indices[k];
            double rate = gpu_rate_array[k];
            double local_rate = ft * rate;

            if (rate >= 0.0) {
                stage_c[i] += local_rate;
                // Only count influx for full (non-ghost) cells
                if (is_full[k]) {
                    local_influx += local_rate * areas[k];
                }
            } else {
                // Negative rate - limit and scale momentum
                double height = stage_c[i] - bed_c[i];
                double actual_rate = (local_rate > -height) ? local_rate : -height;
                double scale_factor = (actual_rate < 0.0) ?
                    (actual_rate + height) / (height + 1.0e-10) : 1.0;

                stage_c[i] += actual_rate;
                xmom_c[i] *= scale_factor;
                ymom_c[i] *= scale_factor;
                // Only count influx for full (non-ghost) cells
                if (is_full[k]) {
                    local_influx += actual_rate * areas[k];
                }
            }
        }
    }

    // Rate array cache stays mapped on GPU for next call

    // Count FLOPs: 8 FLOPs per affected cell
    if (GD->flops.enabled) {
        GD->flops.rate_operator_flops += (uint64_t)op->num_indices * FLOPS_RATE_OPERATOR;
        GD->flops.rate_operator_calls++;
    }

    return local_influx;
}

