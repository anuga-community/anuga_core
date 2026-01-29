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
        RO->num_operators++;
        return op_id;
    }

    // Allocate and copy arrays
    op->indices = (int*)malloc(num_indices * sizeof(int));
    op->areas = (double*)malloc(num_indices * sizeof(double));

    if (!op->indices || !op->areas) {
        fprintf(stderr, "Failed to allocate rate_operator arrays\n");
        if (op->indices) free(op->indices);
        if (op->areas) free(op->areas);
        op->active = 0;
        return -1;
    }

    memcpy(op->indices, indices, num_indices * sizeof(int));
    memcpy(op->areas, areas, num_indices * sizeof(double));

    if (num_full > 0 && full_indices != NULL) {
        op->full_indices = (int*)malloc(num_full * sizeof(int));
        if (op->full_indices) {
            memcpy(op->full_indices, full_indices, num_full * sizeof(int));
        }
    } else {
        op->full_indices = NULL;
    }

    // Map to GPU immediately if GPU is already initialized
    if (GD->gpu_initialized) {
        int ni = op->num_indices;
        int *idx = op->indices;
        double *ar = op->areas;
        #pragma omp target enter data map(to: idx[0:ni], ar[0:ni])
        op->mapped = 1;
    }

    RO->num_operators++;

    //printf("[Rank %d] Rate_operator %d initialized: %d indices, %d full (GPU mapped: %d) "
    //       "indices=%p areas=%p\n",
    //       GD->rank, op_id, num_indices, num_full, op->mapped,
    //       (void*)op->indices, (void*)op->areas);
    //fflush(stdout);

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
        #pragma omp target exit data map(delete: idx[0:ni], ar[0:ni])
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

    op->indices = NULL;
    op->areas = NULL;
    op->full_indices = NULL;
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
        #pragma omp target enter data map(to: idx[0:ni], ar[0:ni])
        op->mapped = 1;
    }

    int num_indices = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict areas = op->areas;

    // Domain arrays (restrict enables better optimization)
    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict xmom_c = GD->D.xmom_centroid_values;
    double * restrict ymom_c = GD->D.ymom_centroid_values;
    double * restrict bed_c = GD->D.bed_centroid_values;

    double local_rate = factor * timestep * rate;
    double local_influx = 0.0;

    if (rate >= 0.0) {
        // Simple positive rate - just add to stage
        // Reduction for mass tracking
        #pragma omp target teams distribute parallel for reduction(+:local_influx)
        for (int k = 0; k < num_indices; k++) {
            int i = indices[k];
            stage_c[i] += local_rate;
            local_influx += local_rate * areas[k];
        }
    } else {
        // Negative rate (extraction) - need to limit and scale momentum
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

            local_influx += actual_rate * areas[k];
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
        #pragma omp target enter data map(to: idx[0:ni], ar[0:ni])
        op->mapped = 1;
    }

    int num_indices = op->num_indices;
    int * restrict indices = op->indices;
    double * restrict areas = op->areas;

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
        #pragma omp target teams distribute parallel for reduction(+:local_influx)
        for (int k = 0; k < num_indices; k++) {
            int i = indices[k];
            double rate = gpu_rate_array[i];
            double local_rate = ft * rate;

            if (rate >= 0.0) {
                stage_c[i] += local_rate;
                local_influx += local_rate * areas[k];
            } else {
                // Negative rate - limit and scale momentum
                double height = stage_c[i] - bed_c[i];
                double actual_rate = (local_rate > -height) ? local_rate : -height;
                double scale_factor = (actual_rate < 0.0) ?
                    (actual_rate + height) / (height + 1.0e-10) : 1.0;

                stage_c[i] += actual_rate;
                xmom_c[i] *= scale_factor;
                ymom_c[i] *= scale_factor;
                local_influx += actual_rate * areas[k];
            }
        }
    } else {
        // gpu_rate_array matches indices size, index with k
        #pragma omp target teams distribute parallel for reduction(+:local_influx)
        for (int k = 0; k < num_indices; k++) {
            int i = indices[k];
            double rate = gpu_rate_array[k];
            double local_rate = ft * rate;

            if (rate >= 0.0) {
                stage_c[i] += local_rate;
                local_influx += local_rate * areas[k];
            } else {
                // Negative rate - limit and scale momentum
                double height = stage_c[i] - bed_c[i];
                double actual_rate = (local_rate > -height) ? local_rate : -height;
                double scale_factor = (actual_rate < 0.0) ?
                    (actual_rate + height) / (height + 1.0e-10) : 1.0;

                stage_c[i] += actual_rate;
                xmom_c[i] *= scale_factor;
                ymom_c[i] *= scale_factor;
                local_influx += actual_rate * areas[k];
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

