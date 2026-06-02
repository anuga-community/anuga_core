// GPU-accelerated max-quantities collection operator.
//
// Tracks the running maximum of stage, depth, speed, and momentum magnitude
// (||(uh, vh)||) over every cell.  All four max arrays are kept device-resident
// between calls so per-timestep cost is one parallel pass with no host<->device
// transfer.  A D2H sync is only needed when the Python operator wants to write
// the current maxima to the SWW file (at yield steps).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "gpu_domain.h"
#include "gpu_device_helpers.h"
#include "gpu_omp_macros.h"

// Initialise the max-quantities operator: allocates host arrays, fills initial
// values, and maps them to the device if GPU is already initialised.
// Returns 0 on success, -1 on allocation failure.
int gpu_max_quantities_init(struct gpu_domain *GD, int n, double velocity_zero_height)
{
    struct max_quantities_info *MQ = &GD->max_qty;

    if (MQ->initialized) return 0;

    MQ->n                   = n;
    MQ->velocity_zero_height = velocity_zero_height;
    MQ->mapped              = 0;

    MQ->max_stage = (double*)malloc(n * sizeof(double));
    MQ->max_depth = (double*)malloc(n * sizeof(double));
    MQ->max_speed = (double*)malloc(n * sizeof(double));
    MQ->max_uh    = (double*)malloc(n * sizeof(double));

    if (!MQ->max_stage || !MQ->max_depth || !MQ->max_speed || !MQ->max_uh) {
        fprintf(stderr, "ERROR: gpu_max_quantities_init: allocation failed\n");
        free(MQ->max_stage); free(MQ->max_depth);
        free(MQ->max_speed); free(MQ->max_uh);
        MQ->max_stage = MQ->max_depth = MQ->max_speed = MQ->max_uh = NULL;
        return -1;
    }

    // stage initialised to large negative; depth/speed/uh initialised to 0
    for (int i = 0; i < n; i++) {
        MQ->max_stage[i] = -1.0e38;
        MQ->max_depth[i] = 0.0;
        MQ->max_speed[i] = 0.0;
        MQ->max_uh[i]    = 0.0;
    }

    if (GD->gpu_initialized) {
        int ni = n;
        double *ms  = MQ->max_stage;
        double *md  = MQ->max_depth;
        double *msp = MQ->max_speed;
        double *mu  = MQ->max_uh;
        #pragma omp target enter data map(to: ms[0:ni], md[0:ni], msp[0:ni], mu[0:ni])
        MQ->mapped = 1;
    }

    MQ->initialized = 1;
    return 0;
}

// Single-pass kernel: reads stage/bed/xmom/ymom (already device-resident via
// gpu_domain_map_arrays) and updates the four device-resident max arrays.
void gpu_max_quantities_update(struct gpu_domain *GD)
{
    struct max_quantities_info *MQ = &GD->max_qty;
    if (!MQ->initialized) return;

    int n   = MQ->n;
    double vzh = MQ->velocity_zero_height;

    double * restrict stage_c = GD->D.stage_centroid_values;
    double * restrict bed_c   = GD->D.bed_centroid_values;
    double * restrict xmom_c  = GD->D.xmom_centroid_values;
    double * restrict ymom_c  = GD->D.ymom_centroid_values;

    double * restrict max_stage = MQ->max_stage;
    double * restrict max_depth = MQ->max_depth;
    double * restrict max_speed = MQ->max_speed;
    double * restrict max_uh    = MQ->max_uh;

    OMP_PARALLEL_LOOP
    for (int i = 0; i < n; i++) {
        double s  = stage_c[i];
        double b  = bed_c[i];
        double xm = xmom_c[i];
        double ym = ymom_c[i];

        double mn = sqrt(xm * xm + ym * ym);
        double d  = s - b;
        if (d < 0.0) d = 0.0;
        double v  = (d > vzh) ? mn / d : 0.0;

        if (s  > max_stage[i]) max_stage[i] = s;
        if (mn > max_uh[i])    max_uh[i]    = mn;
        if (d  > max_depth[i]) max_depth[i] = d;
        if (v  > max_speed[i]) max_speed[i] = v;
    }
}

// Sync the four max arrays from device to host, then copy into caller-supplied
// output buffers (which may be the Python operator's numpy array buffers).
void gpu_max_quantities_get(struct gpu_domain *GD,
                            double *out_stage, double *out_depth,
                            double *out_speed, double *out_uh)
{
    struct max_quantities_info *MQ = &GD->max_qty;
    if (!MQ->initialized) return;

    int n = MQ->n;

    if (MQ->mapped) {
        double *ms  = MQ->max_stage;
        double *md  = MQ->max_depth;
        double *msp = MQ->max_speed;
        double *mu  = MQ->max_uh;
        #pragma omp target update from(ms[0:n], md[0:n], msp[0:n], mu[0:n])
    }

    memcpy(out_stage, MQ->max_stage, n * sizeof(double));
    memcpy(out_depth, MQ->max_depth, n * sizeof(double));
    memcpy(out_speed, MQ->max_speed, n * sizeof(double));
    memcpy(out_uh,    MQ->max_uh,    n * sizeof(double));
}

// Unmap device arrays, free host memory.
void gpu_max_quantities_finalize(struct gpu_domain *GD)
{
    struct max_quantities_info *MQ = &GD->max_qty;
    if (!MQ->initialized) return;

    if (MQ->mapped) {
        int n   = MQ->n;
        double *ms  = MQ->max_stage;
        double *md  = MQ->max_depth;
        double *msp = MQ->max_speed;
        double *mu  = MQ->max_uh;
        #pragma omp target exit data map(delete: ms[0:n], md[0:n], msp[0:n], mu[0:n])
        MQ->mapped = 0;
    }

    free(MQ->max_stage);
    free(MQ->max_depth);
    free(MQ->max_speed);
    free(MQ->max_uh);

    MQ->max_stage = MQ->max_depth = MQ->max_speed = MQ->max_uh = NULL;
    MQ->n           = 0;
    MQ->initialized = 0;
}
