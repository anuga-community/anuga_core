// Analytic bed / initial-stage functions for the benchmark cases.
//
// Shared between the host domain build (setup.c) and the device-side
// initialiser (setup_device.c): the SAME expressions must produce the mesh
// quantities on both paths, so they live here once, marked declare target.
// Bodies are verbatim from the original setup.c; the only change is the
// function-static lookup tables becoming stack constants (same values).

#ifndef ANUGA_BENCH_SETUP_CASES_H
#define ANUGA_BENCH_SETUP_CASES_H

#include <math.h>

#include "setup.h"

// The RIVER case: a valley draining left to right with RIVER_DROP m of fall,
// a parabolic channel of half-width 6% of the domain carved 1.5 m into the
// valley floor, and floodplain banks rising 8 m from the channel edge to the
// domain sides.  The scale factors are relative to length_x/length_y so the
// case works at any --nx/--lenx.
#define RIVER_DROP        4.0    // m, upstream-to-downstream bed drop
#define RIVER_CH_DEPTH    1.5    // m, channel depth below the valley floor
#define RIVER_CH_HALFW    0.06   // fraction of length_y
#define RIVER_BANK_RISE   8.0    // m, floodplain rise from channel edge to side
#define RIVER_DAM_X       0.15   // fraction of length_x: reservoir extent
#define RIVER_FLOW_DEPTH  0.5    // m, initial river depth in the channel

// The BEACH case: a tsunami-style coastal inundation.  Deep ocean on the left
// (x = 0 is a reflective wall), a ramp up to a shoreline whose position varies
// with y (headlands and bays), and an inland slope that stays dry until the
// wave arrives.  The initial condition is a Gaussian hump of sea surface
// offshore, which splits into a shoreward wave and runs up the beach.
// Sea level is stage 0; everything is relative to length_x/length_y.
#define BEACH_DEPTH       20.0   // m, ocean depth
#define BEACH_RAMP_START  0.30   // fraction of length_x: where the ramp begins
#define BEACH_SHORE_X     0.60   // fraction of length_x: mean shoreline
#define BEACH_SHORE_AMP   0.08   // fraction of length_x: headland/bay amplitude
#define BEACH_INLAND_SLOPE 0.05  // dz/dx inland of the shoreline
#define BEACH_HUMP_AMP    3.0    // m, initial sea-surface displacement
#define BEACH_HUMP_X      0.15   // fraction of length_x
#define BEACH_HUMP_R      0.08   // fraction of length_x (Gaussian sigma)

#pragma omp declare target

static inline double bench_beach_bed(const bench_params *P, double x, double y) {
    const double u = x / P->length_x;
    const double v = y / P->length_y;
    const double shore = BEACH_SHORE_X
        + BEACH_SHORE_AMP * sin(4.0 * 3.14159265358979323846 * v);
    if (u < BEACH_RAMP_START) return -BEACH_DEPTH;
    if (u < shore)
        return -BEACH_DEPTH * (shore - u) / (shore - BEACH_RAMP_START);
    return (u - shore) * P->length_x * BEACH_INLAND_SLOPE;
}

static inline double bench_river_bed(const bench_params *P, double x, double y) {
    const double u  = x / P->length_x;
    const double dy = fabs(y - 0.5 * P->length_y);
    const double W  = RIVER_CH_HALFW * P->length_y;
    double z = RIVER_DROP * (1.0 - u);                    // downstream slope
    if (dy < W) {
        const double r = dy / W;
        z -= RIVER_CH_DEPTH * (1.0 - r * r);              // parabolic channel
    } else {
        z += RIVER_BANK_RISE * (dy - W) / (0.5 * P->length_y - W);
    }
    return z;
}

static inline double bench_bed_value(const bench_params *P, double x, double y) {
    if (P->which_case == BENCH_CASE_DAM) return 0.0;
    if (P->which_case == BENCH_CASE_RIVER) return bench_river_bed(P, x, y);
    if (P->which_case == BENCH_CASE_BEACH) return bench_beach_bed(P, x, y);

    // Five Gaussian humps on a gentle downstream slope.  Deterministic, smooth,
    // and tall enough that parts of the domain go dry.
    const double cx[5]  = {0.30, 0.55, 0.70, 0.45, 0.85};
    const double cy[5]  = {0.35, 0.65, 0.25, 0.85, 0.55};
    const double amp[5] = {6.0,  4.0,  5.0,  3.0,  7.0};
    const double rad[5] = {0.08, 0.06, 0.05, 0.07, 0.05};

    const double u = x / P->length_x;
    const double v = y / P->length_y;

    double z = 2.0 * u;   // slope
    for (int i = 0; i < 5; i++) {
        const double du = u - cx[i];
        const double dv = v - cy[i];
        z += amp[i] * exp(-(du * du + dv * dv) / (2.0 * rad[i] * rad[i]));
    }
    return z;
}

static inline double bench_stage_value(const bench_params *P, double x, double y, double z) {
    (void)y;
    switch (P->which_case) {
        case BENCH_CASE_DAM:
            return (x < 0.5 * P->length_x) ? P->dam_height : P->water_level;
        case BENCH_CASE_DAMBUMPS:
            return fmax(z, (x < 0.5 * P->length_x) ? P->dam_height : P->water_level);
        case BENCH_CASE_RIVER: {
            if (x < RIVER_DAM_X * P->length_x)
                return fmax(z, P->dam_height);             // full reservoir
            // Thin river: water surface follows the channel bottom downslope,
            // RIVER_FLOW_DEPTH deep at the centerline; banks stay dry.
            const double u = x / P->length_x;
            const double surf = RIVER_DROP * (1.0 - u) - RIVER_CH_DEPTH
                                + RIVER_FLOW_DEPTH;
            return fmax(z, surf);
        }
        case BENCH_CASE_BEACH: {
            const double du = x / P->length_x - BEACH_HUMP_X;
            const double dv = (y / P->length_y - 0.5) * (P->length_y / P->length_x);
            const double r2 = (du * du + dv * dv)
                            / (2.0 * BEACH_HUMP_R * BEACH_HUMP_R);
            return fmax(z, BEACH_HUMP_AMP * exp(-r2));   // sea level = 0
        }
        case BENCH_CASE_LAKE:
        default:
            return fmax(z, P->water_level);
    }
}

#pragma omp end declare target

#endif  // ANUGA_BENCH_SETUP_CASES_H
