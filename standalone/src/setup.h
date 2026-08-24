// Domain construction for the standalone ANUGA shallow-water benchmark.
//
// Builds a fully populated `struct gpu_domain` (mesh geometry, connectivity,
// quantities, boundary structures) from a bench_mesh, with no Python involved.
// The geometry formulas mirror General_mesh._compute_geometry / Neighbour_mesh
// so the arrays are identical to what ANUGA would hand the same kernels.

#ifndef ANUGA_BENCH_SETUP_H
#define ANUGA_BENCH_SETUP_H

#include "gpu_domain.h"
#include "mesh.h"

// Initial-condition / test cases.
typedef enum {
    BENCH_CASE_DAM = 0,      // flat bed, wet dam break -- all cells wet
    BENCH_CASE_DAMBUMPS,     // bumpy bed, dam break -- exercises wet/dry
    BENCH_CASE_LAKE          // bumpy bed, water at rest -- well-balancedness
} bench_case;

typedef struct {
    // Physics / algorithm parameters (defaults follow ANUGA's DE1 preset).
    double cfl;
    double g;
    double epsilon;
    double H0;                  // == minimum_allowed_height
    double minimum_allowed_height;
    double maximum_allowed_speed;
    double evolve_max_timestep;
    double beta_w, beta_w_dry, beta_uh, beta_uh_dry, beta_vh, beta_vh_dry;
    int    low_froude;
    int    extrapolate_velocity_second_order;
    int    use_sloped_mannings;

    // Problem setup.
    bench_case which_case;
    double     length_x, length_y;
    double     manning;
    double     water_level;     // still-water stage for LAKE / downstream depth
    double     dam_height;      // upstream stage for the dam-break cases
} bench_params;

typedef struct {
    struct gpu_domain GD;
    bench_mesh        mesh;
    void            **allocs;   // every host array we own, for teardown
    int               nallocs;
    int               allocs_cap;
} bench_domain;

void bench_params_defaults(bench_params *P);

// Allocate + fill everything.  Does not touch the device.
void bench_domain_build(bench_domain *B, const bench_mesh *M, const bench_params *P);

// gpu_domain_init + reflective boundary + gpu_domain_map_arrays.
void bench_domain_to_device(bench_domain *B, const bench_params *P, int verbose);

void bench_domain_free(bench_domain *B);

#endif // ANUGA_BENCH_SETUP_H
