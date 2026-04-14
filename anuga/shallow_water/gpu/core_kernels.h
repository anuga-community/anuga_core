// Core kernels for CPU/GPU execution
//
// These functions operate on struct domain* directly (no MPI dependency)
// and use OpenMP for parallelization. When compiled with -DCPU_ONLY_MODE,
// they run on CPU multicore. Otherwise, they can target GPU via OpenMP offload.
//
// Both sw_domain_openmp_ext and sw_domain_gpu_ext use these same kernels.

#ifndef CORE_KERNELS_H
#define CORE_KERNELS_H

#include "sw_domain.h"

// Extrapolate centroid values to edge values (second-order reconstruction)
void core_extrapolate_second_order_edge(struct domain *D);

// Distribute edge values to vertices
void core_distribute_edges_to_vertices(struct domain *D);

// Update conserved quantities with explicit/semi-implicit updates
void core_update_conserved_quantities(struct domain *D, double timestep);

// Backup conserved quantities for RK2 timestepping
void core_backup_conserved_quantities(struct domain *D);

// RK2 combination: Q = a*Q_current + b*Q_backup + c*(something)
void core_saxpy_conserved_quantities(struct domain *D, double a, double b, double c);

// Protect against negative water depths, returns mass error
double core_protect(struct domain *D);

// Fix negative cells (after update)
int core_fix_negative_cells(struct domain *D);

// Manning friction (flat, semi-implicit)
void core_manning_friction_flat_semi_implicit(struct domain *D);

// Manning friction (sloped, semi-implicit)
void core_manning_friction_sloped_semi_implicit(struct domain *D);

// Gravity term
int core_gravity(struct domain *D);

// Gravity term (well-balanced)
int core_gravity_wb(struct domain *D);

// Compute fluxes using central upwind scheme
// Returns minimum timestep, stores boundary flux sum in boundary_flux_sum[substep_count]
// substep_count: which substep of RK timestepping (0 = first, only update timestep on first)
// timestep_fluxcalls: total number of flux calls per timestep (for boundary flux array indexing)
double core_compute_fluxes_central(struct domain *D, int substep_count, int timestep_fluxcalls);

// Compute fluxes using HLLC Riemann solver (Toro 2001).
// Same interface and semantics as core_compute_fluxes_central; only the
// per-edge flux function differs.
double core_compute_fluxes_hllc(struct domain *D, int substep_count, int timestep_fluxcalls);

#endif // CORE_KERNELS_H
