// Python - C extension module for shallow_water.py
//
// To compile (Python2.6):
//  gcc -c swb2_domain_ext.c -I/usr/include/python2.6 -o domain_ext.o -Wall -O
//  gcc -shared swb2_domain_ext.o  -o swb2_domain_ext.so
//
// or use python compile.py
//
// See the module swb_domain.py for more documentation on
// how to use this module
//
//
// Ole Nielsen, GA 2004
// Stephen Roberts, ANU 2009
// Gareth Davies, GA 2011
// Jorge Galvez, NCI 2026 :D

#include "math.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

#include "sw_domain_math.h"
#include "util_ext.h"
#include "sw_domain.h"
#include "anuga_constants.h"

// Core kernels for CPU/GPU (shared with sw_domain_gpu_ext)
#include "core_kernels.h"

// Shared device helper functions (flux, rotation, gradient limiting, etc.)
#include "gpu/gpu_device_helpers.h"

// Flag to use unified kernels (can be toggled for testing)
#define USE_UNIFIED_KERNELS 1

// FIXME: Perhaps use the epsilon used elsewhere.

// Trick to compute n modulo d (n%d in python) when d is a power of 2
anuga_uint __mod_of_power_2(anuga_uint n, anuga_uint d)
{
  return (n & (d - 1));
}

// Wrapper for scalar interface (used by Cython)
// Calls unified gpu_flux_function_central from gpu_device_helpers.h
anuga_int __openmp__flux_function_central(double q_left0, double q_left1, double q_left2,
                                        double q_right0, double q_right1, double q_right2,
                                        double h_left, double h_right,
                                        double hle, double hre,
                                        double n1, double n2,
                                        double epsilon,
                                        double ze,
                                        double g,
                                        double *edgeflux0, double *edgeflux1, double *edgeflux2,
                                        double *max_speed,
                                        double *pressure_flux,
                                        anuga_int low_froude)
{
  double edgeflux[3];
  double q_left[3];
  double q_right[3];

  edgeflux[0] = *edgeflux0;
  edgeflux[1] = *edgeflux1;
  edgeflux[2] = *edgeflux2;

  q_left[0] = q_left0;
  q_left[1] = q_left1;
  q_left[2] = q_left2;

  q_right[0] = q_right0;
  q_right[1] = q_right1;
  q_right[2] = q_right2;

  gpu_flux_function_central(q_left, q_right,
                            h_left, h_right,
                            hle, hre,
                            n1, n2,
                            epsilon, ze, g,
                            edgeflux, max_speed,
                            pressure_flux,
                            low_froude);

  *edgeflux0 = edgeflux[0];
  *edgeflux1 = edgeflux[1];
  *edgeflux2 = edgeflux[2];

  return 0;
}
void inline __adjust_edgeflux_with_weir(double *edgeflux,
                                   const double h_left, double h_right,
                                   const double g, double weir_height,
                                   const double Qfactor,
                                   const double s1, double s2,
                                   const double h1, double h2,
                                   double *max_speed_local)
{
  // Adjust the edgeflux to agree with a weir relation [including
  // subergence], but smoothly vary to shallow water solution when
  // the flow over the weir is much deeper than the weir, or the
  // upstream/downstream water elevations are too similar
  double rw, rw2; // 'Raw' weir fluxes
  double rwRat, hdRat, hdWrRat, scaleFlux, minhd, maxhd;
  double w1, w2; // Weights for averaging
  double newFlux;
  double twothirds = (2.0 / 3.0);
  // Following constants control the 'blending' with the shallow water solution
  // They are now user-defined
  // double s1=0.9; // At this submergence ratio, begin blending with shallow water solution
  // double s2=0.95; // At this submergence ratio, completely use shallow water solution
  // double h1=1.0; // At this (tailwater height above weir) / (weir height) ratio, begin blending with shallow water solution
  // double h2=1.5; // At this (tailwater height above weir) / (weir height) ratio, completely use the shallow water solution

  if ((h_left <= 0.0) && (h_right <= 0.0))
  {
    return;
  }

  minhd = fmin(h_left, h_right);
  maxhd = fmax(h_left, h_right);
  // 'Raw' weir discharge = Qfactor*2/3*H*(2/3*g*H)**0.5
  rw = Qfactor * twothirds * maxhd * sqrt(twothirds * g * maxhd);
  // Factor for villemonte correction
  rw2 = Qfactor * twothirds * minhd * sqrt(twothirds * g * minhd);
  // Useful ratios
  rwRat = rw2 / fmax(rw, 1.0e-100);
  hdRat = minhd / fmax(maxhd, 1.0e-100);

  // (tailwater height above weir)/weir_height ratio

  hdWrRat = minhd / fmax(weir_height, 1.0e-100);

  // Villemonte (1947) corrected weir flow with submergence
  // Q = Q1*(1-Q2/Q1)**0.385
  rw = rw * pow(1.0 - rwRat, 0.385);

  if (h_right > h_left)
  {
    rw *= -1.0;
  }

  if ((hdRat < s2) & (hdWrRat < h2))
  {
    // Rescale the edge fluxes so that the mass flux = desired flux
    // Linearly shift to shallow water solution between hdRat = s1 and s2
    // and between hdWrRat = h1 and h2

    //
    // WEIGHT WITH RAW SHALLOW WATER FLUX BELOW
    // This ensures that as the weir gets very submerged, the
    // standard shallow water equations smoothly take over
    //

    // Weighted average constants to transition to shallow water eqn flow
    w1 = fmin(fmax(hdRat - s1, 0.) / (s2 - s1), 1.0);

    // Adjust again when the head is too deep relative to the weir height
    w2 = fmin(fmax(hdWrRat - h1, 0.) / (h2 - h1), 1.0);

    newFlux = (rw * (1.0 - w1) + w1 * edgeflux[0]) * (1.0 - w2) + w2 * edgeflux[0];

    if (fabs(edgeflux[0]) > 1.0e-100)
    {
      scaleFlux = newFlux / edgeflux[0];
    }
    else
    {
      scaleFlux = 0.;
    }

    scaleFlux = fmax(scaleFlux, 0.);
//

    // FIXME: Do this in a cleaner way
    // IDEA: Compute momentum flux implied by weir relations, and use
    //       those in a weighted average (rather than the rescaling trick here)
    // If we allow the scaling to momentum to be unbounded,
    // velocity spikes can arise for very-shallow-flooded walls
    edgeflux[0] = newFlux;
    edgeflux[1] *= fmin(scaleFlux, 10.);
    edgeflux[2] *= fmin(scaleFlux, 10.);
  }
//
  // Adjust the max speed
  if (fabs(edgeflux[0]) > 0.)
  {
    *max_speed_local = sqrt(g * (maxhd + weir_height)) + fabs(edgeflux[0] / (maxhd + 1.0e-12));
  }
  //*max_speed_local += fabs(edgeflux[0])/(maxhd+1.0e-100);
  //*max_speed_local *= fmax(scaleFlux, 1.0);

}

double __openmp__adjust_edgeflux_with_weir(double *edgeflux0, double *edgeflux1, double *edgeflux2,
                                           double h_left, double h_right,
                                           double g, double weir_height,
                                           double Qfactor,
                                           double s1, double s2,
                                           double h1, double h2,
                                           double *max_speed_local)
{

  double edgeflux[3];
  anuga_int ierr = 0;

  edgeflux[0] = *edgeflux0;
  edgeflux[1] = *edgeflux1;
  edgeflux[2] = *edgeflux2;

   __adjust_edgeflux_with_weir(edgeflux, h_left, h_right,
                                     g, weir_height,
                                     Qfactor, s1, s2, h1, h2,
                                     max_speed_local);
  *edgeflux0 = edgeflux[0];
  *edgeflux1 = edgeflux[1];
  *edgeflux2 = edgeflux[2];

  return ierr;
}

// Apply weir discharge theory correction to the edge flux
void apply_weir_discharge_correction(const struct domain * __restrict D, const EdgeData * __restrict E,
                                     const anuga_int k, const anuga_int ncol_riverwall_hydraulic_properties,
                                     const double g, double *edgeflux, double * __restrict max_speed) {

    anuga_int RiverWall_count = D->edge_river_wall_counter[E->ki];
    anuga_int ii = D->riverwall_rowIndex[RiverWall_count - 1] * ncol_riverwall_hydraulic_properties;

    double Qfactor = D->riverwall_hydraulic_properties[ii];
    double s1 = D->riverwall_hydraulic_properties[ii + 1];
    double s2 = D->riverwall_hydraulic_properties[ii + 2];
    double h1 = D->riverwall_hydraulic_properties[ii + 3];
    double h2 = D->riverwall_hydraulic_properties[ii + 4];

    double weir_height = fmax(D->riverwall_elevation[RiverWall_count - 1] - fmin(E->zl, E->zr), 0.);

    double h_left = fmax(D->stage_centroid_values[k] - E->z_half, 0.);
    double h_right = E->is_boundary
                         ? fmax(E->hc_n + E->zr - E->z_half, 0.)
                         : fmax(D->stage_centroid_values[E->n] - E->z_half, 0.);
    if (D->riverwall_elevation[RiverWall_count - 1] > fmax(E->zc, E->zc_n)) {
        __adjust_edgeflux_with_weir(edgeflux, h_left, h_right, g,
                                    weir_height, Qfactor, s1, s2, h1, h2, max_speed);
    }
}

// TODO: UNIFY WITH core_kernels.c (HIGH PRIORITY)
// GPU kernel exists (core_compute_fluxes_central) but missing features needed for unification:
//   1. Riverwall support (weir discharge adjustments) - see apply_weir_discharge_correction()
//   2. Boundary flux tracking (boundary_flux_sum for boundary_flux_integral_operator)
// Once these are added to core_compute_fluxes_central, enable USE_UNIFIED_KERNELS=1
double _openmp_compute_fluxes_central(const struct domain *__restrict D,
                                      double timestep)
{
  // Local variables
  anuga_int number_of_elements = D->number_of_elements;
  anuga_int n_riverwall_edges = D->number_of_riverwall_edges;
  //printf(" n edges %d \n", n_riverwall_edges);
  // anuga_int KI, KI2, KI3, B, RW, RW5, SubSteps;
  anuga_int substep_count;

  // // FIXME: limiting_threshold is not used for DE1
  anuga_int low_froude = D->low_froude;
  double g = D->g;
  double epsilon = D->epsilon;
  anuga_int ncol_riverwall_hydraulic_properties = D->ncol_riverwall_hydraulic_properties;

  static anuga_int call = 0; // Static local variable flagging already computed flux
  static anuga_int timestep_fluxcalls = 1;
  static anuga_int base_call = 1;

  call++; // Flag 'id' of flux calculation for this timestep

  if (D->timestep_fluxcalls != timestep_fluxcalls)
  {
    timestep_fluxcalls = D->timestep_fluxcalls;
    base_call = call;
  }


  anuga_int boundary_length = D->boundary_length;
  // Which substep of the timestepping method are we on?
  substep_count = (call - base_call) % D->timestep_fluxcalls;

  double local_timestep = 1.0e+100;
  double boundary_flux_sum_substep = 0.0;

      double pressure_flux;
      double max_speed_local;

// For all triangles
#pragma omp parallel for simd default(none) schedule(static) shared(D, substep_count, number_of_elements) \
    firstprivate(ncol_riverwall_hydraulic_properties, epsilon, g, low_froude)\
    private(pressure_flux,  max_speed_local) \
    reduction(min : local_timestep) reduction(+ : boundary_flux_sum_substep)
  for (anuga_int k = 0; k < number_of_elements; k++)
  {
      EdgeData edge_data;
      double edgeflux[3];
    double speed_max_last = 0.0;
    // Set explicit_update to zero for all conserved_quantities.
    // This assumes compute_fluxes called before forcing terms
    D->stage_explicit_update[k] = 0.0;
    D->xmom_explicit_update[k] = 0.0;
    D->ymom_explicit_update[k] = 0.0;

    // Loop through neighbours and compute edge flux for each
    for (anuga_int i = 0; i < 3; i++)
    {
      get_edge_data_central_flux(D,k,i,&edge_data);

      // Edge flux computation (triangle k, edge i)
      if (edge_data.h_left == 0.0 && edge_data.h_right == 0.0)
      {
        // If both heights are zero, then no flux
        edgeflux[0] = 0.0;
        edgeflux[1] = 0.0;
        edgeflux[2] = 0.0;
        max_speed_local = 0.0;
        pressure_flux = 0.0;
      }
      else
      {
        // Compute the fluxes using the central scheme (unified with GPU)
        gpu_flux_function_central(edge_data.ql, edge_data.qr,
                                  edge_data.h_left, edge_data.h_right,
                                  edge_data.hle, edge_data.hre,
                                  edge_data.normal_x, edge_data.normal_y,
                                  epsilon, edge_data.z_half, g,
                                  edgeflux, &max_speed_local, &pressure_flux,
                                  low_froude);
      }

    // Weir flux adjustment
    if (edge_data.is_riverwall) {
      apply_weir_discharge_correction(D, &edge_data, k, ncol_riverwall_hydraulic_properties, g, edgeflux, &max_speed_local);
    }

      // Multiply edgeflux by edgelength
      for (anuga_int j = 0; j < 3; j++)
      {
        edgeflux[j] *= -1.0 * edge_data.length;
      }
      // Update timestep based on edge i and possibly neighbour n
      // NOTE: We should only change the timestep on the 'first substep'
      // of the timestepping method [substep_count==0]
      if (substep_count == 0 && D->tri_full_flag[k] == 1 && max_speed_local > epsilon)
      {
        // Compute the 'edge-timesteps' (useful for setting flux_update_frequency)
        double edge_timestep = D->radii[k] * 1.0 / fmax(max_speed_local, epsilon);
        // Update the timestep
        // Apply CFL condition for triangles joining this edge (triangle k and triangle n)
        // CFL for triangle k
        local_timestep = fmin(local_timestep, edge_timestep);
        speed_max_last = fmax(speed_max_last, max_speed_local);
      }

      D->stage_explicit_update[k] += edgeflux[0];
      D->xmom_explicit_update[k] += edgeflux[1];
      D->ymom_explicit_update[k] += edgeflux[2];
      // If this cell is not a ghost, and the neighbour is a
      // boundary condition OR a ghost cell, then add the flux to the
      // boundary_flux_integral
      if (((edge_data.n < 0) & (D->tri_full_flag[k] == 1)) | ((edge_data.n >= 0) && ((D->tri_full_flag[k] == 1) & (D->tri_full_flag[edge_data.n] == 0))))
      {
        // boundary_flux_sum is an array with length = timestep_fluxcalls
        // For each sub-step, we put the boundary flux sum in.
        boundary_flux_sum_substep += edgeflux[0];
      }

      // bedslope_work contains all gravity related terms
      double pressuregrad_work = edge_data.length * (-g * 0.5 * (edge_data.h_left * edge_data.h_left - edge_data.hle * edge_data.hle - (edge_data.hle + edge_data.hc) * (edge_data.zl - edge_data.zc)) + pressure_flux);
      D->xmom_explicit_update[k] -= D->normals[edge_data.ki2] * pressuregrad_work;
      D->ymom_explicit_update[k] -= D->normals[edge_data.ki2 + 1] * pressuregrad_work;

    } // End edge i (and neighbour n)

    // Keep track of maximal speeds
    if (substep_count == 0){
      D->max_speed[k] = speed_max_last; // max_speed;
    }
    // Normalise triangle k by area and store for when all conserved
    // quantities get updated
    double inv_area = 1.0 / D->areas[k];
    D->stage_explicit_update[k] *= inv_area;
    D->xmom_explicit_update[k] *= inv_area;
    D->ymom_explicit_update[k] *= inv_area;

  } // End triangle k
  //    #pragma omp target exit data map(release:edgeflux)

  //   // Now add up stage, xmom, ymom explicit updates

  // variable to accumulate D->boundary_flux_sum[substep_count]
  D->boundary_flux_sum[substep_count] = boundary_flux_sum_substep;

  // Ensure we only update the timestep on the first call within each rk2/rk3 step
  if (substep_count == 0){
    timestep = local_timestep;
  }

  return timestep;
}

// Protect against the water elevation falling below the triangle bed
double _openmp_protect(const struct domain *__restrict D)
{

  double mass_error = 0.;

  double minimum_allowed_height = D->minimum_allowed_height;

  anuga_int number_of_elements = D->number_of_elements;

  // wc = D->stage_centroid_values;
  // zc = D->bed_centroid_values;
  // wv = D->stage_vertex_values;
  // xmomc = D->xmom_centroid_values;
  // ymomc = D->xmom_centroid_values;
  // areas = D->areas;

  // This acts like minimum_allowed height, but scales with the vertical
  // distance between the bed_centroid_value and the max bed_edge_value of
  // every triangle.
  // double minimum_relative_height=0.05;
  // anuga_int mass_added = 0;

  // Protect against inifintesimal and negative heights
  // if (maximum_allowed_speed < epsilon) {
#pragma omp parallel for schedule(static) reduction(+ : mass_error) firstprivate(minimum_allowed_height)
  for (anuga_int k = 0; k < number_of_elements; k++)
  {
    anuga_int k3 = 3 * k;
    double hc = D->stage_centroid_values[k] - D->bed_centroid_values[k];
    if (hc < minimum_allowed_height * 1.0)
    {
      // Set momentum to zero and ensure h is non negative
      D->xmom_centroid_values[k] = 0.;
      D->xmom_centroid_values[k] = 0.;
      if (hc <= 0.0)
      {
        double bmin = D->bed_centroid_values[k];
        // Minimum allowed stage = bmin

        // WARNING: ADDING MASS if wc[k]<bmin
        if (D->stage_centroid_values[k] < bmin)
        {
          mass_error += (bmin - D->stage_centroid_values[k]) * D->areas[k];
          // mass_added = 1; //Flag to warn of added mass

          D->stage_centroid_values[k] = bmin;

          // FIXME: Set vertex values as well. Seems that this shouldn't be
          // needed. However, from memory this is important at the first
          // time step, for 'dry' areas where the designated stage is
          // less than the bed centroid value
          D->stage_vertex_values[k3] = bmin;     // min(bmin, wc[k]); //zv[3*k]-minimum_allowed_height);
          D->stage_vertex_values[k3 + 1] = bmin; // min(bmin, wc[k]); //zv[3*k+1]-minimum_allowed_height);
          D->stage_vertex_values[k3 + 2] = bmin; // min(bmin, wc[k]); //zv[3*k+2]-minimum_allowed_height);
        }
      }
    }
  }

  // if(mass_added == 1){
  //   printf("Cumulative mass protection: %f m^3 \n", mass_error);
  // }

  return mass_error;
}

// Extrapolation helpers now unified in gpu_device_helpers.h:
// - gpu_find_qmin_and_qmax_dq1_dq2
// - gpu_limit_gradient
// - gpu_calc_edge_values_with_gradient
// - gpu_set_constant_edge_values
// - gpu_compute_qmin_qmax_from_dq1

// Extrapolate second order edge values from centroid values
// Unified: calls core_extrapolate_second_order_edge from core_kernels.c
void _openmp_extrapolate_second_order_edge_sw(struct domain *__restrict D)
{
    core_extrapolate_second_order_edge(D);
}

void _openmp_distribute_edges_to_vertices(struct domain *__restrict D)
{
    // Unified: calls core_distribute_edges_to_vertices from core_kernels.c
    core_distribute_edges_to_vertices(D);
}

void _openmp_manning_friction_flat_semi_implicit(const struct domain *__restrict D)
{
    // Unified: calls core_manning_friction_flat_semi_implicit from core_kernels.c
    core_manning_friction_flat_semi_implicit((struct domain *)D);
}




    

void _openmp_manning_friction_sloped_semi_implicit(const struct domain *__restrict D)
{
    // Unified: calls core_manning_friction_sloped_semi_implicit from core_kernels.c
    core_manning_friction_sloped_semi_implicit((struct domain *)D);
}

// TODO: PORT TO core_kernels.c (MEDIUM PRIORITY)
// Edge-based variant of sloped Manning friction
// Currently only flat and sloped (centroid-based) semi-implicit friction are in core_kernels.c
void _openmp_manning_friction_sloped_semi_implicit_edge_based(const struct domain *__restrict D)
{
  anuga_int k;
  const double one_third = 1.0 / 3.0;
  const double seven_thirds = 7.0 / 3.0;

  anuga_int N = D->number_of_elements;
  const double  g = D->g;
  const double  eps = D->minimum_allowed_height;
  
#pragma omp parallel for simd default(none) shared(D) schedule(static) \
        firstprivate(N, eps, g, seven_thirds, one_third)
for (k = 0; k < N; k++)
  {
    double S, h, z, z0, z1, z2, zs, zx, zy;
    double x0, y0, x1, y1, x2, y2;
    anuga_int k3, k6;

    double w = D->stage_centroid_values[k];
    double uh = D->xmom_centroid_values[k];
    double vh = D->ymom_centroid_values[k];
    double eta = D->friction_centroid_values[k];

    S = 0.0;
    k3 = 3 * k;
    
    // Get bathymetry
    z0 = D->bed_edge_values[k3 + 0];
    z1 = D->bed_edge_values[k3 + 1];
    z2 = D->bed_edge_values[k3 + 2];

    // Compute bed slope
    k6 = 6 * k; // base index

    
    x0 = D->edge_coordinates[k6 + 0];
    y0 = D->edge_coordinates[k6 + 1];
    x1 = D->edge_coordinates[k6 + 2];
    y1 = D->edge_coordinates[k6 + 3];
    x2 = D->edge_coordinates[k6 + 4];
    y2 = D->edge_coordinates[k6 + 5];

    
    if (eta > 1.0e-16)
    {
      _gradient(x0, y0, x1, y1, x2, y2, z0, z1, z2, &zx, &zy);

      zs = sqrt(1.0 + zx * zx + zy * zy);
      z = (z0 + z1 + z2) * one_third;

      h = w - z;
      if (h >= eps)
      {
        S = -g*eta*eta*zs * sqrt((uh*uh + vh*vh));
        S /= pow(h, seven_thirds); 
      }
    }
    D->xmom_semi_implicit_update[k] += S * uh;
    D->ymom_semi_implicit_update[k] += S * vh;
  }
}

// TODO: PORT TO core_kernels.c (LOW PRIORITY)
// Explicit (non semi-implicit) friction - rarely used, semi-implicit is preferred
// Original function for flat friction
void _openmp_manning_friction_flat(const double g, const double eps, const anuga_int N,
                                   double *__restrict w, double *__restrict z_centroid,
                                   double *__restrict uh, double *__restrict vh,
                                   double *__restrict eta, double *__restrict xmom_update, double *__restrict ymom_update)
{

  anuga_int k;
  const double seven_thirds = 7.0 / 3.0;

#pragma omp parallel for schedule(static) firstprivate(eps, g, seven_thirds)
  for (k = 0; k < N; k++)
  {
    double S, h, z, abs_mom;
    abs_mom = sqrt((uh[k] * uh[k] + vh[k] * vh[k]));
    S = 0.0;

    if (eta[k] > eps)
    {
      z = z_centroid[k];
      h = w[k] - z;
      if (h >= eps)
      {
        S = -g * eta[k] * eta[k] * abs_mom;
        S /= pow(h, seven_thirds); 
      }
    }
    xmom_update[k] += S * uh[k];
    ymom_update[k] += S * vh[k];
  }
}


// TODO: PORT TO core_kernels.c (LOW PRIORITY)
// Explicit (non semi-implicit) sloped friction - rarely used
void _openmp_manning_friction_sloped(const double g, const double eps, const anuga_int N,
                                     double *__restrict x_vertex, double *__restrict w, double *__restrict z_vertex,
                                     double *__restrict uh, double *__restrict vh,
                                     double *__restrict eta, double *__restrict xmom_update, double *__restrict ymom_update)
{

  const double one_third = 1.0 / 3.0;
  const double seven_thirds = 7.0 / 3.0;

#pragma omp parallel for schedule(static) firstprivate(eps, g, one_third, seven_thirds)
  for (anuga_int k = 0; k < N; k++)
  {
    double S = 0.0;
    anuga_int k3 = 3 * k;
    // Get bathymetry
    double z0 = z_vertex[k3 + 0];
    double z1 = z_vertex[k3 + 1];
    double z2 = z_vertex[k3 + 2];

    // Compute bed slope
    anuga_int k6 = 6 * k; // base index

    double x0 = x_vertex[k6 + 0];
    double y0 = x_vertex[k6 + 1];
    double x1 = x_vertex[k6 + 2];
    double y1 = x_vertex[k6 + 3];
    double x2 = x_vertex[k6 + 4];
    double y2 = x_vertex[k6 + 5];

    if (eta[k] > eps)
    {
      double zx, zy, zs, z, h;
      _gradient(x0, y0, x1, y1, x2, y2, z0, z1, z2, &zx, &zy);

      zs = sqrt(1.0 + zx * zx + zy * zy);
      z = (z0 + z1 + z2) * one_third;
      h = w[k] - z;
      if (h >= eps)
      {
        S = -g * eta[k] * eta[k] * zs * sqrt((uh[k] * uh[k] + vh[k] * vh[k]));
        S /= pow(h, seven_thirds); 
      }
    }
    xmom_update[k] += S * uh[k];
    ymom_update[k] += S * vh[k];
  }
}

// TODO: PORT TO core_kernels.c (LOW PRIORITY)
// Explicit (non semi-implicit) sloped edge-based friction - rarely used
void _openmp_manning_friction_sloped_edge_based(const double g, const double eps, const anuga_int N,
                                     double *__restrict x_edge, double *__restrict w, double *__restrict z_edge,
                                     double *__restrict uh, double *__restrict vh,
                                     double *__restrict eta, double *__restrict xmom_update, double *__restrict ymom_update)
{

  const double one_third = 1.0 / 3.0;
  const double seven_thirds = 7.0 / 3.0;

#pragma omp parallel for schedule(static) firstprivate(eps, g, one_third, seven_thirds)
  for (anuga_int k = 0; k < N; k++)
  {
    double S = 0.0;
    anuga_int k3 = 3 * k;
    // Get bathymetry
    double z0 = z_edge[k3 + 0];
    double z1 = z_edge[k3 + 1];
    double z2 = z_edge[k3 + 2];

    // Compute bed slope
    anuga_int k6 = 6 * k; // base index

    double x0 = x_edge[k6 + 0];
    double y0 = x_edge[k6 + 1];
    double x1 = x_edge[k6 + 2];
    double y1 = x_edge[k6 + 3];
    double x2 = x_edge[k6 + 4];
    double y2 = x_edge[k6 + 5];

    if (eta[k] > eps)
    {
      double zx, zy, zs, z, h;
      _gradient(x0, y0, x1, y1, x2, y2, z0, z1, z2, &zx, &zy);

      zs = sqrt(1.0 + zx * zx + zy * zy);
      z = (z0 + z1 + z2) * one_third;
      h = w[k] - z;
      if (h >= eps)
      {
        S = -g * eta[k] * eta[k] * zs * sqrt((uh[k] * uh[k] + vh[k] * vh[k]));
        S /= pow(h, seven_thirds); 
      }
    }
    xmom_update[k] += S * uh[k];
    ymom_update[k] += S * vh[k];
  }
}


// Computational function for flux computation
anuga_int _openmp_fix_negative_cells(const struct domain *__restrict D)
{
  anuga_int num_negative_cells = 0;

#pragma omp parallel for schedule(static) reduction(+ : num_negative_cells)
  for (anuga_int k = 0; k < D->number_of_elements; k++)
  {
    if ((D->stage_centroid_values[k] - D->bed_centroid_values[k] < 0.0) & (D->tri_full_flag[k] > 0))
    {
      num_negative_cells = num_negative_cells + 1;
      D->stage_centroid_values[k] = D->bed_centroid_values[k];
      D->xmom_centroid_values[k] = 0.0;
      D->ymom_centroid_values[k] = 0.0;
    }
  }
  return num_negative_cells;
}


anuga_int _openmp_gravity(const struct domain *__restrict D) {

    anuga_int k, N, k3, k6;
    double g, avg_h, zx, zy;
    double x0, y0, x1, y1, x2, y2, z0, z1, z2;

    g = D->g;
    N = D->number_of_elements;

    for (k = 0; k < N; k++) {
        k3 = 3 * k; // base index

        // Get bathymetry
        z0 = (D->bed_vertex_values)[k3 + 0];
        z1 = (D->bed_vertex_values)[k3 + 1];
        z2 = (D->bed_vertex_values)[k3 + 2];

        //printf("z0 %g, z1 %g, z2 %g \n",z0,z1,z2);

        // Get average depth from centroid values
        avg_h = (D->stage_centroid_values)[k] - (D->bed_centroid_values)[k];

        //printf("avg_h  %g \n",avg_h);
        // Compute bed slope
        k6 = 6 * k; // base index

        x0 = (D->vertex_coordinates)[k6 + 0];
        y0 = (D->vertex_coordinates)[k6 + 1];
        x1 = (D->vertex_coordinates)[k6 + 2];
        y1 = (D->vertex_coordinates)[k6 + 3];
        x2 = (D->vertex_coordinates)[k6 + 4];
        y2 = (D->vertex_coordinates)[k6 + 5];

        //printf("x0 %g, y0 %g, x1 %g, y1 %g, x2 %g, y2 %g \n",x0,y0,x1,y1,x2,y2);
        _gradient(x0, y0, x1, y1, x2, y2, z0, z1, z2, &zx, &zy);

        //printf("zx %g, zy %g \n",zx,zy);

        // Update momentum
        (D->xmom_explicit_update)[k] += -g * zx*avg_h;
        (D->ymom_explicit_update)[k] += -g * zy*avg_h;
    }
    return 0;
}

anuga_int _openmp_gravity_wb(const struct domain *__restrict D) {

    anuga_int i, k, N, k3, k6;
    double g, avg_h, wx, wy, fact;
    double x0, y0, x1, y1, x2, y2;
    double hh[3];
    double w0, w1, w2;
    double sidex, sidey, area;
    double n0, n1;

    g = D->g;

    N = D->number_of_elements;
    for (k = 0; k < N; k++) {
        k3 = 3 * k; // base index

        //------------------------------------
        // Calculate side terms -ghw_x term
        //------------------------------------

        // Get vertex stage values for gradient calculation
        w0 = (D->stage_vertex_values)[k3 + 0];
        w1 = (D->stage_vertex_values)[k3 + 1];
        w2 = (D->stage_vertex_values)[k3 + 2];

        // Compute stage slope
        k6 = 6 * k; // base index

        x0 = (D->vertex_coordinates)[k6 + 0];
        y0 = (D->vertex_coordinates)[k6 + 1];
        x1 = (D->vertex_coordinates)[k6 + 2];
        y1 = (D->vertex_coordinates)[k6 + 3];
        x2 = (D->vertex_coordinates)[k6 + 4];
        y2 = (D->vertex_coordinates)[k6 + 5];

        //printf("x0 %g, y0 %g, x1 %g, y1 %g, x2 %g, y2 %g \n",x0,y0,x1,y1,x2,y2);
        _gradient(x0, y0, x1, y1, x2, y2, w0, w1, w2, &wx, &wy);

        avg_h = (D->stage_centroid_values)[k] - (D->bed_centroid_values)[k];

        // Update using -ghw_x term
        (D->xmom_explicit_update)[k] += -g * wx*avg_h;
        (D->ymom_explicit_update)[k] += -g * wy*avg_h;

        //------------------------------------
        // Calculate side terms \sum_i 0.5 g l_i h_i^2 n_i
        //------------------------------------

        // Getself.stage_c = self.domain.quantities['stage'].centroid_values edge depths
        hh[0] = (D->stage_edge_values)[k3 + 0] - (D->bed_edge_values)[k3 + 0];
        hh[1] = (D->stage_edge_values)[k3 + 1] - (D->bed_edge_values)[k3 + 1];
        hh[2] = (D->stage_edge_values)[k3 + 2] - (D->bed_edge_values)[k3 + 2];


        //printf("h0,1,2 %f %f %f\n",hh[0],hh[1],hh[2]);

        // Calculate the side correction term
        sidex = 0.0;
        sidey = 0.0;
        for (i = 0; i < 3; i++) {
            n0 = (D->normals)[k6 + 2 * i];
            n1 = (D->normals)[k6 + 2 * i + 1];

            //printf("n0, n1 %i %g %g\n",i,n0,n1);
            fact = -0.5 * g * hh[i] * hh[i] * (D->edgelengths)[k3 + i];
            sidex = sidex + fact*n0;
            sidey = sidey + fact*n1;
        }

        // Update momentum with side terms
        area = (D->areas)[k];
        (D->xmom_explicit_update)[k] += -sidex / area;
        (D->ymom_explicit_update)[k] += -sidey / area;

    }
    return 0;
}



anuga_int _openmp_update_conserved_quantities(const struct domain *__restrict D,
                                              const double timestep)
{
    // Unified: calls core_update_conserved_quantities from core_kernels.c
    core_update_conserved_quantities((struct domain *)D, timestep);
    return 0;
}

anuga_int _openmp_saxpy_conserved_quantities(const struct domain *__restrict D, 
                                             const double a, 
                                             const double b, 
                                             const double c)
{
  // This function performs a SAXPY operation on the centroid values and backup values.
  //
  // It does a standard SAXPY operation and then multiplies through a constant c.
  // to deal with some numerical issues when using a = 1/3 and b = 2/3 and maintaining
  // positive values.
  

  anuga_int N = D->number_of_elements;
  // double a_c = a / c;
  // double bc_a = b *c /a;
  double c_inv = 1.0 / c;

  #pragma omp parallel for simd schedule(static)
  for (anuga_int i = 0; i < N; i++)
  {
    D->stage_centroid_values[i] = a*D->stage_centroid_values[i] + b*D->stage_backup_values[i];
    D->xmom_centroid_values[i]  = a*D->xmom_centroid_values[i] + b*D->xmom_backup_values[i];
    D->ymom_centroid_values[i]  = a*D->ymom_centroid_values[i] + b*D->ymom_backup_values[i];
  }

  if (c != 1.0)
  {
    #pragma omp parallel for simd schedule(static)
    for (anuga_int i = 0; i < N; i++)
    {
      D->stage_centroid_values[i] *= c_inv;
      D->xmom_centroid_values[i]  *= c_inv;
      D->ymom_centroid_values[i]  *= c_inv;
    }
  }

  // FIXME: Should get this to work as it should be faster than the above
  // // stage
  // anuga_dscal(N, a, D->stage_centroid_values, 1);
  // anuga_daxpy(N, b, D->stage_backup_values, 1, D->stage_centroid_values, 1);
  // if (c != 1.0) {
  //   anuga_dscal(N, c_inv, D->stage_centroid_values, 1);
  // }
  
  // // xmom
  // anuga_dscal(N, a, D->xmom_centroid_values, 1);
  // anuga_daxpy(N, b, D->xmom_backup_values, 1, D->xmom_centroid_values, 1);
  // if (c != 1.0) {
  //   anuga_dscal(N, c_inv, D->xmom_centroid_values, 1);
  // }


  // // ymom
  // anuga_dscal(N, a, D->ymom_centroid_values, 1);
  // anuga_daxpy(N, b, D->ymom_backup_values, 1, D->ymom_centroid_values, 1);
  // if (c != 1.0) {
  //   anuga_dscal(N, c_inv, D->ymom_centroid_values, 1);
  // }

  return 0;
}

anuga_int _openmp_backup_conserved_quantities(const struct domain *__restrict D)
{
    // Unified: calls core_backup_conserved_quantities from core_kernels.c
    core_backup_conserved_quantities((struct domain *)D);
    return 0;
}

void _openmp_set_omp_num_threads(anuga_int num_threads)
{
  // Set the number of threads for OpenMP
  // This is a global setting and will affect all subsequent OpenMP parallel regions
  omp_set_num_threads(num_threads);
}

// PORTED TO GPU: See gpu_evaluate_reflective_boundary() in gpu_boundaries.c
// TODO: NOT UNIFIED - GPU and CPU use different architectural approaches:
//   - GPU: Pre-collects ALL boundary info during init, evaluates all edges in one kernel call
//   - CPU: Python calls this C function per boundary segment/tag, passing arrays each time
// Unifying would require changing how mode 1 handles boundaries (Python-side changes),
// which could break existing functionality. Keep separate implementations for now.
void _openmp_evaluate_reflective_segment(const struct domain *__restrict D, anuga_int N,
   anuga_int *edge_segment, anuga_int *vol_ids, anuga_int *edge_ids){

  anuga_int boundary_length = D->boundary_length;
  anuga_int number_of_edges = N;
  anuga_int number_of_elements = D->number_of_elements; 

    #pragma omp parallel for schedule(static)
     for(anuga_int k = 0; k < number_of_edges; k++){


      // get vol_ids 
      anuga_int edge_segment_id = edge_segment[k];
      anuga_int vid = vol_ids[k];
      anuga_int edge_id = edge_ids[k];
      double n1 = D->normals[vid * 6 + 2 * edge_id];
      double n2 = D->normals[vid * 6 + 2 * edge_id + 1];

      D->stage_boundary_values[edge_segment_id] = D->stage_edge_values[3 * vid + edge_id];
      // the bed is the elevation
      D->bed_boundary_values[edge_segment_id] = D->bed_edge_values[3 * vid + edge_id];
      D->height_boundary_values[edge_segment_id] = D->height_edge_values[3 * vid + edge_id];

      double q1 = D->xmom_edge_values[3 * vid + edge_id];
      double q2 = D->ymom_edge_values[3 * vid + edge_id];

      double r1 = -q1*n1 - q2*n2;
      double r2 = -q1*n2 + q2*n1;

      double x_mom_boundary_value = n1*r1 - n2*r2;
      double y_mom_boundary_value = n2*r1 + n1*r2;

      D->xmom_boundary_values[edge_segment_id] = x_mom_boundary_value;
      D->ymom_boundary_values[edge_segment_id] = y_mom_boundary_value;

	  // FIXME SR: Check that we really need to work with velocities
      q1 = D->xvelocity_edge_values[3 * vid + edge_id];
      q2 = D->yvelocity_edge_values[3 * vid + edge_id];

      r1 = q1*n1 + q2*n2;
      r2 = q1*n2 - q2*n1;

      double x_vel_boundary_value = n1*r1 - n2*r2;
      double y_vel_boundary_value = n2*r1 + n1*r2;

      D->xvelocity_boundary_values[edge_segment_id] = x_vel_boundary_value;
      D->yvelocity_boundary_values[edge_segment_id] = y_vel_boundary_value;

     }


}
