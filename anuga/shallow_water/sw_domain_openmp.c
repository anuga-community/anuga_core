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

// Wrapper for Cython - calls gpu_rotate from gpu_device_helpers.h
anuga_int __rotate(double *q, double n1, double n2)
{
  gpu_rotate(q, n1, n2);
  return 0;
}

// Unified: calls core_compute_fluxes_central from core_kernels.c
// Handles substep tracking via static variables (per-module state)
double _openmp_compute_fluxes_central(const struct domain *__restrict D,
                                      double timestep)
{
  // Static variables for substep tracking
  static anuga_int call = 0;
  static anuga_int timestep_fluxcalls = 1;
  static anuga_int base_call = 1;

  call++;

  if (D->timestep_fluxcalls != timestep_fluxcalls) {
    timestep_fluxcalls = D->timestep_fluxcalls;
    base_call = call;
  }

  // Which substep of the timestepping method are we on?
  int substep_count = (call - base_call) % D->timestep_fluxcalls;

  // Call unified flux computation
  double local_timestep = core_compute_fluxes_central((struct domain *)D, substep_count, D->timestep_fluxcalls);

  // Return timestep only on first substep
  if (substep_count == 0) {
    timestep = local_timestep;
  }

  return timestep;
}

// Protect against the water elevation falling below the triangle bed
// Unified: calls core_protect from core_kernels.c
double _openmp_protect(const struct domain *__restrict D)
{
    return core_protect((struct domain *)D);
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

void _openmp_ader_ck_predictor(struct domain *__restrict D, double dt)
{
    // ADER Cauchy-Kovalewski predictor — calls core_ader_ck_predictor from core_kernels.c
    core_ader_ck_predictor(D, dt);
}

void _openmp_ader_ck_predictor_edge(struct domain *__restrict D, double dt)
{
    // Fused ADER-2 predictor: writes Q^{n+1/2} to edge values, centroids untouched
    core_ader_ck_predictor_edge(D, dt);
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
    // Unified: calls core_saxpy_conserved_quantities from core_kernels.c
    // Computes Q = (a*Q + b*Q_backup) / c
    core_saxpy_conserved_quantities((struct domain *)D, a, b, c);
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

  anuga_int number_of_edges = N;

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
