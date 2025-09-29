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

// FIXME: Perhaps use the epsilon used elsewhere.

// Trick to compute n modulo d (n%d in python) when d is a power of 2
anuga_uint __mod_of_power_2(anuga_uint n, anuga_uint d)
{
  return (n & (d - 1));
}

// Computational function for rotation
anuga_int __rotate(double *q, const double n1, const double n2)
{
  /*Rotate the last  2 coordinates of q (q[1], q[2])
    from x,y coordinates to coordinates based on normal vector (n1, n2).

    Result is returned in array 2x1 r
    To rotate in opposite direction, call rotate with (q, n1, -n2)

    Contents of q are changed by this function */

  double q1, q2;

  // Shorthands
  q1 = q[1]; // x coordinate
  q2 = q[2]; // y coordinate

  // Rotate
  q[1] = n1 * q1 + n2 * q2;
  q[2] = -n2 * q1 + n1 * q2;

  return 0;
}
// general function to replace the repeated if statements for the velocity terms
static inline void compute_velocity_terms(
    const double h, const double h_edge,
    const double uh_raw, const double vh_raw,
    double *__restrict u, double *__restrict uh, double *__restrict v, double *__restrict vh)
{
  if (h_edge > 0.0)
  {
    double inv_h_edge = 1.0 / h_edge;

    *u = uh_raw * inv_h_edge;
    *uh = h * (*u);

    *v = vh_raw * inv_h_edge;
    *vh = h * inv_h_edge * vh_raw;
  }
  else
  {
    *u = 0.0;
    *uh = 0.0;
    *v = 0.0;
    *vh = 0.0;
  }
}

static inline double compute_local_froude(
    const anuga_int low_froude,
    const double u_left, const double u_right,
    const double v_left, const double v_right,
    const double soundspeed_left, const double soundspeed_right)
{
  double numerator = u_right * u_right + u_left * u_left +
                     v_right * v_right + v_left * v_left;
  double denominator = soundspeed_left * soundspeed_left +
                       soundspeed_right * soundspeed_right + 1.0e-10;

  if (low_froude == 1)
  {
    return sqrt(fmax(0.001, fmin(1.0, numerator / denominator)));
  }
  else if (low_froude == 2)
  {
    double fr = sqrt(numerator / denominator);
    return sqrt(fmin(1.0, 0.01 + fmax(fr - 0.01, 0.0)));
  }
  else
  {
    return 1.0;
  }
}

static inline double compute_s_max(const double u_left, const double u_right,
                                   const double c_left, const double c_right)
{
  double s = fmax(u_left + c_left, u_right + c_right);
  return (s < 0.0) ? 0.0 : s;
}

static inline double compute_s_min(const double u_left, const double u_right,
                                   const double c_left, const double c_right)
{
  double s = fmin(u_left - c_left, u_right - c_right);
  return (s > 0.0) ? 0.0 : s;
}

// Innermost flux function (using stage w=z+h)
anuga_int __flux_function_central(double *__restrict q_left, double *__restrict q_right,
                                const double h_left, const double h_right,
                                const double hle, const double hre,
                                const double n1, const double n2,
                                const double epsilon,
                                const double ze,
                                const double g,
                                double *__restrict edgeflux, double *__restrict max_speed,
                                double *__restrict pressure_flux,
                                const anuga_int low_froude)
{

  /*Compute fluxes between volumes for the shallow water wave equation
    cast in terms of the 'stage', w = h+z using
    the 'central scheme' as described in

    Kurganov, Noelle, Petrova. 'Semidiscrete Central-Upwind Schemes For
    Hyperbolic Conservation Laws and Hamilton-Jacobi Equations'.
    Siam J. Sci. Comput. Vol. 23, No. 3, pp. 707-740.

    The implemented formula is given in equation (3.15) on page 714

    FIXME: Several variables in this interface are no longer used, clean up
  */

  double uh_left, vh_left, u_left;
  double uh_right, vh_right, u_right;
  double soundspeed_left, soundspeed_right;
  double denom;
  double v_right, v_left;
  double q_left_rotated[3], q_right_rotated[3], flux_right[3], flux_left[3];


  for (anuga_int i = 0; i < 3; i++)
  {
    // Rotate the conserved quantities to align with the normal vector
    // This is done to align the x- and y-momentum with the x-axis
    q_left_rotated[i] = q_left[i];
    q_right_rotated[i] = q_right[i];
  }

  // Align x- and y-momentum with x-axis
  __rotate(q_left_rotated, n1, n2);
  __rotate(q_right_rotated, n1, n2);

  // Compute speeds in x-direction
  // w_left = q_left_rotated[0];
  uh_left = q_left_rotated[1];
  vh_left = q_left_rotated[2];
  compute_velocity_terms(h_left, hle, q_left_rotated[1], q_left_rotated[2],
                         &u_left, &uh_left, &v_left, &vh_left);

  uh_right = q_right_rotated[1];
  vh_right = q_right_rotated[2];
  compute_velocity_terms(h_right, hre, q_right_rotated[1], q_right_rotated[2],
                         &u_right, &uh_right, &v_right, &vh_right);

  // Maximal and minimal wave speeds
  soundspeed_left = sqrt(g * h_left);
  soundspeed_right = sqrt(g * h_right);
  // Something that scales like the Froude number
  // We will use this to scale the diffusive component of the UH/VH fluxes.
  double local_fr = compute_local_froude(
      low_froude, u_left, u_right, v_left, v_right,
      soundspeed_left, soundspeed_right);

  double s_max = compute_s_max(u_left, u_right, soundspeed_left, soundspeed_right);
  double s_min = compute_s_min(u_left, u_right, soundspeed_left, soundspeed_right);

  // Flux formulas
  flux_left[0] = u_left * h_left;
  flux_left[1] = u_left * uh_left; //+ 0.5*g*h_left*h_left;
  flux_left[2] = u_left * vh_left;

  flux_right[0] = u_right * h_right;
  flux_right[1] = u_right * uh_right; //+ 0.5*g*h_right*h_right;
  flux_right[2] = u_right * vh_right;

  // Flux computation
  denom = s_max - s_min;
  double inverse_denominator = 1.0 / fmax(denom, 1.0e-100);
  double s_max_s_min = s_max * s_min;
  if (denom < epsilon)
  {
    // Both wave speeds are very small
    memset(edgeflux, 0, 3 * sizeof(double));

    *max_speed = 0.0;
    //*pressure_flux = 0.0;
    *pressure_flux = 0.5 * g * 0.5 * (h_left * h_left + h_right * h_right);
  }
  else
  {
    // Maximal wavespeed
    *max_speed = fmax(s_max, -s_min);
    {
      double flux_0 = s_max * flux_left[0] - s_min * flux_right[0];
      flux_0 += s_max_s_min * (fmax(q_right_rotated[0], ze) - fmax(q_left_rotated[0], ze));
      edgeflux[0] = flux_0 * inverse_denominator;

      double flux_1 = s_max * flux_left[1] - s_min * flux_right[1];
      flux_1 += local_fr * s_max_s_min * (uh_right - uh_left);
      edgeflux[1] = flux_1 * inverse_denominator;

      double flux_2 = s_max * flux_left[2] - s_min * flux_right[2];
      flux_2 += local_fr * s_max_s_min * (vh_right - vh_left);
      edgeflux[2] = flux_2 * inverse_denominator;
    }

    // Separate pressure flux, so we can apply different wet-dry hacks to it
    *pressure_flux = 0.5 * g * (s_max * h_left * h_left - s_min * h_right * h_right) * inverse_denominator;

    // Rotate back
    __rotate(edgeflux, n1, -n2);
  }

  return 0;
}

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

  anuga_int ierr;

  edgeflux[0] = *edgeflux0;
  edgeflux[1] = *edgeflux1;
  edgeflux[2] = *edgeflux2;

  q_left[0] = q_left0;
  q_left[1] = q_left1;
  q_left[2] = q_left2;

  q_right[0] = q_right0;
  q_right[1] = q_right1;
  q_right[2] = q_right2;

  ierr = __flux_function_central(q_left, q_right,
                                 h_left, h_right,
                                 hle, hre,
                                 n1, n2,
                                 epsilon,
                                 ze,
                                 g,
                                 edgeflux, max_speed,
                                 pressure_flux,
                                 low_froude);

  *edgeflux0 = edgeflux[0];
  *edgeflux1 = edgeflux[1];
  *edgeflux2 = edgeflux[2];

  return ierr;
}

double __adjust_edgeflux_with_weir(double *edgeflux,
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
    return 0;
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

    edgeflux[0] = newFlux;

    // FIXME: Do this in a cleaner way
    // IDEA: Compute momentum flux implied by weir relations, and use
    //       those in a weighted average (rather than the rescaling trick here)
    // If we allow the scaling to momentum to be unbounded,
    // velocity spikes can arise for very-shallow-flooded walls
    edgeflux[1] *= fmin(scaleFlux, 10.);
    edgeflux[2] *= fmin(scaleFlux, 10.);
  }

  // Adjust the max speed
  if (fabs(edgeflux[0]) > 0.)
  {
    *max_speed_local = sqrt(g * (maxhd + weir_height)) + fabs(edgeflux[0] / (maxhd + 1.0e-12));
  }
  //*max_speed_local += fabs(edgeflux[0])/(maxhd+1.0e-100);
  //*max_speed_local *= fmax(scaleFlux, 1.0);

  return 0;
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
  anuga_int ierr;

  edgeflux[0] = *edgeflux0;
  edgeflux[1] = *edgeflux1;
  edgeflux[2] = *edgeflux2;

  ierr = __adjust_edgeflux_with_weir(edgeflux, h_left, h_right,
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
                                     const double g, double * __restrict edgeflux, double * __restrict max_speed) {

    anuga_int RiverWall_count = D->edge_river_wall_counter[E->ki];
    anuga_int ii = D->riverwall_rowIndex[RiverWall_count - 1] * ncol_riverwall_hydraulic_properties;

    double Qfactor = D->riverwall_hydraulic_properties[ii];
    double s1 = D->riverwall_hydraulic_properties[ii + 1];
    double s2 = D->riverwall_hydraulic_properties[ii + 2];
    double h1 = D->riverwall_hydraulic_properties[ii + 3];
    double h2 = D->riverwall_hydraulic_properties[ii + 4];

    double weir_height = fmax(D->riverwall_elevation[RiverWall_count - 1] - fmin(E->zl, E->zr), 0.);

    double h_left_tmp = fmax(D->stage_centroid_values[k] - E->z_half, 0.);
    double h_right_tmp = E->is_boundary
                         ? fmax(E->hc_n + E->zr - E->z_half, 0.)
                         : fmax(D->stage_centroid_values[E->n] - E->z_half, 0.);

    if (D->riverwall_elevation[RiverWall_count - 1] > fmax(E->zc, E->zc_n)) {
        __adjust_edgeflux_with_weir(edgeflux, h_left_tmp, h_right_tmp, g,
                                    weir_height, Qfactor, s1, s2, h1, h2, max_speed);
    }
}

double _openmp_compute_fluxes_central(const struct domain *__restrict D,
                                      double timestep)
{
  // Local variables 
  anuga_int number_of_elements = D->number_of_elements;
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

  // Which substep of the timestepping method are we on?
  substep_count = (call - base_call) % D->timestep_fluxcalls;

  double local_timestep = 1.0e+100;
  double boundary_flux_sum_substep = 0.0;
  // double max_speed_local;

      double edgeflux[3];
      double pressure_flux;
      double max_speed_local;
      EdgeData edge_data;
// For all triangles
#pragma omp parallel for simd default(none) schedule(static) shared(D, substep_count, number_of_elements) \
    firstprivate(ncol_riverwall_hydraulic_properties, epsilon, g, low_froude)                              \
    private(edgeflux, pressure_flux, max_speed_local, edge_data) \
    reduction(min : local_timestep) reduction(+ : boundary_flux_sum_substep)
  for (anuga_int k = 0; k < number_of_elements; k++)
  {
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
        // Compute the fluxes using the central scheme
        __flux_function_central(edge_data.ql, edge_data.qr,
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

static inline anuga_int __find_qmin_and_qmax_dq1_dq2(const double dq0, const double dq1, const double dq2,
                                                   double *qmin, double *qmax)
{
  // Considering the centroid of an FV triangle and the vertices of its
  // auxiliary triangle, find
  // qmin=min(q)-qc and qmax=max(q)-qc,
  // where min(q) and max(q) are respectively min and max over the
  // four values (at the centroid of the FV triangle and the auxiliary
  // triangle vertices),
  // and qc is the centroid
  // dq0=q(vertex0)-q(centroid of FV triangle)
  // dq1=q(vertex1)-q(vertex0)
  // dq2=q(vertex2)-q(vertex0)

  // This is a simple implementation
  *qmax = fmax(fmax(dq0, fmax(dq0 + dq1, dq0 + dq2)), 0.0);
  *qmin = fmin(fmin(dq0, fmin(dq0 + dq1, dq0 + dq2)), 0.0);

  return 0;
}

static inline anuga_int __limit_gradient(double *__restrict dqv, double qmin, double qmax, const double beta_w)
{
  // Given provisional jumps dqv from the FV triangle centroid to its
  // vertices/edges, and jumps qmin (qmax) between the centroid of the FV
  // triangle and the minimum (maximum) of the values at the auxiliary triangle
  // vertices (which are centroids of neighbour mesh triangles), calculate a
  // multiplicative factor phi by which the provisional vertex jumps are to be
  // limited

  double r = 1000.0;
  //#pragma omp parallel for simd reduction(min : r) default(none) shared(dqv, qmin, qmax, beta_w, TINY)
  double dq_x = dqv[0];
  double dq_y = dqv[1];
  double dq_z = dqv[2];

  if(dq_x < -TINY)
  {
    double r0 = qmin / dq_x;
    r = fmin(r, r0);
  }
  else if (dq_x > TINY)
  {
    double r0 = qmax / dq_x;
    r = fmin(r, r0);
  }
  if(dq_y < -TINY)
  {
    double r0 = qmin / dq_y;
    r = fmin(r, r0);
  }
  else if (dq_y > TINY)
  {
    double r0 = qmax / dq_y;
    r = fmin(r, r0);
  }
  if(dq_z < -TINY)
  {
    double r0 = qmin / dq_z;
    r = fmin(r, r0);
  }
  else if (dq_z > TINY)
  {
    double r0 = qmax / dq_z;
    r = fmin(r, r0);
  }


  double phi = fmin(r * beta_w, 1.0);

  for (anuga_int i = 0; i < 3; i++)
  {
    dqv[i] *= phi;
  }
  return 0;
}

#pragma omp declare simd
static inline void __calc_edge_values_with_gradient(
    const double cv_k, const double cv_k0, const double cv_k1, const double cv_k2,
    const double dxv0, const double dxv1, const double dxv2, const double dyv0, const double dyv1, const double dyv2,
    const double dx1, const double dx2, const double dy1, const double dy2, const double inv_area2,
    const double beta_tmp, double *__restrict edge_values)
{
  double dqv[3];
  double dq0 = cv_k0 - cv_k;
  double dq1 = cv_k1 - cv_k0;
  double dq2 = cv_k2 - cv_k0;

  double a = (dy2 * dq1 - dy1 * dq2) * inv_area2;
  double b = (dx1 * dq2 - dx2 * dq1) * inv_area2;

  dqv[0] = a * dxv0 + b * dyv0;
  dqv[1] = a * dxv1 + b * dyv1;
  dqv[2] = a * dxv2 + b * dyv2;

  double qmin, qmax;
  __find_qmin_and_qmax_dq1_dq2(dq0, dq1, dq2, &qmin, &qmax);
  __limit_gradient(dqv, qmin, qmax, beta_tmp);

  edge_values[0] = cv_k + dqv[0];
  edge_values[1] = cv_k + dqv[1];
  edge_values[2] = cv_k + dqv[2];
}

#pragma omp declare simd
static inline void __set_constant_edge_values(const double cv_k, double *edge_values)
{
  edge_values[0] = cv_k;
  edge_values[1] = cv_k;
  edge_values[2] = cv_k;
}

#pragma omp declare simd
static inline void compute_qmin_qmax_from_dq1(const double dq1, double *qmin, double *qmax)
{
  if (dq1 >= 0.0)
  {
    *qmin = 0.0;
    *qmax = dq1;
  }
  else
  {
    *qmin = dq1;
    *qmax = 0.0;
  }
}


static inline void update_centroid_values(struct domain *__restrict D,
                                          const anuga_int number_of_elements,
                                          const double minimum_allowed_height,
                                          const anuga_int extrapolate_velocity_second_order)
{
#pragma omp parallel for simd default(none) shared(D) schedule(static) \
    firstprivate(number_of_elements, minimum_allowed_height, extrapolate_velocity_second_order)
  for (anuga_int k = 0; k < number_of_elements; ++k)
  {
    double stage = D->stage_centroid_values[k];
    double bed   = D->bed_centroid_values[k];
    double xmom  = D->xmom_centroid_values[k];
    double ymom  = D->ymom_centroid_values[k];

    double dk_local = fmax(stage - bed, 0.0);
    D->height_centroid_values[k] = dk_local;

    anuga_int is_dry = (dk_local <= minimum_allowed_height);
    anuga_int extrapolate = (extrapolate_velocity_second_order == 1) & (dk_local > minimum_allowed_height);

    // Prepare outputs branchless
    double xmom_out = (is_dry) ? 0.0 : xmom;
    double ymom_out = (is_dry) ? 0.0 : ymom;

    double inv_dk = (extrapolate) ? (1.0 / dk_local) : 1.0;

    D->x_centroid_work[k] = (extrapolate) ? xmom_out : 0.0;
    D->y_centroid_work[k] = (extrapolate) ? ymom_out : 0.0;
    D->xmom_centroid_values[k] = xmom_out * inv_dk;
    D->ymom_centroid_values[k] = ymom_out * inv_dk;
  }
}



#pragma omp declare simd
static inline void set_all_edge_values_from_centroid(struct domain *__restrict D, const anuga_int k)
{

  const double stage = D->stage_centroid_values[k];
  const double xmom = D->xmom_centroid_values[k];
  const double ymom = D->ymom_centroid_values[k];
  const double height = D->height_centroid_values[k];

  for (anuga_int i = 0; i < 3; i++)
  {
    anuga_int ki = 3 * k + i;
    D->stage_edge_values[ki] = stage;
    D->xmom_edge_values[ki] = xmom;
    D->ymom_edge_values[ki] = ymom;
    D->height_edge_values[ki] = height;
    D->bed_edge_values[ki] = D->bed_centroid_values[k];
  }
}

#pragma omp declare simd
static inline anuga_int get_internal_neighbour(const struct domain *__restrict D, const anuga_int k)
{
  for (anuga_int i = 0; i < 3; i++)
  {
    anuga_int n = D->surrogate_neighbours[3 * k + i];
    if (n != k)
    {
      return n;
    }
  }
  return -1; // Indicates failure
}

#pragma omp declare simd
static inline void compute_dqv_from_gradient(const double dq1, const double dx2, const double dy2,
                                             const double dxv0, const double dxv1, const double dxv2,
                                             const double dyv0, const double dyv1, const double dyv2,
                                             double dqv[3])
{
  // Calculate the gradient between the centroid of triangle k
  // and that of its neighbour
  double a = dq1 * dx2;
  double b = dq1 * dy2;

  dqv[0] = a * dxv0 + b * dyv0;
  dqv[1] = a * dxv1 + b * dyv1;
  dqv[2] = a * dxv2 + b * dyv2;
}

#pragma omp declare simd
static inline void compute_gradient_projection_between_centroids(
    const struct domain *__restrict D, const anuga_int k, const anuga_int k1,
    double *__restrict dx2, double *__restrict dy2)
{
  double x = D->centroid_coordinates[2 * k + 0];
  double y = D->centroid_coordinates[2 * k + 1];
  double x1 = D->centroid_coordinates[2 * k1 + 0];
  double y1 = D->centroid_coordinates[2 * k1 + 1];

  double dx = x1 - x;
  double dy = y1 - y;
  double area2 = dx * dx + dy * dy;

  if (area2 > 0.0)
  {
    *dx2 = dx / area2;
    *dy2 = dy / area2;
  }
  else
  {
    *dx2 = 0.0;
    *dy2 = 0.0;
  }
}

#pragma omp declare simd
static inline void extrapolate_gradient_limited(
    const double *__restrict centroid_values, double *__restrict edge_values,
    const anuga_int k, const anuga_int k1, const anuga_int k3,
    const double dx2, const double dy2,
    const double dxv0, const double dxv1, const double dxv2,
    const double dyv0, const double dyv1, const double dyv2,
    const double beta)
{
  double dq1 = centroid_values[k1] - centroid_values[k];

  double dqv[3];
  compute_dqv_from_gradient(dq1, dx2, dy2,
                            dxv0, dxv1, dxv2,
                            dyv0, dyv1, dyv2, dqv);

  double qmin, qmax;
  compute_qmin_qmax_from_dq1(dq1, &qmin, &qmax);

  __limit_gradient(dqv, qmin, qmax, beta);

  for (anuga_int i = 0; i < 3; i++)
  {
    edge_values[k3 + i] = centroid_values[k] + dqv[i];
  }
}

#pragma omp declare simd
static inline void interpolate_edges_with_beta(
    const double *__restrict centroid_values,
    double *__restrict edge_values,
    const anuga_int k, const anuga_int k0, const anuga_int k1, const anuga_int k2, const anuga_int k3,
    const double dxv0, const double dxv1, const double dxv2,
    const double dyv0, const double dyv1, const double dyv2,
    const double dx1, const double dx2, const double dy1, const double dy2,
    const double inv_area2,
    const double beta_dry, const double beta_wet, const double hfactor)
{
  double beta = beta_dry + (beta_wet - beta_dry) * hfactor;

  double edge_vals[3];
  if (beta > 0.0)
  {
    __calc_edge_values_with_gradient(
        centroid_values[k],
        centroid_values[k0],
        centroid_values[k1],
        centroid_values[k2],
        dxv0, dxv1, dxv2,
        dyv0, dyv1, dyv2,
        dx1, dx2, dy1, dy2,
        inv_area2,
        beta,
        edge_vals);
  }
  else
  {
    __set_constant_edge_values(centroid_values[k], edge_vals);
  }
  for (anuga_int i = 0; i < 3; i++)
  {
    edge_values[k3 + i] = edge_vals[i];
  }
}

#pragma omp declare simd
static inline void compute_hfactor_and_inv_area(
    const struct domain *__restrict D,
    const anuga_int k, const anuga_int k0, const anuga_int k1, const anuga_int k2,
    const double area2, const double c_tmp, const double d_tmp,
    double *__restrict hfactor, double *__restrict inv_area2)
{
  double hc = D->height_centroid_values[k];
  double h0 = D->height_centroid_values[k0];
  double h1 = D->height_centroid_values[k1];
  double h2 = D->height_centroid_values[k2];

  double hmin = fmin(fmin(h0, fmin(h1, h2)), hc);
  double hmax = fmax(fmax(h0, fmax(h1, h2)), hc);

  double tmp1 = c_tmp * fmax(hmin, 0.0) / fmax(hc, 1.0e-06) + d_tmp;
  double tmp2 = c_tmp * fmax(hc, 0.0) / fmax(hmax, 1.0e-06) + d_tmp;

  *hfactor = fmax(0.0, fmin(tmp1, fmin(tmp2, 1.0)));

  // Smooth shutoff near dry areas
  *hfactor = fmin(1.2 * fmax(hmin - D->minimum_allowed_height, 0.0) /
                      (fmax(hmin, 0.0) + D->minimum_allowed_height),
                  *hfactor);

  *inv_area2 = 1.0 / area2;
}

#pragma omp declare simd
static inline void reconstruct_vertex_values(double *__restrict edge_values, double *__restrict vertex_values, const anuga_int k3)
{
  vertex_values[k3 + 0] = edge_values[k3 + 1] + edge_values[k3 + 2] - edge_values[k3 + 0];
  vertex_values[k3 + 1] = edge_values[k3 + 2] + edge_values[k3 + 0] - edge_values[k3 + 1];
  vertex_values[k3 + 2] = edge_values[k3 + 0] + edge_values[k3 + 1] - edge_values[k3 + 2];
}

#pragma omp declare simd
static inline void compute_edge_diffs(const double x, const double y,
                                      const double xv0, const double yv0,
                                      const double xv1, const double yv1,
                                      const double xv2, const double yv2,
                                      double *__restrict dxv0, double *__restrict dxv1, double *__restrict dxv2,
                                      double *__restrict dyv0, double *__restrict dyv1, double *__restrict dyv2)
{
  *dxv0 = xv0 - x;
  *dxv1 = xv1 - x;
  *dxv2 = xv2 - x;
  *dyv0 = yv0 - y;
  *dyv1 = yv1 - y;
  *dyv2 = yv2 - y;
}

// Computational routine
// Extrapolate second order edge values from centroid values
// This is the current procedure used in evolve loop.
void _openmp_extrapolate_second_order_edge_sw(struct domain *__restrict D)
{
  double minimum_allowed_height = D->minimum_allowed_height;
  anuga_int number_of_elements = D->number_of_elements;
  anuga_int extrapolate_velocity_second_order = D->extrapolate_velocity_second_order;

  // Parameters used to control how the limiter is forced to first-order near
  // wet-dry regions
  double a_tmp = 0.3; // Highest depth ratio with hfactor=1
  double b_tmp = 0.1; // Highest depth ratio with hfactor=0
  double c_tmp = 1.0 / (a_tmp - b_tmp);
  double d_tmp = 1.0 - (c_tmp * a_tmp);

  update_centroid_values(D, number_of_elements, minimum_allowed_height, extrapolate_velocity_second_order);

#pragma omp parallel for simd default(none) schedule(static) \
    shared(D)                                                 \
    firstprivate(number_of_elements, minimum_allowed_height, extrapolate_velocity_second_order, c_tmp, d_tmp)
  for (anuga_int k = 0; k < number_of_elements; k++)
  {
    // // Useful indices
    anuga_int k2 = k * 2;
    anuga_int k3 = k * 3;
    anuga_int k6 = k * 6;

    // Get the edge coordinates
    const double xv0 = D->edge_coordinates[k6 + 0];
    const double yv0 = D->edge_coordinates[k6 + 1];
    const double xv1 = D->edge_coordinates[k6 + 2];
    const double yv1 = D->edge_coordinates[k6 + 3];
    const double xv2 = D->edge_coordinates[k6 + 4];
    const double yv2 = D->edge_coordinates[k6 + 5];

    // Get the centroid coordinates
    const double x = D->centroid_coordinates[k2 + 0];
    const double y = D->centroid_coordinates[k2 + 1];

    // needed in the boundaries section
    double dxv0, dxv1, dxv2;
    double dyv0, dyv1, dyv2;
    compute_edge_diffs(x, y,
                       xv0, yv0,
                       xv1, yv1,
                       xv2, yv2,
                       &dxv0, &dxv1, &dxv2,
                       &dyv0, &dyv1, &dyv2);
    // dxv0 = dxv0;
    // dxv1 = dxv1;
    // dxv2 = dxv2;
    // dyv0 = dyv0;
    // dyv1 = dyv1;
    // dyv2 = dyv2;

    anuga_int k0 = D->surrogate_neighbours[k3 + 0];
    anuga_int k1 = D->surrogate_neighbours[k3 + 1];
    k2 = D->surrogate_neighbours[k3 + 2];

    anuga_int coord_index = 2 * k0;
    double x0 = D->centroid_coordinates[coord_index + 0];
    double y0 = D->centroid_coordinates[coord_index + 1];

    coord_index = 2 * k1;
    double x1 = D->centroid_coordinates[coord_index + 0];
    double y1 = D->centroid_coordinates[coord_index + 1];

    coord_index = 2 * k2;
    double x2 = D->centroid_coordinates[coord_index + 0];
    double y2 = D->centroid_coordinates[coord_index + 1];

    // needed in the boundaries section
    double dx1 = x1 - x0;
    double dx2 = x2 - x0;
    double dy1 = y1 - y0;
    double dy2 = y2 - y0;
    // dx1 = dx1;
    // dx2 = dx2;
    // dy1 = dy1;
    // dy2 = dy2;
    // needed in the boundaries section
    double area2 = dy2 * dx1 - dy1 * dx2;
    // area2 = area2;
    // the calculation of dx0 dx1 dx2 dy0 dy1 dy2 etc could be calculated once and stored 
    // in the domain structure.


    const anuga_int dry =
        ((D->height_centroid_values[k0] < minimum_allowed_height) | (k0 == k)) &
        ((D->height_centroid_values[k1] < minimum_allowed_height) | (k1 == k)) &
        ((D->height_centroid_values[k2] < minimum_allowed_height) | (k2 == k));

    if (dry)
    {
      D->x_centroid_work[k] = 0.0;
      D->xmom_centroid_values[k] = 0.0;
      D->y_centroid_work[k] = 0.0;
      D->ymom_centroid_values[k] = 0.0;
    }

    // int k0 = D->surrogate_neighbours[k3 + 0];
    // int k1 = D->surrogate_neighbours[k3 + 1];
    // k2 = D->surrogate_neighbours[k3 + 2];

    if (D->number_of_boundaries[k] == 3)
    {
      // Very unlikely
      // No neighbourso, set gradient on the triangle to zero
      set_all_edge_values_from_centroid(D, k);
    }
    else if (D->number_of_boundaries[k] <= 1)
    {
      //==============================================
      // Number of boundaries <= 1
      // 'Typical case'
      //==============================================
      double hfactor, inv_area2;
      compute_hfactor_and_inv_area(D, k, k0, k1, k2, area2, c_tmp, d_tmp, &hfactor, &inv_area2);
      // stage
      interpolate_edges_with_beta(D->stage_centroid_values, D->stage_edge_values,
                                  k, k0, k1, k2, k3,
                                  dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                                  dx1, dx2, dy1, dy2, inv_area2,
                                  D->beta_w_dry, D->beta_w, hfactor);
      // height
      interpolate_edges_with_beta(D->height_centroid_values, D->height_edge_values,
                                  k, k0, k1, k2, k3,
                                  dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                                  dx1, dx2, dy1, dy2, inv_area2,
                                  D->beta_w_dry, D->beta_w, hfactor);
      // xmom
      interpolate_edges_with_beta(D->xmom_centroid_values, D->xmom_edge_values,
                                  k, k0, k1, k2, k3,
                                  dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                                  dx1, dx2, dy1, dy2, inv_area2,
                                  D->beta_uh_dry, D->beta_uh, hfactor);
      // ymom
      interpolate_edges_with_beta(D->ymom_centroid_values, D->ymom_edge_values,
                                  k, k0, k1, k2, k3,
                                  dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                                  dx1, dx2, dy1, dy2, inv_area2,
                                  D->beta_vh_dry, D->beta_vh, hfactor);

    } // End number_of_boundaries <=1
    else
    {
      //==============================================
      //  Number of boundaries == 2
      //==============================================
      // One internal neighbour and gradient is in direction of the neighbour's centroid
      // Find the only internal neighbour (k1?)
      k1 = get_internal_neighbour(D, k);
      compute_gradient_projection_between_centroids(D, k, k1, &dx2, &dy2);
      // stage
      extrapolate_gradient_limited(D->stage_centroid_values, D->stage_edge_values,
                                   k, k1, k3, dx2, dy2,
                                   dxv0, dxv1, dxv2,
                                   dyv0, dyv1, dyv2, D->beta_w);
      // height
      extrapolate_gradient_limited(D->height_centroid_values, D->height_edge_values,
                                   k, k1, k3, dx2, dy2,
                                   dxv0, dxv1, dxv2,
                                   dyv0, dyv1, dyv2, D->beta_w);
      // xmom
      extrapolate_gradient_limited(D->xmom_centroid_values, D->xmom_edge_values,
                                   k, k1, k3, dx2, dy2,
                                   dxv0, dxv1, dxv2,
                                   dyv0, dyv1, dyv2, D->beta_w);
      // ymom
      extrapolate_gradient_limited(D->ymom_centroid_values, D->ymom_edge_values,
                                   k, k1, k3, dx2, dy2,
                                   dxv0, dxv1, dxv2,
                                   dyv0, dyv1, dyv2, D->beta_w);

    } // else [number_of_boundaries]

    // If needed, convert from velocity to momenta
    if (D->extrapolate_velocity_second_order == 1)
    {
      // Re-compute momenta at edges
      for (anuga_int i = 0; i < 3; i++)
      {
        double dk = D->height_edge_values[k3 + i];
        D->xmom_edge_values[k3 + i] = D->xmom_edge_values[k3 + i] * dk;
        D->ymom_edge_values[k3 + i] = D->ymom_edge_values[k3 + i] * dk;
      }
    }

    for (anuga_int i = 0; i < 3; i++)
    {
      D->bed_edge_values[k3 + i] = D->stage_edge_values[k3 + i] - D->height_edge_values[k3 + i];
    }

    // This should not be needed, as now the evolve loop should just depend
    // on the edge values, which are reconstructed from the centroid values
    // reconstruct_vertex_values(D->stage_edge_values, D->stage_vertex_values, k3);
    // reconstruct_vertex_values(D->height_edge_values, D->height_vertex_values, k3);
    // reconstruct_vertex_values(D->xmom_edge_values, D->xmom_vertex_values, k3);
    // reconstruct_vertex_values(D->ymom_edge_values, D->ymom_vertex_values, k3);
    // reconstruct_vertex_values(D->bed_edge_values, D->bed_vertex_values, k3);
  }
  // for k=0 to number_of_elements-1
// Fix xmom and ymom centroid values
if(extrapolate_velocity_second_order == 1)
{
#pragma omp parallel for simd schedule(static) firstprivate(extrapolate_velocity_second_order)
  for (anuga_int k = 0; k < D->number_of_elements; k++)
  {
      // Convert velocity back to momenta at centroids
      D->xmom_centroid_values[k] = D->x_centroid_work[k];
      D->ymom_centroid_values[k] = D->y_centroid_work[k];
  }
}

}

void _openmp_distribute_edges_to_vertices(struct domain *__restrict D)
{
  // Distribute edge values to vertices
  anuga_int number_of_elements = D->number_of_elements;

#pragma omp parallel for simd default(none) shared(D) schedule(static) firstprivate(number_of_elements)
  for (anuga_int k = 0; k < number_of_elements; k++)
  {
    anuga_int k3 = 3 * k;

    // Set vertex values from edge values
    reconstruct_vertex_values(D->stage_edge_values, D->stage_vertex_values, k3);
    reconstruct_vertex_values(D->height_edge_values, D->height_vertex_values, k3);
    reconstruct_vertex_values(D->xmom_edge_values, D->xmom_vertex_values, k3);
    reconstruct_vertex_values(D->ymom_edge_values, D->ymom_vertex_values, k3);
    reconstruct_vertex_values(D->bed_edge_values, D->bed_vertex_values, k3);
  
  }
}

void _openmp_manning_friction_flat_semi_implicit(const struct domain *__restrict D)
{

  anuga_int k;

  const anuga_int N = D->number_of_elements;
  const double eps = D->minimum_allowed_height;
  const double g = D->g;
  const double seven_thirds = 7.0 / 3.0;

 
#pragma omp parallel for simd default(none) shared(D) schedule(static) \
        firstprivate(N, eps, g, seven_thirds)

  for (k = 0; k < N; k++)
  {
    double S = 0.0;
    double h;
    double uh = D->xmom_centroid_values[k];
    double vh = D->ymom_centroid_values[k];
    double eta = D->friction_centroid_values[k];
    double abs_mom = sqrt( uh*uh + vh*vh );

    if (eta > 1.0e-15)
    {
      h = D->stage_centroid_values[k] - D->bed_centroid_values[k];
      if (h >= eps)
       {
        S = -g * eta * eta * abs_mom;
        S /= pow(h, seven_thirds); 
       }
      }
    D->xmom_semi_implicit_update[k] += S * D->xmom_centroid_values[k];
    D->ymom_semi_implicit_update[k] += S * D->ymom_centroid_values[k];
  }
}




    

void _openmp_manning_friction_sloped_semi_implicit(const struct domain *__restrict D)
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
    z0 = D->bed_vertex_values[k3 + 0];
    z1 = D->bed_vertex_values[k3 + 1];
    z2 = D->bed_vertex_values[k3 + 2];

    // Compute bed slope
    k6 = 6 * k; // base index

    
    x0 = D->vertex_coordinates[k6 + 0];
    y0 = D->vertex_coordinates[k6 + 1];
    x1 = D->vertex_coordinates[k6 + 2];
    y1 = D->vertex_coordinates[k6 + 3];
    x2 = D->vertex_coordinates[k6 + 4];
    y2 = D->vertex_coordinates[k6 + 5];

    
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


// Old function for extrapolating second order edge values from centroid values
// This function is now replaced by _openmp_extrapolate_second_order_edge_sw
// which uses SIMD and OpenMP for parallelization
// This function is kept for reference and compatibility
void _openmp_extrapolate_second_order_sw(const struct domain *__restrict D) {


  // Domain Variables
    anuga_int number_of_elements;
    double epsilon;
    double minimum_allowed_height;
    double beta_w;
    double beta_w_dry;
    double beta_uh;
    double beta_uh_dry;
    double beta_vh;
    double beta_vh_dry;
    anuga_int* surrogate_neighbours;
    anuga_int* number_of_boundaries;
    double* centroid_coordinates;
    double* stage_centroid_values;
    double* xmom_centroid_values;
    double* ymom_centroid_values;
    double* bed_centroid_values;
    double* vertex_coordinates;
    double* stage_vertex_values;
    double* xmom_vertex_values;
    double* ymom_vertex_values;
    double* bed_vertex_values;
    anuga_int optimise_dry_cells;
    anuga_int extrapolate_velocity_second_order;

    // Local variables
    double a, b; // Gradient vector used to calculate edge values from centroids
    anuga_int k, k0, k1, k2, k3, k6, coord_index, i;
    double x, y, x0, y0, x1, y1, x2, y2, xv0, yv0, xv1, yv1, xv2, yv2; // Vertices of the auxiliary triangle
    double dx1, dx2, dy1, dy2, dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dq0, dq1, dq2, area2, inv_area2;
    double dqv[3], qmin, qmax, hmin, hmax;
    double hc, h0, h1, h2, beta_tmp, hfactor;
    //double dk, dv0, dv1, dv2, de[3], demin, dcmax, r0scale;
    double dk, dv0, dv1, dv2;

    double *xmom_centroid_store;
    double *ymom_centroid_store;
    //double *stage_centroid_store;


    // Associate memory location of Domain varibles with local aliases
    number_of_elements     = D->number_of_elements;
    epsilon                = D->epsilon;
    minimum_allowed_height = D->minimum_allowed_height;
    beta_w                 = D->beta_w;
    beta_w_dry             = D->beta_w_dry;
    beta_uh                = D->beta_uh;
    beta_uh_dry            = D->beta_uh_dry;
    beta_vh                = D->beta_vh;
    beta_vh_dry            = D->beta_vh_dry;
    optimise_dry_cells     = D->optimise_dry_cells;

    extrapolate_velocity_second_order = D->extrapolate_velocity_second_order;

    surrogate_neighbours      = D->surrogate_neighbours;
    number_of_boundaries      = D->number_of_boundaries;
    centroid_coordinates      = D->centroid_coordinates;
    stage_centroid_values     = D->stage_centroid_values;
    xmom_centroid_values      = D->xmom_centroid_values;
    ymom_centroid_values      = D->ymom_centroid_values;
    bed_centroid_values       = D->bed_centroid_values;
    vertex_coordinates        = D->vertex_coordinates;
    stage_vertex_values       = D->stage_vertex_values;
    xmom_vertex_values        = D->xmom_vertex_values;
    ymom_vertex_values        = D->ymom_vertex_values;
    bed_vertex_values         = D->bed_vertex_values;




/*
anuga_int _extrapolate_second_order_sw(anuga_int number_of_elements,
        double epsilon,
        double minimum_allowed_height,
        double beta_w,
        double beta_w_dry,
        double beta_uh,
        double beta_uh_dry,
        double beta_vh,
        double beta_vh_dry,
        anuga_int* surrogate_neighbours,
        anuga_int* number_of_boundaries,
        double* centroid_coordinates,
        double* stage_centroid_values,
        double* xmom_centroid_values,
        double* ymom_centroid_values,
        double* elevation_centroid_values,
        double* vertex_coordinates,
        double* stage_vertex_values,
        double* xmom_vertex_values,
        double* ymom_vertex_values,
        double* elevation_vertex_values,
        anuga_int optimise_dry_cells,
        anuga_int extrapolate_velocity_second_order) {



    // Local variables
    double a, b; // Gradient vector used to calculate vertex values from centroids
    anuga_int k, k0, k1, k2, k3, k6, coord_index, i;
    double x, y, x0, y0, x1, y1, x2, y2, xv0, yv0, xv1, yv1, xv2, yv2; // Vertices of the auxiliary triangle
    double dx1, dx2, dy1, dy2, dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dq0, dq1, dq2, area2, inv_area2;
    double dqv[3], qmin, qmax, hmin, hmax;
    double hc, h0, h1, h2, beta_tmp, hfactor;
    double xmom_centroid_store[number_of_elements], ymom_centroid_store[number_of_elements], dk, dv0, dv1, dv2;
*/

   // Use malloc to avoid putting these variables on the stack, which can cause
   // segfaults in large model runs
    xmom_centroid_store = malloc(number_of_elements*sizeof(double));
    ymom_centroid_store = malloc(number_of_elements*sizeof(double));
    // stage_centroid_store = malloc(number_of_elements*sizeof(double));

    if (extrapolate_velocity_second_order == 1) {
        // Replace momentum centroid with velocity centroid to allow velocity
        // extrapolation This will be changed back at the end of the routine
        for (k = 0; k < number_of_elements; k++) {

            dk = fmax(stage_centroid_values[k] - bed_centroid_values[k], minimum_allowed_height);
            xmom_centroid_store[k] = xmom_centroid_values[k];
            xmom_centroid_values[k] = xmom_centroid_values[k] / dk;

            ymom_centroid_store[k] = ymom_centroid_values[k];
            ymom_centroid_values[k] = ymom_centroid_values[k] / dk;
        }
    }

    // Begin extrapolation routine
    for (k = 0; k < number_of_elements; k++) {
        k3 = k * 3;
        k6 = k * 6;

        if (number_of_boundaries[k] == 3) {
            // No neighbours, set gradient on the triangle to zero

            stage_vertex_values[k3] = stage_centroid_values[k];
            stage_vertex_values[k3 + 1] = stage_centroid_values[k];
            stage_vertex_values[k3 + 2] = stage_centroid_values[k];
            xmom_vertex_values[k3] = xmom_centroid_values[k];
            xmom_vertex_values[k3 + 1] = xmom_centroid_values[k];
            xmom_vertex_values[k3 + 2] = xmom_centroid_values[k];
            ymom_vertex_values[k3] = ymom_centroid_values[k];
            ymom_vertex_values[k3 + 1] = ymom_centroid_values[k];
            ymom_vertex_values[k3 + 2] = ymom_centroid_values[k];

            continue;
        } else {
            // Triangle k has one or more neighbours.
            // Get centroid and vertex coordinates of the triangle

            // Get the vertex coordinates
            xv0 = vertex_coordinates[k6];
            yv0 = vertex_coordinates[k6 + 1];
            xv1 = vertex_coordinates[k6 + 2];
            yv1 = vertex_coordinates[k6 + 3];
            xv2 = vertex_coordinates[k6 + 4];
            yv2 = vertex_coordinates[k6 + 5];

            // Get the centroid coordinates
            coord_index = 2 * k;
            x = centroid_coordinates[coord_index];
            y = centroid_coordinates[coord_index + 1];

            // Store x- and y- differentials for the vertices of
            // triangle k relative to the centroid
            dxv0 = xv0 - x;
            dxv1 = xv1 - x;
            dxv2 = xv2 - x;
            dyv0 = yv0 - y;
            dyv1 = yv1 - y;
            dyv2 = yv2 - y;
        }




        if (number_of_boundaries[k] <= 1) {
            //==============================================
            // Number of boundaries <= 1
            //==============================================


            // If no boundaries, auxiliary triangle is formed
            // from the centroids of the three neighbours
            // If one boundary, auxiliary triangle is formed
            // from this centroid and its two neighbours

            k0 = surrogate_neighbours[k3];
            k1 = surrogate_neighbours[k3 + 1];
            k2 = surrogate_neighbours[k3 + 2];

            // Get the auxiliary triangle's vertex coordinates
            // (really the centroids of neighbouring triangles)
            coord_index = 2 * k0;
            x0 = centroid_coordinates[coord_index];
            y0 = centroid_coordinates[coord_index + 1];

            coord_index = 2 * k1;
            x1 = centroid_coordinates[coord_index];
            y1 = centroid_coordinates[coord_index + 1];

            coord_index = 2 * k2;
            x2 = centroid_coordinates[coord_index];
            y2 = centroid_coordinates[coord_index + 1];

            // Store x- and y- differentials for the vertices
            // of the auxiliary triangle
            dx1 = x1 - x0;
            dx2 = x2 - x0;
            dy1 = y1 - y0;
            dy2 = y2 - y0;

            // Calculate 2*area of the auxiliary triangle
            // The triangle is guaranteed to be counter-clockwise
            area2 = dy2 * dx1 - dy1*dx2;

            // If the mesh is 'weird' near the boundary,
            // the triangle might be flat or clockwise
            // Default to zero gradient
            if (area2 <= 0) {
                //printf("Error negative triangle area \n");
                //return -1;

                stage_vertex_values[k3] = stage_centroid_values[k];
                stage_vertex_values[k3 + 1] = stage_centroid_values[k];
                stage_vertex_values[k3 + 2] = stage_centroid_values[k];
                xmom_vertex_values[k3] = xmom_centroid_values[k];
                xmom_vertex_values[k3 + 1] = xmom_centroid_values[k];
                xmom_vertex_values[k3 + 2] = xmom_centroid_values[k];
                ymom_vertex_values[k3] = ymom_centroid_values[k];
                ymom_vertex_values[k3 + 1] = ymom_centroid_values[k];
                ymom_vertex_values[k3 + 2] = ymom_centroid_values[k];

                continue;
            }

            // Calculate heights of neighbouring cells
            hc = stage_centroid_values[k] - bed_centroid_values[k];
            h0 = stage_centroid_values[k0] - bed_centroid_values[k0];
            h1 = stage_centroid_values[k1] - bed_centroid_values[k1];
            h2 = stage_centroid_values[k2] - bed_centroid_values[k2];
            hmin = fmax(fmax(h0, fmax(h1, h2)), hc);
            //hfactor = hc/(hc + 1.0);

            hfactor = 0.0;
            if (hmin > 0.001) {
                hfactor = (hmin - 0.001) / (hmin + 0.004);
            }

            if (optimise_dry_cells) {
                // Check if linear reconstruction is necessary for triangle k
                // This check will exclude dry cells.

                hmax = fmax(h0, fmax(h1, h2));
                if (hmax < epsilon) {
                    continue;
                }
            }

            //-----------------------------------
            // stage
            //-----------------------------------

            // Calculate the difference between vertex 0 of the auxiliary
            // triangle and the centroid of triangle k
            dq0 = stage_centroid_values[k0] - stage_centroid_values[k];

            // Calculate differentials between the vertices
            // of the auxiliary triangle (centroids of neighbouring triangles)
            dq1 = stage_centroid_values[k1] - stage_centroid_values[k0];
            dq2 = stage_centroid_values[k2] - stage_centroid_values[k0];

            inv_area2 = 1.0 / area2;
            // Calculate the gradient of stage on the auxiliary triangle
            a = dy2 * dq1 - dy1*dq2;
            a *= inv_area2;
            b = dx1 * dq2 - dx2*dq1;
            b *= inv_area2;

            // Calculate provisional jumps in stage from the centroid
            // of triangle k to its vertices, to be limited
            dqv[0] = a * dxv0 + b*dyv0;
            dqv[1] = a * dxv1 + b*dyv1;
            dqv[2] = a * dxv2 + b*dyv2;

            // Now we want to find min and max of the centroid and the
            // vertices of the auxiliary triangle and compute jumps
            // from the centroid to the min and max
            __find_qmin_and_qmax_dq1_dq2(dq0, dq1, dq2, &qmin, &qmax);

            // Playing with dry wet interface
            //hmin = qmin;
            //beta_tmp = beta_w_dry;
            //if (hmin>minimum_allowed_height)
            beta_tmp = beta_w_dry + (beta_w - beta_w_dry) * hfactor;

            //printf("min_alled_height = %f\n",minimum_allowed_height);
            //printf("hmin = %f\n",hmin);
            //printf("beta_w = %f\n",beta_w);
            //printf("beta_tmp = %f\n",beta_tmp);
            // Limit the gradient
            __limit_gradient(dqv, qmin, qmax, beta_tmp);

            //for (i=0;i<3;i++)
            stage_vertex_values[k3 + 0] = stage_centroid_values[k] + dqv[0];
            stage_vertex_values[k3 + 1] = stage_centroid_values[k] + dqv[1];
            stage_vertex_values[k3 + 2] = stage_centroid_values[k] + dqv[2];


            //-----------------------------------
            // xmomentum
            //-----------------------------------

            // Calculate the difference between vertex 0 of the auxiliary
            // triangle and the centroid of triangle k
            dq0 = xmom_centroid_values[k0] - xmom_centroid_values[k];

            // Calculate differentials between the vertices
            // of the auxiliary triangle
            dq1 = xmom_centroid_values[k1] - xmom_centroid_values[k0];
            dq2 = xmom_centroid_values[k2] - xmom_centroid_values[k0];

            // Calculate the gradient of xmom on the auxiliary triangle
            a = dy2 * dq1 - dy1*dq2;
            a *= inv_area2;
            b = dx1 * dq2 - dx2*dq1;
            b *= inv_area2;

            // Calculate provisional jumps in stage from the centroid
            // of triangle k to its vertices, to be limited
            dqv[0] = a * dxv0 + b*dyv0;
            dqv[1] = a * dxv1 + b*dyv1;
            dqv[2] = a * dxv2 + b*dyv2;

            // Now we want to find min and max of the centroid and the
            // vertices of the auxiliary triangle and compute jumps
            // from the centroid to the min and max
            __find_qmin_and_qmax_dq1_dq2(dq0, dq1, dq2, &qmin, &qmax);
            //beta_tmp = beta_uh;
            //if (hmin<minimum_allowed_height)
            //beta_tmp = beta_uh_dry;
            beta_tmp = beta_uh_dry + (beta_uh - beta_uh_dry) * hfactor;

            // Limit the gradient
            __limit_gradient(dqv, qmin, qmax, beta_tmp);

            for (i = 0; i < 3; i++) {
                xmom_vertex_values[k3 + i] = xmom_centroid_values[k] + dqv[i];
            }

            //-----------------------------------
            // ymomentum
            //-----------------------------------

            // Calculate the difference between vertex 0 of the auxiliary
            // triangle and the centroid of triangle k
            dq0 = ymom_centroid_values[k0] - ymom_centroid_values[k];

            // Calculate differentials between the vertices
            // of the auxiliary triangle
            dq1 = ymom_centroid_values[k1] - ymom_centroid_values[k0];
            dq2 = ymom_centroid_values[k2] - ymom_centroid_values[k0];

            // Calculate the gradient of xmom on the auxiliary triangle
            a = dy2 * dq1 - dy1*dq2;
            a *= inv_area2;
            b = dx1 * dq2 - dx2*dq1;
            b *= inv_area2;

            // Calculate provisional jumps in stage from the centroid
            // of triangle k to its vertices, to be limited
            dqv[0] = a * dxv0 + b*dyv0;
            dqv[1] = a * dxv1 + b*dyv1;
            dqv[2] = a * dxv2 + b*dyv2;

            // Now we want to find min and max of the centroid and the
            // vertices of the auxiliary triangle and compute jumps
            // from the centroid to the min and max
            __find_qmin_and_qmax_dq1_dq2(dq0, dq1, dq2, &qmin, &qmax);

            //beta_tmp = beta_vh;
            //
            //if (hmin<minimum_allowed_height)
            //beta_tmp = beta_vh_dry;
            beta_tmp = beta_vh_dry + (beta_vh - beta_vh_dry) * hfactor;

            // Limit the gradient
            __limit_gradient(dqv, qmin, qmax, beta_tmp);

            for (i = 0; i < 3; i++) {
                ymom_vertex_values[k3 + i] = ymom_centroid_values[k] + dqv[i];
            }
        }// End number_of_boundaries <=1
        else {

            //==============================================
            // Number of boundaries == 2
            //==============================================

            // One internal neighbour and gradient is in direction of the neighbour's centroid

            // Find the only internal neighbour (k1?)
            for (k2 = k3; k2 < k3 + 3; k2++) {
                // Find internal neighbour of triangle k
                // k2 indexes the edges of triangle k

                if (surrogate_neighbours[k2] != k) {
                    break;
                }
            }

            // if (k2 == k3 + 3) {
            //     // If we didn't find an internal neighbour
            //     return -1;
            // }

            k1 = surrogate_neighbours[k2];

            // The coordinates of the triangle are already (x,y).
            // Get centroid of the neighbour (x1,y1)
            coord_index = 2 * k1;
            x1 = centroid_coordinates[coord_index];
            y1 = centroid_coordinates[coord_index + 1];

            // Compute x- and y- distances between the centroid of
            // triangle k and that of its neighbour
            dx1 = x1 - x;
            dy1 = y1 - y;

            // Set area2 as the square of the distance
            area2 = dx1 * dx1 + dy1*dy1;

            // Set dx2=(x1-x0)/((x1-x0)^2+(y1-y0)^2)
            // and dy2=(y1-y0)/((x1-x0)^2+(y1-y0)^2) which
            // respectively correspond to the x- and y- gradients
            // of the conserved quantities
            dx2 = 1.0 / area2;
            dy2 = dx2*dy1;
            dx2 *= dx1;


            //-----------------------------------
            // stage
            //-----------------------------------

            // Compute differentials
            dq1 = stage_centroid_values[k1] - stage_centroid_values[k];

            // Calculate the gradient between the centroid of triangle k
            // and that of its neighbour
            a = dq1*dx2;
            b = dq1*dy2;

            // Calculate provisional vertex jumps, to be limited
            dqv[0] = a * dxv0 + b*dyv0;
            dqv[1] = a * dxv1 + b*dyv1;
            dqv[2] = a * dxv2 + b*dyv2;

            // Now limit the jumps
            if (dq1 >= 0.0) {
                qmin = 0.0;
                qmax = dq1;
            } else {
                qmin = dq1;
                qmax = 0.0;
            }

            // Limit the gradient
            __limit_gradient(dqv, qmin, qmax, beta_w);

            //for (i=0; i < 3; i++)
            //{
            stage_vertex_values[k3] = stage_centroid_values[k] + dqv[0];
            stage_vertex_values[k3 + 1] = stage_centroid_values[k] + dqv[1];
            stage_vertex_values[k3 + 2] = stage_centroid_values[k] + dqv[2];
            //}

            //-----------------------------------
            // xmomentum
            //-----------------------------------

            // Compute differentials
            dq1 = xmom_centroid_values[k1] - xmom_centroid_values[k];

            // Calculate the gradient between the centroid of triangle k
            // and that of its neighbour
            a = dq1*dx2;
            b = dq1*dy2;

            // Calculate provisional vertex jumps, to be limited
            dqv[0] = a * dxv0 + b*dyv0;
            dqv[1] = a * dxv1 + b*dyv1;
            dqv[2] = a * dxv2 + b*dyv2;

            // Now limit the jumps
            if (dq1 >= 0.0) {
                qmin = 0.0;
                qmax = dq1;
            } else {
                qmin = dq1;
                qmax = 0.0;
            }

            // Limit the gradient
            __limit_gradient(dqv, qmin, qmax, beta_w);

            //for (i=0;i<3;i++)
            //xmom_vertex_values[k3] = xmom_centroid_values[k] + dqv[0];
            //xmom_vertex_values[k3 + 1] = xmom_centroid_values[k] + dqv[1];
            //xmom_vertex_values[k3 + 2] = xmom_centroid_values[k] + dqv[2];

            for (i = 0; i < 3; i++) {
                xmom_vertex_values[k3 + i] = xmom_centroid_values[k] + dqv[i];
            }

            //-----------------------------------
            // ymomentum
            //-----------------------------------

            // Compute differentials
            dq1 = ymom_centroid_values[k1] - ymom_centroid_values[k];

            // Calculate the gradient between the centroid of triangle k
            // and that of its neighbour
            a = dq1*dx2;
            b = dq1*dy2;

            // Calculate provisional vertex jumps, to be limited
            dqv[0] = a * dxv0 + b*dyv0;
            dqv[1] = a * dxv1 + b*dyv1;
            dqv[2] = a * dxv2 + b*dyv2;

            // Now limit the jumps
            if (dq1 >= 0.0) {
                qmin = 0.0;
                qmax = dq1;
            }
            else {
                qmin = dq1;
                qmax = 0.0;
            }

            // Limit the gradient
            __limit_gradient(dqv, qmin, qmax, beta_w);

            //for (i=0;i<3;i++)
            //ymom_vertex_values[k3] = ymom_centroid_values[k] + dqv[0];
            //ymom_vertex_values[k3 + 1] = ymom_centroid_values[k] + dqv[1];
            //ymom_vertex_values[k3 + 2] = ymom_centroid_values[k] + dqv[2];

            for (i = 0; i < 3; i++) {
                ymom_vertex_values[k3 + i] = ymom_centroid_values[k] + dqv[i];
            }
            //ymom_vertex_values[k3] = ymom_centroid_values[k] + dqv[0];
            //ymom_vertex_values[k3 + 1] = ymom_centroid_values[k] + dqv[1];
            //ymom_vertex_values[k3 + 2] = ymom_centroid_values[k] + dqv[2];
        } // else [number_of_boundaries==2]




    } // for k=0 to number_of_elements-1

    if (extrapolate_velocity_second_order == 1) {
        // Convert back from velocity to momentum
        for (k = 0; k < number_of_elements; k++) {
            k3 = 3 * k;
            //dv0 = fmax(stage_vertex_values[k3]-bed_vertex_values[k3],minimum_allowed_height);
            //dv1 = fmax(stage_vertex_values[k3+1]-bed_vertex_values[k3+1],minimum_allowed_height);
            //dv2 = fmax(stage_vertex_values[k3+2]-bed_vertex_values[k3+2],minimum_allowed_height);
            dv0 = fmax(stage_vertex_values[k3] - bed_vertex_values[k3], 0.);
            dv1 = fmax(stage_vertex_values[k3 + 1] - bed_vertex_values[k3 + 1], 0.);
            dv2 = fmax(stage_vertex_values[k3 + 2] - bed_vertex_values[k3 + 2], 0.);

            //Correct centroid and vertex values
            xmom_centroid_values[k] = xmom_centroid_store[k];
            xmom_vertex_values[k3] = xmom_vertex_values[k3] * dv0;
            xmom_vertex_values[k3 + 1] = xmom_vertex_values[k3 + 1] * dv1;
            xmom_vertex_values[k3 + 2] = xmom_vertex_values[k3 + 2] * dv2;

            ymom_centroid_values[k] = ymom_centroid_store[k];
            ymom_vertex_values[k3] = ymom_vertex_values[k3] * dv0;
            ymom_vertex_values[k3 + 1] = ymom_vertex_values[k3 + 1] * dv1;
            ymom_vertex_values[k3 + 2] = ymom_vertex_values[k3 + 2] * dv2;

        }
    }


    free(xmom_centroid_store);
    free(ymom_centroid_store);
    //free(stage_centroid_store);


}


anuga_int _openmp_update_conserved_quantities(const struct domain *__restrict D, 
                                              const double timestep)
      {
	// Update centroid values based on values stored in
	// explicit_update and semi_implicit_update as well as given timestep


	anuga_int k;
  anuga_int N = D->number_of_elements;
  

	// Divide semi_implicit update by conserved quantity
	#pragma omp parallel for private(k) schedule(static) shared(D) firstprivate(N, timestep)
	for (k=0; k<N; k++) {

    double stage_c, xmom_c, ymom_c;

    double denominator;

		// use previous centroid value
		stage_c = D->stage_centroid_values[k];
		if (stage_c == 0.0) {
			D->stage_semi_implicit_update[k] = 0.0;
		} else {
			D->stage_semi_implicit_update[k] /= stage_c;
		}
 

    xmom_c = D->xmom_centroid_values[k];
		if (xmom_c == 0.0) {
			D->xmom_semi_implicit_update[k] = 0.0;
		} else {
			D->xmom_semi_implicit_update[k] /= xmom_c;
		}

    ymom_c = D->ymom_centroid_values[k];
		if (ymom_c == 0.0) {
			D->ymom_semi_implicit_update[k] = 0.0;
		} else {
			D->ymom_semi_implicit_update[k] /= ymom_c;
		}

		// Explicit updates
		D->stage_centroid_values[k] += timestep*D->stage_explicit_update[k];
    D->xmom_centroid_values[k]  += timestep*D->xmom_explicit_update[k];
    D->ymom_centroid_values[k]  += timestep*D->ymom_explicit_update[k];

		// Semi implicit updates
		denominator = 1.0 - timestep*D->stage_semi_implicit_update[k];
		if (denominator > 0.0) {
			//Update conserved_quantities from semi implicit updates
			D->stage_centroid_values[k] /= denominator;
		}

    denominator = 1.0 - timestep*D->xmom_semi_implicit_update[k];
		if (denominator > 0.0) {
			//Update conserved_quantities from semi implicit updates
			D->xmom_centroid_values[k] /= denominator;
		}

    denominator = 1.0 - timestep*D->ymom_semi_implicit_update[k];
		if (denominator > 0.0) {
			//Update conserved_quantities from semi implicit updates
			D->ymom_centroid_values[k] /= denominator;
		}
		
		// Reset semi_implicit_update here ready for next time step
		D->stage_semi_implicit_update[k] = 0.0;
    D->xmom_semi_implicit_update[k] = 0.0;
    D->ymom_semi_implicit_update[k] = 0.0;
	}

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
  anuga_int k;
  anuga_int N = D->number_of_elements;

  // double stage_tmp[N];
  // double xmom_tmp[N];
  // double ymom_tmp[N];

  #pragma omp parallel for simd default(none) shared(D) schedule(static) firstprivate(N)
  for (k = 0; k < N; k++)
  {
    D->stage_backup_values[k] = D->stage_centroid_values[k];
    D->xmom_backup_values[k]  = D->xmom_centroid_values[k];
    D->ymom_backup_values[k]  = D->ymom_centroid_values[k];

  }

// #pragma omp parallel for simd default(none) shared(D, stage_tmp, xmom_tmp, ymom_tmp) \
//         schedule(static) firstprivate(N)
//   for (k = 0; k < N; k++)
//   {
//     stage_tmp[k] = D->stage_centroid_values[k];
//     xmom_tmp[k]  = D->xmom_centroid_values[k];
//     ymom_tmp[k]  = D->ymom_centroid_values[k];
// }

// #pragma omp parallel for simd default(none) shared(D, stage_tmp, xmom_tmp, ymom_tmp) \
//         schedule(static) firstprivate(N)
//   for (k = 0; k < N; k++)
//   {
//     D->stage_backup_values[k] = stage_tmp[k];
//     D->xmom_backup_values[k]  = xmom_tmp[k];
//     D->ymom_backup_values[k]  = ymom_tmp[k];
// }
  return 0;
}

void _openmp_set_omp_num_threads(anuga_int num_threads)
{
  // Set the number of threads for OpenMP
  // This is a global setting and will affect all subsequent OpenMP parallel regions
  omp_set_num_threads(num_threads);
}

void _openmp_evaluate_reflective_segment(struct domain *D, anuga_int N,
   anuga_int *edge_segment, anuga_int *vol_ids, anuga_int *edge_ids){

    #pragma omp parallel for schedule(static)
     for(int k = 0; k < N; k++){


      // get vol_ids 
      int edge_segment_id = edge_segment[k];
      int vid = vol_ids[k];
      int edge_id = edge_ids[k];
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