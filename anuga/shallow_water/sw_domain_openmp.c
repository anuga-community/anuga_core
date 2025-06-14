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

#if defined(__APPLE__)
// clang doesn't have openmp
#else
#include "omp.h"
#endif

#include "util_ext.h"
#include "sw_domain.h"

const double pi = 3.14159265358979;

// FIXME: Perhaps use the epsilon used elsewhere.
static const double TINY = 1.0e-100; // to avoid machine accuracy problems.

// Trick to compute n modulo d (n%d in python) when d is a power of 2
uint64_t __mod_of_power_2(uint64_t n, uint64_t d)
{
  return (n & (d - 1));
}

// Computational function for rotation
inline int64_t __rotate(double *q, const double n1, const double n2)
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
    const int64_t low_froude,
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
int64_t __flux_function_central(double *__restrict q_left, double *__restrict q_right,
                                const double h_left, const double h_right,
                                const double hle, const double hre,
                                const double n1, const double n2,
                                const double epsilon,
                                const double ze,
                                const double g,
                                double *__restrict edgeflux, double *__restrict max_speed,
                                double *__restrict pressure_flux,
                                const int64_t low_froude)
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


  for (int i = 0; i < 3; i++)
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

int64_t __openmp__flux_function_central(double q_left0, double q_left1, double q_left2,
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
                                        int64_t low_froude)
{

  double edgeflux[3];
  double q_left[3];
  double q_right[3];

  int64_t ierr;

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
  int64_t ierr;

  edgeflux[0] = *edgeflux0;
  edgeflux[1] = *edgeflux1;
  edgeflux[2] = *edgeflux2;

  ierr = __adjust_edgeflux_with_weir(edgeflux0, h_left, h_right,
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
                                     const int k, const int ncol_riverwall_hydraulic_properties,
                                     const double g, double * __restrict edgeflux, double * __restrict max_speed) {

    int RiverWall_count = D->edge_river_wall_counter[E->ki];
    int ii = D->riverwall_rowIndex[RiverWall_count - 1] * ncol_riverwall_hydraulic_properties;

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

double _openmp_compute_fluxes_central(struct domain *D,
                                      double timestep)
{
  // Local variables
  int number_of_elements = D->number_of_elements;
  // int64_t KI, KI2, KI3, B, RW, RW5, SubSteps;
  int64_t substep_count;

  // // FIXME: limiting_threshold is not used for DE1
  int64_t low_froude = D->low_froude;
  double g = D->g;
  double epsilon = D->epsilon;
  int64_t ncol_riverwall_hydraulic_properties = D->ncol_riverwall_hydraulic_properties;

  static int64_t call = 0; // Static local variable flagging already computed flux
  static int64_t timestep_fluxcalls = 1;
  static int64_t base_call = 1;

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

double speed_max_last = 0.0;
      double edgeflux[3];
      double pressure_flux;
      double max_speed_local;
      EdgeData edge_data;
// For all triangles
#pragma omp parallel for simd default(none) schedule(static) shared(D, substep_count, number_of_elements) \
    firstprivate(ncol_riverwall_hydraulic_properties, epsilon, g, low_froude)                              \
    private(speed_max_last, edgeflux, pressure_flux, max_speed_local, edge_data) \
    reduction(min : local_timestep) reduction(+ : boundary_flux_sum_substep)
  for (int k = 0; k < number_of_elements; k++)
  {
    double speed_max_last = 0.0;
    // Set explicit_update to zero for all conserved_quantities.
    // This assumes compute_fluxes called before forcing terms
    D->stage_explicit_update[k] = 0.0;
    D->xmom_explicit_update[k] = 0.0;
    D->ymom_explicit_update[k] = 0.0;

    // Loop through neighbours and compute edge flux for each
    for (int i = 0; i < 3; i++)
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
      for (int j = 0; j < 3; j++)
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
double _openmp_protect(struct domain *D)
{

  double mass_error = 0.;

  double minimum_allowed_height = D->minimum_allowed_height;

  int number_of_elements = D->number_of_elements;

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
  // int64_t mass_added = 0;

  // Protect against inifintesimal and negative heights
  // if (maximum_allowed_speed < epsilon) {
#pragma omp parallel for schedule(static) reduction(+ : mass_error) firstprivate(minimum_allowed_height)
  for (int k = 0; k < number_of_elements; k++)
  {
    int k3 = 3 * k;
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

static inline int64_t __find_qmin_and_qmax_dq1_dq2(const double dq0, const double dq1, const double dq2,
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

static inline int64_t __limit_gradient(double *__restrict dqv, double qmin, double qmax, const double beta_w)
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

  for (int i = 0; i < 3; i++)
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
                                          const int number_of_elements,
                                          const double minimum_allowed_height,
                                          const int extrapolate_velocity_second_order)
{
  double height_tmp[number_of_elements];
  double xmom_tmp[number_of_elements];
  double ymom_tmp[number_of_elements];
  double xwork_tmp[number_of_elements];
  double ywork_tmp[number_of_elements];
#pragma omp parallel for simd default(none) shared(D,height_tmp, xmom_tmp, ymom_tmp, xwork_tmp, ywork_tmp) schedule(static) \
    firstprivate(number_of_elements, minimum_allowed_height, extrapolate_velocity_second_order)
  for (int k = 0; k < number_of_elements; ++k)
  {
    double stage = D->stage_centroid_values[k];
    double bed = D->bed_centroid_values[k];
    double xmom = D->xmom_centroid_values[k];
    double ymom = D->ymom_centroid_values[k];

    double dk_local = fmax(stage - bed, 0.0);
    height_tmp[k] = dk_local;
    //D->height_centroid_values[k] = dk_local;

    int is_dry = (dk_local <= minimum_allowed_height);
    int extrapolate = (extrapolate_velocity_second_order == 1) & (dk_local > minimum_allowed_height);

    // Pre-zero everything
    double xwork = 0.0;
    double ywork = 0.0;
    double xmom_out = is_dry ? 0.0 : xmom;
    double ymom_out = is_dry ? 0.0 : ymom;

    // Store if extrapolating
    if (extrapolate)
    {
      double inv_dk = 1.0 / dk_local;
      xwork = xmom_out;
      xmom_out *= inv_dk;
      ywork = ymom_out;
      ymom_out *= inv_dk;
    }

    xmom_tmp[k] = xmom_out;
    ymom_tmp[k] = ymom_out;
    xwork_tmp[k] = xwork;
    ywork_tmp[k] = ywork;

    // D->x_centroid_work[k] = xwork;
    // D->y_centroid_work[k] = ywork;
    // D->xmom_centroid_values[k] = xmom_out;
    // D->ymom_centroid_values[k] = ymom_out;
  }

    // Second loop: write results back to domain
#pragma omp parallel for simd default(none) shared(D, height_tmp, xmom_tmp, ymom_tmp, xwork_tmp, ywork_tmp) \
    firstprivate(number_of_elements)
  for (int k = 0; k < number_of_elements; ++k)
  {
    D->height_centroid_values[k] = height_tmp[k];
    D->xmom_centroid_values[k]   = xmom_tmp[k];
    D->ymom_centroid_values[k]   = ymom_tmp[k];
    D->x_centroid_work[k]        = xwork_tmp[k];
    D->y_centroid_work[k]        = ywork_tmp[k];
  }
}

#pragma omp declare simd
static inline void set_all_edge_values_from_centroid(struct domain *__restrict D, const int k)
{

  const double stage = D->stage_centroid_values[k];
  const double xmom = D->xmom_centroid_values[k];
  const double ymom = D->ymom_centroid_values[k];
  const double height = D->height_centroid_values[k];

  for (int i = 0; i < 3; i++)
  {
    int ki = 3 * k + i;
    D->stage_edge_values[ki] = stage;
    D->xmom_edge_values[ki] = xmom;
    D->ymom_edge_values[ki] = ymom;
    D->height_edge_values[ki] = height;
    D->bed_edge_values[ki] = D->bed_centroid_values[k];
  }
}

#pragma omp declare simd
static inline int get_internal_neighbour(const struct domain *__restrict D, const int k)
{
  for (int i = 0; i < 3; i++)
  {
    int n = D->surrogate_neighbours[3 * k + i];
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
    const struct domain *__restrict D, const int k, const int k1,
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
    const int k, const int k1, const int k3,
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

  for (int i = 0; i < 3; i++)
  {
    edge_values[k3 + i] = centroid_values[k] + dqv[i];
  }
}

#pragma omp declare simd
static inline void interpolate_edges_with_beta(
    const double *__restrict centroid_values,
    double *__restrict edge_values,
    const int k, const int k0, const int k1, const int k2, const int k3,
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
  for (int i = 0; i < 3; i++)
  {
    edge_values[k3 + i] = edge_vals[i];
  }
}

#pragma omp declare simd
static inline void compute_hfactor_and_inv_area(
    const struct domain *__restrict D,
    const int k, const int k0, const int k1, const int k2,
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
static inline void reconstruct_vertex_values(double *__restrict edge_values, double *__restrict vertex_values, const int k3)
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
int64_t _openmp_extrapolate_second_order_edge_sw(struct domain *__restrict D)
{
  double minimum_allowed_height = D->minimum_allowed_height;
  int number_of_elements = D->number_of_elements;
  int64_t extrapolate_velocity_second_order = D->extrapolate_velocity_second_order;

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
  for (int k = 0; k < number_of_elements; k++)
  {
    // // Useful indices
    int k2 = k * 2;
    int k3 = k * 3;
    int k6 = k * 6;

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

    int k0 = D->surrogate_neighbours[k3 + 0];
    int k1 = D->surrogate_neighbours[k3 + 1];
    k2 = D->surrogate_neighbours[k3 + 2];

    int coord_index = 2 * k0;
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

    const int dry =
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
      for (int i = 0; i < 3; i++)
      {
        double dk = D->height_edge_values[k3 + i];
        D->xmom_edge_values[k3 + i] = D->xmom_edge_values[k3 + i] * dk;
        D->ymom_edge_values[k3 + i] = D->ymom_edge_values[k3 + i] * dk;
      }
    }

    for (int i = 0; i < 3; i++)
    {
      D->bed_edge_values[k3 + i] = D->stage_edge_values[k3 + i] - D->height_edge_values[k3 + i];
    }

    reconstruct_vertex_values(D->stage_edge_values, D->stage_vertex_values, k3);
    reconstruct_vertex_values(D->height_edge_values, D->height_vertex_values, k3);
    reconstruct_vertex_values(D->xmom_edge_values, D->xmom_vertex_values, k3);
    reconstruct_vertex_values(D->ymom_edge_values, D->ymom_vertex_values, k3);
    reconstruct_vertex_values(D->bed_edge_values, D->bed_vertex_values, k3);
  }
  // for k=0 to number_of_elements-1
// Fix xmom and ymom centroid values
if(extrapolate_velocity_second_order == 1)
{
#pragma omp parallel for simd schedule(static) firstprivate(extrapolate_velocity_second_order)
  for (int k = 0; k < D->number_of_elements; k++)
  {
      // Convert velocity back to momenta at centroids
      D->xmom_centroid_values[k] = D->x_centroid_work[k];
      D->ymom_centroid_values[k] = D->y_centroid_work[k];
  }
}
  // Convert velocity back to momenta at centroids


  return 0;
}

void _openmp_manning_friction_flat(const double g, const double eps, const int64_t N,
                                   double *__restrict w, double *__restrict zv,
                                   double *__restrict uh, double *__restrict vh,
                                   double *__restrict eta, double *__restrict xmom_update, double *__restrict ymom_update)
{

  const double seven_thirds = 7.0 / 3.0;

#pragma omp parallel for schedule(static) firstprivate(eps, g, seven_thirds)
  for (int k = 0; k < N; k++)
  {
    double abs_mom = sqrt((uh[k] * uh[k] + vh[k] * vh[k]));
    double S = 0.0;

    if (eta[k] > eps)
    {
      double z = zv[k];
      double h = w[k] - z;
      if (h >= eps)
      {
        S = -g * eta[k] * eta[k] * abs_mom;
        S /= pow(h, seven_thirds); // Expensive (on Ole's home computer)
        // S /= exp((7.0/3.0)*log(h));      //seems to save about 15% over manning_friction
        // S /= h*h*(1 + h/3.0 - h*h/9.0); //FIXME: Could use a Taylor expansion

        // Update momentum
      }
    }
    xmom_update[k] += S * uh[k];
    ymom_update[k] += S * vh[k];
  }
}

void _openmp_manning_friction_sloped(const double g, const double eps, const int64_t N,
                                     double *__restrict x, double *__restrict w, double *__restrict zv,
                                     double *__restrict uh, double *__restrict vh,
                                     double *__restrict eta, double *__restrict xmom_update, double *__restrict ymom_update)
{

  const double one_third = 1.0 / 3.0;
  const double seven_thirds = 7.0 / 3.0;

#pragma omp parallel for schedule(static) firstprivate(eps, g, one_third, seven_thirds)
  for (int k = 0; k < N; k++)
  {
    double S = 0.0;
    int k3 = 3 * k;
    // Get bathymetry
    double z0 = zv[k3 + 0];
    double z1 = zv[k3 + 1];
    double z2 = zv[k3 + 2];

    // Compute bed slope
    int k6 = 6 * k; // base index

    double x0 = x[k6 + 0];
    double y0 = x[k6 + 1];
    double x1 = x[k6 + 2];
    double y1 = x[k6 + 3];
    double x2 = x[k6 + 4];
    double y2 = x[k6 + 5];

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
        S /= pow(h, seven_thirds); // Expensive (on Ole's home computer)
        // S /= exp((7.0/3.0)*log(h));      //seems to save about 15% over manning_friction
        // S /= h*h*(1 + h/3.0 - h*h/9.0); //FIXME: Could use a Taylor expansion
      }
    }
    xmom_update[k] += S * uh[k];
    ymom_update[k] += S * vh[k];
  }
}

// Computational function for flux computation
int64_t _openmp_fix_negative_cells(struct domain *D)
{
  int64_t num_negative_cells = 0;

#pragma omp parallel for schedule(static) reduction(+ : num_negative_cells)
  for (int k = 0; k < D->number_of_elements; k++)
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
