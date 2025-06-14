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
// Stephen Roberts, ANU 2009
// Ole Nielsen, GA 2004
// Gareth Davies, GA 2011

#include "math.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

// #if defined(__APPLE__)
// // clang doesn't have openmp
// #else
// #include "omp.h"
// #endif

#include "sw_domain.h"

const double pi = 3.14159265358979;

// Trick to compute n modulo d (n%d in python) when d is a power of 2
uint64_t __mod_of_power_2(uint64_t n, uint64_t d)
{
  return (n & (d - 1));
}

// Computational function for rotation
int64_t __rotate(double *q, double n1, double n2)
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

// Innermost flux function (using stage w=z+h)
int64_t __flux_function_central(double *q_left, double *q_right,
                            double h_left, double h_right,
                            double hle, double hre,
                            double n1, double n2,
                            double epsilon,
                            double ze,
                            double limiting_threshold,
                            double g,
                            double *edgeflux, double *max_speed,
                            double *pressure_flux, double hc,
                            double hc_n,
                            int64_t low_froude)
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

  int64_t i;

  double uh_left, vh_left, u_left;
  double uh_right, vh_right, u_right;
  double s_min, s_max, soundspeed_left, soundspeed_right;
  double denom, inverse_denominator;
  double tmp, local_fr, v_right, v_left;
  double q_left_rotated[3], q_right_rotated[3], flux_right[3], flux_left[3];

  if (h_left == 0. && h_right == 0.)
  {
    // Quick exit
    memset(edgeflux, 0, 3 * sizeof(double));
    *max_speed = 0.0;
    *pressure_flux = 0.;
    return 0;
  }
  // Copy conserved quantities to protect from modification
  q_left_rotated[0] = q_left[0];
  q_right_rotated[0] = q_right[0];
  q_left_rotated[1] = q_left[1];
  q_right_rotated[1] = q_right[1];
  q_left_rotated[2] = q_left[2];
  q_right_rotated[2] = q_right[2];

  // Align x- and y-momentum with x-axis
  __rotate(q_left_rotated, n1, n2);
  __rotate(q_right_rotated, n1, n2);

  // Compute speeds in x-direction
  // w_left = q_left_rotated[0];
  uh_left = q_left_rotated[1];
  vh_left = q_left_rotated[2];
  if (hle > 0.0)
  {
    tmp = 1.0 / hle;
    u_left = uh_left * tmp; // max(h_left, 1.0e-06);
    uh_left = h_left * u_left;
    v_left = vh_left * tmp; // Only used to define local_fr
    vh_left = h_left * tmp * vh_left;
  }
  else
  {
    u_left = 0.;
    uh_left = 0.;
    vh_left = 0.;
    v_left = 0.;
  }

  // u_left = _compute_speed(&uh_left, &hle,
  //             epsilon, h0, limiting_threshold);

  // w_right = q_right_rotated[0];
  uh_right = q_right_rotated[1];
  vh_right = q_right_rotated[2];
  if (hre > 0.0)
  {
    tmp = 1.0 / hre;
    u_right = uh_right * tmp; // max(h_right, 1.0e-06);
    uh_right = h_right * u_right;
    v_right = vh_right * tmp; // Only used to define local_fr
    vh_right = h_right * tmp * vh_right;
  }
  else
  {
    u_right = 0.;
    uh_right = 0.;
    vh_right = 0.;
    v_right = 0.;
  }
  // u_right = _compute_speed(&uh_right, &hre,
  //               epsilon, h0, limiting_threshold);

  // Maximal and minimal wave speeds
  soundspeed_left = sqrt(g * h_left);
  soundspeed_right = sqrt(g * h_right);
  // soundspeed_left  = sqrt(g*hle);
  // soundspeed_right = sqrt(g*hre);

  // Something that scales like the Froude number
  // We will use this to scale the diffusive component of the UH/VH fluxes.

  // local_fr = sqrt(
  //     max(0.001, min(1.0,
  //         (u_right*u_right + u_left*u_left + v_right*v_right + v_left*v_left)/
  //         (soundspeed_left*soundspeed_left + soundspeed_right*soundspeed_right + 1.0e-10))));
  if (low_froude == 1)
  {
    local_fr = sqrt(
        fmax(0.001, fmin(1.0,
                         (u_right * u_right + u_left * u_left + v_right * v_right + v_left * v_left) /
                             (soundspeed_left * soundspeed_left + soundspeed_right * soundspeed_right + 1.0e-10))));
  }
  else if (low_froude == 2)
  {
    local_fr = sqrt((u_right * u_right + u_left * u_left + v_right * v_right + v_left * v_left) /
                    (soundspeed_left * soundspeed_left + soundspeed_right * soundspeed_right + 1.0e-10));
    local_fr = sqrt(fmin(1.0, 0.01 + fmax(local_fr - 0.01, 0.0)));
  }
  else
  {
    local_fr = 1.0;
  }
  // printf("local_fr %e \n:", local_fr);

  s_max = fmax(u_left + soundspeed_left, u_right + soundspeed_right);
  if (s_max < 0.0)
  {
    s_max = 0.0;
  }

  // if( hc < 1.0e-03){
  //   s_max = 0.0;
  // }

  s_min = fmin(u_left - soundspeed_left, u_right - soundspeed_right);
  if (s_min > 0.0)
  {
    s_min = 0.0;
  }

  // if( hc_n < 1.0e-03){
  //   s_min = 0.0;
  // }

  // Flux formulas
  flux_left[0] = u_left * h_left;
  flux_left[1] = u_left * uh_left; //+ 0.5*g*h_left*h_left;
  flux_left[2] = u_left * vh_left;

  flux_right[0] = u_right * h_right;
  flux_right[1] = u_right * uh_right; //+ 0.5*g*h_right*h_right;
  flux_right[2] = u_right * vh_right;

  // Flux computation
  denom = s_max - s_min;
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

    inverse_denominator = 1.0 / fmax(denom, 1.0e-100);
    for (i = 0; i < 3; i++)
    {
      edgeflux[i] = s_max * flux_left[i] - s_min * flux_right[i];

      // Standard smoothing term
      // edgeflux[i] += 1.0*(s_max*s_min)*(q_right_rotated[i] - q_left_rotated[i]);
      // Smoothing by stage alone can cause high velocities / slow draining for nearly dry cells
      if (i == 0)
        edgeflux[i] += (s_max * s_min) * (fmax(q_right_rotated[i], ze) - fmax(q_left_rotated[i], ze));
      // if(i==0) edgeflux[i] += (s_max*s_min)*(h_right - h_left);
      if (i == 1)
        edgeflux[i] += local_fr * (s_max * s_min) * (uh_right - uh_left);
      if (i == 2)
        edgeflux[i] += local_fr * (s_max * s_min) * (vh_right - vh_left);

      edgeflux[i] *= inverse_denominator;
    }
    // Separate pressure flux, so we can apply different wet-dry hacks to it
    *pressure_flux = 0.5 * g * (s_max * h_left * h_left - s_min * h_right * h_right) * inverse_denominator;

    // Rotate back
    __rotate(edgeflux, n1, -n2);
  }

  return 0;
}

int64_t __openacc__flux_function_central(double q_left0, double q_left1, double q_left2,
                                   double q_right0, double q_right1, double q_right2,
                                   double h_left, double h_right,
                                   double hle, double hre,
                                   double n1, double n2,
                                   double epsilon,
                                   double ze,
                                   double limiting_threshold,
                                   double g,
                                   double *edgeflux0, double *edgeflux1, double *edgeflux2,
                                   double *max_speed,
                                   double *pressure_flux, double hc,
                                   double hc_n,
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
                                 limiting_threshold,
                                 g,
                                 edgeflux, max_speed,
                                 pressure_flux, hc,
                                 hc_n,
                                 low_froude);

  *edgeflux0 = edgeflux[0];
  *edgeflux1 = edgeflux[1];
  *edgeflux2 = edgeflux[2];

  return ierr;
}

double __adjust_edgeflux_with_weir(double *edgeflux,
                                   double h_left, double h_right,
                                   double g, double weir_height,
                                   double Qfactor,
                                   double s1, double s2,
                                   double h1, double h2,
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

double __openacc__adjust_edgeflux_with_weir(double *edgeflux0, double *edgeflux1, double *edgeflux2,
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

// Computational function for flux computation
double _openacc_compute_fluxes_central(struct domain *D,
                                      double timestep)
{
  // Local variables
  int64_t K = D->number_of_elements;
  // int64_t KI, KI2, KI3, B, RW, RW5, SubSteps;
  int64_t substep_count;

  double max_speed_local, length, inv_area, zl, zr;
  double h_left, h_right, z_half; // For andusse scheme
  // FIXME: limiting_threshold is not used for DE1
  double limiting_threshold = 10 * D->H0;
  int64_t low_froude = D->low_froude;
  double g = D->g;
  double epsilon = D->epsilon;
  int64_t ncol_riverwall_hydraulic_properties = D->ncol_riverwall_hydraulic_properties;

  // Workspace (making them static actually made function slightly slower (Ole))
  double ql[3];
  double qr[3];
  double edgeflux[3]; // Work array for summing up fluxes
  double pressuregrad_work;
  double edge_timestep;
  double normal_x, normal_y;
  // static double local_timestep;

  double hle, hre, zc, zc_n, Qfactor, s1, s2, h1, h2;
  double pressure_flux, hc, hc_n;
  double h_left_tmp, h_right_tmp;
  double speed_max_last, weir_height;
  int64_t RiverWall_count;

  //
  int64_t k, i, m, n, ii;
  int64_t ki, nm = 0, ki2; // Index shorthands

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

// For all triangles
#pragma acc  parallel loop \
                                     firstprivate(ncol_riverwall_hydraulic_properties, epsilon, g, low_froude, limiting_threshold) \
                                     private(i, ki, ki2, n, m, nm, ii,                                                  \
                                     max_speed_local, length, inv_area, zl, zr,                                         \
                                     h_left, h_right,                                                                   \
                                     z_half, ql,  pressuregrad_work,                                                    \
                                     qr, edgeflux, edge_timestep, normal_x, normal_y,                                   \
                                     hle, hre, zc, zc_n, Qfactor, s1, s2, h1, h2, pressure_flux, hc, hc_n,              \
                                     h_left_tmp, h_right_tmp, speed_max_last, weir_height, RiverWall_count)             \
                                     reduction(min : local_timestep) reduction(+:boundary_flux_sum_substep)


 
  for (k = 0; k < K; k++)
  {
    speed_max_last = 0.0;
    // Set explicit_update to zero for all conserved_quantities.
    // This assumes compute_fluxes called before forcing terms
    D->stage_explicit_update[k] = 0.0;
    D->xmom_explicit_update[k] = 0.0;
    D->ymom_explicit_update[k] = 0.0;

    // Loop through neighbours and compute edge flux for each
    for (i = 0; i < 3; i++)
    {
      ki = 3 * k + i; // Linear index to edge i of triangle k
      ki2 = 2 * ki;   // k*6 + i*2

      // Get left hand side values from triangle k, edge i
      ql[0] = D->stage_edge_values[ki];
      ql[1] = D->xmom_edge_values[ki];
      ql[2] = D->ymom_edge_values[ki];
      zl    = D->bed_edge_values[ki];
      hle   = D->height_edge_values[ki];

      hc = D->height_centroid_values[k];
      zc = D->bed_centroid_values[k];

      // Get right hand side values either from neighbouring triangle
      // or from boundary array (Quantities at neighbour on nearest face).
      n = D->neighbours[ki];
      hc_n = hc;
      zc_n = D->bed_centroid_values[k];
      if (n < 0)
      {
        // Neighbour is a boundary condition
        m = -n - 1; // Convert negative flag to boundary index

        qr[0] = D->stage_boundary_values[m];
        qr[1] = D->xmom_boundary_values[m];
        qr[2] = D->ymom_boundary_values[m];
        zr = zl;                   // Extend bed elevation to boundary
        hre = fmax(qr[0] - zr, 0.0); // hle;
      }
      else
      {
        // Neighbour is a real triangle
        hc_n = D->height_centroid_values[n];
        zc_n = D->bed_centroid_values[n];

        m = D->neighbour_edges[ki];
        nm = n * 3 + m; // Linear index (triangle n, edge m)

        qr[0] = D->stage_edge_values[nm];
        qr[1] = D->xmom_edge_values[nm];
        qr[2] = D->ymom_edge_values[nm];
        zr = D->bed_edge_values[nm];
        hre = D->height_edge_values[nm];
      }

      // Audusse magic for well balancing
      z_half = fmax(zl, zr);

      // Account for riverwalls
      if (D->edge_flux_type[ki] == 1)
      {
        RiverWall_count = D->edge_river_wall_counter[ki];

        // Set central bed to riverwall elevation
        z_half = fmax(D->riverwall_elevation[RiverWall_count - 1], z_half);
      }

      // Define h left/right for Audusse flux method
      h_left = fmax(hle + zl - z_half, 0.);
      h_right = fmax(hre + zr - z_half, 0.);

      normal_x = D->normals[ki2];
      normal_y = D->normals[ki2 + 1];

      // Edge flux computation (triangle k, edge i)
      __flux_function_central(ql, qr,
                              h_left, h_right,
                              hle, hre,
                              normal_x, normal_y,
                              epsilon, z_half, limiting_threshold, g,
                              edgeflux, &max_speed_local, &pressure_flux,
                              hc, hc_n, low_froude);

      // Force weir discharge to match weir theory
      if (D->edge_flux_type[ki] == 1)
      {

        RiverWall_count = D->edge_river_wall_counter[ki];

        // printf("RiverWall_count %ld\n", RiverWall_count);

        ii = D->riverwall_rowIndex[RiverWall_count - 1] * ncol_riverwall_hydraulic_properties;

        // Get Qfactor index - multiply the idealised weir discharge by this constant factor
        // Get s1, submergence ratio at which we start blending with the shallow water solution
        // Get s2, submergence ratio at which we entirely use the shallow water solution
        // Get h1, tailwater head / weir height at which we start blending with the shallow water solution
        // Get h2, tailwater head / weir height at which we entirely use the shallow water solution
        Qfactor = D->riverwall_hydraulic_properties[ii];
        s1 = D->riverwall_hydraulic_properties[ii + 1];
        s2 = D->riverwall_hydraulic_properties[ii + 2];
        h1 = D->riverwall_hydraulic_properties[ii + 3];
        h2 = D->riverwall_hydraulic_properties[ii + 4];

        weir_height = fmax(D->riverwall_elevation[RiverWall_count - 1] - fmin(zl, zr), 0.); // Reference weir height

        // Use first-order h's for weir -- as the 'upstream/downstream' heads are
        //  measured away from the weir itself
        h_left_tmp = fmax(D->stage_centroid_values[k] - z_half, 0.);

        if (n >= 0)
        {
          h_right_tmp = fmax(D->stage_centroid_values[n] - z_half, 0.);
        }
        else
        {
          h_right_tmp = fmax(hc_n + zr - z_half, 0.);
        }

        // If the weir is not higher than both neighbouring cells, then
        // do not try to match the weir equation. If we do, it seems we
        // can get mass conservation issues (caused by large weir
        // fluxes in such situations)
        if (D->riverwall_elevation[RiverWall_count - 1] > fmax(zc, zc_n))
        {
          // Weir flux adjustment
          __adjust_edgeflux_with_weir(edgeflux, h_left_tmp, h_right_tmp, g,
                                             weir_height, Qfactor,
                                             s1, s2, h1, h2, &max_speed_local);
        }
      }

      // Multiply edgeflux by edgelength
      length = D->edgelengths[ki];
      edgeflux[0] = -edgeflux[0]*length;
      edgeflux[1] = -edgeflux[1]*length;
      edgeflux[2] = -edgeflux[2]*length;

      // bedslope_work contains all gravity related terms
      pressuregrad_work = length * (-g * 0.5 * (h_left * h_left - hle * hle - (hle + hc) * (zl - zc)) + pressure_flux);

      // Update timestep based on edge i and possibly neighbour n
      // NOTE: We should only change the timestep on the 'first substep'
      // of the timestepping method [substep_count==0]
      if (substep_count == 0)
      {

        // Compute the 'edge-timesteps' (useful for setting flux_update_frequency)
        edge_timestep = D->radii[k] *1.0 / fmax(max_speed_local, epsilon);

        // Update the timestep
        if (D->tri_full_flag[k] == 1)
        {
          if (max_speed_local > epsilon)
          {
            // Apply CFL condition for triangles joining this edge (triangle k and triangle n)

            // CFL for triangle k
            local_timestep = fmin(local_timestep, edge_timestep);

            speed_max_last = fmax(speed_max_last, max_speed_local);
          }
        }
      }


      D->stage_explicit_update[k] += edgeflux[0];
      D->xmom_explicit_update[k]  += edgeflux[1];
      D->ymom_explicit_update[k]  += edgeflux[2];

      // If this cell is not a ghost, and the neighbour is a
      // boundary condition OR a ghost cell, then add the flux to the
      // boundary_flux_integral
      if (((n < 0) & (D->tri_full_flag[k] == 1)) | ((n >= 0) && ((D->tri_full_flag[k] == 1) & (D->tri_full_flag[n] == 0))))
      {
        // boundary_flux_sum is an array with length = timestep_fluxcalls
        // For each sub-step, we put the boundary flux sum in.
        boundary_flux_sum_substep += edgeflux[0];
      }

      D->xmom_explicit_update[k] -= D->normals[ki2] * pressuregrad_work;
      D->ymom_explicit_update[k] -= D->normals[ki2 + 1] * pressuregrad_work;

    } // End edge i (and neighbour n)

    // Keep track of maximal speeds
    if (substep_count == 0)
      D->max_speed[k] = speed_max_last; // max_speed;

    // Normalise triangle k by area and store for when all conserved
    // quantities get updated
    inv_area = 1.0 / D->areas[k];
    D->stage_explicit_update[k] *= inv_area;
    D->xmom_explicit_update[k] *= inv_area;
    D->ymom_explicit_update[k] *= inv_area;  

  } // End triangle k



  // variable to accumulate D->boundary_flux_sum[substep_count]
  D->boundary_flux_sum[substep_count] = boundary_flux_sum_substep;  

  // Ensure we only update the timestep on the first call within each rk2/rk3 step
  if (substep_count == 0)
    timestep = local_timestep;

  return timestep;
}

// Computational function for flux computation
// with riverWall_count pulled out of triangle loop
double _compute_fluxes_central_parallel_data_flow(struct domain *D, double timestep)
{

  // Local variables
  double max_speed_local, length, inv_area, zl, zr;
  double h_left, h_right, z_half; // For andusse scheme
  // FIXME: limiting_threshold is not used for DE1
  double limiting_threshold = 10 * D->H0;
  int64_t low_froude = D->low_froude;
  //
  int64_t k, i, m, n, ii;
  int64_t ki, nm = 0, ki2, ki3; // Index shorthands
  // Workspace (making them static actually made function slightly slower (Ole))
  double ql[3], qr[3], edgeflux[3]; // Work array for summing up fluxes
  double bedslope_work;
  static double local_timestep;
  int64_t RiverWall_count, substep_count;
  double hle, hre, zc, zc_n, Qfactor, s1, s2, h1, h2;
  double pressure_flux, hc, hc_n, tmp;
  double h_left_tmp, h_right_tmp;
  static int64_t call = 0; // Static local variable flagging already computed flux
  static int64_t timestep_fluxcalls = 1;
  static int64_t base_call = 1;
  double speed_max_last, weir_height;

  call++; // Flag 'id' of flux calculation for this timestep

  if (D->timestep_fluxcalls != timestep_fluxcalls)
  {
    timestep_fluxcalls = D->timestep_fluxcalls;
    base_call = call;
  }

  // Set explicit_update to zero for all conserved_quantities.
  // This assumes compute_fluxes called before forcing terms

  // #pragma omp parallel for private(k)
  for (k = 0; k < D->number_of_elements; k++)
  {
    D->stage_explicit_update[k] = 0.0;
    D->xmom_explicit_update[k] = 0.0;
    D->ymom_explicit_update[k] = 0.0;
  }
  // memset((char*) D->stage_explicit_update, 0, D->number_of_elements * sizeof (double));
  // memset((char*) D->xmom_explicit_update, 0, D->number_of_elements * sizeof (double));
  // memset((char*) D->ymom_explicit_update, 0, D->number_of_elements * sizeof (double));

  // Counter for riverwall edges
  RiverWall_count = 0;
  // Which substep of the timestepping method are we on?
  substep_count = (call - base_call) % D->timestep_fluxcalls;

  // printf("call = %d substep_count = %d base_call = %d \n",call,substep_count, base_call);

  // Fluxes are not updated every timestep,
  // but all fluxes ARE updated when the following condition holds
  if (D->allow_timestep_increase[0] == 1)
  {
    // We can only increase the timestep if all fluxes are allowed to be updated
    // If this is not done the timestep can't increase (since local_timestep is static)
    local_timestep = 1.0e+100;
  }

  // For all triangles
  // Pull the edge_river_wall count outside parallel loop as in needs to be done sequentially
  // move it to the initiation of the riverwall so only calculated once
  for (k = 0; k < D->number_of_elements; k++)
  {
    for (i = 0; i < 3; i++)
    {
      ki = 3 * k + i;
      D->edge_river_wall_counter[ki] = 0;
      if (D->edge_flux_type[ki] == 1)
      {
        // Update counter of riverwall edges
        RiverWall_count += 1;
        D->edge_river_wall_counter[ki] = RiverWall_count;

        // printf("RiverWall_count %d   edge_counter %d \n", RiverWall_count, D->edge_river_wall_counter[ki]);
      }
    }
  }

  RiverWall_count = 0;

  // For all triangles
  for (k = 0; k < D->number_of_elements; k++)
  {
    speed_max_last = 0.0;

    // Loop through neighbours and compute edge flux for each
    for (i = 0; i < 3; i++)
    {
      ki = 3 * k + i; // Linear index to edge i of triangle k
      ki2 = 2 * ki;   // k*6 + i*2
      ki3 = 3 * ki;

      // Get left hand side values from triangle k, edge i
      ql[0] = D->stage_edge_values[ki];
      ql[1] = D->xmom_edge_values[ki];
      ql[2] = D->ymom_edge_values[ki];
      zl = D->bed_edge_values[ki];
      hc = D->height_centroid_values[k];
      zc = D->bed_centroid_values[k];
      hle = D->height_edge_values[ki];

      // Get right hand side values either from neighbouring triangle
      // or from boundary array (Quantities at neighbour on nearest face).
      n = D->neighbours[ki];
      hc_n = hc;
      zc_n = D->bed_centroid_values[k];
      if (n < 0)
      {
        // Neighbour is a boundary condition
        m = -n - 1; // Convert negative flag to boundary index

        qr[0] = D->stage_boundary_values[m];
        qr[1] = D->xmom_boundary_values[m];
        qr[2] = D->ymom_boundary_values[m];
        zr = zl;                    // Extend bed elevation to boundary
        hre = fmax(qr[0] - zr, 0.); // hle;
      }
      else
      {
        // Neighbour is a real triangle
        hc_n = D->height_centroid_values[n];
        zc_n = D->bed_centroid_values[n];
        m = D->neighbour_edges[ki];
        nm = n * 3 + m; // Linear index (triangle n, edge m)

        qr[0] = D->stage_edge_values[nm];
        qr[1] = D->xmom_edge_values[nm];
        qr[2] = D->ymom_edge_values[nm];
        zr = D->bed_edge_values[nm];
        hre = D->height_edge_values[nm];
      }

      // Audusse magic
      z_half = fmax(zl, zr);

      //// Account for riverwalls
      if (D->edge_flux_type[ki] == 1)
      {
        if (n >= 0 && D->edge_flux_type[nm] != 1)
        {
          printf("Riverwall Error\n");
        }
        // Update counter of riverwall edges == index of
        // riverwall_elevation + riverwall_rowIndex

        // RiverWall_count += 1;
        RiverWall_count = D->edge_river_wall_counter[ki];

        // Set central bed to riverwall elevation
        z_half = fmax(D->riverwall_elevation[RiverWall_count - 1], z_half);
      }

      // Define h left/right for Audusse flux method
      h_left = fmax(hle + zl - z_half, 0.);
      h_right = fmax(hre + zr - z_half, 0.);

      // Edge flux computation (triangle k, edge i)
      __flux_function_central(ql, qr,
                              h_left, h_right,
                              hle, hre,
                              D->normals[ki2], D->normals[ki2 + 1],
                              D->epsilon, z_half, limiting_threshold, D->g,
                              edgeflux, &max_speed_local, &pressure_flux, hc, hc_n, low_froude);

      // Force weir discharge to match weir theory
      if (D->edge_flux_type[ki] == 1)
      {
        ii = D->riverwall_rowIndex[RiverWall_count - 1] * D->ncol_riverwall_hydraulic_properties;

        // Get Qfactor index - multiply the idealised weir discharge by this constant factor
        // Get s1, submergence ratio at which we start blending with the shallow water solution
        // Get s2, submergence ratio at which we entirely use the shallow water solution
        // Get h1, tailwater head / weir height at which we start blending with the shallow water solution
        // Get h2, tailwater head / weir height at which we entirely use the shallow water solution
        Qfactor = D->riverwall_hydraulic_properties[ii];
        s1 = D->riverwall_hydraulic_properties[ii + 1];
        s2 = D->riverwall_hydraulic_properties[ii + 2];
        h1 = D->riverwall_hydraulic_properties[ii + 3];
        h2 = D->riverwall_hydraulic_properties[ii + 4];

        weir_height = fmax(D->riverwall_elevation[RiverWall_count - 1] - fmin(zl, zr), 0.); // Reference weir height

        // Use first-order h's for weir -- as the 'upstream/downstream' heads are
        //  measured away from the weir itself
        h_left_tmp = fmax(D->stage_centroid_values[k] - z_half, 0.);

        if (n >= 0)
        {
          h_right_tmp = fmax(D->stage_centroid_values[n] - z_half, 0.);
        }
        else
        {
          h_right_tmp = fmax(hc_n + zr - z_half, 0.);
        }

        // If the weir is not higher than both neighbouring cells, then
        // do not try to match the weir equation. If we do, it seems we
        // can get mass conservation issues (caused by large weir
        // fluxes in such situations)
        if (D->riverwall_elevation[RiverWall_count - 1] > fmax(zc, zc_n))
        {
          // Weir flux adjustment
          __adjust_edgeflux_with_weir(edgeflux, h_left_tmp, h_right_tmp, D->g,
                                      weir_height, Qfactor,
                                      s1, s2, h1, h2, &max_speed_local);
        }
      }

      // Multiply edgeflux by edgelength
      length = D->edgelengths[ki];
      edgeflux[0] *= length;
      edgeflux[1] *= length;
      edgeflux[2] *= length;

      D->edge_flux_work[ki3 + 0] = -edgeflux[0];
      D->edge_flux_work[ki3 + 1] = -edgeflux[1];
      D->edge_flux_work[ki3 + 2] = -edgeflux[2];

      // bedslope_work contains all gravity related terms
      bedslope_work = length * (-D->g * 0.5 * (h_left * h_left - hle * hle - (hle + hc) * (zl - zc)) + pressure_flux);

      D->pressuregrad_work[ki] = bedslope_work;

      // Update timestep based on edge i and possibly neighbour n
      // NOTE: We should only change the timestep on the 'first substep'
      //  of the timestepping method [substep_count==0]
      if (substep_count == 0)
      {

        // Compute the 'edge-timesteps' (useful for setting flux_update_frequency)
        tmp = 1.0 / fmax(max_speed_local, D->epsilon);
        D->edge_timestep[ki] = D->radii[k] * tmp;

        // Update the timestep
        if (D->tri_full_flag[k] == 1)
        {

          speed_max_last = fmax(speed_max_last, max_speed_local);

          if (max_speed_local > D->epsilon)
          {
            // Apply CFL condition for triangles joining this edge (triangle k and triangle n)

            // CFL for triangle k
            local_timestep = fmin(local_timestep, D->edge_timestep[ki]);

            // if (n >= 0) {
            //     // Apply CFL condition for neigbour n (which is on the ith edge of triangle k)
            //    local_timestep = fmin(local_timestep, D->edge_timestep[nm]);
            // }
          }
        }
      }

    } // End edge i (and neighbour n)

    // Keep track of maximal speeds
    if (substep_count == 0)
      D->max_speed[k] = speed_max_last; // max_speed;

  } // End triangle k

  // Now add up stage, xmom, ymom explicit updates
  for (k = 0; k < D->number_of_elements; k++)
  {
    hc = fmax(D->stage_centroid_values[k] - D->bed_centroid_values[k], 0.);

    for (i = 0; i < 3; i++)
    {
      // FIXME: Make use of neighbours to efficiently set things
      ki = 3 * k + i;
      ki2 = ki * 2;
      ki3 = ki * 3;
      n = D->neighbours[ki];

      D->stage_explicit_update[k] += D->edge_flux_work[ki3 + 0];
      D->xmom_explicit_update[k] += D->edge_flux_work[ki3 + 1];
      D->ymom_explicit_update[k] += D->edge_flux_work[ki3 + 2];

      // If this cell is not a ghost, and the neighbour is a
      // boundary condition OR a ghost cell, then add the flux to the
      // boundary_flux_integral
      if (((n < 0) & (D->tri_full_flag[k] == 1)) | ((n >= 0) && ((D->tri_full_flag[k] == 1) & (D->tri_full_flag[n] == 0))))
      {
        // boundary_flux_sum is an array with length = timestep_fluxcalls
        // For each sub-step, we put the boundary flux sum in.
        D->boundary_flux_sum[substep_count] += D->edge_flux_work[ki3];
      }

      D->xmom_explicit_update[k] -= D->normals[ki2] * D->pressuregrad_work[ki];
      D->ymom_explicit_update[k] -= D->normals[ki2 + 1] * D->pressuregrad_work[ki];

    } // end edge i

    // Normalise triangle k by area and store for when all conserved
    // quantities get updated
    inv_area = 1.0 / D->areas[k];
    D->stage_explicit_update[k] *= inv_area;
    D->xmom_explicit_update[k] *= inv_area;
    D->ymom_explicit_update[k] *= inv_area;

  } // end cell k

  // Ensure we only update the timestep on the first call within each rk2/rk3 step
  if (substep_count == 0)
    timestep = local_timestep;

  return timestep;
}





// Protect against the water elevation falling below the triangle bed
double _openacc_protect(struct domain *D)
{

  int64_t k, k3, K;
  double hc, bmin;
  double mass_error = 0.;

  // double *wc;
  // double *zc;
  // double *wv;
  // double *xmomc;
  // double *ymomc;
  // double *areas;

  double minimum_allowed_height;

  minimum_allowed_height = D->minimum_allowed_height;

  K = D->number_of_elements;

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
// #pragma omp parallel for private(k, k3, hc, bmin ) schedule(static) reduction(+ : mass_error) firstprivate (minimum_allowed_height)
  for (k = 0; k < K; k++)
  {
    k3 = 3*k;
    hc = D->stage_centroid_values[k] - D->bed_centroid_values[k];
    if (hc < minimum_allowed_height * 1.0)
    {
      // Set momentum to zero and ensure h is non negative
      D->xmom_centroid_values[k] = 0.;
      D->xmom_centroid_values[k] = 0.;
      if (hc <= 0.0)
      {
        bmin = D->bed_centroid_values[k];
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


static inline int64_t __find_qmin_and_qmax(double dq0, double dq1, double dq2,
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

static inline int64_t __limit_gradient(double *dqv, double qmin, double qmax, double beta_w)
{
  // Given provisional jumps dqv from the FV triangle centroid to its
  // vertices/edges, and jumps qmin (qmax) between the centroid of the FV
  // triangle and the minimum (maximum) of the values at the auxiliary triangle
  // vertices (which are centroids of neighbour mesh triangles), calculate a
  // multiplicative factor phi by which the provisional vertex jumps are to be
  // limited

  int64_t i;
  double r = 1000.0, r0 = 1.0, phi = 1.0;
  static double TINY = 1.0e-100; // to avoid machine accuracy problems.
  // FIXME: Perhaps use the epsilon used elsewhere.

  // Any provisional jump with magnitude < TINY does not contribute to
  // the limiting process.
  // return 0;

  for (i = 0; i < 3; i++)
  {
    if (dqv[i] < -TINY)
      r0 = qmin / dqv[i];

    if (dqv[i] > TINY)
      r0 = qmax / dqv[i];

    r = fmin(r0, r);
  }

  phi = fmin(r * beta_w, 1.0);
  // phi=1.;
  dqv[0] = dqv[0] * phi;
  dqv[1] = dqv[1] * phi;
  dqv[2] = dqv[2] * phi;

  return 0;
}

static inline void __calc_edge_values(double beta_tmp, double cv_k, double cv_k0, double cv_k1, double cv_k2,
                        double dxv0, double dxv1, double dxv2, double dyv0, double dyv1, double dyv2,
                        double dx1, double dx2, double dy1, double dy2, double inv_area2,
                        double *edge_values)
{
  double dqv[3];
  double dq0, dq1, dq2;
  double a, b;
  double qmin, qmax;

  if (beta_tmp > 0.)
  {
    // Calculate the difference between vertex 0 of the auxiliary
    // triangle and the centroid of triangle k
    dq0 = cv_k0 - cv_k;

    // Calculate differentials between the vertices
    // of the auxiliary triangle (centroids of neighbouring triangles)
    dq1 = cv_k1 - cv_k0;
    dq2 = cv_k2 - cv_k0;

    // Calculate the gradient of stage on the auxiliary triangle
    a = dy2 * dq1 - dy1 * dq2;
    a *= inv_area2;
    b = dx1 * dq2 - dx2 * dq1;
    b *= inv_area2;
    // Calculate provisional jumps in stage from the centroid
    // of triangle k to its vertices, to be limited
    dqv[0] = a * dxv0 + b * dyv0;
    dqv[1] = a * dxv1 + b * dyv1;
    dqv[2] = a * dxv2 + b * dyv2;

    // Now we want to find min and max of the centroid and the
    // vertices of the auxiliary triangle and compute jumps
    // from the centroid to the min and max
    __find_qmin_and_qmax(dq0, dq1, dq2, &qmin, &qmax);

    // Limit the gradient
    __limit_gradient(dqv, qmin, qmax, beta_tmp);

    edge_values[0] = cv_k + dqv[0];
    edge_values[1] = cv_k + dqv[1];
    edge_values[2] = cv_k + dqv[2];
  }
  else
  {
    // Fast alternative when beta_tmp==0
    edge_values[0] = cv_k;
    edge_values[1] = cv_k;
    edge_values[2] = cv_k;
  }
}

static inline void __calc_edge_values_2_bdy(double beta, double cv_k, double cv_k0, 
                        double dxv0, double dxv1, double dxv2, double dyv0, double dyv1, double dyv2,
                        double dx1, double dx2, double dy1, double dy2, double inv_area2,
                        double *edge_values)
{
  double dqv[3];
  double dq1;
  double a, b;
  double qmin, qmax;


  // Compute differentials
  dq1 = cv_k0 - cv_k;

  // Calculate the gradient between the centroid of triangle k
  // and that of its neighbour
  a = dq1 * dx2;
  b = dq1 * dy2;

  // Calculate provisional edge jumps, to be limited
  dqv[0] = a * dxv0 + b * dyv0;
  dqv[1] = a * dxv1 + b * dyv1;
  dqv[2] = a * dxv2 + b * dyv2;

  // Now limit the jumps
  if (dq1 >= 0.0)
  {
    qmin = 0.0;
    qmax = dq1;
  }
  else
  {
    qmin = dq1;
    qmax = 0.0;
  }

  // Limit the gradient
  __limit_gradient(dqv, qmin, qmax, beta);

  edge_values[0] = cv_k + dqv[0];
  edge_values[1] = cv_k + dqv[1];
  edge_values[2] = cv_k + dqv[2];

}





// Computational routine
int64_t _openacc_extrapolate_second_order_edge_sw(struct domain *D)
{

  // Local variables
  double a, b; // Gradient vector used to calculate edge values from centroids
  int64_t k, k0, k1, k2, k3, k6, coord_index, i;
  double x, y, x0, y0, x1, y1, x2, y2, xv0, yv0, xv1, yv1, xv2, yv2; // Vertices of the auxiliary triangle
  double dx1, dx2, dy1, dy2, dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, dq1, area2, inv_area2;
  double dqv[3], qmin, qmax, hmin, hmax;
  double hc, h0, h1, h2, beta_tmp, hfactor;
  double dk, dk_inv, a_tmp, b_tmp, c_tmp, d_tmp;
  double edge_values[3];
  double cv_k, cv_k0, cv_k1, cv_k2;


  double minimum_allowed_height = D->minimum_allowed_height;
  int64_t number_of_elements = D->number_of_elements;
  int64_t extrapolate_velocity_second_order = D->extrapolate_velocity_second_order;


  // Parameters used to control how the limiter is forced to first-order near
  // wet-dry regions
  a_tmp = 0.3; // Highest depth ratio with hfactor=1
  b_tmp = 0.1; // Highest depth ratio with hfactor=0
  c_tmp = 1.0 / (a_tmp - b_tmp);
  d_tmp = 1.0 - (c_tmp * a_tmp);

  // Replace momentum centroid with velocity centroid to allow velocity
  // extrapolation This will be changed back at the end of the routine

  // Need to calculate height xmom and ymom centroid values for all triangles 
  // before extrapolation and limiting

// #pragma omp parallel for simd shared(D) default(none) private(dk, dk_inv) firstprivate(number_of_elements, minimum_allowed_height, extrapolate_velocity_second_order)
    for (k = 0; k < number_of_elements; k++)
    {
    dk = fmax(D->stage_centroid_values[k] - D->bed_centroid_values[k], 0.0);

    D->height_centroid_values[k] = dk;
    D->x_centroid_work[k] = 0.0;
    D->y_centroid_work[k] = 0.0;

    if (dk <= minimum_allowed_height)
      {
        D->x_centroid_work[k] = 0.0;
        D->xmom_centroid_values[k] = 0.0;
        D->y_centroid_work[k] = 0.0;
        D->ymom_centroid_values[k] = 0.0;
      }

    if (extrapolate_velocity_second_order == 1)
    {
      if (dk > minimum_allowed_height)
      {
        dk_inv = 1.0 / dk;
        D->x_centroid_work[k] = D->xmom_centroid_values[k];
        D->xmom_centroid_values[k] = D->xmom_centroid_values[k] * dk_inv;

        D->y_centroid_work[k] = D->ymom_centroid_values[k];
        D->ymom_centroid_values[k] = D->ymom_centroid_values[k] * dk_inv;
      }
    }
    } // end of for



  // Begin extrapolation routine

// #pragma omp parallel for simd private(k0, k1, k2, k3, k6, coord_index, i, \
//                          dx1, dx2, dy1, dy2, dxv0, dxv1, dxv2, dyv0, dyv1, dyv2, \
//                          x_centroid_work, xmom_centroid_values, y_centroid_work, ymom_centroid_values, \
//                          dq1, area2, inv_area2, \
//                          cv_k, cv_k0, cv_k1, cv_k2, edge_values, \
//                          x, y, x0, y0, x1, y1, x2, y2, xv0, yv0, xv1, yv1, xv2, yv2, \
//                          dqv, qmin, qmax, hmin, hmax, \
//                          hc, h0, h1, h2, beta_tmp, hfactor, \
//                          dk, dk_inv, a, b) default(none) shared(D) \
//                          firstprivate(number_of_elements, minimum_allowed_height, extrapolate_velocity_second_order, c_tmp, d_tmp)
  for (k = 0; k < number_of_elements; k++)
  {

    //printf("%ld, %e \n",k, D->height_centroid_values[k]);
    //printf("%ld,  %e, %e, %e, %e \n",k, x_centroid_work,xmom_centroid_values,y_centroid_work,ymom_centroid_values);
    //printf("%ld,  %e, %e, %e, %e \n",k, D->x_centroid_work[k],D->xmom_centroid_values[k],D->y_centroid_work[k],D->ymom_centroid_values[k]);


    // Useful indices
    k2 = k * 2;
    k3 = k * 3;
    k6 = k * 6;

    // Get the edge coordinates
    xv0 = D->edge_coordinates[k6 + 0];
    yv0 = D->edge_coordinates[k6 + 1];
    xv1 = D->edge_coordinates[k6 + 2];
    yv1 = D->edge_coordinates[k6 + 3];
    xv2 = D->edge_coordinates[k6 + 4];
    yv2 = D->edge_coordinates[k6 + 5];

    // Get the centroid coordinates
    x = D->centroid_coordinates[k2 + 0];
    y = D->centroid_coordinates[k2 + 1];

    // Store x- and y- differentials for the edges of
    // triangle k relative to the centroid
    dxv0 = xv0 - x;
    dxv1 = xv1 - x;
    dxv2 = xv2 - x;
    dyv0 = yv0 - y;
    dyv1 = yv1 - y;
    dyv2 = yv2 - y;

    // If no boundaries, auxiliary triangle is formed
    // from the centroids of the three neighbours
    // If one boundary, auxiliary triangle is formed
    // from this centroid and its two neighbours

    k0 = D->surrogate_neighbours[k3 + 0];
    k1 = D->surrogate_neighbours[k3 + 1];
    k2 = D->surrogate_neighbours[k3 + 2];

    // Get the auxiliary triangle's vertex coordinates
    // (normally the centroids of neighbouring triangles)
    coord_index = 2 * k0;
    x0 = D->centroid_coordinates[coord_index + 0];
    y0 = D->centroid_coordinates[coord_index + 1];

    coord_index = 2 * k1;
    x1 = D->centroid_coordinates[coord_index + 0];
    y1 = D->centroid_coordinates[coord_index + 1];

    coord_index = 2 * k2;
    x2 = D->centroid_coordinates[coord_index + 0];
    y2 = D->centroid_coordinates[coord_index + 1];

    // Store x- and y- differentials for the vertices
    // of the auxiliary triangle
    dx1 = x1 - x0;
    dx2 = x2 - x0;
    dy1 = y1 - y0;
    dy2 = y2 - y0;

    // Calculate 2*area of the auxiliary triangle
    // The triangle is guaranteed to be counter-clockwise
    area2 = dy2 * dx1 - dy1 * dx2;

    if (((D->height_centroid_values[k0] < minimum_allowed_height) | (k0 == k)) &
        ((D->height_centroid_values[k1] < minimum_allowed_height) | (k1 == k)) &
        ((D->height_centroid_values[k2] < minimum_allowed_height) | (k2 == k)))
    {
      // printf("Surrounded by dry cells\n");
      D->x_centroid_work[k] = 0.;
      D->xmom_centroid_values[k] = 0.;
      D->y_centroid_work[k] = 0.;
      D->ymom_centroid_values[k] = 0.;
    }

    // Limit the edge values
    if (D->number_of_boundaries[k] == 3)
    {
      // Very unlikely
      // No neighbours, set gradient on the triangle to zero

      //printf("%ld 3 boundaries\n",k);

      D->stage_edge_values[k3 + 0] = D->stage_centroid_values[k];
      D->stage_edge_values[k3 + 1] = D->stage_centroid_values[k];
      D->stage_edge_values[k3 + 2] = D->stage_centroid_values[k];

      D->xmom_edge_values[k3 + 0] = D->xmom_centroid_values[k];
      D->xmom_edge_values[k3 + 1] = D->xmom_centroid_values[k];
      D->xmom_edge_values[k3 + 2] = D->xmom_centroid_values[k];

      D->ymom_edge_values[k3 + 0] = D->ymom_centroid_values[k];
      D->ymom_edge_values[k3 + 1] = D->ymom_centroid_values[k];
      D->ymom_edge_values[k3 + 2] = D->ymom_centroid_values[k];

      dk = D->height_centroid_values[k];
      D->height_edge_values[k3 + 0] = dk;
      D->height_edge_values[k3 + 1] = dk;
      D->height_edge_values[k3 + 2] = dk;

    }
    else if (D->number_of_boundaries[k] <= 1)
    {
      //==============================================
      // Number of boundaries <= 1
      // 'Typical case'
      //==============================================
      //printf("%ld boundaries <= 1\n",k);

      // Calculate heights of neighbouring cells
      hc = D->height_centroid_values[k];
      h0 = D->height_centroid_values[k0];
      h1 = D->height_centroid_values[k1];
      h2 = D->height_centroid_values[k2];

      hmin = fmin(fmin(h0, fmin(h1, h2)), hc);
      hmax = fmax(fmax(h0, fmax(h1, h2)), hc);

      // Look for strong changes in cell depth as an indicator of near-wet-dry
      // Reduce hfactor linearly from 1-0 between depth ratio (hmin/hc) of [a_tmp , b_tmp]
      // NOTE: If we have a more 'second order' treatment in near dry areas (e.g. with b_tmp being negative), then
      //       the water tends to dry more rapidly (which is in agreement with analytical results),
      //       but is also more 'artefacty' in important cases (tendency for high velocities, etc).
      //
      // So hfactor = depth_ratio*(c_tmp) + d_tmp, but is clipped between 0 and 1.
      hfactor = fmax(0., fmin(c_tmp * fmax(hmin, 0.0) / fmax(hc, 1.0e-06) + d_tmp,
                              fmin(c_tmp * fmax(hc, 0.) / fmax(hmax, 1.0e-06) + d_tmp, 1.0)));
      // Set hfactor to zero smothly as hmin--> minimum_allowed_height. This
      // avoids some 'chatter' for very shallow flows
      hfactor = fmin(1.2 * fmax(hmin - D->minimum_allowed_height, 0.) / (fmax(hmin, 0.) + 1. * D->minimum_allowed_height), hfactor);

      inv_area2 = 1.0 / area2;

      //-----------------------------------
      // stage
      //-----------------------------------
      beta_tmp = D->beta_w_dry + (D->beta_w - D->beta_w_dry) * hfactor;

      cv_k  = D->stage_centroid_values[k];
      cv_k0 = D->stage_centroid_values[k0];
      cv_k1 = D->stage_centroid_values[k1];
      cv_k2 = D->stage_centroid_values[k2];

      __calc_edge_values(beta_tmp, 
                         cv_k, 
                         cv_k0,
                         cv_k1,
                         cv_k2,
                         dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                         dx1, dx2, dy1, dy2, inv_area2, edge_values);

      D->stage_edge_values[k3 + 0] = edge_values[0];
      D->stage_edge_values[k3 + 1] = edge_values[1];
      D->stage_edge_values[k3 + 2] = edge_values[2];  

      //-----------------------------------
      // height
      //-----------------------------------

      cv_k  = D->height_centroid_values[k];
      cv_k0 = D->height_centroid_values[k0];
      cv_k1 = D->height_centroid_values[k1];
      cv_k2 = D->height_centroid_values[k2];

      __calc_edge_values(beta_tmp, 
                         cv_k, 
                         cv_k0,
                         cv_k1,
                         cv_k2,
                         dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                         dx1, dx2, dy1, dy2, inv_area2, edge_values);

      D->height_edge_values[k3 + 0] = edge_values[0];
      D->height_edge_values[k3 + 1] = edge_values[1];
      D->height_edge_values[k3 + 2] = edge_values[2]; 

    
      //-----------------------------------
      // xmomentum
      //-----------------------------------

      beta_tmp = D->beta_uh_dry + (D->beta_uh - D->beta_uh_dry) * hfactor;

      cv_k  = D->xmom_centroid_values[k];
      cv_k0 = D->xmom_centroid_values[k0];
      cv_k1 = D->xmom_centroid_values[k1];
      cv_k2 = D->xmom_centroid_values[k2];

      __calc_edge_values(beta_tmp, 
                         cv_k, 
                         cv_k0,
                         cv_k1,
                         cv_k2,
                         dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                         dx1, dx2, dy1, dy2, inv_area2, edge_values);

      D->xmom_edge_values[k3 + 0] = edge_values[0];
      D->xmom_edge_values[k3 + 1] = edge_values[1];
      D->xmom_edge_values[k3 + 2] = edge_values[2]; 

      //-----------------------------------
      // ymomentum
      //-----------------------------------

      beta_tmp = D->beta_vh_dry + (D->beta_vh - D->beta_vh_dry) * hfactor;

      cv_k  = D->ymom_centroid_values[k];
      cv_k0 = D->ymom_centroid_values[k0];
      cv_k1 = D->ymom_centroid_values[k1];
      cv_k2 = D->ymom_centroid_values[k2];

      __calc_edge_values(beta_tmp, 
                         cv_k, 
                         cv_k0,
                         cv_k1,
                         cv_k2,
                         dxv0, dxv1, dxv2, dyv0, dyv1, dyv2,
                         dx1, dx2, dy1, dy2, inv_area2, edge_values);

      D->ymom_edge_values[k3 + 0] = edge_values[0];
      D->ymom_edge_values[k3 + 1] = edge_values[1];
      D->ymom_edge_values[k3 + 2] = edge_values[2]; 

    } // End number_of_boundaries <=1
    else
    {
      //printf("%ld 2 boundaries\n",k);
      //==============================================
      // Number of boundaries == 2
      //==============================================

      // One internal neighbour and gradient is in direction of the neighbour's centroid

      // Find the only internal neighbour (k1?)
      for (k2 = k3; k2 < k3 + 3; k2++)
      {
        // Find internal neighbour of triangle k
        // k2 indexes the edges of triangle k

        if (D->surrogate_neighbours[k2] != k)
        {
          break;
        }
      }

      // if ((k2 == k3 + 3))
      // {
      //   // If we didn't find an internal neighbour
      //   // report_python_error(AT, "Internal neighbour not found");
      //   return -1;
      // }

      k1 = D->surrogate_neighbours[k2];

      // The coordinates of the triangle are already (x,y).
      // Get centroid of the neighbour (x1,y1)
      coord_index = 2 * k1;
      x1 = D->centroid_coordinates[coord_index + 0];
      y1 = D->centroid_coordinates[coord_index + 1];

      // Compute x- and y- distances between the centroid of
      // triangle k and that of its neighbour
      dx1 = x1 - x;
      dy1 = y1 - y;

      // Set area2 as the square of the distance
      area2 = dx1 * dx1 + dy1 * dy1;

      // Set dx2=(x1-x0)/((x1-x0)^2+(y1-y0)^2)
      // and dy2=(y1-y0)/((x1-x0)^2+(y1-y0)^2) which
      // respectively correspond to the x- and y- gradients
      // of the conserved quantities
      dx2 = 1.0 / area2;
      dy2 = dx2 * dy1;
      dx2 *= dx1;

      //-----------------------------------
      // stage
      //-----------------------------------

      // Compute differentials
      dq1 = D->stage_centroid_values[k1] - D->stage_centroid_values[k];

      // Calculate the gradient between the centroid of triangle k
      // and that of its neighbour
      a = dq1 * dx2;
      b = dq1 * dy2;

      // Calculate provisional edge jumps, to be limited
      dqv[0] = a * dxv0 + b * dyv0;
      dqv[1] = a * dxv1 + b * dyv1;
      dqv[2] = a * dxv2 + b * dyv2;

      // Now limit the jumps
      if (dq1 >= 0.0)
      {
        qmin = 0.0;
        qmax = dq1;
      }
      else
      {
        qmin = dq1;
        qmax = 0.0;
      }

      // Limit the gradient
      __limit_gradient(dqv, qmin, qmax, D->beta_w);

      D->stage_edge_values[k3 + 0] = D->stage_centroid_values[k] + dqv[0];
      D->stage_edge_values[k3 + 1] = D->stage_centroid_values[k] + dqv[1];
      D->stage_edge_values[k3 + 2] = D->stage_centroid_values[k] + dqv[2];

      //-----------------------------------
      // height
      //-----------------------------------

      // Compute differentials
      dq1 = D->height_centroid_values[k1] - D->height_centroid_values[k];

      // Calculate the gradient between the centroid of triangle k
      // and that of its neighbour
      a = dq1 * dx2;
      b = dq1 * dy2;

      // Calculate provisional edge jumps, to be limited
      dqv[0] = a * dxv0 + b * dyv0;
      dqv[1] = a * dxv1 + b * dyv1;
      dqv[2] = a * dxv2 + b * dyv2;

      // Now limit the jumps
      if (dq1 >= 0.0)
      {
        qmin = 0.0;
        qmax = dq1;
      }
      else
      {
        qmin = dq1;
        qmax = 0.0;
      }

      // Limit the gradient
      __limit_gradient(dqv, qmin, qmax, D->beta_w);

      D->height_edge_values[k3 + 0] = D->height_centroid_values[k] + dqv[0];
      D->height_edge_values[k3 + 1] = D->height_centroid_values[k] + dqv[1];
      D->height_edge_values[k3 + 2] = D->height_centroid_values[k] + dqv[2];

      //-----------------------------------
      // xmomentum
      //-----------------------------------

      // Compute differentials
      dq1 = D->xmom_centroid_values[k1] - D->xmom_centroid_values[k];

      // Calculate the gradient between the centroid of triangle k
      // and that of its neighbour
      a = dq1 * dx2;
      b = dq1 * dy2;

      // Calculate provisional edge jumps, to be limited
      dqv[0] = a * dxv0 + b * dyv0;
      dqv[1] = a * dxv1 + b * dyv1;
      dqv[2] = a * dxv2 + b * dyv2;

      // Now limit the jumps
      if (dq1 >= 0.0)
      {
        qmin = 0.0;
        qmax = dq1;
      }
      else
      {
        qmin = dq1;
        qmax = 0.0;
      }

      // Limit the gradient
      __limit_gradient(dqv, qmin, qmax, D->beta_w);

      D->xmom_edge_values[k3 + 0] = D->xmom_centroid_values[k] + dqv[0];
      D->xmom_edge_values[k3 + 1] = D->xmom_centroid_values[k] + dqv[1];
      D->xmom_edge_values[k3 + 2] = D->xmom_centroid_values[k] + dqv[2];

      //-----------------------------------
      // ymomentum
      //-----------------------------------

      // Compute differentials
      dq1 = D->ymom_centroid_values[k1] - D->ymom_centroid_values[k];

      // Calculate the gradient between the centroid of triangle k
      // and that of its neighbour
      a = dq1 * dx2;
      b = dq1 * dy2;

      // Calculate provisional edge jumps, to be limited
      dqv[0] = a * dxv0 + b * dyv0;
      dqv[1] = a * dxv1 + b * dyv1;
      dqv[2] = a * dxv2 + b * dyv2;

      // Now limit the jumps
      if (dq1 >= 0.0)
      {
        qmin = 0.0;
        qmax = dq1;
      }
      else
      {
        qmin = dq1;
        qmax = 0.0;
      }

      // Limit the gradient
      __limit_gradient(dqv, qmin, qmax, D->beta_w);

      D->ymom_edge_values[k3 + 0] = D->ymom_centroid_values[k] + dqv[0];
      D->ymom_edge_values[k3 + 1] = D->ymom_centroid_values[k] + dqv[1];
      D->ymom_edge_values[k3 + 2] = D->ymom_centroid_values[k] + dqv[2];

    } // else [number_of_boundaries]

  // printf("%ld, bed    %e, %e, %e\n",k, D->bed_edge_values[k3],D->bed_edge_values[k3 + 1],D->bed_edge_values[k3 + 2] );
  // printf("%ld, stage  %e, %e, %e\n",k, D->stage_edge_values[k3],D->stage_edge_values[k3 + 1],D->stage_edge_values[k3 + 2] );
  // printf("%ld, height %e, %e, %e\n",k, D->height_edge_values[k3],D->height_edge_values[k3 + 1],D->height_edge_values[k3 + 2] );
  // printf("%ld, xmom   %e, %e, %e\n",k, D->xmom_edge_values[k3],D->xmom_edge_values[k3 + 1],D->xmom_edge_values[k3 + 2] );
  // printf("%ld, ymom   %e, %e, %e\n",k, D->ymom_edge_values[k3],D->ymom_edge_values[k3 + 1],D->ymom_edge_values[k3 + 2] );

    // If needed, convert from velocity to momenta
    if (D->extrapolate_velocity_second_order == 1)
    {
      // Re-compute momenta at edges
      for (i = 0; i < 3; i++)
      {
        dk = D->height_edge_values[k3 + i];
        D->xmom_edge_values[k3 + i] = D->xmom_edge_values[k3 + i] * dk;
        D->ymom_edge_values[k3 + i] = D->ymom_edge_values[k3 + i] * dk;
      }
    }

    // Compute new bed elevation
    D->bed_edge_values[k3 + 0] = D->stage_edge_values[k3 + 0] - D->height_edge_values[k3 + 0];
    D->bed_edge_values[k3 + 1] = D->stage_edge_values[k3 + 1] - D->height_edge_values[k3 + 1];
    D->bed_edge_values[k3 + 2] = D->stage_edge_values[k3 + 2] - D->height_edge_values[k3 + 2];


    // FIXME SR: Do we need vertex values every inner timestep?

    // Compute stage vertex values
    D->stage_vertex_values[k3 + 0] = D->stage_edge_values[k3 + 1] + D->stage_edge_values[k3 + 2] - D->stage_edge_values[k3 + 0];
    D->stage_vertex_values[k3 + 1] = D->stage_edge_values[k3 + 0] + D->stage_edge_values[k3 + 2] - D->stage_edge_values[k3 + 1];
    D->stage_vertex_values[k3 + 2] = D->stage_edge_values[k3 + 0] + D->stage_edge_values[k3 + 1] - D->stage_edge_values[k3 + 2];

    // Compute height vertex values
    D->height_vertex_values[k3 + 0] = D->height_edge_values[k3 + 1] + D->height_edge_values[k3 + 2] - D->height_edge_values[k3 + 0];
    D->height_vertex_values[k3 + 1] = D->height_edge_values[k3 + 0] + D->height_edge_values[k3 + 2] - D->height_edge_values[k3 + 1];
    D->height_vertex_values[k3 + 2] = D->height_edge_values[k3 + 0] + D->height_edge_values[k3 + 1] - D->height_edge_values[k3 + 2];

    // Compute momenta at vertices
    D->xmom_vertex_values[k3 + 0] = D->xmom_edge_values[k3 + 1] + D->xmom_edge_values[k3 + 2] - D->xmom_edge_values[k3 + 0];
    D->xmom_vertex_values[k3 + 1] = D->xmom_edge_values[k3 + 0] + D->xmom_edge_values[k3 + 2] - D->xmom_edge_values[k3 + 1];
    D->xmom_vertex_values[k3 + 2] = D->xmom_edge_values[k3 + 0] + D->xmom_edge_values[k3 + 1] - D->xmom_edge_values[k3 + 2];

    D->ymom_vertex_values[k3 + 0] = D->ymom_edge_values[k3 + 1] + D->ymom_edge_values[k3 + 2] - D->ymom_edge_values[k3 + 0];
    D->ymom_vertex_values[k3 + 1] = D->ymom_edge_values[k3 + 0] + D->ymom_edge_values[k3 + 2] - D->ymom_edge_values[k3 + 1];
    D->ymom_vertex_values[k3 + 2] = D->ymom_edge_values[k3 + 0] + D->ymom_edge_values[k3 + 1] - D->ymom_edge_values[k3 + 2];

    
    D->bed_vertex_values[k3 + 0] = D->bed_edge_values[k3 + 1] + D->bed_edge_values[k3 + 2] - D->bed_edge_values[k3 + 0];
    D->bed_vertex_values[k3 + 1] = D->bed_edge_values[k3 + 0] + D->bed_edge_values[k3 + 2] - D->bed_edge_values[k3 + 1];
    D->bed_vertex_values[k3 + 2] = D->bed_edge_values[k3 + 0] + D->bed_edge_values[k3 + 1] - D->bed_edge_values[k3 + 2];



  }   // for k=0 to number_of_elements-1

// Fix xmom and ymom centroid values
// #pragma omp parallel for simd private(k3, i, dk) firstprivate(extrapolate_velocity_second_order)
  for (k = 0; k < D->number_of_elements; k++)
  {
    if (extrapolate_velocity_second_order == 1)
    {
      // Convert velocity back to momenta at centroids
      D->xmom_centroid_values[k] = D->x_centroid_work[k];
      D->ymom_centroid_values[k] = D->y_centroid_work[k];
    }

    // // Compute stage vertex values
    // D->stage_vertex_values[k3] = D->stage_edge_values[k3 + 1] + D->stage_edge_values[k3 + 2] - D->stage_edge_values[k3];
    // D->stage_vertex_values[k3 + 1] = D->stage_edge_values[k3] + D->stage_edge_values[k3 + 2] - D->stage_edge_values[k3 + 1];
    // D->stage_vertex_values[k3 + 2] = D->stage_edge_values[k3] + D->stage_edge_values[k3 + 1] - D->stage_edge_values[k3 + 2];

    // // Compute height vertex values
    // D->height_vertex_values[k3] = D->height_edge_values[k3 + 1] + D->height_edge_values[k3 + 2] - D->height_edge_values[k3];
    // D->height_vertex_values[k3 + 1] = D->height_edge_values[k3] + D->height_edge_values[k3 + 2] - D->height_edge_values[k3 + 1];
    // D->height_vertex_values[k3 + 2] = D->height_edge_values[k3] + D->height_edge_values[k3 + 1] - D->height_edge_values[k3 + 2];

    // // If needed, convert from velocity to momenta
    // if (D->extrapolate_velocity_second_order == 1)
    // {
    //   // Re-compute momenta at edges
    //   for (i = 0; i < 3; i++)
    //   {
    //     dk = D->height_edge_values[k3 + i];
    //     D->xmom_edge_values[k3 + i] = D->xmom_edge_values[k3 + i] * dk;
    //     D->ymom_edge_values[k3 + i] = D->ymom_edge_values[k3 + i] * dk;
    //   }
    // }

    // // Compute momenta at vertices
    // D->xmom_vertex_values[k3 + 0] = D->xmom_edge_values[k3 + 1] + D->xmom_edge_values[k3 + 2] - D->xmom_edge_values[k3 + 0];
    // D->xmom_vertex_values[k3 + 1] = D->xmom_edge_values[k3 + 0] + D->xmom_edge_values[k3 + 2] - D->xmom_edge_values[k3 + 1];
    // D->xmom_vertex_values[k3 + 2] = D->xmom_edge_values[k3 + 0] + D->xmom_edge_values[k3 + 1] - D->xmom_edge_values[k3 + 2];

    // D->ymom_vertex_values[k3 + 0] = D->ymom_edge_values[k3 + 1] + D->ymom_edge_values[k3 + 2] - D->ymom_edge_values[k3 + 0];
    // D->ymom_vertex_values[k3 + 1] = D->ymom_edge_values[k3 + 0] + D->ymom_edge_values[k3 + 2] - D->ymom_edge_values[k3 + 1];
    // D->ymom_vertex_values[k3 + 2] = D->ymom_edge_values[k3 + 0] + D->ymom_edge_values[k3 + 1] - D->ymom_edge_values[k3 + 2];

    // // Compute new bed elevation
    // D->bed_edge_values[k3 + 0] = D->stage_edge_values[k3 + 0] - D->height_edge_values[k3 + 0];
    // D->bed_edge_values[k3 + 1] = D->stage_edge_values[k3 + 1] - D->height_edge_values[k3 + 1];
    // D->bed_edge_values[k3 + 2] = D->stage_edge_values[k3 + 2] - D->height_edge_values[k3 + 2];

    // D->bed_vertex_values[k3 + 0] = D->bed_edge_values[k3 + 1] + D->bed_edge_values[k3 + 2] - D->bed_edge_values[k3 + 0];
    // D->bed_vertex_values[k3 + 1] = D->bed_edge_values[k3 + 0] + D->bed_edge_values[k3 + 2] - D->bed_edge_values[k3 + 1];
    // D->bed_vertex_values[k3 + 2] = D->bed_edge_values[k3 + 0] + D->bed_edge_values[k3 + 1] - D->bed_edge_values[k3 + 2];

  }

  return 0;
}


// Computational function for flux computation
int64_t _openacc_fix_negative_cells(struct domain *D)
{
  int64_t k;
  int64_t tff;
  int64_t num_negative_cells = 0;

  // #pragma omp parallel for private(k, tff) reduction(+:num_negative_cells)
  for (k = 0; k < D->number_of_elements; k++)
  {
    tff = D->tri_full_flag[k];
    if ((D->stage_centroid_values[k] - D->bed_centroid_values[k] < 0.0) & (tff > 0)) 
    {
      num_negative_cells = num_negative_cells + 1;
      D->stage_centroid_values[k] = D->bed_centroid_values[k];
      D->xmom_centroid_values[k] = 0.0;
      D->ymom_centroid_values[k] = 0.0;
    }
  }
  return num_negative_cells;
}
