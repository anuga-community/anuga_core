// C struct for domain and quantities
//
// Stephen Roberts 2012



#ifndef SW_DOMAIN_H
#define SW_DOMAIN_H

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <inttypes.h>
#include "anuga_typedefs.h"

// structures
struct domain {
    // Changing these don't change the data in python object
    anuga_int    number_of_elements;
    anuga_int    boundary_length;
    anuga_int    number_of_riverwall_edges;
    double     epsilon;
    double     H0;
    double     g;
    anuga_int    optimise_dry_cells;
    double     evolve_max_timestep;
    anuga_int    extrapolate_velocity_second_order;
    double     minimum_allowed_height;
    double     maximum_allowed_speed;
    anuga_int    low_froude;


    anuga_int timestep_fluxcalls;

    double beta_w;
    double beta_w_dry;
    double beta_uh;
    double beta_uh_dry;
    double beta_vh;
    double beta_vh_dry;

    anuga_int max_flux_update_frequency;
    anuga_int ncol_riverwall_hydraulic_properties;

    // Changing values in these arrays will change the values in the python object
    anuga_int*   neighbours;
    anuga_int*   neighbour_edges;
    anuga_int*   surrogate_neighbours;
    double*    normals;
    double*    edgelengths;
    double*    radii;
    double*    areas;

    anuga_int*   edge_flux_type;

    anuga_int*   tri_full_flag;
    anuga_int*   already_computed_flux;
    double*    max_speed;

    double* vertex_coordinates;
    double* edge_coordinates;
    double* centroid_coordinates;

    anuga_int* number_of_boundaries;
    double* stage_edge_values;
    double* xmom_edge_values;
    double* ymom_edge_values;
    double* bed_edge_values;
    double* height_edge_values;
    double* xvelocity_edge_values;
    double* yvelocity_edge_values;

    double* stage_centroid_values;
    double* xmom_centroid_values;
    double* ymom_centroid_values;
    double* bed_centroid_values;
    double* height_centroid_values;

    double* stage_vertex_values;
    double* xmom_vertex_values;
    double* ymom_vertex_values;
    double* bed_vertex_values;
    double* height_vertex_values;


    double* stage_boundary_values;
    double* xmom_boundary_values;
    double* ymom_boundary_values;
    double* bed_boundary_values;
    double* height_boundary_values;
    double* xvelocity_boundary_values;
    double* yvelocity_boundary_values;

    double* stage_explicit_update;
    double* xmom_explicit_update;
    double* ymom_explicit_update;

    anuga_int* flux_update_frequency;
    anuga_int* update_next_flux;
    anuga_int* update_extrapolation;
    double* edge_timestep;
    double* edge_flux_work;
    double* neigh_work;
    double* pressuregrad_work;
    double* x_centroid_work;
    double* y_centroid_work;
    double* boundary_flux_sum;

    anuga_int* allow_timestep_increase;

    anuga_int* edge_river_wall_counter;
    double* riverwall_elevation;
    anuga_int* riverwall_rowIndex;
    double* riverwall_hydraulic_properties;

    double* stage_semi_implicit_update;
    double* xmom_semi_implicit_update;
    double* ymom_semi_implicit_update; 
    
    double* friction_centroid_values;

    double* stage_backup_values;
    double* xmom_backup_values;
    double* ymom_backup_values;

};


struct edge {

    anuga_int cell_id;
    anuga_int edge_id;

    // mid point values
    double w;
    double h;
    double z;
    double uh;
    double vh;
    double u;
    double v;

    // vertex values
    double w1;
    double h1;
    double z1;
    double uh1;
    double vh1;
    double u1;
    double v1;

    double w2;
    double h2;
    double z2;
    double uh2;
    double vh2;
    double u2;
    double v2;

};


void get_edge_data(struct edge *E, struct domain *D, anuga_int k, anuga_int i) {
    // fill edge data (conserved and bed) for ith edge of kth triangle

    anuga_int k3i, k3i1, k3i2;

    k3i = 3 * k + i;
    k3i1 = 3 * k + (i + 1) % 3;
    k3i2 = 3 * k + (i + 2) % 3;

    E->cell_id = k;
    E->edge_id = i;

    E->w = D->stage_edge_values[k3i];
    E->z = D->bed_edge_values[k3i];
    E->h = E->w - E->z;
    E->uh = D->xmom_edge_values[k3i];
    E->vh = D->ymom_edge_values[k3i];

    E->w1 = D->stage_vertex_values[k3i1];
    E->z1 = D->bed_vertex_values[k3i1];
    E->h1 = E->w1 - E->z1;
    E->uh1 = D->xmom_vertex_values[k3i1];
    E->vh1 = D->ymom_vertex_values[k3i1];


    E->w2 = D->stage_vertex_values[k3i2];
    E->z2 = D->bed_vertex_values[k3i2];
    E->h2 = E->w2 - E->z2;
    E->uh2 = D->xmom_vertex_values[k3i2];
    E->vh2 = D->ymom_vertex_values[k3i2];

}

anuga_int print_domain_struct(struct domain *D) {


    printf("D->number_of_elements     %" PRId64 "  \n", D->number_of_elements);
    printf("D->boundary_length        %" PRId64 "  \n", D->boundary_length);
    printf("D->number_of_riverwall_edges %" PRId64 "  \n", D->number_of_riverwall_edges);
    printf("D->epsilon                %g \n", D->epsilon);
    printf("D->H0                     %g \n", D->H0);
    printf("D->g                      %g \n", D->g);
    printf("D->optimise_dry_cells     %" PRId64 " \n", D->optimise_dry_cells);
    printf("D->evolve_max_timestep    %g \n", D->evolve_max_timestep);
    printf("D->minimum_allowed_height %g \n", D->minimum_allowed_height);
    printf("D->maximum_allowed_speed  %g \n", D->maximum_allowed_speed);
    printf("D->low_froude             %" PRId64 " \n", D->low_froude);
    printf("D->extrapolate_velocity_second_order %" PRId64 " \n", D->extrapolate_velocity_second_order);
    printf("D->beta_w                 %g \n", D->beta_w);
    printf("D->beta_w_dry             %g \n", D->beta_w_dry);
    printf("D->beta_uh                %g \n", D->beta_uh);
    printf("D->beta_uh_dry            %g \n", D->beta_uh_dry);
    printf("D->beta_vh                %g \n", D->beta_vh);
    printf("D->beta_vh_dry            %g \n", D->beta_vh_dry);



    printf("D->neighbours             %p \n", (void *) D->neighbours);
    printf("D->surrogate_neighbours   %p \n", (void *) D->surrogate_neighbours);
    printf("D->neighbour_edges        %p \n", (void *) D->neighbour_edges);
    printf("D->normals                %p \n", (void *) D->normals);
    printf("D->edgelengths            %p \n", (void *) D->edgelengths);
    printf("D->radii                  %p \n", (void *) D->radii);
    printf("D->areas                  %p \n", (void *) D->areas);
    printf("D->tri_full_flag          %p \n", (void *) D->tri_full_flag);
    printf("D->already_computed_flux  %p \n", (void *) D->already_computed_flux);
    printf("D->vertex_coordinates     %p \n", (void *) D->vertex_coordinates);
    printf("D->edge_coordinates       %p \n", (void *) D->edge_coordinates);
    printf("D->centroid_coordinates   %p \n", (void *) D->centroid_coordinates);
    printf("D->max_speed              %p \n", (void *) D->max_speed);
    printf("D->number_of_boundaries   %p \n", (void *) D->number_of_boundaries);
    printf("D->stage_edge_values      %p \n", (void *) D->stage_edge_values);
    printf("D->xmom_edge_values       %p \n", (void *) D->xmom_edge_values);
    printf("D->ymom_edge_values       %p \n", (void *) D->ymom_edge_values);
    printf("D->bed_edge_values        %p \n", (void *) D->bed_edge_values);
    printf("D->stage_centroid_values  %p \n", (void *) D->stage_centroid_values);
    printf("D->xmom_centroid_values   %p \n", (void *) D->xmom_centroid_values);
    printf("D->ymom_centroid_values   %p \n", (void *) D->ymom_centroid_values);
    printf("D->bed_centroid_values    %p \n", (void *) D->bed_centroid_values);
    printf("D->stage_vertex_values    %p \n", (void *) D->stage_vertex_values);
    printf("D->xmom_vertex_values     %p \n", (void *) D->xmom_vertex_values);
    printf("D->ymom_vertex_values     %p \n", (void *) D->ymom_vertex_values);
    printf("D->bed_vertex_values      %p \n", (void *) D->bed_vertex_values);
    printf("D->height_vertex_values      %p \n", (void *) D->height_vertex_values);
    printf("D->stage_boundary_values  %p \n", (void *) D->stage_boundary_values);
    printf("D->xmom_boundary_values   %p \n", (void *) D->xmom_boundary_values);
    printf("D->ymom_boundary_values   %p \n", (void *) D->ymom_boundary_values);
    printf("D->bed_boundary_values    %p \n", (void *) D->bed_boundary_values);
    printf("D->stage_explicit_update  %p \n", (void *) D->stage_explicit_update);
    printf("D->xmom_explicit_update   %p \n", (void *) D->xmom_explicit_update);
    printf("D->ymom_explicit_update   %p \n", (void *) D->ymom_explicit_update);
    printf("D->edge_river_wall_counter   %p \n",   (void *) D->edge_river_wall_counter);
    printf("D->stage_semi_implicit_update  %p \n", (void *) D->stage_semi_implicit_update);
    printf("D->xmom_semi_implicit_update   %p \n", (void *) D->xmom_semi_implicit_update);
    printf("D->ymom_semi_implicit_update   %p \n", (void *) D->ymom_semi_implicit_update);
    printf("D->friction_centroid_values   %p \n", (void *) D->friction_centroid_values);


    return 0;
}


typedef struct {
    double ql[3], qr[3];
    double zl, zr;
    double hle, hre;
    double h_left, h_right;
    double hc, zc, hc_n, zc_n;
    double z_half;
    double normal_x, normal_y;
    double length;
    int n; // neighbour index
    int ki, ki2;
    bool is_boundary;
    bool is_riverwall;
    int riverwall_index;
} EdgeData;

// Extract edge-related data and organize it into EdgeData
static inline void get_edge_data_central_flux(const struct domain * __restrict D, const int k, const int i, EdgeData * __restrict E) {
    E->ki = 3 * k + i;
    E->ki2 = 2 * E->ki;

    E->ql[0] = D->stage_edge_values[E->ki];
    E->ql[1] = D->xmom_edge_values[E->ki];
    E->ql[2] = D->ymom_edge_values[E->ki];
    E->zl = D->bed_edge_values[E->ki];
    E->hle = D->height_edge_values[E->ki];
    E->length = D->edgelengths[E->ki];

    E->n = D->neighbours[E->ki];
    E->is_boundary = (E->n < 0);
    E->normal_x = D->normals[E->ki2];
    E->normal_y = D->normals[E->ki2 + 1];

    E->hc = D->height_centroid_values[k];
    E->zc = D->bed_centroid_values[k];
    E->hc_n=E->hc;
    E->zc_n=D->bed_centroid_values[k];

    if (E->is_boundary) {
        int m = -E->n - 1;
        E->qr[0] = D->stage_boundary_values[m];
        E->qr[1] = D->xmom_boundary_values[m];
        E->qr[2] = D->ymom_boundary_values[m];
        E->zr = E->zl;
        E->hre = fmax(E->qr[0] - E->zr, 0.0);
    } else {
        E->hc_n = D->height_centroid_values[E->n];
        E->zc_n = D->bed_centroid_values[E->n];
        int m = D->neighbour_edges[E->ki];
        int nm = E->n * 3 + m;
        E->qr[0] = D->stage_edge_values[nm];
        E->qr[1] = D->xmom_edge_values[nm];
        E->qr[2] = D->ymom_edge_values[nm];
        E->zr = D->bed_edge_values[nm];
        E->hre = D->height_edge_values[nm];
    }

    E->z_half = fmax(E->zl, E->zr);

    // Check for riverwall elevation override
    E->is_riverwall = (D->edge_flux_type[E->ki] == 1);
    if (E->is_riverwall) {
        E->riverwall_index = D->edge_river_wall_counter[E->ki] - 1;
        double zwall = D->riverwall_elevation[E->riverwall_index];
        E->z_half = fmax(zwall, E->z_half);
    }

    E->h_left = fmax(E->hle + E->zl - E->z_half, 0.0);
    E->h_right = fmax(E->hre + E->zr - E->z_half, 0.0);
}


#endif
