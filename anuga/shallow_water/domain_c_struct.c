#include "sw_domain.h"

#include <stdlib.h>   // for malloc, free
#include <string.h>   // for memcpy

void say_hi() {
    printf("Hello from C domain!\n");
}   

void init_c_domain(struct domain* D, anuga_int number_of_elements, anuga_int boundary_length) {
    // Initialize the domain structure with the given number of elements and boundary length
    // All of our arrays are in function of these two parameters 
    D->number_of_elements = number_of_elements;
    D->boundary_length = boundary_length;
    


    // for sanity, let's declare the common sizes 
    size_t n = number_of_elements;
    size_t b = boundary_length;
    size_t n3 = 3 * number_of_elements;
    size_t n6 = 6 * number_of_elements;



    // fyi, this is not good code. This will be called every time we create a domain, so it is BAD but right now 
    // I need a working case that will not crash, so I will leave it like this for now.
    // this, should be moved to the start of the simulation so that the allocation happens ONCE 
    

    // domain structures 
    // these follow the order established in sw_domain_openmp_ext.pyx
    D->neighbours = (anuga_int*)malloc(n3 * sizeof(anuga_int));
    D->surrogate_neighbours = (anuga_int*)malloc(n3 * sizeof(anuga_int));
    D->neighbour_edges = (anuga_int*)malloc(n3 * sizeof(anuga_int));
    D->normals = (double*)malloc(n6 * sizeof(double));
    D->edgelengths = (double*)malloc(n3 * sizeof(double));
    D->radii = (double*)malloc(n * sizeof(double));
    D->areas = (double*)malloc(n * sizeof(double));
    D->edge_flux_type = (anuga_int*)malloc(n3 * sizeof(anuga_int));
    D->tri_full_flag = (anuga_int*)malloc(n * sizeof(anuga_int));
    D->already_computed_flux = (anuga_int*)malloc(n3 * sizeof(anuga_int));
    D->vertex_coordinates = (double*)malloc(n6 * sizeof(double));
    D->edge_coordinates = (double*)malloc(n6 * sizeof(double));
    D->centroid_coordinates = (double*)malloc(2*n * sizeof(double));
    D->max_speed = (double*)malloc(n * sizeof(double));
    D->number_of_boundaries = (anuga_int*)malloc(b * sizeof(anuga_int));
    D->flux_update_frequency = (anuga_int*)malloc(n3 * sizeof(anuga_int));
    D->update_next_flux = (anuga_int*)malloc(n3 * sizeof(anuga_int));
    D->update_extrapolation = (anuga_int*)malloc(n*sizeof(anuga_int));
    D->allow_timestep_increase = (anuga_int*)malloc(1*sizeof(anuga_int));
    D->edge_timestep = (double*)malloc(n3 * sizeof(double));
    D->edge_flux_work = (double*)malloc(9*n * sizeof(double));
    D->neigh_work = (double*)malloc(9*n * sizeof(double));
    D->pressuregrad_work = (double*)malloc(3*n * sizeof(double));
    D->x_centroid_work = (double*)malloc(n * sizeof(double));
    D->y_centroid_work = (double*)malloc(n * sizeof(double));
    D->boundary_flux_sum = (double*)malloc(3 * sizeof(double));
    D->edge_river_wall_counter = (anuga_int*)malloc(n3 * sizeof(anuga_int));

    // quantity structures 

    D->stage_edge_values = (double*)malloc(n3 * sizeof(double));
    D->xmom_edge_values = (double*)malloc(n3 * sizeof(double));
    D->ymom_edge_values = (double*)malloc(n3 * sizeof(double));
    D->bed_edge_values = (double*)malloc(n3 * sizeof(double));
    D->height_edge_values = (double*)malloc(n3 * sizeof(double));
    D->xvelocity_edge_values = (double*)malloc(n3 * sizeof(double));
    D->yvelocity_edge_values = (double*)malloc(n3 * sizeof(double));
    D->stage_centroid_values = (double*)malloc(n * sizeof(double));
    D->xmom_centroid_values = (double*)malloc(n * sizeof(double));
    D->ymom_centroid_values = (double*)malloc(n * sizeof(double));
    D->bed_centroid_values = (double*)malloc(n * sizeof(double));
    D->height_centroid_values = (double*)malloc(n * sizeof(double));
    D->friction_centroid_values = (double*)malloc(n * sizeof(double));
    D->stage_backup_values = (double*)malloc(n * sizeof(double));
    D->xmom_backup_values = (double*)malloc(n * sizeof(double));
    D->ymom_backup_values = (double*)malloc(n * sizeof(double));


    // vertex values
    D->stage_vertex_values = (double*)malloc(n3 * sizeof(double));
    D->xmom_vertex_values = (double*)malloc(n3 * sizeof(double));
    D->ymom_vertex_values = (double*)malloc(n3 * sizeof(double));
    D->bed_vertex_values = (double*)malloc(n3 * sizeof(double));
    D->height_vertex_values = (double*)malloc(n3 * sizeof(double));

    // boundary values 
    D->stage_boundary_values = (double*)malloc(b * sizeof(double));
    D->xmom_boundary_values = (double*)malloc(b * sizeof(double));
    D->ymom_boundary_values = (double*)malloc(b * sizeof(double));
    D->bed_boundary_values = (double*)malloc(b * sizeof(double));
    D->height_boundary_values = (double*)malloc(b * sizeof(double));
    D->xvelocity_boundary_values = (double*)malloc(b * sizeof(double));
    D->yvelocity_boundary_values = (double*)malloc(b * sizeof(double));

    // explicit and semi-implicit updates
    D->stage_explicit_update = (double*)malloc(n * sizeof(double));
    D->xmom_explicit_update = (double*)malloc(n * sizeof(double));
    D->ymom_explicit_update = (double*)malloc(n * sizeof(double));
    D->stage_semi_implicit_update = (double*)malloc(n * sizeof(double));
    D->xmom_semi_implicit_update = (double*)malloc(n * sizeof(double));
    D->ymom_semi_implicit_update = (double*)malloc(n * sizeof(double));

    // river wall structures
    D->riverwall_elevation = (double*)malloc(1 * sizeof(double));
    D->riverwall_rowIndex = (anuga_int*)malloc(1 * sizeof(anuga_int));
    D->riverwall_hydraulic_properties = (double*)malloc(1 * sizeof(double));

    D->is_c_domain = true; // Flag to indicate this is a C domain structure
    D->is_initialised = true; // Flag to indicate the domain has been initialised
//     printf("Domain initialised with %d elements and %d boundaries.\n", 
//            number_of_elements, boundary_length);

}


void free_c_domain(struct domain* D) {
    // Free the allocated memory for arrays in the domain structure
    // I've verified that all arrays are freed here, so this should not cause memory leaks
    // HOWEVER, if you add new arrays, please make sure to free them here as well.
    // jorge: there should be a better way to do this....maybe C++? haha
    if(!D->is_initialised) {
        // If the domain is not initialised, there's nothing to free
        return;
    }

    //printf("Freeing domain memory...\n");
    free(D->neighbours);
    free(D->surrogate_neighbours);
    free(D->neighbour_edges);
    free(D->normals);
    free(D->edgelengths);
    free(D->radii);
    free(D->areas);
    free(D->edge_flux_type);
    free(D->tri_full_flag);
    free(D->already_computed_flux);
    free(D->vertex_coordinates);
    free(D->edge_coordinates);
    free(D->centroid_coordinates);
    free(D->max_speed);
    free(D->number_of_boundaries);
    free(D->flux_update_frequency);
    free(D->update_next_flux);
    free(D->update_extrapolation);
    free(D->allow_timestep_increase);
    free(D->edge_timestep);
    free(D->edge_flux_work);
    free(D->neigh_work);
    free(D->pressuregrad_work);
    free(D->x_centroid_work);
    free(D->y_centroid_work);
    free(D->boundary_flux_sum);
    free(D->edge_river_wall_counter);
    free(D->stage_edge_values);
    free(D->xmom_edge_values);
    free(D->ymom_edge_values);
    free(D->bed_edge_values);
    free(D->height_edge_values);
    free(D->xvelocity_edge_values);
    free(D->yvelocity_edge_values);
    free(D->stage_centroid_values);
    free(D->xmom_centroid_values);
    free(D->ymom_centroid_values);
    free(D->bed_centroid_values);
    free(D->height_centroid_values);
    free(D->friction_centroid_values);
    free(D->stage_backup_values);
    free(D->xmom_backup_values);
    free(D->ymom_backup_values);
    free(D->stage_vertex_values);
    free(D->xmom_vertex_values);
    free(D->ymom_vertex_values);
    free(D->bed_vertex_values);
    free(D->height_vertex_values);
    free(D->stage_boundary_values);
    free(D->xmom_boundary_values);
    free(D->ymom_boundary_values);
    free(D->bed_boundary_values);
    free(D->height_boundary_values);
    free(D->xvelocity_boundary_values);
    free(D->yvelocity_boundary_values);
    free(D->stage_explicit_update);
    free(D->xmom_explicit_update);
    free(D->ymom_explicit_update);
    free(D->stage_semi_implicit_update);
    free(D->xmom_semi_implicit_update);
    free(D->ymom_semi_implicit_update);
    free(D->riverwall_elevation);
    free(D->riverwall_rowIndex);
    free(D->riverwall_hydraulic_properties);

    D->is_initialised = false; // Set the initialised flag to false after freeing memory
    //printf("Domain memory freed successfully.\n");

}


void copy_c_domain(struct domain* D, struct domain* source) {

    if(!D->is_initialised) {
        printf("Error: Target Domain is not initialised. Cannot copy.\n");
        return;
    }

    //printf("Copying domain from source to destination...\n");

    D->number_of_elements = source->number_of_elements;
    D->boundary_length = source->boundary_length;
    D->number_of_riverwall_edges = source->number_of_riverwall_edges;
    D->epsilon = source->epsilon;
    D->g = source->g;
    D->timestep_fluxcalls = source->timestep_fluxcalls;

    // reminder, memcpy works as memcpy(destination, source, size)

    //we shall go in the order we initialized things, because my sanity needs to be kept :) 

    // this is awful
    memcpy(D->neighbours, source->neighbours, 
           3 * source->number_of_elements * sizeof(anuga_int));
    memcpy(D->surrogate_neighbours, source->surrogate_neighbours,
           3 * source->number_of_elements * sizeof(anuga_int));
    memcpy(D->neighbour_edges, source->neighbour_edges,
           3 * source->number_of_elements * sizeof(anuga_int));
    memcpy(D->normals, source->normals,
           6 * source->number_of_elements * sizeof(double));
    memcpy(D->edgelengths, source->edgelengths,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->radii, source->radii,
           source->number_of_elements * sizeof(double));
    memcpy(D->areas, source->areas,
           source->number_of_elements * sizeof(double));
    memcpy(D->edge_flux_type, source->edge_flux_type,
           3 * source->number_of_elements * sizeof(anuga_int));
    memcpy(D->tri_full_flag, source->tri_full_flag,
           source->number_of_elements * sizeof(anuga_int));
    memcpy(D->already_computed_flux, source->already_computed_flux,
           3 * source->number_of_elements * sizeof(anuga_int));
    memcpy(D->vertex_coordinates, source->vertex_coordinates,
           6 * source->number_of_elements * sizeof(double));
    memcpy(D->edge_coordinates, source->edge_coordinates,
           6 * source->number_of_elements * sizeof(double));
    memcpy(D->centroid_coordinates, source->centroid_coordinates,
           2 * source->number_of_elements * sizeof(double));
    memcpy(D->max_speed, source->max_speed,
           source->number_of_elements * sizeof(double));
    memcpy(D->number_of_boundaries, source->number_of_boundaries,
           source->boundary_length * sizeof(anuga_int));
    memcpy(D->flux_update_frequency, source->flux_update_frequency,
           3 * source->number_of_elements * sizeof(anuga_int));
    memcpy(D->update_next_flux, source->update_next_flux,
           3 * source->number_of_elements * sizeof(anuga_int));
    memcpy(D->update_extrapolation, source->update_extrapolation,
           source->number_of_elements * sizeof(anuga_int));
    memcpy(D->allow_timestep_increase, source->allow_timestep_increase,
           1 * sizeof(anuga_int));
    memcpy(D->edge_timestep, source->edge_timestep,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->edge_flux_work, source->edge_flux_work,
           9 * source->number_of_elements * sizeof(double));
    memcpy(D->neigh_work, source->neigh_work,
           9 * source->number_of_elements * sizeof(double));
    memcpy(D->pressuregrad_work, source->pressuregrad_work,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->x_centroid_work, source->x_centroid_work,
           source->number_of_elements * sizeof(double));
    memcpy(D->y_centroid_work, source->y_centroid_work,
           source->number_of_elements * sizeof(double));
    memcpy(D->boundary_flux_sum, source->boundary_flux_sum,
           3 * sizeof(double));
    memcpy(D->edge_river_wall_counter, source->edge_river_wall_counter,
           3 * source->number_of_elements * sizeof(anuga_int));

    memcpy(D->stage_edge_values, source->stage_edge_values,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->xmom_edge_values, source->xmom_edge_values,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->ymom_edge_values, source->ymom_edge_values,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->bed_edge_values, source->bed_edge_values,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->height_edge_values, source->height_edge_values,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->xvelocity_edge_values, source->xvelocity_edge_values,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->yvelocity_edge_values, source->yvelocity_edge_values,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->stage_centroid_values, source->stage_centroid_values,
           source->number_of_elements * sizeof(double));
    memcpy(D->xmom_centroid_values, source->xmom_centroid_values,
           source->number_of_elements * sizeof(double));
    memcpy(D->ymom_centroid_values, source->ymom_centroid_values,
           source->number_of_elements * sizeof(double));
    memcpy(D->bed_centroid_values, source->bed_centroid_values,
           source->number_of_elements * sizeof(double));
    memcpy(D->height_centroid_values, source->height_centroid_values,
           source->number_of_elements * sizeof(double));
    memcpy(D->friction_centroid_values, source->friction_centroid_values,
           source->number_of_elements * sizeof(double));
    memcpy(D->stage_backup_values, source->stage_backup_values,
           source->number_of_elements * sizeof(double));
    memcpy(D->xmom_backup_values, source->xmom_backup_values,
           source->number_of_elements * sizeof(double));
    memcpy(D->ymom_backup_values, source->ymom_backup_values,
           source->number_of_elements * sizeof(double));

    memcpy(D->stage_vertex_values, source->stage_vertex_values,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->xmom_vertex_values, source->xmom_vertex_values,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->ymom_vertex_values, source->ymom_vertex_values,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->bed_vertex_values, source->bed_vertex_values,
           3 * source->number_of_elements * sizeof(double));
    memcpy(D->height_vertex_values, source->height_vertex_values,
           3 * source->number_of_elements * sizeof(double));

    memcpy(D->stage_boundary_values, source->stage_boundary_values,
           source->boundary_length * sizeof(double));
    memcpy(D->xmom_boundary_values, source->xmom_boundary_values,
           source->boundary_length * sizeof(double));
    memcpy(D->ymom_boundary_values, source->ymom_boundary_values,
           source->boundary_length * sizeof(double));
    memcpy(D->bed_boundary_values, source->bed_boundary_values,
           source->boundary_length * sizeof(double));
    memcpy(D->height_boundary_values, source->height_boundary_values,
           source->boundary_length * sizeof(double));
    memcpy(D->xvelocity_boundary_values, source->xvelocity_boundary_values,
           source->boundary_length * sizeof(double));
    memcpy(D->yvelocity_boundary_values, source->yvelocity_boundary_values,
           source->boundary_length * sizeof(double));

    memcpy(D->stage_explicit_update, source->stage_explicit_update,
           source->number_of_elements * sizeof(double));
    memcpy(D->xmom_explicit_update, source->xmom_explicit_update,
           source->number_of_elements * sizeof(double));
    memcpy(D->ymom_explicit_update, source->ymom_explicit_update,
           source->number_of_elements * sizeof(double));
    memcpy(D->stage_semi_implicit_update, source->stage_semi_implicit_update,
           source->number_of_elements * sizeof(double));
    memcpy(D->xmom_semi_implicit_update, source->xmom_semi_implicit_update,
           source->number_of_elements * sizeof(double));
    memcpy(D->ymom_semi_implicit_update, source->ymom_semi_implicit_update,
           source->number_of_elements * sizeof(double));

    memcpy(D->riverwall_elevation, source->riverwall_elevation,
           1 * sizeof(double));
    memcpy(D->riverwall_rowIndex, source->riverwall_rowIndex,
           1 * sizeof(anuga_int));
    memcpy(D->riverwall_hydraulic_properties, source->riverwall_hydraulic_properties,
           1 * sizeof(double));


           //printf("Domain copied successfully.\n");

}