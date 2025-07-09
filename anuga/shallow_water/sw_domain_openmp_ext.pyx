#cython: wraparound=False, boundscheck=False, cdivision=True, profile=False, nonecheck=False, overflowcheck=False, cdivision_warnings=False, unraisable_tracebacks=False

import cython
from libc.stdint cimport int64_t

# import both numpy and the Cython declarations for numpy
import sys
import numpy as np
cimport numpy as np


cdef extern from "sw_domain_openmp.c" nogil:
	struct domain:
		# these shouldn't change within a single timestep
		# they are set once before the evolve loop
		int64_t number_of_elements
		int64_t boundary_length
		int64_t number_of_riverwall_edges
		double epsilon
		double H0
		double g
		int64_t optimise_dry_cells
		double evolve_max_timestep
		int64_t extrapolate_velocity_second_order
		double minimum_allowed_height
		double maximum_allowed_speed
		int64_t low_froude
		int64_t timestep_fluxcalls
		double beta_w
		double beta_w_dry
		double beta_uh
		double beta_uh_dry
		double beta_vh
		double beta_vh_dry
		int64_t max_flux_update_frequency
		int64_t ncol_riverwall_hydraulic_properties

		# these pointers are set once the domain is created
		# and are used in the evolve loop
		int64_t* neighbours
		int64_t* neighbour_edges
		int64_t* surrogate_neighbours
		double* normals
		double* edgelengths
		double* radii
		double* areas
		int64_t* edge_flux_type
		int64_t* tri_full_flag
		int64_t* already_computed_flux
		double* max_speed
		double* vertex_coordinates
		double* edge_coordinates
		double* centroid_coordinates
		int64_t* number_of_boundaries
		double* stage_edge_values
		double* xmom_edge_values
		double* ymom_edge_values
		double* bed_edge_values
		double* height_edge_values
		double* xvelocity_edge_values
		double* yvelocity_edge_values
		double* stage_centroid_values
		double* xmom_centroid_values
		double* ymom_centroid_values
		double* bed_centroid_values
		double* height_centroid_values
		double* stage_vertex_values
		double* xmom_vertex_values
		double* ymom_vertex_values
		double* bed_vertex_values
		double* height_vertex_values
		double* stage_boundary_values
		double* xmom_boundary_values
		double* ymom_boundary_values
		double* bed_boundary_values
		double* height_boundary_values
		double* xvelocity_boundary_values
		double* yvelocity_boundary_values
		double* stage_explicit_update
		double* xmom_explicit_update
		double* ymom_explicit_update
		int64_t* flux_update_frequency
		int64_t* update_next_flux
		int64_t* update_extrapolation
		double* edge_timestep
		double* edge_flux_work
		double* neigh_work
		double* pressuregrad_work
		double* x_centroid_work
		double* y_centroid_work
		double* boundary_flux_sum
		int64_t* allow_timestep_increase
		double* riverwall_elevation
		int64_t* riverwall_rowIndex
		double* riverwall_hydraulic_properties
		int64_t* edge_river_wall_counter
		double* stage_semi_implicit_update
		double* xmom_semi_implicit_update
		double* ymom_semi_implicit_update
		double* friction_centroid_values
		double* stage_backup_values
		double* xmom_backup_values
		double* ymom_backup_values



	struct edge:
		pass

	void test_function(float a, float b, int64_t size, int64_t reps)
	int64_t __rotate(double *q, double n1, double n2)
	double _openmp_compute_fluxes_central(domain* D, double timestep)
	double _openmp_protect(domain* D)
	int64_t _openmp_extrapolate_second_order_sw(domain* D)
	int64_t _openmp_extrapolate_second_order_edge_sw(domain* D)
	int64_t _openmp_fix_negative_cells(domain* D)
	int64_t _openmp_gravity(domain *D)
	int64_t _openmp_gravity_wb(domain *D) 
	int64_t _openmp_update_conserved_quantities(domain* D, double timestep)
	void _openmp_manning_friction_flat_semi_implicit(domain *D)
	void _openmp_manning_friction_sloped_semi_implicit(domain *D)
	int64_t _openmp_saxpy_conserved_quantities(domain *D, double a, double b, double c)
	int64_t _openmp_backup_conserved_quantities(domain *D)
	# FIXME SR: Change over to domain* D argument
	void _openmp_manning_friction_flat(double g, double eps, int64_t N, double* w, double* zv, double* uh, double* vh, double* eta, double* xmom, double* ymom)
	void _openmp_manning_friction_sloped(double g, double eps, int64_t N, double* x, double* w, double* zv, double* uh, double* vh, double* eta, double* xmom_update, double* ymom_update)
	void _openmp_evaluate_reflective_segment(domain *D, int64_t N, int64_t *edge_ptr, int64_t *vol_ids_ptr, int64_t *edge_ids_ptr)
	int64_t __flux_function_central(double* ql, double* qr, double h_left,
	double h_right, double hle, double hre, double n1, double n2,
	double epsilon, double ze, double g,
	double* edgeflux, double* max_speed, double* pressure_flux,
	int64_t low_froude)

cdef extern from "domain_c_struct.c" nogil:
	void init_c_domain(domain* D, int64_t number_of_elements, int64_t boundary_length)
	void free_c_domain(domain* D)
	void copy_c_domain(domain* D, domain* source)
	void say_hi()

cdef int64_t pointer_flag = 0
cdef int64_t parameter_flag = 0

cdef inline get_python_domain_parameters(domain *D, object domain_object):

	# these shouldn't change within a single timestep
	# they are set once before the evolve loop
	D.number_of_elements = domain_object.number_of_elements
	D.boundary_length = domain_object.boundary_length 
	D.number_of_riverwall_edges = domain_object.number_of_riverwall_edges
	D.epsilon = domain_object.epsilon
	D.H0 = domain_object.H0
	D.g = domain_object.g
	D.optimise_dry_cells = domain_object.optimise_dry_cells
	D.evolve_max_timestep = domain_object.evolve_max_timestep
	D.minimum_allowed_height = domain_object.minimum_allowed_height
	D.maximum_allowed_speed = domain_object.maximum_allowed_speed
	D.timestep_fluxcalls = domain_object.timestep_fluxcalls
	D.low_froude = domain_object.low_froude
	D.extrapolate_velocity_second_order = domain_object.extrapolate_velocity_second_order
	D.beta_w = domain_object.beta_w
	D.beta_w_dry = domain_object.beta_w_dry
	D.beta_uh = domain_object.beta_uh
	D.beta_uh_dry = domain_object.beta_uh_dry
	D.beta_vh = domain_object.beta_vh
	D.beta_vh_dry = domain_object.beta_vh_dry
	D.max_flux_update_frequency = domain_object.max_flux_update_frequency
		

cdef inline get_python_domain_pointers(domain *D, object domain_object):

	# these arraypointers are set once the domain is created
	# and are used in the evolve loop
	# same order as they are used below
	cdef int64_t[:,::1]   neighbours
	cdef int64_t[:,::1] surrogate_neighbours
	cdef int64_t[:,::1]   neighbour_edges
	cdef double[:,::1] normals
	cdef double[:,::1] edgelengths
	cdef double[::1]   radii
	cdef double[::1]   areas
	cdef int64_t[::1]  edge_flux_type
	cdef int64_t[::1]  tri_full_flag
	cdef int64_t[:,::1] already_computed_flux
	cdef double[:,::1] vertex_coordinates
	cdef double[:,::1] edge_coordinates
	cdef double[:,::1] centroid_coordinates
	cdef double[::1]   max_speed
	cdef int64_t[::1]  number_of_boundaries
	cdef int64_t[::1]  flux_update_frequency
	cdef int64_t[::1]  update_next_flux
	cdef int64_t[::1]  update_extrapolation
	cdef int64_t[::1]  allow_timestep_increase
	cdef double[::1]   edge_timestep
	cdef double[::1]   edge_flux_work
	cdef double[::1]   neigh_work
	cdef double[::1]   pressuregrad_work
	cdef double[::1]   x_centroid_work
	cdef double[::1]   y_centroid_work
	cdef double[::1]   boundary_flux_sum
	cdef int64_t[::1]  edge_river_wall_counter

	cdef int64_t[::1]  riverwall_rowIndex

	cdef double[::1]   riverwall_elevation
	cdef double[:,::1] riverwall_hydraulic_properties
	cdef double[:,::1] edge_values
	cdef double[::1]   centroid_values
	cdef double[:,::1] vertex_values
	cdef double[::1]   boundary_values
	cdef double[::1]   explicit_update
	cdef double[::1]   semi_implicit_update
	
	cdef object quantities
	cdef object riverwallData

	#------------------------------------------------------
	# Domain structures
	#------------------------------------------------------
	# size = 3 * number_of_elements
	neighbours = domain_object.neighbours
	D.neighbours = &neighbours[0,0]
	
	# size = 3 * number_of_elements
	surrogate_neighbours = domain_object.surrogate_neighbours
	D.surrogate_neighbours = &surrogate_neighbours[0,0]

	# size = 3 * number_of_elements
	neighbour_edges = domain_object.neighbour_edges
	D.neighbour_edges = &neighbour_edges[0,0]

	# size = 6 * number_of_elements
	normals = domain_object.normals
	D.normals = &normals[0,0]

	# size = 3 * number_of_elements
	edgelengths = domain_object.edgelengths
	D.edgelengths = &edgelengths[0,0]

	# size = number_of_elements
	radii = domain_object.radii
	D.radii = &radii[0]

	# size = number_of_elements
	areas = domain_object.areas
	D.areas = &areas[0]

	# size = 3 * number_of_elements
	edge_flux_type = domain_object.edge_flux_type
	D.edge_flux_type = &edge_flux_type[0]
	

	# size = number_of_elements
	tri_full_flag = domain_object.tri_full_flag
	D.tri_full_flag = &tri_full_flag[0]

	# size = 3 * number_of_elements
	already_computed_flux = domain_object.already_computed_flux
	D.already_computed_flux = &already_computed_flux[0,0]

	# size = 6 * number_of_elements
	vertex_coordinates = domain_object.vertex_coordinates
	D.vertex_coordinates = &vertex_coordinates[0,0]

	# size = 6 * number_of_elements
	edge_coordinates = domain_object.edge_coordinates
	D.edge_coordinates = &edge_coordinates[0,0]

	# size = 2 * number_of_elements
	centroid_coordinates = domain_object.centroid_coordinates
	D.centroid_coordinates = &centroid_coordinates[0,0]

	# size = number_of_elements
	max_speed = domain_object.max_speed
	D.max_speed = &max_speed[0]

	# size = number_of_elements
	number_of_boundaries = domain_object.number_of_boundaries
	D.number_of_boundaries = &number_of_boundaries[0]

	# size = 3 * number_of_elements
	flux_update_frequency = domain_object.flux_update_frequency
	D.flux_update_frequency = &flux_update_frequency[0]

	# size = 3 * number_of_elements
	update_next_flux = domain_object.update_next_flux
	D.update_next_flux = &update_next_flux[0]

	# size = number_of_elements
	update_extrapolation = domain_object.update_extrapolation
	D.update_extrapolation = &update_extrapolation[0]

	# size = 1
	allow_timestep_increase = domain_object.allow_timestep_increase
	D.allow_timestep_increase = &allow_timestep_increase[0]

	# size = 3 * number_of_elements
	edge_timestep = domain_object.edge_timestep
	D.edge_timestep = &edge_timestep[0]

	# size = 9 * number_of_elements
	edge_flux_work = domain_object.edge_flux_work
	D.edge_flux_work = &edge_flux_work[0]

	# size = 9 * number_of_elements
	neigh_work = domain_object.neigh_work
	D.neigh_work = &neigh_work[0]

	# size = 3 * number_of_elements
	pressuregrad_work = domain_object.pressuregrad_work
	D.pressuregrad_work = &pressuregrad_work[0]

	# size = number_of_elements
	x_centroid_work = domain_object.x_centroid_work
	D.x_centroid_work = &x_centroid_work[0]

	# size = number_of_elements
	y_centroid_work = domain_object.y_centroid_work
	D.y_centroid_work = &y_centroid_work[0]

	# size = 3 
	boundary_flux_sum = domain_object.boundary_flux_sum
	D.boundary_flux_sum = &boundary_flux_sum[0]

	# size = 3 * number_of_elements
	edge_river_wall_counter = domain_object.edge_river_wall_counter
	D.edge_river_wall_counter  = &edge_river_wall_counter[0]

	#------------------------------------------------------
	# Quantity structures
	#------------------------------------------------------
	quantities = domain_object.quantities
	stage = quantities["stage"]
	xmomentum = quantities["xmomentum"]
	ymomentum = quantities["ymomentum"]
	elevation = quantities["elevation"]
	height = quantities["height"]
	friction = quantities["friction"]
	xvelocity = quantities["xvelocity"]
	yvelocity = quantities["yvelocity"]

	# size = 3 * number_of_elements
	edge_values = stage.edge_values
	D.stage_edge_values = &edge_values[0,0]

	# size = 3 * number_of_elements
	edge_values = xmomentum.edge_values
	D.xmom_edge_values = &edge_values[0,0]

	# size = 3 * number_of_elements
	edge_values = ymomentum.edge_values
	D.ymom_edge_values = &edge_values[0,0]

	# size = 3 * number_of_elements
	edge_values = elevation.edge_values
	D.bed_edge_values = &edge_values[0,0]

	# size = 3 * number_of_elements
	edge_values = height.edge_values
	D.height_edge_values = &edge_values[0,0]

	# size = 3 * number_of_elements
	edge_values = xvelocity.edge_values
	D.xvelocity_edge_values = &edge_values[0,0]

	# size = 3 * number_of_elements
	edge_values = yvelocity.edge_values
	D.yvelocity_edge_values = &edge_values[0,0]

	# size = number_of_elements
	centroid_values = stage.centroid_values
	D.stage_centroid_values = &centroid_values[0]

	# size = number_of_elements
	centroid_values = xmomentum.centroid_values
	D.xmom_centroid_values = &centroid_values[0]

	# size = number_of_elements
	centroid_values = ymomentum.centroid_values
	D.ymom_centroid_values = &centroid_values[0]

	# size = number_of_elements
	centroid_values = elevation.centroid_values
	D.bed_centroid_values = &centroid_values[0]

	# size = number_of_elements
	centroid_values = height.centroid_values
	D.height_centroid_values = &centroid_values[0]

	# size = number_of_elements
	centroid_values = friction.centroid_values
	D.friction_centroid_values = &centroid_values[0]	

	# size = number_of_elements
	centroid_values = stage.centroid_backup_values
	D.stage_backup_values = &centroid_values[0]	
	
	# size = number_of_elements
	centroid_values = xmomentum.centroid_backup_values
	D.xmom_backup_values = &centroid_values[0]		
	
	# size = number_of_elements
	centroid_values = ymomentum.centroid_backup_values
	D.ymom_backup_values = &centroid_values[0]	

	#------------------------------------------------------
	# Vertex values
	#------------------------------------------------------

	# size = 3 * number_of_elements
	vertex_values = stage.vertex_values
	D.stage_vertex_values = &vertex_values[0,0]

	# size = 3 * number_of_elements
	vertex_values = xmomentum.vertex_values
	D.xmom_vertex_values = &vertex_values[0,0]

	# size = 3 * number_of_elements
	vertex_values = ymomentum.vertex_values
	D.ymom_vertex_values = &vertex_values[0,0]

	# size = 3 * number_of_elements
	vertex_values = elevation.vertex_values
	D.bed_vertex_values = &vertex_values[0,0]

	# size = 3 * number_of_elements
	vertex_values = height.vertex_values
	D.height_vertex_values = &vertex_values[0,0]


	#------------------------------------------------------
	# Boundary values
	#------------------------------------------------------

	# size = boundary_length
	boundary_values = stage.boundary_values
	D.stage_boundary_values = &boundary_values[0]

	# size = boundary_length
	boundary_values = xmomentum.boundary_values
	D.xmom_boundary_values = &boundary_values[0]

	# size = boundary_length
	boundary_values = ymomentum.boundary_values
	D.ymom_boundary_values = &boundary_values[0]

	# size = boundary_length
	boundary_values = elevation.boundary_values
	D.bed_boundary_values = &boundary_values[0]

	# size = boundary_length
	boundary_values = height.boundary_values
	D.height_boundary_values = &boundary_values[0]

	# size = boundary_length
	boundary_values = xvelocity.boundary_values
	D.xvelocity_boundary_values = &boundary_values[0]

	# size = boundary_length
	boundary_values = yvelocity.boundary_values
	D.yvelocity_boundary_values = &boundary_values[0]

	#------------------------------------------------------
	# Explicit and semi-implicit update values
	#------------------------------------------------------

	# size = number_of_elements
	explicit_update = stage.explicit_update
	D.stage_explicit_update = &explicit_update[0]

	# size = number_of_elements
	explicit_update = xmomentum.explicit_update
	D.xmom_explicit_update = &explicit_update[0]

	# size = number_of_elements
	explicit_update = ymomentum.explicit_update
	D.ymom_explicit_update = &explicit_update[0]

	# size = number_of_elements
	semi_implicit_update = stage.semi_implicit_update
	D.stage_semi_implicit_update = &semi_implicit_update[0]

	# size = number_of_elements
	semi_implicit_update = xmomentum.semi_implicit_update
	D.xmom_semi_implicit_update = &semi_implicit_update[0]

	# size = number_of_elements
	semi_implicit_update = ymomentum.semi_implicit_update
	D.ymom_semi_implicit_update = &semi_implicit_update[0]



	#------------------------------------------------------
	# Riverwall structures
	# 
	# Deal with the case when create_riverwall is called
	# but no reiverwall edges are found.
	#------------------------------------------------------
	riverwallData = domain_object.riverwallData
	# these all seem to be size 1
	try:
		riverwall_elevation = riverwallData.riverwall_elevation
		D.riverwall_elevation = &riverwall_elevation[0]
	except:
		D.riverwall_elevation = NULL

	try:
		riverwall_rowIndex = riverwallData.hydraulic_properties_rowIndex
		D.riverwall_rowIndex = &riverwall_rowIndex[0]
	except:
		D.riverwall_rowIndex = NULL

	try:
		riverwall_hydraulic_properties = riverwallData.hydraulic_properties
		D.riverwall_hydraulic_properties = &riverwall_hydraulic_properties[0,0]
	except:
		D.riverwall_hydraulic_properties = NULL
		

	D.ncol_riverwall_hydraulic_properties = riverwallData.ncol_hydraulic_properties



cdef inline set_python_domain_pointers(domain *D, object domain_object):
	cdef int64_t number_of_elements = domain_object.number_of_elements
	cdef int64_t boundary_length = domain_object.boundary_length
	cdef int i, j

	# 1D helpers
	cdef int idx
	#print("Setting domain pointers from C to python object")

	#------------------------------------------------------
	# Domain structures
	#------------------------------------------------------
	# cdef int64_t[:,::1] neighbours = domain_object.neighbours
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		neighbours[i, j] = D.neighbours[i * 3 + j]

	# cdef int64_t[:,::1] surrogate_neighbours = domain_object.surrogate_neighbours
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		surrogate_neighbours[i, j] = D.surrogate_neighbours[i * 3 + j]

	# cdef int64_t[:,::1] neighbour_edges = domain_object.neighbour_edges
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		neighbour_edges[i, j] = D.neighbour_edges[i * 3 + j]

	# cdef double[:,::1] normals = domain_object.normals
	# for i in range(number_of_elements):
	# 	for j in range(6):
	# 		normals[i, j] = D.normals[i * 6 + j]

	# cdef double[:,::1] edgelengths = domain_object.edgelengths
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		edgelengths[i, j] = D.edgelengths[i * 3 + j]

	# cdef double[::1] radii = domain_object.radii
	# for i in range(number_of_elements):
	# 	radii[i] = D.radii[i]

	# cdef double[::1] areas = domain_object.areas
	# for i in range(number_of_elements):
	# 	areas[i] = D.areas[i]

	# cdef int64_t[::1] edge_flux_type = domain_object.edge_flux_type
	# for i in range(3 * number_of_elements):
	# 	edge_flux_type[i] = D.edge_flux_type[i]

	# cdef int64_t[::1] tri_full_flag = domain_object.tri_full_flag
	# for i in range(number_of_elements):
	# 	tri_full_flag[i] = D.tri_full_flag[i]

	# cdef int64_t[:,::1] already_computed_flux = domain_object.already_computed_flux
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		already_computed_flux[i, j] = D.already_computed_flux[i * 3 + j]
	
	# cdef double[:,::1] vertex_coordinates = domain_object.vertex_coordinates
	# for i in range(number_of_elements):
	# 	for j in range(6):
	# 		vertex_coordinates[i, j] = D.vertex_coordinates[i * 6 + j]
	
	# cdef double[:,::1] edge_coordinates = domain_object.edge_coordinates
	# for i in range(number_of_elements):
	# 	for j in range(6):
	# 		edge_coordinates[i, j] = D.edge_coordinates[i * 6 + j]
	
	# cdef double[:,::1] centroid_coordinates = domain_object.centroid_coordinates
	# for i in range(number_of_elements):
	# 	for j in range(2):
	# 		centroid_coordinates[i, j] = D.centroid_coordinates[i * 2 + j]
	
	cdef double[::1] max_speed = domain_object.max_speed
	for i in range(number_of_elements):
		max_speed[i] = D.max_speed[i]

	# cdef int64_t[::1] number_of_boundaries = domain_object.number_of_boundaries
	# for i in range(number_of_elements):
	# 	number_of_boundaries[i] = D.number_of_boundaries[i]
	
	# cdef int64_t[::1] flux_update_frequency = domain_object.flux_update_frequency
	# for i in range(3 * number_of_elements):
	# 	flux_update_frequency[i] = D.flux_update_frequency[i]
	
	# cdef int64_t[::1] update_next_flux = domain_object.update_next_flux
	# for i in range(3 * number_of_elements):
	# 	update_next_flux[i] = D.update_next_flux[i]
	
	# cdef int64_t[::1] update_extrapolation = domain_object.update_extrapolation
	# for i in range(number_of_elements):
	# 	update_extrapolation[i] = D.update_extrapolation[i]

	# cdef int64_t[::1] allow_timestep_increase = domain_object.allow_timestep_increase
	# allow_timestep_increase[0] = D.allow_timestep_increase[0]

	# cdef double[::1] edge_timestep = domain_object.edge_timestep
	# for i in range(3 * number_of_elements):
	# 	edge_timestep[i] = D.edge_timestep[i]

	# cdef double[::1] edge_flux_work = domain_object.edge_flux_work
	# for i in range(9 * number_of_elements):
	# 	edge_flux_work[i] = D.edge_flux_work[i]

	# cdef double[::1] neigh_work = domain_object.neigh_work
	# for i in range(9 * number_of_elements):
	# 	neigh_work[i] = D.neigh_work[i]

	# cdef double[::1] pressuregrad_work = domain_object.pressuregrad_work
	# for i in range(3 * number_of_elements):
	# 	pressuregrad_work[i] = D.pressuregrad_work[i]
	
	# cdef double[::1] x_centroid_work = domain_object.x_centroid_work
	# for i in range(number_of_elements):
	# 	x_centroid_work[i] = D.x_centroid_work[i]

	# cdef double[::1] y_centroid_work = domain_object.y_centroid_work
	# for i in range(number_of_elements):
	# 	y_centroid_work[i] = D.y_centroid_work[i]
	
	cdef double[::1] boundary_flux_sum = domain_object.boundary_flux_sum
	for i in range(3):
		boundary_flux_sum[i] = D.boundary_flux_sum[i]

	# cdef int64_t[::1] edge_river_wall_counter = domain_object.edge_river_wall_counter
	# for i in range(3 * number_of_elements):
	# 	edge_river_wall_counter[i] = D.edge_river_wall_counter[i]

	# cdef double[:,::1] edge_values = domain_object.quantities["stage"].edge_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		edge_values[i, j] = D.stage_edge_values[i * 3 + j]
	
	# cdef double[:,::1] edge_values_xmom = domain_object.quantities["xmomentum"].edge_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		edge_values_xmom[i, j] = D.xmom_edge_values[i * 3 + j]
	
	# cdef double[:,::1] edge_values_ymom = domain_object.quantities["ymomentum"].edge_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		edge_values_ymom[i, j] = D.ymom_edge_values[i * 3 + j]
	
	# cdef double[:,::1] edge_values_elevation = domain_object.quantities["elevation"].edge_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		edge_values_elevation[i, j] = D.bed_edge_values[i * 3 + j]
	
	# cdef double[:,::1] edge_values_height = domain_object.quantities["height"].edge_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		edge_values_height[i, j] = D.height_edge_values[i * 3 + j]
	
	# cdef double[:,::1] edge_values_xvel = domain_object.quantities["xvelocity"].edge_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		edge_values_xvel[i, j] = D.xvelocity_edge_values[i * 3 + j]

	# cdef double[:,::1] edge_values_yvel = domain_object.quantities["yvelocity"].edge_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		edge_values_yvel[i, j] = D.yvelocity_edge_values[i * 3 + j]
	
	# cdef double[::1] centroid_values = domain_object.quantities["stage"].centroid_values
	# for i in range(number_of_elements):
	# 	centroid_values[i] = D.stage_centroid_values[i]
	
	# cdef double[::1] centroid_values_xmom = domain_object.quantities["xmomentum"].centroid_values
	# for i in range(number_of_elements):
	# 	centroid_values_xmom[i] = D.xmom_centroid_values[i]

	# cdef double[::1] centroid_values_ymom = domain_object.quantities["ymomentum"].centroid_values
	# for i in range(number_of_elements):
	# 	centroid_values_ymom[i] = D.ymom_centroid_values[i]

	# cdef double[::1] centroid_values_elevation = domain_object.quantities["elevation"].centroid_values
	# for i in range(number_of_elements):
	# 	centroid_values_elevation[i] = D.bed_centroid_values[i]

	# cdef double[::1] centroid_values_height = domain_object.quantities["height"].centroid_values
	# for i in range(number_of_elements):
	# 	centroid_values_height[i] = D.height_centroid_values[i]

	# cdef double[::1] centroid_values_friction = domain_object.quantities["friction"].centroid_values
	# for i in range(number_of_elements):
	# 	centroid_values_friction[i] = D.friction_centroid_values[i]

	# cdef double[::1] centroid_values_stage_backup = domain_object.quantities["stage"].centroid_backup_values
	# for i in range(number_of_elements):
	# 	centroid_values_stage_backup[i] = D.stage_backup_values[i]

	# cdef double[::1] centroid_values_xmom_backup = domain_object.quantities["xmomentum"].centroid_backup_values
	# for i in range(number_of_elements):
	# 	centroid_values_xmom_backup[i] = D.xmom_backup_values[i]

	# cdef double[::1] centroid_values_ymom_backup = domain_object.quantities["ymomentum"].centroid_backup_values
	# for i in range(number_of_elements):
	# 	centroid_values_ymom_backup[i] = D.ymom_backup_values[i]

	# cdef double[:,::1] vertex_values_stage = domain_object.quantities["stage"].vertex_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		vertex_values_stage[i, j] = D.stage_vertex_values[i * 3 + j]

	# cdef double[:,::1] vertex_values_xmom = domain_object.quantities["xmomentum"].vertex_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		vertex_values_xmom[i, j] = D.xmom_vertex_values[i * 3 + j]

	# cdef double[:,::1] vertex_values_ymom = domain_object.quantities["ymomentum"].vertex_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		vertex_values_ymom[i, j] = D.ymom_vertex_values[i * 3 + j]

	# cdef double[:,::1] vertex_values_elev = domain_object.quantities["elevation"].vertex_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		vertex_values_elev[i, j] = D.bed_vertex_values[i * 3 + j]

	# cdef double[:,::1] vertex_values_height = domain_object.quantities["height"].vertex_values
	# for i in range(number_of_elements):
	# 	for j in range(3):
	# 		vertex_values_height[i, j] = D.height_vertex_values[i * 3 + j]

	# cdef double[::1] boundary_values = domain_object.quantities["stage"].boundary_values
	# for i in range(boundary_length):
	# 	boundary_values[i] = D.stage_boundary_values[i]

	# cdef double[::1] boundary_values_xmom = domain_object.quantities["xmomentum"].boundary_values
	# for i in range(boundary_length):
	# 	boundary_values_xmom[i] = D.xmom_boundary_values[i]

	# cdef double[::1] boundary_values_ymom = domain_object.quantities["ymomentum"].boundary_values
	# for i in range(boundary_length):
	# 	boundary_values_ymom[i] = D.ymom_boundary_values[i]

	# cdef double[::1] boundary_values_elev = domain_object.quantities["elevation"].boundary_values
	# for i in range(boundary_length):
	# 	boundary_values_elev[i] = D.bed_boundary_values[i]

	# cdef double[::1] boundary_values_height = domain_object.quantities["height"].boundary_values
	# for i in range(boundary_length):
	# 	boundary_values_height[i] = D.height_boundary_values[i]

	# cdef double[::1] boundary_values_xvel = domain_object.quantities["xvelocity"].boundary_values
	# for i in range(boundary_length):
	# 	boundary_values_xvel[i] = D.xvelocity_boundary_values[i]

	# cdef double[::1] boundary_values_yvel = domain_object.quantities["yvelocity"].boundary_values
	# for i in range(boundary_length):
	# 	boundary_values_yvel[i] = D.yvelocity_boundary_values[i]

	cdef double[::1] explicit_update = domain_object.quantities["stage"].explicit_update
	for i in range(number_of_elements):
		explicit_update[i] = D.stage_explicit_update[i]

	cdef double[::1] explicit_update_xmom = domain_object.quantities["xmomentum"].explicit_update
	for i in range(number_of_elements):
		explicit_update_xmom[i] = D.xmom_explicit_update[i]

	cdef double[::1] explicit_update_ymom = domain_object.quantities["ymomentum"].explicit_update
	for i in range(number_of_elements):
		explicit_update_ymom[i] = D.ymom_explicit_update[i]

	# cdef double[::1] semi_implicit_update = domain_object.quantities["stage"].semi_implicit_update
	# for i in range(number_of_elements):
	# 	semi_implicit_update[i] = D.stage_semi_implicit_update[i]

	# cdef double[::1] semi_implicit_update_xmom = domain_object.quantities["xmomentum"].semi_implicit_update
	# for i in range(number_of_elements):
	# 	semi_implicit_update_xmom[i] = D.xmom_semi_implicit_update[i]

	# cdef double[::1] semi_implicit_update_ymom = domain_object.quantities["ymomentum"].semi_implicit_update
	# for i in range(number_of_elements):
	# 	semi_implicit_update_ymom[i] = D.ymom_semi_implicit_update[i]

	#------------------------------------------------------
	# Riverwall structures
	# 
	# Deal with the case when create_riverwall is called
	# but no reiverwall edges are found.
	#------------------------------------------------------
	try:
		D.riverwall_elevation[0] = domain_object.riverwallData.riverwall_elevation[0]
	except:
		pass

	try:
		for i in range(domain_object.riverwallData.hydraulic_properties_rowIndex.shape[0]):
			D.riverwall_rowIndex[i] = domain_object.riverwallData.hydraulic_properties_rowIndex[i]
	except:
		pass

	# try:
	# 	for i in range(domain_object.riverwallData.hydraulic_properties.shape[0]):
	# 		for j in range(domain_object.riverwallData.hydraulic_properties.shape[1]):
	# 			D.riverwall_hydraulic_properties[i, j] = domain_object.riverwallData.hydraulic_properties[i, j]
	# except:
	# 	pass

	D.ncol_riverwall_hydraulic_properties = domain_object.riverwallData.ncol_hydraulic_properties
	#print("Domain pointers set from C to python object")
	


#===============================================================================

def compute_fluxes_ext_central(object domain_object, double timestep):

	cdef domain D
	cdef domain D_malloc

	cdef int64_t number_of_elements
	number_of_elements = domain_object.number_of_elements
	cdef int64_t boundary_length
	boundary_length = domain_object.boundary_length

	# this gets the things from python
	#print("Getting domain parameters from python object")
	get_python_domain_parameters(&D, domain_object)
	#print("Getting domain pointers from python object")
	get_python_domain_pointers(&D, domain_object)
	#print("domain object max speed", [D.max_speed[i] for i in range(number_of_elements)])
	#print("initializing the domain")
	#print("Domain initialized, copying parameters to C domain")
	# let the freak show begin, we will now copy to C 
	#print(" here again ... ")
	with nogil:
		init_c_domain(&D_malloc, number_of_elements, boundary_length)
		copy_c_domain(&D_malloc, &D)
		timestep =  _openmp_compute_fluxes_central(&D_malloc, timestep)

	#print("Time step returned from C: ", timestep)
	set_python_domain_pointers(&D_malloc, domain_object)
	#print("domain object max speed", [D.max_speed[i] for i in range(number_of_elements)])

	with nogil:
		free_c_domain(&D_malloc)
	#print(" here again ... after free_c_domain")
	#sys.exit(1)

	return timestep
def extrapolate_second_order_sw(object domain_object):

	cdef domain D
	cdef int64_t e

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		e = _openmp_extrapolate_second_order_sw(&D)

	if e == -1:
		return None

def extrapolate_second_order_edge_sw(object domain_object):

	cdef domain D
	cdef int64_t e

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		e = _openmp_extrapolate_second_order_edge_sw(&D)

	if e == -1:
		return None

def protect_new(object domain_object):

	cdef domain D

	cdef double mass_error

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		mass_error = _openmp_protect(&D)


	return mass_error

def compute_flux_update_frequency(object domain_object, double timestep):

	pass

def manning_friction_flat_semi_implicit(object domain_object):
	
	cdef domain D

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		_openmp_manning_friction_flat_semi_implicit(&D)

def manning_friction_sloped_semi_implicit(object domain_object):
	
	cdef domain D

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		_openmp_manning_friction_sloped_semi_implicit(&D)


def manning_friction_flat(double g, double eps,
			np.ndarray[double, ndim=1, mode="c"] w not None,
			np.ndarray[double, ndim=1, mode="c"] uh not None,
			np.ndarray[double, ndim=1, mode="c"] vh not None,
			np.ndarray[double, ndim=1, mode="c"] z not None,
			np.ndarray[double, ndim=1, mode="c"] eta not None,
			np.ndarray[double, ndim=1, mode="c"] xmom not None,
			np.ndarray[double, ndim=1, mode="c"] ymom not None):
	
	cdef int64_t N
	
	N = w.shape[0]
	_openmp_manning_friction_flat(g, eps, N, &w[0], &z[0], &uh[0], &vh[0], &eta[0], &xmom[0], &ymom[0])

def manning_friction_sloped(double g, double eps,
		np.ndarray[double, ndim=2, mode="c"] x not None,
		np.ndarray[double, ndim=1, mode="c"] w not None,
		np.ndarray[double, ndim=1, mode="c"] uh not None,
		np.ndarray[double, ndim=1, mode="c"] vh not None,
		np.ndarray[double, ndim=2, mode="c"] z not None,
		np.ndarray[double, ndim=1, mode="c"] eta not None,
		np.ndarray[double, ndim=1, mode="c"] xmom not None,
		np.ndarray[double, ndim=1, mode="c"] ymom not None):
		
	cdef int64_t N
	
	N = w.shape[0]
	_openmp_manning_friction_sloped(g, eps, N, &x[0,0], &w[0], &z[0,0], &uh[0], &vh[0], &eta[0], &xmom[0], &ymom[0])


def fix_negative_cells(object domain_object):

	cdef domain D
	cdef int64_t num_negative_cells

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		num_negative_cells = _openmp_fix_negative_cells(&D)

	return num_negative_cells

def update_conserved_quantities(object domain_object, double timestep):

	cdef domain D
	cdef int64_t num_negative_cells


	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		_openmp_update_conserved_quantities(&D, timestep)
		num_negative_cells = _openmp_fix_negative_cells(&D)

	return num_negative_cells

def saxpy_conserved_quantities(object domain_object, double a, double b, double c):

	cdef domain D


	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		_openmp_saxpy_conserved_quantities(&D, a, b, c)


def backup_conserved_quantities(object domain_object):

	cdef domain D


	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		_openmp_backup_conserved_quantities(&D)	

def evaluate_reflective_segment(object domain_object, np.ndarray[np.int64_t, ndim=1, mode="c"] segment_edges not None, np.ndarray[np.int64_t, ndim=1, mode="c"] vol_ids not None, np.ndarray[np.int64_t, ndim=1, mode="c"] edge_ids not None): 
	cdef domain D
	cdef int64_t N
	N = segment_edges.shape[0]

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)



	with nogil:
		_openmp_evaluate_reflective_segment(&D, N, &segment_edges[0], &vol_ids[0], &edge_ids[0])


def rotate(np.ndarray[double, ndim=1, mode="c"] q not None, np.ndarray[double, ndim=1, mode="c"] normal not None, int64_t direction):
	assert normal.shape[0] == 2, "Normal vector must have 2 components"
	cdef np.ndarray[double, ndim=1, mode="c"] r
	cdef double n1, n2
	n1 = normal[0]
	n2 = normal[1]
	if direction == -1:
		n2 = -n2
	r = np.ascontiguousarray(np.copy(q))
	__rotate(&r[0], n1, n2)
	return r




def flux_function_central(
	np.ndarray[double, ndim=1, mode="c"] normal not None,
	np.ndarray[double, ndim=1, mode="c"] ql not None,
	np.ndarray[double, ndim=1, mode="c"] qr not None,
	double h_left,
	double h_right,
	double hle,
	double hre,
	np.ndarray[double, ndim=1, mode="c"] edgeflux not None,
	double epsilon,
	double ze,
	double g,
	double H0,
	double hc,
	double hc_n,
	int64_t low_froude
):
	cdef double h0, limiting_threshold, max_speed, pressure_flux
	cdef int64_t err

	h0 = H0 * H0
	limiting_threshold = 10 * H0

	err = __flux_function_central(
		&ql[0], &qr[0],
		h_left, h_right, hle, hre, normal[0], normal[1],
		epsilon, ze, g,
		&edgeflux[0], &max_speed, &pressure_flux,
		low_froude
	)

	assert err >= 0, "Discontinuous Elevation"

	return max_speed, pressure_flux

def gravity(object domain_object):
	cdef domain D

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	err = _openmp_gravity(&D)
	if err == -1:
		return None

def gravity_wb(object domain_object):
	cdef domain D
	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)
	err = _openmp_gravity_wb(&D)
	if err == -1:
		return None

def call_c_function_from_python(float a, float b, int64_t size, int64_t reps):
	print("HIIIIII")
	test_function(a, b, size, reps)



