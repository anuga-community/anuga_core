#cython: wraparound=False, boundscheck=False, cdivision=True, profile=False, nonecheck=False, overflowcheck=False, cdivision_warnings=False, unraisable_tracebacks=False

import cython
from libc.stdint cimport int64_t

# import both numpy and the Cython declarations for numpy
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

	int64_t __rotate(double *q, double n1, double n2)
	void _openmp_set_omp_num_threads(int64_t num_threads)
	double _openmp_compute_fluxes_central(domain* D, double timestep)
	double _openmp_protect(domain* D)
	void _openmp_extrapolate_second_order_sw(domain* D)
	void _openmp_extrapolate_second_order_edge_sw(domain* D)
	int64_t _openmp_fix_negative_cells(domain* D)
	int64_t _openmp_gravity(domain *D)
	int64_t _openmp_gravity_wb(domain *D) 
	int64_t _openmp_update_conserved_quantities(domain* D, double timestep)
	void _openmp_manning_friction_flat_semi_implicit(domain *D)
	void _openmp_manning_friction_sloped_semi_implicit(domain *D)
	void _openmp_manning_friction_sloped_semi_implicit_edge_based(domain *D)
	int64_t _openmp_saxpy_conserved_quantities(domain *D, double a, double b, double c)
	int64_t _openmp_backup_conserved_quantities(domain *D)
	void _openmp_distribute_edges_to_vertices(domain *D)
	# FIXME SR: Change over to domain* D argument ?
	void _openmp_manning_friction_flat(double g, double eps, int64_t N, double* w, double* zv, double* uh, double* vh, double* eta, double* xmom, double* ymom)
	void _openmp_manning_friction_sloped(double g, double eps, int64_t N, double* x, double* w, double* zv, double* uh, double* vh, double* eta, double* xmom_update, double* ymom_update)
	void _openmp_manning_friction_sloped_edge_based(double g, double eps, int64_t N, double* x, double* w, double* zv, double* uh, double* vh, double* eta, double* xmom_update, double* ymom_update)
	void _openmp_evaluate_reflective_segment(domain *D, int64_t N, int64_t *edge_ptr, int64_t *vol_ids_ptr, int64_t *edge_ids_ptr)
	int64_t __flux_function_central(double* ql, double* qr, double h_left,
	double h_right, double hle, double hre, double n1, double n2,
	double epsilon, double ze, double g,
	double* edgeflux, double* max_speed, double* pressure_flux,
	int64_t low_froude)


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
	cdef int64_t[:,::1]   neighbours
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
	cdef int64_t[::1]  number_of_boundaries
	cdef int64_t[:,::1] surrogate_neighbours
	cdef double[::1]   max_speed
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
	cdef double[::1]   riverwall_elevation
	cdef int64_t[::1]  riverwall_rowIndex
	cdef double[:,::1] riverwall_hydraulic_properties
	cdef int64_t[::1]  edge_river_wall_counter
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
	neighbours = domain_object.neighbours
	D.neighbours = &neighbours[0,0]
	
	surrogate_neighbours = domain_object.surrogate_neighbours
	D.surrogate_neighbours = &surrogate_neighbours[0,0]

	neighbour_edges = domain_object.neighbour_edges
	D.neighbour_edges = &neighbour_edges[0,0]

	normals = domain_object.normals
	D.normals = &normals[0,0]

	edgelengths = domain_object.edgelengths
	D.edgelengths = &edgelengths[0,0]

	radii = domain_object.radii
	D.radii = &radii[0]

	areas = domain_object.areas
	D.areas = &areas[0]

	edge_flux_type = domain_object.edge_flux_type
	D.edge_flux_type = &edge_flux_type[0]

	tri_full_flag = domain_object.tri_full_flag
	D.tri_full_flag = &tri_full_flag[0]

	already_computed_flux = domain_object.already_computed_flux
	D.already_computed_flux = &already_computed_flux[0,0]

	vertex_coordinates = domain_object.vertex_coordinates
	D.vertex_coordinates = &vertex_coordinates[0,0]

	edge_coordinates = domain_object.edge_coordinates
	D.edge_coordinates = &edge_coordinates[0,0]

	centroid_coordinates = domain_object.centroid_coordinates
	D.centroid_coordinates = &centroid_coordinates[0,0]

	max_speed = domain_object.max_speed
	D.max_speed = &max_speed[0]

	number_of_boundaries = domain_object.number_of_boundaries
	D.number_of_boundaries = &number_of_boundaries[0]

	flux_update_frequency = domain_object.flux_update_frequency
	D.flux_update_frequency = &flux_update_frequency[0]

	update_next_flux = domain_object.update_next_flux
	D.update_next_flux = &update_next_flux[0]

	update_extrapolation = domain_object.update_extrapolation
	D.update_extrapolation = &update_extrapolation[0]

	allow_timestep_increase = domain_object.allow_timestep_increase
	D.allow_timestep_increase = &allow_timestep_increase[0]

	edge_timestep = domain_object.edge_timestep
	D.edge_timestep = &edge_timestep[0]

	edge_flux_work = domain_object.edge_flux_work
	D.edge_flux_work = &edge_flux_work[0]

	neigh_work = domain_object.neigh_work
	D.neigh_work = &neigh_work[0]

	pressuregrad_work = domain_object.pressuregrad_work
	D.pressuregrad_work = &pressuregrad_work[0]

	x_centroid_work = domain_object.x_centroid_work
	D.x_centroid_work = &x_centroid_work[0]

	y_centroid_work = domain_object.y_centroid_work
	D.y_centroid_work = &y_centroid_work[0]

	boundary_flux_sum = domain_object.boundary_flux_sum
	D.boundary_flux_sum = &boundary_flux_sum[0]

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

	edge_values = stage.edge_values
	D.stage_edge_values = &edge_values[0,0]

	edge_values = xmomentum.edge_values
	D.xmom_edge_values = &edge_values[0,0]

	edge_values = ymomentum.edge_values
	D.ymom_edge_values = &edge_values[0,0]

	edge_values = elevation.edge_values
	D.bed_edge_values = &edge_values[0,0]

	edge_values = height.edge_values
	D.height_edge_values = &edge_values[0,0]

	edge_values = xvelocity.edge_values
	D.xvelocity_edge_values = &edge_values[0,0]

	edge_values = yvelocity.edge_values
	D.yvelocity_edge_values = &edge_values[0,0]

	centroid_values = stage.centroid_values
	D.stage_centroid_values = &centroid_values[0]

	centroid_values = xmomentum.centroid_values
	D.xmom_centroid_values = &centroid_values[0]

	centroid_values = ymomentum.centroid_values
	D.ymom_centroid_values = &centroid_values[0]

	centroid_values = elevation.centroid_values
	D.bed_centroid_values = &centroid_values[0]

	centroid_values = height.centroid_values
	D.height_centroid_values = &centroid_values[0]

	centroid_values = friction.centroid_values
	D.friction_centroid_values = &centroid_values[0]	

	centroid_values = stage.centroid_backup_values
	D.stage_backup_values = &centroid_values[0]	
	
	centroid_values = xmomentum.centroid_backup_values
	D.xmom_backup_values = &centroid_values[0]		
	
	centroid_values = ymomentum.centroid_backup_values
	D.ymom_backup_values = &centroid_values[0]	

	#------------------------------------------------------
	# Vertex values
	#------------------------------------------------------

	vertex_values = stage.vertex_values
	D.stage_vertex_values = &vertex_values[0,0]

	vertex_values = xmomentum.vertex_values
	D.xmom_vertex_values = &vertex_values[0,0]

	vertex_values = ymomentum.vertex_values
	D.ymom_vertex_values = &vertex_values[0,0]

	vertex_values = elevation.vertex_values
	D.bed_vertex_values = &vertex_values[0,0]

	vertex_values = height.vertex_values
	D.height_vertex_values = &vertex_values[0,0]


	#------------------------------------------------------
	# Boundary values
	#------------------------------------------------------

	boundary_values = stage.boundary_values
	D.stage_boundary_values = &boundary_values[0]

	boundary_values = xmomentum.boundary_values
	D.xmom_boundary_values = &boundary_values[0]

	boundary_values = ymomentum.boundary_values
	D.ymom_boundary_values = &boundary_values[0]

	boundary_values = elevation.boundary_values
	D.bed_boundary_values = &boundary_values[0]

	boundary_values = height.boundary_values
	D.height_boundary_values = &boundary_values[0]

	boundary_values = xvelocity.boundary_values
	D.xvelocity_boundary_values = &boundary_values[0]

	boundary_values = yvelocity.boundary_values
	D.yvelocity_boundary_values = &boundary_values[0]

	#------------------------------------------------------
	# Explicit and semi-implicit update values
	#------------------------------------------------------

	explicit_update = stage.explicit_update
	D.stage_explicit_update = &explicit_update[0]

	explicit_update = xmomentum.explicit_update
	D.xmom_explicit_update = &explicit_update[0]

	explicit_update = ymomentum.explicit_update
	D.ymom_explicit_update = &explicit_update[0]

	semi_implicit_update = stage.semi_implicit_update
	D.stage_semi_implicit_update = &semi_implicit_update[0]

	semi_implicit_update = xmomentum.semi_implicit_update
	D.xmom_semi_implicit_update = &semi_implicit_update[0]

	semi_implicit_update = ymomentum.semi_implicit_update
	D.ymom_semi_implicit_update = &semi_implicit_update[0]



	#------------------------------------------------------
	# Riverwall structures
	# 
	# Deal with the case when create_riverwall is called
	# but no reiverwall edges are found.
	#------------------------------------------------------
	riverwallData = domain_object.riverwallData

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




#===============================================================================

def set_omp_num_threads(int64_t num_threads):
	"""
	Set the number of OpenMP threads to use.
	"""
	_openmp_set_omp_num_threads(num_threads)

	
def compute_fluxes_ext_central(object domain_object, double timestep):

	cdef domain D


	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		timestep =  _openmp_compute_fluxes_central(&D, timestep)

	return timestep

def extrapolate_second_order_sw(object domain_object):

	cdef domain D
	cdef int64_t e

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		_openmp_extrapolate_second_order_sw(&D)


def distribute_edges_to_vertices(object domain_object):

	cdef domain D
	cdef int64_t e

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		_openmp_distribute_edges_to_vertices(&D)

	

def extrapolate_second_order_edge_sw(object domain_object, distribute_to_vertices=True):

	cdef domain D
	cdef int64_t e

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)


	with nogil:
		_openmp_extrapolate_second_order_edge_sw(&D)

	if distribute_to_vertices:
		with nogil:
			_openmp_distribute_edges_to_vertices(&D)



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

def manning_friction_sloped_semi_implicit_edge_based(object domain_object):
	
	cdef domain D

	get_python_domain_parameters(&D, domain_object)
	get_python_domain_pointers(&D, domain_object)

	with nogil:
		_openmp_manning_friction_sloped_semi_implicit_edge_based(&D)

# FIXME SR: Why is the order of arguments different from the C function?
def manning_friction_flat(double g, double eps,
            np.ndarray[double, ndim=1, mode="c"] w not None,
			np.ndarray[double, ndim=1, mode="c"] uh not None,
			np.ndarray[double, ndim=1, mode="c"] vh not None,
			np.ndarray[double, ndim=1, mode="c"] z_centroid not None,
			np.ndarray[double, ndim=1, mode="c"] eta not None,
			np.ndarray[double, ndim=1, mode="c"] xmom not None,
			np.ndarray[double, ndim=1, mode="c"] ymom not None):
	
	cdef int64_t N
	
	N = w.shape[0]
	_openmp_manning_friction_flat(g, eps, N, &w[0], &z_centroid[0], &uh[0], &vh[0], &eta[0], &xmom[0], &ymom[0])

# FIXME SR: Why is the order of arguments different from the C function?
def manning_friction_sloped(double g, double eps,
        np.ndarray[double, ndim=2, mode="c"] x_vertex not None,
		np.ndarray[double, ndim=1, mode="c"] w not None,
		np.ndarray[double, ndim=1, mode="c"] uh not None,
		np.ndarray[double, ndim=1, mode="c"] vh not None,
		np.ndarray[double, ndim=2, mode="c"] z_vertex not None,
		np.ndarray[double, ndim=1, mode="c"] eta not None,
		np.ndarray[double, ndim=1, mode="c"] xmom not None,
		np.ndarray[double, ndim=1, mode="c"] ymom not None):
		
	cdef int64_t N
	
	N = w.shape[0]
	_openmp_manning_friction_sloped(g, eps, N, &x_vertex[0,0], &w[0], &z_vertex[0,0], &uh[0], &vh[0], &eta[0], &xmom[0], &ymom[0])

# FIXME SR: Why is the order of arguments different from the C function?
def manning_friction_sloped_edge_based(double g, double eps,
        np.ndarray[double, ndim=2, mode="c"] x_edge not None,
		np.ndarray[double, ndim=1, mode="c"] w not None,
		np.ndarray[double, ndim=1, mode="c"] uh not None,
		np.ndarray[double, ndim=1, mode="c"] vh not None,
		np.ndarray[double, ndim=2, mode="c"] z_edge not None,
		np.ndarray[double, ndim=1, mode="c"] eta not None,
		np.ndarray[double, ndim=1, mode="c"] xmom not None,
		np.ndarray[double, ndim=1, mode="c"] ymom not None):
		
	cdef int64_t N
	
	N = w.shape[0]
	_openmp_manning_friction_sloped_edge_based(g, eps, N, &x_edge[0,0], &w[0], &z_edge[0,0], &uh[0], &vh[0], &eta[0], &xmom[0], &ymom[0])


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

