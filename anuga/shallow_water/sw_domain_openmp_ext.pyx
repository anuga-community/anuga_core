#cython: wraparound=False, boundscheck=False, cdivision=True, profile=False, nonecheck=False, overflowcheck=False, cdivision_warnings=False, unraisable_tracebacks=False, warn.unused=False

import cython
from libc.stdint cimport int64_t as anuga_int
from libc.stdint cimport uintptr_t

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# FIXME SR: Should create a python class which holds all of these parameters
# so we don't need to create it before each C function call.

cdef extern from "sw_domain_openmp.c" nogil:
	struct domain:
		# these shouldn't change within a single timestep
		# they are set once before the evolve loop

		anuga_int number_of_elements
		anuga_int boundary_length
		anuga_int number_of_riverwall_edges
		anuga_int optimise_dry_cells
		anuga_int extrapolate_velocity_second_order
		anuga_int low_froude
		anuga_int timestep_fluxcalls
		anuga_int max_flux_update_frequency
		anuga_int ncol_riverwall_hydraulic_properties

		double epsilon
		double H0
		double g
		double evolve_max_timestep
		double evolve_min_timestep
		double minimum_allowed_height
		double maximum_allowed_speed
		double beta_w
		double beta_w_dry
		double beta_uh
		double beta_uh_dry
		double beta_vh
		double beta_vh_dry


		# these pointers are set once the domain is created
		# and are used in the evolve loop
		anuga_int* neighbours
		anuga_int* neighbour_edges
		anuga_int* surrogate_neighbours
		double* normals
		double* edgelengths
		double* radii
		double* areas
		anuga_int* edge_flux_type
		anuga_int* tri_full_flag
		anuga_int* already_computed_flux
		double* max_speed
		double* vertex_coordinates
		double* edge_coordinates
		double* centroid_coordinates
		anuga_int* number_of_boundaries
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
		anuga_int* flux_update_frequency
		anuga_int* update_next_flux
		anuga_int* update_extrapolation
		double* edge_timestep
		double* edge_flux_work
		double* neigh_work
		double* pressuregrad_work
		double* x_centroid_work
		double* y_centroid_work
		double* boundary_flux_sum
		anuga_int* allow_timestep_increase

		anuga_int* edge_river_wall_counter
		double* riverwall_elevation
		anuga_int* riverwall_rowIndex
		double* riverwall_hydraulic_properties

		double* stage_semi_implicit_update
		double* xmom_semi_implicit_update
		double* ymom_semi_implicit_update
		double* friction_centroid_values

		double* stage_backup_values
		double* xmom_backup_values
		double* ymom_backup_values


	struct edge:
		pass

	anuga_int __rotate(double *q, double n1, double n2)
	void _openmp_set_omp_num_threads(anuga_int num_threads)
	double _openmp_compute_fluxes_central(domain* D, double timestep)
	double _openmp_protect(domain* D)
	void _openmp_extrapolate_second_order_edge_sw(domain* D)
	anuga_int _openmp_fix_negative_cells(domain* D)
	anuga_int _openmp_gravity(domain *D)
	anuga_int _openmp_gravity_wb(domain *D) 
	anuga_int _openmp_update_conserved_quantities(domain* D, double timestep)
	void _openmp_manning_friction_flat_semi_implicit(domain *D)
	void _openmp_manning_friction_sloped_semi_implicit(domain *D)
	void _openmp_manning_friction_sloped_semi_implicit_edge_based(domain *D)
	anuga_int _openmp_saxpy_conserved_quantities(domain *D, double a, double b, double c)
	anuga_int _openmp_backup_conserved_quantities(domain *D)
	void _openmp_distribute_edges_to_vertices(domain *D)
	# FIXME SR: Change over to domain* D argument ?
	void _openmp_manning_friction_flat(double g, double eps, anuga_int N, double* w, double* zv, double* uh, double* vh, double* eta, double* xmom, double* ymom)
	void _openmp_manning_friction_sloped(double g, double eps, anuga_int N, double* x, double* w, double* zv, double* uh, double* vh, double* eta, double* xmom_update, double* ymom_update)
	void _openmp_manning_friction_sloped_edge_based(double g, double eps, anuga_int N, double* x, double* w, double* zv, double* uh, double* vh, double* eta, double* xmom_update, double* ymom_update)
	void _openmp_evaluate_reflective_segment(domain *D, anuga_int N, anuga_int *edge_ptr, anuga_int *vol_ids_ptr, anuga_int *edge_ids_ptr)
	anuga_int __openmp__flux_function_central(double q_left0, double q_left1, double q_left2,
	double q_right0, double q_right1, double q_right2,
	double h_left, double h_right, double hle, double hre,
	double n1, double n2, double epsilon, double ze, double g,
	double* edgeflux0, double* edgeflux1, double* edgeflux2,
	double* max_speed, double* pressure_flux, anuga_int low_froude)


cdef anuga_int pointer_flag = 0
cdef anuga_int parameter_flag = 0

cdef inline get_python_domain_parameters(domain *D, object domain_py_object):

	riverwallData = domain_py_object.riverwallData

	# these shouldn't change within a single timestep

	D.number_of_elements = domain_py_object.number_of_elements
	D.boundary_length = domain_py_object.boundary_length
	D.number_of_riverwall_edges = domain_py_object.number_of_riverwall_edges
	D.optimise_dry_cells = domain_py_object.optimise_dry_cells
	D.extrapolate_velocity_second_order = domain_py_object.extrapolate_velocity_second_order
	D.low_froude = domain_py_object.low_froude
	D.timestep_fluxcalls = domain_py_object.timestep_fluxcalls
	D.max_flux_update_frequency = domain_py_object.max_flux_update_frequency

	D.ncol_riverwall_hydraulic_properties = riverwallData.ncol_hydraulic_properties

	D.epsilon = domain_py_object.epsilon
	D.H0 = domain_py_object.H0
	D.g = domain_py_object.g
	D.evolve_max_timestep = domain_py_object.evolve_max_timestep
	D.evolve_min_timestep = domain_py_object.evolve_min_timestep
	D.minimum_allowed_height = domain_py_object.minimum_allowed_height
	D.maximum_allowed_speed = domain_py_object.maximum_allowed_speed
	D.beta_w = domain_py_object.beta_w
	D.beta_w_dry = domain_py_object.beta_w_dry
	D.beta_uh = domain_py_object.beta_uh
	D.beta_uh_dry = domain_py_object.beta_uh_dry
	D.beta_vh = domain_py_object.beta_vh
	D.beta_vh_dry = domain_py_object.beta_vh_dry
		

cdef inline get_python_domain_pointers(domain *D, object domain_py_object):

	# these arraypointers are set once the domain is created
	# and are used in the evolve loop
	cdef anuga_int[:,::1]   neighbours
	cdef anuga_int[:,::1]   neighbour_edges
	cdef double[:,::1] normals
	cdef double[:,::1] edgelengths
	cdef double[::1]   radii
	cdef double[::1]   areas
	cdef anuga_int[::1]  edge_flux_type
	cdef anuga_int[::1]  tri_full_flag
	cdef anuga_int[:,::1] already_computed_flux
	cdef double[:,::1] vertex_coordinates
	cdef double[:,::1] edge_coordinates
	cdef double[:,::1] centroid_coordinates
	cdef anuga_int[::1]  number_of_boundaries
	cdef anuga_int[:,::1] surrogate_neighbours
	cdef double[::1]   max_speed
	cdef anuga_int[::1]  flux_update_frequency
	cdef anuga_int[::1]  update_next_flux
	cdef anuga_int[::1]  update_extrapolation
	cdef anuga_int[::1]  allow_timestep_increase
	cdef double[::1]   edge_timestep
	cdef double[::1]   edge_flux_work
	cdef double[::1]   neigh_work
	cdef double[::1]   pressuregrad_work
	cdef double[::1]   x_centroid_work
	cdef double[::1]   y_centroid_work
	cdef double[::1]   boundary_flux_sum
	cdef double[::1]   riverwall_elevation
	cdef anuga_int[::1]  riverwall_rowIndex
	cdef double[:,::1] riverwall_hydraulic_properties
	cdef anuga_int[::1]  edge_river_wall_counter
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
	neighbours = domain_py_object.neighbours
	D.neighbours = &neighbours[0,0]
	
	surrogate_neighbours = domain_py_object.surrogate_neighbours
	D.surrogate_neighbours = &surrogate_neighbours[0,0]

	neighbour_edges = domain_py_object.neighbour_edges
	D.neighbour_edges = &neighbour_edges[0,0]

	normals = domain_py_object.normals
	D.normals = &normals[0,0]

	edgelengths = domain_py_object.edgelengths
	D.edgelengths = &edgelengths[0,0]

	radii = domain_py_object.radii
	D.radii = &radii[0]

	areas = domain_py_object.areas
	D.areas = &areas[0]

	edge_flux_type = domain_py_object.edge_flux_type
	D.edge_flux_type = &edge_flux_type[0]

	tri_full_flag = domain_py_object.tri_full_flag
	D.tri_full_flag = &tri_full_flag[0]

	already_computed_flux = domain_py_object.already_computed_flux
	D.already_computed_flux = &already_computed_flux[0,0]

	vertex_coordinates = domain_py_object.vertex_coordinates
	D.vertex_coordinates = &vertex_coordinates[0,0]

	edge_coordinates = domain_py_object.edge_coordinates
	D.edge_coordinates = &edge_coordinates[0,0]

	centroid_coordinates = domain_py_object.centroid_coordinates
	D.centroid_coordinates = &centroid_coordinates[0,0]

	max_speed = domain_py_object.max_speed
	D.max_speed = &max_speed[0]

	number_of_boundaries = domain_py_object.number_of_boundaries
	D.number_of_boundaries = &number_of_boundaries[0]

	flux_update_frequency = domain_py_object.flux_update_frequency
	D.flux_update_frequency = &flux_update_frequency[0]

	update_next_flux = domain_py_object.update_next_flux
	D.update_next_flux = &update_next_flux[0]

	update_extrapolation = domain_py_object.update_extrapolation
	D.update_extrapolation = &update_extrapolation[0]

	allow_timestep_increase = domain_py_object.allow_timestep_increase
	D.allow_timestep_increase = &allow_timestep_increase[0]

	edge_timestep = domain_py_object.edge_timestep
	D.edge_timestep = &edge_timestep[0]

	edge_flux_work = domain_py_object.edge_flux_work
	D.edge_flux_work = &edge_flux_work[0]

	neigh_work = domain_py_object.neigh_work
	D.neigh_work = &neigh_work[0]

	pressuregrad_work = domain_py_object.pressuregrad_work
	D.pressuregrad_work = &pressuregrad_work[0]

	x_centroid_work = domain_py_object.x_centroid_work
	D.x_centroid_work = &x_centroid_work[0]

	y_centroid_work = domain_py_object.y_centroid_work
	D.y_centroid_work = &y_centroid_work[0]

	boundary_flux_sum = domain_py_object.boundary_flux_sum
	D.boundary_flux_sum = &boundary_flux_sum[0]

	edge_river_wall_counter = domain_py_object.edge_river_wall_counter
	D.edge_river_wall_counter  = &edge_river_wall_counter[0]

	#------------------------------------------------------
	# Quantity structures
	#------------------------------------------------------
	quantities = domain_py_object.quantities
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
	riverwallData = domain_py_object.riverwallData

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
# DomainStruct: Cython wrapper for domain struct
#===============================================================================

from cpython.ref cimport PyObject
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from libc.stdint cimport uintptr_t

cdef class Domain_C_struct:
	"""
	Cython wrapper stored on the Python Domain object.
	Owns a domain* that is reused between calls.
	"""
	cdef domain *domain_c_struct_ptr
	cdef domain  domain_snapshot          # by-value snapshot
	cdef public object domain_py_object   # keep a reference to the Python Domain
	cdef public object verbose

	def __cinit__(self, object domain_py_object, verbose=False):
		self.domain_c_struct_ptr = <domain*> PyMem_Malloc(sizeof(domain))
		if self.domain_c_struct_ptr == NULL:
			raise MemoryError()

		self.domain_py_object = domain_py_object
		self.verbose = verbose

		# Initial fill from Python Domain
		get_python_domain_parameters(self.domain_c_struct_ptr, self.domain_py_object)
		get_python_domain_pointers(self.domain_c_struct_ptr, self.domain_py_object)

		# Initial snapshot
		self.domain_snapshot = self.domain_c_struct_ptr[0]

		if self.verbose:
			print("Created Domain_C_struct at %#x"
			  % <uintptr_t> self.domain_c_struct_ptr)

	def __dealloc__(self):
		if self.verbose:
			print("Deleting Domain_C_struct at %#x"
			   % <uintptr_t> self.domain_c_struct_ptr)

		if self.domain_c_struct_ptr != NULL:
			PyMem_Free(self.domain_c_struct_ptr)
			self.domain_c_struct_ptr = NULL

	def update_domain_c_struct(self):
		get_python_domain_parameters(self.domain_c_struct_ptr, self.domain_py_object)
		get_python_domain_pointers(self.domain_c_struct_ptr, self.domain_py_object)

	cdef domain* get_domain_c_struct_ptr(self, update_c_domain_struct=False):
		if update_c_domain_struct:
			get_python_domain_parameters(self.domain_c_struct_ptr, self.domain_py_object)
			get_python_domain_pointers(self.domain_c_struct_ptr, self.domain_py_object)

		if self.verbose:
			print("Returning domain_c_struct_ptr at %#x"
			  % <uintptr_t> self.domain_c_struct_ptr)

		return self.domain_c_struct_ptr

	def refresh_snapshot(self):
		"""
		Replace the snapshot with the current struct contents.
		"""
		self.domain_snapshot = self.domain_c_struct_ptr[0]

	def diff_domain_struct(self):
		"""
		Compare current domain struct with the snapshot and print changes.
		"""
		cdef domain* D = self.domain_c_struct_ptr
		cdef domain* S = &self.domain_snapshot

		# --- scalar fields (unchanged from previous example; include as needed) ---
		if D.number_of_elements != S.number_of_elements:
			print("number_of_elements: %d -> %d"
				  % (S.number_of_elements, D.number_of_elements))
		if D.boundary_length != S.boundary_length:
			print("boundary_length: %d -> %d"
				  % (S.boundary_length, D.boundary_length))
		if D.number_of_riverwall_edges != S.number_of_riverwall_edges:
			print("number_of_riverwall_edges: %d -> %d"
				  % (S.number_of_riverwall_edges, D.number_of_riverwall_edges))
		if D.epsilon != S.epsilon:
			print("epsilon: %g -> %g" % (S.epsilon, D.epsilon))
		if D.H0 != S.H0:
			print("H0: %g -> %g" % (S.H0, D.H0))
		if D.g != S.g:
			print("g: %g -> %g" % (S.g, D.g))
		if D.optimise_dry_cells != S.optimise_dry_cells:
			print("optimise_dry_cells: %d -> %d"
				  % (S.optimise_dry_cells, D.optimise_dry_cells))
		if D.evolve_max_timestep != S.evolve_max_timestep:
			print("evolve_max_timestep: %g -> %g"
				  % (S.evolve_max_timestep, D.evolve_max_timestep))
		if D.evolve_min_timestep != S.evolve_min_timestep:
			print("evolve_min_timestep: %g -> %g"
				  % (S.evolve_min_timestep, D.evolve_min_timestep))
		if D.minimum_allowed_height != S.minimum_allowed_height:
			print("minimum_allowed_height: %g -> %g"
				  % (S.minimum_allowed_height, D.minimum_allowed_height))
		if D.maximum_allowed_speed != S.maximum_allowed_speed:
			print("maximum_allowed_speed: %g -> %g"
				  % (S.maximum_allowed_speed, D.maximum_allowed_speed))
		if D.low_froude != S.low_froude:
			print("low_froude: %d -> %d" % (S.low_froude, D.low_froude))
		if D.timestep_fluxcalls != S.timestep_fluxcalls:
			print("timestep_fluxcalls: %d -> %d"
				  % (S.timestep_fluxcalls, D.timestep_fluxcalls))
		if D.beta_w != S.beta_w:
			print("beta_w: %g -> %g" % (S.beta_w, D.beta_w))
		if D.beta_w_dry != S.beta_w_dry:
			print("beta_w_dry: %g -> %g" % (S.beta_w_dry, D.beta_w_dry))
		if D.beta_uh != S.beta_uh:
			print("beta_uh: %g -> %g" % (S.beta_uh, D.beta_uh))
		if D.beta_uh_dry != S.beta_uh_dry:
			print("beta_uh_dry: %g -> %g" % (S.beta_uh_dry, D.beta_uh_dry))
		if D.beta_vh != S.beta_vh:
			print("beta_vh: %g -> %g" % (S.beta_vh, D.beta_vh))
		if D.beta_vh_dry != S.beta_vh_dry:
			print("beta_vh_dry: %g -> %g" % (S.beta_vh_dry, D.beta_vh_dry))
		if D.max_flux_update_frequency != S.max_flux_update_frequency:
			print("max_flux_update_frequency: %d -> %d"
				  % (S.max_flux_update_frequency, D.max_flux_update_frequency))
		if D.ncol_riverwall_hydraulic_properties != S.ncol_riverwall_hydraulic_properties:
			print("ncol_riverwall_hydraulic_properties: %d -> %d"
				  % (S.ncol_riverwall_hydraulic_properties,
					 D.ncol_riverwall_hydraulic_properties))

		# --- pointer fields (all pointers in struct domain) ---
		if <uintptr_t>D.neighbours != <uintptr_t>S.neighbours:
			print("neighbours: %#x -> %#x"
				  % (<uintptr_t>S.neighbours, <uintptr_t>D.neighbours))
		if <uintptr_t>D.neighbour_edges != <uintptr_t>S.neighbour_edges:
			print("neighbour_edges: %#x -> %#x"
				  % (<uintptr_t>S.neighbour_edges, <uintptr_t>D.neighbour_edges))
		if <uintptr_t>D.surrogate_neighbours != <uintptr_t>S.surrogate_neighbours:
			print("surrogate_neighbours: %#x -> %#x"
				  % (<uintptr_t>S.surrogate_neighbours,
					 <uintptr_t>D.surrogate_neighbours))
		if <uintptr_t>D.normals != <uintptr_t>S.normals:
			print("normals: %#x -> %#x"
				  % (<uintptr_t>S.normals, <uintptr_t>D.normals))
		if <uintptr_t>D.edgelengths != <uintptr_t>S.edgelengths:
			print("edgelengths: %#x -> %#x"
				  % (<uintptr_t>S.edgelengths, <uintptr_t>D.edgelengths))
		if <uintptr_t>D.radii != <uintptr_t>S.radii:
			print("radii: %#x -> %#x"
				  % (<uintptr_t>S.radii, <uintptr_t>D.radii))
		if <uintptr_t>D.areas != <uintptr_t>S.areas:
			print("areas: %#x -> %#x"
				  % (<uintptr_t>S.areas, <uintptr_t>D.areas))
		if <uintptr_t>D.edge_flux_type != <uintptr_t>S.edge_flux_type:
			print("edge_flux_type: %#x -> %#x"
				  % (<uintptr_t>S.edge_flux_type, <uintptr_t>D.edge_flux_type))
		if <uintptr_t>D.tri_full_flag != <uintptr_t>S.tri_full_flag:
			print("tri_full_flag: %#x -> %#x"
				  % (<uintptr_t>S.tri_full_flag, <uintptr_t>D.tri_full_flag))
		if <uintptr_t>D.already_computed_flux != <uintptr_t>S.already_computed_flux:
			print("already_computed_flux: %#x -> %#x"
				  % (<uintptr_t>S.already_computed_flux,
					 <uintptr_t>D.already_computed_flux))
		if <uintptr_t>D.max_speed != <uintptr_t>S.max_speed:
			print("max_speed: %#x -> %#x"
				  % (<uintptr_t>S.max_speed, <uintptr_t>D.max_speed))
		if <uintptr_t>D.vertex_coordinates != <uintptr_t>S.vertex_coordinates:
			print("vertex_coordinates: %#x -> %#x"
				  % (<uintptr_t>S.vertex_coordinates,
					 <uintptr_t>D.vertex_coordinates))
		if <uintptr_t>D.edge_coordinates != <uintptr_t>S.edge_coordinates:
			print("edge_coordinates: %#x -> %#x"
				  % (<uintptr_t>S.edge_coordinates,
					 <uintptr_t>D.edge_coordinates))
		if <uintptr_t>D.centroid_coordinates != <uintptr_t>S.centroid_coordinates:
			print("centroid_coordinates: %#x -> %#x"
				  % (<uintptr_t>S.centroid_coordinates,
					 <uintptr_t>D.centroid_coordinates))
		if <uintptr_t>D.number_of_boundaries != <uintptr_t>S.number_of_boundaries:
			print("number_of_boundaries: %#x -> %#x"
				  % (<uintptr_t>S.number_of_boundaries,
					 <uintptr_t>D.number_of_boundaries))
		if <uintptr_t>D.stage_edge_values != <uintptr_t>S.stage_edge_values:
			print("stage_edge_values: %#x -> %#x"
				  % (<uintptr_t>S.stage_edge_values,
					 <uintptr_t>D.stage_edge_values))
		if <uintptr_t>D.xmom_edge_values != <uintptr_t>S.xmom_edge_values:
			print("xmom_edge_values: %#x -> %#x"
				  % (<uintptr_t>S.xmom_edge_values,
					 <uintptr_t>D.xmom_edge_values))
		if <uintptr_t>D.ymom_edge_values != <uintptr_t>S.ymom_edge_values:
			print("ymom_edge_values: %#x -> %#x"
				  % (<uintptr_t>S.ymom_edge_values,
					 <uintptr_t>D.ymom_edge_values))
		if <uintptr_t>D.bed_edge_values != <uintptr_t>S.bed_edge_values:
			print("bed_edge_values: %#x -> %#x"
				  % (<uintptr_t>S.bed_edge_values,
					 <uintptr_t>D.bed_edge_values))
		if <uintptr_t>D.height_edge_values != <uintptr_t>S.height_edge_values:
			print("height_edge_values: %#x -> %#x"
				  % (<uintptr_t>S.height_edge_values,
					 <uintptr_t>D.height_edge_values))
		if <uintptr_t>D.xvelocity_edge_values != <uintptr_t>S.xvelocity_edge_values:
			print("xvelocity_edge_values: %#x -> %#x"
				  % (<uintptr_t>S.xvelocity_edge_values,
					 <uintptr_t>D.xvelocity_edge_values))
		if <uintptr_t>D.yvelocity_edge_values != <uintptr_t>S.yvelocity_edge_values:
			print("yvelocity_edge_values: %#x -> %#x"
				  % (<uintptr_t>S.yvelocity_edge_values,
					 <uintptr_t>D.yvelocity_edge_values))
		if <uintptr_t>D.stage_centroid_values != <uintptr_t>S.stage_centroid_values:
			print("stage_centroid_values: %#x -> %#x"
				  % (<uintptr_t>S.stage_centroid_values,
					 <uintptr_t>D.stage_centroid_values))
		if <uintptr_t>D.xmom_centroid_values != <uintptr_t>S.xmom_centroid_values:
			print("xmom_centroid_values: %#x -> %#x"
				  % (<uintptr_t>S.xmom_centroid_values,
					 <uintptr_t>D.xmom_centroid_values))
		if <uintptr_t>D.ymom_centroid_values != <uintptr_t>S.ymom_centroid_values:
			print("ymom_centroid_values: %#x -> %#x"
				  % (<uintptr_t>S.ymom_centroid_values,
					 <uintptr_t>D.ymom_centroid_values))
		if <uintptr_t>D.bed_centroid_values != <uintptr_t>S.bed_centroid_values:
			print("bed_centroid_values: %#x -> %#x"
				  % (<uintptr_t>S.bed_centroid_values,
					 <uintptr_t>D.bed_centroid_values))
		if <uintptr_t>D.height_centroid_values != <uintptr_t>S.height_centroid_values:
			print("height_centroid_values: %#x -> %#x"
				  % (<uintptr_t>S.height_centroid_values,
					 <uintptr_t>D.height_centroid_values))
		if <uintptr_t>D.stage_vertex_values != <uintptr_t>S.stage_vertex_values:
			print("stage_vertex_values: %#x -> %#x"
				  % (<uintptr_t>S.stage_vertex_values,
					 <uintptr_t>D.stage_vertex_values))
		if <uintptr_t>D.xmom_vertex_values != <uintptr_t>S.xmom_vertex_values:
			print("xmom_vertex_values: %#x -> %#x"
				  % (<uintptr_t>S.xmom_vertex_values,
					 <uintptr_t>D.xmom_vertex_values))
		if <uintptr_t>D.ymom_vertex_values != <uintptr_t>S.ymom_vertex_values:
			print("ymom_vertex_values: %#x -> %#x"
				  % (<uintptr_t>S.ymom_vertex_values,
					 <uintptr_t>D.ymom_vertex_values))
		if <uintptr_t>D.bed_vertex_values != <uintptr_t>S.bed_vertex_values:
			print("bed_vertex_values: %#x -> %#x"
				  % (<uintptr_t>S.bed_vertex_values,
					 <uintptr_t>D.bed_vertex_values))
		if <uintptr_t>D.height_vertex_values != <uintptr_t>S.height_vertex_values:
			print("height_vertex_values: %#x -> %#x"
				  % (<uintptr_t>S.height_vertex_values,
					 <uintptr_t>D.height_vertex_values))
		if <uintptr_t>D.stage_boundary_values != <uintptr_t>S.stage_boundary_values:
			print("stage_boundary_values: %#x -> %#x"
				  % (<uintptr_t>S.stage_boundary_values,
					 <uintptr_t>D.stage_boundary_values))
		if <uintptr_t>D.xmom_boundary_values != <uintptr_t>S.xmom_boundary_values:
			print("xmom_boundary_values: %#x -> %#x"
				  % (<uintptr_t>S.xmom_boundary_values,
					 <uintptr_t>D.xmom_boundary_values))
		if <uintptr_t>D.ymom_boundary_values != <uintptr_t>S.ymom_boundary_values:
			print("ymom_boundary_values: %#x -> %#x"
				  % (<uintptr_t>S.ymom_boundary_values,
					 <uintptr_t>D.ymom_boundary_values))
		if <uintptr_t>D.bed_boundary_values != <uintptr_t>S.bed_boundary_values:
			print("bed_boundary_values: %#x -> %#x"
				  % (<uintptr_t>S.bed_boundary_values,
					 <uintptr_t>D.bed_boundary_values))
		if <uintptr_t>D.height_boundary_values != <uintptr_t>S.height_boundary_values:
			print("height_boundary_values: %#x -> %#x"
				  % (<uintptr_t>S.height_boundary_values,
					 <uintptr_t>D.height_boundary_values))
		if <uintptr_t>D.xvelocity_boundary_values != <uintptr_t>S.xvelocity_boundary_values:
			print("xvelocity_boundary_values: %#x -> %#x"
				  % (<uintptr_t>S.xvelocity_boundary_values,
					 <uintptr_t>D.xvelocity_boundary_values))
		if <uintptr_t>D.yvelocity_boundary_values != <uintptr_t>S.yvelocity_boundary_values:
			print("yvelocity_boundary_values: %#x -> %#x"
				  % (<uintptr_t>S.yvelocity_boundary_values,
					 <uintptr_t>D.yvelocity_boundary_values))
		if <uintptr_t>D.stage_explicit_update != <uintptr_t>S.stage_explicit_update:
			print("stage_explicit_update: %#x -> %#x"
				  % (<uintptr_t>S.stage_explicit_update,
					 <uintptr_t>D.stage_explicit_update))
		if <uintptr_t>D.xmom_explicit_update != <uintptr_t>S.xmom_explicit_update:
			print("xmom_explicit_update: %#x -> %#x"
				  % (<uintptr_t>S.xmom_explicit_update,
					 <uintptr_t>D.xmom_explicit_update))
		if <uintptr_t>D.ymom_explicit_update != <uintptr_t>S.ymom_explicit_update:
			print("ymom_explicit_update: %#x -> %#x"
				  % (<uintptr_t>S.ymom_explicit_update,
					 <uintptr_t>D.ymom_explicit_update))
		if <uintptr_t>D.flux_update_frequency != <uintptr_t>S.flux_update_frequency:
			print("flux_update_frequency: %#x -> %#x"
				  % (<uintptr_t>S.flux_update_frequency,
					 <uintptr_t>D.flux_update_frequency))
		if <uintptr_t>D.update_next_flux != <uintptr_t>S.update_next_flux:
			print("update_next_flux: %#x -> %#x"
				  % (<uintptr_t>S.update_next_flux,
					 <uintptr_t>D.update_next_flux))
		if <uintptr_t>D.update_extrapolation != <uintptr_t>S.update_extrapolation:
			print("update_extrapolation: %#x -> %#x"
				  % (<uintptr_t>S.update_extrapolation,
					 <uintptr_t>D.update_extrapolation))
		if <uintptr_t>D.edge_timestep != <uintptr_t>S.edge_timestep:
			print("edge_timestep: %#x -> %#x"
				  % (<uintptr_t>S.edge_timestep,
					 <uintptr_t>D.edge_timestep))
		if <uintptr_t>D.edge_flux_work != <uintptr_t>S.edge_flux_work:
			print("edge_flux_work: %#x -> %#x"
				  % (<uintptr_t>S.edge_flux_work,
					 <uintptr_t>D.edge_flux_work))
		if <uintptr_t>D.neigh_work != <uintptr_t>S.neigh_work:
			print("neigh_work: %#x -> %#x"
				  % (<uintptr_t>S.neigh_work,
					 <uintptr_t>D.neigh_work))
		if <uintptr_t>D.pressuregrad_work != <uintptr_t>S.pressuregrad_work:
			print("pressuregrad_work: %#x -> %#x"
				  % (<uintptr_t>S.pressuregrad_work,
					 <uintptr_t>D.pressuregrad_work))
		if <uintptr_t>D.x_centroid_work != <uintptr_t>S.x_centroid_work:
			print("x_centroid_work: %#x -> %#x"
				  % (<uintptr_t>S.x_centroid_work,
					 <uintptr_t>D.x_centroid_work))
		if <uintptr_t>D.y_centroid_work != <uintptr_t>S.y_centroid_work:
			print("y_centroid_work: %#x -> %#x"
				  % (<uintptr_t>S.y_centroid_work,
					 <uintptr_t>D.y_centroid_work))
		if <uintptr_t>D.boundary_flux_sum != <uintptr_t>S.boundary_flux_sum:
			print("boundary_flux_sum: %#x -> %#x"
				  % (<uintptr_t>S.boundary_flux_sum,
					 <uintptr_t>D.boundary_flux_sum))
		if <uintptr_t>D.allow_timestep_increase != <uintptr_t>S.allow_timestep_increase:
			print("allow_timestep_increase: %#x -> %#x"
				  % (<uintptr_t>S.allow_timestep_increase,
					 <uintptr_t>D.allow_timestep_increase))
		if <uintptr_t>D.edge_river_wall_counter != <uintptr_t>S.edge_river_wall_counter:
			print("edge_river_wall_counter: %#x -> %#x"
				  % (<uintptr_t>S.edge_river_wall_counter,
					 <uintptr_t>D.edge_river_wall_counter))
		if <uintptr_t>D.riverwall_elevation != <uintptr_t>S.riverwall_elevation:
			print("riverwall_elevation: %#x -> %#x"
				  % (<uintptr_t>S.riverwall_elevation,
					 <uintptr_t>D.riverwall_elevation))
		if <uintptr_t>D.riverwall_rowIndex != <uintptr_t>S.riverwall_rowIndex:
			print("riverwall_rowIndex: %#x -> %#x"
				  % (<uintptr_t>S.riverwall_rowIndex,
					 <uintptr_t>D.riverwall_rowIndex))
		if <uintptr_t>D.riverwall_hydraulic_properties != <uintptr_t>S.riverwall_hydraulic_properties:
			print("riverwall_hydraulic_properties: %#x -> %#x"
				  % (<uintptr_t>S.riverwall_hydraulic_properties,
					 <uintptr_t>D.riverwall_hydraulic_properties))
		if <uintptr_t>D.stage_semi_implicit_update != <uintptr_t>S.stage_semi_implicit_update:
			print("stage_semi_implicit_update: %#x -> %#x"
				  % (<uintptr_t>S.stage_semi_implicit_update,
					 <uintptr_t>D.stage_semi_implicit_update))
		if <uintptr_t>D.xmom_semi_implicit_update != <uintptr_t>S.xmom_semi_implicit_update:
			print("xmom_semi_implicit_update: %#x -> %#x"
				  % (<uintptr_t>S.xmom_semi_implicit_update,
					 <uintptr_t>D.xmom_semi_implicit_update))
		if <uintptr_t>D.ymom_semi_implicit_update != <uintptr_t>S.ymom_semi_implicit_update:
			print("ymom_semi_implicit_update: %#x -> %#x"
				  % (<uintptr_t>S.ymom_semi_implicit_update,
					 <uintptr_t>D.ymom_semi_implicit_update))
		if <uintptr_t>D.friction_centroid_values != <uintptr_t>S.friction_centroid_values:
			print("friction_centroid_values: %#x -> %#x"
				  % (<uintptr_t>S.friction_centroid_values,
					 <uintptr_t>D.friction_centroid_values))
		if <uintptr_t>D.stage_backup_values != <uintptr_t>S.stage_backup_values:
			print("stage_backup_values: %#x -> %#x"
				  % (<uintptr_t>S.stage_backup_values,
					 <uintptr_t>D.stage_backup_values))
		if <uintptr_t>D.xmom_backup_values != <uintptr_t>S.xmom_backup_values:
			print("xmom_backup_values: %#x -> %#x"
				  % (<uintptr_t>S.xmom_backup_values,
					 <uintptr_t>D.xmom_backup_values))
		if <uintptr_t>D.ymom_backup_values != <uintptr_t>S.ymom_backup_values:
			print("ymom_backup_values: %#x -> %#x"
				  % (<uintptr_t>S.ymom_backup_values,
					 <uintptr_t>D.ymom_backup_values))


	def print_domain_struct(self):
		cdef domain* D = self.domain_c_struct_ptr

		print("D.number_of_elements     %d" % D.number_of_elements)
		print("D.boundary_length        %d" % D.boundary_length)
		print("D.number_of_riverwall_edges %d " % D.number_of_riverwall_edges)
		print("D.epsilon                %g" % D.epsilon)
		print("D.H0                     %g" % D.H0)
		print("D.g                      %g" % D.g)
		print("D.optimise_dry_cells     %d" % D.optimise_dry_cells)
		print("D.evolve_max_timestep    %g" % D.evolve_max_timestep)
		print("D.evolve_min_timestep    %g" % D.evolve_min_timestep)
		print("D.minimum_allowed_height %g" % D.minimum_allowed_height)
		print("D.maximum_allowed_speed  %g" % D.maximum_allowed_speed)
		print("D.low_froude             %d" % D.low_froude)
		print("D.extrapolate_velocity_second_order %d" % D.extrapolate_velocity_second_order)
		print("D.beta_w                 %g" % D.beta_w)
		print("D.beta_w_dry             %g" % D.beta_w_dry)
		print("D.beta_uh                %g" % D.beta_uh)
		print("D.beta_uh_dry            %g" % D.beta_uh_dry)
		print("D.beta_vh                %g" % D.beta_vh)
		print("D.beta_vh_dry            %g \n" % D.beta_vh_dry)



		print("D.neighbours             %#x" % <uintptr_t> D.neighbours)
		print("D.surrogate_neighbours   %#x" % <uintptr_t> D.surrogate_neighbours)
		print("D.neighbour_edges        %#x" % <uintptr_t> D.neighbour_edges)
		print("D.normals                %#x" % <uintptr_t> D.normals)
		print("D.edgelengths            %#x" % <uintptr_t> D.edgelengths)
		print("D.radii                  %#x" % <uintptr_t> D.radii)
		print("D.areas                  %#x" % <uintptr_t> D.areas)
		print("D.tri_full_flag          %#x" % <uintptr_t> D.tri_full_flag)
		print("D.already_computed_flux  %#x" % <uintptr_t> D.already_computed_flux)
		print("D.vertex_coordinates     %#x" % <uintptr_t> D.vertex_coordinates)
		print("D.edge_coordinates       %#x" % <uintptr_t> D.edge_coordinates)
		print("D.centroid_coordinates   %#x" % <uintptr_t> D.centroid_coordinates)
		print("D.max_speed              %#x" % <uintptr_t> D.max_speed)
		print("D.number_of_boundaries   %#x" % <uintptr_t> D.number_of_boundaries)
		print("D.stage_edge_values      %#x" % <uintptr_t> D.stage_edge_values)
		print("D.xmom_edge_values       %#x" % <uintptr_t> D.xmom_edge_values)
		print("D.ymom_edge_values       %#x" % <uintptr_t> D.ymom_edge_values)
		print("D.bed_edge_values        %#x" % <uintptr_t> D.bed_edge_values)
		print("D.stage_centroid_values  %#x" % <uintptr_t> D.stage_centroid_values)
		print("D.xmom_centroid_values   %#x" % <uintptr_t> D.xmom_centroid_values)
		print("D.ymom_centroid_values   %#x" % <uintptr_t> D.ymom_centroid_values)
		print("D.bed_centroid_values    %#x" % <uintptr_t> D.bed_centroid_values)
		print("D.stage_vertex_values    %#x" % <uintptr_t> D.stage_vertex_values)
		print("D.xmom_vertex_values     %#x" % <uintptr_t> D.xmom_vertex_values)
		print("D.ymom_vertex_values     %#x" % <uintptr_t> D.ymom_vertex_values)
		print("D.bed_vertex_values      %#x" % <uintptr_t> D.bed_vertex_values)
		print("D.height_vertex_values   %#x" % <uintptr_t> D.height_vertex_values)
		print("D.stage_boundary_values  %#x" % <uintptr_t> D.stage_boundary_values)
		print("D.xmom_boundary_values   %#x" % <uintptr_t> D.xmom_boundary_values)
		print("D.ymom_boundary_values   %#x" % <uintptr_t> D.ymom_boundary_values)
		print("D.bed_boundary_values    %#x" % <uintptr_t> D.bed_boundary_values)
		print("D.stage_explicit_update  %#x" % <uintptr_t> D.stage_explicit_update)
		print("D.xmom_explicit_update   %#x" % <uintptr_t> D.xmom_explicit_update)
		print("D.ymom_explicit_update   %#x" % <uintptr_t> D.ymom_explicit_update)
		print("D.edge_river_wall_counter    %#x" % <uintptr_t> D.edge_river_wall_counter)
		print("D.stage_semi_implicit_update %#x" % <uintptr_t> D.stage_semi_implicit_update)
		print("D.xmom_semi_implicit_update  %#x" % <uintptr_t> D.xmom_semi_implicit_update)
		print("D.ymom_semi_implicit_update  %#x" % <uintptr_t> D.ymom_semi_implicit_update)
		print("D.friction_centroid_values   %#x" % <uintptr_t> D.friction_centroid_values)
		print("D.stage_backup_values        %#x" % <uintptr_t> D.stage_backup_values)
		print("D.xmom_backup_values         %#x" % <uintptr_t> D.xmom_backup_values)
		print("D.ymom_backup_values         %#x" % <uintptr_t> D.ymom_backup_values)


#===============================================================================

def set_omp_num_threads(anuga_int num_threads):
	"""
	Set the number of OpenMP threads to use.
	"""
	_openmp_set_omp_num_threads(num_threads)

	
def setup_Domain_C_struct(object domain_py_object, update_domain_c_struct=False):
	"""
	Setup the Domain_C_struct on the Python Domain object.
	"""
	domain_py_object._Domain_C_struct = Domain_C_struct(domain_py_object)

def update_Domain_C_struct(object domain_py_object):
	"""
	Update the Domain_C_struct on the Python Domain object.
	"""
	if domain_py_object._Domain_C_struct is None:
		setup_Domain_C_struct(domain_py_object)

	domain_py_object._Domain_C_struct.update_domain_c_struct()

cdef domain* get_domain_c_struct_ptr(object domain_py_object, update_domain_c_struct=False):
	"""
	Get the domain* from the Domain_C_struct on the Python Domain object.
	"""
	if domain_py_object._Domain_C_struct is None:
		setup_Domain_C_struct(domain_py_object)

	cdef Domain_C_struct domain_c_struct
	domain_c_struct = domain_py_object._Domain_C_struct
	cdef domain* D = domain_c_struct.get_domain_c_struct_ptr(update_domain_c_struct)

	return D

def compute_fluxes_ext_central(object domain_py_object, 
                               double timestep, 
							   update_domain_c_struct=False):

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		timestep =  _openmp_compute_fluxes_central(D, timestep)

	return timestep

def extrapolate_second_order_sw(object domain_py_object, update_domain_c_struct=False):

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		_openmp_extrapolate_second_order_edge_sw(D)

def distribute_to_edges(object domain_py_object, update_domain_c_struct=False):

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		_openmp_extrapolate_second_order_edge_sw(D)

def distribute_to_edges_and_vertices(object domain_py_object, 
                                    distribute_to_vertices=True, 
									update_domain_c_struct=False):

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		_openmp_extrapolate_second_order_edge_sw(D)

	if distribute_to_vertices:
		with nogil:
			_openmp_distribute_edges_to_vertices(D)


def distribute_edges_to_vertices(object domain_py_object, update_domain_c_struct=False):

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		_openmp_distribute_edges_to_vertices(D)

	

def extrapolate_second_order_edge_sw(object domain_py_object, 
                                     distribute_to_vertices=True,
                                     update_domain_c_struct=False):

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)


	with nogil:
		_openmp_extrapolate_second_order_edge_sw(D)
	if distribute_to_vertices:
		with nogil:
			_openmp_distribute_edges_to_vertices(D)


def protect_new(object domain_py_object, update_domain_c_struct=False):

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	cdef double mass_error

	with nogil:
		mass_error = _openmp_protect(D)


	return mass_error

def compute_flux_update_frequency(object domain_py_object, double timestep):

	pass

def manning_friction_flat_semi_implicit(object domain_py_object, update_domain_c_struct=False):
	
	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		_openmp_manning_friction_flat_semi_implicit(D)

def manning_friction_sloped_semi_implicit(object domain_py_object, update_domain_c_struct=False):
	
	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		_openmp_manning_friction_sloped_semi_implicit(D)

def manning_friction_sloped_semi_implicit_edge_based(object domain_py_object, update_domain_c_struct=False):
	
	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		_openmp_manning_friction_sloped_semi_implicit_edge_based(D)

# FIXME SR: Why is the order of arguments different from the C function?
def manning_friction_flat(double g, double eps,
            np.ndarray[double, ndim=1, mode="c"] w not None,
			np.ndarray[double, ndim=1, mode="c"] uh not None,
			np.ndarray[double, ndim=1, mode="c"] vh not None,
			np.ndarray[double, ndim=1, mode="c"] z_centroid not None,
			np.ndarray[double, ndim=1, mode="c"] eta not None,
			np.ndarray[double, ndim=1, mode="c"] xmom not None,
			np.ndarray[double, ndim=1, mode="c"] ymom not None):
	
	cdef anuga_int N
	
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
		
	cdef anuga_int N
	
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
		
	cdef anuga_int N
	
	N = w.shape[0]
	_openmp_manning_friction_sloped_edge_based(g, eps, N, &x_edge[0,0], &w[0], &z_edge[0,0], &uh[0], &vh[0], &eta[0], &xmom[0], &ymom[0])


def fix_negative_cells(object domain_py_object, update_domain_c_struct=False):

	cdef anuga_int num_negative_cells
	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		num_negative_cells = _openmp_fix_negative_cells(D)

	return num_negative_cells

def update_conserved_quantities(object domain_py_object, 
                                double timestep,
                                update_domain_c_struct=False):

	cdef anuga_int num_negative_cells
	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		_openmp_update_conserved_quantities(D, timestep)
		num_negative_cells = _openmp_fix_negative_cells(D)

	return num_negative_cells

def saxpy_conserved_quantities(object domain_py_object, 
                               double a, double b, double c,
                               update_domain_c_struct=False):

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		_openmp_saxpy_conserved_quantities(D, a, b, c)


def backup_conserved_quantities(object domain_py_object, update_domain_c_struct=False):

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	with nogil:
		_openmp_backup_conserved_quantities(D)	

def evaluate_reflective_segment(object domain_py_object, 
                                np.ndarray[np.int64_t, ndim=1, mode="c"] segment_edges not None, 
								np.ndarray[np.int64_t, ndim=1, mode="c"] vol_ids not None, 
								np.ndarray[np.int64_t, ndim=1, mode="c"] edge_ids not None, 
                                update_domain_c_struct=False): 

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)
	cdef anuga_int N
	N = segment_edges.shape[0]

	with nogil:
		_openmp_evaluate_reflective_segment(D, N, &segment_edges[0], &vol_ids[0], &edge_ids[0])


def rotate(np.ndarray[double, ndim=1, mode="c"] q not None, np.ndarray[double, ndim=1, mode="c"] normal not None, anuga_int direction):

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
	anuga_int low_froude
):
	cdef double h0, limiting_threshold, max_speed, pressure_flux
	cdef anuga_int err

	h0 = H0 * H0
	limiting_threshold = 10 * H0

	err = __openmp__flux_function_central(
		ql[0], ql[1], ql[2],
		qr[0], qr[1], qr[2],
		h_left, h_right, hle, hre,
		normal[0], normal[1], epsilon, ze, g,
		&edgeflux[0], &edgeflux[1], &edgeflux[2],
		&max_speed, &pressure_flux, low_froude
	)

	assert err >= 0, "Discontinuous Elevation"

	return max_speed, pressure_flux

def gravity(object domain_py_object, update_domain_c_struct=False):

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	err = _openmp_gravity(D)
	if err == -1:
		return None

def gravity_wb(object domain_py_object, update_domain_c_struct=False):

	cdef domain* D = get_domain_c_struct_ptr(domain_py_object, update_domain_c_struct=update_domain_c_struct)

	err = _openmp_gravity_wb(D)
	if err == -1:
		return None

