#cython: wraparound=False, boundscheck=False, cdivision=True, profile=False, nonecheck=False, overflowcheck=False, cdivision_warnings=False, unraisable_tracebacks=False, warn.unused=False
"""
Cython wrapper for GPU-accelerated shallow water solver with MPI support.

This module bridges Python (mpi4py, NumPy) to the C GPU implementation.
Key responsibilities:
- Extract MPI_Comm handle from mpi4py
- Convert Python domain arrays to C pointers
- Flatten Python ghost dicts to C arrays for GPU efficiency
"""

import cython
from libc.stdint cimport int64_t, int32_t, uint64_t
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

# Import mpi4py's MPI_Comm type
from mpi4py cimport MPI
from mpi4py.libmpi cimport MPI_Comm

# External C declarations (from gpu/ subdirectory)
cdef extern from "gpu_domain.h" nogil:
    # Forward declare the structs
    struct domain:
        int64_t number_of_elements
        int64_t boundary_length
        double epsilon
        double H0
        double g
        double minimum_allowed_height
        double maximum_allowed_speed
        double evolve_max_timestep
        int64_t low_froude
        int64_t extrapolate_velocity_second_order
        # Beta values for gradient limiting
        double beta_w
        double beta_w_dry
        double beta_uh
        double beta_uh_dry
        double beta_vh
        double beta_vh_dry
        # Centroid values
        double* stage_centroid_values
        double* xmom_centroid_values
        double* ymom_centroid_values
        double* bed_centroid_values
        double* height_centroid_values
        # Edge values
        double* stage_edge_values
        double* xmom_edge_values
        double* ymom_edge_values
        double* bed_edge_values
        double* height_edge_values
        # Updates
        double* stage_explicit_update
        double* xmom_explicit_update
        double* ymom_explicit_update
        double* stage_semi_implicit_update
        double* xmom_semi_implicit_update
        double* ymom_semi_implicit_update
        # Boundary values
        double* stage_boundary_values
        double* xmom_boundary_values
        double* ymom_boundary_values
        double* bed_boundary_values
        double* height_boundary_values
        # Backup for RK2
        double* stage_backup_values
        double* xmom_backup_values
        double* ymom_backup_values
        # Mesh connectivity
        int64_t* neighbours
        int64_t* neighbour_edges
        int64_t* surrogate_neighbours
        int64_t* number_of_boundaries
        double* normals
        double* edgelengths
        double* areas
        double* radii
        double* max_speed
        double* centroid_coordinates
        double* edge_coordinates
        # Work arrays for extrapolation
        double* x_centroid_work
        double* y_centroid_work
        # Friction
        double* friction_centroid_values

    struct halo_exchange:
        int num_neighbors
        int* neighbor_ranks
        int* send_counts
        int* recv_counts
        int total_send_size
        int total_recv_size
        int* flat_send_indices
        int* flat_recv_indices
        double* send_buffer
        double* recv_buffer

    struct gpu_domain:
        domain D
        MPI_Comm comm
        int rank
        int nprocs
        int gpu_initialized
        int device_id
        int gpu_aware_mpi
        halo_exchange halo
        double CFL
        double evolve_max_timestep

    # Function declarations - initialization and cleanup
    int gpu_domain_init(gpu_domain *GD, MPI_Comm comm, int rank, int nprocs)
    void gpu_domain_finalize(gpu_domain *GD)
    int gpu_halo_init(gpu_domain *GD, int num_neighbors, int *neighbor_ranks,
                      int *send_counts, int *recv_counts,
                      int *flat_send_indices, int *flat_recv_indices)
    void gpu_halo_finalize(gpu_domain *GD)

    # GPU memory management
    void gpu_domain_map_arrays(gpu_domain *GD)
    void gpu_remap_boundary_arrays(gpu_domain *GD)
    void gpu_domain_unmap_arrays(gpu_domain *GD)
    void gpu_domain_sync_to_device(gpu_domain *GD)
    void gpu_domain_sync_from_device(gpu_domain *GD)
    void gpu_domain_sync_all_from_device(gpu_domain *GD)
    # Partial sync for sparse triangle updates (Inlet_operator optimization)
    void gpu_domain_sync_partial_from_device(gpu_domain *GD,
                                              int *indices, int num_indices,
                                              double *stage_buf, double *xmom_buf,
                                              double *ymom_buf, double *height_buf)
    void gpu_domain_sync_partial_to_device(gpu_domain *GD,
                                            int *indices, int num_indices,
                                            double *stage_buf, double *xmom_buf,
                                            double *ymom_buf, double *height_buf)
    void gpu_sync_boundary_values(gpu_domain *GD)
    void gpu_sync_edge_values_from_device(gpu_domain *GD)
    int gpu_boundary_edge_sync_init(gpu_domain *GD, int num_boundary_cells, int *boundary_cell_ids)
    void gpu_boundary_edge_sync_finalize(gpu_domain *GD)
    void gpu_boundary_edge_sync(gpu_domain *GD)

    # MPI ghost exchange
    void gpu_exchange_ghosts(gpu_domain *GD)

    # Reflective boundary
    int gpu_reflective_init(gpu_domain *GD, int num_edges,
                            int *boundary_indices, int *vol_ids, int *edge_ids)
    void gpu_reflective_finalize(gpu_domain *GD)
    void gpu_evaluate_reflective_boundary(gpu_domain *GD)

    # Dirichlet boundary
    int gpu_dirichlet_init(gpu_domain *GD, int num_edges,
                           int *boundary_indices, int *vol_ids, int *edge_ids,
                           double stage_value, double xmom_value, double ymom_value)
    void gpu_dirichlet_finalize(gpu_domain *GD)
    void gpu_evaluate_dirichlet_boundary(gpu_domain *GD)

    # Transmissive boundary
    int gpu_transmissive_init(gpu_domain *GD, int num_edges,
                              int *boundary_indices, int *vol_ids, int *edge_ids,
                              int use_centroid)
    void gpu_transmissive_finalize(gpu_domain *GD)
    void gpu_evaluate_transmissive_boundary(gpu_domain *GD)

    # Transmissive_n_momentum_zero_t_momentum_set_stage boundary
    int gpu_transmissive_n_zero_t_init(gpu_domain *GD, int num_edges,
                                       int *boundary_indices, int *vol_ids, int *edge_ids)
    void gpu_transmissive_n_zero_t_finalize(gpu_domain *GD)
    void gpu_transmissive_n_zero_t_set_stage(gpu_domain *GD, double stage_value)
    void gpu_evaluate_transmissive_n_zero_t_boundary(gpu_domain *GD)

    # Time_boundary - time-dependent Dirichlet values
    int gpu_time_boundary_init(gpu_domain *GD, int num_edges,
                               int *boundary_indices, int *vol_ids, int *edge_ids)
    void gpu_time_boundary_finalize(gpu_domain *GD)
    void gpu_time_boundary_set_values(gpu_domain *GD, double stage, double xmom, double ymom)
    void gpu_evaluate_time_boundary(gpu_domain *GD)

    # GPU kernels
    void gpu_extrapolate_second_order(gpu_domain *GD)
    double gpu_compute_fluxes(gpu_domain *GD)
    void gpu_update_conserved_quantities(gpu_domain *GD, double timestep)
    void gpu_backup_conserved_quantities(gpu_domain *GD)
    void gpu_saxpy_conserved_quantities(gpu_domain *GD, double a, double b)
    double gpu_protect(gpu_domain *GD)
    double gpu_compute_water_volume(gpu_domain *GD)
    void gpu_manning_friction(gpu_domain *GD)

    # Full RK2 step
    double gpu_evolve_one_rk2_step(gpu_domain *GD, double max_timestep, int apply_forcing)
    void print_gpu_domain_info(gpu_domain *GD)

    # Rate operators (rain, extraction, etc.)
    int gpu_rate_operator_init(gpu_domain *GD, int num_indices, int *indices,
                               double *areas, int *full_indices, int num_full)
    void gpu_rate_operator_finalize(gpu_domain *GD, int op_id)
    void gpu_rate_operators_finalize_all(gpu_domain *GD)
    double gpu_rate_operator_apply(gpu_domain *GD, int op_id,
                                   double rate, double factor, double timestep)
    double gpu_rate_operator_apply_array(gpu_domain *GD, int op_id,
                                         double *rate_array, int rate_array_size,
                                         int use_indices_into_rate,
                                         int rate_changed,
                                         double factor, double timestep)

    # FLOP counters (Gordon Bell performance profiling)
    void gpu_flop_counters_init(gpu_domain *GD)
    void gpu_flop_counters_reset(gpu_domain *GD)
    void gpu_flop_counters_enable(gpu_domain *GD, int enable)
    void gpu_flop_counters_start_timer(gpu_domain *GD)
    void gpu_flop_counters_stop_timer(gpu_domain *GD)
    uint64_t gpu_flop_counters_get_total(gpu_domain *GD)
    double gpu_flop_counters_get_flops(gpu_domain *GD)
    void gpu_flop_counters_print(gpu_domain *GD)
    uint64_t gpu_flop_counters_get_extrapolate(gpu_domain *GD)
    uint64_t gpu_flop_counters_get_compute_fluxes(gpu_domain *GD)
    uint64_t gpu_flop_counters_get_update(gpu_domain *GD)
    uint64_t gpu_flop_counters_get_protect(gpu_domain *GD)
    uint64_t gpu_flop_counters_get_manning(gpu_domain *GD)
    uint64_t gpu_flop_counters_get_backup(gpu_domain *GD)
    uint64_t gpu_flop_counters_get_saxpy(gpu_domain *GD)
    uint64_t gpu_flop_counters_get_rate_operator(gpu_domain *GD)
    uint64_t gpu_flop_counters_get_ghost_exchange(gpu_domain *GD)
    # MPI reduction for multi-GPU (Gordon Bell)
    uint64_t gpu_flop_counters_get_global_total(gpu_domain *GD)
    double gpu_flop_counters_get_global_flops(gpu_domain *GD)
    void gpu_flop_counters_print_global(gpu_domain *GD)


# ============================================================================
# GPU Domain Wrapper Class
# ============================================================================

cdef class GPUDomain:
    """
    Wrapper class holding the C gpu_domain struct.
    This provides a Python-accessible handle to the GPU domain state.
    """
    cdef gpu_domain GD
    cdef bint initialized
    cdef object python_domain  # Keep reference to prevent GC

    def __cinit__(self):
        self.initialized = False
        self.python_domain = None

    def __dealloc__(self):
        if self.initialized:
            gpu_domain_finalize(&self.GD)

    @property
    def rank(self):
        return self.GD.rank

    @property
    def nprocs(self):
        return self.GD.nprocs

    @property
    def gpu_initialized(self):
        return self.GD.gpu_initialized

    @property
    def device_id(self):
        return self.GD.device_id

    @property
    def gpu_aware_mpi(self):
        return self.GD.gpu_aware_mpi

    @property
    def num_neighbors(self):
        return self.GD.halo.num_neighbors


# ============================================================================
# MPI Communicator Extraction
# ============================================================================

cdef MPI_Comm get_mpi_comm():
    """
    Extract the raw MPI_Comm handle from mpi4py.

    mpi4py exposes the C MPI_Comm via the .ob_mpi attribute on Comm objects.
    This is the key bridge between Python MPI and C MPI calls.
    """
    import anuga.utilities.parallel_abstraction as pypar
    cdef MPI.Comm comm = pypar.comm
    return comm.ob_mpi


cdef int get_mpi_rank():
    """Get current MPI rank."""
    import anuga.utilities.parallel_abstraction as pypar
    # Use function if available (works in both MPI and non-MPI modes)
    if hasattr(pypar, 'rank'):
        return pypar.rank()
    return pypar.myid


cdef int get_mpi_size():
    """Get total number of MPI processes."""
    import anuga.utilities.parallel_abstraction as pypar
    # Use function if available (works in both MPI and non-MPI modes)
    if hasattr(pypar, 'size'):
        return pypar.size()
    return pypar.numprocs


# ============================================================================
# Domain Array Pointer Extraction
# ============================================================================

cdef void get_domain_pointers(gpu_domain *GD, object domain_object):
    """
    Extract NumPy array pointers from Python domain object.

    This mirrors the pattern from sw_domain_openmp_ext.pyx but targets
    the gpu_domain struct.
    """
    cdef domain *D = &GD.D

    # Typed memoryview declarations
    cdef double[::1] stage_cv, xmom_cv, ymom_cv, bed_cv, height_cv
    cdef double[:,::1] stage_ev, xmom_ev, ymom_ev, bed_ev, height_ev
    cdef double[::1] stage_eu, xmom_eu, ymom_eu
    cdef double[::1] stage_siu, xmom_siu, ymom_siu
    cdef double[::1] stage_bv, xmom_bv, ymom_bv
    cdef double[::1] stage_backup, xmom_backup, ymom_backup
    cdef int64_t[:,::1] neighbours, neighbour_edges, surrogate_neighbours
    cdef int64_t[::1] number_of_boundaries
    cdef double[:,::1] normals, edgelengths
    cdef double[::1] areas, radii, max_speed
    cdef double[:,::1] centroid_coords, edge_coords
    cdef double[::1] x_centroid_work, y_centroid_work

    # Get basic parameters
    D.number_of_elements = domain_object.number_of_elements
    D.boundary_length = domain_object.boundary_length
    D.epsilon = domain_object.epsilon
    D.H0 = domain_object.H0
    D.g = domain_object.g
    D.minimum_allowed_height = domain_object.minimum_allowed_height
    D.maximum_allowed_speed = domain_object.maximum_allowed_speed
    D.evolve_max_timestep = domain_object.evolve_max_timestep

    # Copy CFL and evolve_max_timestep to GPU domain struct (used by C RK2 loop)
    GD.CFL = domain_object.CFL
    GD.evolve_max_timestep = domain_object.evolve_max_timestep
    D.low_froude = domain_object.low_froude
    D.extrapolate_velocity_second_order = domain_object.extrapolate_velocity_second_order

    # Beta values for gradient limiting
    D.beta_w = domain_object.beta_w
    D.beta_w_dry = domain_object.beta_w_dry
    D.beta_uh = domain_object.beta_uh
    D.beta_uh_dry = domain_object.beta_uh_dry
    D.beta_vh = domain_object.beta_vh
    D.beta_vh_dry = domain_object.beta_vh_dry

    # Get quantities dict
    quantities = domain_object.quantities

    # Stage
    stage = quantities["stage"]
    stage_cv = stage.centroid_values
    D.stage_centroid_values = &stage_cv[0]
    stage_ev = stage.edge_values
    D.stage_edge_values = &stage_ev[0, 0]
    stage_eu = stage.explicit_update
    D.stage_explicit_update = &stage_eu[0]
    stage_siu = stage.semi_implicit_update
    D.stage_semi_implicit_update = &stage_siu[0]
    stage_bv = stage.boundary_values
    D.stage_boundary_values = &stage_bv[0]

    # X-momentum
    xmom = quantities["xmomentum"]
    xmom_cv = xmom.centroid_values
    D.xmom_centroid_values = &xmom_cv[0]
    xmom_ev = xmom.edge_values
    D.xmom_edge_values = &xmom_ev[0, 0]
    xmom_eu = xmom.explicit_update
    D.xmom_explicit_update = &xmom_eu[0]
    xmom_siu = xmom.semi_implicit_update
    D.xmom_semi_implicit_update = &xmom_siu[0]
    xmom_bv = xmom.boundary_values
    D.xmom_boundary_values = &xmom_bv[0]

    # Y-momentum
    ymom = quantities["ymomentum"]
    ymom_cv = ymom.centroid_values
    D.ymom_centroid_values = &ymom_cv[0]
    ymom_ev = ymom.edge_values
    D.ymom_edge_values = &ymom_ev[0, 0]
    ymom_eu = ymom.explicit_update
    D.ymom_explicit_update = &ymom_eu[0]
    ymom_siu = ymom.semi_implicit_update
    D.ymom_semi_implicit_update = &ymom_siu[0]
    ymom_bv = ymom.boundary_values
    D.ymom_boundary_values = &ymom_bv[0]

    # Elevation (bed)
    elev = quantities["elevation"]
    bed_cv = elev.centroid_values
    D.bed_centroid_values = &bed_cv[0]
    bed_ev = elev.edge_values
    D.bed_edge_values = &bed_ev[0, 0]
    cdef double[::1] bed_bv
    bed_bv = elev.boundary_values
    D.bed_boundary_values = &bed_bv[0]

    # Height
    height = quantities["height"]
    height_cv = height.centroid_values
    D.height_centroid_values = &height_cv[0]
    height_ev = height.edge_values
    D.height_edge_values = &height_ev[0, 0]
    cdef double[::1] height_bv
    height_bv = height.boundary_values
    D.height_boundary_values = &height_bv[0]

    # Mesh connectivity
    neighbours = domain_object.neighbours
    D.neighbours = &neighbours[0, 0]

    neighbour_edges = domain_object.neighbour_edges
    D.neighbour_edges = &neighbour_edges[0, 0]

    surrogate_neighbours = domain_object.surrogate_neighbours
    D.surrogate_neighbours = &surrogate_neighbours[0, 0]

    number_of_boundaries = domain_object.number_of_boundaries
    D.number_of_boundaries = &number_of_boundaries[0]

    normals = domain_object.normals
    D.normals = &normals[0, 0]

    edgelengths = domain_object.edgelengths
    D.edgelengths = &edgelengths[0, 0]

    areas = domain_object.areas
    D.areas = &areas[0]

    radii = domain_object.radii
    D.radii = &radii[0]

    max_speed = domain_object.max_speed
    D.max_speed = &max_speed[0]

    centroid_coords = domain_object.centroid_coordinates
    D.centroid_coordinates = &centroid_coords[0, 0]

    edge_coords = domain_object.edge_coordinates
    D.edge_coordinates = &edge_coords[0, 0]

    # Work arrays for extrapolation
    x_centroid_work = domain_object.x_centroid_work
    D.x_centroid_work = &x_centroid_work[0]

    y_centroid_work = domain_object.y_centroid_work
    D.y_centroid_work = &y_centroid_work[0]

    # Friction
    cdef double[::1] friction_cv
    friction = quantities["friction"]
    friction_cv = friction.centroid_values
    D.friction_centroid_values = &friction_cv[0]

    # Backup arrays for RK2 - these are on each Quantity object
    cdef double[::1] stage_backup_cv, xmom_backup_cv, ymom_backup_cv
    stage_backup_cv = stage.centroid_backup_values
    xmom_backup_cv = xmom.centroid_backup_values
    ymom_backup_cv = ymom.centroid_backup_values
    D.stage_backup_values = &stage_backup_cv[0]
    D.xmom_backup_values = &xmom_backup_cv[0]
    D.ymom_backup_values = &ymom_backup_cv[0]


# ============================================================================
# Halo Structure Building
# ============================================================================

cdef void build_halo_from_dicts(gpu_domain *GD, object domain_object):
    """
    Convert Python ghost dictionaries to flattened C arrays for GPU.

    ANUGA stores ghost exchange info in:
    - domain.full_send_dict[proc] = [indices, global_ids, buffer]
    - domain.ghost_recv_dict[proc] = [indices, global_ids, buffer]

    We flatten these to contiguous arrays for efficient GPU access.
    """
    full_send_dict = domain_object.full_send_dict
    ghost_recv_dict = domain_object.ghost_recv_dict

    # Count neighbors (excluding self)
    my_rank = GD.rank
    neighbors = []
    for proc in full_send_dict:
        if proc != my_rank:
            neighbors.append(proc)

    cdef int num_neighbors = len(neighbors)
    if num_neighbors == 0:
        return

    # Allocate temporary arrays
    cdef np.ndarray[int, ndim=1] neighbor_ranks = np.array(neighbors, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] send_counts = np.zeros(num_neighbors, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] recv_counts = np.zeros(num_neighbors, dtype=np.int32)

    # Count send/recv sizes
    cdef int total_send = 0
    cdef int total_recv = 0
    cdef int ni
    for ni in range(num_neighbors):
        proc = neighbor_ranks[ni]
        send_counts[ni] = len(full_send_dict[proc][0])
        recv_counts[ni] = len(ghost_recv_dict[proc][0])
        total_send += send_counts[ni]
        total_recv += recv_counts[ni]

    # Flatten indices
    cdef np.ndarray[int, ndim=1] flat_send = np.zeros(total_send, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] flat_recv = np.zeros(total_recv, dtype=np.int32)

    cdef int send_offset = 0
    cdef int recv_offset = 0
    cdef int j, idx

    for ni in range(num_neighbors):
        proc = neighbor_ranks[ni]

        # Copy send indices
        send_indices = full_send_dict[proc][0]
        for j in range(send_counts[ni]):
            flat_send[send_offset + j] = send_indices[j]
        send_offset += send_counts[ni]

        # Copy recv indices
        recv_indices = ghost_recv_dict[proc][0]
        for j in range(recv_counts[ni]):
            flat_recv[recv_offset + j] = recv_indices[j]
        recv_offset += recv_counts[ni]

    # Call C function to initialize halo
    gpu_halo_init(GD, num_neighbors,
                  <int*>neighbor_ranks.data,
                  <int*>send_counts.data,
                  <int*>recv_counts.data,
                  <int*>flat_send.data,
                  <int*>flat_recv.data)


# ============================================================================
# Public Python API
# ============================================================================

def init_gpu_domain(object domain_object):
    """
    Initialize GPU domain from Python domain object.

    This is the main entry point called from Python. It:
    1. Extracts MPI_Comm from mpi4py
    2. Creates gpu_domain struct with array pointers
    3. Builds flattened halo exchange structures
    4. Returns a GPUDomain wrapper object

    Parameters
    ----------
    domain_object : anuga.shallow_water.Domain
        The Python domain object (must be already partitioned for MPI)

    Returns
    -------
    GPUDomain
        Wrapper object holding the GPU domain state
    """
    cdef MPI_Comm comm = get_mpi_comm()
    cdef int rank = get_mpi_rank()
    cdef int nprocs = get_mpi_size()

    # Create wrapper object
    gpu_dom = GPUDomain()

    # Initialize C struct with MPI info
    gpu_domain_init(&gpu_dom.GD, comm, rank, nprocs)

    # Extract array pointers from Python domain
    get_domain_pointers(&gpu_dom.GD, domain_object)

    # Build halo exchange structures from ghost dicts
    if hasattr(domain_object, 'full_send_dict') and domain_object.full_send_dict:
        build_halo_from_dicts(&gpu_dom.GD, domain_object)

    # Keep reference to Python domain to prevent GC of arrays
    gpu_dom.python_domain = domain_object
    gpu_dom.initialized = True

    if rank == 0:
        print(f"GPU domain initialized: {gpu_dom.GD.D.number_of_elements} elements")
        print_gpu_domain_info(&gpu_dom.GD)

    return gpu_dom


def map_to_gpu(GPUDomain gpu_dom):
    """
    Map domain arrays to GPU memory.

    Call this once after init_gpu_domain, before starting the evolve loop.
    """
    gpu_domain_map_arrays(&gpu_dom.GD)


def remap_boundary_arrays(GPUDomain gpu_dom):
    """
    Remap boundary arrays to GPU memory.

    Call this after boundaries are initialized (if they weren't initialized
    before map_to_gpu was called).
    """
    gpu_remap_boundary_arrays(&gpu_dom.GD)


def unmap_from_gpu(GPUDomain gpu_dom):
    """
    Unmap domain arrays from GPU memory.

    Call this when done with GPU computation.
    """
    gpu_domain_unmap_arrays(&gpu_dom.GD)


def sync_to_device(GPUDomain gpu_dom):
    """
    Sync centroid values from host to device.

    Call this after Python modifies arrays (e.g., boundary updates).
    """
    gpu_domain_sync_to_device(&gpu_dom.GD)


def sync_from_device(GPUDomain gpu_dom):
    """
    Sync centroid values from device to host.

    Call this before Python needs to read arrays (e.g., at yieldstep for I/O).
    """
    gpu_domain_sync_from_device(&gpu_dom.GD)


def sync_all_from_device(GPUDomain gpu_dom):
    """
    Sync ALL arrays from device to host (for debugging/testing).

    This syncs centroid_values, edge_values, boundary_values,
    explicit_update, semi_implicit_update, and backup_values.
    Use this when you need to inspect intermediate GPU values.
    """
    gpu_domain_sync_all_from_device(&gpu_dom.GD)


def sync_partial_from_device(GPUDomain gpu_dom,
                              np.ndarray[int, ndim=1, mode="c"] indices,
                              np.ndarray[double, ndim=1, mode="c"] stage_buf,
                              np.ndarray[double, ndim=1, mode="c"] xmom_buf,
                              np.ndarray[double, ndim=1, mode="c"] ymom_buf,
                              np.ndarray[double, ndim=1, mode="c"] height_buf):
    """
    Sync specific triangle centroid values FROM GPU to host buffers.

    This is much more efficient than sync_from_device when only a small
    subset of triangles need to be synced (e.g., for Inlet_operator).

    Parameters
    ----------
    gpu_dom : GPUDomain
        The GPU domain wrapper
    indices : ndarray[int]
        Triangle indices to sync
    stage_buf, xmom_buf, ymom_buf, height_buf : ndarray[double]
        Output buffers (must be pre-allocated with size len(indices))
    """
    cdef int num_indices = len(indices)
    if num_indices == 0:
        return
    gpu_domain_sync_partial_from_device(&gpu_dom.GD,
                                         &indices[0], num_indices,
                                         &stage_buf[0], &xmom_buf[0],
                                         &ymom_buf[0], &height_buf[0])


def sync_partial_to_device(GPUDomain gpu_dom,
                            np.ndarray[int, ndim=1, mode="c"] indices,
                            np.ndarray[double, ndim=1, mode="c"] stage_buf,
                            np.ndarray[double, ndim=1, mode="c"] xmom_buf,
                            np.ndarray[double, ndim=1, mode="c"] ymom_buf,
                            np.ndarray[double, ndim=1, mode="c"] height_buf):
    """
    Sync specific triangle centroid values TO GPU from host buffers.

    This is much more efficient than sync_to_device when only a small
    subset of triangles need to be synced (e.g., for Inlet_operator).

    Parameters
    ----------
    gpu_dom : GPUDomain
        The GPU domain wrapper
    indices : ndarray[int]
        Triangle indices to sync
    stage_buf, xmom_buf, ymom_buf, height_buf : ndarray[double]
        Input buffers with values to write to GPU
    """
    cdef int num_indices = len(indices)
    if num_indices == 0:
        return
    gpu_domain_sync_partial_to_device(&gpu_dom.GD,
                                       &indices[0], num_indices,
                                       &stage_buf[0], &xmom_buf[0],
                                       &ymom_buf[0], &height_buf[0])


def sync_boundary_values(GPUDomain gpu_dom):
    """
    Sync boundary values from host to device.

    Call this after Python updates boundary conditions.
    """
    gpu_sync_boundary_values(&gpu_dom.GD)


def sync_edge_values_from_device(GPUDomain gpu_dom):
    """
    Sync ALL edge values from device to host.

    Call this before CPU boundary evaluation needs to read edge values.
    WARNING: This is expensive - use sync_boundary_edge_values for sparse sync.
    """
    gpu_sync_edge_values_from_device(&gpu_dom.GD)


def init_boundary_edge_sync(GPUDomain gpu_dom, np.ndarray[int, ndim=1, mode="c"] boundary_cell_ids):
    """
    Initialize boundary edge sync buffers - call once after boundaries are set.

    This pre-allocates staging buffers on GPU for efficient sparse sync.

    Parameters
    ----------
    gpu_dom : GPUDomain
        The GPU domain wrapper
    boundary_cell_ids : ndarray
        Unique cell IDs that are adjacent to boundaries
    """
    cdef int num_cells = len(boundary_cell_ids)
    cdef int result = gpu_boundary_edge_sync_init(&gpu_dom.GD, num_cells, &boundary_cell_ids[0])
    if result != 0:
        raise RuntimeError("Failed to initialize boundary edge sync")


def boundary_edge_sync(GPUDomain gpu_dom):
    """
    Sync edge values for boundary-adjacent cells from GPU to host.

    Fast operation - uses pre-allocated buffers. Call init_boundary_edge_sync first.
    Call this before CPU boundary evaluation needs to read edge values.
    """
    gpu_boundary_edge_sync(&gpu_dom.GD)


def exchange_ghosts(GPUDomain gpu_dom):
    """
    Exchange ghost cell data between MPI ranks.

    This uses GPU-aware MPI if available, otherwise does efficient
    D2H/H2D transfers of only the small halo buffers.
    """
    gpu_exchange_ghosts(&gpu_dom.GD)


def init_reflective_boundary(GPUDomain gpu_dom, object domain_object):
    """
    Initialize reflective boundary for GPU evaluation.

    Extracts reflective boundary info from domain and maps to GPU.
    Call this once after domain setup.

    Parameters
    ----------
    gpu_dom : GPUDomain
        The GPU domain wrapper
    domain_object : Domain
        The Python domain object
    """
    cdef int num_edges = 0
    cdef np.ndarray[int, ndim=1, mode="c"] boundary_indices
    cdef np.ndarray[int, ndim=1, mode="c"] vol_ids_arr
    cdef np.ndarray[int, ndim=1, mode="c"] edge_ids_arr

    # Collect all reflective boundary edges (may have multiple tags)
    # boundary_map may not be set up yet if called early
    if domain_object.boundary_map is None:
        return

    all_ids = []
    for tag, boundary in domain_object.boundary_map.items():
        if boundary is not None and boundary.__class__.__name__ == 'Reflective_boundary':
            segment_edges = domain_object.tag_boundary_cells.get(tag, None)
            if segment_edges is not None and len(segment_edges) > 0:
                all_ids.extend(segment_edges)

    if len(all_ids) == 0:
        # No reflective boundary, nothing to do
        return

    # Build arrays
    ids = np.array(all_ids, dtype=np.intc)
    num_edges = len(ids)
    boundary_indices = ids
    vol_ids_arr = np.ascontiguousarray(domain_object.boundary_cells[ids], dtype=np.intc)
    edge_ids_arr = np.ascontiguousarray(domain_object.boundary_edges[ids], dtype=np.intc)

    # Call C init
    gpu_reflective_init(&gpu_dom.GD, num_edges,
                        &boundary_indices[0], &vol_ids_arr[0], &edge_ids_arr[0])


def evaluate_reflective_boundary_gpu(GPUDomain gpu_dom):
    """
    Evaluate reflective boundary on GPU.

    This reads edge values and writes boundary values entirely on device.
    No data transfer required.
    """
    gpu_evaluate_reflective_boundary(&gpu_dom.GD)


def init_dirichlet_boundary(GPUDomain gpu_dom, object domain_object):
    """
    Initialize Dirichlet boundary for GPU evaluation.

    Extracts Dirichlet boundary info from domain and prepares for GPU mapping.
    Call this once after domain setup, BEFORE map_to_gpu.
    """
    cdef int num_edges = 0
    cdef np.ndarray[int, ndim=1, mode="c"] boundary_indices
    cdef np.ndarray[int, ndim=1, mode="c"] vol_ids_arr
    cdef np.ndarray[int, ndim=1, mode="c"] edge_ids_arr
    cdef double stage_value = 0.0
    cdef double xmom_value = 0.0
    cdef double ymom_value = 0.0

    if domain_object.boundary_map is None:
        return

    # Find Dirichlet boundaries - note: all Dirichlet boundaries must have same values
    # for GPU evaluation (limitation - could be extended to per-tag values)
    all_ids = []
    for tag, boundary in domain_object.boundary_map.items():
        if boundary is not None and boundary.__class__.__name__ == 'Dirichlet_boundary':
            segment_edges = domain_object.tag_boundary_cells.get(tag, None)
            if segment_edges is not None and len(segment_edges) > 0:
                all_ids.extend(segment_edges)
                # Get Dirichlet values from first boundary found
                if hasattr(boundary, 'dirichlet_values') and len(boundary.dirichlet_values) >= 3:
                    stage_value = float(boundary.dirichlet_values[0])
                    xmom_value = float(boundary.dirichlet_values[1])
                    ymom_value = float(boundary.dirichlet_values[2])

    if len(all_ids) == 0:
        return

    ids = np.array(all_ids, dtype=np.intc)
    num_edges = len(ids)
    boundary_indices = ids
    vol_ids_arr = np.ascontiguousarray(domain_object.boundary_cells[ids], dtype=np.intc)
    edge_ids_arr = np.ascontiguousarray(domain_object.boundary_edges[ids], dtype=np.intc)

    gpu_dirichlet_init(&gpu_dom.GD, num_edges,
                       &boundary_indices[0], &vol_ids_arr[0], &edge_ids_arr[0],
                       stage_value, xmom_value, ymom_value)


def evaluate_dirichlet_boundary_gpu(GPUDomain gpu_dom):
    """
    Evaluate Dirichlet boundary on GPU.

    Sets constant values at boundary, entirely on device.
    No data transfer required.
    """
    gpu_evaluate_dirichlet_boundary(&gpu_dom.GD)


def init_transmissive_boundary(GPUDomain gpu_dom, object domain_object):
    """
    Initialize Transmissive boundary for GPU evaluation.

    Extracts Transmissive boundary info from domain and prepares for GPU mapping.
    Call this once after domain setup, BEFORE map_to_gpu.
    """
    cdef int num_edges = 0
    cdef np.ndarray[int, ndim=1, mode="c"] boundary_indices
    cdef np.ndarray[int, ndim=1, mode="c"] vol_ids_arr
    cdef np.ndarray[int, ndim=1, mode="c"] edge_ids_arr
    cdef int use_centroid = 0

    if domain_object.boundary_map is None:
        return

    all_ids = []
    for tag, boundary in domain_object.boundary_map.items():
        if boundary is not None and boundary.__class__.__name__ == 'Transmissive_boundary':
            segment_edges = domain_object.tag_boundary_cells.get(tag, None)
            if segment_edges is not None and len(segment_edges) > 0:
                all_ids.extend(segment_edges)

    if len(all_ids) == 0:
        return

    # Check if domain uses centroid transmissive BC
    if hasattr(domain_object, 'centroid_transmissive_bc'):
        use_centroid = 1 if domain_object.centroid_transmissive_bc else 0

    ids = np.array(all_ids, dtype=np.intc)
    num_edges = len(ids)
    boundary_indices = ids
    vol_ids_arr = np.ascontiguousarray(domain_object.boundary_cells[ids], dtype=np.intc)
    edge_ids_arr = np.ascontiguousarray(domain_object.boundary_edges[ids], dtype=np.intc)

    gpu_transmissive_init(&gpu_dom.GD, num_edges,
                          &boundary_indices[0], &vol_ids_arr[0], &edge_ids_arr[0],
                          use_centroid)


def evaluate_transmissive_boundary_gpu(GPUDomain gpu_dom):
    """
    Evaluate Transmissive boundary on GPU.

    Copies interior values to boundary, entirely on device.
    No data transfer required.
    """
    gpu_evaluate_transmissive_boundary(&gpu_dom.GD)


def init_transmissive_n_zero_t_boundary(GPUDomain gpu_dom, object domain_object):
    """
    Initialize Transmissive_n_momentum_zero_t_momentum_set_stage boundary for GPU.

    This boundary type sets stage from a (potentially time-varying) function,
    keeps normal momentum, and zeros tangential momentum.

    Call this once after domain setup, BEFORE map_to_gpu.
    """
    cdef int num_edges = 0
    cdef np.ndarray[int, ndim=1, mode="c"] boundary_indices
    cdef np.ndarray[int, ndim=1, mode="c"] vol_ids_arr
    cdef np.ndarray[int, ndim=1, mode="c"] edge_ids_arr

    if domain_object.boundary_map is None:
        return

    all_ids = []
    for tag, boundary in domain_object.boundary_map.items():
        if boundary is not None and boundary.__class__.__name__ == 'Transmissive_n_momentum_zero_t_momentum_set_stage_boundary':
            segment_edges = domain_object.tag_boundary_cells.get(tag, None)
            if segment_edges is not None and len(segment_edges) > 0:
                all_ids.extend(segment_edges)

    if len(all_ids) == 0:
        return

    ids = np.array(all_ids, dtype=np.intc)
    num_edges = len(ids)
    boundary_indices = ids
    vol_ids_arr = np.ascontiguousarray(domain_object.boundary_cells[ids], dtype=np.intc)
    edge_ids_arr = np.ascontiguousarray(domain_object.boundary_edges[ids], dtype=np.intc)

    gpu_transmissive_n_zero_t_init(&gpu_dom.GD, num_edges,
                                   &boundary_indices[0], &vol_ids_arr[0], &edge_ids_arr[0])


def set_transmissive_n_zero_t_stage(GPUDomain gpu_dom, double stage_value):
    """
    Update the stage value for Transmissive_n_zero_t boundary.

    Call this each timestep before evaluate_transmissive_n_zero_t_boundary_gpu.
    """
    gpu_transmissive_n_zero_t_set_stage(&gpu_dom.GD, stage_value)


def evaluate_transmissive_n_zero_t_boundary_gpu(GPUDomain gpu_dom):
    """
    Evaluate Transmissive_n_zero_t boundary on GPU.

    Sets stage from external value, keeps normal momentum, zeros tangential.
    Call set_transmissive_n_zero_t_stage first to set the stage value.
    """
    gpu_evaluate_transmissive_n_zero_t_boundary(&gpu_dom.GD)


def init_time_boundary(GPUDomain gpu_dom, object domain_object):
    """
    Initialize Time_boundary for GPU.

    This boundary type sets conserved quantities from a time-dependent function.
    The function is called from Python each timestep (cheap), and the assignment
    happens on GPU (avoids D2H/H2D transfer overhead).

    Call this once after domain setup, BEFORE map_to_gpu.
    """
    cdef int num_edges = 0
    cdef np.ndarray[int, ndim=1, mode="c"] boundary_indices
    cdef np.ndarray[int, ndim=1, mode="c"] vol_ids_arr
    cdef np.ndarray[int, ndim=1, mode="c"] edge_ids_arr

    if domain_object.boundary_map is None:
        return

    all_ids = []
    for tag, boundary in domain_object.boundary_map.items():
        if boundary is not None and boundary.__class__.__name__ == 'Time_boundary':
            segment_edges = domain_object.tag_boundary_cells.get(tag, None)
            if segment_edges is not None and len(segment_edges) > 0:
                all_ids.extend(segment_edges)

    if len(all_ids) == 0:
        return

    ids = np.array(all_ids, dtype=np.intc)
    num_edges = len(ids)
    boundary_indices = ids
    vol_ids_arr = np.ascontiguousarray(domain_object.boundary_cells[ids], dtype=np.intc)
    edge_ids_arr = np.ascontiguousarray(domain_object.boundary_edges[ids], dtype=np.intc)

    gpu_time_boundary_init(&gpu_dom.GD, num_edges,
                           &boundary_indices[0], &vol_ids_arr[0], &edge_ids_arr[0])


def set_time_boundary_values(GPUDomain gpu_dom, double stage, double xmom, double ymom):
    """
    Update the values for Time_boundary.

    Call this each timestep before evaluate_time_boundary_gpu.
    The values come from calling the Python time-dependent function.
    """
    gpu_time_boundary_set_values(&gpu_dom.GD, stage, xmom, ymom)


def evaluate_time_boundary_gpu(GPUDomain gpu_dom):
    """
    Evaluate Time_boundary on GPU.

    Sets stage and momentum from time-dependent values.
    Call set_time_boundary_values first to set the values.
    """
    gpu_evaluate_time_boundary(&gpu_dom.GD)


def evolve_one_rk2_step_gpu(GPUDomain gpu_dom, double max_timestep, int apply_forcing):
    """
    Execute one RK2 timestep on GPU.

    Parameters
    ----------
    gpu_dom : GPUDomain
        The GPU domain wrapper
    max_timestep : float
        Maximum allowed timestep (respecting yieldstep/finaltime constraints)
    apply_forcing : int
        Whether to apply GPU-compatible forcing terms

    Returns
    -------
    float
        The timestep used
    """
    return gpu_evolve_one_rk2_step(&gpu_dom.GD, max_timestep, apply_forcing)


def finalize_gpu_domain(GPUDomain gpu_dom):
    """
    Clean up GPU domain resources.

    This is called automatically when GPUDomain is garbage collected,
    but can be called explicitly for deterministic cleanup.
    """
    if gpu_dom.initialized:
        gpu_domain_finalize(&gpu_dom.GD)
        gpu_dom.initialized = False


# ============================================================================
# GPU Kernel Wrappers
# ============================================================================

def extrapolate_second_order_gpu(GPUDomain gpu_dom):
    """
    Perform second-order edge extrapolation on GPU.

    This extrapolates centroid values to edge values using a limited
    gradient reconstruction. Handles wet-dry fronts with adaptive limiting.
    """
    gpu_extrapolate_second_order(&gpu_dom.GD)


def compute_fluxes_gpu(GPUDomain gpu_dom):
    """
    Compute fluxes across all edges on GPU.

    Uses the central upwind Kurganov-Noelle-Petrova scheme.

    Returns
    -------
    float
        The local minimum timestep (caller should do MPI_Allreduce for global min)
    """
    return gpu_compute_fluxes(&gpu_dom.GD)


def update_conserved_quantities_gpu(GPUDomain gpu_dom, double timestep):
    """
    Update conserved quantities using explicit and semi-implicit updates.

    Q_new = (Q_old + timestep * explicit_update) / (1 - timestep * semi_implicit_update)
    """
    gpu_update_conserved_quantities(&gpu_dom.GD, timestep)


def backup_conserved_quantities_gpu(GPUDomain gpu_dom):
    """
    Backup centroid values for RK2 timestepping.
    """
    gpu_backup_conserved_quantities(&gpu_dom.GD)


def saxpy_conserved_quantities_gpu(GPUDomain gpu_dom, double a, double b):
    """
    RK2 combination: Q = a*Q_current + b*Q_backup

    Typically called with a=0.5, b=0.5 for standard RK2.
    """
    gpu_saxpy_conserved_quantities(&gpu_dom.GD, a, b)


def protect_gpu(GPUDomain gpu_dom):
    """
    Protect against negative water depths.

    Sets stage = bed where water depth would be negative,
    and zeros momentum where depth is very small.

    Returns
    -------
    float
        Mass error (total volume added to prevent negative depths)
    """
    return gpu_protect(&gpu_dom.GD)


def compute_water_volume_gpu(GPUDomain gpu_dom):
    """
    Compute total water volume on GPU.

    Returns local volume - caller should do MPI_Allreduce for global sum.

    Returns
    -------
    float
        Local water volume (m^3)
    """
    return gpu_compute_water_volume(&gpu_dom.GD)


def manning_friction_gpu(GPUDomain gpu_dom):
    """
    Apply Manning friction on GPU (flat, semi-implicit formulation).

    Adds friction contribution to the semi_implicit_update arrays.
    The friction is then applied during update_conserved_quantities.
    """
    gpu_manning_friction(&gpu_dom.GD)


# ============================================================================
# Rate Operator GPU Support
# ============================================================================

def init_rate_operator(GPUDomain gpu_dom,
                       np.ndarray[int, ndim=1, mode="c"] indices,
                       np.ndarray[double, ndim=1, mode="c"] areas,
                       np.ndarray[int, ndim=1, mode="c"] full_indices=None):
    """
    Initialize a rate operator for GPU execution.

    Parameters
    ----------
    gpu_dom : GPUDomain
        The GPU domain wrapper
    indices : ndarray of int
        Triangle indices where rate is applied
    areas : ndarray of double
        Triangle areas (for mass tracking)
    full_indices : ndarray of int, optional
        Indices of "full" (non-ghost) triangles for mass tracking

    Returns
    -------
    int
        Operator ID (use this to apply rate or finalize)
        Returns -1 on error
    """
    cdef int num_indices = len(indices)
    cdef int num_full = 0
    cdef int *full_ptr = NULL

    if full_indices is not None:
        num_full = len(full_indices)
        full_ptr = &full_indices[0]

    return gpu_rate_operator_init(&gpu_dom.GD, num_indices, &indices[0],
                                  &areas[0], full_ptr, num_full)


def finalize_rate_operator(GPUDomain gpu_dom, int op_id):
    """
    Finalize and free a rate operator.

    Parameters
    ----------
    gpu_dom : GPUDomain
        The GPU domain wrapper
    op_id : int
        Operator ID returned by init_rate_operator
    """
    gpu_rate_operator_finalize(&gpu_dom.GD, op_id)


def finalize_all_rate_operators(GPUDomain gpu_dom):
    """
    Finalize all rate operators.

    Call this during domain cleanup.
    """
    gpu_rate_operators_finalize_all(&gpu_dom.GD)


def apply_rate_operator_gpu(GPUDomain gpu_dom, int op_id,
                            double rate, double factor, double timestep):
    """
    Apply rate operator on GPU.

    This is the GPU equivalent of Rate_operator.__call__().
    Handles both positive rates (rain) and negative rates (extraction).

    Parameters
    ----------
    gpu_dom : GPUDomain
        The GPU domain wrapper
    op_id : int
        Operator ID returned by init_rate_operator
    rate : double
        Rate value in m/s (scalar - get from Python function if time-dependent)
    factor : double
        Conversion factor
    timestep : double
        Current timestep

    Returns
    -------
    double
        Local mass influx (for mass conservation tracking)
    """
    return gpu_rate_operator_apply(&gpu_dom.GD, op_id, rate, factor, timestep)


def apply_rate_operator_array_gpu(GPUDomain gpu_dom, int op_id,
                                  np.ndarray[double, ndim=1, mode="c"] rate_array,
                                  int use_indices_into_rate,
                                  int rate_changed,
                                  double factor, double timestep):
    """
    Apply rate operator with per-cell rate array on GPU.

    This handles quantity-type rates where each cell has its own rate value.

    Parameters
    ----------
    gpu_dom : GPUDomain
        The GPU domain wrapper
    op_id : int
        Operator ID returned by init_rate_operator
    rate_array : ndarray of double
        Per-cell rate values
    use_indices_into_rate : int
        If 1, rate_array is full domain size (index with indices[k])
        If 0, rate_array matches operator indices size (index with k)
    rate_changed : int
        If 1, transfer new data to GPU; if 0, reuse cached data on GPU
    factor : double
        Conversion factor
    timestep : double
        Current timestep

    Returns
    -------
    double
        Local mass influx (for mass conservation tracking)
    """
    cdef int rate_size = len(rate_array)
    return gpu_rate_operator_apply_array(&gpu_dom.GD, op_id,
                                         &rate_array[0], rate_size,
                                         use_indices_into_rate,
                                         rate_changed,
                                         factor, timestep)


# ============================================================================
# FLOP Counter API (Gordon Bell Performance Profiling)
# ============================================================================

def flop_counters_reset(GPUDomain gpu_dom):
    """
    Reset all FLOP counters to zero.

    Call this at the start of the profiling period.
    """
    gpu_flop_counters_reset(&gpu_dom.GD)


def flop_counters_enable(GPUDomain gpu_dom, bint enable):
    """
    Enable or disable FLOP counting.

    Parameters
    ----------
    gpu_dom : GPUDomain
        The GPU domain wrapper
    enable : bool
        True to enable counting, False to disable
    """
    gpu_flop_counters_enable(&gpu_dom.GD, 1 if enable else 0)


def flop_counters_start_timer(GPUDomain gpu_dom):
    """
    Start the FLOP counter timer.

    Call this at the start of the profiling period.
    """
    gpu_flop_counters_start_timer(&gpu_dom.GD)


def flop_counters_stop_timer(GPUDomain gpu_dom):
    """
    Stop the FLOP counter timer.

    Call this at the end of the profiling period.
    """
    gpu_flop_counters_stop_timer(&gpu_dom.GD)


def flop_counters_get_total(GPUDomain gpu_dom):
    """
    Get total FLOP count across all kernels.

    Returns
    -------
    int
        Total FLOPs executed since last reset
    """
    return gpu_flop_counters_get_total(&gpu_dom.GD)


def flop_counters_get_flops(GPUDomain gpu_dom):
    """
    Get FLOP rate (FLOPs per second).

    Call flop_counters_stop_timer first to record elapsed time.

    Returns
    -------
    float
        FLOP/s rate
    """
    return gpu_flop_counters_get_flops(&gpu_dom.GD)


def flop_counters_print(GPUDomain gpu_dom):
    """
    Print detailed FLOP counter summary to stdout.

    Shows per-kernel breakdown and total performance in GFLOP/s.
    """
    gpu_flop_counters_print(&gpu_dom.GD)


def flop_counters_get_stats(GPUDomain gpu_dom):
    """
    Get all FLOP counter statistics as a dictionary.

    Returns
    -------
    dict
        Dictionary with per-kernel FLOPs, total, elapsed time, and GFLOP/s
    """
    cdef uint64_t total = gpu_flop_counters_get_total(&gpu_dom.GD)
    cdef double flops = gpu_flop_counters_get_flops(&gpu_dom.GD)

    return {
        'extrapolate': gpu_flop_counters_get_extrapolate(&gpu_dom.GD),
        'compute_fluxes': gpu_flop_counters_get_compute_fluxes(&gpu_dom.GD),
        'update': gpu_flop_counters_get_update(&gpu_dom.GD),
        'protect': gpu_flop_counters_get_protect(&gpu_dom.GD),
        'manning': gpu_flop_counters_get_manning(&gpu_dom.GD),
        'backup': gpu_flop_counters_get_backup(&gpu_dom.GD),
        'saxpy': gpu_flop_counters_get_saxpy(&gpu_dom.GD),
        'rate_operator': gpu_flop_counters_get_rate_operator(&gpu_dom.GD),
        'ghost_exchange': gpu_flop_counters_get_ghost_exchange(&gpu_dom.GD),
        'total_flops': total,
        'flops_per_second': flops,
        'gflops_per_second': flops / 1.0e9,
        'total_gflops': total / 1.0e9,
    }


# ============================================================================
# Global FLOP Counter API (MPI reduction for multi-GPU)
# ============================================================================

def flop_counters_get_global_total(GPUDomain gpu_dom):
    """
    Get global total FLOP count summed across all MPI ranks/GPUs.

    This performs an MPI_Allreduce to sum FLOPs from all ranks.

    Returns
    -------
    int
        Total FLOPs across all GPUs since last reset
    """
    return gpu_flop_counters_get_global_total(&gpu_dom.GD)


def flop_counters_get_global_flops(GPUDomain gpu_dom):
    """
    Get global FLOP rate (FLOPs per second) across all GPUs.

    Call flop_counters_stop_timer first to record elapsed time.

    Returns
    -------
    float
        Global FLOP/s rate (sum of all GPU FLOPs / elapsed time)
    """
    return gpu_flop_counters_get_global_flops(&gpu_dom.GD)


def flop_counters_print_global(GPUDomain gpu_dom):
    """
    Print global FLOP counter summary to stdout (rank 0 only).

    Shows per-kernel breakdown summed across all GPUs, with percentages.
    Includes total GFLOP/s and per-GPU average.
    """
    gpu_flop_counters_print_global(&gpu_dom.GD)


def flop_counters_get_global_stats(GPUDomain gpu_dom):
    """
    Get global FLOP counter statistics as a dictionary (MPI reduced).

    Returns
    -------
    dict
        Dictionary with global total, elapsed time, GFLOP/s, and per-GPU average
    """
    cdef uint64_t global_total = gpu_flop_counters_get_global_total(&gpu_dom.GD)
    cdef double global_flops = gpu_flop_counters_get_global_flops(&gpu_dom.GD)
    cdef int nprocs = gpu_dom.GD.nprocs

    return {
        'global_total_flops': global_total,
        'global_total_gflops': global_total / 1.0e9,
        'global_flops_per_second': global_flops,
        'global_gflops_per_second': global_flops / 1.0e9,
        'num_gpus': nprocs,
        'per_gpu_gflops_per_second': (global_flops / 1.0e9) / nprocs if nprocs > 0 else 0.0,
    }
