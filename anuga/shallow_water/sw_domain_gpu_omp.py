"""
GPU interface using OpenMP target offloading.

This provides a similar interface to sw_domain_cuda.py but uses
OpenMP target offloading via the sw_domain_gpu_ext Cython module.

Key differences from CUDA approach:
- Arrays stay in NumPy (OpenMP handles device mapping)
- No explicit array copies needed
- Uses MPI for multi-GPU (vs single-GPU CUDA approach)
"""

import numpy as np

class GPU_OMP_interface:
    """
    Interface for GPU-accelerated shallow water solver using OpenMP target offloading.

    Usage:
        domain.set_multiprocessor_mode(2)  # This will create the interface
        # ... normal evolve loop ...
    """

    def __init__(self, domain):
        """
        Initialize the GPU interface.

        Parameters
        ----------
        domain : anuga.shallow_water.Domain
            The domain object (should already be partitioned for MPI)
        """
        self.domain = domain
        self.gpu_dom = None
        self.initialized = False

    def setup(self):
        """
        Initialize GPU domain and map arrays to device.

        Call this once before starting the evolve loop.
        """
        if self.initialized:
            return

        from anuga.shallow_water.sw_domain_gpu_ext import (
            init_gpu_domain, map_to_gpu,
            init_reflective_boundary, init_dirichlet_boundary, init_transmissive_boundary,
            init_transmissive_n_zero_t_boundary, init_time_boundary
        )

        # Initialize GPU domain structure
        self.gpu_dom = init_gpu_domain(self.domain)

        # Initialize all supported boundary types BEFORE mapping to GPU
        # This allows boundary arrays to be mapped together with main arrays,
        # avoiding OpenMP data region issues
        if self.domain.boundary_map is not None:
            init_reflective_boundary(self.gpu_dom, self.domain)
            init_dirichlet_boundary(self.gpu_dom, self.domain)
            init_transmissive_boundary(self.gpu_dom, self.domain)
            init_transmissive_n_zero_t_boundary(self.gpu_dom, self.domain)
            init_time_boundary(self.gpu_dom, self.domain)

        # Map arrays to GPU memory (persistent for simulation)
        # This now includes all boundary arrays if initialized above
        map_to_gpu(self.gpu_dom)

        self.initialized = True

    def finalize(self):
        """
        Clean up GPU resources.

        Call this when done with GPU computation.
        """
        if not self.initialized:
            return

        from anuga.shallow_water.sw_domain_gpu_ext import (
            unmap_from_gpu, finalize_gpu_domain
        )

        unmap_from_gpu(self.gpu_dom)
        finalize_gpu_domain(self.gpu_dom)
        self.initialized = False

    def sync_to_device(self):
        """Sync centroid values from host to device."""
        from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device
        sync_to_device(self.gpu_dom)

    def sync_from_device(self):
        """Sync centroid values from device to host."""
        from anuga.shallow_water.sw_domain_gpu_ext import sync_from_device
        sync_from_device(self.gpu_dom)

    def sync_boundary_values(self):
        """Sync boundary values from host to device."""
        from anuga.shallow_water.sw_domain_gpu_ext import sync_boundary_values
        sync_boundary_values(self.gpu_dom)

    def exchange_ghosts(self):
        """Exchange ghost cells between MPI ranks."""
        from anuga.shallow_water.sw_domain_gpu_ext import exchange_ghosts
        exchange_ghosts(self.gpu_dom)

    # =========================================================================
    # Kernel wrappers - these match the interface expected by shallow_water_domain.py
    # =========================================================================

    def extrapolate_second_order_edge_sw_kernel(self, domain, distribute_to_vertices=False):
        """
        Extrapolate centroid values to edges.

        Note: distribute_to_vertices is ignored (we only do edge-based extrapolation)
        """
        from anuga.shallow_water.sw_domain_gpu_ext import extrapolate_second_order_gpu
        extrapolate_second_order_gpu(self.gpu_dom)

    def compute_fluxes_ext_central_kernel(self, domain, timestep):
        """
        Compute fluxes and return minimum timestep.

        Returns
        -------
        float
            Local minimum timestep (MPI allreduce needed for global min)
        """
        from anuga.shallow_water.sw_domain_gpu_ext import compute_fluxes_gpu
        return compute_fluxes_gpu(self.gpu_dom)

    def protect_against_infinitesimal_and_negative_heights_kernel(self, domain):
        """Protect against negative water depths."""
        from anuga.shallow_water.sw_domain_gpu_ext import protect_gpu
        return protect_gpu(self.gpu_dom)

    def update_conserved_quantities_kernel(self, domain, timestep):
        """Update conserved quantities with explicit and semi-implicit terms."""
        from anuga.shallow_water.sw_domain_gpu_ext import update_conserved_quantities_gpu
        update_conserved_quantities_gpu(self.gpu_dom, timestep)

    def backup_conserved_quantities_kernel(self, domain):
        """Backup centroid values for RK2."""
        from anuga.shallow_water.sw_domain_gpu_ext import backup_conserved_quantities_gpu
        backup_conserved_quantities_gpu(self.gpu_dom)

    def saxpy_conserved_quantities_kernel(self, domain, a, b):
        """RK2 combination: Q = a*Q_current + b*Q_backup."""
        from anuga.shallow_water.sw_domain_gpu_ext import saxpy_conserved_quantities_gpu
        saxpy_conserved_quantities_gpu(self.gpu_dom, a, b)

    def manning_friction_kernel(self, domain):
        """Apply Manning friction (semi-implicit)."""
        from anuga.shallow_water.sw_domain_gpu_ext import manning_friction_gpu
        manning_friction_gpu(self.gpu_dom)
