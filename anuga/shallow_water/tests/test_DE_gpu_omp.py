"""
Tests for GPU (OpenMP target offloading) implementation of ANUGA's shallow water solver.

These tests verify that the GPU implementation produces results matching the CPU implementation.
"""

import unittest
import sys
import numpy as np
import pytest

import anuga
from anuga import Reflective_boundary, Dirichlet_boundary
from anuga import rectangular_cross_domain
from anuga import Inlet_operator


def gpu_available():
    """Check if GPU OpenMP interface is available."""
    try:
        from anuga.shallow_water.sw_domain_gpu_ext import init_gpu_domain
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not gpu_available(), reason="GPU OpenMP interface not available")
class Test_GPU_Kernels(unittest.TestCase):
    """Unit tests for individual GPU kernels."""

    def setUp(self):
        """Create a simple test domain."""
        self.domain = rectangular_cross_domain(10, 10, len1=100., len2=100.)
        self.domain.set_flow_algorithm('DE0')
        self.domain.set_low_froude(0)
        self.domain.set_name('test_gpu')
        self.domain.set_datadir('.')
        self.domain.store = False

        def topography(x, y):
            return -x / 50.0  # Slope from 0 to -2

        self.domain.set_quantity('elevation', topography)
        self.domain.set_quantity('friction', 0.01)
        self.domain.set_quantity('stage', 0.0)

        Br = Reflective_boundary(self.domain)
        Bd = Dirichlet_boundary([-0.5, 0., 0.])
        self.domain.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})

    def tearDown(self):
        import os
        for ext in ['.sww']:
            try:
                os.remove(f'test_gpu{ext}')
            except:
                pass

    def test_flux_kernel(self):
        """Test that GPU flux computation matches CPU."""
        # Run CPU flux computation
        self.domain.set_multiprocessor_mode(1)
        self.domain.distribute_to_vertices_and_edges()
        self.domain.update_boundary()
        self.domain.compute_fluxes()

        cpu_timestep = self.domain.flux_timestep
        cpu_stage_update = self.domain.quantities['stage'].explicit_update.copy()
        cpu_xmom_update = self.domain.quantities['xmomentum'].explicit_update.copy()
        cpu_ymom_update = self.domain.quantities['ymomentum'].explicit_update.copy()

        # Reset and run GPU flux computation
        for qname in ['stage', 'xmomentum', 'ymomentum']:
            self.domain.quantities[qname].explicit_update[:] = 0.0

        self.domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import (
            sync_to_device, sync_all_from_device,
            extrapolate_second_order_gpu, compute_fluxes_gpu,
            evaluate_reflective_boundary_gpu, evaluate_dirichlet_boundary_gpu
        )

        gpu_dom = self.domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)
        extrapolate_second_order_gpu(gpu_dom)
        evaluate_reflective_boundary_gpu(gpu_dom)
        evaluate_dirichlet_boundary_gpu(gpu_dom)
        gpu_timestep = compute_fluxes_gpu(gpu_dom)
        sync_all_from_device(gpu_dom)

        gpu_stage_update = self.domain.quantities['stage'].explicit_update.copy()
        gpu_xmom_update = self.domain.quantities['xmomentum'].explicit_update.copy()
        gpu_ymom_update = self.domain.quantities['ymomentum'].explicit_update.copy()

        # Verify - use atol for near-zero values where rtol alone causes issues
        self.assertAlmostEqual(cpu_timestep, gpu_timestep, places=10,
                               msg=f"Timestep mismatch: CPU={cpu_timestep}, GPU={gpu_timestep}")
        np.testing.assert_allclose(cpu_stage_update, gpu_stage_update, rtol=1e-10, atol=1e-14,
                                   err_msg="Stage explicit_update mismatch")
        np.testing.assert_allclose(cpu_xmom_update, gpu_xmom_update, rtol=1e-10, atol=1e-14,
                                   err_msg="Xmomentum explicit_update mismatch")
        np.testing.assert_allclose(cpu_ymom_update, gpu_ymom_update, rtol=1e-10, atol=1e-14,
                                   err_msg="Ymomentum explicit_update mismatch")

    def test_extrapolate_kernel(self):
        """Test that GPU extrapolation matches CPU."""
        # Set initial conditions
        self.domain.set_quantity('stage', expression='0.1 * x / 100.0')

        # Run CPU extrapolation
        self.domain.set_multiprocessor_mode(1)
        self.domain.distribute_to_vertices_and_edges()

        cpu_stage_edge = self.domain.quantities['stage'].edge_values.copy()

        # Reset edge values and run GPU
        self.domain.quantities['stage'].edge_values[:] = 0.0

        self.domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import (
            sync_to_device, sync_all_from_device,
            protect_gpu, extrapolate_second_order_gpu
        )

        gpu_dom = self.domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)
        protect_gpu(gpu_dom)
        extrapolate_second_order_gpu(gpu_dom)
        sync_all_from_device(gpu_dom)

        gpu_stage_edge = self.domain.quantities['stage'].edge_values.copy()

        # Verify
        np.testing.assert_allclose(cpu_stage_edge, gpu_stage_edge, rtol=1e-10, atol=1e-14,
                                   err_msg="Edge values mismatch after extrapolation")


@pytest.mark.skipif(not gpu_available(), reason="GPU OpenMP interface not available")
class Test_GPU_RK2(unittest.TestCase):
    """Tests for complete RK2 step on GPU."""

    def create_domain(self):
        """Create a test domain."""
        domain = rectangular_cross_domain(10, 10, len1=100., len2=100.)
        domain.set_flow_algorithm('DE0')
        domain.set_low_froude(0)
        domain.set_name('test_rk2')
        domain.set_datadir('.')
        domain.store = False

        def topography(x, y):
            return -x / 50.0

        domain.set_quantity('elevation', topography)
        domain.set_quantity('friction', 0.01)
        domain.set_quantity('stage', 0.0)

        Br = Reflective_boundary(domain)
        Bd = Dirichlet_boundary([-0.5, 0., 0.])
        domain.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})

        return domain

    def test_single_rk2_step(self):
        """Test that a single RK2 step matches between CPU and GPU."""
        # Create two identical domains
        cpu_domain = self.create_domain()
        gpu_domain = self.create_domain()

        # Run CPU
        cpu_domain.set_multiprocessor_mode(1)
        for t in cpu_domain.evolve(yieldstep=0.1, finaltime=0.1):
            pass

        cpu_stage = cpu_domain.quantities['stage'].centroid_values.copy()

        # Run GPU
        gpu_domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device, sync_from_device

        gpu_dom = gpu_domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)

        for t in gpu_domain.evolve(yieldstep=0.1, finaltime=0.1):
            pass

        sync_from_device(gpu_dom)
        gpu_stage = gpu_domain.quantities['stage'].centroid_values.copy()

        # Compare - allow small tolerance for floating point differences
        diff = np.abs(cpu_stage - gpu_stage)
        self.assertLess(diff.max(), 1e-10,
                        f"Stage difference too large: max={diff.max():.2e}")

    def test_multi_step_evolution(self):
        """Test multiple RK2 steps match between CPU and GPU."""
        cpu_domain = self.create_domain()
        gpu_domain = self.create_domain()

        # Run CPU for 1 second
        cpu_domain.set_multiprocessor_mode(1)
        for t in cpu_domain.evolve(yieldstep=0.5, finaltime=1.0):
            pass

        cpu_stage = cpu_domain.quantities['stage'].centroid_values.copy()
        cpu_xmom = cpu_domain.quantities['xmomentum'].centroid_values.copy()

        # Run GPU for 1 second
        gpu_domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device, sync_from_device

        gpu_dom = gpu_domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)

        for t in gpu_domain.evolve(yieldstep=0.5, finaltime=1.0):
            pass

        sync_from_device(gpu_dom)
        gpu_stage = gpu_domain.quantities['stage'].centroid_values.copy()
        gpu_xmom = gpu_domain.quantities['xmomentum'].centroid_values.copy()

        # For longer simulations, allow slightly larger tolerance due to
        # floating point accumulation differences in parallel execution
        stage_diff = np.abs(cpu_stage - gpu_stage)
        xmom_diff = np.abs(cpu_xmom - gpu_xmom)

        self.assertLess(stage_diff.max(), 1e-8,
                        f"Stage difference too large: max={stage_diff.max():.2e}")
        self.assertLess(xmom_diff.max(), 1e-8,
                        f"Xmomentum difference too large: max={xmom_diff.max():.2e}")


@pytest.mark.skipif(not gpu_available(), reason="GPU OpenMP interface not available")
class Test_GPU_Boundaries(unittest.TestCase):
    """Tests for GPU boundary evaluation."""

    def test_reflective_boundary(self):
        """Test reflective boundary on GPU."""
        domain = rectangular_cross_domain(5, 5, len1=50., len2=50.)
        domain.set_flow_algorithm('DE0')
        domain.set_low_froude(0)
        domain.set_name('test_reflective')
        domain.set_datadir('.')
        domain.store = False

        domain.set_quantity('elevation', -1.0)
        domain.set_quantity('friction', 0.01)
        domain.set_quantity('stage', 0.5)  # Water above bed
        domain.set_quantity('xmomentum', 1.0)  # Flow towards boundaries

        Br = Reflective_boundary(domain)
        domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

        # Run CPU
        domain.set_multiprocessor_mode(1)
        domain.distribute_to_vertices_and_edges()
        domain.update_boundary()

        cpu_stage_bv = domain.quantities['stage'].boundary_values.copy()
        cpu_xmom_bv = domain.quantities['xmomentum'].boundary_values.copy()

        # Run GPU
        domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import (
            sync_to_device, sync_all_from_device,
            extrapolate_second_order_gpu, evaluate_reflective_boundary_gpu
        )

        gpu_dom = domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)
        extrapolate_second_order_gpu(gpu_dom)
        evaluate_reflective_boundary_gpu(gpu_dom)
        sync_all_from_device(gpu_dom)

        gpu_stage_bv = domain.quantities['stage'].boundary_values.copy()
        gpu_xmom_bv = domain.quantities['xmomentum'].boundary_values.copy()

        np.testing.assert_allclose(cpu_stage_bv, gpu_stage_bv, rtol=1e-10, atol=1e-14,
                                   err_msg="Reflective boundary stage mismatch")
        np.testing.assert_allclose(cpu_xmom_bv, gpu_xmom_bv, rtol=1e-10, atol=1e-14,
                                   err_msg="Reflective boundary xmomentum mismatch")

    def test_dirichlet_boundary(self):
        """Test Dirichlet boundary on GPU."""
        domain = rectangular_cross_domain(5, 5, len1=50., len2=50.)
        domain.set_flow_algorithm('DE0')
        domain.set_name('test_dirichlet')
        domain.set_datadir('.')
        domain.store = False

        domain.set_quantity('elevation', -1.0)
        domain.set_quantity('friction', 0.01)
        domain.set_quantity('stage', 0.0)

        Br = Reflective_boundary(domain)
        Bd = Dirichlet_boundary([0.5, 0.1, 0.05])  # stage, xmom, ymom
        domain.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})

        # Run GPU evolution
        domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import (
            sync_to_device, sync_all_from_device,
            extrapolate_second_order_gpu, evaluate_reflective_boundary_gpu,
            evaluate_dirichlet_boundary_gpu
        )

        gpu_dom = domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)
        extrapolate_second_order_gpu(gpu_dom)
        evaluate_reflective_boundary_gpu(gpu_dom)
        evaluate_dirichlet_boundary_gpu(gpu_dom)
        sync_all_from_device(gpu_dom)

        # Check that Dirichlet values are applied
        stage_bv = domain.quantities['stage'].boundary_values
        xmom_bv = domain.quantities['xmomentum'].boundary_values
        ymom_bv = domain.quantities['ymomentum'].boundary_values

        # Get indices for 'left' boundary tag
        left_indices = domain.tag_boundary_cells.get('left', [])
        self.assertGreater(len(left_indices), 0, "Should have 'left' boundary edges")

        for idx in left_indices:
            self.assertAlmostEqual(stage_bv[idx], 0.5, places=10)
            self.assertAlmostEqual(xmom_bv[idx], 0.1, places=10)
            self.assertAlmostEqual(ymom_bv[idx], 0.05, places=10)

    def test_transmissive_n_zero_t_boundary(self):
        """Test Transmissive_n_momentum_zero_t_momentum_set_stage_boundary on GPU."""
        domain = rectangular_cross_domain(5, 5, len1=50., len2=50.)
        domain.set_flow_algorithm('DE0')
        domain.set_low_froude(0)
        domain.set_name('test_transmissive')
        domain.set_datadir('.')
        domain.store = False

        domain.set_quantity('elevation', -1.0)
        domain.set_quantity('friction', 0.01)
        domain.set_quantity('stage', 0.0)
        domain.set_quantity('xmomentum', 0.5)

        def tide_function(t):
            return -0.3

        Br = Reflective_boundary(domain)
        Bt = anuga.Transmissive_n_momentum_zero_t_momentum_set_stage_boundary(
            domain, function=tide_function)
        domain.set_boundary({'left': Bt, 'right': Br, 'top': Br, 'bottom': Br})

        # Run one evolve step
        domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device, sync_all_from_device

        gpu_dom = domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)

        for t in domain.evolve(yieldstep=0.1, finaltime=0.1):
            pass

        sync_all_from_device(gpu_dom)

        # Check that stage boundary values are set to tide function value
        stage_bv = domain.quantities['stage'].boundary_values

        # Get indices for 'left' boundary tag
        left_indices = domain.tag_boundary_cells.get('left', [])
        self.assertGreater(len(left_indices), 0, "Should have 'left' boundary edges")

        for idx in left_indices:
            # Stage should be set to tide function value
            self.assertAlmostEqual(stage_bv[idx], -0.3, places=5)


@pytest.mark.skipif(not gpu_available(), reason="GPU OpenMP interface not available")
class Test_GPU_Initialization(unittest.TestCase):
    """Tests for GPU initialization and error handling."""

    def test_boundary_before_gpu_mode(self):
        """Test that boundaries must be set before GPU mode."""
        domain = rectangular_cross_domain(5, 5, len1=50., len2=50.)
        domain.set_flow_algorithm('DE0')
        domain.set_name('test_init')
        domain.set_datadir('.')
        domain.store = False

        domain.set_quantity('elevation', -1.0)
        domain.set_quantity('stage', 0.0)

        # Do NOT set boundaries, then try to enable GPU mode
        # This should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            domain.set_multiprocessor_mode(2)

        self.assertIn("boundaries", str(context.exception).lower())

    def test_correct_initialization_order(self):
        """Test that correct initialization order works."""
        domain = rectangular_cross_domain(5, 5, len1=50., len2=50.)
        domain.set_flow_algorithm('DE0')
        domain.set_name('test_init_ok')
        domain.set_datadir('.')
        domain.store = False

        domain.set_quantity('elevation', -1.0)
        domain.set_quantity('stage', 0.0)

        # Set boundaries FIRST
        Br = Reflective_boundary(domain)
        domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

        # Then enable GPU mode - should work
        domain.set_multiprocessor_mode(2)

        self.assertIsNotNone(domain.gpu_interface)
        self.assertTrue(domain.gpu_interface.initialized)


@pytest.mark.skipif(not gpu_available(), reason="GPU OpenMP interface not available")
class Test_GPU_LargeDomain(unittest.TestCase):
    """Tests with larger domains to verify scaling."""

    def test_large_rectangular_domain(self):
        """Test with ~5000 elements."""
        domain = rectangular_cross_domain(50, 50, len1=100., len2=100.)
        domain.set_flow_algorithm('DE0')
        domain.set_low_froude(0)
        domain.set_name('test_large')
        domain.set_datadir('.')
        domain.store = False

        n_elements = len(domain)
        self.assertGreater(n_elements, 4000, "Domain should have >4000 elements")

        def topography(x, y):
            return -x / 50.0

        domain.set_quantity('elevation', topography)
        domain.set_quantity('friction', 0.01)
        domain.set_quantity('stage', 0.0)

        def tide_function(t):
            return -0.5 + 0.01 * t

        Br = Reflective_boundary(domain)
        Bt = anuga.Transmissive_n_momentum_zero_t_momentum_set_stage_boundary(
            domain, function=tide_function)
        domain.set_boundary({'left': Bt, 'right': Br, 'top': Br, 'bottom': Br})

        # Save initial state
        initial_stage = domain.quantities['stage'].centroid_values.copy()

        # Run CPU
        domain.set_multiprocessor_mode(1)
        for t in domain.evolve(yieldstep=1.0, finaltime=1.0):
            pass

        cpu_stage = domain.quantities['stage'].centroid_values.copy()

        # Reset
        for qname in ['stage', 'xmomentum', 'ymomentum']:
            domain.quantities[qname].centroid_values[:] = 0.0
        domain.quantities['stage'].centroid_values[:] = initial_stage
        domain.set_time(0.0)

        # Run GPU
        domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device, sync_from_device

        sync_to_device(domain.gpu_interface.gpu_dom)

        for t in domain.evolve(yieldstep=1.0, finaltime=1.0):
            pass

        sync_from_device(domain.gpu_interface.gpu_dom)
        gpu_stage = domain.quantities['stage'].centroid_values.copy()

        # Compare
        stage_diff = np.abs(cpu_stage - gpu_stage)
        self.assertLess(stage_diff.max(), 1e-6,
                        f"Large domain stage difference: max={stage_diff.max():.2e}")


@pytest.mark.skipif(not gpu_available(), reason="GPU OpenMP interface not available")
class Test_GPU_InletOperator(unittest.TestCase):
    """Tests for Inlet_operator with GPU acceleration."""

    def create_domain(self, name='test_inlet'):
        """Create a test domain suitable for inlet testing."""
        domain = rectangular_cross_domain(20, 10, len1=200., len2=100.)
        domain.set_flow_algorithm('DE0')
        domain.set_low_froude(0)
        domain.set_name(name)
        domain.set_datadir('.')
        domain.store = False

        def topography(x, y):
            return -x / 100.0  # Gentle slope

        domain.set_quantity('elevation', topography)
        domain.set_quantity('friction', 0.03)
        domain.set_quantity('stage', 0.0)

        Br = Reflective_boundary(domain)
        Bd = Dirichlet_boundary([0, 0, 0])
        domain.set_boundary({'left': Bd, 'right': Bd, 'top': Br, 'bottom': Br})

        return domain

    def test_inlet_operator_basic(self):
        """Test that inlet operator works on GPU."""
        domain = self.create_domain('test_inlet_basic')

        # Add inlet operator - line across left side
        line = [[10.0, 20.0], [10.0, 80.0]]
        Q = 10.0  # m^3/s
        inlet = Inlet_operator(domain, line, Q, verbose=False)

        # Enable GPU mode
        domain.set_multiprocessor_mode(2)

        from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device, sync_from_device
        gpu_dom = domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)

        # Evolve
        for t in domain.evolve(yieldstep=0.5, finaltime=1.0):
            pass

        sync_from_device(gpu_dom)

        # Verify water was added
        water_volume = domain.get_water_volume()
        applied_volume = inlet.total_applied_volume

        self.assertGreater(water_volume, 0, "Water volume should be positive after inlet")
        self.assertGreater(applied_volume, 0, "Inlet should have applied some volume")
        # Volume added should be approximately Q * time = 10 * 1.0 = 10 m^3
        self.assertAlmostEqual(applied_volume, Q * 1.0, delta=1.0,
                               msg=f"Applied volume {applied_volume} should be close to {Q * 1.0}")

    def test_inlet_operator_cpu_gpu_match(self):
        """Test that inlet operator produces same results on CPU and GPU."""
        # Create two identical domains
        cpu_domain = self.create_domain('test_inlet_cpu')
        gpu_domain = self.create_domain('test_inlet_gpu')

        # Add inlet operators with same parameters
        line = [[10.0, 20.0], [10.0, 80.0]]
        Q = 15.0  # m^3/s

        cpu_inlet = Inlet_operator(cpu_domain, line, Q, verbose=False)
        gpu_inlet = Inlet_operator(gpu_domain, line, Q, verbose=False)

        # Run CPU
        cpu_domain.set_multiprocessor_mode(1)
        for t in cpu_domain.evolve(yieldstep=0.5, finaltime=1.0):
            pass

        cpu_stage = cpu_domain.quantities['stage'].centroid_values.copy()
        cpu_volume = cpu_domain.get_water_volume()
        cpu_inlet_volume = cpu_inlet.total_applied_volume

        # Run GPU
        gpu_domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device, sync_from_device
        gpu_dom = gpu_domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)

        for t in gpu_domain.evolve(yieldstep=0.5, finaltime=1.0):
            pass

        sync_from_device(gpu_dom)
        gpu_stage = gpu_domain.quantities['stage'].centroid_values.copy()
        gpu_volume = gpu_domain.get_water_volume()
        gpu_inlet_volume = gpu_inlet.total_applied_volume

        # Compare inlet volumes
        self.assertAlmostEqual(cpu_inlet_volume, gpu_inlet_volume, places=6,
                               msg=f"Inlet volumes differ: CPU={cpu_inlet_volume}, GPU={gpu_inlet_volume}")

        # Compare water volumes
        self.assertAlmostEqual(cpu_volume, gpu_volume, delta=1e-6,
                               msg=f"Water volumes differ: CPU={cpu_volume}, GPU={gpu_volume}")

        # Compare stage values
        stage_diff = np.abs(cpu_stage - gpu_stage)
        self.assertLess(stage_diff.max(), 1e-8,
                        f"Stage difference too large: max={stage_diff.max():.2e}")

    def test_inlet_operator_time_varying_Q(self):
        """Test inlet operator with time-varying discharge on GPU."""
        domain = self.create_domain('test_inlet_timevar')

        # Time-varying discharge function
        def Q_func(t):
            return 5.0 + 10.0 * t  # Increases with time

        line = [[10.0, 20.0], [10.0, 80.0]]
        inlet = Inlet_operator(domain, line, Q_func, verbose=False)

        # Enable GPU mode
        domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device, sync_from_device
        gpu_dom = domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)

        # Evolve
        for t in domain.evolve(yieldstep=0.5, finaltime=1.0):
            pass

        sync_from_device(gpu_dom)

        # Verify water was added
        water_volume = domain.get_water_volume()
        applied_volume = inlet.total_applied_volume

        self.assertGreater(water_volume, 0, "Water volume should be positive")
        self.assertGreater(applied_volume, 0, "Inlet should have applied some volume")
        # With Q = 5 + 10*t, average over [0,1] is about 10, so ~10 m^3 total
        self.assertGreater(applied_volume, 5.0, "Should have applied at least 5 m^3")

    def test_inlet_operator_with_velocity(self):
        """Test inlet operator with specified velocity on GPU."""
        cpu_domain = self.create_domain('test_inlet_vel_cpu')
        gpu_domain = self.create_domain('test_inlet_vel_gpu')

        line = [[10.0, 20.0], [10.0, 80.0]]
        Q = 10.0
        velocity = [0.5, 0.0]  # Velocity in x direction

        cpu_inlet = Inlet_operator(cpu_domain, line, Q, velocity=velocity, verbose=False)
        gpu_inlet = Inlet_operator(gpu_domain, line, Q, velocity=velocity, verbose=False)

        # Run CPU
        cpu_domain.set_multiprocessor_mode(1)
        for t in cpu_domain.evolve(yieldstep=0.5, finaltime=1.0):
            pass

        cpu_xmom = cpu_domain.quantities['xmomentum'].centroid_values.copy()

        # Run GPU
        gpu_domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device, sync_from_device
        gpu_dom = gpu_domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)

        for t in gpu_domain.evolve(yieldstep=0.5, finaltime=1.0):
            pass

        sync_from_device(gpu_dom)
        gpu_xmom = gpu_domain.quantities['xmomentum'].centroid_values.copy()

        # Compare momentum
        xmom_diff = np.abs(cpu_xmom - gpu_xmom)
        self.assertLess(xmom_diff.max(), 1e-8,
                        f"Xmomentum difference too large: max={xmom_diff.max():.2e}")


@pytest.mark.skipif(not gpu_available(), reason="GPU OpenMP interface not available")
class Test_GPU_Riverwall(unittest.TestCase):
    """Tests for riverwall/weir support with GPU acceleration."""

    def create_riverwall_domain(self, name='test_riverwall'):
        """Create a test domain with a riverwall."""
        from anuga import create_mesh_from_regions, create_domain_from_file
        import os

        mesh_filename = f'{name}.msh'

        # Domain polygon
        boundaryPolygon = [[0., 0.], [0., 100.], [100.0, 100.0], [100.0, 0.0]]

        # Riverwall - a wall across the middle of the domain
        riverWall = {'centralWall':
                     [[50., 0.0, -0.0],
                      [50., 45., -0.0],
                      [50., 46., -0.2],  # Dip in the wall
                      [50., 100.0, -0.0]]
                     }

        riverWall_Par = {'centralWall': {'Qfactor': 1.0}}

        # Region points
        regionPtAreas = [[25., 50., 5.0*5.0*0.5],
                         [75., 50., 5.0*5.0*0.5]]

        create_mesh_from_regions(boundaryPolygon,
                                 boundary_tags={'left': [0],
                                                'top': [1],
                                                'right': [2],
                                                'bottom': [3]},
                                 maximum_triangle_area=10.0*10.0*0.5,
                                 minimum_triangle_angle=28.0,
                                 filename=mesh_filename,
                                 breaklines=list(riverWall.values()),
                                 use_cache=False,
                                 verbose=False,
                                 regionPtArea=regionPtAreas)

        domain = create_domain_from_file(mesh_filename)
        domain.set_flow_algorithm('DE0')
        domain.set_low_froude(0)
        domain.set_name(name)
        domain.set_datadir('.')
        domain.store = False

        def topography(x, y):
            return -x / 150.0

        def stagefun(x, y):
            return -0.5

        domain.set_quantity('elevation', topography)
        domain.set_quantity('friction', 0.03)
        domain.set_quantity('stage', stagefun)

        # Create the riverwalls
        domain.riverwallData.create_riverwalls(riverWall, riverWall_Par, verbose=False)

        # Clean up mesh file
        os.remove(mesh_filename)

        return domain

    def test_riverwall_initialization(self):
        """Test that riverwall arrays are properly initialized for GPU."""
        domain = self.create_riverwall_domain('test_rw_init')

        # Set boundaries
        Br = Reflective_boundary(domain)
        domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

        # Enable GPU mode
        domain.set_multiprocessor_mode(2)

        # Verify riverwall arrays are set
        self.assertGreater(domain.number_of_riverwall_edges, 0,
                           "Should have riverwall edges")
        self.assertIsNotNone(domain.edge_flux_type)
        self.assertIsNotNone(domain.riverwallData.riverwall_elevation)

    def test_riverwall_cpu_gpu_match(self):
        """Test that riverwall simulation matches between CPU and GPU."""
        from math import exp

        # Create two identical domains
        cpu_domain = self.create_riverwall_domain('test_rw_cpu')
        gpu_domain = self.create_riverwall_domain('test_rw_gpu')

        # Boundary function
        def boundaryFun(t):
            output = -0.4 * exp(-t / 100.) - 0.1
            return min(output, -0.11)

        # Set boundaries
        Br_cpu = Reflective_boundary(cpu_domain)
        Bt_cpu = anuga.Transmissive_n_momentum_zero_t_momentum_set_stage_boundary(
            domain=cpu_domain, function=boundaryFun)
        cpu_domain.set_boundary({'left': Br_cpu, 'right': Bt_cpu, 'top': Br_cpu, 'bottom': Br_cpu})

        Br_gpu = Reflective_boundary(gpu_domain)
        Bt_gpu = anuga.Transmissive_n_momentum_zero_t_momentum_set_stage_boundary(
            domain=gpu_domain, function=boundaryFun)
        gpu_domain.set_boundary({'left': Br_gpu, 'right': Bt_gpu, 'top': Br_gpu, 'bottom': Br_gpu})

        # Run CPU
        cpu_domain.set_multiprocessor_mode(1)
        for t in cpu_domain.evolve(yieldstep=1.0, finaltime=5.0):
            pass

        cpu_stage = cpu_domain.quantities['stage'].centroid_values.copy()
        cpu_xmom = cpu_domain.quantities['xmomentum'].centroid_values.copy()

        # Run GPU
        gpu_domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device, sync_from_device
        gpu_dom = gpu_domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)

        for t in gpu_domain.evolve(yieldstep=1.0, finaltime=5.0):
            pass

        sync_from_device(gpu_dom)
        gpu_stage = gpu_domain.quantities['stage'].centroid_values.copy()
        gpu_xmom = gpu_domain.quantities['xmomentum'].centroid_values.copy()

        # Compare
        stage_diff = np.abs(cpu_stage - gpu_stage)
        xmom_diff = np.abs(cpu_xmom - gpu_xmom)

        self.assertLess(stage_diff.max(), 1e-6,
                        f"Stage difference too large: max={stage_diff.max():.2e}")
        self.assertLess(xmom_diff.max(), 1e-6,
                        f"Xmomentum difference too large: max={xmom_diff.max():.2e}")

    def test_riverwall_flux_kernel(self):
        """Test that riverwall flux computation matches CPU."""
        domain = self.create_riverwall_domain('test_rw_flux')

        # Set initial conditions that should trigger flow over the wall
        domain.set_quantity('stage', 0.5)  # Water above wall level

        # Set boundaries
        Br = Reflective_boundary(domain)
        Bd = Dirichlet_boundary([0.5, 0., 0.])
        domain.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})

        # Run CPU flux computation
        domain.set_multiprocessor_mode(1)
        domain.distribute_to_vertices_and_edges()
        domain.update_boundary()
        domain.compute_fluxes()

        cpu_timestep = domain.flux_timestep
        cpu_stage_update = domain.quantities['stage'].explicit_update.copy()
        cpu_xmom_update = domain.quantities['xmomentum'].explicit_update.copy()

        # Reset and run GPU flux computation
        for qname in ['stage', 'xmomentum', 'ymomentum']:
            domain.quantities[qname].explicit_update[:] = 0.0

        domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import (
            sync_to_device, sync_all_from_device,
            extrapolate_second_order_gpu, compute_fluxes_gpu,
            evaluate_reflective_boundary_gpu, evaluate_dirichlet_boundary_gpu
        )

        gpu_dom = domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)
        extrapolate_second_order_gpu(gpu_dom)
        evaluate_reflective_boundary_gpu(gpu_dom)
        evaluate_dirichlet_boundary_gpu(gpu_dom)
        gpu_timestep = compute_fluxes_gpu(gpu_dom)
        sync_all_from_device(gpu_dom)

        gpu_stage_update = domain.quantities['stage'].explicit_update.copy()
        gpu_xmom_update = domain.quantities['xmomentum'].explicit_update.copy()

        # Verify - riverwalls can have larger flux differences due to weir formula
        self.assertAlmostEqual(cpu_timestep, gpu_timestep, places=8,
                               msg=f"Timestep mismatch: CPU={cpu_timestep}, GPU={gpu_timestep}")
        np.testing.assert_allclose(cpu_stage_update, gpu_stage_update, rtol=1e-8, atol=1e-12,
                                   err_msg="Stage explicit_update mismatch")
        np.testing.assert_allclose(cpu_xmom_update, gpu_xmom_update, rtol=1e-8, atol=1e-12,
                                   err_msg="Xmomentum explicit_update mismatch")

    def test_riverwall_weir_discharge(self):
        """Test that weir discharge formula is applied correctly."""
        domain = self.create_riverwall_domain('test_rw_weir')

        # Create asymmetric water levels to trigger weir flow
        def asymmetric_stage(x, y):
            # Higher water on left side of wall (x < 50)
            return np.where(x < 50, 0.5, -0.3)

        domain.set_quantity('stage', asymmetric_stage)

        # Set boundaries
        Br = Reflective_boundary(domain)
        domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

        # Run simulation on GPU
        domain.set_multiprocessor_mode(2)
        from anuga.shallow_water.sw_domain_gpu_ext import sync_to_device, sync_from_device

        gpu_dom = domain.gpu_interface.gpu_dom
        sync_to_device(gpu_dom)

        initial_volume = domain.get_water_volume()

        for t in domain.evolve(yieldstep=1.0, finaltime=5.0):
            pass

        sync_from_device(gpu_dom)

        final_volume = domain.get_water_volume()

        # Volume should be conserved (within tolerance)
        self.assertAlmostEqual(initial_volume, final_volume, places=4,
                               msg=f"Volume not conserved: initial={initial_volume}, final={final_volume}")

        # Check that water has redistributed (some flow over the wall)
        stage = domain.quantities['stage'].centroid_values
        x = domain.centroid_coordinates[:, 0]

        left_mask = x < 50
        right_mask = x >= 50

        # Left side should have lower water now (some flowed over)
        # Right side should have higher water
        # This is a qualitative check that the riverwall is working
        left_mean = np.mean(stage[left_mask])
        right_mean = np.mean(stage[right_mask])

        # Initially left was 0.5, right was -0.3
        # After evolution, the difference should be reduced
        self.assertLess(left_mean - right_mean, 0.8,
                        "Water should have flowed over the wall, reducing the level difference")


if __name__ == "__main__":
    unittest.main(verbosity=2)
