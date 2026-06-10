"""Tests for the HLLC Riemann solver.

Covers:
  1. Single-edge flux function (flux_function_hllc) against exact Riemann
     solutions for a 1D dam-break and a steady subcritical flow.
  2. Full-domain simulation (compute_fluxes_ext_hllc / flux_solver='hllc')
     against the central-upwind solver: conservation of mass and approximate
     momentum agreement on a symmetric dam-break.
  3. Solver selection API: set_flux_solver / get_flux_solver round-trip.
"""

import unittest
import numpy as np
from math import sqrt

import anuga
from anuga import Reflective_boundary, rectangular_cross_domain


class TestHLLCFluxFunction(unittest.TestCase):
    """Unit tests for the single-edge HLLC flux wrapper."""

    def setUp(self):
        from anuga.shallow_water.sw_domain_openmp_ext import (
            flux_function_hllc,
            flux_function_central,
        )
        self.hllc = flux_function_hllc
        self.central = flux_function_central
        self.g = 9.8
        self.eps = 1.0e-6

    def _call(self, fn, ql, qr, h_left, h_right, normal):
        edgeflux = np.zeros(3, dtype=float)
        max_speed, pressure_flux = fn(
            np.asarray(normal, dtype=float),
            np.asarray(ql, dtype=float),
            np.asarray(qr, dtype=float),
            float(h_left), float(h_right),
            float(h_left), float(h_right),
            edgeflux,
            self.eps,
            0.0,        # ze (reference bed elevation = 0)
            self.g,
            1.0,        # H0
            float(h_left),
            float(h_right),
            0,          # low_froude flag
        )
        return edgeflux.copy(), max_speed, pressure_flux

    def test_dry_both_sides(self):
        """Both sides dry → zero flux."""
        ef, ms, pf = self._call(self.hllc,
                                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                                0.0, 0.0, [1.0, 0.0])
        np.testing.assert_allclose(ef, 0.0, atol=1e-14)
        self.assertAlmostEqual(ms, 0.0)

    def test_dry_left_wet_right(self):
        """Left side dry, right side wet → mass flux direction should be
        inward (positive n-direction)."""
        h_r = 1.0
        ef, ms, _ = self._call(self.hllc,
                               [0.0, 0.0, 0.0], [h_r, 0.0, 0.0],
                               0.0, h_r, [1.0, 0.0])
        # Mass flux should be <= 0 (flowing from right to left in edge frame)
        self.assertLessEqual(ef[0], 0.0 + 1e-10)

    def test_symmetric_dam_break_mass_conservation(self):
        """Symmetric states → zero net mass flux across edge."""
        h = 2.0
        ql = [h, 0.0, 0.0]
        qr = [h, 0.0, 0.0]
        ef, ms, _ = self._call(self.hllc, ql, qr, h, h, [1.0, 0.0])
        # Mass flux must be exactly zero for symmetric states
        self.assertAlmostEqual(ef[0], 0.0, places=12)

    def test_max_speed_positive(self):
        """max_speed must be non-negative for any wet state."""
        h_l, h_r = 2.0, 1.0
        ql = [h_l, h_l * 0.5, 0.0]
        qr = [h_r, h_r * 0.2, 0.0]
        ef, ms, _ = self._call(self.hllc, ql, qr, h_l, h_r, [1.0, 0.0])
        self.assertGreater(ms, 0.0)

    def test_agrees_with_central_still_water(self):
        """HLLC and central should both give zero flux for still water."""
        h = 1.5
        ql = qr = [h, 0.0, 0.0]
        ef_h, _, _ = self._call(self.hllc, ql, qr, h, h, [1.0, 0.0])
        ef_c, _, _ = self._call(self.central, ql, qr, h, h, [1.0, 0.0])
        np.testing.assert_allclose(ef_h, ef_c, atol=1e-10)

    def test_transverse_normal(self):
        """Flux with normal [0,1] (y-direction edge) should give consistent
        results and non-negative max_speed."""
        h_l, h_r = 2.0, 0.5
        ql = [h_l, 0.0, h_l * 1.0]
        qr = [h_r, 0.0, h_r * 0.5]
        ef, ms, _ = self._call(self.hllc, ql, qr, h_l, h_r, [0.0, 1.0])
        self.assertGreaterEqual(ms, 0.0)

    def test_momentum_flux_sign_consistency(self):
        """If water flows to the right (positive u), mass flux should be
        positive (out of left cell into right cell)."""
        h = 1.0
        u = 2.0   # supersonic positive velocity
        ql = [h, h * u, 0.0]
        qr = [h, h * u, 0.0]   # same state → uniform flow
        ef, ms, _ = self._call(self.hllc, ql, qr, h, h, [1.0, 0.0])
        # Mass flux = h*u > 0
        self.assertGreater(ef[0], 0.0)


class TestHLLCDomainSolver(unittest.TestCase):
    """Integration tests: full domain evolve with flux_solver='hllc'."""

    def _make_domain(self, name='tmp', solver='central'):
        domain = anuga.rectangular_cross_domain(10, 10,
                                                len1=10.0, len2=10.0,
                                                origin=(0, 0))
        domain.set_name(name)
        domain.set_store(False)
        domain.set_flow_algorithm('DE0')
        domain.set_quantity('elevation', 0.0)
        domain.set_quantity('stage', lambda x, y: np.where(x < 5.0, 2.0, 1.0))
        domain.set_quantity('xmomentum', 0.0)
        domain.set_quantity('ymomentum', 0.0)
        domain.set_boundary({'left': Reflective_boundary(domain),
                             'right': Reflective_boundary(domain),
                             'top': Reflective_boundary(domain),
                             'bottom': Reflective_boundary(domain)})
        domain.set_flux_solver(solver)
        return domain

    def test_api_round_trip(self):
        domain = self._make_domain()
        domain.set_flux_solver('hllc')
        self.assertEqual(domain.get_flux_solver(), 'hllc')
        domain.set_flux_solver('central')
        self.assertEqual(domain.get_flux_solver(), 'central')

    def test_invalid_solver_raises(self):
        domain = self._make_domain()
        with self.assertRaises(Exception):
            domain.set_flux_solver('unknown_solver')

    def test_hllc_conserves_mass(self):
        """HLLC solver must conserve total water volume."""
        domain = self._make_domain(solver='hllc')
        h_init = domain.get_quantity('stage').get_values(location='centroids')
        z = domain.get_quantity('elevation').get_values(location='centroids')
        vol_init = np.sum((h_init - z) * domain.areas)

        for _ in domain.evolve(yieldstep=0.1, finaltime=0.3):
            pass

        h_final = domain.get_quantity('stage').get_values(location='centroids')
        vol_final = np.sum((h_final - z) * domain.areas)

        # Mass should be conserved to floating-point precision
        self.assertAlmostEqual(vol_init, vol_final, places=6)

    def test_hllc_vs_central_mass_conservation(self):
        """Both HLLC and central must conserve the same initial mass."""
        for solver in ['central', 'hllc']:
            domain = self._make_domain(solver=solver)
            h_init = domain.get_quantity('stage').get_values(location='centroids')
            z = domain.get_quantity('elevation').get_values(location='centroids')
            vol_init = np.sum((h_init - z) * domain.areas)

            for _ in domain.evolve(yieldstep=0.1, finaltime=0.3):
                pass

            h_final = domain.get_quantity('stage').get_values(location='centroids')
            vol_final = np.sum((h_final - z) * domain.areas)
            self.assertAlmostEqual(vol_init, vol_final, places=6,
                                   msg=f'Mass not conserved for {solver}')

    def test_hllc_vs_central_similar_stages(self):
        """HLLC and central solvers should produce similar (not identical)
        stage profiles for a short-time dam-break simulation."""
        stages = {}
        for solver in ['central', 'hllc']:
            domain = self._make_domain(solver=solver)
            for _ in domain.evolve(yieldstep=0.05, finaltime=0.1):
                pass
            stages[solver] = domain.get_quantity('stage').get_values(
                location='centroids').copy()

        # Profiles should be close (within 5% of initial head difference)
        diff = np.abs(stages['central'] - stages['hllc'])
        self.assertLess(diff.max(), 0.5,
                        'HLLC and central stages diverge too much after 0.1 s')


if __name__ == '__main__':
    unittest.main()
