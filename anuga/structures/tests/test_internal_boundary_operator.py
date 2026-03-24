#!/usr/bin/env python

import unittest

import numpy
import anuga
from anuga.abstract_2d_finite_volumes.mesh_factory import rectangular_cross
from anuga.shallow_water.shallow_water_domain import Domain
from anuga.structures.internal_boundary_operator import Internal_boundary_operator


verbose = False


def make_culvert_domain():
    """Create a simple 10m x 5m rectangular domain for culvert tests."""
    points, vertices, boundary = rectangular_cross(10, 5, len1=10.0, len2=5.0)
    domain = Domain(points, vertices, boundary)
    domain.set_quantity('elevation', 0.0)
    domain.set_quantity('stage', 1.0)
    domain.set_quantity('friction', 0.0)
    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})
    return domain


def simple_ibf(hw, tw):
    """Simple internal boundary function: flow proportional to head difference."""
    return max(0.0, hw - tw) * 1.0


class Test_Internal_boundary_operator(unittest.TestCase):
    """Tests for Internal_boundary_operator."""

    def setUp(self):
        self.domain = make_culvert_domain()

    def tearDown(self):
        pass

    def test_construction(self):
        """Internal_boundary_operator can be constructed without error."""
        op = Internal_boundary_operator(
            self.domain,
            internal_boundary_function=simple_ibf,
            end_points=[[3., 2.5], [7., 2.5]],
            width=1.0,
            verbose=verbose)
        self.assertIsNotNone(op)

    def test_structure_type(self):
        """structure_type attribute is set to 'internal_boundary'."""
        op = Internal_boundary_operator(
            self.domain,
            internal_boundary_function=simple_ibf,
            end_points=[[3., 2.5], [7., 2.5]],
            width=1.0,
            verbose=verbose)
        self.assertEqual(op.structure_type, 'internal_boundary')

    def test_width_stored(self):
        """get_culvert_width() returns the specified width."""
        op = Internal_boundary_operator(
            self.domain,
            internal_boundary_function=simple_ibf,
            end_points=[[3., 2.5], [7., 2.5]],
            width=1.0,
            verbose=verbose)
        self.assertAlmostEqual(op.get_culvert_width(), 1.0)

    def test_calling_operator(self):
        """Calling the operator once does not raise an exception."""
        op = Internal_boundary_operator(
            self.domain,
            internal_boundary_function=simple_ibf,
            end_points=[[3., 2.5], [7., 2.5]],
            width=1.0,
            verbose=verbose)
        self.domain.timestep = 0.1
        self.domain.yieldstep = 1.0
        # Calling the operator should not raise
        op()

    def test_statistics_returns_string(self):
        """statistics() returns a non-empty string."""
        op = Internal_boundary_operator(
            self.domain,
            internal_boundary_function=simple_ibf,
            end_points=[[3., 2.5], [7., 2.5]],
            width=1.0,
            verbose=verbose)
        result = op.statistics()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_Internal_boundary_operator)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
