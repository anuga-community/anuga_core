"""
Tests for Elliptic_operator.

Note: Elliptic_operator does not define __call__ (it inherits the base class
stub that raises NotImplementedError). Tests instead verify construction and
the core linear-algebra methods (elliptic_multiply).
"""

import unittest
import numpy as num
import anuga
from anuga import Reflective_boundary, rectangular_cross_domain
from anuga import Quantity


def make_rect_domain():
    """4x4 rectangular cross domain with Reflective boundaries."""
    domain = rectangular_cross_domain(4, 4, len1=1.0, len2=1.0)
    domain.set_quantity('stage', 0.5)
    domain.set_quantity('elevation', 0.0)
    domain.set_boundary({'left': Reflective_boundary(domain),
                         'right': Reflective_boundary(domain),
                         'top': Reflective_boundary(domain),
                         'bottom': Reflective_boundary(domain)})
    return domain


class Test_elliptic_operator(unittest.TestCase):

    def setUp(self):
        self.domain = make_rect_domain()

    def tearDown(self):
        import os
        try:
            os.remove('domain.sww')
        except OSError:
            pass

    def test_construction(self):
        """Operator builds without error."""
        from anuga.operators.elliptic_operator import Elliptic_operator
        operator = Elliptic_operator(self.domain)
        self.assertIsNotNone(operator)

    def test_domain_time_unchanged_after_construction(self):
        """Construction should not advance domain time."""
        from anuga.operators.elliptic_operator import Elliptic_operator
        Elliptic_operator(self.domain)
        self.assertEqual(self.domain.get_time(), 0.0)

    def test_diffusivity_quantity_values(self):
        """diffusivity quantity should exist and be initialised to 1.0."""
        from anuga.operators.elliptic_operator import Elliptic_operator
        operator = Elliptic_operator(self.domain)
        self.assertIsNotNone(operator.diffusivity)
        self.assertTrue(num.allclose(operator.diffusivity.centroid_values, 1.0),
                        "diffusivity centroid values should all be 1.0")

    def test_elliptic_multiply_returns_array(self):
        """elliptic_multiply on a numpy array should return an array of correct shape."""
        from anuga.operators.elliptic_operator import Elliptic_operator
        operator = Elliptic_operator(self.domain)
        n = len(self.domain)
        x = num.ones(n, dtype=float)
        result = operator.elliptic_multiply(x)
        self.assertEqual(result.shape, (n,),
                         "elliptic_multiply should return array of length n")

    def test_elliptic_multiply_quantity(self):
        """elliptic_multiply on a Quantity should return a Quantity."""
        from anuga.operators.elliptic_operator import Elliptic_operator
        operator = Elliptic_operator(self.domain)
        q_in = Quantity(self.domain)
        q_in.set_values(1.0)
        q_in.set_boundary_values(1.0)
        q_out = operator.elliptic_multiply(q_in)
        self.assertIsInstance(q_out, Quantity)

    def test_parallel_safe_not_defined(self):
        """Elliptic_operator should have parallel_safe from base class."""
        from anuga.operators.elliptic_operator import Elliptic_operator
        operator = Elliptic_operator(self.domain)
        # Base class returns False by default; just confirm the method exists and is callable
        result = operator.parallel_safe()
        self.assertIsInstance(result, bool)

    def test_update_elliptic_matrix_runs(self):
        """update_elliptic_matrix should run without error."""
        from anuga.operators.elliptic_operator import Elliptic_operator
        operator = Elliptic_operator(self.domain)
        operator.update_elliptic_matrix()  # default a=None uses identity diffusivity

    def test_statistics_returns_string(self):
        from anuga.operators.elliptic_operator import Elliptic_operator
        operator = Elliptic_operator(self.domain)
        msg = operator.statistics()
        self.assertIsInstance(msg, str)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(__import__('__main__'))
    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
