#!/usr/bin/env python

import unittest
import warnings

import numpy
import anuga
from anuga.abstract_2d_finite_volumes.mesh_factory import rectangular_cross
from anuga.shallow_water.shallow_water_domain import Domain
from anuga.structures.inlet_enquiry import Inlet_enquiry


verbose = False

# This end-point geometry places enquiry points inside inlet triangles on
# this small test mesh — expected behaviour that raises a UserWarning.
_INLET_WARNING = 'Enquiry point.*is in an inlet triangle'


def make_domain():
    """Create a simple 10m x 5m rectangular domain for testing."""
    points, vertices, boundary = rectangular_cross(10, 5, len1=10.0, len2=5.0)
    domain = Domain(points, vertices, boundary)
    domain.set_quantity('elevation', 0.0)
    domain.set_quantity('stage', 1.0)
    domain.set_quantity('friction', 0.0)
    Br = anuga.Reflective_boundary(domain)
    domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})
    return domain


class Test_Structure_operator(unittest.TestCase):
    """Tests for the Structure_operator base class."""

    def setUp(self):
        self.domain = make_domain()
        self._warning_ctx = warnings.catch_warnings()
        self._warning_ctx.__enter__()
        warnings.filterwarnings('ignore', message=_INLET_WARNING,
                                category=UserWarning)

    def tearDown(self):
        self._warning_ctx.__exit__(None, None, None)

    def _make_operator(self):
        """Helper: create a Structure_operator with all required parameters.

        The culvert runs along the x-axis so that auto-computed enquiry points
        stay inside the 10m x 5m domain.
        """
        return anuga.Structure_operator(
            self.domain,
            end_points=[[3., 2.5], [7., 2.5]],
            width=1.0,
            manning=0.013,
            enquiry_gap=0.0,
            verbose=verbose)

    def test_construction(self):
        """Structure_operator construction warns when enquiry points are in
        inlet triangles (expected for this test mesh geometry)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            op = anuga.Structure_operator(
                self.domain,
                end_points=[[3., 2.5], [7., 2.5]],
                width=1.0,
                manning=0.013,
                enquiry_gap=0.0,
                verbose=verbose)
        self.assertIsNotNone(op)
        inlet_warnings = [w for w in caught
                          if issubclass(w.category, UserWarning)
                          and 'inlet triangle' in str(w.message)]
        self.assertGreater(len(inlet_warnings), 0,
                           'Expected inlet-triangle UserWarning was not raised')

    def test_get_culvert_length(self):
        """get_culvert_length returns a positive value."""
        op = self._make_operator()
        length = op.get_culvert_length()
        self.assertGreater(length, 0.0)

    def test_get_culvert_width(self):
        """get_culvert_width returns the specified width."""
        op = self._make_operator()
        self.assertAlmostEqual(op.get_culvert_width(), 1.0)

    def test_repr_returns_string(self):
        """str() on the operator returns a string (via __repr__ or __str__)."""
        op = self._make_operator()
        result = str(op)
        self.assertIsInstance(result, str)

    def test_statistics_returns_string(self):
        """statistics() returns a non-empty string."""
        op = self._make_operator()
        result = op.statistics()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_discharge_routine_raises(self):
        """Base class discharge_routine raises (NotImplementedError or similar)."""
        op = self._make_operator()
        with self.assertRaises(Exception):
            op.discharge_routine()


class Test_Inlet_enquiry(unittest.TestCase):
    """Tests for the Inlet_enquiry class."""

    def setUp(self):
        self.domain = make_domain()

    def tearDown(self):
        pass

    def test_construction(self):
        """Inlet_enquiry can be constructed without error."""
        region = [[2.5, 0.], [2.5, 2.5]]
        enquiry_pt = [1.5, 2.5]
        inlet = Inlet_enquiry(
            self.domain,
            region=region,
            enquiry_pt=enquiry_pt,
            verbose=verbose)
        self.assertIsNotNone(inlet)

    def test_enquiry_pt_stored(self):
        """enquiry_pt attribute is stored correctly."""
        region = [[2.5, 0.], [2.5, 2.5]]
        enquiry_pt = [1.5, 2.5]
        inlet = Inlet_enquiry(
            self.domain,
            region=region,
            enquiry_pt=enquiry_pt,
            verbose=verbose)
        self.assertTrue(numpy.allclose(inlet.enquiry_pt, enquiry_pt))

    def test_enquiry_index_set(self):
        """enquiry_index is set and is a valid triangle index (>= 0)."""
        region = [[2.5, 0.], [2.5, 2.5]]
        enquiry_pt = [1.5, 2.5]
        inlet = Inlet_enquiry(
            self.domain,
            region=region,
            enquiry_pt=enquiry_pt,
            verbose=verbose)
        self.assertGreaterEqual(inlet.enquiry_index, 0)
        num_triangles = len(self.domain)
        self.assertLess(inlet.enquiry_index, num_triangles)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_Structure_operator)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(Test_Inlet_enquiry))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
