"""
Tests for Collect_max_stage_operator and Collect_max_quantities_operator.
"""

import unittest
import numpy as num
import anuga
from anuga import Reflective_boundary


def make_domain():
    """4-triangle domain, 1m water over 0.5m elevation."""
    a = [0.0, 0.0]; b = [0.0, 2.0]; c = [2.0, 0.0]
    d = [0.0, 4.0]; e = [2.0, 2.0]; f = [4.0, 0.0]
    points = [a, b, c, d, e, f]
    vertices = [[1, 0, 2], [1, 2, 4], [4, 2, 5], [3, 1, 4]]
    domain = anuga.Domain(points, vertices)
    domain.set_quantity('elevation', 0.5)
    domain.set_quantity('stage', 1.0)
    domain.set_quantity('friction', 0.0)
    domain.set_boundary({'exterior': Reflective_boundary(domain)})
    return domain


class Test_collect_max_stage_operator(unittest.TestCase):

    def setUp(self):
        self.domain = make_domain()

    def tearDown(self):
        import os
        try:
            os.remove('domain.sww')
        except OSError:
            pass

    def test_construction(self):
        from anuga.operators.collect_max_stage_operator import Collect_max_stage_operator
        operator = Collect_max_stage_operator(self.domain)
        self.assertIsNotNone(operator)

    def test_max_stage_quantity_initialised(self):
        """max_stage quantity should exist and be initialised to -1e100."""
        from anuga.operators.collect_max_stage_operator import Collect_max_stage_operator
        operator = Collect_max_stage_operator(self.domain)
        self.assertIn('max_stage', self.domain.quantities)
        max_stage_vals = self.domain.quantities['max_stage'].centroid_values
        self.assertTrue(num.all(max_stage_vals <= -1.0e+99),
                        "max_stage should be initialised to -1e100")

    def test_call_updates_max_stage(self):
        """After calling the operator, max_stage should be >= stage."""
        from anuga.operators.collect_max_stage_operator import Collect_max_stage_operator
        operator = Collect_max_stage_operator(self.domain)
        self.domain.timestep = 1.0
        operator()
        max_stage_vals = self.domain.quantities['max_stage'].centroid_values
        stage_vals = self.domain.quantities['stage'].centroid_values
        self.assertTrue(num.all(max_stage_vals >= stage_vals),
                        "max_stage should be >= stage after operator call")

    def test_parallel_safe(self):
        from anuga.operators.collect_max_stage_operator import Collect_max_stage_operator
        operator = Collect_max_stage_operator(self.domain)
        self.assertTrue(operator.parallel_safe())


class Test_collect_max_quantities_operator(unittest.TestCase):

    def setUp(self):
        self.domain = make_domain()

    def tearDown(self):
        import os
        try:
            os.remove('domain.sww')
        except OSError:
            pass

    def test_construction_defaults(self):
        from anuga.operators.collect_max_quantities_operator import Collect_max_quantities_operator
        operator = Collect_max_quantities_operator(self.domain)
        self.assertIsNotNone(operator)

    def test_update_frequency_and_start_time_stored(self):
        from anuga.operators.collect_max_quantities_operator import Collect_max_quantities_operator
        operator = Collect_max_quantities_operator(
            self.domain, update_frequency=3, collection_start_time=5.0)
        self.assertEqual(operator.update_frequency, 3)
        self.assertEqual(operator.collection_start_time, 5.0)

    def test_call_updates_max_stage_when_time_exceeds_start(self):
        """After setting domain time > collection_start_time and calling operator,
        max_stage should be updated (>= -1e30)."""
        from anuga.operators.collect_max_quantities_operator import Collect_max_quantities_operator
        operator = Collect_max_quantities_operator(
            self.domain, update_frequency=1, collection_start_time=0.0)
        # domain.get_time() uses starttime + relative_time
        self.domain.relative_time = 1.0
        operator()
        # max_stage should now have been updated from -max_float to actual stage values
        self.assertTrue(num.all(operator.max_stage >= -1.0e30),
                        "max_stage should be updated after operator call")

    def test_call_no_update_before_start_time(self):
        """Before collection_start_time, max_stage should remain at initialised value."""
        from anuga.operators.collect_max_quantities_operator import Collect_max_quantities_operator
        from anuga.config import max_float
        operator = Collect_max_quantities_operator(
            self.domain, update_frequency=1, collection_start_time=10.0)
        # domain.time defaults to 0.0, which is <= collection_start_time
        operator()
        # max_stage should remain at the initialised -max_float
        self.assertTrue(num.all(operator.max_stage < -1.0e30),
                        "max_stage should not be updated before collection_start_time")

    def test_parallel_safe(self):
        from anuga.operators.collect_max_quantities_operator import Collect_max_quantities_operator
        operator = Collect_max_quantities_operator(self.domain)
        self.assertTrue(operator.parallel_safe())


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromModule(__import__('__main__'))
    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
