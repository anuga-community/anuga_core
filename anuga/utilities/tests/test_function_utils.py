import unittest
import numpy as num

from anuga.utilities.function_utils import determine_function_type


class Test_Function_Utils(unittest.TestCase):

    def test_determine_function_type_scalar(self):

        type = determine_function_type(1.0)

        assert type == 'scalar'

        type = determine_function_type(1)

        assert type == 'scalar'

    def test_determine_function_type_exception(self):

        type = determine_function_type([])

        assert type == 'array'

    def test_determine_function_type_time_only(self):

        def myfunc(t):
            return t*2

        type = determine_function_type(myfunc)

        assert type == 't'

    def test_determine_function_type_spatial_only(self):

        def myfunc(x, y):
            return x+y

        type = determine_function_type(myfunc)

        assert type == 'x,y'

    def test_determine_function_type_spatial_only(self):

        def myfunc(x, y, t):
            return x+y+t

        type = determine_function_type(myfunc)

        assert type == 'x,y,t'


# -------------------------------------------------------------

class Test_Function_Utils_extra(unittest.TestCase):
    """Tests for uncovered paths in function_utils."""

    def test_determine_function_type_none(self):
        """None returns None (line 26)."""
        result = determine_function_type(None)
        self.assertIsNone(result)

    def test_determine_function_type_ndarray(self):
        """ndarray returns 'array' (lines 81-82)."""
        import numpy as np
        result = determine_function_type(np.array([1.0, 2.0]))
        self.assertEqual(result, 'array')

    def test_determine_function_type_t_valueerror(self):
        """Function of t that raises ValueError returns 't' (lines 57-59)."""
        def f(t):
            if isinstance(t, float):
                raise ValueError('out of range')
            raise TypeError('not a float')
        result = determine_function_type(f)
        self.assertEqual(result, 't')

    def test_evaluate_temporal_function_scalar(self):
        """evaluate_temporal_function with non-callable returns value (line 115)."""
        from anuga.utilities.function_utils import evaluate_temporal_function
        result = evaluate_temporal_function(42.0, t=1.0)
        self.assertAlmostEqual(result, 42.0)

    def test_evaluate_temporal_function_callable(self):
        """evaluate_temporal_function with callable (line 89)."""
        from anuga.utilities.function_utils import evaluate_temporal_function
        result = evaluate_temporal_function(lambda t: t * 2.0, t=3.0)
        self.assertAlmostEqual(result, 6.0)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_Function_Utils)
    runner = unittest.TextTestRunner()  # verbosity=2)
    runner.run(suite)
