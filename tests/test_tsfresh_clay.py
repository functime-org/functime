import polars as pl
import pytest
from polars.testing import assert_frame_equal

from functime.feature_extraction.tsfresh_features_clay import (
    autocorrelation,
    count_above,
    count_above_mean,
    count_below,
    count_below_mean,
    has_duplicate,
    has_duplicate_max,
    has_duplicate_min,
)

def test_partial_autocorrelation(self):

        # Test for altering time series
        # len(x) < max_lag
        param = [{"lag": lag} for lag in range(10)]
        x = [1, 2, 1, 2, 1, 2]
        expected_res = [("lag_0", 1.0), ("lag_1", -1.0), ("lag_2", np.nan)]
        res = partial_autocorrelation(x, param=param)
        self.assertAlmostEqual(res[0][1], expected_res[0][1], places=4)
        self.assertAlmostEqual(res[1][1], expected_res[1][1], places=4)
        self.assertIsNaN(res[2][1])

        # Linear signal
        param = [{"lag": lag} for lag in range(10)]
        x = np.linspace(0, 1, 3000)
        expected_res = [("lag_0", 1.0), ("lag_1", 1.0), ("lag_2", 0)]
        res = partial_autocorrelation(x, param=param)
        self.assertAlmostEqual(res[0][1], expected_res[0][1], places=2)
        self.assertAlmostEqual(res[1][1], expected_res[1][1], places=2)
        self.assertAlmostEqual(res[2][1], expected_res[2][1], places=2)

        # Random noise
        np.random.seed(42)
        x = np.random.normal(size=3000)
        param = [{"lag": lag} for lag in range(10)]
        expected_res = [("lag_0", 1.0), ("lag_1", 0), ("lag_2", 0)]
        res = partial_autocorrelation(x, param=param)
        self.assertAlmostEqual(res[0][1], expected_res[0][1], places=1)
        self.assertAlmostEqual(res[1][1], expected_res[1][1], places=1)
        self.assertAlmostEqual(res[2][1], expected_res[2][1], places=1)

        # On a simulated AR process
        np.random.seed(42)
        param = [{"lag": lag} for lag in range(10)]
        # Simulate AR process
        T = 3000
        epsilon = np.random.randn(T)
        x = np.repeat(1.0, T)
        for t in range(T - 1):
            x[t + 1] = 0.5 * x[t] + 2 + epsilon[t]
        expected_res = [("lag_0", 1.0), ("lag_1", 0.5), ("lag_2", 0)]
        res = partial_autocorrelation(x, param=param)
        self.assertAlmostEqual(res[0][1], expected_res[0][1], places=1)
        self.assertAlmostEqual(res[1][1], expected_res[1][1], places=1)
        self.assertAlmostEqual(res[2][1], expected_res[2][1], places=1)

        # Some pathological cases
        param = [{"lag": lag} for lag in range(10)]
        # List of length 1
        res = partial_autocorrelation([1], param=param)
        for lag_no, lag_val in res:
            self.assertIsNaN(lag_val)
        # Empty list
        res = partial_autocorrelation([], param=param)
        for lag_no, lag_val in res:
            self.assertIsNaN(lag_val)
        # List contains only zeros
        res = partial_autocorrelation(np.zeros(100), param=param)
        for lag_no, lag_val in res:
            if lag_no == "lag_0":
                self.assertEqual(lag_val, 1.0)
            else:
                self.assertIsNaN(lag_val)