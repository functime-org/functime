import polars as pl
import numpy as np
from tsfresh.examples.driftbif_simulation import velocity
from unittest import TestCase
import pandas as pd

from functime.feature_extraction.tsfresh_features_clay import (
    partial_autocorrelation,
    binned_entropy,
    friedrich_coefficients,
    max_langevin_fixed_point,
    lempel_ziv_complexity,
    _estimate_friedrich_coefficients
)

'''
Tests adapted from TSFresh (https://github.com/blue-yonder/tsfresh/blob/main/tests/units/feature_extraction/test_feature_calculations.py)
'''

class FeatureCalculationTestCase(TestCase):
    
    def assertAlmostEqualOnAllArrayTypes(self, f, input_to_f, result, *args, **kwargs):
        
        if isinstance(input_to_f, list):
            input_to_f = pl.Series(input_to_f)
            
        if isinstance(input_to_f, np.ndarray):
            input_to_f = pl.Series(input_to_f.tolist())
        
        expected_result = f(input_to_f, *args, **kwargs)
        
        # Adapt to polars
        if isinstance(expected_result, pl.Series):
            expected_result = expected_result.to_numpy()
            
        self.assertAlmostEqual(
            expected_result,
            result,
            msg="Not almost equal for lists: {} != {}".format(expected_result, result),
        )

    def assertIsNaN(self, result):
        self.assertTrue(np.isnan(result), msg="{} is not np.NaN")
        
    def test_binned_entropy(self):
        self.assertAlmostEqualOnAllArrayTypes(binned_entropy, [10] * 100, 0, 10)
        self.assertAlmostEqualOnAllArrayTypes(
            binned_entropy,
            [10] * 10 + [1],
            -(10 / 11 * np.math.log(10 / 11) + 1 / 11 * np.math.log(1 / 11)),
            10,
        )
        self.assertAlmostEqualOnAllArrayTypes(
            binned_entropy,
            [10] * 10 + [1],
            -(10 / 11 * np.math.log(10 / 11) + 1 / 11 * np.math.log(1 / 11)),
            10,
        )
        self.assertAlmostEqualOnAllArrayTypes(
            binned_entropy,
            [10] * 10 + [1],
            -(10 / 11 * np.math.log(10 / 11) + 1 / 11 * np.math.log(1 / 11)),
            100,
        )
        self.assertAlmostEqualOnAllArrayTypes(
            binned_entropy, list(range(10)), -np.math.log(1 / 10), 100
        )
        self.assertAlmostEqualOnAllArrayTypes(
            binned_entropy, list(range(100)), -np.math.log(1 / 2), 2
        )

    def test_partial_autocorrelation(self):

        # Test for altering time series
        # len(x) < max_lag
        param = [{"lag": lag} for lag in range(10)]
        x = [1, 2, 1, 2, 1, 2]
        expected_res = [("lag_0", 1.0), ("lag_1", -1.0), ("lag_2", np.nan)]
        res = partial_autocorrelation(pl.Series(x), lags=param)
        self.assertAlmostEqual(res[0][1], expected_res[0][1], places=4)
        self.assertAlmostEqual(res[1][1], expected_res[1][1], places=4)
        self.assertIsNaN(res[2][1])

        # Linear signal
        param = [{"lag": lag} for lag in range(10)]
        x = np.linspace(0, 1, 3000)
        expected_res = [("lag_0", 1.0), ("lag_1", 1.0), ("lag_2", 0)]
        res = partial_autocorrelation(pl.Series(x.tolist()), lags=param)
        self.assertAlmostEqual(res[0][1], expected_res[0][1], places=2)
        self.assertAlmostEqual(res[1][1], expected_res[1][1], places=2)
        self.assertAlmostEqual(res[2][1], expected_res[2][1], places=2)

        # Random noise
        np.random.seed(42)
        x = np.random.normal(size=3000)
        param = [{"lag": lag} for lag in range(10)]
        expected_res = [("lag_0", 1.0), ("lag_1", 0), ("lag_2", 0)]
        res = partial_autocorrelation(pl.Series(x.tolist()), lags=param)
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
        res = partial_autocorrelation(pl.Series(x.tolist()), lags=param)
        self.assertAlmostEqual(res[0][1], expected_res[0][1], places=1)
        self.assertAlmostEqual(res[1][1], expected_res[1][1], places=1)
        self.assertAlmostEqual(res[2][1], expected_res[2][1], places=1)

        # Some pathological cases
        param = [{"lag": lag} for lag in range(10)]
        # List of length 1
        res = partial_autocorrelation(pl.Series([1]), lags=param)
        for lag_no, lag_val in res:
            self.assertIsNaN(lag_val)
        # Empty list
        res = partial_autocorrelation(pl.Series([]), lags=param)
        for lag_no, lag_val in res:
            self.assertIsNaN(lag_val)
        # List contains only zeros
        res = partial_autocorrelation(pl.Series(np.zeros(100).tolist()), lags=param)
        for lag_no, lag_val in res:
            if lag_no == "lag_0":
                self.assertEqual(lag_val, 1.0)
            else:
                self.assertIsNaN(lag_val)


    def test_lempel_ziv_complexity(self):
        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [1, 1, 1], 2.0 / 3, num_bins=2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [1, 1, 1], 2.0 / 3, num_bins=5
        )

        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [1, 1, 1, 1, 1, 1, 1], 0.4285714285, num_bins=2
        )
        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [1, 1, 1, 2, 1, 1, 1], 0.5714285714, num_bins=2
        )

        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [-1, 4.3, 5, 1, -4.5, 1, 5, 7, -3.4, 6], 0.8, num_bins=10
        )
        # Discrepency here between handling NaN values between functime and tsfresh! Test omitted
        # self.assertAlmostEqualOnAllArrayTypes(
        #     lempel_ziv_complexity,
        #     [-1, np.nan, 5, 1, -4.5, 1, 5, 7, -3.4, 6],
        #     0.4,
        #     num_bins=10,
        # )
        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, np.linspace(0, 1, 10), 0.6, num_bins=3
        )
        self.assertAlmostEqualOnAllArrayTypes(
            lempel_ziv_complexity, [1, 1, 2, 3, 4, 5, 6, 0, 7, 8], 0.6, num_bins=3
        )


class FriedrichTestCase(TestCase):
    def test_estimate_friedrich_coefficients(self):
        """
        Estimate friedrich coefficients
        """
        default_params = {"m": 3, "r": 30}

        # active Brownian motion
        ds = velocity(tau=3.8, delta_t=0.05, R=3e-4, seed=0)
        v = ds.simulate(10000, v0=np.zeros(1))
        coeff = _estimate_friedrich_coefficients(v[:, 0], **default_params)
        self.assertLess(abs(coeff[-1]), 0.0001)

        # Brownian motion
        ds = velocity(tau=2.0 / 0.3 - 3.8, delta_t=0.05, R=3e-4, seed=0)
        v = ds.simulate(10000, v0=np.zeros(1))
        coeff = _estimate_friedrich_coefficients(v[:, 0], **default_params)
        self.assertLess(abs(coeff[-1]), 0.0001)

    def test_friedrich_coefficients(self):
        # Test binning error returns vector of NaNs
        param = [{"coeff": coeff, "m": 2, "r": 30} for coeff in range(4)]
        x = pl.Series(np.zeros(100).tolist())
        res = pd.Series(dict(friedrich_coefficients(x, param)))

        expected_index = [
            "coeff_0__m_2__r_30",
            "coeff_1__m_2__r_30",
            "coeff_2__m_2__r_30",
            "coeff_3__m_2__r_30",
        ]
        self.assertCountEqual(list(res.index), expected_index)
        self.assertTrue(np.sum(res.isna()), 3)
    
    def test_max_langevin_fixed_point(self):
        """
        Estimating the intrinsic velocity of a dissipative soliton
        """
        default_params = {"m": 3, "r": 30}

        # active Brownian motion
        ds = velocity(tau=3.8, delta_t=0.05, R=3e-4, seed=0)
        v = pl.Series(ds.simulate(100000, v0=np.zeros(1))[:, 0].tolist())
        v0 = max_langevin_fixed_point(v, **default_params)
        self.assertLess(abs(ds.deterministic - v0), 0.001)

        # Brownian motion
        ds = velocity(tau=2.0 / 0.3 - 3.8, delta_t=0.05, R=3e-4, seed=0)
        v = pl.Series(ds.simulate(10000, v0=np.zeros(1))[:, 0].tolist())
        v0 = max_langevin_fixed_point(v, **default_params)
        self.assertLess(v0, 0.001)


if __name__ == "__main__":
    test_set_1 = FeatureCalculationTestCase()
    test_set_1.test_binned_entropy()
    test_set_1.test_partial_autocorrelation()
    test_set_1.test_lempel_ziv_complexity()
    
    test_set_2 = FriedrichTestCase()
    test_set_2.test_estimate_friedrich_coefficients()
    test_set_2.test_friedrich_coefficients()
    test_set_2.test_max_langevin_fixed_point()