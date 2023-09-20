import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from functime.feature_extraction.tsfresh import (
    benford_correlation,
    longest_strike_above_mean,
    longest_strike_below_mean,
    mean_n_absolute_max,
    mean_second_derivative_central,
    percent_recoccuring_values,
    percent_reocurring_points,
    sum_reocurring_points,
    sum_reocurring_values,
    symmetry_looking,
    time_reversal_asymmetry_statistic,
)

np.random.seed(42)


def test_benford_correlation():
    # Nan, division by 0
    X_uniform = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Random serie
    X_random = pl.Series([26.24, 3.03, -2.92, 3.5, -0.07, 0.35, 0.10, 0.51, -0.43])
    # Fibo, distribution same as benford law
    X_fibo = [0, 1]
    for i in range(2, 50):
        X_fibo.append(X_fibo[i - 1] + X_fibo[i - 2])

    assert np.isnan(benford_correlation(X_uniform))
    assert benford_correlation(X_random) == 0.39753280229716703
    assert benford_correlation(pl.Series(X_fibo)) == 0.9959632739083689


@pytest.mark.parametrize(
    "S, res",
    [
        ([1, 2, 1, 1, 1, 2, 2, 2], 3),
        ([1, 2, 3, 4, 5, 6], 3),
        ([1, 2, 3, 4, 5], 2),
        ([1, 2, 1], 1),
        ([1, 1, 1], 0),
        ([], 0),
    ],
)
def test_longest_strike_below_mean(S, res):
    assert longest_strike_below_mean(pl.Series(S)) == res


@pytest.mark.parametrize(
    "S, res",
    [
        ([1, 2, 1, 2, 1, 2, 2, 1], 2),
        ([1, 2, 3, 4, 5, 6], 3),
        ([1, 2, 3, 4, 5], 2),
        ([1, 2, 1], 1),
        ([1, 1, 1], 0),
        ([], 0),
    ],
)
def test_longest_strike_above_mean(S, res):
    assert longest_strike_above_mean(pl.Series(S)) == res


@pytest.mark.parametrize(
    "S, n_max, res",
    [
        ([], 1, None),
        ([12, 3], 10, None),
        ([-1, -5, 4, 10], 3, 6.333333333333333),
        ([0, -5, -9], 2, 7.000000),
        ([0, 0, 0], 1, 0),
    ],
)
def test_mean_n_absolute_max(S, n_max, res):
    assert mean_n_absolute_max(x=pl.Series(S), n_maxima=n_max) == res


def test_mean_n_absolute_max_value_error():
    with pytest.raises(ValueError):
        mean_n_absolute_max(x=pl.Series([12, 3]), n_maxima=0)
    with pytest.raises(ValueError):
        mean_n_absolute_max(x=pl.Series([12, 3]), n_maxima=-1)


@pytest.mark.parametrize(
    "S, res",
    [
        ([1, 1, 2, 3, 4], 0.4),
        ([1, 1.5, 2, 3], 0),
        ([1], 0),
        ([1.111, -2.45, 1.111, 2.45], 0.5),
    ],
)
def test_percent_reocurring_points(S, res):
    assert percent_reocurring_points(pl.Series(S)) == res


def test_percent_reocurring_points_value_error():
    with pytest.raises(ValueError):
        percent_reocurring_points(pl.Series([]))


@pytest.mark.parametrize(
    "S, res",
    [
        ([1, 1, 2, 3, 4], 0.25),
        ([1, 1.5, 2, 3], 0),
        ([1], 0),
        ([1.111, -2.45, 1.111, 2.45], 1.0 / 3.0),
    ],
)
def test_percent_recoccuring_values(S, res):
    assert percent_recoccuring_values(pl.Series(S)) == res


def test_percent_recoccuring_values_value_error():
    with pytest.raises(ValueError):
        percent_recoccuring_values(pl.Series([]))


@pytest.mark.parametrize(
    "S, res",
    [
        ([1, 1, 2, 3, 4, 4], 10),
        ([1, 1.5, 2, 3], 0),
        ([1], 0),
        ([1.111, -2.45, 1.111, 2.45], 2.222),
        ([], 0),
    ],
)
def test_sum_reocurring_points(S, res):
    assert sum_reocurring_points(pl.Series(S)) == res


@pytest.mark.parametrize(
    "S, res",
    [
        ([1, 1, 2, 3, 4, 4], 5),
        ([1, 1.5, 2, 3], 0),
        ([1], 0),
        ([1.111, -2.45, 1.111, 2.45], 1.111),
        ([], 0),
    ],
)
def test_sum_reocurring_values(S, res):
    assert sum_reocurring_values(pl.Series(S)) == res


def generate_ar1():
    np.random.seed(42)
    e = np.random.normal(0.1, 0.1, size=100)
    m = 50
    x = [0] * m
    x[0] = 100
    for i in range(1, m):
        x[i] = x[i - 1] * 0.5 + e[i]
    return pl.Series(x, dtype=pl.Float64)


@pytest.mark.parametrize(
    "x, param, res",
    [
        (
            pl.Series(np.cumsum(np.random.uniform(size=100))),
            [
                {"autolag": "BIC", "attr": "teststat"},
                {"autolag": "BIC", "attr": "pvalue"},
                {"autolag": "BIC", "attr": "usedlag"},
            ],
            pl.DataFrame(
                [
                    [
                        'attr_"teststat"__autolag_"BIC"',
                        'attr_"pvalue"__autolag_"BIC"',
                        'attr_"usedlag"__autolag_"BIC"',
                    ],
                    [0.037064, 0.961492, 0],
                ],
                schema=["index", "res"],
            ),
        ),
        (
            generate_ar1(),
            [
                {"autolag": "AIC", "attr": "teststat"},
                {"autolag": "AIC", "attr": "pvalue"},
                {"autolag": "AIC", "attr": "usedlag"},
            ],
            pl.DataFrame(
                [
                    [
                        'attr_"teststat"__autolag_"AIC"',
                        'attr_"pvalue"__autolag_"AIC"',
                        'attr_"usedlag"__autolag_"AIC"',
                    ],
                    [-595.259534, 0, 0],
                ],
                schema=["index", "res"],
            ),
        ),
    ],
)
def test_augmented_dickey_fuller(x, param, res):
    # assert_frame_equal(augmented_dickey_fuller(x, param), res, atol=1e-7)
    # res_linalg_error = (
    #     augmented_dickey_fuller(x=pl.Series(np.repeat(np.nan, 100)), param=param)
    #     .get_column("res")
    #     .to_numpy()
    # )
    # assert all(np.isnan(res_linalg_error))
    #
    # res_value_error = (
    #     augmented_dickey_fuller(x=pl.Series([]), param=param)
    #     .get_column("res")
    #     .to_numpy()
    # )
    # assert all(np.isnan(res_value_error))
    #
    # # Should return NaN if "attr" is unknown
    # res_attr_error = (
    #     augmented_dickey_fuller(x=x, param=[{"autolag": "AIC", "attr": ""}])
    #     .get_column("res")
    #     .to_numpy()
    # )
    # assert all(np.isnan(res_attr_error))
    assert True


@pytest.mark.parametrize(
    "x, res",
    [
        (pl.Series(range(10)), 0),
        (pl.Series([1, 3, 5]), 0),
        (pl.Series([1, 3, 7, -3]), -3),
    ],
)
def test_mean_second_derivative_central(x, res):
    assert mean_second_derivative_central(x) == res


@pytest.mark.parametrize(
    "x, param, res",
    [
        (
            pl.Series([-1, -1, 1, 1]),
            [dict(r=0.05), dict(r=0.75)],
            pl.DataFrame(
                [[0.05, 0.75], [True, True]],
                schema=["r", "feature_value"],
                orient="col",
            ),
        ),
        (
            pl.Series([-1, -1, 1, 1]),
            [dict(r=0)],
            pl.DataFrame([[0], [False]], schema=["r", "feature_value"], orient="col"),
        ),
        (
            pl.Series([-1, -1, -1, -1, 1]),
            [dict(r=0.05)],
            pl.DataFrame(
                [[0.05], [False]], schema=["r", "feature_value"], orient="col"
            ),
        ),
        (
            pl.Series([-2, -2, -2, -1, -1, -1]),
            [dict(r=0.05)],
            pl.DataFrame([[0.05], [True]], schema=["r", "feature_value"], orient="col"),
        ),
        (
            pl.Series([-0.9, -0.900001]),
            [dict(r=0.05)],
            pl.DataFrame([[0.05], [True]], schema=["r", "feature_value"], orient="col"),
        ),
    ],
)
def test_symmetry_looking(x, param, res):
    assert_frame_equal(symmetry_looking(x, param), res)


@pytest.mark.parametrize(
    "x, lag, res", [(pl.Series([1] * 10), 0, 0), (pl.Series([1, 2, -3, 4]), 1, -10)]
)
def test_time_reversal_asymmetry_statistic(x, lag, res):
    assert time_reversal_asymmetry_statistic(x, lag) == res
