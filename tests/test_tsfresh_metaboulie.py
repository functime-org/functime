import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from functime.feature_extraction.tsfresh_metaboulie import (
    _aggregate_on_chunks,
    agg_linear_trend,
    ar_coefficient,
    augmented_dickey_fuller,
    cwt_coefficients,
    mean_second_derivative_central,
    symmetry_looking,
    time_reversal_asymmetry_statistic,
)


@pytest.mark.parametrize(
    "x, f_agg, chunk_len, res",
    [
        (pl.Series([0, 1, 2, 3]), "max", 2, [1, 3]),
        (pl.Series([1, 1, 3, 3]), "max", 2, [1, 3]),
        (pl.Series([0, 1, 2, 3]), "min", 2, [0, 2]),
        (pl.Series([0, 1, 2, 3, 5]), "min", 2, [0, 2, 5]),
        (pl.Series([0, 1, 2, 3]), "mean", 2, [0.5, 2.5]),
        (pl.Series([0, 1, 0, 4, 5]), "mean", 2, [0.5, 2, 5]),
        (pl.Series([0, 1, 0, 4, 5]), "mean", 3, [1 / 3, 4.5]),
        (pl.Series([0, 1, 2, 3, 5, -2]), "median", 2, [0.5, 2.5, 1.5]),
        (pl.Series([-10, 5, 3, -3, 4, -6]), "median", 3, [3, -3]),
        (pl.Series([0, 1, 2, float("nan"), 5]), "median", 2, [0.5, float("nan"), 5]),
    ],
)
def test__aggregate_on_chunks(x, f_agg, chunk_len, res):
    assert_series_equal(
        pl.Series(_aggregate_on_chunks(x, f_agg, chunk_len), dtype=pl.Float64),
        pl.Series(res, dtype=pl.Float64),
    )


@pytest.mark.parametrize(
    "x, param, res",
    [
        (
            pl.Series(range(9)),
            [
                {"attr": "intercept", "chunk_len": 3, "f_agg": "max"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "max"},
                {"attr": "intercept", "chunk_len": 3, "f_agg": "min"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "min"},
                {"attr": "intercept", "chunk_len": 3, "f_agg": "mean"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "mean"},
                {"attr": "intercept", "chunk_len": 3, "f_agg": "median"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "median"},
            ],
            pl.DataFrame(
                [
                    [
                        'attr_"intercept"__chunk_len_3__f_agg_"max"',
                        'attr_"slope"__chunk_len_3__f_agg_"max"',
                        'attr_"intercept"__chunk_len_3__f_agg_"min"',
                        'attr_"slope"__chunk_len_3__f_agg_"min"',
                        'attr_"intercept"__chunk_len_3__f_agg_"mean"',
                        'attr_"slope"__chunk_len_3__f_agg_"mean"',
                        'attr_"intercept"__chunk_len_3__f_agg_"median"',
                        'attr_"slope"__chunk_len_3__f_agg_"median"',
                    ],
                    [2, 3, 0, 3, 1, 3, 1, 3],
                ],
                schema=[("res_index", str), ("res_data", float)],
                orient="col",
            ),
        ),
        (
            pl.Series([float("nan"), float("nan"), float("nan"), -3, -3, -3]),
            [
                {"attr": "intercept", "chunk_len": 3, "f_agg": "max"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "max"},
                {"attr": "intercept", "chunk_len": 3, "f_agg": "min"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "min"},
                {"attr": "intercept", "chunk_len": 3, "f_agg": "mean"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "mean"},
                {"attr": "intercept", "chunk_len": 3, "f_agg": "median"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "median"},
            ],
            pl.DataFrame(
                [
                    [
                        'attr_"intercept"__chunk_len_3__f_agg_"max"',
                        'attr_"slope"__chunk_len_3__f_agg_"max"',
                        'attr_"intercept"__chunk_len_3__f_agg_"min"',
                        'attr_"slope"__chunk_len_3__f_agg_"min"',
                        'attr_"intercept"__chunk_len_3__f_agg_"mean"',
                        'attr_"slope"__chunk_len_3__f_agg_"mean"',
                        'attr_"intercept"__chunk_len_3__f_agg_"median"',
                        'attr_"slope"__chunk_len_3__f_agg_"median"',
                    ],
                    [
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    ],
                ],
                schema=[("res_index", str), ("res_data", float)],
                orient="col",
            ),
        ),
        (
            pl.Series([float("nan"), float("nan"), -3, -3, -3, -3]),
            [
                {"attr": "intercept", "chunk_len": 3, "f_agg": "max"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "max"},
                {"attr": "intercept", "chunk_len": 3, "f_agg": "min"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "min"},
                {"attr": "intercept", "chunk_len": 3, "f_agg": "mean"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "mean"},
                {"attr": "intercept", "chunk_len": 3, "f_agg": "median"},
                {"attr": "slope", "chunk_len": 3, "f_agg": "median"},
            ],
            pl.DataFrame(
                [
                    [
                        'attr_"intercept"__chunk_len_3__f_agg_"max"',
                        'attr_"slope"__chunk_len_3__f_agg_"max"',
                        'attr_"intercept"__chunk_len_3__f_agg_"min"',
                        'attr_"slope"__chunk_len_3__f_agg_"min"',
                        'attr_"intercept"__chunk_len_3__f_agg_"mean"',
                        'attr_"slope"__chunk_len_3__f_agg_"mean"',
                        'attr_"intercept"__chunk_len_3__f_agg_"median"',
                        'attr_"slope"__chunk_len_3__f_agg_"median"',
                    ],
                    [
                        -3,
                        0,
                        -3,
                        0,
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    ],
                ],
                schema=[("res_index", str), ("res_data", float)],
                orient="col",
            ),
        ),
    ],
)
def test_agg_linear_trend(x, param, res):
    assert_frame_equal(agg_linear_trend(x, param), res)


@pytest.mark.parametrize(
    "x, param, res",
    [
        (
            pl.Series(
                [
                    1.0,
                    3.5,
                    9.75,
                    25.375,
                    64.4375,
                    162.09375,
                    406.234375,
                    1016.5859375,
                    2542.46484375,
                    6357.162109375,
                ]
            ),
            [{"k": 1, "coeff": 0}, {"k": 1, "coeff": 1}],
            pl.DataFrame({"coeff_0__k_1": 1.0, "coeff_1__k_1": 2.5}),
        ),
        (
            pl.Series([1.0, 1.0, 2.5, 7.75, 23.125, 66.4375, 187.28125]),
            [
                {"k": 1, "coeff": 0},
                {"k": 1, "coeff": 1},
                {"k": 2, "coeff": 0},
                {"k": 2, "coeff": 1},
                {"k": 2, "coeff": 2},
                {"k": 2, "coeff": 3},
            ],
            pl.DataFrame(
                {
                    "coeff_0__k_1": 0.079705,
                    "coeff_1__k_1": 2.824953,
                    "coeff_0__k_2": 1.0,
                    "coeff_1__k_2": 3.5,
                    "coeff_2__k_2": -2.0,
                    "coeff_3__k_2": float("nan"),
                }
            ),
        ),
    ],
)
def test_ar_coefficient(x, param, res):
    assert_frame_equal(ar_coefficient(x, param), res)


@pytest.mark.parametrize("x, param, res", [()])
def test_augmented_dickey_fuller(x, param, res):
    assert_frame_equal(augmented_dickey_fuller(x, param), res)


@pytest.mark.parametrize(
    "x, param, res",
    [
        (
            pl.Series([0.1, 0.2, 0.3]),
            [
                {"widths": (1, 2, 3), "coeff": 2, "w": 1},
                {"widths": (1, 3), "coeff": 2, "w": 3},
                {"widths": (1, 3), "coeff": 5, "w": 3},
            ],
            pl.DataFrame(
                [
                    [
                        "coeff_2__w_1__widths_(1, 2, 3)",
                        "coeff_2__w_3__widths_(1, 3)",
                        "coeff_5__w_3__widths_(1, 3)",
                    ],
                    [0.260198, 0.234437, float("nan")],
                ],
                schema=["index", "res"],
                orient="col",
            ),
        )
    ],
)
def test_cwt_coefficients(x, param, res):
    assert_frame_equal(cwt_coefficients(x, param), res)


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
