from typing import List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pytest

# AttributeError: module 'polars' has no attribute 'testing'
from polars.testing import assert_frame_equal
from scipy import signal
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer

from functime.preprocessing import (
    boxcox,
    detrend,
    diff,
    fractional_diff,
    lag,
    roll,
    scale,
    yeojohnson,
)


@pytest.fixture
def fruits_example():
    # Entity, time
    fruits = ["apple", "mango", "pineapple", "pear"]
    colors = ["red", "yellow", "yellow", "green"]
    shapes = ["round", "oval", "oval", "oval"]
    dates = pd.date_range("2000-01-01", periods=24)
    freshness_coef = np.array([0.025, 0.05, 0.075, 0.1])
    freshness = 1 - np.array([np.arange(len(dates))] * len(fruits)).T * freshness_coef
    freshness = np.maximum(freshness, 0)
    return fruits, dates, colors, shapes, freshness


@pytest.fixture
def fruits_panel_data(fruits_example):
    # Example values
    fruits, dates, colors, shapes, freshness = fruits_example
    # Prepare test data
    X_panel = pl.DataFrame(
        {
            "fruit": fruits * len(dates),
            "date": dates.repeat(len(fruits)),
            "freshness": freshness.flatten(),
        }
    ).sort(by=["fruit", "date"])
    X_static = pl.DataFrame({"fruit": fruits, "color": colors, "shape": shapes})
    X_ts = pl.DataFrame({"date": dates, "day_of_week": list(dates.day_of_week)})
    return X_panel, X_static, X_ts


def pd_gb_transform(df: pd.DataFrame, model):
    return df.groupby("series_id").apply(lambda s: model.fit_transform(s))


@pytest.mark.benchmark
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_sklearn_impute(strategy, pd_X, benchmark):
    imputer = SimpleImputer(strategy=strategy)
    benchmark(pd_gb_transform, pd_X, imputer)


@pytest.mark.benchmark
def test_sklearn_boxcox(pd_X, benchmark):
    # All values must be stricty positive
    X = pd_X.abs() + 0.001
    transformer = PowerTransformer(method="box-cox", standardize=False)
    benchmark(pd_gb_transform, X, transformer)


@pytest.fixture(
    params=[[1], [1, 2, 3], list(range(7, 15))],
    ids=lambda x: f"lags({min(x)},{max(x)})",
)
def lags(request):
    return request.param


@pytest.fixture(
    params=[
        [2],
        [3, 4, 5],
    ],
    ids=lambda x: "_".join([str(i) for i in x]),
)
def window_sizes(request):
    return request.param


@pytest.fixture(
    params=[
        ["mean"],
        ["std"],
        ["mean", "std"],
    ],
    ids=lambda x: "_".join(x),
)
def stats(request):
    return request.param


def pd_lag(X: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """Lag values in panel pd.DataFrame by `lags` and suffix k-lag
    column names with `_lag_{k}`. Includes original columns.
    """
    gb = X.groupby(level=0)
    X_lags = []
    for _lag in lags:
        X_lag = gb.shift(_lag).add_suffix(f"__lag_{_lag}")
        X_lags.append(X_lag)
    return X.join(pd.concat(X_lags, axis=1))


def pd_roll(X: pd.DataFrame, window_sizes: List[int], stats: str):
    # Pandas currently does not support group_by with rolling on multi-index
    # See: https://github.com/pandas-dev/pandas/issues/34642
    # To bypass this, we perform a few pivots so that the rolling
    # operation is performed on a single index, but the final
    # result is a MultiIndex DataFrame
    X_wide = X.unstack(level=0)
    reducers = {
        "mean": lambda X, w: (X.rolling(w).mean().shift(1), f"rolling_mean_{w}"),
        "std": lambda X, w: (X.rolling(w).std().shift(1), f"rolling_std_{w}"),
    }
    X_window_sizes = []
    for w in window_sizes:
        X_stats = []
        for stat in stats:
            X_stat, suffix = reducers[stat](X_wide, w)
            X_stat.columns = pd.MultiIndex.from_tuples(
                [(f"{x}__{suffix}", y) for x, y in X_stat.columns],
                names=X_stat.columns.names,
            )
            X_stats.append(X_stat)
        X_window = (
            pd.concat(X_stats, axis=1)
            .stack(dropna=False)
            .reorder_levels([1, 0], axis=0)
        )
        X_window_sizes.append(X_window)
    X_new = pd.concat(X_window_sizes, axis=1).sort_index()  # Defensive sort
    return X_new


@pytest.fixture
def lagged_pd_dataframe(
    pd_X: pd.DataFrame, lags: List[int]
) -> Tuple[List[int], pd.DataFrame]:
    return lags, pd_lag(pd_X, lags=lags)


@pytest.fixture
def rolling_pd_dataframe(
    pd_X: pd.DataFrame, window_sizes: List[int], stats: List[str]
) -> Tuple[List[int], List[int], pd.DataFrame]:
    return window_sizes, stats, pd_roll(pd_X, window_sizes=window_sizes, stats=stats)


@pytest.mark.benchmark
def test_pd_lag(pd_X, lags, benchmark):
    benchmark(pd_lag, pd_X, lags=lags)


@pytest.mark.benchmark
def test_pd_roll(pd_X, stats, window_sizes, benchmark):
    benchmark(pd_roll, pd_X, window_sizes=window_sizes, stats=stats)


def test_lag(pd_X, lagged_pd_dataframe, benchmark):
    X = pl.from_pandas(pd_X.reset_index())
    lags, df = lagged_pd_dataframe
    entity_col = X.columns[0]
    result = benchmark(lambda: lag(lags=lags)(X=X.lazy()).collect()).sort(entity_col)
    expected = df.dropna().reset_index().loc[:, result.columns]
    assert_frame_equal(result, pl.DataFrame(expected))


def test_roll(pd_X, rolling_pd_dataframe, benchmark):
    X = pl.from_pandas(pd_X.reset_index()).lazy()
    window_sizes, stats, df = rolling_pd_dataframe
    result = benchmark(
        lambda: roll(window_sizes=window_sizes, stats=stats, freq="1d")(X=X).collect()
    )
    expected = df.reset_index().loc[:, result.columns]
    assert_frame_equal(result, pl.DataFrame(expected), check_exact=False, rtol=0.01)


def test_scale(pd_X):
    entity_col = pd_X.index.names[0]
    numeric_cols = pd_X.select_dtypes(include=["float"]).columns
    pd_X = pd_X.assign(**{col: pd_X[col].abs() for col in numeric_cols}).replace(0, 1)
    expected = pd_X.groupby(entity_col)[numeric_cols].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    X = pl.from_pandas(pd_X.reset_index()).lazy()
    transformer = scale()
    X_new = X.pipe(transformer).collect()
    assert_frame_equal(X_new, pl.DataFrame(expected.reset_index()))
    X_original = X_new.pipe(transformer.invert)
    assert_frame_equal(X_original, X, check_dtype=False)


@pytest.mark.parametrize("sp", [1])
def test_diff(pd_X, sp):
    entity_col, time_col = pd_X.index.names
    idx_cols = (entity_col, time_col)
    numeric_cols = pd_X.select_dtypes(include=["float"]).columns
    expected_X_new = pd_X.groupby(entity_col, group_keys=False)[numeric_cols].diff(
        periods=sp
    )
    X = pl.from_pandas(pd_X.reset_index()).lazy()
    transform = diff(order=1, sp=sp)
    X_new = transform(X=X)
    pd.testing.assert_frame_equal(
        X_new.sort([entity_col, time_col]).collect().to_pandas(),
        expected_X_new.reset_index(),
        check_dtype=False,
        check_categorical=False,
    )
    X_original = transform.invert(X_new)
    assert_frame_equal(
        X_original.sort(idx_cols).collect(),
        X_new.select(idx_cols).join(X, on=idx_cols, how="left").collect(),
        check_dtype=False,
    )


def test_boxcox(pd_X):
    entity_col = pd_X.index.names[0]
    numeric_cols = pd_X.select_dtypes(include=["float"]).columns
    # The Box-Cox transformation can only be applied to strictly positive data
    # Must be > 0
    pd_X = pd_X.assign(**{col: pd_X[col].abs() for col in numeric_cols}).replace(0, 1)

    transformer = PowerTransformer(method="box-cox", standardize=False)
    expected = pd_X.groupby(entity_col)[numeric_cols].transform(
        lambda x: np.concatenate(transformer.fit_transform(x.values.reshape(-1, 1)))
    )
    X = pl.from_pandas(pd_X.reset_index()).lazy()
    transformer = boxcox()
    X_new = X.pipe(transformer).collect()
    assert_frame_equal(X_new, pl.DataFrame(expected.reset_index()))
    X_original = X_new.pipe(transformer.invert)
    assert_frame_equal(X_original, X, check_dtype=False)


def test_yeojohnson(pd_X):
    entity_col = pd_X.index.names[0]
    numeric_cols = pd_X.select_dtypes(include=["float"]).columns
    print(pd_X.info())
    print(pd_X[numeric_cols].head())  # Print the first row of each group
    transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    expected = pd_X.groupby(entity_col)[numeric_cols].transform(
        lambda x: np.concatenate(transformer.fit_transform(x.values.reshape(-1, 1)))
    )
    X = pl.from_pandas(pd_X.reset_index()).lazy()
    transformer = yeojohnson()
    X_new = X.pipe(transformer).collect()
    assert_frame_equal(X_new, pl.DataFrame(expected.reset_index()))
    X_original = X_new.pipe(transformer.invert)
    assert_frame_equal(X_original, X, check_dtype=False)


@pytest.mark.parametrize("method", ["linear", "mean"])
def test_detrend(method, pd_X):
    entity_col = pd_X.index.names[0]
    expected = pd_X.groupby(entity_col).transform(
        signal.detrend, type=method if method == "linear" else "constant"
    )
    X = pl.from_pandas(pd_X.reset_index()).lazy()
    transformer = detrend(method=method, freq="1d")
    X_new = X.pipe(transformer).collect()
    assert_frame_equal(X_new, pl.DataFrame(expected.reset_index()))
    X_original = X_new.pipe(transformer.invert)
    assert_frame_equal(X_original, X, check_dtype=False)


def pd_fractional_diff(df, d, thres):
    """Pandas implementation of fracdiff from Marcos Lopez de Prado."""

    def getWeights_FFD(d, thres):
        # thres>0 drops insignificant weights
        w, k = [1.0], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
        w = np.array(w[::-1]).reshape(-1, 1)
        return w

    def fracDiff_FFD(series, d, thres=1e-5):
        # 1) Compute weights for the longest series
        w = getWeights_FFD(d, thres)
        width = len(w) - 1
        # 2) Apply weights to values
        df = {"time": series[series.columns[0]]}
        for name in series.columns[1:]:
            seriesF = series[[name]].dropna()
            df_ = pd.Series()
            for iloc1 in range(width, seriesF.shape[0]):
                loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
                if not np.isfinite(series.loc[loc1, name]):
                    continue  # exclude NAs
                df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
                df[name] = df_.copy(deep=True)
        df = pd.concat(df, axis=1)
        return df

    numeric_cols = df.select_dtypes(include=["float"]).columns
    cols = [df.index.names[1]]
    cols.extend(numeric_cols)
    return (
        df.reset_index().groupby(df.index.names[0])[cols].apply(fracDiff_FFD, d, thres)
    )


def test_fractional_diff(pd_X):
    X = pl.from_pandas(pd_X.reset_index()).lazy()
    entity_col = pd_X.index.names[0]
    time_col = pd_X.index.names[1]
    transformer = fractional_diff(d=0.5, min_weight=1e-3)
    X_new = X.pipe(transformer).collect()
    expected = (
        pd_fractional_diff(pd_X, d=0.5, thres=1e-3)
        .reset_index()
        .drop(columns="level_1")
    )
    assert_frame_equal(
        X_new.drop_nulls().sort(entity_col, time_col),
        pl.DataFrame(expected).drop_nulls().sort(entity_col, time_col),
    )


### Temporarily commented out. Uncomment when benchmarking is ready. ###
# @pytest.mark.benchmark(group="fractional_diff")
# def test_fractional_diff_benchmark_functime(pd_X, benchmark):
#     X = pl.from_pandas(pd_X.reset_index()).lazy()
#     entity_col = pd_X.index.names[0]
#     time_col = pd_X.index.names[1]
#     transformer = fractional_diff(d=0.5, min_weight=1e-3)
#     X_new = X.pipe(transformer)
#     benchmark(X_new.collect)


# @pytest.mark.benchmark(group="fractional_diff")
# def test_fractional_diff_benchmark_pd(pd_X, benchmark):
#     benchmark(pd_fractional_diff, pd_X, d=0.5, thres=1e-3)
