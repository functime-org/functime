"""Forecasts on series whose future is known exactly.

Each series follows a recurrence that the fitted model can represent exactly, so the
forecast must continue it. A model that reads its lags one step out of place, or
feeds a direct horizon the wrong lag columns, misses by a wide margin.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from functime.forecasting import (
    censored_model,
    knn,
    linear_model,
    zero_inflated_model,
)

LAGS = 3
MAX_HORIZONS = 5
STRATEGIES = ["recursive", "direct", "ensemble"]
# Features are cast to Float32 before fitting
TOL = {"rtol": 1e-3, "atol": 1e-3}


def _trend(start: float, n: int) -> np.ndarray:
    return start + 2.0 * np.arange(n)


def _constant(start: float, n: int) -> np.ndarray:
    return np.full(n, start)


def _ar1(start: float, n: int) -> np.ndarray:
    # Slowly decaying oscillation, so a value one step off has the wrong sign
    y = np.empty(n)
    y[0] = start
    for t in range(1, n):
        y[t] = 1.0 - 0.95 * y[t - 1]
    return y


def _sine(start: float, n: int) -> np.ndarray:
    # Satisfies y_t = 2 cos(w) y_{t-1} - y_{t-2} + c for any phase
    return 20.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / 7 + start)


SERIES = {"trend": _trend, "constant": _constant, "ar1": _ar1, "sine": _sine}


def _panel(fn, n_obs: dict[str, int], starts: dict[str, float]):
    """Entities of different lengths, listed out of order, with exact futures."""
    frames, futures = [], {}
    for entity, n in n_obs.items():
        values = fn(starts[entity], n + MAX_HORIZONS)
        frames.append(
            pl.DataFrame({"entity": entity, "time": np.arange(n), "target": values[:n]})
        )
        futures[entity] = values[n:]
    return pl.concat(frames), futures


def _assert_forecast(y_pred: pl.DataFrame, futures: dict, n_obs: dict, fh: int):
    assert y_pred.columns == ["entity", "time", "target"]
    assert y_pred.height == fh * len(futures)
    for entity, future in futures.items():
        rows = y_pred.filter(pl.col("entity") == entity).sort("time")
        n = n_obs[entity]
        assert rows.get_column("time").to_list() == list(range(n, n + fh))
        np.testing.assert_allclose(rows.get_column("target"), future[:fh], **TOL)


@pytest.mark.parametrize("fh", [1, 3, MAX_HORIZONS])
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("series", SERIES)
def test_linear_model_continues_linear_recurrence(series, strategy, fh):
    n_obs = {"b": 30, "a": 25}
    y, futures = _panel(SERIES[series], n_obs, starts={"a": 10.0, "b": -10.0})
    model = linear_model(
        freq="1i", lags=LAGS, strategy=strategy, max_horizons=MAX_HORIZONS
    )
    y_pred = model.fit(y=y).predict(fh=fh)
    _assert_forecast(y_pred, futures, n_obs, fh)


@pytest.mark.parametrize("fh", [1, 3, MAX_HORIZONS])
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_knn_continues_repeating_pattern(strategy, fh):
    # Every lag window recurs, so its nearest neighbours share the exact next value
    pattern = [1.0, 5.0, 2.0, 8.0]

    def repeat(start: float, n: int) -> np.ndarray:
        return np.roll(pattern, -int(start))[np.arange(n) % len(pattern)]

    n_obs = {"b": 41, "a": 38}
    y, futures = _panel(repeat, n_obs, starts={"a": 0.0, "b": 1.0})
    model = knn(
        freq="1i",
        lags=len(pattern),
        strategy=strategy,
        max_horizons=MAX_HORIZONS,
        n_neighbors=3,
    )
    y_pred = model.fit(y=y).predict(fh=fh)
    _assert_forecast(y_pred, futures, n_obs, fh)


def _tree_regress(X: np.ndarray, y: np.ndarray):
    from sklearn.tree import DecisionTreeRegressor

    return DecisionTreeRegressor(random_state=0).fit(X, y)


def _tree_classify(X: np.ndarray, y: np.ndarray):
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(random_state=0).fit(X, y)


@pytest.mark.parametrize("fh", [1, 3, MAX_HORIZONS])
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    "model,pattern,threshold",
    [
        (censored_model, [1.0, 5.0, 2.0, 8.0], 3.0),
        (zero_inflated_model, [0.0, 5.0, 0.0, 8.0], 0.0),
    ],
    ids=["censored", "zero_inflated"],
)
def test_censored_continues_repeating_pattern(model, pattern, threshold, strategy, fh):
    # Every lag window recurs, so fully grown trees fit each horizon exactly; a
    # horizon that used another horizon's classifier or regressor would miss
    def repeat(start: float, n: int) -> np.ndarray:
        return np.roll(pattern, -int(start))[np.arange(n) % len(pattern)]

    n_obs = {"b": 41, "a": 38}
    y, futures = _panel(repeat, n_obs, starts={"a": 0.0, "b": 1.0})
    kwargs = {"threshold": threshold} if model is censored_model else {}
    forecaster = model(
        freq="1i",
        lags=len(pattern),
        strategy=strategy,
        max_horizons=MAX_HORIZONS,
        regress=_tree_regress,
        classify=_tree_classify,
        **kwargs,
    )
    y_pred = forecaster.fit(y=y).predict(fh=fh)

    proba = y_pred.get_column("threshold_proba")
    _assert_forecast(y_pred.drop("threshold_proba"), futures, n_obs, fh)
    for entity, future in futures.items():
        rows = y_pred.filter(pl.col("entity") == entity).sort("time")
        expected = (future[:fh] > threshold).astype(float)
        np.testing.assert_allclose(rows.get_column("threshold_proba"), expected)
    assert proba.dtype == pl.Float32


def test_censored_conformalize_ignores_threshold_proba():
    rng = np.random.default_rng(0)
    y = pl.concat(
        pl.DataFrame(
            {
                "entity": entity,
                "time": np.arange(40),
                "target": np.where(rng.random(40) < 0.4, 0.0, rng.uniform(1, 9, 40)),
            }
        )
        for entity in ["a", "b"]
    )
    forecaster = zero_inflated_model(
        freq="1i", lags=LAGS, regress=_tree_regress, classify=_tree_classify
    )
    y_pred_qnts = forecaster.conformalize(fh=3, y=y, alphas=[0.1, 0.9])

    assert y_pred_qnts.columns == ["entity", "time", "target", "quantile"]
    # Two entities, three forecast periods, two quantiles
    assert y_pred_qnts.filter(pl.col("time") >= 40).height == 2 * 3 * 2


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_forecast_uses_future_X_for_matching_horizon(strategy):
    # The target depends only on the same-period regressor
    n, fh = 30, 4
    x = np.random.default_rng(0).normal(size=n + fh)
    time = np.arange(n + fh)
    y = pl.DataFrame({"entity": "a", "time": time[:n], "target": 3.0 * x[:n] + 1.0})
    X = pl.DataFrame({"entity": "a", "time": time[:n], "x": x[:n]})
    X_future = pl.DataFrame({"entity": "a", "time": time[n:], "x": x[n:]})

    model = linear_model(
        freq="1i", lags=LAGS, strategy=strategy, max_horizons=MAX_HORIZONS
    )
    y_pred = model.fit(y=y, X=X).predict(fh=fh, X=X_future)

    np.testing.assert_allclose(y_pred.get_column("target"), 3.0 * x[n:] + 1.0, **TOL)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_forecast_invariant_to_input_row_order(strategy):
    n_obs = {"b": 30, "a": 25}
    y, _ = _panel(_ar1, n_obs, starts={"a": 10.0, "b": -10.0})
    shuffled = y.sample(fraction=1.0, shuffle=True, seed=1)

    def forecast(y: pl.DataFrame) -> pl.DataFrame:
        model = linear_model(
            freq="1i", lags=LAGS, strategy=strategy, max_horizons=MAX_HORIZONS
        )
        return model.fit(y=y).predict(fh=3).sort("entity", "time")

    expected, result = forecast(y), forecast(shuffled)
    assert result.select("entity", "time").equals(expected.select("entity", "time"))
    np.testing.assert_allclose(result.get_column("target"), expected["target"], **TOL)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    "freq,time",
    [
        (
            "1d",
            pl.date_range(dt.date(2020, 1, 1), dt.date(2020, 1, 20), "1d", eager=True),
        ),
        (
            "1h",
            pl.datetime_range(
                dt.datetime(2020, 1, 1),
                dt.datetime(2020, 1, 1, 19),
                "1h",
                time_unit="ns",
                eager=True,
            ),
        ),
        ("1i", pl.Series(range(20), dtype=pl.Int32)),
    ],
    ids=["date", "datetime_ns", "int32"],
)
def test_forecast_preserves_time_dtype(freq, time, strategy):
    y = pl.DataFrame({"entity": "a", "time": time, "target": np.arange(20.0)})
    model = linear_model(
        freq=freq, lags=LAGS, strategy=strategy, max_horizons=MAX_HORIZONS
    )
    y_pred = model.fit(y=y).predict(fh=3)

    assert y_pred.schema["time"] == time.dtype
    np.testing.assert_allclose(y_pred.get_column("target"), [20.0, 21.0, 22.0], **TOL)
