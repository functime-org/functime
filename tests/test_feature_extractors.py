from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import signal

from functime.feature_extractors import (
    autoregressive_coefficients,
    cwt_coefficients,
)


@pytest.mark.parametrize("length", np.random.randint(low=1, high=101, size=5))
@pytest.mark.parametrize("widths", [(2,), (2, 5, 10, 20), (2, 5, 10, 20, 30)])
@pytest.mark.parametrize("n_coefficients", np.random.randint(low=1, high=101, size=5))
def test_cwt(length: int, widths: tuple, n_coefficients: int) -> None:
    out = cwt_coefficients(
        pl.Series([1 for _ in range(length)]),
        widths=widths,
        n_coefficients=n_coefficients,
    )
    assert len(out) == min(n_coefficients, length) * len(widths)


def test_cwt_expr_not_implemented() -> None:
    assert cwt_coefficients(pl.col("x")) is NotImplemented


def test_autoregressive_coefficients_recovery() -> None:
    rng = np.random.default_rng(42)
    n = 200
    y = np.zeros(n, dtype=np.float64)
    eps = rng.normal(scale=0.1, size=n)
    for t in range(2, n):
        y[t] = 1.5 + 0.6 * y[t - 1] - 0.25 * y[t - 2] + eps[t]

    s = pl.Series("y", y)
    coeffs = autoregressive_coefficients(s, n_lags=2)

    design = np.column_stack([y[1:-1], y[:-2], np.ones(n - 2)])
    expected = np.linalg.lstsq(design, y[2:], rcond=None)[0]
    np.testing.assert_allclose(coeffs, expected, atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(coeffs, [0.6, -0.25, 1.5], atol=0.1)


def test_autoregressive_coefficients_edge_cases() -> None:
    # Constant series has rank-deficient design matrix and must fall back cleanly.
    const = pl.Series("y", [5.0] * 20)
    coeffs = autoregressive_coefficients(const, n_lags=2)
    assert len(coeffs) == 3
    assert np.all(np.isfinite(coeffs))
    np.testing.assert_allclose(coeffs @ np.array([5.0, 5.0, 1.0]), 5.0, atol=1e-10)

    # Integer and null-containing inputs cast and impute nulls as 0.0 cleanly.
    s_int = pl.Series("y", [1, None, 3, 4, 5, 6, 7], dtype=pl.Int64)
    s_zero = pl.Series("y", [1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=pl.Float64)
    np.testing.assert_allclose(
        autoregressive_coefficients(s_int, n_lags=2),
        autoregressive_coefficients(s_zero, n_lags=2),
    )

    with pytest.raises(ValueError, match="greater than"):
        autoregressive_coefficients(pl.Series("y", [1.0, 2.0]), n_lags=2)


def test_expr_frac_diff_window_and_panel() -> None:
    df = pl.DataFrame(
        {
            "entity_id": ["a"] * 6 + ["b"] * 6,
            "y": [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    out = df.select(
        "entity_id",
        pl.col("y").ts.frac_diff(d=1.0, window_size=1).over("entity_id"),
    )
    a_vals = out.filter(pl.col("entity_id") == "a").get_column("y").to_list()
    b_vals = out.filter(pl.col("entity_id") == "b").get_column("y").to_list()
    # d=1.0 produces exact first difference: [None, y_t - y_{t-1}, ...]
    assert a_vals[0] is None
    np.testing.assert_allclose(a_vals[1:], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert b_vals[0] is None
    np.testing.assert_allclose(b_vals[1:], [0.0, 0.0, 0.0, 0.0, 0.0])

    # Internal nulls propagate instead of crashing, and d=0.0 returns Float64 identity.
    s_null = pl.DataFrame({"y": [1, 2, None, 4, 5]}).select(
        pl.col("y").ts.frac_diff(d=0.0, min_weight=1e-3)
    )
    assert s_null.schema["y"] == pl.Float64
    assert s_null.get_column("y").to_list() == [1.0, 2.0, None, 4.0, 5.0]

    mixed_panel = pl.DataFrame(
        {
            "entity_id": ["a"] * 4 + ["b"] * 4,
            "y": [1.0, 2.0, 4.0, 7.0, 1.0, None, 4.0, 7.0],
        }
    )
    mixed_out = mixed_panel.select(
        "entity_id",
        pl.col("y").ts.frac_diff(d=1.0, window_size=1).over("entity_id"),
    )
    assert mixed_out.filter(pl.col("entity_id") == "a").get_column("y").to_list() == [
        None,
        1.0,
        2.0,
        3.0,
    ]
    assert mixed_out.filter(pl.col("entity_id") == "b").get_column("y").to_list() == [
        None,
        None,
        None,
        3.0,
    ]

    with pytest.raises(ValueError, match="Either min_weight or window_size"):
        df.select(pl.col("y").ts.frac_diff(d=0.5))
    with pytest.raises(ValueError, match="Only one of min_weight or window_size"):
        df.select(pl.col("y").ts.frac_diff(d=0.5, min_weight=1e-3, window_size=5))


@pytest.mark.parametrize(
    "method, scipy_type", [("linear", "linear"), ("mean", "constant")]
)
def test_expr_detrend_panel(method: str, scipy_type: str) -> None:
    a = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0]
    b = [10.0, 8.0, 9.0, 4.0]
    df = pl.DataFrame({"entity_id": ["a"] * 6 + ["b"] * 4, "y": a + b})
    out = df.select(pl.col("y").ts.detrend(method).over("entity_id"))
    expected = np.concatenate(
        [signal.detrend(a, type=scipy_type), signal.detrend(b, type=scipy_type)]
    )
    np.testing.assert_allclose(out.get_column("y").to_numpy(), expected, atol=1e-12)

    with pytest.raises(ValueError, match="Unknown detrend method"):
        pl.col("y").ts.detrend("quadratic")
