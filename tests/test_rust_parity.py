from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from functime import _rust
from functime.feature_extractors import (
    autoregressive_coefficients,
    lempel_ziv_complexity,
)


def _ar_series(n: int = 80, seed: int = 42) -> pl.Series:
    rng = np.random.default_rng(seed)
    vals = np.zeros(n, dtype=np.float64)
    eps = rng.normal(0.0, 0.5, size=n)
    for i in range(2, n):
        vals[i] = 0.6 * vals[i - 1] - 0.25 * vals[i - 2] + 1.5 + eps[i]
    return pl.Series("y", vals)


def _panel_df() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    n_per_group = 60
    rows = []
    for gid, shift in [("a", 0.0), ("b", 5.0), ("c", -3.0)]:
        base = rng.normal(shift, 1.0, size=n_per_group)
        base[40:] += 4.0
        for idx, val in enumerate(base):
            rows.append({"id": gid, "t": idx, "y": float(val)})
    return pl.DataFrame(rows)


def test_pure_python_autoregressive_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_rust, "HAS_RUST", False)

    series = _ar_series()
    for n_lags in (1, 2, 4):
        coeffs = autoregressive_coefficients(series, n_lags=n_lags)
        assert coeffs.shape == (n_lags + 1,)
        assert np.all(np.isfinite(coeffs))

    # Constant and rank-deficient inputs must succeed with exact fit.
    for n_lags in (1, 3):
        const = pl.Series("y", [5.0] * 25)
        coeffs = autoregressive_coefficients(const, n_lags=n_lags)
        assert coeffs.shape == (n_lags + 1,)
        assert np.all(np.isfinite(coeffs))
        assert np.isclose(np.sum(coeffs[:-1]) * 5.0 + coeffs[-1], 5.0)

    # Periodic rank-deficient series (column 1 == column 3).
    periodic = pl.Series("y", [1.0, 2.0] * 20)
    coeffs_p = autoregressive_coefficients(periodic, n_lags=3)
    assert coeffs_p.shape == (4,)
    assert np.all(np.isfinite(coeffs_p))

    # Integer input with nulls and underdetermined length < n_lags + 1.
    int_with_null = pl.Series("y", [1, None, 3, 4, 5, 6, 7], dtype=pl.Int64)
    coeffs_int = autoregressive_coefficients(int_with_null, n_lags=2)
    assert coeffs_int.shape == (3,)
    assert np.all(np.isfinite(coeffs_int))

    short = pl.Series("y", [1.0, 2.0, 3.0])
    coeffs_short = autoregressive_coefficients(short, n_lags=3)
    assert coeffs_short.shape == (4,)
    assert np.all(coeffs_short == 0.0)


def test_pure_python_lempel_ziv_complexity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_rust, "HAS_RUST", False)

    df = pl.DataFrame(
        {
            "id": ["a"] * 10 + ["b"] * 10,
            "v": [0.0, 1.0, 0.0, 1.0, None, 2.0, 0.0, 1.0, 2.0, 0.0]
            + [5.0, 5.0, 0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, None],
            "thr": [0.5] * 20,
        }
    )

    res_select = df.select(
        lz=pl.col("v").ts.lempel_ziv_complexity(threshold=0.5, as_ratio=False),
        lz_ratio=pl.col("v").ts.lempel_ziv_complexity(threshold=0.5, as_ratio=True),
        lz_expr_thr=pl.col("v").ts.lempel_ziv_complexity(
            threshold=pl.col("thr"), as_ratio=False
        ),
    )
    assert res_select.schema["lz"] == pl.UInt32
    assert res_select.schema["lz_ratio"] == pl.Float64
    assert res_select.item(0, "lz") == res_select.item(0, "lz_expr_thr")
    assert np.isclose(res_select.item(0, "lz_ratio"), res_select.item(0, "lz") / 20)

    res_gb = df.group_by("id", maintain_order=True).agg(
        lz=pl.col("v").ts.lempel_ziv_complexity(threshold=0.5, as_ratio=False),
        lz_ratio=pl.col("v").ts.lempel_ziv_complexity(threshold=0.5, as_ratio=True),
    )
    assert res_gb.shape == (2, 3)
    assert res_gb.schema["lz"] == pl.UInt32
    assert np.isclose(res_gb.item(0, "lz_ratio"), res_gb.item(0, "lz") / 10)

    res_over = df.select(
        lz=pl.col("v")
        .ts.lempel_ziv_complexity(threshold=0.5, as_ratio=False)
        .over("id")
    )
    assert res_over.height == 20
    assert res_over.item(0, "lz") == res_gb.item(0, "lz")
    assert res_over.item(15, "lz") == res_gb.item(1, "lz")

    # Series wrapper and edge cases (all nulls, empty).
    s = pl.Series("v", [0, 1, 1, 0, 1, None, 0, 1])
    assert isinstance(lempel_ziv_complexity(s, threshold=0, as_ratio=False), int)
    all_null = pl.DataFrame({"v": pl.Series([None, None, None], dtype=pl.Float64)})
    assert (
        all_null.select(
            pl.col("v").ts.lempel_ziv_complexity(0.0, as_ratio=False)
        ).item()
        == 2
    )
    empty = pl.DataFrame({"v": pl.Series([], dtype=pl.Float64)})
    assert (
        empty.select(pl.col("v").ts.lempel_ziv_complexity(0.0, as_ratio=False)).item()
        == 0
    )


def test_pure_python_cusum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_rust, "HAS_RUST", False)

    rng = np.random.default_rng(123)
    up_shift = np.concatenate([rng.normal(0.0, 0.1, 40), rng.normal(1.5, 0.1, 20)])
    down_shift = np.concatenate([rng.normal(0.0, 0.1, 40), rng.normal(-1.5, 0.1, 20)])
    df = pl.DataFrame(
        {
            "id": ["up"] * 60 + ["down"] * 60,
            "y": np.concatenate([up_shift, down_shift]),
        }
    )

    out_select = df.filter(pl.col("id") == "up").select(
        ev=pl.col("y").ts.cusum(threshold=3.0, warmup_period=20, drift=0.5)
    )
    assert out_select.schema["ev"] == pl.Int32
    assert out_select.get_column("ev").sum() >= 1
    assert out_select.head(20).get_column("ev").sum() == 0

    out_over = df.with_columns(
        ev=pl.col("y").ts.cusum(threshold=3.0, warmup_period=20, drift=0.5).over("id")
    )
    assert out_over.schema["ev"] == pl.Int32
    assert (
        out_over.group_by("id")
        .agg(pl.col("ev").sum())
        .filter(pl.col("id") == "down")
        .item(0, "ev")
        >= 1
    )

    out_gb = df.group_by("id", maintain_order=True).agg(
        ev=pl.col("y").ts.cusum(threshold=3.0, warmup_period=20, drift=0.5)
    )
    assert out_gb.schema["ev"] == pl.List(pl.Int32)
    assert (
        out_gb.explode("ev").get_column("ev").to_list()
        == out_over.get_column("ev").to_list()
    )

    # Nulls, constant warmup, and all-null warmup must not raise.
    res_const = pl.DataFrame({"y": [1.0] * 15 + [5.0] + [1.0] * 5}).select(
        pl.col("y").ts.cusum(threshold=2.0, warmup_period=10, drift=0.0)
    )
    assert res_const.schema["y"] == pl.Int32
    assert res_const.item(15, "y") == 1

    res_null_warmup = pl.DataFrame({"y": [None] * 10 + [1.0, 2.0, 3.0]}).select(
        pl.col("y").ts.cusum(threshold=2.0, warmup_period=5, drift=0.0)
    )
    assert res_null_warmup.schema["y"] == pl.Int32

    res_int = pl.DataFrame({"y": [1, 2, 1, 2, 1, 2, 10, 11, 12]}).select(
        pl.col("y").ts.cusum(threshold=2.0, warmup_period=4, drift=0.0)
    )
    assert res_int.schema["y"] == pl.Int32


def test_pure_python_frac_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_rust, "HAS_RUST", False)

    df = _panel_df()
    res_w = df.with_columns(
        fd=pl.col("y").ts.frac_diff(d=0.5, window_size=5).over("id")
    )
    assert res_w.schema["fd"] == pl.Float64
    for gid in ("a", "b", "c"):
        sub = res_w.filter(pl.col("id") == gid).get_column("fd")
        assert sub[:5].null_count() == 5
        assert sub[5:].null_count() == 0

    res_mw = df.select(pl.col("y").ts.frac_diff(d=0.4, min_weight=1e-2))
    assert res_mw.schema["y"] == pl.Float64
    assert res_mw.height == df.height

    # Null handling in pure Python.
    null_df = pl.DataFrame({"y": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0]})
    res_null = null_df.select(pl.col("y").ts.frac_diff(d=0.5, window_size=2))
    assert res_null.schema["y"] == pl.Float64
    # window_size=2 produces 3 weights (width=2): nulls at indices 0, 1 (leading) and 2, 3, 4 (overlap index 2).
    assert res_null.get_column("y").is_null().to_list() == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]

    # window_size=0, short, empty, integer inputs, and invalid params.
    assert pl.DataFrame({"y": [1, 2, 3, 4]}).select(
        pl.col("y").ts.frac_diff(d=0.5, window_size=0)
    ).get_column("y").to_list() == [1.0, 2.0, 3.0, 4.0]
    assert (
        pl.DataFrame({"y": [1.0, 2.0]})
        .select(pl.col("y").ts.frac_diff(d=0.5, window_size=5))
        .get_column("y")
        .null_count()
        == 2
    )
    assert (
        pl.DataFrame({"y": pl.Series([], dtype=pl.Float64)})
        .select(pl.col("y").ts.frac_diff(d=0.5, window_size=3))
        .height
        == 0
    )

    with pytest.raises(ValueError, match="Only one"):
        pl.col("y").ts.frac_diff(d=0.5, min_weight=0.1, window_size=3)
    with pytest.raises(ValueError, match="Either"):
        pl.col("y").ts.frac_diff(d=0.5)
    with pytest.raises(ValueError, match="non-negative"):
        pl.col("y").ts.frac_diff(d=0.5, window_size=-1)


@pytest.mark.skipif(not _rust.HAS_RUST, reason="functime-rust is not installed")
@pytest.mark.parametrize("n_lags", [1, 2, 5])
def test_rust_parity_autoregressive_coefficients(
    monkeypatch: pytest.MonkeyPatch, n_lags: int
) -> None:
    series_list = [
        _ar_series(),
        pl.Series("y", [5.0] * 30),
        pl.Series("y", [1.0, 2.0, 1.0, 2.0] * 8),
        pl.Series("y", [1.0, None, 2.5, 3.5, None, 5.0, 6.0, 7.0, 8.5, 9.0]),
        pl.Series("y", [1, 3, 2, 5, 4, 7, 6, 9, 8, 11], dtype=pl.Int64),
    ]
    for s in series_list:
        monkeypatch.setattr(_rust, "HAS_RUST", True)
        rust_out = autoregressive_coefficients(s, n_lags=n_lags)
        monkeypatch.setattr(_rust, "HAS_RUST", False)
        py_out = autoregressive_coefficients(s, n_lags=n_lags)
        np.testing.assert_allclose(rust_out, py_out, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _rust.HAS_RUST, reason="functime-rust is not installed")
@pytest.mark.parametrize("as_ratio", [False, True])
def test_rust_parity_lempel_ziv_complexity(
    monkeypatch: pytest.MonkeyPatch, as_ratio: bool
) -> None:
    df = pl.DataFrame(
        {
            "id": ["a"] * 15 + ["b"] * 15 + ["c"] * 15,
            "v": (
                [
                    0.0,
                    1.0,
                    0.0,
                    2.0,
                    None,
                    1.0,
                    0.0,
                    3.0,
                    2.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    2.0,
                ]
                + [5.0] * 15
                + [
                    None,
                    1.0,
                    None,
                    2.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0,
                    None,
                    4.0,
                    1.0,
                    0.0,
                ]
            ),
        }
    )

    def _run(use_rust: bool) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        monkeypatch.setattr(_rust, "HAS_RUST", use_rust)
        sel = df.select(pl.col("v").ts.lempel_ziv_complexity(0.5, as_ratio=as_ratio))
        gb = df.group_by("id", maintain_order=True).agg(
            pl.col("v").ts.lempel_ziv_complexity(0.5, as_ratio=as_ratio)
        )
        ov = df.select(
            pl.col("v").ts.lempel_ziv_complexity(0.5, as_ratio=as_ratio).over("id")
        )
        return sel, gb, ov

    r_sel, r_gb, r_ov = _run(True)
    p_sel, p_gb, p_ov = _run(False)
    assert_frame_equal(r_sel, p_sel)
    assert_frame_equal(r_gb, p_gb)
    assert_frame_equal(r_ov, p_ov)


@pytest.mark.skipif(not _rust.HAS_RUST, reason="functime-rust is not installed")
@pytest.mark.parametrize("drift", [0.0, 0.5])
def test_rust_parity_cusum(monkeypatch: pytest.MonkeyPatch, drift: float) -> None:
    rng = np.random.default_rng(99)
    s1 = list(rng.normal(0.0, 0.2, 30)) + list(rng.normal(2.0, 0.2, 20))
    s2 = list(rng.normal(0.0, 0.2, 30)) + list(rng.normal(-2.0, 0.2, 20))
    s1[5] = None
    s1[35] = None
    df = pl.DataFrame({"id": ["a"] * 50 + ["b"] * 50, "y": s1 + s2})

    def _run(use_rust: bool) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        monkeypatch.setattr(_rust, "HAS_RUST", use_rust)
        expr = pl.col("y").ts.cusum(threshold=3.0, warmup_period=15, drift=drift)
        return (
            df.select(expr),
            df.group_by("id", maintain_order=True).agg(expr),
            df.select(expr.over("id")),
        )

    r_sel, r_gb, r_ov = _run(True)
    p_sel, p_gb, p_ov = _run(False)
    assert_frame_equal(r_sel, p_sel)
    assert_frame_equal(r_gb, p_gb)
    assert_frame_equal(r_ov, p_ov)

    # Constant and all-null warmup edge cases.
    const_df = pl.DataFrame({"y": [2.0] * 12 + [5.0] + [2.0] * 5})
    null_df = pl.DataFrame({"y": [None] * 6 + [1.0, 2.0, 3.0]})
    expr_c = pl.col("y").ts.cusum(threshold=2.0, warmup_period=8, drift=drift)
    expr_n = pl.col("y").ts.cusum(threshold=2.0, warmup_period=4, drift=drift)

    monkeypatch.setattr(_rust, "HAS_RUST", True)
    r_const = const_df.select(expr_c)
    r_null = null_df.select(expr_n)
    monkeypatch.setattr(_rust, "HAS_RUST", False)
    p_const = const_df.select(expr_c)
    p_null = null_df.select(expr_n)
    assert_frame_equal(r_const, p_const)
    assert_frame_equal(r_null, p_null)


@pytest.mark.skipif(not _rust.HAS_RUST, reason="functime-rust is not installed")
@pytest.mark.parametrize(
    ("d", "min_weight", "window_size"),
    [
        (0.5, 1e-2, None),
        (0.35, None, 5),
        (0.0, 1e-3, None),
        (1.0, None, 3),
        (0.5, None, 0),
    ],
)
def test_rust_parity_frac_diff(
    monkeypatch: pytest.MonkeyPatch,
    d: float,
    min_weight: float | None,
    window_size: int | None,
) -> None:
    df = _panel_df()
    null_df = pl.DataFrame(
        {
            "id": ["a"] * 12 + ["b"] * 12,
            "y": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
            + [None, 2.0, 3.0, 4.0, 5.0, None, 7.0, 8.0, 9.0, 10.0, 11.0, None],
        }
    )

    def _run(
        frame: pl.DataFrame, use_rust: bool
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        monkeypatch.setattr(_rust, "HAS_RUST", use_rust)
        expr = pl.col("y").ts.frac_diff(
            d=d, min_weight=min_weight, window_size=window_size
        )
        return (
            frame.select(expr),
            frame.group_by("id", maintain_order=True).agg(expr),
            frame.select(expr.over("id")),
        )

    for frame in (df, null_df):
        r_sel, r_gb, r_ov = _run(frame, True)
        p_sel, p_gb, p_ov = _run(frame, False)
        assert_frame_equal(r_sel, p_sel, rel_tol=1e-10, abs_tol=1e-10)
        assert_frame_equal(r_gb, p_gb, rel_tol=1e-10, abs_tol=1e-10)
        assert_frame_equal(r_ov, p_ov, rel_tol=1e-10, abs_tol=1e-10)
