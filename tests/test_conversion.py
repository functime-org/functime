from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import functime.conversion as conversion
from functime.conversion import X_to_numpy, y_to_numpy
from functime.forecasting._regressors import GradientBoostedTreeRegressor


def test_df_to_ndarray_removed():
    assert not hasattr(conversion, "df_to_ndarray")


def test_X_to_numpy_casts_and_imputes():
    df = pl.DataFrame(
        {
            "entity_id": ["a", "a", "a", "a"],
            "time": [0, 1, 2, 3],
            "f1": [1.0, np.nan, np.inf, 3.0],
            "f2": pl.Series("f2", [10.0, None, -np.inf, 20.0], dtype=pl.Float64),
            "f3": [1, 2, 3, 4],
        }
    )

    for frame in (df, df.lazy()):
        arr = X_to_numpy(frame)
        assert arr.dtype == np.float32
        assert arr.shape == (4, 3)
        assert np.isfinite(arr).all()
        np.testing.assert_allclose(arr[:, 0], [1.0, 2.0, 2.0, 3.0], rtol=1e-6)
        np.testing.assert_allclose(arr[:, 1], [10.0, 15.0, 15.0, 20.0], rtol=1e-6)
        np.testing.assert_allclose(arr[:, 2], [1.0, 2.0, 3.0, 4.0], rtol=1e-6)


def test_y_to_numpy_zero_copy_on_contiguous_float32():
    src = np.arange(128, dtype=np.float32)
    df = pl.DataFrame(
        {
            "entity_id": ["a"] * 128,
            "time": list(range(128)),
            "target": pl.Series("target", src),
        }
    )
    expected = df.get_column("target").to_numpy(allow_copy=False)
    arr = y_to_numpy(df)

    assert arr.dtype == np.float32
    assert arr.shape == (128,)
    assert np.shares_memory(arr, expected)
    assert arr.ctypes.data == expected.ctypes.data
    np.testing.assert_array_equal(arr, expected)


def test_y_to_numpy_chunked_float32():
    chunked = pl.concat(
        [
            pl.Series("target", [1.0, 2.0], dtype=pl.Float32),
            pl.Series("target", [3.0, 4.0], dtype=pl.Float32),
        ],
        rechunk=False,
    )
    assert chunked.n_chunks() == 2
    df = pl.DataFrame(
        {
            "entity_id": ["a"] * 4,
            "time": [0, 1, 2, 3],
            "target": chunked,
        }
    )

    arr = y_to_numpy(df)
    assert arr.dtype == np.float32
    assert arr.shape == (4,)
    np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_y_to_numpy_fallback_cases():
    df_non_finite = pl.DataFrame(
        {
            "entity_id": ["a"] * 6,
            "time": list(range(6)),
            "target": pl.Series(
                "target", [2.0, 6.0, np.nan, np.inf, -np.inf, None], dtype=pl.Float32
            ),
        }
    )
    arr_imputed = y_to_numpy(df_non_finite)
    assert arr_imputed.dtype == np.float32
    np.testing.assert_allclose(arr_imputed, [2.0, 6.0, 4.0, 4.0, 4.0, 4.0], rtol=1e-6)

    for dtype in (pl.Float64, pl.Int64):
        df_cast = pl.DataFrame(
            {
                "entity_id": ["a", "a", "a"],
                "time": [0, 1, 2],
                "target": pl.Series("target", [1, 2, 3], dtype=dtype),
            }
        )
        arr_cast = y_to_numpy(df_cast)
        assert arr_cast.dtype == np.float32
        np.testing.assert_array_equal(
            arr_cast, np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )

    for series in (
        pl.Series("target", [None, None], dtype=pl.Float64),
        pl.Series("target", [np.nan, np.nan], dtype=pl.Float32),
    ):
        df_null = pl.DataFrame(
            {"entity_id": ["a", "a"], "time": [0, 1], "target": series}
        )
        arr_null = y_to_numpy(df_null)
        assert arr_null.dtype == np.float32
        assert arr_null.shape == (2,)
        assert np.isnan(arr_null).all()

    for dtype in (pl.Float32, pl.Float64):
        df_empty = pl.DataFrame(
            {
                "entity_id": pl.Series("entity_id", [], dtype=pl.String),
                "time": pl.Series("time", [], dtype=pl.Int64),
                "target": pl.Series("target", [], dtype=dtype),
            }
        )
        arr_empty = y_to_numpy(df_empty)
        assert arr_empty.dtype == np.float32
        assert arr_empty.shape == (0,)


def test_gbt_regressor_numpy_fit_predict():
    X = pl.DataFrame(
        {
            "entity_id": ["a", "a", "b", "b"],
            "time": [0, 1, 0, 1],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [100.0, 200.0, 300.0, 400.0],
            "flag": [True, False, True, False],
        }
    )
    y = pl.DataFrame(
        {
            "entity_id": ["a", "a", "b", "b"],
            "time": [0, 1, 0, 1],
            "target": [1.5, 2.5, 3.5, 4.5],
        }
    )

    captured: dict[str, object] = {}

    class _StubModel:
        def predict(self, X_pred):
            captured["predict_X"] = X_pred
            return np.full(X_pred.shape[0], 7.0, dtype=np.float32)

    def _regress(X, y, sample_weight=None):
        captured["fit_X"] = X
        captured["fit_y"] = y
        captured["sample_weight"] = sample_weight
        return _StubModel()

    regressor = GradientBoostedTreeRegressor(
        regress=_regress, fit_dtype="numpy", predict_dtype="numpy"
    )
    regressor.fit(X=X, y=y)
    preds = regressor.predict(X)

    assert isinstance(captured["fit_X"], np.ndarray)
    assert captured["fit_X"].dtype == np.float32
    assert captured["fit_X"].shape == (4, 3)

    assert isinstance(captured["fit_y"], np.ndarray)
    assert captured["fit_y"].dtype == np.float32
    np.testing.assert_allclose(captured["fit_y"], [1.5, 2.5, 3.5, 4.5], rtol=1e-6)

    assert isinstance(captured["predict_X"], np.ndarray)
    assert captured["predict_X"].dtype == np.float32
    assert captured["predict_X"].shape == (4, 3)
    np.testing.assert_array_equal(preds, np.full(4, 7.0, dtype=np.float32))


def test_gbt_regressor_arrow_fit_predict():
    pa = pytest.importorskip("pyarrow")

    X = pl.DataFrame(
        {
            "entity_id": ["a", "a", "b", "b"],
            "time": [0, 1, 0, 1],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "cat": pl.Series("cat", ["x", "y", "x", "y"], dtype=pl.Categorical),
        }
    )
    y = pl.DataFrame(
        {
            "entity_id": ["a", "a", "b", "b"],
            "time": [0, 1, 0, 1],
            "target": [10.0, 20.0, 30.0, 40.0],
        }
    )

    captured: dict[str, object] = {}

    class _StubModel:
        def predict(self, X_pred):
            captured["predict_X"] = X_pred
            return np.ones(X_pred.num_rows, dtype=np.float32)

    def _regress(X, y, sample_weight=None):
        captured["fit_X"] = X
        captured["fit_y"] = y
        return _StubModel()

    regressor = GradientBoostedTreeRegressor(
        regress=_regress, fit_dtype="arrow", predict_dtype="arrow"
    )
    regressor.fit(X=X, y=y)
    preds = regressor.predict(X)

    assert isinstance(captured["fit_X"], pa.Table)
    assert isinstance(captured["fit_y"], pa.Table)
    assert captured["fit_X"].num_rows == 4
    assert captured["fit_y"].num_rows == 4
    assert captured["fit_y"].column_names == ["entity_id", "time", "target"]

    assert isinstance(captured["predict_X"], pa.Table)
    assert captured["predict_X"].num_rows == 4
    assert captured["predict_X"].column_names == ["entity_id", "time", "f1", "cat"]
    np.testing.assert_array_equal(preds, np.ones(4, dtype=np.float32))
