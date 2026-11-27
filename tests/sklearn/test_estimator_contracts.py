from __future__ import annotations

import pickle
import warnings
from collections.abc import Sequence
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal
from scipy import sparse
from sklearn import clone, config_context
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_mixin_order, parametrize_with_checks

from functime._frame import TemporalFrame
from functime.sklearn import BaseTemporalTransformer
from functime.sklearn._base import (
    _DataFrameTransformerMixin,
    _TemporalRegressorMixin,
)

_TRANSFORMER_ONLY_NUMPY_CHECKS: dict[str, str] = {
    "check_readonly_memmap_input": "DataFrame-only transformer rejects NumPy arrays.",
    "check_n_features_in_after_fitting": "DataFrame-only transformer rejects NumPy arrays.",
    "check_estimators_overwrite_params": "DataFrame-only transformer rejects NumPy arrays.",
    "check_dont_overwrite_parameters": "DataFrame-only transformer rejects NumPy arrays.",
    "check_estimators_fit_returns_self": "DataFrame-only transformer rejects NumPy arrays.",
    "check_fit_score_takes_y": "DataFrame-only transformer rejects NumPy arrays.",
    "check_estimators_dtypes": "DataFrame-only transformer rejects NumPy arrays.",
    "check_complex_data": "DataFrame-only transformer rejects NumPy arrays with TypeError.",
    "check_dtype_object": "DataFrame-only transformer rejects NumPy arrays.",
    "check_pipeline_consistency": "DataFrame-only transformer rejects NumPy arrays.",
    "check_estimators_pickle": "DataFrame-only transformer rejects NumPy arrays.",
    "check_array_api_input": "DataFrame-only transformer rejects NumPy arrays.",
    "check_f_contiguous_array_estimator": "DataFrame-only transformer rejects NumPy arrays.",
    "check_transformer_data_not_an_array": "DataFrame-only transformer rejects list-of-lists inputs.",
    "check_positive_only_tag_during_fit": "DataFrame-only transformer rejects NumPy arrays.",
    "check_transformer_general": "DataFrame-only transformer rejects NumPy arrays.",
    "check_transformer_preserve_dtypes": "DataFrame-only transformer rejects NumPy arrays.",
    "check_methods_sample_order_invariance": "DataFrame-only transformer rejects NumPy arrays.",
    "check_methods_subset_invariance": "DataFrame-only transformer rejects NumPy arrays.",
    "check_fit2d_1sample": "DataFrame-only transformer rejects NumPy arrays.",
    "check_fit2d_1feature": "DataFrame-only transformer rejects NumPy arrays.",
    "check_dict_unchanged": "DataFrame-only transformer rejects NumPy arrays.",
    "check_fit_idempotent": "DataFrame-only transformer rejects NumPy arrays.",
    "check_fit_check_is_fitted": "DataFrame-only transformer rejects NumPy arrays.",
    "check_n_features_in": "DataFrame-only transformer rejects NumPy arrays.",
    "check_fit2d_predict1d": "DataFrame-only transformer rejects NumPy arrays.",
}


class _DummyTemporalTransformer(BaseTemporalTransformer):
    """Dummy temporal transformer for contract verification."""

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        offset: float = 1.0,
    ) -> None:
        super().__init__(time_col=time_col, entity_col=entity_col)
        self.offset = offset

    def _fit_frame(self, frame: TemporalFrame, y: Any = None) -> None:
        self._fit_calls = getattr(self, "_fit_calls", 0) + 1
        self._state_ = frame.lf.select(frame.target_cols).mean().collect()
        self.feature_names_out_ = tuple(frame.columns)

    def _transform_frame(self, frame: TemporalFrame) -> TemporalFrame:
        self._transform_calls = getattr(self, "_transform_calls", 0) + 1
        return frame.with_exprs(
            [(pl.col(col) + self.offset).alias(col) for col in frame.target_cols]
        )

    def _fit_transform_frame(
        self, frame: TemporalFrame, y: Any = None
    ) -> TemporalFrame:
        self._fit_transform_calls = getattr(self, "_fit_transform_calls", 0) + 1
        self._fit_frame(frame, y)
        return frame.with_exprs(
            [(pl.col(col) + self.offset).alias(col) for col in frame.target_cols]
        )


class _DummyChronologicalLagTransformer(BaseTemporalTransformer):
    """Dummy temporal transformer exercising chronological `.over` execution."""

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        lag: int = 1,
    ) -> None:
        super().__init__(time_col=time_col, entity_col=entity_col)
        self.lag = lag

    def _fit_frame(self, frame: TemporalFrame, y: Any = None) -> None:
        self.feature_names_out_ = (
            *frame.columns,
            *(f"{col}_lag_{self.lag}" for col in frame.target_cols),
        )

    def _transform_frame(self, frame: TemporalFrame) -> TemporalFrame:
        return frame.with_exprs(
            [
                pl.col(col).shift(self.lag).alias(f"{col}_lag_{self.lag}")
                for col in frame.target_cols
            ],
            chronological=True,
        )


class _DummyDataFrameTransformer(_DataFrameTransformerMixin):
    """Dummy DataFrame transformer supporting optional Polars expressions."""

    def __init__(self, scale: float = 2.0, expr: pl.Expr | None = None) -> None:
        self.scale = scale
        self.expr = expr

    def _fit_df(self, df: pl.DataFrame, y: Any = None) -> None:
        self.columns_ = tuple(df.columns)
        self.feature_names_out_ = tuple(df.columns)

    def _transform_df(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.expr is not None:
            return df.with_columns(self.expr)
        return df.with_columns(
            [
                pl.col(col) * self.scale
                for col in self.columns_
                if df.schema[col].is_numeric()
            ]
        )


class _DummyTemporalRegressor(_TemporalRegressorMixin):
    """Dummy temporal regressor for mixin and tag contract verification."""

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col


def assert_temporal_transformer_contracts(
    estimator: BaseTemporalTransformer,
    df_panel: pl.DataFrame,
    *,
    df_univariate: pl.DataFrame | None = None,
    df_multi_entity: pl.DataFrame | None = None,
) -> None:
    """Verify full scikit-learn and DataFrame contracts for a temporal transformer."""
    unfitted = clone(estimator)
    with pytest.raises(NotFittedError):
        unfitted.transform(df_panel)
    with pytest.raises(NotFittedError):
        unfitted.get_feature_names_out()

    params_before = dict(unfitted.get_params(deep=True))
    fitted = unfitted.fit(df_panel)
    assert fitted is unfitted
    assert fitted.get_params(deep=True) == params_before

    # Idempotence and pickle / clone round-trip.
    out_first = fitted.transform(df_panel)
    assert isinstance(out_first, pl.DataFrame)
    fitted_twice = fitted.fit(df_panel)
    assert fitted_twice is fitted
    out_second = fitted_twice.transform(df_panel)
    assert_frame_equal(out_first, out_second)

    unpickled = pickle.loads(pickle.dumps(fitted))
    assert_frame_equal(out_first, unpickled.transform(df_panel))

    # Feature names out and input_features checks.
    names_out = fitted.get_feature_names_out()
    assert list(names_out) == out_first.columns
    names_out_in = fitted.get_feature_names_out(df_panel.columns)
    assert list(names_out_in) == out_first.columns
    with pytest.raises(ValueError, match="feature_names_in_"):
        fitted.get_feature_names_out(["wrong_col"])

    # Containers: Polars DF/LazyFrame, Pandas DF, PyArrow Table, Narwhals DF/LazyFrame.
    pd_panel = df_panel.to_pandas()
    pd_panel.index = pd.Index([f"row_{i}" for i in range(len(pd_panel))])
    pa_tbl = df_panel.to_arrow()
    assert isinstance(pa_tbl, pa.Table)
    containers: list[tuple[str, Any]] = [
        ("pl_df", df_panel),
        ("pl_lf", df_panel.lazy()),
        ("pd_df", pd_panel),
        ("pa_tbl", pa_tbl),
        ("nw_df", nw.from_native(df_panel)),
        ("nw_lf", nw.from_native(df_panel.lazy())),
    ]
    for label, container in containers:
        est = clone(estimator)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transformed_ft = est.fit_transform(container)
            est_fit = clone(estimator).fit(container)
            transformed_t = est_fit.transform(container)
        assert not caught, (
            f"Unexpected warnings for {label}: {[str(w.message) for w in caught]}"
        )
        assert est_fit.n_features_in_ == df_panel.width
        assert list(est_fit.feature_names_in_) == df_panel.columns
        assert isinstance(transformed_ft, pl.DataFrame)
        assert isinstance(transformed_t, pl.DataFrame)
        assert_frame_equal(transformed_ft, out_first)
        assert_frame_equal(transformed_t, out_first)

    # set_output and global config_context precedence.
    est_pd = clone(estimator).fit(pd_panel)
    assert est_pd.set_output(transform=None) is est_pd
    assert est_pd.set_output(transform="pandas") is est_pd
    out_pd = est_pd.transform(pd_panel)
    assert isinstance(out_pd, pd.DataFrame)
    assert list(out_pd.index) == list(pd_panel.index)

    pd_multi = df_panel.to_pandas()
    pd_multi.index = pd.MultiIndex.from_arrays(
        [["g"] * len(pd_multi), list(range(len(pd_multi)))], names=["grp", "row"]
    )
    out_multi_pd = (
        clone(estimator).set_output(transform="pandas").fit_transform(pd_multi)
    )
    assert isinstance(out_multi_pd.index, pd.MultiIndex)
    assert out_multi_pd.index.equals(pd_multi.index)

    assert est_pd.set_output(transform="polars") is est_pd
    assert isinstance(est_pd.transform(pd_panel), pl.DataFrame)
    assert est_pd.set_output(transform="default") is est_pd
    assert isinstance(est_pd.transform(pd_panel), pl.DataFrame)

    with pytest.raises(ValueError, match="Unsupported output container"):
        est_pd.set_output(transform="arrow")  # type: ignore[arg-type]

    with config_context(transform_output="pandas"):
        est_ctx = clone(estimator).fit(pd_panel)
        out_ctx = est_ctx.transform(pd_panel)
        assert isinstance(out_ctx, pd.DataFrame)
        assert list(out_ctx.index) == list(pd_panel.index)

        est_override = clone(estimator).set_output(transform="polars").fit(pd_panel)
        assert isinstance(est_override.transform(pd_panel), pl.DataFrame)

    # Shuffled row preservation.
    shuffled = df_panel.sample(fraction=1.0, shuffle=True, seed=42)
    out_shuffled = clone(estimator).fit_transform(shuffled)
    assert out_shuffled.select([*fitted.entity_cols_, fitted.time_col]).equals(
        shuffled.select([*fitted.entity_cols_, fitted.time_col])
    )

    # Univariate and multi-column entity variants.
    if df_univariate is not None:
        est_uni = clone(estimator).set_params(entity_col=None).fit(df_univariate)
        assert est_uni.entity_cols_ == ()
        out_uni = est_uni.transform(df_univariate)
        assert isinstance(out_uni, pl.DataFrame)
        assert out_uni.height == df_univariate.height

    if df_multi_entity is not None:
        est_multi = (
            clone(estimator)
            .set_params(entity_col=["store", "sku"])
            .fit(df_multi_entity)
        )
        assert est_multi.entity_cols_ == ("store", "sku")
        out_multi = est_multi.transform(df_multi_entity)
        assert isinstance(out_multi, pl.DataFrame)
        assert out_multi.height == df_multi_entity.height

    # Invalid inputs.
    for bad_sparse in (
        sparse.csr_matrix(np.eye(3)),
        sparse.csr_array(np.eye(3)),
    ):
        with pytest.raises(TypeError, match="Sparse"):
            clone(estimator).fit(bad_sparse)

    for bad_1d in (np.array([1.0, 2.0, 3.0]), pl.Series("y", [1.0, 2.0])):
        with pytest.raises(ValueError, match="1D"):
            clone(estimator).fit(bad_1d)

    with pytest.raises(TypeError, match="DataFrame"):
        clone(estimator).fit(np.ones((4, 3)))

    with pytest.raises(ValueError, match="0 sample"):
        clone(estimator).fit(df_panel.head(0))

    with pytest.raises(ValueError, match="0 sample"):
        clone(estimator).fit(pl.DataFrame())

    with pytest.raises(ValueError, match="0 feature"):
        clone(estimator).fit(pd.DataFrame(index=range(3)))

    with pytest.raises(ValueError, match="Missing required"):
        clone(estimator).fit(df_panel.drop(fitted.time_col))

    if fitted.entity_cols_:
        with pytest.raises(ValueError, match="Missing required"):
            clone(estimator).fit(df_panel.drop(fitted.entity_cols_[0]))

        null_entity = df_panel.with_columns(
            pl.when(pl.arange(0, df_panel.height) == 0)
            .then(None)
            .otherwise(pl.col(fitted.entity_cols_[0]))
            .alias(fitted.entity_cols_[0])
        )
        with pytest.raises(ValueError, match="null"):
            clone(estimator).fit(null_entity)

    null_time = df_panel.with_columns(
        pl.when(pl.arange(0, df_panel.height) == 0)
        .then(None)
        .otherwise(pl.col(fitted.time_col))
        .alias(fitted.time_col)
    )
    with pytest.raises(ValueError, match="null"):
        clone(estimator).fit(null_time)

    # Feature name / count mismatch on transform.
    with pytest.raises(ValueError):
        fitted.transform(df_panel.with_columns(pl.lit(1.0).alias("extra_col")))
    first_target = fitted.columns_[0]
    with pytest.raises(ValueError):
        fitted.transform(df_panel.rename({first_target: "renamed_col"}))


def _make_sample_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df_panel = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "time": [1, 2, 1, 2],
            "y": [1.0, 2.0, 10.0, 20.0],
        }
    )
    df_uni = pl.DataFrame({"time": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    df_multi = pl.DataFrame(
        {
            "store": ["s1", "s1", "s2", "s2"],
            "sku": ["k1", "k1", "k2", "k2"],
            "time": [1, 2, 1, 2],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return df_panel, df_uni, df_multi


@parametrize_with_checks(
    [
        _DummyTemporalTransformer(time_col="time", entity_col="id"),
        _DummyDataFrameTransformer(),
    ],
    expected_failed_checks=lambda _est: _TRANSFORMER_ONLY_NUMPY_CHECKS,
)
def test_sklearn_estimator_checks(estimator: Any, check: Any) -> None:
    check(estimator)


def test_temporal_transformer_contracts_suite() -> None:
    df_panel, df_uni, df_multi = _make_sample_frames()
    assert_temporal_transformer_contracts(
        _DummyTemporalTransformer(time_col="time", entity_col="id", offset=2.5),
        df_panel,
        df_univariate=df_uni,
        df_multi_entity=df_multi,
    )


def test_single_pass_fit_and_fit_transform_behavior() -> None:
    df_panel, _, _ = _make_sample_frames()

    est = _DummyTemporalTransformer(time_col="time", entity_col="id", offset=3.0)
    est.fit(df_panel)
    assert getattr(est, "_fit_calls", 0) == 1
    assert getattr(est, "_transform_calls", 0) == 0

    est_ft = _DummyTemporalTransformer(time_col="time", entity_col="id", offset=3.0)
    out = est_ft.fit_transform(df_panel)
    assert getattr(est_ft, "_fit_transform_calls", 0) == 1
    assert getattr(est_ft, "_fit_calls", 0) == 1
    assert getattr(est_ft, "_transform_calls", 0) == 0
    assert out.get_column("y").to_list() == [4.0, 5.0, 13.0, 23.0]


def test_temporal_param_validation_and_failed_fit_leaves_unfitted() -> None:
    df_panel, _, _ = _make_sample_frames()

    for bad_time in ("", 123):
        est = _DummyTemporalTransformer(time_col=bad_time)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="time_col"):
            est.fit(df_panel)

    with pytest.raises(ValueError, match="cannot also appear"):
        _DummyTemporalTransformer(time_col="time", entity_col=["id", "time"]).fit(
            df_panel
        )

    est_missing = _DummyTemporalTransformer(time_col="missing_time", entity_col="id")
    with pytest.raises(ValueError, match="Missing required"):
        est_missing.fit(df_panel)
    with pytest.raises(NotFittedError):
        est_missing.transform(df_panel)


def test_chronological_dummy_temporal_transformer_shuffled_panel_and_univariate() -> (
    None
):
    df_panel = pl.DataFrame(
        {
            "id": ["b", "a", "a", "b"],
            "time": [2, 2, 1, 1],
            "y": [20.0, 2.0, 1.0, 10.0],
        }
    )
    est_panel = _DummyChronologicalLagTransformer(time_col="time", entity_col="id")
    out_panel = est_panel.fit_transform(df_panel)
    assert out_panel.select(["id", "time"]).equals(df_panel.select(["id", "time"]))
    assert out_panel.get_column("y_lag_1").to_list() == [10.0, 1.0, None, None]

    df_uni = pl.DataFrame({"time": [3, 1, 2], "y": [30.0, 10.0, 20.0]})
    est_uni = _DummyChronologicalLagTransformer(time_col="time", entity_col=None)
    out_uni = est_uni.fit_transform(df_uni)
    assert out_uni.select(["time"]).equals(df_uni.select(["time"]))
    assert out_uni.get_column("y_lag_1").to_list() == [20.0, None, 10.0]


def test_temporal_regressor_mixin_tags_and_mro() -> None:
    reg = _DummyTemporalRegressor(time_col="ds", entity_col="series_id")
    check_mixin_order("DummyTemporalRegressor", reg)
    tags = reg.__sklearn_tags__()
    assert tags.estimator_type == "regressor"
    assert tags.target_tags.required is True
    assert tags.target_tags.one_d_labels is True
    assert tags.requires_fit is True
    assert tags.input_tags.two_d_array is True
    assert tags.input_tags.categorical is True
    assert tags.input_tags.string is True
    assert tags.input_tags.allow_nan is True
    assert clone(reg).get_params() == {"entity_col": "series_id", "time_col": "ds"}


def test_fit_feature_names_match_fit_transform_and_failed_core_fit_unfits() -> None:
    df_panel, _, _ = _make_sample_frames()

    class _AddsColumn(_DummyTemporalTransformer):
        def _fit_frame(self, frame, y=None):  # type: ignore[no-untyped-def]
            self.offset_ = 1.0

        def _transform_frame(self, frame):  # type: ignore[no-untyped-def]
            return frame.with_lazy(frame.lf.with_columns(pl.lit(1.0).alias("new")))

        _fit_transform_frame = BaseTemporalTransformer._fit_transform_frame

    via_fit = _AddsColumn(time_col="time", entity_col="id").fit(df_panel)
    via_fit_transform = _AddsColumn(time_col="time", entity_col="id")
    out = via_fit_transform.fit_transform(df_panel)
    assert list(via_fit.get_feature_names_out()) == out.columns
    assert list(via_fit_transform.get_feature_names_out()) == out.columns

    class _FailingFit(_DummyTemporalTransformer):
        def _fit_frame(self, frame, y=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    for method in ("fit", "fit_transform"):
        est = _FailingFit(time_col="time", entity_col="id")
        with pytest.raises(RuntimeError, match="boom"):
            getattr(est, method)(df_panel)
        with pytest.raises(NotFittedError):
            est.transform(df_panel)
