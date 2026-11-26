from __future__ import annotations

import pickle
from collections.abc import Sequence
from datetime import date
from typing import Any, ClassVar, Self

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr
from sklearn.utils.estimator_checks import parametrize_with_checks

from functime._frame import HandleUnknown, TemporalFrame, temporal_transform
from functime.sklearn import BaseTemporalTransformer, sklearn_transformer
from tests.sklearn.test_estimator_contracts import (
    _TRANSFORMER_ONLY_NUMPY_CHECKS,
    assert_temporal_transformer_contracts,
)
from tests.sklearn.test_html_repr import _assert_balanced_html


@temporal_transform(chronological=False)
def _log_transform(
    columns: Sequence[str],
    *,
    offset: float = 0.0,
    base: float | None = None,
) -> list[pl.Expr]:
    out: list[pl.Expr] = []
    for col in columns:
        shifted = pl.col(col).abs() + offset + 1.0
        expr = shifted.log() if base is None else shifted.log(base=base)
        out.append(expr.alias(col))
    return out


@sklearn_transformer(_log_transform)
class _LogTransformer(BaseTemporalTransformer):
    """Apply a positive log transform to temporal target columns.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp column.
    entity_col : str or sequence of str or None, default=None
        Column name(s) identifying series entities.
    columns : sequence of str or None, default=None
        Subset of target columns to transform.
    offset : float, default=0.0
        Non-negative offset added before computing logarithms.
    base : float or None, default=None
        Logarithm base; defaults to natural logarithm when ``None``.
    """

    _extra_class_flag: ClassVar[int] = 1
    _ignored_private: int = 0

    offset: float = 0.0
    base: float | None = None


@temporal_transform(chronological=True)
def _lag_feature(columns: Sequence[str], *, lag: int = 1) -> list[pl.Expr]:
    return [pl.col(col).shift(lag).alias(f"{col}_lag_{lag}") for col in columns]


@sklearn_transformer(_lag_feature)
class _LagTransformer(BaseTemporalTransformer):
    """Create chronological lag features per entity or series."""

    lag: int = 1


@temporal_transform(chronological=False, needs_time_col=True, allow_null_time=True)
def _year_feature(_columns: Sequence[str], *, time_col: str) -> list[pl.Expr]:
    return [pl.col(time_col).dt.year().alias("year")]


@sklearn_transformer(_year_feature)
class _YearTransformer(BaseTemporalTransformer):
    """Extract calendar year from the temporal column."""


class _ScalerCore:
    def __init__(
        self,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        handle_unknown: HandleUnknown = "global",
    ) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.handle_unknown = handle_unknown

    def fit(
        self,
        data: TemporalFrame | Any,
        *,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
    ) -> Self:
        frame = TemporalFrame.parse(
            data, time_col=time_col, entity_col=entity_col, columns=columns
        )
        self._time_col = frame.time_col
        self._entity_col = frame.entity_cols or None
        self._target_cols = frame.target_cols

        agg_exprs = [
            expr
            for col in frame.target_cols
            for expr in (
                pl.col(col).mean().alias(f"__mean_{col}"),
                pl.col(col).std(ddof=0).alias(f"__std_{col}"),
            )
        ]
        self._state_df = frame.group_agg(agg_exprs).collect()
        self._global_df = frame.lf.select(agg_exprs).collect()
        return self

    def transform(self, data: TemporalFrame | Any) -> pl.LazyFrame:
        frame = self._parse_transform_frame(data)
        joined = frame.join_state(self._state_df, handle_unknown=self.handle_unknown)
        exprs = [self._scale_expr(col) for col in self._target_cols]
        drop_cols = [
            name
            for col in self._target_cols
            for name in (f"__mean_{col}", f"__std_{col}")
        ]
        return joined.lf.with_columns(exprs).drop(drop_cols)

    def inverse_transform(self, data: TemporalFrame | Any) -> pl.LazyFrame:
        frame = self._parse_transform_frame(data)
        joined = frame.join_state(self._state_df, handle_unknown=self.handle_unknown)
        exprs = [self._unscale_expr(col) for col in self._target_cols]
        drop_cols = [
            name
            for col in self._target_cols
            for name in (f"__mean_{col}", f"__std_{col}")
        ]
        return joined.lf.with_columns(exprs).drop(drop_cols)

    def _parse_transform_frame(self, data: TemporalFrame | Any) -> TemporalFrame:
        return TemporalFrame.parse(
            data,
            time_col=self._time_col,
            entity_col=self._entity_col,
            columns=self._target_cols,
        )

    def _stat_expr(self, prefix: str, col: str, default: float) -> pl.Expr:
        stat = pl.col(f"{prefix}_{col}")
        if self.handle_unknown == "global":
            fallback = float(self._global_df.get_column(f"{prefix}_{col}").item(0))
            return stat.fill_null(fallback)
        if self.handle_unknown == "identity":
            return stat.fill_null(default)
        return stat

    def _scale_expr(self, col: str) -> pl.Expr:
        expr = pl.col(col)
        if self.with_mean:
            expr = expr - self._stat_expr("__mean", col, 0.0)
        if self.with_std:
            std = self._stat_expr("__std", col, 1.0)
            safe_std = pl.when(std == 0.0).then(1.0).otherwise(std)
            expr = expr / safe_std
        return expr.alias(col)

    def _unscale_expr(self, col: str) -> pl.Expr:
        expr = pl.col(col)
        if self.with_std:
            std = self._stat_expr("__std", col, 1.0)
            safe_std = pl.when(std == 0.0).then(1.0).otherwise(std)
            expr = expr * safe_std
        if self.with_mean:
            expr = expr + self._stat_expr("__mean", col, 0.0)
        return expr.alias(col)


@sklearn_transformer(_ScalerCore)
class _StandardScaler(BaseTemporalTransformer):
    """Standardize target columns per entity or globally."""

    with_mean: bool = True
    with_std: bool = True
    handle_unknown: HandleUnknown = "global"


class _MeanCenterCore:
    def fit(self, data: TemporalFrame, *, columns: Sequence[str]) -> Self:
        self._means = {
            col: float(data.lf.select(pl.col(col).mean()).collect().item())
            for col in columns
        }
        return self

    def transform(self, data: TemporalFrame) -> pl.LazyFrame:
        return data.lf.with_columns(
            [(pl.col(col) - mean).alias(col) for col, mean in self._means.items()]
        )


@sklearn_transformer(_MeanCenterCore)
class _MeanCenterTransformer(BaseTemporalTransformer):
    """Subtract training column means without inverse support."""


class _SpyCore:
    fit_calls: ClassVar[int] = 0
    transform_calls: ClassVar[int] = 0

    @classmethod
    def reset_counts(cls) -> None:
        cls.fit_calls = 0
        cls.transform_calls = 0

    def __init__(self, shift: float = 1.0) -> None:
        self.shift = shift

    def fit(self, data: TemporalFrame) -> Self:
        _SpyCore.fit_calls += 1
        self._cols = data.target_cols
        return self

    def transform(self, data: TemporalFrame) -> pl.LazyFrame:
        _SpyCore.transform_calls += 1
        return data.lf.with_columns(
            [(pl.col(col) + self.shift).alias(col) for col in self._cols]
        )


@sklearn_transformer(_SpyCore)
class _SpyTransformer(BaseTemporalTransformer):
    """Spy transformer verifying single-pass fit_transform execution."""

    shift: float = 1.0


def _sample_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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
        _LogTransformer(time_col="time", entity_col="id"),
        _LagTransformer(time_col="time", entity_col="id"),
        _StandardScaler(time_col="time", entity_col="id"),
        _MeanCenterTransformer(time_col="time", entity_col="id"),
    ],
    expected_failed_checks=lambda _est: _TRANSFORMER_ONLY_NUMPY_CHECKS,
)
def test_decorated_sklearn_estimator_checks(estimator: Any, check: Any) -> None:
    check(estimator)


@pytest.mark.parametrize(
    "estimator",
    [
        _LogTransformer(time_col="time", entity_col="id", offset=1.0),
        _LagTransformer(time_col="time", entity_col="id", lag=1),
        _StandardScaler(time_col="time", entity_col="id"),
        _MeanCenterTransformer(time_col="time", entity_col="id"),
    ],
)
def test_decorated_transformer_contracts_suite(
    estimator: BaseTemporalTransformer,
) -> None:
    df_panel, df_uni, df_multi = _sample_frames()
    assert_temporal_transformer_contracts(
        estimator,
        df_panel,
        df_univariate=df_uni,
        df_multi_entity=df_multi,
    )


def test_stateless_columns_selection_chronological_and_lazy_fit() -> None:
    df = pl.DataFrame(
        {
            "id": ["b", "a", "a", "b"],
            "time": [2, 2, 1, 1],
            "y": [20.0, 2.0, 1.0, 10.0],
            "z": [200.0, 20.0, 10.0, 100.0],
        }
    )
    log_t = _LogTransformer(time_col="time", entity_col="id", columns=["y"])
    assert not hasattr(log_t, "inverse_transform")
    assert not hasattr(log_t, "entity_cols_")
    assert not hasattr(log_t, "columns_")

    out_log = log_t.fit_transform(df)
    assert out_log.get_column("z").to_list() == [200.0, 20.0, 10.0, 100.0]
    assert list(log_t.get_feature_names_out()) == ["id", "time", "y", "z"]

    lag_t = _LagTransformer(time_col="time", entity_col="id", columns=["y"], lag=1)
    out_lag = lag_t.fit_transform(df)
    assert list(lag_t.get_feature_names_out()) == ["id", "time", "y", "z", "y_lag_1"]
    assert out_lag.get_column("y_lag_1").to_list() == [10.0, 1.0, None, None]

    # Stateless _fit_frame stays lazy via schema inference without executing .collect().
    deferred_lf = df.lazy().with_columns(
        pl.col("y").cast(pl.String).str.to_integer().alias("y")
    )
    tf_lazy = TemporalFrame.parse(
        deferred_lf, time_col="time", entity_col="id", columns=["y"]
    )
    lazy_lag = _LagTransformer(time_col="time", entity_col="id", columns=["y"], lag=1)
    lazy_lag._fit_frame(tf_lazy)
    assert lazy_lag.feature_names_out_ == ("id", "time", "y", "z", "y_lag_1")

    # allow_null_time=True and needs_time_col=True propagate from @temporal_transform.
    df_dates = pl.DataFrame(
        {
            "time": [date(2024, 1, 1), None],
            "y": [1.0, 2.0],
        }
    )
    year_t = _YearTransformer(time_col="time")
    out_year = year_t.fit_transform(df_dates)
    assert year_t.get_feature_names_out().tolist() == ["time", "y", "year"]
    assert out_year.get_column("year").to_list() == [2024, None]

    # Invalid `columns` parameter validation.
    with pytest.raises(TypeError, match="sequence"):
        _LogTransformer(time_col="time", columns="y").fit(df)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        _LogTransformer(time_col="time", columns=[]).fit(df)
    with pytest.raises(TypeError, match="non-empty strings"):
        _LogTransformer(time_col="time", columns=["y", ""]).fit(df)
    with pytest.raises(ValueError, match="Duplicate"):
        _LogTransformer(time_col="time", columns=["y", "y"]).fit(df)


def test_pep681_construction_pipeline_column_transformer_and_html_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    est_pos = _LogTransformer("time", "id", ("y",), 2.0, 10.0)
    assert est_pos.get_params() == {
        "time_col": "time",
        "entity_col": "id",
        "columns": ("y",),
        "offset": 2.0,
        "base": 10.0,
    }

    df_panel, _, _ = _sample_frames()
    pipe = Pipeline(
        [
            ("log", _LogTransformer(time_col="time", entity_col="id", offset=1.0)),
            ("lag", _LagTransformer(time_col="time", entity_col="id", lag=1)),
            ("scale", _StandardScaler(time_col="time", entity_col="id")),
        ]
    )
    out = pipe.fit_transform(df_panel)
    assert out.columns == ["id", "time", "y", "y_lag_1"]
    assert pipe.get_feature_names_out().tolist() == ["id", "time", "y", "y_lag_1"]

    ct = ColumnTransformer(
        [
            (
                "temporal",
                _LogTransformer(time_col="time", entity_col="id"),
                ["id", "time", "y"],
            )
        ]
    )
    assert ct.fit_transform(df_panel).shape == (4, 3)

    monkeypatch.setattr(
        _LogTransformer, "__module__", "functime.sklearn.preprocessing._dummy"
    )
    unfitted_html = estimator_html_repr(est_pos)
    assert "functime-org.github.io" in unfitted_html
    assert "Non-negative offset added before computing logarithms." in unfitted_html
    _assert_balanced_html(unfitted_html)
    _assert_balanced_html(estimator_html_repr(pipe.fit(df_panel)))


def test_stateful_inverse_transform_single_pass_and_handle_unknown() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "time": [1, 2, 1, 2],
            "y": [10.0, 20.0, 100.0, 200.0],
        }
    )

    scaler = _StandardScaler(time_col="time", entity_col="id")
    center = _MeanCenterTransformer(time_col="time", entity_col="id")
    assert hasattr(scaler, "inverse_transform")
    assert not hasattr(center, "inverse_transform")

    with pytest.raises(NotFittedError):
        scaler.inverse_transform(df)

    scaled = scaler.fit_transform(df)
    reconstructed = scaler.inverse_transform(scaled)
    assert isinstance(reconstructed, pl.DataFrame)
    assert_frame_equal(reconstructed, df)

    # Pickle and clone preserve stateful fit and inverse_transform.
    restored = pickle.loads(pickle.dumps(scaler))
    assert_frame_equal(restored.inverse_transform(scaled), df)
    assert clone(scaler).get_params() == scaler.get_params()

    # Pandas MultiIndex round-trip on inverse_transform + Pipeline inverse_transform.
    pd_df = df.to_pandas()
    pd_df.index = pd.MultiIndex.from_arrays(
        [["g", "g", "h", "h"], [0, 1, 0, 1]], names=["grp", "row"]
    )
    pipe = Pipeline(
        [("scale", _StandardScaler(time_col="time", entity_col="id"))]
    ).set_output(transform="pandas")
    assert hasattr(pipe, "inverse_transform")
    pipe_scaled = pipe.fit_transform(pd_df)
    assert isinstance(pipe_scaled, pd.DataFrame)
    pipe_inv = pipe.inverse_transform(pipe_scaled)
    assert isinstance(pipe_inv, pd.DataFrame)
    assert pipe_inv.index.equals(pd_df.index)
    assert pipe_inv["y"].tolist() == pytest.approx([10.0, 20.0, 100.0, 200.0])

    # handle_unknown policies on unseen entity "c".
    df_unseen = pl.DataFrame({"id": ["c"], "time": [1], "y": [165.0]})
    out_global = (
        _StandardScaler(entity_col="id", handle_unknown="global")
        .fit(df)
        .transform(df_unseen)
    )
    global_mean = (10.0 + 20.0 + 100.0 + 200.0) / 4.0
    global_std = pl.Series([10.0, 20.0, 100.0, 200.0]).std(ddof=0)
    assert global_std is not None
    assert out_global.get_column("y").item(0) == pytest.approx(
        (165.0 - global_mean) / float(global_std)
    )

    out_identity = (
        _StandardScaler(entity_col="id", handle_unknown="identity")
        .fit(df)
        .transform(df_unseen)
    )
    assert out_identity.get_column("y").item(0) == pytest.approx(165.0)

    out_null = (
        _StandardScaler(entity_col="id", handle_unknown="null")
        .fit(df)
        .transform(df_unseen)
    )
    assert out_null.get_column("y").item(0) is None

    err_scaler = _StandardScaler(entity_col="id", handle_unknown="error").fit(df)
    with pytest.raises(ValueError, match="unseen entity"):
        err_scaler.transform(df_unseen)

    # Single-pass fit_transform verification.
    _SpyCore.reset_counts()
    spy = _SpyTransformer(time_col="time", entity_col="id", shift=2.0)
    out_spy = spy.fit_transform(df)
    assert _SpyCore.fit_calls == 1
    assert _SpyCore.transform_calls == 1
    assert out_spy.get_column("y").to_list() == [12.0, 22.0, 102.0, 202.0]


def test_decorator_validation_errors() -> None:
    with pytest.raises(TypeError, match="BaseTemporalTransformer"):

        @sklearn_transformer(_log_transform)  # type: ignore[arg-type]
        class _NotInherited:
            offset: float = 0.0

    with pytest.raises(TypeError, match="BaseTemporalTransformer"):
        sklearn_transformer(_log_transform)(BaseTemporalTransformer)

    with pytest.raises(TypeError, match="define __init__"):

        @sklearn_transformer(_log_transform)
        class _ManualInit(BaseTemporalTransformer):
            offset: float = 0.0

            def __init__(self) -> None:
                super().__init__()

    with pytest.raises(TypeError, match="@temporal_transform"):

        def _plain_fn(columns: Sequence[str]) -> list[pl.Expr]:
            return [pl.col(c) for c in columns]

        @sklearn_transformer(_plain_fn)
        class _BadTargetFn(BaseTemporalTransformer):
            pass

    with pytest.raises(TypeError, match="fit"):

        class _NoFit:
            def transform(self, data: Any) -> pl.LazyFrame:
                return data

        @sklearn_transformer(_NoFit)
        class _BadTargetCls(BaseTemporalTransformer):
            pass

    with pytest.raises(TypeError, match="not accepted"):

        @sklearn_transformer(_log_transform)
        class _UnknownParam(BaseTemporalTransformer):
            unknown_arg: float = 1.0

    @temporal_transform(chronological=True)
    def _req_op(columns: Sequence[str], *, horizon: int) -> list[pl.Expr]:
        return [pl.col(c).shift(-horizon) for c in columns]

    with pytest.raises(TypeError, match="missing required"):

        @sklearn_transformer(_req_op)
        class _MissingRequiredExpr(BaseTemporalTransformer):
            pass

    class _ReqCore:
        def __init__(self, *, window: int) -> None:
            self.window = window

        def fit(self, data: TemporalFrame) -> Self:
            return self

        def transform(self, data: TemporalFrame) -> pl.LazyFrame:
            return data.lf

    with pytest.raises(TypeError, match="missing required"):

        @sklearn_transformer(_ReqCore)
        class _MissingRequiredCore(BaseTemporalTransformer):
            pass

    with pytest.raises(TypeError, match="not accepted"):

        class _BareCore:
            def fit(self, data: TemporalFrame) -> Self:
                return self

            def transform(self, data: TemporalFrame) -> pl.LazyFrame:
                return data.lf

        @sklearn_transformer(_BareCore)
        class _ExtraForBareCore(BaseTemporalTransformer):
            window: int = 3

    with pytest.raises(TypeError, match="default value"):

        @sklearn_transformer(_log_transform)
        class _NoDefault(BaseTemporalTransformer):
            offset: float

    with pytest.raises(TypeError, match="mutable default"):

        @sklearn_transformer(_log_transform)
        class _MutableDefault(BaseTemporalTransformer):
            offset: Any = [0.0]

    # Static and fitted annotations do not leak into estimator parameters.
    assert set(_LogTransformer().get_params()) == {
        "time_col",
        "entity_col",
        "columns",
        "offset",
        "base",
    }
