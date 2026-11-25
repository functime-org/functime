from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Self

from sklearn.utils.validation import check_is_fitted

from functime._core._narwhals import _validate_temporal_columns
from functime._core.preprocessing import (
    CalendarAttr,
    calendar_features,
    detrend_features,
    diff_features,
    fourier_features,
    frac_diff_features,
    invert_detrend_features,
    invert_diff_features,
    invert_scale_features,
    lag_features,
    rolling_features,
    scale_features,
)
from functime.sklearn._base import _extract_polars_df, _TemporalTransformerMixin


class LagTransformer(_TemporalTransformerMixin):
    """Scikit-learn transformer adding chronological lag features in pure Polars.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    lags : sequence of int, default=(1,)
        Positive lag offsets to generate.
    columns : sequence of str or None, default=None
        Target columns to lag. Defaults to numeric non-index columns seen in ``fit``.
    keep_original : bool, default=True
        Whether to keep existing columns alongside lag features.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        lags: Sequence[int] = (1,),
        columns: Sequence[str] | None = None,
        keep_original: bool = True,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.lags = lags
        self.columns = columns
        self.keep_original = keep_original

    def fit(self, X: Any, y: Any = None) -> Self:
        """Validate temporal schema and compute output feature names."""
        df = _extract_polars_df(self, X, reset=True)
        out, _ = lag_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            lags=self.lags,
            columns=self.columns,
            keep_original=self.keep_original,
        )
        self.feature_names_out_ = tuple(out.columns)
        return self

    def transform(self, X: Any) -> Any:
        """Compute lagged columns per entity while preserving row order."""
        check_is_fitted(self, "feature_names_out_")
        df = _extract_polars_df(self, X, reset=False)
        out, _ = lag_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            lags=self.lags,
            columns=self.columns,
            keep_original=self.keep_original,
        )
        return self._format_df_output(out, X)


class RollingTransformer(_TemporalTransformerMixin):
    """Scikit-learn transformer computing causal rolling window aggregations in Polars.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    window_sizes : sequence of int, default=(3,)
        Window lengths (in rows).
    stats : sequence of {"mean", "std", "min", "max", "median", "sum"}, default=("mean",)
        Rolling aggregations to compute.
    min_samples : int, default=1
        Minimum non-null observations within a window before emitting null.
    shift : int, default=1
        Causal pre-shift applied prior to rolling windows.
    columns : sequence of str or None, default=None
        Columns to aggregate. Defaults to numeric non-index columns.
    keep_original : bool, default=True
        Whether to keep existing columns alongside rolling features.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        window_sizes: Sequence[int] = (3,),
        stats: Sequence[Literal["mean", "std", "min", "max", "median", "sum"]] = (
            "mean",
        ),
        min_samples: int = 1,
        shift: int = 1,
        columns: Sequence[str] | None = None,
        keep_original: bool = True,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.window_sizes = window_sizes
        self.stats = stats
        self.min_samples = min_samples
        self.shift = shift
        self.columns = columns
        self.keep_original = keep_original

    def fit(self, X: Any, y: Any = None) -> Self:
        """Validate temporal schema and compute output feature names."""
        df = _extract_polars_df(self, X, reset=True)
        out, _ = rolling_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            window_sizes=self.window_sizes,
            stats=self.stats,
            min_samples=self.min_samples,
            shift=self.shift,
            columns=self.columns,
            keep_original=self.keep_original,
        )
        self.feature_names_out_ = tuple(out.columns)
        return self

    def transform(self, X: Any) -> Any:
        """Compute rolling aggregations per entity while preserving row order."""
        check_is_fitted(self, "feature_names_out_")
        df = _extract_polars_df(self, X, reset=False)
        out, _ = rolling_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            window_sizes=self.window_sizes,
            stats=self.stats,
            min_samples=self.min_samples,
            shift=self.shift,
            columns=self.columns,
            keep_original=self.keep_original,
        )
        return self._format_df_output(out, X)


class DiffTransformer(_TemporalTransformerMixin):
    """Scikit-learn transformer applying invertible seasonal/ordinary differencing.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    order : int, default=1
        Number of differencing passes.
    sp : int, default=1
        Differencing lag step.
    columns : sequence of str or None, default=None
        Columns to difference. Defaults to numeric non-index columns.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        order: int = 1,
        sp: int = 1,
        columns: Sequence[str] | None = None,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.order = order
        self.sp = sp
        self.columns = columns

    def fit(self, X: Any, y: Any = None) -> Self:
        """Validate schema and record initial anchors required by ``inverse_transform``."""
        df = _extract_polars_df(self, X, reset=True)
        out, state = diff_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            order=self.order,
            sp=self.sp,
            columns=self.columns,
        )
        self.state_ = state
        self.feature_names_out_ = tuple(out.columns)
        return self

    def transform(self, X: Any) -> Any:
        """Apply configured differencing to ``X`` per entity."""
        check_is_fitted(self, "state_")
        df = _extract_polars_df(self, X, reset=False)
        out, _ = diff_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            order=self.order,
            sp=self.sp,
            columns=self.columns,
        )
        return self._format_df_output(out, X)

    def inverse_transform(self, X: Any) -> Any:
        """Invert differencing using the fitted initial anchors."""
        check_is_fitted(self, "state_")
        df = _extract_polars_df(self, X, reset=False)
        out = invert_diff_features(df, state=self.state_)
        return self._format_df_output(out, X)


class PanelStandardScaler(_TemporalTransformerMixin):
    """Standardize numeric columns per panel entity (or globally when univariate).

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    with_mean : bool, default=True
        Subtract per-entity mean.
    with_std : bool, default=True
        Divide by per-entity standard deviation.
    columns : sequence of str or None, default=None
        Columns to scale. Defaults to numeric non-index columns.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        with_mean: bool = True,
        with_std: bool = True,
        columns: Sequence[str] | None = None,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.with_mean = with_mean
        self.with_std = with_std
        self.columns = columns

    def fit(self, X: Any, y: Any = None) -> Self:
        """Compute per-entity mean and standard deviation on ``X``."""
        df = _extract_polars_df(self, X, reset=True)
        out, state = scale_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            with_mean=self.with_mean,
            with_std=self.with_std,
            columns=self.columns,
        )
        self.state_ = state
        self.feature_names_out_ = tuple(out.columns)
        return self

    def transform(self, X: Any) -> Any:
        """Scale ``X`` using statistics learned during ``fit``."""
        check_is_fitted(self, "state_")
        df = _extract_polars_df(self, X, reset=False)
        out, _ = scale_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            with_mean=self.with_mean,
            with_std=self.with_std,
            columns=self.columns,
            state=self.state_,
        )
        return self._format_df_output(out, X)

    def inverse_transform(self, X: Any) -> Any:
        """Invert standardization using statistics learned during ``fit``."""
        check_is_fitted(self, "state_")
        df = _extract_polars_df(self, X, reset=False)
        out = invert_scale_features(df, state=self.state_)
        return self._format_df_output(out, X)


class DetrendTransformer(_TemporalTransformerMixin):
    """Remove per-entity linear or constant trend anchored at fit time ``t0``.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    method : {"linear", "mean"}, default="linear"
        Trend estimator.
    columns : sequence of str or None, default=None
        Columns to detrend. Defaults to numeric non-index columns.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        method: Literal["linear", "mean"] = "linear",
        columns: Sequence[str] | None = None,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.method = method
        self.columns = columns

    def fit(self, X: Any, y: Any = None) -> Self:
        """Fit per-entity trend parameters on ``X``."""
        df = _extract_polars_df(self, X, reset=True)
        out, state = detrend_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            method=self.method,
            columns=self.columns,
        )
        self.state_ = state
        self.feature_names_out_ = tuple(out.columns)
        return self

    def transform(self, X: Any) -> Any:
        """Subtract fitted trend from ``X``."""
        check_is_fitted(self, "state_")
        df = _extract_polars_df(self, X, reset=False)
        out, _ = detrend_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            method=self.method,
            columns=self.columns,
            state=self.state_,
        )
        return self._format_df_output(out, X)

    def inverse_transform(self, X: Any) -> Any:
        """Restore fitted trend to ``X``."""
        check_is_fitted(self, "state_")
        df = _extract_polars_df(self, X, reset=False)
        out = invert_detrend_features(df, state=self.state_)
        return self._format_df_output(out, X)


class CalendarTransformer(_TemporalTransformerMixin):
    """Extract numeric or cyclical sin/cos calendar features from ``time_col``.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the Date or Datetime column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    attrs : sequence of CalendarAttr, default=("month", "weekday", "day")
        Polars datetime attributes to extract.
    cyclical : bool, default=False
        Whether to emit sin/cos encodings for periodic attributes.
    keep_original : bool, default=True
        Whether to retain existing columns alongside calendar features.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        attrs: Sequence[CalendarAttr] = ("month", "weekday", "day"),
        cyclical: bool = False,
        keep_original: bool = True,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.attrs = attrs
        self.cyclical = cyclical
        self.keep_original = keep_original

    def fit(self, X: Any, y: Any = None) -> Self:
        """Validate temporal column type and record output feature names."""
        df = _extract_polars_df(self, X, reset=True)
        out, _ = calendar_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            attrs=self.attrs,
            cyclical=self.cyclical,
            keep_original=self.keep_original,
        )
        self.feature_names_out_ = tuple(out.columns)
        return self

    def transform(self, X: Any) -> Any:
        """Extract configured calendar columns from ``X``."""
        check_is_fitted(self, "feature_names_out_")
        df = _extract_polars_df(self, X, reset=False)
        out, _ = calendar_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            attrs=self.attrs,
            cyclical=self.cyclical,
            keep_original=self.keep_original,
        )
        return self._format_df_output(out, X)


class FourierTransformer(_TemporalTransformerMixin):
    """Extract phase-anchored harmonic Fourier features continuous across train/test.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    period : float, default=12.0
        Seasonal period in step units.
    order : int, default=3
        Number of sine/cosine harmonic pairs.
    freq : str or None, default=None
        Optional cadence hint (e.g., ``"1mo"``) for calendar-month stepping.
    keep_original : bool, default=True
        Whether to retain existing columns alongside Fourier features.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        period: float = 12.0,
        order: int = 3,
        freq: str | None = None,
        keep_original: bool = True,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.period = period
        self.order = order
        self.freq = freq
        self.keep_original = keep_original

    def fit(self, X: Any, y: Any = None) -> Self:
        """Learn phase reference anchor ``t0`` and step cadence on ``X``."""
        df = _extract_polars_df(self, X, reset=True)
        out, state = fourier_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            period=self.period,
            order=self.order,
            freq=self.freq,
            keep_original=self.keep_original,
        )
        self.state_ = state
        self.feature_names_out_ = tuple(out.columns)
        return self

    def transform(self, X: Any) -> Any:
        """Compute phase-continuous Fourier features relative to fit anchor ``t0``."""
        check_is_fitted(self, "state_")
        df = _extract_polars_df(self, X, reset=False)
        out, _ = fourier_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            period=self.period,
            order=self.order,
            freq=self.freq,
            keep_original=self.keep_original,
            state=self.state_,
        )
        return self._format_df_output(out, X)


class FractionalDiffTransformer(_TemporalTransformerMixin):
    """Apply fixed-width window fractional differentiation in pure Polars.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    d : float, default=0.4
        Fractional differencing order.
    min_weight : float, default=1e-4
        Minimum weight magnitude threshold truncating binomial weights.
    window_size : int or None, default=None
        Optional maximum kernel window length.
    columns : sequence of str or None, default=None
        Columns to differentiate. Defaults to numeric non-index columns.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        d: float = 0.4,
        min_weight: float = 1e-4,
        window_size: int | None = None,
        columns: Sequence[str] | None = None,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.d = d
        self.min_weight = min_weight
        self.window_size = window_size
        self.columns = columns

    def fit(self, X: Any, y: Any = None) -> Self:
        """Validate schema and record output feature names."""
        df = _extract_polars_df(self, X, reset=True)
        _validate_temporal_columns(
            df, time_col=self.time_col, entity_col=self.entity_col, columns=self.columns
        )
        out, _ = frac_diff_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            d=self.d,
            min_weight=self.min_weight,
            window_size=self.window_size,
            columns=self.columns,
        )
        self.feature_names_out_ = tuple(out.columns)
        return self

    def transform(self, X: Any) -> Any:
        """Apply fixed-width fractional differentiation to ``X``."""
        check_is_fitted(self, "feature_names_out_")
        df = _extract_polars_df(self, X, reset=False)
        out, _ = frac_diff_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            d=self.d,
            min_weight=self.min_weight,
            window_size=self.window_size,
            columns=self.columns,
        )
        return self._format_df_output(out, X)
