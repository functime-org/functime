from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from sklearn.utils.validation import check_is_fitted

from functime._core.naive import naive_forecast, seasonal_naive_forecast
from functime.sklearn._base import _extract_polars_df, _TemporalRegressorMixin


class NaiveForecaster(_TemporalRegressorMixin):
    """Persistence baseline predicting each entity's final observed value.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col

    def fit(self, X: Any, y: Any) -> Self:
        """Fit per-entity last observed values from ``(X, y)``."""
        df = _extract_polars_df(self, X, reset=True)
        _, state = naive_forecast(
            df,
            y,
            time_col=self.time_col,
            entity_col=self.entity_col,
        )
        self.state_ = state
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict persisted last values aligned to ``X``."""
        check_is_fitted(self, "state_")
        df = _extract_polars_df(self, X, reset=False)
        preds, _ = naive_forecast(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            state=self.state_,
        )
        return preds


class SeasonalNaiveForecaster(_TemporalRegressorMixin):
    """Seasonal-naive baseline cycling each entity's trailing ``sp`` values.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    sp : int, default=1
        Seasonal period length.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        sp: int = 1,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.sp = sp

    def fit(self, X: Any, y: Any) -> Self:
        """Fit per-entity trailing seasonal values of length ``sp`` from ``(X, y)``."""
        df = _extract_polars_df(self, X, reset=True)
        _, state = seasonal_naive_forecast(
            df,
            y,
            time_col=self.time_col,
            entity_col=self.entity_col,
            sp=self.sp,
        )
        self.state_ = state
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict cyclical seasonal-naive values aligned to ``X``."""
        check_is_fitted(self, "state_")
        df = _extract_polars_df(self, X, reset=False)
        preds, _ = seasonal_naive_forecast(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            sp=self.sp,
            state=self.state_,
        )
        return preds
