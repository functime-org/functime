from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Self

from sklearn.utils.validation import check_is_fitted

from functime._core.preprocessing import impute_features
from functime.sklearn._base import _extract_polars_df, _TemporalTransformerMixin


class TemporalImputer(_TemporalTransformerMixin):
    """Impute missing values chronologically per entity without cross-entity leakage.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    strategy : {"forward", "backward", "linear", "mean", "median", "constant"}, default="forward"
        Imputation strategy applied within each series.
    fill_value : float or None, default=None
        Constant fill value when ``strategy="constant"``.
    columns : sequence of str or None, default=None
        Subset of columns to impute. Defaults to numeric non-index columns.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        strategy: Literal[
            "forward", "backward", "linear", "mean", "median", "constant"
        ] = "forward",
        fill_value: float | None = None,
        columns: Sequence[str] | None = None,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns

    def fit(self, X: Any, y: Any = None) -> Self:
        """Fit per-entity fill statistics (when needed) and record output columns."""
        df = _extract_polars_df(self, X, reset=True)
        out, state = impute_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            strategy=self.strategy,
            fill_value=self.fill_value,
            columns=self.columns,
        )
        self.state_ = state
        self.feature_names_out_ = tuple(out.columns)
        return self

    def transform(self, X: Any) -> Any:
        """Impute missing values in ``X`` chronologically within each series."""
        check_is_fitted(self, "state_")
        df = _extract_polars_df(self, X, reset=False)
        out, _ = impute_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            strategy=self.strategy,
            fill_value=self.fill_value,
            columns=self.columns,
            state=self.state_,
        )
        return self._format_df_output(out, X)
