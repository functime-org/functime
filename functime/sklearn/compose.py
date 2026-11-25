from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Self

from sklearn.utils.validation import check_is_fitted

from functime._core.preprocessing import encode_panel_features
from functime.sklearn._base import _extract_polars_df, _TemporalTransformerMixin


class PanelFeatureAdapter(_TemporalTransformerMixin):
    """Bridge panel DataFrames directly into tabular scikit-learn and GBDT estimators.

    Drops raw temporal index columns (by default), encodes ``entity_col`` using
    consistent training-derived categories (``"enum"``, ``"ordinal"``, or ``"drop"``),
    and casts numeric columns to model-friendly ``float32`` values.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    drop_time : bool, default=True
        Whether to drop ``time_col`` from the output feature matrix.
    entity_encoding : {"enum", "ordinal", "drop"}, default="enum"
        Encoding strategy for ``entity_col`` levels.
    cast_float32 : bool, default=True
        Cast numeric non-entity features to ``pl.Float32`` and booleans to
        ``pl.Int8``.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        drop_time: bool = True,
        entity_encoding: Literal["enum", "ordinal", "drop"] = "enum",
        cast_float32: bool = True,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.drop_time = drop_time
        self.entity_encoding = entity_encoding
        self.cast_float32 = cast_float32

    def fit(self, X: Any, y: Any = None) -> Self:
        """Learn stable entity category levels and output schema on ``X``."""
        df = _extract_polars_df(self, X, reset=True)
        out, state = encode_panel_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            drop_time=self.drop_time,
            entity_encoding=self.entity_encoding,
            cast_float32=self.cast_float32,
        )
        self.state_ = state
        self.feature_names_out_ = tuple(out.columns)
        return self

    def transform(self, X: Any) -> Any:
        """Transform ``X`` into a model-ready DataFrame."""
        check_is_fitted(self, "state_")
        df = _extract_polars_df(self, X, reset=False)
        out, _ = encode_panel_features(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            drop_time=self.drop_time,
            entity_encoding=self.entity_encoding,
            cast_float32=self.cast_float32,
            state=self.state_,
        )
        return self._format_df_output(out, X)
