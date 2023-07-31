from typing import Optional

import polars as pl

from functime.base import Forecaster
from functime.ranges import make_future_ranges


class naive(Forecaster):
    """Naive forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars
    max_fh : int
        Max forecast horizon. `fh` in predict cannot exceed this value.
    """

    def __init__(self, freq: str, max_fh: int):
        self.max_fh = max_fh
        super().__init__(freq=freq, lags=1)

    def _fit(self, y: pl.DataFrame, X: Optional[pl.DataFrame] = None):
        max_fh = self.max_fh
        idx_cols = y.columns[:2]
        entity_col = idx_cols[0]
        target_col = y.columns[2]
        y_past = (
            y
            # Sort by entity and time
            .sort(idx_cols)
            # Group by entity then takes the last row
            .groupby(entity_col).tail(1)
        )

        y_pred = (
            pl.concat([y_past] * max_fh).groupby(entity_col).agg(pl.col(target_col))
        )
        artifacts = {"y_pred": y_pred}
        return artifacts

    def predict(self, fh: int, X: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        state = self.state
        entity = state.entity
        target = state.target
        # Cutoffs cannot be lazy
        cutoffs: pl.DataFrame = state.artifacts["__cutoffs"]
        future_ranges = make_future_ranges(
            time_col=state.time,
            cutoffs=cutoffs,
            fh=fh,
            freq=self.freq,
        )
        y_pred_vals = state.artifacts["y_pred"]
        y_pred_vals = (
            y_pred_vals.sort(by=entity)
            .select(pl.col(y_pred_vals.columns[-1]).alias(target).list.head(fh))
            .collect(streaming=True)
        )
        y_pred = pl.concat(
            [future_ranges.sort(by=entity), y_pred_vals], how="horizontal"
        ).explode(pl.all().exclude(entity))
        return y_pred.pipe(self._reset_string_cache)
