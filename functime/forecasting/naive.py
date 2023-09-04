from typing import Optional, Union

import polars as pl

from functime.base import Forecaster
from functime.ranges import make_future_ranges


class naive(Forecaster):
    """Naive forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    """

    def __init__(self, freq: str):
        super().__init__(freq=freq, lags=1)

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        idx_cols = y.columns[:2]
        entity_col = idx_cols[0]
        target_col = y.columns[2]
        # BUG: Cannot run the following in lazy streaming mode?
        # Causes internal error: entered unreachable code
        y_pred = (
            y.sort(idx_cols)
            .set_sorted(idx_cols)
            .groupby(entity_col)
            .agg(pl.col(target_col).last())
        )
        artifacts = {"y_pred": y_pred}
        return artifacts

    def predict(
        self, fh: int, X: Optional[Union[pl.DataFrame, pl.LazyFrame]] = None
    ) -> pl.DataFrame:
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
        y_pred_vals = y_pred_vals.with_columns(pl.col(target).repeat_by(fh))
        y_pred_vals = (
            y_pred_vals.sort(by=entity)
            .select(pl.col(y_pred_vals.columns[-1]).alias(target).list.tail(fh))
            .collect()
        )
        y_pred = pl.concat(
            [future_ranges.sort(by=entity), y_pred_vals], how="horizontal"
        ).explode(pl.all().exclude(entity))
        return y_pred.pipe(self._reset_string_cache)
