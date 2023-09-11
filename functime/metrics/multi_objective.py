"""Module with functions to compute, compare, and optimize multi-objective forecasts.
"""

from dataclasses import dataclass
from functools import partial, reduce
from typing import Optional

import polars as pl
from typing_extensions import Literal

from functime.metrics import (
    mae,
    mase,
    mse,
    overforecast,
    rmse,
    rmsse,
    smape,
    underforecast,
)


@dataclass(frozen=True)
class Metrics:
    mae: float
    mse: float
    smape: float
    rmse: float
    rmsse: float
    mase: float
    overforecast: float
    underforecast: float


def summarize_scores(
    scores: pl.DataFrame, agg_method: Literal["mean", "median"] = "mean"
) -> Metrics:
    """Given a DataFrame of forecast metrics, return a dataclass of metrics aggregated by `agg_method`.

    Parameters
    ----------
    scores : pl.DataFrame
        DataFrame of scores. N rows of entities by M columns of metrics.
    agg_method : str
        Method ("mean", "median") to aggregate scores across entities by.

    Returns
    -------
    metrics : Metrics
        Dataclass of scores aggregated across entities.
    """
    entity_col = scores.columns[0]
    expr = {
        "mean": pl.all().exclude(entity_col).mean(),
        "median": pl.all().exclude(entity_col).median(),
    }
    metrics = scores.select(expr[agg_method]).to_dicts()[0]
    return Metrics(**metrics)


def score_forecast(
    y_true: pl.DataFrame, y_pred: pl.DataFrame, y_train: pl.DataFrame
) -> pl.DataFrame:
    """Return DataFrame of forecast metrics across entities.

    Metrics returned:\n
    - MAE
    - MASE
    - MSE
    - Overforecast
    - RMSE
    - RMSSE
    - SMAPE
    - Underforecast

    Note: SMAPE is used instead of MAPE to avoid potential divide by zero errors.

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.
    y_train : pl.DataFrame
        Observed training values.

    Returns
    -------
    scores : pl.DataFrame
        DataFrame with computed metrics column by column across entities row by row.
    """
    entity_col = y_true.columns[0]
    metrics = [
        mae,
        partial(mase, y_train=y_train),
        mse,
        overforecast,
        rmse,
        partial(rmsse, y_train=y_train),
        smape,
        underforecast,
    ]
    scores = [metric(y_true, y_pred) for metric in metrics]
    scores = reduce(
        lambda df1, df2: df1.lazy().join(df2.lazy(), how="left", on=entity_col), scores
    ).collect()
    return scores


def score_backtest(
    y_true: pl.DataFrame,
    y_preds: pl.DataFrame,
    agg_method: Optional[Literal["mean", "median", "first", "last"]] = None,
) -> pl.DataFrame:
    """Return DataFrame of forecast metrics across entities.

    Metrics returned:\n
    - MAE
    - MASE
    - MSE
    - Overforecast
    - RMSE
    - RMSSE
    - SMAPE
    - Underforecast

    Note: MAPE is excluded to avoid potential divide by zero errors.
    We recommend looking at SMAPE instead.

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_preds : pl.DataFrame
        Stacked predicted values across CV splits.
        DataFrame contains four columns: entity, time, target, "split".
    agg_method : Optional[str] = None
        Method ("mean", "median") to aggregate scores across entities by.
        If None, forecasts in overlapping splits are weighted equally, i.e.
        no aggregation is applied.

    Returns
    -------
    scores : pl.DataFrame
        DataFrame with computed metrics column by column across entities row by row.
    """
    entity_col, time_col, target_col = y_preds.columns[:3]
    expr = {
        "mean": pl.col(target_col).mean(),
        "median": pl.col(target_col).median(),
        "first": pl.col(target_col).first(),
        "last": pl.col(target_col).last(),
    }
    if agg_method:
        y_pred = (
            y_preds.lazy()
            .group_by([entity_col, time_col])
            .agg(expr[agg_method])
            .sort([entity_col, time_col])
            .set_sorted([entity_col, time_col])
            .collect()
        )
        scores = score_forecast(y_true, y_pred, y_train=y_true)
    else:
        scores = score_forecast(y_true, y_preds, y_train=y_true)
    return scores
