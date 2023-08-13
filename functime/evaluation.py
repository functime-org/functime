from functools import partial
from typing import Optional

import numpy as np
import polars as pl
from scipy.stats import norm, normaltest
from typing_extensions import Literal

from functime.base.metric import METRIC_TYPE
from functime.metrics import (
    mae,
    mape,
    mase,
    mse,
    overforecast,
    rmse,
    rmsse,
    smape,
    underforecast,
)


def SORT_BY_TO_METRIC(y_train):
    return {
        "mae": mae,
        "mape": mape,
        "mase": partial(mase, y_train=y_train),
        "mse": mse,
        "overforecast": overforecast,
        "rmse": rmse,
        "rmsse": partial(rmsse, y_train=y_train),
        "smape": smape,
        "underforecast": underforecast,
    }


FORECAST_SORT_BY = Literal[
    # Summary statistics
    "mean",
    "median",
    "std",
    "cv",
    # Metrics
    "mae",
    "mape",
    "mase",
    "mse",
    "rmse",
    "rmsse",
    "smape",
    "smape_original",
    "overforecast",
    "underforecast",
]

RESIDUALS_SORT_BY = Literal["bias", "abs_bias", "normality", "autocor_lb", "autocor_bg"]

FVA_SORT_BY = Literal["naive", "snaive", "linear", "linear_scaled"]


def acf(x: pl.Series, max_lags: int, alpha: float = 0.05):
    n = x.len()
    # Brute force ACF calculation (might be slow for long series and lags)
    acf = [pl.corr(x, x.shift(i), ddof=i) for i in range(1, max_lags + 1)]
    # Calculate variance using Bartlett's formula
    varacf = pl.repeat(1, max_lags + 1, eager=True) / n
    cumsum_var = pl.cumsum_horizontal(varacf.slice(1, n - 1) ** 2)
    varacf = [varacf.get(i) * (1 + 2 * cumsum_var) for i in range(2, max_lags)]
    pff = norm.pff(1 - alpha / 2.0)
    intervals = [pff * np.sqrt(var) for var in varacf]
    return acf, intervals


def ljung_box_test(x: pl.Series, max_lags: int):
    # Brute force ACF calculation (might be slow for long series and lags)
    acf = [pl.corr(x, x.shift(i), ddof=i) for i in range(1, max_lags + 1)]
    n = x.len()
    qstat = n * (n + 2) * np.sum((acf[1:] ** 2) / (n - np.arange(1, max_lags + 1)))
    return qstat


def normality_test(x: pl.Series):
    return normaltest(x.to_numpy())[0]


def _rank_entities_by_stat(y_true: pl.DataFrame, sort_by: str, descending: bool):
    entity_col, _, target_col = y_true.columns[:3]
    ranked_entities = (
        y_true.groupby(entity_col)
        .agg(getattr(pl, sort_by)(target_col))
        .sort(by=sort_by, descending=descending)
        .select([entity_col, sort_by])
    )
    return ranked_entities


def _rank_entities_by_score(
    y_true: pl.DataFrame, y_pred: pl.DataFrame, sort_by: str, descending: bool
):
    scoring = SORT_BY_TO_METRIC(y_true)(sort_by)
    ranked_entities = (
        scoring(y_true=y_true, y_pred=y_pred)
        .sort(by=sort_by, descending=descending)
        .select([y_true.columns[0], sort_by])
    )
    return ranked_entities


def _rank_entities(
    y_pred: pl.DataFrame, y_true: pl.DataFrame, sort_by: str, descending: bool
):
    if sort_by in ["mean", "median", "std", "cv"]:
        ranks = _rank_entities_by_stat(
            y_true=y_true, sort_by=sort_by, descending=descending
        )
    else:
        ranks = _rank_entities_by_score(
            y_true=y_true, y_pred=y_pred, sort_by=sort_by, descending=descending
        )
    return ranks


def rank_forecasts(
    y_pred: pl.DataFrame,
    y_true: pl.DataFrame,
    descending: bool = False,
    sort_by: FORECAST_SORT_BY = "smape",
) -> pl.DataFrame:
    ranks = _rank_entities(
        y=y_true, y_pred=y_pred, sort_by=sort_by, descending=descending
    )
    return ranks


def rank_residuals(
    y_resids: pl.DataFrame,
    sort_by: RESIDUALS_SORT_BY,
    max_lags: int = 12,
    alpha: float = 0.05,
) -> pl.DataFrame:
    entity_col, _, target_col = y_resids.columns[:3]
    sort_by_to_expr = {
        "bias": pl.col(target_col).mean().abs(),
        "abs_bias": pl.col(target_col).mean().abs(),
        "normality": pl.col(target_col).apply(normality_test),
        "autocorr_lb": (
            pl.col(target_col)
            .apply(ljung_box_test, max_lags=max_lags, alpha=alpha)
            .struct["qstat"]
        ),
    }
    ranks = (
        y_resids.groupby(entity_col)
        .agg(sort_by_to_expr[sort_by].alias(sort_by))
        .sort(sort_by)
    )
    return ranks


def rank_fva(
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
    y_pred_bench: Optional[pl.DataFrame] = None,
    scoring: Optional[METRIC_TYPE] = None,
    descending: bool = False,
) -> pl.DataFrame:
    scoring = scoring or smape
    scores = scoring(y_true=y_true, y_pred=y_pred)
    if y_pred_bench is None:
        y_pred_bench = {}
    scores_bench = scoring(y_true=y_true, y_pred=y_pred_bench)
    entity_col, metric_name = scores_bench.columns
    uplift = (
        scores.join(
            scores_bench.rename({metric_name: f"{metric_name}_bench"}),
            how="left",
            on=scores.columns[0],
        )
        .with_columns(
            uplift=pl.col(f"{metric_name}_bench") - pl.col(metric_name),
            has_uplift=pl.col(f"{metric_name}_bench") - pl.col(metric_name) > 0,
        )
        .select([entity_col, "uplift", "has_uplift"])
        .sort("uplift", descending=descending)
    )
    return uplift
