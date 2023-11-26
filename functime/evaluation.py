from functools import partial
from typing import List, Optional

import polars as pl
import polars.selectors as cs
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

RESIDUALS_SORT_BY = Literal["bias", "abs_bias", "normality", "autocorr"]

FVA_SORT_BY = Literal["naive", "snaive", "linear", "linear_scaled"]


def acf_formula(x: pl.Expr, max_lags: int) -> List[pl.Expr]:
    # NOTE: Unsure if lists of expressions are automatically vectorized by the Rust query engine...
    # Brute force adjusted ACF calculation (might be slow for long series and lags)
    n = x.len()
    # Otherwise, the following computation will be faster as eager arrays / series
    acf = [
        pl.corr(x, x.shift(i), ddof=i).alias(f"acorr_{i}")
        for i in range(1, max_lags + 1)
    ]
    return [*acf, n.alias("length")]


def acf_confint_formula(acf: pl.Expr, length: pl.Expr, ppf: float) -> pl.Expr:
    # Calculate variance using Bartlett's formula
    var_acf = (1 + 2 * (acf**2).cum_sum()).alias("var")
    intervals = ppf * ((1.0 / length) * var_acf).sqrt().cast(pl.Float32)
    return intervals.alias("interval")


def acf(X: pl.DataFrame, max_lags: int, alpha: float = 0.05) -> pl.DataFrame:
    entity_col, _, target_col = X.columns
    ppf = norm.ppf(1 - alpha / 2.0)
    result = (
        X.lazy()
        # Defensive downcast and demean
        .with_columns(
            (pl.col(target_col) - pl.col(target_col).mean())
            .over(entity_col)
            .cast(pl.Float32)
        )
        .group_by(entity_col)
        .agg(acf_formula(pl.col(target_col), max_lags=max_lags))
        .select(
            entity_col,
            pl.concat_list(cs.starts_with("acorr_")).alias("acf"),
            pl.col("length"),
        )
        .explode("acf")
        .group_by(entity_col)
        .agg(
            [
                pl.col("acf"),
                ppf
                * (1.0 / pl.col("length").cast(pl.Float32).first())
                .sqrt()
                .alias("interval_1"),
                acf_confint_formula(
                    acf=pl.col("acf"), length=pl.col("length"), ppf=ppf
                ).slice(1, max_lags - 1),
            ]
        )
        .select(
            [
                entity_col,
                "acf",
                pl.concat_list(cs.starts_with("interval")).alias("interval"),
            ]
        )
        .explode(["acf", "interval"])
        .group_by(entity_col)
        .agg(
            [
                pl.col("acf"),
                (pl.col("acf") - pl.col("interval")).alias("confint_lower"),
                (pl.col("acf") + pl.col("interval")).alias("confint_upper"),
            ]
        )
        .with_columns(
            pl.lit([1.0]).list.concat("acf").alias("acf"),
            pl.lit([1.0]).list.concat("confint_lower").alias("confint_lower"),
            pl.lit([1.0]).list.concat("confint_upper").alias("confint_upper"),
        )
        .collect(streaming=True)
    )
    return result


def ljung_box_test(X: pl.DataFrame, max_lags: int):
    def _acf_sqr_ratio(x: pl.Expr):
        n = x.len()
        x = x - x.mean()
        acf = [
            pl.corr(x, x.shift(i), ddof=i).alias(f"acorr_{i}")
            for i in range(1, max_lags + 1)
        ]
        acf_sqr = [x**2 for x in acf]
        acf_sqr_ratio = [x / (n - k) for x, k in zip(acf_sqr, range(1, max_lags + 1))]
        return [*acf_sqr_ratio, n.alias("length")]

    def _qstat_ljung_box(acf: pl.Expr, length: pl.Expr):
        qstats = length * (length + 2) * acf.cum_sum()
        return qstats.alias("qstats")

    entity_col, _, target_col = X.columns[:3]
    results = (
        X.group_by(entity_col)
        .agg(_acf_sqr_ratio(pl.col(target_col)))
        .select(
            entity_col,
            pl.concat_list(cs.starts_with("acorr_")).alias("acf"),
            pl.col("length"),
        )
        .explode("acf")
        .group_by(entity_col)
        .agg(_qstat_ljung_box(pl.col("acf"), pl.col("length")))
    )
    return results


def normality_test(X: pl.DataFrame) -> pl.DataFrame:
    entity_col, _, target_col = X.columns[:3]
    results = X.group_by(entity_col).agg(
        pl.col(target_col)
        .map_elements(lambda s: normaltest(s.to_numpy())[0])
        .alias("normal_test")
    )
    return results


def _rank_entities_by_stat(y_true: pl.DataFrame, sort_by: str, descending: bool):
    entity_col, _, target_col = y_true.columns[:3]
    if sort_by == "mean":
        stats = y_true.group_by(entity_col).agg(
            pl.col(target_col).mean().alias(sort_by)
        )
    elif sort_by == "median":
        stats = y_true.group_by(entity_col).agg(
            pl.col(target_col).median().alias(sort_by)
        )
    elif sort_by == "std":
        stats = y_true.group_by(entity_col).agg(pl.col(target_col).std().alias(sort_by))
    elif sort_by == "cv":
        stats = y_true.group_by(entity_col).agg(
            (pl.col(target_col).std() / pl.col(target_col).mean()).alias(sort_by)
        )
    else:
        raise ValueError(f"`sort_by` method '{sort_by}' not found")
    ranks = stats.sort(by=sort_by, descending=descending).select([entity_col, sort_by])
    return ranks


def _rank_entities_by_score(
    y_true: pl.DataFrame, y_pred: pl.DataFrame, sort_by: str, descending: bool
):
    scoring = SORT_BY_TO_METRIC(y_true)[sort_by]
    ranks = (
        scoring(y_true=y_true, y_pred=y_pred)
        .sort(by=sort_by, descending=descending)
        .select([y_true.columns[0], sort_by])
    )
    return ranks


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


def rank_point_forecasts(
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
    sort_by: FORECAST_SORT_BY = "smape",
    descending: bool = False,
) -> pl.DataFrame:
    """Sorts point forecasts in `y_pred` across entities / time-series by score.

    Parameters
    ----------
    y_true : pl.DataFrame
        Panel DataFrame of observed values.
    y_pred : pl.DataFrame
        Panel DataFrame of point forecasts.
    sort_by : str
        Method to sort forecasts by:\n
        - mean
        - median
        - std
        - cv (coefficient of variation / volatility)
        - mae (mean absolute error)
        - mape (mean absolute percentage error)
        - mase (mean absolute scaled error)
        - mse (mean squared error)
        - rmse (root mean squared error)
        - rmsse (root mean scaled squared error)
        - smape (symmetric mean absolute percentage error)
        - smape_original
        - overforecast
        - underforecast
    descending : bool
        Sort in descending order. Defaults to False.

    Returns
    -------
    ranks : pl.DataFrame
        Cross-sectional DataFrame with two columns: entity name and score.
    """
    ranks = _rank_entities(
        y_true=y_true, y_pred=y_pred, sort_by=sort_by.lower(), descending=descending
    )
    return ranks


def rank_residuals(
    y_resids: pl.DataFrame,
    sort_by: RESIDUALS_SORT_BY = "abs_bias",
    descending: bool = False,
) -> pl.DataFrame:
    """Sorts point forecasts in `y_pred` across entities / time-series by score.

    Parameters
    ----------
    y_resids : pl.DataFrame
        Panel DataFrame of residuals by splits.
    sort_by : str
        Method to sort residuals by: `bias`, `abs_bias` (absolute bias),
        `normality` (via skew and kurtosis tests), or `autocorr` (Ljung-box test for lag = 1).
        Defaults to `abs_bias`.
    descending : bool
        Sort in descending order. Defaults to False.

    Returns
    -------
    ranks : pl.DataFrame
        Cross-sectional DataFrame with two columns: entity name and score.
    """
    entity_col, _, target_col = y_resids.columns[:3]
    y_resids = y_resids.lazy()
    if sort_by == "autocorr":
        ranks = (
            ljung_box_test(y_resids.collect(), max_lags=1)
            .with_columns(pl.col("qstats").list.first())
            .sort("qstats", descending=descending)
            .rename({"qstats": "qstat"})
        )
    elif sort_by == "normality":
        ranks = normality_test(y_resids.collect()).sort(
            "normal_test", descending=descending
        )
    else:
        sort_by_to_expr = {
            "bias": pl.col(target_col).mean().abs(),
            "abs_bias": pl.col(target_col).mean().abs(),
        }
        ranks = (
            y_resids.group_by(entity_col)
            .agg(sort_by_to_expr[sort_by].alias(sort_by))
            .sort(sort_by, descending=descending)
            .collect()
        )
    return ranks


def rank_fva(
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
    y_pred_bench: Optional[pl.DataFrame] = None,
    scoring: Optional[METRIC_TYPE] = None,
    descending: bool = False,
) -> pl.DataFrame:
    """Sorts point forecasts in `y_pred` across entities / time-series by score.

    Parameters
    ----------
    y_true : pl.DataFrame
        Panel DataFrame of observed values.
    y_pred : pl.DataFrame
        Panel DataFrame of point forecasts.
    y_pred_bench : pl.DataFrame
        Panel DataFrame of benchmark forecast values.
    scoring : Optional[metric]
        If None, defaults to SMAPE.
    descending : bool
        Sort in descending order. Defaults to False.

    Returns
    -------
    ranks : pl.DataFrame
        Cross-sectional DataFrame with two columns: entity name and score.
    """
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
