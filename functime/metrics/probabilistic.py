from __future__ import annotations

import numpy as np
import polars as pl

from functime.base import metric


def _score_probabilistic(y_true, y_pred_lower, y_pred_upper, formula: pl.Expr, alias: str):
    """Helper for probabilistic metrics that take lower/upper prediction intervals."""
    entity_col = y_true.columns[0]
    time_col = y_true.columns[1]
    y_true = y_true.rename({y_true.columns[-1]: "actual"})
    y_pred_lower = y_pred_lower.rename({y_pred_lower.columns[-1]: "lower"})
    y_pred_upper = y_pred_upper.rename({y_pred_upper.columns[-1]: "upper"})
    scores = (
        y_true.join(y_pred_lower.select(entity_col, time_col, "lower"), on=[entity_col, time_col], how="left")
        .join(y_pred_upper.select(entity_col, time_col, "upper"), on=[entity_col, time_col], how="left")
        .group_by(entity_col)
        .agg(formula.alias(alias))
    )
    return scores


@metric
def crps(
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
    y_pred_std: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return Continuous Ranked Probability Score (CRPS) assuming Gaussian predictions.

    If `y_pred_std` is not provided, defaults to a standard deviation of 1.0.

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted mean values.
    y_pred_std : pl.DataFrame, optional
        Predicted standard deviations. If None, uses 1.0.

    Returns
    -------
    scores : pl.DataFrame
        CRPS score per entity (lower is better).
    """
    from scipy.stats import norm

    entity_col, time_col = y_true.columns[:2]
    actual = y_true.get_column(y_true.columns[-1]).to_numpy().astype(np.float64)
    predicted = y_pred.rename({y_pred.columns[-1]: y_true.columns[-1]}).select(
        [pl.col(col).cast(dtype) for col, dtype in y_true.schema.items()]
    ).get_column(y_true.columns[-1]).to_numpy().astype(np.float64)

    if y_pred_std is not None:
        std = y_pred_std.get_column(y_pred_std.columns[-1]).to_numpy().astype(np.float64)
    else:
        std = np.ones_like(predicted)

    # CRPS for Gaussian: sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
    # where z = (y - mu) / sigma
    z = (actual - predicted) / std
    crps_vals = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

    result = y_true.select(entity_col).with_columns(pl.lit(crps_vals).alias("crps"))
    scores = result.group_by(entity_col).agg(pl.col("crps").mean())
    return scores


@metric
def interval_coverage(
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
    y_pred_lower: pl.DataFrame | None = None,
    y_pred_upper: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return empirical coverage of prediction intervals.

    The prediction interval is defined by `y_pred_lower` and `y_pred_upper`.
    If only `y_pred` is provided and it contains "lower" and "upper" columns,
    those will be used.

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values (used to determine entity/time columns if lower/upper given separately).
    y_pred_lower : pl.DataFrame, optional
        Lower bound of prediction interval.
    y_pred_upper : pl.DataFrame, optional
        Upper bound of prediction interval.

    Returns
    -------
    scores : pl.DataFrame
        Coverage proportion per entity (values between 0 and 1).
    """
    entity_col, time_col = y_true.columns[:2]
    actual_col = y_true.columns[-1]

    if y_pred_lower is not None and y_pred_upper is not None:
        lower = y_pred_lower.rename({y_pred_lower.columns[-1]: "lower"}).select(entity_col, time_col, "lower")
        upper = y_pred_upper.rename({y_pred_upper.columns[-1]: "upper"}).select(entity_col, time_col, "upper")
    elif "lower" in y_pred.columns and "upper" in y_pred.columns:
        lower = y_pred.select(entity_col, time_col, "lower")
        upper = y_pred.select(entity_col, time_col, "upper")
    else:
        raise ValueError(
            "Must provide either `y_pred_lower` and `y_pred_upper`, "
            "or `y_pred` with 'lower' and 'upper' columns."
        )

    scores = (
        y_true.rename({actual_col: "actual"})
        .join(lower, on=[entity_col, time_col], how="left")
        .join(upper, on=[entity_col, time_col], how="left")
        .group_by(entity_col)
        .agg(
            ((pl.col("actual") >= pl.col("lower")) & (pl.col("actual") <= pl.col("upper")))
            .mean()
            .alias("coverage")
        )
    )
    return scores


@metric
def winkler_score(
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
    y_pred_lower: pl.DataFrame | None = None,
    y_pred_upper: pl.DataFrame | None = None,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Return Winkler interval score.

    The Winkler score penalizes wide intervals and observations outside the interval.

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.
    y_pred_lower : pl.DataFrame, optional
        Lower bound of prediction interval.
    y_pred_upper : pl.DataFrame, optional
        Upper bound of prediction interval.
    alpha : float
        Significance level for the prediction interval. Defaults to 0.05.

    Returns
    -------
    scores : pl.DataFrame
        Winkler score per entity (lower is better).
    """
    entity_col, time_col = y_true.columns[:2]
    actual_col = y_true.columns[-1]

    if y_pred_lower is not None and y_pred_upper is not None:
        lower = y_pred_lower.rename({y_pred_lower.columns[-1]: "lower"}).select(entity_col, time_col, "lower")
        upper = y_pred_upper.rename({y_pred_upper.columns[-1]: "upper"}).select(entity_col, time_col, "upper")
    elif "lower" in y_pred.columns and "upper" in y_pred.columns:
        lower = y_pred.select(entity_col, time_col, "lower")
        upper = y_pred.select(entity_col, time_col, "upper")
    else:
        raise ValueError(
            "Must provide either `y_pred_lower` and `y_pred_upper`, "
            "or `y_pred` with 'lower' and 'upper' columns."
        )

    # Winkler score = interval_width + (2/alpha) * penalty
    # penalty = max(lower - actual, 0) + max(actual - upper, 0)
    scores = (
        y_true.rename({actual_col: "actual"})
        .join(lower, on=[entity_col, time_col], how="left")
        .join(upper, on=[entity_col, time_col], how="left")
        .with_columns(
            (pl.col("upper") - pl.col("lower")).alias("width"),
            (
                pl.when(pl.col("actual") < pl.col("lower"))
                .then((pl.col("lower") - pl.col("actual")) * (2.0 / alpha))
                .when(pl.col("actual") > pl.col("upper"))
                .then((pl.col("actual") - pl.col("upper")) * (2.0 / alpha))
                .otherwise(0.0)
            ).alias("penalty"),
        )
        .group_by(entity_col)
        .agg((pl.col("width") + pl.col("penalty")).mean().alias("winkler"))
    )
    return scores


__all__ = [
    "crps",
    "interval_coverage",
    "winkler_score",
]
