from typing import List, Optional

import polars as pl


def enbpi(
    y_pred: pl.LazyFrame,
    y_resid: pl.LazyFrame,
    alphas: List[float],
) -> pl.DataFrame:
    """Compute prediction intervals using ensemble batch prediction intervals (ENBPI)."""

    # 1. Group residuals by entity
    entity_col, time_col = y_pred.columns[:2]
    y_resid = y_resid.collect()

    # 2. Forecast future prediction intervals: use constant residual quantile
    schema = y_pred.schema
    y_pred_qnts = []
    for alpha in alphas:
        y_pred_qnt = y_pred.join(
            y_resid.group_by(entity_col)
            .agg(pl.col(y_resid.columns[-1]).quantile(alpha).alias("score"))
            .lazy(),
            how="left",
            on=entity_col,
        ).select(
            [
                pl.col(entity_col).cast(schema[entity_col]),
                pl.col(time_col).cast(schema[time_col]),
                pl.col(y_pred.columns[-1]) + pl.col("score"),
                pl.lit(alpha).alias("quantile"),
            ]
        )
        y_pred_qnts.append(y_pred_qnt)

    y_pred_qnts = pl.concat(y_pred_qnts).sort([entity_col, time_col]).collect()
    return y_pred_qnts


def conformalize(
    y_pred: pl.DataFrame,
    y_preds: pl.DataFrame,
    y_resids: pl.DataFrame,
    alphas: Optional[List[float]] = None,
) -> pl.DataFrame:
    """Compute prediction intervals using ensemble batch prediction intervals (ENBPI)."""

    alphas = alphas or [0.1, 0.9]
    entity_col, time_col, target_col = y_pred.columns[:3]
    schema = y_pred.schema
    y_preds = pl.concat(
        [
            y_pred,
            y_preds.select(
                [
                    entity_col,
                    pl.col(time_col).cast(schema[time_col]),
                    pl.col(target_col).cast(schema[target_col]),
                ]
            ),
        ]
    )

    y_preds = y_preds.lazy()
    y_resids = y_resids.select(y_resids.columns[:3]).lazy()
    y_pred_quantiles = enbpi(y_preds, y_resids, alphas)

    # Make alpha base 100
    y_pred_quantiles = y_pred_quantiles.with_columns(
        (pl.col("quantile") * 100).cast(pl.Int16)
    )

    return y_pred_quantiles
