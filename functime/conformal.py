from __future__ import annotations

import polars as pl


def enbpi(
    y_pred: pl.LazyFrame,
    y_resid: pl.LazyFrame,
    alphas: list[float],
) -> pl.DataFrame:
    """Compute prediction intervals using ensemble batch prediction intervals (ENBPI)."""

    # 1. Group residuals by entity
    schema = y_pred.collect_schema()
    entity_col, time_col = schema.names()[:2]
    target_col = schema.names()[-1]
    y_resid_df = y_resid.collect()

    # 2. Forecast future prediction intervals: use constant residual quantile
    y_pred_qnts = []
    for alpha in alphas:
        y_pred_qnt = y_pred.join(
            y_resid_df.group_by(entity_col)
            .agg(pl.col(y_resid_df.columns[-1]).quantile(alpha).alias("score"))
            .lazy(),
            how="left",
            on=entity_col,
            maintain_order="left",
        ).select(
            [
                pl.col(entity_col).cast(schema[entity_col]),
                pl.col(time_col).cast(schema[time_col]),
                pl.col(target_col) + pl.col("score"),
                pl.lit(alpha).alias("quantile"),
            ]
        )
        y_pred_qnts.append(y_pred_qnt)

    y_pred_qnts_df = pl.concat(y_pred_qnts).sort([entity_col, time_col]).collect()
    return y_pred_qnts_df


def conformalize(
    y_pred: pl.DataFrame,
    y_preds: pl.DataFrame,
    y_resids: pl.DataFrame,
    alphas: list[float] | None = None,
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

    y_preds_lazy = y_preds.lazy()
    y_resids_lazy = y_resids.select(y_resids.columns[:3]).lazy()
    y_pred_quantiles = enbpi(y_preds_lazy, y_resids_lazy, alphas)

    # Make alpha base 100
    y_pred_quantiles = y_pred_quantiles.with_columns(
        (pl.col("quantile") * 100).cast(pl.Int16)
    )

    return y_pred_quantiles
