from functools import partial
from typing import List, Mapping, Optional, Union

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy.stats import norm, normaltest
from typing_extensions import Literal

from functime.base.metric import METRIC_TYPE
from functime.metrics import mae, mase, smape

COLOR_PALETTE = {"actual": "#B7B7B7", "forecast": "#1b57f1", "backtest": "#A76EF4"}


def _sort_entities_by_stat(
    y: pl.DataFrame,
    y_pred: pl.DataFrame,
    descending: bool,
    sort_by: Literal["mean", "median", "cv", "smape", "mae", "mase"],
) -> pl.DataFrame:
    # NOTE: Sort might be faster if we agg all other cols into list (so no duplicate agg stats)
    entity_col, time_col, target_col = y.columns
    if sort_by in ["smape", "mae", "mase"]:
        metric_to_scoring = {
            "smape": smape,
            "mae": mae,
            "mase": partial(mase, y_train=y),
        }
        scores = metric_to_scoring[sort_by](y_true=y, y_pred=y_pred)
        y_new = y.join(scores, on=entity_col, how="left")
    else:
        sort_expr = {
            "mean": pl.col(target_col).mean(),
            "median": pl.col(target_col).max(),
            "sum": pl.col(target_col).sum(),
            "cv": pl.col(target_col).std() / pl.col(target_col).mean(),
        }
        y_new = y.with_columns(sort_expr[sort_by].over(entity_col).alias(sort_by))
    y_sorted = y_new.sort(
        by=[sort_by, entity_col, time_col], descending=descending
    ).select(y.columns)
    return y_sorted


def _sample_entities(
    y: pl.DataFrame,
    y_pred: pl.DataFrame,
    top_k: int,
    last_n: int,
    random: bool,
    descending: bool,
    sort_by: Literal["mean", "median", "cv", "smape", "mae", "mase"],
) -> pl.DataFrame:
    entity_col = y.columns[0]
    if random:
        y_sample = (
            y.lazy()
            .groupby(entity_col)
            .agg(pl.all().tail(last_n))
            .sample(n=top_k)
            .collect()
        )
    else:
        y_sample = (
            y.pipe(
                _sort_entities_by_stat,
                y_pred=y_pred,
                descending=descending,
                sort_by=sort_by,
            )
            .lazy()
            .groupby(entity_col, maintain_order=True)
            .agg(pl.all().tail(last_n))
            .head(top_k)
            .collect()
        )
    return y_sample.explode(pl.all().exclude(entity_col))


def _remove_legend_duplicates(fig: go.Figure) -> go.Figure:
    names = set()
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )
    return fig


def plot_forecasts(
    y: pl.DataFrame,
    y_pred: pl.DataFrame,
    n_cols: int = 2,
    top_k: int = 10,
    last_n: int = 48,
    random: bool = False,
    descending: bool = False,
    sort_by: Literal["mean", "median", "sum", "cv", "smape", "mae", "mase"] = "smape",
    **kwargs
) -> go.Figure:

    entity_col, time_col, target_col = y.columns
    y = y.pipe(
        _sample_entities,
        y_pred=y_pred,
        top_k=top_k,
        last_n=last_n,
        random=random,
        descending=descending,
        sort_by=sort_by,
    )
    n_rows = int(np.ceil(top_k / n_cols))
    row_idx = np.repeat(range(n_rows), n_cols)
    entity_ids = y.get_column(entity_col).unique(maintain_order=True)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=entity_ids)

    for i, entity_id in enumerate(entity_ids):
        ts = y.filter(pl.col(entity_col) == entity_id)
        ts_pred = y_pred.filter(pl.col(entity_col) == entity_id)
        row = row_idx[i] + 1
        col = i % n_cols + 1
        # Plot actual
        fig.add_trace(
            go.Scatter(
                x=ts.get_column(time_col),
                y=ts.get_column(target_col),
                name="Actual",
                legendgroup="Actual",
                line=dict(color=COLOR_PALETTE["actual"]),
            ),
            row=row,
            col=col,
        )
        # Plot forecast
        fig.add_trace(
            go.Scatter(
                x=ts_pred.get_column(time_col),
                y=ts_pred.get_column(target_col),
                name="Forecast",
                legendgroup="Forecast",
                line=dict(color=COLOR_PALETTE["forecast"], dash="dash"),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(**kwargs)
    fig = _remove_legend_duplicates(fig)
    return fig


def plot_backtests(
    y: pl.DataFrame,
    y_preds: pl.DataFrame,
    n_cols: int = 2,
    top_k: int = 10,
    last_n: int = 48,
    random: bool = False,
    descending: bool = False,
    sort_by: Literal["mean", "median", "sum", "cv", "smape", "mae", "mase"] = "smape",
    agg_splits_by: Literal["mean", "median"] = "mean",
    **kwargs
) -> go.Figure:
    entity_col, time_col, target_col = y.columns
    agg_by = {"mean": pl.col(target_col).mean(), "median": pl.col(target_col).median()}
    y_pred = (
        y_preds.groupby([entity_col, time_col])
        .agg(agg_by[agg_splits_by])
        .sort([entity_col, time_col])
        .set_sorted([entity_col, time_col])
    )
    y = y.pipe(
        _sample_entities,
        y_pred=y_pred,
        top_k=top_k,
        last_n=last_n,
        random=random,
        descending=descending,
        sort_by=sort_by,
    )
    n_rows = top_k // n_cols
    row_idx = np.repeat(range(n_rows), n_cols)
    entity_ids = y.get_column(entity_col).unique(maintain_order=True)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=entity_ids)

    for i, entity_id in enumerate(entity_ids):
        ts = y.filter(pl.col(entity_col) == entity_id)
        ts_pred = y_pred.filter(pl.col(entity_col) == entity_id)
        row = row_idx[i] + 1
        col = i % n_cols + 1
        # Plot actual
        fig.add_trace(
            go.Scatter(
                x=ts.get_column(time_col),
                y=ts.get_column(target_col),
                name="Actual",
                legendgroup="Actual",
                line=dict(color=COLOR_PALETTE["actual"]),
            ),
            row=row,
            col=col,
        )
        # Plot forecast
        fig.add_trace(
            go.Scatter(
                x=ts_pred.get_column(time_col),
                y=ts_pred.get_column(target_col),
                name="Backtest",
                legendgroup="Backtest",
                line=dict(color=COLOR_PALETTE["backtest"], dash="dash"),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(**kwargs)
    fig = _remove_legend_duplicates(fig)
    return fig


def ljungbox_test(
    x: pl.Series, max_lags: int, alpha: float = 0.05
) -> Mapping[str, Union[List[pl.Expr], pl.Expr]]:
    n = x.len()
    # Brute force ACF calculation (might be slow for long series and lags)
    acf = [pl.corr(x, x.shift(i), ddof=i) for i in range(1, max_lags + 1)]
    # Calculate variance using Bartlett's formula
    varacf = pl.repeat(1, max_lags + 1, eager=True) / n
    cumsum_var = pl.cumsum_horizontal(varacf.slice(1, n - 1) ** 2)
    varacf = [varacf.get(i) * (1 + 2 * cumsum_var) for i in range(2, max_lags)]
    pff = norm.pff(1 - alpha / 2.0)
    intervals = [pff * np.sqrt(var) for var in varacf]
    qstat = n * (n + 2) * np.sum((acf[1:] ** 2) / (n - np.arange(1, max_lags + 1)))
    return {"qstat": qstat, "acf": acf, "intervals": intervals}


def plot_residuals(
    y_resids: pl.DataFrame,
    top_k: int = 4,
    n_bins: Optional[int] = None,
    sort_by: Literal[
        "normality", "bias", "pos_bias", "neg_bias", "autocorr"
    ] = "normality",
) -> go.Figure:

    entity_col, time_col, target_col = y_resids.columns[:3]

    # Rank time-series
    gb = y_resids.select([entity_col, time_col, target_col]).groupby(entity_col)
    if sort_by == "bias":
        y_resids = gb.agg(
            [pl.all(), pl.col(target_col).mean().abs().alias(sort_by)]
        ).sort(by=sort_by)
    elif sort_by == "autocorr":
        pass
    elif sort_by == "pos_bias":
        y_resids = gb.agg(
            [pl.all(), pl.col(target_col).mean().abs().alias(sort_by)]
        ).sort(by=sort_by, descending=True)
    elif sort_by == "neg_bias":
        y_resids = gb.agg([pl.all(), pl.col(target_col).mean().alias(sort_by)]).sort(
            by=sort_by
        )
    else:
        norm_test_expr = (
            pl.col(target_col)
            .apply(lambda x: normaltest(x.to_numpy())[0])
            .alias(sort_by)
        )
        y_resids = (
            gb.agg([pl.all(), norm_test_expr])
            # higher stat == lower p-value
            .sort(by=sort_by, descending=True)
        )
    y_resids = (
        y_resids.head(top_k)
        .explode([time_col, target_col])
        .with_columns(pl.col(target_col).alias("Residuals"))
    )
    fig_dist = px.histogram(
        y_resids,
        x="Residuals",
        y="Residuals",
        color=entity_col,
        marginal="rug",
        histfunc="count",
        nbins=n_bins,
    )
    return fig_dist


def plot_acf(y: pl.DataFrame):
    pass


def plot_residuals_acf(y_resids: pl.DataFrame):
    pass


def plot_comet(
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    y_pred: pl.DataFrame,
    scoring: Optional[METRIC_TYPE] = None,
):
    pass


def plot_fva(y_test: pl.DataFrame, y_pred: pl.DataFrame, y_pred_bench: pl.DataFrame):
    pass


if __name__ == "__main__":

    from functime.cross_validation import train_test_split
    from functime.forecasting import linear_model, snaive

    y = pl.read_parquet("data/commodities.parquet")
    y_train, y_test = train_test_split(test_size=6, eager=True)(y)
    y_pred = snaive(freq="1mo", sp=12)(y=y_train, fh=12)
    y_preds, y_resids = linear_model(freq="1mo", lags=24).backtest(
        y=y, test_size=12, step_size=12, n_splits=5
    )

    fig_forecast = plot_forecasts(y=y_train, y_pred=y_pred, height=1200)
    fig_backtest = plot_backtests(y=y, y_preds=y_preds, height=1200)
    fig_forecast.show()
    fig_backtest.show()

    fig_resids_bias = plot_residuals(y_resids=y_resids, sort_by="bias")
    fig_resids_bias.show()
