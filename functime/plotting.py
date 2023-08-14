from functools import partial
from typing import Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from functime.base.metric import METRIC_TYPE
from functime.evaluation import rank_fva

COLOR_PALETTE = {"actual": "#B7B7B7", "forecast": "#1b57f1", "backtest": "#A76EF4"}
DEFAULT_LAST_N = 64


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
    last_n: int = DEFAULT_LAST_N,
    **kwargs
) -> go.Figure:

    # Get most recent observations
    entity_col, time_col, target_col = y.columns
    y = y.groupby(entity_col).tail(last_n)

    # Organize subplots
    n_series = y.get_column(entity_col).n_unique()
    n_rows = int(np.ceil(n_series / n_cols))
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
    last_n: int = DEFAULT_LAST_N,
    **kwargs
) -> go.Figure:

    # Get most recent observations
    entity_col, time_col, target_col = y.columns
    y = y.groupby(entity_col).tail(last_n)

    # Organize subplots
    n_series = y.get_column(entity_col).n_unique()
    n_rows = n_series // n_cols
    row_idx = np.repeat(range(n_rows), n_cols)
    entity_ids = y.get_column(entity_col).unique(maintain_order=True)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=entity_ids)

    for i, entity_id in enumerate(entity_ids):
        ts = y.filter(pl.col(entity_col) == entity_id)
        ts_pred = y_preds.filter(pl.col(entity_col) == entity_id)
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


def plot_residuals(
    y_resids: pl.DataFrame, n_bins: Optional[int] = None, **kwargs
) -> go.Figure:
    entity_col, _, target_col = y_resids.columns[:3]
    y_resids = y_resids.with_columns(pl.col(target_col).alias("Residuals"))
    fig = px.histogram(
        y_resids,
        x="Residuals",
        y="Residuals",
        color=entity_col,
        marginal="rug",
        histfunc="count",
        nbins=n_bins,
    )
    fig.update_layout(**kwargs)
    return fig


def plot_acf(y: pl.DataFrame, max_lags: int, alpha: float = 0.05):
    pass


def plot_residuals_acf(y_resids: pl.DataFrame):
    pass


def plot_comet(
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    y_pred: pl.DataFrame,
    scoring: Optional[METRIC_TYPE] = None,
    **kwargs
):
    entity_col, _, target_col = y_train.columns
    scoring = scoring or partial(mase, y_train=y_train)
    scores = scoring(y_true=y_test, y_pred=y_pred)
    cvs = y_train.groupby(entity_col).agg(
        pl.col(target_col).var() / pl.col(target_col).mean()
    )
    comet = scores.join(cvs, on=entity_col, how="left").drop_nulls()
    mean_score = scores.get_column(scores.columns[-1]).mean()
    mean_cv = cvs.get_column(cvs.columns[-1]).mean()
    fig = px.scatter(
        comet, x=cvs.columns[-1], y=scores.columns[-1], hover_data=entity_col
    )
    fig.add_hline(y=mean_score)
    fig.add_vline(x=mean_cv)
    fig.update_layout(**kwargs)
    return fig


def plot_fva(
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
    y_pred_bench: pl.DataFrame,
    scoring: Optional[METRIC_TYPE] = None,
    **kwargs
):
    uplift = rank_fva(
        y_true=y_true, y_pred=y_pred, y_pred_bench=y_pred_bench, scoring=scoring
    )
    entity_col, metric_name, metric_bench_name = uplift.columns[:2]
    fig = px.scatter(uplift, x=metric_bench_name, y=metric_name, hover_data=entity_col)
    deg45_line = {
        "type": "line",
        "yref": "paper",
        "xref": "paper",
        "y0": 0,
        "y1": 1,
        "x0": 0,
        "x1": 1,
    }
    fig.update_layout(shapes=[deg45_line])
    fig.update_layout(**kwargs)
    return fig


if __name__ == "__main__":

    from functime.cross_validation import train_test_split
    from functime.forecasting import snaive
    from functime.metrics import mase

    y = pl.read_parquet("data/commodities.parquet")
    entity_col = y.columns[0]
    fh = 24
    y_train, y_test = train_test_split(test_size=fh)(y)
    y_pred = snaive(freq="1mo", sp=12)(y=y_train, fh=fh)
    scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
    top_scoring = scores.sort("mase").get_column(entity_col).head(n=4)

    y = y.filter(pl.col(entity_col).is_in(top_scoring))
    y_pred = y_pred.filter(pl.col(entity_col).is_in(top_scoring))
    fig = plot_forecasts(y=y, y_pred=y_pred, width=1150)
    fig.show()
