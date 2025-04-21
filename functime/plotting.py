from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from functime._plotting import TimeSeriesDisplay
from functime.base.metric import METRIC_TYPE
from functime.metrics import smape

if TYPE_CHECKING:
    from typing import Any


def plot_entities(
    y: pl.DataFrame | pl.LazyFrame,
    **kwargs,
) -> go.Figure:
    """Given panel DataFrame of observed values `y`,
    returns bar chart of entity counts, which is the number of observed values per entity.

    Parameters
    ----------
    y : pl.DataFrame | pl.LazyFrame
        Panel DataFrame of observed values.
    **kwargs
        Additional keyword arguments to pass to a `plotly.graph_objects.Layout` object.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Plotly bar chart.
    """
    entity_col = y.columns[0]

    if isinstance(y, pl.DataFrame):
        y = y.lazy()

    entity_counts = y.group_by(entity_col).agg(pl.len()).sort("len").collect()

    height = kwargs.pop("height", len(entity_counts) * 20)
    title = kwargs.pop("title", "Entities counts")
    template = kwargs.pop("template", "plotly_white")

    layout = go.Layout(
        height=height,
        title=title,
        template=template,
        **kwargs,
    )

    return go.Figure(
        data=go.Bar(
            x=entity_counts.get_column("len"),
            y=entity_counts.get_column(entity_col),
            orientation="h",
        ),
        layout=layout,
    )


# TODO: if num_points is (0,1] than take a percentage of the points
def plot_panel(
    y: pl.DataFrame | pl.LazyFrame,
    *,
    num_series: int | None = None,
    num_cols: int | None = None,
    num_points: int | None = None,
    seed: int | None = None,
    layout_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
):
    """Given panel DataFrames of observed values `y`,
    returns subplots for each individual entity / time-series.

    Parameters
    ----------
    y : Union[pl.DataFrame, pl.LazyFrame]
        Panel DataFrame of observed values.
    num_series : Optional[int]
        Number of entities / time-series to plot. If `None`, plot all entities.
        Defaults to `None`.
    num_points : Optional[int]
        Plot `last_n` most recent values in `y`. If `None`, plot all points.
        Defaults to `None`.
    num_cols : Optional[int]
        Number of columns to arrange subplots. Defaults to 2.
    seed : Optional[int]
        Random seed for sampling entities / time-series. Defaults to `None`.
    layout_kwargs
        Additional keyword arguments to pass to a `plotly.graph_objects.Layout` object.
    line_kwargs
        Additional keyword arguments to pass to a `plotly.graph_objects.Line` object.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Plotly instance of `Figure` with all the subplots.
    """
    if isinstance(y, pl.DataFrame):
        y = y.lazy()

    drawer = TimeSeriesDisplay.from_panel(
        y=y,
        num_cols=num_cols,
        num_series=num_series,
        seed=seed,
        default_title="Entitites line plot",
        **layout_kwargs or {},
    )

    drawer.add_time_series(
        data=y,
        num_points=num_points,
        show_legend=False,
        **line_kwargs or {"color": drawer.DEFAULT_PALETTE["primary"]},
    )

    return drawer.figure


def plot_forecasts(
    *,
    y_true: pl.DataFrame | pl.LazyFrame,
    y_pred: pl.DataFrame | pl.LazyFrame,
    num_series: int | None = None,
    num_cols: int | None = None,
    num_points: int | None = None,
    seed: int | None = None,
    layout_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Given panel DataFrames of observed values `y` and forecasts `y_pred`,
    returns subplots for each individual entity / time-series.

    Parameters
    ----------
    y : Union[pl.DataFrame, pl.LazyFrame]
        Panel DataFrame of observed values.
    num_series : Optional[int]
        Number of entities / time-series to plot. If `None`, plot all entities.
        Defaults to `None`.
    num_points : Optional[int]
        Plot `last_n` most recent values in `y`. If `None`, plot all points.
        Defaults to `None`.
    num_cols : Optional[int]
        Number of columns to arrange subplots. Defaults to 2.
    seed : Optional[int]
        Random seed for sampling entities / time-series.
        Defaults to None.
    layout_kwargs
        Additional keyword arguments to pass to `plotly.graph_objects.Figure.update_layout` or, equivalently, a `plotly.graph_objects.Layout` object.
    line_kwargs
        Additional keyword arguments to pass to a `plotly.graph_objects.Line` object.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Plotly instance of `Figure` with all the subplots.
    """
    if isinstance(y_true, pl.DataFrame):
        y_true = y_true.lazy()

    if isinstance(y_pred, pl.DataFrame):
        y_pred = y_pred.lazy()

    drawer = TimeSeriesDisplay.from_panel(
        y=y_true,
        num_cols=num_cols,
        num_series=num_series,
        seed=seed,
        default_title="Predictions versus actuals",
        **layout_kwargs or {},
    )

    drawer.add_time_series(
        data=y_true,
        num_points=num_points,
        name_on_hover="Actual",
        legend_group="Actual",
        **line_kwargs or {"color": drawer.DEFAULT_PALETTE["actual"]},
    )

    drawer.add_time_series(
        data=y_pred,
        num_points=num_points,
        name_on_hover="Forecast",
        legend_group="Forecast",
        **line_kwargs or {"color": drawer.DEFAULT_PALETTE["primary"], "dash": "dot"},
    )

    return drawer.figure


def plot_backtests(
    y_true: pl.DataFrame | pl.LazyFrame,
    y_preds: pl.DataFrame | pl.LazyFrame,
    *,
    num_series: int | None = None,
    num_cols: int | None = None,
    num_points: int | None = None,
    seed: int | None = None,
    layout_kwargs: dict[str, Any] | None = None,
    line_kwargs: dict[str, Any] | None = None,
):
    if isinstance(y_true, pl.DataFrame):
        y_true = y_true.lazy()

    if isinstance(y_preds, pl.DataFrame):
        y_preds = y_preds.lazy()

    drawer = TimeSeriesDisplay.from_panel(
        y=y_true,
        num_cols=num_cols,
        num_series=num_series,
        seed=seed,
        default_title="Predictions versus actuals",
        **layout_kwargs or {},
    )

    drawer.add_time_series(
        data=y_true,
        num_points=num_points,
        name_on_hover="Actual",
        legend_group="Actual",
        **line_kwargs or {"color": drawer.DEFAULT_PALETTE["actual"]},
    )

    for name, y_pred in y_preds.collect().group_by(["split"], maintain_order=True):
        drawer.add_time_series(
            data=y_pred.lazy(),
            num_points=num_points,
            name_on_hover=f"Split {name[0]}",  # pyright: ignore[reportIndexIssue]
            legend_group=f"Split {name[0]}",  # pyright: ignore[reportIndexIssue]
            **line_kwargs or {"dash": "dot"},
        )

    return drawer.figure


def plot_residuals(
    y_resids: pl.DataFrame | pl.LazyFrame,
    n_bins: int | None = None,
    **kwargs,
) -> go.Figure:
    """Given panel DataFrame of residuals across splits `y_resids`,
    returns binned counts plot of forecast residuals colored by entity / time-series.

    Useful for residuals analysis (bias and normality) at scale.

    Parameters
    ----------
    y_resids : Union[pl.DataFrame, pl.LazyFrame]
        Panel DataFrame of forecast residuals (i.e. observed less forecast).
    n_bins : int
        Number of bins.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Plotly histogram.
    """
    entity_col, _, target_col = y_resids.columns[:3]

    if isinstance(y_resids, pl.DataFrame):
        y_resids = y_resids.lazy()

    y_resids = y_resids.with_columns(pl.col(target_col).alias("Residuals")).collect()

    fig = px.histogram(
        y_resids,
        x="Residuals",
        y="Residuals",
        color=entity_col,
        marginal="rug",
        histfunc="count",
        nbins=n_bins,
    )

    template = kwargs.pop("template", "plotly_white")

    fig.update_layout(template=template, **kwargs)
    return fig


def plot_comet(
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    y_pred: pl.DataFrame,
    scoring: METRIC_TYPE | None = None,
    **kwargs,
):
    """Given a train-test-split of panel data (`y_train`, `y_test`) and forecast `y_pred`,
    returns a Comet plot i.e. scatterplot of volatility per entity in `y_train` against the forecast scores.

    Parameters
    ----------
    y_train : pl.DataFrame
        Panel DataFrame of train dataset.
    y_test : pl.DataFrame
        Panel DataFrame of test dataset.
    y_pred : pl.DataFrame
        Panel DataFrame of forecasted values to score against `y_test`.
    scoring : Optional[metric]
        If None, defaults to SMAPE.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Plotly scatterplot.
    """
    entity_col, _, target_col = y_train.columns[:3]

    scoring = scoring or smape

    # FIX: this fails when scoring is not SMAPE.
    scores = scoring(y_true=y_test, y_pred=y_pred)

    cvs = y_train.group_by(entity_col).agg(
        (pl.col(target_col).var() / pl.col(target_col).mean()).alias("CV")
    )

    comet = scores.join(cvs, on=entity_col, how="left").drop_nulls()
    mean_score = scores.get_column(scores.columns[-1]).mean()
    mean_cv = cvs.get_column(cvs.columns[-1]).mean()
    fig = px.scatter(
        comet, x=cvs.columns[-1], y=scores.columns[-1], hover_data=entity_col
    )
    fig.add_hline(y=mean_score)
    fig.add_vline(x=mean_cv)

    template = kwargs.pop("template", "plotly_white")

    fig.update_layout(template=template, **kwargs)
    return fig


def plot_fva(
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
    y_pred_bench: pl.DataFrame,
    scoring: METRIC_TYPE | None = None,
    **kwargs,
):
    """Given two panel data forecasts `y_pred` and `y_pred_bench`,
    returns scatterplot of benchmark scores against forecast scores.
    Each dot represents a single entity / time-series.

    Parameters
    ----------
    y_true : pl.DataFrame
        Panel DataFrame of test dataset.
    y_pred : pl.DataFrame
        Panel DataFrame of forecasted values.
    y_pred_bench : pl.DataFrame
        Panel DataFrame of benchmark forecast values.
    scoring : Optional[metric]
        If None, defaults to SMAPE.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Plotly scatterplot.
    """
    scoring = scoring or smape
    scores = scoring(y_true=y_true, y_pred=y_pred)
    scores_bench = scoring(y_true=y_true, y_pred=y_pred_bench)
    entity_col, metric_name = scores_bench.columns
    x_title = f"Benchmark ({metric_name})"
    y_title = f"Forecast ({metric_name})"
    uplift = scores.rename({metric_name: y_title}).join(
        scores_bench.rename({metric_name: x_title}),
        how="left",
        on=scores.columns[0],
    )
    fig = px.scatter(uplift, x=x_title, y=y_title, hover_data=entity_col)
    deg45_line = {
        "type": "line",
        "yref": "paper",
        "xref": "paper",
        "y0": 0,
        "y1": 1,
        "x0": 0,
        "x1": 1,
    }
    max_score = max(
        scores.get_column(metric_name).max(), scores_bench.get_column(metric_name).max()
    )
    min_score = min(
        scores.get_column(metric_name).min(), scores_bench.get_column(metric_name).min()
    )

    template = kwargs.pop("template", "plotly_white")

    fig.update_layout(
        shapes=[deg45_line],
        xaxis={"range": [min_score, max_score]},
        yaxis={"range": [min_score, max_score]},
        template=template,
    )
    fig.update_layout(**kwargs)
    return fig
