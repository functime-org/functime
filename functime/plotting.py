from typing import Optional, Union

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from functime.base.metric import METRIC_TYPE
from functime.metrics import smape

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


def _set_subplot_default_kwargs(kwargs: dict, n_rows: int, n_cols: int) -> dict:
    """
    Sets or adjusts plot layout properties based on the number of rows and columns in subplots,
    ensuring a fixed size for subplots and additional space for titles and other elements.
    Default values are applied only if not already specified by the user.

    Parameters:
    - kwargs (dict): The original keyword arguments dictionary passed to the plotting function.
    - n_rows (int): Number of rows in the subplot.
    - n_cols (int): Number of columns in the subplot.

    Returns:
    - dict: Updated keyword arguments with adjusted layout properties.

    The function ensures the following:
    - A fixed size for each subplot.
    - Additional space for plot titles and other elements.
    - The overall figure size is dynamically adjusted based on the subplot configuration.
    """
    # Fixed size for each subplot
    subplot_width = 250  # width for each subplot column
    subplot_height = 200 # height for each subplot row

    # Additional space for titles and other elements
    additional_space_vertical = 100   # space for titles, labels, etc.
    additional_space_horizontal = 100 # additional horizontal space if needed

    # Calculate total width and height
    total_width = subplot_width * n_cols + additional_space_horizontal
    total_height = subplot_height * n_rows + additional_space_vertical

    # Apply defaults if not already specified by the user
    kwargs.setdefault("width", total_width)
    kwargs.setdefault("height", total_height)
    kwargs.setdefault("template", "plotly_white")

    return kwargs


def plot_entities(
    y: Union[pl.DataFrame, pl.LazyFrame],
    **kwargs,
) -> go.Figure:
    """Given panel DataFrame of observed values `y`,
    returns bar chart of entity counts, which is the number of observed values per entity.

    Parameters
    ----------
    y : pl.DataFrame | pl.LazyFrame
        Panel DataFrame of observed values.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Plotly bar chart.
    """
    entity_col = y.columns[0]

    if isinstance(y, pl.DataFrame):
        y = y.lazy()

    entity_counts = y.group_by(entity_col).agg(pl.count()).collect()

    height = kwargs.pop("height", len(entity_counts) * 20)
    title = kwargs.pop("title", "Entities counts")
    template = kwargs.pop("template", "plotly_white")

    return px.bar(
        data_frame=entity_counts,
        x="count",
        y=entity_col,
        orientation="h",
    ).update_layout(
        height=height,
        title=title,
        template=template,
        **kwargs,
    )


def plot_panel(
    y: Union[pl.DataFrame, pl.LazyFrame],
    *,
    n_series: int = 10,
    seed: int | None = None,
    n_cols: int = 2,
    last_n: int = DEFAULT_LAST_N,
    **kwargs,
):
    """Given panel DataFrames of observed values `y`,
    returns subplots for each individual entity / time-series.

    Parameters
    ----------
    y : Union[pl.DataFrame, pl.LazyFrame]
        Panel DataFrame of observed values.
    n_series : int
        Number of entities / time-series to plot.
        Defaults to 10.
    seed : int | None
        Random seed for sampling entities / time-series.
        Defaults to None.
    n_cols : int
        Number of columns to arrange subplots.
        Defaults to 2.
    last_n : int
        Plot `last_n` most recent values in `y` and `y_pred`.
        Defaults to 64.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Plotly subplots.
    """
    entity_col, time_col, target_col = y.columns[:3]

    if isinstance(y, pl.DataFrame):
        y = y.lazy()

    entities = y.select(pl.col(entity_col).unique(maintain_order=True)).collect()

    entities_sample = entities.to_series().sample(n_series, seed=seed)

    # Get most recent observations
    y = (
        y.filter(pl.col(entity_col).is_in(entities_sample))
        .group_by(entity_col)
        .tail(last_n)
        .collect()
    )

    # Organize subplots
    n_rows = n_series // n_cols + (n_series % n_cols > 0)
    row_idx = np.repeat(range(n_rows), n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=entities_sample)

    for i, entity_id in enumerate(entities_sample):
        ts = y.filter(pl.col(entity_col) == entity_id)
        row = row_idx[i] + 1
        col = i % n_cols + 1
        # Plot actual
        fig.add_trace(
            go.Scatter(
                x=ts.get_column(time_col),
                y=ts.get_column(target_col),
                name="Time-series",
                legendgroup="Time-series",
                line=dict(color=COLOR_PALETTE["forecast"]),
            ),
            row=row,
            col=col,
        )

    # Set default kwargs for plotting if user did not provide these
    kwargs = _set_subplot_default_kwargs(kwargs=kwargs, n_rows=n_rows, n_cols=n_cols)

    # Tidy up the plot
    fig.update_layout(**kwargs)
    fig = _remove_legend_duplicates(fig)

    return fig


def plot_forecasts(
    y_true: Union[pl.DataFrame, pl.LazyFrame],
    y_pred: pl.DataFrame,
    *,
    n_series: int = 10,
    seed: int | None = None,
    n_cols: int = 2,
    last_n: int = DEFAULT_LAST_N,
    **kwargs,
) -> go.Figure:
    """Given panel DataFrames of observed values `y` and forecasts `y_pred`,
    returns subplots for each individual entity / time-series.

    Parameters
    ----------
    y_true : Union[pl.DataFrame, pl.LazyFrame]
        Panel DataFrame of observed values.
    y_pred : pl.DataFrame
        Panel DataFrame of forecasted values.
    n_series : int
        Number of entities / time-series to plot.
        Defaults to 10.
    seed : int | None
        Random seed for sampling entities / time-series.
        Defaults to None.
    n_cols : int
        Number of columns to arrange subplots.
        Defaults to 2.
    last_n : int
        Plot `last_n` most recent values in `y` and `y_pred`.
        Defaults to 64.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Plotly subplots.
    """
    entity_col, time_col, target_col = y_true.columns[:3]

    if isinstance(y_true, pl.DataFrame):
        y_true = y_true.lazy()

    # Get the unique entities
    entities = y_true.select(pl.col(entity_col).unique(maintain_order=True)).collect()
    # Get sampled entities
    entities_sample = entities.to_series().sample(n_series, seed=seed)

    # Get the most recent observations for the sampled entities
    y = (
        y_true.filter(pl.col(entity_col).is_in(entities_sample))
        .group_by(entity_col)
        .tail(last_n)
        .collect()
    )

    # Organize subplots
    n_rows = n_series // n_cols + (n_series % n_cols > 0)
    row_idx = np.repeat(range(n_rows), n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=entities_sample)

    for i, entity_id in enumerate(entities_sample):
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

    # Set default kwargs for plotting if user did not provide these
    kwargs = _set_subplot_default_kwargs(kwargs=kwargs, n_rows=n_rows, n_cols=n_cols)

    # Tidy up the plot
    fig.update_layout(**kwargs)
    fig = _remove_legend_duplicates(fig)
    return fig


def plot_backtests(
    y_true: Union[pl.DataFrame, pl.LazyFrame],
    y_preds: pl.DataFrame,
    *,
    n_series: int = 10,
    seed: int | None = None,
    n_cols: int = 2,
    last_n: int = DEFAULT_LAST_N,
    **kwargs,
) -> go.Figure:
    """Given panel DataFrame of observed values `y` and backtests across splits `y_pred`,
    returns subplots for each individual entity / time-series.

    Parameters
    ----------
    y_true : Union[pl.DataFrame, pl.LazyFrame]
        Panel DataFrame of observed values.
    y_preds : pl.DataFrame
        Panel DataFrame of backtested values.
    n_series : int
        Number of entities / time-series to plot.
        Defaults to 10.
    seed : int | None
        Random seed for sampling entities / time-series.
        Defaults to None.
    n_cols : int
        Number of columns to arrange subplots.
        Defaults to 2.
    last_n : int
        Plot `last_n` most recent values in `y` and `y_pred`.
        Defaults to 64.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Plotly subplots.
    """
    entity_col, time_col, target_col = y_true.columns[:3]

    if isinstance(y_true, pl.DataFrame):
        y_true = y_true.lazy()

    # Get most recent observations
    entities = y_true.select(pl.col(entity_col).unique(maintain_order=True)).collect()

    entities_sample = entities.to_series().sample(n_series, seed=seed)

    # Get most recent observations
    y = (
        y_true.filter(pl.col(entity_col).is_in(entities_sample))
        .group_by(entity_col)
        .tail(last_n)
        .collect()
    )

    # Organize subplots
    n_rows = n_series // n_cols + (n_series % n_cols)
    row_idx = np.repeat(range(n_rows), n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=entities_sample)

    for i, entity_id in enumerate(entities_sample):
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

    # Set default kwargs for plotting if user did not provide these
    kwargs = _set_subplot_default_kwargs(kwargs=kwargs, n_rows=n_rows, n_cols=n_cols)

    # Tidy up the plot
    fig.update_layout(**kwargs)
    fig = _remove_legend_duplicates(fig)
    return fig


def plot_residuals(
    y_resids: Union[pl.DataFrame, pl.LazyFrame],
    n_bins: Optional[int] = None,
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
    scoring: Optional[METRIC_TYPE] = None,
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
    scoring: Optional[METRIC_TYPE] = None,
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
