"""TimeSeriesDisplay class to draw panel datasets plots."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import (
        ClassVar,
        TypedDict,
    )

    from typing_extensions import Self

    class DefaultPalette(TypedDict):
        """Alias of functime deafult color palette.

        `forecast` is aliased to `primary` and `actual` is aliased to `secondary`.
        """

        primary: str
        secondary: str
        actual: str
        forecast: str
        backtest: str


class TimeSeriesDisplay:
    DEFAULT_PALETTE: ClassVar[DefaultPalette] = {
        "primary": "#1b57f1",
        "forecast": "#1b57f1",
        "actual": "#B7B7B7",
        "secondary": "#A76EF4",
        "backtest": "#A76EF4",
    }

    def __init__(
        self,
        *,
        entities: Collection[str],
        num_cols: int,
        num_series: int,
        num_rows: int,
        default_title: str,
        **kwargs,
    ):
        """Initialize a time series display.

        The initialisation defines a `plotly.graphic_objects.Figure` figure with subplots.

        Parameters
        ----------
        entities : Collection[str]
            Entities to plot in the subplot grid.
        num_cols : int
            Number of columns in the subplot grid.
        num_series : int
            Number of series in the subplot grid.
        num_rows : int
            Number of rows in the subplot grid.
        default_title : str
            Default title for the figure.
        kwargs
            Additional keyword arguments to pass to `plotly.graph_objects.Layout` object.


        """

        self.num_cols = num_cols
        self.num_series = num_series
        self.num_rows = num_rows
        self.entities = entities

        title_space = 100
        height = kwargs.pop("height", num_rows * 200 + title_space)
        template = kwargs.pop("template", "plotly_white")
        title = kwargs.pop("title", default_title)

        layout = go.Layout(
            height=height,
            title=title,
            template=template,
            **kwargs,
        )

        self.figure = make_subplots(
            figure=go.Figure(layout=layout),
            rows=self.num_rows,
            cols=self.num_cols,
            subplot_titles=self.entities,
        )

    @classmethod
    def from_panel(
        cls,
        *,
        y: pl.LazyFrame,
        num_cols: int | None = None,
        num_series: int | None = None,
        seed: int | None = None,
        default_title: str,
        **kwargs,
    ):
        """Initialize a time series display from a Panel LazyFrame.

        The initialisation defines a `plotly.graphic_objects.Figure` figure with the given
        number of columns and rows, and the entities in the data.

        Parameters
        ----------
        y : pl.LazyFrame
            Panel LazyFrame time series data.
        num_cols : Optional[int]
            Number of columns in the subplot grid. Defaults to 2.
        num_series : Optional[int]
            Number of series in the subplot grid. If `None`, displays all series.
            Defaults to `None`.
        seed : Optional[int]
            Seed for the random sample of entities to plot. Defaults to `None`.
        default_title : str
            Default title for the figure.
        **kwargs
            Additional keyword arguments to pass to `plotly.graph_objects.Figure` object.

        Returns
        -------
        self : Self
            Instance of `TimeSeriesDisplay`.

        Raises
        ------
        ValueError
            If any of `num_cols` or `num_series` is not a positive integer.
        """
        if num_cols is not None and num_cols < 1:
            raise ValueError("Number of columns must be a positive integer")
        if num_series is not None and num_series < 1:
            raise ValueError("Number of series must be a positive integer")

        n_cols = num_cols or 2

        sample_entities = get_chosen_entities(
            y=y,
            num_series=num_series,
            seed=seed,
        )

        n_series = len(sample_entities)

        n_rows = get_num_rows(
            num_series=n_series,
            num_cols=n_cols,
        )

        return cls(
            entities=sample_entities,
            num_cols=n_cols,
            num_series=n_series,
            num_rows=n_rows,
            default_title=default_title,
            **kwargs,
        )

    def add_time_series(
        self: Self,
        *,
        data: pl.LazyFrame,
        num_points: int | None = None,
        name_on_hover: str | None = None,
        legend_group: str | None = None,
        **kwargs,
    ) -> Self:
        """Add a time series to the subplot grid.

        Parameters
        ----------
        data : pl.LazyFrame
            Panel LazyFrame time series data.
        num_points : Optional[int]
            Number of data points to plot. If `None`, plot all points.
            Defaults to `None`
        name_on_hover : Optional[str]
            Text that will be displayed on hover. Defaults to the name of the target column.
        legend_group : Optional[str]
            Legend group the trace belongs to. Defaults to `None`.
        **kwargs
            Additional keyword arguments to pass to `plotly.graph_objects.Line` object.

        Returns
        -------
        self : Self
            Instance of `TimeSeriesDisplay`.

        Raises
        ------
        ValueError
            If `num_points` is not a positive integer.
        """

        if num_points is not None and num_points < 0:
            raise ValueError("Number of points must be a positive integer")

        entity_col = data.columns[0]

        num_entities = data.select(pl.col(entity_col).n_unique()).collect().item()

        if len(self.entities) == num_entities:
            y = data
        else:
            y = data.filter(pl.col(entity_col).is_in(self.entities))

        if num_points is not None:
            y = y.group_by(entity_col).tail(num_points)

        self.figure = add_traces(
            figure=self.figure,
            y=y.collect(),
            name_on_hover=name_on_hover,
            legend_group=legend_group,
            num_cols=self.num_cols,
            entities=self.entities,
            **kwargs,
        )

        return self


def add_traces(
    *,
    figure: go.Figure,
    y: pl.DataFrame,
    entities: Collection[str],
    num_cols: int,
    show_legend: bool = True,
    name_on_hover: str | None = None,
    legend_group: str | None = None,
    **kwargs,
) -> go.Figure:
    """Add scatterplot traces to a `Figure` instance.

    The function needs to know the number of columns in the subplot grid to
    place a trace in the correct position.

    Parameters
    ----------
    figure : go.Figure
        Plotly figure to add traces to.
    y : pl.DataFrame
        Panel DataFrame.
    entities : Collection[str]
        Entities to plot in the subplot grid.
    num_cols : int
        Number of columns in the subplot grid.
    show_legend : bool
        Whether to show the legend for the trace. Defaults to `True`.
    name_on_hover : Optional[str]
        Text that will be displayed on hover. Defaults to the name of the target column, in title case.
    legend_group : Optional[str]
        Legend group the trace belongs to. Defaults to `None`.
    **kwargs
        Additional keyword arguments to pass to `plotly.graph_objects.Line` object.

    Returns
    -------
    figure : go.Figure
        Updated Plotly figure.
    """
    entity_col, time_col, target_col = y.columns[:3]

    for i, entity in enumerate(entities):
        ts = y.filter(pl.col(entity_col) == entity)

        row, col = get_subplot_grid_position(element=i, num_cols=num_cols)

        figure.add_trace(
            go.Scatter(
                x=ts.get_column(time_col),
                y=ts.get_column(target_col),
                name=name_on_hover or target_col.title(),
                legendgroup=legend_group,
                line=kwargs,
                showlegend=show_legend if i == 0 else False,
                mode="lines",
            ),
            row=row,
            col=col,
        )

    return figure


def get_chosen_entities(
    *,
    y: pl.LazyFrame,
    num_series: int | None = None,
    seed: int | None = None,
):
    """Sample entities to plot in a subplot grid, given the data.

    If `seed` is `None`, it returns the first `num_series` entities in the data. Alternatively, it returns a random sample.

    Parameters
    ----------
    y : pl.LazyFrame
        Panel LazyFrame time series data.
    num_series : int
        Number of series to sample.
    seed : Optional[int]
        Seed for the random sample. Defaults to `None`, i.e. no sampling.

    Returns
    -------
    entities : pl.Series
        Series of sampled entities.

    Raises
    ------
    ValueError
        If `num_series` is bigger than the total number of entities in the data, or if `num_series` is less than 1.

    Example
    -------
    >>> get_chosen_entities(y=pl.DataFrame({"entity": ["a", "b", "c", "d", "e"], "value": [1, 2, 3, 4, 5]}), n_series=2)
    pl.Series(['a', 'b'], name='entity')
    >>> get_chosen_entities(y=pl.DataFrame({"entity": ["a", "b", "c", "d", "e"], "value": [1, 2, 3, 4, 5]}), n_series=0)
    pl.Series(['a', 'b', 'c', 'd', 'e'], name='entity')
    """
    if num_series is not None and num_series <= 0:
        raise ValueError("Number of series must be a positive integer")

    entity_col = y.columns[0]

    num_entities = y.select(pl.col(entity_col).n_unique()).collect().item()

    if num_series is not None and num_series > num_entities:
        raise ValueError(
            f"Number of series ({num_series}) is greater than the total number of entities ({num_entities})"
        )

    entities = (
        y.select(pl.col(entity_col).unique(maintain_order=True))
        .collect()
        .to_series()
        .sort()
    )

    if num_series is None or num_series == num_entities:
        if seed is not None:
            logging.info(
                "`num_series` is equal to the total number of entities, while `seed` is set."
                "There will be no random sampling."
            )
        return entities.to_list()

    if seed is not None:
        return entities.sample(num_series, seed=seed).to_list()
    return entities.slice(0, num_series).to_list()


def get_num_rows(
    *,
    num_series: int,
    num_cols: int,
) -> int:
    """Get the number of rows in a subplot grid, given the number of series and number of columns.

    Parameters
    ----------
    num_series : int
        Number of series in the subplot grid.
    num_cols : int
        Number of columns in the subplot grid.

    Returns
    -------
    int
        Number of rows in the subplot grid.

    Example
    -------
    >>> get_num_rows(num_series=10, num_cols=2)
    5
    >>> get_num_rows(num_series=5, num_cols=2)
    3
    """
    if any([num_series < 1, num_cols < 1]):
        raise ValueError("Number of series and columns must be a positive integer")

    num_rows, remainder = divmod(num_series, num_cols)
    if remainder != 0:
        return num_rows + 1
    return num_rows


def get_subplot_grid_position(
    *,
    element: int,
    num_cols: int,
) -> tuple[int, int]:
    """Get the row and column index of the subplot at the given element index.

    Need to add 1 because the grid indexes in a plotly subplot are 1-based.

    Parameters
    ----------
    element : int
        Element index in the subplot grid.
    num_cols : int
        Number of columns in the subplot grid.

    Returns
    -------
    Tuple[int, int]
        Row and column index of the subplot.

    Example
    -------
    >>> get_subplot_grid_position(element=0, num_cols=2)
    (1, 1)
    >>> get_subplot_grid_position(element=1, num_cols=2)
    (1, 2)
    >>> get_subplot_grid_position(element=2, num_cols=2)
    (2, 1)
    >>> get_subplot_grid_position(element=3, num_cols=2)
    (2, 2)
    """
    row_index, col_index = divmod(element, num_cols)
    return row_index + 1, col_index + 1
