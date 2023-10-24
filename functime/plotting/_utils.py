"""Utilities for plotting."""

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import List, Literal, Optional, Tuple, Union, overload

    import plotly.graph_objects as go


def _remove_legend_duplicates(fig: go.Figure) -> go.Figure:
    names = set()
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )
    return fig


@overload
def _sample_entities(
    y: Union[pl.DataFrame, pl.LazyFrame],
    *,
    k: int = 6,
    seed: Optional[int] = None,
    return_entities: Literal[True],
) -> Tuple[pl.LazyFrame, List[str]]:
    ...


@overload
def _sample_entities(
    y: Union[pl.DataFrame, pl.LazyFrame],
    *,
    k: int = 6,
    seed: Optional[int] = None,
    return_entities: Literal[False],
) -> pl.LazyFrame:
    ...


def _sample_entities(
    y: Union[pl.DataFrame, pl.LazyFrame],
    *,
    k: int = 6,
    seed: Optional[int] = None,
    return_entities: bool = False,
) -> Union[pl.LazyFrame, Tuple[pl.LazyFrame, List[str]]]:
    """Sample k entities from panel DataFrame `y`."""
    if isinstance(y, pl.DataFrame):
        y = y.lazy()

    entity_col = y.columns[0]

    entities = (
        y.select(pl.col(entity_col).unique().sample(k, seed=seed)).collect().to_series()
    )

    if return_entities:
        return y.filter(pl.col(entity_col).is_in(entities)), sorted(entities)

    return y.filter(pl.col(entity_col).is_in(entities))


def _get_entities(
    y: Union[pl.DataFrame, pl.LazyFrame],
    *,
    ids: str | Collection[str],
) -> pl.LazyFrame:
    """Get entities from panel DataFrame `y`."""
    if isinstance(y, pl.DataFrame):
        y = y.lazy()

    entity_col = y.columns[0]

    if isinstance(ids, str):
        return y.filter(entity_col == ids)
    return y.filter(pl.col(entity_col).is_in(ids))
