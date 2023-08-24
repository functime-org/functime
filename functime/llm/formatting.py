from typing import Mapping, Optional, Tuple, Union

import polars as pl
from typing_extensions import Literal

FORMAT_T = Literal["markdown_bullet_list", "freeform"]


def format_dataframes(data: Union[pl.DataFrame, Mapping[str, pl.DataFrame]]) -> str:
    """Convert 1 or more polars DataFrames to Markdown"""
    outer = "\n\n```\n{}\n```\n\n"
    if isinstance(data, pl.DataFrame):
        inner = f"{data.to_pandas().to_markdown(index=False)}"
    else:
        inner = "\n\n".join(
            f"### {key}\n{df.to_pandas().to_markdown(index=False)}"
            for key, df in data.items()
        )
    return outer.format(inner)


def format_instructions(format: FORMAT_T) -> Tuple[str, str]:
    if format == "markdown_bullet_list":
        return (
            "Analyze the following time series data in 8-10 bulletpoints.",
            "{{ Insert unordered Markdown list here }}",
        )
    if format == "freeform":
        return (
            "Analyze the following time series data.",
            "{{ Insert your response here }}",
        )
    raise ValueError(f"Invalid formatting option: {format}")


def univariate_panel_to_wide(
    panel_df: pl.DataFrame,
    *,
    entity_col: Optional[str] = None,
    time_col: Optional[str] = None,
    target_col: Optional[str] = None,
    drop_nulls: bool = False,
    shrink_dtype: bool = False,
) -> pl.DataFrame:
    if entity_col is None:
        entity_col = panel_df.columns[0]
    if time_col is None:
        time_col = panel_df.columns[1]
    if target_col is None:
        target_col = panel_df.columns[2]
    wide_df = panel_df.pivot(
        index=time_col,
        values=target_col,
        columns=entity_col,
        aggregate_function=None,
        sort_columns=True,
    ).sort(time_col)
    if drop_nulls:
        wide_df = wide_df.drop_nulls()
    if shrink_dtype:
        wide_df = wide_df.select(pl.all().shrink_dtype())
    return wide_df
