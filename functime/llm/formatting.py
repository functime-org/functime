from typing import Mapping, Tuple, Union

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
