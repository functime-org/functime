from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import narwhals as nw
import numpy as np
import polars as pl


def _is_dataframe_like(data: Any) -> bool:
    if isinstance(data, pl.DataFrame | pl.LazyFrame | nw.DataFrame | nw.LazyFrame):
        return True
    try:
        native = nw.from_native(data, pass_through=True)
    except TypeError:
        return False
    return isinstance(native, nw.DataFrame | nw.LazyFrame)


def _to_polars_df(data: Any) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    if isinstance(data, nw.DataFrame):
        return data.to_polars()
    if isinstance(data, nw.LazyFrame):
        return data.collect().to_polars()
    try:
        native = nw.from_native(data, pass_through=True)
    except TypeError as exc:
        raise TypeError(
            f"Expected a Polars, Pandas, Arrow, or Narwhals DataFrame; got {type(data).__name__}."
        ) from exc
    if isinstance(native, nw.LazyFrame):
        return native.collect().to_polars()
    if isinstance(native, nw.DataFrame):
        return native.to_polars()
    raise TypeError(f"Expected a DataFrame container; got {type(data).__name__}.")


def _normalize_entity_cols(entity_col: str | Sequence[str] | None) -> list[str]:
    if entity_col is None:
        return []
    if isinstance(entity_col, str):
        return [entity_col]
    return list(entity_col)


def _validate_temporal_columns(
    df: pl.DataFrame,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    entity_cols = _normalize_entity_cols(entity_col)
    missing = [col for col in [*entity_cols, time_col] if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required temporal/entity column(s) {missing} in DataFrame columns {df.columns}."
        )
    if df.height == 0:
        raise ValueError("Input DataFrame has 0 rows; at least 1 sample is required.")
    index_set = set(entity_cols) | {time_col}
    if columns is None:
        target_cols = [col for col in df.columns if col not in index_set]
    else:
        unknown = [col for col in columns if col not in df.columns]
        if unknown:
            raise ValueError(
                f"Specified column(s) {unknown} not found in DataFrame columns {df.columns}."
            )
        target_cols = list(columns)
    return entity_cols, target_cols


def _coerce_target_series(y: Any, n_rows: int, name: str = "target") -> pl.Series:
    if isinstance(y, pl.Series):
        series = y.rename(name)
    elif isinstance(y, nw.Series):
        series = y.to_polars().rename(name)
    elif isinstance(y, np.ndarray):
        if y.ndim != 1:
            raise ValueError(f"Expected a 1D target array; got shape {y.shape}.")
        series = pl.Series(name, y)
    elif isinstance(y, pl.DataFrame):
        if y.width != 1:
            raise ValueError(
                f"Expected a 1-column DataFrame for y; got {y.width} columns."
            )
        series = y.get_column(y.columns[0]).rename(name)
    else:
        try:
            native = nw.from_native(y, pass_through=True, series_only=True)
        except TypeError:
            native = y
        if isinstance(native, nw.Series):
            series = native.to_polars().rename(name)
        else:
            arr = np.asarray(native)
            if arr.ndim != 1:
                raise ValueError(
                    f"Expected a 1D target sequence; got shape {arr.shape}."
                )
            series = pl.Series(name, arr)

    if len(series) != n_rows:
        raise ValueError(
            f"Target length ({len(series)}) does not match X row count ({n_rows})."
        )
    return series


def _format_output(
    df: pl.DataFrame,
    output_format: str = "default",
    original: Any = None,
) -> Any:
    if output_format in ("default", "polars"):
        return df
    if output_format == "pandas":
        pdf = df.to_pandas()
        if original is not None and hasattr(original, "index"):
            orig_index = original.index
            if len(orig_index) == len(pdf):
                pdf.index = orig_index
        return pdf
    raise ValueError(
        f"Unsupported output container {output_format!r}. Expected 'default', 'polars', or 'pandas'."
    )
