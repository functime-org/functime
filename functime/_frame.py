from __future__ import annotations

from typing import Any

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
import polars as pl
from narwhals.typing import IntoFrame
from typing_extensions import TypeIs


def to_polars_lazy(data: Any) -> pl.LazyFrame:
    if isinstance(data, pl.LazyFrame):
        return data
    if isinstance(data, pl.DataFrame):
        return data.lazy()
    try:
        nw_frame = nw.from_native(data)
    except TypeError as exc:
        raise TypeError(
            f"Expected a DataFrame container; got {type(data).__name__}."
        ) from exc
    if isinstance(nw_frame, nw.LazyFrame):
        if nw_frame.implementation is nw.Implementation.POLARS:
            return nw_frame.to_native()
        return nw_frame.collect().to_polars().lazy()
    return nw_frame.to_polars().lazy()


def is_frame_like(data: Any) -> TypeIs[IntoFrame]:
    return nwd.is_into_dataframe(data) or nwd.is_into_lazyframe(data)


def coerce_target_series(y: Any, n_rows: int, name: str = "target") -> pl.Series:
    if isinstance(y, pl.Series):
        series = y.rename(name)
    elif nwd.is_into_series(y):
        series = nw.from_native(y, series_only=True).to_polars().rename(name)
    elif is_frame_like(y):
        df = to_polars_lazy(y).collect()
        if df.width != 1:
            raise ValueError(
                f"Expected a 1-column DataFrame for y; got {df.width} columns."
            )
        series = df.get_column(df.columns[0]).rename(name)
    else:
        arr = np.asarray(y)
        if arr.ndim != 1:
            raise ValueError(f"Expected a 1D target array; got shape {arr.shape}.")
        series = pl.Series(name, arr)

    if len(series) != n_rows:
        raise ValueError(
            f"Target length ({len(series)}) does not match X row count ({n_rows})."
        )
    return series
