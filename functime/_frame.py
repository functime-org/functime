from __future__ import annotations

from typing import Any

# Use narwhals.stable.v2 for API stability across versions, while keeping
# narwhals.dependencies/typing so top-level narwhals and v1/v2 containers are all accepted.
import narwhals.dependencies as nwd
import narwhals.stable.v2 as nw
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
        # Non-Polars lazy backends (e.g. DuckDB, Dask, Ibis) must materialize;
        # backend="polars" avoids intermediate PyArrow/Pandas conversions where supported.
        return nw_frame.collect(backend="polars").to_polars().lazy()
    _validate_no_meaningful_index(nw_frame)
    return nw_frame.to_polars().lazy()


def is_frame_like(data: Any) -> TypeIs[IntoFrame]:
    return nwd.is_into_dataframe(data) or nwd.is_into_lazyframe(data)


def coerce_target_series(y: Any, n_rows: int, name: str = "target") -> pl.Series:
    if isinstance(y, pl.Series):
        series = y.rename(name)
    elif nwd.is_into_series(y):
        nw_series = nw.from_native(y, series_only=True)
        _validate_no_meaningful_index(nw_series)
        series = nw_series.to_polars().rename(name)
    elif is_frame_like(y):
        lf = to_polars_lazy(y)
        width = lf.collect_schema().len()
        if width != 1:
            raise ValueError(
                f"Expected a 1-column DataFrame for y; got {width} columns."
            )
        series = lf.collect().to_series().rename(name)
    else:
        arr = np.asarray(y)
        if arr.ndim != 1:
            # sklearn (n, 1) DataConversionWarning/coercion belongs in functime.sklearn via column_or_1d(..., warn=True).
            raise ValueError(f"Expected a 1D target array; got shape {arr.shape}.")
        series = pl.Series(name, arr)

    if series.dtype == pl.Object:
        # Object columns pass construction but fail later inside most Polars expressions.
        raise TypeError(
            "Target has an unsupported object dtype; cast y to a numeric or string dtype first."
        )
    if len(series) != n_rows:
        raise ValueError(
            f"Target length ({len(series)}) does not match X row count ({n_rows})."
        )
    return series


def _validate_no_meaningful_index(nw_obj: nw.DataFrame[Any] | nw.Series[Any]) -> None:
    idx = nw.maybe_get_index(nw_obj)
    if idx is None:
        return
    nlevels = getattr(idx, "nlevels", 1)
    name = getattr(idx, "name", None)
    dtype_kind = getattr(getattr(idx, "dtype", None), "kind", None)
    if nlevels > 1 or name is not None or dtype_kind not in ("i", "u"):
        names = getattr(idx, "names", None)
        detail = f" (names={list(names)!r})" if names is not None else ""
        raise ValueError(
            f"Pandas-like {type(nw_obj.to_native()).__name__} has a meaningful "
            f"{type(idx).__name__}{detail} "
            "that would be dropped on conversion. Call `df.reset_index()` before "
            "passing it to functime."
        )
