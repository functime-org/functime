from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Literal, cast

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
import polars as pl
from narwhals.typing import IntoFrame
from typing_extensions import TypeIs

type HandleUnknown = Literal["global", "identity", "null", "error"]


class SortState(IntEnum):
    UNKNOWN = 0
    CHRONOLOGICAL = 1
    ENTITY_TIME_SORTED = 2


@dataclass(frozen=True, slots=True)
class TemporalFrame:
    lf: pl.LazyFrame
    time_col: str
    entity_cols: tuple[str, ...]
    target_cols: tuple[str, ...]
    sort_state: SortState = SortState.UNKNOWN

    @property
    def is_panel(self) -> bool:
        return bool(self.entity_cols)

    @property
    def columns(self) -> list[str]:
        return self.lf.collect_schema().names()

    @classmethod
    def parse(
        cls,
        data: TemporalFrame | Any,
        *,
        time_col: str,
        entity_col: str | Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        allow_null_time: bool = False,
        allow_null_entity: bool = False,
        assume_sorted: bool = False,
    ) -> TemporalFrame:
        entity_cols = validate_temporal_params(time_col, entity_col)
        if isinstance(data, cls):
            target_cols = _validate_schema_and_targets(
                data.lf,
                time_col=time_col,
                entity_cols=entity_cols,
                columns=columns,
                allow_null_time=allow_null_time,
                allow_null_entity=allow_null_entity,
            )
            same_axes = data.time_col == time_col and data.entity_cols == entity_cols
            base_sort = data.sort_state if same_axes else SortState.UNKNOWN
            sort_state = (
                max(base_sort, SortState.CHRONOLOGICAL) if assume_sorted else base_sort
            )
            return cls(
                lf=data.lf,
                time_col=time_col,
                entity_cols=entity_cols,
                target_cols=target_cols,
                sort_state=sort_state,
            )

        if isinstance(data, pl.DataFrame):
            target_cols = _validate_schema_and_targets(
                data,
                time_col=time_col,
                entity_cols=entity_cols,
                columns=columns,
                allow_null_time=allow_null_time,
                allow_null_entity=allow_null_entity,
            )
            # Capture O(1) flags and contiguous single-entity order before wrapping as LazyFrame.
            check_scan = (
                len(entity_cols) == 1
                and data.get_column(entity_cols[0]).flags["SORTED_ASC"]
            )
            base_sort = detect_sort_state(
                data, entity_cols, time_col, check_scan=check_scan
            )
            lf = data.lazy()
        else:
            lf = to_polars_lazy(data)
            target_cols = _validate_schema_and_targets(
                lf,
                time_col=time_col,
                entity_cols=entity_cols,
                columns=columns,
                allow_null_time=allow_null_time,
                allow_null_entity=allow_null_entity,
            )
            base_sort = SortState.UNKNOWN

        sort_state = (
            max(base_sort, SortState.CHRONOLOGICAL) if assume_sorted else base_sort
        )
        return cls(
            lf=lf,
            time_col=time_col,
            entity_cols=entity_cols,
            target_cols=target_cols,
            sort_state=sort_state,
        )

    def over(self, expr: pl.Expr, *, chronological: bool = False) -> pl.Expr:
        if not chronological or self.sort_state >= SortState.CHRONOLOGICAL:
            return expr.over(self.entity_cols) if self.entity_cols else expr
        if self.entity_cols:
            return expr.over(self.entity_cols, order_by=self.time_col)
        return expr.over(pl.lit(0), order_by=self.time_col)

    def with_exprs(
        self, exprs: Sequence[pl.Expr], *, chronological: bool = False
    ) -> TemporalFrame:
        wrapped = [self.over(expr, chronological=chronological) for expr in exprs]
        return replace(self, lf=self.lf.with_columns(wrapped))

    def with_lazy(
        self, lf: pl.LazyFrame, *, preserves_order: bool = True
    ) -> TemporalFrame:
        return replace(
            self,
            lf=lf,
            sort_state=self.sort_state if preserves_order else SortState.UNKNOWN,
        )

    def sort_entity_time(self) -> TemporalFrame:
        if self.sort_state >= SortState.ENTITY_TIME_SORTED:
            return self
        return replace(
            self,
            lf=sort_entity_time(self.lf, self.entity_cols, self.time_col),
            sort_state=SortState.ENTITY_TIME_SORTED,
        )

    def group_agg(
        self,
        exprs: Sequence[pl.Expr],
        *,
        chronological: bool = False,
        maintain_order: bool = False,
    ) -> pl.LazyFrame:
        target = (
            self.sort_entity_time()
            if chronological and self.sort_state < SortState.CHRONOLOGICAL
            else self
        )
        if not target.entity_cols:
            return target.lf.select(exprs)
        return target.lf.group_by(
            target.entity_cols, maintain_order=maintain_order
        ).agg(exprs)

    def join_state(
        self,
        stats_df: pl.DataFrame,
        *,
        handle_unknown: HandleUnknown = "global",
    ) -> TemporalFrame:
        if handle_unknown not in _VALID_HANDLE_UNKNOWN:
            raise ValueError(
                f"Invalid `handle_unknown`={handle_unknown!r}. Expected one of {_VALID_HANDLE_UNKNOWN}."
            )
        if not self.entity_cols:
            if stats_df.height != 1:
                raise ValueError(
                    f"Expected 1-row state DataFrame for univariate input; got {stats_df.height} rows."
                )
            if stats_df.width == 0:
                return self
            out = self.lf.with_columns(
                [
                    pl.lit(stats_df.get_column(col).item(0))
                    .cast(stats_df.schema[col])
                    .alias(col)
                    for col in stats_df.columns
                ]
            )
            return replace(self, lf=out)

        left_schema = self.lf.collect_schema()
        left_names = list(left_schema.names())
        right_names = list(stats_df.columns)
        left = self.lf
        right = stats_df.lazy()

        join_keys: list[str] = []
        temp_keys: list[str] = []
        for col in self.entity_cols:
            left_dtype = left_schema[col]
            right_dtype = stats_df.schema[col]
            if left_dtype == right_dtype:
                join_keys.append(col)
            elif _is_string_like(left_dtype) and _is_string_like(right_dtype):
                # Join via temporary String keys so the input entity column preserves its original dtype.
                tmp = nw.generate_temporary_column_name(
                    n_bytes=8,
                    columns=[*left_names, *right_names, *join_keys, *temp_keys],
                    prefix=f"__ft_join_{col}_",
                )
                left = left.with_columns(pl.col(col).cast(pl.String).alias(tmp))
                right = right.with_columns(pl.col(col).cast(pl.String).alias(tmp)).drop(
                    col
                )
                join_keys.append(tmp)
                temp_keys.append(tmp)
            elif left_dtype.is_integer() and right_dtype.is_integer():
                right = right.with_columns(pl.col(col).cast(left_dtype, strict=False))
                join_keys.append(col)
            else:
                raise TypeError(
                    f"Incompatible dtypes for entity column {col!r}: input has {left_dtype}, state has {right_dtype}."
                )

        if handle_unknown == "error":
            unseen = (
                left.select(join_keys)
                .unique()
                .join(right.select(join_keys), on=join_keys, how="anti")
                .collect()
            )
            if unseen.height > 0:
                sample = unseen.head(5).rows()
                raise ValueError(
                    f"Found {unseen.height} unseen entity key(s) not present in fitted state (showing up to 5): {sample}."
                )

        joined = left.join(right, on=join_keys, how="left", maintain_order="left")
        if temp_keys:
            joined = joined.drop(temp_keys)
        return replace(self, lf=joined)

    def collect(self) -> pl.DataFrame:
        return self.lf.collect()


def temporal_transform(
    *,
    chronological: bool = False,
    needs_time_col: bool = False,
    allow_null_time: bool = False,
    allow_null_entity: bool = False,
) -> Callable[
    [Callable[..., Sequence[pl.Expr]]],
    Callable[..., pl.LazyFrame],
]:
    def decorator(
        expr_fn: Callable[..., Sequence[pl.Expr]],
    ) -> Callable[..., pl.LazyFrame]:
        @functools.wraps(expr_fn)
        def wrapper(
            data: TemporalFrame | Any,
            *,
            time_col: str = "time",
            entity_col: str | Sequence[str] | None = None,
            columns: Sequence[str] | None = None,
            assume_sorted: bool = False,
            **kwargs: Any,
        ) -> pl.LazyFrame:
            tf = TemporalFrame.parse(
                data,
                time_col=time_col,
                entity_col=entity_col,
                columns=columns,
                allow_null_time=allow_null_time,
                allow_null_entity=allow_null_entity,
                assume_sorted=assume_sorted,
            )
            exprs = (
                expr_fn(tf.target_cols, time_col=tf.time_col, **kwargs)
                if needs_time_col
                else expr_fn(tf.target_cols, **kwargs)
            )
            return tf.with_exprs(exprs, chronological=chronological).lf

        cast(Any, wrapper).expr_fn = expr_fn
        return wrapper

    return decorator


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


def detect_sort_state(
    df: pl.DataFrame,
    entity_cols: tuple[str, ...],
    time_col: str,
    *,
    check_scan: bool = False,
) -> SortState:
    if df.height <= 1:
        return SortState.ENTITY_TIME_SORTED
    if df.get_column(time_col).flags["SORTED_ASC"]:
        # Univariate sorted series and [time, entity]-sorted panels share time SORTED_ASC.
        return (
            SortState.ENTITY_TIME_SORTED if not entity_cols else SortState.CHRONOLOGICAL
        )
    if not check_scan:
        return SortState.UNKNOWN
    if not entity_cols:
        return (
            SortState.ENTITY_TIME_SORTED
            if df.get_column(time_col).is_sorted()
            else SortState.UNKNOWN
        )
    if len(entity_cols) == 1 and df.get_column(entity_cols[0]).flags["SORTED_ASC"]:
        # Contiguous entity blocks let a single shift validate intra-entity monotonicity without hashing.
        entity = entity_cols[0]
        ok = bool(
            df.select(
                (
                    (pl.col(entity) != pl.col(entity).shift(1))
                    | (pl.col(time_col) >= pl.col(time_col).shift(1))
                )
                .fill_null(True)
                .all()
            ).item()
        )
        return SortState.ENTITY_TIME_SORTED if ok else SortState.UNKNOWN
    ok = bool(df.select(pl.col(time_col).is_sorted().over(entity_cols).all()).item())
    return SortState.CHRONOLOGICAL if ok else SortState.UNKNOWN


def sort_entity_time(
    lf: pl.LazyFrame, entity_cols: tuple[str, ...], time_col: str
) -> pl.LazyFrame:
    return lf.sort([*entity_cols, time_col], maintain_order=True)


def normalize_entity_cols(entity_col: str | Sequence[str] | None) -> tuple[str, ...]:
    if entity_col is None:
        return ()
    if isinstance(entity_col, str):
        if not entity_col:
            raise ValueError("`entity_col` cannot be an empty string.")
        return (entity_col,)
    if isinstance(entity_col, (bytes, bytearray)):
        raise TypeError(
            "`entity_col` must be str, a sequence of str, or None; got bytes."
        )
    if not isinstance(entity_col, Sequence):
        raise TypeError(
            f"`entity_col` must be str, a sequence of str, or None; got {type(entity_col).__name__}."
        )
    cols = tuple(entity_col)
    if not cols:
        raise ValueError(
            "`entity_col` sequence cannot be empty; pass None for univariate data."
        )
    if any(not isinstance(col, str) or not col for col in cols):
        raise TypeError("All entries in `entity_col` must be non-empty strings.")
    if len(set(cols)) != len(cols):
        raise ValueError(f"Duplicate column names in `entity_col`: {cols}.")
    return cols


def validate_temporal_params(
    time_col: str, entity_col: str | Sequence[str] | None = None
) -> tuple[str, ...]:
    if not isinstance(time_col, str) or not time_col:
        raise ValueError("`time_col` must be a non-empty string.")
    entity_cols = normalize_entity_cols(entity_col)
    if time_col in entity_cols:
        raise ValueError(
            f"`time_col` ({time_col!r}) cannot also appear in `entity_col`."
        )
    return entity_cols


def clean_float_expr(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Float64).fill_nan(None)


_VALID_HANDLE_UNKNOWN: tuple[str, ...] = ("global", "identity", "null", "error")


def _is_string_like(dtype: pl.DataType) -> bool:
    return (
        dtype == pl.String
        or dtype == pl.Categorical
        or isinstance(dtype, (pl.Categorical, pl.Enum))
    )


def _validate_schema_and_targets(
    raw_frame: pl.DataFrame | pl.LazyFrame,
    *,
    time_col: str,
    entity_cols: tuple[str, ...],
    columns: Sequence[str] | None,
    allow_null_time: bool,
    allow_null_entity: bool,
) -> tuple[str, ...]:
    col_names = (
        raw_frame.columns
        if isinstance(raw_frame, pl.DataFrame)
        else raw_frame.collect_schema().names()
    )
    missing = [col for col in (*entity_cols, time_col) if col not in col_names]
    if missing:
        raise ValueError(
            f"Missing required temporal/entity column(s) {missing} in DataFrame columns {col_names}."
        )
    if isinstance(raw_frame, pl.DataFrame):
        if raw_frame.height == 0:
            raise ValueError(
                "Input DataFrame has 0 rows; at least 1 sample is required."
            )
        if not allow_null_time and raw_frame.get_column(time_col).null_count() > 0:
            raise ValueError(f"`time_col` ({time_col!r}) contains null values.")
        if not allow_null_entity:
            for col in entity_cols:
                if raw_frame.get_column(col).null_count() > 0:
                    raise ValueError(f"`entity_col` ({col!r}) contains null values.")

    index_set = set(entity_cols) | {time_col}
    if columns is None:
        return tuple(col for col in col_names if col not in index_set)
    unknown = [col for col in columns if col not in col_names]
    if unknown:
        raise ValueError(
            f"Specified column(s) {unknown} not found in DataFrame columns {col_names}."
        )
    return tuple(columns)
