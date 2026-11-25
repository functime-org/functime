from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from functime._core._narwhals import _to_polars_df, _validate_temporal_columns


def _unique_sorted_timestamps(df: pl.DataFrame, time_col: str) -> pl.Series:
    return df.get_column(time_col).unique().sort()


def temporal_train_test_split(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    test_size: int | float = 0.25,
    gap: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a univariate or panel DataFrame into chronological train and test sets.

    Splits across unique sorted timestamps so every entity shares the same
    time cutoff and no future observations leak into the training set.

    Parameters
    ----------
    data : DataFrame-like
        Polars, Pandas, Arrow, or Narwhals DataFrame.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional entity identifier column(s) defining panel groups.
    test_size : int or float, default=0.25
        Number of trailing unique timestamps (``int >= 1``) or fraction
        (``0 < float < 1``) reserved for the test split.
    gap : int, default=0
        Number of unique timestamps omitted between train end and test start.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        ``(train_df, test_df)`` preserving relative row ordering within each split.
    """
    df = _to_polars_df(data)
    _validate_temporal_columns(df, time_col=time_col, entity_col=entity_col)
    if gap < 0:
        raise ValueError(f"`gap` must be >= 0, got {gap}.")

    unique_times = _unique_sorted_timestamps(df, time_col)
    n_times = len(unique_times)
    if isinstance(test_size, float):
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"Float `test_size` must lie in (0, 1), got {test_size}.")
        n_test = max(1, int(np.ceil(n_times * test_size)))
    elif isinstance(test_size, int):
        if test_size < 1:
            raise ValueError(f"Integer `test_size` must be >= 1, got {test_size}.")
        n_test = test_size
    else:
        raise TypeError(
            f"`test_size` must be int or float, got {type(test_size).__name__}."
        )

    n_train = n_times - gap - n_test
    if n_train < 1:
        raise ValueError(
            f"Not enough unique timestamps ({n_times}) for test_size={n_test} and gap={gap}."
        )

    train_times = unique_times.slice(0, n_train).implode()
    test_times = unique_times.slice(n_train + gap, n_test).implode()
    return (
        df.filter(pl.col(time_col).is_in(train_times)),
        df.filter(pl.col(time_col).is_in(test_times)),
    )


def expanding_window_indices(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    n_splits: int = 5,
    test_size: int = 1,
    step_size: int = 1,
    gap: int = 0,
    max_train_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute expanding-window positional train/test index pairs ordered by time.

    Parameters
    ----------
    data : DataFrame-like
        Polars, Pandas, Arrow, or Narwhals DataFrame.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional entity identifier column(s) defining panel groups.
    n_splits : int, default=5
        Number of cross-validation folds.
    test_size : int, default=1
        Number of unique timestamps in each test fold.
    step_size : int, default=1
        Number of unique timestamps advancing between consecutive test folds.
    gap : int, default=0
        Number of timestamps skipped between the end of train and start of test.
    max_train_size : int or None, default=None
        Optional cap on trailing unique timestamps included in each training fold.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Sequence of ``(train_idx, test_idx)`` 1D integer NumPy arrays.
    """
    return _window_indices(
        data=data,
        time_col=time_col,
        entity_col=entity_col,
        n_splits=n_splits,
        test_size=test_size,
        step_size=step_size,
        gap=gap,
        max_train_size=max_train_size,
    )


def sliding_window_indices(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    n_splits: int = 5,
    test_size: int = 1,
    step_size: int = 1,
    gap: int = 0,
    window_size: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute fixed-width sliding-window positional train/test index pairs.

    Parameters
    ----------
    data : DataFrame-like
        Polars, Pandas, Arrow, or Narwhals DataFrame.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional entity identifier column(s) defining panel groups.
    n_splits : int, default=5
        Number of cross-validation folds.
    test_size : int, default=1
        Number of unique timestamps in each test fold.
    step_size : int, default=1
        Number of unique timestamps advancing between consecutive test folds.
    gap : int, default=0
        Number of timestamps skipped between the end of train and start of test.
    window_size : int, default=10
        Fixed number of trailing unique timestamps included in each training fold.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Sequence of ``(train_idx, test_idx)`` 1D integer NumPy arrays.
    """
    if window_size < 1:
        raise ValueError(f"`window_size` must be >= 1, got {window_size}.")
    return _window_indices(
        data=data,
        time_col=time_col,
        entity_col=entity_col,
        n_splits=n_splits,
        test_size=test_size,
        step_size=step_size,
        gap=gap,
        max_train_size=window_size,
    )


def _window_indices(
    *,
    data: Any,
    time_col: str,
    entity_col: str | Sequence[str] | None,
    n_splits: int,
    test_size: int,
    step_size: int,
    gap: int,
    max_train_size: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 1:
        raise ValueError(f"`n_splits` must be >= 1, got {n_splits}.")
    if test_size < 1:
        raise ValueError(f"`test_size` must be >= 1, got {test_size}.")
    if step_size < 1:
        raise ValueError(f"`step_size` must be >= 1, got {step_size}.")
    if gap < 0:
        raise ValueError(f"`gap` must be >= 0, got {gap}.")
    if max_train_size is not None and max_train_size < 1:
        raise ValueError(f"`max_train_size` must be >= 1, got {max_train_size}.")

    df = _to_polars_df(data)
    _validate_temporal_columns(df, time_col=time_col, entity_col=entity_col)

    unique_times = _unique_sorted_timestamps(df, time_col)
    n_times = len(unique_times)
    required_min = (n_splits - 1) * step_size + test_size + gap + 1
    if n_times < required_min:
        raise ValueError(
            f"Not enough unique timestamps ({n_times}) for n_splits={n_splits}, "
            f"test_size={test_size}, step_size={step_size}, gap={gap} "
            f"(requires at least {required_min})."
        )

    indexed = df.select(pl.col(time_col)).with_row_index("__pos")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        test_end = n_times - (n_splits - 1 - k) * step_size
        test_start = test_end - test_size
        train_end = test_start - gap
        train_start = (
            0 if max_train_size is None else max(0, train_end - max_train_size)
        )

        train_times = unique_times.slice(train_start, train_end - train_start).implode()
        test_times = unique_times.slice(test_start, test_end - test_start).implode()

        train_idx = (
            indexed.filter(pl.col(time_col).is_in(train_times))
            .get_column("__pos")
            .to_numpy()
        )
        test_idx = (
            indexed.filter(pl.col(time_col).is_in(test_times))
            .get_column("__pos")
            .to_numpy()
        )
        folds.append((train_idx.astype(np.intp), test_idx.astype(np.intp)))
    return folds
