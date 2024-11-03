from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from typing import (
        Literal,
        Mapping,
        Optional,
        Tuple,
        Union,
    )

    from functime.type_aliases import PolarsFrame

import numpy as np
import polars as pl

__all__ = [
    "train_test_split",
    "expanding_window_split",
    "sliding_window_split",
]


@overload
def train_test_split(
    test_size: Union[int, float] = ...,
    eager: Literal[True] = ...,
) -> Callable[[PolarsFrame], Tuple[pl.DataFrame, pl.DataFrame]]: ...


@overload
def train_test_split(
    test_size: Union[int, float] = ...,
    eager: Literal[False] = ...,
) -> Callable[[PolarsFrame], Tuple[pl.DataFrame, pl.DataFrame]]: ...


@overload
def train_test_split(
    test_size: Union[int, float] = ...,
    eager: bool = ...,
) -> Callable[
    [PolarsFrame],
    Union[Tuple[pl.DataFrame, pl.DataFrame], Tuple[pl.LazyFrame, pl.LazyFrame]],
]: ...


def train_test_split(
    test_size: Union[int, float] = 0.25,
    eager: bool = False,
) -> Callable[
    [PolarsFrame],
    Union[Tuple[pl.DataFrame, pl.DataFrame], Tuple[pl.LazyFrame, pl.LazyFrame]],
]:
    """Return a time-ordered train set and test set given `test_size`.

    Parameters
    ----------
    test_size : int | float, default=0.25
        Number or fraction of test samples.
    eager : bool, default=False
        If True, evaluate immediately and returns tuple of train-test `DataFrame`.

    Returns
    -------
    splitter : Union[EagerSplitter, LazySplitter]
        Function that takes a panel DataFrame, or LazyFrame, and returns:
        * A tuple of train / test LazyFrames, if `eager=False`.
        * A tuple of train / test DataFrames, if `eager=True`.
    """
    if isinstance(test_size, float):
        if test_size < 0 or test_size > 1:
            raise ValueError("`test_size` must be between 0 and 1")
    elif isinstance(test_size, int):
        if test_size < 0:
            raise ValueError("`test_size` must be greater than 0")
    else:
        raise TypeError("`test_size` must be int or float")

    def splitter(
        X: PolarsFrame,
    ) -> Union[Tuple[pl.DataFrame, pl.DataFrame], Tuple[pl.LazyFrame, pl.LazyFrame]]:
        """Split the data into train and test sets."""

        return _splitter_train_test(
            X=X,
            test_size=test_size,
            eager=eager,
        )

    return splitter


@overload
def _splitter_train_test(
    X: PolarsFrame,
    test_size: Union[int, float],
    eager: Literal[True] = ...,
) -> Tuple[pl.DataFrame, pl.DataFrame]: ...


@overload
def _splitter_train_test(
    X: PolarsFrame,
    test_size: Union[int, float],
    eager: Literal[False] = ...,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]: ...


@overload
def _splitter_train_test(
    X: PolarsFrame,
    test_size: Union[int, float],
    eager: bool = ...,
) -> Union[Tuple[pl.DataFrame, pl.DataFrame], Tuple[pl.LazyFrame, pl.LazyFrame]]: ...


def _splitter_train_test(
    X: PolarsFrame,
    test_size: Union[int, float],
    eager: bool = False,
) -> Union[Tuple[pl.DataFrame, pl.DataFrame], Tuple[pl.LazyFrame, pl.LazyFrame]]:
    if isinstance(X, pl.DataFrame):
        X = X.lazy()

    entity_col = X.columns[0]

    max_size = (
        X.group_by(entity_col).agg(pl.len()).select(pl.min("len")).collect().item()
    )

    if isinstance(test_size, int) and test_size > max_size:
        raise ValueError(
            "`test_size` must be less than the number of samples of the smallest entity"
        )

    train_length = (
        pl.len() - test_size
        if isinstance(test_size, int)
        else (pl.len() * (1 - test_size)).cast(int)
    )
    test_length = pl.len() - train_length

    train_split = (
        X.group_by(entity_col)
        .agg(pl.all().slice(offset=0, length=train_length))
        .explode(pl.all().exclude(entity_col))
    )
    test_split = (
        X.group_by(entity_col)
        .agg(pl.all().slice(offset=train_length, length=test_length))
        .explode(pl.all().exclude(entity_col))
    )
    if eager:
        train_split, test_split = pl.collect_all([train_split, test_split])
        return train_split, test_split
    return train_split, test_split


def expanding_window_split(
    test_size: int, n_splits: int = 5, step_size: int = 1, eager: bool = False
):
    """Return train/test splits using expanding window splitter.

    Split time series repeatedly into an growing training set and a fixed-size test set.
    For example, given `test_size = 3`, `n_splits = 5` and `step_size = 1`,
    the train `o`s and test `x`s folds can be visualized as:

    ```
    | o o o x x x - - - - |
    | o o o o x x x - - - |
    | o o o o o x x x - - |
    | o o o o o o x x x - |
    | o o o o o o o x x x |
    ```

    Parameters
    ----------
    test_size : int
        Number of test samples for each split.
    n_splits : int, default=5
        Number of splits.
    step_size : int, default=1
        Step size between windows.
    eager : bool, default=False
        If True return DataFrames. Otherwise, return LazyFrames.

    Returns
    -------
    splitter : Callable[pl.LazyFrame, Mapping[int, Tuple[pl.LazyFrame, pl.LazyFrame]]]
        Function that takes a panel LazyFrame and Dict of (train, test) splits, where
        the key represents the split number (1,2,...,n_splits) and the value is a tuple of LazyFrames.
    """

    def split(X: pl.LazyFrame) -> pl.LazyFrame:
        splits = _window_split(X, test_size, n_splits, step_size)
        if eager:
            splits = {i: pl.collect_all(s) for i, s in splits.items()}
        return splits

    return split


def sliding_window_split(
    test_size: int,
    n_splits: int = 5,
    step_size: int = 1,
    window_size: int = 10,
    eager: bool = False,
):
    """Return train/test splits using sliding window splitter.
    Split time series repeatedly into a fixed-length training and test set.
    For example, given `test_size = 3`, `n_splits = 5`, `step_size = 1` and `window_size=5`
    the train `o`s and test `x`s folds can be visualized as:

    ```
    | o o o o o x x x - - - - |
    | - o o o o o x x x - - - |
    | - - o o o o o x x x - - |
    | - - - o o o o o x x x - |
    | - - - - o o o o o x x x |
    ```

    Parameters
    ----------
    test_size : int
        Number of test samples for each split.
    n_splits : int, default=5
        Number of splits.
    step_size : int, default=1
        Step size between windows.
    window_size: int, default=10
        Window size for training.
    eager : bool, default=False
        If True return DataFrames. Otherwise, return LazyFrames.

    Returns
    -------
    splitter : Callable[pl.LazyFrame, Mapping[int, Tuple[pl.LazyFrame, pl.LazyFrame]]]
        Function that takes a panel LazyFrame and Dict of (train, test) splits, where
        the key represents the split number (1,2,...,n_splits) and the value is a tuple of LazyFrames.
    """

    def split(X: pl.LazyFrame) -> pl.LazyFrame:
        splits = _window_split(X, test_size, n_splits, step_size, window_size)
        if eager:
            splits = {i: pl.collect_all(s) for i, s in splits.items()}
        return splits

    return split


def _window_split(
    X: pl.LazyFrame,
    test_size: int,
    n_splits: int,
    step_size: int,
    window_size: Optional[int] = None,
) -> Mapping[int, Tuple[pl.LazyFrame, pl.LazyFrame]]:
    X = X.lazy()  # Defensive
    backward_steps = np.arange(1, n_splits) * step_size + test_size
    cutoffs = np.flip(np.concatenate([np.array([test_size]), backward_steps]))
    entity_col = X.collect_schema().names()[0]

    # TODO: split in two functions?
    if window_size:
        # Sliding window CV
        train_exprs = [
            pl.all().slice(pl.len() - cutoff - window_size, window_size)
            for cutoff in cutoffs
        ]
    else:
        # Expanding window CV
        train_exprs = [pl.all().slice(0, pl.len() - cutoff) for cutoff in cutoffs]

    test_exprs = [pl.all().slice(-cutoffs[i], test_size) for i in range(n_splits)]
    train_test_exprs = zip(train_exprs, test_exprs)

    splits = {}
    for i, train_test_expr in enumerate(train_test_exprs):
        train_expr, test_expr = train_test_expr
        train_split = (
            X.group_by(entity_col).agg(train_expr).explode(pl.all().exclude(entity_col))
        )
        test_split = (
            X.group_by(entity_col).agg(test_expr).explode(pl.all().exclude(entity_col))
        )
        splits[i] = train_split, test_split
    return splits
