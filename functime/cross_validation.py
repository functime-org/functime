from typing import List, Optional, Tuple

import numpy as np
import polars as pl


def get_split(
    entity_col: str,
    time_col: str,
    target_col: str,
    y_splits: pl.LazyFrame,
    X_splits: Optional[pl.LazyFrame] = None,
    feature_cols: Optional[List[str]] = None,
):

    y_cols = [entity_col, time_col, target_col]
    X_cols = None if X_splits is None else [entity_col, time_col, *feature_cols]

    def _get_split(splits: pl.DataFrame, i: int) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        array_cols = pl.all().exclude(entity_col)
        train_cols = [entity_col, pl.col(f"^*__train_{i}$")]
        test_cols = [entity_col, pl.col(f"^*__test_{i}$")]
        train_split = splits.select(train_cols).explode(array_cols).lazy()
        test_split = splits.select(test_cols).explode(array_cols).lazy()
        return train_split, test_split

    def get_y_split(i: int):
        y_train, y_test = _get_split(y_splits, i)
        # Remove split label suffix
        y_train = y_train.rename({x: y for x, y in zip(y_train.columns, y_cols)})
        y_test = y_test.rename({x: y for x, y in zip(y_test.columns, y_cols)})
        return y_train, y_test, None, None

    def get_X_y_split(i: int):
        y_train, y_test = _get_split(y_splits, i)
        X_train, X_test = _get_split(X_splits, i)
        # Remove split label suffix
        y_train = y_train.rename({x: y for x, y in zip(y_train.columns, y_cols)})
        y_test = y_test.rename({x: y for x, y in zip(y_test.columns, y_cols)})
        X_train = X_train.rename({x: y for x, y in zip(X_train.columns, X_cols)})
        X_test = X_test.rename({x: y for x, y in zip(X_test.columns, X_cols)})
        return y_train, y_test, X_train, X_test

    fn = get_y_split if X_splits is None else get_X_y_split
    return fn


def train_test_split(test_size: int):
    """Return train/test splits.

    Parameters
    ----------
    test_size : int
        Number of test samples.

    Returns
    -------
    splitter : Callable[pl.LazyFrame, pl.LazyFrame]
        Function that takes a panel LazyFrame and returns a LazyFrame of train / test splits.
    """

    def split(X: pl.LazyFrame) -> pl.LazyFrame:
        X = X.lazy()  # Defensive
        train_split = (
            X.groupby(X.columns[0]).agg(pl.all().slice(0, pl.count() - test_size))
            .explode(pl.all().exclude(X.columns[0]))
        )
        test_split = (
            X.groupby(X.columns[0]).agg(pl.all().slice(-1 * test_size, test_size))
            .explode(pl.all().exclude(X.columns[0]))
        )
        return train_split, test_split

    return split


def _window_split(
    X: pl.LazyFrame,
    test_size: int,
    n_splits: int,
    step_size: int,
    window_size: Optional[int] = None,
) -> pl.LazyFrame:
    X = X.lazy()  # Defensive
    backward_steps = np.arange(1, n_splits) * step_size + test_size
    cutoffs = np.flip(np.concatenate([np.array([test_size]), backward_steps]))
    if window_size:
        # Sliding window CV
        train_exprs = [
            pl.all()
            .slice(pl.count() - x - window_size, window_size)
            .suffix(f"__train_{i}")
            for i, x in enumerate(cutoffs)
        ]
    else:
        # Expanding window CV
        train_exprs = [
            pl.all().slice(0, pl.count() - x).suffix(f"__train_{i}")
            for i, x in enumerate(cutoffs)
        ]
    test_exprs = [
        pl.all().slice(-cutoffs[i], test_size).suffix(f"__test_{i}")
        for i in range(n_splits)
    ]
    exprs = zip(train_exprs, test_exprs)
    splits = X.groupby(X.columns[0]).agg(
        [window for windows in exprs for window in windows]
    )
    return splits


def expanding_window_split(
    test_size: int,
    n_splits: int = 5,
    step_size: int = 1,
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

    Returns
    -------
    splitter : Callable[pl.LazyFrame, pl.LazyFrame]
        Function that takes a panel LazyFrame and returns a LazyFrame of train / test splits.
    """

    def split(X: pl.LazyFrame) -> pl.LazyFrame:
        return _window_split(X, test_size, n_splits, step_size)

    return split


def sliding_window_split(
    test_size: int,
    n_splits: int = 5,
    step_size: int = 1,
    window_size: int = 10,
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

    Returns
    -------
    splitter : Callable[pl.LazyFrame, pl.LazyFrame]
        Function that takes a panel LazyFrame and returns a LazyFrame of train / test splits.
    """

    def split(X: pl.LazyFrame) -> pl.LazyFrame:
        return _window_split(X, test_size, n_splits, step_size, window_size)

    return split
