from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from functime._core.model_selection import (
    expanding_window_indices,
    sliding_window_indices,
    temporal_train_test_split,
)


def train_test_split(
    data: Any,
    *,
    time_col: str = "time",
    entity_col: str | Sequence[str] | None = None,
    test_size: int | float = 0.25,
    gap: int = 0,
):
    """Split a univariate or panel DataFrame chronologically into train and test sets.

    Parameters
    ----------
    data : DataFrame-like
        Polars, Pandas, Arrow, or Narwhals DataFrame.
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    test_size : int or float, default=0.25
        Number or fraction of trailing unique timestamps reserved for testing.
    gap : int, default=0
        Number of unique timestamps omitted between train end and test start.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        ``(train_df, test_df)``.
    """
    return temporal_train_test_split(
        data,
        time_col=time_col,
        entity_col=entity_col,
        test_size=test_size,
        gap=gap,
    )


class ExpandingWindowSplit(BaseCrossValidator):
    """Expanding-window temporal cross-validator for univariate and panel data.

    Splits folds along sorted unique values of ``time_col`` so all entities share
    identical temporal cutoffs with no future leakage.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    n_splits : int, default=5
        Number of temporal folds.
    test_size : int, default=1
        Number of unique timestamps in each test fold.
    step_size : int, default=1
        Number of unique timestamps between consecutive test folds.
    gap : int, default=0
        Number of timestamps skipped between train end and test start.
    max_train_size : int or None, default=None
        Optional maximum number of trailing timestamps included in each training fold.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        n_splits: int = 5,
        test_size: int = 1,
        step_size: int = 1,
        gap: int = 0,
        max_train_size: int | None = None,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.n_splits = n_splits
        self.test_size = test_size
        self.step_size = step_size
        self.gap = gap
        self.max_train_size = max_train_size

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate ``(train_indices, test_indices)`` positional splits ordered by time.

        Parameters
        ----------
        X : DataFrame-like
            Input panel or univariate DataFrame.
        y : Any, default=None
            Ignored; accepted for scikit-learn compatibility.
        groups : Any, default=None
            Ignored; accepted for scikit-learn compatibility.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            Positional integer row index arrays into ``X``.
        """
        folds = expanding_window_indices(
            X,
            time_col=self.time_col,
            entity_col=self.entity_col,
            n_splits=self.n_splits,
            test_size=self.test_size,
            step_size=self.step_size,
            gap=self.gap,
            max_train_size=self.max_train_size,
        )
        yield from folds


class SlidingWindowSplit(BaseCrossValidator):
    """Fixed-width sliding-window temporal cross-validator for univariate and panel data.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    n_splits : int, default=5
        Number of temporal folds.
    test_size : int, default=1
        Number of unique timestamps in each test fold.
    step_size : int, default=1
        Number of unique timestamps between consecutive test folds.
    gap : int, default=0
        Number of timestamps skipped between train end and test start.
    window_size : int, default=10
        Fixed number of unique timestamps in each training window.
    """

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        n_splits: int = 5,
        test_size: int = 1,
        step_size: int = 1,
        gap: int = 0,
        window_size: int = 10,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col
        self.n_splits = n_splits
        self.test_size = test_size
        self.step_size = step_size
        self.gap = gap
        self.window_size = window_size

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate ``(train_indices, test_indices)`` positional splits ordered by time.

        Parameters
        ----------
        X : DataFrame-like
            Input panel or univariate DataFrame.
        y : Any, default=None
            Ignored; accepted for scikit-learn compatibility.
        groups : Any, default=None
            Ignored; accepted for scikit-learn compatibility.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            Positional integer row index arrays into ``X``.
        """
        folds = sliding_window_indices(
            X,
            time_col=self.time_col,
            entity_col=self.entity_col,
            n_splits=self.n_splits,
            test_size=self.test_size,
            step_size=self.step_size,
            gap=self.gap,
            window_size=self.window_size,
        )
        yield from folds
