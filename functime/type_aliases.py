"""Type aliases."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import polars as pl

# general
type PolarsFrame = pl.DataFrame | pl.LazyFrame

# cross validation
type EagerSplitter = Callable[
    [pl.DataFrame | pl.LazyFrame], tuple[pl.DataFrame, pl.DataFrame]
]
type LazySplitter = Callable[
    [pl.DataFrame | pl.LazyFrame], tuple[pl.LazyFrame, pl.LazyFrame]
]

# feature extraction
type DetrendMethod = Literal["linear", "mean"]
