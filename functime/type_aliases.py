"""Type aliases."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

import polars as pl

# general
PolarsFrame: TypeAlias = pl.DataFrame | pl.LazyFrame

# cross validation
EagerSplitter: TypeAlias = Callable[
    [pl.DataFrame | pl.LazyFrame], tuple[pl.DataFrame, pl.DataFrame]
]
LazySplitter: TypeAlias = Callable[
    [pl.DataFrame | pl.LazyFrame], tuple[pl.LazyFrame, pl.LazyFrame]
]

# feature extraction
DetrendMethod: TypeAlias = Literal["linear", "mean"]
