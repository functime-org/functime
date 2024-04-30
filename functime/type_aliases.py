"""Type aliases."""

from __future__ import annotations

import sys
from typing import (
    Callable,
    Literal,
    Tuple,
    TypeAlias,
    Union,
)

import polars as pl

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:  # 3.9
    from typing_extensions import TypeAlias

# general
PolarsFrame: TypeAlias = Union[pl.DataFrame, pl.LazyFrame]

# cross validation
EagerSplitter: TypeAlias = Callable[
    [Union[pl.DataFrame, pl.LazyFrame]], Tuple[pl.DataFrame, pl.DataFrame]
]
LazySplitter: TypeAlias = Callable[
    [Union[pl.DataFrame, pl.LazyFrame]], Tuple[pl.LazyFrame, pl.LazyFrame]
]


# feature extraction
DetrendMethod: TypeAlias = Literal["linear", "mean"]
