from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import polars as pl

from functime.base.model import (
    _enforce_string_cache,
    _reset_string_cache,
    _set_string_cache,
)

METRIC_TYPE = Callable[
    [pl.LazyFrame | pl.DataFrame, pl.LazyFrame | pl.DataFrame], pl.DataFrame
]


# Simple wrapper to collect y_true, y_pred if lazy
def metric(score: Callable):
    @wraps(score)
    def _score(
        y_true: pl.LazyFrame | pl.DataFrame,
        y_pred: pl.LazyFrame | pl.DataFrame,
        *args,
        **kwargs,
    ) -> pl.DataFrame:
        if isinstance(y_true, pl.LazyFrame):
            y_true = y_true.collect(streaming=True)

        if isinstance(y_pred, pl.LazyFrame):
            y_pred = y_pred.collect(streaming=True)

        y_true, entity_col_dtype, string_cache, inv_string_cache = y_true.pipe(
            _set_string_cache
        )
        y_pred = y_pred.pipe(_enforce_string_cache, string_cache=string_cache)
        # Coerce column names and dtypes
        cols = y_true.columns
        y_pred = y_pred.rename({x: y for x, y in zip(y_pred.columns, cols)}).select(
            [pl.col(col).cast(dtype) for col, dtype in y_true.schema.items()]
        )

        if "y_train" in kwargs:
            y_train = (
                kwargs["y_train"]
                .lazy()
                .collect(streaming=True)
                .pipe(_enforce_string_cache, string_cache=string_cache)
            )
            kwargs["y_train"] = y_train

        scores = score(y_true, y_pred, *args, **kwargs).pipe(
            _reset_string_cache,
            inv_string_cache=inv_string_cache,
            return_dtype=entity_col_dtype,
        )
        return scores

    return _score
