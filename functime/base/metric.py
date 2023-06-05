from functools import wraps
from typing import Callable, Mapping, Union

import polars as pl


def _set_string_cache(df: pl.DataFrame) -> pl.DataFrame:
    entity_col = df.columns[0]
    entities = df.get_column(entity_col).unique()
    string_cache = {entity: i for i, entity in enumerate(entities)}
    entity_col_dtype = df.schema[entity_col]
    if entity_col_dtype == pl.Categorical:
        # Reset categorical to string type
        df = df.with_columns(pl.col(entity_col).cast(pl.Utf8))
    df_new = df.with_columns(
        pl.col(entity_col).map_dict(string_cache, return_dtype=pl.Int32)
    )
    inv_string_cache = {i: entity for entity, i in string_cache.items()}
    return df_new, string_cache, inv_string_cache


def _enforce_string_cache(
    df: pl.DataFrame, string_cache: Mapping[Union[int, str], int]
) -> pl.DataFrame:
    entity_col = df.columns[0]
    if df.schema[entity_col] == pl.Categorical:
        # Reset categorical to string type
        df = df.with_columns(pl.col(entity_col).cast(pl.Utf8))
    return df.with_columns(
        pl.col(entity_col).map_dict(string_cache, return_dtype=pl.Int32)
    )


def _reset_string_cache(
    df: pl.DataFrame, inv_string_cache: Mapping[int, Union[int, str]], return_dtype
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(df.columns[0]).map_dict(inv_string_cache, return_dtype=return_dtype)
    )


# Simple wrapper to collect y_true, y_pred if lazy
def metric(score: Callable):
    @wraps(score)
    def _score(
        y_true: Union[pl.LazyFrame, pl.DataFrame],
        y_pred: Union[pl.LazyFrame, pl.DataFrame],
        *args,
        **kwargs,
    ) -> pl.DataFrame:

        entity_col_dtype = y_true.schema[y_true.columns[0]]
        if isinstance(y_true, pl.LazyFrame):
            y_true = y_true.collect()

        if isinstance(y_pred, pl.LazyFrame):
            y_pred = y_pred.collect()

        y_true, string_cache, inv_string_cache = y_true.pipe(_set_string_cache)
        y_pred = y_pred.pipe(_enforce_string_cache, string_cache=string_cache)
        # Coerce columnn names and dtypes
        cols = y_true.columns
        y_pred = y_pred.rename({x: y for x, y in zip(y_pred.columns, cols)}).select(
            [pl.col(col).cast(dtype) for col, dtype in y_true.schema.items()]
        )

        if "y_train" in kwargs:
            y_train = (
                kwargs["y_train"]
                .lazy()
                .collect()
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
