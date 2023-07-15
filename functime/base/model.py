from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Union

import polars as pl


def _set_string_cache(df: pl.DataFrame):
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
    return df_new, entity_col_dtype, string_cache, inv_string_cache


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


class Regressor(Protocol):
    def fit(self, X, y, sample_weight=None):
        ...

    def predict(self, X):
        ...


class Classifier(Protocol):
    def fit(self, X, y, sample_weight=None):
        ...

    def predict_proba(self, X):
        ...


# functime uses immutable dataclases to store artifacts returned by `fit`/
# These artifacts are bundled together with y, X metadata (e.g. column names)
# and passed into the `predict` function(s).


@dataclass(frozen=True)
class ModelState:
    entity: str
    time: str
    artifacts: Mapping[str, Any]


class Model:
    """A functime Model definition."""

    def __init__(self):
        self.state = None
        self.entity_col_dtype = None
        self.string_cache = {}
        self.inv_string_cache = {}

    def _set_string_cache(self, df: pl.DataFrame) -> pl.DataFrame:
        df_new, entity_col_dtype, string_cache, inv_string_cache = _set_string_cache(df)
        self.entity_col_dtype = entity_col_dtype
        self.string_cache = string_cache
        self.inv_string_cache = inv_string_cache
        return df_new

    def _enforce_string_cache(self, df: pl.DataFrame) -> pl.DataFrame:
        return _enforce_string_cache(df, self.string_cache)

    def _reset_string_cache(self, df: pl.DataFrame) -> pl.DataFrame:
        return _reset_string_cache(df, self.inv_string_cache, self.entity_col_dtype)
