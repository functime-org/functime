import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

import polars as pl


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
    """A functime Model definition.

    Inherited by `functime` model types:
    - check
    - classifier
    - clusterer
    - forecaster
    - transformer
    """

    def __init__(self, model: Callable, *args, **kwargs):
        self.model = model
        self.args = args
        self.kwargs = kwargs
        self.state = None
        self.entity_col_dtype = None
        self.string_cache = {}
        self.inv_string_cache = {}

    def _set_string_cache(self, df: pl.DataFrame) -> pl.DataFrame:
        entity_col = df.columns[0]
        entity_col_dtype = df.schema[entity_col]
        entities = df.get_column(entity_col).unique()
        string_cache = {entity: i for i, entity in enumerate(entities)}
        if entity_col_dtype == pl.Categorical:
            # Reset categorical to string type
            df = df.with_columns(pl.col(entity_col).cast(pl.Utf8))
        df_new = df.with_columns(
            pl.col(entity_col).map_dict(string_cache, return_dtype=pl.Int32)
        )
        self.entity_col_dtype = entity_col_dtype
        self.string_cache = string_cache
        self.inv_string_cache = {i: entity for entity, i in string_cache.items()}
        return df_new

    def _enforce_string_cache(self, df: pl.DataFrame) -> pl.DataFrame:
        entity_col = df.columns[0]
        if df.schema[entity_col] == pl.Categorical:
            # Reset categorical to string type
            df = df.with_columns(pl.col(entity_col).cast(pl.Utf8))
        return df.with_columns(
            pl.col(entity_col).map_dict(self.string_cache, return_dtype=pl.Int32)
        )

    def _reset_string_cache(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(df.columns[0]).map_dict(
                self.inv_string_cache, return_dtype=self.entity_col_dtype
            )
        )

    @property
    def func(self):
        return self.model(*self.args, **self.kwargs)

    @property
    def params(self):
        model = self.model
        kwargs = self.kwargs
        sig = inspect.signature(model)
        model_params = sig.parameters
        model_args = list(model_params.keys())
        params = {
            **{
                k: kwargs.get(k, v.default)
                for k, v in model_params.items()
                if k != "kwargs"
            },
            **{model_args[i]: p for i, p in enumerate(self.args)},
        }
        return params
