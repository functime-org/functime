import inspect
from functools import cached_property, wraps
from typing import Callable, Tuple, TypeVar, Union

import polars as pl
from typing_extensions import ParamSpec

from functime.base.model import ModelState

P = ParamSpec("P")  # The parameters of the Model
R = TypeVar("R")

DF_TYPE = Union[pl.LazyFrame, pl.DataFrame]


class Transformer:
    """A transformer."""

    def __init__(self, model: Callable, *args, **kwargs):
        self.model = model
        self.args = args
        self.kwargs = kwargs
        self.state = None

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

    def __call__(self, X: DF_TYPE):
        return self.transform(X)

    @cached_property
    def is_invertible(self):
        return isinstance(self.func, Tuple)

    def transform(self, X: DF_TYPE) -> pl.LazyFrame:
        X = X.lazy()
        transform = self.func[0] if self.is_invertible else self.func
        artifacts = transform(X)
        state = ModelState(entity=X.columns[0], time=X.columns[1], artifacts=artifacts)
        self.state = state
        return artifacts["X_new"]

    def invert(self, X: DF_TYPE) -> pl.LazyFrame:
        if not self.is_invertible:
            raise ValueError("`invert` is not supported for this transformer.")
        invert = self.func[1]
        return invert(state=self.state, X=X.lazy())

    def transform_new(self, X: DF_TYPE) -> pl.LazyFrame:
        transform = self.func[2]
        X_new = transform(state=self.state, X=X.lazy())
        return X_new


def transformer(model: Callable[P, R]):
    @wraps(model)
    def _transformer(*args: P.args, **kwargs: P.kwargs) -> Transformer:
        return Transformer(model, *args, **kwargs)

    return _transformer
