from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import cached_property, wraps

import polars as pl

from functime.base.model import ModelState

DF_TYPE = pl.LazyFrame | pl.DataFrame


def _restore_transformer(
    factory: Callable,
    args: tuple,
    kwargs: dict,
    state: ModelState | None,
) -> Transformer:
    obj = factory(*args, **kwargs)
    obj.state = state
    return obj


def _restore_raw_transformer(
    transf: Callable,
    args: tuple,
    kwargs: dict,
    state: ModelState | None,
) -> Transformer:
    obj = Transformer(transf, *args, **kwargs)
    obj.state = state
    return obj


class Transformer:
    """A transformer."""

    def __init__(self, transf: Callable, *args, **kwargs):
        self.transf = transf
        self.args = args
        self.kwargs = kwargs
        self.state = None
        self._factory: Callable | None = None

    def __reduce__(self):
        # `@wraps` replaces the module attribute with the wrapper, so stdlib
        # `pickle` must reference the decorated factory rather than `self.transf`.
        if self._factory is not None:
            return (
                _restore_transformer,
                (self._factory, self.args, self.kwargs, self.state),
            )
        return (
            _restore_raw_transformer,
            (self.transf, self.args, self.kwargs, self.state),
        )

    @property
    def func(self):
        return self.transf(*self.args, **self.kwargs)

    @property
    def params(self):
        transf = self.transf
        kwargs = self.kwargs
        sig = inspect.signature(transf)
        params = sig.parameters
        args = list(params.keys())
        params = {
            **{k: kwargs.get(k, v.default) for k, v in params.items() if k != "kwargs"},
            **{args[i]: p for i, p in enumerate(self.args)},
        }
        return params

    def __call__(self, X: DF_TYPE):
        return self.transform(X)

    @cached_property
    def is_invertible(self):
        return isinstance(self.func, tuple)

    def transform(self, X: DF_TYPE) -> pl.LazyFrame:
        X = X.lazy()
        X_columns = X.collect_schema().names()
        transform = self.func[0] if self.is_invertible else self.func
        artifacts = transform(X)
        state = ModelState(entity=X_columns[0], time=X_columns[1], artifacts=artifacts)
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


def transformer[**P, R](transf: Callable[P, R]):
    @wraps(transf)
    def _transformer(*args: P.args, **kwargs: P.kwargs) -> Transformer:
        obj = Transformer(transf, *args, **kwargs)
        obj._factory = _transformer
        return obj

    return _transformer
