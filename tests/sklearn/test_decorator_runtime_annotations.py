# ruff: noqa: I002
# No `from __future__ import annotations` here on purpose: on Python 3.14+ (PEP 649)
# class annotations are then evaluated lazily and absent from cls.__dict__.
from collections.abc import Sequence
from typing import Any, Self

import polars as pl
import pytest

from functime._frame import TemporalFrame, temporal_transform
from functime.sklearn import BaseTemporalTransformer, sklearn_transformer


@temporal_transform()
def _add(columns: Sequence[str], *, k: float = 0.0) -> list[pl.Expr]:
    return [(pl.col(c) + k).alias(c) for c in columns]


@sklearn_transformer(_add)
class _Add(BaseTemporalTransformer):
    k: float = 5.0


class _NullTolerantBase:
    _allow_null_time = True


class _PassthroughCore(_NullTolerantBase):
    def fit(self, data: TemporalFrame | Any, *, time_col: str = "time") -> Self:
        return self

    def transform(self, data: TemporalFrame) -> pl.LazyFrame:
        return data.lf


@sklearn_transformer(_PassthroughCore)
class _Passthrough(BaseTemporalTransformer):
    pass


def test_runtime_annotations_become_init_params() -> None:
    df = pl.DataFrame({"time": [1, 2], "y": [1.0, 2.0]})
    assert _Add().get_params()["k"] == 5.0
    assert _Add().fit_transform(df).get_column("y").to_list() == [6.0, 7.0]
    assert _Add(k=1.0).fit_transform(df).get_column("y").to_list() == [2.0, 3.0]
    with pytest.raises(TypeError):
        _Add(unknown=1)  # type: ignore[call-arg]


def test_inherited_core_null_flags_are_honoured() -> None:
    df = pl.DataFrame({"time": [1, None], "y": [1.0, 2.0]})
    assert _Passthrough().fit_transform(df).height == 2
