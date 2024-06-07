from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from functime.feature_extractors import cwt_coefficients


@pytest.mark.parametrize("length", np.random.random_integers(low=1, high=100, size=5))
@pytest.mark.parametrize("widths", [(2,), (2, 5, 10, 20), (2, 5, 10, 20, 30)])
@pytest.mark.parametrize(
    "n_coefficients", np.random.random_integers(low=1, high=100, size=5)
)
def test_cwt(length: int, widths: tuple, n_coefficients: int) -> None:
    out = cwt_coefficients(
        pl.Series([1 for _ in range(length)]),
        widths=widths,
        n_coefficients=n_coefficients,
    )
    assert len(out) == min(n_coefficients, length) * len(widths)
