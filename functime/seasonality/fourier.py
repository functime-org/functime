import numpy as np
import polars as pl

from functime.base import transformer


@transformer
def add_fourier_terms(sp: int, K: int):
    """Fourier features for time series seasonality.

    Fourier Series terms can be used as explanatory variables for the cases of multiple
    seasonal periods and or complex / long seasonal periods.

    The implementation is based on the Fourier function from the R [forecast package](https://pkg.robjhyndman.com/forecast/reference/fourier.html).

    Parameters
    ----------
    sp: int
        Seasonal period.
    K : int
        Maximum order(s) of Fourier terms.
        Must be less than `sp`.
    """

    if K > sp:
        raise ValueError("`K` must be less than `sp`")

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        cos_terms = [
            np.cos(2 * np.pi * k * pl.col("fourier_coef")).alias(f"cos_{sp}_{k}")
            for k in range(1, K + 1)
        ]
        sin_terms = [
            np.sin(2 * np.pi * k * pl.col("fourier_coef")).alias(f"sin_{sp}_{k}")
            for k in range(1, K + 1)
        ]
        X_new = X.with_columns(
            (pl.col(time_col).arg_sort().mod(sp).over(entity_col) / sp).alias(
                "fourier_coef"
            )
        ).select([*X.columns, *cos_terms, *sin_terms])
        artifacts = {"X_new": X_new}
        return artifacts

    return transform
