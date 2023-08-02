import numpy as np
import polars as pl

from functime.base import transformer


@transformer
def add_fourier_terms(sp: int, K: int):
    """Fourier features for time series seasonality.

    Fourier Series terms can be used as explanatory variables for the cases of multiple
    seasonal periods and or complex / long seasonal periods [1]_, [2]_. For every
    seasonal period, :math:`sp` and fourier term :math:`k` pair there are 2 fourier
    terms sin_sp_k and cos_sp_k:
        - sin_sp_k = :math:`sin(\frac{2 \\pi k t}{sp})`
        - cos_sp_k = :math:`cos(\frac{2 \\pi k t}{sp})`

    Where :math:`t` is the number of time steps elapsed from the beginning of the time series.

    The implementation is based on the fourier function from the R forecast package [3]_

    Parameters
    ----------
    sp: int
        Seasonal period.
    K : int
        Maximum order(s) of Fourier terms.
        Must be less than `sp`.

    References
    ----------
    .. [1] Hyndsight - Forecasting with long seasonal periods:
        https://robjhyndman.com/hyndsight/longseasonality/
    .. [2] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and
        practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3.
        Accessed on August 14th 2022.
    .. [3] https://pkg.robjhyndman.com/forecast/reference/fourier.html
    """

    if K > sp:
        raise ValueError("`K` must be less than `sp`")

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        n_cos_terms = K // 2 + K % 2
        n_sin_terms = K // 2
        cos_terms = [
            np.cos(2 * np.pi * k * pl.col("fourier_coef")).alias(f"cos_{sp}_{k}")
            for k in range(1, n_cos_terms + 1)
        ]
        sin_terms = [
            np.sin(2 * np.pi * k * pl.col("fourier_coef")).alias(f"sin_{sp}_{k}")
            for k in range(1, n_sin_terms + 1)
        ]
        X_new = X.with_columns(
            (pl.col(time_col).to_physical().over(entity_col) / sp).alias("fourier_coef")
        ).select([entity_col, time_col, *cos_terms, *sin_terms])
        artifacts = {"X_new": X_new}
        return artifacts

    return transform
