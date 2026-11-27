from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from functime.base import Forecaster
from functime.base.forecaster import ForecastStrategy
from functime.forecasting._ar import fit_autoreg
from functime.forecasting._regressors import CensoredRegressor


def default_regress(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingRegressor

    regressor = HistGradientBoostingRegressor()
    regressor.fit(X=X, y=y)
    return regressor


def default_classify(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingClassifier

    classifier = HistGradientBoostingClassifier()
    classifier.fit(X=X, y=y)
    return classifier


class censored_model(Forecaster):
    """Censored forecaster given `threshold` parameter (defaults to 0.0).

    Two separate forecasters are fit above and below a certain threshold.
    The forecasts are then combined using a binary classifier where: {0: "below_threshold", 1: "above_threshold"}.
    Forecasts include a `threshold_proba` column with the classifier's probability that each
    forecast is above the threshold.
    """

    def __init__(
        self,
        freq: str | None,
        lags: int,
        max_horizons: int | None = None,
        strategy: ForecastStrategy | None = None,
        threshold: float = 0.0,
        regress: Callable | None = None,
        classify: Callable | None = None,
        **kwargs,
    ):
        self.threshold = threshold
        self.regress = regress or default_regress
        self.classify = classify or default_classify
        return super().__init__(
            freq=freq, lags=lags, max_horizons=max_horizons, strategy=strategy, **kwargs
        )

    def _fit(self, y: pl.LazyFrame, X: pl.LazyFrame | None = None):
        def regress(X: pl.DataFrame, y: pl.DataFrame) -> CensoredRegressor:
            # A new regressor per call, so each direct horizon keeps its own fit
            censored_regressor = CensoredRegressor(
                threshold=self.threshold,
                regress=self.regress,
                classify=self.classify,
            )
            return censored_regressor.fit(X=X, y=y)

        return fit_autoreg(
            regress=regress,
            lags=self.lags,
            y=y,
            X=X,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
        )


class zero_inflated_model(censored_model):
    """Censored forecaster with threshold at 0."""

    def __init__(
        self,
        freq: str | None,
        lags: int,
        max_horizons: int | None = None,
        strategy: ForecastStrategy | None = None,
        regress: Callable | None = None,
        classify: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            freq=freq,
            lags=lags,
            max_horizons=max_horizons,
            strategy=strategy,
            threshold=0.0,
            regress=regress,
            classify=classify,
            **kwargs,
        )
