from typing import Callable, Optional, Union

import numpy as np
import polars as pl

from functime.base import Forecaster
from functime.base.forecaster import FORECAST_STRATEGIES
from functime.conversion import X_to_numpy, y_to_numpy
from functime.forecasting._ar import fit_autoreg
from functime.forecasting._reduction import make_reduction
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
    """

    def __init__(
        self,
        freq: Union[str, None],
        lags: int,
        max_horizons: Optional[int] = None,
        strategy: FORECAST_STRATEGIES = None,
        threshold: float = 0.0,
        regress: Optional[Callable] = None,
        classify: Optional[Callable] = None,
        **kwargs
    ):
        self.threshold = threshold
        self.regress = regress or default_regress
        self.classify = classify or default_classify
        return super().__init__(
            freq=freq, lags=lags, max_horizons=max_horizons, strategy=strategy, **kwargs
        )

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):

        # 1. Fit classifier
        target_col = y.columns[-1]
        X_y_final = (
            make_reduction(lags=self.lags, y=y, X=X)
            .with_columns(
                pl.when(pl.col(target_col) > self.threshold)
                .then(1)
                .otherwise(0)
                .alias(target_col)
            )
            .lazy()
        )
        X_final, y_final = pl.collect_all(
            [
                X_y_final.select(pl.all().exclude(target_col)),
                X_y_final.select([*X_y_final.columns[:2], target_col]),
            ]
        )
        fitted_classifier = self.classify(X=X_to_numpy(X_final), y=y_to_numpy(y_final))
        # 2. Fit forecast model on non-zeros
        censored_regressor = CensoredRegressor(
            threshold=self.threshold,
            regress=self.regress,
            predict_proba=fitted_classifier.predict_proba,
        )
        forecast_artifacts = fit_autoreg(
            regress=censored_regressor.fit,
            lags=self.lags,
            y=y,
            X=X,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
        )
        # 3. Collect artifacts
        artifacts = {"classifier": fitted_classifier, **forecast_artifacts}
        return artifacts


class zero_inflated_model(censored_model):
    """Censored forecaster with threshold at 0."""

    def __init__(
        self,
        freq: Union[str, None],
        lags: int,
        max_horizons: Optional[int] = None,
        strategy: FORECAST_STRATEGIES = None,
        regress: Optional[Callable] = None,
        classify: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(
            freq=freq,
            lags=lags,
            max_horizons=max_horizons,
            strategy=strategy,
            threshold=0.0,
            regress=regress,
            classify=classify,
            **kwargs
        )
