from typing import Optional

import polars as pl

from functime.base import Forecaster
from functime.forecasting._ar import fit_autoreg
from functime.forecasting._regressors import SklearnRegressor


def _knn(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.neighbors import KNeighborsRegressor

        regressor = SklearnRegressor(
            regressor=KNeighborsRegressor(**kwargs, n_jobs=-1),
        )
        return regressor.fit(X=X, y=y)

    return regress


class knn(Forecaster):
    """Autoregressive k-nearest neighbors.

    Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    """

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _knn(**self.kwargs)
        return fit_autoreg(
            regress=regress,
            y=y,
            X=X,
            lags=self.lags,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
        )
