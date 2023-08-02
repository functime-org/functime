from typing import Optional

import polars as pl

from functime.base import Forecaster
from functime.forecasting._ar import fit_autoreg
from functime.forecasting._regressors import SklearnRegressor


def _linear_model(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.linear_model import LinearRegression

        regressor = SklearnRegressor(
            regressor=LinearRegression(**kwargs, copy_X=False),
        )
        return regressor.fit(X=X, y=y)

    return regress


def _lasso(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.linear_model import Lasso

        regressor = SklearnRegressor(
            regressor=Lasso(**kwargs, tol=0.001, copy_X=False, max_iter=10000),
        )
        return regressor.fit(X=X, y=y)

    return regress


def _ridge(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.linear_model import Ridge

        regressor = SklearnRegressor(
            regressor=Ridge(**kwargs, tol=0.001, copy_X=False, max_iter=10000)
        )
        return regressor.fit(X=X, y=y)

    return regress


def _elastic_net(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.linear_model import ElasticNet

        regressor = SklearnRegressor(
            regressor=ElasticNet(**kwargs, tol=0.001, copy_X=False, max_iter=10000)
        )
        return regressor.fit(X=X, y=y)

    return regress


def _lasso_cv(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.linear_model import LassoCV

        regressor = SklearnRegressor(
            regressor=LassoCV(**kwargs, max_iter=10000, n_jobs=-1),
        )
        return regressor.fit(X=X, y=y)

    return regress


def _ridge_cv(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.linear_model import RidgeCV

        regressor = SklearnRegressor(regressor=RidgeCV(**kwargs))
        return regressor.fit(X=X, y=y)

    return regress


def _elastic_net_cv(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        from sklearn.linear_model import ElasticNetCV

        regressor = SklearnRegressor(regressor=ElasticNetCV(**kwargs, n_jobs=-1))
        return regressor.fit(X=X, y=y)

    return regress


class linear_model(Forecaster):
    """Autoregressive linear forecaster.

    Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        kwargs = self.kwargs
        # Check dummy variable trap
        if (
            X is not None
            and len(X.select(pl.col(pl.Categorical)).columns) > 0
            and kwargs.get("fit_intercept") is True
        ):
            raise ValueError(
                "Dummy variable trap! Must set `fit_intercept=False` if X contains categorical columns."
            )

        regress = _linear_model(**kwargs)
        return fit_autoreg(
            regress=regress,
            y=y,
            X=X,
            lags=self.lags,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
        )


class lasso(Forecaster):
    """Autoregressive LASSO forecaster.

    Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
    """

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _lasso(**self.kwargs)
        return fit_autoreg(
            regress=regress,
            y=y,
            X=X,
            lags=self.lags,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
        )


class ridge(Forecaster):
    """Autoregressive Ridge forecaster.

    Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
    """

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _ridge(**self.kwargs)
        return fit_autoreg(
            regress=regress,
            y=y,
            X=X,
            lags=self.lags,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
        )


class elastic_net(Forecaster):
    """Autoregressive ElasticNet forecaster.

    Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
    """

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _elastic_net(**self.kwargs)
        return fit_autoreg(
            regress=regress,
            y=y,
            X=X,
            lags=self.lags,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
        )


class lasso_cv(Forecaster):
    """Autoregressive LassoCV forecaster.

    Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.LassoCV
    """

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _lasso_cv(**self.kwargs)
        return fit_autoreg(
            regress=regress,
            y=y,
            X=X,
            lags=self.lags,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
        )


class ridge_cv(Forecaster):
    """Autoregressive RidgeCV forecaster.

    Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.RidgeCV
    """

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _ridge_cv(**self.kwargs)
        return fit_autoreg(
            regress=regress,
            y=y,
            X=X,
            lags=self.lags,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
        )


class elastic_net_cv(Forecaster):
    """Autoregressive ElasticNetCV forecaster.

    Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNetCV
    """

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _elastic_net_cv(**self.kwargs)
        return fit_autoreg(
            regress=regress,
            y=y,
            X=X,
            lags=self.lags,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
        )
