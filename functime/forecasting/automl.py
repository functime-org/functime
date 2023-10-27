from abc import abstractmethod
from functools import partial
from typing import Any, Mapping, Optional, Union

import polars as pl
from flaml import tune
from typing_extensions import Literal

from functime.base.forecaster import FORECAST_STRATEGIES, SUPPORTED_FREQ, Forecaster
from functime.base.transformer import Transformer
from functime.forecasting.knn import knn
from functime.forecasting.linear import elastic_net, lasso, linear_model, ridge

try:
    from functime.forecasting.lightgbm import lightgbm
except ImportError:
    pass


class AutoForecaster(Forecaster):
    """Forecaster with automated hyperparameter tuning and lags selection.

    Parameters
    ----------
    freq : str
        Offset alias as dictated.
    min_lags : int
        Minimum number of lagged target values.
    max_lags : int
        Maximum number of lagged target values.
    max_horizons: Optional[int]
        Maximum number of horizons to predict directly.
        Only applied if `strategy` equals "direct" or "ensemble".
    strategy : Optional[str]
        Forecasting strategy. Currently supports "recursive", "direct",
        and "ensemble" of both recursive and direct strategies.
    test_size : int
        Number of lags.
    step_size : int
        Step size between backtest windows.
    n_splits : int
        Number of backtest splits.
    time_budget : int
        Maximum time budgeted to train each forecaster per window and set of hyperparameters.
    search_space : Optional[dict]
        Equivalent to `config` in [FLAML](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function#search-space)
    points_to_evaluate : Optional[dict]
        Equivalent to `points_to_evaluate` in [FLAML](https://microsoft.github.io/FLAML/docs/Use-Cases/Tune-User-Defined-Function#warm-start)
    num_samples : int
        Number of hyper-parameter sets to test. -1 means unlimited (until `time_budget` is exhausted.)
    target_transform : Optional[Transformer]
        functime transformer to apply to `y` before fit. The transform is inverted at predict time.
    feature_transform : Optional[Transformer]
        functime transformer to apply to `X` before fit and predict.
    **kwargs : Mapping[str, Any]
        Additional keyword arguments passed into underlying sklearn-compatible regressor.
    """

    def __init__(
        self,
        freq: Union[str, None],
        min_lags: int = 3,
        max_lags: int = 12,
        max_horizons: Optional[int] = None,
        strategy: FORECAST_STRATEGIES = None,
        test_size: int = 1,
        step_size: int = 1,
        n_splits: int = 5,
        time_budget: int = 5,
        search_space: Optional[Mapping[str, Any]] = None,
        points_to_evaluate: Optional[Mapping[str, Any]] = None,
        num_samples: int = -1,
        target_transform: Optional[Transformer] = None,
        feature_transform: Optional[Transformer] = None,
        **kwargs,
    ):
        if freq not in SUPPORTED_FREQ:
            raise ValueError(f"Offset {freq} not supported")

        self.freq = freq
        self.min_lags = min_lags
        self.max_lags = max_lags
        self.max_horizons = max_horizons
        self.strategy = strategy
        self.test_size = test_size
        self.step_size = step_size
        self.n_splits = n_splits
        self.time_budget = time_budget
        self.search_space = search_space
        self.points_to_evaluate = points_to_evaluate
        self.num_samples = num_samples
        self.target_transform = target_transform
        self.feature_transform = feature_transform
        self.kwargs = kwargs

    @property
    @abstractmethod
    def forecaster(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def best_params(self):
        return self.state.artifacts["best_params"]

    @property
    def default_search_space(self):
        return None

    @property
    def default_points_to_evaluate(self):
        return None

    @property
    def low_cost_partial_config(self):
        return None

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        from functime.forecasting._ar import fit_cv

        return fit_cv(
            y=y,
            X=X,
            forecaster_cls=partial(self.forecaster, **self.kwargs),
            freq=self.freq,
            min_lags=self.min_lags,
            max_lags=self.max_lags,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
            test_size=self.test_size,
            step_size=self.step_size,
            n_splits=self.n_splits,
            time_budget=self.time_budget,
            search_space=self.search_space or self.default_search_space,
            points_to_evaluate=self.points_to_evaluate
            or self.default_points_to_evaluate,
            num_samples=self.num_samples,
            low_cost_partial_config=self.low_cost_partial_config,
        )

    def _predict(self, fh: int, X: Optional[pl.LazyFrame] = None):
        from functime.forecasting._ar import predict_autoreg

        return predict_autoreg(state=self.state, fh=fh, X=X)

    def backtest(
        self,
        y: Union[pl.LazyFrame, pl.DataFrame],
        X: Optional[pl.LazyFrame] = None,
        test_size: int = 1,
        step_size: int = 1,
        n_splits: int = 5,
        window_size: int = 10,
        strategy: Literal["expanding", "sliding"] = "expanding",
    ):
        # Get base forecaster with fixed best params
        forecaster_cls = self.forecaster
        if self.state is None:
            raise ValueError("Must `.fit` AutoForecaster before `.backtest`")
        best_params = self.state.artifacts["best_params"]
        forecaster = forecaster_cls(**best_params)
        y_preds, y_resids = forecaster.backtest(
            y=y,
            X=X,
            test_size=test_size,
            step_size=step_size,
            n_splits=n_splits,
            window_size=window_size,
            strategy=strategy,
        )
        return y_preds, y_resids


class auto_lightgbm(AutoForecaster):
    """LightGBM forecaster with automated lags and hyperparamters selection."""

    DEFAULT_TREE_DEPTH = 8

    @property
    def forecaster(self):
        return lightgbm

    @property
    def default_search_space(self):
        max_depth = self.kwargs.get("max_depth", 0)
        return {
            "reg_alpha": tune.loguniform(0.001, 20.0),
            "reg_lambda": tune.loguniform(0.001, 20.0),
            "num_leaves": tune.randint(
                2, 2**max_depth if max_depth > 0 else 2**self.DEFAULT_TREE_DEPTH
            ),
            "colsample_bytree": tune.uniform(0.4, 1.0),
            "subsample": tune.uniform(0.4, 1.0),
            "subsample_freq": tune.randint(1, 7),
            "min_child_samples": tune.qlograndint(5, 100, 5),
            "n_estimators": tune.qrandint(60, 400, 20),
        }

    @property
    def default_points_to_evaluate(self):
        return [
            {
                "num_leaves": 31,
                "colsample_bytree": 1.0,
                "subsample": 1.0,
                "min_child_samples": 20,
            }
        ]

    @property
    def low_cost_partial_config(self):
        return {"n_estimators": 50, "num_leaves": 2}


class auto_knn(AutoForecaster):
    """KNN forecaster with automated lags and hyperparamters selection."""

    @property
    def forecaster(self):
        return knn

    @property
    def default_search_space(self):
        return {"leaf_size": tune.choice([30, 60, 120, 400])}

    @property
    def low_cost_partial_config(self):
        return {"leaf_size": 400}


class auto_linear_model(AutoForecaster):
    """Autoregressive linear forecaster with automated lags selection."""

    @property
    def forecaster(self):
        return linear_model


class auto_lasso(AutoForecaster):
    """LASSO forecaster with automated lags and hyperparameters selection."""

    @property
    def forecaster(self):
        return lasso

    @property
    def default_search_space(self):
        return {
            "alpha": tune.loguniform(0.001, 20.0),
            "fit_intercept": tune.choice([True, False]),
        }

    @property
    def low_cost_partial_config(self):
        return {"alpha": 1.0}


class auto_ridge(AutoForecaster):
    """Ridge forecaster with automated lags and hyperparameters selection."""

    @property
    def forecaster(self):
        return ridge

    @property
    def default_search_space(self):
        return {
            "alpha": tune.loguniform(0.001, 20.0),
            "fit_intercept": tune.choice([True, False]),
        }

    @property
    def low_cost_partial_config(self):
        return {"alpha": 1.0}


class auto_elastic_net(AutoForecaster):
    """ElasticNet forecaster with automated lags and hyperparameters selection."""

    @property
    def forecaster(self):
        return elastic_net

    @property
    def default_search_space(self):
        return {
            "alpha": tune.loguniform(0.001, 20.0),
            "l1_ratio": tune.uniform(0, 1.0),
            "fit_intercept": tune.choice([True, False]),
        }

    @property
    def low_cost_partial_config(self):
        return {"alpha": 1.0}
