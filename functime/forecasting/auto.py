from typing import Any, Literal, Mapping, Optional

from .base import ForecasterClient

FORECAST_STRATEGIES = Optional[Literal["direct", "recursive", "ensemble"]]


class BaseAutoForecaster(ForecasterClient):
    def __init__(
        self,
        fh: int,
        freq: str,
        min_lags: int = 3,
        max_lags: int = 12,
        max_horizons: Optional[int] = None,
        strategy: FORECAST_STRATEGIES = None,
        step_size: int = 1,
        n_splits: int = 5,
        time_budget: int = 5,
        search_space: Optional[Mapping[str, Any]] = None,
        points_to_evaluate: Optional[Mapping[str, Any]] = None,
        num_samples: int = -1,
        **kwargs,
    ):
        super().__init__(
            fh=fh,
            freq=freq,
            min_lags=min_lags,
            max_lags=max_lags,
            max_horizons=max_horizons,
            strategy=strategy,
            step_size=step_size,
            n_splits=n_splits,
            time_budget=time_budget,
            search_space=search_space,
            points_to_evaluate=points_to_evaluate,
            num_samples=num_samples,
            **kwargs,
        )


class AutoElasticNet(BaseAutoForecaster):
    """ElasticNet forecaster with automated hyperparameter tuning.

    Parameters
    ----------
    fh : int
        Number of lags.
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
    """

    model = "auto_elastic_net"


class AutoKNN(BaseAutoForecaster):
    """K-nearest neighbors forecaster with automated hyperparameter tuning.

    Parameters
    ----------
    fh : int
        Number of lags.
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
    """

    model = "auto_knn"


class AutoLasso(BaseAutoForecaster):
    """LASSO regression forecaster with automated hyperparameter tuning.

    Parameters
    ----------
    fh : int
        Number of lags.
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
    """

    model = "auto_lasso"


class AutoLightGBM(BaseAutoForecaster):
    """LightGBM forecaster with automated hyperparameter tuning.

    Parameters
    ----------
    fh : int
        Number of lags.
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
    """

    model = "auto_lightgbm"


class AutoLinearModel(BaseAutoForecaster):
    """Linear autoregressive forecaster with automated hyperparameter tuning.

    Parameters
    ----------
    fh : int
        Number of lags.
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
    """

    model = "auto_linear_model"


class AutoRidge(BaseAutoForecaster):
    """Ridge regression forecaster with automated hyperparameter tuning.

    Parameters
    ----------
    fh : int
        Number of lags.
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
    """

    model = "auto_ridge"
