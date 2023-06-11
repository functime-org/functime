from typing import Literal, Optional, Union

from ._base import ForecasterClient

FORECAST_STRATEGIES = Optional[Literal["direct", "recursive", "ensemble"]]


class BaseForecaster(ForecasterClient):
    def __init__(
        self,
        freq: Union[str, None],
        lags: int,
        max_horizons: Optional[int] = None,
        strategy: FORECAST_STRATEGIES = None,
        **kwargs,
    ):
        super().__init__(
            freq=freq, lags=lags, max_horizons=max_horizons, strategy=strategy, **kwargs
        )


class ElasticNet(BaseForecaster):
    """ElasticNet forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    lags : int
        Number of lagged target variables.
    max_horizons: Optional[int]
        Maximum number of horizons to predict directly.
        Only applied if `strategy` equals "direct" or "ensemble".
    strategy : Optional[str]
        Forecasting strategy. Currently supports "recursive", "direct",
        and "ensemble" of both recursive and direct strategies.
    """

    model = "elastic_net"


class KNN(BaseForecaster):
    """K-nearest neighbors forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    lags : int
        Number of lagged target variables.
    max_horizons: Optional[int]
        Maximum number of horizons to predict directly.
        Only applied if `strategy` equals "direct" or "ensemble".
    strategy : Optional[str]
        Forecasting strategy. Currently supports "recursive", "direct",
        and "ensemble" of both recursive and direct strategies.
    """

    model = "knn"


class Lasso(BaseForecaster):
    """LASSO regression forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    lags : int
        Number of lagged target variables.
    max_horizons: Optional[int]
        Maximum number of horizons to predict directly.
        Only applied if `strategy` equals "direct" or "ensemble".
    strategy : Optional[str]
        Forecasting strategy. Currently supports "recursive", "direct",
        and "ensemble" of both recursive and direct strategies.
    """

    model = "lasso"


class LightGBM(BaseForecaster):
    """LightGBM forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    lags : int
        Number of lagged target variables.
    max_horizons: Optional[int]
        Maximum number of horizons to predict directly.
        Only applied if `strategy` equals "direct" or "ensemble".
    strategy : Optional[str]
        Forecasting strategy. Currently supports "recursive", "direct",
        and "ensemble" of both recursive and direct strategies.
    """

    model = "lightgbm"


class LinearModel(BaseForecaster):
    """Linear autoregressive forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    lags : int
        Number of lagged target variables.
    max_horizons: Optional[int]
        Maximum number of horizons to predict directly.
        Only applied if `strategy` equals "direct" or "ensemble".
    strategy : Optional[str]
        Forecasting strategy. Currently supports "recursive", "direct",
        and "ensemble" of both recursive and direct strategies.
    """

    model = "linear_model"


class Ridge(BaseForecaster):
    """Ridge regression forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    lags : int
        Number of lagged target variables.
    max_horizons: Optional[int]
        Maximum number of horizons to predict directly.
        Only applied if `strategy` equals "direct" or "ensemble".
    strategy : Optional[str]
        Forecasting strategy. Currently supports "recursive", "direct",
        and "ensemble" of both recursive and direct strategies.
    """

    model = "ridge"
