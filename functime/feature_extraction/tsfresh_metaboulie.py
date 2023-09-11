from typing import Dict, List

import numpy as np
import polars as pl
from statsmodels.tsa.ar_model import AutoReg


def symmetry_looking(x: pl.Series, param: List[Dict[str, float]]) -> pl.DataFrame:
    """
    Boolean variable denoting if the distribution of x *looks symmetric*. This is the case if

    .. math::

        | mean(X)-median(X)| < r * (max(X)-min(X))

    :param x: the time series to calculate the feature of
    :type x: pl.Series
    :param param: contains dictionaries {"r": x} with x (float) is the percentage of the range to compare with
    :type param: list
    :return: the value of this feature with different r values
    :return type: pl.DataFrame
    """
    mean_median_difference = abs(x.mean() - x.median())
    max_min_difference = x.max() - x.min()
    result = pl.DataFrame(
        {
            "r": [r["r"] for r in param],
            "feature_value": [
                mean_median_difference < r["r"] * max_min_difference for r in param
            ],
        }
    )
    return result


def time_reversal_asymmetry_statistic(x: pl.Series, lag: int) -> float:
    """
    Returns the time reversal asymmetry statistic.

    This function calculates the value of

    .. math::

        \\frac{1}{n-2lag} \\sum_{i=1}^{n-2lag} x_{i + 2 \\cdot lag}^2 \\cdot x_{i + lag} - x_{i + lag} \\cdot  x_{i}^2

    which is

    .. math::

        \\mathbb{E}[L^2(X)^2 \\cdot L(X) - L(X) \\cdot X^2]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a
    promising feature to extract from time series.

    .. rubric:: References

    |  [1] Fulcher, B.D., Jones, N.S. (2014).
    |  Highly comparative feature-based time-series classification.
    |  Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.

    :param x: the time series to calculate the feature of
    :type x: pl.Series
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    n = len(x)
    if 2 * lag >= n:
        raise ValueError("lag must be less than n/2")
    else:
        one_lag = x.shift(-lag)
        two_lag = x.shift(-2 * lag)
        result = (
            (two_lag * two_lag * one_lag - one_lag * x * x).head(n - 2 * lag).mean()
        )

    return result


def ar_coefficient(x: pl.Series, param: List[Dict[str, int]]) -> pl.DataFrame:
    """
    This feature calculator fits the unconditional maximum likelihood
    of an autoregressive AR(k) process.
    The k parameter is the maximum lag of the process

    .. math::

        X_{t}=\\varphi_0 +\\sum _{{i=1}}^{k}\\varphi_{i}X_{{t-i}}+\\varepsilon_{t}

    For the configurations from param which should contain the maxlag "k" and such an AR process is calculated. Then
    the coefficients :math:`\\varphi_{i}` whose index :math:`i` contained from "coeff" are returned.

    :param x: the time series to calculate the feature of
    :type x: pl.Series
    :param param: contains dictionaries {"coeff": x, "k": y} with x,y int
    :type param: list
    :return x: the different feature values
    :return type: pandas.Series
    """
    calculated_ar_params = {}
    res = {}

    for parameter_combination in param:
        k = parameter_combination["k"]
        p = parameter_combination["coeff"]

        column_name = f"coeff_{p}__k_{k}"

        if k not in calculated_ar_params:
            try:
                calculated_AR = AutoReg(x.to_list(), lags=k, trend="c")
                calculated_ar_params[k] = calculated_AR.fit().params
            except (ZeroDivisionError, np.linalg.LinAlgError, ValueError):
                calculated_ar_params[k] = [np.NaN] * k

        mod = calculated_ar_params[k]
        if p <= k:
            try:
                res[column_name] = mod[p]
            except IndexError:
                res[column_name] = 0
        else:
            res[column_name] = np.NaN

    return pl.DataFrame(data=res)
