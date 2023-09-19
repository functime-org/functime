from collections import defaultdict
from math import ceil
from typing import Dict, List, Union

import polars as pl
from scipy.signal import cwt, ricker
from scipy.stats import linregress


def _aggregate_on_chunks(x: pl.Series, f_agg: str, chunk_len: int) -> list:
    """
    Takes the time series x and constructs a lower sampled version of it by applying the aggregation function f_agg on
    consecutive chunks of length chunk_len

    :param x: the time series to calculate the aggregation of
    :type x: polars.Series
    :param f_agg: The name of the aggregation function that should be an attribute of the polars.Series
    :type f_agg: str
    :param chunk_len: The size of the chunks where to aggregate the time series
    :type chunk_len: int
    :return: A list of the aggregation function over the chunks
    :return type: list
    """
    return [
        getattr(x[i * chunk_len : (i + 1) * chunk_len], f_agg)()
        for i in range(ceil(len(x) / chunk_len))
    ]


def agg_linear_trend(
    x: pl.Series, param: List[Dict[str, Union[str, int, str]]]
) -> pl.DataFrame:
    """
    Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
    the sequence from 0 up to the number of chunks minus one.

    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.

    The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
    "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.

    The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.

    Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

    :param x: the time series to calculate the feature of
    :type x: polars.Series
    :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
    :type param: list
    :return: the different feature values
    :return type: polars.DataFrame
    """

    calculated_agg = defaultdict(dict)
    res_data = []
    res_index = []

    for parameter_combination in param:
        chunk_len = parameter_combination["chunk_len"]
        f_agg = parameter_combination["f_agg"]

        if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
            if chunk_len >= len(x):
                calculated_agg[f_agg][chunk_len] = float("nan")
            else:
                aggregate_result = _aggregate_on_chunks(x, f_agg, chunk_len)
                lin_reg_result = linregress(
                    range(len(aggregate_result)), aggregate_result
                )
                calculated_agg[f_agg][chunk_len] = lin_reg_result

        attr = parameter_combination["attr"]

        if chunk_len >= len(x):
            res_data.append(float("nan"))
        else:
            res_data.append(getattr(calculated_agg[f_agg][chunk_len], attr))

        res_index.append(
            'attr_"{}"__chunk_len_{}__f_agg_"{}"'.format(attr, chunk_len, f_agg)
        )

        res = [res_index, res_data]

    return pl.DataFrame(res, schema=["res_index", "res_data"], orient="col")


def ar_coefficient(
    x: pl.Series, param: List[Dict[str, Union[int, int]]]
) -> pl.DataFrame:
    """
    This feature calculator fits the unconditional maximum likelihood
    of an autoregressive AR(k) process.
    The k parameter is the maximum lag of the process

    .. math::

        X_{t}=\\varphi_0 +\\sum _{{i=1}}^{k}\\varphi_{i}X_{{t-i}}+\\varepsilon_{t}

    For the configurations from param which should contain the maxlag "k" and such an AR process is calculated. Then
    the coefficients :math:`\\varphi_{i}` whose index :math:`i` contained from "coeff" are returned.

    :param x: the time series to calculate the feature of
    :type x: polars.Series
    :param param: contains dictionaries {"coeff": x, "k": y} with x,y int
    :type param: list
    :return x: the different feature values
    :return type: polars.DataFrame
    """
    return NotImplemented


def augmented_dickey_fuller(
    x: pl.Series, param: List[Dict[str, Union[str, str]]]
) -> pl.DataFrame:
    """
    Does the time series have a unit root?

    The Augmented Dickey-Fuller test is a hypothesis test which checks whether a unit root is present in a time
    series sample. This feature calculator returns the value of the respective test statistic.

    See the statsmodels implementation for references and more details.

    :param x: the time series to calculate the feature of
    :type x: polars.Series
    :param param: contains dictionaries {"attr": x, "autolag": y} with x str, either "teststat", "pvalue" or "usedlag"
                  and with y str, either of "AIC", "BIC", "t-stats" or None (See the documentation of adfuller() in
                  statsmodels).
    :type param: list
    :return: the value of this feature
    :return type: polars.DataFrame
    """
    return NotImplemented


def cwt_coefficients(
    x: pl.Series, param: List[Dict[str, Union[List[int], int, int]]]
) -> pl.DataFrame:
    """
    Calculates a Continuous wavelet transform for the Ricker wavelet, also known as the "Mexican hat wavelet" which is
    defined by

    .. math::
        \\frac{2}{\\sqrt{3a} \\pi^{\\frac{1}{4}}} (1 - \\frac{x^2}{a^2}) exp(-\\frac{x^2}{2a^2})

    where :math:`a` is the width parameter of the wavelet function.

    This feature calculator takes three different parameter: widths, coeff and w. The feature calculator takes all the
    different widths arrays and then calculates the cwt one time for each different width array. Then the values for the
    different coefficient for coeff and width w are returned. (For each dic in param one feature is returned)

    :param x: the time series to calculate the feature of
    :type x: polars.Series
    :param param: contains dictionaries {"widths":x, "coeff": y, "w": z} with x array of int and y,z int
    :type param: list
    :return: the different feature values
    :return type: polars.DataFrame
    """
    calculated_cwt = {}
    res = []
    indices = []

    for parameter_combination in param:
        widths = tuple(parameter_combination["widths"])
        w = parameter_combination["w"]
        coeff = parameter_combination["coeff"]

        if widths not in calculated_cwt:
            calculated_cwt[widths] = cwt(x, ricker, widths)

        calculated_cwt_for_widths = calculated_cwt[widths]

        indices += [f"coeff_{coeff}__w_{w}__widths_{widths}"]

        i = widths.index(w)
        if calculated_cwt_for_widths.shape[1] <= coeff:
            res += [float("nan")]
        else:
            res += [calculated_cwt_for_widths[i, coeff]]

    data = [indices, res]

    return pl.DataFrame(data, schema=["index", "res"], orient="col")


def mean_second_derivative_central(x: pl.Series) -> float:
    """
    Returns the mean value of a central approximation of the second derivative

    .. math::

        \\frac{1}{2(n-2)} \\sum_{i=1,\\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)

    :param x: the time series to calculate the feature of
    :type x: polars.Series
    :return: the value of this feature
    :return type: float | float("nan")
    """
    return (
        (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2))
        if len(x) > 2
        else float("nan")
    )


def symmetry_looking(x: pl.Series, param: List[Dict[str, float]]) -> pl.DataFrame:
    """
    Boolean variable denoting if the distribution of x *looks symmetric*. This is the case if

    .. math::

        | mean(X)-median(X)| < r * (max(X)-min(X))

    :param x: the time series to calculate the feature of
    :type x: polars.Series
    :param param: contains dictionaries {"r": x} with x (float) is the percentage of the range to compare with
    :type param: list
    :return: the value of this feature with different r values
    :return type: polars.DataFrame
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
    :type x: polars.Series
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
