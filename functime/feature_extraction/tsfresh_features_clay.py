import polars as pl
from collections import defaultdict
import numpy as np
from statsmodels.tsa.stattools import pacf

def binned_entropy(x: pl.Series, max_bins: int = None) -> float:
    """
    Documentation and code adapted from TSFresh (https://github.com/blue-yonder/tsfresh)

    First bins the values of x into max_bins equidistant bins.
    Then calculates the value of

    .. math::

        - \\sum_{k=0}^{min(max\\_bins, len(x))} p_k log(p_k) \\cdot \\mathbf{1}_{(p_k > 0)}

    where :math:`p_k` is the percentage of samples in bin :math:`k`.

    :param x: the time series to calculate the feature of
    :type x: polars.Series
    :param max_bins: the maximal number of bins
    """
    # Can't calculate entropy of if there are null values
    if x.is_null().any():
        return pl.lit(None)

    hist_df = x.hist(bin_count=max_bins)
    
    return hist_df.select( (pl.col('^.*_count$') / x.len())).filter(
        pl.col('^.*_count$') > 0
    ).to_series().entropy()
    

# Not implemented, as we decided to avoid FFT features for now
def fourier_entropy(x: pl.Series) -> float:
    return None


def _estimate_friedrich_coefficients(x, m, r):
    """
    Documentation and code adapted from TSFresh (https://github.com/blue-yonder/tsfresh)

    Coefficients of polynomial :math:`h(x)`, which has been fitted to
    the deterministic dynamics of Langevin model
    .. math::
        \\dot{x}(t) = h(x(t)) + \\mathcal{N}(0,R)

    As described by

        Friedrich et al. (2000): Physics Letters A 271, p. 217-222
        *Extracting model equations from experimental data*

    For short time-series this method is highly dependent on the parameters.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param m: order of polynomial to fit for estimating fixed points of dynamics
    :type m: int
    :param r: number of quantiles to use for averaging
    :type r: float

    :return: coefficients of polynomial of deterministic dynamics
    :return type: ndarray
    """
    assert m > 0, "Order of polynomial need to be positive integer, found {}".format(m)

    df = pl.DataFrame({"signal": x[:-1], "delta": np.diff(x)})
    try:
        #df["quantiles"] = pd.qcut(df.signal, r)
        df = df.with_columns(pl.col('signal').qcut(r).alias('quantiles'))
    except (ValueError, IndexError, pl.exceptions.DuplicateError):
        return [np.NaN] * (m + 1)

    result = df.groupby("quantiles").agg([
        pl.col("signal").mean().alias("x_mean"),
        pl.col("delta").mean().alias("y_mean")
        ]).drop_nulls()

    try:
        return np.polyfit(result["x_mean"], result["y_mean"], deg=m)
    except (np.linalg.LinAlgError, ValueError):
        return [np.NaN] * (m + 1)


def friedrich_coefficients(x: pl.Series, params: list) -> pl.Series:
    """
    Documentation and code adapted from TSFresh (https://github.com/blue-yonder/tsfresh)

    Coefficients of polynomial :math:`h(x)`, which has been fitted to
    the deterministic dynamics of Langevin model

    .. math::
        \\dot{x}(t) = h(x(t)) + \\mathcal{N}(0,R)

    as described by [1].

    For short time-series this method is highly dependent on the parameters.

    .. rubric:: References

    |  [1] Friedrich et al. (2000): Physics Letters A 271, p. 217-222
    |  *Extracting model equations from experimental data*

    :params x: the time series to calculate the feature of
    :type x: polars.Series
    :param params: contains dictionaries {"m": x, "r": y, "coeff": z} with x being positive integer,
                  the order of polynomial to fit for estimating fixed points of
                  dynamics, y positive float, the number of quantiles to use for averaging and finally z,
                  a positive integer corresponding to the returned coefficient
    :type params: list
    :return: the different feature values
    :return type: polars.Series of type Struct
    """
    # calculated is dictionary storing the calculated coefficients {m: {r: friedrich_coefficients}}
    calculated = defaultdict(dict)
    # res is a dictionary containing the results {"m_10__r_2__coeff_3": 15.43}
    res = {}

    for parameter_combination in params:
        m = parameter_combination["m"]
        r = parameter_combination["r"]
        coeff = parameter_combination["coeff"]

        assert coeff >= 0, "Coefficients must be positive or zero. Found {}".format(
            coeff
        )

        # calculate the current friedrich coefficients if they do not exist yet
        if m not in calculated or r not in calculated[m]:
            calculated[m][r] = _estimate_friedrich_coefficients(x.to_numpy(), m, r)

        try:
            res["coeff_{}__m_{}__r_{}".format(coeff, m, r)] = calculated[m][r][coeff]
        except IndexError:
            res["coeff_{}__m_{}__r_{}".format(coeff, m, r)] = pl.lit(None)
    return pl.Series([(key, value) for key, value in res.items()])


def lempel_ziv_complexity(x: pl.Series, num_bins: int) -> float:
    """
    Documentation and code adapted from TSFresh (https://github.com/blue-yonder/tsfresh)

    Calculate a complexity estimate based on the Lempel-Ziv compression
    algorithm.

    The complexity is defined as the number of dictionary entries (or sub-words) needed
    to encode the time series when viewed from left to right.
    For this, the time series is first binned into the given number of bins.
    Then it is converted into sub-words with different prefixes.
    The number of sub-words needed for this divided by the length of the time
    series is the complexity estimate.

    For example, if the time series (after binning in only 2 bins) would look like "100111",
    the different sub-words would be 1, 0, 01 and 11 and therefore the result is 4/6 = 0.66.

    Ref: https://github.com/Naereen/Lempel-Ziv_Complexity/blob/master/src/lempel_ziv_complexity.py

    """
    bins = pl.Series('seq', np.linspace(x.min(), x.max(), num_bins + 1)[1:].tolist())
    sequence = bins.search_sorted(x, side="left")

    sub_strings = set()
    n = sequence.len()

    ind = 0
    inc = 1
    while ind + inc <= n:
        # convert to tuple in order to make it hashable
        sub_str = tuple(sequence[ind : ind + inc])
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings) / n


def max_langevin_fixed_point(x: pl.Series, r: int, m: int) -> float:
    """
    Documentation and code adapted from TSFresh (https://github.com/blue-yonder/tsfresh)

    Largest fixed point of dynamics  :math:argmax_x {h(x)=0}` estimated from polynomial :math:`h(x)`,
    which has been fitted to the deterministic dynamics of Langevin model

    .. math::
        \\dot(x)(t) = h(x(t)) + R \\mathcal(N)(0,1)

    as described by

        Friedrich et al. (2000): Physics Letters A 271, p. 217-222
        *Extracting model equations from experimental data*

    For short time-series this method is highly dependent on the parameters.

    :param x: the time series to calculate the feature of
    :type x: polars.Series
    :param m: order of polynomial to fit for estimating fixed points of dynamics
    :type m: int
    :param r: number of quantiles to use for averaging
    :type r: float

    :return: Largest fixed point of deterministic dynamics
    :return type: float
    """

    coeff = _estimate_friedrich_coefficients(x.to_numpy(), m, r)

    try:
        max_fixed_point = np.max(np.real(np.roots(coeff)))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

    return max_fixed_point


def partial_autocorrelation(x: pl.Series, lags: dict) -> pl.Series:
    """
    Documentation and code adapted from TSFresh (https://github.com/blue-yonder/tsfresh)

    Calculates the value of the partial autocorrelation function at the given lag.

    The lag `k` partial autocorrelation of a time series :math:`\\lbrace x_t, t = 1 \\ldots T \\rbrace` equals the
    partial correlation of :math:`x_t` and :math:`x_{t-k}`, adjusted for the intermediate variables
    :math:`\\lbrace x_{t-1}, \\ldots, x_{t-k+1} \\rbrace` ([1]).

    Following [2], it can be defined as

    .. math::

        \\alpha_k = \\frac{ Cov(x_t, x_{t-k} | x_{t-1}, \\ldots, x_{t-k+1})}
        {\\sqrt{ Var(x_t | x_{t-1}, \\ldots, x_{t-k+1}) Var(x_{t-k} | x_{t-1}, \\ldots, x_{t-k+1} )}}

    with (a) :math:`x_t = f(x_{t-1}, \\ldots, x_{t-k+1})` and (b) :math:`x_{t-k} = f(x_{t-1}, \\ldots, x_{t-k+1})`
    being AR(k-1) models that can be fitted by OLS. Be aware that in (a), the regression is done on past values to
    predict :math:`x_t` whereas in (b), future values are used to calculate the past value :math:`x_{t-k}`.
    It is said in [1] that "for an AR(p), the partial autocorrelations [ :math:`\\alpha_k` ] will be nonzero for `k<=p`
    and zero for `k>p`."
    With this property, it is used to determine the lag of an AR-Process.

    .. rubric:: References

    |  [1] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
    |  Time series analysis: forecasting and control. John Wiley & Sons.
    |  [2] https://onlinecourses.science.psu.edu/stat510/node/62

    :param x: the time series to calculate the feature of
    :type x: polars.Series
    :param lags: contains dictionaries {"lag": val} with int val indicating the lag to be returned
    :type param: list
    :return: the value of this feature
    :return type: float
    """
    # Check the difference between demanded lags by param and possible lags to calculate (depends on len(x))
    max_demanded_lag = max([lag["lag"] for lag in lags])
    n = x.len()

    # Check if list is too short to make calculations
    if n <= 1:
        pacf_coeffs = [np.nan] * (max_demanded_lag + 1)
    else:
        # https://github.com/statsmodels/statsmodels/pull/6846
        # PACF limits lag length to 50% of sample size.
        if max_demanded_lag >= n // 2:
            max_lag = n // 2 - 1
        else:
            max_lag = max_demanded_lag
        if max_lag > 0:
            pacf_coeffs = list(pacf(x.to_numpy(), method="ld", nlags=max_lag))
            pacf_coeffs = pacf_coeffs + [np.nan] * max(0, (max_demanded_lag - max_lag))
        else:
            pacf_coeffs = [np.nan] * (max_demanded_lag + 1)

    return pl.Series([("lag_{}".format(lag["lag"]), pacf_coeffs[lag["lag"]]) for lag in lags])