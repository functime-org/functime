import polars as pl


def change_quantiles(
    x: pl.Expr, ql: float, qh: float, isabs: bool, f_agg: str
) -> pl.Expr:
    """
    First fixes a corridor given by the quantiles ql and qh of the distribution of x.
    Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

    Think about selecting a corridor on the
    y-Axis and only calculating the mean of the absolute change of the time series inside this corridor.

    Parameters
    ----------
    x: pl.Expr
        the time series to calculate the feature of
    ql : float
        the lower quantile of the corridor
        Must be less than `qh`.
    qh : float
        the upper quantile of the corridor
        Must be greater than `ql`.
    isabs : bool
        should the absolute differences be taken instead?
    f_agg : str
        the aggregator function that is applied to the differences in the bin
    """
    if isabs:
        x = (
            x.diff()
            .abs()
            .filter(
                x.is_between(
                    x.quantile(ql, interpolation="linear"),
                    x.quantile(qh, interpolation="linear"),
                ).and_(
                    x.is_between(
                        x.quantile(ql, interpolation="linear"),
                        x.quantile(qh, interpolation="linear"),
                    ).shift_and_fill(fill_value=False, periods=1)
                )
            )
        )
    else:
        x = x.diff().filter(
            x.is_between(
                x.quantile(ql, interpolation="linear"),
                x.quantile(qh, interpolation="linear"),
            ).and_(
                x.is_between(
                    x.quantile(ql, interpolation="linear"),
                    x.quantile(qh, interpolation="linear"),
                ).shift_and_fill(fill_value=False, periods=1)
            )
        )
    if f_agg == "std":
        return getattr(x, f_agg)(ddof=0).fill_null(0.0)
    else:
        return getattr(x, f_agg)().fill_null(0.0)


def mean_abs_change(x: pl.Expr) -> pl.Expr:
    """
    Compute mean absolute change.

    Parameters
    ----------
    x : pl.Expr
        The time series to compute the feature of.
    """
    return x.diff(null_behavior="drop").abs().mean()


def mean_change(x: pl.Expr) -> pl.Expr:
    """
    Compute mean change.

    Parameters
    ----------
    x : pl.Expr
        The time series to compute the feature of.
    """
    return x.diff(null_behavior="drop").mean()


def number_crossing_m(x: pl.Expr, m: float) -> pl.Expr:
    """
    Calculates the number of crossings of x on m. A crossing is defined as two sequential values where the first value
    is lower than m and the next is greater, or vice-versa. If you set m to zero, you will get the number of zero
    crossings.

    Parameters
    ----------
    x : pl.Expr
        the time series to calculate the feature of.
    m : float
        the crossing value.
    """
    return x.gt(m).cast(pl.Int8).diff(null_behavior="drop").abs().eq(1).sum()


def var_greater_than_std(x: pl.Expr) -> pl.Expr:
    """
    Is variance higher than the standard deviation?

    Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
    being larger than 1

    Parameters
    ----------
    x : pl.Expr
        the time series to calculate the feature of
    """
    y = x.var(ddof=0)
    return y > y.sqrt()


def first_location_of_maximum(x: pl.Expr) -> pl.Expr:
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr
        the time series to calculate the feature of
    """
    return x.arg_max() / x.len()


def last_location_of_maximum(x: pl.Expr) -> pl.Expr:
    """
    Returns the last location of the maximum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr
        the time series to calculate the feature of
    """
    return (x.len() - x.reverse().arg_max()) / x.len()


def first_location_of_minimum(x: pl.Expr) -> pl.Expr:
    """
    Returns the first location of the minimum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr
        the time series to calculate the feature of
    """
    return x.arg_min() / x.len()


def last_location_of_minimum(x: pl.Expr) -> pl.Expr:
    """
    Returns the last location of the minimum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr
        the time series to calculate the feature of
    """
    return (x.len() - x.reverse().arg_min()) / x.len()


def autocorrelation(x: pl.Expr, lag: int) -> pl.Expr:
    """Calculate the autocorrelation of a Polars Expression for a specified lag.

    The autocorrelation measures the linear dependence between a timeseries and a lagged version of itself.

    Parameters
    ----------
    x : pl.Expr
        The Polars Expression for which the autocorrelation will be calculated.
    lag : int
        The lag at which to calculate the autocorrelation. Must be a non-negative integer.

    Returns
    -------
    pl.Expr
        A Polars Expression representing the autocorrelation at the given lag. If `lag` is 0, a constant
        expression with a value of 1.0 is returned.

    Raises
    ------
    Exception
        If `lag` is a negative number, an exception is raised.

    Examples
    --------
    >>> import polars as pl
    >>> data = pl.DataFrame({'A': [1, 2, 3, 2, 4]})
    >>> expr = data.select(pl.autocorrelation(data['A'], 1).alias('autocorr'))
    >>> print(expr)
       autocorr
    0  0.214286

    Notes
    -----
    - This function calculates the autocorrelation using https://en.wikipedia.org/wiki/Autocorrelation#Estimation
    - If `lag` is 0, the autocorrelation is always 1.0, as it represents the correlation of the timeseries with itself.

    See Also
    --------
    pl.Expr.shift : Shift a Polars Expression by a specified number of periods.
    pl.Expr.drop_nulls : Remove rows with null values from a Polars Expression.
    pl.Expr.sub : Subtract a value or Expression from a Polars Expression.
    pl.Expr.dot : Calculate the dot product of two Polars Expressions.
    pl.Expr.truediv : Perform element-wise true division on a Polars Expression.
    pl.Expr.count : Count the non-null elements in a Polars Expression.
    pl.Expr.var : Calculate the variance of a Polars Expression.

    """
    if lag < 0:
        raise Exception("Lag cannot be a negative number")

    if lag == 0:
        return pl.lit(1.0)

    return (
        x.shift(periods=-lag)
        .drop_nulls()
        .sub(x.mean())
        .dot(x.shift(periods=lag).drop_nulls().sub(x.mean()))
        .truediv((x.count() - lag).mul(x.var(ddof=0)))
    )


def count_below(x: pl.Expr, t: float) -> pl.Expr:
    """Calculate the percentage of values below or equal to a threshold in a Polars Expression.

    This function computes the percentage of values in the input Polars Expression `x` that are less than
    or equal to the specified threshold `t`.

    Parameters
    ----------
    x : pl.Expr
        The Polars Expression containing the values to be counted.
    t : float
        The threshold value for comparison.

    Returns
    -------
    pl.Expr
        A Polars Expression representing the percentage of values in `x` that are less than or equal to `t`.

    Examples
    --------
    >>> import polars as pl
    >>> data = pl.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7]})
    >>> expr = count_below(data['A'], 4)
    >>> print(expr)
    57.14285714285714

    Notes
    -----
    - This function filters the values in the input Polars Expression `x` using the condition `x <= t`, counts
      the number of values that satisfy the condition, and then computes the percentage relative to the total
      number of values in `x`.
    - The result is expressed as a percentage, which is a floating-point number between 0 and 100.

    See Also
    --------
    pl.Expr.filter : Filter a Polars Expression based on a condition.
    pl.Expr.count : Count the number of elements in a Polars Expression.
    pl.Expr.truediv : Perform element-wise true division on a Polars Expression.
    pl.Expr.mul : Multiply a Polars Expression by a scalar or another Expression.

    """
    return x.filter(x <= t).count().truediv(x.count()).mul(100)


def count_above(x: pl.Expr, t: float) -> pl.Expr:
    """Calculate the percentage of values above or equal to a threshold in a Polars Expression.

    This function computes the percentage of values in the input Polars Expression `x` that are greater than
    or equal to the specified threshold `t`.

    Parameters
    ----------
    x : pl.Expr
        The Polars Expression containing the values to be counted.
    t : float
        The threshold value for comparison.

    Returns
    -------
    pl.Expr
        A Polars Expression representing the percentage of values in `x` that are greater than or equal to `t`.

    Examples
    --------
    >>> import polars as pl
    >>> data = pl.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7]})
    >>> expr = count_above(data['A'], 4)
    >>> print(expr)
    42.857142857142854

    Notes
    -----
    - This function filters the values in the input Polars Expression `x` using the condition `x >= t`, counts
      the number of values that satisfy the condition, and then computes the percentage relative to the total
      number of values in `x`.
    - The result is expressed as a percentage, which is a floating-point number between 0 and 100.

    See Also
    --------
    pl.Expr.filter : Filter a Polars Expression based on a condition.
    pl.Expr.count : Count the number of elements in a Polars Expression.
    pl.Expr.truediv : Perform element-wise true division on a Polars Expression.
    pl.Expr.mul : Multiply a Polars Expression by a scalar or another Expression.

    """
    return x.filter(x >= t).count().truediv(x.count()).mul(100)


def count_below_mean(x: pl.Expr) -> pl.Expr:
    """Count the number of values in a Polars Expression that are below the mean.

    This function filters the values in the input Polars Expression `x` and counts the number of values that are
    less than the mean of the expression.

    Parameters
    ----------
    x : pl.Expr
        The Polars Expression containing the values to be counted.

    Returns
    -------
    pl.Expr
        A Polars Expression representing the count of values in `x` that are below the mean.

    Examples
    --------
    >>> import polars as pl
    >>> data = pl.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7]})
    >>> expr = count_below_mean(data['A'])
    >>> print(expr)
    3

    Notes
    -----
    - This function filters the values in the input Polars Expression `x` using the condition `x < x.mean()`,
      and then counts the number of values that satisfy the condition.
    - The result is an integer representing the count of values below the mean.

    See Also
    --------
    pl.Expr.filter : Filter a Polars Expression based on a condition.
    pl.Expr.count : Count the number of elements in a Polars Expression.
    pl.Expr.mean : Calculate the mean (average) of a Polars Expression.

    """
    return x.filter(x < x.mean()).count()


def count_above_mean(x: pl.Expr) -> pl.Expr:
    """Count the number of values in a Polars Expression that are above the mean.

    This function filters the values in the input Polars Expression `x` and counts the number of values that are
    greater than the mean of the expression.

    Parameters
    ----------
    x : pl.Expr
        The Polars Expression containing the values to be counted.

    Returns
    -------
    pl.Expr
        A Polars Expression representing the count of values in `x` that are above the mean.

    Examples
    --------
    >>> import polars as pl
    >>> data = pl.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7]})
    >>> expr = count_above_mean(data['A'])
    >>> print(expr)
    3

    Notes
    -----
    - This function filters the values in the input Polars Expression `x` using the condition `x > x.mean()`,
      and then counts the number of values that satisfy the condition.
    - The result is an integer representing the count of values above the mean.

    See Also
    --------
    pl.Expr.filter : Filter a Polars Expression based on a condition.
    pl.Expr.count : Count the number of elements in a Polars Expression.
    pl.Expr.mean : Calculate the mean (average) of a Polars Expression.

    """
    return x.filter(x > x.mean()).count()


def has_duplicate(x: pl.Expr) -> pl.Expr:
    """Check if a Polars Expression contains any duplicate values.

    This function checks whether the input Polars Expression `x` contains any duplicate values.

    Parameters
    ----------
    x : pl.Expr
        The Polars Expression to be checked for duplicates.

    Returns
    -------
    pl.Expr
        A boolean Polars Expression indicating whether there are duplicate values in `x`.
        Returns True if duplicates exist, otherwise False.

    Examples
    --------
    >>> import polars as pl
    >>> data = pl.DataFrame({'A': [1, 2, 2, 3, 4, 4]})
    >>> expr = has_duplicate(data['A'])
    >>> print(expr)
    True

    Notes
    -----
    - This function uses the `is_duplicated` method to identify duplicate values within the input Polars Expression.
    - The result is a boolean expression, where True indicates the presence of duplicates and False indicates no duplicates.

    See Also
    --------
    pl.Expr.is_duplicated : Check for duplicate values in a Polars Expression.
    pl.Expr.any : Check if any elements in a boolean Polars Expression are True.

    """
    return x.is_duplicated().any()


# helper function should not be exposed
def _has_duplicate_of_value(x: pl.Expr, t: float) -> pl.Expr:
    """Check if a value exists as a duplicate in a Polars Series.

    Parameters
    ----------
    x : pl.Series
        The input Polars Series to search for duplicates in.
    t : float
        The value to check for duplicates of within the Series.

    Returns
    -------
    bool
        Returns True if duplicates of the specified value `t` exist in the
        input Series `x`, otherwise returns False.

    Examples
    --------
    >>> import polars as pl
    >>> data = pl.DataFrame({'A': [1, 2, 3, 2, 4]})
    >>> series = data['A']
    >>> result = _has_duplicate_of_value(series, 2)
    >>> print(result)
    True

    See Also
    --------
    pl.Series.filter : Filter a Series using a boolean expression.
    pl.Series.is_duplicated : Check for duplicate values in a Series.
    pl.Series.any : Check if any elements in a boolean Series are True.

    """
    return x.filter(x == t).is_duplicated().any()


def has_duplicate_max(x: pl.Expr) -> pl.Expr:
    """Check if a Polars Expression contains duplicate values equal to its maximum value.

    This function checks whether the input Polars Expression `x` contains any duplicate values equal to its maximum value.

    Parameters
    ----------
    x : pl.Expr
        The Polars Expression to be checked for duplicates.

    Returns
    -------
    pl.Expr
        A boolean Polars Expression indicating whether there are duplicate values in `x` equal to its maximum value.
        Returns True if such duplicates exist, otherwise False.

    Examples
    --------
    >>> import polars as pl
    >>> data = pl.DataFrame({'A': [1, 2, 2, 3, 4, 4]})
    >>> expr = has_duplicate_max(data['A'])
    >>> print(expr)
    True

    Notes
    -----
    - This function first calculates the maximum value in the input Polars Expression `x` using the `max` method.
    - It then checks for duplicates of that maximum value using the `_has_duplicate_of_value` function.
    - The result is a boolean expression, where True indicates the presence of such duplicates and False indicates none.

    See Also
    --------
    _has_duplicate_of_value : Check for duplicate values equal to a specified value in a Polars Expression.
    pl.Expr.max : Calculate the maximum value in a Polars Expression.
    pl.Expr.is_duplicated : Check for duplicate values in a Polars Expression.
    pl.Expr.any : Check if any elements in a boolean Polars Expression are True.

    """
    return _has_duplicate_of_value(x, x.max())


def has_duplicate_min(x: pl.Expr) -> pl.Expr:
    """Check if a Polars Expression contains duplicate values equal to its minimum value.

    This function checks whether the input Polars Expression `x` contains any duplicate values equal to its minimum value.

    Parameters
    ----------
    x : pl.Expr
        The Polars Expression to be checked for duplicates.

    Returns
    -------
    pl.Expr
        A boolean Polars Expression indicating whether there are duplicate values in `x` equal to its minimum value.
        Returns True if such duplicates exist, otherwise False.

    Examples
    --------
    >>> import polars as pl
    >>> data = pl.DataFrame({'A': [1, 2, 2, 3, 4, 4]})
    >>> expr = has_duplicate_min(data['A'])
    >>> print(expr)
    True

    Notes
    -----
    - This function first calculates the minimum value in the input Polars Expression `x` using the `min` method.
    - It then checks for duplicates of that minimum value using the `_has_duplicate_of_value` function.
    - The result is a boolean expression, where True indicates the presence of such duplicates and False indicates none.

    See Also
    --------
    _has_duplicate_of_value : Check for duplicate values equal to a specified value in a Polars Expression.
    pl.Expr.min : Calculate the minimum value in a Polars Expression.
    pl.Expr.is_duplicated : Check for duplicate values in a Polars Expression.
    pl.Expr.any : Check if any elements in a boolean Polars Expression are True.

    """
    return _has_duplicate_of_value(x, x.min())


def benfords_correlation(x: pl.Series) -> float:
    """
    Returns the correlation between the first digit distribution of the input time series and the Newcomb-Benford's Law
    distribution [1][2].

    Parameters
    ----------
    x : pl.Series
        The time series to calculate the feature of.

    Returns
    -------
    float
        The value of this feature.

    Notes
    -----
    The Newcomb-Benford distribution for d that is the leading digit of the number {1, 2, 3, 4, 5, 6, 7, 8, 9} is given by:

    .. math::

        P(d) = \\log_{10}\\left(1 + \\frac{1}{d}\\right)

    References
    ----------
    [1] Hill, T. P. (1995). A Statistical Derivation of the Significant-Digit Law. Statistical Science.
    [2] Hill, T. P. (1995). The significant-digit phenomenon. The American Mathematical Monthly.
    [3] Benford, F. (1938). The law of anomalous numbers. Proceedings of the American philosophical society.
    [4] Newcomb, S. (1881). Note on the frequency of use of the different digits in natural numbers. American Journal of
        mathematics.
    """
    X = (x / (10 ** x.abs().log10().floor())).abs().floor()
    df_corr = pl.DataFrame(
        [
            [X.eq(i).mean() for i in pl.int_range(1, 10, eager=True)],
            (1 + 1 / pl.int_range(1, 10, eager=True)).log10(),
        ]
    ).corr()
    return df_corr[0, 1]


def _get_length_sequences_where(x: pl.Series) -> pl.Series:
    """
    Calculates the length of all sub-sequences where the series x is either True or 1.

    Parameters
    ----------
    x : pl.Series
        A pl.Series containing only 1, True, 0 and False values.

    Returns
    -------
    pl.Series
        A pl.Series with the length of all sub-sequences where the series is either True or False. If no ones or Trues
        contained, the list [0] is returned.

    Examples
    --------
    >>> x = pl.Series([0,1,0,0,1,1,1,0,0,1,0,1,1])
    >>> _get_length_sequences_where(x)
    >>> shape: (4,)
        Series: 'count' [u32]
        [
            2
            1
            1
            3
        ]

    >>> x = pl.Series([0,True,0,0,True,True,True,0,0,True,0,True,True])
    >>> _get_length_sequences_where(x)
    >>> shape: (4,)
        Series: 'count' [u32]
        [
            1
            2
            1
            3
        ]
    """
    X = (
        x.alias("orig")
        .to_frame()
        .with_columns(shift=pl.col("orig").shift(periods=1))
        .with_columns(mask=pl.col("orig").ne(pl.col("shift")).fill_null(0).cumsum())
        .filter(pl.col("orig") == 1)
        .group_by(pl.col("mask"), maintain_order=True)
        .count()
    )["count"]
    return X


def longest_strike_below_mean(x: pl.Series) -> float:
    """
    Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x.

    Parameters
    ----------
    x : pl.Series
        The time series to calculate the feature of.

    Returns
    -------
    float
        The value of this feature.
    """
    if not x.is_empty():
        X = _get_length_sequences_where(x.cast(pl.Float64) < x.mean())
    else:
        return 0
    return X.max() if X.len() > 0 else 0


def longest_strike_above_mean(x: pl.Series) -> float:
    """
    Returns the length of the longest consecutive subsequence in x that is greater than the mean of x.

    Parameters
    ----------
    x : pl.Series
        The time series to calculate the feature of.

    Returns
    -------
    float
        The value of this feature.

    Examples
    --------
    >>> x = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> longest_strike_above_mean(x)
    5.0

    >>> x = pl.Series([5, 4, 3, 2, 1])
    >>> longest_strike_above_mean(x)
    0.0

    >>> x = pl.Series([1, 2, 3, 4, 5, 4, 3, 2, 1])
    >>> longest_strike_above_mean(x)
    5.0

    Notes
    -----
    A consecutive subsequence is a sequence of consecutive elements in the time series. The function calculates the mean
    of the time series and then finds the longest consecutive subsequence that is greater than the mean. If there is no
    such subsequence, the function returns 0.
    """
    if not x.is_empty():
        X = _get_length_sequences_where(x.cast(pl.Float64) > x.mean())
    else:
        return 0
    return X.max() if X.len() > 0 else 0


def mean_n_absolute_max(x: pl.Series, n_maxima: int) -> float:
    """
    Calculates the arithmetic mean of the n absolute maximum values of the time series.

    Parameters
    ----------
    x : pl.Series
        The time series to calculate the feature of.
    n_maxima : int
        The number of maxima to consider.

    Returns
    -------
    float
        The value of this feature.
    """
    if n_maxima <= 0:
        raise ValueError("The number of maxima should be > 0.")
    return (
        x.abs().sort(descending=True)[:n_maxima].mean() if x.len() > n_maxima else None
    )


def percent_reocurring_points(x: pl.Series) -> float:
    """
    Returns the percentage of non-unique data points in the time series. Non-unique data points are those that occur
    more than once in the time series.

    The percentage is calculated as follows:

        # of data points occurring more than once / # of all data points

    This means the ratio is normalized to the number of data points in the time series, in contrast to the
    `percent_recoccuring_values` function.

    Parameters
    ----------
    x : pl.Series
        The time series to calculate the feature of.

    Returns
    -------
    float
        The value of this feature.
    """
    if x.is_empty():
        raise ValueError("The serie is empty.")
    X = x.value_counts().filter(pl.col("counts") > 1).sum()
    return X[0, "counts"] / x.len()


def percent_recoccuring_values(x: pl.Series) -> float:
    """
    Returns the percentage of values that are present in the time series more than once.

    The percentage is calculated as follows:

        len(different values occurring more than once) / len(different values)

    This means the percentage is normalized to the number of unique values in the time series, in contrast to the
    `percent_reocurring_points` function.

    Parameters
    ----------
    x : pl.Series
        The time series to calculate the feature of.

    Returns
    -------
    float
        The value of this feature.
    """
    if x.is_empty():
        raise ValueError("The serie is empty.")
    X = x.value_counts().filter(pl.col("counts") > 1)
    return X.shape[0] / x.n_unique()


def sum_reocurring_points(x: pl.Series) -> float:
    """
    Returns the sum of all data points that are present in the time series more than once.

    For example, `sum_reocurring_points(pl.Series([2, 2, 2, 2, 1]))` returns 8, as 2 is a reoccurring value, so all 2's
    are summed up.

    This is in contrast to the `sum_reocurring_values` function, where each reoccuring value is only counted once.

    Parameters
    ----------
    x : pl.Series
        The time series to calculate the feature of.

    Returns
    -------
    float
        The value of this feature.
    """
    X = x.value_counts().filter(pl.col("counts") > 1)
    return X[:, 0].dot(X[:, 1])


def sum_reocurring_values(x: pl.Series) -> float:
    """
    Returns the sum of all values that are present in the time series more than once.

    For example, `sum_reocurring_values(pl.Series([2, 2, 2, 2, 1]))` returns 2, as 2 is a reoccurring value, so it is
    summed up with all other reoccuring values (there is none), so the result is 2.

    This is in contrast to the `sum_reocurring_points` function, where each reoccuring value is only counted as often
    as it is present in the data.

    Parameters
    ----------
    x : pl.Series
        The time series to calculate the feature of.

    Returns
    -------
    float
        The value of this feature.
    """
    X = x.value_counts().filter(pl.col("counts") > 1).sum()
    return X[0, 0]


def mean_second_derivative_central(x: pl.Series) -> float:
    """
    Returns the mean value of a central approximation of the second derivative.

    .. math::

    \\frac{1}{2(n-2)} \\sum_{i=1}^{n-1} 0.5 (x_{i+2} - 2 x_{i+1} + x_i)

    where n is the length of the time series x

    Parameters
    ----------
    x : pl.Series
        A time series to calculate the feature of.

    Returns
    -------
    float
        The value of the central approximation of the second derivative of x.
    """
    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2))


def symmetry_looking(x: pl.Series, r: float) -> pl.DataFrame:
    """Check if the distribution of x *looks symmetric*.

    A distribution is considered symmetric if:

    .. math::

    | mean(X)-median(X) | < r * (max(X)-min(X))

    Parameters
    ----------
    x : polars.Series
        The time series to calculate the feature of.
    r : float
        Multiplier on distance between max and min.

    Returns
    -------
    bool : True if symmetric.
    """
    mean_median_difference = abs(x.mean() - x.median())
    max_min_difference = x.max() - x.min()
    return mean_median_difference < r["r"] * max_min_difference


def time_reversal_asymmetry_statistic(x: pl.Series, lag: int) -> float:
    """
    Returns the time reversal asymmetry statistic.

    This function calculates the value of:

    .. math::

        \\frac{1}{n-2lag} \\sum_{i=1}^{n-2lag} x_{i + 2 \\cdot lag}^2 \\cdot x_{i + lag} - x_{i + lag} \\cdot  x_{i}^2

    which is

    .. math::

        \\mathbb{E}[L^2(X)^2 \\cdot L(X) - L(X) \\cdot X^2]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a
    promising feature to extract from time series.

    Parameters
    ----------
    x : pl.Series
        The time series to calculate the feature of.
    lag : int
        The lag that should be used in the calculation of the feature.

    Returns
    -------
    float
        The value of the calculated feature.

    References
    ----------
    [1] Fulcher, B.D., Jones, N.S. (2014). Highly comparative feature-based time-series classification.
        Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.
    """
    n = len(x)
    one_lag = x.shift(-lag)
    two_lag = x.shift(-2 * lag)
    result = (two_lag * two_lag * one_lag - one_lag * x * x).head(n - 2 * lag).mean()
    return result
