import polars as pl

def benford_correlation(x: pl.Series)-> float:
    """
     Useful for anomaly detection applications [1][2]. Returns the correlation from first digit distribution when
     compared to the Newcomb-Benford's Law distribution [3][4].

     .. math::

         P(d)=\\log_{10}\\left(1+\\frac{1}{d}\\right)

     where :math:`P(d)` is the Newcomb-Benford distribution for :math:`d` that is the leading digit of the number
     {1, 2, 3, 4, 5, 6, 7, 8, 9}.

     .. rubric:: References

     |  [1] A Statistical Derivation of the Significant-Digit Law, Theodore P. Hill, Statistical Science, 1995
     |  [2] The significant-digit phenomenon, Theodore P. Hill, The American Mathematical Monthly, 1995
     |  [3] The law of anomalous numbers, Frank Benford, Proceedings of the American philosophical society, 1938
     |  [4] Note on the frequency of use of the different digits in natural numbers, Simon Newcomb, American Journal of
     |  mathematics, 1881

    :param x: the time series to calculate the feature of
    :type x: pl.Series
    :return: the value of this feature
    :return type: float
    """
    X = (
        (x / (10 ** x.abs().log10().floor()))
        .abs()
        .floor()
    )
    df_corr = pl.DataFrame(
        [
            [X.eq(i).mean() for i in pl.int_range(1, 10, eager=True)],
            (1+1/pl.int_range(1, 10, eager=True)).log10()
        ]
    ).corr()
    return df_corr[0,1]

def _get_length_sequences_where(x: pl.Series)-> pl.Series:
    """
    This method calculates the length of all sub-sequences where the serie x is either True or 1.

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

    :param x: A pl.Series containing only 1, True, 0 and False values
    :return: A pl.Series with the length of all sub-sequences where the series is either True or False. If no ones or Trues
    contained, the list [0] is returned.
    """
    X = (
        x
        .alias("orig")
        .to_frame()
        .with_columns(
            shift=pl.col("orig").shift(periods=1)
        )
        .with_columns(
            mask=pl.col("orig").ne(pl.col("shift")).fill_null(0).cumsum()
        )
        .filter(pl.col("orig") == 1)
        .group_by(pl.col("mask"), maintain_order=True).count()
    )["count"]
    return X

def longest_strike_below_mean(x: pl.Series)-> float:
    """
    Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x

    :param x: the time series to calculate the feature of
    :type x: pl.Series
    :return: the value of this feature
    :return type: float
    """
    if not x.is_empty():
        X = _get_length_sequences_where(
            (x.cast(pl.Float64) < x.mean())
        )
    else:
        return 0
    return X.max() if X.len() > 0 else 0


def longest_strike_above_mean(x: pl.Series)-> float:
    """
    Returns the length of the longest consecutive subsequence in x that is greater than the mean of x

    :param x: the time series to calculate the feature of
    :type x: pl.Series
    :return: the value of this feature
    :return type: float
    """
    if not x.is_empty():
        X = _get_length_sequences_where(
            (x.cast(pl.Float64) > x.mean())
        )
    else:
        return 0
    return X.max() if X.len() > 0 else 0

def mean_n_absolute_max(x: pl.Series, n_maxima: int)-> float:
    """
    Calculates the arithmetic mean of the n absolute maximum values of the time series.

    :param x: the time series to calculate the feature of
    :type x: pl.Series
    :param n_maxima: the number of maxima, which should be considered
    :type n_maxima: int

    :return: the value of this feature
    :return type: float
    """
    if n_maxima <= 0:
        raise ValueError("The number of maxima should be > 0.")
    return x.abs().sort(descending=True)[:n_maxima].mean() if x.len() > n_maxima else None

def percent_reocurring_points(x: pl.Series)-> float:
    """
    Returns the percentage of non-unique data points. Non-unique means that they are
    contained another time in the time series again.

        # of data points occurring more than once / # of all data points

    This means the ratio is normalized to the number of data points in the time series,
    in contrast to the percent_recoccuring_values.

    :param x: the time series to calculate the feature of
    :type x: pl.Series
    :return: the value of this feature
    :return type: float
    """
    if x.is_empty():
        raise ValueError("The serie is empty.")
    X = (
        x
        .value_counts()
        .filter(pl.col("counts") > 1).sum()
    )
    return X[0, "counts"] / x.len()

def percent_recoccuring_values(x: pl.Series)-> float:
    """
    Returns the percentage of values that are present in the time series
    more than once.

        len(different values occurring more than once) / len(different values)

    This means the percentage is normalized to the number of unique values,
    in contrast to the percent_reocurring_points.

    :param x: the time series to calculate the feature of
    :type x: pl.Series
    :return: the value of this feature
    :return type: float
    """
    if x.is_empty():
        raise ValueError("The serie is empty.")
    X = (
        x
        .value_counts()
        .filter(pl.col("counts") > 1)
    )
    return X.shape[0] / x.n_unique()

def sum_reocurring_points(x: pl.Series)-> float:
    """
    Returns the sum of all data points, that are present in the time series
    more than once.

    For example

        sum_reocurring_points(pl.Series([2, 2, 2, 2, 1])) = 8

    as 2 is a reoccurring value, so all 2's are summed up.

    This is in contrast to ``sum_reocurring_values``,
    where each reoccuring value is only counted once.

    :param x: the time series to calculate the feature of
    :type x: pl.Series
    :return: the value of this feature
    :return type: float
    """
    X = (
        x
        .value_counts()
        .filter(pl.col("counts") > 1)
    )
    return X[:,0].dot(X[:,1])

def sum_reocurring_values(x: pl.Series)-> float:
    """
    Returns the sum of all values, that are present in the time series
    more than once.

    For example

        sum_reocurring_values(pl.Series([2, 2, 2, 2, 1])) = 2

    as 2 is a reoccurring value, so it is summed up with all
    other reoccuring values (there is none), so the result is 2.

    This is in contrast to ``sum_reocurring_points``,
    where each reoccuring value is only counted as often as
    it is present in the data.

    :param x: the time series to calculate the feature of
    :type x: pl.Series
    :return: the value of this feature
    :return type: float
    """
    X = (
        x
        .value_counts()
        .filter(pl.col("counts") > 1).sum()
    )
    return X[0,0]
