import polars as pl


# helper function should not be exposed
def has_duplicate_of_value(x: pl.Expr, t: float) -> pl.Expr:
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
    >>> result = has_duplicate_of_value(series, 2)
    >>> print(result)
    True

    See Also
    --------
    pl.Series.filter : Filter a Series using a boolean expression.
    pl.Series.is_duplicated : Check for duplicate values in a Series.
    pl.Series.any : Check if any elements in a boolean Series are True.

    """
    return x.filter(x == t).is_duplicated().any()


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
    - It then checks for duplicates of that maximum value using the `has_duplicate_of_value` function.
    - The result is a boolean expression, where True indicates the presence of such duplicates and False indicates none.

    See Also
    --------
    has_duplicate_of_value : Check for duplicate values equal to a specified value in a Polars Expression.
    pl.Expr.max : Calculate the maximum value in a Polars Expression.
    pl.Expr.is_duplicated : Check for duplicate values in a Polars Expression.
    pl.Expr.any : Check if any elements in a boolean Polars Expression are True.

    """
    return has_duplicate_of_value(x, x.max())


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
    - It then checks for duplicates of that minimum value using the `has_duplicate_of_value` function.
    - The result is a boolean expression, where True indicates the presence of such duplicates and False indicates none.

    See Also
    --------
    has_duplicate_of_value : Check for duplicate values equal to a specified value in a Polars Expression.
    pl.Expr.min : Calculate the minimum value in a Polars Expression.
    pl.Expr.is_duplicated : Check for duplicate values in a Polars Expression.
    pl.Expr.any : Check if any elements in a boolean Polars Expression are True.

    """
    return has_duplicate_of_value(x, x.min())
