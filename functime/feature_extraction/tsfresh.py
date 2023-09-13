import polars as pl


# helper function should not be exposed
def has_duplicate_of_value(x: pl.Expr, t: float) -> pl.Expr:
    """
    Check if a Polars Series has duplicates of a specific value.

    Parameters:
    - x (pl.Expr): The input Polars Series to be checked for duplicates.
    - t (float): The target value to check for duplicates.

    Returns:
    - pl.Expr: A Boolean Polars Series indicating whether duplicates of 't' exist in 'x'.

    This function checks if the input Polars Series 'x' contains duplicates of the specified value 't'.
    It performs the following steps:

    1. Filters 'x' to retain only the elements equal to 't'.
    2. Determines if there are any duplicates within the filtered Series.
    3. Returns a Boolean Polars Series indicating the presence of duplicates.

    Example:
    If 'x' contains [1.0, 2.0, 3.0, 1.0, 4.0], and 't' is 1.0, the function will return True,
    as there is a duplicate of the value 1.0 in 'x'.

    Note:
    - Make sure 'x' is a valid Polars Series.
    """
    return x.filter(x == t).is_duplicated().any()


def autocorrelation(x: pl.Expr, lag: int) -> pl.Expr:
    """
    Compute the autocorrelation coefficient at a given lag 'l' for a given input time series 'x'.

    Parameters:
    -----------
    x : pl.Expr
        The input time series data.
    l : int
        The lag at which the autocorrelation is to be computed. Should be a non-negative integer.

    Returns:
    --------
    pl.Expr
        The autocorrelation coefficient at the specified lag 'l' for the input time series 'x'.

    Raises:
    -------
    Exception
        If the provided 'lag' value 'l' is negative.

    Notes:
    ------
    The autocorrelation coefficient at lag 'l' is calculated using the following formula:

    .. math::
        R(l) = \\frac{\\sum_{t=0}^{n-l-1}(x[t] - \\bar{x})(x[t+l] - \\bar{x})}{(n-l) \\cdot \\text{Var}(x)}

    Where:
    - 'n' is the length of the time series 'x'.
    - 'x[t]' represents the value of 'x' at time 't'.
    - '\\bar{x}' is the mean of the time series 'x'.
    - '\\text{Var}(x)' is the variance of the time series 'x'.

    If 'l' is 0, the autocorrelation at lag 0 is 1.0, indicating perfect correlation with itself.

    Example:
    --------
    >>> x = pl.DataFrame({"values": [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> lag = 2
    >>> autocorrelation_result = autocorrelation(x['values'], lag)
    >>> print(autocorrelation_result)
    0.25
    """
    if lag < 0:
        raise Exception("Lag cannot be a negative number")

    if lag == 0:
        return pl.lit(1.0)

    return (
        x.slice(offset=0, length=x.count() - lag)
        .sub(x.mean())
        .dot(x.slice(offset=lag, length=x.count() - lag).sub(x.mean()))
        .truediv((x.count() - lag).mul(x.var(ddof=0)))
    )


def count_below(x: pl.Expr, t: float) -> pl.Expr:
    """
    Calculate the percentage of values below or equal to a specified threshold in a Polars Series.

    Parameters:
    - x (pl.Expr): The input Polars Series for which the percentage is calculated.
    - t (float): The threshold value for comparison.

    Returns:
    - pl.Expr: A Polars Series representing the percentage of values in 'x' that are below or equal to 't'.

    This function calculates the percentage of values in the input Polars Series 'x' that are below or equal to
    the specified threshold 't'. The calculation proceeds as follows:

    1. Filters 'x' to retain only the elements that are less than or equal to 't'.
    2. Counts the number of elements that meet the condition.
    3. Divides the count by the total number of elements in 'x'.
    4. Multiplies the result by 100 to express the percentage.

    Example:
    If 'x' contains [10.0, 15.0, 20.0, 5.0, 10.0], and 't' is 12.0, the function will return 60.0,
    as 60% of the values in 'x' (3 out of 5) are below or equal to 12.0.

    Note:
    - Make sure 'x' is a valid Polars Series.
    """
    return x.filter(x <= t).count().truediv(x.count()).mul(100)


def count_above(x: pl.Expr, t: float) -> pl.Expr:
    """
    Calculate the percentage of values above or equal to a specified threshold in a Polars Series.

    Parameters:
    - x (pl.Expr): The input Polars Series for which the percentage is calculated.
    - t (float): The threshold value for comparison.

    Returns:
    - pl.Expr: A Polars Series representing the percentage of values in 'x' that are above or equal to 't'.

    This function calculates the percentage of values in the input Polars Series 'x' that are above or equal to
    the specified threshold 't'. The calculation proceeds as follows:

    1. Filters 'x' to retain only the elements that are greater than or equal to 't'.
    2. Counts the number of elements that meet the condition.
    3. Divides the count by the total number of elements in 'x'.
    4. Multiplies the result by 100 to express the percentage.

    Example:
    If 'x' contains [10.0, 15.0, 20.0, 5.0, 10.0], and 't' is 12.0, the function will return 40.0,
    as 40% of the values in 'x' (2 out of 5) are above or equal to 12.0.

    Note:
    - Make sure 'x' is a valid Polars Series.
    """
    return x.filter(x >= t).count().truediv(x.count()).mul(100)


def count_below_mean(x: pl.Expr) -> pl.Expr:
    """
    Count the number of values in a Polars Series that are below the mean.

    Parameters:
    - x (pl.Expr): The input Polars Series for which the count is calculated.

    Returns:
    - pl.Expr: A Polars Series representing the count of values in 'x' that are below the mean.

    This function calculates the count of values in the input Polars Series 'x' that are below the mean value of 'x'.
    The calculation proceeds as follows:

    1. Filters 'x' to retain only the elements that are less than the mean of 'x'.
    2. Counts the number of elements that meet the condition.

    Example:
    If 'x' contains [10.0, 15.0, 20.0, 5.0, 10.0], the mean of 'x' is 12.0.
    The function will return 2, as there are 2 values in 'x' (5.0 and 10.0) that are below the mean (12.0).

    Note:
    - Make sure 'x' is a valid Polars Series.
    """
    return x.filter(x < x.mean()).count()


def count_above_mean(x: pl.Expr) -> pl.Expr:
    """
    Count the number of values in a Polars Series that are above or equal to the mean.

    Parameters:
    - x (pl.Expr): The input Polars Series for which the count is calculated.

    Returns:
    - pl.Expr: A Polars Series representing the count of values in 'x' that are above or equal to the mean.

    This function calculates the count of values in the input Polars Series 'x' that are above or equal to
    the mean value of 'x'. The calculation proceeds as follows:

    1. Filters 'x' to retain only the elements that are greater than or equal to the mean of 'x'.
    2. Counts the number of elements that meet the condition.

    Example:
    If 'x' contains [10.0, 15.0, 20.0, 5.0, 10.0], the mean of 'x' is 12.0.
    The function will return 3, as there are 3 values in 'x' (10.0, 15.0, and 20.0) that are above or equal to the mean (12.0).

    Note:
    - Make sure 'x' is a valid Polars Series.
    """
    return x.filter(x >= x.mean()).count()


def has_duplicate(x: pl.Expr) -> pl.Expr:
    """
    Check if a Polars Series contains any duplicate values.

    Parameters:
    - x (pl.Expr): The input Polars Series to be checked for duplicates.

    Returns:
    - pl.Expr: A Boolean Polars Series indicating whether any duplicates exist in 'x'.

    This function checks if the input Polars Series 'x' contains any duplicate values.
    It performs the following steps:

    1. Determines if there are any duplicate values within 'x'.
    2. Returns a Boolean Polars Series indicating the presence of duplicates.

    Example:
    If 'x' contains [10.0, 15.0, 10.0, 5.0, 20.0], the function will return True,
    as there are duplicate values (10.0) in 'x'.

    Note:
    - Make sure 'x' is a valid Polars Series.
    """
    return x.is_duplicated().any()


def has_duplicate_max(x: pl.Expr) -> pl.Expr:
    """
    Check if a Polars Series contains duplicates of its maximum value.

    Parameters:
    - x (pl.Expr): The input Polars Series to be checked for duplicates of its maximum value.

    Returns:
    - pl.Expr: A Boolean Polars Series indicating whether duplicates of the maximum value exist in 'x'.

    This function checks if the input Polars Series 'x' contains duplicates of its maximum value.
    It performs the following steps:

    1. Calculates the maximum value in 'x'.
    2. Uses the 'has_duplicate_of_value' function to check if there are duplicates of the maximum value in 'x'.
    3. Returns a Boolean Polars Series indicating the presence of duplicates.

    Example:
    If 'x' contains [10.0, 15.0, 10.0, 5.0, 20.0], and the maximum value in 'x' is 20.0, the function will return True,
    as there are duplicate values (10.0) equal to the maximum value in 'x'.

    Note:
    - Make sure 'x' is a valid Polars Series.
    - The 'has_duplicate_of_value' function is used internally to perform the duplicate check.
    """
    return has_duplicate_of_value(x, x.max())


def has_duplicate_min(x: pl.Expr) -> pl.Expr:
    """
    Check if a Polars Series contains duplicates of its minimum value.

    Parameters:
    - x (pl.Expr): The input Polars Series to be checked for duplicates of its minimum value.

    Returns:
    - pl.Expr: A Boolean Polars Series indicating whether duplicates of the minimum value exist in 'x'.

    This function checks if the input Polars Series 'x' contains duplicates of its minimum value.
    It performs the following steps:

    1. Calculates the minimum value in 'x'.
    2. Uses the 'has_duplicate_of_value' function to check if there are duplicates of the minimum value in 'x'.
    3. Returns a Boolean Polars Series indicating the presence of duplicates.

    Example:
    If 'x' contains [10.0, 15.0, 5.0, 5.0, 20.0], and the minimum value in 'x' is 5.0, the function will return True,
    as there are duplicate values (5.0) equal to the minimum value in 'x'.

    Note:
    - Make sure 'x' is a valid Polars Series.
    - The 'has_duplicate_of_value' function is used internally to perform the duplicate check.
    """
    return has_duplicate_of_value(x, x.min())
