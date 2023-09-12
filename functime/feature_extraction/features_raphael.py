import polars as pl


def change_quantiles(
    x: pl.Expr, ql: float, qh: float, isabs: bool, f_agg: str
) -> pl.Expr:
    """
    First fixes a corridor given by the quantiles ql and qh of the distribution of x.
    Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

    Think about selecting a corridor on the
    y-Axis and only calculating the mean of the absolute change of the time series inside this corridor.

    Args:
        x (pl.Expr): the time series to calculate the feature of
        ql (float): the lower quantile of the corridor
        qh (float): the higher quantile of the corridor
        isabs (bool): should the absolute differences be taken?
        f_agg (str): the aggregator function that is applied to the differences in the bin

    Returns:
        pl.Expr: the value of this feature
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

    Args:
        x (pl.Expr): The time series to compute the feature of.

    Returns:
        pl.Expr: The mean absolute change of the time series
    """
    return x.diff(null_behavior="drop").abs().mean()


def mean_change(x: pl.Expr) -> pl.Expr:
    """
    Compute mean change.

    Args:
        x (pl.Expr): The time series to compute the feature of.

    Returns:
        pl.Expr: The mean change of the time series
    """
    return x.diff(null_behavior="drop").mean()


def number_crossing_m(x: pl.Expr, m: float) -> pl.Expr:
    """
    Calculates the number of crossings of x on m. A crossing is defined as two sequential values where the first value
    is lower than m and the next is greater, or vice-versa. If you set m to zero, you will get the number of zero
    crossings.

    Args:
        x (pl.Expr): the time series to calculate the feature of.
        m (float): the crossing value.

    Returns:
        pl.Expr: how many times x crosses m.
    """
    return x.gt(m).cast(pl.Int8).diff(null_behavior="drop").abs().eq(1).sum()


def var_greater_than_std(x: pl.Expr) -> pl.Expr:
    """
    Is variance higher than the standard deviation?

    Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
    being larger than 1

    Args:
        x (pl.Expr): the time series to calculate the feature of

    Returns:
        pl.Expr: the value of this feature
    """
    y = x.var(ddof=0)
    return y > y.sqrt()


def first_location_of_maximum(x: pl.Expr) -> pl.Expr:
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    Args:
        x (pl.Expr): the time series to calculate the feature of

    Returns:
        pl.Expr: the first location of maximum as float fraction
    """
    return x.arg_max() / x.len()


def last_location_of_maximum(x: pl.Expr) -> pl.Expr:
    """
    Returns the last location of the maximum value of x.
    The position is calculated relatively to the length of x.

    Args:
        x (pl.Expr): the time series to calculate the feature of

    Returns:
        pl.Expr: the last location of maximum as float fraction
    """
    return (x.len() - x.reverse().arg_max()) / x.len()


def first_location_of_minimum(x: pl.Expr) -> pl.Expr:
    """
    Returns the first location of the minimum value of x.
    The position is calculated relatively to the length of x.

    Args:
        x (pl.Expr): the time series to calculate the feature of

    Returns:
        pl.Expr: the first location of minimum as float fraction
    """
    return x.arg_min() / x.len()


def last_location_of_minimum(x: pl.Expr) -> pl.Expr:
    """
    Returns the last location of the minimum value of x.
    The position is calculated relatively to the length of x.

    Args:
        x (pl.Expr): the time series to calculate the feature of

    Returns:
        pl.Expr: the last location of minimum as float fraction
    """
    return (x.len() - x.reverse().arg_min()) / x.len()
