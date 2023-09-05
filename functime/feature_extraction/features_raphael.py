import polars as pl
from typing import Union

def change_quantiles(x: pl.Expr,
                    ql: float,
                    qh: float,
                    isabs: bool,
                    f_agg: str) -> pl.Expr:
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
        x = x.filter(
        x.is_between(x.quantile(ql), x.quantile(qh))
    ).diff(null_behavior="drop").abs()
    else:
        x = x.filter(
        x.is_between(x.quantile(ql), x.quantile(qh))
    ).diff(null_behavior="drop")
    return getattr(x, f_agg)()