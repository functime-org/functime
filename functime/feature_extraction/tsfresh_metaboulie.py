import polars as pl


def symmetry_looking(x: pl.Series, param: list) -> pl.DataFrame:
    """
    Boolean variable denoting if the distribution of x *looks symmetric*. This is the case if

    .. math::

        | mean(X)-median(X)| < r * (max(X)-min(X))

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
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
            "feature_value": mean_median_difference < pl.col("r") * max_min_difference,
        }
    )
    return result
