"""Fetch sample datasets."""

from typing import Union

import polars as pl
from typing_extensions import Literal, overload


@overload
def fetch_commodities(mode: Literal["eager"]) -> pl.DataFrame:
    ...


@overload
def fetch_commodities(mode: Literal["lazy"]) -> pl.LazyFrame:
    ...


def fetch_commodities(
    mode: Literal["eager", "lazy"]
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """Read the commodities data.

    The data is a panel datasets with 70 entities with monthly observations from 1960 to 2023.

    Parameters
    ----------
    mode : Literal["eager", "lazy"]
        Whether to return a DataFrame or a LazyFrame.

    Return
    ------
    pl.DataFrame | pl.LazyFrame
        A DataFrame if mode is "eager" or a LazyFrame if mode is "lazy".
    """
    url = "https://github.com/neocortexdb/functime/raw/main/data/commodities.parquet"
    y = pl.scan_parquet(url).with_columns(pl.col("time").cast(pl.Date))

    if mode == "eager":
        return y.collect()
    return y
