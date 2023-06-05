from functools import reduce
from typing import List, Optional

import polars as pl
from holidays import country_holidays
from typing_extensions import Literal

from functime.base import transformer
from functime.ranges import make_future_ranges


@transformer
def add_calendar_effects(
    attrs: List[
        Literal["minute", "hour", "day", "weekday", "week", "month", "quarter", "year"]
    ],
):
    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        time_col = pl.col(X.columns[1])
        X_new = X.with_columns(
            [
                getattr(time_col.dt, attr)()
                .alias(attr)
                .cast(pl.Utf8)
                .cast(pl.Categorical)
                for attr in attrs
            ]
        )
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def add_holiday_effects(country_codes: List[str], freq: str):
    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        time_col = X.columns[1]
        dt_min_max = X.select(
            [pl.col(time_col).min().alias("min"), pl.col(time_col).max().alias("max")]
        ).collect(streaming=True)
        dt_min, dt_max = dt_min_max[0, "min"], dt_min_max[0, "max"]
        years = range(dt_min.year, dt_max.year + 1)
        # Instantiate countries mapping
        countries = [country_holidays(code, years=years) for code in country_codes]
        # Add holiday effects and cast as categorical
        dt_range = (
            pl.date_range(dt_min, dt_max, interval="1d", eager=True)
            .alias(time_col)
            .to_frame()
            .lazy()
        )
        holidays = [
            dt_range.with_columns(
                pl.col(time_col)
                .apply(country.get)
                .cast(pl.Utf8)
                .str.to_lowercase()
                .str.replace_all("'", "")
                .str.replace_all("-", "")
                .str.replace_all(" ", "_")
                .cast(pl.Categorical)
                .alias(f"holiday__{country_codes[i]}")
            )
            for i, country in enumerate(countries)
        ]
        holidays = (
            reduce(lambda df1, df2: (df1.join(df2, how="inner", on=time_col)), holidays)
            .groupby_dynamic(time_col, every=freq)
            .agg(pl.all().exclude(time_col).drop_nulls().first())
            .with_columns(pl.col(time_col).cast(X.schema[time_col]))
        )
        X_new = X.join(holidays, how="left", on=time_col)
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


def make_future_calendar_effects(
    idx: pl.DataFrame,
    attrs: List[str],
    fh: int,
    freq: Optional[str] = None,
):
    entity_col, time_col = idx.columns[:2]
    cutoffs = idx.groupby(entity_col).agg(pl.col(time_col).max().alias("low"))
    future_idx = make_future_ranges(
        time_col=time_col,
        cutoffs=cutoffs,
        fh=fh,
        freq=freq,
    ).explode(time_col)
    transf = add_calendar_effects(attrs)
    return transf(future_idx)


def make_future_holiday_effects(
    idx: pl.DataFrame,
    country_codes: List[str],
    fh: int,
    freq: Optional[str] = None,
):
    entity_col, time_col = idx.columns[:2]
    cutoffs = idx.groupby(entity_col).agg(pl.col(time_col).max().alias("low"))
    future_idx = make_future_ranges(
        time_col=time_col,
        cutoffs=cutoffs,
        fh=fh,
        freq=freq,
    ).explode(time_col)
    transf = add_holiday_effects(country_codes, freq=freq)
    return transf(future_idx)
