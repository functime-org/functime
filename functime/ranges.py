from typing import Optional

import polars as pl
from polars.type_aliases import TimeUnit

from functime.offsets import _strip_freq_alias


def make_future_ranges(
    time_col: str,
    cutoffs: pl.DataFrame,
    fh: int,
    freq: str,
    time_unit: Optional[TimeUnit] = None,
) -> pl.DataFrame:
    """Return pl.DataFrame with columns entity_col, time_col.

    DataFrame has shape (n_entities, 2) and dtypes (str, list[date]), (str, list[datetime]), or (str, list[int]).
    """
    entity_col = cutoffs.columns[0]
    if freq.endswith("i"):
        return cutoffs.select(
            pl.col(entity_col),
            pl.int_ranges(
                pl.col("low") + 1,
                pl.col("low") + fh + 1,
                step=int(freq[:-1]),
                eager=False,
            ).alias(time_col),
        )
    else:
        offset_n, offset_alias = _strip_freq_alias(freq)
        # Make date ranges
        return cutoffs.select(
            pl.col(entity_col),
            pl.date_ranges(
                pl.col("low"),
                pl.col("low")
                .dt.offset_by(f"{fh * offset_n}{offset_alias}")
                .alias("high"),
                interval=freq,
                closed="right",
                time_unit=time_unit or "us",
                eager=False,
            ).alias(time_col),
        )
