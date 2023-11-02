from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal

from functime.seasonality.calendar import (
    add_calendar_effects,
    add_holiday_effects,
)


def test_add_calendar_effects():
    """
    Creates new calendar effect columns based on datetime column in a time series DataFrame.
    """
    with pl.StringCache():
        data = pl.DataFrame(
            {
                "country": ["US", "US", "US"],
                "datetime": [
                    datetime(2023, 1, 1, 0, 0, 0),
                    datetime(2023, 2, 1, 0, 0, 0),
                    datetime(2023, 3, 1, 0, 0, 0),
                ],
            }
        ).lazy()
        calendar_cols = ["minute", "hour", "day", "week", "month", "quarter", "year"]
        result = add_calendar_effects(calendar_cols)(data)
        expected = pl.DataFrame(
            {
                "country": ["US", "US", "US"],
                "datetime": [
                    datetime(2023, 1, 1, 0, 0, 0),
                    datetime(2023, 2, 1, 0, 0, 0),
                    datetime(2023, 3, 1, 0, 0, 0),
                ],
                "minute": [0, 0, 0],
                "hour": [0, 0, 0],
                "day": [1, 1, 1],
                "week": [52, 5, 9],
                "month": [1, 2, 3],
                "quarter": [1, 1, 1],
                "year": [2023, 2023, 2023],
            }
        ).with_columns(pl.all().exclude("datetime").cast(pl.Utf8).cast(pl.Categorical))
        assert_frame_equal(result.collect(), expected, check_dtype=False)


def test_add_holiday_effects():
    """
    Creates country holiday columns based on the datetime column.
    The function will return null values for datetime without holidays
    """
    with pl.StringCache():
        data = pl.DataFrame(
            {
                "country": ["US", "US", "US", "US", "US", "US"],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 11, 26),
                    datetime(2020, 12, 25),
                    datetime(2021, 1, 1),
                    datetime(2021, 1, 2),
                ],
            }
        ).lazy()
        country_codes = ["US", "UK"]
        result = add_holiday_effects(country_codes)(data)
        expected = pl.DataFrame(
            {
                "country": ["US", "US", "US", "US", "US", "US"],
                "datetime": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 11, 26),
                    datetime(2020, 12, 25),
                    datetime(2021, 1, 1),
                    datetime(2021, 1, 2),
                ],
                "holiday__US": [
                    "new_years_day",
                    None,
                    "thanksgiving",
                    "christmas_day",
                    "new_years_day",
                    None,
                ],
                "holiday__UK": [
                    "new_years_day",
                    None,
                    None,
                    "christmas_day",
                    "new_years_day",
                    None,
                ],
            }
        ).with_columns(pl.all().exclude("datetime").cast(pl.Utf8).cast(pl.Categorical))
        assert_frame_equal(result.collect(), expected, check_dtype=False)
