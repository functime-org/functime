import polars as pl
import pytest
from polars.testing import assert_frame_equal

from functime.feature_extraction.tsfresh_achasol import (
    autocorrelation,
    count_above,
    count_above_mean,
    count_below,
    count_below_mean,
    has_duplicate,
    has_duplicate_max,
    has_duplicate_min,
)


@pytest.fixture(params=[pl.DataFrame, pl.LazyFrame])
def rental_car_dataset(request):
    df_cls = request.param
    return df_cls, df_cls(
        {
            "volvo": [10, 12, 12, 14, 17, 22],
            "tesla": [4, 5, 3, 2, 3, 2],
            "ford": [10, 10, 0, 0, 1, 21],
            "mitsubishi": [7, 7, 8, 8, 9, 11],
            "toyota": [7, 8, 9, 10, 11, 12],
            "mercedes": [7.0, 8.0, 3.0, 10.0, 100.0, 100.0],
            "audi": [1.0, 1.0, 3.0, 99.0, 100.0, 101.0],
        }
    )


def test_autocorrelation(rental_car_dataset):
    df_cls, dataset = rental_car_dataset
    result = dataset.select(
        autocorrelation(pl.col("volvo"), 1).alias("autocorrelation-volvo"),
        autocorrelation(pl.col("tesla"), 2).alias("autocorrelation-tesla"),
        autocorrelation(pl.col("ford"), 1).alias("autocorrelation-ford"),
        autocorrelation(pl.col("mitsubishi"), 0).alias("autocorrelation-mitsubishi"),
    )

    expected = df_cls(
        {
            "autocorrelation-volvo": [0.4554973821989529],
            "autocorrelation-tesla": [-0.19512195121951229],
            "autocorrelation-ford": [-0.017241379310344827],
            "autocorrelation-mitsubishi": [1.0],
        }
    )

    assert_frame_equal(result, expected)


def test_count_below(rental_car_dataset):
    df_cls, dataset = rental_car_dataset
    result = dataset.select(
        count_below(pl.col("volvo"), 15).alias("count-below-volvo"),
        count_below(pl.col("tesla"), 0).alias("count-below-tesla"),
        count_below(pl.col("ford"), 10).alias("count-below-ford"),
        count_below(pl.col("mitsubishi"), 12).alias("count-below-mitsubishi"),
    )

    expected = df_cls(
        {
            "count-below-volvo": [200 / 3],
            "count-below-tesla": [0.0],
            "count-below-ford": [500 / 6],
            "count-below-mitsubishi": [100.0],
        }
    )

    assert_frame_equal(result, expected)


def test_count_above(rental_car_dataset):
    df_cls, dataset = rental_car_dataset
    result = dataset.select(
        count_above(pl.col("volvo"), 15).alias("count-above-volvo"),
        count_above(pl.col("tesla"), 0).alias("count-above-tesla"),
        count_above(pl.col("ford"), 3).alias("count-above-ford"),
        count_above(pl.col("mitsubishi"), 8).alias("count-above-mitsubishi"),
    )

    expected = df_cls(
        {
            "count-above-volvo": [200 / 6],
            "count-above-tesla": [100.0],
            "count-above-ford": [300 / 6],
            "count-above-mitsubishi": [400 / 6],
        }
    )
    assert_frame_equal(result, expected)


def test_count_below_mean(rental_car_dataset):
    df_cls, dataset = rental_car_dataset
    result = dataset.select(
        count_below_mean(pl.col("volvo")).alias("count_below_mean-volvo"),
        count_below_mean(pl.col("tesla")).alias("count_below_mean-tesla"),
        count_below_mean(pl.col("ford")).alias("count_below_mean-ford"),
        count_below_mean(pl.col("mitsubishi")).alias("count_below_mean-mitsubishi"),
    )

    expected = df_cls(
        {
            "count_below_mean-volvo": [4],
            "count_below_mean-tesla": [4],
            "count_below_mean-ford": [3],
            "count_below_mean-mitsubishi": [4],
        }
    )
    assert_frame_equal(result, expected, check_dtype=False)


def test_count_above_mean(rental_car_dataset):
    df_cls, dataset = rental_car_dataset
    result = dataset.select(
        count_above_mean(pl.col("volvo")).alias("count_above_mean-volvo"),
        count_above_mean(pl.col("tesla")).alias("count_above_mean-tesla"),
        count_above_mean(pl.col("ford")).alias("count_above_mean-ford"),
        count_above_mean(pl.col("mitsubishi")).alias("count_above_mean-mitsubishi"),
    )

    expected = df_cls(
        {
            "count_above_mean-volvo": [2],
            "count_above_mean-tesla": [2],
            "count_above_mean-ford": [3],
            "count_above_mean-mitsubishi": [2],
        }
    )
    assert_frame_equal(result, expected, check_dtype=False)


def test_has_duplicate(rental_car_dataset):
    df_cls, dataset = rental_car_dataset
    result = dataset.select(
        has_duplicate(pl.col("volvo")).alias("has_duplicate-volvo"),
        has_duplicate(pl.col("tesla")).alias("has_duplicate-tesla"),
        has_duplicate(pl.col("ford")).alias("has_duplicate-ford"),
        has_duplicate(pl.col("toyota")).alias("has_duplicate-toyota"),
    )

    expected = df_cls(
        {
            "has_duplicate-volvo": [True],
            "has_duplicate-tesla": [True],
            "has_duplicate-ford": [True],
            "has_duplicate-toyota": [False],
        }
    )

    assert_frame_equal(result, expected)


def test_has_duplicate_max(rental_car_dataset):
    df_cls, dataset = rental_car_dataset
    result = dataset.select(
        has_duplicate_max(pl.col("mercedes")).alias("has_duplicate_max-mercedes"),
        has_duplicate_max(pl.col("audi")).alias("has_duplicate_max-audi"),
        has_duplicate_max(pl.col("tesla")).alias("has_duplicate_max-tesla"),
        has_duplicate_max(pl.col("toyota")).alias("has_duplicate_max-toyota"),
    )

    expected = df_cls(
        {
            "has_duplicate_max-mercedes": [True],
            "has_duplicate_max-audi": [False],
            "has_duplicate_max-tesla": [False],
            "has_duplicate_max-toyota": [False],
        }
    )

    assert_frame_equal(result, expected)


def test_has_duplicate_min(rental_car_dataset):
    df_cls, dataset = rental_car_dataset
    result = dataset.select(
        has_duplicate_min(pl.col("mercedes")).alias("has_duplicate_min-mercedes"),
        has_duplicate_min(pl.col("audi")).alias("has_duplicate_min-audi"),
        has_duplicate_min(pl.col("tesla")).alias("has_duplicate_min-tesla"),
        has_duplicate_min(pl.col("toyota")).alias("has_duplicate_min-toyota"),
    )

    expected = df_cls(
        {
            "has_duplicate_min-mercedes": [False],
            "has_duplicate_min-audi": [True],
            "has_duplicate_min-tesla": [True],
            "has_duplicate_min-toyota": [False],
        }
    )

    assert_frame_equal(result, expected)
