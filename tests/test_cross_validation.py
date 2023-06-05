import polars as pl
import pytest
from polars.testing import assert_frame_equal

from functime.cross_validation import (
    expanding_window_split,
    sliding_window_split,
    train_test_split,
)


@pytest.fixture(params=[6, 12], ids=lambda x: f"test_size({x})")
def test_size(request):
    return request.param


@pytest.fixture(params=[3, 5, 10], ids=lambda x: f"n_splits({x})")
def n_splits(request):
    return request.param


@pytest.fixture(params=[1, 3], ids=lambda x: f"step_size({x})")
def step_size(request):
    return request.param


def test_train_test_split(test_size, pl_y, benchmark):

    splits = benchmark(lambda y: train_test_split(test_size)(y).collect(), pl_y)

    y = pl_y.collect()
    entity_col, time_col, value_col = y.columns
    assert splits.columns == [
        entity_col,
        f"{time_col}__train",
        f"{value_col}__train",
        f"{time_col}__test",
        f"{value_col}__test",
    ]

    # Check train window lengths
    train_lengths = splits.select(
        [entity_col, pl.col(f"{time_col}__train").arr.lengths()]
    ).sort(by=entity_col)
    expected_train_lengths = (
        y.groupby(entity_col)
        .agg(pl.count().alias(f"{time_col}__train"))
        .with_columns(pl.col(f"{time_col}__train") - test_size)
        .sort(by=entity_col)
    )
    assert_frame_equal(train_lengths, expected_train_lengths)

    # Check test window lengths = test_size
    test_lengths = splits.select(
        pl.col(f"{time_col}__test").arr.lengths().alias("count")
    )
    assert (test_lengths == test_size).select(pl.all().all())[0, 0]


def test_expanding_window_split(test_size, n_splits, step_size, pl_y, benchmark):

    cv = expanding_window_split(
        test_size=test_size, n_splits=n_splits, step_size=step_size
    )
    splits = benchmark(lambda y: cv(y).collect(), pl_y)

    y = pl_y.collect()
    entity_col, time_col, value_col = y.columns
    cv_split_cols = []
    for i in range(n_splits):
        cv_split_cols += [
            f"{time_col}__train_{i}",
            f"{value_col}__train_{i}",
            f"{time_col}__test_{i}",
            f"{value_col}__test_{i}",
        ]

    assert splits.columns == [entity_col, *cv_split_cols]

    # Check test window lengths = test_size
    test_lengths = splits.select([pl.col(f"^{time_col}__test_.*$").arr.lengths()])
    expected = pl.DataFrame({col: [True] for col in test_lengths.columns})
    assert_frame_equal((test_lengths == test_size).select(pl.all().all()), expected)


def test_sliding_window_split(test_size, n_splits, step_size, pl_y, benchmark):

    cv = sliding_window_split(
        test_size=test_size,
        n_splits=n_splits,
        step_size=step_size,
    )
    splits = benchmark(lambda y: cv(y).collect(), pl_y)

    y = pl_y.collect()
    entity_col, time_col, value_col = y.columns
    cv_split_cols = []
    for i in range(n_splits):
        cv_split_cols += [
            f"{time_col}__train_{i}",
            f"{value_col}__train_{i}",
            f"{time_col}__test_{i}",
            f"{value_col}__test_{i}",
        ]
    assert splits.columns == [entity_col, *cv_split_cols]

    # Check test window lengths = test_size
    test_lengths = splits.select([pl.col(f"^{time_col}__test_.*$").arr.lengths()])
    expected = pl.DataFrame({col: [True] for col in test_lengths.columns})
    assert_frame_equal((test_lengths == test_size).select(pl.all().all()), expected)
