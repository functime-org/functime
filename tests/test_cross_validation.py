import polars as pl
import pytest

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
    def _split(y):
        y_train, y_test = train_test_split(test_size)(y)
        return pl.collect_all([y_train, y_test])

    y_train, y_test = benchmark(_split, pl_y)

    # Check column names
    entity_col, time_col = pl_y.columns[:2]
    assert y_train.columns == y_test.columns

    # Check train window lengths
    ts_lengths = (
        pl_y.group_by(entity_col, maintain_order=True)
        .agg(pl.col(time_col).count())
        .collect()
    )
    train_lengths = y_train.group_by(entity_col, maintain_order=True).agg(
        pl.col(time_col).count()
    )
    assert (
        (ts_lengths.select("time") - train_lengths.select("time")) == test_size
    ).select(pl.all().all())[0, 0]

    # Check test window lengths
    test_lengths = y_test.group_by(entity_col).agg(pl.col(time_col).count())
    assert (test_lengths.select("time") == test_size).select(pl.all().all())[0, 0]


def test_expanding_window_split(test_size, n_splits, step_size, pl_y, benchmark):
    def _split(y):
        cv = expanding_window_split(
            test_size=test_size, n_splits=n_splits, step_size=step_size
        )
        splits = cv(y)
        return {i: pl.collect_all(s) for i, s in splits.items()}

    splits = benchmark(_split, pl_y)
    entity_col, time_col = pl_y.columns[:2]

    for split in splits.values():
        _, y_test = split
        # Check test window lengths
        test_lengths = y_test.group_by(entity_col, maintain_order=True).agg(
            pl.col(time_col).count()
        )
        assert (test_lengths.select("time") == test_size).select(pl.all().all())[0, 0]


def test_sliding_window_split(test_size, n_splits, step_size, pl_y, benchmark):
    def _split(y):
        cv = sliding_window_split(
            test_size=test_size,
            n_splits=n_splits,
            step_size=step_size,
        )
        splits = cv(y)
        return {i: pl.collect_all(s) for i, s in splits.items()}

    splits = benchmark(_split, pl_y)
    entity_col, time_col = pl_y.columns[:2]

    for split in splits.values():
        _, y_test = split
        # Check test window lengths
        test_lengths = y_test.group_by(entity_col, maintain_order=True).agg(
            pl.col(time_col).count()
        )
        assert (test_lengths.select("time") == test_size).select(pl.all().all())[0, 0]
