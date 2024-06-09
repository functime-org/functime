from __future__ import annotations

import polars as pl
import pytest

from functime._plotting import (
    get_chosen_entities,
    get_num_rows,
    get_subplot_grid_position,
)


@pytest.fixture
def mock_dataframe():
    data = {
        "entity": ["A", "A", "B", "B", "C", "C"],
        "time": [1, 2, 1, 2, 1, 2],
        "value": [10, 20, 30, 40, 50, 60],
    }
    return pl.LazyFrame(data)


def test_get_chosen_entities_not_random(mock_dataframe):
    actual = get_chosen_entities(y=mock_dataframe, num_series=3, seed=None)
    expected = ["A", "B", "C"]
    assert actual == expected


@pytest.mark.parametrize("n_series, seed", [(3, 42), (2, 42)])
def test_get_chosen_entities_random(mock_dataframe, n_series, seed):
    expected = (
        mock_dataframe.select(pl.col("entity").unique(maintain_order=True))
        .collect()
        .sample(n_series, seed=seed)
        .to_series()
        .to_list()
    )

    actual = get_chosen_entities(
        y=mock_dataframe,
        num_series=n_series,
        seed=seed,
    )

    assert actual == expected


@pytest.mark.parametrize(
    "n_series, n_cols, expected_rows",
    [
        (10, 2, 5),  # 10 series in 2 columns > 5 rows
        (10, 1, 10),  # All series in one column > 10 rows
        (10, 3, 4),  # Series not exactly divisible by columns
        (10, 10, 1),  # Each series in its own column
        (10, 15, 1),  # More columns than series
    ],
)
def test_get_num_rows(n_series, n_cols, expected_rows):
    assert get_num_rows(num_series=n_series, num_cols=n_cols) == expected_rows


@pytest.mark.parametrize(
    "n_series, n_cols",
    [(0, 1), (1, 0), (0, 0)],
)
def test_get_num_rows_raises_value_error(n_series, n_cols):
    with pytest.raises(ValueError):
        get_num_rows(num_series=n_series, num_cols=n_cols)


@pytest.mark.parametrize(
    "i, n_cols, expected_row_col",
    [
        (0, 3, (1, 1)),  # First series & 3 cols pos = 1,1
        (1, 3, (1, 2)),  # Second series & 3 cols pos = 1,2
        (2, 3, (1, 3)),  # Third series & 3 cols pos = 1,3
        (3, 3, (2, 1)),  # Fourth series, start of second row
        (26, 3, (9, 3)),  # 27th series in a 3-column layout
        (27, 3, (10, 1)),  # 28th series, starts a new row
        (160, 7, (23, 7)),  # 161st series, ends in the last col
        (161, 7, (24, 1)),  # 162nd series, starts a new row
    ],
)
def test_get_subplot_grid_position(i, n_cols, expected_row_col):
    assert get_subplot_grid_position(element=i, num_cols=n_cols) == expected_row_col
