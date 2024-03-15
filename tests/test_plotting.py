import polars as pl
from functime import plotting
import pytest

def test_set_subplot_default_kwargs_no_existing_kwargs():
    kwargs = {}
    updated_kwargs = plotting._set_subplot_default_kwargs(kwargs, 2, 3)

    assert updated_kwargs["width"] == 250 * 3 + 100 # default width * cols + space
    assert updated_kwargs["height"] == 200 * 2 + 100 # default height * rows + space
    assert updated_kwargs["template"] == "plotly_white"


def test_set_subplot_default_kwargs_with_one_defined_kwarg():
    kwargs = {"width": 800, "some_other_kwarg": "value"}
    updated_kwargs = plotting._set_subplot_default_kwargs(kwargs, 2, 3)

    assert updated_kwargs["width"] == 800  # Should remain unchanged
    assert updated_kwargs["height"] ==  200 * 2 + 100 # default height * rows + space
    assert updated_kwargs["some_other_kwarg"] == "value"


@pytest.mark.parametrize("n_series, n_cols, expected_rows", [
    (10, 2, 5),   # 10 series in 2 columns > 5 rows
    (10, 1, 10),  # All series in one column > 10 rows
    (10, 3, 4),   # Series not exactly divisible by columns
    (10, 10, 1),  # Each series in its own column
    (10, 15, 1),  # More columns than series
])
def test_calculate_subplot_n_rows(n_series, n_cols, expected_rows):
    assert plotting._calculate_subplot_n_rows(n_series, n_cols) == expected_rows


@pytest.mark.parametrize("n_series, n_cols", [
    (0, 2),  # No series
    (10, 0),  # Zero columns
    (-1, 2),  # Negative series
    (10, -2), # Negative columns
])


def test_calculate_subplot_n_rows_errors(n_series, n_cols):
    with pytest.raises(ValueError):
        plotting._calculate_subplot_n_rows(n_series, n_cols)


def create_mock_dataframe():
    # Create a mock DataFrame for testing
    data = {
        "entity": ["A", "A", "B", "B", "C", "C"],
        "time": [1, 2, 1, 2, 1, 2],
        "value": [10, 20, 30, 40, 50, 60]
    }
    return pl.DataFrame(data)


@pytest.mark.parametrize("n_series, last_n, expected_entities", [
    (2, 1, {"A", "B"}),  # Test with 2 series, last 1 record
    (3, 2, {"A", "B", "C"}),  # Test with all series, last 2 records
    (4, 2, {"A", "B", "C"}),  # More series than available
])


def test_prepare_data_for_subplots(n_series, last_n, expected_entities):
    df = create_mock_dataframe()
    entities_sample, _, y_filtered = plotting._prepare_data_for_subplots(df, n_series, last_n, seed=1)

    # Check if the correct entities are sampled
    assert set(entities_sample) == expected_entities

    # Check if the data is correctly filtered
    for entity in entities_sample:
        assert y_filtered.filter(pl.col("entity") == entity).height <= last_n


@pytest.mark.parametrize("i, n_cols, expected_row_col", [
    (0, 3, (1, 1)),  # First series & 3 cols pos = 1,1
    (1, 3, (1, 2)),  # Second series & 3 cols pos = 1,2
    (2, 3, (1, 3)),  # Third series & 3 cols pos = 1,3
    (3, 3, (2, 1)),  # Fourth series, start of second row
    (26, 3, (9, 3)), # 27th series in a 3-column layout
    (27, 3, (10, 1)), # 28th series, starts a new row
    (160, 7, (23, 7)), # 161st series, ends in the last col
    (161, 7, (24, 1)), # 162nd series, starts a new row
    
])
def test_get_subplot_grid_position(i, n_cols, expected_row_col):
    assert plotting._get_subplot_grid_position(i, n_cols) == expected_row_col