import polars as pl
from polars.testing import assert_series_equal
from functime.feature_extraction.tsfresh_mathieu import benford_correlation, _get_length_sequences_where, longest_strike_below_mean, longest_strike_above_mean, mean_n_absolute_max, percent_reocurring_points, percent_recoccuring_values, sum_reocurring_points, sum_reocurring_values


@pytest.mark.parametrize("x, res", [
    (pl.Series([26.24, 3.03, -2.92, 3.5, -0.07, 0.35, 0.10, 0.51, -0.43]), 0.39), 
    (pl.Series([]), 1),
])
def test_benford_correlation():
    pass


def test__get_length_sequences_where():
    pass