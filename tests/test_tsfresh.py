import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal
import inspect

# percent_recoccuring_values,
from functime.feature_extraction.tsfresh import (
    absolute_energy,
    absolute_maximum,
    absolute_sum_of_changes,
    approximate_entropy,
    autocorrelation,
    autoregressive_coefficients,
    binned_entropy,
    benford_correlation,
    first_location_of_maximum,
    first_location_of_minimum,
    has_duplicate,
    has_duplicate_max,
    has_duplicate_min,
    index_mass_quantile,
    last_location_of_maximum,
    last_location_of_minimum,
    c3,
    change_quantiles,
    cid_ce,
    count_above,
    count_above_mean,
    count_below,
    count_below_mean,
    longest_streak_above_mean,
    longest_streak_below_mean,
    mean_n_absolute_max,
    mean_second_derivative_central,
    percent_reocurring_points,
    sum_reocurring_points,
    sum_reocurring_values,
    number_peaks,
    symmetry_looking,
    time_reversal_asymmetry_statistic,
    approximate_entropy,
    percent_reoccuring_values,
    lempel_ziv_complexity,
    range_over_mean,
    range_change,
    longest_winning_streak,
    longest_streak_above,
    longest_losing_streak,
    longest_streak_below,
    max_abs_change,
)

np.random.seed(42)

@pytest.mark.parametrize("S, res", [
    ([-5, 0, 1], [26]),
    ([0], [0]),
    ([-1, 2, -3], [14]),
    ([-1, 1.3], [2.6900000000000004]),
    ([1], [1])
])
def test_abolute_energy(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            absolute_energy(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            absolute_energy(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert absolute_energy(pl.Series(S)) == res[0]


@pytest.mark.parametrize("S, res", [
    ([-5, 0, 1], [5]),
    ([0], [0]),
    ([-1.0, 2.0, -3.0], [3.0]),
])
def test_absolute_maximum(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            absolute_maximum(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("max", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            absolute_maximum(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("max", res))
    )
    assert absolute_maximum(pl.Series(S)) == res[0]


@pytest.mark.parametrize("S, res", [
    ([1, 1, 1, 1, 2, 1], [2]),
    ([1.4, -1.3, 1.7, -1.2], [8.6]),
    ([1], [0])
])
def test_absolute_sum_of_changes(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            absolute_sum_of_changes(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            absolute_sum_of_changes(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert absolute_sum_of_changes(pl.Series(S)) == res[0]


@pytest.mark.parametrize("S, res, m, r, scale", [
    ([1], 0, 2, 0.5, False),
    ([12, 13, 15, 16, 17] * 10, 0.282456191276673, 2, 0.9, True),
    ([1.4, -1.3, 1.7, -1.2], 0.0566330122651324, 2, 0.5, False),
    ([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 0.002223871246127107, 2, 0.5, False),
    ([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1], 0.47133806162842484, 2, 0.5, False),
    ([85, 80, 89] * 17, 1.099654110658932e-05, 2, 3, False),
    ([85, 80, 89] * 17, 0.0, 2, 3, True)
])
def test_approximate_entropy(S, res, m, r, scale):
    assert approximate_entropy(x = pl.Series(S), run_length=m, filtering_level=r, scale_by_std=scale) == res

@pytest.mark.parametrize("S, res, n_lags", [
    ([1, 2, 1, 2, 1, 2], [-1.0], 1),
    ([1, 2, 1, 2, 1, 2], [1.0], 2),
    ([1, 2, 1, 2, 1, 2], [1.0], 4),
    ([0, 1, 2, 0, 1, 2], [-0.75], 2)
])
def test_autocorrelation(S, res, n_lags):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            autocorrelation(pl.col("a"), n_lags=n_lags)
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            autocorrelation(pl.col("a"), n_lags=n_lags)
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert autocorrelation(pl.Series(S), n_lags=n_lags) == res[0]

@pytest.mark.parametrize("S, res, n_lags", [
    ([1, 2, 1, 2, 1, 2], [1.0], 0)
])
def test_autocorrelation_shortcut(S, res, n_lags):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            autocorrelation(pl.col("a"), n_lags=n_lags)
        ),
        pl.DataFrame(pl.Series("literal", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            autocorrelation(pl.col("a"), n_lags=n_lags)
        ).collect(),
        pl.DataFrame(pl.Series("literal", res))
    )
    assert autocorrelation(pl.Series(S), n_lags=n_lags) == res[0]



@pytest.mark.parametrize("S, res, bin_count", [
    ([10] * 100, [-0.0], 10),
    ([10] * 10 + [1], [0.30463609734923813], 10),
    (list(range(10)), [2.302585092994046], 100)
])
def test_binned_entropy(S, res, bin_count):
    # Doesn't work for lazy mode
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            binned_entropy(pl.col("a"), bin_count=bin_count)
        ),
        pl.DataFrame(pl.Series("counts", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            binned_entropy(pl.col("a"), bin_count=bin_count)
        ).collect(),
        pl.DataFrame(pl.Series("counts", res))
    )
    assert binned_entropy(pl.Series(S), bin_count=bin_count) == res[0]

@pytest.mark.parametrize("S, res, n_lags", [
    ([1, 2, -3, 4], [-15.0], 1),
    ([1]*10, [1.0], 1),
    ([1]*10, [1.0], 2),
    ([1]*10, [1.0], 3)
])
def test_c3(S, res, n_lags):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            c3(pl.col("a"), n_lags=n_lags)
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            c3(pl.col("a"), n_lags=n_lags)
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert c3(pl.Series(S), n_lags=n_lags) == res[0]

@pytest.mark.parametrize("S, res, n_lags", [
    ([1, 2, -3, 4], [np.nan], 2),
    ([1, 2, -3, 4], [0.0], 3)
])
def test_c3_not_define(S, res, n_lags):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            c3(pl.col("a"), n_lags=n_lags)
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            c3(pl.col("a"), n_lags=n_lags)
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert np.isnan(c3(pl.Series(S), n_lags=n_lags))

@pytest.mark.parametrize("S, res, q_low, q_high, is_abs", [
    ([0, 1, -9, 0, 0, 1, 0], [[1, 0, 1, 1]], 0.1, 0.9, True),
    ([0, 1, -9, 0, 0, 1, 0], [[1, 0, 1, -1]], 0.1, 0.9, False),
    (list(range(10)), [[1, 1, 1]], 0.25, 0.75, True)
])
def test_change_quantiles(S, res, q_low, q_high, is_abs):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            change_quantiles(pl.col("a"), q_low=q_low, q_high=q_high, is_abs=is_abs)
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            change_quantiles(pl.col("a"), q_low=q_low, q_high=q_high, is_abs=is_abs)
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_series_equal(
        change_quantiles(pl.Series(S), q_low=q_low, q_high=q_high, is_abs=is_abs),
        pl.Series(res[0])
    )

@pytest.mark.parametrize("S, res, normalize", [
    ([1, 1, 1], [0.0], False),
    ([0, 4], [2.0], True),
    ([100, 104], [2.0], True),
    ([-4.33, -1.33, 2.67], [5.0], False)
])
def test_cid_ce(S, res, normalize):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            cid_ce(pl.col("a"), normalize=normalize)
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            cid_ce(pl.col("a"), normalize=normalize)
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert cid_ce(pl.Series(S), normalize=normalize) == res[0]


@pytest.mark.parametrize("S, res, normalize", [
    ([1, 1, 1], [np.nan], True)
])
def test_cid_ce_nan_case(S, res, normalize):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            cid_ce(pl.col("a"), normalize=normalize)
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            cid_ce(pl.col("a"), normalize=normalize)
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    np.isnan(cid_ce(pl.Series(S), normalize=normalize) == res[0])

@pytest.mark.parametrize("S, res, threshold", [
    ([0.1, 0.2, 0.3] * 3, [200 / 3], 0.2),
    ([1] * 10, [100.0], 1.0),
    (list(range(10)), [100.0], 0),
    (list(range(10)), [50.0], 5)
])
def test_count_above(S, res, threshold):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            count_above(pl.col("a"), threshold=threshold)
        ),
        pl.DataFrame(pl.Series("literal", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            count_above(pl.col("a"), threshold=threshold)
        ).collect(),
        pl.DataFrame(pl.Series("literal", res))
    )
    assert count_above(pl.Series(S), threshold=threshold) == res[0]

@pytest.mark.parametrize("S, res, threshold", [
    ([0.1, 0.2, 0.3] * 3, [200 / 3], 0.2),
    ([1] * 10, [100.0], 1),
    (list(range(10)), [60.0], 5),
    (list(range(10)), [10.0], 0)
])
def test_count_below(S, res, threshold):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            count_below(pl.col("a"), threshold=threshold)
        ),
        pl.DataFrame(pl.Series("literal", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            count_below(pl.col("a"), threshold=threshold)
        ).collect(),
        pl.DataFrame(pl.Series("literal", res))
    )
    assert count_below(pl.Series(S), threshold=threshold) == res[0]


@pytest.mark.parametrize("S, res", [
    ([1, 2, 1, 2, 1, 2], [3]),
    ([1, 1, 1, 1, 1, 2], [1]),
    ([1, 1, 1, 1, 1], [0])
])
def test_count_above_mean(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            count_above_mean(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res, dtype=pl.UInt32))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            count_above_mean(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res, dtype=pl.UInt32))
    )
    assert count_above_mean(pl.Series(S, dtype=pl.UInt32)) == res[0]

@pytest.mark.parametrize("S, res", [
    ([1, 2, 1, 2, 1, 2], [3]),
    ([1, 1, 1, 1, 1, 2], [5]),
    ([1, 1, 1, 1, 1], [0])
])
def test_count_below_mean(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            count_below_mean(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res, dtype=pl.UInt32))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            count_below_mean(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res, dtype=pl.UInt32))
    )
    assert count_below_mean(pl.Series(S, dtype=pl.UInt32)) == res[0]

@pytest.mark.parametrize("S, res", [
    ([1, 2, 1, 2, 1], [0.2]),
    ([1.5, 2.6, 1.8, 2.1, 1.0], [0.2]),
    ([2, 1, 1, 1, 1], [0.0]),
    ([1, 1, 1, 1, 1], [0.0])
])
def test_first_location_of_maximum(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            first_location_of_maximum(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            first_location_of_maximum(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert first_location_of_maximum(pl.Series(S)) == res[0]

@pytest.mark.parametrize("S, res", [
    ([1, 2, 1, 2, 1], [0.0]),
    ([2, 1, 1, 1, 2], [0.2]),
    ([2.7, 1.05, 1.2, 1.068, 2.3], [0.2]),
    ([1, 1, 1, 1, 1], [0.0])
])
def test_first_location_of_minimum(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            first_location_of_minimum(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            first_location_of_minimum(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert first_location_of_minimum(pl.Series(S)) == res[0]


@pytest.mark.parametrize("S, res", [
    ([2.1, 0, 0, 2.1, 1.1], [True]),
    ([2.1, 0, 4, 2, 1.1], [False])
])
def test_has_duplicate(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            has_duplicate(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            has_duplicate(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert has_duplicate(pl.Series(S)) == res[0]


@pytest.mark.parametrize("S, res", [
    ([-2.1, 0, 0, -2.1, 1.1], [True]),
    ([2.1, 0, -1, 2, 1.1], [False]),
    ([1, 1, 1, 1], [True]),
    ([0], [False])
])
def test_has_duplicate_min(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            has_duplicate_min(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            has_duplicate_min(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert has_duplicate_min(pl.Series(S)) == res[0]

@pytest.mark.parametrize("S, res", [
    ([2.1, 0, 0, 2.1, 1.1], [True]),
    ([2.1, 0, 0, 2, 1.1], [False]),
    ([1, 1, 1, 1], [True]),
    ([0], [False])
])
def test_has_duplicate_max(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            has_duplicate_max(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            has_duplicate_max(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert has_duplicate_max(pl.Series(S)) == res[0]


@pytest.mark.parametrize("S, res, q", [
    ([1] * 101, [0.504950495049505], 0.5),
    ([0, 1, 1, 0, 0, 1, 0, 0], [0.25], 0.3),
    ([0, 1, 1, 0, 0, 1, 0, 0], [0.375], 0.6),
    ([0, 1, 1, 0, 0, 1, 0, 0], [0.75], 0.9)
])
def test_index_mass_quantile(S, res, q):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            index_mass_quantile(pl.col("a"), q)
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            index_mass_quantile(pl.col("a"), q)
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )
    assert index_mass_quantile(pl.Series(S), q) == res[0]

@pytest.mark.parametrize("S, res", [
    ([1, 2, 1, 2, 1], [1.0]),
    ([1, 2, 1, 2, 2], [0.6]),
    ([2.7, 1.05, 1.2, 1.068, 2.3], [0.4]),
    ([2, 1, 1, 1, 2], [0.8])
])
def test_last_location_of_minimum(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            last_location_of_minimum(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("literal", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            last_location_of_minimum(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("literal", res))
    )
    assert last_location_of_minimum(pl.Series(S)) == res[0]


@pytest.mark.parametrize("S, res", [
    ([1, 2, 1, 2, 1], [0.8]),
    ([1, 2, 1, 1, 2], [1.0]),
    ([2.7, 1.05, 1.2, 1.068, 2.3], [0.19999999999999996]),
    ([2, 1, 1, 1, 1], [0.19999999999999996])
])
def test_last_location_of_maximum(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            last_location_of_maximum(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("literal", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            last_location_of_maximum(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("literal", res))
    )
    assert last_location_of_maximum(pl.Series(S)) == res[0]

def test_benford_correlation():
    # Nan, division by 0
    X_uniform = pl.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9]
    })
    X_uniform_lazy = pl.LazyFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9]
    })
    # Random serie
    X_random = pl.DataFrame({
        "a": [26.24, 3.03, -2.92, 3.5, -0.07, 0.35, 0.10, 0.51, -0.43]
    })
    X_random_lazy = pl.LazyFrame({
        "a": [26.24, 3.03, -2.92, 3.5, -0.07, 0.35, 0.10, 0.51, -0.43]
    })
    # Fibo, distribution same as benford law
    l_fibo = [0, 1]
    for i in range(2, 50):
        l_fibo.append(l_fibo[i - 1] + l_fibo[i - 2])
    
    X_fibo = pl.DataFrame({
        "a": l_fibo
    })

    X_fibo_lazy = pl.LazyFrame({
        "a": l_fibo
    })
    assert_frame_equal(
        X_uniform.select(
            benford_correlation(pl.col("a"))
        ),
        pl.DataFrame({"counts": [np.nan]})
    )
    assert_frame_equal(
        X_uniform_lazy.select(
            benford_correlation(pl.col("a"))
        ).collect(),
        pl.DataFrame({"counts": [np.nan]})
    )
    assert_frame_equal(
        X_random.select(
            benford_correlation(pl.col("a"))
        ),
        pl.DataFrame({"counts": [0.39753280229716703]})
    )
    assert_frame_equal(
        X_random_lazy.select(
            benford_correlation(pl.col("a"))
        ).collect(),
        pl.DataFrame({"counts": [0.39753280229716703]})
    )
    assert_frame_equal(
        X_fibo.select(
            benford_correlation(pl.col("a"))
        ),
        pl.DataFrame({"counts": [0.9959632739083689]})
    )
    assert_frame_equal(
        X_fibo_lazy.select(
            benford_correlation(pl.col("a"))
        ).collect(),
        pl.DataFrame({"counts": [0.9959632739083689]})
    )

@pytest.mark.parametrize("S, res", [
    ([1, 2, 1, 1, 1, 2, 2, 2], [3]),
    ([1, 2, 3, 4, 5, 6], [3]),
    ([1, 2, 3, 4, 5], [2]),
    ([1, 2, 1], [1]),
    ([1, 1, 1], [0]),
    ([], [0])
])
def test_longest_streak_below_mean(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            longest_streak_below_mean(pl.col("a")).alias("lengths")
        ),
        pl.DataFrame(pl.Series("lengths", res, dtype=pl.UInt64))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            longest_streak_below_mean(pl.col("a")).alias("lengths")
        ).collect(),
        pl.DataFrame(pl.Series("lengths", res, dtype=pl.UInt64))
    )


@pytest.mark.parametrize("S, res", [
    ([1, 2, 1, 2, 1, 2, 2, 1], [2]),
    ([1, 2, 3, 4, 5, 6], [3]),
    ([1, 2, 3, 4, 5], [2]),
    ([1, 2, 1], [1]),
    ([1, 1, 1], [0]),
    ([], [0])
])
def test_longest_streak_above_mean(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            longest_streak_above_mean(pl.col("a")).alias("lengths")
        ),
        pl.DataFrame(pl.Series("lengths", res, dtype=pl.UInt64))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            longest_streak_above_mean(pl.col("a")).alias("lengths")
        ).collect(),
        pl.DataFrame(pl.Series("lengths", res, dtype=pl.UInt64))
    )

@pytest.mark.parametrize("S, n_max, res", [
    ([], 1, [None]),
    ([12, 3], 10, [7.5]),
    ([-1, -5, 4, 10], 3, [6.333333]),
    ([0, -5, -9], 2, [7.0]),
    ([0, 0, 0], 1, [0.0])
])
def test_mean_n_absolute_max(S, n_max, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            mean_n_absolute_max(pl.col("a"), n_maxima=n_max).cast(pl.Float64)
        ),
        pl.DataFrame(pl.Series("a", res, dtype=pl.Float64))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            mean_n_absolute_max(pl.col("a"), n_maxima=n_max).cast(pl.Float64)
        ).collect(),
        pl.DataFrame(pl.Series("a", res, dtype=pl.Float64))
    )

def test_mean_n_absolute_max_value_error():
    with pytest.raises(ValueError):
        mean_n_absolute_max(
            x = pl.Series([12, 3]),
            n_maxima = 0
        )
    with pytest.raises(ValueError):
        mean_n_absolute_max(
            x = pl.Series([12, 3]),
            n_maxima = -1
        )


@pytest.mark.parametrize("S, res", [
    ([1, 1, 2, 3, 4], [0.4]),
    ([1, 1.5, 2, 3], [0]),
    ([1], [0]),
    ([1.111, -2.45, 1.111, 2.45], [0.5]),
    ([], [np.nan])
])
def test_percent_reoccuring_values(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            percent_reoccuring_values(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res, dtype=pl.Float64))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            percent_reoccuring_values(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res, dtype=pl.Float64))
    )


@pytest.mark.parametrize("S, res", [
    ([1, 1, 2, 3, 4], [0.25]),
    ([1, 1.5, 2, 3], [0]),
    ([1], [0]),
    ([1.111, -2.45, 1.111, 2.45], [1.0 / 3.0])
])
def test_percent_reoccuring_values(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            percent_reoccuring_values(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res, dtype=pl.Float64))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            percent_reoccuring_values(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res, dtype=pl.Float64))
    )


@pytest.mark.parametrize("S, res", [
    ([1, 1, 2, 3, 4, 4], [10]),
    ([1, 1.5, 2, 3], [0.0]),
    ([1], [0]),
    ([1.111, -2.45, 1.111, 2.45], [2.222])
])
def test_sum_reocurring_points(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            sum_reocurring_points(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            sum_reocurring_points(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )


@pytest.mark.parametrize("S, res", [
    ([1, 1, 2, 3, 4, 4], [5]),
    ([1, 1.5, 2, 3], [0.0]),
    ([1], [0]),
    ([1.111, -2.45, 1.111, 2.45], [1.111])
])
def test_sum_reocurring_values(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            sum_reocurring_values(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            sum_reocurring_values(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res))
    )


@pytest.mark.parametrize("S, res", [
    ([1, 1, 2, 3, 4], [0.4]),
    ([1, 1.5, 2, 3], [0]),
    ([1], [0]),
    ([1.111, -2.45, 1.111, 2.45], [0.5]),
    ([], [np.nan])
])
def test_percent_reocurring_points(S, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            percent_reocurring_points(pl.col("a"))
        ),
        pl.DataFrame(pl.Series("a", res, dtype=pl.Float64))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            percent_reocurring_points(pl.col("a"))
        ).collect(),
        pl.DataFrame(pl.Series("a", res, dtype=pl.Float64))
    )


@pytest.mark.parametrize("S, n, res", [
    ([0, 5, 2, 3, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1], 1, [3]),
    ([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1], 2, [2]),
    ([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1], 3, [2]),
    ([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1], 4, [1])
])
def test_number_peaks(S, n, res):
    assert_frame_equal(
        pl.DataFrame(
            {"a": S}
        ).select(
            number_peaks(pl.col("a"), n).alias("a")
        ),
        pl.DataFrame(pl.Series("a", res, dtype=pl.UInt32))
    )
    assert_frame_equal(
        pl.LazyFrame(
            {"a": S}
        ).select(
            number_peaks(pl.col("a"), n).alias("a")
        ).collect(),
        pl.DataFrame(pl.Series("a", res, dtype=pl.UInt32))
    )


def generate_ar1():
    np.random.seed(42)
    e = np.random.normal(0.1, 0.1, size=100)
    m = 50
    x = [0] * m
    x[0] = 100
    for i in range(1, m):
        x[i] = x[i - 1] * 0.5 + e[i]
    return pl.Series(x, dtype=pl.Float64)


@pytest.mark.parametrize(
    "x, param, res",
    [
        (
            pl.Series(np.cumsum(np.random.uniform(size=100))),
            [
                {"autolag": "BIC", "attr": "teststat"},
                {"autolag": "BIC", "attr": "pvalue"},
                {"autolag": "BIC", "attr": "usedlag"},
            ],
            pl.DataFrame(
                [
                    [
                        'attr_"teststat"__autolag_"BIC"',
                        'attr_"pvalue"__autolag_"BIC"',
                        'attr_"usedlag"__autolag_"BIC"',
                    ],
                    [0.037064, 0.961492, 0],
                ],
                schema=["index", "res"],
            ),
        ),
        (
            generate_ar1(),
            [
                {"autolag": "AIC", "attr": "teststat"},
                {"autolag": "AIC", "attr": "pvalue"},
                {"autolag": "AIC", "attr": "usedlag"},
            ],
            pl.DataFrame(
                [
                    [
                        'attr_"teststat"__autolag_"AIC"',
                        'attr_"pvalue"__autolag_"AIC"',
                        'attr_"usedlag"__autolag_"AIC"',
                    ],
                    [-595.259534, 0, 0],
                ],
                schema=["index", "res"],
            ),
        ),
    ],
)
def test_augmented_dickey_fuller(x, param, res):
    # assert_frame_equal(augmented_dickey_fuller(x, param), res, atol=1e-7)
    # res_linalg_error = (
    #     augmented_dickey_fuller(x=pl.Series(np.repeat(np.nan, 100)), param=param)
    #     .get_column("res")
    #     .to_numpy()
    # )
    # assert all(np.isnan(res_linalg_error))
    #
    # res_value_error = (
    #     augmented_dickey_fuller(x=pl.Series([]), param=param)
    #     .get_column("res")
    #     .to_numpy()
    # )
    # assert all(np.isnan(res_value_error))
    #
    # # Should return NaN if "attr" is unknown
    # res_attr_error = (
    #     augmented_dickey_fuller(x=x, param=[{"autolag": "AIC", "attr": ""}])
    #     .get_column("res")
    #     .to_numpy()
    # )
    # assert all(np.isnan(res_attr_error))
    assert True


@pytest.mark.parametrize(
    "x, res",
    [
        (pl.Series(range(10)), pl.Series([0.0])),
        (pl.Series([1, 3, 5]), pl.Series([0.0])),
        (pl.Series([1, 3, 7, -3]), pl.Series([-3.0])),
    ],
)
def test_mean_second_derivative_central(x, res):
    assert_series_equal(
        mean_second_derivative_central(x),
        res
    )
# This test needs to be rewritten..
@pytest.mark.parametrize(
    "x, r, res",
    [
        (
            pl.Series([-1, -1, 1, 1]),
            0.05,
            True
        ),
        (
            pl.Series([-2, -1, 0, 1, 1]),
            0.05,
            False
        ),
        (
            pl.Series([-2, -1, 0, 1, 1]),
            0.1,
            True
        ),
    ],
)
def test_symmetry_looking(x, r, res):
    ans = symmetry_looking(x, r)
    assert ans == res

    df = x.to_frame()
    assert_frame_equal(
        df.select(
            symmetry_looking(pl.col(x.name), ratio=r)
        ),
        pl.DataFrame({x.name:[res]})
    )


@pytest.mark.parametrize(
    "x, lag, res", [(pl.Series([1] * 10), 0, 0), (pl.Series([1, 2, -3, 4]), 1, -10)]
)
def test_time_reversal_asymmetry_statistic(x, lag, res):
    assert time_reversal_asymmetry_statistic(x, lag) == res


def test_lempel_ziv_complexity():
    a = pl.Series([1,0,0,1,1,1,1,0,1,1,0,0,0,0,1,0])
    assert lempel_ziv_complexity(a, threshold = 0) * len(a) == 8
    a = pl.Series([1,0,0,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0])
    assert lempel_ziv_complexity(a, threshold = 0) * len(a) == 9
    a = pl.Series([1,0,0,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0])
    assert lempel_ziv_complexity(a, threshold = 0) * len(a) == 10


@pytest.mark.parametrize("S, res", [
    (list(range(100)), 99),
    ([0, 0 , 0, 0, -1, 2, -3, 1], 3),
    (list(range(100, 0, -1)), 0)
])
def test_longest_streak_above(S, res):

    x = pl.Series(S)
    assert longest_streak_above(x, threshold=0) == res
    df = x.to_frame()
    assert_frame_equal(
        df.select(
            longest_streak_above(pl.col(x.name), threshold=0).alias(x.name).cast(pl.Int64)
        ),
        pl.DataFrame({x.name:[res]})
    )

    assert_frame_equal(
        df.lazy().select(
            longest_streak_above(pl.col(x.name), threshold=0).alias(x.name).cast(pl.Int64)
        ).collect(),
        pl.DataFrame({x.name:[res]})
    )

@pytest.mark.parametrize("S, res", [
    (list(range(100)), 0),
    ([0, 0, 0, 0, -1, 2, -3, 1], 4),
    (list(range(100, 0, -1)), 99)
])
def test_longest_streak_below(S, res):

    x = pl.Series(S)
    assert longest_streak_below(x, threshold=0) == res
    df = x.to_frame()
    assert_frame_equal(
        df.select(
            longest_streak_below(pl.col(x.name), threshold=0).alias(x.name).cast(pl.Int64)
        ),
        pl.DataFrame({x.name:[res]})
    )

    assert_frame_equal(
        df.lazy().select(
            longest_streak_below(pl.col(x.name), threshold=0).alias(x.name).cast(pl.Int64)
        ).collect(),
        pl.DataFrame({x.name:[res]})
    )

@pytest.mark.parametrize("S, res", [
    (list(range(100)), 1),
    ([0, -100, 1,2,3,4,5,6,7,8,9], 101),
    ([-50, -100, 200, 3, 9, 12], 300)
])
def test_max_abs_change(S, res):

    x = pl.Series(S)
    assert max_abs_change(x) == res
    df = x.to_frame()
    assert_frame_equal(
        df.select(
            max_abs_change(pl.col(x.name)).alias(x.name)
        ),
        pl.DataFrame({x.name:[res]})
    )
    assert_frame_equal(
        df.lazy().select(
            max_abs_change(pl.col(x.name)).alias(x.name)
        ).collect(),
        pl.DataFrame({x.name:[res]})
    )

@pytest.mark.parametrize("S, res", [
    ([1, 1, 1, 1, 1], 0.),
    ([1, 2, 3, 4, 5, 6, 7], 1.5),
    ([1], 0.),
    ([0.1, 0.2, 0.8, 0.9], 1.6)
])
def test_range_over_mean_and_range(S, res):

    # The tests here are non-exhaustive, but is good enough
    x = pl.Series(S)
    assert range_over_mean(x) == res
    range_ = (np.max(S) - np.min(S))
    range_chg_pct = range_ / np.min(S)
    assert range_change(x, percentage=False) == range_
    assert range_change(x, percentage=True) == range_chg_pct
    df = x.to_frame()
    assert_frame_equal(
        df.select(
            range_change(pl.col(x.name), percentage=False)
        ),
        pl.DataFrame({x.name:[range_]})
    )
    assert_frame_equal(
        df.select(
            range_over_mean(pl.col(x.name))
        ),
        pl.DataFrame({x.name:[res]})
    )
    assert_frame_equal(
        df.lazy().select(
            range_change(pl.col(x.name), percentage=True)
        ).collect(),
        pl.DataFrame({x.name:[range_chg_pct]})
    )
    assert_frame_equal(
        df.lazy().select(
            range_over_mean(pl.col(x.name))
        ).collect(),
        pl.DataFrame({x.name:[res]})
    )


