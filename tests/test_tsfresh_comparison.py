from functools import partial
from itertools import product

import numpy as np
import polars as pl
import pytest
from tsfresh.feature_extraction import feature_calculators as tsfresh

from functime.feature_extraction.tsfresh import (  # fft_coefficients,
    absolute_energy,
    absolute_maximum,
    absolute_sum_of_changes,
    approximate_entropies,
    augmented_dickey_fuller,
    autocorrelation,
    autoregressive_coefficients,
    benford_correlation,
    binned_entropy,
    c3,
    change_quantiles,
    cid_ce,
    count_above,
    count_above_mean,
    count_below,
    count_below_mean,
    cwt_coefficients,
    energy_ratios,
    first_location_of_maximum,
    first_location_of_minimum,
    fourier_entropy,
    friedrich_coefficients,
    has_duplicate,
    has_duplicate_max,
    has_duplicate_min,
    index_mass_quantile,
    large_standard_deviation,
    last_location_of_maximum,
    last_location_of_minimum,
    lempel_ziv_complexity,
    linear_trend,
    longest_strike_above_mean,
    longest_strike_below_mean,
    mean_abs_change,
    mean_change,
    mean_n_absolute_max,
    mean_second_derivative_central,
    number_crossings,
    number_cwt_peaks,
    number_peaks,
    partial_autocorrelation,
    percent_recoccuring_values,
    percent_reocurring_points,
    permutation_entropy,
    range_count,
    ratio_beyond_r_sigma,
    ratio_n_unique_to_length,
    root_mean_square,
    sample_entropy,
    spkt_welch_density,
    sum_reocurring_points,
    sum_reocurring_values,
    symmetry_looking,
    time_reversal_asymmetry_statistic,
)


@pytest.fixture
def ts(pl_y):
    pl_y = pl_y.collect()
    return pl_y.get_column(pl_y.columns[-1]).rename("y")


@pytest.fixture
def ts_lazy(ts):
    return lambda func: ts.to_frame().lazy().select(func(pl.col("y"))).collect().item()


@pytest.fixture
def pd_ts(ts):
    return ts.to_pandas()


def test_absolute_energy(ts, ts_lazy, pd_ts):
    expected = tsfresh.abs_energy(pd_ts)
    assert absolute_energy(ts) == expected
    assert ts_lazy(absolute_energy) == expected


def test_absolute_maximum(ts, ts_lazy, pd_ts):
    expected = tsfresh.absolute_maximum(pd_ts)
    assert absolute_maximum(ts) == expected
    assert ts_lazy(absolute_energy) == expected


def test_absolute_sum_of_changes(ts, ts_lazy, pd_ts):
    expected = tsfresh.absolute_sum_of_changes(pd_ts)
    assert absolute_sum_of_changes(ts) == expected
    assert ts_lazy(absolute_sum_of_changes) == expected


def test_approximate_entropies(ts, ts_lazy, pd_ts):
    run_length = 2
    filtering_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    result = approximate_entropies(
        x=ts, filtering_levels=filtering_levels, run_length=run_length
    )
    result_lazy = ts_lazy(
        partial(
            approximate_entropies,
            filtering_levels=filtering_levels,
            run_length=run_length,
        )
    )
    expected = [tsfresh.approximate_entropy(pd_ts, r=r, m=2) for r in filtering_levels]
    assert result == expected
    assert result_lazy == expected


def test_augmented_dickey_fuller(ts, ts_lazy, pd_ts):
    expected = tsfresh.augmented_dickey_fuller(pd_ts, param={"attr": "teststat"})
    assert augmented_dickey_fuller(ts) == expected
    assert ts_lazy(augmented_dickey_fuller) == expected


def test_autocorrelation(ts, ts_lazy, pd_ts):
    n_lags = 12
    expected = tsfresh.autocorrelation(pd_ts, lag=n_lags)
    assert autocorrelation(ts, n_lags=n_lags) == expected
    assert ts_lazy(partial(autocorrelation, n_lags=n_lags)) == expected


def test_autoregressive_coefficients(ts, ts_lazy, pd_ts):
    n_lags = 12
    expected = [
        tsfresh.ar_coefficient(pd_ts, param={"coeff": i, "k": n_lags})
        for i in range(n_lags)
    ]
    assert autoregressive_coefficients(ts, n_lags=n_lags) == expected
    assert ts_lazy(partial(autoregressive_coefficients, n_lags=n_lags)) == expected


def test_benfold_correlation(ts, ts_lazy, pd_ts):
    expected = tsfresh.benford_correlation(pd_ts)
    assert benford_correlation(ts) == expected
    assert ts_lazy(benford_correlation) == expected


def test_binned_entropy(ts, ts_lazy, pd_ts):
    expected = tsfresh.binned_entropy(pd_ts, max_bins=10)
    assert binned_entropy(ts, bin_count=10) == expected
    assert ts_lazy(partial(binned_entropy, bin_count=10)) == expected


def test_c3(ts, ts_lazy, pd_ts):
    n_lags = 12
    expected = tsfresh.c3(pd_ts, lag=n_lags)
    assert c3(ts, n_lags=n_lags) == expected
    assert ts_lazy(partial(c3, n_lags=n_lags)) == expected


@pytest.mark.parametrize(
    "ql,qh,is_abs", [(0.1, 0.9, False), (0.2, 0.8, True), (0.3, 0.7, True)]
)
def test_change_quantiles(ql, qh, is_abs, ts, ts_lazy, pd_ts):
    expected = tsfresh.change_quantiles(
        pd_ts, ql=ql, qh=qh, is_abs=is_abs, f_agg="mean"
    )
    assert np.mean(change_quantiles(ts, ql=ql, qh=qh, is_abs=is_abs)) == expected
    assert np.mean(ts_lazy(partial(c3, ql=ql, qh=qh, is_abs=is_abs))) == expected


@pytest.mark.parametrize("normalize", [False, True])
def test_cid_ce(normalize, ts, ts_lazy, pd_ts):
    expected = tsfresh.cid_ce(pd_ts, normalize=normalize)
    assert cid_ce(ts, normalize=normalize) == expected
    assert ts_lazy(cid_ce) == expected


def test_count_above(ts, ts_lazy, pd_ts):
    threshold = 0
    expected = tsfresh.count_above(pd_ts, t=threshold)
    assert count_above(ts, threshold=threshold) == expected
    assert ts_lazy(partial(count_below, threshold=threshold)) == expected


def test_count_above_mean(ts, ts_lazy, pd_ts):
    expected = tsfresh.count_above_mean(pd_ts)
    assert count_above_mean(ts) == expected
    assert ts_lazy(count_above_mean) == expected


def test_count_below(ts, ts_lazy, pd_ts):
    threshold = 0
    expected = tsfresh.count_below(pd_ts, t=threshold)
    assert count_below(ts, threshold=threshold) == expected
    assert ts_lazy(partial(count_below, threshold=threshold)) == expected


def test_count_below_mean(ts, ts_lazy, pd_ts):
    expected = tsfresh.count_below_mean(pd_ts)
    assert count_below_mean(ts) == expected
    assert ts_lazy(count_below_mean) == expected


def test_cwt_coefficients(ts, ts_lazy, pd_ts):
    widths = [2, 5, 10, 20]
    n_coefficients = 3
    expected = [
        tsfresh.cwt_coefficients(pd_ts, param={"widths": widths, "coeff": i, "w": w})
        for i, w in product(widths, range(n_coefficients))
    ]
    assert (
        cwt_coefficients(ts, widths=widths, n_coefficients=n_coefficients) == expected
    )
    assert (
        ts_lazy(partial(cwt_coefficients, widths=widths, n_coefficients=n_coefficients))
        == expected
    )


def test_energy_ratios(ts, ts_lazy, pd_ts):
    n_chunks = 10
    expected = [
        tsfresh.energy_ratio_by_chunks(
            pd_ts, param={"num_segments": n_chunks, "segment_focus": i}
        )
        for i in range(n_chunks)
    ]
    assert energy_ratios(ts, n_chunks=10) == expected
    assert ts_lazy(energy_ratios(n_chunks=10)) == expected


def test_first_location_of_maximum(ts, ts_lazy, pd_ts):
    expected = tsfresh.first_location_of_maximum(pd_ts)
    assert first_location_of_maximum(ts) == expected
    assert ts_lazy(first_location_of_maximum) == expected


def test_first_location_of_minimum(ts, ts_lazy, pd_ts):
    expected = tsfresh.first_location_of_minimum(pd_ts)
    assert first_location_of_minimum(ts) == expected
    assert ts_lazy(first_location_of_minimum) == expected


def test_fourier_entropy(ts, ts_lazy, pd_ts):
    expected = tsfresh.fourier_entropy(pd_ts, bins=10)
    assert fourier_entropy(ts) == expected
    assert ts_lazy(partial(fourier_entropy, n_bins=10)) == expected


def test_friedrich_coefficients(ts, ts_lazy, pd_ts):
    expected = tsfresh.friedrich_coefficients(
        x=pd_ts, param={"m": 3, "n_quantiles": 30}
    )
    assert friedrich_coefficients(ts) == expected
    assert ts_lazy(friedrich_coefficients) == expected


def test_has_duplicate(ts, ts_lazy, pd_ts):
    expected = tsfresh.has_duplicate(pd_ts)
    assert has_duplicate(ts) == expected
    assert ts_lazy(has_duplicate) == expected


def test_has_duplicate_max(ts, ts_lazy, pd_ts):
    expected = tsfresh.has_duplicate_max(pd_ts)
    assert has_duplicate_max(ts) == expected
    assert ts_lazy(has_duplicate_max) == expected


def test_has_duplicate_min(ts, ts_lazy, pd_ts):
    expected = tsfresh.has_duplicate_min(pd_ts)
    assert has_duplicate_min(ts) == expected
    assert ts_lazy(has_duplicate_min) == expected


def test_index_mass_quantile(ts, ts_lazy, pd_ts):
    q = 0.5
    expected = tsfresh.index_mass_quantile(pd_ts, param={"q": q})
    assert index_mass_quantile(ts, q=q) == expected
    assert ts_lazy(partial(index_mass_quantile, q=q)) == expected


def test_large_standard_deviation(ts, ts_lazy, pd_ts):
    ratio = 0.25
    expected = tsfresh.large_standard_deviation(pd_ts, r=ratio)
    assert large_standard_deviation(ts) == expected
    assert ts_lazy(large_standard_deviation) == expected


def test_last_location_of_maximum(ts, ts_lazy, pd_ts):
    expected = tsfresh.last_location_of_maximum(pd_ts)
    assert last_location_of_maximum(ts) == expected
    assert ts_lazy(last_location_of_maximum) == expected


def test_last_location_of_minimum(ts, ts_lazy, pd_ts):
    expected = tsfresh.last_location_of_minimum(pd_ts)
    assert last_location_of_minimum(ts) == expected
    assert ts_lazy(last_location_of_minimum) == expected


def test_lempel_ziv_complexity(ts, ts_lazy, pd_ts):
    expected = tsfresh.lempel_ziv_complexity(pd_ts, bins=10)
    assert lempel_ziv_complexity(ts, n_bins=10) == expected
    assert ts_lazy(lempel_ziv_complexity, n_bins=10) == expected


def test_linear_trend(ts, ts_lazy, pd_ts):
    expected_results = tsfresh.linear_trend(
        pd_ts, param=[{"attr": "slope"}, {"attr": "intercept"}, {"attr": "rss"}]
    )
    expected = {
        "slope": expected_results[0][1],
        "intercept": expected_results[1][1],
        "rss": expected_results[2][1],
    }
    assert linear_trend(ts) == expected
    assert ts_lazy(linear_trend) == expected


def test_longest_strike_above_mean(ts, ts_lazy, pd_ts):
    expected = tsfresh.longest_strike_above_mean(pd_ts)
    assert longest_strike_above_mean(ts) == expected
    assert ts_lazy(longest_strike_above_mean) == expected


def test_longest_strike_below_mean(ts, ts_lazy, pd_ts):
    expected = tsfresh.longest_strike_below_mean(pd_ts)
    assert longest_strike_below_mean(ts) == expected
    assert ts_lazy(longest_strike_below_mean) == expected


def test_mean_abs_change(ts, ts_lazy, pd_ts):
    expected = tsfresh.mean_abs_change(pd_ts)
    assert mean_abs_change(ts) == expected
    assert ts_lazy(mean_abs_change) == expected


def test_mean_change(ts, ts_lazy, pd_ts):
    expected = tsfresh.mean_change(pd_ts)
    assert mean_change(ts) == expected
    assert ts_lazy(mean_change) == expected


def test_mean_n_absolute_max(ts, ts_lazy, pd_ts):
    expected = tsfresh.mean_n_absolute_max(pd_ts, number_of_maxima=3)
    assert mean_n_absolute_max(ts, n_maxima=3) == expected
    assert ts_lazy(partial(mean_n_absolute_max, n_maxima=3)) == expected


def test_mean_second_derivative_central(ts, ts_lazy, pd_ts):
    expected = tsfresh.mean_second_derivative_central(pd_ts)
    assert mean_second_derivative_central(ts) == expected
    assert ts_lazy(mean_second_derivative_central) == expected


def test_number_crossings(ts, ts_lazy, pd_ts):
    expected = tsfresh.number_crossing_m(pd_ts, m=0.0)
    assert number_crossings(ts, crossing_value=0.0) == expected
    assert ts_lazy(partial(number_crossings, crossing_value=0.0)) == expected


def test_number_cwt_peaks(ts, ts_lazy, pd_ts):
    expected = tsfresh.number_cwt_peaks(pd_ts, m=5)
    assert number_cwt_peaks(ts, max_width=5) == expected
    assert ts_lazy(partial(number_cwt_peaks, max_width=5)) == expected


def test_number_peaks(ts, ts_lazy, pd_ts):
    expected = tsfresh.number_peaks(pd_ts, n=5)
    assert number_peaks(ts, number_peaks=5) == expected
    assert ts_lazy(partial(number_peaks, number_peaks=5)) == expected


def test_partial_autocorrelation(ts, ts_lazy, pd_ts):
    expected = tsfresh.partial_autocorrelation(pd_ts, param={"lag": 6})
    assert partial_autocorrelation(ts, n_lags=6) == expected
    assert ts_lazy(partial(partial_autocorrelation, n_lags=6)) == expected


def test_percent_recoccuring_values(ts, ts_lazy, pd_ts):
    expected = tsfresh.percentage_of_reoccurring_datapoints_to_all_datapoints(pd_ts)
    assert percent_reocurring_points(ts) == expected
    assert ts_lazy(percent_reocurring_points) == expected


def test_percent_reocurring_points(ts, ts_lazy, pd_ts):
    expected = tsfresh.percentage_of_reoccurring_values_to_all_values(pd_ts)
    assert percent_recoccuring_values(ts) == expected
    assert ts_lazy(percent_recoccuring_values) == expected


def test_permutation_entropy(ts, ts_lazy, pd_ts):
    expected = tsfresh.permutation_entropy(pd_ts, tau=1, dimension=3)
    assert permutation_entropy(ts, tau=1, n_dims=3) == expected
    assert ts_lazy(partial(permutation_entropy, tau=1, n_dims=3)) == expected


def test_range_count(ts, ts_lazy, pd_ts):
    expected = tsfresh.range_count(pd_ts, min=0, max=100)
    assert range_count(ts, lower=0, upper=100) == expected
    assert ts_lazy(partial(permutation_entropy, lower=0, upper=100)) == expected


def test_ratio_beyond_r_sigma(ts, ts_lazy, pd_ts):
    expected = tsfresh.ratio_beyond_r_sigma(pd_ts, r=0.25)
    assert ratio_beyond_r_sigma(ts, ratio=0.25) == expected
    assert ts_lazy(partial(ratio_beyond_r_sigma, ratio=0.25)) == expected


def test_ratio_n_unique_to_length(ts, ts_lazy, pd_ts):
    expected = tsfresh.ratio_value_number_to_time_series_length(pd_ts)
    assert ratio_n_unique_to_length(ts) == expected
    assert ts_lazy(ratio_n_unique_to_length) == expected


def test_root_mean_square(ts, ts_lazy, pd_ts):
    expected = tsfresh.root_mean_square(pd_ts)
    assert root_mean_square(ts) == expected
    assert ts_lazy(root_mean_square) == expected


def test_sample_entropy(ts, ts_lazy, pd_ts):
    expected = tsfresh.sample_entropy(pd_ts)
    assert sample_entropy(ts) == expected
    assert ts_lazy(sample_entropy) == expected


def test_spkt_welch_density(ts, ts_lazy, pd_ts):
    n_coefficients = 12
    expected = [
        tsfresh.spkt_welch_density(pd_ts, param={"coeff": i})
        for i in range(n_coefficients)
    ]
    assert spkt_welch_density(ts, n_cofficients=12) == expected
    assert ts_lazy(partial(spkt_welch_density, n_cofficients=12)) == expected


def test_sum_reocurring_points(ts, ts_lazy, pd_ts):
    expected = tsfresh.sum_reocurring_points(pd_ts)
    assert sum_reocurring_points(ts) == expected
    assert ts_lazy(sum_reocurring_points) == expected


def test_sum_reocurring_values(ts, ts_lazy, pd_ts):
    expected = tsfresh.sum_reocurring_values(pd_ts)
    assert sum_reocurring_values(ts) == expected
    assert ts_lazy(sum_reocurring_values) == expected


def test_symmetry_looking(ts, ts_lazy, pd_ts):
    ratio = 0.25
    expected = tsfresh.symmetry_looking(x=pd_ts, param={"r": ratio})
    assert symmetry_looking(ts, ratio=ratio) == expected
    assert ts_lazy(partial(symmetry_looking, ratio=ratio)) == expected


def test_time_reversal_asymmetry_statistic(ts, ts_lazy, pd_ts):
    expected = tsfresh.time_reversal_asymmetry_statistic(x=pd_ts, lag=6)
    assert time_reversal_asymmetry_statistic(ts, n_lags=6) == expected
    assert ts_lazy(partial(time_reversal_asymmetry_statistic, n_lags=6)) == expected


def test_variation_coefficient():
    expected = tsfresh.number_peaks(pd_ts)
    assert number_peaks(ts) == expected
    assert ts_lazy(number_peaks) == expected
