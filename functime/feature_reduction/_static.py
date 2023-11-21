import polars as pl
import functime.feature_extractors as fe

DEFAULT = [
    []
]


ALL_SMALL = [
    ["absolute_energy", {}],
    ["absolute_maximum", {}],
    ["absolute_sum_of_changes", {}],
    ["lempel_ziv_complexity", {"threshold": (pl.col("value").max() - pl.col("value").min()) / 2}],
    # ["augmented_dickey_fuller", "param"]
    ["autocorrelation", {"n_lags": 4}],
    # ["autoregressive_coefficients", {"n_lags": 4}],
    ["benford_correlation", {}],
    ["binned_entropy", {"bin_count": 10}],
    ["c3", {"n_lags": 10}],
    ["change_quantiles", {"q_low": 0.1, "q_high": 0.9, "is_abs": True}],
    ["cid_ce", {"normalize": True}],
    ["count_above", {"threshold": 0.0}],
    ["count_above_mean", {}],
    ["count_below", {"threshold": 0.0}],
    ["count_below_mean", {}],
    # ["cwt_coefficients", {"widths": (1, 2, 3), "n_coefficients": 2}]
    ["energy_ratios", {"n_chunks": 6}],
    ["first_location_of_maximum", {}],
    ["first_location_of_minimum", {}],
    # ["fourier_entropy", {"n_bins": 10}],
    # ["friedrich_coefficients", {"polynomial_order": 3, "n_quantiles": 30, "params": [{"m": 3, "r": 30}]}]
    ["has_duplicate", {}],
    ["has_duplicate_max", {}],
    ["has_duplicate_min", {}],
    ["index_mass_quantile", {"q": 0.5}],
    ["large_standard_deviation", {"ratio": 0.25}],
    ["last_location_of_maximum", {}],
    ["last_location_of_minimum", {}],
    # ["lempel_ziv_complexity", {"n_bins": 5}]
    # ["linear_trend", {}],
    ["longest_streak_above_mean", {}],
    ["longest_streak_below_mean", {}],
    ["mean_abs_change", {}],
    ["mean_change", {}],
    ["mean_n_absolute_max", {"n_maxima": 20}],
    ["mean_second_derivative_central", {}],
    ["number_crossings", {"crossing_value": 0.0}],
    ["number_peaks", {"support": 5, "n": 5}],
    # ["partial_autocorrelation", "param"]
    ["percent_reoccurring_values", {}],
    ["percent_reoccurring_points", {}],
    ["permutation_entropy", {"tau": 1, "n_dims": 3}],
    ["range_count", {"lower": 0, "upper": 9, "closed": "none"}],
    ["ratio_beyond_r_sigma", {"ratio": 2}],
    ["ratio_n_unique_to_length", {}],
    ["root_mean_square", {}],
    ["sum_reoccurring_points", {}],
    ["sum_reoccurring_values", {}],
    ["symmetry_looking", {"ratio": 0.25}],
    ["time_reversal_asymmetry_statistic", {"n_lags": 3}],
    ["variation_coefficient", {}],
    ["var_gt_std", {}]
]

ALL_MEDIUM = [
    ["number_peaks", {"support": 2}],
    ["mean_n_absolute_max", {"n_maxima": 10}],
    ["root_mean_square", {}],
    ["count_above_mean", {}],
    ["first_location_of_minimum", {}],
    ["first_location_of_maximum", {}]
]
