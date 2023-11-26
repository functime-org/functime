import polars as pl
from functime.feature_extractors import FeatureExtractor

def get_small(col_values: str):
    features_small = [
        pl.col(col_values).max().alias("max"),
        pl.col(col_values).min().alias("min"),
        pl.col(col_values).mean().alias("mean"),
        pl.col(col_values).quantile(0.25).alias("quantile_0.25"),
        pl.col(col_values).quantile(0.75).alias("quantile_0.75"),
        pl.col(col_values).quantile(0.1).alias("quantile_0.1"),
        pl.col(col_values).quantile(0.9).alias("quantile_0.9"),
        pl.col(col_values).median().alias("median"),
        pl.col(col_values).std().alias("std"),
        pl.col(col_values).ts.absolute_energy().alias("absolute_energy"),
        pl.col(col_values).ts.absolute_maximum().alias("absolute_maximum"),
        pl.col(col_values).ts.absolute_sum_of_changes().alias("absolute_sum_of_changes"),
        pl.col(col_values).ts.lempel_ziv_complexity(threshold=(pl.col(col_values).max() - pl.col(col_values).min()) / 2).alias("lempel_ziv_complexity_threshold_max-min_div_2"),
        # pl.col(col_values).ts.augmented_dickey_fuller("param").alias("augmented_dickey_fuller_param"),
        pl.col(col_values).ts.autocorrelation(n_lags=4).alias("autocorrelation_n_lags_4"),
        # pl.col(col_values).ts.autoregressive_coefficients(n_lags=4).alias("autoregressive_coefficients_n_lags_4"),
        pl.col(col_values).ts.benford_correlation().alias("benford_correlation"),
        pl.col(col_values).ts.binned_entropy(bin_count=10).alias("binned_entropy_bin_count_10"),
        pl.col(col_values).ts.c3(n_lags=10).alias("c3_n_lags_10"),
        # (
        #     pl.col(col_values).ts.change_quantiles(q_low=0.1, q_high=0.9, is_abs=True)
        #     .explode()
        #     .list
        #     .mean()
        # ),
        pl.col(col_values).ts.cid_ce(normalize=True).alias("cid_ce_normalize_True"),
        pl.col(col_values).ts.count_above(threshold=0.0).alias("count_above_threshold_0.0"),
        pl.col(col_values).ts.count_above_mean().alias("count_above_mean"),
        pl.col(col_values).ts.count_below(threshold=0.0).alias("count_below_threshold_0.0"),
        pl.col(col_values).ts.count_below_mean().alias("count_below_mean"),
        # pl.col(col_values).ts.cwt_coefficients(widths=(1, 2, 3), n_coefficients=2).alias("cwt_coefficients_widths_1_2_3_n_coefficients_2"),
        # (
        #     pl.col(col_values).ts.energy_ratios(n_chunks=6)
        #     .explode()
        #     .list
        #     .to_struct(
        #         fields=[f"energy_ratios_n_chunks_6_{i}" for i in range(6)]
        #     )
        #     .alias("energy_ratios_n_chunks_6")
        # ),
        pl.col(col_values).ts.first_location_of_maximum().alias("first_location_of_maximum"),
        pl.col(col_values).ts.first_location_of_minimum().alias("first_location_of_minimum"),
        # pl.col(col_values).ts.fourier_entropy(n_bins=10).alias("fourier_entropy_n_bins_10"),
        # pl.col(col_values).ts.friedrich_coefficients(polynomial_order=3, n_quantiles=30, params=[{"m": 3, "r": 30}]).alias("friedrich_coefficients_polynomial_order_3_n_quantiles_30_params_m_3_r_30"),
        pl.col(col_values).ts.has_duplicate().alias("has_duplicate"),
        pl.col(col_values).ts.has_duplicate_max().alias("has_duplicate_max"),
        pl.col(col_values).ts.has_duplicate_min().alias("has_duplicate_min"),
        pl.col(col_values).ts.index_mass_quantile(q=0.5).alias("index_mass_quantile_q_0.5"),
        pl.col(col_values).ts.large_standard_deviation(ratio=0.25).alias("large_standard_deviation_ratio_0.25"),
        pl.col(col_values).ts.last_location_of_maximum().alias("last_location_of_maximum"),
        pl.col(col_values).ts.last_location_of_minimum().alias("last_location_of_minimum"),
        # pl.col(col_values).ts.lempel_ziv_complexity(n_bins=5).alias("lempel_ziv_complexity_n_bins_5"),
        pl.col(col_values).ts.linear_trend().struct.field("slope").alias("linear_trend_slope"),
        pl.col(col_values).ts.linear_trend().struct.field("intercept").alias("linear_trend_intercept"),
        pl.col(col_values).ts.linear_trend().struct.field("rss").alias("linear_trend_rss"),
        pl.col(col_values).ts.longest_streak_above_mean().alias("longest_streak_above_mean"),
        pl.col(col_values).ts.longest_streak_below_mean().alias("longest_streak_below_mean"),
        pl.col(col_values).ts.mean_abs_change().alias("mean_abs_change"),
        pl.col(col_values).ts.mean_change().alias("mean_change"),
        pl.col(col_values).ts.mean_n_absolute_max(n_maxima=20).alias("mean_n_absolute_max_n_maxima_20"),
        pl.col(col_values).ts.mean_second_derivative_central().alias("mean_second_derivative_central"),
        pl.col(col_values).ts.number_crossings(crossing_value=0.0).alias("number_crossings_crossing_value_0.0"),
        pl.col(col_values).ts.number_peaks(support=5).alias("number_peaks_support_5"),
        # pl.col(col_values).ts.partial_autocorrelation("param").alias("partial_autocorrelation_param"),
        pl.col(col_values).ts.percent_reoccurring_values().alias("percent_reoccurring_values"),
        pl.col(col_values).ts.percent_reoccurring_points().alias("percent_reoccurring_points"),
        pl.col(col_values).ts.permutation_entropy(tau=1, n_dims=3).alias("permutation_entropy_tau_1_n_dims_3"),
        pl.col(col_values).ts.range_count(lower=0, upper=9, closed="none").alias("range_count_lower_0_upper_9_closed_none"),
        pl.col(col_values).ts.ratio_beyond_r_sigma(ratio=2).alias("ratio_beyond_r_sigma_ratio_2"),
        pl.col(col_values).ts.ratio_n_unique_to_length().alias("ratio_n_unique_to_length"),
        pl.col(col_values).ts.root_mean_square().alias("root_mean_square"),
        pl.col(col_values).ts.sum_reoccurring_points().alias("sum_reoccurring_points"),
        pl.col(col_values).ts.sum_reoccurring_values().alias("sum_reoccurring_values"),
        pl.col(col_values).ts.symmetry_looking(ratio=0.25).alias("symmetry_looking_ratio_0.25"),
        pl.col(col_values).ts.time_reversal_asymmetry_statistic(n_lags=3).alias("time_reversal_asymmetry_statistic_n_lags_3"),
        pl.col(col_values).ts.variation_coefficient().alias("variation_coefficient"),
        pl.col(col_values).ts.var_gt_std().alias("var_gt_std")
    ]
    return features_small

def get_medium(col_values: str):
    features_medium = [
        pl.col(col_values).max().alias("max"),
        pl.col(col_values).min().alias("min"),
        pl.col(col_values).mean().alias("mean"),
        pl.col(col_values).quantile(0.25).alias("quantile_0.25"),
        pl.col(col_values).quantile(0.75).alias("quantile_0.75"),
        pl.col(col_values).quantile(0.1).alias("quantile_0.1"),
        pl.col(col_values).quantile(0.9).alias("quantile_0.9"),
        pl.col(col_values).median().alias("median"),
        pl.col(col_values).std().alias("std"),
        pl.col(col_values).ts.absolute_energy().alias("absolute_energy"),
        pl.col(col_values).ts.absolute_maximum().alias("absolute_maximum"),
        pl.col(col_values).ts.absolute_sum_of_changes().alias("absolute_sum_of_changes"),
        pl.col(col_values).ts.lempel_ziv_complexity(threshold=(pl.col(col_values).max() - pl.col(col_values).min()) / 2).alias("lempel_ziv_complexity_threshold_max-min_div_2"),
        pl.col(col_values).ts.lempel_ziv_complexity(threshold=pl.col(col_values).median()).alias("lempel_ziv_complexity_threshold_median"),
        pl.col(col_values).ts.lempel_ziv_complexity(threshold=pl.col(col_values).mean()).alias("lempel_ziv_complexity_threshold_mean"),
        pl.col(col_values).ts.lempel_ziv_complexity(threshold=pl.col(col_values).quantile(0.25)).alias("lempel_ziv_complexity_threshold_quantile_0.25"),
        pl.col(col_values).ts.lempel_ziv_complexity(threshold=pl.col(col_values).quantile(0.75)).alias("lempel_ziv_complexity_threshold_quantile_0.75"),
        # pl.col(col_values).ts.augmented_dickey_fuller("param").alias("augmented_dickey_fuller_param"),
        pl.col(col_values).ts.autocorrelation(n_lags=1).alias("autocorrelation_n_lags_1"),
        pl.col(col_values).ts.autocorrelation(n_lags=2).alias("autocorrelation_n_lags_2"),
        pl.col(col_values).ts.autocorrelation(n_lags=3).alias("autocorrelation_n_lags_3"),
        pl.col(col_values).ts.autocorrelation(n_lags=4).alias("autocorrelation_n_lags_4"),
        pl.col(col_values).ts.autocorrelation(n_lags=5).alias("autocorrelation_n_lags_5"),
        pl.col(col_values).ts.autocorrelation(n_lags=6).alias("autocorrelation_n_lags_6"),
        pl.col(col_values).ts.autocorrelation(n_lags=7).alias("autocorrelation_n_lags_7"),
        pl.col(col_values).ts.autocorrelation(n_lags=8).alias("autocorrelation_n_lags_8"),
        pl.col(col_values).ts.autocorrelation(n_lags=9).alias("autocorrelation_n_lags_9"),
        pl.col(col_values).ts.autocorrelation(n_lags=10).alias("autocorrelation_n_lags_10"),
        pl.col(col_values).ts.autocorrelation(n_lags=11).alias("autocorrelation_n_lags_11"),
        pl.col(col_values).ts.autocorrelation(n_lags=12).alias("autocorrelation_n_lags_12"),
        pl.col(col_values).ts.autocorrelation(n_lags=13).alias("autocorrelation_n_lags_13"),
        pl.col(col_values).ts.autocorrelation(n_lags=14).alias("autocorrelation_n_lags_14"),
        pl.col(col_values).ts.autocorrelation(n_lags=15).alias("autocorrelation_n_lags_15"),
        # pl.col(col_values).ts.autoregressive_coefficients(n_lags=4).alias("autoregressive_coefficients_n_lags_4"),
        pl.col(col_values).ts.benford_correlation().alias("benford_correlation"),
        pl.col(col_values).ts.binned_entropy(bin_count=10).alias("binned_entropy_bin_count_10"),
        pl.col(col_values).ts.binned_entropy(bin_count=15).alias("binned_entropy_bin_count_15"),
        pl.col(col_values).ts.binned_entropy(bin_count=20).alias("binned_entropy_bin_count_20"),
        pl.col(col_values).ts.binned_entropy(bin_count=25).alias("binned_entropy_bin_count_25"),
        pl.col(col_values).ts.binned_entropy(bin_count=30).alias("binned_entropy_bin_count_30"),
        # pl.col(col_values).ts.binned_entropy(
        #     bin_count= (
        #             2*(pl.col(col_values).quantile(0.75)-pl.col(col_values).quantile(0.25))/pl.col(col_values).len()
        #         )
        #     ).alias("binned_entropy_bin_count_Freedman_Diaconis_rule"),
        pl.col(col_values).ts.c3(n_lags=5).alias("c3_n_lags_5"),
        pl.col(col_values).ts.c3(n_lags=10).alias("c3_n_lags_10"),
        # (
        #     pl.col(col_values).ts.change_quantiles(q_low=0.1, q_high=0.9, is_abs=True)
        #     .explode()
        #     .list
        #     .mean()
        # ),
        pl.col(col_values).ts.cid_ce(normalize=True).alias("cid_ce_normalize_True"),
        pl.col(col_values).ts.count_above(threshold=0.0).alias("count_above_threshold_0.0"),
        pl.col(col_values).ts.count_above_mean().alias("count_above_mean"),
        pl.col(col_values).ts.count_below(threshold=0.0).alias("count_below_threshold_0.0"),
        pl.col(col_values).ts.count_below_mean().alias("count_below_mean"),
        # pl.col(col_values).ts.cwt_coefficients(widths=(1, 2, 3), n_coefficients=2).alias("cwt_coefficients_widths_1_2_3_n_coefficients_2"),
        # (
        #     pl.col(col_values).ts.energy_ratios(n_chunks=6)
        #     .explode()
        #     .list
        #     .to_struct(
        #         fields=[f"energy_ratios_n_chunks_6_{i}" for i in range(6)]
        #     )
        #     .alias("energy_ratios_n_chunks_6")
        # ),
        pl.col(col_values).ts.first_location_of_maximum().alias("first_location_of_maximum"),
        pl.col(col_values).ts.first_location_of_minimum().alias("first_location_of_minimum"),
        # pl.col(col_values).ts.fourier_entropy(n_bins=10).alias("fourier_entropy_n_bins_10"),
        # pl.col(col_values).ts.friedrich_coefficients(polynomial_order=3, n_quantiles=30, params=[{"m": 3, "r": 30}]).alias("friedrich_coefficients_polynomial_order_3_n_quantiles_30_params_m_3_r_30"),
        pl.col(col_values).ts.has_duplicate().alias("has_duplicate"),
        pl.col(col_values).ts.has_duplicate_max().alias("has_duplicate_max"),
        pl.col(col_values).ts.has_duplicate_min().alias("has_duplicate_min"),
        pl.col(col_values).ts.index_mass_quantile(q=0.5).alias("index_mass_quantile_q_0.5"),
        pl.col(col_values).ts.large_standard_deviation(ratio=0.25).alias("large_standard_deviation_ratio_0.25"),
        pl.col(col_values).ts.last_location_of_maximum().alias("last_location_of_maximum"),
        pl.col(col_values).ts.last_location_of_minimum().alias("last_location_of_minimum"),
        # pl.col(col_values).ts.lempel_ziv_complexity(n_bins=5).alias("lempel_ziv_complexity_n_bins_5"),
        pl.col(col_values).ts.linear_trend().struct.field("slope").alias("linear_trend_slope"),
        pl.col(col_values).ts.linear_trend().struct.field("intercept").alias("linear_trend_intercept"),
        pl.col(col_values).ts.linear_trend().struct.field("rss").alias("linear_trend_rss"),
        pl.col(col_values).ts.longest_streak_above_mean().alias("longest_streak_above_mean"),
        pl.col(col_values).ts.longest_streak_below_mean().alias("longest_streak_below_mean"),
        pl.col(col_values).ts.mean_abs_change().alias("mean_abs_change"),
        pl.col(col_values).ts.mean_change().alias("mean_change"),
        pl.col(col_values).ts.mean_n_absolute_max(n_maxima=5).alias("mean_n_absolute_max_n_maxima_5"),
        pl.col(col_values).ts.mean_n_absolute_max(n_maxima=10).alias("mean_n_absolute_max_n_maxima_10"),
        pl.col(col_values).ts.mean_n_absolute_max(n_maxima=15).alias("mean_n_absolute_max_n_maxima_15"),
        pl.col(col_values).ts.mean_n_absolute_max(n_maxima=20).alias("mean_n_absolute_max_n_maxima_20"),
        pl.col(col_values).ts.mean_second_derivative_central().alias("mean_second_derivative_central"),
        pl.col(col_values).ts.number_crossings(crossing_value=0.0).alias("number_crossings_crossing_value_0.0"),
        pl.col(col_values).ts.number_peaks(support=2).alias("number_peaks_support_2"),
        pl.col(col_values).ts.number_peaks(support=4).alias("number_peaks_support_4"),
        pl.col(col_values).ts.number_peaks(support=6).alias("number_peaks_support_6"),
        pl.col(col_values).ts.number_peaks(support=8).alias("number_peaks_support_8"),
        pl.col(col_values).ts.number_peaks(support=10).alias("number_peaks_support_10"),
        # pl.col(col_values).ts.partial_autocorrelation("param").alias("partial_autocorrelation_param"),
        pl.col(col_values).ts.percent_reoccurring_values().alias("percent_reoccurring_values"),
        pl.col(col_values).ts.percent_reoccurring_points().alias("percent_reoccurring_points"),
        pl.col(col_values).ts.permutation_entropy(tau=1, n_dims=3).alias("permutation_entropy_tau_1_n_dims_3"),
        pl.col(col_values).ts.permutation_entropy(tau=1, n_dims=4).alias("permutation_entropy_tau_1_n_dims_4"),
        pl.col(col_values).ts.permutation_entropy(tau=1, n_dims=5).alias("permutation_entropy_tau_1_n_dims_5"),
        pl.col(col_values).ts.permutation_entropy(tau=1, n_dims=6).alias("permutation_entropy_tau_1_n_dims_6"),
        pl.col(col_values).ts.range_count(lower=0.1, upper=0.9, closed="none").alias("range_count_lower_q_0.1_upper_q_0.9_closed_none"),
        pl.col(col_values).ts.ratio_beyond_r_sigma(ratio=1).alias("ratio_beyond_r_sigma_ratio_1"),
        pl.col(col_values).ts.ratio_beyond_r_sigma(ratio=2).alias("ratio_beyond_r_sigma_ratio_2"),
        pl.col(col_values).ts.ratio_beyond_r_sigma(ratio=3).alias("ratio_beyond_r_sigma_ratio_3"),
        pl.col(col_values).ts.ratio_beyond_r_sigma(ratio=4).alias("ratio_beyond_r_sigma_ratio_4"),
        pl.col(col_values).ts.ratio_n_unique_to_length().alias("ratio_n_unique_to_length"),
        pl.col(col_values).ts.root_mean_square().alias("root_mean_square"),
        pl.col(col_values).ts.sum_reoccurring_points().alias("sum_reoccurring_points"),
        pl.col(col_values).ts.sum_reoccurring_values().alias("sum_reoccurring_values"),
        pl.col(col_values).ts.symmetry_looking(ratio=0.25).alias("symmetry_looking_ratio_0.25"),
        pl.col(col_values).ts.time_reversal_asymmetry_statistic(n_lags=3).alias("time_reversal_asymmetry_statistic_n_lags_3"),
        pl.col(col_values).ts.variation_coefficient().alias("variation_coefficient"),
        pl.col(col_values).ts.var_gt_std().alias("var_gt_std")
    ]
    return features_medium

def get_large():
    pass
