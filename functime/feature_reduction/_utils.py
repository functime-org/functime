import polars as pl
from functime.feature_extractors import FeatureExtractor

def get_small(col_values: str):
    ALL_SMALL = [
        pl.col(col_values).ts.absolute_energy().alias("absolute_energy"),
        pl.col(col_values).ts.absolute_maximum().alias("absolute_maximum"),
        pl.col(col_values).ts.absolute_sum_of_changes().alias("absolute_sum_of_changes"),
        pl.col(col_values).ts.lempel_ziv_complexity(threshold=(pl.col("price").max() - pl.col("price").min()) / 2).alias("lempel_ziv_complexity_threshold_max-min_div_2"),
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
        # pl.col(col_values).ts.linear_trend().alias("linear_trend"),
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
    return ALL_SMALL

def get_medium():
    pass

def get_large():
    pass
