import perfplot
from typing import Union, Callable
import polars as pl
import pandas as pd
from functime.feature_extraction import tsfresh as func_feat
from tsfresh.feature_extraction import feature_calculators as ts_feat

# list (functime_function, tsfresh_function, params)
_FUNC_PARAMS_BENCH  = [
    (func_feat.absolute_energy, ts_feat.abs_energy, None, None),
    (func_feat.absolute_maximum, ts_feat.absolute_maximum, None, None),
    (func_feat.absolute_sum_of_changes, ts_feat.absolute_sum_of_changes, None, None),
    (func_feat.approximate_entropies, ts_feat.approximate_entropy, {"run_lenght": 2, "filtering_levels": 0.5}, {"m": 2, "r": 0.5}),
    # (func_feat.augmented_dickey_fuller, ts_feat.augmented_dickey_fuller, "param")
    (func_feat.autocorrelation, ts_feat.autocorrelation, {"n_lags": 4}, {"lag": 4}),
    # (func_feat.autoregressive_coefficients, ts_feat.ar_coefficient, "param"),
    (func_feat.benford_correlation2, ts_feat.benford_correlation, None, None),
    (func_feat.binned_entropy, ts_feat.binned_entropy, {"bin_count": 10}, {"max_bins": 10}),
    (func_feat.c3, ts_feat.c3, {"n_lags": 10}, {"lag": 10}),
    (func_feat.change_quantiles, ts_feat.change_quantiles, {"q_low": 0.1, "q_high": 0.9, "is_abs": True}, {"ql": 0.1, "qh": 0.9, "isabs": True, "f_agg": "mean"}),
    (func_feat.cid_ce, ts_feat.cid_ce, {"normalize": True}, {"normalize": True}),
    (func_feat.count_above, ts_feat.count_above, {"threshold": 0.0}, {"t": 0.0}),
    (func_feat.count_above_mean, ts_feat.count_above_mean, None, None),
    (func_feat.count_below, ts_feat.count_below, {"threshold": 0.0}, {"t": 0.0}),
    (func_feat.count_below_mean, ts_feat.count_below_mean, None, None),
    # (func_feat.cwt_coefficients, ts_feat.cwt_coefficients, "param"),
    (func_feat.energy_ratios, ts_feat.energy_ratio_by_chunks, {"n_chunks": 10}, {"param": None}),
    (func_feat.first_location_of_maximum, ts_feat.first_location_of_maximum, None, None),
    (func_feat.first_location_of_minimum, ts_feat.first_location_of_minimum, None, None),
    (func_feat.fourier_entropy, ts_feat.fourier_entropy, {"n_bins": 10}, {"bins": 10}),
    # (func_feat.friedrich_coefficients, ts_feat.friedrich_coefficients, {"polynomial_order": 3, "n_quantiles": 30}, {"params": [{"m": 3, "r": 30}]}),
    (func_feat.has_duplicate, ts_feat.has_duplicate, None, None),
    (func_feat.has_duplicate_max, ts_feat.has_duplicate_max, None, None),
    (func_feat.has_duplicate_min, ts_feat.has_duplicate_min, None, None),
    (func_feat.index_mass_quantile, ts_feat.index_mass_quantile, {"q": 0.5}, {"params": [{"q": 0.5}]}),
    (func_feat.large_standard_deviation, ts_feat.large_standard_deviation, {"ratio": 0.25}, {"r": 0.25}),
    (func_feat.last_location_of_maximum, ts_feat.last_location_of_maximum, None, None),
    (func_feat.last_location_of_minimum, ts_feat.last_location_of_minimum, None, None),
    (func_feat.lempel_ziv_complexity, ts_feat.lempel_ziv_complexity, {"n_bins": 5}, {"bins": 5}),
    (func_feat.linear_trend, ts_feat.linear_trend, None, {"params": [{"attr": "slope"}, {"attr": "intercept"}, {"attr": "rss"}]}),
    (func_feat.longest_strike_above_mean, ts_feat.longest_strike_above_mean, None, None),
    (func_feat.longest_strike_below_mean, ts_feat.longest_strike_below_mean, None, None),
    (func_feat.mean_abs_change, ts_feat.mean_abs_change, None, None),
    (func_feat.mean_change, ts_feat.mean_change, None, None),
    (func_feat.mean_n_absolute_max, ts_feat.mean_n_absolute_max, {"n_maxima": 20}, {"number_of_maxima": 20}),
    (func_feat.mean_second_derivative_central, ts_feat.mean_second_derivative_central, None, None),
    (func_feat.number_crossings, ts_feat.number_crossing_m, {"crossing_value": 0.0}, {"m": 0.0}),
    # (func_feat.number_cwt_peaks, ts_feat.number_cwt_peaks, "n"),
    (func_feat.number_peaks, ts_feat.number_peaks, {"support": 5}, {"n": 5}),
    # (func_feat.partial_autocorrelation, ts_feat.partial_autocorrelation, "param"),
    (func_feat.percent_reoccuring_values, ts_feat.percentage_of_reoccurring_values_to_all_values, None, None),
    (func_feat.percent_reocurring_points, ts_feat.percentage_of_reoccurring_datapoints_to_all_datapoints, None, None),
    # (func_feat.permutation_entropy, ts_feat.permutation_entropy, "tau, dimension"),
    # (func_feat.range_count, ts_feat.range_count, "min, max"),
    # (func_feat.ratio_beyond_r_sigma, ts_feat.ratio_beyond_r_sigma, "r"),
    (func_feat.ratio_n_unique_to_length, ts_feat.ratio_value_number_to_time_series_length, None, None),
    (func_feat.root_mean_square, ts_feat.root_mean_square, None, None),
    (func_feat.sample_entropy, ts_feat.sample_entropy, None, None),
    # (func_feat.spkt_welch_density, ts_feat.spkt_welch_density, "param"),
    (func_feat.sum_reocurring_points, ts_feat.sum_of_reoccurring_data_points, None, None),
    (func_feat.sum_reocurring_values, ts_feat.sum_of_reoccurring_values, None, None),
    # (func_feat.symmetry_looking, ts_feat.symmetry_looking, "param"),
    # (func_feat.time_reversal_asymmetry_statistic, ts_feat.time_reversal_asymmetry_statistic, "lag"),
]

_M4_DATASET = "data/M4_daily.parquet"

DF_PANDAS = pd.melt(pd.read_parquet(_M4_DATASET)).drop(columns=["variable"]).dropna().reset_index(drop=True)
DF_PL_EAGER = pl.from_pandas(DF_PANDAS)
DF_PL_LAZY = DF_PL_EAGER.lazy()

def slice_dataframe(df_pandas: pd.DataFrame, df_pl_eager: pl.DataFrame, df_pl_lazy: pl.LazyFrame, n: int):
    return df_pandas.head(n), df_pl_eager.head(n), df_pl_lazy.head(n)


def benchmark(functime_feat: Callable, tsfresh_feat: Callable, functime_params: dict, tsfresh_params: dict):
    out = perfplot.bench(
        setup = lambda n: slice_dataframe(DF_PANDAS, DF_PL_EAGER, DF_PL_LAZY, n),
        kernels = [
            lambda x, _y, _z: tsfresh_feat(x["value"]) if tsfresh_params else tsfresh_feat(x["value"], **tsfresh_params),
            lambda _x, y, _z: y.select(functime_feat(pl.col("value"))) if functime_params else y.select(functime_feat(pl.col("value"), **functime_params)) ,
            lambda _x, _y, z: z.select(functime_feat(pl.col("value"))).collect() if functime_params else z.select(functime_feat(pl.col("value"), **functime_params)).collect()
        ],
        n_range = [2**k for k in range(24)],
        labels=["tsfresh", "eager", "lazy"],
    )
    return out

out = benchmark(functime_feat = _FUNC_PARAMS_BENCH[0][0], tsfresh_feat = _FUNC_PARAMS_BENCH[0][1], functime_params=_FUNC_PARAMS_BENCH[0][2], tsfresh_params = _FUNC_PARAMS_BENCH[0][3])
out.show()
# out.save("perf.png", transparent=True, bbox_inches="tight")