import perfplot
from typing import Union, Callable
import polars as pl
import pandas as pd
from functime.feature_extraction import tsfresh as f_ts
from tsfresh.feature_extraction import feature_calculators as ts
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# list (functime_function, tsfresh_function, params)
_FUNC_PARAMS_BENCH  = [
    (f_ts.absolute_energy, ts.abs_energy, {}, {}),
    (f_ts.absolute_maximum, ts.absolute_maximum, {}, {}),
    (f_ts.absolute_sum_of_changes, ts.absolute_sum_of_changes, {}, {}),
    (f_ts.approximate_entropy, ts.approximate_entropy, {"run_length": 2, "filtering_level": 0.5}, {"m": 2, "r": 0.5}),
    # (f_ts.augmented_dickey_fuller, ts.augmented_dickey_fuller, "param")
    (f_ts.autocorrelation, ts.autocorrelation, {"n_lags": 4}, {"lag": 4}),
    (f_ts.autoregressive_coefficients, ts.ar_coefficient, {"n_lags": 4}, {"param": [{"coeff": i, "k": 4}] for i in range(5)}),
    (f_ts.benford_correlation2, ts.benford_correlation, {}, {}),
    (f_ts.benford_correlation, ts.benford_correlation, {}, {}),
    (f_ts.binned_entropy, ts.binned_entropy, {"bin_count": 10}, {"max_bins": 10}),
    (f_ts.c3, ts.c3, {"n_lags": 10}, {"lag": 10}),
    (f_ts.change_quantiles, ts.change_quantiles, {"q_low": 0.1, "q_high": 0.9, "is_abs": True}, {"ql": 0.1, "qh": 0.9, "isabs": True, "f_agg": "mean"}),
    (f_ts.cid_ce, ts.cid_ce, {"normalize": True}, {"normalize": True}),
    (f_ts.count_above, ts.count_above, {"threshold": 0.0}, {"t": 0.0}),
    (f_ts.count_above_mean, ts.count_above_mean, {}, {}),
    (f_ts.count_below, ts.count_below, {"threshold": 0.0}, {"t": 0.0}),
    (f_ts.count_below_mean, ts.count_below_mean, {}, {}),
    # (f_ts.cwt_coefficients, ts.cwt_coefficients, {"widths": (1, 2, 3), "n_coefficients": 2},{"param": {"widths": (1, 2, 3), "coeff": 2, "w": 1}}),
    (f_ts.energy_ratios, ts.energy_ratio_by_chunks, {"n_chunks": 6}, {"param": [{"num_segments": 6, "segment_focus": i} for i in range(6)]}),
    (f_ts.first_location_of_maximum, ts.first_location_of_maximum, {}, {}),
    (f_ts.first_location_of_minimum, ts.first_location_of_minimum, {}, {}),
    # (f_ts.fourier_entropy, ts.fourier_entropy, {"n_bins": 10}, {"bins": 10}),
    # (f_ts.friedrich_coefficients, ts.friedrich_coefficients, {"polynomial_order": 3, "n_quantiles": 30}, {"params": [{"m": 3, "r": 30}]}),
    (f_ts.has_duplicate, ts.has_duplicate, {}, {}),
    (f_ts.has_duplicate_max, ts.has_duplicate_max, {}, {}),
    (f_ts.has_duplicate_min, ts.has_duplicate_min, {}, {}),
    (f_ts.index_mass_quantile, ts.index_mass_quantile, {"q": 0.5}, {"param": [{"q": 0.5}]}),
    (f_ts.large_standard_deviation, ts.large_standard_deviation, {"ratio": 0.25}, {"r": 0.25}),
    (f_ts.last_location_of_maximum, ts.last_location_of_maximum, {}, {}),
    (f_ts.last_location_of_minimum, ts.last_location_of_minimum, {}, {}),
    # (f_ts.lempel_ziv_complexity, ts.lempel_ziv_complexity, {"n_bins": 5}, {"bins": 5}),
    # (f_ts.linear_trend, ts.linear_trend, {}, {"params": [{"attr": "slope"}, {"attr": "intercept"}]}),
    (f_ts.longest_streak_above_mean, ts.longest_strike_above_mean, {}, {}),
    (f_ts.longest_streak_below_mean, ts.longest_strike_below_mean, {}, {}),
    (f_ts.mean_abs_change, ts.mean_abs_change, {}, {}),
    (f_ts.mean_change, ts.mean_change, {}, {}),
    (f_ts.mean_n_absolute_max, ts.mean_n_absolute_max, {"n_maxima": 20}, {"number_of_maxima": 20}),
    (f_ts.mean_second_derivative_central, ts.mean_second_derivative_central, {}, {}),
    (f_ts.number_crossings, ts.number_crossing_m, {"crossing_value": 0.0}, {"m": 0.0}),
    (f_ts.number_cwt_peaks, ts.number_cwt_peaks, {"max_width: 5"}, {"n": 5}),
    (f_ts.number_peaks, ts.number_peaks, {"support": 5}, {"n": 5}),
    # (f_ts.partial_autocorrelation, ts.partial_autocorrelation, "param"),
    (f_ts.percent_reoccuring_values, ts.percentage_of_reoccurring_values_to_all_values, {}, {}),
    (f_ts.percent_reocurring_points, ts.percentage_of_reoccurring_datapoints_to_all_datapoints, {}, {}),
    (f_ts.permutation_entropy, ts.permutation_entropy, {"tau": 1,"n_dims": 3}, {"tau": 1,"dimension": 3}),
    (f_ts.range_count, ts.range_count, {"lower": 0, "upper": 9, "closed": 'none'}, {"min": 0, "max": 9}),
    (f_ts.ratio_beyond_r_sigma, ts.ratio_beyond_r_sigma, {"ratio": 2}, {"r": 2}),
    (f_ts.ratio_n_unique_to_length, ts.ratio_value_number_to_time_series_length, {}, {}),
    (f_ts.root_mean_square, ts.root_mean_square, {}, {}),
    (f_ts.sample_entropy, ts.sample_entropy, {}, {}),
    (f_ts.spkt_welch_density, ts.spkt_welch_density, {"n_coeffs": 10}, {"param": [{"coeff": i} for i in range(10)]}),
    (f_ts.sum_reocurring_points, ts.sum_of_reoccurring_data_points, {}, {}),
    (f_ts.sum_reocurring_values, ts.sum_of_reoccurring_values, {}, {}),
    (f_ts.symmetry_looking, ts.symmetry_looking, {"ratio": 0.25}, {"param": [{"r": 0.25}]}),
    (f_ts.time_reversal_asymmetry_statistic, ts.time_reversal_asymmetry_statistic, {"n_lags": 3}, {"lag": 3}),
    (f_ts.variation_coefficient, ts.variation_coefficient, {}, {}),
    (f_ts.var_gt_std, ts.variance_larger_than_standard_deviation, {}, {})
]

_M4_DATASET = "data/M4_daily.parquet"

DF_PANDAS = pd.melt(pd.read_parquet(_M4_DATASET)).drop(columns=["variable"]).dropna().reset_index(drop=True)
DF_PL_EAGER = pl.from_pandas(DF_PANDAS)
DF_PL_LAZY = DF_PL_EAGER.lazy()

def slice_dataframe(df_pandas: pd.DataFrame, df_pl_eager: pl.DataFrame, df_pl_lazy: pl.LazyFrame, n: int):
    return df_pandas.head(n), df_pl_eager.head(n), df_pl_lazy.head(n)


def benchmark(f_feat: Callable, ts_feat: Callable, f_params: dict, ts_params: dict, expr: bool):
    upper_n = 24
    if f_feat.__name__ in ("binned_entropy", "approximate_entropy", "permutation_entropy", "sample_entropy"):
        upper_n = 14
    benchmark = perfplot.bench(
        setup = lambda n: slice_dataframe(DF_PANDAS, DF_PL_EAGER, DF_PL_LAZY, n)[:2],
        kernels = [
            lambda x, _y: ts_feat(x["value"], **ts_params),
            lambda _x, y: f_feat(y["value"], **f_params) if not expr else y.select(f_feat(pl.col("value"), **f_params))
        ],
        n_range = [2**k for k in range(6, upper_n)],
        equality_check=False,
        labels=["tsfresh", "tsfresh", "functime"]
    )
    return benchmark


def create_df_benchmarks(params: list[tuple])-> pl.DataFrame:
    bench_df = pl.DataFrame(schema={"id": pl.Utf8, "n": pl.Int64, "tsfresh": pl.Float64, "functime": pl.Float64, "X_speed": pl.Float64})
    for x in params:
        try:
            print("Feature: {}".format(x))
            bench = benchmark(
                f_feat = x[0],
                ts_feat = x[1],
                f_params = x[2],
                ts_params = x[3],
                expr = False
            )
            bench_df = pl.concat([
                pl.DataFrame({
                    "id": [x[0].__name__]*len(bench.n_range),
                    "n": bench.n_range,
                    "tsfresh": bench.timings_s[0],
                    "functime": bench.timings_s[1],
                    "X_speed": bench.timings_s[0] / bench.timings_s[1]                
                }),
                bench_df]
            )
        except:
            print("Failure for feature: {}".format(x))
    return bench_df


df_benchs = create_df_benchmarks(params = _FUNC_PARAMS_BENCH[:2])
