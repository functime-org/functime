import perfplot
from typing import Union, Callable
import polars as pl
import pandas as pd
from functime.feature_extraction import tsfresh as func_feat
from tsfresh.feature_extraction import feature_calculators as ts_feat
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# list (functime_function, tsfresh_function, params)
_FUNC_PARAMS_BENCH  = [
    (func_feat.absolute_energy, ts_feat.abs_energy, {}, {}),
    (func_feat.absolute_maximum, ts_feat.absolute_maximum, {}, {}),
    (func_feat.absolute_sum_of_changes, ts_feat.absolute_sum_of_changes, {}, {}),
    (func_feat.approximate_entropy, ts_feat.approximate_entropy, {"run_length": 2, "filtering_level": 0.5}, {"m": 2, "r": 0.5}),
    # (func_feat.augmented_dickey_fuller, ts_feat.augmented_dickey_fuller, "param")
    (func_feat.autocorrelation, ts_feat.autocorrelation, {"n_lags": 4}, {"lag": 4}),
    (func_feat.autoregressive_coefficients, ts_feat.ar_coefficient, {"n_lags": 4}, {"param": [{"coeff": i, "k": 4}] for i in range(5)}),
    (func_feat.benford_correlation2, ts_feat.benford_correlation, {}, {}),
    (func_feat.benford_correlation, ts_feat.benford_correlation, {}, {}),
    (func_feat.binned_entropy, ts_feat.binned_entropy, {"bin_count": 10}, {"max_bins": 10}),
    (func_feat.c3, ts_feat.c3, {"n_lags": 10}, {"lag": 10}),
    (func_feat.change_quantiles, ts_feat.change_quantiles, {"q_low": 0.1, "q_high": 0.9, "is_abs": True}, {"ql": 0.1, "qh": 0.9, "isabs": True, "f_agg": "mean"}),
    (func_feat.cid_ce, ts_feat.cid_ce, {"normalize": True}, {"normalize": True}),
    (func_feat.count_above, ts_feat.count_above, {"threshold": 0.0}, {"t": 0.0}),
    (func_feat.count_above_mean, ts_feat.count_above_mean, {}, {}),
    (func_feat.count_below, ts_feat.count_below, {"threshold": 0.0}, {"t": 0.0}),
    (func_feat.count_below_mean, ts_feat.count_below_mean, {}, {}),
    # (func_feat.cwt_coefficients, ts_feat.cwt_coefficients, {"widths": (1, 2, 3), "n_coefficients": 2},{"param": {"widths": (1, 2, 3), "coeff": 2, "w": 1}}),
    (func_feat.energy_ratios, ts_feat.energy_ratio_by_chunks, {"n_chunks": 6}, {"param": [{"num_segments": 6, "segment_focus": i} for i in range(6)]}),
    (func_feat.first_location_of_maximum, ts_feat.first_location_of_maximum, {}, {}),
    (func_feat.first_location_of_minimum, ts_feat.first_location_of_minimum, {}, {}),
    # (func_feat.fourier_entropy, ts_feat.fourier_entropy, {"n_bins": 10}, {"bins": 10}),
    # (func_feat.friedrich_coefficients, ts_feat.friedrich_coefficients, {"polynomial_order": 3, "n_quantiles": 30}, {"params": [{"m": 3, "r": 30}]}),
    (func_feat.has_duplicate, ts_feat.has_duplicate, {}, {}),
    (func_feat.has_duplicate_max, ts_feat.has_duplicate_max, {}, {}),
    (func_feat.has_duplicate_min, ts_feat.has_duplicate_min, {}, {}),
    (func_feat.index_mass_quantile, ts_feat.index_mass_quantile, {"q": 0.5}, {"param": [{"q": 0.5}]}),
    (func_feat.large_standard_deviation, ts_feat.large_standard_deviation, {"ratio": 0.25}, {"r": 0.25}),
    (func_feat.last_location_of_maximum, ts_feat.last_location_of_maximum, {}, {}),
    (func_feat.last_location_of_minimum, ts_feat.last_location_of_minimum, {}, {}),
    # (func_feat.lempel_ziv_complexity, ts_feat.lempel_ziv_complexity, {"n_bins": 5}, {"bins": 5}),
    # (func_feat.linear_trend, ts_feat.linear_trend, {}, {"params": [{"attr": "slope"}, {"attr": "intercept"}]}),
    (func_feat.longest_strike_above_mean, ts_feat.longest_strike_above_mean, {}, {}),
    (func_feat.longest_strike_below_mean, ts_feat.longest_strike_below_mean, {}, {}),
    (func_feat.mean_abs_change, ts_feat.mean_abs_change, {}, {}),
    (func_feat.mean_change, ts_feat.mean_change, {}, {}),
    (func_feat.mean_n_absolute_max, ts_feat.mean_n_absolute_max, {"n_maxima": 20}, {"number_of_maxima": 20}),
    (func_feat.mean_second_derivative_central, ts_feat.mean_second_derivative_central, {}, {}),
    (func_feat.number_crossings, ts_feat.number_crossing_m, {"crossing_value": 0.0}, {"m": 0.0}),
    (func_feat.number_cwt_peaks, ts_feat.number_cwt_peaks, {"max_width: 5"}, {"n": 5}),
    (func_feat.number_peaks, ts_feat.number_peaks, {"support": 5}, {"n": 5}),
    # (func_feat.partial_autocorrelation, ts_feat.partial_autocorrelation, "param"),
    (func_feat.percent_reoccuring_values, ts_feat.percentage_of_reoccurring_values_to_all_values, {}, {}),
    (func_feat.percent_reocurring_points, ts_feat.percentage_of_reoccurring_datapoints_to_all_datapoints, {}, {}),
    (func_feat.permutation_entropy, ts_feat.permutation_entropy, {"tau": 1,"n_dims": 3}, {"tau": 1,"dimension": 3}),
    (func_feat.range_count, ts_feat.range_count, {"lower": 0, "upper": 9, "closed": 'none'}, {"min": 0, "max": 9}),
    (func_feat.ratio_beyond_r_sigma, ts_feat.ratio_beyond_r_sigma, {"ratio": 2}, {"r": 2}),
    (func_feat.ratio_n_unique_to_length, ts_feat.ratio_value_number_to_time_series_length, {}, {}),
    (func_feat.root_mean_square, ts_feat.root_mean_square, {}, {}),
    (func_feat.sample_entropy, ts_feat.sample_entropy, {}, {}),
    (func_feat.spkt_welch_density, ts_feat.spkt_welch_density, {"n_coeffs": 10}, {"param": [{"coeff": i} for i in range(10)]}),
    (func_feat.sum_reocurring_points, ts_feat.sum_of_reoccurring_data_points, {}, {}),
    (func_feat.sum_reocurring_values, ts_feat.sum_of_reoccurring_values, {}, {}),
    (func_feat.symmetry_looking, ts_feat.symmetry_looking, {"ratio": 0.25}, {"param": [{"r": 0.25}]}),
    (func_feat.time_reversal_asymmetry_statistic, ts_feat.time_reversal_asymmetry_statistic, {"n_lags": 3}, {"lag": 3}),
    (func_feat.variation_coefficient, ts_feat.variation_coefficient, {}, {}),
    (func_feat.var_gt_std, ts_feat.variance_larger_than_standard_deviation, {}, {})
]

_M4_DATASET = "data/M4_daily.parquet"

DF_PANDAS = pd.melt(pd.read_parquet(_M4_DATASET)).drop(columns=["variable"]).dropna().reset_index(drop=True)
DF_PL_EAGER = pl.from_pandas(DF_PANDAS)
DF_PL_LAZY = DF_PL_EAGER.lazy()

def slice_dataframe(df_pandas: pd.DataFrame, df_pl_eager: pl.DataFrame, df_pl_lazy: pl.LazyFrame, n: int):
    return df_pandas.head(n), df_pl_eager.head(n), df_pl_lazy.head(n)


def benchmark(functime_feat: Callable, tsfresh_feat: Callable, functime_params: dict, tsfresh_params: dict):
    benchmark = perfplot.bench(
        setup = lambda n: slice_dataframe(DF_PANDAS, DF_PL_EAGER, DF_PL_LAZY, n),
        kernels = [
            lambda x, _y, _z: tsfresh_feat(x["value"], **tsfresh_params),
            lambda _x, y, _z: y.select(functime_feat(pl.col("value"), **functime_params)),
            lambda _x, _y, z: z.select(functime_feat(pl.col("value"), **functime_params)).collect()
        ],
        n_range = [2**k for k in range(4, 24, 2)],
        labels=["tsfresh", "eager", "lazy"],
    )
    return benchmark


def all_benchmarks(params: list[tuple])-> list:
    res = []
    for x in params:
        try:
            bench = benchmark(
                functime_feat = x[0],
                tsfresh_feat = x[1],
                functime_params = x[2],
                tsfresh_params = x[3]
            )
            res.append({
                "function": x[0].__name__,
                "bench": pl.DataFrame({
                    "n": bench.n_range,
                    "tsfresh": bench.timings_s[0],
                    "eager": bench.timings_s[1],
                    "lazy": bench.timings_s[2]                       
                })
            })
        except:
            pass
    return res


res = all_benchmarks(params = _FUNC_PARAMS_BENCH[:2])

fig = make_subplots(rows=8, cols=1, subplot_titles=['feature {}'.format(i["function"]) for i in res])

# Iterate through the DataFrames and add traces to the subplots
for i, d in enumerate(res[:16]):
    df = d["bench"]
    # Add traces for each column in the DataFrame
    for column in df.columns[1:]:
        trace = go.Scatter(x=df['n'], y=df[column], mode='lines', name=column, legendgroup="{}".format(i+1))
        fig.add_trace(trace, row=i+1, col=1)

# Update layout and show the plot
fig.update_layout(height=1200, width=500, title_text='Subplots of 8 DataFrames', legend_tracegroupgap = 80)
fig.show()

# out.save("perf.png")