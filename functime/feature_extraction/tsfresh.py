from itertools import product
from typing import List, Mapping, Optional

import bottleneck as bn
import numpy as np
import polars as pl
import math
from numpy.linalg import lstsq
from scipy.signal import ricker, welch

from functime.base import transformer


# def absolute_energy(x: pl.Expr) -> pl.Expr:
#     return x.dot(x)


# def absolute_maximum(x: pl.Expr) -> pl.Expr:
#     return x.abs().max()


# def absolute_sum_of_changes(x: pl.Expr) -> pl.Expr:
#     return x.diff(n=1, null_behavior="drop").abs().sum()


# def autocorr(x: pl.Expr, max_lags: int) -> List[pl.Expr]:
#     acf = [pl.corr(x, x.shift(i), ddof=i) for i in range(1, max_lags + 1)]
#     return acf


# def linear_trend(x: pl.Expr) -> Mapping[str, pl.Expr]:
#     x_range = pl.int_range(1, x.len() + 1)
#     beta = pl.cov(x, x_range) / x.var()
#     alpha = x.mean() - beta * x_range.mean()
#     resid = x - beta * x_range + alpha
#     rss = resid.pow(2).sum()
#     return {"slope": beta, "intercept": alpha, "rss": rss}


# def _phis(x: pl.Expr, m: int, N: int, rs: List[float]) -> List[pl.Expr]:
#     n = N - m + 1
#     x_runs = [x.slice(i, m) for i in range(n)]
#     max_dists = [(x_i - x_j).max() for x_i, x_j in product(x_runs, x_runs)]
#     phis = []
#     for r in rs:
#         r_comparisons = [d.le(r) for d in max_dists]
#         counts = [
#             (pl.sum_horizontal(r_comparisons[i : i + n]) / n).log()
#             for i in range(0, n**2, n)
#         ]
#         phis.append((1 / n) * pl.sum_horizontal(counts))
#     return phis


# def approximate_entropy(x: pl.Expr, sigma: float, m: int = 2) -> List[pl.Expr]:
#     r_multipliers = [0.1, 0.3, 0.5, 0.7, 0.9]  # Hardcoded
#     rs = [sigma * multiplier for multiplier in r_multipliers]
#     N = x.len()
#     phis_m = _phis(x, m=m, N=N, rs=rs)
#     phis_m_plus_1 = _phis(x, m=m + 1, N=N, rs=rs)
#     entropies = [phis_m[i] - phis_m_plus_1[i] for i in range(len(phis_m))]
#     return entropies


# # NOT PURE
# def ar_coefficients(x: pl.Series, max_lags: int) -> List[pl.Expr]:
#     """AR(p) with drift."""
#     X = np.vstack(
#         [
#             np.asarray([x.shift(i).slice(max_lags) for i in range(1, max_lags + 1)]),
#             np.ones(x.len() - max_lags),
#         ]
#     ).T
#     y = np.asarray(x.slice(max_lags))
#     return lstsq(X, y, rcond=None)[0]


# # NOT PURE
# def augmented_dickey_fuller(x: pl.Series, x_diff: pl.Series, max_lags: int) -> pl.Expr:
#     """Return test statistic for Augmented Dickey Fuller with drift."""
#     k = x_diff.len()
#     X = np.vstack(
#         [
#             x.slice(max_lags),
#             np.asarray(
#                 [x_diff.shift(i).slice(max_lags) for i in range(1, max_lags + 1)]
#             ),
#             np.ones(k - max_lags),
#         ]
#     ).T
#     y = x_diff.slice(max_lags).to_numpy(zero_copy_only=True)
#     coeffs, resids, _, _ = lstsq(X, y, rcond=None)
#     mse = bn.nansum(resids**2) / (k - X.shape[1])
#     x_arr = np.asarray(x).T, np.asarray(x)
#     cov = mse * np.linalg.inv(np.dot(x_arr.T, x))
#     stderrs = np.sqrt(np.diag(cov))
#     return coeffs[0] / stderrs[0]


# # NOT PURE
# def binned_entropy(x: pl.Series, bin_count: int = 10) -> pl.Expr:
#     histogram = x.hist(bin_count=bin_count)
#     counts = histogram.get_column(histogram.columns[-1])
#     probs = counts / x.len()
#     probs = pl.when(probs == 0.0).then(pl.lit(1.0)).otherwise(probs)
#     return (probs * probs.log()).sum().mul(-1)


# def c3(x: pl.Expr, lags: Optional[List[int]] = None) -> Mapping[str, pl.Expr]:
#     """Requires all time series length to be > 2 * lag."""
#     lags = lags or [1, 2, 3, 4]
#     n = x.len()
#     measures = {}
#     for lag in lags:
#         k = n - 2 * lag
#         measure = (
#             pl.sum_horizontal(
#                 [
#                     x.list.get(i + 2 * lag) * x.list.get(i + lag) * x.list.get(i)
#                     for i in range(k)
#                 ]
#             )
#             / k
#         )
#         measures[str(lag)] = measure
#     return measures


# def change_quantiles(
#     x: pl.Expr,
#     q_lower: Optional[List[float]] = None,
#     q_upper: Optional[List[float]] = None,
# ) -> List[pl.Struct]:
#     q_lower = q_lower or [0.0, 0.2, 0.4, 0.6, 0.8]
#     q_upper = q_lower or [0.2, 0.4, 0.6, 0.8, 1.0]
#     change_stats = []
#     x_diff = x.diff().abs()
#     quantiles = [
#         (lower, upper) for lower, upper in product(q_lower, q_upper) if lower < upper
#     ]
#     for q in quantiles:
#         bin_labels = x.qcut(q=q, labels=[str(i) for i in range(len(q))]).to_physical()
#         bin_labels_0 = bin_labels == 0
#         # Count changes that start and end inside the corridor
#         changes_idx = bin_labels_0 & bin_labels_0.shift(1)
#         x_diff_i = x_diff.filter(changes_idx is True)
#         stat = {"mean": x_diff_i.mean(), "var": x_diff_i.var()}
#         change_stats.append(stat)

#     return change_stats


# def cid_ce(x: pl.Expr) -> pl.Expr:
#     return ((x - x.shift(-1)) ** 2).sum() ** (1 / 2)


# def ppv(x: pl.Expr, t: int = 0) -> Mapping[str, pl.Expr]:
#     return {"above": x.ge(t).mean(), "below": x.le(t).mean()}


# def ppv_mean(x: pl.Expr) -> Mapping[str, pl.Expr]:
#     bias = x.mean()
#     return {"above": x.ge(bias).mean(), "below": x.le(bias).mean()}


# # NOT PURE
# def cwt(
#     x: pl.Series, widths: Optional[List[int]] = None, n_coefficients: int = 14
# ) -> Mapping[str, pl.Expr]:
#     widths = widths or [2, 5, 10, 20]
#     convolution = np.empty((len(widths), x.len()), dtype=np.float32)
#     for i, width in enumerate(widths):
#         points = np.min([10 * width, x.len()])
#         wavelet_x = np.conj(ricker(points, width)[::-1])
#         convolution[i] = np.convolve(x.to_numpy(zero_copy_only=True), wavelet_x)
#     outputs = {}
#     for coeff_idx in range(min(n_coefficients, convolution.shape[1])):
#         for width in widths:
#             outputs[f"{coeff_idx}_{width}"] = convolution[
#                 widths.index(width), coeff_idx
#             ]
#     return outputs


# def energy_ratio_by_chunks(x: pl.Expr, n_segments: int = 10) -> List[pl.Expr]:
#     full_energy = (x**2).sum().alias("full_energy")
#     n = x.len()
#     chunk_size = n // n_segments
#     ratios = []
#     for i in pl.int_range(0, n, chunk_size):
#         ratios.append((x.slice(i, chunk_size)) ** 2.0).sum().div(full_energy)
#     return ratios


# def first_location_of_maximum(x: pl.Expr):
#     return x.arg_max()


# def first_location_of_minimum(x: pl.Expr):
#     return x.arg_min()


# def last_location_of_maximum(x: pl.Expr):
#     return 1.0 - x.reverse().arg_max() / x.len()


# def last_location_of_minimum(x: pl.Expr):
#     return 1.0 - x.reverse().arg_min() / x.len()


# # NOT PURE
# def fourier_entropy(x: pl.Series, bins: Optional[List[int]] = None):
#     bins = bins or [2, 3, 5, 10, 100]
#     _, pxx = welch(x, nperseg=min(x.len(), 256))
#     return binned_entropy(pxx / bn.nanmax(pxx), bins)


# # NOT PURE
# def friedrich_coefficients(x: pl.Series, m: int = 3, r: int = 30):
#     X = (
#         x.alias("signal")
#         .to_frame()
#         .with_columns(
#             delta=x.diff().alias("delta"),
#             quantile=x.qcut(q=r, labels=[str(i) for i in range(r)]),
#         )
#         .lazy()
#     )
#     X_means = (
#         X.groupby("quantile")
#         .agg([pl.all().mean()])
#         .drop_nulls()
#         .collect(streaming=True)
#     )
#     coeffs = np.polyfit(
#         X_means.get_column("signal").to_numpy(zero_copy_only=True),
#         X_means.get_column("delta").to_numpy(zero_copy_only=True),
#         deg=m,
#     )
#     return coeffs


# def has_duplicate(x: pl.Expr):
#     return x.n_unique() < x.len()


# def has_duplicate_max(x: pl.Expr):
#     return (x == x.max()).sum() > 1


# def has_duplicate_min(x: pl.Expr):
#     return (x == x.min()).sum() > 1


# def index_mass_quantile(x: pl.Expr, quantiles: Optional[List[float]] = None):
#     quantiles = quantiles or [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
#     x_abs = x.abs()
#     x_sum = x.sum()
#     n = x.len()
#     mass_center = x_abs.cumsum() / x_sum
#     return [((mass_center >= q) + 1).arg_max() / n for q in quantiles]


# def kurtosis(x: pl.Expr):
#     return x.kurtosis()


# def large_std(x: pl.Expr):
#     x_std = x.std()
#     x_interval = x.max() - x.min()
#     return [x_std > (r * x_interval) for r in np.linspace(0.05, 0.95, 19)]


# # NOT PURE
# def lempel_ziv_complexity(x: pl.Series, bins: Optional[List[int]] = None):
#     bins = bins or [2, 3, 5, 10, 100]
#     complexities = []
#     for b in bins:
#         seq = x.search_sorted(
#             element=np.linspace(x.min(), x.max(), b + 1)[1:], side="left"
#         ).to_numpy()
#         sub_strs = set()
#         n = x.len()
#         ind, inc = 0, 1
#         while True:
#             if ind + inc > n:
#                 break
#             sub_str = seq[ind : ind + inc]
#             if sub_str in sub_strs:
#                 inc += 1
#             else:
#                 sub_strs.add(sub_str)
#                 ind += inc
#                 inc = 1
#         complexities.append(len(sub_str) / n)

#     return complexities


# def length(x: pl.Expr):
#     return x.len()


# def longest_strike_above_mean(x: pl.Expr):
#     pass


# def longest_strike_below_mean(x: pl.Expr):
#     pass


# def max_langevin_fixed_point(x: pl.Expr, m: int, r: float):
#     pass


# def maximum(x: pl.Expr):
#     return x.max()


# def mean(x: pl.Expr):
#     return x.mean()


# def mean_abs_change(x: pl.Expr):
#     return x.diff().abs().mean()


# def mean_change(x: pl.Expr):
#     return (x.last() - x.first()) / (x.len() - 1)


# def mean_n_absolute_max(x: pl.Expr, n_maxima: int):
#     pass


# def mean_second_derivative_central(x: pl.Expr):
#     return (x.last() - x.take(x.len() - 1) - x.take(1) + x.first()) / (
#         2 * (x.len() - 2)
#     )


# def median(x: pl.Expr):
#     return x.median()


# def minimum(x: pl.Expr):
#     return x.min()


# def number_crossing_threshold(x: pl.Expr, threshold: float = 0.0):
#     signs = (x > threshold).diff().sign()
#     return (signs != signs.shift()).sum()


# def number_cwt_peaks(x: pl.Expr, max_width: int):
#     pass


# def number_peaks(x: pl.Expr, support: int):
#     pass


def permutation_entropy(
    x: pl.Expr,
    tau:int=1,
    n_dims:int=3,
    base:float=math.e,
    normalize:bool=False
) -> pl.Expr:
    '''
    Computes permutation entropy.

    Paramters
    ---------
    tau : int
        The embedding time delay which controls the number of time periods between elements 
        of each of the new column vectors.
    n_dims : int, > 1
        The embedding dimension which controls the length of each of the new column vectors
    base : float
        The base for log in the entropy computation
    normalize : bool
        Whether to normalize in the entropy computation

    Reference
    ---------
        https://www.aptech.com/blog/permutation-entropy/
        https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#tsfresh.feature_extraction.feature_calculators.permutation_entropy
    '''

    # CSE should take care of x.shift(-n_dims+1).is_not_null() ?
    # If input is eager, then in the divide statement we don't need need
    # a lazy expression to compute the remaining length.
    max_shift = -n_dims + 1
    out = (
        pl.concat_list(x, *(x.shift(-i) for i in range(1,n_dims))) # create list columns
        .take_every(tau) # take every tau
        .filter(x.shift(max_shift).is_not_null()) # This is a filter because length of df is unknown
        .list.eval(pl.element().rank(method="ordinal")) # for each inner list, do an argsort
        .value_counts() # groupby and count, but returns a struct
        .struct.field("counts") # extract the field named "counts"
        / x.shift(max_shift).is_not_null().sum() # get probabilities, alt:(pl.count() + max_shift)
    ).entropy(base=base, normalize=normalize).suffix("_permutation_entropy")

    return out

def quantile(x: pl.Expr, q: float):
    return x.quantile(q)


def query_similarity_count(x: pl.Expr, query, threshold, normalize):
    pass


def ratio_beyond_r_sigma(x: pl.Expr, r: float):
    '''
    Returns the ration of values in the series that is beyond r*std from mean on both sides.

    Parameters
    ----------
    r : float
        The scaling factor for std
    '''
    expr = (
        pl.when(
            (x < x.mean() - pl.lit(r) * x.std())
            | (x > x.mean() + pl.lit(r) * x.std())
        ).then(pl.lit(1, dtype=pl.UInt32))
        .otherwise(pl.lit(0, dtype=pl.UInt32))
        .sum()
        / pl.count()
    )
    return expr

def ratio_n_unique_to_length(x: pl.Expr):
    return x.n_unique() / x.len()


def root_mean_square(x: pl.Expr):
    # Try (x.dot(x) / pl.count()).sqrt() and compare performance.
    # dot generally has pretty good performance
    return (x**2).mean().sqrt()


def _into_sequential_chunks(x:pl.Series, m:int) -> np.ndarray:
    n_rows = x.len() - m + 1
    matrix = []
    for i, values in enumerate(zip(x, *(x.shift(-i) for i in range(1,m)))):
        if i < n_rows:
            matrix.append(values)
    return np.asarray(matrix)

def sample_entropy(x: pl.Series, r:float=0.2) -> float:
    
    threshold = r * x.std(ddof=0)
    m = 2
    mat_m = _into_sequential_chunks(x, m)
    b = 0
    mat_m_p1 = _into_sequential_chunks(x, m+1)
    a = 0

    b = sum(
        np.sum((np.abs(mat_m[i, :] - mat_m[i+1:, :])).max(axis=1) <= threshold)
        for i in range(mat_m.shape[0]-1)
    )
    # for i in range(mat_m.shape[0]-1):
    #     row = mat_m[i, :] # numpy slice view, no copy
    #     to_compare = mat_m[i+1:, :] # numpy slice view, no copy
    #     b += np.sum((np.abs(row - to_compare)).max(axis=1) <= threshold)

    a = sum(
        np.sum((np.abs(mat_m_p1[i, :] - mat_m_p1[i+1:, :])).max(axis=1) <= threshold)
        for i in range(mat_m_p1.shape[0]-1)
    )
    # for i in range(mat_m_p1.shape[0]-1):
    #     row = mat_m_p1[i, :] # numpy slice view, no copy
    #     to_compare = mat_m_p1[i+1:, :] # numpy slice view, no copy
    #     a += np.sum((np.abs(row - to_compare)).max(axis=1) <= threshold)

    #print(b)
    # print(a)
    return np.log(b / a) # -ln(a/b) = ln(b/a)

def skewness(x: pl.Expr):
    return x.skew()


def spkt_welch_density(x: pl.Expr, coeff: int):
    pass


def standard_deviation(x: pl.Expr):
    return x.std()


def sum_reocurring_points(x: pl.Expr):
    pass


def sum_reocurring_values(x: pl.Expr):
    pass


def sum_values(x: pl.Expr):
    return x.sum()


def is_approx_symmetric(x: pl.Expr, r: float):
    return (x.mean() - x.median()).abs() < r * (x.max() - x.min())


def time_reversal_asymmetry_statistic(x: pl.Expr):
    pass


def variance(x: pl.Expr):
    return x.var()


def var_gt_std(x: pl.Expr):
    return x.var() > x.std()


def coefficient_of_variation(x: pl.Expr):
    return x.var() / x.mean()


# FFT Features


def fft_mean(x_fft: pl.List(pl.Struct(fields=(pl.Float32, pl.Float32)))) -> pl.Float32:
    pass


def fft_var(x_fft: pl.List(pl.Struct(fields=(pl.Float32, pl.Float32)))) -> pl.Float32:
    pass


def fft_skew(x_fft: pl.List(pl.Struct(fields=(pl.Float32, pl.Float32)))) -> pl.Float32:
    pass


def fft_kurtosis(
    x_fft: pl.List(pl.Struct(fields=(pl.Float32, pl.Float32)))
) -> pl.Float32:
    pass


def fft_coefficients(
    x_fft: pl.List(pl.Struct(fields=(pl.Float32, pl.Float32)))
) -> List[pl.Float32]:
    pass


# Feature extractor


@transformer
def add_tsfresh_features(max_lags: int, window_size: int, max_bins: int = 10):
    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        # Compute FFT
        pass

    return transform
