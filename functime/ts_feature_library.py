import math
import polars as pl
import functime.feature_extraction.tsfresh as f
from polars.type_aliases import ClosedInterval

@pl.api.register_expr_namespace("ts")
class FeatureLibrary:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def absolute_energy(self) -> pl.Expr:
        '''
        Compute the absolute energy of a time series.

        Returns
        -------
        An expression of the output
        '''
        return f.absolute_energy(self._expr)

    def absolute_maximum(self) -> pl.Expr:
        """
        Compute the absolute maximum of a time series.

        Returns
        -------
        An expression of the output
        """
        return f.absolute_maximum(self._expr)

    def absolute_sum_of_changes(self) -> pl.Expr:
        """
        Compute the absolute sum of changes of a time series.

        Returns
        -------
        An expression of the output
        """
        return f.absolute_sum_of_changes(self._expr)

    def autocorrelation(self, n_lags: int) -> pl.Expr:
        """
        Calculate the autocorrelation for a specified lag. The autocorrelation measures the linear 
        dependence between a time-series and a lagged version of itself.

        Parameters
        ----------
        n_lags : int
            The lag at which to calculate the autocorrelation. Must be a non-negative integer.

        Returns
        -------
        An expression of the output
        """
        return f.autocorrelation(self._expr, n_lags)

    def root_mean_square(self)-> pl.Expr:
        """
        Calculate the root mean square.

        Returns
        -------
        An expression of the output
        """
        return f.root_mean_square(self._expr)

    def benford_correlation(self) -> pl.Expr:
        """
        Returns the correlation between the first digit distribution of the input time series and
        the Newcomb-Benford's Law distribution.

        Returns
        -------
        An expression of the output
        """
        return f.benford_correlation(self._expr)

    def benford_correlation2(self) -> pl.Expr:
        """
        Returns the correlation between the first digit distribution of the input time series and
        the Newcomb-Benford's Law distribution. This version may be numerically unstable due to float
        point precision issue, but is faster for bigger data.

        Returns
        -------
        An expression of the output
        """
        return f.benford_correlation2(self._expr)

    def binned_entropy(self, bin_count: int = 10) -> pl.Expr:
        """
        Calculates the entropy of a binned histogram for a given time series.

        Parameters
        ----------
        bin_count : int, optional
            The number of bins to use in the histogram. Default is 10.

        Returns
        -------
        An expression of the output
        """
        return f.binned_entropy(self._expr, bin_count)

    def c3(self, n_lags: int) -> pl.Expr:
        """
        Measure of non-linearity in the time series using c3 statistics.

        Parameters
        ----------
        n_lags : int
            The lag that should be used in the calculation of the feature.

        Returns
        -------
        An expression of the output
        """
        return f.c3(self._expr, n_lags)

    def change_quantiles(
        self,
        q_low: float,
        q_high: float,
        is_abs: bool = True
    ) -> pl.Expr:
        """
        First fixes a corridor given by the quantiles ql and qh of the distribution of x.
        Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

        Parameters
        ----------
        q_low : float
            The lower quantile of the corridor. Must be less than `q_high`.
        q_high : float
            The upper quantile of the corridor. Must be greater than `q_low`.
        is_abs : bool
            If True, takes absolute difference.

        Returns
        -------
        An expression of the output
        """
        return f.change_quantiles(self._expr, q_low, q_high, is_abs)

    def cid_ce(self, normalize: bool = False) -> pl.Expr:
        """
        Computes estimate of time-series complexity[^1].

        A more complex time series has more peaks and valleys.
        This feature is calculated by:

        Parameters
        ----------
        normalize : bool, optional
            If True, z-normalizes the time-series before computing the feature.
            Default is False.

        Returns
        -------
        An expression of the output
        """
        return f.cid_ce(self._expr, normalize)

    def count_above(self, threshold: float = 0.0) -> pl.Expr:
        """
        Calculate the percentage of values above or equal to a threshold.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return f.count_above(self._expr, threshold)

    def count_above_mean(self) -> pl.Expr:
        """
        Count the number of values that are above the mean.

        Parameters
        ----------
        x : pl.Expr | pl.Series
            Input time-series.

        Returns
        -------
        An expression of the output
        """
        return f.count_above_mean(self._expr)

    def count_below(self, threshold: float = 0.0) -> pl.Expr:
        """
        Calculate the percentage of values below or equal to a threshold.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return f.count_below(self._expr, threshold)


    def count_below_mean(self) -> pl.Expr:
        """
        Count the number of values that are below the mean.

        Returns
        -------
        An expression of the output
        """
        return f.count_below_mean(self._expr)

    def energy_ratios(self, n_chunks: int = 10) -> pl.Expr:
        """
        Calculates sum of squares over the whole series for `n_chunks` equally segmented parts of the time-series.
        All ratios for all chunks will be returned at once.

        Parameters
        ----------
        n_chunks : int, optional
            The number of equally segmented parts to divide the time-series into. Default is 10.

        Returns
        -------
        An expression of the output
        """
        return f.energy_ratios(self._expr, n_chunks)

    def first_location_of_maximum(self) -> pl.Expr:
        """
        Returns the first location of the maximum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return f.first_location_of_maximum(self._expr)

    def first_location_of_minimum(self) -> pl.Expr:
        """
        Returns the first location of the minimum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return f.first_location_of_minimum(self._expr)

    def has_duplicate(self) -> pl.Expr:
        """
        Check if the time-series contains any duplicate values.

        Returns
        -------
        An expression of the output
        """
        return f.has_duplicate(self._expr)

    def has_duplicate_max(self) -> pl.Expr:
        """
        Check if the time-series contains any duplicate values equal to its maximum value.

        Returns
        -------
        An expression of the output
        """
        return f.has_duplicate_max(self._expr)

    def has_duplicate_min(self) -> pl.Expr:
        """
        Check if the time-series contains duplicate values equal to its minimum value.

        Returns
        -------
        An expression of the output
        """
        return f.has_duplicate_min(self._expr)

    def index_mass_quantile(self, q: float) -> pl.Expr:
        """
        Calculates the relative index i of time series x where q% of the mass of x lies left of i.
        For example for q = 50% this feature calculator will return the mass center of the time series.

        Parameters
        ----------
        q : float
            The quantile.

        Returns
        -------
        An expression of the output
        """
        return f.index_mass_quantile(self._expr)

    def large_standard_deviation(self, ratio: float = 0.25) -> pl.Expr:
        """
        Checks if the time-series has a large standard deviation: `std(x) > r * (max(X)-min(X))`.

        As a heuristic, the standard deviation should be a forth of the range of the values.

        Parameters
        ----------
        ratio : float
            The ratio of the interval to compare with.

        Returns
        -------
        An expression of the output
        """
        return f.large_standard_deviation(self._expr, ratio)

    def last_location_of_maximum(self) -> pl.Expr:
        """
        Returns the last location of the maximum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return f.last_location_of_maximum(self._expr)

    def last_location_of_minimum(self) -> pl.Expr:
        """
        Returns the last location of the minimum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return f.last_location_of_minimum(self._expr)

    def linear_trend(self) -> pl.Expr:
        """
        Compute the slope, intercept, and RSS of the linear trend.

        Returns
        -------
        An expression of the output
        """
        return f.linear_trend(self._expr)

    def longest_streak_above_mean(self) -> pl.Expr:
        """
        Returns the length of the longest consecutive subsequence in x that is greater than the mean of x.

        Returns
        -------
        An expression of the output
        """
        return f.longest_streak_above_mean(self._expr)

    def longest_streak_below_mean(self) -> pl.Expr:
        """
        Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x.

        Returns
        -------
        An expression of the output
        """
        return f.longest_streak_below_mean(self._expr)
    
    def longest_streak_above(self, threshold: float) -> pl.Expr:
        """
        Returns the longest streak of changes >= threshold of the time series. A change
        is counted when (x_t+1 - x_t) >= threshold. Note that the streaks here
        are about the changes for consecutive values in the time series, not the individual values.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return f.longest_streak_above(self._expr, threshold)

    def longest_streak_below(self, threshold: float) -> pl.Expr:
        """
        Returns the longest streak of changes <= threshold of the time series. A change
        is counted when (x_t+1 - x_t) <= threshold. Note that the streaks here
        are about the changes for consecutive values in the time series, not the individual values.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return f.longest_streak_below(self._expr, threshold)

    def mean_abs_change(self) -> pl.Expr:
        """
        Compute mean absolute change.

        Returns
        -------
        An expression of the output
        """
        return f.mean_abs_change(self._expr)
    
    def max_abs_change(self)-> pl.Expr:
        """
        Compute the maximum absolute change from X_t to X_t+1.

        Returns
        -------
        An expression of the output
        """
        return f.max_abs_change(self._expr)

    def mean_change(self) -> pl.Expr:
        """
        Compute mean change.

        Returns
        -------
        An expression of the output
        """
        return f.mean_change(self._expr)

    def mean_n_absolute_max(self, n_maxima: int) -> pl.Expr:
        """
        Calculates the arithmetic mean of the n absolute maximum values of the time series.

        Parameters
        ----------
        n_maxima : int
            The number of maxima to consider.

        Returns
        -------
        An expression of the output
        """
        return f.mean_n_absolute_max(self._expr, n_maxima)

    def mean_second_derivative_central(self) -> pl.Expr:
        """
        Returns the mean value of a central approximation of the second derivative.

        Returns
        -------
        An expression of the output
        """
        return f.mean_second_derivative_central(self._expr)

    def number_crossings(self, crossing_value: float = 0.0) -> pl.Expr:
        """
        Calculates the number of crossings of x on m, where m is the crossing value.

        A crossing is defined as two sequential values where the first value is lower than m and the next is greater, 
        or vice-versa. If you set m to zero, you will get the number of zero crossings.

        Parameters
        ----------
        crossing_value : float
            The crossing value. Defaults to 0.0.

        Returns
        -------
        An expression of the output
        """
        return f.number_crossings(self._expr, crossing_value)

    def percent_reocurring_points(self) -> pl.Expr:
        """
        Returns the percentage of non-unique data points in the time series. Non-unique data points are those that occur
        more than once in the time series.

        The percentage is calculated as follows:

            # of data points occurring more than once / # of all data points

        This means the ratio is normalized to the number of data points in the time series, in contrast to the
        `percent_reoccuring_values` function.

        Returns
        -------
        An expression of the output
        """
        return f.percent_reocurring_points(self._expr)

    def percent_reoccuring_values(self) -> pl.Expr:
        """
        Returns the percentage of values that are present in the time series more than once.

        The percentage is calculated as follows:

            len(different values occurring more than once) / len(different values)

        This means the percentage is normalized to the number of unique values in the time series, in contrast to the
        `percent_reocurring_points` function.

        Returns
        -------
        An expression of the output
        """
        return f.percent_reoccuring_values(self._expr)

    def number_peaks(self, support: int) -> pl.Expr:
        """
        Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
        subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

        Hence in the sequence

        x = [3, 0, 0, 4, 0, 0, 13]

        4 is a peak of support 1 and 2 because in the subsequences

        [0, 4, 0]
        [0, 0, 4, 0, 0]

        4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
        and its bigger than 4.

        Parameters
        ----------
        support : int
            Support of the peak

        Returns
        -------
        An expression of the output
        """
        return f.number_peaks(self._expr, support)

    def permutation_entropy(
        self,
        tau: int = 1,
        n_dims: int = 3,
        base: float = math.e,
    ) -> pl.Expr:
        """
        Computes permutation entropy.

        Parameters
        ----------
        tau : int
            The embedding time delay which controls the number of time periods between elements
            of each of the new column vectors.
        n_dims : int, > 1
            The embedding dimension which controls the length of each of the new column vectors
        base : float
            The base for log in the entropy computation

        Returns
        -------
        An expression of the output
        """
        return f.permutation_entropy(self._expr, tau, n_dims, base)

    def range_count(
        self,
        lower: float,
        upper: float,
        closed: ClosedInterval = "left"
    ) -> pl.Expr:
        """
        Computes values of input expression that is between lower (inclusive) and upper (exclusive).

        Parameters
        ----------
        lower : float
            The lower bound, inclusive
        upper : float
            The upper bound, exclusive
        closed : ClosedInterval
            Whether or not the boundaries should be included/excluded

        Returns
        -------
        An expression of the output
        """
        return f.range_count(self._expr, lower, upper, closed)

    def ratio_beyond_r_sigma(self, ratio: float = 0.25) -> pl.Expr:
        """
        Returns the ratio of values in the series that is beyond r*std from mean on both sides.

        Parameters
        ----------
        ratio : float
            The scaling factor for std

        Returns
        -------
        An expression of the output
        """
        return f.ratio_beyond_r_sigma(self._expr, ratio)

    # Originally named: `sum_of_reoccurring_data_points`
    def sum_reocurring_points(self) -> pl.Expr:
        """
        Returns the sum of all data points that are present in the time series more than once.

        For example, `sum_reocurring_points(pl.Series([2, 2, 2, 2, 1]))` returns 8, as 2 is a reoccurring value, so all 2's
        are summed up.

        This is in contrast to the `sum_reocurring_values` function, where each reoccuring value is only counted once.

        Returns
        -------
        An expression of the output
        """
        return f.sum_reocurring_points(self._expr)

    # Originally named: `sum_of_reoccurring_values`
    def sum_reocurring_values(self) -> pl.Expr:
        """
        Returns the sum of all values that are present in the time series more than once.

        For example, `sum_reocurring_values(pl.Series([2, 2, 2, 2, 1]))` returns 2, as 2 is a reoccurring value, so it is
        summed up with all other reoccuring values (there is none), so the result is 2.

        This is in contrast to the `sum_reocurring_points` function, where each reoccuring value is only counted as often
        as it is present in the data.

        Returns
        -------
        An expression of the output
        """
        return f.sum_reocurring_values(self._expr)

    def symmetry_looking(self, ratio: float = 0.25) -> pl.Expr:
        """
        Check if the distribution of x looks symmetric.

        A distribution is considered symmetric if: `| mean(X)-median(X) | < ratio * (max(X)-min(X))`

        Parameters
        ----------
        ratio : float
            Multiplier on distance between max and min.

        Returns
        -------
        An expression of the output
        """
        return f.symmetry_looking(self._expr, ratio)


    def time_reversal_asymmetry_statistic(self, n_lags: int) -> pl.Expr:
        """
        Returns the time reversal asymmetry statistic.

        Parameters
        ----------
        n_lags : int
            The lag that should be used in the calculation of the feature.

        Returns
        -------
        An expression of the output
        """
        return f.time_reversal_asymmetry_statistic(self._expr, n_lags)

    def variation_coefficient(self) -> pl.Expr:
        """
        Calculate the coefficient of variation (CV).

        Returns
        -------
        An expression of the output
        """
        return f.variation_coefficient(self._expr)

    def var_gt_std(self, ddof: int = 1) -> pl.Expr:
        """
        Is the variance >= std? In other words, is var >= 1?

        Parameters
        ----------
        ddof : int
            Delta Degrees of Freedom used when computing var/std.

        Returns
        -------
        An expression of the output
        """
        return f.var_gt_std(self._expr, ddof)

    def harmonic_mean(self) -> pl.Expr:
        """
        Returns the harmonic mean of the expression

        Returns
        -------
        An expression of the output
        """
        return f.harmonic_mean(self._expr)
    
    def range_over_mean(self)-> pl.Expr:
        """
        Returns the range (max - min) over mean of the time series.

        Returns
        -------
        An expression of the output
        """
        return f.range_over_mean(self._expr)

    def range_change(self, percentage: bool = True)-> pl.Expr:
        """
        Returns the range (max - min) over mean of the time series.
        
        Parameters
        ----------
        percentage : bool
            Compute the percentage if set to True

        Returns
        -------
        An expression of the output
        """
        return f.range_change(self._expr, percentage)
    
    def streak_length_stats(self, above: bool, threshold: float)-> pl.Expr:
        """
        Returns some statistics of the length of the streaks of the time series. Note that the streaks here
        are about the changes for consecutive values in the time series, not the individual values.

        The statistics include: min length, max length, average length, std of length,
        10-percentile length, median length, 90-percentile length, and mode of the length. If input is Series,
        a dictionary will be returned. If input is an expression, the expression will evaluate to a struct
        with the fields ordered by the statistics.

        Parameters
        ----------
        above: bool
            Above (>=) or below (<=) the given threshold
        threshold
            The threshold for the change (x_t+1 - x_t) to be counted

        Returns
        -------
        An expression of the output
        """
        return f.streak_length_stats(self._expr, above, threshold)

    def longest_winning_streak(self)-> pl.Expr:
        """
        Returns the longest winning streak of the time series. A win is counted when
        (x_t+1 - x_t) >= 0

        Returns
        -------
        An expression of the output
        """
        return f.longest_winning_streak(self._expr)

    def longest_losing_streak(self)-> pl.Expr:
        """
        Returns the longest losing streak of the time series. A loss is counted when
        (x_t+1 - x_t) <= 0

        Returns
        -------
        An expression of the output
        """
        return f.longest_losing_streak(self._expr)