import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller, kpss


def check_stationarity(y: pl.DataFrame, alpha: float = 0.05):
    y_arr = y.get_column(y.columns[-1])
    _, adf_pval, _ = adfuller(y_arr, regression="n")
    _, kpss_pval, _ = kpss(y_arr, regression="ct")
    reject_adf_null = adf_pval < alpha
    # kpss test returns p-value within interval [0.01, 0.1]
    reject_kpss_null = kpss_pval == 0.01
    if not reject_adf_null and reject_kpss_null:
        # ADF: Under null, cannot reject null that series has unit root (non-stationary)
        # KPSS: Under null, reject null that series is trend stationary (non-stationary)
        res = False, None, adf_pval, kpss_pval
    elif reject_adf_null and not reject_kpss_null:
        # ADF: Under null, reject null that series has unit root (stationary)
        # KPSS: Under null, cannot reject null that series is trend stationary (stationary)
        res = True, None, adf_pval, kpss_pval
    elif reject_adf_null and reject_kpss_null:
        # ADF: Under null, reject null that series has unit root (stationary)
        # KPSS: Under null, reject null that series is trend stationary (non-stationary)
        res = False, "trend", adf_pval, kpss_pval
    else:
        # ADF: Under null, cannot reject null that series has unit root (non-stationary)
        # KPSS: Under null, cannot reject null that series is trend stationary (stationary)
        res = False, "diff", adf_pval, kpss_pval
    return res


def check_excess_risks(risks: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    entity_col = scores.columns[0]
    stat_col = risks.columns[-1]
    metric_col = scores.columns[-1]
    risks_arr = (
        risks.sort(entity_col).get_column(stat_col).to_numpy(zero_copy_only=True)
    )
    scores_arr = (
        scores.sort(entity_col).get_column(metric_col).to_numpy(zero_copy_only=True)
    )
    slope, intercept = np.polyfit(risks_arr, scores_arr, 1)
    res = (
        risks.with_columns((slope * pl.col(stat_col) + intercept).alias("trend"))
        .join(scores, on=entity_col)
        .with_columns((pl.col(metric_col) - pl.col("trend")).alias("distance"))
        .with_columns((pl.col("distance") > 0).alias("is_excess"))
        # Sort by highest "risk"
        .sort("distance", reverse=True)
        .select(
            [
                entity_col,
                metric_col,
                stat_col,
                "trend",
                "distance",
                "is_excess",
            ]
        )
    )
    return res
