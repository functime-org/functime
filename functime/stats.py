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
