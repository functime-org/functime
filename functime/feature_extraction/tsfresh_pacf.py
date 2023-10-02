import polars as pl
from scipy.linalg import lstsq
from feature_extraction.tsfresh import autocorrelation, TIME_SERIES_T, FLOAT_EXPR

def partial_autocorrelation(x: TIME_SERIES_T, lag: int) -> FLOAT_EXPR:
    """
    Calculates the partial autocorrelation (https://en.wikipedia.org/wiki/Partial_autocorrelation_function) of time series `x` at the given lag.
    Follows the algorithm described in https://timeseriesreasoning.com/contents/partial-auto-correlation/
    
    params:
    x: the time series to calculate the feature of
    lag: the lag to calculate the partial autocorrelation for
    
    returns:
    partial autocorrelation value at the given lag
    """
    # PACF limits lag length to 50% of sample size.
    if lag >= x.len() // 2:
        print('USER WARNING: lag is too large, setting to 50% of sample size')
        lag = x.len() // 2 - 1
    elif lag == 0:
        return 1.0
    elif lag == 1:
        return autocorrelation(x, 1)
        
    df = pl.DataFrame(x)
    name = df.columns[0]
    
    # Get various shifts of the series, up to lag
    df = df.with_columns(
        [pl.lit(1).alias('bias')]
        + 
        [
        pl.col(name).shift(i).alias(f"-{i}") for i in range(1, lag+1)
    ]).drop_nulls()
    M = df.to_numpy()
    
    # Calculate residuals of shift 0 and shift lag on the lags matrix
    fit_0 = lstsq(M[:, 1:-1], M[:, 0], overwrite_a=True, overwrite_b=True)[0]
    residuals_0 = M[:, 0] - M[:, 1:-1] @ fit_0
    fit_lag = lstsq(M[:, 1:-1], M[:, -1], overwrite_a=True, overwrite_b=True)[0]
    residuals_lag = M[:, -1] - M[:, 1:-1] @ fit_lag
    
    # Return the correlation between the residuals
    return df.with_columns([
        pl.Series(residuals_0).alias('res_0'),
        pl.Series(residuals_lag).alias('res_lag')
    ]).select([
        pl.corr('res_0', 'res_lag', method='pearson')
    ])[0,0]