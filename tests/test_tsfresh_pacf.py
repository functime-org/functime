import polars as pl
import numpy as np
from feature_extraction.tsfresh_pacf import partial_autocorrelation
from statsmodels.tsa.stattools import pacf

def test_partial_autocorrelation():
    ts = np.random.rand(1000)
    x = pl.Series(ts)
    lags = [10, 50, 100, 450]
    assert all([abs(partial_autocorrelation(x, lag) - pacf(ts, nlags=lag, method='ld')[lag]) < 0.05 for lag in lags])
    print('partial_autocorrelation test passed')