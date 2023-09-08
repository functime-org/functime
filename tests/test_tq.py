import polars as pl
import numpy as np
import pytest 
import tsfresh.feature_extraction.feature_calculators as fc
import functime.feature_extraction.tsfresh as ft

COL = "price"

@pytest.mark.parametrize("execution_times", [5])
def test_sample_entropy(execution_times:int):
    test_df = pl.read_parquet("data/commodities.parquet")
    for _ in range(execution_times):
        df = test_df.sample(n=2000)
        pl_s = df[COL]
        numpy_s = pl_s.to_numpy()

        tsfresh = round(float(fc.sample_entropy(numpy_s)), 12)
        rewrite = round(float(ft.sample_entropy(pl_s, r = 0.2)), 12)

        assert tsfresh == rewrite
