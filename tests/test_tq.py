import polars as pl
import numpy as np
import pytest 
import tsfresh.feature_extraction.feature_calculators as fc
import functime.feature_extraction.tq_feature_extractions as ft

COL = "price"

@pytest.mark.parametrize("execution_times", [5])
def test_sample_entropy_against_tsfresh(execution_times:int):
    test_df = pl.read_parquet("data/commodities.parquet")
    for _ in range(execution_times):
        df = test_df.sample(n=2000)
        pl_s = df[COL]
        numpy_s = pl_s.to_numpy()

        tsfresh = round(float(fc.sample_entropy(numpy_s)), 12)
        rewrite = round(float(ft.sample_entropy(pl_s, r = 0.2)), 12)

        assert tsfresh == rewrite

def test_permutation_entropy_against_tsfresh():
    test_df = pl.read_parquet("data/commodities.parquet")

    numpy_s = test_df[COL].to_numpy()

    tsfresh1 = round(fc.permutation_entropy(numpy_s, tau=1, dimension=3), 12)
    rewrite1 = test_df.select(
        ft.permutation_entropy(pl.col(COL), tau=1, n_dims=3)
    ).item(0,0)
    rewrite1 = round(rewrite1, 12)
    assert tsfresh1 == rewrite1

    tsfresh2 = round(fc.permutation_entropy(numpy_s, tau=2, dimension=4), 12)
    rewrite2 = test_df.select(
        ft.permutation_entropy(pl.col(COL), tau=2, n_dims=4)
    ).item(0,0)
    rewrite2 = round(rewrite2, 12)
    assert tsfresh2 == rewrite2

def test_range_count_against_tsfresh():
    test_df = pl.read_parquet("data/commodities.parquet")

    numpy_s = test_df[COL].to_numpy()

    tsfresh1 = round(fc.range_count(numpy_s, min=500, max=1000), 12)
    rewrite1 = test_df.select(
        ft.range_count(pl.col(COL), lower=500, upper=1000)
    ).item(0,0)
    rewrite1 = round(rewrite1, 12)
    assert tsfresh1 == rewrite1

    tsfresh2 = round(fc.range_count(numpy_s, min=1200, max=1300), 12)
    rewrite2 = test_df.select(
        ft.range_count(pl.col(COL), lower=1200, upper=1300)
    ).item(0,0)
    rewrite2 = round(rewrite2, 12)
    assert tsfresh2 == rewrite2

@pytest.mark.parametrize("bounds", [(5,10), (1,2), (23,25)])
def test_range_count(bounds:tuple[int, int]):
    test_df = pl.DataFrame({
        "a": range(100)
    })
    lower = bounds[0]
    upper = bounds[1]
    ans = test_df.select(
        ft.range_count(pl.col("a"), lower=lower, upper=upper)
    ).item(0,0)
    assert ans == upper-lower
