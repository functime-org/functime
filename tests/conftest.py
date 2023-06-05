import numpy as np
import pandas as pd
import polars as pl
import pytest


@pytest.fixture(params=[250, 1000], ids=lambda x: f"n_periods({x})")
def n_periods(request):
    return request.param


@pytest.fixture(params=[50, 500], ids=lambda x: f"n_entities({x})")
def n_entities(request):
    return request.param


@pytest.fixture
def pd_X(n_periods, n_entities):
    """Return panel pd.DataFrame with sin, cos, and tan columns and time,
    entity multi-index. Used to benchmark polars vs pandas.
    """
    entity_idx = [f"x{i}" for i in range(n_entities)]
    time_idx = pd.date_range("2020-01-01", periods=n_periods, freq="1D", name="time")
    multi_row = [(entity, timestamp) for entity in entity_idx for timestamp in time_idx]
    idx = pd.MultiIndex.from_tuples(multi_row, names=["series_id", "time"])
    sin_x = np.sin(np.arange(0, n_entities * n_periods))
    cos_x = np.cos(np.arange(0, n_entities * n_periods))
    tan_x = np.tan(np.arange(0, n_entities * n_periods))
    X = pd.DataFrame({"open": sin_x, "close": cos_x, "volume": tan_x}, index=idx)
    return X.sort_index()


@pytest.fixture
def pd_y(pd_X):
    return pd_X.loc[:, "close"]


@pytest.fixture
def pl_y(pd_y):
    return pl.from_pandas(pd_y.reset_index()).lazy()
