import subprocess
from dataclasses import dataclass
from typing import Optional

import polars as pl
import pytest
from joblib import Parallel, delayed

from functime.forecasting import LinearModel

STRATEGIES = ["recursive", "direct", "ensemble"]


@pytest.fixture(autouse=True)
def delete_deployed_models():
    yield
    subprocess.call("functime deploy remove --all")


@dataclass
class DatasetPath:
    y: str
    X: Optional[str] = None
    X_future: Optional[str] = None


@dataclass
class ForecastParams:
    lags: int
    min_lags: int
    max_lags: int
    fh: int
    freq: Optional[str] = None


@dataclass
class Dataset:
    y: pl.DataFrame
    params: ForecastParams
    X: Optional[pl.DataFrame] = None
    X_future: Optional[pl.DataFrame] = None


@pytest.fixture(
    params=[
        (
            "commodities",
            "data/commodities.parquet",
        ),
        (
            "m4_1d",
            DatasetPath("data/m4_1d_train.parquet"),
            ForecastParams(lags=30, min_lags=24, max_lags=30, fh=30),
        ),
        (
            "m4_1h",
            DatasetPath("data/m4_1h_train.parquet"),
            ForecastParams(lags=24, min_lags=20, max_lags=24, fh=24),
        ),
        (
            "m4_1mo",
            DatasetPath("data/m4_1mo_train.parquet"),
            ForecastParams(lags=12, min_lags=8, max_lags=12, fh=12),
        ),
        (
            "m4_1w",
            DatasetPath("data/m4_1w_train.parquet"),
            ForecastParams(lags=6, min_lags=4, max_lags=6, fh=6),
        ),
        (
            "m4_3mo",
            DatasetPath("data/m4_3mo_train.parquet"),
            ForecastParams(lags=4, min_lags=2, max_lags=4, fh=4),
        ),
        (
            "m5_sample",
            DatasetPath(
                "data/m5_y_train_sample.parquet",
                "data/m5_X_train_sample.parquet",
                "data/m5_X_test_sample.parquet",
            ),
            ForecastParams(lags=4, min_lags=2, max_lags=4, fh=4),
        ),
        # ("m5_full", dict(y="data/m5_y_train.parquet", X="data/m5_X_train.parquet"), dict(lags=4, min_lags=2, max_lags=4, fh=4)),
    ],
    ids=lambda params: params[0],
)
def dataset(request):
    _, dataset_path, params = request.param
    y = pl.read_parquet(dataset_path.y)
    X = pl.read_parquet(dataset_path.X) if dataset_path.X else None
    X_future = pl.read_parquet(dataset_path.X_future) if dataset_path.X_future else None
    dataset = Dataset(y=y, X=X, X_future=X_future, params=params)
    return dataset


def test_fit_predict(dataset):
    pass


def test_fit_then_predict(dataset):
    pass


def test_fit_predict_with_kwargs(dataset):
    pass


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategies():
    pass


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_auto_strategies():
    pass


def test_different_stub_same_model():
    model_a = LinearModel(
        lags=dataset.lags,
        freq=dataset.freq,
    )
    model_a.fit(y=dataset.y)
    model_b = LinearModel(
        lags=dataset.lags,
        freq=dataset.freq,
    )
    model_b.fit(y=dataset.y)
    assert model_a.stub_id != model_b.stub_id


def test_from_deployment():
    """Identical predictions for in-memory and from deployed forecasters."""
    forecaster = LinearModel(
        lags=dataset.lags,
        freq=dataset.freq,
    )
    forecaster.fit(y=dataset.y, X=dataset.X)


@pytest.mark.parametrize("n_iter", [2, 4, 16])
def test_sequential_inference(n_iter, fitted_forecaster):
    for _ in range(n_iter):
        fitted_forecaster.predict(fh=dataset.fh, X=dataset.X_future)


@pytest.mark.parametrize("n_iter", [2, 4, 16])
def test_parallel_inference(n_iter, fitted_forecaster):
    Parallel(n_jobs=-1)(
        delayed(fitted_forecaster.predict)(dataset.fh, dataset.X_future)
        for _ in range(n_iter)
    )


def test_different_model_from_deployed():
    """Raises HTTPException with status code 400"""
    pass


def test_usage_limits():
    pass
