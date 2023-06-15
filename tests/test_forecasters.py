import subprocess
from dataclasses import dataclass
from typing import Optional

import polars as pl
import pytest
from fastapi import HTTPException
from joblib import Parallel, delayed
from polars.testing import assert_frame_equal

from functime.forecasting import AutoLasso, AutoLinearModel, Lasso, LinearModel
from functime.metrics import smape


@pytest.fixture(params=["recursive", "direct", "ensemble"])
def strategy(request):
    return request.param


@pytest.fixture(autouse=True)
def delete_deployed_models():
    yield
    subprocess.call(["functime", "deploy", "remove", "--all"])


@pytest.fixture
def test_dataset():
    y = pl.read_parquet("data/tourism.parquet")
    freq = "1mo"
    return y, freq


@dataclass
class DatasetPath:
    y: str
    y_test: pl.DataFrame
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
    y_test: pl.DataFrame
    params: ForecastParams
    X: Optional[pl.DataFrame] = None
    X_future: Optional[pl.DataFrame] = None


@pytest.fixture(
    params=[
        (
            "m4_1d",
            DatasetPath("data/m4_1d_train.parquet", "data/m4_1d_test.parquet"),
            ForecastParams(lags=30, min_lags=24, max_lags=30, fh=30),
        ),
        (
            "m4_1h",
            DatasetPath("data/m4_1h_train.parquet", "data/m4_1h_test.parquet"),
            ForecastParams(lags=24, min_lags=20, max_lags=24, fh=24),
        ),
        (
            "m4_1mo",
            DatasetPath("data/m4_1mo_train.parquet", "data/m4_1mo_test.parquet"),
            ForecastParams(lags=12, min_lags=8, max_lags=12, fh=12),
        ),
        (
            "m4_1w",
            DatasetPath("data/m4_1w_train.parquet", "data/m4_1w_test.parquet"),
            ForecastParams(lags=6, min_lags=4, max_lags=6, fh=6),
        ),
        (
            "m4_3mo",
            DatasetPath("data/m4_3mo_train.parquet", "data/m4_3mo_test.parquet"),
            ForecastParams(lags=4, min_lags=2, max_lags=4, fh=4),
        ),
        (
            "m5_sample",
            DatasetPath(
                "data/m5_y_train_sample.parquet",
                "data/m5_y_test_sample.parquet",
                "data/m5_X_train_sample.parquet",
                "data/m5_X_test_sample.parquet",
            ),
            ForecastParams(lags=28, min_lags=24, max_lags=28, fh=28),
        ),
        # (
        #     "m5_full",
        #     DatasetPath(
        #         "data/m5_y_train.parquet",
        #         "data/m5_y_test.parquet",
        #         "data/m5_X_train.parquet",
        #         "data/m5_X_test.parquet",
        #     ),
        #     ForecastParams(lags=28, min_lags=24, max_lags=28, fh=28),
        # ),
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


@pytest.fixture(
    params=[
        ("linear", lambda dataset: LinearModel(lags=dataset.lags, freq=dataset.freq)),
        (
            "auto_linear",
            lambda dataset: AutoLinearModel(
                freq=dataset.freq, min_lags=dataset.min_lags, max_lags=dataset.max_lags
            ),
        ),
    ],
    ids=lambda params: params[0],
)
def forecaster(request):
    return request.param[1]


def test_fit_predict(dataset, forecaster):
    model = forecaster(dataset)
    y_pred = model.fit_predict(
        y=dataset.y, X=dataset.X, X_future=dataset.X_future, fh=dataset.fh
    )
    score = smape(y_pred=y_pred, y_true=dataset.y_test)
    assert score < 0.4


def test_fit_then_predict(dataset):
    model = forecaster(dataset)
    model.fit(y=dataset.y, X=dataset.X)
    y_pred = model.predict(fh=dataset.fh, X=dataset.X_future)
    score = smape(y_pred=y_pred, y_true=dataset.y_test)
    assert score < 0.4


def test_forecaster_with_kwargs(test_dataset):
    y, freq = test_dataset
    alpha = 0.001
    model = Lasso(freq=freq, lags=3, alpha=alpha, fit_intercept=False).fit(y=y)
    regressor_params = model.get_regressor_params()
    assert regressor_params["alpha"] == alpha
    assert regressor_params["fit_intercept"] is False


def test_auto_forecaster_with_kwargs(test_dataset):
    y, freq = test_dataset
    max_iter = 50
    model = AutoLasso(freq=freq, max_iter=max_iter).fit(y)
    regressor_params = model.get_regressor_params()
    assert regressor_params["max_iter"] == max_iter


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


def test_from_deployment(test_dataset):
    """Identical predictions for in-memory and from deployed forecasters."""
    y, freq = test_dataset
    model = LinearModel(
        freq=freq,
        lags=3,
    )
    model.fit(y=y)
    stub_id = model.stub_id
    # From deployed
    fitted_model = LinearModel.from_deployed(stub_id=stub_id)
    assert_frame_equal(model.predict(fh=3), fitted_model.predict(fh=3))


def test_different_model_from_deployed(test_dataset):
    """Raises HTTPException with status code 400"""
    y, freq = test_dataset
    forecaster = LinearModel(
        freq=freq,
        lags=3,
    )
    forecaster.fit(y=y)
    stub_id = forecaster.stub_id
    with pytest.raises(HTTPException):
        Lasso.from_deployed(stub_id=stub_id)


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


def test_fit_deploys_model():
    """Check estimator gets deployed in modal and redis"""
    pass


def test_fit_updates_mb_usage():
    """Check that fit updates usage by the correct mb amount
    The total mb amount should be the sum of bytes of y + X + X_future
    """
    pass


def test_predict_updates_forecasts_usage():
    """Check that predict updates the forecast usage by the number of entities in y"""
    pass


def test_maxed_out_data_usage():
    """Raises HTTPException: Payment required"""
    pass


def test_maxed_out_prediction_usage():
    """Raises HTTPException: Payment required"""
    pass
