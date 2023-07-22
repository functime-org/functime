import logging
from functools import partial
from typing import Mapping

import numpy as np
import polars as pl
import pytest
from lightgbm import LGBMRegressor
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearnex import patch_sklearn

from functime.forecasting import lightgbm, linear_model
from functime.metrics import mae, mase, mse, rmse, rmsse, smape
from functime.preprocessing import scale

patch_sklearn()


METRICS_TO_TEST = [smape, rmse, rmsse, mae, mase, mse]
TTEST_SIG_LEVEL = 0.10  # Two tailed


@pytest.fixture(
    params=[
        ("linear", lambda: LinearRegression(fit_intercept=True)),
        ("lgbm", lambda: LGBMRegressor(force_col_wise=True, tree_learner="serial")),
    ],
    ids=lambda x: x[0],
)
def regressor(request):
    return request.param


@pytest.fixture(
    params=[("linear", linear_model), ("lgbm", lightgbm)], ids=lambda x: x[0]
)
def forecaster(request):
    return request.param


@pytest.fixture(params=[3, 8, 16, 32, 64], ids=lambda x: f"lags({x})")
def lags_to_test(request):
    return request.param


def score_forecasts(
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
    y_train: pl.DataFrame,
) -> Mapping[str, np.ndarray]:
    """Return mapping of metric name to scores across time-series."""
    scores = {}
    for metric in METRICS_TO_TEST:
        metric_name = metric.__name__
        if metric_name in ["rmsse", "mase"]:
            metric = partial(metric, y_train=y_train)
        scores[metric_name] = (
            metric(y_true=y_true, y_pred=y_pred)
            .get_column(metric_name)
            .to_numpy(zero_copy_only=True)
        )
    return scores


# 6 mins timeout
@pytest.mark.benchmark
def test_mlforecast_on_m4(pd_m4_dataset, regressor, lags_to_test, benchmark, request):
    from joblib import cpu_count
    from mlforecast import MLForecast
    from mlforecast.target_transforms import LocalStandardScaler

    y_train, y_test, fh, freq, entity_col, time_col = pd_m4_dataset
    estimator_name, estimator = regressor

    def fit_predict():
        model = MLForecast(
            models=[estimator()],
            lags=list(range(lags_to_test)),
            num_threads=cpu_count(),
            target_transforms=[LocalStandardScaler()],
        )
        model.fit(
            y_train,
            id_col=entity_col,
            time_col=time_col,
            target_col=y_train.columns[-1],
        )
        y_pred = model.predict(fh)
        return y_pred

    y_pred = benchmark(fit_predict)
    scores = score_forecasts(
        y_true=pl.DataFrame(y_test),
        y_pred=pl.DataFrame(y_pred),
        y_train=pl.DataFrame(y_train),
    )
    cache_id = f"m4_{freq}_{lags_to_test}_{estimator_name}"
    logging.info(
        "mlforecast M4 (freq={freq}, lags={lags}) scores:\n{scores}",
        freq=freq,
        lags=lags_to_test,
        scores=scores,
    )
    request.config.cache.set(cache_id, scores)


@pytest.mark.benchmark
def test_mlforecast_on_m5(pd_m5_dataset, regressor, lags_to_test, benchmark, request):
    from joblib import cpu_count
    from mlforecast import MLForecast
    from mlforecast.target_transforms import LocalStandardScaler

    X_y_train, X_y_test, X_test, fh, entity_col, time_col = pd_m5_dataset
    estimator_name, estimator = regressor

    def fit_predict():
        model = MLForecast(
            models=[estimator],
            freq="D",
            lags=list(range(lags_to_test)),
            num_threads=cpu_count(),
            target_transforms=[LocalStandardScaler()],
        )
        model.fit(
            X_y_train.reset_index(),
            id_col=entity_col,
            time_col=time_col,
            target_col="quantity_sold",
            keep_last_n=lags_to_test,
        )
        y_pred = model.predict(fh, dynamic_dfs=[X_test])
        return y_pred

    y_pred = benchmark(fit_predict)
    scores = score_forecasts(
        y_true=pl.DataFrame(X_y_test.loc[:, entity_col, time_col, "quantity_sold"]),
        y_pred=pl.DataFrame(y_pred),
        y_train=pl.DataFrame(X_y_train.loc[:, entity_col, time_col, "quantity_sold"]),
    )
    cache_id = f"mlforecast_m5_{lags_to_test}_{estimator_name}_scores"
    request.config.cache.set(cache_id, scores)


@pytest.mark.benchmark
def test_functime_on_m4(m4_dataset, forecaster, lags_to_test, benchmark, request):
    y_train, y_test, fh, freq = m4_dataset
    model_name, model = forecaster
    y_pred = benchmark(
        lambda: model(lags=lags_to_test, freq=freq, target_transform=scale())(
            y=y_train, fh=fh
        )
    )

    # Score forceasts
    scores = score_forecasts(y_true=y_test, y_pred=y_pred, y_train=y_train)
    mlforecast_scores = request.config.cache.get(
        f"m4_{freq}_{lags_to_test}_{model_name}"
    )

    for metric_name, baseline_scores in mlforecast_scores.items():
        # Compare mean scores with t-test
        res = ttest_ind(a=scores[metric_name], b=baseline_scores)
        assert res.pvalue > TTEST_SIG_LEVEL


@pytest.mark.benchmark
def test_functime_on_m5(m5_dataset, forecaster, lags_to_test, benchmark, request):
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    model_name, model = forecaster
    y_pred = benchmark(
        lambda: model(lags=lags_to_test, freq=freq, target_transform=scale())(
            y=y_train, X_train=X_train, X_future=X_test, fh=fh
        )
    )

    # Check no missing time-series
    entity_col = y_pred.columns[0]
    assert pl.testing.assert_frame_equal(
        y_train.select(pl.col(entity_col).unique()).collect(),
        y_pred.select(pl.col(entity_col).unique()).collect(),
    )

    # Score forceasts
    scores = score_forecasts(y_true=y_test, y_pred=y_pred)

    mlforecast_scores = request.config.cache.get(
        f"mlforecast_m5_{lags_to_test}_{model_name}_scores"
    )
    for metric_name, baseline_scores in mlforecast_scores.items():
        # Compare mean scores with t-test
        res = ttest_ind(a=scores[metric_name], b=baseline_scores)
        assert res.pvalue > TTEST_SIG_LEVEL
