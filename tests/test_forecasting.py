import cloudpickle
import numpy as np
import polars as pl
import pytest
from sklearnex import patch_sklearn

from functime.forecasting import (  # ann,
    auto_elastic_net,
    auto_lightgbm,
    catboost,
    censored_model,
    elastic_net,
    elite,
    flaml_lightgbm,
    lightgbm,
    linear_model,
    xgboost,
    zero_inflated_model,
)
from functime.metrics import rmsse, smape
from functime.preprocessing import scale

patch_sklearn()


DEFAULT_LAGS = 12
DIRECT_KWARGS = {"max_horizons": 28, "strategy": "direct"}
ENSEMBLE_KWARGS = {"max_horizons": 28, "strategy": "ensemble"}


# fmt: off
FORECASTERS_TO_TEST = [
    # ("ann", lambda freq: ann(lags=DEFAULT_LAGS, freq=freq)),
    # ("direct__ann", lambda freq: ann(lags=DEFAULT_LAGS, freq=freq)),
    # ("ensemble__ann", lambda freq: ann(lags=DEFAULT_LAGS, freq=freq)),
    ("linear", lambda freq: linear_model(lags=DEFAULT_LAGS, freq=freq, target_transform=scale())),
    ("direct__linear", lambda freq: linear_model(lags=DEFAULT_LAGS, freq=freq, target_transform=scale(), **DIRECT_KWARGS)),
    ("ensemble__linear", lambda freq: linear_model(lags=DEFAULT_LAGS, freq=freq, target_transform=scale(), **DIRECT_KWARGS)),
    ("catboost", lambda freq: catboost(lags=DEFAULT_LAGS, freq=freq, iterations=10)),
    ("xgboost", lambda freq: xgboost(lags=DEFAULT_LAGS, freq=freq, num_boost_round=10)),
    ("lgbm", lambda freq: lightgbm(lags=DEFAULT_LAGS, freq=freq, num_iterations=10)),
    ("flaml_lgbm", lambda freq: flaml_lightgbm(lags=DEFAULT_LAGS, freq=freq, custom_hp={"lgbm": {"num_iterations": {"domain": 10}}})),
    ("direct__lgbm", lambda freq: lightgbm(lags=DEFAULT_LAGS, freq=freq, num_iterations=10, **DIRECT_KWARGS)),
    ("ensemble__lgbm", lambda freq: lightgbm(lags=DEFAULT_LAGS, freq=freq, num_iterations=10, **DIRECT_KWARGS)),
]
# fmt: on


@pytest.fixture(params=FORECASTERS_TO_TEST, ids=lambda model: model[0])
def forecaster(request):
    return request.param[1]


@pytest.fixture(
    params=[
        (
            "auto_elastic_net",
            lambda freq: auto_elastic_net(
                test_size=1, freq=freq, min_lags=3, max_lags=6
            ),
        ),
        (
            "auto_lgbm",
            lambda freq: auto_lightgbm(test_size=1, freq=freq, min_lags=3, max_lags=6),
        ),
    ],
    ids=lambda model: model[0],
)
def auto_forecaster(request):
    return request.param[1]


def test_forecaster_cloudpickle():
    y = pl.DataFrame(
        {
            "entity": ["a"] * 12 + ["b"] * 12,
            "time": list(range(12)) + list(range(12)),
            "target": [i + np.random.normal() for i in range(24)],
        }
    )
    forecaster = elastic_net(freq="1i", lags=3).fit(y=y)
    y_pred = forecaster.predict(fh=3)
    pickle = cloudpickle.dumps(forecaster)
    unpickled_forecaster = cloudpickle.loads(pickle)
    assert (
        smape(y_pred, unpickled_forecaster.predict(fh=3)).get_column("smape").sum()
        < 0.001
    )


def test_auto_cloudpickle():
    y = pl.DataFrame(
        {
            "entity": ["a"] * 12 + ["b"] * 12,
            "time": list(range(12)) + list(range(12)),
            "target": [i + np.random.normal() for i in range(24)],
        }
    )
    forecaster = auto_elastic_net(freq="1i", min_lags=3, max_lags=6).fit(y=y)
    y_pred = forecaster.predict(fh=3)
    pickle = cloudpickle.dumps(forecaster)
    unpickled_forecaster = cloudpickle.loads(pickle)
    assert (
        smape(y_pred, unpickled_forecaster.predict(fh=3)).get_column("smape").sum()
        < 0.001
    )


def _check_missing_values(df_x: pl.LazyFrame, df_y: pl.LazyFrame, col: str):
    pl.testing.assert_series_equal(
        df_x.select(pl.col(col).unique()).collect().get_column(col).sort(),
        df_y.select(pl.col(col).unique()).collect().get_column(col).sort(),
    )


def _check_m4_score(y_test, y_pred, threshold: float = 0.3):
    score = smape(y_test, y_pred).get_column("smape").mean()
    assert score < threshold


def _check_m5_score(y_test, y_pred, y_train, threshold: float = 2.0):
    score = rmsse(y_test, y_pred, y_train=y_train).get_column("rmsse").mean()
    assert score < threshold


def test_forecaster_on_m4(forecaster, m4_dataset):
    """Run global models against the M4 competition datasets and check overall SMAPE
    (i.e. averaged across all time-series) is less than 0.3
    """
    y_train, y_test, fh, _ = m4_dataset
    freq = None  # I.e. test set starts at 1,2,3...,fh
    y_pred = forecaster(freq=freq)(y=y_train, fh=fh)
    entity_col = y_pred.columns[0]
    _check_missing_values(y_train.lazy(), y_pred.lazy(), entity_col)
    _check_m4_score(y_test, y_pred)


def test_auto_on_m4(auto_forecaster, m4_dataset):
    y_train, y_test, fh, _ = m4_dataset
    freq = None  # I.e. test set starts at 1,2,3...,fh
    y_pred = auto_forecaster(freq=freq)(y=y_train, fh=fh)
    entity_col = y_pred.columns[0]
    _check_missing_values(y_train.lazy(), y_pred.lazy(), entity_col)
    _check_m4_score(y_test, y_pred)


@pytest.mark.multivariate
def test_forecaster_on_m5(forecaster, m5_dataset):
    """Run global models against the M5 (Walmart) competition dataset and check
    overall RMSSE (i.e. averaged across all time-series) is less than 2.
    """
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    y_pred = forecaster(freq)(y=y_train, X=X_train, fh=fh, X_future=X_test)
    entity_col = y_pred.columns[0]
    _check_missing_values(y_train.lazy(), y_pred.lazy(), entity_col)
    _check_m5_score(y_test, y_pred, y_train)


@pytest.mark.multivariate
def test_auto_on_m5(auto_forecaster, m5_dataset):
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    y_pred = auto_forecaster(freq=freq)(y=y_train, X=X_train, fh=fh, X_future=X_test)
    entity_col = y_pred.columns[0]
    _check_missing_values(y_train.lazy(), y_pred.lazy(), entity_col)
    _check_m5_score(y_test, y_pred, y_train)


def simple_regress(X: np.ndarray, y: np.ndarray):
    import sklearn
    from sklearn.linear_model import LinearRegression

    with sklearn.config_context(assume_finite=False):
        regressor = LinearRegression()
        regressor.fit(X=X, y=y)
    return regressor


def simple_classify(X: np.ndarray, y: np.ndarray):
    import sklearn
    from sklearn.linear_model import LogisticRegression

    with sklearn.config_context(assume_finite=False):
        regressor = LogisticRegression()
        regressor.fit(X=X, y=y)
    return regressor


@pytest.mark.multivariate
@pytest.mark.parametrize("threshold", [5, 10])
def test_censored_model_on_m5(threshold, m5_dataset):
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    idx_cols = y_train.columns[:2]
    X_train = X_train.with_columns(
        pl.all().exclude(idx_cols).to_physical().cast(pl.Float32).fill_null("mean")
    )
    X_test = X_test.with_columns(
        pl.all().exclude(idx_cols).to_physical().cast(pl.Float32).fill_null("mean")
    )
    y_pred = censored_model(
        lags=3,
        threshold=threshold,
        freq=freq,
        regress=simple_regress,
        classify=simple_classify,
    )(y=y_train, X=X_train, fh=fh, X_future=X_test)
    # Check column names
    assert y_pred.columns == [*y_train.columns[:3], "threshold_proba"]
    # Check no missing time-series
    entity_col = y_pred.columns[0]
    _check_missing_values(y_train.lazy(), y_pred.lazy(), entity_col)
    # Check score
    score = (
        rmsse(y_test, y_pred.select(y_train.columns[:3]), y_train=y_train)
        .get_column("rmsse")
        .mean()
    )
    assert score < 2


@pytest.mark.multivariate
def test_zero_inflated_model_on_m5(m5_dataset):
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    y_pred = zero_inflated_model(
        lags=3, freq=freq, regress=simple_regress, classify=simple_classify
    )(y=y_train, X=X_train, fh=fh, X_future=X_test)
    # Check column names
    assert y_pred.columns == [*y_train.columns[:3], "threshold_proba"]
    # Check no missing time-series
    entity_col = y_pred.columns[0]
    _check_missing_values(y_train.lazy(), y_pred.lazy(), entity_col)
    # Check score
    score = (
        rmsse(y_test, y_pred.select(y_train.columns[:3]), y_train=y_train)
        .get_column("rmsse")
        .mean()
    )
    assert score < 2


def test_elite_on_m4(m4_dataset, m4_freq_to_sp, m4_freq_to_lags):
    y_train, y_test, fh, freq = m4_dataset
    y_pred = elite(freq=None, lags=m4_freq_to_lags[freq], sp=m4_freq_to_sp[freq])(
        y=y_train, fh=fh
    )
    _check_m4_score(y_test, y_pred)
