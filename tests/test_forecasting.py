import logging

import cloudpickle
import numpy as np
import polars as pl
import pytest

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
    naive,
    xgboost,
    zero_inflated_model,
)
from functime.metrics import rmsse, smape, smape_original
from functime.preprocessing import detrend, diff, scale
from functime.seasonality import add_fourier_terms

DEFAULT_LAGS = 12
DIRECT_KWARGS = {"max_horizons": 28, "strategy": "direct"}
ENSEMBLE_KWARGS = {"max_horizons": 28, "strategy": "ensemble"}


# fmt: off
FORECASTERS_TO_TEST = [
    # ("ann", lambda freq: ann(lags=DEFAULT_LAGS, freq=freq)),
    # ("direct__ann", lambda freq: ann(lags=DEFAULT_LAGS, freq=freq, **DIRECT_KWARGS)),
    # ("ensemble__ann", lambda freq: ann(lags=DEFAULT_LAGS, freq=freq, **ENSEMBLE_KWARGS)),
    ("linear", lambda freq: linear_model(lags=DEFAULT_LAGS, freq=freq, target_transform=detrend(freq=freq))),
    ("direct__linear", lambda freq: linear_model(lags=DEFAULT_LAGS, freq=freq, target_transform=detrend(freq=freq), **DIRECT_KWARGS)),
    ("ensemble__linear", lambda freq: linear_model(lags=DEFAULT_LAGS, freq=freq, target_transform=detrend(freq=freq), **ENSEMBLE_KWARGS)),
    ("catboost", lambda freq: catboost(lags=DEFAULT_LAGS, freq=freq, iterations=5)),
    ("xgboost", lambda freq: xgboost(lags=DEFAULT_LAGS, freq=freq, num_boost_round=5)),
    ("lgbm", lambda freq: lightgbm(lags=DEFAULT_LAGS, freq=freq, num_iterations=5)),
    ("flaml_lgbm", lambda freq: flaml_lightgbm(lags=DEFAULT_LAGS, freq=freq, custom_hp={"lgbm": {"num_iterations": {"domain": 5}}})),
    ("direct__lgbm", lambda freq: lightgbm(lags=DEFAULT_LAGS, freq=freq, num_iterations=5, **DIRECT_KWARGS)),
    ("ensemble__lgbm", lambda freq: lightgbm(lags=DEFAULT_LAGS, freq=freq, num_iterations=5, **ENSEMBLE_KWARGS)),
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


# def _check_missing_values(df_x: pl.LazyFrame, df_y: pl.LazyFrame, col: str):
#     pl.testing.assert_series_equal(
#         df_x.select(pl.col(col).unique()).collect().get_column(col).sort(),
#         df_y.select(pl.col(col).unique()).collect().get_column(col).sort(),
#     )


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
    y_pred = forecaster(freq="1i")(y=y_train, fh=fh)
    # _check_missing_values(y_train.lazy(), y_pred.lazy(), y_pred.columns[0])
    _check_m4_score(y_test, y_pred)


def test_auto_on_m4(auto_forecaster, m4_dataset):
    y_train, y_test, fh, _ = m4_dataset
    y_pred = auto_forecaster(freq="1i")(y=y_train, fh=fh)
    # _check_missing_values(y_train.lazy(), y_pred.lazy(), y_pred.columns[0])
    _check_m4_score(y_test, y_pred)


@pytest.mark.multivariate
def test_forecaster_on_m5(forecaster, m5_dataset, benchmark):
    """Run global models against the M5 (Walmart) competition dataset and check
    overall RMSSE (i.e. averaged across all time-series) is less than 2.
    """
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    y_pred = benchmark(
        lambda: forecaster(freq)(y=y_train, X=X_train, fh=fh, X_future=X_test)
    )
    # entity_col = y_pred.columns[0]
    # _check_missing_values(y_train.lazy(), y_pred.lazy(), entity_col)
    _check_m5_score(y_test, y_pred, y_train)


@pytest.mark.multivariate
def test_auto_on_m5(auto_forecaster, m5_dataset):
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    y_pred = auto_forecaster(freq=freq)(y=y_train, X=X_train, fh=fh, X_future=X_test)
    # entity_col = y_pred.columns[0]
    # _check_missing_values(y_train.lazy(), y_pred.lazy(), entity_col)
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
    # # Check no missing time-series
    # entity_col = y_pred.columns[0]
    # _check_missing_values(y_train.lazy(), y_pred.lazy(), entity_col)
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
    # # Check no missing time-series
    # entity_col = y_pred.columns[0]
    # _check_missing_values(y_train.lazy(), y_pred.lazy(), entity_col)
    # Check score
    score = (
        rmsse(y_test, y_pred.select(y_train.columns[:3]), y_train=y_train)
        .get_column("rmsse")
        .mean()
    )
    assert score < 2


@pytest.mark.skip("WIP")
def test_elite_on_m4(m4_dataset, m4_freq_to_lags, m4_freq_to_sp):
    y_train, y_test, fh, freq = m4_dataset
    lags = m4_freq_to_lags[freq]
    sp = m4_freq_to_sp[freq]
    y_pred = elite(freq="1i", lags=lags, sp=sp, scoring=smape_original)(
        y=y_train, fh=fh
    )
    y_pred_naive = naive(freq="1i")(y=y_train, fh=fh)

    # Score
    elite_scores = smape_original(y_true=y_test, y_pred=y_pred)
    naive_scores = smape_original(y_true=y_test, y_pred=y_pred_naive)
    scores = elite_scores.join(naive_scores, suffix="_naive", on=y_train.columns[0])

    # Compare scores (forecast value add)
    fva = (
        scores.lazy()
        .with_columns(
            [
                (pl.col("smape_original_naive") - pl.col("smape_original")).alias(
                    "fva"
                ),
                ((pl.col("smape_original_naive") - pl.col("smape_original")) > 0).alias(
                    "is_value_add"
                ),
            ]
        )
        .collect(streaming=True)
    )

    logging.info(fva.filter(pl.col("fva") > 0).describe())
    logging.info(fva.filter(pl.col("fva") < 0).describe())
    logging.info(fva.filter(pl.col("fva") == 0).describe())

    elite_mean_score = elite_scores.get_column("smape_original").mean()
    naive_mean_score = naive_scores.get_column("smape_original").mean()
    assert elite_mean_score < naive_mean_score
    assert fva.get_column("is_value_add").sum() == len(fva)


def test_conformalize_non_crossing_m4(m4_dataset):
    y_train, _, fh, _ = m4_dataset
    entity_col, time_col, target_col = y_train.columns[:3]
    y_preds = linear_model(freq="1i", lags=12).conformalize(
        y=y_train, fh=fh, alphas=[0.1, 0.9], drop_short=True
    )
    y_pred_qnt_10 = (
        y_preds.sort([entity_col, time_col])
        .filter(pl.col("quantile") == 10)
        .get_column(target_col)
    )
    y_pred_qnt_90 = (
        y_preds.sort([entity_col, time_col])
        .filter(pl.col("quantile") == 90)
        .get_column(target_col)
    )
    np.testing.assert_array_less(y_pred_qnt_10.to_numpy(), y_pred_qnt_90.to_numpy())


@pytest.mark.skip("Memory leak")
def test_conformalize_non_crossing_m5(m5_dataset):
    y_train, X_train, _, X_test, fh, freq = m5_dataset
    entity_col, time_col, target_col = y_train.columns[:3]
    y_preds = linear_model(freq=freq, lags=12).conformalize(
        y=y_train, X=X_train, X_future=X_test, fh=fh, alphas=[0.1, 0.9], drop_short=True
    )
    y_pred_qnt_10 = (
        y_preds.sort([entity_col, time_col])
        .filter(pl.col("quantile") == 10)
        .get_column(target_col)
    )
    y_pred_qnt_90 = (
        y_preds.sort([entity_col, time_col])
        .filter(pl.col("quantile") == 90)
        .get_column(target_col)
    )
    np.testing.assert_array_less(y_pred_qnt_10.to_numpy(), y_pred_qnt_90.to_numpy())


@pytest.mark.parametrize(
    "target,feature",
    [
        (scale(), None),
        (diff(order=1, fill_strategy="backward"), None),
        ([diff(order=1, fill_strategy="backward"), scale()], None),
        (
            [
                detrend(method="linear", freq="1d"),
                diff(order=1, fill_strategy="backward"),
            ],
            None,
        ),
        (None, add_fourier_terms(sp=12, K=3)),
        (
            [diff(order=1, fill_strategy="backward"), scale()],
            add_fourier_terms(sp=3, K=3),
        ),
    ],
)
def test_chained_transforms(target, feature, m5_dataset):
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    forecaster = linear_model(
        freq=freq, lags=12, target_transform=target, feature_transform=feature
    )
    y_pred = forecaster(y=y_train, fh=fh, X=X_train, X_future=X_test)
    _check_m5_score(y_test, y_pred, y_train)
