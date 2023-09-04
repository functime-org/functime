import logging
from functools import partial
from typing import List, Mapping

import numpy as np
import polars as pl
import pytest
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearnex import patch_sklearn

from functime.forecasting import linear_model
from functime.metrics import mae, mase, mse, rmse, rmsse, smape
from functime.preprocessing import scale

patch_sklearn()


METRICS_TO_TEST = [smape, rmse, rmsse, mae, mase, mse]
TTEST_SIG_LEVEL = 0.20  # Two tailed


@pytest.fixture(params=[8, 16, 32, 64], ids=lambda x: f"lags_{x:02d}")
def lags_to_test(request):
    return request.param


@pytest.fixture(
    params=[("linear", lambda: LinearRegression(fit_intercept=True))],
    ids=lambda x: x[0],
)
def regressor(request):
    return request.param


@pytest.fixture(params=[("linear", linear_model)], ids=lambda x: x[0])
def forecaster(request):
    return request.param


def get_short_series(X: pl.LazyFrame, min_counts: int) -> pl.DataFrame:
    entity_col, time_col = X.columns[:2]
    short_ts = (
        X.groupby(entity_col)
        .agg(pl.col(time_col).count().alias("count"))
        .filter(pl.col("count") <= min_counts)
        .collect(streaming=True)
    )
    if len(short_ts) > 0:
        n_dropped = len(short_ts)
        n_series = X.select(entity_col).collect().n_unique()
        logging.info("Short %s / %s time series:\n%s", n_dropped, n_series, short_ts)
    return short_ts.get_column(entity_col)


@pytest.fixture
def m4_dataset_no_missing(m4_dataset, lags_to_test):
    """M4 dataset with short time series dropped."""
    y_train, y_test, fh, freq = m4_dataset
    entity_col = y_train.columns[0]
    series_to_drop = get_short_series(y_train, min_counts=lags_to_test)
    # Filter out short series
    y_train_new = y_train.filter(~pl.col(entity_col).is_in(series_to_drop))
    y_test_new = y_test.filter(~pl.col(entity_col).is_in(series_to_drop))
    return y_train_new, y_test_new, fh, freq, lags_to_test


@pytest.fixture
def m5_dataset_no_missing(m5_dataset, lags_to_test):
    """M5 dataset with short time series dropped."""
    y_train, X_train, y_test, X_test, fh, freq = m5_dataset
    entity_col = y_train.columns[0]
    series_to_drop = get_short_series(y_train, min_counts=lags_to_test)
    # Filter out short series
    y_train_new = y_train.filter(~pl.col(entity_col).is_in(series_to_drop))
    X_train_new = X_train.filter(~pl.col(entity_col).is_in(series_to_drop))
    y_test_new = y_test.filter(~pl.col(entity_col).is_in(series_to_drop))
    X_test_new = X_test.filter(~pl.col(entity_col).is_in(series_to_drop))
    return y_train_new, X_train_new, y_test_new, X_test_new, fh, freq, lags_to_test


@pytest.fixture
def pd_m4_dataset(m4_dataset_no_missing):
    y_train, y_test, fh, freq, lags = m4_dataset_no_missing
    entity_col, time_col = y_train.columns[:2]
    pd_y_train = y_train.collect().to_pandas()
    pd_y_test = y_test.collect().to_pandas()
    return pd_y_train, pd_y_test, fh, freq, lags, entity_col, time_col


@pytest.fixture
def pd_m5_dataset(m5_dataset_no_missing):
    y_train, X_train, y_test, X_test, fh, freq, lags = m5_dataset_no_missing
    entity_col, time_col = y_train.columns[:2]
    pd_y_train = y_train.collect().to_pandas()
    pd_y_test = y_test.collect().to_pandas()
    pd_X_train = X_train.collect().to_pandas()
    pd_X_test = X_test.collect().to_pandas()
    pd_X_y = pd_y_train.merge(pd_X_train, how="left", on=[entity_col, time_col])
    return (
        pd_X_y,
        pd_y_test,
        pd_X_test,
        fh,
        freq[-1].upper(),
        lags,
        entity_col,
        time_col,
    )


def summarize_scores(scores: Mapping[str, np.ndarray]) -> Mapping[str, float]:
    return {k: sum(arr) / len(arr) for k, arr in scores.items()}


def score_forecasts(
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
    y_train: pl.DataFrame,
) -> Mapping[str, List[float]]:
    """Return mapping of metric name to scores across time-series."""

    # Defensive sort and coerce time column
    entity_col, time_col = y_true.columns[:2]
    y_true = y_true.sort([entity_col, time_col])
    y_pred = pl.concat(
        [
            y_true.select([entity_col, time_col]),
            y_pred.sort([entity_col, time_col]).select(y_pred.columns[-1]),
        ],
        how="horizontal",
    )

    scores = {}
    for metric in METRICS_TO_TEST:
        metric_name = metric.__name__
        if metric_name in ["rmsse", "mase"]:
            metric = partial(metric, y_train=y_train)
        scores[metric_name] = (
            metric(y_true=y_true, y_pred=y_pred)
            .get_column(metric_name)
            .to_list()  # So that it's JSON serializable for pytest cache
        )
    return scores


# 6 mins timeout
@pytest.mark.benchmark
def test_mlforecast_on_m4(regressor, pd_m4_dataset, benchmark, request):
    from joblib import cpu_count
    from mlforecast import MLForecast
    from mlforecast.target_transforms import LocalStandardScaler

    y_train, y_test, fh, freq, lags, entity_col, time_col = pd_m4_dataset
    regressor_name, regressor_cls = regressor

    def fit_predict():
        forecaster = MLForecast(
            models=[regressor_cls()],
            lags=list(range(lags)),
            num_threads=cpu_count(),
            target_transforms=[LocalStandardScaler()],
        )
        forecaster.fit(
            y_train,
            id_col=entity_col,
            time_col=time_col,
            target_col=y_train.columns[-1],
        )
        y_pred = forecaster.predict(fh)
        return y_pred

    y_pred = benchmark(fit_predict)
    scores = score_forecasts(
        y_true=pl.DataFrame(y_test),
        y_pred=pl.DataFrame(y_pred),
        y_train=pl.DataFrame(y_train),
    )
    logging.info(
        "Baseline scores (freq=%s, lags=%s): %s", freq, lags, summarize_scores(scores)
    )

    cache_id = f"baseline_m4_{freq}_{lags}_{regressor_name}"
    request.config.cache.set(cache_id, scores)


@pytest.mark.benchmark
def test_mlforecast_on_m5(regressor, pd_m5_dataset, benchmark):
    from joblib import cpu_count
    from mlforecast import MLForecast
    from mlforecast.target_transforms import LocalStandardScaler

    X_y_train, _, X_test, fh, freq, lags, entity_col, time_col = pd_m5_dataset
    _, regressor_cls = regressor

    # NOTE: We create a sklearn pipeline with imputer because
    # lags > 3 raises input X contains NaN, infinity or a
    # value too large for dtype('float64').

    pipeline = Pipeline([("impute", SimpleImputer()), ("regressor", regressor_cls())])

    def fit_predict():
        forecaster = MLForecast(
            models=[pipeline],
            freq=freq,
            lags=list(range(lags)),
            num_threads=cpu_count(),
            target_transforms=[LocalStandardScaler()],
        )
        forecaster.fit(
            X_y_train.reset_index(),
            id_col=entity_col,
            time_col=time_col,
            target_col="quantity_sold",
            keep_last_n=lags,
        )
        y_pred = forecaster.predict(fh, dynamic_dfs=[X_test])
        return y_pred

    benchmark(fit_predict)


@pytest.mark.benchmark
def test_functime_on_m4(forecaster, m4_dataset_no_missing, benchmark, request):
    y_train, y_test, fh, freq, lags = m4_dataset_no_missing
    forecaster_name, forecaster_cls = forecaster
    y_pred = benchmark(
        lambda: forecaster_cls(lags=lags, freq="1i", target_transform=scale())(
            y=y_train, fh=fh
        )
    )

    # Score forceasts
    scores = score_forecasts(y_true=y_test.collect(), y_pred=y_pred, y_train=y_train)
    mlforecast_scores = request.config.cache.get(
        f"baseline_m4_{freq}_{lags}_{forecaster_name}", None
    )

    for metric_name, baseline_scores in mlforecast_scores.items():
        functime_scores = scores[metric_name]
        assert len(functime_scores) == len(baseline_scores)
        mean_functime_score = np.mean(functime_scores)
        mean_baseline_score = np.mean(baseline_scores)
        if mean_functime_score > mean_baseline_score:
            res = ttest_ind(a=functime_scores, b=baseline_scores)
            assert res.pvalue > TTEST_SIG_LEVEL


@pytest.mark.benchmark
def test_functime_on_m5(forecaster, m5_dataset_no_missing, benchmark):
    y_train, X_train, _, X_test, fh, freq, lags = m5_dataset_no_missing
    _, forecaster_cls = forecaster
    benchmark(
        lambda: forecaster_cls(lags=lags, freq=freq, target_transform=scale())(
            y=y_train, X=X_train, X_future=X_test, fh=fh
        )
    )


FREQ_TO_LONG_FH = {
    "1d": 365,
    "1w": 52,
    "1mo": 120,
    "3mo": 24,
    "1y": 10,
}


@pytest.fixture
def pd_m4_dataset_long_fh(pd_m4_dataset):
    y_train, y_test, _, freq, lags, entity_col, time_col = pd_m4_dataset
    return y_train, y_test, FREQ_TO_LONG_FH[freq], freq, lags, entity_col, time_col


@pytest.fixture
def m4_dataset_no_missing_long_fh(m4_dataset_no_missing):
    y_train, y_test, _, freq, lags = m4_dataset_no_missing
    return y_train, y_test, FREQ_TO_LONG_FH[freq], freq, lags


@pytest.mark.benchmark
def test_mlforecast_on_m4_long_fh(regressor, pd_m4_dataset_long_fh, benchmark):
    from joblib import cpu_count
    from mlforecast import MLForecast
    from mlforecast.target_transforms import LocalStandardScaler

    y_train, _, fh, _, lags, entity_col, time_col = pd_m4_dataset_long_fh
    _, regressor_cls = regressor

    def fit_predict():
        forecaster = MLForecast(
            models=[regressor_cls()],
            lags=list(range(lags)),
            num_threads=cpu_count(),
            target_transforms=[LocalStandardScaler()],
        )
        forecaster.fit(
            y_train,
            id_col=entity_col,
            time_col=time_col,
            target_col=y_train.columns[-1],
        )
        y_pred = forecaster.predict(fh)
        return y_pred

    benchmark(fit_predict)


@pytest.mark.benchmark
def test_functime_on_m4_long_fh(forecaster, m4_dataset_no_missing_long_fh, benchmark):
    y_train, _, fh, _, lags = m4_dataset_no_missing_long_fh
    _, forecaster_cls = forecaster
    benchmark(
        lambda: forecaster_cls(lags=lags, freq="1i", target_transform=scale())(
            y=y_train, fh=fh
        )
    )
