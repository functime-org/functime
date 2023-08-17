from functools import partial

import polars as pl
import pytest
import statsmodels.api as sm
from polars.testing import assert_frame_equal
from statsmodels.tsa.stattools import acf as sm_acf

from functime.cross_validation import train_test_split
from functime.evaluation import (
    acf,
    ljung_box_test,
    rank_forecasts,
    rank_fva,
    rank_residuals,
)
from functime.forecasting import linear_model, snaive
from functime.preprocessing import scale

MAX_LAGS = 24


@pytest.fixture
def commodities_dataset():
    y = pl.read_parquet("data/commodities.parquet")
    y_train, y_test = train_test_split(test_size=6, eager=True)(y)
    return y_train, y_test


@pytest.fixture
def commodities_forecast(commodities_dataset):
    y_train, _ = commodities_dataset
    y_pred_bench = snaive(freq="1mo", sp=12)(y=y_train, fh=12)
    y_pred = linear_model(freq="1mo", lags=12, target_transform=scale())(
        y=y_train, fh=12
    )
    y_preds, y_resids = linear_model(
        freq="1mo", lags=MAX_LAGS, target_transform=scale()
    ).backtest(y=y_train, test_size=12, step_size=12, n_splits=5)
    return y_pred_bench, y_pred, y_preds, y_resids


def test_acf(commodities_dataset):
    y_train, _ = commodities_dataset
    entity_col, _, target_col = y_train.columns
    # Result
    acf_result = acf(y_train, max_lags=MAX_LAGS)
    # Expected
    sm_acf_values = (
        y_train.groupby(entity_col, maintain_order=True)
        .agg(
            pl.col(target_col)
            .apply(
                lambda s: sm_acf(s, nlags=MAX_LAGS, adjusted=True, alpha=0.05)[
                    0
                ].tolist()
            )
            .alias("acf"),
            pl.col(target_col)
            .apply(
                lambda s: sm_acf(s, nlags=MAX_LAGS, adjusted=True, alpha=0.05)[
                    1
                ].tolist()
            )
            .alias("confint")
            .flatten(),
        )
        .select(
            [
                entity_col,
                pl.col("acf").cast(pl.List(pl.Float32)),
                pl.concat_list(
                    [
                        pl.col("confint").list.get(i)
                        for i in range(0, MAX_LAGS * 2 + 2, 2)
                    ]
                ).alias("confint_lower"),
                pl.concat_list(
                    [
                        pl.col("confint").list.get(i + 1)
                        for i in range(0, MAX_LAGS * 2 + 2, 2)
                    ]
                ).alias("confint_upper"),
            ]
        )
    )
    print(acf_result.sort(entity_col))
    print(sm_acf_values.sort(entity_col))
    assert_frame_equal(
        acf_result.sort(entity_col),
        sm_acf_values.sort(entity_col),
    )


def test_ljung_box(commodities_dataset):
    y_train, _ = commodities_dataset
    entity_col, _, target_col = y_train.columns
    # Result
    acf_result = ljung_box_test(y_train, max_lags=MAX_LAGS)
    # Expected
    sm_ljungbox = partial(sm.stats.acorr_ljungbox, lags=MAX_LAGS)
    sm_lb_values = y_train.groupby(entity_col, maintain_order=True).agg(
        pl.col(target_col)
        .apply(lambda s: sm_ljungbox(s).loc[:, "lb_stat"].to_numpy().tolist())
        .alias("qstats"),
    )
    print(acf_result.sort(entity_col))
    print(sm_lb_values.sort(entity_col))
    assert_frame_equal(
        acf_result.sort(entity_col),
        sm_lb_values.sort(entity_col),
    )


@pytest.mark.parametrize("sort_by", ["mean", "median", "std", "cv", "smape"])
def test_rank_forecasts(commodities_forecast, sort_by):
    y_test, _, y_pred, _, _ = commodities_forecast
    ranks = rank_forecasts(y_pred=y_pred, y_true=y_test, sort_by=sort_by)
    print(ranks)
    raise ValueError


@pytest.mark.parametrize("sort_by", ["mean", "median", "std", "cv", "smape"])
def test_rank_backtests(commodities_forecast, sort_by):
    y_test, _, _, y_preds, _ = commodities_forecast
    ranks = rank_forecasts(y_pred=y_preds, y_true=y_test, sort_by=sort_by)
    print(ranks)
    raise ValueError


@pytest.mark.parametrize("sort_by", ["bias", "abs_bias", "normality", "autocorr"])
def test_rank_residuals(commodities_forecast, sort_by):
    y_test, _, _, y_preds, _ = commodities_forecast
    ranks = rank_residuals(y_pred=y_preds, y_true=y_test, sort_by=sort_by)
    print(ranks)
    raise ValueError


def test_rank_fva(commodities_forecast):
    y_test, y_pred_bench, y_pred, _, _ = commodities_forecast
    ranks = rank_fva(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_bench=y_pred_bench,
    )
    print(ranks)
    raise ValueError
