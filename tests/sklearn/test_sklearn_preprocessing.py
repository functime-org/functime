from __future__ import annotations

import pickle
from datetime import date, timedelta

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from sklearn import config_context
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from functime.forecasting import NaiveForecaster, SeasonalNaiveForecaster
from functime.sklearn import (
    CalendarTransformer,
    DetrendTransformer,
    DiffTransformer,
    FourierTransformer,
    FractionalDiffTransformer,
    LagTransformer,
    PanelFeatureAdapter,
    PanelStandardScaler,
    RollingTransformer,
    TemporalImputer,
)


def _panel_interleaved() -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "entity_id": ["b", "a", "b", "a", "b", "a", "b", "a"],
            "time": [2, 1, 1, 2, 4, 3, 3, 4],
            "val": [20.0, 1.0, 10.0, 2.0, 40.0, 3.0, 30.0, 4.0],
        }
    )
    return df


def test_lag_and_rolling_preserve_interleaved_row_order():
    df = _panel_interleaved()
    lag_tf = LagTransformer(time_col="time", entity_col="entity_id", lags=(1, 2))
    out_lag = lag_tf.fit_transform(df)
    assert out_lag.height == df.height
    assert out_lag.select(["entity_id", "time", "val"]).equals(df)
    assert lag_tf.get_feature_names_out().tolist() == [
        "entity_id",
        "time",
        "val",
        "val__lag_1",
        "val__lag_2",
    ]
    # Verify chronological values despite shuffled input rows (e.g., row 0 is (b, t=2, v=20) -> lag1=10)
    assert out_lag.get_column("val__lag_1").to_list()[0] == 10.0

    roll_tf = RollingTransformer(
        time_col="time",
        entity_col="entity_id",
        window_sizes=(2,),
        stats=("mean", "sum"),
        shift=1,
    )
    out_roll = roll_tf.fit_transform(df)
    assert out_roll.select(["entity_id", "time", "val"]).equals(df)
    # Row index 6 is (b, t=3): past 2 shifted values are [10, 20] -> sum=30, mean=15
    assert out_roll.get_column("val__rolling_mean_2").to_list()[6] == 15.0
    assert out_roll.get_column("val__rolling_sum_2").to_list()[6] == 30.0


def test_diff_scale_detrend_inverse_transform_roundtrips():
    df = _panel_interleaved()
    for tf in (
        DiffTransformer(time_col="time", entity_col="entity_id", order=1, sp=1),
        DiffTransformer(time_col="time", entity_col="entity_id", order=2, sp=1),
        PanelStandardScaler(time_col="time", entity_col="entity_id"),
        DetrendTransformer(time_col="time", entity_col="entity_id", method="linear"),
        DetrendTransformer(time_col="time", entity_col="entity_id", method="mean"),
    ):
        transformed = tf.fit_transform(df)
        restored = tf.inverse_transform(transformed)
        assert_frame_equal(
            restored,
            df,
            check_exact=False,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )

    # Also test univariate Detrend extrapolation out-of-sample
    uni_train = pl.DataFrame({"time": [0, 1, 2, 3], "val": [5.0, 7.0, 9.0, 11.0]})
    uni_test = pl.DataFrame({"time": [4, 5], "val": [13.0, 15.0]})
    dt = DetrendTransformer(time_col="time", entity_col=None, method="linear").fit(
        uni_train
    )
    assert np.allclose(
        dt.transform(uni_test).get_column("val").to_numpy(), [0.0, 0.0], atol=1e-6
    )


def test_fourier_phase_continuity_numeric_and_monthly_dates():
    # Numeric index
    num_df = pl.DataFrame(
        {"time": list(range(24)), "val": [float(i) for i in range(24)]}
    )
    ft = FourierTransformer(
        time_col="time", period=12.0, order=2, keep_original=False
    ).fit(num_df.head(15))
    tr_feat = ft.transform(num_df.slice(0, 3))
    te_feat = ft.transform(num_df.slice(12, 3))
    assert np.allclose(
        tr_feat.drop("time").to_numpy(),
        te_feat.drop("time").to_numpy(),
        atol=1e-12,
    )

    # Monthly Date continuity across train/test boundary
    dates = pl.date_range(
        date(2020, 1, 1), date(2022, 12, 1), interval="1mo", eager=True
    )
    mo_df = pl.DataFrame({"time": dates, "val": [float(i) for i in range(len(dates))]})
    ft_mo = FourierTransformer(
        time_col="time", period=12.0, order=1, freq="1mo", keep_original=False
    ).fit(mo_df.head(24))
    jan_2020 = ft_mo.transform(mo_df.slice(0, 1)).drop("time").to_numpy()
    jan_2022 = ft_mo.transform(mo_df.slice(24, 1)).drop("time").to_numpy()
    assert np.allclose(jan_2020, jan_2022, atol=1e-12)


def test_calendar_frac_diff_and_imputer():
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    df = pl.DataFrame(
        {
            "time": dates,
            "val": [1.0, None, 3.0, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    imp = TemporalImputer(time_col="time", strategy="linear")
    df_imp = imp.fit_transform(df)
    assert df_imp.get_column("val").null_count() == 0
    assert df_imp.get_column("val").to_list()[1] == 2.0

    cal = CalendarTransformer(
        time_col="time", attrs=("month", "weekday"), cyclical=True
    )
    df_cal = cal.fit_transform(df_imp)
    assert {"month__sin", "month__cos", "weekday__sin", "weekday__cos"} <= set(
        df_cal.columns
    )

    fd = FractionalDiffTransformer(time_col="time", d=0.5, window_size=4)
    df_fd = fd.fit_transform(df_imp)
    assert len(df_fd) == len(df_imp)
    assert df_fd.get_column("val").drop_nulls().len() == len(df_imp) - 3


def test_narwhals_inputs_set_output_clone_and_pickle():
    df_pl = pl.DataFrame(
        {
            "region": ["eu", "eu", "us", "us"],
            "store": ["s1", "s1", "s2", "s2"],
            "time": [0, 1, 0, 1],
            "val": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df_pd = df_pl.to_pandas()
    df_pa = df_pl.to_arrow()

    tf = LagTransformer(time_col="time", entity_col=["region", "store"], lags=(1,))
    out_pl = tf.fit_transform(df_pl)
    out_pd_via_default = tf.transform(df_pd)
    out_pa_via_default = tf.transform(df_pa)
    assert isinstance(out_pl, pl.DataFrame)
    assert_frame_equal(out_pl, out_pd_via_default)
    assert_frame_equal(out_pl, out_pa_via_default)

    # set_output on estimator, pipeline, and global config_context
    tf_pandas = clone(tf).set_output(transform="pandas").fit(df_pd)
    res_pd = tf_pandas.transform(df_pd)
    assert hasattr(res_pd, "to_numpy") and list(res_pd.columns) == list(out_pl.columns)

    with config_context(transform_output="pandas"):
        pipe = make_pipeline(
            LagTransformer(time_col="time", entity_col=["region", "store"], lags=(1,)),
            PanelFeatureAdapter(time_col="time", entity_col=["region", "store"]),
        ).fit(df_pl)
        piped = pipe.transform(df_pl)
        assert piped.index is not None
        assert "time" not in piped.columns

    # Pickle roundtrip + naive forecasters
    y = df_pl.get_column("val").to_numpy()
    naive = NaiveForecaster(time_col="time", entity_col=["region", "store"]).fit(
        df_pl, y
    )
    restored_naive = pickle.loads(pickle.dumps(naive))
    preds = restored_naive.predict(df_pl)
    np.testing.assert_allclose(preds, [20.0, 20.0, 40.0, 40.0])

    snaive = SeasonalNaiveForecaster(
        time_col="time", entity_col=["region", "store"], sp=2
    ).fit(df_pl, y)
    np.testing.assert_allclose(snaive.predict(df_pl), [10.0, 20.0, 30.0, 40.0])
