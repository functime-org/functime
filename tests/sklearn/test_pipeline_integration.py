from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from functime.sklearn import (
    CalendarTransformer,
    ExpandingWindowSplit,
    FourierTransformer,
    LagTransformer,
    PanelFeatureAdapter,
    RollingTransformer,
    TemporalImputer,
)


def _make_synthetic_panel(
    n_entities: int = 3, n_steps: int = 48
) -> tuple[pl.DataFrame, np.ndarray]:
    dates = [date(2022, 1, 1) + timedelta(days=30 * i) for i in range(n_steps)]
    entities: list[str] = []
    times: list[date] = []
    signal: list[float] = []
    targets: list[float] = []
    for e_idx in range(n_entities):
        eid = f"series_{e_idx}"
        base = 10.0 * (e_idx + 1)
        for t_idx, d in enumerate(dates):
            val = base + 0.5 * t_idx + 3.0 * np.sin(2.0 * np.pi * t_idx / 12.0)
            entities.append(eid)
            times.append(d)
            signal.append(float(val))
            targets.append(float(val + 1.0))
    X = pl.DataFrame({"entity_id": entities, "time": times, "value": signal})
    return X, np.asarray(targets, dtype=np.float32)


def _feature_pipeline(
    entity_encoding: str,
    *,
    as_pandas: bool = False,
):
    steps = [
        CalendarTransformer(
            time_col="time", entity_col="entity_id", attrs=("month",), cyclical=True
        ),
        FourierTransformer(
            time_col="time", entity_col="entity_id", period=12.0, order=2, freq="1mo"
        ),
        LagTransformer(time_col="time", entity_col="entity_id", lags=(1, 2)),
        RollingTransformer(
            time_col="time", entity_col="entity_id", window_sizes=(3,), stats=("mean",)
        ),
        TemporalImputer(time_col="time", entity_col="entity_id", strategy="backward"),
        PanelFeatureAdapter(
            time_col="time",
            entity_col="entity_id",
            entity_encoding=entity_encoding,  # type: ignore[arg-type]
        ),
    ]
    pipe = make_pipeline(*steps)
    if as_pandas:
        pipe.set_output(transform="pandas")
    return pipe


def test_hist_gradient_boosting_polars_enum_pipeline():
    X, y = _make_synthetic_panel()
    cv = ExpandingWindowSplit(
        time_col="time", entity_col="entity_id", n_splits=3, test_size=4, step_size=4
    )
    pipe = make_pipeline(
        _feature_pipeline("enum", as_pandas=False),
        HistGradientBoostingRegressor(
            categorical_features="from_dtype",
            min_samples_leaf=1,
            max_iter=15,
            random_state=0,
        ),
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_lightgbm_ordinal_and_pandas_cat_pipeline():
    lightgbm = pytest.importorskip("lightgbm")
    X, y = _make_synthetic_panel()
    cv = ExpandingWindowSplit(
        time_col="time", entity_col="entity_id", n_splits=3, test_size=4, step_size=4
    )

    pipe_polars = make_pipeline(
        _feature_pipeline("ordinal", as_pandas=False),
        lightgbm.LGBMRegressor(
            n_estimators=15, min_child_samples=1, verbose=-1, random_state=0
        ),
    )
    scores_pl = cross_val_score(
        pipe_polars, X, y, cv=cv, scoring="neg_mean_absolute_error"
    )
    assert np.isfinite(scores_pl).all()

    pipe_pandas = make_pipeline(
        _feature_pipeline("enum", as_pandas=True),
        lightgbm.LGBMRegressor(
            n_estimators=15, min_child_samples=1, verbose=-1, random_state=0
        ),
    )
    scores_pd = cross_val_score(
        pipe_pandas, X, y, cv=cv, scoring="neg_mean_absolute_error"
    )
    assert np.isfinite(scores_pd).all()


def test_xgboost_polars_enum_pipeline():
    xgboost = pytest.importorskip("xgboost")
    X, y = _make_synthetic_panel()
    cv = ExpandingWindowSplit(
        time_col="time", entity_col="entity_id", n_splits=3, test_size=4, step_size=4
    )
    pipe = make_pipeline(
        _feature_pipeline("enum", as_pandas=False),
        xgboost.XGBRegressor(n_estimators=15, enable_categorical=True, random_state=0),
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_catboost_ordinal_and_pandas_cat_pipeline():
    catboost = pytest.importorskip("catboost")
    X, y = _make_synthetic_panel()
    cv = ExpandingWindowSplit(
        time_col="time", entity_col="entity_id", n_splits=3, test_size=4, step_size=4
    )

    pipe_polars = make_pipeline(
        _feature_pipeline("ordinal", as_pandas=False),
        catboost.CatBoostRegressor(iterations=15, verbose=0, random_seed=0),
    )
    scores_pl = cross_val_score(
        pipe_polars, X, y, cv=cv, scoring="neg_mean_absolute_error"
    )
    assert np.isfinite(scores_pl).all()

    # CatBoostRegressor mutates `cat_features` internally, breaking sklearn `clone` in
    # `cross_val_score`; evaluate the equivalent CV folds directly via fit/predict.
    fe_pandas = _feature_pipeline("enum", as_pandas=True)
    for tr_idx, te_idx in cv.split(X):
        model = catboost.CatBoostRegressor(
            iterations=15, cat_features=["entity_id"], verbose=0, random_seed=0
        )
        X_tr_pd = fe_pandas.fit_transform(X[tr_idx], y[tr_idx])
        X_te_pd = fe_pandas.transform(X[te_idx])
        model.fit(X_tr_pd, y[tr_idx])
        preds = model.predict(X_te_pd)
        assert preds.shape == (len(te_idx),)
        assert np.isfinite(preds).all()
