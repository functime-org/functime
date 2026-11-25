from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from functime.sklearn import (
    ExpandingWindowSplit,
    LagTransformer,
    PanelFeatureAdapter,
    SlidingWindowSplit,
    TemporalImputer,
    train_test_split,
)


def _make_panel_df(n_steps: int = 20, multi_entity: bool = False) -> pl.DataFrame:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_steps)]
    if not multi_entity:
        return pl.DataFrame(
            {
                "entity_id": ["e0"] * n_steps + ["e1"] * n_steps,
                "time": dates + dates,
                "x": [float(i) for i in range(n_steps)]
                + [float(100 + i) for i in range(n_steps)],
            }
        )
    rows_r, rows_s, rows_t, rows_x = [], [], [], []
    for reg in ("north", "south"):
        for store in ("s1", "s2"):
            for i, d in enumerate(dates):
                rows_r.append(reg)
                rows_s.append(store)
                rows_t.append(d)
                rows_x.append(float(i + (10 if reg == "south" else 0)))
    return pl.DataFrame(
        {"region": rows_r, "store": rows_s, "time": rows_t, "x": rows_x}
    )


def test_train_test_split_univariate_and_panel_gap():
    uni = pl.DataFrame({"time": list(range(20)), "x": [float(i) for i in range(20)]})
    tr, te = train_test_split(uni, time_col="time", test_size=4, gap=2)
    assert tr.get_column("time").to_list() == list(range(14))
    assert te.get_column("time").to_list() == list(range(16, 20))

    panel = _make_panel_df(n_steps=20, multi_entity=True)
    tr_p, te_p = train_test_split(
        panel, time_col="time", entity_col=["region", "store"], test_size=0.2, gap=1
    )
    assert tr_p.get_column("time").max() < te_p.get_column("time").min()
    assert tr_p.height == 4 * 15
    assert te_p.height == 4 * 4


def test_expanding_and_sliding_window_splits_panel_and_pandas():
    panel = _make_panel_df(n_steps=24, multi_entity=True)
    ev = ExpandingWindowSplit(
        time_col="time",
        entity_col=["region", "store"],
        n_splits=3,
        test_size=2,
        step_size=2,
        gap=1,
    )
    assert ev.get_n_splits() == 3

    folds = list(ev.split(panel))
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert len(set(train_idx) & set(test_idx)) == 0
        t_train_max = panel[train_idx].get_column("time").max()
        t_test_min = panel[test_idx].get_column("time").min()
        assert t_train_max is not None and t_test_min is not None
        assert t_train_max < t_test_min
        assert panel[test_idx].get_column("time").n_unique() == 2

    sv = SlidingWindowSplit(
        time_col="time",
        entity_col=["region", "store"],
        n_splits=3,
        test_size=2,
        step_size=2,
        window_size=6,
    )
    pdf = panel.to_pandas()
    for tr_idx, te_idx in sv.split(pdf):
        assert panel[tr_idx].get_column("time").n_unique() == 6
        assert panel[te_idx].get_column("time").n_unique() == 2


def test_cross_val_score_with_temporal_pipeline():
    n_steps = 40
    df = _make_panel_df(n_steps=n_steps, multi_entity=False)
    y = df.get_column("x").to_numpy() * 2.0 + 3.0

    cv = ExpandingWindowSplit(
        time_col="time",
        entity_col="entity_id",
        n_splits=3,
        test_size=4,
        step_size=4,
    )
    pipe = make_pipeline(
        LagTransformer(time_col="time", entity_col="entity_id", lags=(1, 2)),
        TemporalImputer(time_col="time", entity_col="entity_id", strategy="backward"),
        PanelFeatureAdapter(
            time_col="time",
            entity_col="entity_id",
            entity_encoding="ordinal",
        ),
        Ridge(),
    )
    scores = cross_val_score(pipe, df, y, cv=cv, scoring="neg_mean_absolute_error")
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_split_validation_errors():
    df = pl.DataFrame({"time": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    cv = ExpandingWindowSplit(time_col="time", n_splits=5, test_size=2)
    with pytest.raises(ValueError, match="Not enough unique timestamps"):
        list(cv.split(df))
    with pytest.raises(ValueError, match="Missing required"):
        train_test_split(df, time_col="missing_col", test_size=1)
