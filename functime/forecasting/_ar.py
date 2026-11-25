from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
import polars as pl
from tqdm import trange

from functime.base.forecaster import ForecastStrategy
from functime.forecasting._reduction import (
    make_direct_reduction,
    make_reduction,
    make_y_lag,
)


class _RegressFn(Protocol):
    def __call__(self, X: Any, y: Any) -> Any: ...


def fit_recursive(
    regress: _RegressFn,
    lags: int,
    y: pl.LazyFrame,
    X: pl.LazyFrame | None = None,
) -> Mapping[str, Any]:
    # 1. Impose AR structure
    y_columns = y.collect_schema().names()
    target_col = y_columns[-1]
    X_y_final = make_reduction(lags=lags, y=y, X=X).lazy()
    X_y_final_columns = X_y_final.collect_schema().names()
    X_final, y_final = pl.collect_all(
        [
            X_y_final.select(pl.all().exclude(target_col)),
            X_y_final.select([*X_y_final_columns[:2], target_col]),
        ]
    )
    # 2. Fit
    fitted_regressor = regress(X=X_final, y=y_final)
    # 3. Collect artifacts
    y_lag = make_y_lag(X_y_final, target_col=y_columns[-1], lags=lags)
    artifacts = {
        "regressor": fitted_regressor,
        "y_lag": y_lag.collect(engine="streaming"),
    }
    return artifacts


def fit_direct(
    regress: _RegressFn,
    lags: int,
    max_horizons: int,
    y: pl.LazyFrame,
    X: pl.LazyFrame | None = None,
) -> Mapping[str, Any]:
    idx_cols = y.collect_schema().names()[:2]
    target_col = y.collect_schema().names()[-1]
    feature_cols = X.collect_schema().names()[2:] if X is not None else []
    # 1. Impose AR structure
    X_y_final = make_direct_reduction(lags=lags, max_horizons=max_horizons, y=y, X=X)
    # 2. Fit
    fitted_regressors = []
    for i in trange(1, max_horizons + 1, desc="Fitting direct forecasters:"):
        selected_lags = range(i, lags + i)
        lag_cols = [f"{target_col}__lag_{j}" for j in selected_lags]
        X_final = X_y_final.select([*idx_cols, *lag_cols, *feature_cols])
        y_final = X_y_final.select([*idx_cols, target_col])
        fitted_regressor = regress(X=X_final, y=y_final)
        fitted_regressors.append(fitted_regressor)
    # 3. Collect artifacts
    y_lag = make_y_lag(
        X_y_final, target_col=y.collect_schema().names()[-1], lags=lags + max_horizons
    )
    artifacts = {
        "regressors": fitted_regressors,
        "y_lag": y_lag.collect(engine="streaming"),
    }
    return artifacts


def fit_autoreg(
    regress: _RegressFn,
    lags: int,
    y: pl.DataFrame | pl.LazyFrame,
    X: pl.DataFrame | pl.LazyFrame | None = None,
    max_horizons: int | None = None,
    strategy: ForecastStrategy | None = None,
) -> Mapping[str, Any]:
    y = y.lazy()
    X = X.lazy() if X is not None else X
    strategy = strategy or "recursive"
    if strategy in ["direct", "ensemble"] and max_horizons is None:
        raise ValueError(
            "If `strategy` is set as 'direct' or 'ensemble', then `max_horizons` must be set"
            " in the forecaster's kwargs upon initialization."
        )
    if strategy == "recursive":
        artifacts = fit_recursive(regress=regress, lags=lags, y=y, X=X)
    elif strategy == "direct":
        artifacts = fit_direct(
            regress=regress, lags=lags, max_horizons=max_horizons, y=y, X=X
        )
    elif strategy == "ensemble":
        artifacts = {
            "recursive": fit_recursive(regress=regress, lags=lags, y=y, X=X),
            "direct": fit_direct(
                regress=regress, lags=lags, max_horizons=max_horizons, y=y, X=X
            ),
        }
    else:
        raise ValueError(f"Cannot recognize `strategy` '{strategy}'")
    return artifacts


# NOTE: REMEMBER exogenous X DOES NOT HAVE TIME_COL
# (values are aggregated into list before being passed into predict)


def predict_recursive(
    state,
    fh: int,
    X: pl.DataFrame | None = None,
) -> pl.DataFrame:
    artifacts = state.artifacts
    if "recursive" in artifacts.keys():
        artifacts = state.artifacts["recursive"]
    regressor = artifacts["regressor"]
    entity_col = state.entity
    y_lag: pl.DataFrame = artifacts["y_lag"].sort(entity_col).set_sorted(entity_col)
    if X is not None:
        X = X.group_by(entity_col).agg(pl.all()).sort(entity_col).set_sorted(entity_col)

    lag_cols = y_lag.columns[2:]
    lead_col = lag_cols[0]

    def _get_x_y_slice(y_lag: pl.DataFrame, i: int):
        x_y_slice = y_lag.select(
            [entity_col, pl.all().exclude(entity_col).list.get(-1)]
        )
        if X is not None:
            x = X.select([entity_col, pl.all().exclude(entity_col).list.get(i)])
            x_y_slice = x_y_slice.join(
                x, on=entity_col, how="left", maintain_order="left"
            )
        return x_y_slice

    is_censored = getattr(regressor, "is_censored", False)
    weights = np.zeros((fh, len(y_lag))) if is_censored else None

    for i in range(fh):
        # 1. Get most recent features
        x_y_slice = _get_x_y_slice(y_lag=y_lag, i=i)
        # 2. Predict
        y_pred_i = regressor.predict(x_y_slice)
        if is_censored:
            y_pred_i, weights_i = y_pred_i
            weights[i] = weights_i
        # 3. Update AR structure
        y_shifted = [
            pl.col(lag_cols[i]).alias(lag_cols[i + 1])
            for i in range(0, len(lag_cols) - 1)
        ]
        y_new = pl.col(lead_col).list.concat(pl.Series(y_pred_i))
        y_lag = y_lag.with_columns([y_new, *y_shifted])

    pred_cols = [entity_col, pl.col(lead_col).list.tail(fh).alias(state.target)]
    y_pred = y_lag.select(pred_cols)

    if is_censored:
        weights = pl.DataFrame(np.stack(weights, axis=1).astype(np.float32)).select(
            pl.concat_list(pl.all()).alias("threshold_proba")
        )
        y_pred = pl.concat([y_pred, weights], how="horizontal")

    return y_pred


# NOTE: REMEMBER exogenous X DOES NOT HAVE TIME_COL
# (values are aggregated into list before being passed into predict)


def predict_direct(state, fh: int, X: pl.DataFrame | None = None) -> pl.DataFrame:
    entity_col = state.entity
    time_col = state.time
    target_col = state.target
    regressor_cols = X.columns[1:] if X is not None else []
    artifacts = state.artifacts
    if "direct" in artifacts.keys():
        artifacts = state.artifacts["direct"]
    regressors = artifacts["regressors"]
    max_horizons = len(regressors)
    if fh > max_horizons:
        raise ValueError(
            "`fh` must be less than or equal to `max_horizons` in model parameters."
            f" Expected `fh <= {max_horizons}`, got `{fh}`."
        )

    y_lag: pl.DataFrame = artifacts["y_lag"].sort(entity_col).set_sorted(entity_col)
    if X is not None:
        X = X.group_by(entity_col).agg(pl.all()).sort(entity_col).set_sorted(entity_col)
    lags = (y_lag.width - 1) - max_horizons

    n_entities = len(y_lag)
    y_pred = np.empty((n_entities, fh))
    is_censored = getattr(regressors[0], "predict_proba", None)
    weights = np.zeros((fh, n_entities)) if is_censored else None

    for i in range(fh):
        selected_lags = range(i + 1, lags + i)
        lag_cols = [pl.col(f"{target_col}__lag_{j}").list.get(i) for j in selected_lags]
        x = y_lag.select([entity_col, pl.col(time_col).list.get(i), *lag_cols])
        if X is not None:
            x_slice = X.select([entity_col, pl.col(regressor_cols).list.get(i)])
            x = x.join(x_slice, on=entity_col, how="left", maintain_order="left")
        # Predict
        y_pred_i = regressors[i].predict(x)
        # Censored forecast adjustment
        if is_censored:
            y_pred_i, weights_i = y_pred_i
            weights[i] = weights_i
        y_pred[:, i] = y_pred_i

    y_pred = (
        pl.DataFrame(y_pred)
        .select(pl.concat_list(pl.all()).alias(target_col))
        .with_columns(y_lag.get_column(entity_col))
        .select([entity_col, target_col])
    )
    if is_censored:
        weights = pl.DataFrame(np.stack(weights, axis=1).astype()).select(
            pl.concat_list(pl.all()).alias("threshold_proba")
        )
        y_pred = pl.concat([y_pred, weights], how="horizontal")

    return y_pred


# NOTE: REMEMBER exogenous X DOES NOT HAVE TIME_COL
# (values are aggregated into list before being passed into predict)


def predict_autoreg(
    state,
    fh: int,
    X: pl.DataFrame | pl.LazyFrame | None = None,
) -> pl.DataFrame:
    strategy = state.strategy
    time_col = state.time
    predict_kwargs = {
        "state": state,
        "fh": fh,
        "X": X.drop(time_col).lazy().collect() if X is not None else None,
    }

    if strategy == "recursive":
        y_pred = predict_recursive(**predict_kwargs)
    elif strategy == "direct":
        y_pred = predict_direct(**predict_kwargs)
    elif strategy == "ensemble":
        target_col = state.target
        y_pred_rec = predict_recursive(**predict_kwargs).rename(
            {target_col: "recursive"}
        )
        y_pred_dir = predict_direct(**predict_kwargs).rename({target_col: "direct"})
        y_pred = (
            y_pred_rec.join(y_pred_dir, on=state.entity, maintain_order="left")
            .explode(["recursive", "direct"])
            .select(
                [
                    state.entity,
                    ((pl.col("recursive") + pl.col("direct")) / 2).alias(target_col),
                ]
            )
            .group_by(state.entity, maintain_order=True)
            .agg(target_col)
        )
    else:
        raise ValueError(f"Cannot recognize `strategy` '{strategy}'")
    return y_pred
