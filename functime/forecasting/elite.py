from typing import Any, Mapping, Optional, Union

import polars as pl
from sklearn.linear_model import LassoLarsIC
from tqdm import tqdm

from functime.backtesting import backtest
from functime.base.forecaster import Forecaster
from functime.base.metric import METRIC_TYPE
from functime.conversion import X_to_numpy, y_to_numpy
from functime.cross_validation import expanding_window_split
from functime.forecasting.linear import lasso, linear_model, ridge
from functime.forecasting.naive import naive
from functime.metrics import rmse
from functime.offsets import freq_to_sp


class elite(Forecaster):
    """Global ELITE forecasting procedure with expanding windows for cross-validation.

    Uses `sklearn.linear_model.LassoLarsIC` for stacking.

    Inspired by the ensemble approach taken by DoorDash:
    https://doordash.engineering/2023/06/20/how-doordash-built-an-ensemble-learning-model-for-time-series-forecasting/

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    lags : int
        Number of lagged target variables.
    max_fh : Optional[int]
        Max forecast horizon (required for `naive` forecaster).
        If None, defaults to `test_size`.
    sp : Optional[int]
        Seasonal periods; length of one seasonal cycle.
    forecasters : Optional[Mapping[str, Forecaster]]
        Mapping of name to forecaster class to fit.
        A `naive` forecaster is always fit as the fallback.
    model_kwargs : Optional[Mapping[str, Mapping[str, Any]]]
        Mapping of forecaster name to model kwargs passed into the underlying sklearn-compatible regressor.
    top_k : Optional[int]
        Select top k performing forecasters from cross-validation to ensemble.
        Defaults to 4 if None.
    scoring : Optional[metric]
        If None, defaults to RMSE.
    test_size : Optional[int]
        Number of test samples for each split.
        If None, defaults to equal to one seasonal period given `freq`
        (e.g. `test_size=12` if freq is monthly `1mo`).
    step_size : Optional[int]
        Step size between windows.
    n_splits : Optional[int]
        Number of splits. Defaults to 3.
    **kwargs : Mapping[str, Any]
        Additional keyword arguments passed into the final stacking regressor (i.e. `sklearn.linear_model.LassoLarsIC`)
    """

    def __init__(
        self,
        freq: Union[str, None],
        lags: int,
        max_fh: Optional[int] = None,
        sp: Optional[int] = None,
        forecasters: Optional[Mapping[str, Forecaster]] = None,
        model_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        top_k: Optional[int] = None,
        scoring: Optional[METRIC_TYPE] = None,
        test_size: Optional[int] = None,
        step_size: Optional[int] = None,
        n_splits: Optional[int] = None,
        **kwargs,
    ):
        self.max_fh = max_fh
        self.sp = sp
        self.forecasters = forecasters or {
            # # "Seasonality" models
            # "knn": knn,
            # AR linear models
            "linear": linear_model,
            "ridge": ridge,
            "lasso": lasso,
            # # AR models with Fourier terms
            # # AR models with box-cox scaling
            # "linear_boxcox": linear_model,
            # "ridge_boxcox": lasso,
            # "lasso_boxcox": ridge,
            # # AR models with box-cox scaling and Fourier terms
            # "linear_boxcox_fourier": linear_model,
            # "ridge_boxcox_fourier": lasso,
            # "lasso_boxcox_fourier": ridge,
            # # Linear detrended AR models
            # "linear_detrend": linear_model,
            # "ridge_detrend": lasso,
            # "lasso_detrend": ridge,
        }
        self.model_kwargs = model_kwargs or {}
        self.top_k = top_k or 4
        self.scoring = scoring
        self.test_size = test_size
        self.step_size = step_size
        self.n_splits = n_splits
        super().__init__(freq=freq, lags=lags, **kwargs)

    def _get_X_stack(
        self,
        y_pred: pl.DataFrame,
        best_models: pl.DataFrame,
        X: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        top_k = self.top_k
        entity_col, time_col, target_col = y_pred.columns[:3]
        X_stack = (
            best_models.select([entity_col, "model_name"])
            .explode("model_name")
            .lazy()
            .join(y_pred, how="left", on=[entity_col, "model_name"])
            .groupby([entity_col, time_col])
            .agg(pl.col(target_col))
            .select(
                [
                    entity_col,
                    time_col,
                    *[
                        pl.col(target_col).list.get(i).alias(f"model_{i}")
                        for i in range(top_k)
                    ],
                ]
            )
            .sort([entity_col, time_col])
            .set_sorted([entity_col, time_col])
            .collect(streaming=True)
        )
        if X is not None:
            X_stack = (
                X_stack.lazy()
                .join(X.lazy(), on=[entity_col, time_col], how="left")
                .collect(streaming=True)
            )
        return X_stack

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        freq = self.freq
        lags = self.lags
        top_k = self.top_k
        model_kwargs = self.model_kwargs
        score = self.scoring or rmse
        metric_name = score.__name__
        entity_col, time_col, target_col = y.columns

        # 1. Cross validation
        sp = self.sp or freq_to_sp(freq=freq)
        test_size = self.test_size or sp
        max_fh = self.max_fh or test_size
        step_size = self.step_size or sp
        n_splits = self.n_splits or 3
        cv = expanding_window_split(
            test_size=test_size,
            step_size=step_size,
            n_splits=n_splits,
        )
        cv_y_preds = {}
        # NOTE: Parallelized version available in Cloud
        for model_name, forecaster_cls in (pbar := tqdm(self.forecasters.items())):
            pbar.set_description(f"Cross-validating [forecaster={model_name}]")
            # TODO: Investigate using residuals to quantify model uncertainty
            forecaster = forecaster_cls(
                freq=freq, lags=lags, **model_kwargs.get(model_name, {})
            )
            y_preds = backtest(forecaster=forecaster, y=y, cv=cv, residualize=False)
            cv_y_preds[model_name] = y_preds

        # 2. Fit naive (fallback)
        cv_y_preds["naive"] = backtest(
            forecaster=naive(freq=freq, max_fh=test_size), y=y, cv=cv, residualize=False
        )

        # 3. Score individual forecasters
        cv_scores = []
        for model_name, y_preds in (pbar := tqdm(cv_y_preds.items())):
            pbar.set_description(f"Scoring [forecaster={model_name}]")
            for split in range(n_splits):
                y_pred = y_preds.filter(pl.col("split") == split).drop("split").lazy()
                y_true = (
                    y_pred.select([entity_col, time_col])
                    .join(y.lazy(), on=[entity_col, time_col], how="left")
                    .collect(streaming=True)
                )
                scores = score(y_pred=y_pred, y_true=y_true).with_columns(
                    [
                        pl.lit(split).alias("split"),
                        pl.lit(model_name).alias("model_name"),
                    ]
                )
                cv_scores.append(scores)

        # 4. Select top N best models from CV
        scores = (
            pl.concat(cv_scores)
            .sort([entity_col, metric_name, "split"])
            .set_sorted([entity_col, metric_name, "split"])
        )
        best_models = (
            scores.lazy()
            .groupby([entity_col, "model_name"])
            # Compute average score across splits
            .agg(pl.col(metric_name).mean())
            .sort([entity_col, metric_name])
            .set_sorted([entity_col, metric_name])
            # Select top K scores
            .groupby(entity_col, maintain_order=True)
            .agg([pl.col("model_name").head(top_k), metric_name])
            .collect(streaming=True)
        )
        full_y_preds = pl.concat(
            [
                y_preds.lazy().with_columns(pl.lit(model_name).alias("model_name"))
                for model_name, y_preds in cv_y_preds.items()
            ]
        )

        # 5. Prepare ensemble (stacked regression) inputs
        X_stack = self._get_X_stack(y_pred=full_y_preds, best_models=best_models, X=X)
        y_stack = (
            X_stack.select([entity_col, time_col])
            .lazy()
            .join(y.lazy(), on=[entity_col, time_col], how="left")
            .collect(streaming=True)
        )

        # 6. Fit final regressor
        regressor = LassoLarsIC(**self.kwargs).fit(
            X=X_to_numpy(X_stack), y=y_to_numpy(y_stack)
        )

        # 7. Fit forecasters
        forecasters = {}
        for model_name, forecaster_cls in (pbar := tqdm(self.forecasters.items())):
            pbar.set_description(f"Refitting [forecaster={model_name}]")
            forecaster = forecaster_cls(
                freq=freq, lags=lags, **model_kwargs.get(model_name, {})
            ).fit(y=y)
            forecasters[model_name] = forecaster
        forecasters["naive"] = naive(freq=freq, max_fh=max_fh).fit(y=y)

        artifacts = {
            "scores": scores,
            "best_models": best_models,
            "forecasters": forecasters,
            "final_regressor": regressor,
        }
        return artifacts

    def predict(self, fh: int, X: Optional[pl.LazyFrame] = None):
        state = self.state
        entity_col = state.entity
        time_col = state.time
        target_col = state.target

        # 1. Get individual forecasts
        forecasters = state.artifacts["forecasters"]
        forecasts = {}
        for model_name, forecaster in (pbar := tqdm(forecasters.items())):
            pbar.set_description(f"Forecast [forecaster={model_name}]")
            forecasts[model_name] = forecaster.predict(fh=fh)

        # 2. Prepare ensemble (stacked regression) input
        best_models = state.artifacts["best_models"]
        full_y_pred = pl.concat(
            [
                y_pred.lazy().with_columns(pl.lit(model_name).alias("model_name"))
                for model_name, y_pred in forecasts.items()
            ]
        )
        X_stack = self._get_X_stack(y_pred=full_y_pred, best_models=best_models, X=X)

        # 3. Predict using final regressor
        final_regressor = state.artifacts["final_regressor"]
        y_pred_values = final_regressor.predict(X=X_to_numpy(X_stack))
        y_pred = (
            X_stack.select([entity_col, time_col])
            .with_columns(pl.lit(y_pred_values).alias(target_col))
            .pipe(self._reset_string_cache)
        )
        return y_pred
