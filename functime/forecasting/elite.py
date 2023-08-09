from functools import partial
from typing import Any, Mapping, Optional, Union

import polars as pl
import polars.selectors as cs
from sklearn.linear_model import LassoLarsIC
from tqdm import tqdm
from typing_extensions import Literal

from functime.backtesting import backtest
from functime.base.forecaster import Forecaster
from functime.base.metric import METRIC_TYPE
from functime.conversion import X_to_numpy, y_to_numpy
from functime.cross_validation import expanding_window_split
from functime.feature_extraction import add_fourier_terms
from functime.forecasting.knn import knn
from functime.forecasting.linear import lasso_cv, linear_model, ridge_cv
from functime.forecasting.naive import naive
from functime.metrics import mae
from functime.offsets import freq_to_sp
from functime.preprocessing import coerce_dtypes, detrend, diff, scale


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
    sp : Optional[int]
        Seasonal periods; length of one seasonal cycle.
    forecasters : Optional[Mapping[str, Forecaster]]
        Mapping of name to forecaster class to fit.
        A `naive` forecaster is always fit as the fallback.
    model_kwargs : Optional[Mapping[str, Mapping[str, Any]]]
        Mapping of forecaster name to model kwargs passed into the underlying sklearn-compatible regressor.
    ensemble_strategy : Literal["lasso", "log_lasso", "mean"]
        Strategy to stack base forecasts.
    top_k : Optional[int]
        Select top k performing forecasters from cross-validation to ensemble.
        Defaults to 4 if None.
    scoring : Optional[metric]
        If None, defaults to MAE.
    test_size : Optional[int]
        Number of test samples for each split.
        If None, defaults to 1.
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
        sp: Optional[int] = None,
        forecasters: Optional[Mapping[str, Forecaster]] = None,
        model_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        ensemble_strategy: Literal["lasso", "log_lasso", "mean"] = "mean",
        top_k: Optional[int] = None,
        scoring: Optional[METRIC_TYPE] = None,
        test_size: Optional[int] = None,
        step_size: Optional[int] = None,
        n_splits: Optional[int] = None,
        **kwargs,
    ):
        self.sp = sp or freq_to_sp(freq=freq)[0]
        self.forecasters = forecasters or {
            # "Seasonality" models
            "knn": partial(knn, n_neighbors=lags // 2),
            "knn_scaled": partial(knn, n_neighbors=lags // 2),
            "knn_detrend_linear": partial(
                knn, n_neighbors=lags // 2, target_transform=detrend(method="linear")
            ),
            # AR linear models
            "linear": linear_model,
            "ridge": ridge_cv,
            "lasso": lasso_cv,
            # AR linear models without drift
            "linear_no_drift": partial(linear_model, fit_intercept=False),
            "ridge_no_drift": partial(ridge_cv, fit_intercept=False),
            "lasso_no_drift": partial(lasso_cv, fit_intercept=False),
            # AR models with local scaling
            "linear_scaled": partial(linear_model, target_transform=scale()),
            "ridge_scaled": partial(ridge_cv, target_transform=scale()),
            "lasso_scaled": partial(lasso_cv, target_transform=scale()),
            # AR models with first differences
            "linear_diff": partial(linear_model, target_transform=diff(order=1)),
            "ridge_diff": partial(ridge_cv, target_transform=diff(order=1)),
            "lasso_diff": partial(lasso_cv, target_transform=diff(order=1)),
            # AR models with Fourier terms (defaults to K=6)
            "linear_fourier": partial(
                linear_model, feature_transform=add_fourier_terms(sp=self.sp, K=6)
            ),
            "ridge_fourier": partial(
                ridge_cv, feature_transform=add_fourier_terms(sp=self.sp, K=6)
            ),
            "lasso_fourier": partial(
                lasso_cv, feature_transform=add_fourier_terms(sp=self.sp, K=6)
            ),
            "linear_scaled_fourier": partial(
                linear_model,
                target_transform=scale(),
                feature_transform=add_fourier_terms(sp=self.sp, K=6),
            ),
            "ridge_scaled_fourier": partial(
                ridge_cv,
                target_transform=scale(),
                feature_transform=add_fourier_terms(sp=self.sp, K=6),
            ),
            "lasso_scaled_fourier": partial(
                lasso_cv,
                target_transform=scale(),
                feature_transform=add_fourier_terms(sp=self.sp, K=6),
            ),
            # Linear detrended AR models
            "linear_detrend_linear": partial(
                linear_model, target_transform=detrend(method="linear")
            ),
            "ridge_detrend_linear": partial(
                ridge_cv, target_transform=detrend(method="linear")
            ),
            "lasso_detrend_linear": partial(
                lasso_cv, target_transform=detrend(method="linear")
            ),
            # Mean detrended models
            "linear_detrend_mean": partial(
                linear_model, target_transform=detrend(method="mean")
            ),
            "ridge_detrend_mean": partial(
                ridge_cv, target_transform=detrend(method="mean")
            ),
            "lasso_detrend_mean": partial(
                lasso_cv, target_transform=detrend(method="mean")
            ),
            # Linear detrended fourier AR models
            "linear_detrend_linear_fourier": partial(
                linear_model,
                target_transform=detrend(method="linear"),
                feature_transform=add_fourier_terms(sp=self.sp, K=12),
            ),
            "ridge_detrend_linear_fourier": partial(
                ridge_cv,
                target_transform=detrend(method="linear"),
                feature_transform=add_fourier_terms(sp=self.sp, K=12),
            ),
            "lasso_detrend_linear_fourier": partial(
                lasso_cv,
                target_transform=detrend(method="linear"),
                feature_transform=add_fourier_terms(sp=self.sp, K=12),
            ),
        }

        self.model_kwargs = model_kwargs or {}
        self.ensemble_strategy = ensemble_strategy
        self.top_k = top_k or 12
        self.scoring = scoring
        self.test_size = test_size or 1
        self.step_size = step_size or self.test_size
        self.n_splits = n_splits or 3
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
        X_stack = X_stack.with_columns(
            trend=pl.col(time_col).arg_sort().over(entity_col)
        )
        return X_stack

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        freq = self.freq
        lags = self.lags
        top_k = self.top_k
        model_kwargs = self.model_kwargs
        score = self.scoring or mae
        metric_name = score.__name__
        entity_col, time_col, target_col = y.columns

        # 1. Cross validation
        test_size = self.test_size
        step_size = self.step_size
        n_splits = self.n_splits
        cv = expanding_window_split(
            test_size=test_size,
            step_size=step_size,
            n_splits=n_splits,
        )
        cv_y_preds = {}
        schema = y.schema
        forecasters = {**self.forecasters, "naive": naive}
        # NOTE: Parallelized version available in Cloud
        for model_name, forecaster_cls in (pbar := tqdm(forecasters.items())):
            pbar.set_description(f"Cross-validating [forecaster={model_name}]")
            # TODO: Investigate using residuals to quantify model uncertainty
            if model_name != "naive":
                forecaster = forecaster_cls(
                    freq=freq, lags=lags, **model_kwargs.get(model_name, {})
                )
            else:
                forecaster = forecaster_cls(freq=freq)
            y_preds = backtest(forecaster=forecaster, y=y, cv=cv, residualize=False)
            cv_y_preds[model_name] = y_preds.pipe(coerce_dtypes(schema)).collect()

        # 2. Score individual forecasters
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

        # 3. Select top N best models from CV
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
            .agg([pl.col("model_name").head(top_k), pl.col(metric_name).head(top_k)])
            .collect(streaming=True)
        )
        full_y_preds = pl.concat(
            [
                y_preds.lazy().with_columns(pl.lit(model_name).alias("model_name"))
                for model_name, y_preds in cv_y_preds.items()
            ]
        )

        # 4. Prepare ensemble (stacked regression) inputs
        X_stack = self._get_X_stack(y_pred=full_y_preds, best_models=best_models, X=X)
        y_stack = (
            X_stack.select([entity_col, time_col])
            .lazy()
            .join(y.lazy(), on=[entity_col, time_col], how="left")
            .collect(streaming=True)
        )

        final_regressor = None
        if self.ensemble_strategy in ["lasso", "log_lasso"]:
            # 5. Fit final regressor
            final_regressor = LassoLarsIC(**self.kwargs).fit(
                X=X_to_numpy(X_stack), y=y_to_numpy(y_stack)
            )

        # 6. Fit forecasters
        fitted_forecasters = {}
        for model_name, forecaster_cls in (pbar := tqdm(forecasters.items())):
            pbar.set_description(f"Refitting [forecaster={model_name}]")
            if model_name != "naive":
                forecaster = forecaster_cls(
                    freq=freq, lags=lags, **model_kwargs.get(model_name, {})
                ).fit(y=y)
            else:
                forecaster = forecaster_cls(freq=freq).fit(y=y)
            fitted_forecasters[model_name] = forecaster

        artifacts = {
            "scores": scores,
            "best_models": best_models,
            "forecasters": fitted_forecasters,
            "final_regressor": final_regressor,
        }
        return artifacts

    def predict(self, fh: int, X: Optional[pl.LazyFrame] = None):
        state = self.state
        entity_col = state.entity
        time_col = state.time
        target_col = state.target
        schema = state.target_schema

        # 1. Get individual forecasts
        forecasters = state.artifacts["forecasters"]
        forecasts = {}
        for model_name, forecaster in (pbar := tqdm(forecasters.items())):
            pbar.set_description(f"Forecast [forecaster={model_name}]")
            y_pred = forecaster.predict(fh=fh).pipe(coerce_dtypes(schema)).collect()
            forecasts[model_name] = y_pred

        # 2. Prepare ensemble (stacked regression) input
        best_models = state.artifacts["best_models"]
        full_y_pred = pl.concat(
            [
                y_pred.lazy().with_columns(pl.lit(model_name).alias("model_name"))
                for model_name, y_pred in forecasts.items()
            ]
        )
        X_stack = self._get_X_stack(y_pred=full_y_pred, best_models=best_models, X=X)

        # 3. Predict using final stacker
        if self.ensemble_strategy == "mean":
            y_pred = X_stack.select(
                [
                    entity_col,
                    time_col,
                    (pl.sum_horizontal(cs.starts_with("model_")) / self.top_k).alias(
                        target_col
                    ),
                ]
            )
        else:
            final_regressor = state.artifacts["final_regressor"]
            y_pred_values = final_regressor.predict(X=X_to_numpy(X_stack))
            y_pred = (
                X_stack.select([entity_col, time_col])
                .with_columns(pl.lit(y_pred_values).alias(target_col))
                .pipe(coerce_dtypes(schema))
                .collect()
            )

        random_walk_series = (
            best_models.select([entity_col, pl.col("model_name").list.first()])
            .filter(pl.col("model_name") == "naive")
            .get_column(entity_col)
        )
        naive_forecasts = forecasts["naive"].filter(
            pl.col(entity_col).is_in(random_walk_series)
        )
        ensemble_forecasts = y_pred.filter(
            ~pl.col(entity_col).is_in(random_walk_series)
        )
        y_pred = pl.concat([naive_forecasts, ensemble_forecasts])

        return y_pred.pipe(self._reset_string_cache)
