"""
Fit-predict regressors with special needs.
"""

from typing import Callable, Optional, Union

import numpy as np
import polars as pl
import sklearn
from typing_extensions import Literal

from functime.conversion import X_to_numpy, y_to_numpy
from functime.preprocessing import PL_NUMERIC_COLS


class GradientBoostedTreeRegressor:
    def __init__(
        self,
        regress,
        weight_transform: Optional[Callable] = None,
        fit_dtype: Literal["numpy", "arrow"] = None,
        predict_dtype: Union[Literal["numpy", "arrow"], Callable] = None,
    ):
        self.regress = regress
        self.regressor = None
        self.weight_transform = weight_transform
        self.fit_dtype = fit_dtype or "numpy"
        self.predict_dtype = predict_dtype or "numpy"
        self.label_to_cat = {}

    def _preproc_X(self, X: pl.DataFrame) -> pl.DataFrame:
        entity_col = X.columns[0]
        X_new = X.with_columns(
            pl.col([pl.Categorical, pl.Boolean]).exclude(entity_col).to_physical()
        )
        return X_new

    def fit(self, X: pl.DataFrame, y: pl.DataFrame):
        weight_transform = self.weight_transform
        sample_weight = None
        if weight_transform is not None:
            sample_weight = y.pipe(weight_transform)

        X = self._preproc_X(X)

        if self.fit_dtype == "numpy":
            X_coerced = X_to_numpy(X)
            y_coerced = y_to_numpy(X)
        elif self.fit_dtype == "arrow":
            X_coerced = X.to_arrow()
            y_coerced = y.to_arrow()
        else:
            raise ValueError(f"`fit_dtype` not supported: {self.fit_dtype}")

        self.regressor = self.regress(
            X=X_coerced, y=y_coerced, sample_weight=sample_weight
        )
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        X = self._preproc_X(X)
        if self.predict_dtype == "numpy":
            X_coerced = X_to_numpy(X)
        elif self.predict_dtype == "arrow":
            X_coerced = X.to_arrow
        elif isinstance(self.predict_dtype, Callable):
            X_coerced = self.predict_dtype(X)
        y_pred = self.regressor.predict(X_coerced)
        return y_pred


class SklearnRegressor:
    def __init__(self, regressor):
        self.regressor = regressor

    def _preproc_X(self, X: pl.DataFrame):
        entity_col, time_col = X.columns[:2]
        X_new = X.select(
            [
                entity_col,
                time_col,
                PL_NUMERIC_COLS(entity_col, time_col),
                pl.col(pl.Categorical).exclude(entity_col).to_physical(),
                pl.col(pl.Boolean).cast(pl.Int8),
            ]
        )
        return X_new

    def fit(self, X: pl.DataFrame, y: pl.DataFrame):
        X_new = self._preproc_X(X).lazy()
        # Regress
        with sklearn.config_context(assume_finite=True):
            # NOTE: We can assume finite due to preproc
            self.regressor = self.regressor.fit(X=X_to_numpy(X_new), y=y_to_numpy(y))
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        X_new = self._preproc_X(X).lazy()
        with sklearn.config_context(assume_finite=True):
            # NOTE: We can assume finite due to preproc
            y_pred = self.regressor.predict(X_to_numpy(X_new))
        return y_pred


class CensoredRegressor:
    def __init__(
        self,
        threshold: Union[int, float],
        regress,
        predict_proba,
    ):
        self.threshold = threshold
        self.regress = regress
        self.predict_proba = predict_proba
        self.regressors = None

    @property
    def is_censored(self):
        return True

    def fit(self, X: pl.DataFrame, y: pl.DataFrame):
        idx_cols = X.columns[:2]
        target_col = y.columns[-1]
        threshold = self.threshold
        X_y_above = X.join(y, on=idx_cols).filter(pl.col(target_col) > threshold)
        y_above = X_y_above.select([*idx_cols, target_col])
        X_above = X_y_above.select(pl.all().exclude(target_col))
        if threshold == 0:
            self.regressors = (
                self.regress(X_to_numpy(X_above), y_to_numpy(y_above)),
                None,
            )
        else:
            X_y_below = X.join(y, on=idx_cols).filter(pl.col(target_col) <= threshold)
            y_below = X_y_below.select([*idx_cols, target_col])
            X_below = X_y_below.select(pl.all().exclude(target_col))
            fitted_model_above = self.regress(X_to_numpy(X_above), y_to_numpy(y_above))
            fitted_model_below = self.regress(X_to_numpy(X_below), y_to_numpy(y_below))
            self.regressors = fitted_model_above, fitted_model_below
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        weights = self.predict_proba(X_to_numpy(X))
        regress_above, regress_below = self.regressors
        y_pred = weights[:, 1] * regress_above.predict(X_to_numpy(X))
        if abs(self.threshold) > 0:
            y_pred += weights[:, 0] * regress_below.predict(X_to_numpy(X))
        return y_pred, weights[:, 1]


class FLAMLRegressor:
    """FLAML AutoML regressor."""

    def __init__(
        self,
        time_budget: Optional[int] = None,
        max_iter: Optional[int] = None,
        metric: Optional[str] = None,
        **kwargs,
    ):
        self.time_budget = time_budget or 30
        self.max_iter = max_iter or 30
        self.metric = metric or "rmse"
        self.kwargs = kwargs
        self.tuner = None

    def fit(self, X: pl.DataFrame, y: pl.DataFrame):
        from flaml import AutoML

        feat_cols = X.columns[2:]
        target_col = y.columns[-1]
        tuner_kwargs = {
            # Fixed settings
            "task": "regression",
            "split_type": "time",
            # Explicit settings
            "time_budget": self.time_budget,
            "max_iter": self.max_iter,
            "metric": self.metric,
            # Additional kwargs
            **self.kwargs,
        }

        tuner = AutoML(**tuner_kwargs)
        tuner.fit(
            X_train=X.select(feat_cols).to_pandas(),
            y_train=y.get_column(target_col).to_pandas(),
        )
        self.tuner = tuner
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self.tuner.predict(X=X.to_pandas())
