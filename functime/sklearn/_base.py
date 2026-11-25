from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any, Literal, Self

import numpy as np
import polars as pl
from scipy import sparse
from sklearn import get_config
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from functime._core._narwhals import (
    _format_output,
    _is_dataframe_like,
    _to_polars_df,
)

# scikit-learn>=1.6 tag and validate_data APIs may lag in bundled static type stubs.
_tags_mod: Any = import_module("sklearn.utils._tags")
_val_mod: Any = import_module("sklearn.utils.validation")
_InputTags = _tags_mod.InputTags
_TargetTags = _tags_mod.TargetTags
_validate_data = _val_mod.validate_data

_SUPPORTED_OUTPUTS = ("default", "pandas", "polars")


def _check_non_empty_dense(X: Any) -> None:
    if sparse.issparse(X):
        raise TypeError(
            "Sparse data was passed, but dense tabular DataFrame data is required."
        )
    if hasattr(X, "shape"):
        shape = X.shape
        if len(shape) == 1:
            raise ValueError(
                f"Expected 2D DataFrame input, got 1D array with shape {shape}."
            )
        if len(shape) >= 2:
            if shape[0] == 0:
                raise ValueError(
                    f"0 sample(s) (shape={shape}) while a minimum of 1 is required."
                )
            if shape[1] == 0:
                raise ValueError(
                    f"0 feature(s) (shape={shape}) while a minimum of 1 is required."
                )
    elif hasattr(X, "__len__") and len(X) == 0:
        raise ValueError("0 sample(s) while a minimum of 1 is required.")


def _extract_polars_df(estimator: Any, X: Any, *, reset: bool) -> pl.DataFrame:
    _check_non_empty_dense(X)
    if not _is_dataframe_like(X):
        raise TypeError(
            f"{type(estimator).__name__} expects a Narwhals-compatible DataFrame "
            f"(Polars, Pandas, or Arrow) containing column {estimator.time_col!r}; "
            f"got {type(X).__name__}."
        )
    _validate_data(estimator, X, reset=reset, skip_check_array=True)
    return _to_polars_df(X)


class _TemporalTransformerMixin(
    TransformerMixin,
    BaseEstimator,
    auto_wrap_output_keys=None,
):
    time_col: str
    entity_col: str | Sequence[str] | None
    feature_names_out_: tuple[str, ...]

    def __sklearn_tags__(self) -> Any:
        parent_base: Any = super()
        tags = parent_base.__sklearn_tags__()
        tags.input_tags = _InputTags(
            two_d_array=True,
            categorical=True,
            string=True,
            allow_nan=True,
        )
        tags.requires_fit = True
        return tags

    def set_output(
        self, *, transform: Literal["default", "pandas", "polars"] | None = None
    ) -> Self:
        """Configure the output container type of ``transform`` and ``fit_transform``.

        Parameters
        ----------
        transform : {"default", "pandas", "polars"} or None, default=None
            Container returned by ``transform`` (``"default"`` returns a Polars
            DataFrame).

        Returns
        -------
        Self
            Estimator instance.
        """
        if transform is None:
            return self
        if transform not in _SUPPORTED_OUTPUTS:
            raise ValueError(
                f"output config must be in {list(_SUPPORTED_OUTPUTS)}, got {transform!r}."
            )
        self._sklearn_output_config = {"transform": transform}
        return self

    def _resolve_output_format(self) -> str:
        local = getattr(self, "_sklearn_output_config", {})
        if "transform" in local:
            return str(local["transform"])
        global_fmt = get_config().get("transform_output", "default")
        return str(global_fmt) if global_fmt in _SUPPORTED_OUTPUTS else "default"

    def _format_df_output(self, df: pl.DataFrame, original: Any) -> Any:
        return _format_output(df, self._resolve_output_format(), original=original)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored; included for scikit-learn signature compatibility.

        Returns
        -------
        np.ndarray of str
            Output column names produced by ``transform``.
        """
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


class _TemporalRegressorMixin(RegressorMixin, BaseEstimator):
    time_col: str
    entity_col: str | Sequence[str] | None

    def __sklearn_tags__(self) -> Any:
        parent_base: Any = super()
        tags = parent_base.__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.target_tags = _TargetTags(required=True, one_d_labels=True)
        tags.input_tags = _InputTags(
            two_d_array=True,
            categorical=True,
            string=True,
            allow_nan=True,
        )
        tags.requires_fit = True
        return tags
