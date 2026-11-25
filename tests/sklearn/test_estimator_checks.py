from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import parametrize_with_checks

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

_DATAFRAME_REASON = (
    "functime.sklearn estimators require a DataFrame (Polars/Pandas/Arrow) with explicit `time_col`; "
    "2D NumPy-only array checks do not apply."
)

_COMMON_NUMPY_2D_CHECKS = {
    "check_complex_data": _DATAFRAME_REASON,
    "check_dict_unchanged": _DATAFRAME_REASON,
    "check_dont_overwrite_parameters": _DATAFRAME_REASON,
    "check_dtype_object": _DATAFRAME_REASON,
    "check_estimators_dtypes": _DATAFRAME_REASON,
    "check_estimators_fit_returns_self": _DATAFRAME_REASON,
    "check_estimators_overwrite_params": _DATAFRAME_REASON,
    "check_estimators_pickle": _DATAFRAME_REASON,
    "check_f_contiguous_array_estimator": _DATAFRAME_REASON,
    "check_fit2d_1feature": _DATAFRAME_REASON,
    "check_fit2d_1sample": _DATAFRAME_REASON,
    "check_fit2d_predict1d": _DATAFRAME_REASON,
    "check_fit_check_is_fitted": _DATAFRAME_REASON,
    "check_fit_idempotent": _DATAFRAME_REASON,
    "check_fit_score_takes_y": _DATAFRAME_REASON,
    "check_methods_sample_order_invariance": _DATAFRAME_REASON,
    "check_methods_subset_invariance": _DATAFRAME_REASON,
    "check_n_features_in": _DATAFRAME_REASON,
    "check_n_features_in_after_fitting": _DATAFRAME_REASON,
    "check_pipeline_consistency": _DATAFRAME_REASON,
    "check_positive_only_tag_during_fit": _DATAFRAME_REASON,
    "check_readonly_memmap_input": _DATAFRAME_REASON,
}

_TRANSFORMER_ONLY_NUMPY_CHECKS = {
    **_COMMON_NUMPY_2D_CHECKS,
    "check_transformer_data_not_an_array": _DATAFRAME_REASON,
    "check_transformer_general": _DATAFRAME_REASON,
    "check_transformer_preserve_dtypes": _DATAFRAME_REASON,
}

_REGRESSOR_ONLY_NUMPY_CHECKS = {
    **_COMMON_NUMPY_2D_CHECKS,
    "check_regressor_data_not_an_array": _DATAFRAME_REASON,
    "check_regressors_int": _DATAFRAME_REASON,
    "check_regressors_no_decision_function": _DATAFRAME_REASON,
    "check_regressors_train": _DATAFRAME_REASON,
    "check_requires_y_none": _DATAFRAME_REASON,
    "check_supervised_y_2d": _DATAFRAME_REASON,
    "check_supervised_y_no_nan": _DATAFRAME_REASON,
}


def _expected_failed_checks(estimator: BaseEstimator) -> dict[str, str]:
    if isinstance(estimator, (NaiveForecaster, SeasonalNaiveForecaster)):
        return dict(_REGRESSOR_ONLY_NUMPY_CHECKS)
    return dict(_TRANSFORMER_ONLY_NUMPY_CHECKS)


_ESTIMATORS = [
    LagTransformer(),
    RollingTransformer(),
    DiffTransformer(),
    PanelStandardScaler(),
    DetrendTransformer(),
    CalendarTransformer(),
    FourierTransformer(),
    FractionalDiffTransformer(),
    TemporalImputer(),
    PanelFeatureAdapter(),
    NaiveForecaster(),
    SeasonalNaiveForecaster(),
]


@parametrize_with_checks(
    _ESTIMATORS,
    expected_failed_checks=_expected_failed_checks,
)
def test_functime_sklearn_estimator_checks(estimator, check):
    check(estimator)
