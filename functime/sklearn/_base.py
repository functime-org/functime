from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import math
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, Literal, Self, cast

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
import polars as pl
import sklearn.utils as _sk_utils
import sklearn.utils.validation as _sk_validation
from scipy import sparse
from sklearn import get_config
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from functime._frame import (
    TemporalFrame,
    is_frame_like,
    to_polars_lazy,
    validate_temporal_params,
)

TemporalRegime = Literal["row_stateless", "row_stateful", "entity_reducing"]


class _FunctimeHTMLReprMixin:
    _doc_link_module: str = "functime"

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        estimator = cast(BaseEstimator, self)
        params = estimator.get_params(deep=False)
        init_sig = inspect.signature(self.__init__).parameters
        init_defaults = {name: param.default for name, param in init_sig.items()}

        if not any(
            _needs_safe_display(val)
            for val in (*params.values(), *init_defaults.values())
        ):
            return BaseEstimator.__repr__(estimator, N_CHAR_MAX=N_CHAR_MAX)

        # Bypass sklearn's _EstimatorPrettyPrinter when params include Polars expressions or callables,
        # since sklearn invokes `default == inspect._empty` (TypeError on pl.Expr) and raw hex reprs.
        print_changed_only = bool(get_config().get("print_changed_only", True))
        ordered_names = [name for name in init_defaults if name in params]
        ordered_names.extend(name for name in params if name not in init_defaults)

        rendered: list[str] = []
        for name in ordered_names:
            val = params[name]
            is_changed = name not in init_defaults or not _safe_param_equal(
                val, init_defaults[name]
            )
            if not print_changed_only or is_changed:
                rendered.append(f"{name}={_format_param_value(val)!r}")

        text = f"{self.__class__.__name__}({', '.join(rendered)})"
        if len("".join(text.split())) > N_CHAR_MAX and N_CHAR_MAX > 10:
            text = f"{text[: max(0, N_CHAR_MAX - 4)]}...)"
        return text

    @property
    def _doc_link_template(self) -> str:
        return (
            "https://functime-org.github.io/functime/api-reference/"
            "{page}/#{estimator_module}.{estimator_name}"
        )

    def _doc_link_url_param_generator(self) -> dict[str, str]:
        parts = [
            part
            for part in self.__class__.__module__.split(".")
            if not part.startswith("_")
        ]
        page = "-".join(parts[1:]).replace("_", "-") if len(parts) > 1 else "index"
        return {
            "page": page,
            "estimator_module": ".".join(parts),
            "estimator_name": self.__class__.__name__,
        }

    @available_if(lambda _est: _ParamsDict is not None)
    def _get_params_html(self, deep: bool = True, doc_link: str = "") -> Any:
        estimator = cast(BaseEstimator, self)
        out = estimator.get_params(deep=deep)
        init_sig = inspect.signature(self.__init__).parameters
        init_defaults = {name: param.default for name, param in init_sig.items()}

        ordered_names = [name for name in init_defaults if name in out]
        ordered_names.extend(name for name in out if name not in init_defaults)

        non_default_params: list[str] = []
        default_params: list[str] = []
        for name in ordered_names:
            val = out[name]
            if name not in init_defaults or not _safe_param_equal(
                val, init_defaults[name]
            ):
                non_default_params.append(name)
            else:
                default_params.append(name)

        formatted = {
            name: _format_param_value(out[name])
            for name in (*non_default_params, *default_params)
        }
        # estimator_class/doc_link were added to ParamsDict after scikit-learn 1.7.
        extra = {"estimator_class": self.__class__, "doc_link": doc_link}
        return _ParamsDict(
            params=formatted,
            non_default=tuple(non_default_params),
            **{k: v for k, v in extra.items() if k in _PARAMS_DICT_KWARGS},
        )


class _DataFrameTransformerMixin(
    _FunctimeHTMLReprMixin, TransformerMixin, BaseEstimator, auto_wrap_output_keys=None
):
    feature_names_out_: tuple[str, ...]
    _sklearn_output_config: dict[str, str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        kwargs.setdefault("auto_wrap_output_keys", None)
        super().__init_subclass__(**kwargs)

    def __sklearn_tags__(self) -> Any:
        tags = cast(Any, super()).__sklearn_tags__()
        tags.input_tags = _InputTags(
            two_d_array=True,
            categorical=True,
            string=True,
            allow_nan=True,
        )
        tags.requires_fit = True
        return tags

    def fit(self, X: Any, y: Any = None) -> Self:
        """Fit the transformer on an input DataFrame.

        Parameters
        ----------
        X : DataFrame-like
            Input Polars, Pandas, Arrow, or Narwhals DataFrame or LazyFrame.
        y : Any, default=None
            Optional target ignored by unsupervised transformers.

        Returns
        -------
        Self
            The fitted estimator instance.
        """
        self._validate_params()
        df, _ = self._validate_and_to_polars(X, reset=True)
        self.__dict__.pop("feature_names_out_", None)
        self._fit_df(df, y)
        if "feature_names_out_" not in self.__dict__:
            self.feature_names_out_ = tuple(df.columns)
        return self

    def transform(self, X: Any) -> Any:
        """Transform an input DataFrame using the fitted transformer state.

        Parameters
        ----------
        X : DataFrame-like
            Input Polars, Pandas, Arrow, or Narwhals DataFrame or LazyFrame.

        Returns
        -------
        DataFrame
            Transformed output in the configured container format.
        """
        check_is_fitted(self)
        df, pandas_index = self._validate_and_to_polars(X, reset=False)
        out = self._transform_df(df)
        return self._format_df_output(out, pandas_index=pandas_index)

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Fit to data and return the transformed DataFrame in a single pass.

        Parameters
        ----------
        X : DataFrame-like
            Input Polars, Pandas, Arrow, or Narwhals DataFrame or LazyFrame.
        y : Any, default=None
            Optional target ignored by unsupervised transformers.
        **fit_params : Any
            Additional parameters accepted for scikit-learn compatibility.

        Returns
        -------
        DataFrame
            Transformed output in the configured container format.
        """
        self._validate_params()
        df, pandas_index = self._validate_and_to_polars(X, reset=True)
        self.__dict__.pop("feature_names_out_", None)
        out = self._fit_transform_df(df, y)
        self.feature_names_out_ = tuple(out.columns)
        return self._format_df_output(out, pandas_index=pandas_index)

    def get_feature_names_out(
        self, input_features: Sequence[str] | np.ndarray | None = None
    ) -> np.ndarray:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names validated against ``feature_names_in_`` when
            provided.

        Returns
        -------
        np.ndarray
            Array of output column names produced by ``transform``.
        """
        check_is_fitted(self, "feature_names_out_")
        if input_features is not None:
            _check_feature_names_in(self, input_features)
        return np.asarray(self.feature_names_out_, dtype=object)

    def set_output(
        self, *, transform: Literal["default", "pandas", "polars"] | None = None
    ) -> Self:
        """Configure the output container returned by `transform` and `fit_transform`.

        Parameters
        ----------
        transform : {"default", "pandas", "polars"} or None, default=None
            Target output container type. Passing ``None`` leaves the current
            configuration unchanged.

        Returns
        -------
        Self
            The estimator instance.
        """
        if transform is None:
            return self
        if transform not in ("default", "pandas", "polars"):
            raise ValueError(
                f"Unsupported output container {transform!r}. Expected 'default', 'polars', or 'pandas'."
            )
        self._sklearn_output_config = {"transform": transform}
        return self

    def _validate_params(self) -> None:
        return None

    def _validate_and_to_polars(
        self, X: Any, *, reset: bool
    ) -> tuple[pl.DataFrame, Any]:
        df, pandas_index = _coerce_nonempty_polars_df(X)
        _validate_data(self, df, reset=reset, skip_check_array=True)
        return df, pandas_index

    def _resolve_output_format(self) -> str:
        config = getattr(self, "_sklearn_output_config", None)
        if isinstance(config, dict) and "transform" in config:
            return config["transform"]
        return str(get_config().get("transform_output", "default"))

    def _format_df_output(self, df: pl.DataFrame, *, pandas_index: Any = None) -> Any:
        return _format_sklearn_output(
            df, self._resolve_output_format(), pandas_index=pandas_index
        )

    def _fit_df(self, df: pl.DataFrame, y: Any = None) -> None:
        raise NotImplementedError

    def _transform_df(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def _fit_transform_df(self, df: pl.DataFrame, y: Any = None) -> pl.DataFrame:
        self._fit_df(df, y)
        return self._transform_df(df)


class BaseTemporalTransformer(_DataFrameTransformerMixin):
    """Base class for scikit-learn-compatible temporal transformers.

    Operates uniformly across univariate time series (``entity_col=None``)
    and multi-entity panels (``entity_col=...``) while keeping internal
    computations lazy in Polars until collecting at the scikit-learn boundary.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp column.
    entity_col : str or sequence of str or None, default=None
        Column name(s) identifying series entities, or ``None`` for univariate
        time series.

    Notes
    -----
    All temporal transformer estimators—including classes wrapped by
    ``@sklearn_transformer``—must inherit explicitly from
    ``BaseTemporalTransformer`` for three reasons:

    1. **Python runtime, pickling, and cloning:** Mutating ``__bases__``
       raises ``TypeError`` in CPython because the C-level deallocator differs
       from ``object``, and returning a dynamically synthesized wrapper class
       breaks ``pickle``, ``sklearn.base.clone``, and ``super()``. Explicit
       inheritance allows decorators to configure the class cleanly in place.
    2. **Static typing and IDE support:** PEP 681 ``@dataclass_transform``
       describes generated ``__init__`` signatures, but cannot inject
       inherited methods such as ``fit``, ``transform``, ``fit_transform``,
       ``set_output``, or ``get_feature_names_out`` onto an unrelated class.
    3. **Immediate scikit-learn clarity:** The class signature makes
       ``BaseEstimator`` and ``TransformerMixin`` compatibility explicit for
       ``Pipeline``, ``ColumnTransformer``, and ``FeatureUnion`` usage.
    """

    _temporal_regime: ClassVar[TemporalRegime] = "row_stateless"
    _allow_null_time: ClassVar[bool] = False
    _allow_null_entity: ClassVar[bool] = False

    time_col: str
    entity_col: str | Sequence[str] | None
    entity_cols_: tuple[str, ...]
    columns_: tuple[str, ...]

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
    ) -> None:
        self.time_col = time_col
        self.entity_col = entity_col

    def fit(self, X: Any, y: Any = None) -> Self:
        """Fit the temporal transformer on an input DataFrame.

        Parameters
        ----------
        X : DataFrame-like
            Input Polars, Pandas, Arrow, or Narwhals DataFrame or LazyFrame.
        y : Any, default=None
            Optional target ignored by unsupervised transformers.

        Returns
        -------
        Self
            The fitted estimator instance.
        """
        self._validate_params()
        with _clear_fitted_state_on_error(self):
            frame, _ = self._validate_and_parse_frame(X, reset=True)
            self.__dict__.pop("feature_names_out_", None)
            self._fit_frame(frame, y)
            if "feature_names_out_" not in self.__dict__:
                # Resolve from the lazy output schema so fit() agrees with fit_transform() for column-adding cores.
                self.feature_names_out_ = tuple(
                    self._transform_frame(frame).lf.collect_schema().names()
                )
        return self

    def transform(self, X: Any) -> Any:
        """Transform a temporal DataFrame using the fitted transformer state.

        Parameters
        ----------
        X : DataFrame-like
            Input Polars, Pandas, Arrow, or Narwhals DataFrame or LazyFrame.

        Returns
        -------
        DataFrame
            Transformed output in the configured container format.
        """
        check_is_fitted(self)
        frame, pandas_index = self._validate_and_parse_frame(X, reset=False)
        out_df = self._transform_frame(frame).collect()
        return self._format_df_output(out_df, pandas_index=pandas_index)

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Fit to temporal data and return the transformed DataFrame in a single pass.

        Parameters
        ----------
        X : DataFrame-like
            Input Polars, Pandas, Arrow, or Narwhals DataFrame or LazyFrame.
        y : Any, default=None
            Optional target ignored by unsupervised transformers.
        **fit_params : Any
            Additional parameters accepted for scikit-learn compatibility.

        Returns
        -------
        DataFrame
            Transformed output in the configured container format.
        """
        self._validate_params()
        with _clear_fitted_state_on_error(self):
            frame, pandas_index = self._validate_and_parse_frame(X, reset=True)
            self.__dict__.pop("feature_names_out_", None)
            out_df = self._fit_transform_frame(frame, y).collect()
            self.feature_names_out_ = tuple(out_df.columns)
        return self._format_df_output(out_df, pandas_index=pandas_index)

    def _validate_params(self) -> None:
        validate_temporal_params(self.time_col, self.entity_col)

    def _validate_and_parse_frame(
        self, X: Any, *, reset: bool
    ) -> tuple[TemporalFrame, Any]:
        df, pandas_index = _coerce_nonempty_polars_df(X)
        if reset:
            # Parse temporal columns before validate_data so failed fit leaves no partial fitted attrs.
            frame = TemporalFrame.parse(
                df,
                time_col=self.time_col,
                entity_col=self.entity_col,
                columns=getattr(self, "columns", None),
                allow_null_time=self._allow_null_time,
                allow_null_entity=self._allow_null_entity,
            )
            _validate_data(self, df, reset=True, skip_check_array=True)
            self.entity_cols_ = frame.entity_cols
            self.columns_ = frame.target_cols
            return frame, pandas_index

        _validate_data(self, df, reset=False, skip_check_array=True)
        frame = TemporalFrame.parse(
            df,
            time_col=self.time_col,
            entity_col=self.entity_col,
            columns=self.columns_,
            allow_null_time=self._allow_null_time,
            allow_null_entity=self._allow_null_entity,
        )
        return frame, pandas_index

    def _fit_frame(self, frame: TemporalFrame, y: Any = None) -> None:
        raise NotImplementedError

    def _transform_frame(self, frame: TemporalFrame) -> TemporalFrame:
        raise NotImplementedError

    def _fit_transform_frame(
        self, frame: TemporalFrame, y: Any = None
    ) -> TemporalFrame:
        self._fit_frame(frame, y)
        return self._transform_frame(frame)


class _TemporalRegressorMixin(_FunctimeHTMLReprMixin, RegressorMixin, BaseEstimator):
    time_col: str
    entity_col: str | Sequence[str] | None

    def __sklearn_tags__(self) -> Any:
        tags = cast(Any, super()).__sklearn_tags__()
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


def _coerce_nonempty_polars_df(X: Any) -> tuple[pl.DataFrame, Any]:
    if sparse.issparse(X):
        raise TypeError(
            "Sparse matrices and arrays are not supported; pass a dense DataFrame."
        )
    if nwd.is_into_series(X) or (
        not is_frame_like(X) and hasattr(X, "ndim") and X.ndim == 1
    ):
        raise ValueError(
            "Expected a 2D DataFrame; got 1D input. Reshape your data or pass a DataFrame."
        )
    if not is_frame_like(X):
        # Preserve sklearn's empty-input ValueError contract before rejecting non-DataFrames.
        if hasattr(X, "shape") and len(X.shape) == 2:
            if X.shape[0] == 0:
                raise ValueError(
                    f"Found array with 0 sample(s) (shape={X.shape}) while a minimum of 1 is required."
                )
            if X.shape[1] == 0:
                raise ValueError(
                    f"Found array with 0 feature(s) (shape={X.shape}) while a minimum of 1 is required."
                )
        raise TypeError(
            f"Expected a Polars, Pandas, Arrow, or Narwhals DataFrame; got {type(X).__name__}."
        )

    pandas_index = _capture_pandas_index(X)
    # Eagerly convert LazyFrame/Narwhals inputs before validate_data so feature names are always recorded.
    if isinstance(X, pl.DataFrame):
        df = X
    elif pandas_index is not None:
        # The index is restored on output, so the core's dropped-index guard does not apply here.
        df = nw.from_native(X, eager_only=True).to_polars()
    else:
        df = to_polars_lazy(X).collect()
    if df.height == 0:
        raise ValueError(
            f"Found DataFrame with 0 sample(s) (shape={df.shape}) while a minimum of 1 is required."
        )
    if df.width == 0:
        raise ValueError(
            f"Found DataFrame with 0 feature(s) (shape={df.shape}) while a minimum of 1 is required."
        )
    return df, pandas_index


@contextlib.contextmanager
def _clear_fitted_state_on_error(estimator: Any) -> Iterator[None]:
    try:
        yield
    except BaseException:
        # A partially fitted estimator would pass check_is_fitted and fail later with AttributeError.
        for attr in [
            k for k in vars(estimator) if k.endswith("_") and not k.startswith("_")
        ]:
            delattr(estimator, attr)
        raise


def _capture_pandas_index(X: Any) -> Any:
    if isinstance(X, nw.DataFrame):
        native = X.to_native()
        return nw.maybe_get_index(X) if nwd.is_pandas_like_dataframe(native) else None
    if not isinstance(X, nw.LazyFrame) and nwd.is_pandas_like_dataframe(X):
        return nw.maybe_get_index(nw.from_native(X, eager_only=True))
    return None


def _format_sklearn_output(
    df: pl.DataFrame, output_format: str, *, pandas_index: Any = None
) -> Any:
    if output_format in ("default", "polars"):
        return df
    if output_format == "pandas":
        pdf = df.to_pandas()
        if pandas_index is not None and len(pandas_index) == len(pdf):
            # Restore original Pandas Index/MultiIndex without importing pandas in functime._frame.
            pdf.index = pandas_index
        return pdf
    raise ValueError(
        f"Unsupported output container {output_format!r}. Expected 'default', 'polars', or 'pandas'."
    )


class _DisplayToken:
    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        self._text = text

    def __repr__(self) -> str:
        return self._text


def _is_scalar_nan(val: Any) -> bool:
    return isinstance(val, (float, np.floating)) and math.isnan(float(val))


def _format_callable_label(obj: Any) -> str:
    if isinstance(obj, functools.partial):
        inner = _format_callable_label(obj.func)
        parts = [inner]
        parts.extend(repr(_format_param_value(arg)) for arg in obj.args)
        parts.extend(f"{k}={_format_param_value(v)!r}" for k, v in obj.keywords.items())
        return f"partial({', '.join(parts)})"
    if isinstance(obj, type):
        return obj.__name__
    name = getattr(obj, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return f"{obj.__class__.__name__}(...)"


def _format_param_value(val: Any) -> Any:
    if isinstance(val, BaseEstimator):
        return val
    if isinstance(val, pl.Expr):
        return _DisplayToken(f"pl.Expr({val})")
    if isinstance(val, pl.DataFrame):
        return _DisplayToken(f"pl.DataFrame({val.shape[0]}x{val.shape[1]})")
    if isinstance(val, pl.Series):
        return _DisplayToken(
            f"pl.Series({val.name!r}, len={len(val)}, dtype={val.dtype})"
        )
    if isinstance(val, functools.partial) or (
        callable(val) and not isinstance(val, type)
    ):
        return _DisplayToken(_format_callable_label(val))
    if isinstance(val, tuple):
        return tuple(_format_param_value(item) for item in val)
    if isinstance(val, list):
        return [_format_param_value(item) for item in val]
    if isinstance(val, dict):
        return {k: _format_param_value(v) for k, v in val.items()}
    return val


def _needs_safe_display(val: Any) -> bool:
    if isinstance(val, BaseEstimator):
        return False
    if isinstance(val, (pl.Expr, pl.DataFrame, pl.Series, functools.partial)):
        return True
    if callable(val) and not isinstance(val, type):
        return True
    if isinstance(val, (tuple, list)):
        return any(_needs_safe_display(item) for item in val)
    if isinstance(val, dict):
        return any(_needs_safe_display(item) for item in val.values())
    return False


def _safe_param_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if a is inspect._empty or b is inspect._empty:
        return False
    if isinstance(a, pl.Expr) or isinstance(b, pl.Expr):
        return isinstance(a, pl.Expr) and isinstance(b, pl.Expr) and a.meta.eq(b)
    if isinstance(a, pl.DataFrame) or isinstance(b, pl.DataFrame):
        return (
            isinstance(a, pl.DataFrame) and isinstance(b, pl.DataFrame) and a.equals(b)
        )
    if isinstance(a, pl.Series) or isinstance(b, pl.Series):
        return isinstance(a, pl.Series) and isinstance(b, pl.Series) and a.equals(b)
    if isinstance(a, BaseEstimator) or isinstance(b, BaseEstimator):
        return (
            isinstance(a, BaseEstimator)
            and isinstance(b, BaseEstimator)
            and type(a) is type(b)
            and _safe_param_equal(a.get_params(deep=False), b.get_params(deep=False))
        )
    if isinstance(a, functools.partial) or isinstance(b, functools.partial):
        return (
            isinstance(a, functools.partial)
            and isinstance(b, functools.partial)
            and _safe_param_equal(a.func, b.func)
            and _safe_param_equal(a.args, b.args)
            and _safe_param_equal(a.keywords, b.keywords)
        )
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_safe_param_equal(x, y) for x, y in zip(a, b, strict=True))
        )
    if isinstance(a, dict) or isinstance(b, dict):
        return (
            type(a) is type(b)
            and a.keys() == b.keys()
            and all(_safe_param_equal(a[k], b[k]) for k in a)
        )
    if _is_scalar_nan(a) and _is_scalar_nan(b):
        return True
    if callable(a) or callable(b):
        return bool(a == b)
    try:
        return np.array_equal(a, b)
    except Exception:
        try:
            return bool(a == b)
        except Exception:
            return False


def _load_visual_block() -> Any:
    for mod_name in (
        "sklearn.utils._repr_html.estimator",
        "sklearn.utils._estimator_html_repr",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        block_cls = getattr(mod, "_VisualBlock", None)
        if block_cls is not None:
            return block_cls
    return None


def _load_params_dict() -> Any:
    try:
        mod = importlib.import_module("sklearn.utils._repr_html.params")
    except ImportError:
        return None
    return getattr(mod, "ParamsDict", None)


_VisualBlock: Any = _load_visual_block()
_ParamsDict: Any = _load_params_dict()
_PARAMS_DICT_KWARGS: frozenset[str] = (
    frozenset(inspect.signature(_ParamsDict).parameters)
    if _ParamsDict is not None
    else frozenset()
)
# Pyrefly's bundled scikit-learn stubs predate 1.6 tags/validate_data APIs.
_InputTags: Any = cast(Any, _sk_utils).InputTags
_TargetTags: Any = cast(Any, _sk_utils).TargetTags
_check_feature_names_in: Any = cast(Any, _sk_validation)._check_feature_names_in
_validate_data: Any = cast(Any, _sk_validation).validate_data
