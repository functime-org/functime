from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast, get_origin

import polars as pl

from functime._frame import TemporalFrame, TemporalTransformSpec
from functime.sklearn._base import BaseTemporalTransformer

_T = TypeVar("_T", bound=BaseTemporalTransformer)
_BASE_PARAM_NAMES: frozenset[str] = frozenset({"time_col", "entity_col", "columns"})


def sklearn_transformer(
    target: Callable[..., Any] | type[Any],
) -> Callable[[type[_T]], type[_T]]:
    """Wrap a stateless transform or stateful core class as a sklearn estimator.

    Binds either a ``@temporal_transform`` expression function or a pure-Polars
    stateful core class implementing ``fit`` and ``transform`` (and optionally
    ``inverse_transform``) onto a ``BaseTemporalTransformer`` subclass in
    place.

    Parameters
    ----------
    target : callable or type
        Either a function decorated with ``@temporal_transform`` or a pure
        Polars stateful class exposing ``fit`` and ``transform`` methods.

    Returns
    -------
    callable
        Class decorator that wires ``__init__`` and temporal frame lifecycle
        hooks onto a ``BaseTemporalTransformer`` subclass.

    Notes
    -----
    Decorated classes must inherit explicitly from ``BaseTemporalTransformer``,
    declare extra estimator parameters as annotated class attributes with
    immutable defaults, and omit a manual ``__init__`` definition.
    """

    def decorator(cls: type[_T]) -> type[_T]:
        _validate_decorated_class(cls)
        all_params, extra_param_names = _extract_params(cls)

        spec = getattr(target, "_temporal_spec", None)
        if isinstance(spec, TemporalTransformSpec):
            _wire_stateless_transform(cls, spec, extra_param_names)
        elif (
            isinstance(target, type)
            and callable(getattr(target, "fit", None))
            and callable(getattr(target, "transform", None))
        ):
            _wire_stateful_transform(cls, target, extra_param_names)
        else:
            raise TypeError(
                "@sklearn_transformer target must be a @temporal_transform function "
                "or a stateful class defining `fit` and `transform`."
            )

        cls.__init__ = _build_init(cls, all_params)  # type: ignore[method-assign]
        return cls

    return decorator


def _validate_decorated_class(cls: Any) -> None:
    if (
        not isinstance(cls, type)
        or not issubclass(cls, BaseTemporalTransformer)
        or cls is BaseTemporalTransformer
    ):
        raise TypeError(
            "@sklearn_transformer requires a subclass of BaseTemporalTransformer "
            "(for CPython runtime safety, pickling, clone(), and static typing)."
        )
    if "__init__" in cls.__dict__:
        raise TypeError(
            f"{cls.__name__} cannot define __init__ when decorated with "
            "@sklearn_transformer; declare parameters as annotated class attributes instead."
        )


def _is_classvar_annotation(annotation: Any) -> bool:
    if isinstance(annotation, str):
        text = annotation.strip()
        return (
            text == "ClassVar"
            or text.startswith("ClassVar[")
            or ".ClassVar[" in text
            or text.endswith(".ClassVar")
        )
    origin = get_origin(annotation)
    return getattr(origin, "__name__", None) == "ClassVar" or (
        getattr(annotation, "__name__", None) == "ClassVar"
    )


def _extract_params(
    cls: type[BaseTemporalTransformer],
) -> tuple[list[tuple[str, Any, Any]], tuple[str, ...]]:
    collected: dict[str, tuple[Any, Any]] = {}
    base_defaults: dict[str, Any] = {
        "time_col": BaseTemporalTransformer.time_col,
        "entity_col": BaseTemporalTransformer.entity_col,
        "columns": BaseTemporalTransformer.columns,
    }
    mro_classes = [
        klass
        for klass in reversed(cls.__mro__)
        if isinstance(klass, type) and issubclass(klass, BaseTemporalTransformer)
    ]

    for klass in mro_classes:
        annotations = klass.__dict__.get("__annotations__", {})
        for name, ann in annotations.items():
            if name.startswith("_") or name.endswith("_"):
                continue
            if _is_classvar_annotation(ann):
                continue
            if name in klass.__dict__:
                default = klass.__dict__[name]
            elif name in collected:
                default = collected[name][1]
            elif name in base_defaults:
                default = base_defaults[name]
            else:
                raise TypeError(
                    f"Parameter {name!r} on {cls.__name__} must declare a default value "
                    "for scikit-learn compatibility."
                )

            if isinstance(default, (list, dict, set)):
                raise TypeError(
                    f"Parameter {name!r} on {cls.__name__} has mutable default "
                    f"{default!r}; use an immutable default such as tuple or None."
                )
            collected[name] = (ann, default)

    params = [(name, ann, default) for name, (ann, default) in collected.items()]
    extra_names = tuple(name for name in collected if name not in _BASE_PARAM_NAMES)
    return params, extra_names


def _build_init(
    cls: type[BaseTemporalTransformer],
    params: list[tuple[str, Any, Any]],
) -> Callable[..., None]:
    sig_parts = ["self"]
    assign_lines: list[str] = []
    namespace: dict[str, Any] = {}
    annotations: dict[str, Any] = {"return": None}

    for name, ann, default in params:
        if not name.isidentifier():
            raise ValueError(f"Invalid parameter identifier: {name!r}.")
        default_key = f"_d_{name}"
        namespace[default_key] = default
        sig_parts.append(f"{name}={default_key}")
        assign_lines.append(f"    self.{name} = {name}")
        annotations[name] = ann

    body = "\n".join(assign_lines) if assign_lines else "    pass"
    code = f"def __init__({', '.join(sig_parts)}):\n{body}\n"
    exec(code, namespace)

    init_fn: Callable[..., None] = namespace["__init__"]
    init_fn.__qualname__ = f"{cls.__qualname__}.__init__"
    init_fn.__module__ = cls.__module__
    init_fn.__annotations__ = annotations
    return init_fn


def _validate_extra_params_against_signature(
    cls: type[BaseTemporalTransformer],
    extra_param_names: tuple[str, ...],
    sig: inspect.Signature,
    *,
    ignored_params: frozenset[str],
    target_label: str,
) -> None:
    params = list(sig.parameters.values())
    has_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
    accepted = {
        p.name for p in params if p.kind is not inspect.Parameter.VAR_KEYWORD
    } - ignored_params
    required = {
        p.name
        for p in params
        if p.name not in ignored_params
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and p.default is inspect.Parameter.empty
    }

    extra_set = set(extra_param_names)
    if not has_varkw:
        unexpected = sorted(extra_set - accepted)
        if unexpected:
            raise TypeError(
                f"{cls.__name__} declares parameter(s) {unexpected} not accepted by {target_label}."
            )
    missing = sorted(required - extra_set)
    if missing:
        raise TypeError(
            f"{cls.__name__} is missing required parameter(s) {missing} of {target_label}."
        )


def _wire_stateless_transform(
    cls: type[BaseTemporalTransformer],
    spec: TemporalTransformSpec,
    extra_param_names: tuple[str, ...],
) -> None:
    sig = inspect.signature(spec.expr_fn)
    sig_params = list(sig.parameters.values())
    if not sig_params:
        raise TypeError(
            f"{spec.expr_fn.__name__} must accept `columns` as its first parameter."
        )
    ignored = {sig_params[0].name}
    if spec.needs_time_col:
        has_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig_params)
        if "time_col" not in sig.parameters and not has_varkw:
            raise TypeError(
                f"{spec.expr_fn.__name__} sets needs_time_col=True but does not accept `time_col`."
            )
        ignored.add("time_col")

    _validate_extra_params_against_signature(
        cls,
        extra_param_names,
        sig,
        ignored_params=frozenset(ignored),
        target_label=f"{spec.expr_fn.__name__}()",
    )

    if "_temporal_regime" not in cls.__dict__:
        cls._temporal_regime = "row_stateless"
    if "_allow_null_time" not in cls.__dict__:
        cls._allow_null_time = spec.allow_null_time
    if "_allow_null_entity" not in cls.__dict__:
        cls._allow_null_entity = spec.allow_null_entity

    def _transform_frame(
        self: BaseTemporalTransformer, frame: TemporalFrame
    ) -> TemporalFrame:
        kwargs = {name: getattr(self, name) for name in extra_param_names}
        exprs = (
            spec.expr_fn(frame.target_cols, time_col=frame.time_col, **kwargs)
            if spec.needs_time_col
            else spec.expr_fn(frame.target_cols, **kwargs)
        )
        return frame.with_exprs(exprs, chronological=spec.chronological)

    def _fit_frame(
        self: BaseTemporalTransformer, frame: TemporalFrame, y: Any = None
    ) -> None:
        out = self._transform_frame(frame)
        self.feature_names_out_ = tuple(out.lf.collect_schema().names())

    def _fit_transform_frame(
        self: BaseTemporalTransformer, frame: TemporalFrame, y: Any = None
    ) -> TemporalFrame:
        out = self._transform_frame(frame)
        self.feature_names_out_ = tuple(out.lf.collect_schema().names())
        return out

    if "_transform_frame" not in cls.__dict__:
        cls._transform_frame = _transform_frame  # type: ignore[method-assign]
    if "_fit_frame" not in cls.__dict__:
        cls._fit_frame = _fit_frame  # type: ignore[method-assign]
    if "_fit_transform_frame" not in cls.__dict__:
        cls._fit_transform_frame = _fit_transform_frame  # type: ignore[method-assign]


def _wire_stateful_transform(
    cls: type[BaseTemporalTransformer],
    core_cls: type[Any],
    extra_param_names: tuple[str, ...],
) -> None:
    if core_cls.__init__ is object.__init__:
        if extra_param_names:
            raise TypeError(
                f"{cls.__name__} declares parameter(s) {list(extra_param_names)} "
                f"not accepted by {core_cls.__name__}.__init__()."
            )
    else:
        init_sig = inspect.signature(core_cls.__init__)
        init_params = list(init_sig.parameters.values())
        ignored = frozenset({init_params[0].name}) if init_params else frozenset()
        _validate_extra_params_against_signature(
            cls,
            extra_param_names,
            init_sig,
            ignored_params=ignored,
            target_label=f"{core_cls.__name__}.__init__()",
        )

    fit_sig = inspect.signature(core_cls.fit)
    fit_params = fit_sig.parameters
    fit_has_varkw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in fit_params.values()
    )
    accepts_time = "time_col" in fit_params or fit_has_varkw
    accepts_entity = "entity_col" in fit_params or fit_has_varkw
    accepts_columns = "columns" in fit_params or fit_has_varkw
    accepts_y = "y" in fit_params or fit_has_varkw

    if "_temporal_regime" not in cls.__dict__:
        cls._temporal_regime = getattr(core_cls, "_temporal_regime", "row_stateful")
    if (
        "_allow_null_time" not in cls.__dict__
        and "_allow_null_time" in core_cls.__dict__
    ):
        cls._allow_null_time = bool(getattr(core_cls, "_allow_null_time", False))
    if (
        "_allow_null_entity" not in cls.__dict__
        and "_allow_null_entity" in core_cls.__dict__
    ):
        cls._allow_null_entity = bool(getattr(core_cls, "_allow_null_entity", False))

    def _fit_core(
        self: BaseTemporalTransformer, frame: TemporalFrame, y: Any = None
    ) -> Any:
        init_kwargs = {name: getattr(self, name) for name in extra_param_names}
        core = core_cls(**init_kwargs)
        fit_kwargs: dict[str, Any] = {}
        if accepts_time:
            fit_kwargs["time_col"] = frame.time_col
        if accepts_entity:
            fit_kwargs["entity_col"] = frame.entity_cols or None
        if accepts_columns:
            fit_kwargs["columns"] = frame.target_cols
        if accepts_y:
            fit_kwargs["y"] = y
        fitted = core.fit(frame, **fit_kwargs)
        resolved = core if fitted is None else fitted
        self._core_ = resolved  # type: ignore[attr-defined]
        return resolved

    def _fit_frame(
        self: BaseTemporalTransformer, frame: TemporalFrame, y: Any = None
    ) -> None:
        fitted = _fit_core(self, frame, y)
        names = getattr(fitted, "feature_names_out", None)
        if names is not None:
            self.feature_names_out_ = tuple(names)
        else:
            out = self._transform_frame(frame)
            self.feature_names_out_ = tuple(out.lf.collect_schema().names())

    def _transform_frame(
        self: BaseTemporalTransformer, frame: TemporalFrame
    ) -> TemporalFrame:
        core = cast(Any, self)._core_
        return _to_temporal_frame(frame, core.transform(frame))

    def _fit_transform_frame(
        self: BaseTemporalTransformer, frame: TemporalFrame, y: Any = None
    ) -> TemporalFrame:
        _fit_core(self, frame, y)
        return self._transform_frame(frame)

    if "_fit_frame" not in cls.__dict__:
        cls._fit_frame = _fit_frame  # type: ignore[method-assign]
    if "_transform_frame" not in cls.__dict__:
        cls._transform_frame = _transform_frame  # type: ignore[method-assign]
    if "_fit_transform_frame" not in cls.__dict__:
        cls._fit_transform_frame = _fit_transform_frame  # type: ignore[method-assign]

    if (
        callable(getattr(core_cls, "inverse_transform", None))
        and "_inverse_transform_frame" not in cls.__dict__
    ):

        def _inverse_transform_frame(
            self: BaseTemporalTransformer, frame: TemporalFrame
        ) -> TemporalFrame:
            core = cast(Any, self)._core_
            return _to_temporal_frame(frame, core.inverse_transform(frame))

        cls._inverse_transform_frame = _inverse_transform_frame  # type: ignore[method-assign]


def _to_temporal_frame(frame: TemporalFrame, result: Any) -> TemporalFrame:
    if isinstance(result, TemporalFrame):
        return result
    if isinstance(result, pl.LazyFrame):
        return frame.with_lazy(result, preserves_order=False)
    if isinstance(result, pl.DataFrame):
        return frame.with_lazy(result.lazy(), preserves_order=False)
    raise TypeError(
        "Core transformer methods must return a TemporalFrame, pl.LazyFrame, "
        f"or pl.DataFrame; got {type(result).__name__}."
    )
