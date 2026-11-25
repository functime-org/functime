from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import polars as pl

from functime._core._narwhals import _to_polars_df, _validate_temporal_columns

_CALENDAR_ATTR_PERIODS: dict[str, float] = {
    "minute": 60.0,
    "hour": 24.0,
    "day": 31.0,
    "weekday": 7.0,
    "week": 53.0,
    "month": 12.0,
    "quarter": 4.0,
    "ordinal_day": 366.0,
}

CalendarAttr = Literal[
    "minute",
    "hour",
    "day",
    "weekday",
    "week",
    "month",
    "quarter",
    "year",
    "ordinal_day",
]


def calendar_features(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    attrs: Sequence[CalendarAttr] = ("month", "weekday", "day"),
    cyclical: bool = False,
    keep_original: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    """Extract numeric (and optionally sin/cos cyclical) calendar features.

    Parameters
    ----------
    data : DataFrame-like
        Input DataFrame with a Date or Datetime ``time_col``.
    time_col : str
        Name of the temporal column.
    entity_col : str, sequence of str, or None, default=None
        Optional entity identifier column(s) validated against schema.
    attrs : sequence of str, default=("month", "weekday", "day")
        Calendar attributes extracted via ``pl.col(time_col).dt.<attr>()``.
    cyclical : bool, default=False
        When ``True``, emit ``{attr}__sin`` and ``{attr}__cos`` instead of raw
        integers for periodic attributes (preserving raw ``year``).
    keep_original : bool, default=True
        Whether to keep input columns alongside extracted features.

    Returns
    -------
    tuple[pl.DataFrame, list[str]]
        Transformed DataFrame and list of newly generated column names.
    """
    df = _to_polars_df(data)
    entity_cols, _ = _validate_temporal_columns(
        df, time_col=time_col, entity_col=entity_col
    )
    time_dtype = df.schema[time_col]
    if not isinstance(time_dtype, pl.Date | pl.Datetime):
        raise TypeError(
            f"`calendar_features` requires Date or Datetime `time_col`; got {time_dtype}."
        )

    attrs_list = list(attrs)
    if not attrs_list:
        raise ValueError("`attrs` must contain at least one calendar attribute.")

    allowed = set(_CALENDAR_ATTR_PERIODS) | {"year"}
    unknown = [a for a in attrs_list if a not in allowed]
    if unknown:
        raise ValueError(
            f"Unsupported calendar attribute(s) {unknown}; allowed: {sorted(allowed)}."
        )

    exprs: list[pl.Expr] = []
    new_names: list[str] = []
    t_expr = pl.col(time_col)
    for attr in attrs_list:
        val_expr = getattr(t_expr.dt, attr)().cast(pl.Int32)
        if cyclical and attr in _CALENDAR_ATTR_PERIODS:
            period = _CALENDAR_ATTR_PERIODS[attr]
            angle = 2.0 * np.pi * val_expr.cast(pl.Float64) / period
            sin_name = f"{attr}__sin"
            cos_name = f"{attr}__cos"
            new_names.extend([sin_name, cos_name])
            exprs.append(angle.sin().alias(sin_name))
            exprs.append(angle.cos().alias(cos_name))
        else:
            new_names.append(attr)
            exprs.append(val_expr.alias(attr))

    if keep_original:
        return df.with_columns(exprs), new_names
    return df.with_columns(exprs).select(
        [*entity_cols, time_col, *new_names]
    ), new_names


def fourier_features(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    period: float = 12.0,
    order: int = 3,
    freq: str | None = None,
    keep_original: bool = True,
    state: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Compute phase-anchored Fourier terms continuous across fit and transform.

    Anchors the harmonic phase on the reference start timestamp ``t0`` and step
    unit learned during fit so future out-of-sample rows remain strictly in
    phase with training rows regardless of row subsetting.

    Parameters
    ----------
    data : DataFrame-like
        Input DataFrame.
    time_col : str
        Name of the timestamp or numeric time index column.
    entity_col : str, sequence of str, or None, default=None
        Optional entity identifier column(s) validated against schema.
    period : float, default=12.0
        Seasonal period in step units (e.g., ``12`` for monthly annual cycle,
        ``24`` for hourly diurnal cycle).
    order : int, default=3
        Number of sine/cosine harmonic pairs ``k = 1..order``.
    freq : str or None, default=None
        Optional Polars cadence hint (e.g., ``"1mo"``, ``"1d"``, ``"1h"``) for
        exact calendar-month stepping.
    keep_original : bool, default=True
        Whether to keep input columns alongside Fourier terms.
    state : dict[str, Any] or None, default=None
        Pre-fitted phase anchor state containing ``t0`` and ``step_us``.

    Returns
    -------
    tuple[pl.DataFrame, dict[str, Any]]
        Transformed DataFrame and fitted phase state dictionary.
    """
    if period <= 0:
        raise ValueError(f"`period` must be > 0, got {period}.")
    if order < 1:
        raise ValueError(f"`order` must be >= 1, got {order}.")

    df = _to_polars_df(data)
    entity_cols, _ = _validate_temporal_columns(
        df, time_col=time_col, entity_col=entity_col
    )

    if state is None:
        time_dtype = df.schema[time_col]
        t0 = df.get_column(time_col).min()
        step_us = 1.0
        if isinstance(time_dtype, pl.Date | pl.Datetime):
            diffs = (
                df.get_column(time_col)
                .unique()
                .sort()
                .cast(pl.Datetime("us"))
                .diff()
                .drop_nulls()
                .dt.total_microseconds()
            )
            positive_diffs = diffs.filter(diffs > 0)
            if len(positive_diffs) > 0:
                med = positive_diffs.median()
                if isinstance(med, int | float) and float(med) > 0:
                    step_us = float(med)
            else:
                step_us = 86_400_000_000.0
        fitted_state: dict[str, Any] = {
            "time_col": time_col,
            "period": float(period),
            "order": order,
            "freq": freq,
            "t0": t0,
            "time_dtype": time_dtype,
            "step_us": step_us,
        }
    else:
        fitted_state = state

    t0 = fitted_state["t0"]
    time_dtype = fitted_state["time_dtype"]
    step_us = fitted_state["step_us"]
    eff_period = fitted_state["period"]
    eff_order = fitted_state["order"]
    eff_freq = fitted_state["freq"]

    if isinstance(time_dtype, pl.Date | pl.Datetime):
        if eff_freq is not None and eff_freq.lower().endswith("mo"):
            n_mo = int(eff_freq.lower().rstrip("mo") or "1")
            dt = pl.col(time_col).cast(pl.Datetime("us"))
            step_expr = (
                (dt.dt.year() - t0.year) * 12 + (dt.dt.month() - t0.month)
            ).cast(pl.Float64) / float(n_mo)
        else:
            step_expr = (
                pl.col(time_col).cast(pl.Datetime("us"))
                - pl.lit(t0).cast(pl.Datetime("us"))
            ).dt.total_microseconds().cast(pl.Float64) / step_us
    else:
        step_expr = pl.col(time_col).cast(pl.Float64) - float(t0)

    p_tag = int(eff_period) if float(eff_period).is_integer() else eff_period
    exprs: list[pl.Expr] = []
    new_names: list[str] = []
    for k in range(1, eff_order + 1):
        angle = 2.0 * np.pi * k * step_expr / eff_period
        cos_name = f"cos_{p_tag}_{k}"
        sin_name = f"sin_{p_tag}_{k}"
        new_names.extend([cos_name, sin_name])
        exprs.append(angle.cos().alias(cos_name))
        exprs.append(angle.sin().alias(sin_name))

    fitted_state = {**fitted_state, "feature_names_out": new_names}
    if keep_original:
        return df.with_columns(exprs), fitted_state
    return df.with_columns(exprs).select(
        [*entity_cols, time_col, *new_names]
    ), fitted_state
