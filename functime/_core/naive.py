from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from functime._core._narwhals import (
    _coerce_target_series,
    _to_polars_df,
    _validate_temporal_columns,
)


def naive_forecast(
    data: Any,
    y: Any = None,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict the last observed target value per entity (or globally).

    Parameters
    ----------
    data : DataFrame-like
        Feature/index DataFrame specifying ``time_col`` and optional ``entity_col``.
    y : array-like, Series, or None, default=None
        1D target values aligned with ``data`` during fitting (ignored when ``state`` is given).
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    state : dict[str, Any] or None, default=None
        Pre-fitted state dictionary. When ``None``, fit on ``(data, y)``.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        1D float64 predictions matching rows of ``data`` and fitted state.
    """
    return seasonal_naive_forecast(
        data,
        y,
        time_col=time_col,
        entity_col=entity_col,
        sp=1,
        state=state,
    )


def seasonal_naive_forecast(
    data: Any,
    y: Any = None,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    sp: int = 1,
    state: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict seasonal-naive target values per entity (or globally).

    Stores the final ``sp`` chronological observations per entity at fit time and
    cycles through them cyclically relative to each entity's training length or
    row position at prediction time.

    Parameters
    ----------
    data : DataFrame-like
        Feature/index DataFrame containing ``time_col`` and optional ``entity_col``.
    y : array-like, Series, or None, default=None
        1D target values aligned with ``data`` during fitting.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    sp : int, default=1
        Seasonal period (``1`` reduces to last-value naive persistence).
    state : dict[str, Any] or None, default=None
        Pre-fitted state dictionary. When ``None``, fit on ``(data, y)``.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        1D float64 predictions matching rows of ``data`` and fitted state.
    """
    if sp < 1:
        raise ValueError(f"`sp` must be >= 1, got {sp}.")

    df = _to_polars_df(data)
    entity_cols, _ = _validate_temporal_columns(
        df, time_col=time_col, entity_col=entity_col
    )

    if state is None:
        if y is None:
            raise ValueError(
                "`y` must be provided when fitting without pre-existing state."
            )
        target_s = _coerce_target_series(y, n_rows=df.height, name="__target__")
        work = (
            df.select([*entity_cols, time_col])
            .with_columns(target_s.cast(pl.Float64))
            .sort([*entity_cols, time_col], maintain_order=True)
        )
        last_global = float(
            work.get_column("__target__").drop_nulls().tail(1).item()
            if work.get_column("__target__").drop_nulls().len() > 0
            else 0.0
        )
        global_tail = work.get_column("__target__").tail(sp).to_list() or [last_global]

        entity_tails: dict[tuple[Any, ...], list[float]] = {}
        if entity_cols:
            grouped = work.group_by(entity_cols, maintain_order=True).agg(
                pl.col("__target__").tail(sp).alias("__tail")
            )
            for row in grouped.iter_rows(named=True):
                key = tuple(row[col] for col in entity_cols)
                vals = [
                    float(v) if v is not None else last_global for v in row["__tail"]
                ]
                entity_tails[key] = vals or [last_global]
        fitted_state: dict[str, Any] = {
            "time_col": time_col,
            "entity_cols": entity_cols,
            "sp": sp,
            "entity_tails": entity_tails,
            "global_tail": [
                float(v) if v is not None else last_global for v in global_tail
            ],
            "fallback": last_global,
        }
    else:
        fitted_state = state

    eff_entity_cols: list[str] = fitted_state["entity_cols"]
    tails: dict[tuple[Any, ...], list[float]] = fitted_state["entity_tails"]
    global_tail_vals: list[float] = fitted_state["global_tail"]
    fallback: float = fitted_state["fallback"]

    if not eff_entity_cols:
        order = (
            df.select(time_col)
            .with_row_index("__pos")
            .sort(time_col, maintain_order=True)
            .get_column("__pos")
            .to_list()
        )
        preds = np.empty(df.height, dtype=np.float64)
        m = len(global_tail_vals)
        for rank, orig_idx in enumerate(order):
            preds[orig_idx] = global_tail_vals[rank % m] if m > 0 else fallback
        return preds, fitted_state

    sorted_rows = (
        df.select([*eff_entity_cols, time_col])
        .with_row_index("__pos")
        .sort([*eff_entity_cols, time_col], maintain_order=True)
    )
    preds = np.empty(df.height, dtype=np.float64)
    seen_counts: dict[tuple[Any, ...], int] = {}
    for row in sorted_rows.iter_rows(named=True):
        key = tuple(row[c] for c in eff_entity_cols)
        idx = seen_counts.get(key, 0)
        seen_counts[key] = idx + 1
        seq = tails.get(key, global_tail_vals)
        preds[row["__pos"]] = seq[idx % len(seq)] if seq else fallback
    return preds, fitted_state
