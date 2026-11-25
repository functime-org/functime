from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import polars as pl
import polars.selectors as cs

from functime._core._narwhals import _to_polars_df, _validate_temporal_columns

_ROW_IDX = "__functime_row_idx__"


def _with_temporal_sort(
    df: pl.DataFrame,
    entity_cols: list[str],
    time_col: str,
) -> pl.DataFrame:
    sort_keys = [*entity_cols, time_col]
    return df.with_row_index(_ROW_IDX).sort(sort_keys, maintain_order=True)


def _restore_row_order(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(_ROW_IDX).drop(_ROW_IDX)


def _over_entity(expr: pl.Expr, entity_cols: list[str]) -> pl.Expr:
    return expr.over(entity_cols) if entity_cols else expr


def _numeric_target_cols(df: pl.DataFrame, target_cols: Sequence[str]) -> list[str]:
    numeric_set = set(df.select(cs.numeric()).columns)
    return [col for col in target_cols if col in numeric_set]


def _relative_time_expr(time_col: str, t0: Any, time_dtype: pl.DataType) -> pl.Expr:
    if isinstance(time_dtype, pl.Date | pl.Datetime):
        # Convert to days so long historical spans stay numerically conditioned for OLS.
        return (
            pl.col(time_col).cast(pl.Datetime("us"))
            - pl.lit(t0).cast(pl.Datetime("us"))
        ).dt.total_microseconds().cast(pl.Float64) / 86_400_000_000.0
    return pl.col(time_col).cast(pl.Float64) - float(t0)


def lag_features(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    lags: Sequence[int] = (1,),
    columns: Sequence[str] | None = None,
    keep_original: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    """Compute lagged columns per entity while preserving row order.

    Parameters
    ----------
    data : DataFrame-like
        Input DataFrame.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    lags : sequence of int, default=(1,)
        Positive integer lag offsets to compute.
    columns : sequence of str or None, default=None
        Subset of columns to lag. Defaults to all non-index numeric columns.
    keep_original : bool, default=True
        Whether to keep existing columns alongside generated lag columns.

    Returns
    -------
    tuple[pl.DataFrame, list[str]]
        Transformed DataFrame and list of newly generated lag column names.
    """
    df = _to_polars_df(data)
    entity_cols, target_cols = _validate_temporal_columns(
        df, time_col=time_col, entity_col=entity_col, columns=columns
    )
    lag_cols = _numeric_target_cols(df, target_cols) if columns is None else target_cols
    lags_list = list(lags)
    if not lags_list or any(lag <= 0 for lag in lags_list):
        raise ValueError(
            f"`lags` must be a non-empty sequence of positive ints; got {lags}."
        )

    exprs: list[pl.Expr] = []
    new_names: list[str] = []
    for col in lag_cols:
        for lag in lags_list:
            alias = f"{col}__lag_{lag}"
            new_names.append(alias)
            exprs.append(_over_entity(pl.col(col).shift(lag), entity_cols).alias(alias))

    sorted_df = _with_temporal_sort(df, entity_cols, time_col)
    if keep_original:
        result = _restore_row_order(sorted_df.with_columns(exprs))
    else:
        kept_idx = [*entity_cols, time_col]
        result = _restore_row_order(
            sorted_df.with_columns(exprs).select([_ROW_IDX, *kept_idx, *new_names])
        )
    return result, new_names


def rolling_features(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    window_sizes: Sequence[int] = (3,),
    stats: Sequence[Literal["mean", "std", "min", "max", "median", "sum"]] = ("mean",),
    min_samples: int = 1,
    shift: int = 1,
    columns: Sequence[str] | None = None,
    keep_original: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    """Compute causal rolling window summary features in pure Polars.

    By default ``shift=1`` lags series by one step before applying the rolling
    window so row ``t`` only depends on strictly past observations.

    Parameters
    ----------
    data : DataFrame-like
        Input DataFrame.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    window_sizes : sequence of int, default=(3,)
        Window lengths (in rows).
    stats : sequence of {"mean", "std", "min", "max", "median", "sum"}, default=("mean",)
        Rolling aggregations to compute.
    min_samples : int, default=1
        Minimum non-null observations within a window before emitting null.
    shift : int, default=1
        Causal pre-shift applied before the rolling reduction (``>= 0``).
    columns : sequence of str or None, default=None
        Columns to aggregate. Defaults to non-index numeric columns.
    keep_original : bool, default=True
        Whether to retain input columns alongside generated rolling features.

    Returns
    -------
    tuple[pl.DataFrame, list[str]]
        Transformed DataFrame and list of newly generated rolling column names.
    """
    df = _to_polars_df(data)
    entity_cols, target_cols = _validate_temporal_columns(
        df, time_col=time_col, entity_col=entity_col, columns=columns
    )
    roll_cols = (
        _numeric_target_cols(df, target_cols) if columns is None else target_cols
    )
    win_list = list(window_sizes)
    stats_list = list(stats)
    if not win_list or any(w <= 0 for w in win_list):
        raise ValueError(
            f"`window_sizes` must contain positive ints; got {window_sizes}."
        )
    if shift < 0:
        raise ValueError(f"`shift` must be >= 0, got {shift}.")

    allowed = {"mean", "std", "min", "max", "median", "sum"}
    unknown = [s for s in stats_list if s not in allowed]
    if not stats_list or unknown:
        raise ValueError(
            f"Unsupported rolling stat(s) {unknown}; allowed: {sorted(allowed)}."
        )

    exprs: list[pl.Expr] = []
    new_names: list[str] = []
    for col in roll_cols:
        base = pl.col(col).shift(shift) if shift > 0 else pl.col(col)
        for w in win_list:
            for stat in stats_list:
                alias = f"{col}__rolling_{stat}_{w}"
                new_names.append(alias)
                ms = min(min_samples, w)
                if stat == "mean":
                    r_expr = base.rolling_mean(window_size=w, min_samples=ms)
                elif stat == "std":
                    r_expr = base.rolling_std(window_size=w, min_samples=ms)
                elif stat == "min":
                    r_expr = base.rolling_min(window_size=w, min_samples=ms)
                elif stat == "max":
                    r_expr = base.rolling_max(window_size=w, min_samples=ms)
                elif stat == "median":
                    r_expr = base.rolling_median(window_size=w, min_samples=ms)
                else:
                    r_expr = base.rolling_sum(window_size=w, min_samples=ms)
                exprs.append(_over_entity(r_expr, entity_cols).alias(alias))

    sorted_df = _with_temporal_sort(df, entity_cols, time_col)
    if keep_original:
        result = _restore_row_order(sorted_df.with_columns(exprs))
    else:
        kept_idx = [*entity_cols, time_col]
        result = _restore_row_order(
            sorted_df.with_columns(exprs).select([_ROW_IDX, *kept_idx, *new_names])
        )
    return result, new_names


def diff_features(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    order: int = 1,
    sp: int = 1,
    columns: Sequence[str] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Apply seasonal/ordinary differencing per entity in chronological order.

    Parameters
    ----------
    data : DataFrame-like
        Input DataFrame.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    order : int, default=1
        Number of differencing passes.
    sp : int, default=1
        Differencing lag step.
    columns : sequence of str or None, default=None
        Columns to difference. Defaults to non-index numeric columns.

    Returns
    -------
    tuple[pl.DataFrame, dict[str, Any]]
        Differenced DataFrame and state dictionary needed by :func:`invert_diff_features`.
    """
    if order < 1:
        raise ValueError(f"`order` must be >= 1, got {order}.")
    if sp < 1:
        raise ValueError(f"`sp` must be >= 1, got {sp}.")

    df = _to_polars_df(data)
    entity_cols, target_cols = _validate_temporal_columns(
        df, time_col=time_col, entity_col=entity_col, columns=columns
    )
    diff_cols = (
        _numeric_target_cols(df, target_cols) if columns is None else target_cols
    )
    sorted_df = _with_temporal_sort(df, entity_cols, time_col)

    anchors: list[pl.DataFrame] = []
    cur = sorted_df
    for stage_idx in range(order):
        base = stage_idx * sp
        if entity_cols:
            # Earlier differencing stages leave leading nulls; slice past them for valid anchors.
            anchor = (
                cur.with_columns(
                    pl.int_range(0, pl.len()).over(entity_cols).alias("__ig")
                )
                .filter((pl.col("__ig") >= base) & (pl.col("__ig") < base + sp))
                .select([*entity_cols, *diff_cols])
            )
        else:
            anchor = cur.slice(base, sp).select(diff_cols)
        anchors.append(anchor)
        diff_exprs = [
            _over_entity(pl.col(col).diff(n=sp), entity_cols).alias(col)
            for col in diff_cols
        ]
        cur = cur.with_columns(diff_exprs)

    state = {
        "time_col": time_col,
        "entity_cols": entity_cols,
        "columns": diff_cols,
        "order": order,
        "sp": sp,
        "anchors": anchors,
    }
    return _restore_row_order(cur), state


def _reconstruct_diff_series(
    diffs: Sequence[Any],
    anc_list: Sequence[Any],
    sp: int,
    base: int,
) -> list[Any]:
    buf: list[Any] = []
    for idx, val in enumerate(diffs):
        if idx < base:
            restored = val
        elif idx < base + sp:
            anc_idx = idx - base
            restored = anc_list[anc_idx] if anc_idx < len(anc_list) else val
        else:
            prev = buf[idx - sp]
            restored = None if (val is None or prev is None) else prev + val
        buf.append(restored)
    return buf


def invert_diff_features(data: Any, *, state: dict[str, Any]) -> pl.DataFrame:
    """Invert differencing produced by :func:`diff_features` on aligned rows.

    Parameters
    ----------
    data : DataFrame-like
        Differenced DataFrame aligned on ``[entity_col, time_col]``.
    state : dict[str, Any]
        State returned by :func:`diff_features`.

    Returns
    -------
    pl.DataFrame
        Undifferenced DataFrame in input row order.
    """
    df = _to_polars_df(data)
    time_col: str = state["time_col"]
    entity_cols: list[str] = state["entity_cols"]
    diff_cols: list[str] = state["columns"]
    sp: int = state["sp"]
    anchors: list[pl.DataFrame] = state["anchors"]

    sorted_df = _with_temporal_sort(df, entity_cols, time_col)
    cur = sorted_df
    for stage_idx in range(len(anchors) - 1, -1, -1):
        stage_anchor = anchors[stage_idx]
        base = stage_idx * sp
        if entity_cols:
            grouped_anchor = stage_anchor.group_by(
                entity_cols, maintain_order=True
            ).agg([pl.col(col).alias(f"__anc_{col}") for col in diff_cols])
            cur = cur.join(
                grouped_anchor, on=entity_cols, how="left", maintain_order="left"
            )

        for col in diff_cols:
            if entity_cols:
                anc_col = f"__anc_{col}"
                vals_df = cur.select([_ROW_IDX, *entity_cols, col, anc_col])
                reconstructed: dict[int, Any] = {}
                for _, group in vals_df.group_by(entity_cols, maintain_order=True):
                    row_ids = group.get_column(_ROW_IDX).to_list()
                    diffs = group.get_column(col).to_list()
                    raw_anc = group.get_column(anc_col).item(0)
                    anc_list = (
                        raw_anc.to_list()
                        if isinstance(raw_anc, pl.Series)
                        else (raw_anc or [])
                    )
                    buf = _reconstruct_diff_series(diffs, anc_list, sp, base)
                    for row_id, restored in zip(row_ids, buf):
                        reconstructed[row_id] = restored
                restored_col = [
                    reconstructed[rid] for rid in cur.get_column(_ROW_IDX).to_list()
                ]
                cur = cur.with_columns(pl.Series(col, restored_col)).drop(anc_col)
            else:
                diffs = cur.get_column(col).to_list()
                anc_list = stage_anchor.get_column(col).to_list()
                buf = _reconstruct_diff_series(diffs, anc_list, sp, base)
                cur = cur.with_columns(pl.Series(col, buf))

    return _restore_row_order(cur)


def scale_features(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    with_mean: bool = True,
    with_std: bool = True,
    columns: Sequence[str] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Standardize numeric columns per panel entity (or globally if univariate).

    Parameters
    ----------
    data : DataFrame-like
        Input DataFrame.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional entity identifier column(s).
    with_mean : bool, default=True
        Center features around entity mean.
    with_std : bool, default=True
        Scale features to unit entity standard deviation.
    columns : sequence of str or None, default=None
        Columns to scale. Defaults to non-index numeric columns.
    state : dict[str, Any] or None, default=None
        Pre-fitted state dictionary. When ``None``, fit statistics on ``data``.

    Returns
    -------
    tuple[pl.DataFrame, dict[str, Any]]
        Scaled DataFrame and fitted statistics dictionary.
    """
    if not with_mean and not with_std:
        raise ValueError("At least one of `with_mean` or `with_std` must be True.")

    df = _to_polars_df(data)
    entity_cols, target_cols = _validate_temporal_columns(
        df, time_col=time_col, entity_col=entity_col, columns=columns
    )
    if state is None:
        scale_cols = (
            _numeric_target_cols(df, target_cols) if columns is None else target_cols
        )
        stat_exprs: list[pl.Expr] = []
        for col in scale_cols:
            if with_mean:
                stat_exprs.append(
                    pl.col(col)
                    .cast(pl.Float64)
                    .mean()
                    .fill_null(0.0)
                    .alias(f"__mean_{col}")
                )
            if with_std:
                raw_std = pl.col(col).cast(pl.Float64).std(ddof=0)
                stat_exprs.append(
                    pl.when(raw_std.is_null() | (raw_std == 0.0))
                    .then(1.0)
                    .otherwise(raw_std)
                    .alias(f"__std_{col}")
                )
        if entity_cols:
            stats_df = df.group_by(entity_cols, maintain_order=True).agg(stat_exprs)
            fallback_df = df.select(stat_exprs)
        else:
            stats_df = df.select(stat_exprs)
            fallback_df = stats_df
        fitted_state: dict[str, Any] = {
            "entity_cols": entity_cols,
            "columns": scale_cols,
            "with_mean": with_mean,
            "with_std": with_std,
            "stats_df": stats_df,
            "fallback": fallback_df.row(0, named=True)
            if fallback_df.height > 0
            else {},
        }
    else:
        fitted_state = state
        scale_cols = fitted_state["columns"]
        entity_cols = fitted_state["entity_cols"]

    stats_df = fitted_state["stats_df"]
    fallback: dict[str, float] = fitted_state["fallback"]
    out = (
        df.join(stats_df, on=entity_cols, how="left", maintain_order="left")
        if entity_cols
        else df.with_columns(
            [pl.lit(stats_df.get_column(c).item(0)).alias(c) for c in stats_df.columns]
        )
    )

    update_exprs: list[pl.Expr] = []
    drop_cols: list[str] = (
        list(stats_df.columns)
        if not entity_cols
        else [c for c in stats_df.columns if c not in entity_cols]
    )
    for col in scale_cols:
        expr = pl.col(col).cast(pl.Float64)
        if fitted_state["with_mean"]:
            m_col = f"__mean_{col}"
            expr = expr - pl.col(m_col).fill_null(fallback.get(m_col, 0.0))
        if fitted_state["with_std"]:
            s_col = f"__std_{col}"
            expr = expr / pl.col(s_col).fill_null(fallback.get(s_col, 1.0))
        update_exprs.append(expr.alias(col))

    result = out.with_columns(update_exprs).drop(drop_cols)
    return result, fitted_state


def invert_scale_features(data: Any, *, state: dict[str, Any]) -> pl.DataFrame:
    """Invert scaling applied by :func:`scale_features`.

    Parameters
    ----------
    data : DataFrame-like
        Scaled DataFrame.
    state : dict[str, Any]
        Fitted statistics state returned by :func:`scale_features`.

    Returns
    -------
    pl.DataFrame
        Unscaled DataFrame in input row order.
    """
    df = _to_polars_df(data)
    entity_cols: list[str] = state["entity_cols"]
    scale_cols: list[str] = state["columns"]
    stats_df: pl.DataFrame = state["stats_df"]
    fallback: dict[str, float] = state["fallback"]

    out = (
        df.join(stats_df, on=entity_cols, how="left", maintain_order="left")
        if entity_cols
        else df.with_columns(
            [pl.lit(stats_df.get_column(c).item(0)).alias(c) for c in stats_df.columns]
        )
    )
    drop_cols = (
        list(stats_df.columns)
        if not entity_cols
        else [c for c in stats_df.columns if c not in entity_cols]
    )

    update_exprs: list[pl.Expr] = []
    for col in scale_cols:
        expr = pl.col(col).cast(pl.Float64)
        if state["with_std"]:
            s_col = f"__std_{col}"
            expr = expr * pl.col(s_col).fill_null(fallback.get(s_col, 1.0))
        if state["with_mean"]:
            m_col = f"__mean_{col}"
            expr = expr + pl.col(m_col).fill_null(fallback.get(m_col, 0.0))
        update_exprs.append(expr.alias(col))

    return out.with_columns(update_exprs).drop(drop_cols)


def detrend_features(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    method: Literal["linear", "mean"] = "linear",
    columns: Sequence[str] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Remove per-entity linear or constant mean trend anchored at fit time ``t0``.

    Parameters
    ----------
    data : DataFrame-like
        Input DataFrame.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    method : {"linear", "mean"}, default="linear"
        Trend estimator.
    columns : sequence of str or None, default=None
        Columns to detrend. Defaults to non-index numeric columns.
    state : dict[str, Any] or None, default=None
        Pre-fitted trend parameters. Fits parameters when ``None``.

    Returns
    -------
    tuple[pl.DataFrame, dict[str, Any]]
        Detrended DataFrame and fitted trend state.
    """
    if method not in ("linear", "mean"):
        raise ValueError(f"`method` must be 'linear' or 'mean', got {method!r}.")

    df = _to_polars_df(data)
    entity_cols, target_cols = _validate_temporal_columns(
        df, time_col=time_col, entity_col=entity_col, columns=columns
    )

    if state is None:
        detrend_cols = (
            _numeric_target_cols(df, target_cols) if columns is None else target_cols
        )
        time_dtype = df.schema[time_col]
        t0 = df.get_column(time_col).min()
        work = df.with_columns(
            _relative_time_expr(time_col, t0, time_dtype).alias("__t")
        )

        agg_exprs: list[pl.Expr] = []
        for col in detrend_cols:
            y = pl.col(col).cast(pl.Float64)
            if method == "mean":
                agg_exprs.append(y.mean().fill_null(0.0).alias(f"__alpha_{col}"))
                agg_exprs.append(pl.lit(0.0).alias(f"__beta_{col}"))
            else:
                t_val = pl.when(y.is_not_null()).then(pl.col("__t")).otherwise(None)
                t_centered = t_val - t_val.mean()
                denom = t_centered.pow(2).sum()
                beta = (
                    pl.when(denom.is_null() | (denom == 0.0))
                    .then(0.0)
                    .otherwise((t_centered * (y - y.mean())).sum() / denom)
                    .fill_null(0.0)
                )
                alpha = (
                    y.mean().fill_null(0.0) - beta * t_val.mean().fill_null(0.0)
                ).fill_null(0.0)
                agg_exprs.append(alpha.alias(f"__alpha_{col}"))
                agg_exprs.append(beta.alias(f"__beta_{col}"))

        params_df = (
            work.group_by(entity_cols, maintain_order=True).agg(agg_exprs)
            if entity_cols
            else work.select(agg_exprs)
        )
        fitted_state: dict[str, Any] = {
            "time_col": time_col,
            "entity_cols": entity_cols,
            "columns": detrend_cols,
            "method": method,
            "t0": t0,
            "time_dtype": time_dtype,
            "params_df": params_df,
        }
    else:
        fitted_state = state
        detrend_cols = fitted_state["columns"]
        entity_cols = fitted_state["entity_cols"]

    params_df: pl.DataFrame = fitted_state["params_df"]
    t0 = fitted_state["t0"]
    time_dtype = fitted_state["time_dtype"]
    work = df.with_columns(_relative_time_expr(time_col, t0, time_dtype).alias("__t"))
    work = (
        work.join(params_df, on=entity_cols, how="left", maintain_order="left")
        if entity_cols
        else work.with_columns(
            [
                pl.lit(params_df.get_column(c).item(0)).alias(c)
                for c in params_df.columns
            ]
        )
    )
    drop_cols = ["__t", *(c for c in params_df.columns if c not in entity_cols)]
    sub_exprs = [
        (
            pl.col(col).cast(pl.Float64)
            - (
                pl.col(f"__alpha_{col}").fill_null(0.0)
                + pl.col(f"__beta_{col}").fill_null(0.0) * pl.col("__t")
            )
        ).alias(col)
        for col in detrend_cols
    ]
    return work.with_columns(sub_exprs).drop(drop_cols), fitted_state


def invert_detrend_features(data: Any, *, state: dict[str, Any]) -> pl.DataFrame:
    """Restore linear or constant trend removed by :func:`detrend_features`.

    Parameters
    ----------
    data : DataFrame-like
        Detrended DataFrame.
    state : dict[str, Any]
        Fitted trend state from :func:`detrend_features`.

    Returns
    -------
    pl.DataFrame
        DataFrame with trend restored.
    """
    df = _to_polars_df(data)
    time_col: str = state["time_col"]
    entity_cols: list[str] = state["entity_cols"]
    detrend_cols: list[str] = state["columns"]
    params_df: pl.DataFrame = state["params_df"]
    work = df.with_columns(
        _relative_time_expr(time_col, state["t0"], state["time_dtype"]).alias("__t")
    )
    work = (
        work.join(params_df, on=entity_cols, how="left", maintain_order="left")
        if entity_cols
        else work.with_columns(
            [
                pl.lit(params_df.get_column(c).item(0)).alias(c)
                for c in params_df.columns
            ]
        )
    )
    drop_cols = ["__t", *(c for c in params_df.columns if c not in entity_cols)]
    add_exprs = [
        (
            pl.col(col).cast(pl.Float64)
            + pl.col(f"__alpha_{col}").fill_null(0.0)
            + pl.col(f"__beta_{col}").fill_null(0.0) * pl.col("__t")
        ).alias(col)
        for col in detrend_cols
    ]
    return work.with_columns(add_exprs).drop(drop_cols)


def _ffd_weights(d: float, min_weight: float, window_size: int | None) -> list[float]:
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * ((d - k + 1) / k)
        if abs(w) < min_weight:
            break
        if window_size is not None and len(weights) >= window_size:
            break
        weights.append(w)
        k += 1
        if k > 10_000:
            break
    return weights


def frac_diff_features(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    d: float = 0.4,
    min_weight: float = 1e-4,
    window_size: int | None = None,
    columns: Sequence[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Apply fixed-width window fractional differentiation in pure Polars.

    Parameters
    ----------
    data : DataFrame-like
        Input DataFrame.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    d : float, default=0.4
        Differencing order in ``(0, 1]`` (or any non-negative float).
    min_weight : float, default=1e-4
        Weight threshold truncating the binomially decaying kernel.
    window_size : int or None, default=None
        Optional hard cap on kernel length.
    columns : sequence of str or None, default=None
        Columns to fractionally differentiate. Defaults to numeric non-index columns.

    Returns
    -------
    tuple[pl.DataFrame, list[str]]
        Fractionally differentiated DataFrame and transformed column names.
    """
    if min_weight <= 0:
        raise ValueError(f"`min_weight` must be > 0, got {min_weight}.")
    if window_size is not None and window_size < 1:
        raise ValueError(f"`window_size` must be >= 1, got {window_size}.")

    df = _to_polars_df(data)
    entity_cols, target_cols = _validate_temporal_columns(
        df, time_col=time_col, entity_col=entity_col, columns=columns
    )
    diff_cols = (
        _numeric_target_cols(df, target_cols) if columns is None else target_cols
    )
    weights = _ffd_weights(d=d, min_weight=min_weight, window_size=window_size)

    sorted_df = _with_temporal_sort(df, entity_cols, time_col)
    exprs: list[pl.Expr] = []
    for col in diff_cols:
        terms = [
            pl.lit(w) * _over_entity(pl.col(col).cast(pl.Float64).shift(k), entity_cols)
            for k, w in enumerate(weights)
        ]
        expr = terms[0]
        for term in terms[1:]:
            expr = expr + term
        exprs.append(expr.alias(col))
    return _restore_row_order(sorted_df.with_columns(exprs)), diff_cols


def impute_features(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    strategy: Literal[
        "forward", "backward", "linear", "mean", "median", "constant"
    ] = "forward",
    fill_value: float | None = None,
    columns: Sequence[str] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Impute missing values chronologically per entity without cross-entity leakage.

    Parameters
    ----------
    data : DataFrame-like
        Input DataFrame.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional panel entity identifier column(s).
    strategy : {"forward", "backward", "linear", "mean", "median", "constant"}, default="forward"
        Imputation method applied within each series.
    fill_value : float or None, default=None
        Constant fallback when ``strategy="constant"``.
    columns : sequence of str or None, default=None
        Columns to impute. Defaults to non-index numeric columns.
    state : dict[str, Any] or None, default=None
        Pre-fitted summary statistics state (for ``"mean"`` or ``"median"``).

    Returns
    -------
    tuple[pl.DataFrame, dict[str, Any]]
        Imputed DataFrame in input row order and fitted state.
    """
    allowed = {"forward", "backward", "linear", "mean", "median", "constant"}
    if strategy not in allowed:
        raise ValueError(
            f"Unsupported imputation strategy {strategy!r}; allowed: {sorted(allowed)}."
        )
    if strategy == "constant" and fill_value is None:
        raise ValueError("`fill_value` must be provided when `strategy='constant'`.")

    df = _to_polars_df(data)
    entity_cols, target_cols = _validate_temporal_columns(
        df, time_col=time_col, entity_col=entity_col, columns=columns
    )
    imp_cols = _numeric_target_cols(df, target_cols) if columns is None else target_cols
    sorted_df = _with_temporal_sort(df, entity_cols, time_col)

    if state is None:
        stats_df: pl.DataFrame | None = None
        fallback: dict[str, float] = {}
        if strategy in ("mean", "median"):
            agg = [
                (
                    pl.col(c).cast(pl.Float64).mean()
                    if strategy == "mean"
                    else pl.col(c).cast(pl.Float64).median()
                ).alias(f"__fill_{c}")
                for c in imp_cols
            ]
            stats_df = (
                sorted_df.group_by(entity_cols, maintain_order=True).agg(agg)
                if entity_cols
                else sorted_df.select(agg)
            )
            fb_df = sorted_df.select(agg)
            fallback = fb_df.row(0, named=True) if fb_df.height > 0 else {}
        fitted_state: dict[str, Any] = {
            "entity_cols": entity_cols,
            "columns": imp_cols,
            "strategy": strategy,
            "fill_value": fill_value,
            "stats_df": stats_df,
            "fallback": fallback,
        }
    else:
        fitted_state = state
        imp_cols = fitted_state["columns"]
        entity_cols = fitted_state["entity_cols"]

    if fitted_state["strategy"] in ("mean", "median"):
        stats_df = fitted_state["stats_df"]
        assert stats_df is not None
        fallback = fitted_state["fallback"]
        joined = (
            sorted_df.join(stats_df, on=entity_cols, how="left", maintain_order="left")
            if entity_cols
            else sorted_df.with_columns(
                [
                    pl.lit(stats_df.get_column(c).item(0)).alias(c)
                    for c in stats_df.columns
                ]
            )
        )
        drop_cols = (
            list(stats_df.columns)
            if not entity_cols
            else [c for c in stats_df.columns if c not in entity_cols]
        )
        fill_exprs = [
            pl.col(c)
            .fill_null(
                pl.col(f"__fill_{c}").fill_null(fallback.get(f"__fill_{c}", 0.0))
            )
            .alias(c)
            for c in imp_cols
        ]
        result = _restore_row_order(joined.with_columns(fill_exprs).drop(drop_cols))
        return result, fitted_state

    imp_exprs: list[pl.Expr] = []
    for c in imp_cols:
        col_expr = pl.col(c)
        if strategy == "forward":
            filled = _over_entity(col_expr.forward_fill(), entity_cols)
        elif strategy == "backward":
            filled = _over_entity(col_expr.backward_fill(), entity_cols)
        elif strategy == "linear":
            filled = _over_entity(col_expr.interpolate(), entity_cols)
        else:
            filled = col_expr.fill_null(fill_value)
        imp_exprs.append(filled.alias(c))

    return _restore_row_order(sorted_df.with_columns(imp_exprs)), fitted_state


def encode_panel_features(
    data: Any,
    *,
    time_col: str,
    entity_col: str | Sequence[str] | None = None,
    drop_time: bool = True,
    entity_encoding: Literal["enum", "ordinal", "drop"] = "enum",
    cast_float32: bool = True,
    state: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Prepare a panel DataFrame for downstream tabular estimators and GBDTs.

    Parameters
    ----------
    data : DataFrame-like
        Input DataFrame.
    time_col : str
        Name of the timestamp/order column.
    entity_col : str, sequence of str, or None, default=None
        Optional entity identifier column(s).
    drop_time : bool, default=True
        Whether to drop ``time_col`` from the output matrix.
    entity_encoding : {"enum", "ordinal", "drop"}, default="enum"
        Encoding applied to ``entity_col`` columns: ``"enum"`` freezes training
        levels into ``pl.Enum``, ``"ordinal"`` emits dense ``Int32`` codes
        compatible with raw Polars inputs across HGB/LightGBM/XGBoost/CatBoost,
        and ``"drop"`` removes entity columns.
    cast_float32 : bool, default=True
        Cast numeric non-entity features to ``pl.Float32`` and booleans to
        ``pl.Int8``, mapping non-finite values safely.
    state : dict[str, Any] or None, default=None
        Pre-fitted entity category mapping dictionary.

    Returns
    -------
    tuple[pl.DataFrame, dict[str, Any]]
        Model-ready Polars DataFrame and fitted category state.
    """
    if entity_encoding not in ("enum", "ordinal", "drop"):
        raise ValueError(
            f"`entity_encoding` must be 'enum', 'ordinal', or 'drop'; got {entity_encoding!r}."
        )

    df = _to_polars_df(data)
    entity_cols, _ = _validate_temporal_columns(
        df, time_col=time_col, entity_col=entity_col
    )

    if state is None:
        categories: dict[str, list[str]] = {}
        for col in entity_cols:
            cats = sorted(
                df.get_column(col).cast(pl.String).drop_nulls().unique().to_list()
            )
            categories[col] = cats
        fitted_state: dict[str, Any] = {
            "time_col": time_col,
            "entity_cols": entity_cols,
            "drop_time": drop_time,
            "entity_encoding": entity_encoding,
            "cast_float32": cast_float32,
            "categories": categories,
        }
    else:
        fitted_state = state
        categories = fitted_state["categories"]
        entity_cols = fitted_state["entity_cols"]

    out = df.drop(time_col) if fitted_state["drop_time"] else df
    entity_set = set(entity_cols)
    if entity_encoding == "drop" and entity_cols:
        out = out.drop(entity_cols)
    elif entity_cols:
        enc_exprs: list[pl.Expr] = []
        for col in entity_cols:
            cats = categories[col]
            if entity_encoding == "enum":
                # Cast unknown test entities to null instead of failing on unseen levels.
                s_expr = pl.col(col).cast(pl.String)
                enc_exprs.append(
                    pl.when(s_expr.is_in(cats))
                    .then(s_expr)
                    .otherwise(None)
                    .cast(pl.Enum(cats))
                    .alias(col)
                )
            else:
                mapping = {cat: idx for idx, cat in enumerate(cats)}
                enc_exprs.append(
                    pl.col(col)
                    .cast(pl.String)
                    .replace_strict(mapping, default=-1, return_dtype=pl.Int32)
                    .alias(col)
                )
        out = out.with_columns(enc_exprs)

    if fitted_state["cast_float32"]:
        cast_exprs: list[pl.Expr] = []
        for col, dtype in out.schema.items():
            if col in entity_set or col == time_col:
                continue
            if dtype == pl.Boolean:
                cast_exprs.append(pl.col(col).cast(pl.Int8))
            elif dtype.is_numeric():
                f_col = pl.col(col).cast(pl.Float32)
                cast_exprs.append(
                    pl.when(f_col.is_infinite())
                    .then(np.nan)
                    .otherwise(f_col)
                    .alias(col)
                )
        if cast_exprs:
            out = out.with_columns(cast_exprs)

    return out, fitted_state
