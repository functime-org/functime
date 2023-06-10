from typing import List, Mapping, Union

import polars as pl
from scipy.stats import boxcox_normmax
from typing_extensions import Literal

from functime.base import transformer
from functime.base.model import ModelState
from functime.offsets import _strip_freq_alias

PL_FLOAT_DTYPES = [pl.Float32, pl.Float64]
PL_INT_DTYPES = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
PL_NUMERIC_DTYPES = [*PL_INT_DTYPES, *PL_FLOAT_DTYPES]
PL_FLOAT_COLS = pl.col(PL_FLOAT_DTYPES)
PL_INT_COLS = pl.col(PL_INT_DTYPES)


def PL_NUMERIC_COLS(*exclude):
    return pl.col(PL_NUMERIC_DTYPES).exclude(exclude)


@transformer
def coerce_dtypes(schema: Mapping[str, pl.DataType]):
    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        X_new = X.cast({pl.col(col).cast(dtype) for col, dtype in schema.items()})
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def resample(freq: str, agg_method: str, impute_method: Union[str, int, float]):
    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col, target_col = X.columns
        agg_exprs = {
            "sum": pl.sum(target_col),
            "mean": pl.mean(target_col),
            "median": pl.median(target_col),
        }
        X_new = (
            # Defensive resampling
            X.lazy()
            .groupby_dynamic(time_col, every=freq, by=entity_col)
            .agg(agg_exprs[agg_method])
            # Must defensive sort columns otherwise time_col and target_col
            # positions are incorrectly swapped in lazy
            .select([entity_col, time_col, target_col])
            # Reindex full (entity, time) index
            .pipe(reindex_panel(freq=freq, sort=True))
            # Impute gaps after reindex
            .pipe(impute(impute_method))
            # Defensive fill null with 0 for impute method `ffill`
            .fill_null(0)
        )
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def lag(lags: List[int]):
    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col = X.columns[0]
        time_col = X.columns[1]
        max_lag = max(lags)
        lagged_series = [
            (
                pl.all()
                .exclude([entity_col, time_col])
                .shift(lag)
                .over(entity_col)
                .suffix(f"__lag_{lag}")
            )
            for lag in lags
        ]
        X_new = (
            # Pre-sorting seems to improve performance by ~20%
            X.sort(by=[entity_col, time_col])
            .select(
                pl.col(entity_col).set_sorted(),
                pl.col(time_col).set_sorted(),
                *lagged_series,
            )
            .groupby(entity_col)
            .agg(pl.all().slice(max_lag))
            .explode(pl.all().exclude(entity_col))
        )
        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def roll(
    window_sizes: List[int],
    stats: List[Literal["mean", "min", "max", "mlm", "sum", "std", "cv"]],
    freq: str,
):
    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        offset_n, offset_alias = _strip_freq_alias(freq)
        values = pl.all().exclude([entity_col, time_col])
        stat_exprs = {
            "mean": lambda w: values.mean().suffix(f"__rolling_mean_{w}"),
            "min": lambda w: values.min().suffix(f"__rolling_min_{w}"),
            "max": lambda w: values.max().suffix(f"__rolling_max_{w}"),
            "mlm": lambda w: (values.max() - values.min()).suffix(f"__rolling_mlm_{w}"),
            "sum": lambda w: values.sum().suffix(f"__rolling_sum_{w}"),
            "std": lambda w: values.std().suffix(f"__rolling_std_{w}"),
            "cv": lambda w: (values.std() / values.mean()).suffix(f"__rolling_cv_{w}"),
        }
        # Degrees of freedom
        X_all = [
            (
                X.sort([entity_col, time_col])
                .groupby_dynamic(
                    index_column=time_col,
                    by=entity_col,
                    every=freq,
                    period=f"{offset_n * w}{offset_alias}",
                )
                .agg([stat_exprs[stat](w) for stat in stats])
                # NOTE: Must lag by 1 to avoid data leakage
                .select([entity_col, time_col, values.shift(w).over(entity_col)])
            )
            for w in window_sizes
        ]
        # Join all window lazy operations
        X_rolling = X_all[0]
        for X_window in X_all[1:]:
            X_rolling = X_rolling.join(X_window, on=[entity_col, time_col], how="outer")
        # Defensive join to match original X index
        X_new = X.join(X_rolling, on=[entity_col, time_col], how="left").select(
            X_rolling.columns
        )

        artifacts = {"X_new": X_new}
        return artifacts

    return transform


@transformer
def scale(use_mean: bool = True, use_std: bool = True, rescale_bool: bool = True):

    if not (use_mean or use_std):
        raise ValueError("At least one of `use_mean` or `use_std` must be set to True")

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        idx_cols = X.columns[:2]
        entity_col, time_col = idx_cols
        cols = X.select(PL_NUMERIC_COLS(entity_col, time_col)).columns
        boolean_cols = None
        _mean = None
        _std = None
        if use_mean:
            X = X.with_columns(
                PL_NUMERIC_COLS(entity_col, time_col)
                .mean()
                .over(entity_col)
                .suffix("_mean")
            )
            mean_cols = [col for col in X.columns if col.endswith("_mean")]
            _mean = X.select([*idx_cols, *mean_cols])
            X = X.select(
                idx_cols + [pl.col(col) - pl.col(f"{col}_mean") for col in cols]
            )
        if use_std:
            X = X.with_columns(
                PL_NUMERIC_COLS(entity_col, time_col)
                .std()
                .over(entity_col)
                .suffix("_std")
            )
            std_cols = [col for col in X.columns if col.endswith("_std")]
            _std = X.select([*idx_cols, *std_cols])
            X = X.select(
                idx_cols + [pl.col(col) / pl.col(f"{col}_std") for col in cols]
            )
        expr = pl.all()
        if rescale_bool:
            # Original boolean column names
            boolean_cols = X.select(pl.col(pl.Boolean)).columns
            # Minmax rescale boolean cols [-1, 1]
            expr = [expr, pl.col(pl.Boolean).cast(pl.Int8) * 2 - 1]
        X_new = X.select(expr)
        artifacts = {
            "X_new": X_new,
            "boolean_cols": boolean_cols,
            "_mean": _mean,
            "_std": _std,
        }
        return artifacts

    def invert(state: ModelState, X: pl.LazyFrame) -> pl.LazyFrame:
        idx_cols = X.columns[:2]
        cols = X.select(PL_NUMERIC_COLS(state.time)).columns
        if use_std:
            _std = state.artifacts["_std"]
            X = X.join(_std, on=idx_cols, how="left").select(
                idx_cols + [pl.col(col) * pl.col(f"{col}_std") for col in cols]
            )
        if use_mean:
            _mean = state.artifacts["_mean"]
            X = X.join(_mean, on=idx_cols, how="left").select(
                idx_cols + [pl.col(col) + pl.col(f"{col}_mean") for col in cols]
            )
        expr = pl.all()
        if rescale_bool:
            # Minmax rescale boolean cols [-1, 1]
            boolean_cols = pl.col(state.artifacts["boolean_cols"])
            expr = [expr, (boolean_cols + 1).cast(pl.Int8)]
        X_new = X.select(expr)
        return X_new

    return transform, invert


@transformer
def impute(
    method: Union[
        Literal["mean", "median", "fill", "ffill", "bfill", "interpolate"],
        Union[int, float],
    ]
):
    def method_to_expr(entity_col, time_col):
        """Fill-in methods."""
        return {
            "mean": PL_NUMERIC_COLS(entity_col, time_col).fill_null(
                PL_NUMERIC_COLS(entity_col, time_col).mean().over(entity_col)
            ),
            "median": PL_NUMERIC_COLS(entity_col, time_col).fill_null(
                PL_NUMERIC_COLS(entity_col, time_col).median().over(entity_col)
            ),
            # "mode": PL_NUMERIC_COLS(entity_col, time_col).fill_null(PL_NUMERIC_COLS(entity_col, time_col).mode().over(entity_col)),
            "fill": [
                PL_FLOAT_COLS.fill_null(PL_FLOAT_COLS.mean().over(entity_col)),
                PL_INT_COLS.fill_null(PL_INT_COLS.median().over(entity_col)),
                # pl.col([pl.Categorical, pl.Boolean]).fill_null(
                #     PL_FLOAT_COLS.mode().over(entity_col)
                # ),
            ],
            "ffill": PL_NUMERIC_COLS(entity_col, time_col)
            .fill_null(strategy="forward")
            .over(entity_col),
            "bfill": PL_NUMERIC_COLS(entity_col, time_col)
            .fill_null(strategy="backward")
            .over(entity_col),
            "interpolate": PL_NUMERIC_COLS(entity_col, time_col)
            .interpolate()
            .over(entity_col),
        }

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col, time_col = X.columns[:2]
        if isinstance(method, int) or isinstance(method, float):
            expr = PL_NUMERIC_COLS(entity_col, time_col).fill_null(pl.lit(method))
        else:
            expr = method_to_expr(entity_col, time_col)[method]
        X_new = X.with_columns(expr)
        return {"X_new": X_new}

    return transform


@transformer
def diff(order: int, sp: int = 1):
    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        def _diff(X):
            X_new = (
                X.groupby(entity_col, maintain_order=True)
                .agg([pl.col(time_col), PL_FLOAT_COLS - PL_FLOAT_COLS.shift(sp)])
                .explode(pl.all().exclude(entity_col))
            )
            return X_new

        idx_cols = X.columns[:2]
        entity_col = idx_cols[0]
        time_col = idx_cols[1]
        X = X.with_columns(pl.col(pl.Categorical).cast(pl.Utf8))

        X_first, X_last = pl.collect_all(
            [
                X.groupby(entity_col).head(1),
                X.groupby(entity_col).tail(1),
            ]
        )
        for _ in range(order):
            X = _diff(X)

        # Drop null
        artifacts = {
            "X_new": X.drop_nulls(),
            "X_first": X_first.lazy(),
            "X_last": X_last.lazy(),
        }
        return artifacts

    def invert(
        state: ModelState, X: pl.LazyFrame, from_last: bool = False
    ) -> pl.LazyFrame:
        def _inverse_diff(X: pl.LazyFrame):
            X_new = (
                X.groupby(entity_col, maintain_order=True)
                .agg([pl.col(time_col), PL_FLOAT_COLS.cumsum()])
                .explode(pl.all().exclude(entity_col))
            )
            return X_new

        artifacts = state.artifacts
        entity_col = X.columns[0]
        time_col = X.columns[1]
        idx_cols = entity_col, time_col

        X_cutoff = artifacts["X_last"] if from_last else artifacts["X_first"]
        X_new = pl.concat([X, X_cutoff], how="diagonal").drop_nulls().sort(idx_cols)
        for _ in range(order):
            X_new = _inverse_diff(X_new)  # noqa: F821

        return X.select(idx_cols).join(X_new, on=idx_cols, how="left")

    return transform, invert


@transformer
def boxcox(method: str = "mle"):
    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        idx_cols = X.columns[:2]
        entity_col, time_col = idx_cols
        gb = X.groupby(X.columns[0])
        # Step 1. Compute optimal lambdas
        lmbds = gb.agg(
            PL_NUMERIC_COLS(entity_col, time_col)
            .apply(lambda x: boxcox_normmax(x, method=method))
            .cast(pl.Float64())
        )
        # Step 2. Transform
        cols = X.select(PL_NUMERIC_COLS(entity_col, time_col)).columns
        X_new = X.join(lmbds, on=entity_col, how="left", suffix="_lmbd").select(
            idx_cols
            + [
                pl.when(pl.col(f"{col}_lmbd") == 0)
                .then(pl.col(col).log())
                .otherwise(
                    (pl.col(col) ** pl.col(f"{col}_lmbd") - 1) / pl.col(f"{col}_lmbd")
                )
                for col in cols
            ]
        )
        artifacts = {"X_new": X_new, "lmbds": lmbds}
        return artifacts

    def invert(state: ModelState, X: pl.LazyFrame) -> pl.LazyFrame:
        idx_cols = X.columns[:2]
        lmbds = state.artifacts["lmbds"]
        cols = X.select(PL_NUMERIC_COLS(state.time)).columns
        X_new = X.join(lmbds, on=X.columns[0], how="left", suffix="_lmbd").select(
            idx_cols
            + [
                pl.when(pl.col(f"{col}_lmbd") == 0)
                .then(pl.col(col).exp())
                .otherwise(
                    (pl.col(f"{col}_lmbd") * pl.col(col) + 1)
                    ** (1 / pl.col(f"{col}_lmbd"))
                )
                for col in cols
            ]
        )
        return X_new

    return transform, invert


@transformer
def reindex_panel(freq: str, sort: bool = False):
    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        # Create new index
        entity_col = X.columns[0]
        time_col = X.columns[1]
        dtypes = X.dtypes[:2]

        with pl.StringCache():
            entities = X.collect().get_column(entity_col).unique().to_frame()
            dates = X.collect().get_column(time_col)
            timestamps = pl.date_range(
                dates.min(), dates.max(), interval=freq
            ).to_frame(name=time_col)

            full_idx = entities.join(timestamps, how="cross")
            # Defensive cast dtypes to be consistent with df
            full_idx = full_idx.select(
                [pl.col(col).cast(dtypes[i]) for i, col in enumerate(full_idx.columns)]
            )

            # Outer join
            X_new = (
                # Must collect before join otherwise will hit error:
                # Joins/or comparisons on categorical dtypes can only happen if they are created under the same global string cache.
                X.collect().join(full_idx, on=[entity_col, time_col], how="outer")
            )

        if sort:
            X_new = X_new.sort([entity_col, time_col]).with_columns(
                [pl.col(entity_col).set_sorted(), pl.col(time_col).set_sorted()]
            )

        artifacts = {"X_new": X_new.lazy()}
        return artifacts

    return transform


@transformer
def zero_pad(freq: str, include_null: bool = True, include_nan: bool = True):
    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        """Reindex panel then fill nulls / nans with 0."""
        target_cols = X.columns[2:]
        transform = reindex_panel(freq=freq)

        if include_null and include_nan:
            expr = [pl.col(col).fill_null(0).fill_nan(0) for col in target_cols]
        elif include_null:
            expr = [pl.col(col).fill_null(0) for col in target_cols]
        else:
            expr = [pl.col(col).fill_nan(0) for col in target_cols]

        X_new = transform(X=X).with_columns(expr)
        artifacts = {"X_new": X_new.lazy()}
        return artifacts

    return transform
