from __future__ import annotations

import numpy as np
import polars as pl


def df_to_ndarray(df: pl.DataFrame) -> np.ndarray:
    """Zero-copy spill-to-disk Polars DataFrame to numpy ndarray."""
    return df.to_numpy()


def X_to_numpy(X: pl.DataFrame) -> np.ndarray:
    X_arr = (
        X.lazy()
        .drop(pl.nth([0, 1]))
        .select(pl.all().cast(pl.Float32))
        .select(
            pl.when(pl.all().is_infinite() | pl.all().is_nan())
            .then(None)
            .otherwise(pl.all())
            .name.keep()
        )
        # TODO: Support custom group_by imputation
        .fill_null(strategy="mean")  # Do not fill backward (data leak)
        .collect(streaming=True)
        .pipe(df_to_ndarray)
    )
    return X_arr


def y_to_numpy(y: pl.DataFrame) -> np.ndarray:
    y_arr = (
        y.lazy()
        .select(pl.col(y.columns[-1]).cast(pl.Float32))
        .select(
            pl.when(pl.all().is_infinite() | pl.all().is_nan())
            .then(None)
            .otherwise(pl.all())
            .name.keep()
        )
        # TODO: Support custom group_by imputation
        .fill_null(strategy="mean")  # Do not fill backward (data leak)
        .collect(streaming=True)
        .get_column(y.columns[-1])
        .to_numpy()  # TODO: Cannot require zero-copy array?
    )
    return y_arr
