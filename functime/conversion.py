import tempfile
from datetime import datetime
from typing import Optional

import dask.array as da
import numpy as np
import polars as pl
import zarr


def df_to_ndarray(df: pl.DataFrame, n_groups: Optional[int] = None) -> np.ndarray:
    """Zero-copy spill-to-disk Polars DataFrame to numpy ndarray."""
    columns = df.columns
    df = df.select(pl.all().cast(pl.Float32))  # Defensive type cast

    chunks = (df.shape[0], 1)  # Chunk columnar
    if n_groups:
        chunks = (n_groups, df.shape[1])  # Chunk by group

    with tempfile.TemporaryDirectory() as tempdir:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        file_path = f"{tempdir}/{timestamp}.zarr"
        X = zarr.open_array(
            store=file_path,
            mode="w",
            shape=df.shape,
            chunks=chunks,
            dtype=np.float32,
            chunk_store=f"{file_path}/chunks",
        )
        for i, col in enumerate(columns):
            series = df.get_column(col)
            x = series.to_numpy(zero_copy_only=True)
            X[:, i] = x
        X = da.from_zarr(X).compute()
    return X


def X_to_numpy(X: pl.DataFrame) -> np.ndarray:
    X_arr = (
        X.lazy()
        .select(pl.col(X.columns[2:]).cast(pl.Float32))
        .select(
            pl.when(pl.all().is_infinite() | pl.all().is_nan())
            .then(None)
            .otherwise(pl.all())
            .keep_name()
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
            .keep_name()
        )
        # TODO: Support custom group_by imputation
        .fill_null(strategy="mean")  # Do not fill backward (data leak)
        .collect(streaming=True)
        .get_column(y.columns[-1])
        .to_numpy(zero_copy_only=True)
    )
    return y_arr
