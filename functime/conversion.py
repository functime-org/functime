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


if __name__ == "__main__":

    from timeit import default_timer

    df = pl.DataFrame({f"x{i}": pl.arange(0, 1_000_000, eager=True) for i in range(48)})
    start = default_timer()
    X = df_to_ndarray(df)
    print(default_timer() - start)  # 1 second
