from __future__ import annotations

import numpy as np
import polars as pl


def wide_to_long(
    df: pl.DataFrame,
    id_col: str = "entity_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pl.DataFrame:
    """Convert a wide-format DataFrame to long (panel) format.

    In wide format, each entity is a column and each row is a timestamp.
    In long format, there are three columns: entity, time, and value.

    Parameters
    ----------
    df : pl.DataFrame
        Wide-format DataFrame where the first column is the time index
        and remaining columns are entities.
    id_col : str
        Name for the entity column in the output. Defaults to "entity_id".
    time_col : str
        Name of the time column (must exist in the input). Defaults to "date".
    value_col : str
        Name for the value column in the output. Defaults to "value".

    Returns
    -------
    pl.DataFrame
        Long-format panel DataFrame with columns [id_col, time_col, value_col].
    """
    # Determine the time column — use time_col if present, otherwise first column
    if time_col in df.columns:
        time_column = time_col
    else:
        time_column = df.columns[0]

    entity_columns = [c for c in df.columns if c != time_column]
    result = df.unpivot(
        on=entity_columns,
        index=time_column,
        variable_name=id_col,
        value_name=value_col,
    )
    # Rename the time column if needed
    if time_column != time_col:
        result = result.rename({time_column: time_col})
    # Reorder to [entity, time, value]
    return result.select(id_col, time_col, value_col)


def long_to_wide(
    df: pl.DataFrame,
    id_col: str | None = None,
    time_col: str | None = None,
    value_col: str | None = None,
) -> pl.DataFrame:
    """Convert a long (panel) format DataFrame to wide format.

    In long format, there are entity, time, and value columns.
    In wide format, each entity becomes its own column and each row is a timestamp.

    Parameters
    ----------
    df : pl.DataFrame
        Long-format panel DataFrame with at least 3 columns:
        entity, time, value (in that order if col names not specified).
    id_col : str, optional
        Name of the entity column. Defaults to first column.
    time_col : str, optional
        Name of the time column. Defaults to second column.
    value_col : str, optional
        Name of the value column. Defaults to third column.

    Returns
    -------
    pl.DataFrame
        Wide-format DataFrame where each entity is a column.
    """
    cols = df.columns
    id_col = id_col or cols[0]
    time_col = time_col or cols[1]
    value_col = value_col or cols[2]

    result = df.pivot(
        on=id_col,
        index=time_col,
        values=value_col,
    ).sort(time_col)
    return result


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
        .collect(engine="streaming")
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
        .collect(engine="streaming")
        .get_column(y.columns[-1])
        .to_numpy()  # TODO: Cannot require zero-copy array?
    )
    return y_arr
