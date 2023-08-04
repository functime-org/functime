import os
from io import BytesIO
from typing import Optional

import numpy as np
import polars as pl
import pyarrow as pa
import requests
from typing_extensions import Literal


def _pyarrow_table_to_bytes(table: pa.Table) -> BytesIO:
    with pa.BufferOutputStream() as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        bytes_data = sink.getvalue().to_pybytes()
    return BytesIO(bytes_data)


def embed(
    X: pl.DataFrame,
    model: Literal[
        "rocket", "bazooka", "bazooka-turbo", "hindsight", "hindsight-turbo"
    ] = "bazooka",
    user_id: Optional[str] = None,
) -> np.ndarray:
    """Creates temporal embeddings representing the input univariate
    (single dimensional / channel) or multivariate (high dimensional / channel) time-series.

    Parameters
    ----------
    X : pl.DataFrame
        Panel data of time-series.
    model : str
        ID of the temporal embeddings model to use.
        Current supports the following model IDs:\n
        - `rocket`: for univariate time-series
        - `bazooka`: for multivariate time-series
        - `rocket`: for univariate time-series
        - `rocket`: for univariate time-series
        - `rocket`: for univariate time-series
    user_id : Optional[str]
        A unique identifier representing your end-user,
        which can help functime Cloud to monitor and detect abuse.

    Returns
    -------
    embs : np.ndarray
        2D Numpy array of temporal embeddings.
        The returned shape depends on the selected `model`:\n
        - `rocket`: `n_entities` by 9,996 dimensions
        - `bazooka`: `n_entities` by 9,996 dimensions
        - `bazooka-turbo`: `n_entities` by 49,728 dimensions
        - `hindsight`: (`n_entities` x `n_timestamps`) by 9,996 dimensions
        - `hindsight-turbo`: `(`n_entities` x `n_timestamps`)` by 49,728 dimensions

    Raises
    ------
    KeyError : if missing environment variables `FUNCTIME_API_URL` and `FUNCTIME_API_TOKEN`.
    ValueError : if the shape of the input DataFrame `X` does not align with the selected `model`.
    """

    try:
        api_url = os.environ["FUNCTIME_API_URL"]
        api_token = os.environ["FUNCTIME_API_TOKEN"]
    except KeyError as err:
        raise KeyError(
            "Missing environment variables 'FUNCTIME_API_URL' and 'FUNCTIME_API_TOKEN'"
        ) from err

    n_channels = X.shape[1] - 2
    if model == "rocket" and n_channels > 1:
        raise ValueError(
            f"Selected univariate embeddings model 'rocket', but found multivariate panel data with {n_channels} features."
        )
    elif model != "rocket" and n_channels == 1:
        raise ValueError(
            f"Select multiviarate embeddings model '{model}', but found univariate panel data."
        )

    embs_bytes = requests.get(
        f"{api_url}/embed",
        headers={"Authorization": f"Bearer {api_token}"},
        files={"X": _pyarrow_table_to_bytes(X.to_arrow())},
        params={"model": model, "user_id": user_id},
    )
    embs = np.from_buffer(embs_bytes)

    return embs
