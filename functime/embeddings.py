from io import BytesIO

import numpy as np

from functime.io.client import FunctimeH2Client


def embed(
    X: np.ndarray,
    model: str = "minirocket",
    **kwargs,
) -> np.ndarray:
    """Create embeddings from a 2D array of time series.

    Parameters
    ----------
    X : np.ndarray
        2D array where each row represents a choronological time series.
        The array shape (n_rows, n_cols) where n_rows is the number of
        time series and n_cols is the number of timepoints for each time
        series. There must not be any missing values or NaNs in the array.

    model : str
        The embedding model to use. Currently only supports "minirocket".

    """
    endpoint = "/embed"

    kwargs = kwargs or {}
    arr_bytes = BytesIO(X.tobytes())
    dtype = str(X.dtype)
    rows, cols = X.shape
    kwargs.update({"model": model, "dtype": dtype, "rows": rows, "cols": cols})

    with FunctimeH2Client(msg="Creating embeddings") as client:
        response = client.post(
            endpoint,
            files={"X": arr_bytes},
            params=kwargs,
        )
    res_json = response.json()
    # Reconstruct the np.array from the json
    emb = np.array(res_json["embeddings"], dtype=res_json["dtype"]).reshape(
        (res_json["rows"], res_json["cols"])
    )
    return emb
