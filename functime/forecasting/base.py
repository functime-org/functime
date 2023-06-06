from io import BytesIO
from typing import Literal, Optional, Self, Union

import httpx
import pandas as pd
import polars as pl
import pyarrow as pa

from functime.config import API_CALL_TIMEOUT, FUNCTIME_SERVER_URL
from functime.io.auth import require_token
from functime.io.serialize import deserialize_bytes, serialize_bytes

FORECAST_STRATEGIES = Optional[Literal["direct", "recursive", "naive"]]
DF_TYPE = Union[pl.LazyFrame, pl.DataFrame, pa.Table, pd.DataFrame]
SUPPORTED_FORECASTERS = [
    "auto_elastic_net",
    "auto_knn",
    "auto_lasso",
    "auto_lightgbm",
    "auto_linear_model",
    "auto_ridge",
    "elastic_net",
    "knn",
    "lasso",
    "lightgbm",
    "linear_model",
    "ridge",
]


class ForecasterClient:
    """Functime forecaster client"""

    _server_url = FUNCTIME_SERVER_URL
    _stub_id: Optional[str] = None

    def __init__(self, **kwargs):
        self.model_kwargs = kwargs

    def __call__(
        self,
        y: DF_TYPE,
        fh: int,
        freq: Optional[str] = None,
        X: Optional[DF_TYPE] = None,
        X_future: Optional[DF_TYPE] = None,
    ) -> pl.DataFrame:
        return self.fit_predict(y=y, fh=fh, freq=freq, X=X, X_future=X_future)

    @classmethod
    def from_deployed(cls, stub_id: str, **kwargs) -> Self:
        """Load a ForecasterClient from a deployed estimator."""
        # Pull model metadata?
        response = _api_call(
            url=cls._server_url,
            endpoint="/stub/from_deployed",
            estimator_id=stub_id,
            model_id=cls.model,
        )
        response_json = response.json()
        kwargs.update(response_json["model_kwargs"])
        _cls = cls(**kwargs)
        _cls._stub_id = stub_id
        return _cls

    @property
    def stub_id(self) -> str:
        return self._stub_id

    @property
    def is_fitted(self) -> bool:
        return self._stub_id is not None

    def fit(self, y: DF_TYPE, X: Optional[DF_TYPE] = None) -> Self:
        """Fit the forecaster to the data and return the estimator ID."""
        y = coerce_df_to_pa_table(y)
        if X is not None:
            X = coerce_df_to_pa_table(X)
        response = _api_call(
            url=self._server_url,
            endpoint="/fit",
            y=y,
            X=X,
            model_id=self.model,
            **{k: v for k, v in self.model_kwargs.items() if v is not None},
        )
        response_json = response.json()
        self._stub_id = response_json["estimator_id"]
        return self

    def predict(
        self, fh: int, freq: Optional[str] = None, X: Optional[DF_TYPE] = None
    ) -> pl.DataFrame:
        """Predict using the forecaster."""
        if not self.is_fitted:
            raise RuntimeError("Forecaster has not been fitted yet.")
        if X is not None:
            X = coerce_df_to_pa_table(X)
        response = _api_call(
            url=self._server_url,
            endpoint="/predict",
            estimator_id=self._stub_id,
            fh=fh,
            freq=freq,
            X=X,
            model_id=self.model,
            **{k: v for k, v in self.model_kwargs.items() if v is not None},
        )
        y_pred_bytes = response.content
        y_pred_arrow = deserialize_bytes(y_pred_bytes)
        return pl.from_arrow(y_pred_arrow)

    def fit_predict(
        self,
        y: pa.Table,
        fh: int,
        freq: Optional[str] = None,
        X: Optional[pa.Table] = None,
        X_future: Optional[pa.Table] = None,
    ) -> pl.DataFrame:
        y = coerce_df_to_pa_table(y)
        if X is not None:
            X = coerce_df_to_pa_table(X)
        if X_future is not None:
            X_future = coerce_df_to_pa_table(X_future)
        response = _api_call(
            url=self._server_url,
            endpoint="/fit_predict",
            y=y,
            fh=fh,
            freq=freq,
            X=X,
            X_future=X_future,
            model_id=self.model,
            **{k: v for k, v in self.model_kwargs.items() if v is not None},
        )
        table_bytes = response.content
        table = deserialize_bytes(table_bytes)
        return pl.from_arrow(table)


def coerce_df_to_pa_table(df: DF_TYPE) -> pa.Table:
    if isinstance(df, pa.Table):
        return df
    if isinstance(df, pl.DataFrame):
        return df.to_arrow()
    if isinstance(df, pl.LazyFrame):
        return df.collect().to_arrow()
    if isinstance(df, pd.DataFrame):
        return pa.Table.from_pandas(df)
    raise TypeError(f"Unsupported type: {type(df)}")


@require_token
def _api_call(token, *, url: str, endpoint: str, **kwargs):
    headers = {"Authorization": f"Bearer {token}"}
    # kwargs is flat
    files = {}
    y = kwargs.pop("y", None)
    X = kwargs.pop("X", None)
    X_future = kwargs.pop("X_future", None)
    if y is not None:
        y_bytes = serialize_bytes(y)
        files["y"] = BytesIO(y_bytes)
    if X is not None:
        X_bytes = serialize_bytes(X)
        files["X"] = BytesIO(X_bytes)
    if X_future is not None:
        X_future_bytes = serialize_bytes(X_future)
        files["X_future"] = BytesIO(X_future_bytes)
    with httpx.Client(http2=True) as client:
        response = client.post(
            url + endpoint,
            headers=headers,
            files=files or None,
            params=kwargs,
            timeout=kwargs.get("timeout", API_CALL_TIMEOUT),
        )
        response.raise_for_status()
    return response
