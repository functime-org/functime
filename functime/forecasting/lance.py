from typing import Optional

import lance
import numpy as np
import polars as pl
from tqdm import tqdm
from typing_extensions import Literal

from functime.base import Forecaster
from functime.forecasting._ar import fit_autoreg


class ANNRegressor:
    """Approximate-nearest neighbors regressor built on Lance.

    Reference:
    https://lancedb.github.io/lance/api/python/lance.html#module-lance.dataset
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        index_type: Literal["IVF_PQ"] = "IVF_PQ",
        metric: Literal["L2", "cosine", "dot"] = "L2",
        num_partitions: int = 256,
        num_sub_vectors: Optional[int] = None,
        ivf_centroids: Optional[np.ndarray] = None,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        **kwargs
    ):
        self.uri = uri or "functime_embs/knn.lance"
        self.index_type = index_type
        self.metric = metric
        self.num_partitions = num_partitions
        self.num_sub_vectors = num_sub_vectors
        self.ivf_centroids = ivf_centroids
        self.nprobes = nprobes
        self.refine_factor = refine_factor
        self.kwargs = kwargs
        self._dataset = None

    def fit(self, X: pl.DataFrame, y: pl.DataFrame):
        idx_cols = y.columns[:2]
        feat_cols = X.columns[2:]
        n_dims = len(feat_cols)
        df = (
            y.join(X, how="left", on=idx_cols)
            .with_columns(
                pl.all().exclude(idx_cols).to_physical().cast(pl.Float32)
            )  # Defensive
            .select(
                pl.col(y.columns[-1]).alias("label"),
                pl.concat_list(feat_cols)
                .alias("emb")
                .cast(pl.Array(width=n_dims, inner=pl.Float32)),
            )
        )
        dataset = lance.write_dataset(df.to_arrow(), uri=self.uri, mode="overwrite")
        dataset.create_index(
            "emb",
            index_type=self.index_type,
            metric=self.metric,
            num_partitions=self.num_partitions,
            ivf_centroids=self.ivf_centroids,
            # Must satisfy contraints:
            # 1. (n_dims / num_sub_vectors) % 8 == 0
            # 2. n_dims % num_sub_vectors == 0
            num_sub_vectors=self.num_sub_vectors or n_dims // 8,
        )
        self._dataset = dataset
        return self

    def predict(self, X: pl.DataFrame):
        dataset = self._dataset
        feat_cols = X.columns[2:]
        embs = (
            X.select(pl.concat_list(feat_cols).alias("emb")).get_column("emb").to_list()
        )
        labels = np.zeros(shape=X.shape[0], dtype=np.float32)
        # TODO: Parallelize
        for i, emb in tqdm(enumerate(embs), desc="ANN search"):
            labels[i] = dataset.to_table(
                columns=["label"],
                nearest={
                    "column": "emb",
                    "q": emb,
                    "k": 1,
                    "nprobes": self.nprobes,
                    "refine_factor": self.refine_factor,
                },
            )["label"][0].as_py()
        return labels


def _ann(**kwargs):
    def regress(X: pl.DataFrame, y: pl.DataFrame):
        regressor = ANNRegressor(**kwargs)
        return regressor.fit(X=X, y=y)

    return regress


class ann(Forecaster):
    """Autoregressive approximate nearest neighbors built on Lance."""

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        regress = _ann(**self.kwargs)
        return fit_autoreg(
            regress=regress,
            y=y,
            X=X,
            lags=self.lags,
            max_horizons=self.max_horizons,
            strategy=self.strategy,
        )
