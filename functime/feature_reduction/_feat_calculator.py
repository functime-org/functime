import polars as pl
from functime.feature_extractors import FeatureExtractor
import sys
import logging
import polars.selectors as cs

logger = logging.getLogger(__name__)


class FeatureCalculator:
    def __init__(self, col_values: str):
        self.features = []
        self.X_features = None
        self.col_values = col_values

    def add_feature(self, feature: pl.Expr):
        self.features.append(feature)

    def add_multi_features(self, features: list[pl.Expr]):
        self.features += features

    def calculate_features(self, X: pl.DataFrame)-> pl.DataFrame:
        id = X.columns[0]
        self.X_features = (
            X
            .group_by(pl.col(id), maintain_order=True)
            .agg(
                self.features
            )
        ).cast({cs.numeric(): pl.Float64, cs.by_dtype(pl.Boolean): pl.Float64})
        ## Need to add postprocessing steps because explode() doesn't within group_by
        #self.X_features.unnest(~cs.numeric() & ~cs.string() & ~cs.by_dtype(pl.Boolean))
        return self.X_features
