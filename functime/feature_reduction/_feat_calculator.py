import polars as pl
from functime.feature_extractors import FeatureExtractor
import sys
import logging

logger = logging.getLogger(__name__)

class FeatureCalculator:
    def __init__(self):
        self.features = []
        self.X_features = None

    def add_feature(self, feature: str, params: dict):
        self.features.append(
            [
                feature,
                params
            ]
        )

    @staticmethod
    def _clean_feature(x: list[str, dict] | list[str]):
        if len(x) <= 2:
            if len(x) == 1:
                return True
        else:
            logger.error(
                "The format of the features is incorrect. Please refer to the documentation."
            )
            sys.exit(1)

    @staticmethod
    def _check_feature(feat_name: str):
        if not hasattr(FeatureExtractor, feat_name) and not hasattr(pl.Expr, feat_name):
            logger.error(
                f"The feature `{feat_name}` is not available for both polars and functime."
            )
            sys.exit(1)

    def add_multi_features(self, features: list[list[str, dict]]):
        for i, x in enumerate(features):
            if self._clean_feature(x):
                features[i].append({})
            if 0 < len(x) <= 2:
                self._check_feature(feat_name = x[0])
            if features.count(x) > 1:
                features.remove(x)
        
        self.features += features

    def rm_feature(self, feature):
        elem_to_rm = [
            sub for sub in self.features 
            if sub[0] == feature
        ]
        if len(elem_to_rm) > 0:
            for x in elem_to_rm:
                self.features.remove(x)

    def calculate_features(self, X: pl.DataFrame):
        id = X.columns[0]
        feat_values = X.columns[-1]
        feat_agg = [
            getattr(pl.col(feat_values).ts, x[0])(**x[1])
            .alias(f"{x[0]}{''.join(f'_{param}_{val}' for param, val in x[1].items())}")
            if hasattr(FeatureExtractor, x[0])
            else
            getattr(pl.col(feat_values), x[0])(**x[1])
            .alias(f"{x[0]}{''.join(f'_{param}_{val}' for param, val in x[1].items())}")
            for x in self.features
        ]
        self.X_features = (
            X
            .group_by(
                pl.col(id),
                maintain_order=True
            )
            .agg(
                feat_agg
            )
        )
        print(self.X_features)
        return self.X_features