import polars as pl
from functime import feature_extractors as fe


class FeatureCalculator:
    def __init__(self):
        self.features = []

    def add_feature(self, feature: str, params: dict):
        self.features.append(
            [
                feature,
                params
            ]
        )

    def add_multi_features(self, features: list[list[str, dict]]):
        for x in features:
            self.features.append(
                [
                    x[0],
                    x[1]
                ]
            )

    def rm_feature(self, feature):
        elem_to_rm = [
            sub for sub in self.features 
            if sub[0] == feature
        ]
        if len(elem_to_rm) > 0:
            for x in elem_to_rm:
                self.features.remove(x)

    def calculate_features(self, X: pl.DataFrame):
        X_features = (
            X
            .transpose(include_header=True, header_name="id")
            .melt(id_vars="id")
            .group_by(
                pl.col("id")
            )
            .agg(
                [
                    getattr(pl.col("value").ts, x[0])().alias(x[0])
                    for x in self.features
                ]
            )
        )
        return X_features