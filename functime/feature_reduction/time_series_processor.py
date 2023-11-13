import polars as pl
import logging
from functime.feature_reduction._feat_calculator import FeatureCalculator
from functime.feature_reduction._dim_reducer import DimensionReducer

class TimeSeriesProcessor:
    def __init__(self, features: str = "default", model: str = "PCA"):
        self.feature_calculator = FeatureCalculator()
        self.dimension_reducer = DimensionReducer()
        self.model = model

        if features == "default":
            self.feature_calculator.add_multi_features(
                [
                    ["root_mean_square", {}],
                    ["count_above_mean", {}],
                    ["first_location_of_minimum", {}],
                    ["first_location_of_maximum", {}]
                ]
            )

    def add_feature(self, feature: str, params: dict = {}):
        self.feature_calculator.add_feature(feature, params)

    def add_multi_features(self, features: list[list[str, dict]]):
        self.feature_calculator.add_multi_features(features)

    def rm_feature(self, feature: str):
        self.feature_calculator.rm_feature(feature)
    
    def fit(self, X: pl.DataFrame, dim: int = 2, **kwargs):
        X_features = self.feature_calculator.calculate_features(X)
        print(X_features)
        if self.model == "PCA":
            self.dimension_reducer.fit_pca(
                X_features.select(
                    pl.exclude("id")
                ),
                dim,
                **kwargs
            )
            return self.dimension_reducer.state_model
        elif self.model == "TSNE":
            pass
        else:
            logging.info(
                "The dimension algorithm requested has not been implemented yet."
            )
    
    def fit_transform(self, X: pl.DataFrame, dim: int = 2, **kwargs):
        X_features = self.feature_calculator.calculate_features(X)
        if self.model == "PCA":
            self.dimension_reducer.fit_pca(
                X_features.select(
                    pl.exclude("id")
                ),
                dim,
                **kwargs
            )
            return self.dimension_reducer.state_model.transform(X_features.select(
                    pl.exclude("id")
                ))
        elif self.model == "TSNE":
            pass
        else:
            logging.info(
                "The dimension algorithm requested has not been implemented yet."
            )


s1 = pl.Series([1,2,3,4,5]*1000000)
s2 = pl.Series([1,2,3,3,3]*1000000)
s3 = pl.Series([1,2,3,4,5]*1000000)
s4 = pl.Series([1,2,3,3,3]*1000000)
s5 = pl.Series([1,2,3,4,5]*1000000)
s6 = pl.Series([1,2,3,3,3]*1000000)

df = pl.DataFrame(
    {"a": s1, "b": s2, "c": s3, "d": s4, "e": s5, "f": s6}
)

ts_proc = TimeSeriesProcessor(features= "default", model = "PCA")

ts_proc.fit_transform(
    X = df
)