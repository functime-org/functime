import polars as pl
import logging
import sys
from functime.feature_reduction._feat_calculator import FeatureCalculator
from functime.feature_reduction._dim_reducer import DimensionReducer

logger = logging.getLogger(__name__)

class features_dim_reduction:
    def __init__(self, features: str = "", model: str = "PCA"):
        self.feature_calculator = FeatureCalculator()
        self.dimension_reducer = DimensionReducer()
        self.model = model

        if features == "default":
            self.feature_calculator.add_multi_features(
                [
                    ["number_peaks", {"support": 2}],
                    ["mean_n_absolute_max", {"n_maxima": 10}],
                    ["root_mean_square", {}],
                    ["count_above_mean", {}],
                    ["first_location_of_minimum", {}],
                    ["first_location_of_maximum", {}]
                ]
            )
        elif features == "all":
            pass

    def add_feature(self, feature: str, params: dict = {}):
        self.feature_calculator.add_feature(feature, params)
        return self

    def add_multi_features(self, features: list[list[str, dict]]):
        self.feature_calculator.add_multi_features(features)
        return self

    def rm_feature(self, feature: str):
        self.feature_calculator.rm_feature(feature)
        return self
    
    def calculate_features(self, X: pl.DataFrame) -> pl.DataFrame:
        return self.feature_calculator.calculate_features(X = X)

    def X_features(self):
        return self.feature_calculator.X_features
    
    def X_reduced(self):
        id = self.feature_calculator.X_features.columns[0]
        return self.dimension_reducer.state_model.transform(
            self.feature_calculator.X_features.select(
                pl.exclude(id)
            )
        )
    
    def fit(self, X: pl.DataFrame, dim: int = 2, **kwargs):
        id = X.columns[0]
        X_features = (
            self.calculate_features(X)
            .select(
                pl.exclude(id)
            )
        )
        if self.model == "PCA":
            self.dimension_reducer.fit_pca(
                X_features,
                dim,
                **kwargs
            )
            return self.dimension_reducer.state_model
        elif self.model == "TSNE":
            pass
        else:
            logger.info(
                "The dimension algorithm requested has not been implemented yet."
            )
    
    def fit_transform(self, X: pl.DataFrame, dim: int = 2, **kwargs)-> pl.DataFrame:
        id = X.columns[0]
        X_features = (
            self.calculate_features(X)
            .select(
                pl.exclude(id)
            )
        )
        if self.model == "PCA":
            self.dimension_reducer.fit_pca(
                X_features,
                dim,
                **kwargs
            )
            return self.dimension_reducer.state_model.transform(X_features)
        elif self.model == "TSNE":
            pass
        else:
            logger.info(
                "The dimension algorithm requested has not been implemented yet."
            )



df = pl.read_parquet("data/sp500.parquet")
ts_proc = features_dim_reduction(model = "PCA")

fitted_pca = (
    ts_proc
    .add_multi_features(
        [
            ["number_peaks", {"support": 2}],
            ["number_peaks", {"support": 2}],
            ["mean_n_absolute_max", {"n_maxima": 10}],
            ["max"]
        ]
    )
    .add_feature(
        feature="number_peaks",
        params = {"support": 4}
    )
    .fit(X = df, dim = 3)
)

print(fitted_pca)

# # Use sklearn parameters
print(fitted_pca.explained_variance_ratio_)

# Get the X_reduced
X_reduced = ts_proc.X_reduced()
print(X_reduced)

# Get the table of the features
print(ts_proc.X_features())

