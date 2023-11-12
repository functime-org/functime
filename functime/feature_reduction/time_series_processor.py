import polars as pl
import logging
from functime.feature_reduction._feat_calculator import FeatureCalculator
from functime.feature_reduction._dim_reducer import DimensionReducer


class TimeSeriesProcessor:
    def __init__(self, features: str = "custom", model: str = "PCA"):
        self.feature_calculator = FeatureCalculator()
        self.dimension_reducer = DimensionReducer()
        self.features = features
        self.model = model

        if features ==  "custom":
            self.feature_calculator.add_feature(
                "root_mean_square",
                "ratio_n_unique_to_length"
            )
    def add_feature(self, feature: str, **kwargs):
        self.feature_calculator.add_feature(feature, **kwargs)

    def add_features(self, name):
        self.feature_calculator.add_features(name)
    
    def fit(self, X: pl.DataFrame, dim: int, **kwargs):
        if self.model == "PCA":
            self.dimension_reducer.fit_pca(
                X,
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


    # Other methods to coordinate the overall process...