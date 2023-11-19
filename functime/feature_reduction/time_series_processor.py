import polars as pl
import logging
from functime.feature_reduction._feat_calculator import FeatureCalculator
from functime.feature_reduction._dim_reducer import DimensionReducer

class features_dim_reduction:
    def __init__(self, features: str = "default", model: str = "PCA", format: str = "wide"):
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
        return self.dimension_reducer.state_model.transform(
            self.feature_calculator.X_features.select(
                pl.exclude("id")
            )
        )
    
    def fit(self, X: pl.DataFrame, dim: int = 2, **kwargs):
        X_features = (
            self.calculate_features(X)
            .select(
                pl.exclude("id")
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
            logging.info(
                "The dimension algorithm requested has not been implemented yet."
            )
    
    def fit_transform(self, X: pl.DataFrame, dim: int = 2, **kwargs)-> pl.DataFrame:
        X_features = (
            self.calculate_features(X)
            .select(
                pl.exclude("id")
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
            logging.info(
                "The dimension algorithm requested has not been implemented yet."
            )

s1 = pl.Series([1,2,3,4,5]*10000)
s2 = pl.Series([1,2,3,3,3]*10000)
s3 = pl.Series([1,2,3,4,5]*10000)
s4 = pl.Series([1,2,3,3,3]*10000)
s5 = pl.Series([1,2,3,4,5]*10000)
s6 = pl.Series([1,2,3,3,3]*10000)

df = pl.DataFrame(
    {"a": s1, "b": s2, "c": s3, "d": s4, "e": s5, "f": s6}
)

# ts_proc = features_dim_reduction(features= "default", model = "PCA")

# fitted_pca = (
#     ts_proc
#     .add_multi_features(
#         [
#             ["number_peaks", {"support": 2}],
#             ["mean_n_absolute_max", {"n_maxima": 10}]
#         ]
#     )
#     .fit(X = df)
# )

# # Use sklearn parameters
# fitted_pca.explained_variance_ratio_

# # Get the X_reduced
# X_reduced = ts_proc.X_reduced()

# # Get the table of the features
# ts_proc.X_features()


# df_res = (
#     df
#     .transpose(include_header=True, header_name="id")
#     .melt(id_vars="id")
#     .with_columns([
#         pl.col("value").ts.number_peaks(support=2).over(pl.col("id")).alias("nb_peaks"),
#         pl.col("value").ts.mean_n_absolute_max(n_maxima=10).over(pl.col("id")).alias("mean_abs")
#     ])
# )

# print(df_res)
