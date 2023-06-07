import polars as pl

from functime.forecasting import AutoLightGBM

y = pl.read_parquet("./data/m4_1w_train.parquet")
y = y.with_columns(pl.col("series").cast(pl.Categorical))
forecaster = AutoLightGBM(freq=None).fit_predict(fh=30, y=y)
print(forecaster)
