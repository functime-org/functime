import polars as pl

from functime.forecasting import LinearModel

y = pl.read_parquet("./data/m4_1w_train.parquet")
y = y.with_columns(pl.col("series").cast(pl.Categorical))
model = LinearModel(lags=12).fit(y=y)
print(model.stub_id)
