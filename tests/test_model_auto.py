import polars as pl

from functime.forecasting import AutoLightGBM

## Test that the last used time is updated

y = pl.read_parquet("./data/m4_1w_train.parquet")
y = y.with_columns(pl.col("series").cast(pl.Categorical))

a = AutoLightGBM(freq=None).fit(y=y)
print("Stub id: ", a.stub_id)

a_y_pred = a.predict(fh=30)
print(f"pred 1: {a_y_pred}")
