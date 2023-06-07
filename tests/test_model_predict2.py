import polars as pl

from functime.forecasting import LinearModel

y = pl.read_parquet("./data/m4_1w_train.parquet")
y = y.with_columns(pl.col("series").cast(pl.Categorical))

a = LinearModel(12, None).fit(y=y)
print("A Stub id: ", a.stub_id)

b = LinearModel(12, None).fit(y=y)
print("B Stub id: ", b.stub_id)

c = LinearModel.from_deployed(a.stub_id)
print("C Stub id: ", c.stub_id)

a_y_pred = a.predict(fh=30)
print(f"A pred 1: {a_y_pred}")
b_y_pred = b.predict(fh=30)
print(f"B pred 1: {b_y_pred}")
c_y_pred = c.predict(fh=30)
print(f"A pred 2: {c_y_pred}")
