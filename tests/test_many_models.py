import time

import polars as pl

from functime.forecasting import KNN, ElasticNet, Lasso, LinearModel, Ridge

y = pl.read_parquet("./data/m4_1w_train.parquet")
y = y.with_columns(pl.col("series").cast(pl.Categorical))
model = LinearModel(lags=12, freq=None).fit(y=y)

a = LinearModel.from_deployed(stub_id=model.stub_id)


print("A Stub id: ", a.stub_id)
a_y_pred = a.predict(fh=30)
print(f"A pred 1: {a_y_pred}")


# This should fail
try:
    r = Ridge.from_deployed(stub_id=model.stub_id)
except Exception as e:
    print(e)
    time.sleep(3)

r = Ridge(lags=12, freq=None).fit(y=y)
print("R Stub id: ", r.stub_id)
r_y_pred = r.predict(fh=30)
print(f"R pred 1: {r_y_pred}")


# Do the same for the other models
k = KNN(lags=12, freq=None).fit(y=y)
print("K Stub id: ", k.stub_id)
k_y_pred = k.predict(fh=30)
print(f"K pred 1: {k_y_pred}")

e = ElasticNet(lags=12, freq=None).fit(y=y)
print("E Stub id: ", e.stub_id)
e_y_pred = e.predict(fh=30)
print(f"E pred 1: {e_y_pred}")

L = Lasso(lags=12, freq=None).fit(y=y)
print("L Stub id: ", L.stub_id)
L_y_pred = L.predict(fh=30)
print(f"L pred 1: {L_y_pred}")


k2 = KNN.from_deployed(stub_id=k.stub_id)
print("K2 Stub id: ", k2.stub_id)
k2_y_pred = k2.predict(fh=30)
print(f"K2 pred 1: {k2_y_pred}")
