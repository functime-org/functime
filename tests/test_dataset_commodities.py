import polars as pl

from functime.cross_validation import train_test_split
from functime.feature_extraction import add_calendar_effects, add_holiday_effects
from functime.forecasting import AutoLinearModel
from functime.metrics import mase

# Specify forecast horizon (the number of periods to predict into the future)
# and the time-series frequency
fh = 3
freq = "1mo"

# Load example data
y = pl.read_parquet("./data/commodities.parquet")
entity_col, time_col = y.columns[:2]

# Add calendar and holiday effects
X = (
    y.select([entity_col, time_col])
    .pipe(add_calendar_effects(["month"]))
    .pipe(add_holiday_effects(country_codes=["US"], freq="1d"))
    .collect()
)

# Time series split
y_train, y_test = y.pipe(train_test_split(test_size=fh))
X_train, X_test = X.pipe(train_test_split(test_size=fh))
print(y_train.collect(), y_test.collect())

# Specify model
model = AutoLinearModel(
    freq="1mo", min_lags=3, max_lags=3, test_size=3, max_horizons=3, strategy="ensemble"
)

# Fit then predict
model.fit(y=y_train)
y_pred = model.predict(fh=3)

# Score forecasts in parallel
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
print(scores.select(pl.col("mase").mean()))

# Fit then predict
model.fit(y=y_train, X=X_train)
y_pred = model.predict(fh=3, X=X_test)

# Score forecasts in parallel
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
print(scores.select(pl.col("mase").mean()))
