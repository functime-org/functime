from timeit import default_timer

import polars as pl

from functime.cross_validation import train_test_split
from functime.forecasting import linear_model, naive, snaive
from functime.metrics import smape
from functime.preprocessing import scale
from functime.seasonality import add_fourier_terms

start_time = default_timer()

# Load data
y = pl.read_parquet("hf://datasets/functime-org/datasets/commodities.parquet")
entity_col, time_col = y.columns[:2]
X = y.select([entity_col, time_col]).pipe(add_fourier_terms(sp=12, K=6)).collect()

print("🎯 Target variable (y):\n", y)
print("📉 Exogenous variables (X):\n", X)

# Train-test splits
test_size = 3
freq = "1mo"
y_train, y_test = train_test_split(test_size)(y)
X_train, X_test = train_test_split(test_size)(X)

# Paralleized naive forecasts!
y_pred_naive = naive(freq="1mo")(y=y_train, fh=3)
y_pred_snaive = snaive(freq="1mo", sp=12)(y=y_train, fh=3)


# Univariate time-series fit and predict
lags = 12
forecaster = linear_model(freq=freq, lags=lags)
forecaster.fit(y=y_train)
# Predict
y_pred = forecaster.predict(fh=test_size)
# Score
scores = smape(y_true=y_test, y_pred=y_pred)
print("✅ Predictions (univariate):\n", y_pred.sort(entity_col))
print("💯 Scores (univariate):\n", scores.sort("smape"))
print("💯 Scores summary (univariate):\n", scores.select("smape").describe())

# Multivariate
forecaster = linear_model(freq=freq, lags=lags)
forecaster.fit(y=y_train, X=X_train)
# Predict
y_pred = forecaster.predict(fh=test_size, X=X_test)
# Score
scores_with_exog = smape(y_true=y_test, y_pred=y_pred)

print("✅ Predictions (multivariate):\n", y_pred.sort(entity_col))
print("💯 Scores (multivariate):\n", scores_with_exog.sort("smape"))
print(
    "💯 Scores summary (multivariate):\n", scores_with_exog.select("smape").describe()
)

# Check uplift from Fourier features
uplift = (
    scores_with_exog.join(scores, on=entity_col, suffix="_univar")
    .with_columns(
        uplift=pl.col("smape_univar") - pl.col("smape"),
        has_uplift=pl.col("smape_univar") - pl.col("smape") > 0,
    )
    .select([entity_col, "uplift", "has_uplift"])
)

# NOTE: Fourier features lead to uplift for ~20% of commodities
# However, at the expense of an overall mean and variance SMAPE
# (likely due to overfitting on seasonal features)

print("💯 Uplift:\n", uplift.sort("uplift", descending=True))
print("💯 Proportion with uplift:", uplift.get_column("has_uplift").mean())

# "Direct" strategy forecasting
# Predict using the "functional" API
y_pred = linear_model(freq=freq, lags=lags, max_horizons=test_size, strategy="direct")(
    y=y_train, fh=test_size
)

# "Ensemble" strategy forecasting and backtesting
y_preds = linear_model(
    freq=freq, lags=lags, max_horizons=test_size, strategy="ensemble"
).backtest(y=y_train, X=X_train)
print("✅ Backtests:", y_preds)

# Forecast with target transforms and feature transforms
forecaster = linear_model(
    freq="1mo",
    lags=24,
    target_transform=scale(),
    feature_transform=add_fourier_terms(sp=12, K=6),
)
y_pred = forecaster(y=y_train, fh=test_size)

elapsed_time = default_timer() - start_time
print(f"⏱️ Elapsed time: {elapsed_time}")
